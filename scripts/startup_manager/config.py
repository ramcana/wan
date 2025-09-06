"""
Configuration models for the startup manager using Pydantic.
Handles loading and validation of startup_config.json with environment variable overrides.
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator


class ServerConfig(BaseModel):
    """Base configuration for server instances."""
    host: str = Field(default="localhost", description="Server host address")
    port: Optional[int] = Field(default=None, description="Server port (auto-assigned if None)")
    auto_port: bool = Field(default=True, description="Automatically find available port if specified port is taken")
    timeout: int = Field(default=30, ge=1, le=300, description="Server startup timeout in seconds")


class BackendConfig(ServerConfig):
    """Configuration for FastAPI backend server."""
    port: Optional[int] = Field(default=8000, description="Backend server port")
    reload: bool = Field(default=True, description="Enable auto-reload for development")
    log_level: str = Field(default="info", pattern="^(debug|info|warning|error|critical)$")
    workers: int = Field(default=1, ge=1, le=8, description="Number of worker processes")
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ['debug', 'info', 'warning', 'error', 'critical']
        if v.lower() not in valid_levels:
            raise ValueError(f'log_level must be one of: {", ".join(valid_levels)}')
        return v.lower()


class FrontendConfig(ServerConfig):
    """Configuration for React frontend server."""
    port: Optional[int] = Field(default=3000, description="Frontend server port")
    open_browser: bool = Field(default=True, description="Automatically open browser on startup")
    hot_reload: bool = Field(default=True, description="Enable hot module replacement")


class LoggingConfig(BaseModel):
    """Configuration for logging system."""
    level: str = Field(default="info", pattern="^(debug|info|warning|error|critical)$")
    file_enabled: bool = Field(default=True, description="Enable logging to file")
    console_enabled: bool = Field(default=True, description="Enable console logging")
    max_file_size: int = Field(default=10485760, ge=1048576, description="Maximum log file size in bytes (default 10MB)")
    backup_count: int = Field(default=5, ge=1, le=20, description="Number of backup log files to keep")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log format string")
    
    @field_validator('level')
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ['debug', 'info', 'warning', 'error', 'critical']
        if v.lower() not in valid_levels:
            raise ValueError(f'log_level must be one of: {", ".join(valid_levels)}')
        return v.lower()


class RecoveryConfig(BaseModel):
    """Configuration for recovery and error handling."""
    enabled: bool = Field(default=True, description="Enable automatic recovery mechanisms")
    max_retry_attempts: int = Field(default=3, ge=1, le=10, description="Maximum retry attempts for recovery actions")
    retry_delay: float = Field(default=2.0, ge=0.1, le=30.0, description="Base delay between retry attempts in seconds")
    exponential_backoff: bool = Field(default=True, description="Use exponential backoff for retries")
    auto_kill_processes: bool = Field(default=False, description="Automatically kill conflicting processes")
    fallback_ports: List[int] = Field(default=[8080, 8081, 8082, 3001, 3002, 3003], description="Fallback ports to try")


class EnvironmentConfig(BaseModel):
    """Configuration for environment validation."""
    python_min_version: str = Field(default="3.8.0", description="Minimum required Python version")
    node_min_version: str = Field(default="16.0.0", description="Minimum required Node.js version")
    npm_min_version: str = Field(default="8.0.0", description="Minimum required npm version")
    check_virtual_env: bool = Field(default=True, description="Check if running in virtual environment")
    validate_dependencies: bool = Field(default=True, description="Validate that all dependencies are installed")
    auto_install_missing: bool = Field(default=False, description="Automatically install missing dependencies")


class SecurityConfig(BaseModel):
    """Configuration for security settings."""
    allow_admin_elevation: bool = Field(default=True, description="Allow automatic admin elevation when needed")
    firewall_auto_exception: bool = Field(default=False, description="Automatically add firewall exceptions")
    trusted_port_range: tuple[int, int] = Field(default=(8000, 9000), description="Trusted port range for servers")
    
    @field_validator('trusted_port_range')
    @classmethod
    def validate_port_range(cls, v):
        if len(v) != 2 or v[0] >= v[1] or v[0] < 1024 or v[1] > 65535:
            raise ValueError('trusted_port_range must be a tuple of (start, end) with start < end and both in range 1024-65535')
        return v


class StartupConfig(BaseModel):
    """Main startup configuration containing all server and system settings."""
    backend: BackendConfig = Field(default_factory=BackendConfig)
    frontend: FrontendConfig = Field(default_factory=FrontendConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    recovery: RecoveryConfig = Field(default_factory=RecoveryConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    
    # Legacy fields for backward compatibility
    retry_attempts: int = Field(default=3, ge=1, le=10, description="Number of retry attempts for failed operations (deprecated, use recovery.max_retry_attempts)")
    retry_delay: float = Field(default=2.0, ge=0.1, le=30.0, description="Delay between retry attempts in seconds (deprecated, use recovery.retry_delay)")
    verbose_logging: bool = Field(default=False, description="Enable verbose logging output (deprecated, use logging.level)")
    auto_fix_issues: bool = Field(default=True, description="Automatically attempt to fix detected issues (deprecated, use recovery.enabled)")
    
    model_config = ConfigDict(
        env_prefix="WAN22_",
        env_file=".env",
        case_sensitive=False,
        env_nested_delimiter="__"
    )
    
    @model_validator(mode='after')
    def sync_legacy_fields(self):
        """Sync legacy fields with new structured config for backward compatibility."""
        # If legacy fields are explicitly set, update new config
        if hasattr(self, '_legacy_retry_attempts_set'):
            self.recovery.max_retry_attempts = self.retry_attempts
        if hasattr(self, '_legacy_retry_delay_set'):
            self.recovery.retry_delay = self.retry_delay
        if hasattr(self, '_legacy_verbose_logging_set') and self.verbose_logging:
            self.logging.level = "debug"
        if hasattr(self, '_legacy_auto_fix_issues_set'):
            self.recovery.enabled = self.auto_fix_issues
        
        return self


class ConfigLoader:
    """Handles loading and validation of startup configuration with environment overrides."""
    
    def __init__(self, config_path: Optional[Path] = None, env_prefix: str = "WAN22_"):
        self.config_path = config_path or Path("startup_config.json")
        self.env_prefix = env_prefix
        self._config: Optional[StartupConfig] = None
        self._original_config_data: Optional[Dict[str, Any]] = None
    
    def load_config(self, apply_env_overrides: bool = True) -> StartupConfig:
        """Load configuration from file with optional environment variable overrides."""
        config_data = {}
        
        # Load from file if exists
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    self._original_config_data = config_data.copy()
            except (json.JSONDecodeError, ValueError) as e:
                raise ValueError(f"Invalid configuration file {self.config_path}: {e}")
        
        # Apply environment variable overrides if requested
        if apply_env_overrides:
            env_overrides = self._get_env_overrides()
            config_data = self._merge_config_data(config_data, env_overrides)
        
        try:
            self._config = StartupConfig(**config_data)
        except ValueError as e:
            raise ValueError(f"Configuration validation failed: {e}")
        
        # Create default configuration file if it doesn't exist
        if not self.config_path.exists():
            self.save_config()
        
        return self._config
    
    def _get_env_overrides(self) -> Dict[str, Any]:
        """Extract configuration overrides from environment variables."""
        overrides = {}
        
        for key, value in os.environ.items():
            if key.startswith(self.env_prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(self.env_prefix):].lower()
                
                # Handle nested configuration (e.g., WAN22_BACKEND__PORT)
                if '__' in config_key:
                    parts = config_key.split('__')
                    current = overrides
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = self._parse_env_value(value)
                else:
                    overrides[config_key] = self._parse_env_value(value)
        
        return overrides
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate Python type."""
        # Handle boolean values
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        elif value.lower() in ('false', 'no', '0', 'off'):
            return False
        
        # Handle numeric values
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Handle JSON values (for lists, dicts)
        if value.startswith(('[', '{')):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # Return as string
        return value
    
    def _merge_config_data(self, base_config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration data with overrides."""
        result = base_config.copy()
        
        for key, value in overrides.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config_data(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        if self._config is None:
            raise ValueError("No configuration loaded to save")
        
        config_dict = self._config.model_dump()
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate current configuration and return comprehensive validation results."""
        if self._config is None:
            raise ValueError("No configuration loaded to validate")
        
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "info": []
        }
        
        # Port validation
        self._validate_ports(validation_results)
        
        # Timeout validation
        self._validate_timeouts(validation_results)
        
        # Security validation
        self._validate_security(validation_results)
        
        # Environment validation
        self._validate_environment_config(validation_results)
        
        # Recovery configuration validation
        self._validate_recovery_config(validation_results)
        
        # Logging configuration validation
        self._validate_logging_config(validation_results)
        
        return validation_results
    
    def _validate_ports(self, results: Dict[str, Any]) -> None:
        """Validate port configurations."""
        # Check for port conflicts
        if (self._config.backend.port and self._config.frontend.port and 
            self._config.backend.port == self._config.frontend.port):
            results["valid"] = False
            results["errors"].append(
                f"Backend and frontend cannot use the same port: {self._config.backend.port}"
            )
        
        # Check for reasonable port ranges
        for server_name, config in [("backend", self._config.backend), ("frontend", self._config.frontend)]:
            if config.port:
                if config.port < 1024:
                    results["warnings"].append(
                        f"{server_name.capitalize()} port {config.port} is in privileged range (<1024), may require admin rights"
                    )
                elif config.port > 65535:
                    results["errors"].append(
                        f"{server_name.capitalize()} port {config.port} is invalid (>65535)"
                    )
                    results["valid"] = False
        
        # Check trusted port range
        trusted_start, trusted_end = self._config.security.trusted_port_range
        for server_name, config in [("backend", self._config.backend), ("frontend", self._config.frontend)]:
            if config.port and not (trusted_start <= config.port <= trusted_end):
                results["warnings"].append(
                    f"{server_name.capitalize()} port {config.port} is outside trusted range ({trusted_start}-{trusted_end})"
                )
    
    def _validate_timeouts(self, results: Dict[str, Any]) -> None:
        """Validate timeout configurations."""
        if self._config.backend.timeout < 10:
            results["warnings"].append(
                "Backend timeout is very low (<10s), may cause startup failures"
            )
        
        if self._config.frontend.timeout < 10:
            results["warnings"].append(
                "Frontend timeout is very low (<10s), may cause startup failures"
            )
    
    def _validate_security(self, results: Dict[str, Any]) -> None:
        """Validate security configurations."""
        if self._config.security.firewall_auto_exception:
            results["warnings"].append(
                "Automatic firewall exception creation is enabled - ensure this is intended for your environment"
            )
        
        if not self._config.security.allow_admin_elevation:
            results["info"].append(
                "Admin elevation is disabled - some recovery actions may not be available"
            )
    
    def _validate_environment_config(self, results: Dict[str, Any]) -> None:
        """Validate environment configuration."""
        if self._config.environment.auto_install_missing:
            results["warnings"].append(
                "Automatic dependency installation is enabled - this may modify your system"
            )
        
        if not self._config.environment.check_virtual_env:
            results["info"].append(
                "Virtual environment checking is disabled"
            )
    
    def _validate_recovery_config(self, results: Dict[str, Any]) -> None:
        """Validate recovery configuration."""
        if not self._config.recovery.enabled:
            results["warnings"].append(
                "Automatic recovery is disabled - manual intervention may be required for failures"
            )
        
        if self._config.recovery.auto_kill_processes:
            results["warnings"].append(
                "Automatic process killing is enabled - this may terminate other applications"
            )
        
        if self._config.recovery.max_retry_attempts > 5:
            results["warnings"].append(
                f"High retry attempt count ({self._config.recovery.max_retry_attempts}) may cause long delays"
            )
    
    def _validate_logging_config(self, results: Dict[str, Any]) -> None:
        """Validate logging configuration."""
        if self._config.logging.max_file_size < 1048576:  # 1MB
            results["warnings"].append(
                "Log file size limit is very small (<1MB), logs may rotate frequently"
            )
        
        if self._config.logging.backup_count > 10:
            results["info"].append(
                f"High backup count ({self._config.logging.backup_count}) will use more disk space"
            )
    
    def get_effective_config(self) -> Dict[str, Any]:
        """Get the effective configuration including environment overrides."""
        if self._config is None:
            raise ValueError("No configuration loaded")
        
        return self._config.model_dump()
    
    def get_env_overrides_summary(self) -> Dict[str, Any]:
        """Get summary of active environment variable overrides."""
        if self._original_config_data is None:
            return {}
        
        current_config = self.get_effective_config()
        overrides = {}
        
        # Compare original and current config to identify overrides
        def find_differences(original, current, path=""):
            for key, value in current.items():
                current_path = f"{path}.{key}" if path else key
                
                if key not in original:
                    overrides[current_path] = {"value": value, "source": "environment"}
                elif isinstance(value, dict) and isinstance(original[key], dict):
                    find_differences(original[key], value, current_path)
                elif original[key] != value:
                    overrides[current_path] = {
                        "original": original[key],
                        "value": value,
                        "source": "environment"
                    }
        
        find_differences(self._original_config_data, current_config)
        return overrides
    
    def export_config_for_ci(self, format: str = "env") -> str:
        """Export configuration in format suitable for CI/CD systems."""
        if self._config is None:
            raise ValueError("No configuration loaded")
        
        if format == "env":
            return self._export_as_env_vars()
        elif format == "json":
            return json.dumps(self.get_effective_config(), indent=2)
        elif format == "yaml":
            try:
                import yaml
                return yaml.dump(self.get_effective_config(), default_flow_style=False)
            except ImportError:
                raise ValueError("PyYAML is required for YAML export")
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_as_env_vars(self) -> str:
        """Export configuration as environment variables."""
        lines = []
        config_dict = self.get_effective_config()
        
        def flatten_dict(d, parent_key="", sep="__"):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        
        flat_config = flatten_dict(config_dict)
        
        for key, value in flat_config.items():
            env_key = f"{self.env_prefix}{key.upper()}"
            if isinstance(value, bool):
                env_value = "true" if value else "false"
            elif isinstance(value, (list, dict)):
                env_value = json.dumps(value)
            else:
                env_value = str(value)
            
            lines.append(f"{env_key}={env_value}")
        
        return "\n".join(lines)
    
    def create_deployment_config(self, deployment_type: str = "production") -> Dict[str, Any]:
        """Create optimized configuration for specific deployment types."""
        if self._config is None:
            raise ValueError("No configuration loaded")
        
        base_config = self.get_effective_config()
        
        if deployment_type == "production":
            # Production optimizations
            base_config["logging"]["level"] = "warning"
            base_config["backend"]["reload"] = False
            base_config["frontend"]["hot_reload"] = False
            base_config["frontend"]["open_browser"] = False
            base_config["recovery"]["auto_kill_processes"] = False
            base_config["security"]["allow_admin_elevation"] = False
            
        elif deployment_type == "development":
            # Development optimizations
            base_config["logging"]["level"] = "debug"
            base_config["backend"]["reload"] = True
            base_config["frontend"]["hot_reload"] = True
            base_config["frontend"]["open_browser"] = True
            base_config["recovery"]["enabled"] = True
            
        elif deployment_type == "ci":
            # CI/CD optimizations
            base_config["logging"]["level"] = "info"
            base_config["backend"]["reload"] = False
            base_config["frontend"]["hot_reload"] = False
            base_config["frontend"]["open_browser"] = False
            base_config["recovery"]["auto_kill_processes"] = True
            base_config["environment"]["auto_install_missing"] = True
            
        elif deployment_type == "testing":
            # Testing optimizations
            base_config["logging"]["level"] = "debug"
            base_config["backend"]["timeout"] = 60
            base_config["frontend"]["timeout"] = 60
            base_config["recovery"]["max_retry_attempts"] = 1
            base_config["recovery"]["retry_delay"] = 0.5
            
        return base_config
    
    @property
    def config(self) -> Optional[StartupConfig]:
        """Get current configuration."""
        return self._config


def load_config(config_path: Optional[Path] = None, apply_env_overrides: bool = True, apply_preferences: bool = True) -> StartupConfig:
    """Convenience function to load startup configuration with optional preference application."""
    loader = ConfigLoader(config_path)
    config = loader.load_config(apply_env_overrides)
    
    if apply_preferences:
        try:
            from .preferences import PreferenceManager
            pref_manager = PreferenceManager()
            config = pref_manager.apply_preferences_to_config(config)
        except ImportError:
            pass  # Preferences module not available
    
    return config


def create_default_config_file(config_path: Optional[Path] = None) -> Path:
    """Create a default configuration file with comprehensive settings."""
    path = config_path or Path("startup_config.json")
    
    default_config = StartupConfig()
    config_dict = default_config.model_dump()
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    return path