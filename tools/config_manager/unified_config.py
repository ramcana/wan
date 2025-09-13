"""
Unified Configuration Schema

This module defines the comprehensive configuration schema for the WAN22 project,
including all system, service, and environment settings with validation rules.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from enum import Enum
import json
import yaml


class LogLevel(Enum):
    """Supported logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class QuantizationLevel(Enum):
    """Supported model quantization levels"""
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    FP32 = "fp32"


class Environment(Enum):
    """Supported deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class SystemConfig:
    """Core system configuration"""
    name: str = "WAN22 Video Generation System"
    version: str = "2.2.0"
    debug: bool = False
    log_level: LogLevel = LogLevel.INFO
    environment: Environment = Environment.DEVELOPMENT
    
    # Performance settings
    max_queue_size: int = 10
    stats_refresh_interval: int = 5
    
    # Directory settings
    output_directory: str = "outputs"
    models_directory: str = "models"
    loras_directory: str = "loras"
    temp_directory: str = "temp"
    cache_directory: str = "cache"


@dataclass
class APIConfig:
    """API server configuration"""
    host: str = "localhost"
    port: int = 8000
    auto_port: bool = True
    debug: bool = False
    reload: bool = True
    workers: int = 1
    timeout: int = 300
    
    # CORS settings
    cors_origins: List[str] = field(default_factory=lambda: [
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ])
    
    # Request limits
    max_request_size: str = "100MB"
    
    # Rate limiting
    rate_limiting_enabled: bool = True
    requests_per_minute: int = 120


@dataclass
class DatabaseConfig:
    """Database configuration"""
    url: str = "sqlite:///./generation_tasks.db"
    echo: bool = False
    pool_size: int = 20
    max_overflow: int = 30
    
    # Connection settings
    connect_timeout: int = 30
    query_timeout: int = 60


@dataclass
class ModelConfig:
    """Model management configuration"""
    base_path: str = "models"
    cache_dir: str = "./model_cache"
    
    # Model settings
    t2v_model: str = "Wan2.2-T2V-A14B"
    i2v_model: str = "Wan2.2-I2V-A14B"
    ti2v_model: str = "Wan2.2-TI2V-5B"
    default_model: str = "t2v-A14B"
    
    # Optimization settings
    auto_optimize: bool = True
    enable_offloading: bool = True
    vram_management: bool = True
    quantization_enabled: bool = True
    quantization_level: QuantizationLevel = QuantizationLevel.BF16
    preload_models: bool = False
    model_cache_size: int = 2
    
    # Download settings
    auto_download_models: bool = True
    download_timeout: int = 1800
    max_concurrent_downloads: int = 2
    auto_cleanup: bool = True
    
    # Supported types
    supported_types: List[str] = field(default_factory=lambda: [
        "t2v-A14B",
        "i2v-A14B", 
        "ti2v-5B"
    ])


@dataclass
class HardwareConfig:
    """Hardware optimization configuration"""
    auto_detect: bool = True
    optimize_for_hardware: bool = True
    
    # Memory settings
    vram_limit_gb: int = 14
    max_vram_usage_gb: int = 14
    cpu_threads: int = 32
    
    # Optimization flags
    enable_mixed_precision: bool = True
    enable_attention_slicing: bool = True
    enable_memory_efficient_attention: bool = True
    enable_xformers: bool = True
    enable_offload: bool = False
    enable_sequential_offload: bool = False
    
    # VAE settings
    vae_tile_size: int = 512
    vae_tile_size_range: List[int] = field(default_factory=lambda: [256, 1024])
    
    # Attention settings
    attention_slice_size: str = "auto"
    
    # Performance features
    enable_torch_compile: bool = True
    torch_compile_mode: str = "default"
    enable_cuda_graphs: bool = False
    memory_format: str = "channels_last"
    enable_flash_attention: bool = True


@dataclass
class GenerationConfig:
    """Video generation configuration"""
    mode: str = "real"
    enable_real_models: bool = True
    fallback_to_mock: bool = False
    
    # Generation limits
    max_concurrent_generations: int = 1
    max_concurrent_jobs: int = 3
    generation_timeout_minutes: int = 45
    default_timeout: int = 600
    
    # Default settings
    default_resolution: str = "1280x720"
    default_steps: int = 50
    default_duration: int = 4
    default_fps: int = 24
    
    # Supported resolutions
    supported_resolutions: List[str] = field(default_factory=lambda: [
        "854x480",
        "480x854", 
        "1280x720",
        "1280x704",
        "1920x1080"
    ])
    
    # Progress tracking
    enable_progress_tracking: bool = True


@dataclass
class UIConfig:
    """User interface configuration"""
    max_file_size_mb: int = 10
    gallery_thumbnail_size: int = 256
    
    # Supported formats
    supported_image_formats: List[str] = field(default_factory=lambda: [
        "PNG", "JPG", "JPEG", "WebP"
    ])


@dataclass
class FrontendConfig:
    """Frontend application configuration"""
    host: str = "localhost"
    port: int = 3000
    auto_port: bool = True
    timeout: int = 30
    open_browser: bool = True
    hot_reload: bool = True
    
    # Build settings
    output_dir: str = "./frontend/dist"
    public_path: str = "/"
    source_maps: bool = True


@dataclass
class WebSocketConfig:
    """WebSocket configuration"""
    enable_progress_updates: bool = True
    detailed_progress: bool = True
    resource_monitoring: bool = True
    update_interval_seconds: float = 0.5
    vram_monitoring: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/application.log"
    file_enabled: bool = True
    console_enabled: bool = True
    max_file_size: str = "50MB"
    backup_count: int = 5
    log_retention_days: int = 30


@dataclass
class SecurityConfig:
    """Security configuration"""
    secret_key: str = "change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    
    # Authentication
    authentication_enabled: bool = False
    token_expiry: int = 3600
    
    # CORS
    cors_enabled: bool = True
    allow_credentials: bool = False
    
    # Admin settings
    allow_admin_elevation: bool = True
    firewall_auto_exception: bool = False
    trusted_port_range: List[int] = field(default_factory=lambda: [8000, 9000])


@dataclass
class PerformanceConfig:
    """Performance monitoring configuration"""
    # Monitoring settings
    enabled: bool = True
    metrics_interval: int = 60
    sample_interval_seconds: float = 30.0
    max_history_samples: int = 100
    
    # Warning thresholds
    vram_warning_threshold: float = 0.9
    vram_warning_percent: int = 90
    cpu_warning_percent: int = 95
    memory_warning_percent: int = 85
    
    # Target performance
    target_720p_time_minutes: int = 9
    target_1080p_time_minutes: int = 17
    
    # Monitoring flags
    cpu_monitoring_enabled: bool = False
    disk_io_monitoring_enabled: bool = False
    network_monitoring_enabled: bool = False


@dataclass
class RecoveryConfig:
    """Error recovery configuration"""
    enabled: bool = True
    max_retry_attempts: int = 3
    retry_delay: float = 2.0
    exponential_backoff: bool = True
    auto_kill_processes: bool = False
    auto_fix_issues: bool = True
    
    # Fallback settings
    fallback_ports: List[int] = field(default_factory=lambda: [
        8080, 8081, 8082, 3001, 3002, 3003
    ])


@dataclass
class EnvironmentValidationConfig:
    """Environment validation configuration"""
    python_min_version: str = "3.8.0"
    node_min_version: str = "16.0.0"
    npm_min_version: str = "8.0.0"
    check_virtual_env: bool = True
    validate_dependencies: bool = True
    auto_install_missing: bool = False


@dataclass
class PromptEnhancementConfig:
    """Prompt enhancement configuration"""
    max_prompt_length: int = 500
    enable_basic_quality: bool = True
    enable_vace_detection: bool = True
    enable_cinematic_enhancement: bool = True
    enable_style_detection: bool = True
    max_quality_keywords: int = 3
    max_cinematic_keywords: int = 3
    max_style_keywords: int = 2


@dataclass
class FeatureFlags:
    """Feature flags configuration"""
    model_management: bool = True
    performance_monitoring: bool = True
    websocket_updates: bool = True
    auto_optimization: bool = True
    prompt_enhancement: bool = True
    error_recovery: bool = True
    health_monitoring: bool = True


@dataclass
class EnvironmentOverrides:
    """Environment-specific configuration overrides"""
    system: Optional[Dict[str, Any]] = None
    api: Optional[Dict[str, Any]] = None
    database: Optional[Dict[str, Any]] = None
    models: Optional[Dict[str, Any]] = None
    hardware: Optional[Dict[str, Any]] = None
    generation: Optional[Dict[str, Any]] = None
    logging: Optional[Dict[str, Any]] = None
    security: Optional[Dict[str, Any]] = None
    performance: Optional[Dict[str, Any]] = None
    features: Optional[Dict[str, Any]] = None


@dataclass
class UnifiedConfig:
    """
    Unified configuration schema for the WAN22 project.
    
    This class provides a comprehensive configuration system that consolidates
    all scattered configuration files into a single, validated structure.
    """
    
    # Core configuration sections
    system: SystemConfig = field(default_factory=SystemConfig)
    api: APIConfig = field(default_factory=APIConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    websocket: WebSocketConfig = field(default_factory=WebSocketConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    environment_validation: EnvironmentValidationConfig = field(default_factory=EnvironmentValidationConfig)
    prompt_enhancement: PromptEnhancementConfig = field(default_factory=PromptEnhancementConfig)
    features: FeatureFlags = field(default_factory=FeatureFlags)
    
    # Environment-specific overrides
    environments: Dict[str, EnvironmentOverrides] = field(default_factory=dict)
    
    # Metadata
    config_version: str = "1.0.0"
    last_updated: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        result = {}
        
        # Convert each section to dict
        for field_name, field_value in self.__dict__.items():
            if hasattr(field_value, '__dict__'):
                # Handle dataclass fields
                result[field_name] = {
                    k: v.value if isinstance(v, Enum) else v
                    for k, v in field_value.__dict__.items()
                }
            elif isinstance(field_value, dict):
                # Handle dictionary fields
                result[field_name] = field_value
            else:
                # Handle primitive fields
                result[field_name] = field_value.value if isinstance(field_value, Enum) else field_value
        
        return result
    
    def to_json(self, indent: int = 2) -> str:
        """Convert configuration to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def to_yaml(self) -> str:
        """Convert configuration to YAML string"""
        return yaml.dump(self.to_dict(), default_flow_style=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedConfig':
        """Create configuration from dictionary"""
        config = cls()
        
        # Update each section if present in data
        for section_name, section_data in data.items():
            if hasattr(config, section_name) and isinstance(section_data, dict):
                section_obj = getattr(config, section_name)
                if hasattr(section_obj, '__dict__'):
                    # Update dataclass fields
                    for key, value in section_data.items():
                        if hasattr(section_obj, key):
                            # Handle enum conversions
                            field_type = type(getattr(section_obj, key))
                            if hasattr(field_type, '__bases__') and Enum in field_type.__bases__:
                                try:
                                    value = field_type(value)
                                except ValueError:
                                    pass  # Keep original value if enum conversion fails
                            setattr(section_obj, key, value)
                else:
                    # Handle non-dataclass fields
                    setattr(config, section_name, section_data)
        
        return config
    
    @classmethod
    def from_json(cls, json_str: str) -> 'UnifiedConfig':
        """Create configuration from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'UnifiedConfig':
        """Create configuration from YAML string"""
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'UnifiedConfig':
        """Load configuration from file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        content = file_path.read_text(encoding='utf-8')
        
        if file_path.suffix.lower() in ['.yaml', '.yml']:
            return cls.from_yaml(content)
        elif file_path.suffix.lower() == '.json':
            return cls.from_json(content)
        else:
            raise ValueError(f"Unsupported configuration file format: {file_path.suffix}")
    
    def save_to_file(self, file_path: Union[str, Path], format: str = 'auto') -> None:
        """Save configuration to file"""
        file_path = Path(file_path)
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine format
        if format == 'auto':
            format = file_path.suffix.lower().lstrip('.')
        
        if format in ['yaml', 'yml']:
            content = self.to_yaml()
        elif format == 'json':
            content = self.to_json()
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        file_path.write_text(content, encoding='utf-8')
    
    def apply_environment_overrides(self, environment: Union[str, Environment]) -> 'UnifiedConfig':
        """Apply environment-specific overrides"""
        if isinstance(environment, str):
            env_key = environment
        else:
            env_key = environment.value
        
        if env_key not in self.environments:
            return self
        
        overrides = self.environments[env_key]
        if not overrides:
            return self
        
        # Create a copy to avoid modifying the original
        config_dict = self.to_dict()
        
        # Apply overrides
        for section_name, section_overrides in overrides.__dict__.items():
            if section_overrides and section_name in config_dict:
                if isinstance(config_dict[section_name], dict):
                    config_dict[section_name].update(section_overrides)
                else:
                    config_dict[section_name] = section_overrides
        
        return self.from_dict(config_dict)
    
    def get_config_path(self, path: str) -> Any:
        """Get configuration value by dot-separated path"""
        parts = path.split('.')
        current = self
        
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                raise KeyError(f"Configuration path not found: {path}")
        
        return current
    
    def set_config_path(self, path: str, value: Any) -> None:
        """Set configuration value by dot-separated path"""
        parts = path.split('.')
        current = self
        
        # Navigate to the parent object
        for part in parts[:-1]:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                raise KeyError(f"Configuration path not found: {path}")
        
        # Set the final value
        final_key = parts[-1]
        if hasattr(current, final_key):
            setattr(current, final_key, value)
        else:
            raise KeyError(f"Configuration path not found: {path}")
