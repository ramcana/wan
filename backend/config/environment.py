"""
Environment-specific configuration management.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class Environment(str, Enum):
    """Application environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

class EnvironmentConfig:
    """Manages environment-specific configuration."""
    
    def __init__(self, env: Optional[str] = None):
        self.env = Environment(env or os.getenv("APP_ENV", "development"))
        self.config = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration based on environment."""
        # Base configuration
        base_config_path = Path("config.json")
        if base_config_path.exists():
            with open(base_config_path, 'r') as f:
                self.config = json.load(f)
        
        # Environment-specific configuration
        env_config_path = Path(f"config_{self.env.value}.json")
        if env_config_path.exists():
            with open(env_config_path, 'r') as f:
                env_config = json.load(f)
                self._merge_config(env_config)
        
        # Environment variables override
        self._apply_env_overrides()
        
        logger.info(f"Loaded configuration for environment: {self.env.value}")
    
    def _merge_config(self, env_config: Dict[str, Any]) -> None:
        """Merge environment-specific config with base config."""
        def deep_merge(base: Dict, override: Dict) -> Dict:
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        self.config = deep_merge(self.config, env_config)
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        env_mappings = {
            # API settings
            "API_HOST": ("api_settings", "host"),
            "API_PORT": ("api_settings", "port"),
            "API_CORS_ORIGINS": ("api_settings", "cors_origins"),
            
            # Database settings
            "DATABASE_URL": ("database_settings", "url"),
            "DATABASE_POOL_SIZE": ("database_settings", "pool_size"),
            
            # Model settings
            "MODEL_CACHE_DIR": ("model_settings", "model_cache_dir"),
            "DEFAULT_MODEL": ("model_settings", "default_model"),
            "QUANTIZATION_MODE": ("model_settings", "quantization_mode"),
            
            # Storage settings
            "OUTPUTS_DIR": ("storage_settings", "outputs_dir"),
            "TEMP_DIR": ("storage_settings", "temp_dir"),
            "MAX_STORAGE_GB": ("storage_settings", "max_storage_gb"),
            
            # Logging settings
            "LOG_LEVEL": ("logging_settings", "log_level"),
            "LOG_FILE": ("logging_settings", "log_file"),
            
            # Performance settings
            "MAX_WORKERS": ("performance_settings", "max_workers"),
            "WORKER_TIMEOUT": ("performance_settings", "worker_timeout"),
            
            # Security settings
            "SECRET_KEY": ("security_settings", "secret_key"),
            "JWT_SECRET": ("security_settings", "jwt_secret"),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                if section not in self.config:
                    self.config[section] = {}
                
                # Type conversion
                if key in ["port", "pool_size", "max_storage_gb", "max_workers", "worker_timeout"]:
                    try:
                        value = int(value)
                    except ValueError:
                        logger.warning(f"Invalid integer value for {env_var}: {value}")
                        continue
                elif key == "cors_origins":
                    value = [origin.strip() for origin in value.split(",")]
                elif value.lower() in ["true", "false"]:
                    value = value.lower() == "true"
                
                self.config[section][key] = value
                logger.info(f"Applied environment override: {env_var} -> {section}.{key}")
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(section, {}).get(key, default)
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        return self.config.get(section, {})
    
    def set(self, section: str, key: str, value: Any) -> None:
        """Set configuration value."""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.env == Environment.DEVELOPMENT
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.env == Environment.PRODUCTION
    
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.env == Environment.TESTING
    
    def get_database_url(self) -> str:
        """Get database URL based on environment."""
        if self.is_testing():
            return "sqlite:///test.db"
        elif self.is_development():
            return self.get("database_settings", "url", "sqlite:///wan22_dev.db")
        else:
            return self.get("database_settings", "url", "sqlite:///wan22.db")
    
    def get_log_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        log_level = self.get("logging_settings", "log_level", "INFO")
        log_file = self.get("logging_settings", "log_file", "backend/logs/app.log")
        
        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": log_level,
                    "formatter": "default",
                    "stream": "ext://sys.stdout"
                }
            },
            "loggers": {
                "": {
                    "level": log_level,
                    "handlers": ["console"]
                }
            }
        }
        
        # Add file handler for non-testing environments
        if not self.is_testing():
            # Ensure log directory exists
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            config["handlers"]["file"] = {
                "class": "logging.handlers.RotatingFileHandler",
                "level": log_level,
                "formatter": "detailed",
                "filename": log_file,
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5
            }
            config["loggers"][""]["handlers"].append("file")
        
        return config
    
    def get_cors_origins(self) -> list:
        """Get CORS origins based on environment."""
        if self.is_development():
            return [
                "http://localhost:3000",
                "http://localhost:5173",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:5173"
            ]
        elif self.is_production():
            return self.get("api_settings", "cors_origins", ["https://your-domain.com"])
        else:
            return ["*"]  # Allow all for testing/staging
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration."""
        return {
            "host": self.get("api_settings", "host", "0.0.0.0"),
            "port": self.get("api_settings", "port", 8000),
            "cors_origins": self.get_cors_origins(),
            "max_file_size_mb": self.get("api_settings", "max_file_size_mb", 100),
            "workers": self.get("performance_settings", "max_workers", 4),
            "timeout": self.get("performance_settings", "worker_timeout", 300)
        }
    
    def get_storage_config(self) -> Dict[str, Any]:
        """Get storage configuration."""
        base_dir = "backend" if not self.is_testing() else "test_backend"
        
        return {
            "outputs_dir": self.get("storage_settings", "outputs_dir", f"{base_dir}/outputs"),
            "thumbnails_dir": self.get("storage_settings", "thumbnails_dir", f"{base_dir}/outputs/thumbnails"),
            "temp_dir": self.get("storage_settings", "temp_dir", f"{base_dir}/temp"),
            "max_storage_gb": self.get("storage_settings", "max_storage_gb", 50)
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "default_model": self.get("model_settings", "default_model", "T2V-A14B"),
            "model_cache_dir": self.get("model_settings", "model_cache_dir", "models"),
            "enable_model_offloading": self.get("model_settings", "enable_model_offloading", True),
            "quantization_mode": self.get("model_settings", "quantization_mode", "fp16")
        }
    
    def get_optimization_config(self) -> Dict[str, Any]:
        """Get optimization configuration."""
        return {
            "vram_optimization": self.get("optimization_settings", "vram_optimization", True),
            "cpu_offload": self.get("optimization_settings", "cpu_offload", False),
            "vae_tile_size": self.get("optimization_settings", "vae_tile_size", 512),
            "enable_attention_slicing": self.get("optimization_settings", "enable_attention_slicing", True)
        }
    
    def save_config(self, path: Optional[str] = None) -> None:
        """Save current configuration to file."""
        if path is None:
            path = f"config_{self.env.value}.json"
        
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Configuration saved to {path}")

# Global configuration instance
config = EnvironmentConfig()

def get_config() -> EnvironmentConfig:
    """Get global configuration instance."""
    return config

def reload_config() -> None:
    """Reload configuration from files."""
    global config
    config = EnvironmentConfig()
    logger.info("Configuration reloaded")