#!/usr/bin/env python3
"""
Configuration migration script to migrate from existing systems to FastAPI integration.
Handles merging configurations from different sources and ensuring compatibility.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigurationMigrator:
    """Handles configuration migration from existing systems."""
    
    def __init__(self):
        self.backend_path = Path(__file__).parent.parent
        self.config_sources = {
            "fastapi_config": self.backend_path / "config.json",
            "wan22_config": self.backend_path / "main_config.json",
            "local_install_config": self.backend_path.parent / "local_installation" / "config.json",
            "model_config": self.backend_path / "models" / "model_config.json"
        }
        self.merged_config = {}
        
    def load_existing_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load all existing configuration files."""
        configs = {}
        
        for config_name, config_path in self.config_sources.items():
            try:
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        configs[config_name] = json.load(f)
                    logger.info(f"Loaded {config_name} from {config_path}")
                else:
                    logger.warning(f"Configuration file not found: {config_path}")
                    configs[config_name] = {}
            except Exception as e:
                logger.error(f"Failed to load {config_name}: {e}")
                configs[config_name] = {}
        
        return configs
    
    def create_default_config(self) -> Dict[str, Any]:
        """Create default configuration structure."""
        return {
            "generation": {
                "mode": "real",
                "enable_real_models": True,
                "fallback_to_mock": True,
                "auto_download_models": True,
                "max_concurrent_generations": 2,
                "generation_timeout_minutes": 30
            },
            "models": {
                "base_path": "models",
                "auto_optimize": True,
                "enable_offloading": True,
                "vram_management": True,
                "quantization_enabled": True,
                "supported_types": ["t2v-A14B", "i2v-A14B", "ti2v-5B"],
                "default_model": "t2v-A14B"
            },
            "hardware": {
                "auto_detect": True,
                "optimize_for_hardware": True,
                "vram_limit_gb": None,
                "cpu_threads": None,
                "enable_mixed_precision": True
            },
            "websocket": {
                "enable_progress_updates": True,
                "detailed_progress": True,
                "resource_monitoring": True,
                "update_interval_seconds": 1.0
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "debug": False,
                "cors_origins": ["http://localhost:3000"],
                "rate_limiting": {
                    "enabled": True,
                    "requests_per_minute": 60
                }
            },
            "database": {
                "url": "sqlite:///./generation_tasks.db",
                "echo": False,
                "pool_size": 10
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "logs/fastapi_backend.log"
            },
            "security": {
                "secret_key": "your-secret-key-here",
                "algorithm": "HS256",
                "access_token_expire_minutes": 30
            }
        }
    
    def merge_wan22_config(self, configs: Dict[str, Dict[str, Any]]) -> None:
        """Merge WAN2.2 configuration settings."""
        wan22_config = configs.get("wan22_config", {})
        
        if wan22_config:
            # Merge model settings
            if "models" in wan22_config:
                wan22_models = wan22_config["models"]
                self.merged_config["models"].update({
                    "base_path": wan22_models.get("model_path", self.merged_config["models"]["base_path"]),
                    "quantization_enabled": wan22_models.get("enable_quantization", True),
                    "enable_offloading": wan22_models.get("enable_offloading", True)
                })
            
            # Merge hardware settings
            if "hardware" in wan22_config:
                wan22_hardware = wan22_config["hardware"]
                self.merged_config["hardware"].update({
                    "vram_limit_gb": wan22_hardware.get("vram_limit_gb"),
                    "cpu_threads": wan22_hardware.get("cpu_threads"),
                    "enable_mixed_precision": wan22_hardware.get("mixed_precision", True)
                })
            
            # Merge generation settings
            if "generation" in wan22_config:
                wan22_gen = wan22_config["generation"]
                self.merged_config["generation"].update({
                    "generation_timeout_minutes": wan22_gen.get("timeout_minutes", 30),
                    "max_concurrent_generations": wan22_gen.get("max_concurrent", 2)
                })
            
            logger.info("Merged WAN2.2 configuration settings")
    
    def merge_local_install_config(self, configs: Dict[str, Dict[str, Any]]) -> None:
        """Merge local installation configuration settings."""
        local_config = configs.get("local_install_config", {})
        
        if local_config:
            # Merge download settings
            if "download" in local_config:
                download_config = local_config["download"]
                self.merged_config["models"].update({
                    "auto_download_models": download_config.get("auto_download", True),
                    "download_timeout_minutes": download_config.get("timeout_minutes", 60)
                })
            
            # Merge model paths
            if "paths" in local_config:
                paths_config = local_config["paths"]
                self.merged_config["models"]["base_path"] = paths_config.get(
                    "models_dir", self.merged_config["models"]["base_path"]
                )
            
            # Merge optimization settings
            if "optimization" in local_config:
                opt_config = local_config["optimization"]
                self.merged_config["hardware"].update({
                    "auto_detect": opt_config.get("auto_detect_hardware", True),
                    "optimize_for_hardware": opt_config.get("auto_optimize", True)
                })
            
            logger.info("Merged local installation configuration settings")
    
    def merge_model_config(self, configs: Dict[str, Dict[str, Any]]) -> None:
        """Merge model-specific configuration settings."""
        model_config = configs.get("model_config", {})
        
        if model_config:
            # Merge model definitions
            if "models" in model_config:
                models = model_config["models"]
                supported_types = []
                
                for model_id, model_info in models.items():
                    if model_info.get("enabled", True):
                        supported_types.append(model_id)
                
                if supported_types:
                    self.merged_config["models"]["supported_types"] = supported_types
                    self.merged_config["models"]["default_model"] = supported_types[0]
            
            # Merge model parameters
            if "parameters" in model_config:
                params = model_config["parameters"]
                self.merged_config["models"].update({
                    "default_steps": params.get("steps", 20),
                    "default_guidance_scale": params.get("guidance_scale", 7.5),
                    "default_resolution": params.get("resolution", "720p")
                })
            
            logger.info("Merged model configuration settings")
    
    def merge_fastapi_config(self, configs: Dict[str, Dict[str, Any]]) -> None:
        """Merge existing FastAPI configuration settings."""
        fastapi_config = configs.get("fastapi_config", {})
        
        if fastapi_config:
            # Preserve existing API settings
            if "api" in fastapi_config:
                api_config = fastapi_config["api"]
                self.merged_config["api"].update(api_config)
            
            # Preserve database settings
            if "database" in fastapi_config:
                db_config = fastapi_config["database"]
                self.merged_config["database"].update(db_config)
            
            # Preserve security settings
            if "security" in fastapi_config:
                security_config = fastapi_config["security"]
                self.merged_config["security"].update(security_config)
            
            logger.info("Merged existing FastAPI configuration settings")
    
    def validate_merged_config(self) -> List[str]:
        """Validate the merged configuration and return any issues."""
        issues = []
        
        # Check required sections
        required_sections = ["generation", "models", "hardware", "api", "database"]
        for section in required_sections:
            if section not in self.merged_config:
                issues.append(f"Missing required section: {section}")
        
        # Validate model paths
        models_config = self.merged_config.get("models", {})
        model_path = Path(models_config.get("base_path", "models"))
        if not model_path.exists():
            try:
                model_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created models directory: {model_path}")
            except Exception as e:
                issues.append(f"Cannot create models directory {model_path}: {e}")
        
        # Validate supported model types
        supported_types = models_config.get("supported_types", [])
        if not supported_types:
            issues.append("No supported model types configured")
        
        # Validate API settings
        api_config = self.merged_config.get("api", {})
        port = api_config.get("port", 8000)
        if not isinstance(port, int) or port < 1 or port > 65535:
            issues.append(f"Invalid API port: {port}")
        
        # Validate database URL
        db_config = self.merged_config.get("database", {})
        db_url = db_config.get("url", "")
        if not db_url:
            issues.append("Database URL not configured")
        
        return issues
    
    def backup_existing_config(self) -> bool:
        """Backup existing configuration file."""
        config_path = self.config_sources["fastapi_config"]
        backup_path = config_path.with_suffix(".backup.json")
        
        try:
            if config_path.exists():
                import shutil
                shutil.copy2(config_path, backup_path)
                logger.info(f"Backed up existing config to {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to backup config: {e}")
            return False
    
    def save_merged_config(self) -> bool:
        """Save the merged configuration to file."""
        config_path = self.config_sources["fastapi_config"]
        
        try:
            # Ensure directory exists
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write merged configuration
            with open(config_path, 'w') as f:
                json.dump(self.merged_config, f, indent=2)
            
            logger.info(f"Saved merged configuration to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save merged config: {e}")
            return False
    
    def run_migration(self) -> bool:
        """Run the complete configuration migration."""
        logger.info("Starting configuration migration...")
        
        # Load existing configurations
        configs = self.load_existing_configs()
        
        # Start with default configuration
        self.merged_config = self.create_default_config()
        
        # Merge configurations in order of priority (lowest to highest)
        self.merge_local_install_config(configs)
        self.merge_wan22_config(configs)
        self.merge_model_config(configs)
        self.merge_fastapi_config(configs)  # Highest priority
        
        # Validate merged configuration
        issues = self.validate_merged_config()
        if issues:
            logger.error("Configuration validation failed:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return False
        
        # Backup existing configuration
        if not self.backup_existing_config():
            logger.warning("Failed to backup existing configuration")
        
        # Save merged configuration
        if not self.save_merged_config():
            logger.error("Failed to save merged configuration")
            return False
        
        logger.info("Configuration migration completed successfully!")
        return True

def main():
    """Main migration function."""
    migrator = ConfigurationMigrator()
    
    try:
        success = migrator.run_migration()
        
        if success:
            print("\n✅ Configuration migration completed successfully!")
            print("Your configuration has been merged and is ready for use.")
        else:
            print("\n❌ Configuration migration failed.")
            print("Please check the logs and resolve any issues.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Migration interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Migration failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()