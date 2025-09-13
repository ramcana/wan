"""
Comprehensive configuration management system.
Integrates configuration generation, validation, optimization, and backup functionality.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from interfaces import HardwareProfile, ValidationResult, InstallationError, ErrorCategory
from base_classes import BaseInstallationComponent
from generate_config import ConfigurationEngine
from config_validator import ConfigurationValidator


class ConfigurationManager(BaseInstallationComponent):
    """Comprehensive configuration management system."""
    
    def __init__(self, installation_path: str):
        super().__init__(installation_path)
        self.config_engine = ConfigurationEngine(installation_path)
        self.config_validator = ConfigurationValidator(installation_path)
        self.config_file = self.installation_path / "config" / "wan22_config.json"
        self.default_config_file = self.installation_path / "resources" / "default_config.json"
    
    def create_optimized_configuration(self, hardware_profile: HardwareProfile, 
                                     variant: str = "balanced") -> Dict[str, Any]:
        """Create an optimized configuration for the given hardware profile."""
        try:
            self.logger.info(f"Creating {variant} configuration for hardware profile")
            
            # Generate base configuration
            if variant == "balanced":
                config = self.config_engine.generate_config(hardware_profile)
            else:
                variants = self.config_engine.create_config_variants(hardware_profile)
                config = variants.get(variant, variants["balanced"])
            
            # Validate the generated configuration
            validation_result = self.config_validator.validate_configuration(config, hardware_profile)
            
            if not validation_result.success:
                self.logger.warning("Generated configuration failed validation, applying fixes")
                config = self._fix_validation_errors(config, validation_result)
            
            # Apply optimizations
            optimized_config = self.config_validator.optimize_configuration(config, hardware_profile)
            
            # Final validation
            final_validation = self.config_validator.validate_configuration(optimized_config, hardware_profile)
            
            if not final_validation.success:
                raise InstallationError(
                    "Failed to create valid configuration after optimization",
                    ErrorCategory.CONFIGURATION,
                    ["Use default configuration", "Check hardware detection results"]
                )
            
            # Add creation metadata
            optimized_config["metadata"] = optimized_config.get("metadata", {})
            optimized_config["metadata"]["created_by"] = "ConfigurationManager"
            optimized_config["metadata"]["variant"] = variant
            optimized_config["metadata"]["validation_passed"] = True
            
            self.logger.info(f"Successfully created optimized {variant} configuration")
            return optimized_config
            
        except Exception as e:
            raise InstallationError(
                f"Failed to create optimized configuration: {str(e)}",
                ErrorCategory.CONFIGURATION,
                ["Use default configuration", "Check hardware profile"]
            )
    
    def save_configuration(self, config: Dict[str, Any], 
                          create_backup: bool = True) -> bool:
        """Save configuration to file with optional backup."""
        try:
            # Create backup if requested and config file exists
            if create_backup and self.config_file.exists():
                backup_path = self.config_validator.create_backup(str(self.config_file))
                self.logger.info(f"Configuration backup created: {backup_path}")
            
            # Ensure config directory exists
            self.ensure_directory(self.config_file.parent)
            
            # Save configuration
            success = self.config_engine.save_config(config, str(self.config_file))
            
            if success:
                self.logger.info(f"Configuration saved to: {self.config_file}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False
    
    def load_configuration(self) -> Optional[Dict[str, Any]]:
        """Load configuration from file."""
        try:
            if self.config_file.exists():
                config = self.load_json_file(self.config_file)
                self.logger.info(f"Configuration loaded from: {self.config_file}")
                return config
            else:
                self.logger.warning("Configuration file not found")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            return None
    
    def validate_current_configuration(self, hardware_profile: Optional[HardwareProfile] = None) -> ValidationResult:
        """Validate the current configuration file."""
        config = self.load_configuration()
        
        if not config:
            return ValidationResult(
                success=False,
                message="No configuration file found",
                details={"error": "Configuration file does not exist"}
            )
        
        return self.config_validator.validate_configuration(config, hardware_profile)
    
    def optimize_current_configuration(self, hardware_profile: HardwareProfile) -> bool:
        """Optimize the current configuration file."""
        try:
            config = self.load_configuration()
            
            if not config:
                self.logger.error("No configuration to optimize")
                return False
            
            # Create backup before optimization
            backup_path = self.config_validator.create_backup(str(self.config_file), "pre_optimization_backup.json")
            self.logger.info(f"Pre-optimization backup created: {backup_path}")
            
            # Optimize configuration
            optimized_config = self.config_validator.optimize_configuration(config, hardware_profile)
            
            # Save optimized configuration
            success = self.save_configuration(optimized_config, create_backup=False)
            
            if success:
                self.logger.info("Configuration optimized and saved")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to optimize configuration: {e}")
            return False
    
    def reset_to_default(self, hardware_profile: HardwareProfile) -> bool:
        """Reset configuration to hardware-optimized defaults."""
        try:
            # Create backup of current configuration
            if self.config_file.exists():
                backup_path = self.config_validator.create_backup(str(self.config_file), "pre_reset_backup.json")
                self.logger.info(f"Pre-reset backup created: {backup_path}")
            
            # Generate new default configuration
            default_config = self.create_optimized_configuration(hardware_profile, "balanced")
            
            # Save new configuration
            success = self.save_configuration(default_config, create_backup=False)
            
            if success:
                self.logger.info("Configuration reset to optimized defaults")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to reset configuration: {e}")
            return False
    
    def get_configuration_status(self, hardware_profile: Optional[HardwareProfile] = None) -> Dict[str, Any]:
        """Get comprehensive status of the current configuration."""
        status = {
            "config_exists": self.config_file.exists(),
            "config_path": str(self.config_file),
            "validation_result": None,
            "recommendations": [],
            "backups_available": len(self.config_validator.list_backups()),
            "last_modified": None
        }
        
        if status["config_exists"]:
            # Get file modification time
            stat = self.config_file.stat()
            status["last_modified"] = stat.st_mtime
            
            # Validate configuration
            validation_result = self.validate_current_configuration(hardware_profile)
            status["validation_result"] = {
                "success": validation_result.success,
                "message": validation_result.message,
                "error_count": len(validation_result.details.get("errors", [])) if validation_result.details else 0,
                "warning_count": len(validation_result.warnings or [])
            }
            
            # Get optimization recommendations
            if hardware_profile:
                config = self.load_configuration()
                if config:
                    status["recommendations"] = self.config_validator.get_optimization_recommendations(config, hardware_profile)
        
        return status
    
    def repair_configuration(self, hardware_profile: HardwareProfile) -> bool:
        """Attempt to repair a broken or invalid configuration."""
        try:
            config = self.load_configuration()
            
            if not config:
                self.logger.info("No configuration found, creating new one")
                return self.reset_to_default(hardware_profile)
            
            # Validate current configuration
            validation_result = self.config_validator.validate_configuration(config, hardware_profile)
            
            if validation_result.success:
                self.logger.info("Configuration is already valid")
                return True
            
            # Create backup before repair
            backup_path = self.config_validator.create_backup(str(self.config_file), "pre_repair_backup.json")
            self.logger.info(f"Pre-repair backup created: {backup_path}")
            
            # Attempt to fix validation errors
            repaired_config = self._fix_validation_errors(config, validation_result)
            
            # Apply optimizations to ensure stability
            optimized_config = self.config_validator.optimize_configuration(repaired_config, hardware_profile)
            
            # Final validation
            final_validation = self.config_validator.validate_configuration(optimized_config, hardware_profile)
            
            if not final_validation.success:
                self.logger.warning("Repair failed, resetting to defaults")
                return self.reset_to_default(hardware_profile)
            
            # Save repaired configuration
            success = self.save_configuration(optimized_config, create_backup=False)
            
            if success:
                self.logger.info("Configuration repaired successfully")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to repair configuration: {e}")
            return self.reset_to_default(hardware_profile)
    
    def _fix_validation_errors(self, config: Dict[str, Any], 
                             validation_result: ValidationResult) -> Dict[str, Any]:
        """Attempt to fix validation errors in configuration."""
        fixed_config = config.copy()
        
        if not validation_result.details or "errors" not in validation_result.details:
            return fixed_config
        
        errors = validation_result.details["errors"]
        
        for error in errors:
            error_lower = error.lower()
            
            # Fix quantization errors
            if "invalid quantization" in error_lower:
                fixed_config.setdefault("system", {})["default_quantization"] = "fp16"
                self.logger.info("Fixed quantization setting to fp16")
            
            # Fix VAE tile size errors
            if "invalid vae tile size" in error_lower:
                fixed_config.setdefault("system", {})["vae_tile_size"] = 256
                self.logger.info("Fixed VAE tile size to 256")
            
            # Fix worker thread errors
            if "worker threads must be" in error_lower:
                fixed_config.setdefault("system", {})["worker_threads"] = 4
                self.logger.info("Fixed worker threads to 4")
            
            # Fix queue size errors
            if "queue size must be" in error_lower:
                fixed_config.setdefault("system", {})["max_queue_size"] = 5
                self.logger.info("Fixed queue size to 5")
            
            # Fix model precision errors
            if "invalid model precision" in error_lower:
                fixed_config.setdefault("models", {})["model_precision"] = "fp16"
                self.logger.info("Fixed model precision to fp16")
            
            # Fix missing sections
            if "missing required" in error_lower:
                if "system" in error_lower:
                    fixed_config.setdefault("system", {})
                elif "optimization" in error_lower:
                    fixed_config.setdefault("optimization", {})
                elif "models" in error_lower:
                    fixed_config.setdefault("models", {})
        
        # Ensure all required keys exist with safe defaults
        fixed_config.setdefault("system", {}).update({
            "default_quantization": fixed_config.get("system", {}).get("default_quantization", "fp16"),
            "enable_offload": fixed_config.get("system", {}).get("enable_offload", True),
            "vae_tile_size": fixed_config.get("system", {}).get("vae_tile_size", 256),
            "max_queue_size": fixed_config.get("system", {}).get("max_queue_size", 5),
            "worker_threads": fixed_config.get("system", {}).get("worker_threads", 4)
        })
        
        fixed_config.setdefault("optimization", {}).update({
            "cpu_threads": fixed_config.get("optimization", {}).get("cpu_threads", 4),
            "memory_pool_gb": fixed_config.get("optimization", {}).get("memory_pool_gb", 4)
        })
        
        fixed_config.setdefault("models", {}).update({
            "cache_models": fixed_config.get("models", {}).get("cache_models", False),
            "preload_models": fixed_config.get("models", {}).get("preload_models", False),
            "model_precision": fixed_config.get("models", {}).get("model_precision", "fp16")
        })
        
        return fixed_config
    
    def create_configuration_report(self, hardware_profile: HardwareProfile) -> Dict[str, Any]:
        """Create a comprehensive configuration report."""
        report = {
            "timestamp": self._get_timestamp(),
            "hardware_profile": {
                "cpu": f"{hardware_profile.cpu.model} ({hardware_profile.cpu.cores}C/{hardware_profile.cpu.threads}T)",
                "memory": f"{hardware_profile.memory.total_gb}GB {hardware_profile.memory.type}",
                "gpu": hardware_profile.gpu.model if hardware_profile.gpu else "None",
                "storage": f"{hardware_profile.storage.available_gb}GB {hardware_profile.storage.type}"
            },
            "configuration_status": self.get_configuration_status(hardware_profile),
            "available_variants": ["balanced", "performance", "memory_conservative", "quality_focused"],
            "backup_info": {
                "backup_count": len(self.config_validator.list_backups()),
                "backups": self.config_validator.list_backups()[:5]  # Show last 5 backups
            }
        }
        
        # Add current configuration details if available
        config = self.load_configuration()
        if config:
            report["current_configuration"] = {
                "quantization": config.get("system", {}).get("default_quantization"),
                "cpu_threads": config.get("optimization", {}).get("cpu_threads"),
                "memory_pool_gb": config.get("optimization", {}).get("memory_pool_gb"),
                "max_vram_usage_gb": config.get("optimization", {}).get("max_vram_usage_gb"),
                "worker_threads": config.get("system", {}).get("worker_threads"),
                "variant": config.get("metadata", {}).get("variant", "unknown")
            }
        
        return report
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for metadata."""
        from datetime import datetime
        return datetime.now().isoformat()


def create_configuration_for_hardware(hardware_profile: HardwareProfile, 
                                    installation_path: str = ".",
                                    variant: str = "balanced") -> Dict[str, Any]:
    """Standalone function to create configuration for given hardware."""
    manager = ConfigurationManager(installation_path)
    return manager.create_optimized_configuration(hardware_profile, variant)


def validate_configuration_file(config_path: str, 
                               hardware_profile: Optional[HardwareProfile] = None) -> ValidationResult:
    """Standalone function to validate a configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        validator = ConfigurationValidator(".")
        return validator.validate_configuration(config, hardware_profile)
        
    except Exception as e:
        return ValidationResult(
            success=False,
            message=f"Failed to validate configuration file: {str(e)}",
            details={"exception": str(e)}
        )
