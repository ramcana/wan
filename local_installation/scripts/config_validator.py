"""
Configuration validation and optimization system.
Validates configuration settings and provides optimization recommendations.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from interfaces import (
    ValidationResult, HardwareProfile, InstallationError, ErrorCategory
)
from base_classes import BaseInstallationComponent


class ConfigurationValidator(BaseInstallationComponent):
    """Validates and optimizes configuration settings."""
    
    def __init__(self, installation_path: str):
        super().__init__(installation_path)
        self.config_dir = self.installation_path / "config"
        self.backup_dir = self.installation_path / "config_backups"
        
        # Configuration limits and safe ranges
        self.limits = {
            "cpu_threads": {"min": 1, "max": 256, "safe_max_ratio": 0.9},
            "worker_threads": {"min": 1, "max": 64, "safe_max": 32},
            "memory_pool_gb": {"min": 1, "max": 512, "safe_max_ratio": 0.8},
            "max_vram_usage_gb": {"min": 1, "max": 80, "safe_max_ratio": 0.9},
            "vae_tile_size": {"min": 64, "max": 1024, "valid_sizes": [64, 128, 256, 384, 512, 768, 1024]},
            "max_queue_size": {"min": 1, "max": 100, "safe_max": 50},
            "cache_size_gb": {"min": 1, "max": 100, "safe_max": 50},
            "temp_space_gb": {"min": 1, "max": 200, "safe_max": 100}
        }
    
    def validate_configuration(self, config: Dict[str, Any], 
                             hardware_profile: Optional[HardwareProfile] = None) -> ValidationResult:
        """Validate configuration against hardware limits and safety constraints."""
        try:
            errors = []
            warnings = []
            
            # Validate structure
            structure_errors = self._validate_config_structure(config)
            errors.extend(structure_errors)
            
            # Validate system settings
            system_errors, system_warnings = self._validate_system_settings(config.get("system", {}))
            errors.extend(system_errors)
            warnings.extend(system_warnings)
            
            # Validate optimization settings
            opt_errors, opt_warnings = self._validate_optimization_settings(
                config.get("optimization", {}), hardware_profile
            )
            errors.extend(opt_errors)
            warnings.extend(opt_warnings)
            
            # Validate model settings
            model_errors, model_warnings = self._validate_model_settings(config.get("models", {}))
            errors.extend(model_errors)
            warnings.extend(model_warnings)
            
            # Check for conflicting settings
            conflict_warnings = self._check_setting_conflicts(config)
            warnings.extend(conflict_warnings)
            
            success = len(errors) == 0
            message = "Configuration validation passed" if success else f"Configuration validation failed with {len(errors)} errors"
            
            return ValidationResult(
                success=success,
                message=message,
                details={
                    "errors": errors,
                    "warnings": warnings,
                    "validation_timestamp": datetime.now().isoformat()
                },
                warnings=warnings
            )
            
        except Exception as e:
            return ValidationResult(
                success=False,
                message=f"Configuration validation failed: {str(e)}",
                details={"exception": str(e)}
            )
    
    def _validate_config_structure(self, config: Dict[str, Any]) -> List[str]:
        """Validate that configuration has required structure."""
        errors = []
        
        required_sections = ["system", "optimization", "models"]
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required configuration section: {section}")
        
        # Validate system section
        if "system" in config:
            required_system_keys = [
                "default_quantization", "enable_offload", "vae_tile_size", 
                "max_queue_size", "worker_threads"
            ]
            for key in required_system_keys:
                if key not in config["system"]:
                    errors.append(f"Missing required system setting: {key}")
        
        # Validate optimization section
        if "optimization" in config:
            required_opt_keys = ["cpu_threads", "memory_pool_gb"]
            for key in required_opt_keys:
                if key not in config["optimization"]:
                    errors.append(f"Missing required optimization setting: {key}")
        
        return errors
    
    def _validate_system_settings(self, system_config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Validate system configuration settings."""
        errors = []
        warnings = []
        
        # Validate quantization
        valid_quantizations = ["bf16", "fp16", "int8", "int4"]
        quantization = system_config.get("default_quantization")
        if quantization and quantization not in valid_quantizations:
            errors.append(f"Invalid quantization '{quantization}'. Valid options: {valid_quantizations}")
        
        # Validate VAE tile size
        vae_tile_size = system_config.get("vae_tile_size")
        if vae_tile_size:
            if not isinstance(vae_tile_size, int):
                errors.append("VAE tile size must be an integer")
            elif vae_tile_size not in self.limits["vae_tile_size"]["valid_sizes"]:
                valid_sizes = self.limits["vae_tile_size"]["valid_sizes"]
                errors.append(f"Invalid VAE tile size {vae_tile_size}. Valid sizes: {valid_sizes}")
        
        # Validate worker threads
        worker_threads = system_config.get("worker_threads")
        if worker_threads:
            if not isinstance(worker_threads, int) or worker_threads < 1:
                errors.append("Worker threads must be a positive integer")
            elif worker_threads > self.limits["worker_threads"]["safe_max"]:
                warnings.append(f"Worker threads ({worker_threads}) exceeds recommended maximum ({self.limits['worker_threads']['safe_max']})")
        
        # Validate queue size
        queue_size = system_config.get("max_queue_size")
        if queue_size:
            if not isinstance(queue_size, int) or queue_size < 1:
                errors.append("Queue size must be a positive integer")
            elif queue_size > self.limits["max_queue_size"]["safe_max"]:
                warnings.append(f"Queue size ({queue_size}) may cause memory issues")
        
        return errors, warnings
    
    def _validate_optimization_settings(self, opt_config: Dict[str, Any], 
                                      hardware_profile: Optional[HardwareProfile]) -> Tuple[List[str], List[str]]:
        """Validate optimization configuration settings."""
        errors = []
        warnings = []
        
        # Validate CPU threads
        cpu_threads = opt_config.get("cpu_threads")
        if cpu_threads:
            if not isinstance(cpu_threads, int) or cpu_threads < 1:
                errors.append("CPU threads must be a positive integer")
            elif hardware_profile:
                max_safe_threads = int(hardware_profile.cpu.threads * self.limits["cpu_threads"]["safe_max_ratio"])
                if cpu_threads > max_safe_threads:
                    warnings.append(f"CPU threads ({cpu_threads}) exceeds safe limit for your CPU ({max_safe_threads})")
        
        # Validate memory pool
        memory_pool = opt_config.get("memory_pool_gb")
        if memory_pool:
            if not isinstance(memory_pool, (int, float)) or memory_pool < 1:
                errors.append("Memory pool must be a positive number")
            elif hardware_profile:
                max_safe_memory = hardware_profile.memory.available_gb * self.limits["memory_pool_gb"]["safe_max_ratio"]
                if memory_pool > max_safe_memory:
                    warnings.append(f"Memory pool ({memory_pool}GB) may cause system instability. Available: {hardware_profile.memory.available_gb}GB")
        
        # Validate VRAM usage
        vram_usage = opt_config.get("max_vram_usage_gb")
        if vram_usage and hardware_profile and hardware_profile.gpu:
            if not isinstance(vram_usage, (int, float)) or vram_usage < 1:
                errors.append("VRAM usage must be a positive number")
            else:
                max_safe_vram = hardware_profile.gpu.vram_gb * self.limits["max_vram_usage_gb"]["safe_max_ratio"]
                if vram_usage > max_safe_vram:
                    warnings.append(f"VRAM usage ({vram_usage}GB) may cause GPU memory errors. GPU VRAM: {hardware_profile.gpu.vram_gb}GB")
        
        return errors, warnings
    
    def _validate_model_settings(self, model_config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Validate model configuration settings."""
        errors = []
        warnings = []
        
        # Validate model precision
        precision = model_config.get("model_precision")
        if precision:
            valid_precisions = ["bf16", "fp16", "int8", "int4"]
            if precision not in valid_precisions:
                errors.append(f"Invalid model precision '{precision}'. Valid options: {valid_precisions}")
        
        # Validate boolean settings
        boolean_settings = ["cache_models", "preload_models"]
        for setting in boolean_settings:
            value = model_config.get(setting)
            if value is not None and not isinstance(value, bool):
                errors.append(f"Setting '{setting}' must be true or false")
        
        # Check for conflicting model settings
        if model_config.get("preload_models") and not model_config.get("cache_models"):
            warnings.append("Preloading models without caching may cause performance issues")
        
        return errors, warnings
    
    def _check_setting_conflicts(self, config: Dict[str, Any]) -> List[str]:
        """Check for conflicting configuration settings."""
        warnings = []
        
        system = config.get("system", {})
        optimization = config.get("optimization", {})
        models = config.get("models", {})
        
        # Check quantization conflicts
        system_quant = system.get("default_quantization")
        model_precision = models.get("model_precision")
        if system_quant and model_precision and system_quant != model_precision:
            warnings.append(f"System quantization ({system_quant}) differs from model precision ({model_precision})")
        
        # Check memory allocation conflicts
        memory_pool = optimization.get("memory_pool_gb", 0)
        cache_size = system.get("cache_size_gb", 0)
        temp_space = system.get("temp_space_gb", 0)
        
        total_allocated = memory_pool + cache_size + temp_space
        if total_allocated > 64:  # Arbitrary high threshold
            warnings.append(f"Total memory allocation ({total_allocated}GB) is very high and may cause issues")
        
        # Check GPU settings without GPU
        if optimization.get("max_vram_usage_gb", 0) > 0 and not system.get("enable_gpu_acceleration", False):
            warnings.append("VRAM usage configured but GPU acceleration is disabled")
        
        return warnings
    
    def optimize_configuration(self, config: Dict[str, Any], 
                             hardware_profile: HardwareProfile) -> Dict[str, Any]:
        """Optimize configuration based on validation results and hardware profile."""
        optimized_config = config.copy()
        
        # Apply safety limits
        optimized_config = self._apply_safety_limits(optimized_config, hardware_profile)
        
        # Apply performance optimizations
        optimized_config = self._apply_performance_optimizations(optimized_config, hardware_profile)
        
        # Apply stability improvements
        optimized_config = self._apply_stability_improvements(optimized_config)
        
        # Add optimization metadata
        optimized_config["metadata"] = optimized_config.get("metadata", {})
        optimized_config["metadata"]["optimization_applied"] = True
        optimized_config["metadata"]["optimization_timestamp"] = datetime.now().isoformat()
        
        return optimized_config
    
    def _apply_safety_limits(self, config: Dict[str, Any], 
                           hardware_profile: HardwareProfile) -> Dict[str, Any]:
        """Apply safety limits to prevent system instability."""
        # Limit CPU threads
        cpu_threads = config.get("optimization", {}).get("cpu_threads", 1)
        max_safe_cpu = int(hardware_profile.cpu.threads * 0.85)
        if cpu_threads > max_safe_cpu:
            config["optimization"]["cpu_threads"] = max_safe_cpu
            self.logger.info(f"Limited CPU threads to {max_safe_cpu} for stability")
        
        # Limit memory usage
        memory_pool = config.get("optimization", {}).get("memory_pool_gb", 1)
        max_safe_memory = hardware_profile.memory.available_gb * 0.75
        if memory_pool > max_safe_memory:
            config["optimization"]["memory_pool_gb"] = int(max_safe_memory)
            self.logger.info(f"Limited memory pool to {int(max_safe_memory)}GB for stability")
        
        # Limit VRAM usage
        if hardware_profile.gpu:
            vram_usage = config.get("optimization", {}).get("max_vram_usage_gb", 1)
            max_safe_vram = hardware_profile.gpu.vram_gb * 0.85
            if vram_usage > max_safe_vram:
                config["optimization"]["max_vram_usage_gb"] = int(max_safe_vram)
                self.logger.info(f"Limited VRAM usage to {int(max_safe_vram)}GB for stability")
        
        return config
    
    def _apply_performance_optimizations(self, config: Dict[str, Any], 
                                       hardware_profile: HardwareProfile) -> Dict[str, Any]:
        """Apply performance optimizations based on hardware capabilities."""
        # Optimize for high-end systems
        if hardware_profile.cpu.cores >= 16 and hardware_profile.memory.total_gb >= 32:
            config["system"]["enable_parallel_processing"] = True
            config["system"]["batch_processing"] = True
        
        # Optimize for SSD storage
        if "ssd" in hardware_profile.storage.type.lower():
            config["system"]["enable_disk_cache"] = True
            config["system"]["cache_strategy"] = "aggressive"
        
        # Optimize for high VRAM GPUs
        if hardware_profile.gpu and hardware_profile.gpu.vram_gb >= 12:
            config["models"]["enable_model_caching"] = True
            config["system"]["gpu_memory_strategy"] = "preload"
        
        return config
    
    def _apply_stability_improvements(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply stability improvements to configuration."""
        # Ensure reasonable queue sizes
        queue_size = config.get("system", {}).get("max_queue_size", 5)
        if queue_size > 20:
            config["system"]["max_queue_size"] = 20
        
        # Enable error recovery
        config["system"]["enable_error_recovery"] = True
        config["system"]["auto_retry_failed_operations"] = True
        
        # Set conservative timeouts
        config["system"]["operation_timeout_seconds"] = 300
        config["system"]["model_load_timeout_seconds"] = 120
        
        return config
    
    def create_backup(self, config_path: str, backup_name: Optional[str] = None) -> str:
        """Create a backup of the configuration file."""
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                raise InstallationError(
                    f"Configuration file not found: {config_path}",
                    ErrorCategory.CONFIGURATION,
                    ["Check file path", "Generate new configuration"]
                )
            
            # Create backup directory
            self.ensure_directory(self.backup_dir)
            
            # Generate backup name
            if not backup_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"config_backup_{timestamp}.json"
            
            backup_path = self.backup_dir / backup_name
            
            # Copy configuration file
            shutil.copy2(config_file, backup_path)
            
            self.logger.info(f"Configuration backup created: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            raise InstallationError(
                f"Failed to create configuration backup: {str(e)}",
                ErrorCategory.SYSTEM,
                ["Check file permissions", "Ensure sufficient disk space"]
            )
    
    def restore_backup(self, backup_path: str, config_path: str) -> bool:
        """Restore configuration from backup."""
        try:
            backup_file = Path(backup_path)
            config_file = Path(config_path)
            
            if not backup_file.exists():
                raise InstallationError(
                    f"Backup file not found: {backup_path}",
                    ErrorCategory.CONFIGURATION,
                    ["Check backup path", "List available backups"]
                )
            
            # Validate backup file
            with open(backup_file, 'r') as f:
                json.load(f)  # This will raise an exception if invalid JSON
            
            # Create backup of current config before restoring
            if config_file.exists():
                current_backup = self.create_backup(str(config_file), "pre_restore_backup.json")
                self.logger.info(f"Current configuration backed up to: {current_backup}")
            
            # Restore from backup
            shutil.copy2(backup_file, config_file)
            
            self.logger.info(f"Configuration restored from: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore configuration backup: {e}")
            return False
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List available configuration backups."""
        backups = []
        
        if not self.backup_dir.exists():
            return backups
        
        for backup_file in self.backup_dir.glob("*.json"):
            try:
                stat = backup_file.stat()
                backups.append({
                    "name": backup_file.name,
                    "path": str(backup_file),
                    "size_bytes": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            except Exception as e:
                self.logger.warning(f"Error reading backup file {backup_file}: {e}")
        
        # Sort by creation time (newest first)
        backups.sort(key=lambda x: x["created"], reverse=True)
        return backups
    
    def cleanup_old_backups(self, keep_count: int = 5) -> int:
        """Clean up old backup files, keeping only the most recent ones."""
        backups = self.list_backups()
        
        if len(backups) <= keep_count:
            return 0
        
        deleted_count = 0
        backups_to_delete = backups[keep_count:]
        
        for backup in backups_to_delete:
            try:
                Path(backup["path"]).unlink()
                deleted_count += 1
                self.logger.info(f"Deleted old backup: {backup['name']}")
            except Exception as e:
                self.logger.warning(f"Failed to delete backup {backup['name']}: {e}")
        
        return deleted_count
    
    def get_optimization_recommendations(self, config: Dict[str, Any], 
                                       hardware_profile: HardwareProfile) -> List[str]:
        """Get optimization recommendations for the current configuration."""
        recommendations = []
        
        # Check CPU utilization
        cpu_threads = config.get("optimization", {}).get("cpu_threads", 1)
        available_threads = hardware_profile.cpu.threads
        if cpu_threads < available_threads * 0.5:
            recommendations.append(f"Consider increasing CPU threads from {cpu_threads} to {int(available_threads * 0.8)} for better performance")
        
        # Check memory utilization
        memory_pool = config.get("optimization", {}).get("memory_pool_gb", 1)
        available_memory = hardware_profile.memory.available_gb
        if memory_pool < available_memory * 0.3:
            recommendations.append(f"Consider increasing memory pool from {memory_pool}GB to {int(available_memory * 0.5)}GB")
        
        # Check GPU utilization
        if hardware_profile.gpu:
            vram_usage = config.get("optimization", {}).get("max_vram_usage_gb", 1)
            available_vram = hardware_profile.gpu.vram_gb
            if vram_usage < available_vram * 0.6:
                recommendations.append(f"Consider increasing VRAM usage from {vram_usage}GB to {int(available_vram * 0.8)}GB")
        
        # Check quantization settings
        quantization = config.get("system", {}).get("default_quantization", "int8")
        if hardware_profile.gpu and hardware_profile.gpu.vram_gb >= 12 and quantization == "int8":
            recommendations.append("Consider using fp16 quantization for better quality with your high-VRAM GPU")
        
        # Check caching settings
        if hardware_profile.gpu and hardware_profile.gpu.vram_gb >= 16:
            if not config.get("models", {}).get("cache_models", False):
                recommendations.append("Enable model caching for better performance with your high-VRAM GPU")
        
        return recommendations
