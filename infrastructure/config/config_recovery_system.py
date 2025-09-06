"""
Configuration Recovery System

This module provides configuration recovery functionality including restoration from 
known good defaults, configuration change reporting, and trust_remote_code handling.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from config_validator import (
    ValidationSeverity,
    ValidationMessage,
    ValidationResult,
    ConfigValidator
)


class RecoveryAction(Enum):
    """Types of recovery actions"""
    RESTORE_FROM_BACKUP = "restore_from_backup"
    RESTORE_FROM_DEFAULTS = "restore_from_defaults"
    MERGE_WITH_DEFAULTS = "merge_with_defaults"
    REPAIR_CORRUPTION = "repair_corruption"
    UPDATE_TRUST_REMOTE_CODE = "update_trust_remote_code"


@dataclass
class RecoveryResult:
    """Result of configuration recovery operation"""
    success: bool
    action_taken: RecoveryAction
    original_config_path: Optional[str] = None
    backup_path: Optional[str] = None
    restored_config_path: Optional[str] = None
    changes_made: List[str] = None
    messages: List[ValidationMessage] = None
    
    def __post_init__(self):
        if self.changes_made is None:
            self.changes_made = []
        if self.messages is None:
            self.messages = []


@dataclass
class ConfigChange:
    """Represents a configuration change"""
    timestamp: datetime
    config_path: str
    change_type: str
    old_value: Any
    new_value: Any
    reason: str


class ConfigRecoverySystem:
    """
    Configuration recovery system for WAN22.
    
    Provides functionality to recover from corrupted configurations,
    restore from known good defaults, and manage trust_remote_code settings.
    """
    
    def __init__(self, backup_dir: Optional[Path] = None, recovery_dir: Optional[Path] = None):
        """
        Initialize the configuration recovery system.
        
        Args:
            backup_dir: Directory for configuration backups
            recovery_dir: Directory for recovery operations
        """
        self.logger = logging.getLogger(__name__)
        self.backup_dir = backup_dir or Path("config_backups")
        self.recovery_dir = recovery_dir or Path("config_recovery")
        
        # Ensure directories exist
        self.backup_dir.mkdir(exist_ok=True)
        self.recovery_dir.mkdir(exist_ok=True)
        
        # Initialize validator
        self.validator = ConfigValidator(backup_dir=self.backup_dir)
        
        # Track configuration changes
        self.changes_log_path = self.recovery_dir / "config_changes.json"
        self.changes_history = self._load_changes_history()
        
        # Define known good defaults
        self.default_configs = self._define_default_configs()
    
    def _load_changes_history(self) -> List[Dict[str, Any]]:
        """Load configuration changes history"""
        try:
            if self.changes_log_path.exists():
                with open(self.changes_log_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            self.logger.warning(f"Failed to load changes history: {e}")
            return []
    
    def _save_changes_history(self):
        """Save configuration changes history"""
        try:
            with open(self.changes_log_path, 'w', encoding='utf-8') as f:
                json.dump(self.changes_history, f, indent=2, default=str, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save changes history: {e}")
    
    def _define_default_configs(self) -> Dict[str, Dict[str, Any]]:
        """Define known good default configurations"""
        return {
            "main_config": {
                "system": {
                    "default_quantization": "bf16",
                    "enable_offload": True,
                    "vae_tile_size": 256,
                    "max_queue_size": 10,
                    "stats_refresh_interval": 5
                },
                "directories": {
                    "output_directory": "outputs",
                    "models_directory": "models",
                    "loras_directory": "loras"
                },
                "generation": {
                    "default_resolution": "1280x720",
                    "default_steps": 50,
                    "default_duration": 4,
                    "default_fps": 24,
                    "max_prompt_length": 500,
                    "supported_resolutions": [
                        "854x480",
                        "480x854", 
                        "1280x720",
                        "1280x704",
                        "1920x1080"
                    ]
                },
                "models": {
                    "t2v_model": "Wan2.2-T2V-A14B",
                    "i2v_model": "Wan2.2-I2V-A14B",
                    "ti2v_model": "Wan2.2-TI2V-5B"
                },
                "optimization": {
                    "default_quantization": "bf16",
                    "quantization_levels": ["fp16", "bf16", "int8"],
                    "vae_tile_size_range": [128, 512],
                    "max_vram_usage_gb": 12,
                    "enable_offload": True,
                    "vae_tile_size": 256
                },
                "ui": {
                    "max_file_size_mb": 10,
                    "supported_image_formats": ["PNG", "JPG", "JPEG", "WebP"],
                    "gallery_thumbnail_size": 256
                },
                "performance": {
                    "target_720p_time_minutes": 9,
                    "target_1080p_time_minutes": 17,
                    "vram_warning_threshold": 0.9,
                    "cpu_warning_percent": 95,
                    "memory_warning_percent": 85,
                    "vram_warning_percent": 90,
                    "sample_interval_seconds": 30.0,
                    "max_history_samples": 100,
                    "cpu_monitoring_enabled": False,
                    "disk_io_monitoring_enabled": False,
                    "network_monitoring_enabled": False
                },
                "prompt_enhancement": {
                    "max_prompt_length": 500,
                    "enable_basic_quality": True,
                    "enable_vace_detection": True,
                    "enable_cinematic_enhancement": True,
                    "enable_style_detection": True,
                    "max_quality_keywords": 3,
                    "max_cinematic_keywords": 3,
                    "max_style_keywords": 2
                }
            },
            "model_index": {
                "_class_name": "WanPipeline",
                "_diffusers_version": "0.21.0",
                "text_encoder": ["transformers", "CLIPTextModel"],
                "tokenizer": ["transformers", "CLIPTokenizer"],
                "unet": ["diffusers", "UNet2DConditionModel"],
                "vae": ["diffusers", "AutoencoderKL"],
                "scheduler": ["diffusers", "PNDMScheduler"]
            },
            "vae_config": {
                "_class_name": "AutoencoderKL",
                "_diffusers_version": "0.21.0",
                "act_fn": "silu",
                "block_out_channels": [128, 256, 512, 512],
                "down_block_types": [
                    "DownEncoderBlock2D",
                    "DownEncoderBlock2D", 
                    "DownEncoderBlock2D",
                    "DownEncoderBlock2D"
                ],
                "in_channels": 3,
                "latent_channels": 4,
                "layers_per_block": 2,
                "norm_num_groups": 32,
                "out_channels": 3,
                "sample_size": 512,
                "scaling_factor": 0.18215,
                "up_block_types": [
                    "UpDecoderBlock2D",
                    "UpDecoderBlock2D",
                    "UpDecoderBlock2D", 
                    "UpDecoderBlock2D"
                ]
            }
        }
    
    def recover_config(self, config_path: Union[str, Path], recovery_strategy: Optional[str] = None) -> RecoveryResult:
        """
        Recover a corrupted or invalid configuration.
        
        Args:
            config_path: Path to configuration file to recover
            recovery_strategy: Specific recovery strategy to use (optional)
            
        Returns:
            RecoveryResult with recovery information
        """
        config_path = Path(config_path)
        
        try:
            # First, validate the current configuration
            validation_result = self.validator.validate_config_file(config_path)
            
            # If configuration is valid, check if cleanup was performed
            if validation_result.is_valid:
                if validation_result.cleaned_attributes:
                    # Log cleanup as a change
                    self._log_config_change(
                        config_path=str(config_path),
                        change_type="cleanup_attributes",
                        old_value="config_with_problematic_attributes",
                        new_value="cleaned_config",
                        reason=f"Cleaned up {len(validation_result.cleaned_attributes)} problematic attributes"
                    )
                    
                    return RecoveryResult(
                        success=True,
                        action_taken=RecoveryAction.REPAIR_CORRUPTION,
                        original_config_path=str(config_path),
                        backup_path=validation_result.backup_path,
                        changes_made=[f"Cleaned attribute: {attr}" for attr in validation_result.cleaned_attributes],
                        messages=[ValidationMessage(
                            severity=ValidationSeverity.INFO,
                            code="ATTRIBUTES_CLEANED",
                            message=f"Configuration cleaned up, {len(validation_result.cleaned_attributes)} attributes removed",
                            field_path=str(config_path)
                        )]
                    )
                else:
                    # No cleanup needed
                    return RecoveryResult(
                        success=True,
                        action_taken=RecoveryAction.REPAIR_CORRUPTION,
                        original_config_path=str(config_path),
                        messages=[ValidationMessage(
                            severity=ValidationSeverity.INFO,
                            code="NO_RECOVERY_NEEDED",
                            message="Configuration is already valid",
                            field_path=str(config_path)
                        )]
                    )
            
            # Determine recovery strategy
            if recovery_strategy is None:
                recovery_strategy = self._determine_recovery_strategy(validation_result)
            
            # Execute recovery based on strategy
            if recovery_strategy == "restore_from_backup":
                return self._restore_from_backup(config_path)
            elif recovery_strategy == "restore_from_defaults":
                return self._restore_from_defaults(config_path)
            elif recovery_strategy == "merge_with_defaults":
                return self._merge_with_defaults(config_path)
            elif recovery_strategy == "repair_corruption":
                return self._repair_corruption(config_path, validation_result)
            else:
                # Default to merge with defaults
                return self._merge_with_defaults(config_path)
                
        except Exception as e:
            self.logger.error(f"Recovery failed for {config_path}: {e}")
            return RecoveryResult(
                success=False,
                action_taken=RecoveryAction.REPAIR_CORRUPTION,
                original_config_path=str(config_path),
                messages=[ValidationMessage(
                    severity=ValidationSeverity.CRITICAL,
                    code="RECOVERY_FAILED",
                    message=f"Recovery operation failed: {e}",
                    field_path=str(config_path)
                )]
            )
    
    def _determine_recovery_strategy(self, validation_result: ValidationResult) -> str:
        """Determine the best recovery strategy based on validation results"""
        
        # Check for critical errors that require full restoration
        critical_errors = [msg for msg in validation_result.messages 
                          if msg.severity == ValidationSeverity.CRITICAL]
        
        if critical_errors:
            # Check if it's a JSON parsing error
            json_errors = [msg for msg in critical_errors if "JSON" in msg.code]
            if json_errors:
                return "restore_from_backup"  # Try backup first for JSON errors
            else:
                return "restore_from_defaults"  # Use defaults for other critical errors
        
        # Check for multiple errors
        error_count = len([msg for msg in validation_result.messages 
                          if msg.severity == ValidationSeverity.ERROR])
        
        if error_count > 5:
            return "restore_from_defaults"  # Too many errors, use defaults
        elif error_count > 0:
            return "merge_with_defaults"  # Some errors, merge with defaults
        else:
            return "repair_corruption"  # Only warnings, try to repair
    
    def _restore_from_backup(self, config_path: Path) -> RecoveryResult:
        """Restore configuration from the most recent backup"""
        
        try:
            # Find the most recent backup
            backup_pattern = f"{config_path.stem}_*.backup{config_path.suffix}"
            backup_files = list(self.backup_dir.glob(backup_pattern))
            
            if not backup_files:
                # No backup found, fall back to defaults
                return self._restore_from_defaults(config_path)
            
            # Sort by modification time, get most recent
            most_recent_backup = max(backup_files, key=lambda p: p.stat().st_mtime)
            
            # Validate the backup before restoring
            backup_validation = self.validator.validate_config_file(most_recent_backup)
            
            if not backup_validation.is_valid:
                # Backup is also corrupted, use defaults
                return self._restore_from_defaults(config_path)
            
            # Create a backup of the current (corrupted) file
            corrupted_backup_path = self._create_corrupted_backup(config_path)
            
            # Restore from backup
            shutil.copy2(most_recent_backup, config_path)
            
            # Log the change
            self._log_config_change(
                config_path=str(config_path),
                change_type="restore_from_backup",
                old_value="corrupted_config",
                new_value=str(most_recent_backup),
                reason="Configuration was corrupted, restored from backup"
            )
            
            return RecoveryResult(
                success=True,
                action_taken=RecoveryAction.RESTORE_FROM_BACKUP,
                original_config_path=str(config_path),
                backup_path=str(corrupted_backup_path),
                restored_config_path=str(config_path),
                changes_made=[f"Restored from backup: {most_recent_backup.name}"],
                messages=[ValidationMessage(
                    severity=ValidationSeverity.INFO,
                    code="RESTORED_FROM_BACKUP",
                    message=f"Configuration restored from backup: {most_recent_backup.name}",
                    field_path=str(config_path)
                )]
            )
            
        except Exception as e:
            self.logger.error(f"Failed to restore from backup: {e}")
            # Fall back to defaults
            return self._restore_from_defaults(config_path)
    
    def _restore_from_defaults(self, config_path: Path) -> RecoveryResult:
        """Restore configuration from known good defaults"""
        
        try:
            # Determine config type
            config_type = self._determine_config_type(config_path)
            
            if config_type not in self.default_configs:
                return RecoveryResult(
                    success=False,
                    action_taken=RecoveryAction.RESTORE_FROM_DEFAULTS,
                    original_config_path=str(config_path),
                    messages=[ValidationMessage(
                        severity=ValidationSeverity.ERROR,
                        code="NO_DEFAULT_CONFIG",
                        message=f"No default configuration available for {config_type}",
                        field_path=str(config_path)
                    )]
                )
            
            # Create backup of corrupted file
            corrupted_backup_path = self._create_corrupted_backup(config_path)
            
            # Write default configuration
            default_config = self.default_configs[config_type]
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            
            # Log the change
            self._log_config_change(
                config_path=str(config_path),
                change_type="restore_from_defaults",
                old_value="corrupted_config",
                new_value=config_type,
                reason="Configuration was corrupted, restored from defaults"
            )
            
            return RecoveryResult(
                success=True,
                action_taken=RecoveryAction.RESTORE_FROM_DEFAULTS,
                original_config_path=str(config_path),
                backup_path=str(corrupted_backup_path),
                restored_config_path=str(config_path),
                changes_made=[f"Restored from defaults: {config_type}"],
                messages=[ValidationMessage(
                    severity=ValidationSeverity.INFO,
                    code="RESTORED_FROM_DEFAULTS",
                    message=f"Configuration restored from defaults: {config_type}",
                    field_path=str(config_path)
                )]
            )
            
        except Exception as e:
            self.logger.error(f"Failed to restore from defaults: {e}")
            return RecoveryResult(
                success=False,
                action_taken=RecoveryAction.RESTORE_FROM_DEFAULTS,
                original_config_path=str(config_path),
                messages=[ValidationMessage(
                    severity=ValidationSeverity.CRITICAL,
                    code="RESTORE_FAILED",
                    message=f"Failed to restore from defaults: {e}",
                    field_path=str(config_path)
                )]
            )
    
    def _merge_with_defaults(self, config_path: Path) -> RecoveryResult:
        """Merge current configuration with defaults to fill missing values"""
        
        try:
            # Load current configuration
            with open(config_path, 'r', encoding='utf-8') as f:
                current_config = json.load(f)
            
            # Determine config type and get defaults
            config_type = self._determine_config_type(config_path)
            
            if config_type not in self.default_configs:
                return self._restore_from_defaults(config_path)
            
            default_config = self.default_configs[config_type]
            
            # Create backup
            backup_path = self.validator.create_backup(config_path)
            
            # Merge configurations
            merged_config, changes = self._deep_merge_configs(current_config, default_config)
            
            # Write merged configuration
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(merged_config, f, indent=2, ensure_ascii=False)
            
            # Log the change
            self._log_config_change(
                config_path=str(config_path),
                change_type="merge_with_defaults",
                old_value="partial_config",
                new_value="merged_config",
                reason="Configuration had missing values, merged with defaults"
            )
            
            return RecoveryResult(
                success=True,
                action_taken=RecoveryAction.MERGE_WITH_DEFAULTS,
                original_config_path=str(config_path),
                backup_path=backup_path,
                restored_config_path=str(config_path),
                changes_made=changes,
                messages=[ValidationMessage(
                    severity=ValidationSeverity.INFO,
                    code="MERGED_WITH_DEFAULTS",
                    message=f"Configuration merged with defaults, {len(changes)} changes made",
                    field_path=str(config_path)
                )]
            )
            
        except Exception as e:
            self.logger.error(f"Failed to merge with defaults: {e}")
            return self._restore_from_defaults(config_path)
    
    def _repair_corruption(self, config_path: Path, validation_result: ValidationResult) -> RecoveryResult:
        """Attempt to repair minor configuration corruption"""
        
        try:
            # Create backup
            backup_path = self.validator.create_backup(config_path)
            
            # Load current configuration
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            changes_made = []
            
            # Fix specific validation errors
            for message in validation_result.messages:
                if message.severity in [ValidationSeverity.ERROR, ValidationSeverity.WARNING]:
                    if message.suggested_value is not None:
                        # Apply suggested fix
                        field_parts = message.field_path.split('.')
                        self._set_nested_value(config_data, field_parts, message.suggested_value)
                        changes_made.append(f"Fixed {message.field_path}: {message.current_value} -> {message.suggested_value}")
            
            # Write repaired configuration
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            # Log the change
            self._log_config_change(
                config_path=str(config_path),
                change_type="repair_corruption",
                old_value="corrupted_values",
                new_value="repaired_values",
                reason="Configuration had validation errors, applied suggested fixes"
            )
            
            return RecoveryResult(
                success=True,
                action_taken=RecoveryAction.REPAIR_CORRUPTION,
                original_config_path=str(config_path),
                backup_path=backup_path,
                restored_config_path=str(config_path),
                changes_made=changes_made,
                messages=[ValidationMessage(
                    severity=ValidationSeverity.INFO,
                    code="REPAIRED_CORRUPTION",
                    message=f"Configuration repaired, {len(changes_made)} fixes applied",
                    field_path=str(config_path)
                )]
            )
            
        except Exception as e:
            self.logger.error(f"Failed to repair corruption: {e}")
            return self._merge_with_defaults(config_path)
    
    def update_trust_remote_code_setting(self, config_path: Union[str, Path], 
                                        enable_trust: bool, 
                                        model_names: Optional[List[str]] = None) -> RecoveryResult:
        """
        Update trust_remote_code settings for specific models.
        
        Args:
            config_path: Path to configuration file
            enable_trust: Whether to enable trust_remote_code
            model_names: List of model names to update (optional)
            
        Returns:
            RecoveryResult with update information
        """
        config_path = Path(config_path)
        
        try:
            # Load current configuration
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Create backup
            backup_path = self.validator.create_backup(config_path)
            
            changes_made = []
            
            # Add trust_remote_code section if it doesn't exist
            if "trust_remote_code" not in config_data:
                config_data["trust_remote_code"] = {}
            
            # Update trust settings
            if model_names:
                for model_name in model_names:
                    config_data["trust_remote_code"][model_name] = enable_trust
                    changes_made.append(f"Set trust_remote_code for {model_name}: {enable_trust}")
            else:
                # Global setting
                config_data["trust_remote_code"]["global"] = enable_trust
                changes_made.append(f"Set global trust_remote_code: {enable_trust}")
            
            # Write updated configuration
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            # Log the change
            self._log_config_change(
                config_path=str(config_path),
                change_type="update_trust_remote_code",
                old_value="previous_trust_settings",
                new_value=enable_trust,
                reason=f"Updated trust_remote_code settings for {len(model_names) if model_names else 1} models"
            )
            
            return RecoveryResult(
                success=True,
                action_taken=RecoveryAction.UPDATE_TRUST_REMOTE_CODE,
                original_config_path=str(config_path),
                backup_path=backup_path,
                restored_config_path=str(config_path),
                changes_made=changes_made,
                messages=[ValidationMessage(
                    severity=ValidationSeverity.INFO,
                    code="TRUST_REMOTE_CODE_UPDATED",
                    message=f"Updated trust_remote_code settings, {len(changes_made)} changes made",
                    field_path=str(config_path)
                )]
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update trust_remote_code settings: {e}")
            return RecoveryResult(
                success=False,
                action_taken=RecoveryAction.UPDATE_TRUST_REMOTE_CODE,
                original_config_path=str(config_path),
                messages=[ValidationMessage(
                    severity=ValidationSeverity.ERROR,
                    code="TRUST_UPDATE_FAILED",
                    message=f"Failed to update trust_remote_code settings: {e}",
                    field_path=str(config_path)
                )]
            )
    
    def get_config_changes_report(self, config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        Generate a report of configuration changes.
        
        Args:
            config_path: Specific config path to report on (optional)
            
        Returns:
            Dictionary containing changes report
        """
        changes = self.changes_history
        
        if config_path:
            config_path_str = str(Path(config_path))
            changes = [change for change in changes if change.get("config_path") == config_path_str]
        
        # Group changes by type
        changes_by_type = {}
        for change in changes:
            change_type = change.get("change_type", "unknown")
            if change_type not in changes_by_type:
                changes_by_type[change_type] = []
            changes_by_type[change_type].append(change)
        
        # Generate summary
        total_changes = len(changes)
        recent_changes = [change for change in changes 
                         if datetime.fromisoformat(change.get("timestamp", "1970-01-01T00:00:00")) 
                         > datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)]
        
        return {
            "total_changes": total_changes,
            "recent_changes": len(recent_changes),
            "changes_by_type": {k: len(v) for k, v in changes_by_type.items()},
            "detailed_changes": changes,
            "generated_at": datetime.now().isoformat()
        }
    
    def _determine_config_type(self, config_path: Path) -> str:
        """Determine the type of configuration file"""
        filename = config_path.name.lower()
        
        if filename == "config.json":
            return "main_config"
        elif filename == "model_index.json":
            return "model_index"
        elif "vae" in str(config_path).lower() and filename == "config.json":
            return "vae_config"
        else:
            return "main_config"  # Default fallback
    
    def _create_corrupted_backup(self, config_path: Path) -> Path:
        """Create a backup of the corrupted configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{config_path.stem}_corrupted_{timestamp}{config_path.suffix}"
        backup_path = self.recovery_dir / backup_filename
        
        try:
            shutil.copy2(config_path, backup_path)
            self.logger.info(f"Corrupted config backed up to: {backup_path}")
            return backup_path
        except Exception as e:
            self.logger.error(f"Failed to backup corrupted config: {e}")
            raise
    
    def _deep_merge_configs(self, current: Dict[str, Any], defaults: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Deep merge current configuration with defaults"""
        merged = current.copy()
        changes = []
        
        def merge_recursive(current_dict, default_dict, path=""):
            for key, default_value in default_dict.items():
                current_path = f"{path}.{key}" if path else key
                
                if key not in current_dict:
                    # Missing key, add from defaults
                    current_dict[key] = default_value
                    changes.append(f"Added missing key: {current_path}")
                elif isinstance(default_value, dict) and isinstance(current_dict[key], dict):
                    # Recursive merge for nested dictionaries
                    merge_recursive(current_dict[key], default_value, current_path)
                # If key exists and types match, keep current value
        
        merge_recursive(merged, defaults)
        return merged, changes
    
    def _set_nested_value(self, config_data: Dict[str, Any], field_parts: List[str], value: Any):
        """Set a nested value in configuration data"""
        current = config_data
        
        # Navigate to the parent of the target field
        for part in field_parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Set the final value
        if field_parts:
            current[field_parts[-1]] = value
    
    def _log_config_change(self, config_path: str, change_type: str, old_value: Any, new_value: Any, reason: str):
        """Log a configuration change"""
        change = {
            "timestamp": datetime.now().isoformat(),
            "config_path": config_path,
            "change_type": change_type,
            "old_value": str(old_value),
            "new_value": str(new_value),
            "reason": reason
        }
        
        self.changes_history.append(change)
        self._save_changes_history()


def recover_config(config_path: Union[str, Path], 
                  recovery_strategy: Optional[str] = None,
                  backup_dir: Optional[Path] = None,
                  recovery_dir: Optional[Path] = None) -> RecoveryResult:
    """
    Convenience function to recover a configuration file.
    
    Args:
        config_path: Path to configuration file to recover
        recovery_strategy: Specific recovery strategy to use (optional)
        backup_dir: Directory for backups (optional)
        recovery_dir: Directory for recovery operations (optional)
        
    Returns:
        RecoveryResult with recovery information
    """
    recovery_system = ConfigRecoverySystem(backup_dir, recovery_dir)
    return recovery_system.recover_config(config_path, recovery_strategy)


def update_trust_remote_code(config_path: Union[str, Path], 
                           enable_trust: bool,
                           model_names: Optional[List[str]] = None,
                           backup_dir: Optional[Path] = None,
                           recovery_dir: Optional[Path] = None) -> RecoveryResult:
    """
    Convenience function to update trust_remote_code settings.
    
    Args:
        config_path: Path to configuration file
        enable_trust: Whether to enable trust_remote_code
        model_names: List of model names to update (optional)
        backup_dir: Directory for backups (optional)
        recovery_dir: Directory for recovery operations (optional)
        
    Returns:
        RecoveryResult with update information
    """
    recovery_system = ConfigRecoverySystem(backup_dir, recovery_dir)
    return recovery_system.update_trust_remote_code_setting(config_path, enable_trust, model_names)


def format_recovery_result(result: RecoveryResult) -> str:
    """
    Format recovery result for display.
    
    Args:
        result: RecoveryResult to format
        
    Returns:
        Formatted string representation
    """
    lines = []
    
    if result.success:
        lines.append("✅ Configuration recovery successful")
        lines.append(f"🔧 Action taken: {result.action_taken.value}")
    else:
        lines.append("❌ Configuration recovery failed")
        lines.append(f"🔧 Attempted action: {result.action_taken.value}")
    
    if result.backup_path:
        lines.append(f"📁 Backup created: {result.backup_path}")
    
    if result.changes_made:
        lines.append(f"📝 Changes made ({len(result.changes_made)}):")
        for change in result.changes_made:
            lines.append(f"  - {change}")
    
    if result.messages:
        lines.append("\n📋 Messages:")
        for msg in result.messages:
            icon = {"critical": "🚨", "error": "❌", "warning": "⚠️", "info": "ℹ️"}[msg.severity.value]
            lines.append(f"  {icon} {msg.code}: {msg.message}")
    
    return "\n".join(lines)