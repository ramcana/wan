"""
WAN22 Configuration Migration System

This module handles migration of configuration files between different versions,
ensuring backward compatibility and smooth upgrades.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
from dataclasses import asdict
from datetime import datetime

from wan22_config_manager import WAN22Config, ConfigurationManager

logger = logging.getLogger(__name__)


class ConfigurationMigration:
    """Handles configuration migration between versions"""
    
    # Version migration chain
    MIGRATION_CHAIN = [
        "0.1.0",
        "0.2.0", 
        "0.3.0",
        "0.4.0",
        "0.5.0",
        "1.0.0"
    ]
    
    def __init__(self):
        """Initialize migration system"""
        self.migrations: Dict[str, Callable] = {
            "0.1.0": self._migrate_to_0_1_0,
            "0.2.0": self._migrate_to_0_2_0,
            "0.3.0": self._migrate_to_0_3_0,
            "0.4.0": self._migrate_to_0_4_0,
            "0.5.0": self._migrate_to_0_5_0,
            "1.0.0": self._migrate_to_1_0_0
        }
    
    def migrate_config(self, config_data: Dict[str, Any], target_version: str = "1.0.0") -> Dict[str, Any]:
        """Migrate configuration data to target version
        
        Args:
            config_data: Configuration data to migrate
            target_version: Target version to migrate to
            
        Returns:
            Migrated configuration data
        """
        current_version = config_data.get("version", "0.0.0")
        
        if current_version == target_version:
            logger.info(f"Configuration already at target version {target_version}")
            return config_data
        
        logger.info(f"Migrating configuration from {current_version} to {target_version}")
        
        # Find migration path
        migration_path = self._get_migration_path(current_version, target_version)
        if not migration_path:
            logger.warning(f"No migration path found from {current_version} to {target_version}")
            return config_data
        
        # Apply migrations in sequence
        migrated_data = config_data.copy()
        for version in migration_path:
            if version in self.migrations:
                logger.info(f"Applying migration to {version}")
                migrated_data = self.migrations[version](migrated_data)
                migrated_data["version"] = version
                migrated_data["updated_at"] = datetime.now().isoformat()
                logger.debug(f"After migration to {version}, sections: {list(migrated_data.keys())}")
        
        return migrated_data
    
    def backup_config(self, config_path: str) -> str:
        """Create backup of configuration before migration
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Path to backup file
        """
        config_file = Path(config_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = config_file.parent / f"{config_file.stem}_backup_{timestamp}.json"
        
        try:
            if config_file.exists():
                backup_path.write_text(config_file.read_text())
                logger.info(f"Configuration backed up to {backup_path}")
            return str(backup_path)
        except Exception as e:
            logger.error(f"Failed to backup configuration: {e}")
            raise
    
    def validate_migration(self, original_data: Dict[str, Any], migrated_data: Dict[str, Any]) -> List[str]:
        """Validate migration results
        
        Args:
            original_data: Original configuration data
            migrated_data: Migrated configuration data
            
        Returns:
            List of validation warnings/errors
        """
        warnings = []
        
        # Check version was updated
        original_version = original_data.get("version", "0.0.0")
        migrated_version = migrated_data.get("version", "0.0.0")
        
        if original_version == migrated_version:
            warnings.append("Version was not updated during migration")
        
        # Check essential sections exist
        essential_sections = ["optimization", "pipeline", "security", "compatibility", "user_preferences"]
        for section in essential_sections:
            if section not in migrated_data:
                warnings.append(f"Essential section '{section}' missing after migration")
        
        # Check for data loss (simplified check)
        original_keys = set(self._flatten_dict(original_data).keys())
        migrated_keys = set(self._flatten_dict(migrated_data).keys())
        
        lost_keys = original_keys - migrated_keys
        if lost_keys:
            warnings.append(f"Potential data loss detected: {list(lost_keys)[:5]}...")
        
        return warnings
    
    def _get_migration_path(self, from_version: str, to_version: str) -> List[str]:
        """Get migration path between versions
        
        Args:
            from_version: Starting version
            to_version: Target version
            
        Returns:
            List of versions to migrate through
        """
        try:
            from_idx = self.MIGRATION_CHAIN.index(from_version)
            to_idx = self.MIGRATION_CHAIN.index(to_version)
            
            if from_idx >= to_idx:
                return []
            
            return self.MIGRATION_CHAIN[from_idx + 1:to_idx + 1]
        except ValueError:
            # If from_version is not in chain, try to find a suitable starting point
            if from_version not in self.MIGRATION_CHAIN and to_version in self.MIGRATION_CHAIN:
                # For unknown versions, start from the beginning of the chain
                to_idx = self.MIGRATION_CHAIN.index(to_version)
                return self.MIGRATION_CHAIN[:to_idx + 1]
            
            logger.error(f"Unknown version in migration path: {from_version} -> {to_version}")
            return []
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
        """Flatten nested dictionary for comparison"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    # Migration functions for each version
    
    def _migrate_to_0_1_0(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate to version 0.1.0 - Initial structure"""
        migrated = data.copy()
        
        # Add basic structure if missing, preserving existing settings
        if "optimization" not in migrated:
            migrated["optimization"] = {
                "strategy": "auto",
                "enable_mixed_precision": True,
                "enable_cpu_offload": False
            }
        else:
            # Ensure required fields exist while preserving custom ones
            opt = migrated["optimization"]
            if "strategy" not in opt:
                opt["strategy"] = "auto"
            if "enable_mixed_precision" not in opt:
                opt["enable_mixed_precision"] = True
            if "enable_cpu_offload" not in opt:
                opt["enable_cpu_offload"] = False
        
        if "pipeline" not in migrated:
            migrated["pipeline"] = {
                "selection_mode": "auto",
                "enable_fallback": True
            }
        else:
            # Ensure required fields exist while preserving custom ones
            pipe = migrated["pipeline"]
            if "selection_mode" not in pipe:
                pipe["selection_mode"] = "auto"
            if "enable_fallback" not in pipe:
                pipe["enable_fallback"] = True
        
        return migrated
    
    def _migrate_to_0_2_0(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate to version 0.2.0 - Add security settings"""
        migrated = data.copy()
        
        # Ensure basic structure exists (in case 0.1.0 was skipped)
        if "optimization" not in migrated:
            migrated["optimization"] = {
                "strategy": "auto",
                "enable_mixed_precision": True,
                "enable_cpu_offload": False
            }
        
        if "pipeline" not in migrated:
            migrated["pipeline"] = {
                "selection_mode": "auto",
                "enable_fallback": True
            }
        
        # Add security section
        if "security" not in migrated:
            migrated["security"] = {
                "security_level": "moderate",
                "trust_remote_code": True,
                "trusted_sources": ["huggingface.co", "hf.co"]
            }
        
        # Migrate old trust_remote_code setting if exists
        if "trust_remote_code" in migrated:
            migrated["security"]["trust_remote_code"] = migrated.pop("trust_remote_code")
        
        return migrated
    
    def _migrate_to_0_3_0(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate to version 0.3.0 - Add compatibility settings"""
        migrated = data.copy()
        
        # Add compatibility section
        if "compatibility" not in migrated:
            migrated["compatibility"] = {
                "enable_architecture_detection": True,
                "enable_vae_validation": True,
                "enable_component_validation": True,
                "strict_validation": False
            }
        
        # Migrate old validation settings
        if "enable_validation" in migrated:
            migrated["compatibility"]["enable_component_validation"] = migrated.pop("enable_validation")
        
        return migrated
    
    def _migrate_to_0_4_0(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate to version 0.4.0 - Add user preferences"""
        migrated = data.copy()
        
        # Add user preferences section
        if "user_preferences" not in migrated:
            migrated["user_preferences"] = {
                "default_output_format": "mp4",
                "preferred_video_codec": "h264",
                "default_fps": 24.0,
                "enable_progress_indicators": True,
                "verbose_logging": False
            }
        
        # Migrate old user settings
        if "output_format" in migrated:
            migrated["user_preferences"]["default_output_format"] = migrated.pop("output_format")
        
        if "verbose" in migrated:
            migrated["user_preferences"]["verbose_logging"] = migrated.pop("verbose")
        
        return migrated
    
    def _migrate_to_0_5_0(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate to version 0.5.0 - Enhanced optimization settings"""
        migrated = data.copy()
        
        # Enhance optimization section
        if "optimization" in migrated:
            opt = migrated["optimization"]
            
            # Add new optimization settings
            if "enable_chunked_processing" not in opt:
                opt["enable_chunked_processing"] = False
            
            if "max_chunk_size" not in opt:
                opt["max_chunk_size"] = 8
            
            if "vram_threshold_mb" not in opt:
                opt["vram_threshold_mb"] = 8192
            
            if "enable_vae_tiling" not in opt:
                opt["enable_vae_tiling"] = False
            
            if "vae_tile_size" not in opt:
                opt["vae_tile_size"] = 512
            
            if "custom_optimizations" not in opt:
                opt["custom_optimizations"] = {}
        
        # Enhance pipeline section
        if "pipeline" in migrated:
            pipe = migrated["pipeline"]
            
            if "pipeline_timeout_seconds" not in pipe:
                pipe["pipeline_timeout_seconds"] = 300
            
            if "max_retry_attempts" not in pipe:
                pipe["max_retry_attempts"] = 3
            
            if "custom_pipeline_paths" not in pipe:
                pipe["custom_pipeline_paths"] = {}
        
        return migrated
    
    def _migrate_to_1_0_0(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate to version 1.0.0 - Final structure"""
        migrated = data.copy()
        
        # Add experimental features section
        if "experimental_features" not in migrated:
            migrated["experimental_features"] = {}
        
        # Add custom settings section
        if "custom_settings" not in migrated:
            migrated["custom_settings"] = {}
        
        # Enhance security section
        if "security" in migrated:
            sec = migrated["security"]
            
            if "enable_sandboxing" not in sec:
                sec["enable_sandboxing"] = False
            
            if "sandbox_timeout_seconds" not in sec:
                sec["sandbox_timeout_seconds"] = 60
            
            if "allow_local_code_execution" not in sec:
                sec["allow_local_code_execution"] = True
            
            if "code_signature_verification" not in sec:
                sec["code_signature_verification"] = False
        
        # Enhance compatibility section
        if "compatibility" in migrated:
            comp = migrated["compatibility"]
            
            if "cache_detection_results" not in comp:
                comp["cache_detection_results"] = True
            
            if "detection_cache_ttl_hours" not in comp:
                comp["detection_cache_ttl_hours"] = 24
            
            if "enable_diagnostic_collection" not in comp:
                comp["enable_diagnostic_collection"] = True
            
            if "diagnostic_output_dir" not in comp:
                comp["diagnostic_output_dir"] = "diagnostics"
        
        # Enhance user preferences section
        if "user_preferences" in migrated:
            prefs = migrated["user_preferences"]
            
            if "auto_cleanup_temp_files" not in prefs:
                prefs["auto_cleanup_temp_files"] = True
            
            if "max_concurrent_generations" not in prefs:
                prefs["max_concurrent_generations"] = 1
            
            if "notification_preferences" not in prefs:
                prefs["notification_preferences"] = {
                    "generation_complete": True,
                    "error_notifications": True,
                    "optimization_suggestions": True
                }
        
        # Add timestamps if missing
        if "created_at" not in migrated:
            migrated["created_at"] = datetime.now().isoformat()
        
        if "updated_at" not in migrated:
            migrated["updated_at"] = datetime.now().isoformat()
        
        return migrated


class MigrationManager:
    """Manages configuration migrations for ConfigurationManager"""
    
    def __init__(self, config_manager: ConfigurationManager):
        """Initialize migration manager
        
        Args:
            config_manager: ConfigurationManager instance to manage
        """
        self.config_manager = config_manager
        self.migration = ConfigurationMigration()
    
    def migrate_if_needed(self, target_version: str = "1.0.0") -> bool:
        """Migrate configuration if needed
        
        Args:
            target_version: Target version to migrate to
            
        Returns:
            True if migration was performed, False if not needed
        """
        config_path = str(self.config_manager.config_path)
        
        if not Path(config_path).exists():
            logger.info("No configuration file exists, no migration needed")
            return False
        
        try:
            # Load current configuration
            with open(config_path, 'r', encoding='utf-8') as f:
                current_data = json.load(f)
            
            current_version = current_data.get("version", "0.0.0")
            
            if current_version == target_version:
                logger.info(f"Configuration already at target version {target_version}")
                return False
            
            logger.info(f"Migration needed from {current_version} to {target_version}")
            
            # Create backup
            backup_path = self.migration.backup_config(config_path)
            
            # Perform migration
            migrated_data = self.migration.migrate_config(current_data, target_version)
            
            # Validate migration
            warnings = self.migration.validate_migration(current_data, migrated_data)
            if warnings:
                logger.warning(f"Migration warnings: {warnings}")
            
            # Save migrated configuration
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(migrated_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration migrated successfully to {target_version}")
            logger.info(f"Backup saved to {backup_path}")
            
            # Reload configuration in manager
            self.config_manager._config = None
            self.config_manager.load_config()
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration migration failed: {e}")
            raise
    
    def rollback_migration(self, backup_path: str) -> bool:
        """Rollback migration using backup file
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if rollback successful, False otherwise
        """
        try:
            backup_file = Path(backup_path)
            config_path = self.config_manager.config_path
            
            if not backup_file.exists():
                logger.error(f"Backup file not found: {backup_path}")
                return False
            
            # Restore from backup
            config_path.write_text(backup_file.read_text())
            
            # Reload configuration
            self.config_manager._config = None
            self.config_manager.load_config()
            
            logger.info(f"Configuration rolled back from {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Configuration rollback failed: {e}")
            return False
    
    def get_migration_info(self) -> Dict[str, Any]:
        """Get information about current configuration and available migrations
        
        Returns:
            Dictionary with migration information
        """
        config_path = str(self.config_manager.config_path)
        
        info = {
            "config_exists": Path(config_path).exists(),
            "current_version": "0.0.0",
            "target_version": "1.0.0",
            "migration_needed": False,
            "migration_path": [],
            "backup_files": []
        }
        
        try:
            if info["config_exists"]:
                with open(config_path, 'r', encoding='utf-8') as f:
                    current_data = json.load(f)
                
                info["current_version"] = current_data.get("version", "0.0.0")
                info["migration_needed"] = info["current_version"] != info["target_version"]
                
                if info["migration_needed"]:
                    info["migration_path"] = self.migration._get_migration_path(
                        info["current_version"], 
                        info["target_version"]
                    )
            
            # Find backup files
            config_dir = Path(config_path).parent
            backup_pattern = f"{Path(config_path).stem}_backup_*.json"
            info["backup_files"] = [str(f) for f in config_dir.glob(backup_pattern)]
            
        except Exception as e:
            logger.error(f"Failed to get migration info: {e}")
            info["error"] = str(e)
        
        return info


def migrate_configuration(config_manager: ConfigurationManager, target_version: str = "1.0.0") -> bool:
    """Convenience function to migrate configuration
    
    Args:
        config_manager: ConfigurationManager instance
        target_version: Target version to migrate to
        
    Returns:
        True if migration was performed, False if not needed
    """
    migration_manager = MigrationManager(config_manager)
    return migration_manager.migrate_if_needed(target_version)