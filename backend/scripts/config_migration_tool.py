#!/usr/bin/env python3
"""
Configuration Migration Tool for Enhanced Model Management

This tool helps migrate existing configurations to the new enhanced model
management configuration format, with validation and backup capabilities.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.enhanced_model_config import (
    ConfigurationManager, EnhancedModelConfiguration,
    UserPreferences, AdminPolicies, FeatureFlagConfig
)
from core.config_validation import ConfigurationValidator, ValidationResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigurationMigrationTool:
    """Tool for migrating and managing enhanced model configurations"""
    
    def __init__(self):
        self.validator = ConfigurationValidator()
    
    def migrate_configuration(self, source_path: str, target_path: str, backup: bool = True) -> bool:
        """
        Migrate configuration from source to target path
        
        Args:
            source_path: Path to source configuration file
            target_path: Path to target configuration file
            backup: Whether to create backup of existing target
            
        Returns:
            True if migration successful, False otherwise
        """
        try:
            source_file = Path(source_path)
            target_file = Path(target_path)
            
            if not source_file.exists():
                logger.error(f"Source configuration file not found: {source_path}")
                return False
            
            # Create backup if requested and target exists
            if backup and target_file.exists():
                backup_path = target_file.with_suffix(f'.backup.{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
                target_file.rename(backup_path)
                logger.info(f"Created backup: {backup_path}")
            
            # Load source configuration
            with open(source_file, 'r') as f:
                source_config = json.load(f)
            
            # Detect configuration format and migrate
            migrated_config = self._detect_and_migrate(source_config)
            
            # Validate migrated configuration
            validation_result = self._validate_migrated_config(migrated_config)
            
            if not validation_result.is_valid:
                logger.error("Migrated configuration is invalid:")
                for error in validation_result.errors:
                    logger.error(f"  - {error.field}: {error.message}")
                return False
            
            # Save migrated configuration
            config_manager = ConfigurationManager(target_path)
            config_manager.config = migrated_config
            
            if config_manager.save_configuration():
                logger.info(f"Configuration migrated successfully to: {target_path}")
                
                # Report warnings if any
                if validation_result.warnings:
                    logger.warning("Migration completed with warnings:")
                    for warning in validation_result.warnings:
                        logger.warning(f"  - {warning.field}: {warning.message}")
                
                return True
            else:
                logger.error("Failed to save migrated configuration")
                return False
                
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False
    
    def validate_configuration(self, config_path: str) -> bool:
        """
        Validate existing configuration file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            config_file = Path(config_path)
            
            if not config_file.exists():
                logger.error(f"Configuration file not found: {config_path}")
                return False
            
            # Load configuration using manager (handles migration automatically)
            config_manager = ConfigurationManager(config_path)
            
            # Validate user preferences
            prefs_result = config_manager.validate_user_preferences(
                config_manager.get_user_preferences()
            )
            
            # Validate admin policies
            policies_result = config_manager.validate_admin_policies(
                config_manager.get_admin_policies()
            )
            
            # Report results
            total_errors = len(prefs_result.errors) + len(policies_result.errors)
            total_warnings = len(prefs_result.warnings) + len(policies_result.warnings)
            
            if total_errors == 0:
                logger.info("Configuration validation passed")
                if total_warnings > 0:
                    logger.warning(f"Configuration has {total_warnings} warnings:")
                    for warning in prefs_result.warnings + policies_result.warnings:
                        logger.warning(f"  - {warning.field}: {warning.message}")
                return True
            else:
                logger.error(f"Configuration validation failed with {total_errors} errors:")
                for error in prefs_result.errors + policies_result.errors:
                    logger.error(f"  - {error.field}: {error.message}")
                
                if total_warnings > 0:
                    logger.warning(f"Configuration also has {total_warnings} warnings:")
                    for warning in prefs_result.warnings + policies_result.warnings:
                        logger.warning(f"  - {warning.field}: {warning.message}")
                
                return False
                
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False
    
    def create_default_configuration(self, config_path: str) -> bool:
        """
        Create default configuration file
        
        Args:
            config_path: Path where to create configuration file
            
        Returns:
            True if creation successful, False otherwise
        """
        try:
            config_file = Path(config_path)
            
            if config_file.exists():
                logger.error(f"Configuration file already exists: {config_path}")
                return False
            
            # Create configuration manager (will create default config)
            config_manager = ConfigurationManager(config_path)
            
            logger.info(f"Default configuration created at: {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create default configuration: {e}")
            return False
    
    def _detect_and_migrate(self, config_data: Dict[str, Any]) -> EnhancedModelConfiguration:
        """Detect configuration format and migrate to current version"""
        
        # Check if it's already enhanced model configuration
        if 'config_schema_version' in config_data:
            logger.info("Configuration is already in enhanced format")
            return self._load_enhanced_config(config_data)
        
        # Check if it's legacy model configuration
        elif 'model_config' in config_data or 'models' in config_data:
            logger.info("Detected legacy model configuration, migrating...")
            return self._migrate_from_legacy_model_config(config_data)
        
        # Check if it's basic application configuration
        elif 'download_settings' in config_data or 'user_preferences' in config_data:
            logger.info("Detected basic application configuration, migrating...")
            return self._migrate_from_basic_config(config_data)
        
        # Unknown format, create default with available data
        else:
            logger.warning("Unknown configuration format, creating default configuration")
            return self._create_default_with_data(config_data)
    
    def _load_enhanced_config(self, config_data: Dict[str, Any]) -> EnhancedModelConfiguration:
        """Load existing enhanced configuration"""
        # Use configuration manager to handle deserialization
        temp_manager = ConfigurationManager()
        return temp_manager._deserialize_config(config_data)
    
    def _migrate_from_legacy_model_config(self, config_data: Dict[str, Any]) -> EnhancedModelConfiguration:
        """Migrate from legacy model configuration format"""
        config = EnhancedModelConfiguration()
        
        # Extract download settings
        if 'download_settings' in config_data:
            download_settings = config_data['download_settings']
            config.user_preferences.download_config.max_retries = download_settings.get('max_retries', 3)
            config.user_preferences.download_config.bandwidth_limit_mbps = download_settings.get('bandwidth_limit')
        
        # Extract model preferences
        if 'preferred_models' in config_data:
            config.user_preferences.preferred_models = config_data['preferred_models']
        
        if 'blocked_models' in config_data:
            config.user_preferences.blocked_models = config_data['blocked_models']
        
        # Extract storage settings
        if 'storage_settings' in config_data:
            storage_settings = config_data['storage_settings']
            config.user_preferences.storage_config.max_storage_gb = storage_settings.get('max_storage_gb')
            config.user_preferences.storage_config.cleanup_threshold_percent = storage_settings.get('cleanup_threshold', 85.0)
        
        return config
    
    def _migrate_from_basic_config(self, config_data: Dict[str, Any]) -> EnhancedModelConfiguration:
        """Migrate from basic application configuration"""
        config = EnhancedModelConfiguration()
        
        # Extract user preferences if available
        if 'user_preferences' in config_data:
            user_prefs = config_data['user_preferences']
            
            # Map automation level
            if 'automation_level' in user_prefs:
                from core.enhanced_model_config import AutomationLevel
                level_map = {
                    'manual': AutomationLevel.MANUAL,
                    'semi_automatic': AutomationLevel.SEMI_AUTOMATIC,
                    'automatic': AutomationLevel.FULLY_AUTOMATIC,
                    'full': AutomationLevel.FULLY_AUTOMATIC
                }
                config.user_preferences.automation_level = level_map.get(
                    user_prefs['automation_level'], 
                    AutomationLevel.SEMI_AUTOMATIC
                )
        
        # Extract admin settings if available
        if 'admin_settings' in config_data:
            admin_settings = config_data['admin_settings']
            config.admin_policies.max_user_storage_gb = admin_settings.get('max_user_storage_gb')
            config.admin_policies.require_approval_for_updates = admin_settings.get('require_approval', False)
        
        return config
    
    def _create_default_with_data(self, config_data: Dict[str, Any]) -> EnhancedModelConfiguration:
        """Create default configuration with any available data"""
        config = EnhancedModelConfiguration()
        
        # Try to extract any useful settings
        if 'max_retries' in config_data:
            config.user_preferences.download_config.max_retries = config_data['max_retries']
        
        if 'bandwidth_limit' in config_data:
            config.user_preferences.download_config.bandwidth_limit_mbps = config_data['bandwidth_limit']
        
        return config
    
    def _validate_migrated_config(self, config: EnhancedModelConfiguration) -> ValidationResult:
        """Validate migrated configuration"""
        # Validate user preferences
        prefs_result = self.validator.validate_user_preferences(config.user_preferences)
        
        # Validate admin policies
        policies_result = self.validator.validate_admin_policies(config.admin_policies)
        
        # Validate feature flags
        flags_result = self.validator.validate_feature_flags(config.feature_flags)
        
        # Combine results
        all_errors = prefs_result.errors + policies_result.errors + flags_result.errors
        all_warnings = prefs_result.warnings + policies_result.warnings + flags_result.warnings
        
        return ValidationResult(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings
        )


def main():
    """Main CLI interface for configuration migration tool"""
    parser = argparse.ArgumentParser(
        description="Enhanced Model Configuration Migration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Migrate existing configuration
  python config_migration_tool.py migrate old_config.json new_config.json
  
  # Validate configuration
  python config_migration_tool.py validate config.json
  
  # Create default configuration
  python config_migration_tool.py create-default config.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Migrate command
    migrate_parser = subparsers.add_parser('migrate', help='Migrate configuration file')
    migrate_parser.add_argument('source', help='Source configuration file')
    migrate_parser.add_argument('target', help='Target configuration file')
    migrate_parser.add_argument('--no-backup', action='store_true', help='Skip backup creation')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate configuration file')
    validate_parser.add_argument('config', help='Configuration file to validate')
    
    # Create default command
    create_parser = subparsers.add_parser('create-default', help='Create default configuration')
    create_parser.add_argument('config', help='Path for new configuration file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    tool = ConfigurationMigrationTool()
    
    if args.command == 'migrate':
        success = tool.migrate_configuration(
            args.source, 
            args.target, 
            backup=not args.no_backup
        )
        return 0 if success else 1
    
    elif args.command == 'validate':
        success = tool.validate_configuration(args.config)
        return 0 if success else 1
    
    elif args.command == 'create-default':
        success = tool.create_default_configuration(args.config)
        return 0 if success else 1
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())