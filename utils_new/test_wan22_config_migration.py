"""
Tests for WAN22 Configuration Migration System

This module tests the configuration migration functionality for handling
version upgrades and backward compatibility.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from wan22_config_migration import (
    ConfigurationMigration,
    MigrationManager,
    migrate_configuration
)
from wan22_config_manager import ConfigurationManager, WAN22Config


class TestConfigurationMigration:
    """Test configuration migration functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.migration = ConfigurationMigration()
    
    def test_migration_chain(self):
        """Test migration chain is properly defined"""
        assert len(self.migration.MIGRATION_CHAIN) > 0
        assert "1.0.0" in self.migration.MIGRATION_CHAIN
        
        # Check chain is ordered
        for i in range(len(self.migration.MIGRATION_CHAIN) - 1):
            current = self.migration.MIGRATION_CHAIN[i]
            next_version = self.migration.MIGRATION_CHAIN[i + 1]
            assert current < next_version
    
    def test_get_migration_path(self):
        """Test getting migration path between versions"""
        # Normal migration path
        path = self.migration._get_migration_path("0.1.0", "0.3.0")
        assert path == ["0.2.0", "0.3.0"]
        
        # Same version
        path = self.migration._get_migration_path("0.2.0", "0.2.0")
        assert path == []
        
        # Backward migration (not supported)
        path = self.migration._get_migration_path("0.3.0", "0.1.0")
        assert path == []
        
        # Unknown version - should migrate from beginning
        path = self.migration._get_migration_path("0.0.1", "1.0.0")
        assert len(path) > 0
        assert "1.0.0" in path
    
    def test_migrate_config_same_version(self):
        """Test migration when already at target version"""
        data = {"version": "1.0.0", "test": "value"}
        result = self.migration.migrate_config(data, "1.0.0")
        
        assert result == data
    
    def test_migrate_config_unknown_version(self):
        """Test migration with unknown starting version"""
        data = {"version": "0.0.1", "test": "value"}
        result = self.migration.migrate_config(data, "1.0.0")
        
        # Should migrate to target version
        assert result["version"] == "1.0.0"
        # Should preserve custom data
        assert result["test"] == "value"
    
    def test_migrate_to_0_1_0(self):
        """Test migration to version 0.1.0"""
        data = {"version": "0.0.0"}
        result = self.migration._migrate_to_0_1_0(data)
        
        assert "optimization" in result
        assert "pipeline" in result
        assert result["optimization"]["strategy"] == "auto"
        assert result["pipeline"]["selection_mode"] == "auto"
    
    def test_migrate_to_0_2_0(self):
        """Test migration to version 0.2.0"""
        data = {
            "version": "0.1.0",
            "trust_remote_code": False,
            "optimization": {"strategy": "auto"}
        }
        result = self.migration._migrate_to_0_2_0(data)
        
        assert "security" in result
        assert result["security"]["trust_remote_code"] is False
        assert "trust_remote_code" not in result  # Should be moved
    
    def test_migrate_to_0_3_0(self):
        """Test migration to version 0.3.0"""
        data = {
            "version": "0.2.0",
            "enable_validation": True
        }
        result = self.migration._migrate_to_0_3_0(data)
        
        assert "compatibility" in result
        assert result["compatibility"]["enable_component_validation"] is True
        assert "enable_validation" not in result  # Should be moved
    
    def test_migrate_to_0_4_0(self):
        """Test migration to version 0.4.0"""
        data = {
            "version": "0.3.0",
            "output_format": "webm",
            "verbose": True
        }
        result = self.migration._migrate_to_0_4_0(data)
        
        assert "user_preferences" in result
        assert result["user_preferences"]["default_output_format"] == "webm"
        assert result["user_preferences"]["verbose_logging"] is True
        assert "output_format" not in result
        assert "verbose" not in result
    
    def test_migrate_to_0_5_0(self):
        """Test migration to version 0.5.0"""
        data = {
            "version": "0.4.0",
            "optimization": {"strategy": "auto"},
            "pipeline": {"selection_mode": "auto"}
        }
        result = self.migration._migrate_to_0_5_0(data)
        
        # Check new optimization settings
        opt = result["optimization"]
        assert "enable_chunked_processing" in opt
        assert "max_chunk_size" in opt
        assert "vram_threshold_mb" in opt
        assert "enable_vae_tiling" in opt
        
        # Check new pipeline settings
        pipe = result["pipeline"]
        assert "pipeline_timeout_seconds" in pipe
        assert "max_retry_attempts" in pipe
        assert "custom_pipeline_paths" in pipe
    
    def test_migrate_to_1_0_0(self):
        """Test migration to version 1.0.0"""
        data = {
            "version": "0.5.0",
            "security": {"security_level": "moderate"},
            "compatibility": {"enable_architecture_detection": True},
            "user_preferences": {"default_fps": 24.0}
        }
        result = self.migration._migrate_to_1_0_0(data)
        
        # Check new top-level sections
        assert "experimental_features" in result
        assert "custom_settings" in result
        
        # Check enhanced security settings
        sec = result["security"]
        assert "enable_sandboxing" in sec
        assert "sandbox_timeout_seconds" in sec
        assert "allow_local_code_execution" in sec
        assert "code_signature_verification" in sec
        
        # Check enhanced compatibility settings
        comp = result["compatibility"]
        assert "cache_detection_results" in comp
        assert "detection_cache_ttl_hours" in comp
        assert "enable_diagnostic_collection" in comp
        assert "diagnostic_output_dir" in comp
        
        # Check enhanced user preferences
        prefs = result["user_preferences"]
        assert "auto_cleanup_temp_files" in prefs
        assert "max_concurrent_generations" in prefs
        assert "notification_preferences" in prefs
        
        # Check timestamps
        assert "created_at" in result
        assert "updated_at" in result
    
    def test_full_migration_chain(self):
        """Test complete migration from earliest to latest version"""
        data = {"version": "0.1.0"}
        result = self.migration.migrate_config(data, "1.0.0")
        
        assert result["version"] == "1.0.0"
        assert "optimization" in result
        assert "pipeline" in result
        assert "security" in result
        assert "compatibility" in result
        assert "user_preferences" in result
        assert "experimental_features" in result
        assert "custom_settings" in result
    
    def test_validate_migration(self):
        """Test migration validation"""
        original = {
            "version": "0.1.0",
            "optimization": {"strategy": "auto"},
            "test_setting": "value"
        }
        
        migrated = {
            "version": "1.0.0",
            "optimization": {"strategy": "auto", "enable_mixed_precision": True},
            "pipeline": {"selection_mode": "auto"},
            "security": {"security_level": "moderate"},
            "compatibility": {"enable_architecture_detection": True},
            "user_preferences": {"default_fps": 24.0}
        }
        
        warnings = self.migration.validate_migration(original, migrated)
        
        # Should warn about potential data loss (test_setting)
        assert len(warnings) > 0
        assert any("data loss" in warning.lower() for warning in warnings)
    
    def test_flatten_dict(self):
        """Test dictionary flattening utility"""
        nested = {
            "a": 1,
            "b": {
                "c": 2,
                "d": {
                    "e": 3
                }
            }
        }
        
        flattened = self.migration._flatten_dict(nested)
        
        assert flattened["a"] == 1
        assert flattened["b.c"] == 2
        assert flattened["b.d.e"] == 3


class TestMigrationManager:
    """Test migration manager functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigurationManager(self.temp_dir)
        self.migration_manager = MigrationManager(self.config_manager)
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_migrate_if_needed_no_config(self):
        """Test migration when no config file exists"""
        result = self.migration_manager.migrate_if_needed()
        assert result is False
    
    def test_migrate_if_needed_current_version(self):
        """Test migration when already at current version"""
        # Create current version config
        config = WAN22Config()
        self.config_manager.save_config(config)
        
        result = self.migration_manager.migrate_if_needed("1.0.0")
        assert result is False
    
    def test_migrate_if_needed_old_version(self):
        """Test migration from old version"""
        # Create old version config
        old_config = {
            "version": "0.1.0",
            "optimization": {"strategy": "auto"}
        }
        
        with open(self.config_manager.config_path, 'w') as f:
            json.dump(old_config, f)
        
        result = self.migration_manager.migrate_if_needed("1.0.0")
        assert result is True
        
        # Verify migration
        config = self.config_manager.get_config()
        assert config.version == "1.0.0"
        assert hasattr(config, 'security')
        assert hasattr(config, 'compatibility')
        assert hasattr(config, 'user_preferences')
    
    def test_backup_config(self):
        """Test configuration backup"""
        # Create config file
        config = WAN22Config()
        self.config_manager.save_config(config)
        
        backup_path = self.migration_manager.migration.backup_config(
            str(self.config_manager.config_path)
        )
        
        assert Path(backup_path).exists()
        assert "backup" in backup_path
        
        # Verify backup content
        with open(backup_path, 'r') as f:
            backup_data = json.load(f)
        assert backup_data["version"] == "1.0.0"
    
    def test_rollback_migration(self):
        """Test migration rollback"""
        # Create original config
        original_config = {
            "version": "0.1.0",
            "test_setting": "original_value"
        }
        
        with open(self.config_manager.config_path, 'w') as f:
            json.dump(original_config, f)
        
        # Create backup
        backup_path = self.migration_manager.migration.backup_config(
            str(self.config_manager.config_path)
        )
        
        # Perform migration
        self.migration_manager.migrate_if_needed("1.0.0")
        
        # Verify migration occurred
        config = self.config_manager.get_config()
        assert config.version == "1.0.0"
        
        # Rollback
        success = self.migration_manager.rollback_migration(backup_path)
        assert success is True
        
        # Verify rollback
        with open(self.config_manager.config_path, 'r') as f:
            rolled_back = json.load(f)
        assert rolled_back["version"] == "0.1.0"
        assert rolled_back["test_setting"] == "original_value"
    
    def test_rollback_missing_backup(self):
        """Test rollback with missing backup file"""
        success = self.migration_manager.rollback_migration("nonexistent_backup.json")
        assert success is False
    
    def test_get_migration_info_no_config(self):
        """Test getting migration info when no config exists"""
        info = self.migration_manager.get_migration_info()
        
        assert info["config_exists"] is False
        assert info["current_version"] == "0.0.0"
        assert info["target_version"] == "1.0.0"
        assert info["migration_needed"] is False
        assert info["migration_path"] == []
        assert len(info["backup_files"]) == 0
    
    def test_get_migration_info_with_config(self):
        """Test getting migration info with existing config"""
        # Create old version config
        old_config = {
            "version": "0.2.0",
            "optimization": {"strategy": "auto"}
        }
        
        with open(self.config_manager.config_path, 'w') as f:
            json.dump(old_config, f)
        
        info = self.migration_manager.get_migration_info()
        
        assert info["config_exists"] is True
        assert info["current_version"] == "0.2.0"
        assert info["target_version"] == "1.0.0"
        assert info["migration_needed"] is True
        assert len(info["migration_path"]) > 0
        assert "0.3.0" in info["migration_path"]
        assert "1.0.0" in info["migration_path"]
    
    def test_get_migration_info_with_backups(self):
        """Test getting migration info with backup files"""
        # Create config and backup
        config = WAN22Config()
        self.config_manager.save_config(config)
        
        backup_path = self.migration_manager.migration.backup_config(
            str(self.config_manager.config_path)
        )
        
        info = self.migration_manager.get_migration_info()
        
        assert len(info["backup_files"]) == 1
        assert backup_path in info["backup_files"]


class TestMigrationIntegration:
    """Integration tests for migration system"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigurationManager(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_migrate_configuration_function(self):
        """Test convenience migration function"""
        # Create old version config
        old_config = {
            "version": "0.1.0",
            "optimization": {"strategy": "memory"}
        }
        
        with open(self.config_manager.config_path, 'w') as f:
            json.dump(old_config, f)
        
        # Migrate using convenience function
        result = migrate_configuration(self.config_manager, "1.0.0")
        assert result is True
        
        # Verify migration
        config = self.config_manager.get_config()
        assert config.version == "1.0.0"
        assert config.optimization.strategy.value == "memory"  # Should preserve custom value
    
    def test_complete_migration_workflow(self):
        """Test complete migration workflow"""
        # 1. Create old configuration
        old_config = {
            "version": "0.1.0",
            "optimization": {"strategy": "performance"},
            "trust_remote_code": False,
            "output_format": "webm",
            "verbose": True
        }
        
        with open(self.config_manager.config_path, 'w') as f:
            json.dump(old_config, f)
        
        migration_manager = MigrationManager(self.config_manager)
        
        # 2. Get migration info
        info = migration_manager.get_migration_info()
        assert info["migration_needed"] is True
        assert info["current_version"] == "0.1.0"
        
        # 3. Perform migration
        result = migration_manager.migrate_if_needed("1.0.0")
        assert result is True
        
        # 4. Verify migration results
        config = self.config_manager.get_config()
        assert config.version == "1.0.0"
        
        # Check migrated values
        assert config.optimization.strategy.value == "performance"
        assert config.security.trust_remote_code is False
        assert config.user_preferences.default_output_format == "webm"
        assert config.user_preferences.verbose_logging is True
        
        # Check new sections exist
        assert hasattr(config, 'compatibility')
        assert hasattr(config, 'experimental_features')
        assert hasattr(config, 'custom_settings')
        
        # 5. Verify backup was created
        info = migration_manager.get_migration_info()
        assert len(info["backup_files"]) == 1
    
    def test_migration_preserves_custom_settings(self):
        """Test that migration preserves custom user settings"""
        # Create config with custom settings
        old_config = {
            "version": "0.3.0",
            "optimization": {
                "strategy": "custom",
                "custom_setting": "user_value"
            },
            "user_preferences": {
                "default_fps": 60.0,
                "custom_preference": "user_choice"
            },
            "custom_section": {
                "user_data": "important_value"
            }
        }
        
        with open(self.config_manager.config_path, 'w') as f:
            json.dump(old_config, f)
        
        # Migrate
        migration_manager = MigrationManager(self.config_manager)
        result = migration_manager.migrate_if_needed("1.0.0")
        assert result is True
        
        # Load and verify custom settings preserved
        with open(self.config_manager.config_path, 'r') as f:
            migrated_data = json.load(f)
        
        assert migrated_data["optimization"]["custom_setting"] == "user_value"
        assert migrated_data["user_preferences"]["default_fps"] == 60.0
        assert migrated_data["user_preferences"]["custom_preference"] == "user_choice"
        assert migrated_data["custom_section"]["user_data"] == "important_value"
    
    def test_migration_error_handling(self):
        """Test migration error handling"""
        # Create corrupted config file
        with open(self.config_manager.config_path, 'w') as f:
            f.write("invalid json content {")
        
        migration_manager = MigrationManager(self.config_manager)
        
        # Should raise exception for corrupted file
        with pytest.raises(Exception):
            migration_manager.migrate_if_needed("1.0.0")


        assert True  # TODO: Add proper assertion

if __name__ == "__main__":
    pytest.main([__file__])
