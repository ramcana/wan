"""
Integration tests for user preference management system.
"""

import json
import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from scripts.startup_manager.preferences import (
    UserPreferences, ConfigurationVersion, PreferenceManager, load_preferences
)
from scripts.startup_manager.config import StartupConfig, ConfigLoader


class TestUserPreferences:
    """Test cases for UserPreferences model."""
    
    def test_default_preferences(self):
        """Test default preference values."""
        prefs = UserPreferences()
        
        assert prefs.auto_open_browser is True
        assert prefs.show_progress_bars is True
        assert prefs.verbose_output is False
        assert prefs.confirm_destructive_actions is True
        assert prefs.preferred_recovery_strategy == "auto"
        assert prefs.auto_retry_failed_operations is True
        assert prefs.max_auto_retries == 3
        assert prefs.preferred_backend_port is None
        assert prefs.preferred_frontend_port is None
        assert prefs.allow_port_auto_increment is True
        assert prefs.allow_admin_elevation is True
        assert prefs.trust_local_processes is False
        assert prefs.keep_detailed_logs is True
        assert prefs.log_retention_days == 30
        assert prefs.enable_experimental_features is False
        assert prefs.startup_timeout_multiplier == 1.0
    
    def test_custom_preferences(self):
        """Test creating preferences with custom values."""
        prefs = UserPreferences(
            auto_open_browser=False,
            verbose_output=True,
            preferred_backend_port=8080,
            preferred_frontend_port=3001,
            max_auto_retries=5,
            startup_timeout_multiplier=1.5
        )
        
        assert prefs.auto_open_browser is False
        assert prefs.verbose_output is True
        assert prefs.preferred_backend_port == 8080
        assert prefs.preferred_frontend_port == 3001
        assert prefs.max_auto_retries == 5
        assert prefs.startup_timeout_multiplier == 1.5
    
    def test_preference_validation(self):
        """Test preference validation constraints."""
        # Valid values
        prefs = UserPreferences(
            preferred_recovery_strategy="manual",
            max_auto_retries=1,
            preferred_backend_port=8080,
            log_retention_days=365,
            startup_timeout_multiplier=0.5
        )
        
        assert prefs.preferred_recovery_strategy == "manual"
        assert prefs.max_auto_retries == 1
        assert prefs.preferred_backend_port == 8080
        assert prefs.log_retention_days == 365
        assert prefs.startup_timeout_multiplier == 0.5
        
        # Invalid values should raise validation errors
        with pytest.raises(ValueError):
            UserPreferences(preferred_recovery_strategy="invalid")
        
        with pytest.raises(ValueError):
            UserPreferences(max_auto_retries=0)
        
        with pytest.raises(ValueError):
            UserPreferences(preferred_backend_port=500)  # Below 1024
        
        with pytest.raises(ValueError):
            UserPreferences(log_retention_days=0)
        
        with pytest.raises(ValueError):
            UserPreferences(startup_timeout_multiplier=0.1)  # Below 0.5


class TestConfigurationVersion:
    """Test cases for ConfigurationVersion model."""
    
    def test_default_version_info(self):
        """Test default version information."""
        version_info = ConfigurationVersion(version="2.0.0")
        
        assert version_info.version == "2.0.0"
        assert version_info.startup_manager_version == "2.0.0"
        assert isinstance(version_info.created_at, datetime)
        assert version_info.migration_notes == []
    
    def test_custom_version_info(self):
        """Test creating version info with custom values."""
        created_time = datetime.now() - timedelta(days=1)
        version_info = ConfigurationVersion(
            version="1.5.0",
            created_at=created_time,
            startup_manager_version="2.0.0",
            migration_notes=["Migrated from 1.0.0", "Updated preferences"]
        )
        
        assert version_info.version == "1.5.0"
        assert version_info.created_at == created_time
        assert version_info.startup_manager_version == "2.0.0"
        assert len(version_info.migration_notes) == 2


class TestPreferenceManager:
    """Test cases for PreferenceManager class."""
    
    def test_load_preferences_creates_default(self):
        """Test that loading preferences creates default file when none exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prefs_dir = Path(temp_dir) / "preferences"
            
            manager = PreferenceManager(prefs_dir)
            preferences = manager.load_preferences()
            
            assert isinstance(preferences, UserPreferences)
            assert (prefs_dir / "preferences.json").exists()
            assert preferences.auto_open_browser is True  # Default value
    
    def test_load_preferences_from_existing_file(self):
        """Test loading preferences from existing file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prefs_dir = Path(temp_dir) / "preferences"
            prefs_dir.mkdir(parents=True)
            
            # Create preferences file
            prefs_data = {
                "auto_open_browser": False,
                "verbose_output": True,
                "preferred_backend_port": 8080,
                "max_auto_retries": 5
            }
            
            prefs_file = prefs_dir / "preferences.json"
            with open(prefs_file, 'w') as f:
                json.dump(prefs_data, f)
            
            manager = PreferenceManager(prefs_dir)
            preferences = manager.load_preferences()
            
            assert preferences.auto_open_browser is False
            assert preferences.verbose_output is True
            assert preferences.preferred_backend_port == 8080
            assert preferences.max_auto_retries == 5
    
    def test_load_preferences_handles_corrupted_file(self):
        """Test that corrupted preferences file is handled gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prefs_dir = Path(temp_dir) / "preferences"
            prefs_dir.mkdir(parents=True)
            
            # Create corrupted preferences file
            prefs_file = prefs_dir / "preferences.json"
            with open(prefs_file, 'w') as f:
                f.write("{ invalid json }")
            
            manager = PreferenceManager(prefs_dir)
            preferences = manager.load_preferences()
            
            # Should create default preferences
            assert isinstance(preferences, UserPreferences)
            assert preferences.auto_open_browser is True  # Default value
            
            # Should backup corrupted file
            backup_dir = prefs_dir / "backups"
            assert backup_dir.exists()
            corrupted_files = list(backup_dir.glob("corrupted_preferences_*.json"))
            assert len(corrupted_files) == 1
    
    def test_save_preferences(self):
        """Test saving preferences to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prefs_dir = Path(temp_dir) / "preferences"
            
            manager = PreferenceManager(prefs_dir)
            preferences = manager.load_preferences()
            
            # Modify preferences
            preferences.auto_open_browser = False
            preferences.verbose_output = True
            preferences.preferred_backend_port = 8080
            
            manager.save_preferences()
            
            # Verify file was saved correctly
            prefs_file = prefs_dir / "preferences.json"
            assert prefs_file.exists()
            
            with open(prefs_file, 'r') as f:
                saved_data = json.load(f)
            
            assert saved_data["auto_open_browser"] is False
            assert saved_data["verbose_output"] is True
            assert saved_data["preferred_backend_port"] == 8080
    
    def test_load_version_info_creates_default(self):
        """Test that loading version info creates default when none exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prefs_dir = Path(temp_dir) / "preferences"
            
            manager = PreferenceManager(prefs_dir)
            version_info = manager.load_version_info()
            
            assert isinstance(version_info, ConfigurationVersion)
            assert version_info.version == "2.0.0"
            assert (prefs_dir / "version.json").exists()
    
    def test_load_version_info_from_existing_file(self):
        """Test loading version info from existing file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prefs_dir = Path(temp_dir) / "preferences"
            prefs_dir.mkdir(parents=True)
            
            # Create version file
            version_data = {
                "version": "1.5.0",
                "created_at": "2024-01-01T12:00:00",
                "startup_manager_version": "2.0.0",
                "migration_notes": ["Test migration"]
            }
            
            version_file = prefs_dir / "version.json"
            with open(version_file, 'w') as f:
                json.dump(version_data, f)
            
            manager = PreferenceManager(prefs_dir)
            version_info = manager.load_version_info()
            
            assert version_info.version == "1.5.0"
            assert version_info.startup_manager_version == "2.0.0"
            assert len(version_info.migration_notes) == 1
    
    def test_create_backup(self):
        """Test creating configuration backup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prefs_dir = Path(temp_dir) / "preferences"
            
            # Create startup config file
            config_file = Path(temp_dir) / "startup_config.json"
            with open(config_file, 'w') as f:
                json.dump({"backend": {"port": 8000}}, f)
            
            # Change to temp directory so startup_config.json is found
            original_cwd = Path.cwd()
            try:
                import os
                os.chdir(temp_dir)
                
                manager = PreferenceManager(prefs_dir)
                manager.load_preferences()
                manager.load_version_info()
                
                backup_path = manager.create_backup("test_backup")
                
                assert backup_path.exists()
                assert (backup_path / "preferences.json").exists()
                assert (backup_path / "version.json").exists()
                assert (backup_path / "startup_config.json").exists()
                assert (backup_path / "manifest.json").exists()
                
                # Verify manifest
                with open(backup_path / "manifest.json", 'r') as f:
                    manifest = json.load(f)
                
                assert "created_at" in manifest
                assert "files" in manifest
                assert len(manifest["files"]) == 3
                
            finally:
                os.chdir(original_cwd)
    
    def test_restore_backup(self):
        """Test restoring configuration from backup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prefs_dir = Path(temp_dir) / "preferences"
            
            # Create startup config file
            config_file = Path(temp_dir) / "startup_config.json"
            with open(config_file, 'w') as f:
                json.dump({"backend": {"port": 8000}}, f)
            
            original_cwd = Path.cwd()
            try:
                import os
                os.chdir(temp_dir)
                
                manager = PreferenceManager(prefs_dir)
                
                # Load and modify preferences
                preferences = manager.load_preferences()
                preferences.auto_open_browser = False
                preferences.verbose_output = True
                manager.save_preferences()
                
                # Create backup
                backup_path = manager.create_backup("test_backup")
                
                # Modify preferences again
                preferences.auto_open_browser = True
                preferences.verbose_output = False
                preferences.preferred_backend_port = 8080
                manager.save_preferences()
                
                # Restore backup
                success = manager.restore_backup("test_backup")
                assert success is True
                
                # Verify restoration
                restored_preferences = manager.load_preferences()
                assert restored_preferences.auto_open_browser is False
                assert restored_preferences.verbose_output is True
                assert restored_preferences.preferred_backend_port is None
                
            finally:
                os.chdir(original_cwd)
    
    def test_list_backups(self):
        """Test listing available backups."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prefs_dir = Path(temp_dir) / "preferences"
            
            manager = PreferenceManager(prefs_dir)
            manager.load_preferences()
            
            # Create multiple backups
            backup1 = manager.create_backup("backup_1")
            backup2 = manager.create_backup("backup_2")
            
            backups = manager.list_backups()
            
            assert len(backups) >= 2
            backup_names = [b["name"] for b in backups]
            assert "backup_1" in backup_names
            assert "backup_2" in backup_names
            
            # Verify backup info structure
            for backup in backups:
                assert "name" in backup
                assert "path" in backup
                assert "created_at" in backup
                assert "description" in backup
                assert "files" in backup
    
    def test_cleanup_old_backups(self):
        """Test cleaning up old backups."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prefs_dir = Path(temp_dir) / "preferences"
            
            manager = PreferenceManager(prefs_dir)
            manager.load_preferences()
            
            # Create multiple backups
            for i in range(5):
                manager.create_backup(f"backup_{i}")
            
            # Verify all backups exist
            backups_before = manager.list_backups()
            assert len(backups_before) == 5
            
            # Clean up, keeping only 3
            removed_count = manager.cleanup_old_backups(keep_count=3)
            assert removed_count == 2
            
            # Verify only 3 backups remain
            backups_after = manager.list_backups()
            assert len(backups_after) == 3
    
    def test_migrate_configuration_no_migration_needed(self):
        """Test migration when no migration is needed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prefs_dir = Path(temp_dir) / "preferences"
            
            manager = PreferenceManager(prefs_dir)
            
            # Set current version to target version
            version_info = ConfigurationVersion(version="2.0.0")
            manager._version_info = version_info
            manager.save_version_info()
            
            # Attempt migration
            migrated = manager.migrate_configuration("2.0.0")
            assert migrated is False  # No migration needed
    
    def test_migrate_configuration_from_1x_to_2x(self):
        """Test migration from 1.x to 2.0.0."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prefs_dir = Path(temp_dir) / "preferences"
            
            # Create old configuration file
            config_file = Path(temp_dir) / "startup_config.json"
            old_config = {
                "backend": {"port": 8000},
                "frontend": {"port": 3000},
                "verbose_logging": True,
                "auto_fix_issues": False
            }
            with open(config_file, 'w') as f:
                json.dump(old_config, f)
            
            original_cwd = Path.cwd()
            try:
                import os
                os.chdir(temp_dir)
                
                manager = PreferenceManager(prefs_dir)
                
                # Set old version
                version_info = ConfigurationVersion(version="1.0.0")
                manager._version_info = version_info
                manager.save_version_info()
                
                # Perform migration
                migrated = manager.migrate_configuration("2.0.0")
                assert migrated is True
                
                # Verify migration results
                updated_version = manager.load_version_info()
                assert updated_version.version == "2.0.0"
                assert len(updated_version.migration_notes) > 0
                
                # Verify preferences were updated
                preferences = manager.load_preferences()
                assert preferences.verbose_output is True  # Migrated from verbose_logging
                assert preferences.auto_retry_failed_operations is False  # Migrated from auto_fix_issues
                
            finally:
                os.chdir(original_cwd)
    
    def test_apply_preferences_to_config(self):
        """Test applying user preferences to startup configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prefs_dir = Path(temp_dir) / "preferences"
            
            manager = PreferenceManager(prefs_dir)
            
            # Create preferences with custom values
            preferences = UserPreferences(
                preferred_backend_port=8080,
                preferred_frontend_port=3001,
                auto_open_browser=False,
                allow_port_auto_increment=False,
                max_auto_retries=5,
                auto_retry_failed_operations=False,
                allow_admin_elevation=False,
                verbose_output=True,
                startup_timeout_multiplier=1.5
            )
            
            manager._preferences = preferences
            
            # Create base configuration
            config = StartupConfig()
            
            # Apply preferences
            updated_config = manager.apply_preferences_to_config(config)
            
            # Verify preferences were applied
            assert updated_config.backend.port == 8080
            assert updated_config.frontend.port == 3001
            assert updated_config.frontend.open_browser is False
            assert updated_config.backend.auto_port is False
            assert updated_config.frontend.auto_port is False
            assert updated_config.recovery.max_retry_attempts == 5
            assert updated_config.recovery.enabled is False
            assert updated_config.security.allow_admin_elevation is False
            assert updated_config.logging.level == "debug"
            assert updated_config.backend.timeout == int(30 * 1.5)  # 45
            assert updated_config.frontend.timeout == int(30 * 1.5)  # 45


class TestPreferenceUtilities:
    """Test cases for preference utility functions."""
    
    def test_load_preferences_convenience_function(self):
        """Test convenience function for loading preferences."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prefs_dir = Path(temp_dir) / "preferences"
            
            preferences = load_preferences(prefs_dir)
            
            assert isinstance(preferences, UserPreferences)
            assert preferences.auto_open_browser is True  # Default value
            assert (prefs_dir / "preferences.json").exists()


if __name__ == "__main__":
    pytest.main([__file__])