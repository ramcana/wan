"""
Tests for preference CLI functionality.
"""

import pytest
import tempfile
from pathlib import Path
from click.testing import CliRunner

from scripts.startup_manager.preference_cli import preferences_cli
from scripts.startup_manager.preferences import PreferenceManager, UserPreferences


class TestPreferenceCLI:
    """Test cases for preference CLI commands."""
    
    def test_show_preferences(self):
        """Test showing current preferences."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prefs_dir = Path(temp_dir) / "preferences"
            
            # Create preferences with custom values
            manager = PreferenceManager(prefs_dir)
            preferences = manager.load_preferences()
            preferences.auto_open_browser = False
            preferences.verbose_output = True
            preferences.preferred_backend_port = 8080
            manager.save_preferences()
            
            runner = CliRunner()
            result = runner.invoke(preferences_cli, ['show', '--preferences-dir', str(prefs_dir)])
            
            assert result.exit_code == 0
            assert "Current User Preferences" in result.output
            assert "auto_open_browser" in result.output
            assert "False" in result.output
            assert "8080" in result.output
    
    def test_reset_preferences(self):
        """Test resetting preferences to defaults."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prefs_dir = Path(temp_dir) / "preferences"
            
            # Create preferences with custom values
            manager = PreferenceManager(prefs_dir)
            preferences = manager.load_preferences()
            preferences.auto_open_browser = False
            preferences.verbose_output = True
            preferences.preferred_backend_port = 8080
            manager.save_preferences()
            
            runner = CliRunner()
            # Simulate user confirming reset
            result = runner.invoke(preferences_cli, ['reset', '--preferences-dir', str(prefs_dir)], input='y\n')
            
            assert result.exit_code == 0
            assert "Preferences reset to defaults" in result.output
            
            # Verify preferences were reset
            updated_preferences = manager.load_preferences()
            assert updated_preferences.auto_open_browser is True  # Default
            assert updated_preferences.verbose_output is False  # Default
            assert updated_preferences.preferred_backend_port is None  # Default
    
    def test_backup_preferences(self):
        """Test creating a backup of preferences."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prefs_dir = Path(temp_dir) / "preferences"
            
            manager = PreferenceManager(prefs_dir)
            manager.load_preferences()
            
            runner = CliRunner()
            result = runner.invoke(preferences_cli, ['backup', '--preferences-dir', str(prefs_dir)], input='test_backup\n')
            
            assert result.exit_code == 0
            assert "Backup created at:" in result.output
            
            # Verify backup was created
            backups = manager.list_backups()
            backup_names = [b["name"] for b in backups]
            assert "test_backup" in backup_names
    
    def test_cleanup_backups(self):
        """Test cleaning up old backups."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prefs_dir = Path(temp_dir) / "preferences"
            
            manager = PreferenceManager(prefs_dir)
            manager.load_preferences()
            
            # Create multiple backups
            for i in range(5):
                manager.create_backup(f"backup_{i}")
            
            runner = CliRunner()
            result = runner.invoke(preferences_cli, ['cleanup', '--preferences-dir', str(prefs_dir), '--keep', '3'])
            
            assert result.exit_code == 0
            assert "Cleaned up 2 old backups" in result.output
            
            # Verify only 3 backups remain
            backups = manager.list_backups()
            assert len(backups) == 3
    
    def test_migrate_configuration(self):
        """Test configuration migration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prefs_dir = Path(temp_dir) / "preferences"
            
            manager = PreferenceManager(prefs_dir)
            
            # Set old version
            from scripts.startup_manager.preferences import ConfigurationVersion
            version_info = ConfigurationVersion(version="1.0.0")
            manager._version_info = version_info
            manager.save_version_info()
            
            runner = CliRunner()
            result = runner.invoke(preferences_cli, ['migrate', '--preferences-dir', str(prefs_dir), '--target-version', '2.0.0'], input='y\n')
            
            assert result.exit_code == 0
            # Migration should succeed or indicate no migration needed
            assert ("Successfully migrated" in result.output or "No migration needed" in result.output)


if __name__ == "__main__":
    pytest.main([__file__])