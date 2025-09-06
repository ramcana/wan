"""
Test suite for VersionManager
Tests version checking, update downloading, and migration functionality.
"""

import os
import json
import tempfile
import shutil
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add the scripts directory to the path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from version_manager import VersionManager, UpdateInfo, MigrationInfo


class TestVersionManager(unittest.TestCase):
    """Test cases for VersionManager."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.version_manager = VersionManager(self.test_dir, dry_run=False)
        
        # Create test version file
        version_data = {
            "version": "1.0.0",
            "installed_at": "2024-01-01T00:00:00",
            "last_update_check": None,
            "update_channel": "stable",
            "auto_update": False
        }
        
        with open(self.version_manager.version_file, 'w') as f:
            json.dump(version_data, f)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_get_current_version(self):
        """Test getting current version."""
        version = self.version_manager.get_current_version()
        self.assertEqual(version, "1.0.0")
    
    def test_get_current_version_no_file(self):
        """Test getting current version when file doesn't exist."""
        os.remove(self.version_manager.version_file)
        version = self.version_manager.get_current_version()
        self.assertEqual(version, "1.0.0")  # Default version
    
    @patch('requests.get')
    def test_check_for_updates_available(self, mock_get):
        """Test checking for updates when update is available."""
        # Mock GitHub API response
        mock_response = Mock()
        mock_response.json.return_value = {
            'tag_name': 'v1.1.0',
            'html_url': 'https://github.com/test/repo/releases/tag/v1.1.0',
            'body': 'New features and bug fixes',
            'published_at': '2024-01-15T00:00:00Z',
            'prerelease': False,
            'assets': [
                {
                    'name': 'wan22-installer.zip',
                    'download_url': 'https://github.com/test/repo/releases/download/v1.1.0/installer.zip',
                    'size': 1024000
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        update_info = self.version_manager.check_for_updates()
        
        self.assertEqual(update_info.current_version, "1.0.0")
        self.assertEqual(update_info.latest_version, "1.1.0")
        self.assertTrue(update_info.update_available)
        self.assertEqual(update_info.size_bytes, 1024000)
    
    @patch('requests.get')
    def test_check_for_updates_no_update(self, mock_get):
        """Test checking for updates when no update is available."""
        # Mock GitHub API response with same version
        mock_response = Mock()
        mock_response.json.return_value = {
            'tag_name': 'v1.0.0',
            'html_url': 'https://github.com/test/repo/releases/tag/v1.0.0',
            'body': 'Current release',
            'published_at': '2024-01-01T00:00:00Z',
            'prerelease': False,
            'assets': []
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        update_info = self.version_manager.check_for_updates()
        
        self.assertEqual(update_info.current_version, "1.0.0")
        self.assertEqual(update_info.latest_version, "1.0.0")
        self.assertFalse(update_info.update_available)
    
    @patch('requests.get')
    def test_check_for_updates_network_error(self, mock_get):
        """Test checking for updates with network error."""
        mock_get.side_effect = Exception("Network error")
        
        update_info = self.version_manager.check_for_updates()
        
        self.assertEqual(update_info.current_version, "1.0.0")
        self.assertEqual(update_info.latest_version, "1.0.0")
        self.assertFalse(update_info.update_available)
        self.assertIn("Network error", update_info.release_notes)
    
    @patch('requests.get')
    def test_download_update(self, mock_get):
        """Test downloading an update."""
        # Create mock update info
        update_info = UpdateInfo(
            current_version="1.0.0",
            latest_version="1.1.0",
            update_available=True,
            release_url="https://github.com/test/repo/releases/tag/v1.1.0",
            release_notes="Test update",
            download_url="https://github.com/test/repo/releases/download/v1.1.0/installer.zip",
            published_at="2024-01-15T00:00:00Z",
            prerelease=False,
            size_bytes=1024
        )
        
        # Mock download response
        mock_response = Mock()
        mock_response.headers = {'content-length': '1024'}
        mock_response.iter_content.return_value = [b'test data chunk']
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Mock progress callback
        progress_callback = Mock()
        
        download_path = self.version_manager.download_update(update_info, progress_callback)
        
        self.assertTrue(os.path.exists(download_path))
        self.assertTrue(download_path.endswith('.zip'))
        progress_callback.assert_called()
    
    def test_backup_current_installation(self):
        """Test creating backup of current installation."""
        # Create some test files
        test_config = self.test_dir + "/config.json"
        with open(test_config, 'w') as f:
            json.dump({"test": "config"}, f)
        
        test_scripts_dir = Path(self.test_dir) / "scripts"
        test_scripts_dir.mkdir()
        (test_scripts_dir / "test_script.py").write_text("# test script")
        
        snapshot_id = self.version_manager.backup_current_installation()
        
        self.assertIsNotNone(snapshot_id)
        self.assertTrue(snapshot_id.startswith("snapshot_"))
        
        # Verify backup was created
        snapshots = self.version_manager.rollback_manager.list_snapshots()
        self.assertEqual(len(snapshots), 1)
        self.assertIn("Pre-update backup", snapshots[0].description)
    
    def test_create_migration_template(self):
        """Test creating migration script template."""
        script_path = self.version_manager.create_migration_template("1.0.0", "1.1.0")
        
        self.assertTrue(os.path.exists(script_path))
        self.assertTrue(script_path.endswith("migrate_v1.0.0_to_v1.1.0.py"))
        
        # Verify template content
        with open(script_path, 'r') as f:
            content = f.read()
            self.assertIn("def migrate(context):", content)
            self.assertIn("def validate_migration(context):", content)
            self.assertIn("v1.0.0 to v1.1.0", content)
    
    def test_find_migration_scripts(self):
        """Test finding applicable migration scripts."""
        # Create test migration scripts
        migrations_dir = self.version_manager.migrations_dir
        migrations_dir.mkdir(exist_ok=True)
        
        # Create migration from 1.0.0 to 1.1.0
        script1 = migrations_dir / "migrate_v1.0.0_to_v1.1.0.py"
        script1.write_text("# Migration script 1")
        
        # Create migration from 1.1.0 to 1.2.0
        script2 = migrations_dir / "migrate_v1.1.0_to_v1.2.0.py"
        script2.write_text("# Migration script 2")
        
        # Find migrations from 1.0.0 to 1.2.0
        migrations = self.version_manager._find_migration_scripts("1.0.0", "1.2.0")
        
        self.assertEqual(len(migrations), 2)
        self.assertEqual(migrations[0].from_version, "1.0.0")
        self.assertEqual(migrations[0].to_version, "1.1.0")
        self.assertEqual(migrations[1].from_version, "1.1.0")
        self.assertEqual(migrations[1].to_version, "1.2.0")
    
    def test_update_version_info(self):
        """Test updating version information."""
        self.version_manager.update_version_info("1.1.0")
        
        # Verify version file was updated
        with open(self.version_manager.version_file, 'r') as f:
            version_data = json.load(f)
        
        self.assertEqual(version_data['version'], "1.1.0")
        self.assertEqual(version_data['previous_version'], "1.0.0")
        self.assertIn('updated_at', version_data)
    
    def test_get_version_history(self):
        """Test getting version history."""
        # Create a snapshot to simulate version history
        self.version_manager.rollback_manager.create_snapshot(
            "Pre-update backup (v1.0.0)",
            "pre-update"
        )
        
        history = self.version_manager.get_version_history()
        
        self.assertGreater(len(history), 0)
        # Should have current version
        current_versions = [h for h in history if h.get('current')]
        self.assertEqual(len(current_versions), 1)
        self.assertEqual(current_versions[0]['version'], "1.0.0")


class TestVersionManagerDryRun(unittest.TestCase):
    """Test VersionManager in dry run mode."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.version_manager = VersionManager(self.test_dir, dry_run=True)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_dry_run_download_update(self):
        """Test download update in dry run mode."""
        update_info = UpdateInfo(
            current_version="1.0.0",
            latest_version="1.1.0",
            update_available=True,
            release_url="https://github.com/test/repo/releases/tag/v1.1.0",
            release_notes="Test update",
            download_url="https://github.com/test/repo/releases/download/v1.1.0/installer.zip",
            published_at="2024-01-15T00:00:00Z",
            prerelease=False
        )
        
        download_path = self.version_manager.download_update(update_info)
        
        # Should return a path but not actually download
        self.assertIsNotNone(download_path)
        self.assertTrue(download_path.endswith('.zip'))
    
    def test_dry_run_run_migration(self):
        """Test running migration in dry run mode."""
        result = self.version_manager.run_migration("1.0.0", "1.1.0")
        
        # Should succeed without actually running migration
        self.assertTrue(result)
    
    def test_dry_run_update_version_info(self):
        """Test updating version info in dry run mode."""
        # Should not raise exception
        self.version_manager.update_version_info("1.1.0")
        
        # Version file should not be created in dry run
        self.assertFalse(self.version_manager.version_file.exists())


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    unittest.main()