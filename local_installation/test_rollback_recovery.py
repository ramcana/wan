"""
Test suite for enhanced RollbackManager recovery functionality
Tests recovery procedures, cleanup, and emergency backup features.
"""

import os
import json
import tempfile
import shutil
import unittest
from unittest.mock import Mock, patch
from pathlib import Path

# Add the scripts directory to the path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from rollback_manager import RollbackManager, RollbackSnapshot


class TestRollbackRecovery(unittest.TestCase):
    """Test cases for RollbackManager recovery functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.rollback_manager = RollbackManager(self.test_dir, dry_run=False)
        
        # Create some test files and directories
        self.test_config = Path(self.test_dir) / "config.json"
        self.test_config.write_text('{"test": "config"}')
        
        self.test_scripts_dir = Path(self.test_dir) / "scripts"
        self.test_scripts_dir.mkdir()
        (self.test_scripts_dir / "test_script.py").write_text("# test script")
        
        self.test_models_dir = Path(self.test_dir) / "models"
        self.test_models_dir.mkdir()
        (self.test_models_dir / "test_model.bin").write_text("fake model data")
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_create_recovery_point(self):
        """Test creating a comprehensive recovery point."""
        snapshot_id = self.rollback_manager.create_recovery_point("Test recovery point")
        
        self.assertIsNotNone(snapshot_id)
        self.assertTrue(snapshot_id.startswith("snapshot_"))
        
        # Verify recovery point was created
        snapshots = self.rollback_manager.list_snapshots()
        self.assertEqual(len(snapshots), 1)
        self.assertIn("Recovery point", snapshots[0].description)
        self.assertEqual(snapshots[0].phase, "recovery")
    
    def test_create_emergency_backup(self):
        """Test creating emergency backup."""
        snapshot_id = self.rollback_manager.create_emergency_backup()
        
        self.assertIsNotNone(snapshot_id)
        
        # Verify emergency backup was created
        snapshots = self.rollback_manager.list_snapshots()
        self.assertEqual(len(snapshots), 1)
        self.assertIn("Emergency backup", snapshots[0].description)
        self.assertEqual(snapshots[0].phase, "emergency")
    
    def test_recover_from_failed_installation_success(self):
        """Test successful recovery from failed installation."""
        # Create a pre-installation snapshot
        pre_install_id = self.rollback_manager.create_snapshot(
            "Pre-installation backup",
            "pre-install",
            files_to_backup=["config.json"],
            dirs_to_backup=["scripts"]
        )
        
        # Simulate installation failure by modifying files
        self.test_config.write_text('{"corrupted": "config"}')
        (self.test_scripts_dir / "corrupted_script.py").write_text("# corrupted")
        
        # Attempt recovery
        result = self.rollback_manager.recover_from_failed_installation("dependencies")
        
        self.assertTrue(result)
        
        # Verify files were restored
        with open(self.test_config, 'r') as f:
            config = json.load(f)
        self.assertEqual(config["test"], "config")
        
        # Verify post-recovery snapshot was created
        snapshots = self.rollback_manager.list_snapshots()
        post_recovery_snapshots = [s for s in snapshots if "post-recovery" in s.description.lower()]
        self.assertEqual(len(post_recovery_snapshots), 1)
    
    def test_recover_from_failed_installation_no_snapshots(self):
        """Test recovery when no snapshots are available."""
        result = self.rollback_manager.recover_from_failed_installation("models")
        
        self.assertFalse(result)
    
    def test_get_recovery_recommendations_no_snapshots(self):
        """Test recovery recommendations when no snapshots exist."""
        recommendations = self.rollback_manager.get_recovery_recommendations()
        
        self.assertEqual(len(recommendations), 1)
        self.assertEqual(recommendations[0]['type'], 'no_snapshots')
        self.assertEqual(recommendations[0]['priority'], 'high')
    
    def test_get_recovery_recommendations_with_snapshots(self):
        """Test recovery recommendations with various snapshot types."""
        # Create different types of snapshots
        self.rollback_manager.create_snapshot("Pre-installation backup", "pre-install")
        self.rollback_manager.create_snapshot("Regular backup", "backup")
        
        # Create many snapshots to trigger cleanup recommendation
        for i in range(12):
            self.rollback_manager.create_snapshot(f"Backup {i}", "backup")
        
        recommendations = self.rollback_manager.get_recovery_recommendations()
        
        # Should have pre-install recommendation
        pre_install_recs = [r for r in recommendations if r['type'] == 'pre_install_available']
        self.assertEqual(len(pre_install_recs), 1)
        
        # Should have cleanup recommendation
        cleanup_recs = [r for r in recommendations if r['type'] == 'cleanup_needed']
        self.assertEqual(len(cleanup_recs), 1)
    
    def test_cleanup_failed_installation_artifacts(self):
        """Test cleaning up failed installation artifacts."""
        # Create some temporary artifacts
        temp_download = Path(self.test_dir) / "temp_download"
        temp_download.mkdir()
        (temp_download / "test_file.txt").write_text("temp data")
        
        temp_file = Path(self.test_dir) / "installation.tmp"
        temp_file.write_text("temp installation data")
        
        partial_file = Path(self.test_dir) / "model.partial"
        partial_file.write_text("partial download")
        
        # Run cleanup
        result = self.rollback_manager.cleanup_failed_installation_artifacts()
        
        self.assertTrue(result)
        
        # Verify artifacts were cleaned up
        self.assertFalse(temp_download.exists())
        self.assertFalse(temp_file.exists())
        self.assertFalse(partial_file.exists())
        
        # Verify legitimate files were not touched
        self.assertTrue(self.test_config.exists())
        self.assertTrue(self.test_scripts_dir.exists())
    
    def test_validate_snapshot_valid(self):
        """Test validating a valid snapshot."""
        # Create a snapshot
        snapshot_id = self.rollback_manager.create_snapshot(
            "Test snapshot",
            "test",
            files_to_backup=["config.json"],
            dirs_to_backup=["scripts"]
        )
        
        # Validate the snapshot
        validation = self.rollback_manager.validate_snapshot(snapshot_id)
        
        self.assertTrue(validation['valid'])
        self.assertEqual(len(validation['issues']), 0)
    
    def test_validate_snapshot_invalid(self):
        """Test validating an invalid snapshot."""
        # Create a snapshot
        snapshot_id = self.rollback_manager.create_snapshot(
            "Test snapshot",
            "test",
            files_to_backup=["config.json"]
        )
        
        # Corrupt the snapshot by deleting backup files
        snapshot_dir = self.rollback_manager.snapshots_dir / snapshot_id
        for file_path in snapshot_dir.rglob('*'):
            if file_path.is_file() and file_path.name != 'snapshot.json':
                file_path.unlink()
        
        # Validate the snapshot
        validation = self.rollback_manager.validate_snapshot(snapshot_id)
        
        self.assertFalse(validation['valid'])
        self.assertGreater(len(validation['issues']), 0)
    
    def test_get_recovery_recommendations_invalid_snapshots(self):
        """Test recovery recommendations with invalid snapshots."""
        # Create a snapshot
        snapshot_id = self.rollback_manager.create_snapshot(
            "Test snapshot",
            "test",
            files_to_backup=["config.json"]
        )
        
        # Corrupt the snapshot
        snapshot_dir = self.rollback_manager.snapshots_dir / snapshot_id
        for file_path in snapshot_dir.rglob('*'):
            if file_path.is_file() and file_path.name != 'snapshot.json':
                file_path.unlink()
        
        recommendations = self.rollback_manager.get_recovery_recommendations()
        
        # Should detect invalid snapshots
        invalid_recs = [r for r in recommendations if r['type'] == 'invalid_snapshots']
        self.assertEqual(len(invalid_recs), 1)
        self.assertIn(snapshot_id, invalid_recs[0]['snapshot_ids'])


class TestRollbackRecoveryDryRun(unittest.TestCase):
    """Test RollbackManager recovery functionality in dry run mode."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.rollback_manager = RollbackManager(self.test_dir, dry_run=True)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_dry_run_recover_from_failed_installation(self):
        """Test recovery in dry run mode."""
        result = self.rollback_manager.recover_from_failed_installation("test")
        
        # Should succeed without actually doing anything
        self.assertTrue(result)
    
    def test_dry_run_cleanup_failed_installation_artifacts(self):
        """Test cleanup in dry run mode."""
        result = self.rollback_manager.cleanup_failed_installation_artifacts()
        
        # Should succeed without actually doing anything
        self.assertTrue(result)
    
    def test_dry_run_create_emergency_backup(self):
        """Test emergency backup in dry run mode."""
        # Should not raise exception
        snapshot_id = self.rollback_manager.create_emergency_backup()
        
        # Should return a snapshot ID but not create actual files
        self.assertIsNotNone(snapshot_id)


class TestRollbackRecoveryIntegration(unittest.TestCase):
    """Integration tests for rollback and recovery system."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.rollback_manager = RollbackManager(self.test_dir, dry_run=False)
        
        # Create a more complex test environment
        self.config_file = Path(self.test_dir) / "config.json"
        self.config_file.write_text('{"version": "1.0.0", "settings": {"debug": false}}')
        
        self.version_file = Path(self.test_dir) / "version.json"
        self.version_file.write_text('{"version": "1.0.0", "installed_at": "2024-01-01"}')
        
        self.scripts_dir = Path(self.test_dir) / "scripts"
        self.scripts_dir.mkdir()
        (self.scripts_dir / "main.py").write_text("# main script")
        (self.scripts_dir / "utils.py").write_text("# utilities")
        
        self.models_dir = Path(self.test_dir) / "models"
        self.models_dir.mkdir()
        (self.models_dir / "model1.bin").write_text("model data 1")
        (self.models_dir / "model2.bin").write_text("model data 2")
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_full_recovery_workflow(self):
        """Test complete recovery workflow from installation failure."""
        # Step 1: Create initial recovery point
        initial_snapshot = self.rollback_manager.create_recovery_point("Initial state")
        
        # Step 2: Simulate installation process with intermediate snapshots
        pre_update_snapshot = self.rollback_manager.create_snapshot(
            "Pre-update backup (v1.0.0)",
            "pre-update",
            files_to_backup=["config.json", "version.json"],
            dirs_to_backup=["scripts", "models"]
        )
        
        # Step 3: Simulate installation failure
        # Corrupt configuration
        self.config_file.write_text('{"corrupted": true}')
        
        # Delete some scripts
        (self.scripts_dir / "utils.py").unlink()
        
        # Create temporary artifacts
        temp_dir = Path(self.test_dir) / "temp_download"
        temp_dir.mkdir()
        (temp_dir / "partial_model.tmp").write_text("incomplete download")
        
        # Step 4: Attempt recovery
        recovery_result = self.rollback_manager.recover_from_failed_installation("models")
        
        self.assertTrue(recovery_result)
        
        # Step 5: Verify recovery
        # Check configuration was restored
        with open(self.config_file, 'r') as f:
            config = json.load(f)
        self.assertEqual(config["version"], "1.0.0")
        self.assertFalse(config["settings"]["debug"])
        
        # Check scripts were restored
        self.assertTrue((self.scripts_dir / "utils.py").exists())
        
        # Check models were restored
        self.assertTrue((self.models_dir / "model1.bin").exists())
        self.assertTrue((self.models_dir / "model2.bin").exists())
        
        # Step 6: Clean up artifacts
        cleanup_result = self.rollback_manager.cleanup_failed_installation_artifacts()
        self.assertTrue(cleanup_result)
        
        # Verify temp artifacts were cleaned up
        self.assertFalse(temp_dir.exists())
        
        # Step 7: Verify post-recovery state
        snapshots = self.rollback_manager.list_snapshots()
        post_recovery_snapshots = [s for s in snapshots if "post-recovery" in s.description.lower()]
        self.assertEqual(len(post_recovery_snapshots), 1)
        
        # Step 8: Get recovery recommendations
        recommendations = self.rollback_manager.get_recovery_recommendations()
        
        # Should have recommendations about available snapshots
        self.assertGreater(len(recommendations), 0)
        
        # Should have pre-install recommendation
        pre_install_recs = [r for r in recommendations if r['type'] == 'pre_install_available']
        self.assertGreater(len(pre_install_recs), 0)


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    unittest.main()