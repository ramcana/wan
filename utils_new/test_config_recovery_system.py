"""
Test suite for ConfigRecoverySystem class

Tests configuration recovery, restoration from defaults, and trust_remote_code handling.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from config_recovery_system import (
    ConfigRecoverySystem,
    RecoveryAction,
    RecoveryResult,
    recover_config,
    update_trust_remote_code,
    format_recovery_result
)
from config_validator import ValidationSeverity, ValidationMessage


class TestConfigRecoverySystem:
    """Test cases for ConfigRecoverySystem class"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.backup_dir = self.temp_dir / "backups"
        self.recovery_dir = self.temp_dir / "recovery"
        
        self.recovery_system = ConfigRecoverySystem(
            backup_dir=self.backup_dir,
            recovery_dir=self.recovery_dir
        )
        
        # Create a valid test configuration
        self.valid_config = {
            "system": {
                "default_quantization": "bf16",
                "enable_offload": True,
                "vae_tile_size": 256
            },
            "directories": {
                "output_directory": "outputs",
                "models_directory": "models"
            },
            "generation": {
                "default_resolution": "1280x720",
                "default_steps": 50
            }
        }
        
        # Create a corrupted configuration
        self.corrupted_config = {
            "system": {
                "default_quantization": "invalid_value",
                "enable_offload": "not_boolean"
            }
            # Missing required sections
        }
        
        # Create test config file
        self.test_config_path = self.temp_dir / "config.json"
    
    def teardown_method(self):
        """Clean up test environment"""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_restore_from_defaults_success(self):
        """Test successful restoration from defaults"""
        # Write corrupted config
        with open(self.test_config_path, 'w') as f:
            json.dump(self.corrupted_config, f)
        
        # Recover configuration
        result = self.recovery_system.recover_config(self.test_config_path)
        
        # Verify recovery was successful
        assert result.success
        assert result.action_taken in [RecoveryAction.RESTORE_FROM_DEFAULTS, RecoveryAction.MERGE_WITH_DEFAULTS]
        assert result.backup_path is not None
        assert len(result.changes_made) > 0
        
        # Verify config was restored
        with open(self.test_config_path, 'r') as f:
            restored_config = json.load(f)
        
        # Should have required sections
        assert "system" in restored_config
        assert "directories" in restored_config
        assert "generation" in restored_config
    
    def test_restore_from_backup_success(self):
        """Test successful restoration from backup"""
        # Create a valid config first
        with open(self.test_config_path, 'w') as f:
            json.dump(self.valid_config, f)
        
        # Create backup
        backup_path = self.recovery_system.validator.create_backup(self.test_config_path)
        
        # Corrupt the config
        with open(self.test_config_path, 'w') as f:
            f.write("invalid json content")
        
        # Recover configuration
        result = self.recovery_system.recover_config(self.test_config_path)
        
        # Verify recovery was successful
        assert result.success
        assert result.action_taken == RecoveryAction.RESTORE_FROM_BACKUP
        
        # Verify config was restored from backup
        with open(self.test_config_path, 'r') as f:
            restored_config = json.load(f)
        
        assert restored_config == self.valid_config
    
    def test_merge_with_defaults(self):
        """Test merging partial config with defaults"""
        # Create partial config
        partial_config = {
            "system": {
                "default_quantization": "bf16"
                # Missing other required fields
            }
            # Missing other required sections
        }
        
        with open(self.test_config_path, 'w') as f:
            json.dump(partial_config, f)
        
        # Recover configuration
        result = self.recovery_system.recover_config(self.test_config_path)
        
        # Verify recovery was successful
        assert result.success
        assert result.action_taken == RecoveryAction.MERGE_WITH_DEFAULTS
        assert len(result.changes_made) > 0
        
        # Verify config was merged
        with open(self.test_config_path, 'r') as f:
            merged_config = json.load(f)
        
        # Should have original value
        assert merged_config["system"]["default_quantization"] == "bf16"
        # Should have added missing sections
        assert "directories" in merged_config
        assert "generation" in merged_config
    
    def test_trust_remote_code_update(self):
        """Test updating trust_remote_code settings"""
        # Create config without trust settings
        with open(self.test_config_path, 'w') as f:
            json.dump(self.valid_config, f)
        
        # Update trust settings
        result = self.recovery_system.update_trust_remote_code_setting(
            self.test_config_path, 
            enable_trust=True,
            model_names=["model1", "model2"]
        )
        
        # Verify update was successful
        assert result.success
        assert result.action_taken == RecoveryAction.UPDATE_TRUST_REMOTE_CODE
        assert len(result.changes_made) == 2  # Two models updated
        
        # Verify trust settings were added
        with open(self.test_config_path, 'r') as f:
            updated_config = json.load(f)
        
        assert "trust_remote_code" in updated_config
        assert updated_config["trust_remote_code"]["model1"] is True
        assert updated_config["trust_remote_code"]["model2"] is True
    
    def test_trust_remote_code_global_setting(self):
        """Test updating global trust_remote_code setting"""
        # Create config without trust settings
        with open(self.test_config_path, 'w') as f:
            json.dump(self.valid_config, f)
        
        # Update global trust setting
        result = self.recovery_system.update_trust_remote_code_setting(
            self.test_config_path, 
            enable_trust=False
        )
        
        # Verify update was successful
        assert result.success
        assert result.action_taken == RecoveryAction.UPDATE_TRUST_REMOTE_CODE
        
        # Verify global trust setting was added
        with open(self.test_config_path, 'r') as f:
            updated_config = json.load(f)
        
        assert "trust_remote_code" in updated_config
        assert updated_config["trust_remote_code"]["global"] is False
    
    def test_config_changes_report(self):
        """Test configuration changes reporting"""
        # Create config and make some changes
        with open(self.test_config_path, 'w') as f:
            json.dump(self.corrupted_config, f)
        
        # Recover configuration (this will log changes)
        self.recovery_system.recover_config(self.test_config_path)
        
        # Update trust settings (this will log more changes)
        self.recovery_system.update_trust_remote_code_setting(
            self.test_config_path, 
            enable_trust=True,
            model_names=["test_model"]
        )
        
        # Get changes report
        report = self.recovery_system.get_config_changes_report()
        
        # Verify report structure
        assert "total_changes" in report
        assert "recent_changes" in report
        assert "changes_by_type" in report
        assert "detailed_changes" in report
        assert "generated_at" in report
        
        # Should have at least 2 changes
        assert report["total_changes"] >= 2
        
        # Should have different change types
        assert len(report["changes_by_type"]) >= 1
    
    def test_config_changes_report_specific_file(self):
        """Test configuration changes report for specific file"""
        # Create config and make changes
        with open(self.test_config_path, 'w') as f:
            json.dump(self.corrupted_config, f)
        
        # Recover configuration
        self.recovery_system.recover_config(self.test_config_path)
        
        # Get changes report for specific file
        report = self.recovery_system.get_config_changes_report(self.test_config_path)
        
        # Verify report is filtered to specific file
        assert report["total_changes"] >= 1
        for change in report["detailed_changes"]:
            assert change["config_path"] == str(self.test_config_path)
    
    def test_no_recovery_needed(self):
        """Test when configuration is already valid"""
        # Create valid config
        with open(self.test_config_path, 'w') as f:
            json.dump(self.valid_config, f)
        
        # Attempt recovery
        result = self.recovery_system.recover_config(self.test_config_path)
        
        # Should indicate no recovery needed
        assert result.success
        assert len([msg for msg in result.messages if msg.code == "NO_RECOVERY_NEEDED"]) > 0
    
    def test_recovery_failure_handling(self):
        """Test handling of recovery failures"""
        # Create non-existent config path
        non_existent_path = self.temp_dir / "non_existent.json"
        
        # Attempt recovery
        result = self.recovery_system.recover_config(non_existent_path)
        
        # Should handle failure gracefully
        assert not result.success
        assert len(result.messages) > 0
        assert any(msg.severity == ValidationSeverity.CRITICAL for msg in result.messages)


def test_convenience_functions():
    """Test convenience functions"""
    temp_dir = Path(tempfile.mkdtemp())
    test_config_path = temp_dir / "config.json"
    
    try:
        # Create corrupted config
        corrupted_config = {"invalid": "config"}
        with open(test_config_path, 'w') as f:
            json.dump(corrupted_config, f)
        
        # Test recover_config convenience function
        result = recover_config(test_config_path)
        assert isinstance(result, RecoveryResult)
        
        # Test update_trust_remote_code convenience function
        result = update_trust_remote_code(test_config_path, True, ["test_model"])
        assert isinstance(result, RecoveryResult)
        
        # Test format_recovery_result function
        formatted = format_recovery_result(result)
        assert isinstance(formatted, str)
        assert len(formatted) > 0
        
    finally:
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)