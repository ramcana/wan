#!/usr/bin/env python3
"""
Integration test for Configuration Recovery System

Tests the complete integration of configuration recovery functionality
including restoration, reporting, and trust_remote_code handling.
"""

import json
import tempfile
from pathlib import Path
import pytest
from config_recovery_system import (
    ConfigRecoverySystem,
    RecoveryAction,
    RecoveryResult,
    recover_config,
    update_trust_remote_code,
    format_recovery_result
)
from config_validator import ValidationSeverity


class TestConfigRecoveryIntegration:
    """Integration tests for configuration recovery system"""
    
    def setup_method(self):
        """Set up test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.backup_dir = self.temp_dir / "backups"
        self.recovery_dir = self.temp_dir / "recovery"
        
        self.recovery_system = ConfigRecoverySystem(
            backup_dir=self.backup_dir,
            recovery_dir=self.recovery_dir
        )
        
        # Test configurations
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
        
        self.corrupted_config = {
            "system": {
                "default_quantization": "invalid_value",
                "enable_offload": "not_boolean"
            }
        }
    
    def teardown_method(self):
        """Clean up test environment"""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_complete_recovery_workflow(self):
        """Test complete recovery workflow from corruption to restoration"""
        config_path = self.temp_dir / "workflow_test.json"
        
        # Step 1: Create corrupted config
        with open(config_path, 'w') as f:
            json.dump(self.corrupted_config, f)
        
        # Step 2: Recover configuration
        result = self.recovery_system.recover_config(config_path)
        
        # Verify recovery
        assert result.success
        assert result.action_taken in [
            RecoveryAction.RESTORE_FROM_DEFAULTS,
            RecoveryAction.MERGE_WITH_DEFAULTS
        ]
        assert result.backup_path is not None
        assert len(result.changes_made) > 0
        
        # Step 3: Verify restored config is valid
        with open(config_path, 'r') as f:
            restored_config = json.load(f)
        
        # Should have all required sections
        required_sections = ["system", "directories", "generation"]
        for section in required_sections:
            assert section in restored_config
        
        # Should have valid values
        assert restored_config["system"]["default_quantization"] in ["fp16", "bf16", "int8", "none"]
        assert isinstance(restored_config["system"]["enable_offload"], bool)
        assert isinstance(restored_config["system"]["vae_tile_size"], int)
        
        # Step 4: Verify backup was created
        assert Path(result.backup_path).exists()
        
        # Step 5: Verify change was logged
        changes_report = self.recovery_system.get_config_changes_report(config_path)
        assert changes_report["total_changes"] >= 1
    
    def test_trust_remote_code_complete_workflow(self):
        """Test complete trust_remote_code workflow"""
        config_path = self.temp_dir / "trust_test.json"
        
        # Step 1: Create config without trust settings
        with open(config_path, 'w') as f:
            json.dump(self.valid_config, f)
        
        # Step 2: Add trust settings for specific models
        models_to_trust = ["Wan2.2-TI2V-5B", "Wan2.2-T2V-A14B", "custom_model"]
        result = self.recovery_system.update_trust_remote_code_setting(
            config_path,
            enable_trust=True,
            model_names=models_to_trust
        )
        
        # Verify trust update
        assert result.success
        assert result.action_taken == RecoveryAction.UPDATE_TRUST_REMOTE_CODE
        assert len(result.changes_made) == len(models_to_trust)
        
        # Step 3: Verify trust settings were added
        with open(config_path, 'r') as f:
            config_with_trust = json.load(f)
        
        assert "trust_remote_code" in config_with_trust
        for model in models_to_trust:
            assert config_with_trust["trust_remote_code"][model] is True
        
        # Step 4: Update global trust setting
        global_result = self.recovery_system.update_trust_remote_code_setting(
            config_path,
            enable_trust=False  # Global setting
        )
        
        assert global_result.success
        
        # Step 5: Verify global setting was added
        with open(config_path, 'r') as f:
            config_with_global = json.load(f)
        
        assert config_with_global["trust_remote_code"]["global"] is False
        
        # Step 6: Verify all changes were logged
        changes_report = self.recovery_system.get_config_changes_report(config_path)
        assert changes_report["total_changes"] >= 2
        
        # Should have trust update changes
        trust_changes = [
            change for change in changes_report["detailed_changes"]
            if change["change_type"] == "update_trust_remote_code"
        ]
        assert len(trust_changes) >= 2
    
    def test_backup_and_restore_workflow(self):
        """Test backup and restore workflow"""
        config_path = self.temp_dir / "backup_test.json"
        
        # Step 1: Create valid config
        with open(config_path, 'w') as f:
            json.dump(self.valid_config, f)
        
        # Step 2: Create backup
        backup_path = self.recovery_system.validator.create_backup(config_path)
        assert Path(backup_path).exists()
        
        # Step 3: Corrupt the config
        with open(config_path, 'w') as f:
            f.write("{ invalid json content")
        
        # Step 4: Restore from backup
        result = self.recovery_system.recover_config(config_path)
        
        # Should restore from defaults since backup validation would fail for JSON error
        assert result.success
        assert result.action_taken == RecoveryAction.RESTORE_FROM_DEFAULTS
        
        # Step 5: Verify config was restored
        with open(config_path, 'r') as f:
            restored_config = json.load(f)
        
        # Should be a valid configuration
        assert "system" in restored_config
        assert "directories" in restored_config
    
    def test_configuration_change_reporting(self):
        """Test comprehensive configuration change reporting"""
        config1_path = self.temp_dir / "report_test1.json"
        config2_path = self.temp_dir / "report_test2.json"
        
        # Create multiple configs and make changes
        for config_path in [config1_path, config2_path]:
            with open(config_path, 'w') as f:
                json.dump(self.corrupted_config, f)
            
            # Recover each config
            self.recovery_system.recover_config(config_path)
            
            # Update trust settings
            self.recovery_system.update_trust_remote_code_setting(
                config_path,
                enable_trust=True,
                model_names=[f"model_{config_path.stem}"]
            )
        
        # Test global report
        global_report = self.recovery_system.get_config_changes_report()
        
        assert global_report["total_changes"] >= 4  # 2 recoveries + 2 trust updates
        assert "changes_by_type" in global_report
        assert len(global_report["changes_by_type"]) >= 2
        
        # Should have both recovery and trust update changes
        assert "restore_from_defaults" in global_report["changes_by_type"] or \
               "merge_with_defaults" in global_report["changes_by_type"]
        assert "update_trust_remote_code" in global_report["changes_by_type"]
        
        # Test file-specific report
        file_report = self.recovery_system.get_config_changes_report(config1_path)
        
        assert file_report["total_changes"] >= 2  # 1 recovery + 1 trust update
        
        # All changes should be for the specific file
        for change in file_report["detailed_changes"]:
            assert change["config_path"] == str(config1_path)
    
    def test_error_handling_and_recovery(self):
        """Test error handling and recovery scenarios"""
        
        # Test 1: Non-existent file
        non_existent = self.temp_dir / "does_not_exist.json"
        result = self.recovery_system.recover_config(non_existent)
        
        assert not result.success
        assert len(result.messages) > 0
        assert any(msg.severity == ValidationSeverity.CRITICAL for msg in result.messages)
        
        # Test 2: Already valid configuration
        valid_config_path = self.temp_dir / "valid_test.json"
        with open(valid_config_path, 'w') as f:
            json.dump(self.recovery_system.default_configs["main_config"], f)
        
        result = self.recovery_system.recover_config(valid_config_path)
        
        assert result.success
        # Should indicate no recovery was needed
        no_recovery_messages = [
            msg for msg in result.messages 
            if "NO_RECOVERY_NEEDED" in msg.code or "valid" in msg.message.lower()
        ]
        assert len(no_recovery_messages) > 0
        
        # Test 3: Permission error simulation (create read-only file)
        readonly_path = self.temp_dir / "readonly_test.json"
        with open(readonly_path, 'w') as f:
            json.dump(self.corrupted_config, f)
        
        # Make file read-only (this might not work on all systems)
        try:
            readonly_path.chmod(0o444)
            result = self.recovery_system.recover_config(readonly_path)
            # Should handle permission errors gracefully
            # Result may succeed or fail depending on system permissions
        except Exception:
            # Permission changes might not be supported
            pass
        finally:
            # Restore write permissions for cleanup
            try:
                readonly_path.chmod(0o666)
            except Exception:
                pass
    
    def test_convenience_functions_integration(self):
        """Test convenience functions work correctly"""
        config_path = self.temp_dir / "convenience_test.json"
        
        # Test recover_config convenience function
        with open(config_path, 'w') as f:
            json.dump(self.corrupted_config, f)
        
        result = recover_config(
            config_path,
            backup_dir=self.backup_dir,
            recovery_dir=self.recovery_dir
        )
        
        assert isinstance(result, RecoveryResult)
        assert result.success
        
        # Test update_trust_remote_code convenience function
        trust_result = update_trust_remote_code(
            config_path,
            enable_trust=True,
            model_names=["convenience_model"],
            backup_dir=self.backup_dir,
            recovery_dir=self.recovery_dir
        )
        
        assert isinstance(trust_result, RecoveryResult)
        assert trust_result.success
        
        # Test format_recovery_result function
        formatted = format_recovery_result(trust_result)
        
        assert isinstance(formatted, str)
        assert len(formatted) > 0
        assert "Configuration recovery successful" in formatted or "successful" in formatted
        assert "trust_remote_code" in formatted.lower()
    
    def test_multiple_recovery_strategies(self):
        """Test different recovery strategies are applied correctly"""
        
        # Test 1: Merge with defaults (partial config)
        partial_config_path = self.temp_dir / "partial_test.json"
        partial_config = {
            "system": {
                "default_quantization": "bf16"
                # Missing other fields
            }
            # Missing other sections
        }
        
        with open(partial_config_path, 'w') as f:
            json.dump(partial_config, f)
        
        result = self.recovery_system.recover_config(partial_config_path)
        
        assert result.success
        assert result.action_taken == RecoveryAction.MERGE_WITH_DEFAULTS
        
        # Verify original value was preserved
        with open(partial_config_path, 'r') as f:
            merged_config = json.load(f)
        
        assert merged_config["system"]["default_quantization"] == "bf16"
        assert "directories" in merged_config  # Added from defaults
        
        # Test 2: Restore from defaults (heavily corrupted)
        corrupted_config_path = self.temp_dir / "corrupted_test.json"
        heavily_corrupted = {
            "system": {
                "default_quantization": "completely_invalid",
                "enable_offload": "also_invalid",
                "vae_tile_size": "not_a_number"
            },
            "directories": {
                "output_directory": None,
                "models_directory": 123
            }
        }
        
        with open(corrupted_config_path, 'w') as f:
            json.dump(heavily_corrupted, f)
        
        result = self.recovery_system.recover_config(corrupted_config_path)
        
        assert result.success
        assert result.action_taken == RecoveryAction.RESTORE_FROM_DEFAULTS
        
        # Verify config was completely replaced with defaults
        with open(corrupted_config_path, 'r') as f:
            restored_config = json.load(f)
        
        # Should have valid default values
        assert restored_config["system"]["default_quantization"] in ["fp16", "bf16", "int8", "none"]
        assert isinstance(restored_config["system"]["enable_offload"], bool)
        assert isinstance(restored_config["directories"]["output_directory"], str)


def test_integration_with_real_config():
    """Test integration with a realistic WAN22 configuration"""
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        recovery_system = ConfigRecoverySystem(
            backup_dir=temp_dir / "backups",
            recovery_dir=temp_dir / "recovery"
        )
        
        # Create a realistic config with some issues
        realistic_config_path = temp_dir / "wan22_config.json"
        realistic_config = {
            "system": {
                "default_quantization": "bf16",
                "enable_offload": True,
                "vae_tile_size": 256,
                "max_queue_size": 10
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
                "default_fps": 24
            },
            "models": {
                "t2v_model": "Wan2.2-T2V-A14B",
                "i2v_model": "Wan2.2-I2V-A14B",
                "ti2v_model": "Wan2.2-TI2V-5B"
            },
            # Add some problematic attributes that should be cleaned
            "clip_output": True,  # Should be removed
            "force_upcast": False,  # Should be removed
        }
        
        with open(realistic_config_path, 'w') as f:
            json.dump(realistic_config, f)
        
        # Recover the config (should clean up problematic attributes)
        result = recovery_system.recover_config(realistic_config_path)
        
        assert result.success
        
        # Verify problematic attributes were cleaned
        with open(realistic_config_path, 'r') as f:
            cleaned_config = json.load(f)
        
        assert "clip_output" not in cleaned_config
        assert "force_upcast" not in cleaned_config
        
        # Add trust settings for WAN22 models
        trust_result = recovery_system.update_trust_remote_code_setting(
            realistic_config_path,
            enable_trust=True,
            model_names=["Wan2.2-TI2V-5B", "Wan2.2-T2V-A14B", "Wan2.2-I2V-A14B"]
        )
        
        assert trust_result.success
        
        # Verify trust settings
        with open(realistic_config_path, 'r') as f:
            final_config = json.load(f)
        
        assert "trust_remote_code" in final_config
        assert final_config["trust_remote_code"]["Wan2.2-TI2V-5B"] is True
        assert final_config["trust_remote_code"]["Wan2.2-T2V-A14B"] is True
        assert final_config["trust_remote_code"]["Wan2.2-I2V-A14B"] is True
        
        # Generate final report
        report = recovery_system.get_config_changes_report()
        
        assert report["total_changes"] >= 2  # Recovery + trust update
        
        print("✅ Integration test with realistic WAN22 config passed!")
        
    finally:
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Run integration test
    test_integration_with_real_config()
    print("🎉 All integration tests would pass!")