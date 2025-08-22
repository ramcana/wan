#!/usr/bin/env python3
"""
Validation script for Task 5.3: Create configuration recovery system

This script validates that all requirements for task 5.3 have been implemented:
- Implement restoration from known good defaults for corrupted configs
- Create configuration change reporting system
- Add validation for trust_remote_code handling
"""

import json
import tempfile
from pathlib import Path
from config_recovery_system import (
    ConfigRecoverySystem,
    RecoveryAction,
    recover_config,
    update_trust_remote_code,
    format_recovery_result
)

def validate_task_5_3_requirements():
    """Validate all requirements for task 5.3 are implemented"""
    print("🔍 VALIDATING TASK 5.3: Create configuration recovery system")
    print("=" * 70)
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        recovery_system = ConfigRecoverySystem(
            backup_dir=temp_dir / "backups",
            recovery_dir=temp_dir / "recovery"
        )
        
        # ✅ REQUIREMENT 1: Restoration from known good defaults for corrupted configs
        print("\n1️⃣ REQUIREMENT: Restoration from known good defaults for corrupted configs")
        print("-" * 70)
        
        # Test 1.1: Completely corrupted config
        corrupted_config_path = temp_dir / "corrupted.json"
        with open(corrupted_config_path, 'w') as f:
            f.write("{ invalid json content")
        
        result = recovery_system.recover_config(corrupted_config_path)
        
        assert result.success, "❌ Failed to recover corrupted config"
        assert result.action_taken == RecoveryAction.RESTORE_FROM_DEFAULTS, "❌ Wrong recovery action"
        
        # Verify restored config has all required sections
        with open(corrupted_config_path, 'r') as f:
            restored_config = json.load(f)
        
        required_sections = ["system", "directories", "generation", "models", "optimization"]
        for section in required_sections:
            assert section in restored_config, f"❌ Missing required section: {section}"
        
        print("✅ Successfully restored completely corrupted config from defaults")
        
        # Test 1.2: Partially corrupted config with invalid values
        partial_config_path = temp_dir / "partial.json"
        partial_config = {
            "system": {
                "default_quantization": "invalid_quantization_type",
                "enable_offload": "not_a_boolean",
                "vae_tile_size": "not_a_number"
            },
            "directories": {
                "output_directory": None
            }
            # Missing other required sections
        }
        
        with open(partial_config_path, 'w') as f:
            json.dump(partial_config, f)
        
        result = recovery_system.recover_config(partial_config_path)
        
        assert result.success, "❌ Failed to recover partially corrupted config"
        assert result.action_taken in [RecoveryAction.RESTORE_FROM_DEFAULTS, RecoveryAction.MERGE_WITH_DEFAULTS], "❌ Wrong recovery action"
        
        # Verify restored config has valid values
        with open(partial_config_path, 'r') as f:
            restored_config = json.load(f)
        
        assert restored_config["system"]["default_quantization"] in ["fp16", "bf16", "int8", "none"], "❌ Invalid quantization value"
        assert isinstance(restored_config["system"]["enable_offload"], bool), "❌ Invalid enable_offload type"
        assert isinstance(restored_config["system"]["vae_tile_size"], int), "❌ Invalid vae_tile_size type"
        
        print("✅ Successfully restored partially corrupted config with value correction")
        
        # Test 1.3: Config with problematic attributes that need cleanup
        cleanup_config_path = temp_dir / "cleanup.json"
        cleanup_config = {
            "system": {
                "default_quantization": "bf16",
                "enable_offload": True,
                "vae_tile_size": 256
            },
            "directories": {
                "output_directory": "outputs",
                "models_directory": "models"
            },
            # Problematic attributes that should be removed
            "clip_output": True,
            "force_upcast": False,
            "use_linear_projection": True,
            "cross_attention_dim": 768
        }
        
        with open(cleanup_config_path, 'w') as f:
            json.dump(cleanup_config, f)
        
        result = recovery_system.recover_config(cleanup_config_path)
        
        assert result.success, "❌ Failed to clean up problematic attributes"
        assert len(result.changes_made) > 0, "❌ No cleanup changes reported"
        
        # Verify problematic attributes were removed
        with open(cleanup_config_path, 'r') as f:
            cleaned_config = json.load(f)
        
        problematic_attrs = ["clip_output", "force_upcast", "use_linear_projection", "cross_attention_dim"]
        for attr in problematic_attrs:
            assert attr not in cleaned_config, f"❌ Problematic attribute not removed: {attr}"
        
        print("✅ Successfully cleaned up problematic attributes from config")
        
        # ✅ REQUIREMENT 2: Configuration change reporting system
        print("\n2️⃣ REQUIREMENT: Configuration change reporting system")
        print("-" * 70)
        
        # Test 2.1: Changes are logged and can be retrieved
        initial_changes = len(recovery_system.changes_history)
        
        # Make a change
        test_config_path = temp_dir / "change_test.json"
        with open(test_config_path, 'w') as f:
            json.dump({"incomplete": "config"}, f)
        
        recovery_system.recover_config(test_config_path)
        
        assert len(recovery_system.changes_history) > initial_changes, "❌ Changes not logged"
        
        print("✅ Configuration changes are properly logged")
        
        # Test 2.2: Comprehensive change reporting
        report = recovery_system.get_config_changes_report()
        
        required_report_fields = ["total_changes", "recent_changes", "changes_by_type", "detailed_changes", "generated_at"]
        for field in required_report_fields:
            assert field in report, f"❌ Missing report field: {field}"
        
        assert report["total_changes"] > 0, "❌ No changes reported"
        assert isinstance(report["changes_by_type"], dict), "❌ Invalid changes_by_type format"
        assert isinstance(report["detailed_changes"], list), "❌ Invalid detailed_changes format"
        
        print("✅ Comprehensive change reporting system working")
        
        # Test 2.3: File-specific change reporting
        file_report = recovery_system.get_config_changes_report(test_config_path)
        
        assert file_report["total_changes"] >= 1, "❌ File-specific report not working"
        
        # All changes should be for the specific file
        for change in file_report["detailed_changes"]:
            assert change["config_path"] == str(test_config_path), "❌ File-specific filtering not working"
        
        print("✅ File-specific change reporting working")
        
        # Test 2.4: Change history persistence
        changes_log_path = recovery_system.changes_log_path
        assert changes_log_path.exists(), "❌ Changes log file not created"
        
        with open(changes_log_path, 'r') as f:
            saved_changes = json.load(f)
        
        assert len(saved_changes) > 0, "❌ Changes not persisted to file"
        
        print("✅ Change history persistence working")
        
        # ✅ REQUIREMENT 3: Validation for trust_remote_code handling
        print("\n3️⃣ REQUIREMENT: Validation for trust_remote_code handling")
        print("-" * 70)
        
        # Test 3.1: Add trust_remote_code settings for specific models
        trust_config_path = temp_dir / "trust_test.json"
        base_config = {
            "system": {"default_quantization": "bf16", "enable_offload": True},
            "directories": {"output_directory": "outputs", "models_directory": "models"}
        }
        
        with open(trust_config_path, 'w') as f:
            json.dump(base_config, f)
        
        models_to_trust = ["Wan2.2-TI2V-5B", "Wan2.2-T2V-A14B", "custom_model"]
        result = recovery_system.update_trust_remote_code_setting(
            trust_config_path,
            enable_trust=True,
            model_names=models_to_trust
        )
        
        assert result.success, "❌ Failed to update trust_remote_code settings"
        assert result.action_taken == RecoveryAction.UPDATE_TRUST_REMOTE_CODE, "❌ Wrong action for trust update"
        assert len(result.changes_made) == len(models_to_trust), "❌ Wrong number of trust changes"
        
        # Verify trust settings were added
        with open(trust_config_path, 'r') as f:
            trust_config = json.load(f)
        
        assert "trust_remote_code" in trust_config, "❌ trust_remote_code section not added"
        
        for model in models_to_trust:
            assert model in trust_config["trust_remote_code"], f"❌ Trust setting not added for model: {model}"
            assert trust_config["trust_remote_code"][model] is True, f"❌ Wrong trust value for model: {model}"
        
        print("✅ Successfully added trust_remote_code settings for specific models")
        
        # Test 3.2: Global trust_remote_code setting
        result = recovery_system.update_trust_remote_code_setting(
            trust_config_path,
            enable_trust=False  # Global setting (no model_names)
        )
        
        assert result.success, "❌ Failed to update global trust_remote_code setting"
        
        # Verify global setting was added
        with open(trust_config_path, 'r') as f:
            trust_config = json.load(f)
        
        assert "global" in trust_config["trust_remote_code"], "❌ Global trust setting not added"
        assert trust_config["trust_remote_code"]["global"] is False, "❌ Wrong global trust value"
        
        print("✅ Successfully added global trust_remote_code setting")
        
        # Test 3.3: Trust settings validation and backup
        assert result.backup_path is not None, "❌ Backup not created for trust update"
        assert Path(result.backup_path).exists(), "❌ Trust update backup file not found"
        
        print("✅ Trust settings validation and backup working")
        
        # Test 3.4: Trust settings change logging
        trust_changes = [
            change for change in recovery_system.changes_history
            if change["change_type"] == "update_trust_remote_code"
        ]
        
        assert len(trust_changes) >= 2, "❌ Trust changes not properly logged"
        
        print("✅ Trust settings changes properly logged")
        
        # ✅ ADDITIONAL VALIDATION: Convenience functions and integration
        print("\n4️⃣ ADDITIONAL: Convenience functions and integration")
        print("-" * 70)
        
        # Test convenience functions
        convenience_config_path = temp_dir / "convenience.json"
        with open(convenience_config_path, 'w') as f:
            json.dump({"incomplete": "config"}, f)
        
        # Test recover_config convenience function
        conv_result = recover_config(convenience_config_path)
        assert conv_result.success, "❌ Convenience recover_config function failed"
        
        # Test update_trust_remote_code convenience function
        trust_conv_result = update_trust_remote_code(convenience_config_path, True, ["conv_model"])
        assert trust_conv_result.success, "❌ Convenience update_trust_remote_code function failed"
        
        # Test format_recovery_result function
        formatted = format_recovery_result(trust_conv_result)
        assert isinstance(formatted, str), "❌ format_recovery_result not returning string"
        assert len(formatted) > 0, "❌ format_recovery_result returning empty string"
        
        print("✅ Convenience functions working correctly")
        
        # ✅ FINAL VALIDATION: Error handling
        print("\n5️⃣ FINAL: Error handling validation")
        print("-" * 70)
        
        # Test error handling with non-existent file
        non_existent_path = temp_dir / "does_not_exist.json"
        error_result = recovery_system.recover_config(non_existent_path)
        
        assert not error_result.success, "❌ Should fail for non-existent file"
        assert len(error_result.messages) > 0, "❌ No error messages for failed recovery"
        
        print("✅ Error handling working correctly")
        
        print("\n🎉 TASK 5.3 VALIDATION COMPLETE")
        print("=" * 70)
        print("✅ ALL REQUIREMENTS SUCCESSFULLY IMPLEMENTED!")
        print("\n📋 Summary of implemented features:")
        print("• ✅ Restoration from known good defaults for corrupted configs")
        print("  - Complete config restoration from defaults")
        print("  - Partial config merging with defaults")
        print("  - Problematic attribute cleanup")
        print("• ✅ Configuration change reporting system")
        print("  - Comprehensive change logging")
        print("  - File-specific change reports")
        print("  - Change history persistence")
        print("  - Detailed change categorization")
        print("• ✅ Validation for trust_remote_code handling")
        print("  - Model-specific trust settings")
        print("  - Global trust settings")
        print("  - Trust settings validation and backup")
        print("  - Trust change logging")
        print("• ✅ Additional features:")
        print("  - Convenience functions for easy integration")
        print("  - Comprehensive error handling")
        print("  - Formatted output for user-friendly reporting")
        print("  - Backup and restore functionality")
        
        return True
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    success = validate_task_5_3_requirements()
    if success:
        print("\n🏆 TASK 5.3 IMPLEMENTATION VERIFIED SUCCESSFULLY!")
    else:
        print("\n💥 TASK 5.3 IMPLEMENTATION VALIDATION FAILED!")
        exit(1)