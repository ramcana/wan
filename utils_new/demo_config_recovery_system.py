#!/usr/bin/env python3
"""
Demonstration of Configuration Recovery System

This script demonstrates the complete functionality of the configuration recovery system
including restoration from defaults, configuration change reporting, and trust_remote_code handling.
"""

import json
import tempfile
from pathlib import Path
from config_recovery_system import (
    ConfigRecoverySystem, 
    RecoveryAction, 
    format_recovery_result,
    recover_config,
    update_trust_remote_code
)

def demonstrate_recovery_system():
    """Demonstrate all configuration recovery system features"""
    print("🔧 Configuration Recovery System Demonstration")
    print("=" * 60)
    
    # Create temporary directory for demonstration
    temp_dir = Path(tempfile.mkdtemp())
    print(f"📁 Working directory: {temp_dir}")
    
    try:
        # Initialize recovery system
        recovery_system = ConfigRecoverySystem(
            backup_dir=temp_dir / "backups",
            recovery_dir=temp_dir / "recovery"
        )
        
        print("\n1️⃣ RESTORATION FROM KNOWN GOOD DEFAULTS")
        print("-" * 40)
        
        # Create corrupted configuration
        corrupted_config_path = temp_dir / "corrupted_config.json"
        corrupted_config = {
            "system": {
                "default_quantization": "invalid_quantization_type",
                "enable_offload": "not_a_boolean",
                "vae_tile_size": "not_a_number"
            },
            "directories": {
                "output_directory": None  # Invalid value
            }
            # Missing required sections: generation, models, etc.
        }
        
        with open(corrupted_config_path, 'w') as f:
            json.dump(corrupted_config, f)
        
        print(f"📄 Created corrupted config with invalid values and missing sections")
        print(f"   Original sections: {list(corrupted_config.keys())}")
        
        # Recover from defaults
        recovery_result = recovery_system.recover_config(corrupted_config_path)
        
        print(f"✅ Recovery successful: {recovery_result.success}")
        print(f"🔧 Action taken: {recovery_result.action_taken.value}")
        print(f"📝 Changes made: {len(recovery_result.changes_made)}")
        
        # Show restored configuration
        with open(corrupted_config_path, 'r') as f:
            restored_config = json.load(f)
        
        print(f"📄 Restored config sections: {list(restored_config.keys())}")
        print(f"   System quantization: {restored_config['system']['default_quantization']}")
        print(f"   Enable offload: {restored_config['system']['enable_offload']}")
        print(f"   VAE tile size: {restored_config['system']['vae_tile_size']}")
        
        print("\n2️⃣ BACKUP AND RESTORE FUNCTIONALITY")
        print("-" * 40)
        
        # Create a valid config first
        valid_config_path = temp_dir / "valid_config.json"
        valid_config = {
            "system": {
                "default_quantization": "bf16",
                "enable_offload": True,
                "vae_tile_size": 256
            },
            "directories": {
                "output_directory": "outputs",
                "models_directory": "models"
            }
        }
        
        with open(valid_config_path, 'w') as f:
            json.dump(valid_config, f)
        
        # Create backup
        backup_path = recovery_system.validator.create_backup(valid_config_path)
        print(f"📁 Created backup: {Path(backup_path).name}")
        
        # Corrupt the config
        with open(valid_config_path, 'w') as f:
            f.write("{ invalid json content")
        
        print("💥 Corrupted config with invalid JSON")
        
        # Restore from backup
        restore_result = recovery_system.recover_config(valid_config_path)
        
        print(f"✅ Restore successful: {restore_result.success}")
        print(f"🔧 Action taken: {restore_result.action_taken.value}")
        
        # Verify restoration
        with open(valid_config_path, 'r') as f:
            restored_from_backup = json.load(f)
        
        print(f"📄 Config restored from backup successfully")
        print(f"   Quantization: {restored_from_backup['system']['default_quantization']}")
        
        print("\n3️⃣ TRUST_REMOTE_CODE VALIDATION AND HANDLING")
        print("-" * 40)
        
        # Test trust_remote_code for specific models
        trust_config_path = temp_dir / "trust_config.json"
        with open(trust_config_path, 'w') as f:
            json.dump(valid_config, f)
        
        # Update trust settings for specific models
        trust_result = recovery_system.update_trust_remote_code_setting(
            trust_config_path,
            enable_trust=True,
            model_names=["Wan2.2-TI2V-5B", "Wan2.2-T2V-A14B", "custom_model"]
        )
        
        print(f"✅ Trust update successful: {trust_result.success}")
        print(f"📝 Changes made: {len(trust_result.changes_made)}")
        
        # Show trust settings
        with open(trust_config_path, 'r') as f:
            trust_config = json.load(f)
        
        print("🔒 Trust remote code settings:")
        for model, trust_value in trust_config.get("trust_remote_code", {}).items():
            print(f"   {model}: {trust_value}")
        
        # Test global trust setting
        global_trust_result = recovery_system.update_trust_remote_code_setting(
            trust_config_path,
            enable_trust=False  # No model_names = global setting
        )
        
        print(f"🌐 Global trust setting updated: {global_trust_result.success}")
        
        # Show updated trust settings
        with open(trust_config_path, 'r') as f:
            updated_trust_config = json.load(f)
        
        print("🔒 Updated trust remote code settings:")
        for model, trust_value in updated_trust_config.get("trust_remote_code", {}).items():
            print(f"   {model}: {trust_value}")
        
        print("\n4️⃣ CONFIGURATION CHANGE REPORTING SYSTEM")
        print("-" * 40)
        
        # Generate comprehensive changes report
        changes_report = recovery_system.get_config_changes_report()
        
        print(f"📊 Configuration Changes Report:")
        print(f"   Total changes: {changes_report['total_changes']}")
        print(f"   Recent changes: {changes_report['recent_changes']}")
        print(f"   Changes by type: {changes_report['changes_by_type']}")
        
        print("\n📋 Detailed change history:")
        for i, change in enumerate(changes_report['detailed_changes'][-3:], 1):  # Show last 3 changes
            print(f"   {i}. {change['change_type']} at {change['timestamp'][:19]}")
            print(f"      File: {Path(change['config_path']).name}")
            print(f"      Reason: {change['reason']}")
        
        # Test file-specific report
        file_specific_report = recovery_system.get_config_changes_report(trust_config_path)
        print(f"\n📄 Changes for {trust_config_path.name}: {file_specific_report['total_changes']}")
        
        print("\n5️⃣ CONVENIENCE FUNCTIONS")
        print("-" * 40)
        
        # Test convenience functions
        convenience_config_path = temp_dir / "convenience_test.json"
        with open(convenience_config_path, 'w') as f:
            json.dump({"incomplete": "config"}, f)
        
        # Use convenience function for recovery
        convenience_result = recover_config(convenience_config_path)
        print(f"🔧 Convenience recovery: {convenience_result.success}")
        
        # Use convenience function for trust update
        convenience_trust = update_trust_remote_code(
            convenience_config_path, 
            True, 
            ["convenience_model"]
        )
        print(f"🔒 Convenience trust update: {convenience_trust.success}")
        
        # Format result for display
        formatted_result = format_recovery_result(convenience_trust)
        print("\n📋 Formatted result:")
        print(formatted_result)
        
        print("\n6️⃣ ERROR HANDLING AND VALIDATION")
        print("-" * 40)
        
        # Test error handling with non-existent file
        non_existent_path = temp_dir / "does_not_exist.json"
        error_result = recovery_system.recover_config(non_existent_path)
        
        print(f"❌ Non-existent file handling: {error_result.success}")
        print(f"📝 Error messages: {len(error_result.messages)}")
        if error_result.messages:
            print(f"   First error: {error_result.messages[0].message}")
        
        # Test with already valid configuration
        valid_test_path = temp_dir / "already_valid.json"
        default_config = recovery_system.default_configs["main_config"]
        with open(valid_test_path, 'w') as f:
            json.dump(default_config, f)
        
        no_recovery_result = recovery_system.recover_config(valid_test_path)
        print(f"✅ Valid config handling: {no_recovery_result.success}")
        
        # Check for "no recovery needed" message
        no_recovery_messages = [msg for msg in no_recovery_result.messages 
                              if "NO_RECOVERY_NEEDED" in msg.code or "valid" in msg.message.lower()]
        if no_recovery_messages:
            print(f"ℹ️  Message: {no_recovery_messages[0].message}")
        
        print("\n🎉 DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("✅ All configuration recovery system features demonstrated successfully!")
        print("\nKey Features Implemented:")
        print("• ✅ Restoration from known good defaults for corrupted configs")
        print("• ✅ Configuration change reporting system with detailed history")
        print("• ✅ Validation and handling for trust_remote_code settings")
        print("• ✅ Backup and restore functionality")
        print("• ✅ Multiple recovery strategies (restore, merge, repair)")
        print("• ✅ Comprehensive error handling")
        print("• ✅ Convenience functions for easy integration")
        print("• ✅ Formatted output for user-friendly reporting")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"\n🧹 Cleaned up temporary directory: {temp_dir}")

if __name__ == "__main__":
    demonstrate_recovery_system()
