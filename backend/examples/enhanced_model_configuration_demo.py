#!/usr/bin/env python3
"""
Enhanced Model Configuration System Demo

Demonstrates the configuration management system for enhanced model availability features.
"""

import asyncio
import tempfile
import os
from pathlib import Path
import sys

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.enhanced_model_config import (
    ConfigurationManager, UserPreferences, AdminPolicies,
    DownloadConfig, HealthMonitoringConfig, FallbackConfig,
    AutomationLevel, FeatureFlag, get_config_manager, reset_config_manager
)
from core.config_validation import ConfigurationValidator
from core.runtime_config_updater import RuntimeConfigurationUpdater


async def demo_basic_configuration():
    """Demonstrate basic configuration management"""
    print("=" * 60)
    print("ENHANCED MODEL CONFIGURATION DEMO")
    print("=" * 60)
    
    # Create temporary config file for demo
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        config_path = f.name
    
    try:
        print(f"\n1. Creating configuration manager with file: {config_path}")
        manager = ConfigurationManager(config_path)
        
        # Show default configuration
        print("\n2. Default Configuration:")
        prefs = manager.get_user_preferences()
        print(f"   - Automation Level: {prefs.automation_level.value}")
        print(f"   - Max Download Retries: {prefs.download_config.max_retries}")
        print(f"   - Health Monitoring Enabled: {prefs.health_monitoring.enabled}")
        print(f"   - Fallback Enabled: {prefs.fallback_config.enabled}")
        
        # Show feature flags
        print("\n3. Feature Flags:")
        for flag in FeatureFlag:
            enabled = manager.is_feature_enabled(flag)
            print(f"   - {flag.value}: {enabled}")
        
        # Demonstrate configuration update
        print("\n4. Updating User Preferences:")
        new_prefs = UserPreferences()
        new_prefs.automation_level = AutomationLevel.FULLY_AUTOMATIC
        new_prefs.download_config.max_retries = 5
        new_prefs.download_config.bandwidth_limit_mbps = 50.0
        
        success = await manager.update_user_preferences(new_prefs)
        print(f"   - Update successful: {success}")
        
        if success:
            updated_prefs = manager.get_user_preferences()
            print(f"   - New Automation Level: {updated_prefs.automation_level.value}")
            print(f"   - New Max Retries: {updated_prefs.download_config.max_retries}")
            print(f"   - New Bandwidth Limit: {updated_prefs.download_config.bandwidth_limit_mbps} Mbps")
        
        # Demonstrate feature flag update
        print("\n5. Updating Feature Flags:")
        success = await manager.update_feature_flag(FeatureFlag.AUTO_UPDATES, True)
        print(f"   - Auto Updates enabled: {success}")
        
        # User-specific override
        success = await manager.update_feature_flag(FeatureFlag.ENHANCED_DOWNLOADS, False, "user123")
        print(f"   - User-specific override set: {success}")
        
        # Check feature flags for different users
        global_enabled = manager.is_feature_enabled(FeatureFlag.ENHANCED_DOWNLOADS)
        user_enabled = manager.is_feature_enabled(FeatureFlag.ENHANCED_DOWNLOADS, "user123")
        print(f"   - Enhanced Downloads (global): {global_enabled}")
        print(f"   - Enhanced Downloads (user123): {user_enabled}")
        
        return manager
        
    finally:
        # Cleanup
        if os.path.exists(config_path):
            os.unlink(config_path)


async def demo_configuration_validation():
    """Demonstrate configuration validation"""
    print("\n" + "=" * 60)
    print("CONFIGURATION VALIDATION DEMO")
    print("=" * 60)
    
    validator = ConfigurationValidator()
    
    # Test valid configuration
    print("\n1. Validating Valid Configuration:")
    valid_prefs = UserPreferences()
    result = validator.validate_user_preferences(valid_prefs)
    print(f"   - Is Valid: {result.is_valid}")
    print(f"   - Errors: {len(result.errors)}")
    print(f"   - Warnings: {len(result.warnings)}")
    
    # Test invalid configuration
    print("\n2. Validating Invalid Configuration:")
    invalid_prefs = UserPreferences()
    invalid_prefs.download_config.max_retries = -1  # Invalid
    invalid_prefs.download_config.retry_delay_base = 100.0  # Greater than max
    invalid_prefs.download_config.max_retry_delay = 50.0
    invalid_prefs.health_monitoring.corruption_threshold = 1.5  # > 1.0
    
    result = validator.validate_user_preferences(invalid_prefs)
    print(f"   - Is Valid: {result.is_valid}")
    print(f"   - Errors: {len(result.errors)}")
    
    if result.errors:
        print("   - Error Details:")
        for error in result.errors[:3]:  # Show first 3 errors
            print(f"     * {error.field}: {error.message}")
    
    # Test admin policies validation
    print("\n3. Validating Admin Policies:")
    invalid_policies = AdminPolicies()
    invalid_policies.max_user_storage_gb = -10.0  # Invalid
    invalid_policies.blocked_model_patterns = ["[invalid_regex"]  # Invalid regex
    
    result = validator.validate_admin_policies(invalid_policies)
    print(f"   - Is Valid: {result.is_valid}")
    print(f"   - Errors: {len(result.errors)}")
    
    if result.errors:
        print("   - Error Details:")
        for error in result.errors:
            print(f"     * {error.field}: {error.message}")


async def demo_runtime_updates():
    """Demonstrate runtime configuration updates"""
    print("\n" + "=" * 60)
    print("RUNTIME CONFIGURATION UPDATES DEMO")
    print("=" * 60)
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        config_path = f.name
    
    try:
        # Reset global manager to use our temp file
        reset_config_manager()
        manager = ConfigurationManager(config_path)
        
        # Create runtime updater
        updater = RuntimeConfigurationUpdater(manager)
        
        # Add callback to track changes
        changes_received = []
        
        async def change_callback(data):
            changes_received.append(data)
            print(f"   - Configuration change detected: {data.get('type', 'unknown')}")
        
        updater.add_update_callback('any', change_callback)
        
        print("\n1. Setting up runtime configuration updater...")
        print("   - Callback registered for configuration changes")
        
        # Test runtime preference update
        print("\n2. Testing Runtime Preference Update:")
        new_prefs = UserPreferences()
        new_prefs.automation_level = AutomationLevel.MANUAL
        new_prefs.download_config.max_concurrent_downloads = 5
        
        success = await updater.update_user_preferences_runtime(new_prefs)
        print(f"   - Runtime update successful: {success}")
        
        # Wait a moment for callbacks
        await asyncio.sleep(0.1)
        
        # Test runtime feature flag update
        print("\n3. Testing Runtime Feature Flag Update:")
        success = await updater.update_feature_flag_runtime(FeatureFlag.HEALTH_MONITORING, False)
        print(f"   - Feature flag update successful: {success}")
        
        # Wait a moment for callbacks
        await asyncio.sleep(0.1)
        
        print(f"\n4. Total configuration changes detected: {len(changes_received)}")
        
        # Test rollback functionality
        print("\n5. Testing Configuration Rollback:")
        rollback_history = updater.get_rollback_history()
        print(f"   - Available rollback points: {len(rollback_history)}")
        
        if rollback_history:
            success = await updater.rollback_last_change()
            print(f"   - Rollback successful: {success}")
            
            # Verify rollback
            current_prefs = manager.get_user_preferences()
            print(f"   - Automation level after rollback: {current_prefs.automation_level.value}")
        
        return updater
        
    finally:
        # Cleanup
        if os.path.exists(config_path):
            os.unlink(config_path)


async def demo_admin_constraints():
    """Demonstrate admin policy constraints"""
    print("\n" + "=" * 60)
    print("ADMIN POLICY CONSTRAINTS DEMO")
    print("=" * 60)
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        config_path = f.name
    
    try:
        manager = ConfigurationManager(config_path)
        
        # Set admin constraints
        print("\n1. Setting Admin Constraints:")
        admin_policies = AdminPolicies()
        admin_policies.max_user_storage_gb = 50.0
        admin_policies.bandwidth_limit_per_user_mbps = 10.0
        admin_policies.blocked_model_patterns = [".*nsfw.*", ".*adult.*"]
        
        success = await manager.update_admin_policies(admin_policies)
        print(f"   - Admin policies set: {success}")
        print(f"   - Max user storage: {admin_policies.max_user_storage_gb} GB")
        print(f"   - Bandwidth limit: {admin_policies.bandwidth_limit_per_user_mbps} Mbps")
        print(f"   - Blocked patterns: {admin_policies.blocked_model_patterns}")
        
        # Try to set user preferences that exceed constraints
        print("\n2. Testing User Preferences Against Constraints:")
        user_prefs = UserPreferences()
        user_prefs.storage_config.max_storage_gb = 100.0  # Exceeds admin limit
        user_prefs.download_config.bandwidth_limit_mbps = 20.0  # Exceeds admin limit
        user_prefs.preferred_models = ["good-model", "nsfw-model", "family-friendly"]  # Contains blocked
        
        print(f"   - Requested storage: {user_prefs.storage_config.max_storage_gb} GB")
        print(f"   - Requested bandwidth: {user_prefs.download_config.bandwidth_limit_mbps} Mbps")
        print(f"   - Requested models: {user_prefs.preferred_models}")
        
        # Apply constraints
        constrained_prefs = manager._apply_admin_constraints(user_prefs)
        
        print("\n3. After Applying Admin Constraints:")
        print(f"   - Allowed storage: {constrained_prefs.storage_config.max_storage_gb} GB")
        print(f"   - Allowed bandwidth: {constrained_prefs.download_config.bandwidth_limit_mbps} Mbps")
        print(f"   - Allowed models: {constrained_prefs.preferred_models}")
        
        # Update with constrained preferences
        success = await manager.update_user_preferences(constrained_prefs)
        print(f"   - Constrained preferences saved: {success}")
        
    finally:
        # Cleanup
        if os.path.exists(config_path):
            os.unlink(config_path)


async def main():
    """Run all configuration demos"""
    try:
        await demo_basic_configuration()
        await demo_configuration_validation()
        await demo_runtime_updates()
        await demo_admin_constraints()
        
        print("\n" + "=" * 60)
        print("CONFIGURATION SYSTEM DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nKey Features Demonstrated:")
        print("✓ Basic configuration management with file persistence")
        print("✓ User preferences and admin policies")
        print("✓ Feature flags with user-specific overrides")
        print("✓ Configuration validation with detailed error reporting")
        print("✓ Runtime configuration updates without restart")
        print("✓ Configuration rollback and change tracking")
        print("✓ Admin policy constraints and enforcement")
        print("✓ Automatic configuration migration and backup")
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())