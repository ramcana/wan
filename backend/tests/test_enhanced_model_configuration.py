"""
Tests for Enhanced Model Configuration Management System

Tests configuration management, validation, feature flags, and API endpoints.
"""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from backend.core.enhanced_model_config import (
    ConfigurationManager, EnhancedModelConfiguration,
    UserPreferences, AdminPolicies, FeatureFlagConfig,
    DownloadConfig, HealthMonitoringConfig, FallbackConfig,
    AnalyticsConfig, UpdateConfig, NotificationConfig, StorageConfig,
    AutomationLevel, FeatureFlag, get_config_manager, reset_config_manager
)
from backend.core.config_validation import ConfigurationValidator, ValidationResult


class TestEnhancedModelConfiguration:
    """Test configuration data classes and basic functionality"""
    
    def test_default_configuration_creation(self):
        """Test creating default configuration"""
        config = EnhancedModelConfiguration()
        
        assert config.version == "1.0.0"
        assert config.config_schema_version == "1.0.0"
        assert isinstance(config.user_preferences, UserPreferences)
        assert isinstance(config.admin_policies, AdminPolicies)
        assert isinstance(config.feature_flags, FeatureFlagConfig)
        assert isinstance(config.last_updated, datetime)
    
    def test_user_preferences_defaults(self):
        """Test default user preferences"""
        prefs = UserPreferences()
        
        assert prefs.automation_level == AutomationLevel.SEMI_AUTOMATIC
        assert isinstance(prefs.download_config, DownloadConfig)
        assert isinstance(prefs.health_monitoring, HealthMonitoringConfig)
        assert isinstance(prefs.fallback_config, FallbackConfig)
        assert isinstance(prefs.analytics_config, AnalyticsConfig)
        assert isinstance(prefs.update_config, UpdateConfig)
        assert isinstance(prefs.notification_config, NotificationConfig)
        assert isinstance(prefs.storage_config, StorageConfig)
        assert prefs.preferred_models == []
        assert prefs.blocked_models == []
    
    def test_download_config_defaults(self):
        """Test download configuration defaults"""
        config = DownloadConfig()
        
        assert config.max_retries == 3
        assert config.retry_delay_base == 2.0
        assert config.max_retry_delay == 300.0
        assert config.bandwidth_limit_mbps is None
        assert config.max_concurrent_downloads == 2
        assert config.enable_resume is True
        assert config.enable_pause is True
        assert config.chunk_size_mb == 10.0
        assert config.verify_integrity is True
        assert config.cleanup_failed_downloads is True
    
    def test_feature_flag_defaults(self):
        """Test feature flag defaults"""
        flags = FeatureFlagConfig()
        
        # Check that all feature flags are present
        for flag in FeatureFlag:
            assert flag.value in flags.flags
        
        # Check specific defaults
        assert flags.flags[FeatureFlag.ENHANCED_DOWNLOADS.value] is True
        assert flags.flags[FeatureFlag.HEALTH_MONITORING.value] is True
        assert flags.flags[FeatureFlag.INTELLIGENT_FALLBACK.value] is True
        assert flags.flags[FeatureFlag.USAGE_ANALYTICS.value] is True
        assert flags.flags[FeatureFlag.AUTO_UPDATES.value] is False
        assert flags.flags[FeatureFlag.REAL_TIME_NOTIFICATIONS.value] is True
        
        assert flags.rollout_percentage == {}
        assert flags.user_overrides == {}


class TestConfigurationManager:
    """Test configuration manager functionality"""
    
    @pytest.fixture
    def temp_config_file(self):
        """Create temporary config file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        yield config_path
        Path(config_path).unlink(missing_ok=True)
    
    @pytest.fixture
    def config_manager(self, temp_config_file):
        """Create configuration manager with temporary file"""
        manager = ConfigurationManager(temp_config_file)
        yield manager
        # Cleanup
        Path(temp_config_file).unlink(missing_ok=True)
    
    def test_configuration_manager_initialization(self, config_manager):
        """Test configuration manager initialization"""
        assert isinstance(config_manager.config, EnhancedModelConfiguration)
        assert config_manager.config_path.exists()
    
    def test_save_and_load_configuration(self, config_manager):
        """Test saving and loading configuration"""
        # Modify configuration
        config_manager.config.user_preferences.automation_level = AutomationLevel.FULLY_AUTOMATIC
        config_manager.config.user_preferences.download_config.max_retries = 5
        
        # Save configuration
        assert config_manager.save_configuration() is True
        
        # Create new manager with same file
        new_manager = ConfigurationManager(str(config_manager.config_path))
        
        # Verify loaded configuration
        assert new_manager.config.user_preferences.automation_level == AutomationLevel.FULLY_AUTOMATIC
        assert new_manager.config.user_preferences.download_config.max_retries == 5
    
    @pytest.mark.asyncio
    async def test_update_user_preferences(self, config_manager):
        """Test updating user preferences"""
        new_prefs = UserPreferences()
        new_prefs.automation_level = AutomationLevel.FULLY_AUTOMATIC
        new_prefs.download_config.max_retries = 5
        
        success = await config_manager.update_user_preferences(new_prefs)
        assert success is True
        
        # Verify update
        assert config_manager.config.user_preferences.automation_level == AutomationLevel.FULLY_AUTOMATIC
        assert config_manager.config.user_preferences.download_config.max_retries == 5
    
    @pytest.mark.asyncio
    async def test_update_admin_policies(self, config_manager):
        """Test updating admin policies"""
        new_policies = AdminPolicies()
        new_policies.max_user_storage_gb = 100.0
        new_policies.require_approval_for_updates = True
        
        success = await config_manager.update_admin_policies(new_policies)
        assert success is True
        
        # Verify update
        assert config_manager.config.admin_policies.max_user_storage_gb == 100.0
        assert config_manager.config.admin_policies.require_approval_for_updates is True
    
    @pytest.mark.asyncio
    async def test_update_feature_flag(self, config_manager):
        """Test updating feature flags"""
        # Test global flag update
        success = await config_manager.update_feature_flag(FeatureFlag.AUTO_UPDATES, True)
        assert success is True
        assert config_manager.config.feature_flags.flags[FeatureFlag.AUTO_UPDATES.value] is True
        
        # Test user-specific override
        success = await config_manager.update_feature_flag(FeatureFlag.ENHANCED_DOWNLOADS, False, "user123")
        assert success is True
        assert config_manager.config.feature_flags.user_overrides["user123"][FeatureFlag.ENHANCED_DOWNLOADS.value] is False
    
    def test_is_feature_enabled(self, config_manager):
        """Test feature flag checking"""
        # Test global flag
        assert config_manager.is_feature_enabled(FeatureFlag.ENHANCED_DOWNLOADS) is True
        assert config_manager.is_feature_enabled(FeatureFlag.AUTO_UPDATES) is False
        
        # Test user-specific override
        config_manager.config.feature_flags.user_overrides["user123"] = {
            FeatureFlag.ENHANCED_DOWNLOADS.value: False
        }
        
        assert config_manager.is_feature_enabled(FeatureFlag.ENHANCED_DOWNLOADS, "user123") is False
        assert config_manager.is_feature_enabled(FeatureFlag.ENHANCED_DOWNLOADS, "user456") is True
    
    def test_rollout_percentage_feature_flags(self, config_manager):
        """Test A/B testing with rollout percentages"""
        # Set 50% rollout for a feature
        config_manager.config.feature_flags.rollout_percentage[FeatureFlag.AUTO_UPDATES.value] = 50.0
        config_manager.config.feature_flags.flags[FeatureFlag.AUTO_UPDATES.value] = True
        
        # Test with different user IDs (hash-based assignment should be consistent)
        user1_enabled = config_manager.is_feature_enabled(FeatureFlag.AUTO_UPDATES, "user1")
        user2_enabled = config_manager.is_feature_enabled(FeatureFlag.AUTO_UPDATES, "user2")
        
        # Same user should get consistent result
        assert config_manager.is_feature_enabled(FeatureFlag.AUTO_UPDATES, "user1") == user1_enabled
        assert config_manager.is_feature_enabled(FeatureFlag.AUTO_UPDATES, "user2") == user2_enabled
    
    @pytest.mark.asyncio
    async def test_configuration_observers(self, config_manager):
        """Test configuration change observers"""
        observer_calls = []
        
        async def test_observer(change_type, section, old_value, new_value):
            observer_calls.append((change_type, section, old_value, new_value))
        
        config_manager.add_observer(test_observer)
        
        # Update preferences
        new_prefs = UserPreferences()
        new_prefs.automation_level = AutomationLevel.FULLY_AUTOMATIC
        
        await config_manager.update_user_preferences(new_prefs)
        
        # Verify observer was called
        assert len(observer_calls) == 1
        assert observer_calls[0][0] == "user_preferences"
        assert observer_calls[0][1] == "preferences"
    
    def test_admin_constraints_application(self, config_manager):
        """Test application of admin constraints to user preferences"""
        # Set admin constraints
        config_manager.config.admin_policies.max_user_storage_gb = 50.0
        config_manager.config.admin_policies.bandwidth_limit_per_user_mbps = 10.0
        
        # Create user preferences that exceed constraints
        prefs = UserPreferences()
        prefs.storage_config.max_storage_gb = 100.0  # Exceeds admin limit
        prefs.download_config.bandwidth_limit_mbps = 20.0  # Exceeds admin limit
        
        # Apply constraints
        constrained_prefs = config_manager._apply_admin_constraints(prefs)
        
        # Verify constraints were applied
        assert constrained_prefs.storage_config.max_storage_gb == 50.0
        assert constrained_prefs.download_config.bandwidth_limit_mbps == 10.0


class TestConfigurationValidation:
    """Test configuration validation functionality"""
    
    @pytest.fixture
    def validator(self):
        """Create configuration validator"""
        return ConfigurationValidator()
    
    def test_valid_user_preferences(self, validator):
        """Test validation of valid user preferences"""
        prefs = UserPreferences()
        result = validator.validate_user_preferences(prefs)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_invalid_download_config(self, validator):
        """Test validation of invalid download configuration"""
        prefs = UserPreferences()
        prefs.download_config.max_retries = -1  # Invalid
        prefs.download_config.retry_delay_base = 100.0  # Greater than max_retry_delay
        prefs.download_config.max_retry_delay = 50.0
        
        result = validator.validate_user_preferences(prefs)
        
        assert result.is_valid is False
        assert len(result.errors) >= 2
        
        # Check specific errors
        error_fields = [error.field for error in result.errors]
        assert any("max_retries" in field for field in error_fields)
        assert any("retry_delay_base" in field for field in error_fields)
    
    def test_invalid_health_monitoring_config(self, validator):
        """Test validation of invalid health monitoring configuration"""
        prefs = UserPreferences()
        prefs.health_monitoring.corruption_threshold = 1.5  # > 1.0
        prefs.health_monitoring.performance_degradation_threshold = -0.1  # < 0.0
        
        result = validator.validate_user_preferences(prefs)
        
        assert result.is_valid is False
        assert len(result.errors) >= 2
    
    def test_invalid_model_lists(self, validator):
        """Test validation of invalid model lists"""
        prefs = UserPreferences()
        prefs.preferred_models = ["valid-model", "invalid/model/name!"]
        prefs.blocked_models = ["valid-model"]  # Overlap with preferred
        
        result = validator.validate_user_preferences(prefs)
        
        assert result.is_valid is False
        assert len(result.errors) >= 2
    
    def test_valid_admin_policies(self, validator):
        """Test validation of valid admin policies"""
        policies = AdminPolicies()
        result = validator.validate_admin_policies(policies)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_invalid_admin_policies(self, validator):
        """Test validation of invalid admin policies"""
        policies = AdminPolicies()
        policies.max_user_storage_gb = -10.0  # Invalid
        policies.bandwidth_limit_per_user_mbps = 0.0  # Invalid
        policies.blocked_model_patterns = ["[invalid_regex"]  # Invalid regex
        
        result = validator.validate_admin_policies(policies)
        
        assert result.is_valid is False
        assert len(result.errors) >= 3
    
    def test_feature_flag_validation(self, validator):
        """Test validation of feature flags"""
        flags = FeatureFlagConfig()
        flags.flags["invalid_flag"] = True  # Unknown flag
        flags.rollout_percentage["test_flag"] = 150.0  # Invalid percentage
        
        result = validator.validate_feature_flags(flags)
        
        assert result.is_valid is False
        assert len(result.errors) >= 1
        assert len(result.warnings) >= 1


class TestConfigurationMigration:
    """Test configuration migration functionality"""
    
    @pytest.fixture
    def temp_config_file(self):
        """Create temporary config file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        yield config_path
        Path(config_path).unlink(missing_ok=True)
    
    def test_migration_from_unversioned(self, temp_config_file):
        """Test migration from unversioned configuration"""
        # Create old format configuration
        old_config = {
            "version": "1.0.0",
            "user_preferences": {
                "automation_level": "semi_automatic",
                "download_config": {
                    "max_retries": 3
                }
            }
            # Missing feature_flags and admin_policies
        }
        
        with open(temp_config_file, 'w') as f:
            json.dump(old_config, f)
        
        # Load with configuration manager (should trigger migration)
        manager = ConfigurationManager(temp_config_file)
        
        # Verify migration added missing sections
        assert hasattr(manager.config, 'feature_flags')
        assert hasattr(manager.config, 'admin_policies')
        assert manager.config.config_schema_version == "1.0.0"
    
    def test_backup_creation(self, temp_config_file):
        """Test that backups are created when saving configuration"""
        manager = ConfigurationManager(temp_config_file)
        
        # Save configuration multiple times
        manager.save_configuration()
        manager.save_configuration()
        
        # Check that backup files were created
        backup_dir = Path(temp_config_file).parent
        backups = list(backup_dir.glob(f'{Path(temp_config_file).stem}.backup.*.json'))
        
        assert len(backups) >= 1


class TestGlobalConfigurationManager:
    """Test global configuration manager functionality"""
    
    def test_get_config_manager_singleton(self):
        """Test that get_config_manager returns singleton"""
        reset_config_manager()  # Ensure clean state
        
        manager1 = get_config_manager()
        manager2 = get_config_manager()
        
        assert manager1 is manager2
    
    def test_reset_config_manager(self):
        """Test resetting global configuration manager"""
        manager1 = get_config_manager()
        reset_config_manager()
        manager2 = get_config_manager()
        
        assert manager1 is not manager2


@pytest.mark.asyncio
class TestConfigurationAPI:
    """Test configuration API endpoints"""
    
    @pytest.fixture
    def mock_config_manager(self):
        """Mock configuration manager for API testing"""
        manager = Mock(spec=ConfigurationManager)
        manager.config = Mock()
        manager.config.last_updated = datetime.now()
        manager.config.version = "1.0.0"
        manager.config.config_schema_version = "1.0.0"
        
        # Mock methods
        manager.get_user_preferences.return_value = UserPreferences()
        manager.get_admin_policies.return_value = AdminPolicies()
        manager.update_user_preferences = AsyncMock(return_value=True)
        manager.update_admin_policies = AsyncMock(return_value=True)
        manager.update_feature_flag = AsyncMock(return_value=True)
        manager.is_feature_enabled.return_value = True
        manager.validate_user_preferences.return_value = ValidationResult(True, [], [])
        manager.validate_admin_policies.return_value = ValidationResult(True, [], [])
        
        return manager
    
    @patch('backend.api.enhanced_model_configuration.get_config_manager')
    async def test_get_user_preferences_endpoint(self, mock_get_manager, mock_config_manager):
        """Test get user preferences API endpoint"""
        mock_get_manager.return_value = mock_config_manager
        
        from backend.api.enhanced_model_configuration import get_user_preferences
        
        result = await get_user_preferences("test_user")
        
        assert "user_id" in result
        assert "preferences" in result
        assert "last_updated" in result
        assert result["user_id"] == "test_user"
    
    @patch('backend.api.enhanced_model_configuration.get_config_manager')
    async def test_update_user_preferences_endpoint(self, mock_get_manager, mock_config_manager):
        """Test update user preferences API endpoint"""
        mock_get_manager.return_value = mock_config_manager
        
        from backend.api.enhanced_model_configuration import update_user_preferences
        
        preferences_data = {
            "automation_level": "fully_automatic",
            "download_config": {
                "max_retries": 5
            }
        }
        
        result = await update_user_preferences(preferences_data, "test_user")
        
        assert result["success"] is True
        assert "message" in result
        assert result["user_id"] == "test_user"
        
        # Verify manager was called
        mock_config_manager.update_user_preferences.assert_called_once()
    
    @patch('backend.api.enhanced_model_configuration.get_config_manager')
    async def test_update_feature_flag_endpoint(self, mock_get_manager, mock_config_manager):
        """Test update feature flag API endpoint"""
        mock_get_manager.return_value = mock_config_manager
        
        from backend.api.enhanced_model_configuration import update_feature_flag
        
        result = await update_feature_flag("enhanced_downloads", True, False, "test_user", True)
        
        assert result["success"] is True
        assert result["flag"] == "enhanced_downloads"
        assert result["enabled"] is True
        
        # Verify manager was called
        mock_config_manager.update_feature_flag.assert_called_once()
    
    @patch('backend.api.enhanced_model_configuration.get_config_manager')
    async def test_validate_preferences_endpoint(self, mock_get_manager, mock_config_manager):
        """Test validate preferences API endpoint"""
        mock_get_manager.return_value = mock_config_manager
        
        from backend.api.enhanced_model_configuration import validate_user_preferences
        
        preferences_data = {
            "automation_level": "semi_automatic",
            "download_config": {
                "max_retries": 3
            }
        }
        
        result = await validate_user_preferences(preferences_data, "test_user")
        
        assert "is_valid" in result
        assert "errors" in result
        assert "warnings" in result
        assert result["is_valid"] is True
        
        # Verify manager was called
        mock_config_manager.validate_user_preferences.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])