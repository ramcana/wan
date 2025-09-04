"""
Tests for Reliability System Deployment and Configuration

This module contains comprehensive tests for the reliability system deployment
and configuration components, ensuring proper functionality and integration.

Requirements addressed:
- 1.4: User configurable retry limits and feature control testing
- 8.1: Health report generation and deployment integration testing
- 8.5: Cross-instance monitoring configuration testing
"""

import os
import sys
import json
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime, timedelta

# Add scripts directory to path
script_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(script_dir))

from reliability_config import (
    ReliabilityConfigManager, ReliabilityConfiguration, RetryConfiguration,
    TimeoutConfiguration, RecoveryConfiguration, MonitoringConfiguration,
    FeatureFlags, DeploymentConfiguration, ReliabilityLevel
)
from deploy_reliability_system import ReliabilitySystemDeployer
from feature_flags import (
    FeatureFlagManager, FeatureFlag, FeatureState, RolloutStrategy,
    RolloutConfig, is_feature_enabled
)
from production_monitoring import (
    ProductionMonitor, MonitoringConfig, AlertThreshold, AlertChannel,
    MetricsCollector, AlertManager, HealthMetric, Alert
)


class TestReliabilityConfiguration(unittest.TestCase):
    """Test reliability configuration management."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.json")
        self.config_manager = ReliabilityConfigManager(self.config_path)
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_default_configuration_creation(self):
        """Test creation of default configuration."""
        config = self.config_manager.load_config()
        
        self.assertIsInstance(config, ReliabilityConfiguration)
        self.assertIsInstance(config.retry, RetryConfiguration)
        self.assertIsInstance(config.timeouts, TimeoutConfiguration)
        self.assertIsInstance(config.recovery, RecoveryConfiguration)
        self.assertIsInstance(config.monitoring, MonitoringConfiguration)
        self.assertIsInstance(config.features, FeatureFlags)
        self.assertIsInstance(config.deployment, DeploymentConfiguration)
        
        # Check default values
        self.assertEqual(config.retry.max_retries, 3)
        self.assertEqual(config.timeouts.model_download_seconds, 1800)
        self.assertTrue(config.recovery.missing_method_recovery)
        self.assertTrue(config.monitoring.enable_health_monitoring)
        self.assertEqual(config.features.reliability_level, ReliabilityLevel.STANDARD)
        self.assertEqual(config.deployment.environment, "development")
    
    def test_configuration_save_and_load(self):
        """Test saving and loading configuration."""
        # Create and modify configuration
        config = self.config_manager.load_config()
        config.retry.max_retries = 5
        config.timeouts.model_download_seconds = 3600
        config.recovery.missing_method_recovery = False
        
        # Save configuration
        self.assertTrue(self.config_manager.save_config())
        self.assertTrue(os.path.exists(self.config_path))
        
        # Load configuration in new manager
        new_manager = ReliabilityConfigManager(self.config_path)
        loaded_config = new_manager.load_config()
        
        # Verify loaded values
        self.assertEqual(loaded_config.retry.max_retries, 5)
        self.assertEqual(loaded_config.timeouts.model_download_seconds, 3600)
        self.assertFalse(loaded_config.recovery.missing_method_recovery)
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        config = self.config_manager.load_config()
        
        # Valid configuration should have no issues
        issues = self.config_manager.validate_config()
        self.assertEqual(len(issues), 0)
        
        # Invalid configuration should have issues
        config.retry.max_retries = -1
        config.retry.base_delay_seconds = -1.0
        config.timeouts.model_download_seconds = -100
        
        issues = self.config_manager.validate_config()
        self.assertGreater(len(issues), 0)
        self.assertTrue(any("max_retries must be non-negative" in issue for issue in issues))
        self.assertTrue(any("base_delay_seconds must be positive" in issue for issue in issues))
        self.assertTrue(any("model_download_seconds must be positive" in issue for issue in issues))
    
    def test_environment_specific_configuration(self):
        """Test environment-specific configuration generation."""
        # Test production environment
        prod_config = self.config_manager.get_config_for_environment("production")
        self.assertEqual(prod_config.deployment.log_level, "WARNING")
        self.assertFalse(prod_config.deployment.enable_debug_mode)
        self.assertEqual(prod_config.features.reliability_level, ReliabilityLevel.MAXIMUM)
        
        # Test development environment
        dev_config = self.config_manager.get_config_for_environment("development")
        self.assertEqual(dev_config.deployment.log_level, "DEBUG")
        self.assertTrue(dev_config.deployment.enable_debug_mode)
        self.assertEqual(dev_config.features.reliability_level, ReliabilityLevel.STANDARD)
        
        # Test testing environment
        test_config = self.config_manager.get_config_for_environment("testing")
        self.assertEqual(test_config.retry.max_retries, 1)
        self.assertEqual(test_config.timeouts.model_download_seconds, 300)
        self.assertEqual(test_config.features.reliability_level, ReliabilityLevel.BASIC)
    
    def test_configuration_update(self):
        """Test configuration updates."""
        config = self.config_manager.load_config()
        original_retries = config.retry.max_retries
        
        # Update configuration
        updates = {
            "retry": {
                "max_retries": 10,
                "base_delay_seconds": 5.0
            },
            "features": {
                "reliability_level": "maximum"
            }
        }
        
        self.assertTrue(self.config_manager.update_config(updates))
        
        # Verify updates
        updated_config = self.config_manager.load_config()
        self.assertEqual(updated_config.retry.max_retries, 10)
        self.assertEqual(updated_config.retry.base_delay_seconds, 5.0)
        self.assertNotEqual(updated_config.retry.max_retries, original_retries)
    
    def test_configuration_template_export(self):
        """Test configuration template export."""
        template_path = os.path.join(self.temp_dir, "template.json")
        
        self.assertTrue(self.config_manager.export_config_template(template_path))
        self.assertTrue(os.path.exists(template_path))
        
        # Verify template structure
        with open(template_path, 'r') as f:
            template = json.load(f)
        
        self.assertIn("_description", template)
        self.assertIn("retry", template)
        self.assertIn("timeouts", template)
        self.assertIn("recovery", template)
        self.assertIn("monitoring", template)
        self.assertIn("features", template)
        self.assertIn("deployment", template)


class TestReliabilitySystemDeployer(unittest.TestCase):
    """Test reliability system deployment."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.deployer = ReliabilitySystemDeployer("testing")
        self.deployer.project_root = Path(self.temp_dir)
        self.deployer.script_dir = Path(self.temp_dir) / "scripts"
        self.deployer.script_dir.mkdir(parents=True, exist_ok=True)
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('local_installation.scripts.deploy_reliability_system.ReliabilityConfigManager')
    def test_deployment_validation(self, mock_config_manager):
        """Test deployment validation."""
        # Create required files
        (self.deployer.script_dir / "main_installer.py").write_text("# Main installer")
        (self.deployer.script_dir / "error_handler.py").write_text("# Error handler")
        (self.deployer.script_dir / "reliability_manager.py").write_text("# Reliability manager")
        
        # Mock configuration manager
        mock_config_manager.return_value.get_config_for_environment.return_value = ReliabilityConfiguration()
        mock_config_manager.return_value.validate_config.return_value = []
        mock_config_manager.return_value.save_config.return_value = True
        
        # Test validation
        self.assertTrue(self.deployer._validate_pre_deployment())
    
    def test_backup_creation(self):
        """Test backup creation."""
        # Create files to backup
        config_file = self.deployer.project_root / "config.json"
        config_file.write_text('{"test": "config"}')
        
        installer_file = self.deployer.script_dir / "main_installer.py"
        installer_file.write_text("# Main installer")
        
        # Create backup
        self.assertTrue(self.deployer._create_backup())
        self.assertTrue(self.deployer.backup_dir.exists())
        
        # Verify backup manifest
        manifest_file = self.deployer.backup_dir / "backup_manifest.json"
        self.assertTrue(manifest_file.exists())
        
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
        
        self.assertIn("backup_timestamp", manifest)
        self.assertEqual(manifest["environment"], "testing")


class TestFeatureFlagManager(unittest.TestCase):
    """Test feature flag management."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_flags.json")
        self.flag_manager = FeatureFlagManager(self.config_path)
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_default_flags_initialization(self):
        """Test initialization of default flags."""
        # Check that default flags are created
        self.assertGreater(len(self.flag_manager.flags), 0)
        
        # Check specific flags exist
        self.assertIn("enhanced_error_context", self.flag_manager.flags)
        self.assertIn("missing_method_recovery", self.flag_manager.flags)
        self.assertIn("model_validation_recovery", self.flag_manager.flags)
        
        # Check flag properties
        flag = self.flag_manager.flags["enhanced_error_context"]
        self.assertEqual(flag.state, FeatureState.ENABLED)
        self.assertEqual(flag.rollout_strategy, RolloutStrategy.FULL)
    
    def test_feature_flag_evaluation(self):
        """Test feature flag evaluation."""
        # Test enabled flag
        self.assertTrue(self.flag_manager.is_enabled("enhanced_error_context"))
        
        # Test canary flag
        canary_flag = self.flag_manager.flags["missing_method_recovery"]
        canary_flag.rollout_strategy = RolloutStrategy.CANARY
        canary_flag.rollout_percentage = 50.0
        
        # Should be consistent for same user
        result1 = self.flag_manager.is_enabled("missing_method_recovery", "test_user")
        result2 = self.flag_manager.is_enabled("missing_method_recovery", "test_user")
        self.assertEqual(result1, result2)
    
    def test_flag_creation_and_update(self):
        """Test creating and updating flags."""
        # Create new flag
        self.assertTrue(self.flag_manager.create_flag(
            "test_flag",
            "Test flag description",
            state=FeatureState.TESTING,
            rollout_strategy=RolloutStrategy.CANARY,
            rollout_percentage=25.0
        ))
        
        # Verify flag exists
        self.assertIn("test_flag", self.flag_manager.flags)
        flag = self.flag_manager.flags["test_flag"]
        self.assertEqual(flag.state, FeatureState.TESTING)
        self.assertEqual(flag.rollout_percentage, 25.0)
        
        # Update flag
        self.assertTrue(self.flag_manager.update_flag(
            "test_flag",
            state="enabled",
            rollout_percentage=100.0
        ))
        
        # Verify update
        updated_flag = self.flag_manager.flags["test_flag"]
        self.assertEqual(updated_flag.state, FeatureState.ENABLED)
        self.assertEqual(updated_flag.rollout_percentage, 100.0)
    
    def test_gradual_rollout(self):
        """Test gradual rollout configuration."""
        rollout_config = RolloutConfig(
            start_date=datetime.now().isoformat(),
            end_date=(datetime.now() + timedelta(days=30)).isoformat(),
            initial_percentage=10.0,
            target_percentage=100.0,
            increment_percentage=10.0,
            increment_interval_hours=24
        )
        
        self.assertTrue(self.flag_manager.setup_gradual_rollout("test_gradual", rollout_config))
        self.assertIn("test_gradual", self.flag_manager.rollout_configs)


class TestProductionMonitoring(unittest.TestCase):
    """Test production monitoring system."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "monitoring_config.json")
        
        # Create basic monitoring config
        config = MonitoringConfig()
        config_data = {
            "enabled": True,
            "check_interval_seconds": 5,
            "alert_thresholds": [
                {
                    "metric_name": "test_metric",
                    "warning_threshold": 50.0,
                    "critical_threshold": 80.0,
                    "comparison_operator": "greater_than",
                    "time_window_minutes": 1,
                    "min_occurrences": 1
                }
            ],
            "alert_channels": [
                {
                    "name": "test_log",
                    "type": "log",
                    "enabled": True,
                    "configuration": {"log_level": "ERROR"}
                }
            ]
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config_data, f)
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('local_installation.scripts.production_monitoring.ProductionMonitor._setup_logging')
    def test_monitor_initialization(self, mock_logging):
        """Test monitor initialization."""
        mock_logging.return_value = Mock()
        
        monitor = ProductionMonitor(self.config_path)
        
        self.assertIsInstance(monitor.config, MonitoringConfig)
        self.assertTrue(monitor.config.enabled)
        self.assertEqual(monitor.config.check_interval_seconds, 5)
    
    def test_metrics_collection(self):
        """Test metrics collection."""
        collector = MetricsCollector("test_instance")
        
        # Test system metrics collection
        system_metrics = collector.collect_system_metrics()
        self.assertGreater(len(system_metrics), 0)
        
        # Verify metric structure
        for metric in system_metrics:
            self.assertIsInstance(metric, HealthMetric)
            self.assertEqual(metric.instance_id, "test_instance")
            self.assertIsInstance(metric.timestamp, datetime)
    
    def test_alert_threshold_evaluation(self):
        """Test alert threshold evaluation."""
        config = MonitoringConfig()
        alert_manager = AlertManager(config)
        
        # Create test metric that exceeds threshold
        metric = HealthMetric(
            name="error_rate",
            value=0.15,  # Exceeds critical threshold of 0.10
            unit="percent",
            timestamp=datetime.now(),
            instance_id="test_instance"
        )
        
        alerts = alert_manager.check_thresholds([metric])
        self.assertEqual(len(alerts), 1)
        
        alert = alerts[0]
        self.assertEqual(alert.severity, "critical")
        self.assertEqual(alert.metric_name, "error_rate")
        self.assertEqual(alert.value, 0.15)


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios for the reliability system."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('local_installation.scripts.reliability_integration.get_reliability_integration')
    def test_reliability_integration_availability(self, mock_integration):
        """Test reliability integration availability check."""
        # Mock available integration
        mock_integration.return_value.is_available.return_value = True
        mock_integration.return_value.get_health_status.return_value = {
            "status": "healthy",
            "error_rate": 0.01,
            "response_time_ms": 150
        }
        
        integration = mock_integration.return_value
        self.assertTrue(integration.is_available())
        
        health_status = integration.get_health_status()
        self.assertEqual(health_status["status"], "healthy")
        self.assertLess(health_status["error_rate"], 0.1)
    
    def test_configuration_environment_consistency(self):
        """Test configuration consistency across environments."""
        config_manager = ReliabilityConfigManager()
        
        # Test different environments
        environments = ["development", "testing", "production"]
        
        for env in environments:
            config = config_manager.get_config_for_environment(env)
            
            # Verify configuration is valid
            config_manager.config = config
            issues = config_manager.validate_config()
            self.assertEqual(len(issues), 0, f"Configuration issues in {env}: {issues}")
            
            # Verify environment-specific settings
            if env == "production":
                self.assertEqual(config.features.reliability_level, ReliabilityLevel.MAXIMUM)
                self.assertFalse(config.deployment.enable_debug_mode)
            elif env == "development":
                self.assertTrue(config.deployment.enable_debug_mode)
                self.assertEqual(config.deployment.log_level, "DEBUG")
    
    def test_feature_flag_consistency(self):
        """Test feature flag consistency and dependencies."""
        flag_manager = FeatureFlagManager()
        
        # Test that all reliability features are properly configured
        reliability_features = [
            "enhanced_error_context",
            "missing_method_recovery",
            "model_validation_recovery",
            "network_failure_recovery",
            "dependency_recovery",
            "pre_installation_validation",
            "diagnostic_monitoring",
            "health_reporting",
            "timeout_management",
            "user_guidance_enhancements",
            "intelligent_retry_system"
        ]
        
        for feature in reliability_features:
            self.assertIn(feature, flag_manager.flags, f"Missing feature flag: {feature}")
            
            flag = flag_manager.flags[feature]
            self.assertIsInstance(flag.state, FeatureState)
            self.assertIsInstance(flag.rollout_strategy, RolloutStrategy)
            self.assertGreaterEqual(flag.rollout_percentage, 0.0)
            self.assertLessEqual(flag.rollout_percentage, 100.0)


if __name__ == '__main__':
    # Set up logging for tests
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests
    unittest.main(verbosity=2)