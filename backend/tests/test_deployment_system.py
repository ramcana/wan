"""
Test Suite for Enhanced Model Availability Deployment System

This module contains comprehensive tests for the deployment automation,
validation, rollback, and monitoring systems.
"""

import os
import sys
import json
import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import deployment components
try:
    from scripts.deployment.deployment_validator import EnhancedModelAvailabilityValidator
    from scripts.deployment.rollback_manager import RollbackManager, RollbackType
    from scripts.deployment.monitoring_setup import EnhancedModelAvailabilityMonitor
    from scripts.deployment.config_backup_restore import ConfigurationBackupManager, BackupType
    from scripts.deployment.model_migration import ModelMigrationManager
    from scripts.deployment.deploy import EnhancedModelAvailabilityDeployer
except ImportError as e:
    pytest.skip(f"Deployment components not available: {e}", allow_module_level=True)

class TestDeploymentValidator:
    """Test deployment validation functionality"""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance"""
        return EnhancedModelAvailabilityValidator()
    
    @pytest.mark.asyncio
    async def test_environment_validation(self, validator):
        """Test environment validation"""
        # Mock environment checks
        with patch('sys.version_info', (3, 9, 0)):
            with patch('shutil.disk_usage', return_value=(100*1024**3, 50*1024**3, 50*1024**3)):
                validator.results = []
                await validator._validate_environment()
                
                # Check results
                python_result = next((r for r in validator.results if r.check_name == "python_version"), None)
                assert python_result is not None
                assert python_result.success
                
                disk_result = next((r for r in validator.results if r.check_name == "disk_space"), None)
                assert disk_result is not None
                assert disk_result.success
    
    @pytest.mark.asyncio
    async def test_component_validation(self, validator):
        """Test core component validation"""
        validator.results = []
        await validator._validate_core_components()
        
        # Should have results for each component
        component_results = [r for r in validator.results if r.check_name.startswith("component_")]
        assert len(component_results) > 0
    
    @pytest.mark.asyncio
    async def test_full_validation_report(self, validator):
        """Test complete validation report generation"""
        report = await validator.validate_deployment()
        
        assert hasattr(report, 'overall_success')
        assert hasattr(report, 'critical_failures')
        assert hasattr(report, 'warnings')
        assert hasattr(report, 'results')
        assert isinstance(report.results, list)

class TestRollbackManager:
    """Test rollback functionality"""
    
    @pytest.fixture
    def temp_backup_dir(self):
        """Create temporary backup directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def rollback_manager(self, temp_backup_dir):
        """Create rollback manager with temp directory"""
        return RollbackManager(backup_dir=temp_backup_dir)
    
    @pytest.mark.asyncio
    async def test_create_rollback_point(self, rollback_manager):
        """Test rollback point creation"""
        # Create some test files
        test_config = Path("test_config.json")
        test_config.write_text('{"test": true}')
        
        try:
            rollback_id = await rollback_manager.create_rollback_point(
                "Test rollback point",
                RollbackType.CONFIGURATION_ONLY
            )
            
            assert rollback_id is not None
            assert rollback_id in rollback_manager.rollback_points
            
            rollback_point = rollback_manager.rollback_points[rollback_id]
            assert rollback_point.description == "Test rollback point"
            assert rollback_point.rollback_type == RollbackType.CONFIGURATION_ONLY
            
        finally:
            test_config.unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_rollback_execution(self, rollback_manager):
        """Test rollback execution"""
        # Create test configuration
        test_config = Path("test_config.json")
        original_content = '{"original": true}'
        test_config.write_text(original_content)
        
        try:
            # Create rollback point
            rollback_id = await rollback_manager.create_rollback_point(
                "Test rollback",
                RollbackType.CONFIGURATION_ONLY
            )
            
            # Modify the file
            test_config.write_text('{"modified": true}')
            
            # Execute rollback
            result = await rollback_manager.execute_rollback(rollback_id)
            
            assert result.success
            assert result.rollback_point_id == rollback_id
            
            # Verify file was restored
            restored_content = test_config.read_text()
            assert "original" in restored_content
            
        finally:
            test_config.unlink(missing_ok=True)

class TestConfigurationBackupManager:
    """Test configuration backup and restore"""
    
    @pytest.fixture
    def temp_backup_dir(self):
        """Create temporary backup directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def backup_manager(self, temp_backup_dir):
        """Create backup manager with temp directory"""
        return ConfigurationBackupManager(backup_dir=temp_backup_dir)
    
    @pytest.mark.asyncio
    async def test_create_backup(self, backup_manager):
        """Test backup creation"""
        # Create test configuration files
        test_files = ["config.json", "test_config.json"]
        for file_name in test_files:
            Path(file_name).write_text(f'{{"test": "{file_name}"}}')
        
        try:
            backup_id = await backup_manager.create_backup(
                BackupType.CUSTOM,
                "Test backup",
                custom_files=test_files
            )
            
            assert backup_id is not None
            assert backup_id in backup_manager.manifests
            
            manifest = backup_manager.manifests[backup_id]
            assert manifest.backup_type == BackupType.CUSTOM
            assert len(manifest.files) == len(test_files)
            
        finally:
            for file_name in test_files:
                Path(file_name).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_restore_backup(self, backup_manager):
        """Test backup restoration"""
        # Create and backup test file
        test_file = Path("test_restore.json")
        original_content = '{"original": true}'
        test_file.write_text(original_content)
        
        try:
            # Create backup
            backup_id = await backup_manager.create_backup(
                BackupType.CUSTOM,
                "Test restore backup",
                custom_files=[str(test_file)]
            )
            
            # Modify file
            test_file.write_text('{"modified": true}')
            
            # Restore backup
            result = await backup_manager.restore_backup(backup_id, force=True)
            
            assert result.success
            assert result.files_restored > 0
            
            # Verify restoration
            restored_content = test_file.read_text()
            assert "original" in restored_content
            
        finally:
            test_file.unlink(missing_ok=True)

class TestModelMigrationManager:
    """Test model migration functionality"""
    
    @pytest.fixture
    def temp_models_dir(self):
        """Create temporary models directory"""
        temp_dir = tempfile.mkdtemp()
        models_dir = Path(temp_dir) / "models"
        models_dir.mkdir()
        
        # Create test model
        test_model_dir = models_dir / "test_model"
        test_model_dir.mkdir()
        (test_model_dir / "model.safetensors").write_bytes(b"fake model data")
        
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_discover_existing_models(self, temp_models_dir):
        """Test model discovery"""
        migration_manager = ModelMigrationManager(
            old_models_dir=str(Path(temp_models_dir) / "models"),
            new_models_dir=str(Path(temp_models_dir) / "new_models")
        )
        
        models = await migration_manager._discover_existing_models()
        
        assert len(models) > 0
        assert "test_model" in models
        assert models["test_model"]["type"] == "directory"
    
    @pytest.mark.asyncio
    async def test_migrate_single_model(self, temp_models_dir):
        """Test single model migration"""
        migration_manager = ModelMigrationManager(
            old_models_dir=str(Path(temp_models_dir) / "models"),
            new_models_dir=str(Path(temp_models_dir) / "new_models")
        )
        
        model_info = {
            "path": str(Path(temp_models_dir) / "models" / "test_model"),
            "files": [str(Path(temp_models_dir) / "models" / "test_model" / "model.safetensors")],
            "size_mb": 0.001,
            "type": "directory"
        }
        
        result = await migration_manager._migrate_single_model("test_model", model_info)
        
        assert result.success
        assert result.model_id == "test_model"
        assert len(result.actions_taken) > 0

class TestEnhancedModelAvailabilityMonitor:
    """Test monitoring system"""
    
    @pytest.fixture
    def monitor(self):
        """Create monitor instance"""
        return EnhancedModelAvailabilityMonitor()
    
    def test_metrics_collection(self, monitor):
        """Test metrics collection"""
        # Record test metric
        monitor.metrics_collector.record_metric(
            "test_metric",
            monitor.metrics_collector.MetricType.GAUGE,
            42.0,
            {"label": "test"},
            "Test metric"
        )
        
        # Retrieve metric
        metric = monitor.metrics_collector.get_metric("test_metric")
        assert metric is not None
        assert metric.value == 42.0
        assert metric.labels["label"] == "test"
    
    def test_alert_rules(self, monitor):
        """Test alert rule functionality"""
        from scripts.deployment.monitoring_setup import AlertRule, AlertLevel
        
        # Add test alert rule
        rule = AlertRule(
            name="test_alert",
            metric_name="test_metric",
            condition=">",
            threshold=50.0,
            level=AlertLevel.WARNING,
            description="Test alert"
        )
        
        monitor.alert_manager.add_alert_rule(rule)
        
        # Record metric that should trigger alert
        monitor.metrics_collector.record_metric(
            "test_metric",
            monitor.metrics_collector.MetricType.GAUGE,
            60.0
        )
        
        # Check alerts
        monitor.alert_manager.check_alerts()
        
        assert "test_alert" in monitor.alert_manager.active_alerts

class TestEnhancedModelAvailabilityDeployer:
    """Test deployment orchestration"""
    
    @pytest.fixture
    def temp_config(self):
        """Create temporary deployment config"""
        config = {
            "setup_monitoring": False,
            "cleanup_after_deployment": False,
            "auto_rollback_on_failure": False
        }
        
        config_file = Path("test_deployment_config.json")
        config_file.write_text(json.dumps(config))
        
        yield str(config_file)
        config_file.unlink(missing_ok=True)
    
    @pytest.fixture
    def deployer(self, temp_config):
        """Create deployer instance"""
        return EnhancedModelAvailabilityDeployer(temp_config)
    
    @pytest.mark.asyncio
    async def test_pre_deployment_validation(self, deployer):
        """Test pre-deployment validation"""
        # Mock validation to pass
        with patch.object(deployer, '_pre_deployment_validation', new_callable=AsyncMock):
            await deployer._pre_deployment_validation()
            # Should not raise exception
    
    @pytest.mark.asyncio
    async def test_dry_run_deployment(self, deployer):
        """Test dry run deployment"""
        # Mock all deployment phases
        with patch.object(deployer, '_pre_deployment_validation', new_callable=AsyncMock), \
             patch.object(deployer, '_execute_migration', new_callable=AsyncMock) as mock_migration, \
             patch.object(deployer, '_deploy_core_system', new_callable=AsyncMock), \
             patch.object(deployer, '_post_deployment_validation', new_callable=AsyncMock) as mock_post_val, \
             patch.object(deployer, '_final_health_check', new_callable=AsyncMock) as mock_health:
            
            mock_migration.return_value = {}
            mock_post_val.return_value = {}
            mock_health.return_value = {"healthy": True}
            
            result = await deployer.deploy(dry_run=True, skip_backup=True)
            
            assert result.deployment_id is not None
            assert len(result.phases_completed) > 0

# Integration tests
class TestDeploymentIntegration:
    """Integration tests for deployment system"""
    
    @pytest.mark.asyncio
    async def test_health_check_api_integration(self):
        """Test health check API integration"""
        try:
            from api.deployment_health import get_system_health
            
            # Mock the health check functions
            with patch('api.deployment_health.check_database_health', new_callable=AsyncMock), \
                 patch('api.deployment_health.check_file_system_health', new_callable=AsyncMock):
                
                response = await get_system_health()
                
                assert hasattr(response, 'overall_status')
                assert hasattr(response, 'components')
                assert isinstance(response.components, list)
                
        except ImportError:
            pytest.skip("Health check API not available")
    
    @pytest.mark.asyncio
    async def test_deployment_validation_integration(self):
        """Test deployment validation integration"""
        try:
            from api.deployment_health import validate_deployment
            
            response = await validate_deployment()
            
            assert hasattr(response, 'deployment_valid')
            assert hasattr(response, 'validation_results')
            assert isinstance(response.validation_results, list)
            
        except ImportError:
            pytest.skip("Deployment validation API not available")

# Performance tests
class TestDeploymentPerformance:
    """Performance tests for deployment system"""
    
    @pytest.mark.asyncio
    async def test_validation_performance(self):
        """Test validation performance"""
        validator = EnhancedModelAvailabilityValidator()
        
        start_time = datetime.now()
        report = await validator.validate_deployment()
        duration = (datetime.now() - start_time).total_seconds()
        
        # Validation should complete within reasonable time
        assert duration < 30.0  # 30 seconds max
        assert report is not None
    
    @pytest.mark.asyncio
    async def test_backup_performance(self):
        """Test backup performance"""
        with tempfile.TemporaryDirectory() as temp_dir:
            backup_manager = ConfigurationBackupManager(backup_dir=temp_dir)
            
            # Create test files
            test_files = []
            for i in range(10):
                test_file = Path(f"test_config_{i}.json")
                test_file.write_text(f'{{"test": {i}}}')
                test_files.append(str(test_file))
            
            try:
                start_time = datetime.now()
                backup_id = await backup_manager.create_backup(
                    BackupType.CUSTOM,
                    "Performance test backup",
                    custom_files=test_files
                )
                duration = (datetime.now() - start_time).total_seconds()
                
                # Backup should complete quickly
                assert duration < 10.0  # 10 seconds max
                assert backup_id is not None
                
            finally:
                for test_file in test_files:
                    Path(test_file).unlink(missing_ok=True)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])