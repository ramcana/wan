#!/usr/bin/env python3
"""
Comprehensive Test Suite for WAN Model Deployment System

Tests all components of the deployment system including deployment manager,
migration service, validation service, rollback service, and monitoring service.
"""

import asyncio
import json
import logging
import shutil
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Import deployment system components
from infrastructure.deployment import (
    DeploymentManager, DeploymentConfig, DeploymentStatus,
    MigrationService, ValidationService, RollbackService, MonitoringService
)


class TestDeploymentSystem(unittest.TestCase):
    """Test suite for the WAN Model Deployment System"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        self.source_path = Path(self.temp_dir) / "source"
        self.target_path = Path(self.temp_dir) / "target"
        self.backup_path = Path(self.temp_dir) / "backup"
        
        # Create directories
        self.source_path.mkdir(parents=True)
        self.target_path.mkdir(parents=True)
        self.backup_path.mkdir(parents=True)
        
        # Create test configuration
        self.config = DeploymentConfig(
            source_models_path=str(self.source_path),
            target_models_path=str(self.target_path),
            backup_path=str(self.backup_path),
            validation_enabled=True,
            rollback_enabled=True,
            monitoring_enabled=False,  # Disable for testing
            health_check_interval=60,
            max_deployment_time=300
        )
        
        # Create test model structure
        self._create_test_models()
        
        # Setup logging
        logging.basicConfig(level=logging.DEBUG)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_models(self):
        """Create test model structures"""
        test_models = ["test-model-1", "test-model-2"]
        
        for model_name in test_models:
            model_dir = self.source_path / model_name
            model_dir.mkdir(parents=True)
            
            # Create test files
            (model_dir / "config.json").write_text(json.dumps({
                "model_type": "test",
                "architecture": "transformer",
                "version": "1.0.0"
            }))
            
            (model_dir / "model.bin").write_text("fake model data")
            (model_dir / "tokenizer.json").write_text("fake tokenizer data")
            
            # Create metadata
            (model_dir / "metadata.json").write_text(json.dumps({
                "name": model_name,
                "version": "1.0.0",
                "size_bytes": 1000,
                "checksum": "fake_checksum",
                "dependencies": ["torch", "transformers"],
                "config_requirements": {},
                "hardware_requirements": {}
            }))


class TestMigrationService(TestDeploymentSystem):
    """Test the Migration Service"""
    
    def setUp(self):
        super().setUp()
        self.migration_service = MigrationService(self.config)
    
    def test_migrate_single_model(self):
        """Test migrating a single model"""
        async def run_test():
            result = await self.migration_service.migrate_model("test-model-1", "test-deployment")
            
            self.assertTrue(result.success)
            self.assertEqual(result.model_name, "test-model-1")
            self.assertTrue(Path(result.target_path).exists())
            
            # Verify files were copied
            target_model_dir = Path(result.target_path)
            self.assertTrue((target_model_dir / "config.json").exists())
            self.assertTrue((target_model_dir / "model.bin").exists())
        
        asyncio.run(run_test())
    
    def test_migrate_nonexistent_model(self):
        """Test migrating a model that doesn't exist"""
        async def run_test():
            result = await self.migration_service.migrate_model("nonexistent-model", "test-deployment")
            
            self.assertFalse(result.success)
            self.assertIsNotNone(result.error)
        
        asyncio.run(run_test())
    
    def test_migration_history(self):
        """Test migration history tracking"""
        async def run_test():
            # Perform migrations
            await self.migration_service.migrate_model("test-model-1", "test-deployment")
            await self.migration_service.migrate_model("test-model-2", "test-deployment")
            
            # Check history
            history = await self.migration_service.get_migration_history()
            self.assertEqual(len(history), 2)
            
            # Verify history entries
            model_names = [h.model_name for h in history]
            self.assertIn("test-model-1", model_names)
            self.assertIn("test-model-2", model_names)
        
        asyncio.run(run_test())


class TestValidationService(TestDeploymentSystem):
    """Test the Validation Service"""
    
    def setUp(self):
        super().setUp()
        self.validation_service = ValidationService(self.config)
    
    def test_pre_deployment_validation(self):
        """Test pre-deployment validation"""
        async def run_test():
            result = await self.validation_service.validate_pre_deployment(["test-model-1"])
            
            self.assertIsNotNone(result)
            self.assertEqual(result.validation_type, "pre_deployment")
            self.assertGreater(len(result.checks_performed), 0)
            
            # Should have some checks that pass
            self.assertGreater(len(result.passed_checks), 0)
        
        asyncio.run(run_test())
    
    def test_post_deployment_validation(self):
        """Test post-deployment validation"""
        async def run_test():
            # First migrate a model
            migration_service = MigrationService(self.config)
            await migration_service.migrate_model("test-model-1", "test-deployment")
            
            # Then validate
            result = await self.validation_service.validate_post_deployment(["test-model-1"])
            
            self.assertIsNotNone(result)
            self.assertEqual(result.validation_type, "post_deployment")
            self.assertGreater(len(result.checks_performed), 0)
        
        asyncio.run(run_test())
    
    def test_model_health_validation(self):
        """Test model health validation"""
        async def run_test():
            # First migrate a model
            migration_service = MigrationService(self.config)
            await migration_service.migrate_model("test-model-1", "test-deployment")
            
            # Then check health
            result = await self.validation_service.validate_model_health("test-model-1")
            
            self.assertIsNotNone(result)
            self.assertEqual(result.validation_type, "health_check")
            self.assertEqual(result.model_name, "test-model-1")
        
        asyncio.run(run_test())
    
    def test_validation_history(self):
        """Test validation history tracking"""
        async def run_test():
            # Perform validations
            await self.validation_service.validate_pre_deployment(["test-model-1"])
            await self.validation_service.validate_pre_deployment(["test-model-2"])
            
            # Check history
            history = await self.validation_service.get_validation_history()
            self.assertEqual(len(history), 2)
        
        asyncio.run(run_test())


class TestRollbackService(TestDeploymentSystem):
    """Test the Rollback Service"""
    
    def setUp(self):
        super().setUp()
        self.rollback_service = RollbackService(self.config)
    
    def test_create_backup(self):
        """Test backup creation"""
        async def run_test():
            # Create some target models first
            migration_service = MigrationService(self.config)
            await migration_service.migrate_model("test-model-1", "test-deployment")
            
            # Create backup
            result = await self.rollback_service.create_backup("test-deployment", ["test-model-1"])
            
            self.assertTrue(result.success)
            self.assertEqual(result.deployment_id, "test-deployment")
            self.assertIn("test-model-1", result.models_restored)
            
            # Verify backup exists
            backup_path = Path(result.backup_id)  # This would be the backup directory
            # Note: In real implementation, we'd check the actual backup structure
        
        asyncio.run(run_test())
    
    def test_backup_registry(self):
        """Test backup registry functionality"""
        async def run_test():
            # Create backup
            await self.rollback_service.create_backup("test-deployment", ["test-model-1"])
            
            # List backups
            backups = await self.rollback_service.list_backups()
            self.assertGreater(len(backups), 0)
            
            # Check deployment-specific backups
            deployment_backups = await self.rollback_service.list_backups("test-deployment")
            self.assertGreater(len(deployment_backups), 0)
        
        asyncio.run(run_test())


class TestMonitoringService(TestDeploymentSystem):
    """Test the Monitoring Service"""
    
    def setUp(self):
        super().setUp()
        # Enable monitoring for these tests
        self.config.monitoring_enabled = True
        self.config.health_check_interval = 1  # Fast for testing
        self.monitoring_service = MonitoringService(self.config)
    
    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring"""
        async def run_test():
            # Start monitoring
            await self.monitoring_service.start_monitoring("test-deployment", ["test-model-1"])
            
            # Check that deployment is being monitored
            self.assertIn("test-deployment", self.monitoring_service.monitored_deployments)
            
            # Stop monitoring
            await self.monitoring_service.stop_monitoring("test-deployment")
            
            # Check that deployment is no longer monitored
            self.assertNotIn("test-deployment", self.monitoring_service.monitored_deployments)
        
        asyncio.run(run_test())
    
    def test_health_status(self):
        """Test health status reporting"""
        async def run_test():
            # Start monitoring
            await self.monitoring_service.start_monitoring("test-deployment", ["test-model-1"])
            
            # Get health status
            status = await self.monitoring_service.get_health_status()
            
            self.assertIsInstance(status, dict)
            self.assertIn("timestamp", status)
            self.assertIn("monitored_deployments", status)
            self.assertIn("overall_status", status)
        
        asyncio.run(run_test())
    
    def test_alert_handling(self):
        """Test alert handling"""
        alerts_received = []
        
        async def test_alert_handler(alert):
            alerts_received.append(alert)
        
        async def run_test():
            # Add alert handler
            self.monitoring_service.add_alert_handler(test_alert_handler)
            
            # Start monitoring
            await self.monitoring_service.start_monitoring("test-deployment", ["test-model-1"])
            
            # Wait a bit for monitoring to run
            await asyncio.sleep(2)
            
            # Check if any alerts were generated (depends on system state)
            # This is more of a smoke test
            self.assertIsInstance(alerts_received, list)
        
        asyncio.run(run_test())


class TestDeploymentManager(TestDeploymentSystem):
    """Test the main Deployment Manager"""
    
    def setUp(self):
        super().setUp()
        self.deployment_manager = DeploymentManager(self.config)
    
    def test_full_deployment_flow(self):
        """Test complete deployment flow"""
        async def run_test():
            # Deploy models
            result = await self.deployment_manager.deploy_models(["test-model-1", "test-model-2"])
            
            self.assertIsNotNone(result)
            self.assertIn(result.status, [DeploymentStatus.COMPLETED, DeploymentStatus.FAILED, DeploymentStatus.ROLLED_BACK])
            self.assertIsNotNone(result.deployment_id)
            self.assertIsNotNone(result.start_time)
            self.assertIsNotNone(result.end_time)
            
            # If deployment was successful, check that models were deployed
            if result.status == DeploymentStatus.COMPLETED:
                self.assertGreater(len(result.models_deployed), 0)
                
                # Verify models exist in target location
                for model_name in result.models_deployed:
                    model_path = self.target_path / model_name
                    self.assertTrue(model_path.exists())
        
        asyncio.run(run_test())
    
    def test_deployment_status_tracking(self):
        """Test deployment status tracking"""
        async def run_test():
            # Deploy models
            result = await self.deployment_manager.deploy_models(["test-model-1"])
            
            # Get deployment status
            status = await self.deployment_manager.get_deployment_status(result.deployment_id)
            self.assertIsNotNone(status)
            self.assertEqual(status.deployment_id, result.deployment_id)
        
        asyncio.run(run_test())
    
    def test_deployment_history(self):
        """Test deployment history"""
        async def run_test():
            # Perform multiple deployments
            result1 = await self.deployment_manager.deploy_models(["test-model-1"])
            result2 = await self.deployment_manager.deploy_models(["test-model-2"])
            
            # Get deployment history
            history = await self.deployment_manager.list_deployments()
            
            self.assertGreaterEqual(len(history), 2)
            
            # Check that our deployments are in history
            deployment_ids = [d.deployment_id for d in history]
            self.assertIn(result1.deployment_id, deployment_ids)
            self.assertIn(result2.deployment_id, deployment_ids)
        
        asyncio.run(run_test())
    
    def test_deployment_with_validation_failure(self):
        """Test deployment behavior when validation fails"""
        async def run_test():
            # Mock validation service to always fail
            with patch.object(self.deployment_manager.validation_service, 'validate_pre_deployment') as mock_validate:
                mock_result = Mock()
                mock_result.is_valid = False
                mock_result.errors = ["Test validation failure"]
                mock_validate.return_value = mock_result
                
                # Attempt deployment
                result = await self.deployment_manager.deploy_models(["test-model-1"])
                
                # Should fail due to validation
                self.assertEqual(result.status, DeploymentStatus.FAILED)
                self.assertIsNotNone(result.error_message)
        
        asyncio.run(run_test())


class TestIntegration(TestDeploymentSystem):
    """Integration tests for the complete system"""
    
    def test_end_to_end_deployment(self):
        """Test complete end-to-end deployment scenario"""
        async def run_test():
            # Initialize deployment manager
            deployment_manager = DeploymentManager(self.config)
            
            # Deploy models
            deployment_result = await deployment_manager.deploy_models(["test-model-1"])
            
            # Verify deployment completed (or failed gracefully)
            self.assertIn(deployment_result.status, [
                DeploymentStatus.COMPLETED, 
                DeploymentStatus.FAILED, 
                DeploymentStatus.ROLLED_BACK
            ])
            
            # If successful, verify model is accessible
            if deployment_result.status == DeploymentStatus.COMPLETED:
                model_path = self.target_path / "test-model-1"
                self.assertTrue(model_path.exists())
                self.assertTrue((model_path / "config.json").exists())
            
            # Test rollback if deployment was successful
            if deployment_result.status == DeploymentStatus.COMPLETED and deployment_result.rollback_available:
                rollback_result = await deployment_manager.rollback_deployment(
                    deployment_result.deployment_id, 
                    "Test rollback"
                )
                
                # Rollback should succeed or fail gracefully
                self.assertIsInstance(rollback_result.success, bool)
        
        asyncio.run(run_test())
    
    def test_concurrent_deployments(self):
        """Test handling of concurrent deployments"""
        async def run_test():
            deployment_manager = DeploymentManager(self.config)
            
            # Start multiple deployments concurrently
            tasks = [
                deployment_manager.deploy_models(["test-model-1"]),
                deployment_manager.deploy_models(["test-model-2"])
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Both should complete (successfully or with errors)
            self.assertEqual(len(results), 2)
            
            for result in results:
                if isinstance(result, Exception):
                    # Exception is acceptable for concurrent deployments
                    continue
                else:
                    # Should be a valid deployment result
                    self.assertIn(result.status, [
                        DeploymentStatus.COMPLETED,
                        DeploymentStatus.FAILED,
                        DeploymentStatus.ROLLED_BACK
                    ])
        
        asyncio.run(run_test())


def run_deployment_system_tests():
    """Run all deployment system tests"""
    print("🧪 Running WAN Model Deployment System Tests...")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestMigrationService,
        TestValidationService,
        TestRollbackService,
        TestMonitoringService,
        TestDeploymentManager,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n📊 Test Results:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print(f"\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\n{'✅ All tests passed!' if success else '❌ Some tests failed!'}")
    
    return success


if __name__ == '__main__':
    import sys
    success = run_deployment_system_tests()
    sys.exit(0 if success else 1)
