"""
End-to-end integration tests for the reliability system integration.
Tests the complete integration of reliability components with the installation system.
"""

import unittest
import tempfile
import shutil
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the scripts directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from main_installer import MainInstaller
from integrated_installer import IntegratedInstaller
from reliability_manager import ReliabilityManager
from pre_installation_validator import PreInstallationValidator
from interfaces import InstallationPhase, ValidationResult, HardwareProfile


class TestReliabilityIntegration(unittest.TestCase):
    """Test reliability system integration with installation components."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.installation_path = Path(self.test_dir)
        
        # Create logs directory
        (self.installation_path / "logs").mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)
        
        # Mock arguments
        self.mock_args = Mock()
        self.mock_args.silent = True
        self.mock_args.verbose = False
        self.mock_args.dev_mode = False
        self.mock_args.skip_models = True
        self.mock_args.custom_path = None
        self.mock_args.force_reinstall = False
        self.mock_args.dry_run = True
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_main_installer_reliability_integration(self):
        """Test that MainInstaller properly integrates with ReliabilityManager."""
        installer = MainInstaller(str(self.installation_path), self.mock_args)
        
        # Verify reliability manager is initialized
        self.assertIsInstance(installer.reliability_manager, ReliabilityManager)
        
        # Verify components are wrapped
        self.assertIsNotNone(installer.error_handler)
        self.assertIsNotNone(installer.flow_controller)
        
        # Verify pre-validator is initialized
        self.assertIsInstance(installer.pre_validator, PreInstallationValidator)
    
    @patch('scripts.main_installer.IntegratedInstaller')
    def test_pre_installation_validation_integration(self, mock_integrated_installer):
        """Test pre-installation validation integration."""
        # Mock the integrated installer to avoid actual installation
        mock_installer_instance = Mock()
        mock_installer_instance.run_complete_installation.return_value = True
        mock_integrated_installer.return_value = mock_installer_instance
        
        installer = MainInstaller(str(self.installation_path), self.mock_args)
        
        # Mock validation results
        with patch.object(installer.pre_validator, 'validate_system_requirements') as mock_sys_val, \
             patch.object(installer.pre_validator, 'validate_network_connectivity') as mock_net_val, \
             patch.object(installer.pre_validator, 'validate_permissions') as mock_perm_val, \
             patch.object(installer.pre_validator, 'validate_existing_installation') as mock_exist_val, \
             patch.object(installer.pre_validator, 'generate_validation_report') as mock_report:
            
            # Setup successful validation results
            mock_sys_val.return_value = ValidationResult(success=True, message="")
            mock_net_val.return_value = ValidationResult(success=True, message="")
            mock_perm_val.return_value = ValidationResult(success=True, message="")
            mock_exist_val.return_value = ValidationResult(success=True, message="")
            mock_report.return_value = {"status": "success"}
            
            # Run installation
            result = installer.run_installation()
            
            # Verify validation methods were called
            mock_sys_val.assert_called_once()
            mock_net_val.assert_called_once()
            mock_perm_val.assert_called_once()
            mock_exist_val.assert_called_once()
            mock_report.assert_called_once()
            
            # Verify installation proceeded
            self.assertTrue(result)
    
    def test_integrated_installer_component_wrapping(self):
        """Test that IntegratedInstaller properly wraps components with reliability."""
        installer = IntegratedInstaller(str(self.installation_path), self.mock_args)
        
        # Verify reliability manager is initialized
        self.assertIsInstance(installer.reliability_manager, ReliabilityManager)
        
        # Verify core components are wrapped
        self.assertIsNotNone(installer.flow_controller)
        self.assertIsNotNone(installer.error_handler)
        self.assertIsNotNone(installer.progress_reporter)
        
        # Test component wrapping method
        mock_component = Mock()
        wrapped = installer._wrap_component_with_reliability(
            mock_component, "test_component", "test_id"
        )
        
        self.assertIsNotNone(wrapped)
    
    def test_reliability_metrics_tracking(self):
        """Test that reliability metrics are properly tracked during installation."""
        installer = MainInstaller(str(self.installation_path), self.mock_args)
        
        # Simulate some operations
        installer.reliability_manager.track_reliability_metrics(
            "test_component", "test_operation", True, 1.5
        )
        installer.reliability_manager.track_reliability_metrics(
            "test_component", "test_operation", False, 2.0
        )
        
        # Get metrics
        metrics = installer.reliability_manager.get_reliability_metrics()
        
        # Verify metrics are tracked
        self.assertIsNotNone(metrics)
        self.assertGreater(metrics.total_method_calls, 0)
    
    def test_backward_compatibility(self):
        """Test that reliability integration maintains backward compatibility."""
        # Test that installation can still work without reliability features
        installer = MainInstaller(str(self.installation_path), self.mock_args)
        
        # Verify that all expected attributes exist
        self.assertTrue(hasattr(installer, 'error_handler'))
        self.assertTrue(hasattr(installer, 'flow_controller'))
        self.assertTrue(hasattr(installer, 'progress_reporter'))
        self.assertTrue(hasattr(installer, 'reliability_manager'))
        
        # Verify that the installer can be initialized without errors
        self.assertIsNotNone(installer)
    
    def test_reliability_report_generation(self):
        """Test reliability report generation after installation."""
        installer = MainInstaller(str(self.installation_path), self.mock_args)
        
        # Mock some metrics
        installer.reliability_manager.track_reliability_metrics(
            "test_component", "test_operation", True, 1.0
        )
        
        # Generate reliability report
        installer._generate_reliability_report()
        
        # Verify report file was created
        report_path = self.installation_path / "logs" / "reliability_report.json"
        self.assertTrue(report_path.exists())
    
    def test_failure_report_generation(self):
        """Test failure report generation when installation fails."""
        installer = MainInstaller(str(self.installation_path), self.mock_args)
        
        # Generate failure report
        installer._generate_failure_report()
        
        # Verify failure report file was created
        report_path = self.installation_path / "logs" / "failure_report.json"
        self.assertTrue(report_path.exists())


if __name__ == '__main__':
    unittest.main()
