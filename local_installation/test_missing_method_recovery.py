"""
Unit tests for missing method detection and recovery system.
Tests the fallback implementations and compatibility shims.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.append('scripts')

from missing_method_recovery import MissingMethodRecovery
from interfaces import InstallationError, ErrorCategory


class MockModelDownloader:
    """Mock ModelDownloader class for testing."""
    
    def __init__(self):
        self.models_dir = "models"
        self.MODEL_CONFIG = {
            'WAN2.2-T2V-A14B': {'size': '28GB'},
            'WAN2.2-I2V-A14B': {'size': '28GB'},
            'WAN2.2-TI2V-5B': {'size': '10GB'}
        }
    
    def download_wan22_models(self):
        return True
    
    def verify_model_integrity(self):
        return True


class MockDependencyManager:
    """Mock DependencyManager class for testing."""
    
    def __init__(self):
        pass
    
    def create_virtual_environment(self):
        return True


class MockConfigurationEngine:
    """Mock ConfigurationEngine class for testing."""
    
    def __init__(self):
        pass
    
    def generate_config(self):
        return {"test": "config"}


class MockValidationResult:
    """Mock ValidationResult class for testing."""
    
    def __init__(self, success=True):
        self.success = success


class TestMissingMethodRecovery(unittest.TestCase):
    """Test missing method recovery functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.recovery = MissingMethodRecovery(self.temp_dir)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test proper initialization of recovery system."""
        self.assertIsInstance(self.recovery.fallback_methods, dict)
        self.assertIsInstance(self.recovery.compatibility_shims, dict)
        self.assertIsInstance(self.recovery.version_info, dict)
        
        # Check that fallback methods are registered
        self.assertIn('ModelDownloader', self.recovery.fallback_methods)
        self.assertIn('DependencyManager', self.recovery.fallback_methods)
        self.assertIn('ConfigurationEngine', self.recovery.fallback_methods)
        
        # Check that compatibility shims are registered
        self.assertIn('ModelDownloader', self.recovery.compatibility_shims)
        self.assertIn('DependencyManager', self.recovery.compatibility_shims)
    
    def test_version_detection(self):
        """Test software version detection."""
        versions = self.recovery.version_info
        
        # Should detect Python version
        self.assertIn('python', versions)
        self.assertRegex(versions['python'], r'\d+\.\d+\.\d+')
        
        # Should attempt to detect package versions
        self.assertIn('huggingface_hub', versions)
        self.assertIn('torch', versions)
        self.assertIn('psutil', versions)
    
    def test_compatibility_shim_success(self):
        """Test successful compatibility shim usage."""
        mock_downloader = MockModelDownloader()
        
        # Test that compatibility shim works
        result = self.recovery._try_compatibility_shim(
            mock_downloader, 'ModelDownloader', 'download_models_parallel'
        )
        
        self.assertTrue(result)
    
    def test_fallback_method_success(self):
        """Test successful fallback method usage."""
        mock_downloader = MockModelDownloader()
        
        # Test fallback for get_required_models
        result = self.recovery._try_fallback_method(
            mock_downloader, 'ModelDownloader', 'get_required_models'
        )
        
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
    
    def test_dynamic_injection(self):
        """Test dynamic method injection."""
        mock_downloader = MockModelDownloader()
        
        # Test dynamic injection
        success = self.recovery._try_dynamic_injection(
            mock_downloader, 'ModelDownloader', 'get_required_models'
        )
        
        self.assertTrue(success)
        self.assertTrue(hasattr(mock_downloader, 'get_required_models'))
        
        # Test that injected method works
        result = mock_downloader.get_required_models()
        self.assertIsInstance(result, list)
    
    def test_handle_missing_method_success(self):
        """Test successful missing method handling."""
        mock_downloader = MockModelDownloader()
        
        # Test handling missing method - this should use fallback since MockModelDownloader
        # doesn't have get_required_models method
        result = self.recovery.handle_missing_method(
            mock_downloader, 'get_required_models'
        )
        
        self.assertIsInstance(result, list)
        self.assertIn('WAN2.2-T2V-A14B', result)
    
    def test_handle_missing_method_failure(self):
        """Test missing method handling when all strategies fail."""
        mock_obj = Mock()
        mock_obj.__class__.__name__ = 'UnknownClass'
        
        # Should raise InstallationError when all strategies fail
        with self.assertRaises(InstallationError) as context:
            self.recovery.handle_missing_method(mock_obj, 'unknown_method')
        
        self.assertEqual(context.exception.category, ErrorCategory.SYSTEM)
        self.assertIn('could not be recovered automatically', context.exception.message)
    
    def test_fallback_get_required_models(self):
        """Test fallback implementation for get_required_models."""
        mock_downloader = MockModelDownloader()
        
        result = self.recovery._fallback_get_required_models(mock_downloader)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)  # Should return 3 models from MODEL_CONFIG
        self.assertIn('WAN2.2-T2V-A14B', result)
    
    def test_fallback_download_models_parallel(self):
        """Test fallback implementation for download_models_parallel."""
        mock_downloader = MockModelDownloader()
        
        result = self.recovery._fallback_download_models_parallel(mock_downloader)
        
        self.assertTrue(result)
    
    def test_fallback_verify_all_models(self):
        """Test fallback implementation for verify_all_models."""
        mock_downloader = MockModelDownloader()
        
        result = self.recovery._fallback_verify_all_models(mock_downloader)
        
        self.assertTrue(result)
    
    def test_fallback_create_optimized_venv(self):
        """Test fallback implementation for create_optimized_virtual_environment."""
        mock_manager = MockDependencyManager()
        
        result = self.recovery._fallback_create_optimized_venv(mock_manager)
        
        self.assertTrue(result)
    
    def test_fallback_generate_base_configuration(self):
        """Test fallback implementation for generate_base_configuration."""
        mock_engine = MockConfigurationEngine()
        
        result = self.recovery._fallback_generate_base_configuration(mock_engine)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result, {"test": "config"})
    
    def test_fallback_meets_minimum(self):
        """Test fallback implementation for meets_minimum."""
        mock_result = MockValidationResult(success=True)
        
        result = self.recovery._fallback_meets_minimum(mock_result)
        
        self.assertTrue(result)
        
        # Test with failed validation
        mock_result_failed = MockValidationResult(success=False)
        result_failed = self.recovery._fallback_meets_minimum(mock_result_failed)
        
        self.assertFalse(result_failed)
    
    def test_validate_software_versions(self):
        """Test software version validation."""
        issues = self.recovery.validate_software_versions()
        
        self.assertIsInstance(issues, list)
        # Issues list may be empty if all versions are acceptable
    
    @patch('pkg_resources.get_distribution')
    def test_version_detection_with_missing_packages(self, mock_get_dist):
        """Test version detection when packages are missing."""
        from pkg_resources import DistributionNotFound
        mock_get_dist.side_effect = DistributionNotFound()
        
        recovery = MissingMethodRecovery(self.temp_dir)
        
        # Should handle missing packages gracefully
        self.assertIn('huggingface_hub', recovery.version_info)
        self.assertEqual(recovery.version_info['huggingface_hub'], 'not_installed')
    
    def test_get_recovery_statistics(self):
        """Test recovery statistics generation."""
        stats = self.recovery.get_recovery_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('supported_classes', stats)
        self.assertIn('supported_methods', stats)
        self.assertIn('version_info', stats)
        
        # Check that all expected classes are supported
        supported_classes = stats['supported_classes']
        self.assertIn('ModelDownloader', supported_classes)
        self.assertIn('DependencyManager', supported_classes)
        self.assertIn('ConfigurationEngine', supported_classes)


class TestMissingMethodRecoveryIntegration(unittest.TestCase):
    """Integration tests for missing method recovery."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.recovery = MissingMethodRecovery(self.temp_dir)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_real_world_scenario_model_downloader(self):
        """Test real-world scenario with ModelDownloader missing methods."""
        # Create a mock object that's missing the methods we saw in the error log
        mock_downloader = Mock()
        mock_downloader.__class__.__name__ = 'ModelDownloader'
        mock_downloader.models_dir = self.temp_dir
        mock_downloader.MODEL_CONFIG = {
            'WAN2.2-T2V-A14B': {'size': '28GB'},
            'WAN2.2-I2V-A14B': {'size': '28GB'}
        }
        
        # Add some methods that exist
        mock_downloader.download_wan22_models = Mock(return_value=True)
        mock_downloader.verify_model_integrity = Mock(return_value=True)
        
        # Test missing get_required_models
        result = self.recovery.handle_missing_method(
            mock_downloader, 'get_required_models'
        )
        
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        
        # Test missing download_models_parallel
        result = self.recovery.handle_missing_method(
            mock_downloader, 'download_models_parallel'
        )
        
        self.assertTrue(result)
        mock_downloader.download_wan22_models.assert_called_once()
        
        # Test missing verify_all_models
        result = self.recovery.handle_missing_method(
            mock_downloader, 'verify_all_models'
        )
        
        self.assertTrue(result)
        mock_downloader.verify_model_integrity.assert_called_once()
    
    def test_real_world_scenario_dependency_manager(self):
        """Test real-world scenario with DependencyManager missing methods."""
        mock_manager = Mock()
        mock_manager.__class__.__name__ = 'DependencyManager'
        mock_manager.create_virtual_environment = Mock(return_value=True)
        
        # Test missing create_optimized_virtual_environment
        result = self.recovery.handle_missing_method(
            mock_manager, 'create_optimized_virtual_environment'
        )
        
        self.assertTrue(result)
        mock_manager.create_virtual_environment.assert_called_once()
    
    def test_real_world_scenario_validation_result(self):
        """Test real-world scenario with ValidationResult missing methods."""
        mock_result = Mock()
        mock_result.__class__.__name__ = 'ValidationResult'
        mock_result.success = True
        
        # Test missing meets_minimum
        result = self.recovery.handle_missing_method(
            mock_result, 'meets_minimum'
        )
        
        self.assertTrue(result)
    
    def test_error_log_scenarios(self):
        """Test specific scenarios from the error log."""
        # Scenario 1: ModelDownloader missing get_required_models
        mock_downloader = Mock()
        mock_downloader.__class__.__name__ = 'ModelDownloader'
        mock_downloader.MODEL_CONFIG = {'WAN2.2-T2V-A14B': {}}
        
        try:
            result = self.recovery.handle_missing_method(
                mock_downloader, 'get_required_models'
            )
            self.assertIsInstance(result, list)
        except InstallationError:
            self.fail("Should not raise InstallationError for known missing method")
        
        # Scenario 2: DependencyManager missing create_optimized_virtual_environment
        mock_manager = Mock()
        mock_manager.__class__.__name__ = 'DependencyManager'
        mock_manager.create_virtual_environment = Mock(return_value=True)
        
        try:
            result = self.recovery.handle_missing_method(
                mock_manager, 'create_optimized_virtual_environment'
            )
            self.assertTrue(result)
        except InstallationError:
            self.fail("Should not raise InstallationError for known missing method")


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
