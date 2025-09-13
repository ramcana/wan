"""
Test suite for VRAM Configuration Manager

Tests the fallback configuration system, validation, persistent storage,
and GPU selection interface functionality.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from vram_config_manager import (
    VRAMConfigManager, VRAMConfigProfile, GPUSelectionCriteria,
    create_fallback_config, get_gpu_selection_ui, validate_vram_config
)
from vram_manager import GPUInfo, VRAMUsage, VRAMDetectionError
from datetime import datetime


class TestVRAMConfigManager(unittest.TestCase):
    """Test cases for VRAM Configuration Manager"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = VRAMConfigManager(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        self.config_manager.cleanup()
        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_manual_vram_config_creation(self):
        """Test creating manual VRAM configuration"""
        # Test valid configuration
        gpu_mapping = {0: 16, 1: 24}
        success, errors = self.config_manager.create_manual_vram_config(
            gpu_mapping, "test_config", "Test configuration"
        )
        
        self.assertTrue(success)
        self.assertEqual(len(errors), 0)
        self.assertIn("test_config", self.config_manager.profiles)
        
        # Verify profile was created correctly
        profile = self.config_manager.profiles["test_config"]
        self.assertEqual(profile.manual_vram_gb, gpu_mapping)
        self.assertEqual(profile.description, "Test configuration")

        assert True  # TODO: Add proper assertion
    
    def test_manual_vram_config_validation(self):
        """Test validation of manual VRAM configurations"""
        # Test valid configuration
        valid_config = {0: 16, 1: 24}
        is_valid, errors = self.config_manager.validate_manual_vram_config(valid_config)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
        # Test invalid configurations
        test_cases = [
            ({}, ["GPU VRAM mapping cannot be empty"]),
            ({-1: 16}, ["GPU index cannot be negative: -1"]),
            ({0: 0}, ["VRAM amount must be positive for GPU 0: 0GB"]),
            ({0: 200}, ["VRAM amount too high for GPU 0: 200GB"]),
            ({"invalid": 16}, ["GPU index must be integer"]),
            ({0: "invalid"}, ["VRAM amount must be numeric"]),
        ]
        
        for config, expected_error_keywords in test_cases:
            is_valid, errors = self.config_manager.validate_manual_vram_config(config)
            self.assertFalse(is_valid)
            self.assertGreater(len(errors), 0)
            # Check if any expected keyword is in any error message
            error_text = " ".join(errors).lower()
            self.assertTrue(any(keyword.lower() in error_text for keyword in expected_error_keywords))

        assert True  # TODO: Add proper assertion
    
    @patch('vram_config_manager.VRAMManager')
    def test_gpu_selection_interface(self, mock_vram_manager):
        """Test GPU selection interface functionality"""
        # Mock GPU detection
        mock_gpus = [
            GPUInfo(0, "RTX 4080", 16384, "535.98", "12.2", 65.0, 25.0, 250.0),
            GPUInfo(1, "RTX 3090", 24576, "535.98", "12.2", 70.0, 45.0, 350.0),
        ]
        
        mock_vram_manager.return_value.detect_vram_capacity.return_value = mock_gpus
        mock_vram_manager.return_value.get_current_vram_usage.return_value = [
            VRAMUsage(0, 4096, 12288, 16384, 25.0, datetime.now()),
            VRAMUsage(1, 8192, 16384, 24576, 33.3, datetime.now()),
        ]
        
        # Create new manager with mocked VRAM manager
        config_manager = VRAMConfigManager(self.temp_dir)
        
        # Test interface without criteria
        interface_data = config_manager.get_gpu_selection_interface()
        
        self.assertIn('available_gpus', interface_data)
        self.assertEqual(len(interface_data['available_gpus']), 2)
        
        # Check GPU data structure
        gpu_data = interface_data['available_gpus'][0]
        required_fields = ['index', 'name', 'total_memory_gb', 'suitability_score']
        for field in required_fields:
            self.assertIn(field, gpu_data)
        
        # Test interface with criteria
        criteria = GPUSelectionCriteria(min_vram_gb=20)
        interface_data = config_manager.get_gpu_selection_interface(criteria)
        
        # Should only include RTX 3090 (24GB)
        filtered_gpus = [gpu for gpu in interface_data['available_gpus'] if gpu['total_memory_gb'] >= 20]
        self.assertGreater(len(filtered_gpus), 0)

        assert True  # TODO: Add proper assertion
    
    def test_gpu_selection(self):
        """Test GPU selection functionality"""
        # Mock available GPUs
        with patch.object(self.config_manager.vram_manager, 'get_available_gpus') as mock_get_gpus:
            mock_get_gpus.return_value = [
                GPUInfo(0, "RTX 4080", 16384, "535.98"),
                GPUInfo(1, "RTX 3090", 24576, "535.98"),
            ]
            
            with patch.object(self.config_manager.vram_manager, 'set_preferred_gpu') as mock_set_gpu:
                # Test valid GPU selection
                success, message = self.config_manager.select_gpu(0)
                self.assertTrue(success)
                self.assertIn("RTX 4080", message)
                mock_set_gpu.assert_called_once_with(0)
                
                # Test invalid GPU selection
                success, message = self.config_manager.select_gpu(99)
                self.assertFalse(success)
                self.assertIn("not found", message)

        assert True  # TODO: Add proper assertion
    
    def test_multi_gpu_support(self):
        """Test multi-GPU support functionality"""
        with patch.object(self.config_manager.vram_manager, 'enable_multi_gpu') as mock_enable:
            # Test enabling multi-GPU
            success, message = self.config_manager.enable_multi_gpu_support(True)
            self.assertTrue(success)
            self.assertIn("enabled", message)
            mock_enable.assert_called_once_with(True)
            
            # Test disabling multi-GPU
            success, message = self.config_manager.enable_multi_gpu_support(False)
            self.assertTrue(success)
            self.assertIn("disabled", message)

        assert True  # TODO: Add proper assertion
    
    def test_profile_management(self):
        """Test VRAM configuration profile management"""
        # Test profile creation
        success, message = self.config_manager.create_profile(
            "test_profile", "Test profile", {0: 16}, preferred_gpu=0
        )
        self.assertTrue(success)
        self.assertIn("test_profile", self.config_manager.profiles)
        
        # Test duplicate profile creation
        success, message = self.config_manager.create_profile(
            "test_profile", "Duplicate profile"
        )
        self.assertFalse(success)
        self.assertIn("already exists", message)
        
        # Test profile loading
        with patch.object(self.config_manager.vram_manager, 'set_manual_vram_config'):
            with patch.object(self.config_manager.vram_manager, 'set_preferred_gpu'):
                success, message = self.config_manager.load_profile("test_profile")
                self.assertTrue(success)
                self.assertEqual(self.config_manager.current_profile, "test_profile")
        
        # Test profile listing
        profiles = self.config_manager.list_profiles()
        self.assertIn("test_profile", profiles)
        self.assertTrue(profiles["test_profile"]["is_current"])
        
        # Test profile deletion
        success, message = self.config_manager.delete_profile("test_profile")
        self.assertTrue(success)
        self.assertNotIn("test_profile", self.config_manager.profiles)
        self.assertIsNone(self.config_manager.current_profile)

        assert True  # TODO: Add proper assertion
    
    def test_fallback_config_options(self):
        """Test fallback configuration options"""
        options = self.config_manager.get_fallback_config_options()
        
        # Check required sections
        required_sections = ['common_gpu_configs', 'multi_gpu_examples', 'validation_rules']
        for section in required_sections:
            self.assertIn(section, options)
        
        # Check common GPU configs
        common_configs = options['common_gpu_configs']
        self.assertIn('RTX 4080', common_configs)
        self.assertEqual(common_configs['RTX 4080'], {'0': 16})
        
        # Check validation rules
        rules = options['validation_rules']
        self.assertIn('min_vram_gb', rules)
        self.assertIn('max_vram_gb', rules)

        assert True  # TODO: Add proper assertion
    
    def test_config_export_import(self):
        """Test configuration export and import"""
        # Create test profile
        self.config_manager.create_profile("export_test", "Export test profile", {0: 16})
        
        # Test export
        export_path = os.path.join(self.temp_dir, "export_test.json")
        success, message = self.config_manager.export_config(export_path)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(export_path))
        
        # Verify export content
        with open(export_path, 'r') as f:
            export_data = json.load(f)
        
        self.assertIn('profiles', export_data)
        self.assertIn('export_test', export_data['profiles'])
        
        # Test import
        new_manager = VRAMConfigManager(tempfile.mkdtemp())
        success, message = new_manager.import_config(export_path)
        self.assertTrue(success)
        self.assertIn("export_test", new_manager.profiles)
        
        new_manager.cleanup()

        assert True  # TODO: Add proper assertion
    
    def test_system_status(self):
        """Test system status reporting"""
        status = self.config_manager.get_system_status()
        
        # Check required fields
        required_fields = ['detection_summary', 'profiles_count', 'config_valid']
        for field in required_fields:
            self.assertIn(field, status)
        
        self.assertIsInstance(status['profiles_count'], int)
        self.assertIsInstance(status['config_valid'], bool)

        assert True  # TODO: Add proper assertion
    
    def test_gpu_suitability_scoring(self):
        """Test GPU suitability scoring algorithm"""
        # Create test GPU
        gpu = GPUInfo(0, "RTX 4080", 16384, "535.98", "12.2", 65.0, 25.0, 250.0)
        
        # Test without criteria
        score = self.config_manager._calculate_gpu_suitability(gpu, None)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)
        
        # Test with criteria
        criteria = GPUSelectionCriteria(min_vram_gb=16, preferred_models=["RTX 4080"])
        score_with_criteria = self.config_manager._calculate_gpu_suitability(gpu, criteria)
        self.assertGreater(score_with_criteria, score)  # Should score higher with matching criteria

        assert True  # TODO: Add proper assertion
    
    def test_gpu_filtering_by_criteria(self):
        """Test GPU filtering by selection criteria"""
        gpus = [
            GPUInfo(0, "RTX 4080", 16384, "535.98", "12.2"),
            GPUInfo(1, "RTX 3090", 24576, "535.98", "12.2"),
            GPUInfo(2, "GTX 1660", 6144, "535.98", "12.2"),
        ]
        
        # Test VRAM filtering
        criteria = GPUSelectionCriteria(min_vram_gb=16)
        filtered = self.config_manager._filter_gpus_by_criteria(gpus, criteria)
        self.assertEqual(len(filtered), 2)  # RTX 4080 and RTX 3090
        
        # Test model filtering
        criteria = GPUSelectionCriteria(preferred_models=["RTX 4080"])
        filtered = self.config_manager._filter_gpus_by_criteria(gpus, criteria)
        self.assertEqual(len(filtered), 1)  # Only RTX 4080
        
        # Test exclusion filtering
        criteria = GPUSelectionCriteria(exclude_models=["GTX"])
        filtered = self.config_manager._filter_gpus_by_criteria(gpus, criteria)
        self.assertEqual(len(filtered), 2)  # Exclude GTX 1660


        assert True  # TODO: Add proper assertion

class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
    shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('vram_config_manager.VRAMConfigManager')
    def test_create_fallback_config(self, mock_manager_class):
        """Test create_fallback_config utility function"""
        mock_manager = Mock()
        mock_manager.create_manual_vram_config.return_value = (True, [])
        mock_manager_class.return_value = mock_manager
        
        gpu_mapping = {0: 16}
        success, errors = create_fallback_config(gpu_mapping)
        
        self.assertTrue(success)
        self.assertEqual(len(errors), 0)
        mock_manager.create_manual_vram_config.assert_called_once_with(gpu_mapping)

        assert True  # TODO: Add proper assertion
    
    @patch('vram_config_manager.VRAMConfigManager')
    def test_validate_vram_config(self, mock_manager_class):
        """Test validate_vram_config utility function"""
        mock_manager = Mock()
        mock_manager.validate_manual_vram_config.return_value = (True, [])
        mock_manager_class.return_value = mock_manager
        
        gpu_mapping = {0: 16}
        is_valid, errors = validate_vram_config(gpu_mapping)
        
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        mock_manager.validate_manual_vram_config.assert_called_once_with(gpu_mapping)

        assert True  # TODO: Add proper assertion
    
    @patch('vram_config_manager.VRAMConfigManager')
    def test_get_gpu_selection_ui(self, mock_manager_class):
        """Test get_gpu_selection_ui utility function"""
        mock_manager = Mock()
        mock_interface_data = {'available_gpus': [], 'recommendations': []}
        mock_manager.get_gpu_selection_interface.return_value = mock_interface_data
        mock_manager_class.return_value = mock_manager
        
        interface_data = get_gpu_selection_ui()
        
        self.assertEqual(interface_data, mock_interface_data)
        mock_manager.get_gpu_selection_interface.assert_called_once()


        assert True  # TODO: Add proper assertion

class TestVRAMConfigProfile(unittest.TestCase):
    """Test VRAM configuration profile data class"""
    
    def test_profile_creation(self):
        """Test creating VRAM configuration profile"""
        profile = VRAMConfigProfile(
            name="test_profile",
            description="Test profile",
            manual_vram_gb={0: 16, 1: 24},
            preferred_gpu=0,
            enable_multi_gpu=True
        )
        
        self.assertEqual(profile.name, "test_profile")
        self.assertEqual(profile.description, "Test profile")
        self.assertEqual(profile.manual_vram_gb, {0: 16, 1: 24})
        self.assertEqual(profile.preferred_gpu, 0)
        self.assertTrue(profile.enable_multi_gpu)


        assert True  # TODO: Add proper assertion

class TestGPUSelectionCriteria(unittest.TestCase):
    """Test GPU selection criteria data class"""
    
    def test_criteria_creation(self):
        """Test creating GPU selection criteria"""
        criteria = GPUSelectionCriteria(
            min_vram_gb=16,
            preferred_models=["RTX 4080", "RTX 4090"],
            exclude_models=["GTX"],
            require_cuda=True
        )
        
        self.assertEqual(criteria.min_vram_gb, 16)
        self.assertEqual(criteria.preferred_models, ["RTX 4080", "RTX 4090"])
        self.assertEqual(criteria.exclude_models, ["GTX"])
        self.assertTrue(criteria.require_cuda)


        assert True  # TODO: Add proper assertion

if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests
    
    # Run tests
    unittest.main(verbosity=2)
