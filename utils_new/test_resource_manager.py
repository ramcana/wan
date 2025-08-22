"""
Unit tests for VRAM optimization and resource management functionality
Tests proactive VRAM checking, parameter optimization, and memory cleanup strategies
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
from dataclasses import asdict

# Mock torch module if not available
if 'torch' not in sys.modules:
    torch_mock = MagicMock()
    torch_mock.cuda.is_available.return_value = False
    sys.modules['torch'] = torch_mock

# Mock psutil if not available
if 'psutil' not in sys.modules:
    psutil_mock = MagicMock()
    sys.modules['psutil'] = psutil_mock

from resource_manager import (
    VRAMOptimizer, ResourceStatus, OptimizationLevel, VRAMInfo, SystemResourceInfo,
    ResourceRequirement, OptimizationSuggestion, get_resource_manager,
    check_vram_availability, estimate_resource_requirements, 
    optimize_parameters_for_resources, cleanup_memory, get_system_resource_info
)

class TestVRAMInfo(unittest.TestCase):
    """Test VRAMInfo dataclass"""
    
    def test_vram_info_creation(self):
        """Test VRAMInfo creation and serialization"""
        vram_info = VRAMInfo(
            total_mb=8192,
            allocated_mb=4096,
            cached_mb=1024,
            free_mb=3072,
            utilization_percent=62.5
        )
        
        self.assertEqual(vram_info.total_mb, 8192)
        self.assertEqual(vram_info.allocated_mb, 4096)
        self.assertEqual(vram_info.free_mb, 3072)
        
        # Test serialization
        vram_dict = vram_info.to_dict()
        self.assertIn("total_mb", vram_dict)
        self.assertIn("utilization_percent", vram_dict)
        self.assertEqual(vram_dict["total_mb"], 8192)

class TestResourceRequirement(unittest.TestCase):
    """Test ResourceRequirement dataclass"""
    
    def test_resource_requirement_creation(self):
        """Test ResourceRequirement creation and serialization"""
        requirement = ResourceRequirement(
            model_type="t2v-A14B",
            resolution="720p",
            steps=50,
            duration=4,
            vram_mb=6000,
            ram_mb=3000,
            estimated_time_seconds=45
        )
        
        self.assertEqual(requirement.model_type, "t2v-A14B")
        self.assertEqual(requirement.vram_mb, 6000)
        self.assertEqual(requirement.optimization_level, OptimizationLevel.BASIC)
        
        # Test serialization
        req_dict = requirement.to_dict()
        self.assertIn("model_type", req_dict)
        self.assertIn("optimization_level", req_dict)
        self.assertEqual(req_dict["model_type"], "t2v-A14B")

class TestOptimizationSuggestion(unittest.TestCase):
    """Test OptimizationSuggestion dataclass"""
    
    def test_optimization_suggestion_creation(self):
        """Test OptimizationSuggestion creation and serialization"""
        suggestion = OptimizationSuggestion(
            parameter="resolution",
            current_value="1080p",
            suggested_value="720p",
            reason="Reduce VRAM usage",
            impact="Significant VRAM reduction",
            vram_savings_mb=2000
        )
        
        self.assertEqual(suggestion.parameter, "resolution")
        self.assertEqual(suggestion.vram_savings_mb, 2000)
        
        # Test serialization
        suggestion_dict = suggestion.to_dict()
        self.assertIn("parameter", suggestion_dict)
        self.assertIn("vram_savings_mb", suggestion_dict)
        self.assertEqual(suggestion_dict["parameter"], "resolution")

class TestVRAMOptimizer(unittest.TestCase):
    """Test VRAMOptimizer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            "optimization": {
                "max_vram_usage_percent": 90,
                "memory_safety_margin_mb": 1024
            }
        }
        
        # Mock GPU availability
        self.gpu_available_patcher = patch('resource_manager.torch.cuda.is_available')
        self.mock_gpu_available = self.gpu_available_patcher.start()
        self.mock_gpu_available.return_value = True
        
        # Mock GPU properties
        self.gpu_props_patcher = patch('resource_manager.torch.cuda.get_device_properties')
        self.mock_gpu_props = self.gpu_props_patcher.start()
        mock_props = Mock()
        mock_props.total_memory = 8 * 1024 * 1024 * 1024  # 8GB
        self.mock_gpu_props.return_value = mock_props
        
        # Mock GPU device name
        self.gpu_name_patcher = patch('resource_manager.torch.cuda.get_device_name')
        self.mock_gpu_name = self.gpu_name_patcher.start()
        self.mock_gpu_name.return_value = "Test GPU"
        
        # Mock GPU device count
        self.gpu_count_patcher = patch('resource_manager.torch.cuda.device_count')
        self.mock_gpu_count = self.gpu_count_patcher.start()
        self.mock_gpu_count.return_value = 1
        
        # Mock memory functions
        self.memory_allocated_patcher = patch('resource_manager.torch.cuda.memory_allocated')
        self.mock_memory_allocated = self.memory_allocated_patcher.start()
        self.mock_memory_allocated.return_value = 2 * 1024 * 1024 * 1024  # 2GB
        
        self.memory_reserved_patcher = patch('resource_manager.torch.cuda.memory_reserved')
        self.mock_memory_reserved = self.memory_reserved_patcher.start()
        self.mock_memory_reserved.return_value = 3 * 1024 * 1024 * 1024  # 3GB
        
        # Mock psutil
        self.psutil_patcher = patch('resource_manager.psutil.virtual_memory')
        self.mock_psutil = self.psutil_patcher.start()
        mock_memory = Mock()
        mock_memory.total = 16 * 1024 * 1024 * 1024  # 16GB
        mock_memory.available = 8 * 1024 * 1024 * 1024  # 8GB
        mock_memory.percent = 50.0
        self.mock_psutil.return_value = mock_memory
        
        # Mock CPU usage
        self.cpu_patcher = patch('resource_manager.psutil.cpu_percent')
        self.mock_cpu = self.cpu_patcher.start()
        self.mock_cpu.return_value = 25.0
        
        # Mock disk usage
        self.disk_patcher = patch('resource_manager.psutil.disk_usage')
        self.mock_disk = self.disk_patcher.start()
        mock_disk = Mock()
        mock_disk.free = 100 * 1024 * 1024 * 1024  # 100GB
        self.mock_disk.return_value = mock_disk
        
        self.optimizer = VRAMOptimizer(self.config)
    
    def tearDown(self):
        """Clean up test fixtures"""
        self.optimizer.stop_monitoring()
        self.gpu_available_patcher.stop()
        self.gpu_props_patcher.stop()
        self.gpu_name_patcher.stop()
        self.gpu_count_patcher.stop()
        self.memory_allocated_patcher.stop()
        self.memory_reserved_patcher.stop()
        self.psutil_patcher.stop()
        self.cpu_patcher.stop()
        self.disk_patcher.stop()
    
    def test_initialization(self):
        """Test VRAMOptimizer initialization"""
        self.assertTrue(self.optimizer.gpu_available)
        self.assertEqual(self.optimizer.gpu_count, 1)
        self.assertEqual(self.optimizer.gpu_name, "Test GPU")
        self.assertEqual(self.optimizer.total_vram, 8 * 1024 * 1024 * 1024)
    
    def test_get_vram_info(self):
        """Test VRAM information retrieval"""
        vram_info = self.optimizer.get_vram_info()
        
        self.assertIsInstance(vram_info, VRAMInfo)
        self.assertEqual(vram_info.total_mb, 8192)  # 8GB in MB
        self.assertEqual(vram_info.allocated_mb, 2048)  # 2GB in MB
        self.assertEqual(vram_info.cached_mb, 3072)  # 3GB in MB
        self.assertEqual(vram_info.free_mb, 5120)  # 8GB - 3GB = 5GB in MB
        self.assertAlmostEqual(vram_info.utilization_percent, 37.5, places=1)
    
    def test_get_vram_info_no_gpu(self):
        """Test VRAM info when no GPU is available"""
        self.mock_gpu_available.return_value = False
        optimizer_no_gpu = VRAMOptimizer(self.config)
        
        vram_info = optimizer_no_gpu.get_vram_info()
        self.assertEqual(vram_info.total_mb, 0)
        self.assertEqual(vram_info.free_mb, 0)
        self.assertEqual(vram_info.utilization_percent, 0)
    
    def test_get_system_resource_info(self):
        """Test system resource information retrieval"""
        resource_info = self.optimizer.get_system_resource_info()
        
        self.assertIsInstance(resource_info, SystemResourceInfo)
        self.assertIsInstance(resource_info.vram, VRAMInfo)
        self.assertEqual(resource_info.ram_total_gb, 16)
        self.assertEqual(resource_info.ram_available_gb, 8)
        self.assertEqual(resource_info.ram_usage_percent, 50.0)
        self.assertEqual(resource_info.cpu_usage_percent, 25.0)
        self.assertAlmostEqual(resource_info.disk_free_gb, 100, places=1)
    
    def test_check_vram_availability_sufficient(self):
        """Test VRAM availability check with sufficient memory"""
        available, message = self.optimizer.check_vram_availability(4000)  # 4GB
        
        self.assertTrue(available)
        self.assertIn("Sufficient VRAM available", message)
        self.assertIn("5120", message)  # Free VRAM amount
    
    def test_check_vram_availability_insufficient(self):
        """Test VRAM availability check with insufficient memory"""
        available, message = self.optimizer.check_vram_availability(6000)  # 6GB
        
        self.assertFalse(available)
        self.assertIn("Insufficient VRAM", message)
        self.assertIn("short", message)
    
    def test_check_vram_availability_no_gpu(self):
        """Test VRAM availability check with no GPU"""
        self.mock_gpu_available.return_value = False
        optimizer_no_gpu = VRAMOptimizer(self.config)
        
        available, message = optimizer_no_gpu.check_vram_availability(1000)
        
        self.assertFalse(available)
        self.assertEqual(message, "No GPU available")
    
    def test_estimate_resource_requirements_t2v(self):
        """Test resource requirement estimation for T2V model"""
        requirement = self.optimizer.estimate_resource_requirements(
            model_type="t2v-A14B",
            resolution="720p",
            steps=50,
            duration=4,
            lora_count=0
        )
        
        self.assertIsInstance(requirement, ResourceRequirement)
        self.assertEqual(requirement.model_type, "t2v-A14B")
        self.assertEqual(requirement.resolution, "720p")
        self.assertEqual(requirement.steps, 50)
        self.assertEqual(requirement.duration, 4)
        self.assertGreater(requirement.vram_mb, 0)
        self.assertGreater(requirement.ram_mb, 0)
        self.assertGreater(requirement.estimated_time_seconds, 0)
    
    def test_estimate_resource_requirements_different_resolutions(self):
        """Test resource estimation with different resolutions"""
        req_480p = self.optimizer.estimate_resource_requirements("t2v-A14B", "480p", 50)
        req_720p = self.optimizer.estimate_resource_requirements("t2v-A14B", "720p", 50)
        req_1080p = self.optimizer.estimate_resource_requirements("t2v-A14B", "1080p", 50)
        
        # Higher resolution should require more resources
        self.assertLess(req_480p.vram_mb, req_720p.vram_mb)
        self.assertLess(req_720p.vram_mb, req_1080p.vram_mb)
    
    def test_estimate_resource_requirements_with_lora(self):
        """Test resource estimation with LoRA"""
        req_no_lora = self.optimizer.estimate_resource_requirements("t2v-A14B", "720p", 50, lora_count=0)
        req_with_lora = self.optimizer.estimate_resource_requirements("t2v-A14B", "720p", 50, lora_count=2)
        
        # LoRA should increase resource requirements
        self.assertLess(req_no_lora.vram_mb, req_with_lora.vram_mb)
        self.assertLess(req_no_lora.ram_mb, req_with_lora.ram_mb)
    
    def test_optimize_parameters_for_resources_sufficient(self):
        """Test parameter optimization with sufficient resources"""
        params = {
            "model_type": "t2v-A14B",
            "resolution": "720p",
            "steps": 50,
            "duration": 4,
            "lora_config": {}
        }
        
        optimized_params, suggestions = self.optimizer.optimize_parameters_for_resources(params)
        
        self.assertIsInstance(optimized_params, dict)
        self.assertIsInstance(suggestions, list)
        # With sufficient resources, minimal changes expected
        self.assertEqual(optimized_params["resolution"], "720p")
    
    def test_optimize_parameters_for_resources_insufficient_vram(self):
        """Test parameter optimization with insufficient VRAM"""
        # Simulate low VRAM
        self.mock_memory_allocated.return_value = 7 * 1024 * 1024 * 1024  # 7GB allocated
        self.mock_memory_reserved.return_value = 7.5 * 1024 * 1024 * 1024  # 7.5GB reserved
        
        params = {
            "model_type": "t2v-A14B",
            "resolution": "1080p",
            "steps": 60,
            "duration": 4,
            "lora_config": {}
        }
        
        optimized_params, suggestions = self.optimizer.optimize_parameters_for_resources(params)
        
        # Should suggest optimizations
        self.assertGreater(len(suggestions), 0)
        # Should reduce resolution or steps
        self.assertTrue(
            optimized_params["resolution"] != "1080p" or 
            optimized_params["steps"] < 60 or
            "optimization_settings" in optimized_params
        )
    
    def test_cleanup_memory(self):
        """Test memory cleanup functionality"""
        with patch('resource_manager.torch.cuda.empty_cache') as mock_empty_cache, \
             patch('resource_manager.gc.collect') as mock_gc_collect:
            
            result = self.optimizer.cleanup_memory(aggressive=False)
            
            self.assertIsInstance(result, dict)
            self.assertIn("vram_before_mb", result)
            self.assertIn("vram_after_mb", result)
            self.assertIn("actions_taken", result)
            
            # Should call cleanup functions
            mock_empty_cache.assert_called()
            mock_gc_collect.assert_called()
            
            # Should record actions
            self.assertIn("cleared_gpu_cache", result["actions_taken"])
            self.assertIn("garbage_collection", result["actions_taken"])
    
    def test_cleanup_memory_aggressive(self):
        """Test aggressive memory cleanup"""
        with patch('resource_manager.torch.cuda.empty_cache') as mock_empty_cache, \
             patch('resource_manager.torch.cuda.reset_peak_memory_stats') as mock_reset_stats, \
             patch('resource_manager.gc.collect') as mock_gc_collect:
            
            result = self.optimizer.cleanup_memory(aggressive=True)
            
            # Should perform additional aggressive cleanup
            self.assertIn("aggressive_gpu_cleanup", result["actions_taken"])
            self.assertIn("reset_memory_stats", result["actions_taken"])
            self.assertIn("aggressive_garbage_collection", result["actions_taken"])
            
            # Should call additional functions
            mock_reset_stats.assert_called()
            # gc.collect should be called multiple times
            self.assertGreater(mock_gc_collect.call_count, 1)
    
    def test_get_resource_status(self):
        """Test resource status determination"""
        # Test optimal status
        status = self.optimizer.get_resource_status()
        self.assertEqual(status, ResourceStatus.AVAILABLE)  # 37.5% VRAM, 50% RAM
        
        # Test critical status
        self.mock_memory_allocated.return_value = 7.8 * 1024 * 1024 * 1024  # 96% VRAM
        self.mock_memory_reserved.return_value = 7.8 * 1024 * 1024 * 1024
        
        status = self.optimizer.get_resource_status()
        self.assertEqual(status, ResourceStatus.CRITICAL)
    
    def test_resource_monitoring(self):
        """Test resource monitoring functionality"""
        # Resource monitoring should start automatically
        self.assertTrue(self.optimizer.monitoring_enabled)
        
        # Should have monitoring thread
        self.assertIsNotNone(self.optimizer.monitoring_thread)
        
        # Test stopping monitoring
        self.optimizer.stop_monitoring()
        self.assertFalse(self.optimizer.monitoring_enabled)
    
    def test_get_resource_history(self):
        """Test resource history retrieval"""
        # Add some mock history
        mock_resource_info = SystemResourceInfo(
            vram=VRAMInfo(8192, 2048, 3072, 5120, 37.5),
            ram_total_gb=16,
            ram_available_gb=8,
            ram_usage_percent=50.0,
            cpu_usage_percent=25.0,
            disk_free_gb=100
        )
        self.optimizer.resource_history = [mock_resource_info] * 5
        
        history = self.optimizer.get_resource_history(last_n=3)
        
        self.assertEqual(len(history), 3)
        self.assertIsInstance(history[0], dict)
        self.assertIn("vram", history[0])
        self.assertIn("ram_total_gb", history[0])

class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock GPU availability
        self.gpu_available_patcher = patch('resource_manager.torch.cuda.is_available')
        self.mock_gpu_available = self.gpu_available_patcher.start()
        self.mock_gpu_available.return_value = True
        
        # Mock other GPU functions
        self.gpu_props_patcher = patch('resource_manager.torch.cuda.get_device_properties')
        self.mock_gpu_props = self.gpu_props_patcher.start()
        mock_props = Mock()
        mock_props.total_memory = 8 * 1024 * 1024 * 1024
        self.mock_gpu_props.return_value = mock_props
        
        self.gpu_name_patcher = patch('resource_manager.torch.cuda.get_device_name')
        self.mock_gpu_name = self.gpu_name_patcher.start()
        self.mock_gpu_name.return_value = "Test GPU"
        
        self.gpu_count_patcher = patch('resource_manager.torch.cuda.device_count')
        self.mock_gpu_count = self.gpu_count_patcher.start()
        self.mock_gpu_count.return_value = 1
        
        self.memory_allocated_patcher = patch('resource_manager.torch.cuda.memory_allocated')
        self.mock_memory_allocated = self.memory_allocated_patcher.start()
        self.mock_memory_allocated.return_value = 2 * 1024 * 1024 * 1024
        
        self.memory_reserved_patcher = patch('resource_manager.torch.cuda.memory_reserved')
        self.mock_memory_reserved = self.memory_reserved_patcher.start()
        self.mock_memory_reserved.return_value = 3 * 1024 * 1024 * 1024
        
        # Mock psutil
        self.psutil_patcher = patch('resource_manager.psutil.virtual_memory')
        self.mock_psutil = self.psutil_patcher.start()
        mock_memory = Mock()
        mock_memory.total = 16 * 1024 * 1024 * 1024
        mock_memory.available = 8 * 1024 * 1024 * 1024
        mock_memory.percent = 50.0
        self.mock_psutil.return_value = mock_memory
        
        self.cpu_patcher = patch('resource_manager.psutil.cpu_percent')
        self.mock_cpu = self.cpu_patcher.start()
        self.mock_cpu.return_value = 25.0
        
        self.disk_patcher = patch('resource_manager.psutil.disk_usage')
        self.mock_disk = self.disk_patcher.start()
        mock_disk = Mock()
        mock_disk.free = 100 * 1024 * 1024 * 1024
        self.mock_disk.return_value = mock_disk
    
    def tearDown(self):
        """Clean up test fixtures"""
        # Reset global resource manager
        import resource_manager
        resource_manager._resource_manager = None
        
        self.gpu_available_patcher.stop()
        self.gpu_props_patcher.stop()
        self.gpu_name_patcher.stop()
        self.gpu_count_patcher.stop()
        self.memory_allocated_patcher.stop()
        self.memory_reserved_patcher.stop()
        self.psutil_patcher.stop()
        self.cpu_patcher.stop()
        self.disk_patcher.stop()
    
    def test_get_resource_manager(self):
        """Test global resource manager retrieval"""
        manager1 = get_resource_manager()
        manager2 = get_resource_manager()
        
        # Should return the same instance (singleton)
        self.assertIs(manager1, manager2)
        self.assertIsInstance(manager1, VRAMOptimizer)
    
    def test_check_vram_availability_function(self):
        """Test convenience function for VRAM availability check"""
        available, message = check_vram_availability(4000)
        
        self.assertIsInstance(available, bool)
        self.assertIsInstance(message, str)
        self.assertTrue(available)  # Should be available with our mock setup
    
    def test_estimate_resource_requirements_function(self):
        """Test convenience function for resource estimation"""
        requirement = estimate_resource_requirements("t2v-A14B", "720p", 50)
        
        self.assertIsInstance(requirement, ResourceRequirement)
        self.assertEqual(requirement.model_type, "t2v-A14B")
        self.assertEqual(requirement.resolution, "720p")
    
    def test_optimize_parameters_for_resources_function(self):
        """Test convenience function for parameter optimization"""
        params = {
            "model_type": "t2v-A14B",
            "resolution": "720p",
            "steps": 50,
            "duration": 4,
            "lora_config": {}
        }
        
        optimized_params, suggestions = optimize_parameters_for_resources(params)
        
        self.assertIsInstance(optimized_params, dict)
        self.assertIsInstance(suggestions, list)
    
    def test_cleanup_memory_function(self):
        """Test convenience function for memory cleanup"""
        with patch('resource_manager.torch.cuda.empty_cache'), patch('resource_manager.gc.collect'):
            result = cleanup_memory()
            
            self.assertIsInstance(result, dict)
            self.assertIn("actions_taken", result)
    
    def test_get_system_resource_info_function(self):
        """Test convenience function for system resource info"""
        resource_info = get_system_resource_info()
        
        self.assertIsInstance(resource_info, SystemResourceInfo)
        self.assertIsInstance(resource_info.vram, VRAMInfo)

class TestErrorHandling(unittest.TestCase):
    """Test error handling in resource management"""
    
    def test_vram_info_with_cuda_error(self):
        """Test VRAM info retrieval with CUDA errors"""
        config = {"optimization": {}}
        
        with patch('resource_manager.torch.cuda.is_available', return_value=True), \
             patch('resource_manager.torch.cuda.memory_allocated', side_effect=RuntimeError("CUDA error")):
            
            optimizer = VRAMOptimizer(config)
            vram_info = optimizer.get_vram_info()
            
            # Should return zero values on error
            self.assertEqual(vram_info.total_mb, 0)
            self.assertEqual(vram_info.free_mb, 0)
    
    def test_resource_estimation_with_invalid_model(self):
        """Test resource estimation with invalid model type"""
        config = {"optimization": {}}
        
        with patch('resource_manager.torch.cuda.is_available', return_value=False):
            optimizer = VRAMOptimizer(config)
            
            # Should handle unknown model gracefully
            requirement = optimizer.estimate_resource_requirements(
                "unknown-model", "720p", 50
            )
            
            self.assertIsInstance(requirement, ResourceRequirement)
            self.assertGreater(requirement.vram_mb, 0)  # Should return conservative estimate
    
    def test_parameter_optimization_with_error(self):
        """Test parameter optimization with errors"""
        config = {"optimization": {}}
        
        with patch('resource_manager.torch.cuda.is_available', return_value=False):
            optimizer = VRAMOptimizer(config)
            
            # Should handle errors gracefully
            params = {"invalid": "params"}
            optimized_params, suggestions = optimizer.optimize_parameters_for_resources(params)
            
            self.assertIsInstance(optimized_params, dict)
            self.assertIsInstance(suggestions, list)

if __name__ == '__main__':
    unittest.main()