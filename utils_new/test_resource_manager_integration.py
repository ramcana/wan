"""
Integration tests for resource manager functionality
Tests the actual functionality without mocking external dependencies
"""

import unittest
import sys
from unittest.mock import Mock, patch, MagicMock

# Mock torch and psutil if not available
if 'torch' not in sys.modules:
    torch_mock = MagicMock()
    torch_mock.cuda.is_available.return_value = False
    torch_mock.cuda.memory_allocated.return_value = 0
    torch_mock.cuda.memory_reserved.return_value = 0
    torch_mock.cuda.get_device_properties.return_value.total_memory = 0
    torch_mock.cuda.get_device_name.return_value = "No GPU"
    torch_mock.cuda.device_count.return_value = 0
    torch_mock.cuda.empty_cache = Mock()
    torch_mock.cuda.reset_peak_memory_stats = Mock()
    sys.modules['torch'] = torch_mock

if 'psutil' not in sys.modules:
    psutil_mock = MagicMock()
    memory_mock = Mock()
    memory_mock.total = 16 * 1024 * 1024 * 1024  # 16GB
    memory_mock.available = 8 * 1024 * 1024 * 1024  # 8GB
    memory_mock.percent = 50.0
    psutil_mock.virtual_memory.return_value = memory_mock
    psutil_mock.cpu_percent.return_value = 25.0
    disk_mock = Mock()
    disk_mock.free = 100 * 1024 * 1024 * 1024  # 100GB
    psutil_mock.disk_usage.return_value = disk_mock
    sys.modules['psutil'] = psutil_mock

if 'gc' not in sys.modules:
    gc_mock = MagicMock()
    gc_mock.collect = Mock()
    sys.modules['gc'] = gc_mock

from resource_manager import (
    VRAMOptimizer, ResourceStatus, OptimizationLevel, VRAMInfo, SystemResourceInfo,
    ResourceRequirement, OptimizationSuggestion, get_resource_manager,
    check_vram_availability, estimate_resource_requirements, 
    optimize_parameters_for_resources, cleanup_memory, get_system_resource_info
)

class TestResourceManagerIntegration(unittest.TestCase):
    """Integration tests for resource manager"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            "optimization": {
                "max_vram_usage_percent": 90,
                "memory_safety_margin_mb": 1024
            }
        }
    
    def test_vram_optimizer_creation(self):
        """Test VRAMOptimizer can be created"""
        optimizer = VRAMOptimizer(self.config)
        self.assertIsInstance(optimizer, VRAMOptimizer)
        self.assertEqual(optimizer.max_vram_usage_percent, 90)
        self.assertEqual(optimizer.memory_safety_margin_mb, 1024)
    
    def test_vram_info_structure(self):
        """Test VRAMInfo data structure"""
        vram_info = VRAMInfo(
            total_mb=8192,
            allocated_mb=4096,
            cached_mb=1024,
            free_mb=3072,
            utilization_percent=62.5
        )
        
        # Test serialization
        vram_dict = vram_info.to_dict()
        expected_keys = ["total_mb", "allocated_mb", "cached_mb", "free_mb", "utilization_percent"]
        for key in expected_keys:
            self.assertIn(key, vram_dict)
    
    def test_resource_requirement_structure(self):
        """Test ResourceRequirement data structure"""
        requirement = ResourceRequirement(
            model_type="t2v-A14B",
            resolution="720p",
            steps=50,
            duration=4,
            vram_mb=6000,
            ram_mb=3000,
            estimated_time_seconds=45
        )
        
        # Test serialization
        req_dict = requirement.to_dict()
        expected_keys = ["model_type", "resolution", "steps", "duration", "vram_mb", "ram_mb", "estimated_time_seconds", "optimization_level"]
        for key in expected_keys:
            self.assertIn(key, req_dict)
    
    def test_optimization_suggestion_structure(self):
        """Test OptimizationSuggestion data structure"""
        suggestion = OptimizationSuggestion(
            parameter="resolution",
            current_value="1080p",
            suggested_value="720p",
            reason="Reduce VRAM usage",
            impact="Significant VRAM reduction",
            vram_savings_mb=2000
        )
        
        # Test serialization
        suggestion_dict = suggestion.to_dict()
        expected_keys = ["parameter", "current_value", "suggested_value", "reason", "impact", "vram_savings_mb"]
        for key in expected_keys:
            self.assertIn(key, suggestion_dict)
    
    def test_system_resource_info_structure(self):
        """Test SystemResourceInfo data structure"""
        vram_info = VRAMInfo(8192, 4096, 1024, 3072, 62.5)
        resource_info = SystemResourceInfo(
            vram=vram_info,
            ram_total_gb=16,
            ram_available_gb=8,
            ram_usage_percent=50.0,
            cpu_usage_percent=25.0,
            disk_free_gb=100
        )
        
        # Test serialization
        info_dict = resource_info.to_dict()
        expected_keys = ["vram", "ram_total_gb", "ram_available_gb", "ram_usage_percent", "cpu_usage_percent", "disk_free_gb"]
        for key in expected_keys:
            self.assertIn(key, info_dict)
    
    def test_resource_estimation_logic(self):
        """Test resource estimation logic"""
        optimizer = VRAMOptimizer(self.config)
        
        # Test T2V model estimation
        req_t2v = optimizer.estimate_resource_requirements("t2v-A14B", "720p", 50)
        self.assertEqual(req_t2v.model_type, "t2v-A14B")
        self.assertEqual(req_t2v.resolution, "720p")
        self.assertEqual(req_t2v.steps, 50)
        self.assertGreater(req_t2v.vram_mb, 0)
        self.assertGreater(req_t2v.ram_mb, 0)
        
        # Test I2V model estimation
        req_i2v = optimizer.estimate_resource_requirements("i2v-A14B", "720p", 50)
        self.assertEqual(req_i2v.model_type, "i2v-A14B")
        
        # Test TI2V model estimation
        req_ti2v = optimizer.estimate_resource_requirements("ti2v-5B", "720p", 50)
        self.assertEqual(req_ti2v.model_type, "ti2v-5B")
        
        # TI2V should require less VRAM than T2V and I2V
        self.assertLess(req_ti2v.vram_mb, req_t2v.vram_mb)
        self.assertLess(req_ti2v.vram_mb, req_i2v.vram_mb)
    
    def test_resolution_scaling(self):
        """Test resource scaling with different resolutions"""
        optimizer = VRAMOptimizer(self.config)
        
        req_480p = optimizer.estimate_resource_requirements("t2v-A14B", "480p", 50)
        req_720p = optimizer.estimate_resource_requirements("t2v-A14B", "720p", 50)
        req_1080p = optimizer.estimate_resource_requirements("t2v-A14B", "1080p", 50)
        
        # Higher resolution should require more resources
        self.assertLess(req_480p.vram_mb, req_720p.vram_mb)
        self.assertLess(req_720p.vram_mb, req_1080p.vram_mb)
        
        # Time should also scale with resolution
        self.assertLess(req_480p.estimated_time_seconds, req_720p.estimated_time_seconds)
        self.assertLess(req_720p.estimated_time_seconds, req_1080p.estimated_time_seconds)
    
    def test_step_scaling(self):
        """Test resource scaling with different step counts"""
        optimizer = VRAMOptimizer(self.config)
        
        req_25_steps = optimizer.estimate_resource_requirements("t2v-A14B", "720p", 25)
        req_50_steps = optimizer.estimate_resource_requirements("t2v-A14B", "720p", 50)
        req_100_steps = optimizer.estimate_resource_requirements("t2v-A14B", "720p", 100)
        
        # More steps should require more resources
        self.assertLess(req_25_steps.vram_mb, req_50_steps.vram_mb)
        self.assertLess(req_50_steps.vram_mb, req_100_steps.vram_mb)
        
        # Time should scale with steps
        self.assertLess(req_25_steps.estimated_time_seconds, req_50_steps.estimated_time_seconds)
        self.assertLess(req_50_steps.estimated_time_seconds, req_100_steps.estimated_time_seconds)
    
    def test_lora_overhead(self):
        """Test LoRA overhead calculation"""
        optimizer = VRAMOptimizer(self.config)
        
        req_no_lora = optimizer.estimate_resource_requirements("t2v-A14B", "720p", 50, lora_count=0)
        req_with_lora = optimizer.estimate_resource_requirements("t2v-A14B", "720p", 50, lora_count=2)
        
        # LoRA should increase resource requirements
        self.assertLess(req_no_lora.vram_mb, req_with_lora.vram_mb)
        self.assertLess(req_no_lora.ram_mb, req_with_lora.ram_mb)
    
    def test_parameter_optimization_basic(self):
        """Test basic parameter optimization"""
        optimizer = VRAMOptimizer(self.config)
        
        params = {
            "model_type": "t2v-A14B",
            "resolution": "720p",
            "steps": 50,
            "duration": 4,
            "lora_config": {}
        }
        
        optimized_params, suggestions = optimizer.optimize_parameters_for_resources(params)
        
        # Should return valid results
        self.assertIsInstance(optimized_params, dict)
        self.assertIsInstance(suggestions, list)
        
        # Should preserve basic structure
        self.assertIn("model_type", optimized_params)
        self.assertIn("resolution", optimized_params)
        self.assertIn("steps", optimized_params)
    
    def test_memory_cleanup_structure(self):
        """Test memory cleanup result structure"""
        optimizer = VRAMOptimizer(self.config)
        
        result = optimizer.cleanup_memory(aggressive=False)
        
        # Should return proper structure
        expected_keys = ["vram_before_mb", "vram_after_mb", "vram_freed_mb", 
                        "ram_before_mb", "ram_after_mb", "ram_freed_mb", "actions_taken"]
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Actions should be recorded
        self.assertIsInstance(result["actions_taken"], list)
    
    def test_resource_status_determination(self):
        """Test resource status determination logic"""
        optimizer = VRAMOptimizer(self.config)
        
        # Should return a valid status
        status = optimizer.get_resource_status()
        self.assertIsInstance(status, ResourceStatus)
        self.assertIn(status, [ResourceStatus.OPTIMAL, ResourceStatus.AVAILABLE, 
                              ResourceStatus.LIMITED, ResourceStatus.INSUFFICIENT, 
                              ResourceStatus.CRITICAL])
    
    def test_convenience_functions(self):
        """Test convenience functions work"""
        # Test global resource manager
        manager1 = get_resource_manager()
        manager2 = get_resource_manager()
        self.assertIs(manager1, manager2)  # Should be singleton
        
        # Test convenience functions
        available, message = check_vram_availability(4000)
        self.assertIsInstance(available, bool)
        self.assertIsInstance(message, str)
        
        requirement = estimate_resource_requirements("t2v-A14B", "720p", 50)
        self.assertIsInstance(requirement, ResourceRequirement)
        
        params = {"model_type": "t2v-A14B", "resolution": "720p", "steps": 50}
        optimized_params, suggestions = optimize_parameters_for_resources(params)
        self.assertIsInstance(optimized_params, dict)
        self.assertIsInstance(suggestions, list)
        
        cleanup_result = cleanup_memory()
        self.assertIsInstance(cleanup_result, dict)
        
        resource_info = get_system_resource_info()
        self.assertIsInstance(resource_info, SystemResourceInfo)

class TestResourceManagerErrorHandling(unittest.TestCase):
    """Test error handling in resource manager"""
    
    def test_invalid_model_type(self):
        """Test handling of invalid model types"""
        config = {"optimization": {}}
        optimizer = VRAMOptimizer(config)
        
        # Should handle unknown model gracefully
        requirement = optimizer.estimate_resource_requirements("unknown-model", "720p", 50)
        self.assertIsInstance(requirement, ResourceRequirement)
        self.assertGreater(requirement.vram_mb, 0)  # Should return conservative estimate
    
    def test_invalid_resolution(self):
        """Test handling of invalid resolutions"""
        config = {"optimization": {}}
        optimizer = VRAMOptimizer(config)
        
        # Should handle unknown resolution gracefully
        requirement = optimizer.estimate_resource_requirements("t2v-A14B", "unknown-res", 50)
        self.assertIsInstance(requirement, ResourceRequirement)
        self.assertGreater(requirement.vram_mb, 0)
    
    def test_edge_case_parameters(self):
        """Test handling of edge case parameters"""
        config = {"optimization": {}}
        optimizer = VRAMOptimizer(config)
        
        # Test with extreme values
        requirement = optimizer.estimate_resource_requirements("t2v-A14B", "720p", 1, duration=1)
        self.assertIsInstance(requirement, ResourceRequirement)
        self.assertGreater(requirement.vram_mb, 0)
        
        requirement = optimizer.estimate_resource_requirements("t2v-A14B", "720p", 1000, duration=100)
        self.assertIsInstance(requirement, ResourceRequirement)
        self.assertGreater(requirement.vram_mb, 0)
    
    def test_parameter_optimization_with_invalid_params(self):
        """Test parameter optimization with invalid parameters"""
        config = {"optimization": {}}
        optimizer = VRAMOptimizer(config)
        
        # Should handle missing or invalid parameters gracefully
        params = {"invalid": "params"}
        optimized_params, suggestions = optimizer.optimize_parameters_for_resources(params)
        
        self.assertIsInstance(optimized_params, dict)
        self.assertIsInstance(suggestions, list)

if __name__ == '__main__':
    unittest.main()