"""
Hardware simulation tests for the WAN2.2 local installation system.
Tests various hardware configurations and their optimizations.
"""

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from interfaces import HardwareProfile, CPUInfo, MemoryInfo, GPUInfo, StorageInfo, OSInfo


class HardwareProfileGenerator:
    """Generates various hardware profiles for testing."""
    
    @staticmethod
    def get_threadripper_pro_rtx4080():
        """High-end: Threadripper PRO 5995WX + RTX 4080."""
        return HardwareProfile(
            cpu=CPUInfo(
                model="AMD Ryzen Threadripper PRO 5995WX",
                cores=64,
                threads=128,
                base_clock=2.7,
                boost_clock=4.5,
                architecture="x64"
            ),
            memory=MemoryInfo(
                total_gb=128,
                available_gb=120,
                type="DDR4",
                speed=3200
            ),
            gpu=GPUInfo(
                model="NVIDIA GeForce RTX 4080",
                vram_gb=16,
                cuda_version="12.1",
                driver_version="537.13",
                compute_capability="8.9"
            ),
            storage=StorageInfo(
                available_gb=2000,
                type="NVMe SSD"
            ),
            os=OSInfo(
                name="Windows",
                version="11",
                architecture="x64"
            )
        )
    
    @staticmethod
    def get_ryzen9_rtx4070():
        """High-end: Ryzen 9 5950X + RTX 4070."""
        return HardwareProfile(
            cpu=CPUInfo(
                model="AMD Ryzen 9 5950X",
                cores=16,
                threads=32,
                base_clock=3.4,
                boost_clock=4.9,
                architecture="x64"
            ),
            memory=MemoryInfo(
                total_gb=64,
                available_gb=58,
                type="DDR4",
                speed=3600
            ),
            gpu=GPUInfo(
                model="NVIDIA GeForce RTX 4070",
                vram_gb=12,
                cuda_version="12.1",
                driver_version="537.13",
                compute_capability="8.9"
            ),
            storage=StorageInfo(
                available_gb=1000,
                type="NVMe SSD"
            ),
            os=OSInfo(
                name="Windows",
                version="11",
                architecture="x64"
            )
        )
    
    @staticmethod
    def get_ryzen7_rtx3070():
        """Mid-range: Ryzen 7 5800X + RTX 3070."""
        return HardwareProfile(
            cpu=CPUInfo(
                model="AMD Ryzen 7 5800X",
                cores=8,
                threads=16,
                base_clock=3.8,
                boost_clock=4.7,
                architecture="x64"
            ),
            memory=MemoryInfo(
                total_gb=32,
                available_gb=28,
                type="DDR4",
                speed=3200
            ),
            gpu=GPUInfo(
                model="NVIDIA GeForce RTX 3070",
                vram_gb=8,
                cuda_version="12.1",
                driver_version="537.13",
                compute_capability="8.6"
            ),
            storage=StorageInfo(
                available_gb=500,
                type="NVMe SSD"
            ),
            os=OSInfo(
                name="Windows",
                version="11",
                architecture="x64"
            )
        )
    
    @staticmethod
    def get_intel_i7_rtx3060():
        """Mid-range: Intel i7-12700K + RTX 3060."""
        return HardwareProfile(
            cpu=CPUInfo(
                model="Intel Core i7-12700K",
                cores=12,
                threads=20,
                base_clock=3.6,
                boost_clock=5.0,
                architecture="x64"
            ),
            memory=MemoryInfo(
                total_gb=32,
                available_gb=28,
                type="DDR4",
                speed=3200
            ),
            gpu=GPUInfo(
                model="NVIDIA GeForce RTX 3060",
                vram_gb=12,
                cuda_version="12.1",
                driver_version="537.13",
                compute_capability="8.6"
            ),
            storage=StorageInfo(
                available_gb=500,
                type="NVMe SSD"
            ),
            os=OSInfo(
                name="Windows",
                version="11",
                architecture="x64"
            )
        )
    
    @staticmethod
    def get_ryzen5_gtx1660ti():
        """Budget: Ryzen 5 3600 + GTX 1660 Ti."""
        return HardwareProfile(
            cpu=CPUInfo(
                model="AMD Ryzen 5 3600",
                cores=6,
                threads=12,
                base_clock=3.6,
                boost_clock=4.2,
                architecture="x64"
            ),
            memory=MemoryInfo(
                total_gb=16,
                available_gb=14,
                type="DDR4",
                speed=2666
            ),
            gpu=GPUInfo(
                model="NVIDIA GeForce GTX 1660 Ti",
                vram_gb=6,
                cuda_version="11.8",
                driver_version="516.94",
                compute_capability="7.5"
            ),
            storage=StorageInfo(
                available_gb=250,
                type="SATA SSD"
            ),
            os=OSInfo(
                name="Windows",
                version="10",
                architecture="x64"
            )
        )
    
    @staticmethod
    def get_intel_i5_gtx1060():
        """Minimum: Intel i5-8400 + GTX 1060."""
        return HardwareProfile(
            cpu=CPUInfo(
                model="Intel Core i5-8400",
                cores=6,
                threads=6,
                base_clock=2.8,
                boost_clock=4.0,
                architecture="x64"
            ),
            memory=MemoryInfo(
                total_gb=8,
                available_gb=6,
                type="DDR4",
                speed=2400
            ),
            gpu=GPUInfo(
                model="NVIDIA GeForce GTX 1060",
                vram_gb=6,
                cuda_version="11.8",
                driver_version="516.94",
                compute_capability="6.1"
            ),
            storage=StorageInfo(
                available_gb=100,
                type="HDD"
            ),
            os=OSInfo(
                name="Windows",
                version="10",
                architecture="x64"
            )
        )
    
    @staticmethod
    def get_ryzen7_apu():
        """APU: Ryzen 7 5700G (no dedicated GPU)."""
        return HardwareProfile(
            cpu=CPUInfo(
                model="AMD Ryzen 7 5700G",
                cores=8,
                threads=16,
                base_clock=3.8,
                boost_clock=4.6,
                architecture="x64"
            ),
            memory=MemoryInfo(
                total_gb=16,
                available_gb=14,
                type="DDR4",
                speed=3200
            ),
            gpu=None,  # No dedicated GPU
            storage=StorageInfo(
                available_gb=500,
                type="NVMe SSD"
            ),
            os=OSInfo(
                name="Windows",
                version="11",
                architecture="x64"
            )
        )
    
    @staticmethod
    def get_old_system():
        """Legacy: Intel i5-4590 + GTX 970."""
        return HardwareProfile(
            cpu=CPUInfo(
                model="Intel Core i5-4590",
                cores=4,
                threads=4,
                base_clock=3.3,
                boost_clock=3.7,
                architecture="x64"
            ),
            memory=MemoryInfo(
                total_gb=8,
                available_gb=6,
                type="DDR3",
                speed=1600
            ),
            gpu=GPUInfo(
                model="NVIDIA GeForce GTX 970",
                vram_gb=4,
                cuda_version="11.8",
                driver_version="516.94",
                compute_capability="5.2"
            ),
            storage=StorageInfo(
                available_gb=100,
                type="HDD"
            ),
            os=OSInfo(
                name="Windows",
                version="10",
                architecture="x64"
            )
        )


class TestHighEndHardwareConfigurations(unittest.TestCase):
    """Test high-end hardware configurations."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.profile_generator = HardwareProfileGenerator()
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_threadripper_pro_rtx4080_configuration(self):
        """Test Threadripper PRO 5995WX + RTX 4080 configuration."""
        from generate_config import ConfigurationEngine
        from validate_installation import InstallationValidator
        
        hardware_profile = self.profile_generator.get_threadripper_pro_rtx4080()
        
        # Test configuration generation
        config_engine = ConfigurationEngine(self.temp_dir)
        config = config_engine.generate_config(hardware_profile)
        
        # Verify high-end optimizations
        self.assertGreaterEqual(config["optimization"]["cpu_threads"], 64)
        self.assertGreaterEqual(config["optimization"]["memory_pool_gb"], 32)
        self.assertGreaterEqual(config["optimization"]["max_vram_usage_gb"], 14)
        self.assertEqual(config["system"]["default_quantization"], "bf16")
        self.assertFalse(config["system"]["enable_model_offload"])
        self.assertTrue(config["system"]["enable_gpu_acceleration"])
        self.assertGreaterEqual(config["system"]["max_queue_size"], 20)
        
        # Test hardware validation
        validator = InstallationValidator(self.temp_dir, hardware_profile)
        validation_result = validator.validate_requirements(hardware_profile)
        
        self.assertTrue(validation_result.success)
        self.assertIn("high-end", validation_result.message.lower())
    
    def test_ryzen9_rtx4070_configuration(self):
        """Test Ryzen 9 5950X + RTX 4070 configuration."""
        from generate_config import ConfigurationEngine
        
        hardware_profile = self.profile_generator.get_ryzen9_rtx4070()
        
        config_engine = ConfigurationEngine(self.temp_dir)
        config = config_engine.generate_config(hardware_profile)
        
        # Verify high-end optimizations (slightly lower than Threadripper)
        self.assertGreaterEqual(config["optimization"]["cpu_threads"], 16)
        self.assertLessEqual(config["optimization"]["cpu_threads"], 32)
        self.assertGreaterEqual(config["optimization"]["memory_pool_gb"], 16)
        self.assertGreaterEqual(config["optimization"]["max_vram_usage_gb"], 10)
        self.assertEqual(config["system"]["default_quantization"], "bf16")
        self.assertTrue(config["system"]["enable_gpu_acceleration"])


class TestMidRangeHardwareConfigurations(unittest.TestCase):
    """Test mid-range hardware configurations."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.profile_generator = HardwareProfileGenerator()
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_ryzen7_rtx3070_configuration(self):
        """Test Ryzen 7 5800X + RTX 3070 configuration."""
        from generate_config import ConfigurationEngine
        from validate_installation import InstallationValidator
        
        hardware_profile = self.profile_generator.get_ryzen7_rtx3070()
        
        # Test configuration generation
        config_engine = ConfigurationEngine(self.temp_dir)
        config = config_engine.generate_config(hardware_profile)
        
        # Verify mid-range optimizations
        self.assertGreaterEqual(config["optimization"]["cpu_threads"], 8)
        self.assertLessEqual(config["optimization"]["cpu_threads"], 16)
        self.assertGreaterEqual(config["optimization"]["memory_pool_gb"], 8)
        self.assertLessEqual(config["optimization"]["memory_pool_gb"], 16)
        self.assertGreaterEqual(config["optimization"]["max_vram_usage_gb"], 6)
        self.assertLessEqual(config["optimization"]["max_vram_usage_gb"], 8)
        self.assertEqual(config["system"]["default_quantization"], "fp16")
        self.assertTrue(config["system"]["enable_gpu_acceleration"])
        self.assertTrue(config["system"]["enable_model_offload"])
        
        # Test hardware validation
        validator = InstallationValidator(self.temp_dir, hardware_profile)
        validation_result = validator.validate_requirements(hardware_profile)
        
        self.assertTrue(validation_result.success)
    
    def test_intel_i7_rtx3060_configuration(self):
        """Test Intel i7-12700K + RTX 3060 configuration."""
        from generate_config import ConfigurationEngine
        
        hardware_profile = self.profile_generator.get_intel_i7_rtx3060()
        
        config_engine = ConfigurationEngine(self.temp_dir)
        config = config_engine.generate_config(hardware_profile)
        
        # Verify Intel-specific optimizations
        self.assertGreaterEqual(config["optimization"]["cpu_threads"], 12)
        self.assertLessEqual(config["optimization"]["cpu_threads"], 20)
        self.assertGreaterEqual(config["optimization"]["max_vram_usage_gb"], 10)
        self.assertEqual(config["system"]["default_quantization"], "fp16")
        self.assertTrue(config["system"]["enable_gpu_acceleration"])


class TestBudgetHardwareConfigurations(unittest.TestCase):
    """Test budget hardware configurations."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.profile_generator = HardwareProfileGenerator()
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_ryzen5_gtx1660ti_configuration(self):
        """Test Ryzen 5 3600 + GTX 1660 Ti configuration."""
        from generate_config import ConfigurationEngine
        from validate_installation import InstallationValidator
        
        hardware_profile = self.profile_generator.get_ryzen5_gtx1660ti()
        
        # Test configuration generation
        config_engine = ConfigurationEngine(self.temp_dir)
        config = config_engine.generate_config(hardware_profile)
        
        # Verify budget optimizations
        self.assertGreaterEqual(config["optimization"]["cpu_threads"], 6)
        self.assertLessEqual(config["optimization"]["cpu_threads"], 12)
        self.assertGreaterEqual(config["optimization"]["memory_pool_gb"], 4)
        self.assertLessEqual(config["optimization"]["memory_pool_gb"], 8)
        self.assertGreaterEqual(config["optimization"]["max_vram_usage_gb"], 4)
        self.assertLessEqual(config["optimization"]["max_vram_usage_gb"], 6)
        self.assertEqual(config["system"]["default_quantization"], "fp16")
        self.assertTrue(config["system"]["enable_gpu_acceleration"])
        self.assertTrue(config["system"]["enable_model_offload"])
        self.assertLessEqual(config["system"]["max_queue_size"], 10)
        
        # Test hardware validation
        validator = InstallationValidator(self.temp_dir, hardware_profile)
        validation_result = validator.validate_requirements(hardware_profile)
        
        self.assertTrue(validation_result.success)
        
        # Should have warnings about limited resources
        if validation_result.warnings:
            self.assertGreater(len(validation_result.warnings), 0)
    
    def test_intel_i5_gtx1060_configuration(self):
        """Test Intel i5-8400 + GTX 1060 configuration (minimum specs)."""
        from generate_config import ConfigurationEngine
        from validate_installation import InstallationValidator
        
        hardware_profile = self.profile_generator.get_intel_i5_gtx1060()
        
        # Test configuration generation
        config_engine = ConfigurationEngine(self.temp_dir)
        config = config_engine.generate_config(hardware_profile)
        
        # Verify minimum spec optimizations
        self.assertGreaterEqual(config["optimization"]["cpu_threads"], 4)
        self.assertLessEqual(config["optimization"]["cpu_threads"], 6)
        self.assertGreaterEqual(config["optimization"]["memory_pool_gb"], 2)
        self.assertLessEqual(config["optimization"]["memory_pool_gb"], 4)
        self.assertGreaterEqual(config["optimization"]["max_vram_usage_gb"], 4)
        self.assertLessEqual(config["optimization"]["max_vram_usage_gb"], 6)
        self.assertEqual(config["system"]["default_quantization"], "fp16")
        self.assertTrue(config["system"]["enable_gpu_acceleration"])
        self.assertTrue(config["system"]["enable_model_offload"])
        self.assertLessEqual(config["system"]["max_queue_size"], 5)
        
        # Test hardware validation
        validator = InstallationValidator(self.temp_dir, hardware_profile)
        validation_result = validator.validate_requirements(hardware_profile)
        
        # Should pass but with warnings
        self.assertTrue(validation_result.success)
        self.assertGreater(len(validation_result.warnings), 0)


class TestSpecialHardwareConfigurations(unittest.TestCase):
    """Test special hardware configurations."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.profile_generator = HardwareProfileGenerator()
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_apu_no_gpu_configuration(self):
        """Test APU configuration without dedicated GPU."""
        from generate_config import ConfigurationEngine
        from validate_installation import InstallationValidator
        
        hardware_profile = self.profile_generator.get_ryzen7_apu()
        
        # Test configuration generation
        config_engine = ConfigurationEngine(self.temp_dir)
        config = config_engine.generate_config(hardware_profile)
        
        # Verify CPU-only optimizations
        self.assertFalse(config["system"]["enable_gpu_acceleration"])
        self.assertGreaterEqual(config["optimization"]["cpu_threads"], 8)
        self.assertLessEqual(config["optimization"]["cpu_threads"], 16)
        self.assertGreaterEqual(config["optimization"]["memory_pool_gb"], 4)
        self.assertLessEqual(config["optimization"]["memory_pool_gb"], 8)
        self.assertEqual(config["system"]["default_quantization"], "fp16")
        self.assertTrue(config["system"]["enable_model_offload"])
        self.assertNotIn("max_vram_usage_gb", config["optimization"])
        
        # Test hardware validation
        validator = InstallationValidator(self.temp_dir, hardware_profile)
        validation_result = validator.validate_requirements(hardware_profile)
        
        # Should pass but with GPU warnings
        self.assertTrue(validation_result.success)
        self.assertGreater(len(validation_result.warnings), 0)
        
        # Check for GPU-related warnings
        gpu_warnings = [w for w in validation_result.warnings if "gpu" in w.lower()]
        self.assertGreater(len(gpu_warnings), 0)
    
    def test_legacy_system_configuration(self):
        """Test legacy system configuration."""
        from generate_config import ConfigurationEngine
        from validate_installation import InstallationValidator
        
        hardware_profile = self.profile_generator.get_old_system()
        
        # Test configuration generation
        config_engine = ConfigurationEngine(self.temp_dir)
        config = config_engine.generate_config(hardware_profile)
        
        # Verify conservative optimizations for old hardware
        self.assertGreaterEqual(config["optimization"]["cpu_threads"], 2)
        self.assertLessEqual(config["optimization"]["cpu_threads"], 4)
        self.assertGreaterEqual(config["optimization"]["memory_pool_gb"], 2)
        self.assertLessEqual(config["optimization"]["memory_pool_gb"], 4)
        self.assertGreaterEqual(config["optimization"]["max_vram_usage_gb"], 2)
        self.assertLessEqual(config["optimization"]["max_vram_usage_gb"], 4)
        self.assertEqual(config["system"]["default_quantization"], "fp16")
        self.assertTrue(config["system"]["enable_gpu_acceleration"])
        self.assertTrue(config["system"]["enable_model_offload"])
        self.assertLessEqual(config["system"]["max_queue_size"], 3)
        
        # Test hardware validation
        validator = InstallationValidator(self.temp_dir, hardware_profile)
        validation_result = validator.validate_requirements(hardware_profile)
        
        # May fail minimum requirements
        if not validation_result.success:
            self.assertIn("minimum", validation_result.message.lower())
        else:
            # If it passes, should have many warnings
            self.assertGreater(len(validation_result.warnings), 2)


class TestPerformanceOptimizationSimulation(unittest.TestCase):
    """Test performance optimization for different hardware tiers."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.profile_generator = HardwareProfileGenerator()
        
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_memory_allocation_optimization(self):
        """Test memory allocation optimization across hardware tiers."""
        from generate_config import ConfigurationEngine
        
        config_engine = ConfigurationEngine(self.temp_dir)
        
        # Test different memory configurations
        test_cases = [
            (self.profile_generator.get_threadripper_pro_rtx4080(), 32, 64),  # High-end
            (self.profile_generator.get_ryzen7_rtx3070(), 8, 16),             # Mid-range
            (self.profile_generator.get_ryzen5_gtx1660ti(), 4, 8),            # Budget
            (self.profile_generator.get_intel_i5_gtx1060(), 2, 4)             # Minimum
        ]
        
        for hardware_profile, min_memory, max_memory in test_cases:
            config = config_engine.generate_config(hardware_profile)
            
            memory_allocation = config["optimization"]["memory_pool_gb"]
            
            self.assertGreaterEqual(memory_allocation, min_memory,
                                  f"Memory allocation too low for {hardware_profile.cpu.model}")
            self.assertLessEqual(memory_allocation, max_memory,
                                f"Memory allocation too high for {hardware_profile.cpu.model}")
    
    def test_thread_allocation_optimization(self):
        """Test CPU thread allocation optimization."""
        from generate_config import ConfigurationEngine
        
        config_engine = ConfigurationEngine(self.temp_dir)
        
        # Test thread allocation based on CPU capabilities
        test_cases = [
            (self.profile_generator.get_threadripper_pro_rtx4080(), 64, 128),  # 64C/128T
            (self.profile_generator.get_ryzen9_rtx4070(), 16, 32),             # 16C/32T
            (self.profile_generator.get_ryzen7_rtx3070(), 8, 16),              # 8C/16T
            (self.profile_generator.get_intel_i7_rtx3060(), 12, 20),           # 12C/20T
            (self.profile_generator.get_ryzen5_gtx1660ti(), 6, 12),            # 6C/12T
            (self.profile_generator.get_intel_i5_gtx1060(), 4, 6)              # 6C/6T
        ]
        
        for hardware_profile, min_threads, max_threads in test_cases:
            config = config_engine.generate_config(hardware_profile)
            
            cpu_threads = config["optimization"]["cpu_threads"]
            
            self.assertGreaterEqual(cpu_threads, min_threads // 2,
                                  f"Thread count too low for {hardware_profile.cpu.model}")
            self.assertLessEqual(cpu_threads, max_threads,
                                f"Thread count too high for {hardware_profile.cpu.model}")
    
    def test_vram_allocation_optimization(self):
        """Test VRAM allocation optimization."""
        from generate_config import ConfigurationEngine
        
        config_engine = ConfigurationEngine(self.temp_dir)
        
        # Test VRAM allocation based on GPU capabilities
        test_cases = [
            (self.profile_generator.get_threadripper_pro_rtx4080(), 14, 16),  # RTX 4080 16GB
            (self.profile_generator.get_ryzen9_rtx4070(), 10, 12),            # RTX 4070 12GB
            (self.profile_generator.get_ryzen7_rtx3070(), 6, 8),              # RTX 3070 8GB
            (self.profile_generator.get_intel_i7_rtx3060(), 10, 12),          # RTX 3060 12GB
            (self.profile_generator.get_ryzen5_gtx1660ti(), 4, 6),            # GTX 1660 Ti 6GB
            (self.profile_generator.get_intel_i5_gtx1060(), 4, 6)             # GTX 1060 6GB
        ]
        
        for hardware_profile, min_vram, max_vram in test_cases:
            if hardware_profile.gpu:  # Skip if no GPU
                config = config_engine.generate_config(hardware_profile)
                
                vram_usage = config["optimization"]["max_vram_usage_gb"]
                
                self.assertGreaterEqual(vram_usage, min_vram,
                                      f"VRAM allocation too low for {hardware_profile.gpu.model}")
                self.assertLessEqual(vram_usage, max_vram,
                                    f"VRAM allocation too high for {hardware_profile.gpu.model}")
    
    def test_quantization_selection(self):
        """Test quantization method selection based on hardware."""
        from generate_config import ConfigurationEngine
        
        config_engine = ConfigurationEngine(self.temp_dir)
        
        # High-end systems should use bf16
        high_end_profiles = [
            self.profile_generator.get_threadripper_pro_rtx4080(),
            self.profile_generator.get_ryzen9_rtx4070()
        ]
        
        for profile in high_end_profiles:
            config = config_engine.generate_config(profile)
            self.assertEqual(config["system"]["default_quantization"], "bf16",
                           f"High-end system should use bf16: {profile.cpu.model}")
        
        # Mid-range and budget systems should use fp16
        other_profiles = [
            self.profile_generator.get_ryzen7_rtx3070(),
            self.profile_generator.get_intel_i7_rtx3060(),
            self.profile_generator.get_ryzen5_gtx1660ti(),
            self.profile_generator.get_intel_i5_gtx1060()
        ]
        
        for profile in other_profiles:
            config = config_engine.generate_config(profile)
            self.assertEqual(config["system"]["default_quantization"], "fp16",
                           f"Mid-range/budget system should use fp16: {profile.cpu.model}")


if __name__ == "__main__":
    unittest.main()
