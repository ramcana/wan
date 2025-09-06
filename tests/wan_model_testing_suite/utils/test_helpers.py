"""
Test helper utilities for WAN model testing
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from unittest.mock import Mock, MagicMock
import json
import time
import random


class MockWANModel:
    """Mock WAN model for testing"""
    
    def __init__(self, model_type: str = "t2v-A14B", **kwargs):
        self.model_type = model_type
        self.is_loaded = True
        self.device = kwargs.get("device", "cpu")
        self.precision = kwargs.get("precision", "fp32")
        self.capabilities = self._create_mock_capabilities()
        self.progress_tracker = Mock()
        
        # Mock methods
        self.generate = Mock(side_effect=self._mock_generate)
        self.generate_batch = Mock(side_effect=self._mock_generate_batch)
        self.validate_generation_params = Mock(return_value=(True, []))
        self.get_memory_usage = Mock(return_value={"allocated_gb": 8.5, "cached_gb": 1.2})
        self.optimize_for_hardware = Mock()
        self.enable_cpu_offload = Mock()
        self.enable_quantization = Mock()
        self.preprocess_image = Mock(side_effect=self._mock_preprocess_image)
        self._encode_text = Mock(side_effect=self._mock_encode_text)
        self._encode_image = Mock(side_effect=self._mock_encode_image)
    
    def _create_mock_capabilities(self):
        """Create mock capabilities based on model type"""
        capabilities = Mock()
        
        if "t2v" in self.model_type.lower():
            capabilities.supports_text_conditioning = True
            capabilities.supports_image_conditioning = False
            capabilities.supports_dual_conditioning = False
        elif "i2v" in self.model_type.lower():
            capabilities.supports_text_conditioning = True
            capabilities.supports_image_conditioning = True
            capabilities.supports_dual_conditioning = True
        elif "ti2v" in self.model_type.lower():
            capabilities.supports_text_conditioning = True
            capabilities.supports_image_conditioning = True
            capabilities.supports_dual_conditioning = True
        
        capabilities.supports_lora = True
        capabilities.supports_cpu_offload = True
        capabilities.supports_quantization = True
        capabilities.supports_chunked_inference = True
        capabilities.max_frames = 16
        capabilities.max_resolution = (1280, 720)
        capabilities.min_resolution = (256, 256)
        capabilities.supported_precisions = ["fp32", "fp16", "bf16"]
        
        if "5B" in self.model_type:
            capabilities.estimated_vram_gb = 6.5
            capabilities.parameter_count = 5_000_000_000
        else:
            capabilities.estimated_vram_gb = 10.5
            capabilities.parameter_count = 14_000_000_000
        
        return capabilities
    
    def _mock_generate(self, params):
        """Mock generation method"""
        # Simulate generation time based on parameters
        base_time = 0.1
        if hasattr(params, 'num_frames'):
            base_time *= params.num_frames / 16
        if hasattr(params, 'width') and hasattr(params, 'height'):
            pixel_factor = (params.width * params.height) / (512 * 512)
            base_time *= pixel_factor
        
        time.sleep(base_time)
        
        # Create mock result
        result = Mock()
        result.success = True
        result.generation_time = base_time
        result.video_path = f"/tmp/mock_video_{int(time.time())}.mp4"
        result.metadata = {
            "model": self.model_type,
            "frames": getattr(params, 'num_frames', 16),
            "resolution": f"{getattr(params, 'width', 512)}x{getattr(params, 'height', 512)}",
            "fps": getattr(params, 'fps', 8.0)
        }
        result.frames = [Mock() for _ in range(getattr(params, 'num_frames', 16))]
        
        return result
    
    def _mock_generate_batch(self, params_list):
        """Mock batch generation method"""
        results = []
        for i, params in enumerate(params_list):
            result = self._mock_generate(params)
            result.video_path = f"/tmp/mock_batch_video_{i}_{int(time.time())}.mp4"
            results.append(result)
        return results
    
    def _mock_preprocess_image(self, image):
        """Mock image preprocessing"""
        # Return a mock tensor
        mock_tensor = Mock()
        mock_tensor.shape = (1, 3, 512, 512)
        mock_tensor.device = self.device
        return mock_tensor
    
    def _mock_encode_text(self, prompt, negative_prompt=None):
        """Mock text encoding"""
        mock_positive = Mock()
        mock_positive.shape = (1, 256, 1536 if "A14B" in self.model_type else 1024)
        
        mock_negative = Mock()
        mock_negative.shape = (1, 256, 1536 if "A14B" in self.model_type else 1024)
        
        return mock_positive, mock_negative
    
    def _mock_encode_image(self, image):
        """Mock image encoding"""
        hidden_dim = 1536 if "A14B" in self.model_type else 1024
        
        mock_global = Mock()
        mock_global.shape = (1, hidden_dim)
        
        mock_spatial = Mock()
        mock_spatial.shape = (1, 256, hidden_dim)
        
        return mock_global, mock_spatial


class MockHardwareInfo:
    """Mock hardware information for testing"""
    
    @staticmethod
    def get_rtx4080_specs():
        """Get RTX 4080 specifications"""
        return {
            "gpu_name": "NVIDIA GeForce RTX 4080",
            "gpu_memory_gb": 16.0,
            "cuda_cores": 9728,
            "rt_cores": 76,
            "tensor_cores": 304,
            "base_clock_mhz": 2205,
            "boost_clock_mhz": 2505,
            "memory_bandwidth_gbps": 716.8,
            "cuda_compute_capability": "8.9",
            "cuda_version": "12.1",
            "driver_version": "535.98"
        }
    
    @staticmethod
    def get_threadripper_pro_specs():
        """Get Threadripper PRO specifications"""
        return {
            "cpu_name": "AMD Ryzen Threadripper PRO 5975WX",
            "cpu_cores": 32,
            "cpu_threads": 64,
            "base_clock_ghz": 3.6,
            "boost_clock_ghz": 4.5,
            "cache_l3_mb": 128,
            "memory_channels": 8,
            "memory_capacity_gb": 128,
            "memory_speed_mhz": 3200,
            "memory_bandwidth_gbps": 204.8,
            "pcie_lanes": 128,
            "tdp_watts": 280,
            "architecture": "Zen 3",
            "process_node": "7nm",
            "numa_nodes": 2
        }
    
    @staticmethod
    def get_combined_specs():
        """Get combined RTX 4080 + Threadripper PRO specifications"""
        rtx_specs = MockHardwareInfo.get_rtx4080_specs()
        threadripper_specs = MockHardwareInfo.get_threadripper_pro_specs()
        
        return {**rtx_specs, **threadripper_specs}


class TestDataGenerator:
    """Generate test data for WAN model testing"""
    
    @staticmethod
    def generate_sample_prompts(count: int = 10) -> List[str]:
        """Generate sample text prompts"""
        base_prompts = [
            "A cat playing in a garden",
            "A dog running on the beach",
            "A bird flying in the sky",
            "A car driving down a road",
            "A person walking in the rain",
            "A sunset over the mountains",
            "A city skyline at night",
            "A forest with tall trees",
            "A river flowing through rocks",
            "A butterfly on a flower"
        ]
        
        # Add variations
        variations = [
            "beautiful", "stunning", "amazing", "incredible", "breathtaking",
            "cinematic", "artistic", "realistic", "detailed", "high quality"
        ]
        
        prompts = []
        for i in range(count):
            base = base_prompts[i % len(base_prompts)]
            if random.random() > 0.5:
                variation = random.choice(variations)
                base = f"{variation} {base}"
            prompts.append(base)
        
        return prompts
    
    @staticmethod
    def generate_generation_params(model_type: str = "t2v", **overrides) -> Dict[str, Any]:
        """Generate sample generation parameters"""
        base_params = {
            "num_frames": 16,
            "width": 512,
            "height": 512,
            "fps": 8.0,
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "seed": random.randint(0, 2**32 - 1)
        }
        
        if model_type.startswith("t2v"):
            base_params.update({
                "prompt": "A cat playing in a garden",
                "negative_prompt": "blurry, low quality"
            })
        elif model_type.startswith("i2v"):
            base_params.update({
                "image": "mock_image_tensor",
                "prompt": "A cat playing in a garden",
                "negative_prompt": "blurry, low quality",
                "image_guidance_scale": 1.5
            })
        elif model_type.startswith("ti2v"):
            base_params.update({
                "image": "mock_image_tensor",
                "prompt": "A cat playing in a garden",
                "negative_prompt": "blurry, low quality",
                "image_guidance_scale": 1.5,
                "text_guidance_scale": 7.5,
                "interpolation_strength": 1.0
            })
        
        # Apply overrides
        base_params.update(overrides)
        
        return base_params
    
    @staticmethod
    def create_mock_image_tensor(width: int = 512, height: int = 512, channels: int = 3):
        """Create a mock image tensor"""
        mock_tensor = Mock()
        mock_tensor.shape = (1, channels, height, width)
        mock_tensor.device = "cpu"
        mock_tensor.dtype = "float32"
        return mock_tensor
    
    @staticmethod
    def create_test_video_metadata(num_frames: int = 16, width: int = 512, height: int = 512) -> Dict[str, Any]:
        """Create test video metadata"""
        return {
            "frames": num_frames,
            "width": width,
            "height": height,
            "fps": 8.0,
            "duration_seconds": num_frames / 8.0,
            "format": "mp4",
            "codec": "h264",
            "bitrate_kbps": 2000,
            "file_size_mb": (num_frames * width * height * 3) / (1024 * 1024)  # Rough estimate
        }


class TempDirectoryManager:
    """Manage temporary directories for testing"""
    
    def __init__(self, prefix: str = "wan_test_"):
        self.prefix = prefix
        self.temp_dirs: List[Path] = []
    
    def create_temp_dir(self, suffix: str = "") -> Path:
        """Create a temporary directory"""
        temp_dir = Path(tempfile.mkdtemp(prefix=self.prefix, suffix=suffix))
        self.temp_dirs.append(temp_dir)
        return temp_dir
    
    def create_temp_file(self, content: str = "", suffix: str = ".txt", dir_path: Optional[Path] = None) -> Path:
        """Create a temporary file"""
        if dir_path is None:
            dir_path = self.create_temp_dir()
        
        temp_file = dir_path / f"temp_file_{int(time.time())}{suffix}"
        temp_file.write_text(content)
        return temp_file
    
    def cleanup(self):
        """Clean up all temporary directories"""
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        self.temp_dirs.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class MockProgressCallback:
    """Mock progress callback for testing"""
    
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []
        self.call_count = 0
    
    def __call__(self, step: int, total_steps: int, message: str = "", **kwargs):
        """Mock callback implementation"""
        self.call_count += 1
        call_info = {
            "step": step,
            "total_steps": total_steps,
            "message": message,
            "progress_percent": (step / total_steps) * 100 if total_steps > 0 else 0,
            "timestamp": time.time(),
            **kwargs
        }
        self.calls.append(call_info)
    
    def get_progress_history(self) -> List[Dict[str, Any]]:
        """Get progress call history"""
        return self.calls.copy()
    
    def reset(self):
        """Reset callback state"""
        self.calls.clear()
        self.call_count = 0


class ConfigurationHelper:
    """Helper for managing test configurations"""
    
    @staticmethod
    def create_model_config(model_type: str = "t2v-A14B", **overrides) -> Dict[str, Any]:
        """Create a model configuration"""
        base_config = {
            "model_name": model_type,
            "model_type": model_type,
            "device": "cpu",
            "precision": "fp32",
            "enable_optimization": False,
            "cache_dir": "/tmp/test_cache",
            "max_memory_gb": 8.0,
            "enable_cpu_offload": False,
            "enable_quantization": False,
            "batch_size": 1,
            "num_workers": 1
        }
        
        # Model-specific configurations
        if "5B" in model_type:
            base_config.update({
                "max_memory_gb": 6.0,
                "enable_optimization": True
            })
        elif "A14B" in model_type:
            base_config.update({
                "max_memory_gb": 12.0,
                "enable_cpu_offload": True
            })
        
        # Apply overrides
        base_config.update(overrides)
        
        return base_config
    
    @staticmethod
    def create_hardware_config(hardware_type: str = "rtx4080") -> Dict[str, Any]:
        """Create a hardware configuration"""
        if hardware_type.lower() == "rtx4080":
            return MockHardwareInfo.get_rtx4080_specs()
        elif hardware_type.lower() == "threadripper_pro":
            return MockHardwareInfo.get_threadripper_pro_specs()
        elif hardware_type.lower() == "combined":
            return MockHardwareInfo.get_combined_specs()
        else:
            return {
                "gpu_name": "Generic GPU",
                "gpu_memory_gb": 8.0,
                "cpu_name": "Generic CPU",
                "cpu_cores": 8,
                "memory_capacity_gb": 32
            }


class AssertionHelper:
    """Helper for common test assertions"""
    
    @staticmethod
    def assert_generation_result(result, expected_frames: int = 16, expected_resolution: Tuple[int, int] = (512, 512)):
        """Assert generation result properties"""
        assert result is not None
        assert result.success is True
        assert result.video_path is not None
        assert result.generation_time > 0
        assert result.metadata is not None
        
        if hasattr(result, 'frames'):
            assert len(result.frames) == expected_frames
        
        if "resolution" in result.metadata:
            expected_res_str = f"{expected_resolution[0]}x{expected_resolution[1]}"
            assert result.metadata["resolution"] == expected_res_str
    
    @staticmethod
    def assert_model_capabilities(capabilities, model_type: str):
        """Assert model capabilities"""
        assert capabilities is not None
        
        if "t2v" in model_type.lower():
            assert capabilities.supports_text_conditioning is True
            assert capabilities.supports_image_conditioning is False
        elif "i2v" in model_type.lower():
            assert capabilities.supports_text_conditioning is True
            assert capabilities.supports_image_conditioning is True
        elif "ti2v" in model_type.lower():
            assert capabilities.supports_text_conditioning is True
            assert capabilities.supports_image_conditioning is True
            assert capabilities.supports_dual_conditioning is True
        
        assert capabilities.max_frames > 0
        assert capabilities.max_resolution[0] > 0
        assert capabilities.max_resolution[1] > 0
        assert capabilities.parameter_count > 0
    
    @staticmethod
    def assert_benchmark_result(result, min_duration: float = 0.0, max_duration: float = 300.0):
        """Assert benchmark result properties"""
        assert result is not None
        assert result.duration_seconds >= min_duration
        assert result.duration_seconds <= max_duration
        assert result.memory_usage_mb >= 0
        assert result.cpu_usage_percent >= 0
        assert result.cpu_usage_percent <= 100
    
    @staticmethod
    def assert_hardware_optimization(config, hardware_type: str):
        """Assert hardware optimization configuration"""
        assert config is not None
        
        if hardware_type.lower() == "rtx4080":
            # RTX 4080 should not need CPU offload due to 16GB VRAM
            assert config.get("enable_cpu_offload", True) is False
            assert config.get("precision") in ["fp16", "bf16"]  # Should use mixed precision
            assert config.get("enable_xformers", False) is True  # Should use memory optimizations
        elif hardware_type.lower() == "threadripper_pro":
            # Threadripper PRO should leverage CPU capabilities
            assert config.get("num_workers", 1) > 4  # Should use many workers
            assert config.get("enable_cpu_offload", False) is True  # Should use CPU offload
            assert config.get("numa_optimization", False) is True  # Should optimize for NUMA


def create_test_environment() -> Dict[str, Any]:
    """Create a complete test environment"""
    return {
        "temp_manager": TempDirectoryManager(),
        "data_generator": TestDataGenerator(),
        "config_helper": ConfigurationHelper(),
        "assertion_helper": AssertionHelper(),
        "mock_hardware": MockHardwareInfo(),
        "progress_callback": MockProgressCallback()
    }