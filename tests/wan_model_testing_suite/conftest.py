"""
Pytest configuration and fixtures for WAN Model Testing Suite
"""

import os
import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import Mock, MagicMock

# Test configuration
TEST_CONFIG = {
    "enable_hardware_tests": os.getenv("WAN_TEST_HARDWARE", "false").lower() == "true",
    "enable_performance_tests": os.getenv("WAN_TEST_PERFORMANCE", "false").lower() == "true",
    "enable_integration_tests": os.getenv("WAN_TEST_INTEGRATION", "true").lower() == "true",
    "benchmark_iterations": int(os.getenv("WAN_BENCHMARK_ITERATIONS", "10")),
    "test_timeout": int(os.getenv("WAN_TEST_TIMEOUT", "300")),  # 5 minutes
}


@pytest.fixture(scope="session")
def test_config():
    """Test configuration fixture"""
    return TEST_CONFIG


@pytest.fixture(scope="session")
def temp_dir():
    """Temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_torch():
    """Mock torch for testing without PyTorch dependency"""
    mock_torch = Mock()
    
    # Mock tensor class
    class MockTensor:
        def __init__(self, shape=(1, 1, 1, 1), device="cpu"):
            self.shape = shape
            self.device = device
        
        def cuda(self):
            self.device = "cuda"
            return self
        
        def cpu(self):
            self.device = "cpu"
            return self
        
        def half(self):
            return self
        
        def size(self, dim=None):
            return self.shape[dim] if dim is not None else self.shape
        
        def view(self, *args):
            return MockTensor(args)
        
        def permute(self, *args):
            return self
        
        def contiguous(self):
            return self
        
        def mean(self, dim=None, keepdim=False):
            return self
        
        def __add__(self, other):
            return self
        
        def __mul__(self, other):
            return self
    
    mock_torch.Tensor = MockTensor
    mock_torch.randn = lambda *args, **kwargs: MockTensor()
    mock_torch.zeros = lambda *args, **kwargs: MockTensor()
    mock_torch.ones = lambda *args, **kwargs: MockTensor()
    mock_torch.cat = lambda tensors, dim=0: MockTensor()
    mock_torch.stack = lambda tensors, dim=0: MockTensor()
    
    return mock_torch


@pytest.fixture
def mock_model_config():
    """Mock model configuration"""
    return {
        "model_name": "test-model",
        "model_type": "t2v",
        "device": "cpu",
        "precision": "fp32",
        "enable_optimization": False,
        "cache_dir": "/tmp/test_cache",
        "max_memory_gb": 8.0,
    }


@pytest.fixture
def sample_generation_params():
    """Sample generation parameters for testing"""
    return {
        "prompt": "A cat playing in a garden",
        "negative_prompt": "blurry, low quality",
        "num_frames": 16,
        "width": 512,
        "height": 512,
        "fps": 8.0,
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        "seed": 42,
    }


@pytest.fixture
def mock_progress_callback():
    """Mock progress callback for testing"""
    callback = Mock()
    callback.return_value = None
    return callback


@pytest.fixture
def mock_hardware_info():
    """Mock hardware information"""
    return {
        "gpu_name": "NVIDIA GeForce RTX 4080",
        "gpu_memory_gb": 16.0,
        "cpu_name": "AMD Ryzen Threadripper PRO 5975WX",
        "cpu_cores": 32,
        "system_memory_gb": 128.0,
        "cuda_available": True,
        "cuda_version": "12.1",
        "torch_version": "2.1.0",
    }


@pytest.fixture
def sample_image_tensor(mock_torch):
    """Sample image tensor for testing"""
    return mock_torch.randn(1, 3, 512, 512)


@pytest.fixture
def sample_text_tokens(mock_torch):
    """Sample text tokens for testing"""
    return mock_torch.ones(1, 77, dtype=mock_torch.long)


@pytest.fixture
def mock_wan_model():
    """Mock WAN model for testing"""
    model = Mock()
    model.model_type = "t2v-A14B"
    model.is_loaded = True
    model.device = "cpu"
    model.precision = "fp32"
    model.capabilities = Mock()
    model.capabilities.supports_text_conditioning = True
    model.capabilities.supports_image_conditioning = False
    model.capabilities.max_frames = 16
    model.capabilities.max_resolution = (1280, 720)
    model.capabilities.estimated_vram_gb = 10.5
    model.capabilities.parameter_count = 14_000_000_000
    
    # Mock methods
    model.generate = Mock()
    model.validate_generation_params = Mock(return_value=(True, []))
    model.get_memory_usage = Mock(return_value={"allocated_gb": 8.5, "cached_gb": 1.2})
    model.optimize_for_hardware = Mock()
    model.enable_cpu_offload = Mock()
    model.enable_quantization = Mock()
    
    return model


@pytest.fixture
def benchmark_metrics():
    """Benchmark metrics structure"""
    return {
        "generation_time_seconds": 0.0,
        "memory_usage_gb": 0.0,
        "throughput_fps": 0.0,
        "quality_score": 0.0,
        "gpu_utilization_percent": 0.0,
        "cpu_utilization_percent": 0.0,
    }


# Pytest markers for test categorization
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance benchmark tests")
    config.addinivalue_line("markers", "hardware: Hardware-specific tests")
    config.addinivalue_line("markers", "slow: Slow-running tests")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")
    config.addinivalue_line("markers", "cpu: CPU-only tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on configuration"""
    # Skip hardware tests if not enabled
    if not TEST_CONFIG["enable_hardware_tests"]:
        skip_hardware = pytest.mark.skip(reason="Hardware tests disabled")
        for item in items:
            if "hardware" in item.keywords:
                item.add_marker(skip_hardware)
    
    # Skip performance tests if not enabled
    if not TEST_CONFIG["enable_performance_tests"]:
        skip_performance = pytest.mark.skip(reason="Performance tests disabled")
        for item in items:
            if "performance" in item.keywords:
                item.add_marker(skip_performance)
    
    # Skip integration tests if not enabled
    if not TEST_CONFIG["enable_integration_tests"]:
        skip_integration = pytest.mark.skip(reason="Integration tests disabled")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Setup test environment"""
    # Set test environment variables
    monkeypatch.setenv("WAN_TEST_MODE", "true")
    monkeypatch.setenv("WAN_LOG_LEVEL", "DEBUG")
    
    # Mock external dependencies if not available
    try:
        import torch
    except ImportError:
        # Mock torch module
        import sys
        sys.modules['torch'] = Mock()
        sys.modules['torch.nn'] = Mock()
        sys.modules['torch.nn.functional'] = Mock()
    
    try:
        import numpy
    except ImportError:
        # Mock numpy module
        import sys
        sys.modules['numpy'] = Mock()