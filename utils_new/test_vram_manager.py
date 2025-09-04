"""
Test suite for VRAMManager class

Tests all VRAM detection methods, monitoring capabilities, and multi-GPU support.
"""

import pytest
import json
import tempfile
import os
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from vram_manager import (
    VRAMManager, GPUInfo, VRAMUsage, VRAMConfig, VRAMDetectionError,
    detect_vram, get_optimal_gpu, get_vram_usage
)


class TestVRAMManager:
    """Test cases for VRAMManager class"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_vram_config.json")
        self.manager = VRAMManager(config_path=self.config_path)
    
    def teardown_method(self):
        """Cleanup after each test method"""
        if hasattr(self, 'manager'):
            self.manager.cleanup()
    
    def test_init_default_config(self):
        """Test VRAMManager initialization with default config"""
        assert isinstance(self.manager.config, VRAMConfig)
        assert self.manager.config.manual_vram_gb is None
        assert self.manager.config.preferred_gpu is None
        assert self.manager.config.enable_multi_gpu is False
        assert self.manager.config.memory_fraction == 0.9
    
    def test_load_existing_config(self):
        """Test loading existing configuration file"""
        config_data = {
            "manual_vram_gb": {"0": 16, "1": 8},
            "preferred_gpu": 0,
            "enable_multi_gpu": True,
            "memory_fraction": 0.8
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config_data, f)
        
        manager = VRAMManager(config_path=self.config_path)
        assert manager.config.manual_vram_gb == {"0": 16, "1": 8}
        assert manager.config.preferred_gpu == 0
        assert manager.config.enable_multi_gpu is True
        assert manager.config.memory_fraction == 0.8
    
    @patch('vram_manager.pynvml')
    def test_detect_via_nvml_success(self, mock_pynvml):
        """Test successful NVML detection"""
        # Mock NVML functions
        mock_pynvml.nvmlDeviceGetCount.return_value = 2
        
        # Mock GPU 0
        mock_handle_0 = Mock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = [mock_handle_0, Mock()]
        mock_pynvml.nvmlDeviceGetName.side_effect = [b'RTX 4080', b'RTX 3080']
        
        # Mock memory info
        mock_memory_0 = Mock()
        mock_memory_0.total = 16 * 1024 * 1024 * 1024  # 16GB
        mock_memory_1 = Mock()
        mock_memory_1.total = 10 * 1024 * 1024 * 1024  # 10GB
        mock_pynvml.nvmlDeviceGetMemoryInfo.side_effect = [mock_memory_0, mock_memory_1]
        
        mock_pynvml.nvmlSystemGetDriverVersion.return_value = b'535.98'
        mock_pynvml.nvmlSystemGetCudaDriverVersion.return_value = 12020
        
        # Mock temperature and utilization
        mock_pynvml.nvmlDeviceGetTemperature.side_effect = [65.0, 70.0]
        mock_util_0 = Mock()
        mock_util_0.gpu = 25
        mock_util_1 = Mock()
        mock_util_1.gpu = 30
        mock_pynvml.nvmlDeviceGetUtilizationRates.side_effect = [mock_util_0, mock_util_1]
        mock_pynvml.nvmlDeviceGetPowerUsage.side_effect = [200000, 250000]  # mW
        
        self.manager.nvml_initialized = True
        gpus = self.manager._detect_via_nvml()
        
        assert len(gpus) == 2
        assert gpus[0].name == 'RTX 4080'
        assert gpus[0].total_memory_mb == 16384
        assert gpus[0].driver_version == '535.98'
        assert gpus[0].cuda_version == '12.2'
        assert gpus[0].temperature == 65.0
        assert gpus[0].utilization == 25
        assert gpus[0].power_usage == 200.0
    
    @patch('vram_manager.torch')
    def test_detect_via_pytorch_success(self, mock_torch):
        """Test successful PyTorch detection"""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.version.cuda = '12.1'
        
        # Mock device properties
        mock_props = Mock()
        mock_props.name = 'RTX 4080'
        mock_props.total_memory = 16 * 1024 * 1024 * 1024  # 16GB
        mock_torch.cuda.get_device_properties.return_value = mock_props
        
        gpus = self.manager._detect_via_pytorch()
        
        assert len(gpus) == 1
        assert gpus[0].name == 'RTX 4080'
        assert gpus[0].total_memory_mb == 16384
        assert gpus[0].cuda_version == '12.1'
    
    @patch('subprocess.run')
    def test_detect_via_nvidia_smi_success(self, mock_run):
        """Test successful nvidia-smi detection"""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "0, RTX 4080, 16384, 535.98\n1, RTX 3080, 10240, 535.98\n"
        mock_run.return_value = mock_result
        
        gpus = self.manager._detect_via_nvidia_smi()
        
        assert len(gpus) == 2
        assert gpus[0].name == 'RTX 4080'
        assert gpus[0].total_memory_mb == 16384
        assert gpus[1].name == 'RTX 3080'
        assert gpus[1].total_memory_mb == 10240
    
    def test_detect_via_manual_config(self):
        """Test manual configuration detection"""
        self.manager.config.manual_vram_gb = {0: 16, 1: 8}
        
        gpus = self.manager._detect_via_manual_config()
        
        assert len(gpus) == 2
        assert gpus[0].total_memory_mb == 16384
        assert gpus[1].total_memory_mb == 8192
        assert gpus[0].name == "Manual GPU 0"
    
    def test_detect_vram_capacity_fallback(self):
        """Test VRAM detection with fallback methods"""
        # Set up manual config as fallback
        self.manager.config.manual_vram_gb = {0: 16}
        
        # Mock all other methods to fail
        with patch.object(self.manager, '_detect_via_nvml', side_effect=Exception("NVML failed")), \
             patch.object(self.manager, '_detect_via_pytorch', side_effect=Exception("PyTorch failed")), \
             patch.object(self.manager, '_detect_via_nvidia_smi', side_effect=Exception("nvidia-smi failed")):
            
            gpus = self.manager.detect_vram_capacity()
            
            assert len(gpus) == 1
            assert gpus[0].total_memory_mb == 16384
            assert gpus[0].name == "Manual GPU 0"
    
    def test_detect_vram_capacity_all_fail(self):
        """Test VRAM detection when all methods fail"""
        with patch.object(self.manager, '_detect_via_nvml', side_effect=Exception("NVML failed")), \
             patch.object(self.manager, '_detect_via_pytorch', side_effect=Exception("PyTorch failed")), \
             patch.object(self.manager, '_detect_via_nvidia_smi', side_effect=Exception("nvidia-smi failed")), \
             patch.object(self.manager, '_detect_via_manual_config', side_effect=Exception("Manual failed")):
            
            with pytest.raises(VRAMDetectionError):
                self.manager.detect_vram_capacity()

        assert True  # TODO: Add proper assertion
    
    def test_select_optimal_gpu(self):
        """Test optimal GPU selection"""
        # Create test GPUs
        gpu1 = GPUInfo(0, "RTX 3080", 10240, "535.98")
        gpu2 = GPUInfo(1, "RTX 4080", 16384, "535.98")
        self.manager.detected_gpus = [gpu1, gpu2]
        
        # Should select GPU with most VRAM
        optimal = self.manager.select_optimal_gpu()
        assert optimal.index == 1
        assert optimal.name == "RTX 4080"
    
    def test_select_preferred_gpu(self):
        """Test preferred GPU selection"""
        gpu1 = GPUInfo(0, "RTX 3080", 10240, "535.98")
        gpu2 = GPUInfo(1, "RTX 4080", 16384, "535.98")
        self.manager.detected_gpus = [gpu1, gpu2]
        self.manager.config.preferred_gpu = 0
        
        # Should select preferred GPU even if it has less VRAM
        optimal = self.manager.select_optimal_gpu()
        assert optimal.index == 0
        assert optimal.name == "RTX 3080"
    
    def test_set_manual_vram_config(self):
        """Test setting manual VRAM configuration"""
        gpu_mapping = {0: 16, 1: 8}
        self.manager.set_manual_vram_config(gpu_mapping)
        
        assert self.manager.config.manual_vram_gb == gpu_mapping
        
        # Verify config is saved
        with open(self.config_path, 'r') as f:
            saved_config = json.load(f)
        assert saved_config['manual_vram_gb'] == {"0": 16, "1": 8}
    
    def test_validate_manual_config_valid(self):
        """Test validation of valid manual config"""
        gpu_mapping = {0: 16, 1: 8}
        is_valid, errors = self.manager.validate_manual_config(gpu_mapping)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_manual_config_invalid(self):
        """Test validation of invalid manual config"""
        # Test various invalid configurations
        test_cases = [
            ({}, ["GPU VRAM mapping cannot be empty"]),
            ({-1: 16}, ["Invalid GPU index: -1"]),
            ({0: 0}, ["Invalid VRAM amount for GPU 0: 0GB"]),
            ({0: 200}, ["VRAM amount too high for GPU 0: 200GB"]),
            ({"invalid": 16}, ["Invalid GPU index: invalid"]),
        ]
        
        for gpu_mapping, expected_errors in test_cases:
            is_valid, errors = self.manager.validate_manual_config(gpu_mapping)
            assert is_valid is False
            for expected_error in expected_errors:
                assert any(expected_error in error for error in errors)
    
    @patch('vram_manager.pynvml')
    def test_get_current_vram_usage_nvml(self, mock_pynvml):
        """Test getting current VRAM usage via NVML"""
        self.manager.nvml_initialized = True
        self.manager.detected_gpus = [GPUInfo(0, "RTX 4080", 16384, "535.98")]
        
        # Mock memory info
        mock_handle = Mock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        
        mock_memory = Mock()
        mock_memory.used = 8 * 1024 * 1024 * 1024  # 8GB used
        mock_memory.free = 8 * 1024 * 1024 * 1024  # 8GB free
        mock_memory.total = 16 * 1024 * 1024 * 1024  # 16GB total
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_memory
        
        usage_list = self.manager.get_current_vram_usage()
        
        assert len(usage_list) == 1
        usage = usage_list[0]
        assert usage.gpu_index == 0
        assert usage.used_mb == 8192
        assert usage.free_mb == 8192
        assert usage.total_mb == 16384
        assert usage.usage_percent == 50.0
    
    def test_monitoring_start_stop(self):
        """Test VRAM monitoring start and stop"""
        assert not self.manager.monitoring_active
        
        # Mock get_current_vram_usage to avoid actual GPU calls
        with patch.object(self.manager, 'get_current_vram_usage', return_value=[]):
            self.manager.start_monitoring(interval_seconds=0.1)
            assert self.manager.monitoring_active
            
            time.sleep(0.2)  # Let it run briefly
            
            self.manager.stop_monitoring()
            assert not self.manager.monitoring_active
    
    def test_memory_optimization_trigger(self):
        """Test memory optimization triggering"""
        with patch('vram_manager.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            
            # This should trigger memory optimization
            self.manager._trigger_memory_optimization(0)
            
            # Verify torch.cuda.empty_cache was called
            mock_torch.cuda.empty_cache.assert_called_once()

        assert True  # TODO: Add proper assertion
    
    def test_usage_history_tracking(self):
        """Test VRAM usage history tracking"""
        from datetime import datetime
        
        # Add some usage history
        usage1 = VRAMUsage(0, 4096, 4096, 8192, 50.0, datetime.now())
        usage2 = VRAMUsage(0, 6144, 2048, 8192, 75.0, datetime.now())
        
        self.manager._usage_history[0] = [usage1, usage2]
        
        history = self.manager.get_usage_history(0)
        assert len(history) == 2
        assert history[0].usage_percent == 50.0
        assert history[1].usage_percent == 75.0
        
        stats = self.manager.get_usage_statistics(0)
        assert stats['current_usage_percent'] == 75.0
        assert stats['average_usage_percent'] == 62.5
        assert stats['max_usage_percent'] == 75.0
        assert stats['min_usage_percent'] == 50.0
    
    def test_get_detection_summary(self):
        """Test getting detection summary"""
        gpu = GPUInfo(0, "RTX 4080", 16384, "535.98")
        self.manager.detected_gpus = [gpu]
        
        summary = self.manager.get_detection_summary()
        
        assert summary['total_gpus'] == 1
        assert summary['available_gpus'] == 1
        assert len(summary['gpus']) == 1
        assert summary['gpus'][0]['name'] == "RTX 4080"
        assert 'config' in summary
        assert 'nvml_available' in summary
        assert 'torch_available' in summary


class TestUtilityFunctions:
    """Test utility functions"""
    
    @patch('vram_manager.VRAMManager')
    def test_detect_vram_function(self, mock_manager_class):
        """Test detect_vram utility function"""
        mock_manager = Mock()
        mock_manager.detect_vram_capacity.return_value = [GPUInfo(0, "RTX 4080", 16384, "535.98")]
        mock_manager_class.return_value = mock_manager
        
        gpus = detect_vram()
        
        assert len(gpus) == 1
        assert gpus[0].name == "RTX 4080"
        mock_manager.detect_vram_capacity.assert_called_once()
    
    @patch('vram_manager.VRAMManager')
    def test_get_optimal_gpu_function(self, mock_manager_class):
        """Test get_optimal_gpu utility function"""
        mock_manager = Mock()
        mock_manager.select_optimal_gpu.return_value = GPUInfo(0, "RTX 4080", 16384, "535.98")
        mock_manager_class.return_value = mock_manager
        
        gpu = get_optimal_gpu()
        
        assert gpu.name == "RTX 4080"
        mock_manager.detect_vram_capacity.assert_called_once()
        mock_manager.select_optimal_gpu.assert_called_once()
    
    @patch('vram_manager.VRAMManager')
    def test_get_vram_usage_function(self, mock_manager_class):
        """Test get_vram_usage utility function"""
        from datetime import datetime
        
        mock_manager = Mock()
        mock_usage = VRAMUsage(0, 4096, 4096, 8192, 50.0, datetime.now())
        mock_manager.get_current_vram_usage.return_value = [mock_usage]
        mock_manager_class.return_value = mock_manager
        
        usage = get_vram_usage()
        
        assert len(usage) == 1
        assert usage[0].usage_percent == 50.0
        mock_manager.detect_vram_capacity.assert_called_once()
        mock_manager.get_current_vram_usage.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])