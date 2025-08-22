#!/usr/bin/env python3
"""
Unit tests for HardwareOptimizer component
Tests hardware detection and optimization for RTX 4080 and Threadripper PRO
"""

import unittest
import tempfile
import json
import os
from unittest.mock import patch, MagicMock, PropertyMock
from pathlib import Path

from hardware_optimizer import (
    HardwareOptimizer, HardwareProfile, OptimalSettings, OptimizationResult
)


class TestHardwareOptimizer(unittest.TestCase):
    """Test cases for HardwareOptimizer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "config.json"
        
        # Create minimal config file
        with open(self.config_path, 'w') as f:
            json.dump({}, f)
        
        self.optimizer = HardwareOptimizer(config_path=str(self.config_path))
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test HardwareOptimizer initialization"""
        self.assertIsInstance(self.optimizer, HardwareOptimizer)
        self.assertEqual(str(self.optimizer.config_path), str(self.config_path))
        self.assertIsNone(self.optimizer.hardware_profile)
        self.assertIsNone(self.optimizer.optimal_settings)
    
    @patch('hardware_optimizer.torch')
    @patch('hardware_optimizer.psutil')
    @patch('hardware_optimizer.platform')
    def test_detect_hardware_profile_rtx_4080(self, mock_platform, mock_psutil, mock_torch):
        """Test hardware profile detection for RTX 4080 system"""
        # Mock system info
        mock_platform.processor.return_value = "AMD Ryzen 9 7950X"
        mock_psutil.cpu_count.return_value = 16
        
        mock_memory = MagicMock()
        mock_memory.total = 64 * 1024**3  # 64GB
        mock_psutil.virtual_memory.return_value = mock_memory
        
        # Mock GPU info
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 4080"
        
        mock_props = MagicMock()
        mock_props.total_memory = 16 * 1024**3  # 16GB
        mock_torch.cuda.get_device_properties.return_value = mock_props
        mock_torch.version.cuda = "12.1"
        
        profile = self.optimizer.detect_hardware_profile()
        
        self.assertIsInstance(profile, HardwareProfile)
        self.assertIn("Ryzen", profile.cpu_model)
        self.assertEqual(profile.cpu_cores, 16)
        self.assertEqual(profile.total_memory_gb, 64)
        self.assertEqual(profile.gpu_model, "NVIDIA GeForce RTX 4080")
        self.assertEqual(profile.vram_gb, 16)
        self.assertEqual(profile.cuda_version, "12.1")
        self.assertTrue(profile.is_rtx_4080)
        self.assertFalse(profile.is_threadripper_pro)
    
    @patch('hardware_optimizer.torch')
    @patch('hardware_optimizer.psutil')
    @patch('hardware_optimizer.platform')
    def test_detect_hardware_profile_threadripper_pro(self, mock_platform, mock_psutil, mock_torch):
        """Test hardware profile detection for Threadripper PRO system"""
        # Mock system info
        mock_platform.processor.return_value = "AMD Ryzen Threadripper PRO 5995WX"
        mock_psutil.cpu_count.return_value = 64
        
        mock_memory = MagicMock()
        mock_memory.total = 128 * 1024**3  # 128GB
        mock_psutil.virtual_memory.return_value = mock_memory
        
        # Mock GPU info
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 4080"
        
        mock_props = MagicMock()
        mock_props.total_memory = 16 * 1024**3  # 16GB
        mock_torch.cuda.get_device_properties.return_value = mock_props
        mock_torch.version.cuda = "12.1"
        
        profile = self.optimizer.detect_hardware_profile()
        
        self.assertIsInstance(profile, HardwareProfile)
        self.assertIn("Threadripper PRO", profile.cpu_model)
        self.assertEqual(profile.cpu_cores, 64)
        self.assertEqual(profile.total_memory_gb, 128)
        self.assertTrue(profile.is_rtx_4080)
        self.assertTrue(profile.is_threadripper_pro)
    
    def test_generate_rtx_4080_settings(self):
        """Test optimal settings generation for RTX 4080"""
        profile = HardwareProfile(
            cpu_model="AMD Ryzen 9 7950X",
            cpu_cores=16,
            total_memory_gb=64,
            gpu_model="NVIDIA GeForce RTX 4080",
            vram_gb=16,
            cuda_version="12.1",
            driver_version="531.79",
            is_rtx_4080=True,
            is_threadripper_pro=False
        )
        
        settings = self.optimizer.generate_rtx_4080_settings(profile)
        
        self.assertIsInstance(settings, OptimalSettings)
        self.assertEqual(settings.tile_size, (512, 512))
        self.assertEqual(settings.vae_tile_size, (256, 256))  # As specified in requirements
        self.assertEqual(settings.batch_size, 2)  # For 16GB VRAM
        self.assertTrue(settings.enable_cpu_offload)
        self.assertTrue(settings.text_encoder_offload)
        self.assertTrue(settings.vae_offload)
        self.assertTrue(settings.enable_tensor_cores)
        self.assertTrue(settings.use_fp16)
        self.assertTrue(settings.use_bf16)
        self.assertEqual(settings.memory_fraction, 0.9)
        self.assertTrue(settings.gradient_checkpointing)
        self.assertTrue(settings.enable_xformers)
        self.assertEqual(settings.num_threads, min(16, 8))  # Optimal for most workloads
    
    def test_generate_threadripper_pro_settings(self):
        """Test optimal settings generation for Threadripper PRO"""
        profile = HardwareProfile(
            cpu_model="AMD Ryzen Threadripper PRO 5995WX",
            cpu_cores=64,
            total_memory_gb=128,
            gpu_model="NVIDIA GeForce RTX 4080",
            vram_gb=16,
            cuda_version="12.1",
            driver_version="531.79",
            is_rtx_4080=True,
            is_threadripper_pro=True
        )
        
        with patch.object(self.optimizer, '_detect_numa_nodes', return_value=[0, 1]):
            with patch.object(self.optimizer, '_generate_cpu_affinity', return_value=list(range(32))):
                settings = self.optimizer.generate_threadripper_pro_settings(profile)
        
        self.assertIsInstance(settings, OptimalSettings)
        self.assertEqual(settings.tile_size, (512, 512))
        self.assertEqual(settings.vae_tile_size, (384, 384))  # Larger for powerful CPU
        self.assertEqual(settings.batch_size, 4)  # Higher batch size with CPU support
        self.assertTrue(settings.enable_cpu_offload)
        self.assertFalse(settings.text_encoder_offload)  # Keep on GPU with powerful CPU
        self.assertFalse(settings.vae_offload)  # Keep on GPU with powerful CPU
        self.assertEqual(settings.memory_fraction, 0.95)  # Higher with CPU support
        self.assertFalse(settings.gradient_checkpointing)  # Disable with abundant CPU resources
        self.assertEqual(settings.num_threads, min(64, 32))  # Optimal threading
        self.assertEqual(settings.numa_nodes, [0, 1])
        self.assertEqual(settings.parallel_workers, min(8, 64 // 8))
        self.assertTrue(settings.enable_numa_optimization)
        self.assertEqual(settings.preprocessing_threads, min(16, 64 // 4))
        self.assertEqual(settings.io_threads, min(4, 64 // 16))
    
    @patch('subprocess.run')
    def test_detect_numa_nodes_linux(self, mock_run):
        """Test NUMA node detection on Linux"""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "available: 2 nodes (0-1)\nnode 0 cpus: 0 1 2 3\nnode 1 cpus: 4 5 6 7"
        mock_run.return_value = mock_result
        
        with patch('os.path.exists', return_value=True):
            numa_nodes = self.optimizer._detect_numa_nodes()
        
        self.assertEqual(numa_nodes, [0, 1])
    
    @patch('subprocess.run')
    @patch('hardware_optimizer.platform.system')
    def test_detect_numa_nodes_windows(self, mock_system, mock_run):
        """Test NUMA node detection on Windows"""
        mock_system.return_value = 'Windows'
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NumberOfProcessors\n64"
        mock_run.return_value = mock_result
        
        numa_nodes = self.optimizer._detect_numa_nodes()
        
        self.assertEqual(numa_nodes, [0, 1])  # Default assumption for Threadripper PRO
    
    def test_detect_numa_nodes_fallback(self):
        """Test NUMA node detection fallback"""
        with patch('subprocess.run', side_effect=Exception("Command failed")):
            numa_nodes = self.optimizer._detect_numa_nodes()
        
        self.assertEqual(numa_nodes, [0, 1])  # Default assumption
    
    def test_generate_cpu_affinity_large_system(self):
        """Test CPU affinity generation for large system"""
        affinity = self.optimizer._generate_cpu_affinity(64)
        
        self.assertIsInstance(affinity, list)
        self.assertEqual(len(affinity), 32)  # 16 from each NUMA node
        self.assertIn(0, affinity)  # Should include first cores
        self.assertIn(32, affinity)  # Should include cores from second NUMA node
    
    def test_generate_cpu_affinity_small_system(self):
        """Test CPU affinity generation for small system"""
        affinity = self.optimizer._generate_cpu_affinity(16)
        
        self.assertIsInstance(affinity, list)
        self.assertEqual(len(affinity), 16)  # All cores
        self.assertEqual(affinity, list(range(16)))
    
    @patch('hardware_optimizer.torch')
    @patch('hardware_optimizer.psutil')
    def test_apply_rtx_4080_optimizations(self, mock_psutil, mock_torch):
        """Test applying RTX 4080 optimizations"""
        profile = HardwareProfile(
            cpu_model="AMD Ryzen 9 7950X",
            cpu_cores=16,
            total_memory_gb=64,
            gpu_model="NVIDIA GeForce RTX 4080",
            vram_gb=16,
            cuda_version="12.1",
            driver_version="531.79",
            is_rtx_4080=True,
            is_threadripper_pro=False
        )
        
        # Mock torch CUDA availability
        mock_torch.cuda.is_available.return_value = True
        mock_torch.backends.cudnn = MagicMock()
        mock_torch.backends.cuda.matmul = MagicMock()
        
        result = self.optimizer.apply_rtx_4080_optimizations(profile)
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertTrue(result.success)
        self.assertGreater(len(result.optimizations_applied), 0)
        self.assertIsInstance(result.settings, OptimalSettings)
        
        # Verify tensor cores were enabled
        self.assertTrue(any("Tensor Cores" in opt for opt in result.optimizations_applied))
        
        # Verify CUDA memory fraction was set
        mock_torch.cuda.set_per_process_memory_fraction.assert_called()
        
        # Verify threading was configured
        mock_torch.set_num_threads.assert_called()
    
    @patch('hardware_optimizer.torch')
    @patch('hardware_optimizer.psutil')
    @patch('hardware_optimizer.os.environ', {})
    def test_apply_threadripper_pro_optimizations(self, mock_psutil, mock_torch):
        """Test applying Threadripper PRO optimizations"""
        profile = HardwareProfile(
            cpu_model="AMD Ryzen Threadripper PRO 5995WX",
            cpu_cores=64,
            total_memory_gb=128,
            gpu_model="NVIDIA GeForce RTX 4080",
            vram_gb=16,
            cuda_version="12.1",
            driver_version="531.79",
            is_rtx_4080=True,
            is_threadripper_pro=True
        )
        
        # Mock torch CUDA availability
        mock_torch.cuda.is_available.return_value = True
        mock_torch.backends.cudnn = MagicMock()
        mock_torch.backends.cuda.matmul = MagicMock()
        
        # Mock psutil Process for CPU affinity
        mock_process = MagicMock()
        mock_psutil.Process.return_value = mock_process
        
        with patch.object(self.optimizer, '_detect_numa_nodes', return_value=[0, 1]):
            with patch.object(self.optimizer, '_apply_numa_optimizations'):
                result = self.optimizer.apply_threadripper_pro_optimizations(profile)
        
        self.assertIsInstance(result, OptimizationResult)
        self.assertTrue(result.success)
        self.assertGreater(len(result.optimizations_applied), 0)
        self.assertIsInstance(result.settings, OptimalSettings)
        
        # Verify multi-threading environment variables were set
        self.assertTrue(any("multi-threading" in opt for opt in result.optimizations_applied))
        
        # Verify CPU affinity was attempted
        mock_process.cpu_affinity.assert_called()
        
        # Verify PyTorch optimizations
        mock_torch.set_num_threads.assert_called()
        mock_torch.set_num_interop_threads.assert_called()
    
    @patch('hardware_optimizer.numa', create=True)
    @patch('hardware_optimizer.NUMA_AVAILABLE', True)
    def test_apply_numa_optimizations(self, mock_numa):
        """Test NUMA optimizations application"""
        settings = OptimalSettings(
            tile_size=(512, 512),
            batch_size=4,
            enable_cpu_offload=True,
            enable_tensor_cores=True,
            memory_fraction=0.95,
            num_threads=32,
            enable_xformers=True,
            vae_tile_size=(384, 384),
            text_encoder_offload=False,
            vae_offload=False,
            use_fp16=True,
            use_bf16=True,
            gradient_checkpointing=False,
            numa_nodes=[0, 1],
            enable_numa_optimization=True
        )
        
        with patch.dict('os.environ', {}, clear=True):
            self.optimizer._apply_numa_optimizations(settings)
        
        # Verify NUMA library calls
        mock_numa.set_preferred_node.assert_called_with(0)
        mock_numa.set_interleave_mask.assert_called_with([0, 1])
        
        # Verify environment variables were set
        self.assertEqual(os.environ.get('NUMA_PREFERRED_NODE'), '0')
        self.assertEqual(os.environ.get('NUMA_INTERLEAVE_NODES'), '0,1')
    
    def test_configure_parallel_preprocessing(self):
        """Test parallel preprocessing configuration"""
        with patch('hardware_optimizer.psutil.cpu_count', return_value=64):
            with patch('hardware_optimizer.multiprocessing.set_start_method'):
                config = self.optimizer.configure_parallel_preprocessing()
        
        self.assertIsInstance(config, dict)
        self.assertIn('preprocessing_workers', config)
        self.assertIn('io_workers', config)
        self.assertIn('batch_processing_workers', config)
        
        # For 64 cores, should use min(8, max(2, 64//8)) = 8 workers
        self.assertEqual(config['preprocessing_workers'], 8)
        self.assertEqual(config['io_workers'], 4)  # min(4, max(1, 8//2))
        self.assertEqual(config['batch_processing_workers'], 2)  # min(2, max(1, 8//4))
    
    def test_configure_parallel_preprocessing_custom_workers(self):
        """Test parallel preprocessing configuration with custom worker count"""
        config = self.optimizer.configure_parallel_preprocessing(num_workers=4)
        
        self.assertEqual(config['preprocessing_workers'], 4)
        self.assertEqual(config['io_workers'], 2)
        self.assertEqual(config['batch_processing_workers'], 1)
    
    def test_configure_vae_tiling(self):
        """Test VAE tiling configuration"""
        result = self.optimizer.configure_vae_tiling((256, 256))
        
        self.assertTrue(result)
        self.assertEqual(self.optimizer.vae_tile_size, (256, 256))
    
    def test_configure_cpu_offloading(self):
        """Test CPU offloading configuration"""
        config = self.optimizer.configure_cpu_offloading(text_encoder=True, vae=False)
        
        self.assertEqual(config['text_encoder_offload'], True)
        self.assertEqual(config['vae_offload'], False)
        self.assertEqual(self.optimizer.cpu_offload_config, config)
    
    def test_get_memory_optimization_settings_rtx_4080(self):
        """Test memory optimization settings for RTX 4080 (16GB)"""
        settings = self.optimizer.get_memory_optimization_settings(16)
        
        self.assertFalse(settings['enable_attention_slicing'])
        self.assertFalse(settings['enable_vae_slicing'])
        self.assertTrue(settings['enable_cpu_offload'])
        self.assertEqual(settings['batch_size'], 2)
        self.assertEqual(settings['tile_size'], (512, 512))
        self.assertEqual(settings['vae_tile_size'], (256, 256))
    
    def test_get_memory_optimization_settings_mid_range(self):
        """Test memory optimization settings for mid-range GPU (12GB)"""
        settings = self.optimizer.get_memory_optimization_settings(12)
        
        self.assertTrue(settings['enable_attention_slicing'])
        self.assertTrue(settings['enable_vae_slicing'])
        self.assertTrue(settings['enable_cpu_offload'])
        self.assertEqual(settings['batch_size'], 1)
        self.assertEqual(settings['tile_size'], (384, 384))
        self.assertEqual(settings['vae_tile_size'], (192, 192))
    
    def test_get_memory_optimization_settings_low_vram(self):
        """Test memory optimization settings for low VRAM GPU (8GB)"""
        settings = self.optimizer.get_memory_optimization_settings(8)
        
        self.assertTrue(settings['enable_attention_slicing'])
        self.assertTrue(settings['enable_vae_slicing'])
        self.assertTrue(settings['enable_cpu_offload'])
        self.assertEqual(settings['batch_size'], 1)
        self.assertEqual(settings['tile_size'], (256, 256))
        self.assertEqual(settings['vae_tile_size'], (128, 128))
    
    def test_save_optimization_profile(self):
        """Test saving optimization profile"""
        profile = HardwareProfile(
            cpu_model="Test CPU",
            cpu_cores=8,
            total_memory_gb=32,
            gpu_model="Test GPU",
            vram_gb=8,
            cuda_version="12.1",
            driver_version="531.79"
        )
        
        settings = OptimalSettings(
            tile_size=(512, 512),
            batch_size=2,
            enable_cpu_offload=True,
            enable_tensor_cores=True,
            memory_fraction=0.9,
            num_threads=8,
            enable_xformers=True,
            vae_tile_size=(256, 256),
            text_encoder_offload=True,
            vae_offload=True,
            use_fp16=True,
            use_bf16=True,
            gradient_checkpointing=True
        )
        
        self.optimizer.hardware_profile = profile
        self.optimizer.optimal_settings = settings
        
        profile_path = Path(self.temp_dir) / "profile.json"
        result = self.optimizer.save_optimization_profile(str(profile_path))
        
        self.assertTrue(result)
        self.assertTrue(profile_path.exists())
        
        # Verify saved content
        with open(profile_path, 'r') as f:
            saved_data = json.load(f)
        
        self.assertIn('hardware_profile', saved_data)
        self.assertIn('optimal_settings', saved_data)
        self.assertIn('timestamp', saved_data)
        self.assertEqual(saved_data['hardware_profile']['cpu_model'], "Test CPU")
    
    def test_load_optimization_profile(self):
        """Test loading optimization profile"""
        profile_data = {
            'hardware_profile': {
                'cpu_model': 'Test CPU',
                'cpu_cores': 8,
                'total_memory_gb': 32,
                'gpu_model': 'Test GPU',
                'vram_gb': 8,
                'cuda_version': '12.1',
                'driver_version': '531.79',
                'is_rtx_4080': False,
                'is_threadripper_pro': False
            },
            'optimal_settings': {
                'tile_size': [512, 512],  # List format from JSON
                'batch_size': 2,
                'enable_cpu_offload': True,
                'enable_tensor_cores': True,
                'memory_fraction': 0.9,
                'num_threads': 8,
                'enable_xformers': True,
                'vae_tile_size': [256, 256],  # List format from JSON
                'text_encoder_offload': True,
                'vae_offload': True,
                'use_fp16': True,
                'use_bf16': True,
                'gradient_checkpointing': True,
                'numa_nodes': None,
                'cpu_affinity': None,
                'parallel_workers': 1,
                'enable_numa_optimization': False,
                'preprocessing_threads': 1,
                'io_threads': 1
            },
            'timestamp': '2024-01-01T12:00:00'
        }
        
        profile_path = Path(self.temp_dir) / "profile.json"
        with open(profile_path, 'w') as f:
            json.dump(profile_data, f)
        
        result = self.optimizer.load_optimization_profile(str(profile_path))
        
        self.assertTrue(result)
        self.assertIsNotNone(self.optimizer.hardware_profile)
        self.assertIsNotNone(self.optimizer.optimal_settings)
        
        # Verify loaded data
        self.assertEqual(self.optimizer.hardware_profile.cpu_model, "Test CPU")
        self.assertEqual(self.optimizer.optimal_settings.tile_size, (512, 512))  # Converted to tuple
        self.assertEqual(self.optimizer.optimal_settings.vae_tile_size, (256, 256))  # Converted to tuple
    
    def test_generate_optimal_settings_default(self):
        """Test optimal settings generation for unrecognized hardware"""
        profile = HardwareProfile(
            cpu_model="Generic CPU",
            cpu_cores=8,
            total_memory_gb=16,
            gpu_model="Generic GPU",
            vram_gb=8,
            cuda_version="12.1",
            driver_version="531.79",
            is_rtx_4080=False,
            is_threadripper_pro=False
        )
        
        settings = self.optimizer.generate_optimal_settings(profile)
        
        self.assertIsInstance(settings, OptimalSettings)
        self.assertEqual(settings.tile_size, (384, 384))  # Default settings
        self.assertEqual(settings.batch_size, 1)
        self.assertTrue(settings.enable_cpu_offload)
        self.assertFalse(settings.enable_tensor_cores)  # Conservative default
        self.assertEqual(settings.memory_fraction, 0.8)
        self.assertEqual(settings.num_threads, min(8, 4))  # Conservative threading
        self.assertTrue(settings.enable_xformers)
        self.assertEqual(settings.vae_tile_size, (192, 192))
        self.assertTrue(settings.text_encoder_offload)
        self.assertTrue(settings.vae_offload)
        self.assertTrue(settings.use_fp16)
        self.assertFalse(settings.use_bf16)  # Conservative default
        self.assertTrue(settings.gradient_checkpointing)
    
    def test_apply_hardware_optimizations_auto_detect(self):
        """Test applying hardware optimizations with auto-detection"""
        with patch.object(self.optimizer, 'detect_hardware_profile') as mock_detect:
            with patch.object(self.optimizer, 'apply_rtx_4080_optimizations') as mock_rtx:
                mock_profile = HardwareProfile(
                    cpu_model="Test CPU",
                    cpu_cores=16,
                    total_memory_gb=64,
                    gpu_model="NVIDIA GeForce RTX 4080",
                    vram_gb=16,
                    cuda_version="12.1",
                    driver_version="531.79",
                    is_rtx_4080=True,
                    is_threadripper_pro=False
                )
                mock_detect.return_value = mock_profile
                mock_rtx.return_value = OptimizationResult(True, [], 0.0, 0, [], [])
                
                result = self.optimizer.apply_hardware_optimizations()
                
                mock_detect.assert_called_once()
                mock_rtx.assert_called_once_with(mock_profile)
                self.assertTrue(result.success)


class TestHardwareProfile(unittest.TestCase):
    """Test cases for HardwareProfile dataclass"""
    
    def test_hardware_profile_creation(self):
        """Test HardwareProfile creation"""
        profile = HardwareProfile(
            cpu_model="AMD Ryzen Threadripper PRO 5995WX",
            cpu_cores=64,
            total_memory_gb=128,
            gpu_model="NVIDIA GeForce RTX 4080",
            vram_gb=16,
            cuda_version="12.1",
            driver_version="531.79",
            is_rtx_4080=True,
            is_threadripper_pro=True
        )
        
        self.assertEqual(profile.cpu_model, "AMD Ryzen Threadripper PRO 5995WX")
        self.assertEqual(profile.cpu_cores, 64)
        self.assertEqual(profile.total_memory_gb, 128)
        self.assertEqual(profile.gpu_model, "NVIDIA GeForce RTX 4080")
        self.assertEqual(profile.vram_gb, 16)
        self.assertEqual(profile.cuda_version, "12.1")
        self.assertEqual(profile.driver_version, "531.79")
        self.assertTrue(profile.is_rtx_4080)
        self.assertTrue(profile.is_threadripper_pro)


class TestOptimalSettings(unittest.TestCase):
    """Test cases for OptimalSettings dataclass"""
    
    def test_optimal_settings_creation(self):
        """Test OptimalSettings creation with all parameters"""
        settings = OptimalSettings(
            tile_size=(512, 512),
            batch_size=2,
            enable_cpu_offload=True,
            enable_tensor_cores=True,
            memory_fraction=0.9,
            num_threads=16,
            enable_xformers=True,
            vae_tile_size=(256, 256),
            text_encoder_offload=True,
            vae_offload=True,
            use_fp16=True,
            use_bf16=True,
            gradient_checkpointing=True,
            numa_nodes=[0, 1],
            cpu_affinity=list(range(32)),
            parallel_workers=8,
            enable_numa_optimization=True,
            preprocessing_threads=16,
            io_threads=4
        )
        
        self.assertEqual(settings.tile_size, (512, 512))
        self.assertEqual(settings.batch_size, 2)
        self.assertTrue(settings.enable_cpu_offload)
        self.assertTrue(settings.enable_tensor_cores)
        self.assertEqual(settings.memory_fraction, 0.9)
        self.assertEqual(settings.num_threads, 16)
        self.assertTrue(settings.enable_xformers)
        self.assertEqual(settings.vae_tile_size, (256, 256))
        self.assertTrue(settings.text_encoder_offload)
        self.assertTrue(settings.vae_offload)
        self.assertTrue(settings.use_fp16)
        self.assertTrue(settings.use_bf16)
        self.assertTrue(settings.gradient_checkpointing)
        self.assertEqual(settings.numa_nodes, [0, 1])
        self.assertEqual(settings.cpu_affinity, list(range(32)))
        self.assertEqual(settings.parallel_workers, 8)
        self.assertTrue(settings.enable_numa_optimization)
        self.assertEqual(settings.preprocessing_threads, 16)
        self.assertEqual(settings.io_threads, 4)


class TestOptimizationResult(unittest.TestCase):
    """Test cases for OptimizationResult dataclass"""
    
    def test_optimization_result_success(self):
        """Test OptimizationResult for successful optimization"""
        settings = OptimalSettings(
            tile_size=(512, 512),
            batch_size=2,
            enable_cpu_offload=True,
            enable_tensor_cores=True,
            memory_fraction=0.9,
            num_threads=16,
            enable_xformers=True,
            vae_tile_size=(256, 256),
            text_encoder_offload=True,
            vae_offload=True,
            use_fp16=True,
            use_bf16=True,
            gradient_checkpointing=True
        )
        
        result = OptimizationResult(
            success=True,
            optimizations_applied=["Enabled Tensor Cores", "Set CUDA memory fraction"],
            performance_improvement=25.5,
            memory_savings=2048,
            warnings=["Minor compatibility issue"],
            errors=[],
            settings=settings
        )
        
        self.assertTrue(result.success)
        self.assertEqual(len(result.optimizations_applied), 2)
        self.assertEqual(result.performance_improvement, 25.5)
        self.assertEqual(result.memory_savings, 2048)
        self.assertEqual(len(result.warnings), 1)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(result.settings, settings)
    
    def test_optimization_result_failure(self):
        """Test OptimizationResult for failed optimization"""
        result = OptimizationResult(
            success=False,
            optimizations_applied=["Attempted tensor core setup"],
            performance_improvement=0.0,
            memory_savings=0,
            warnings=[],
            errors=["CUDA not available", "Hardware incompatibility"]
        )
        
        self.assertFalse(result.success)
        self.assertEqual(len(result.optimizations_applied), 1)
        self.assertEqual(result.performance_improvement, 0.0)
        self.assertEqual(result.memory_savings, 0)
        self.assertEqual(len(result.warnings), 0)
        self.assertEqual(len(result.errors), 2)
        self.assertIsNone(result.settings)


if __name__ == '__main__':
    unittest.main()