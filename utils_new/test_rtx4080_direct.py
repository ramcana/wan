#!/usr/bin/env python3
"""
Direct test of RTX 4080 optimization functionality
"""

import sys
import os
import logging
from unittest.mock import Mock, patch

# Set up logging
logging.basicConfig(level=logging.INFO)

# Execute the hardware_optimizer.py file directly
exec(open('hardware_optimizer.py').read())

def test_rtx_4080_functionality():
    """Test RTX 4080 specific functionality"""
    print("Testing RTX 4080 Hardware Optimizer functionality...")
    
    # Create test profile
    rtx_4080_profile = HardwareProfile(
        cpu_model="AMD Ryzen Threadripper PRO 5995WX",
        cpu_cores=64,
        total_memory_gb=128,
        gpu_model="NVIDIA GeForce RTX 4080",
        vram_gb=16,
        cuda_version="12.1",
        driver_version="537.13",
        is_rtx_4080=True,
        is_threadripper_pro=True
    )
    
    # Create optimizer
    optimizer = HardwareOptimizer()
    
    # Test RTX 4080 settings generation
    print("1. Testing RTX 4080 settings generation...")
    settings = optimizer.generate_rtx_4080_settings(rtx_4080_profile)
    
    # Verify RTX 4080 specific settings
    assert settings.vae_tile_size == (256, 256), f"Expected VAE tile size (256, 256), got {settings.vae_tile_size}"
    assert settings.tile_size == (512, 512), f"Expected tile size (512, 512), got {settings.tile_size}"
    assert settings.enable_tensor_cores == True, "Tensor cores should be enabled"
    assert settings.text_encoder_offload == True, "Text encoder offload should be enabled"
    assert settings.vae_offload == True, "VAE offload should be enabled"
    assert settings.enable_cpu_offload == True, "CPU offload should be enabled"
    assert settings.use_bf16 == True, "BF16 should be enabled for RTX 4080"
    assert settings.batch_size == 2, f"Expected batch size 2 for 16GB VRAM, got {settings.batch_size}"
    assert settings.memory_fraction == 0.9, f"Expected memory fraction 0.9, got {settings.memory_fraction}"
    
    print("✓ RTX 4080 settings generation passed")
    
    # Test VAE tiling configuration
    print("2. Testing VAE tiling configuration...")
    result = optimizer.configure_vae_tiling((256, 256))
    assert result == True, "VAE tiling configuration should succeed"
    assert optimizer.vae_tile_size == (256, 256), "VAE tile size should be stored correctly"
    
    print("✓ VAE tiling configuration passed")
    
    # Test CPU offloading configuration
    print("3. Testing CPU offloading configuration...")
    config = optimizer.configure_cpu_offloading(text_encoder=True, vae=True)
    assert config['text_encoder_offload'] == True, "Text encoder offload should be True"
    assert config['vae_offload'] == True, "VAE offload should be True"
    assert optimizer.cpu_offload_config == config, "CPU offload config should be stored"
    
    print("✓ CPU offloading configuration passed")
    
    # Test memory optimization settings for RTX 4080
    print("4. Testing memory optimization settings...")
    settings_16gb = optimizer.get_memory_optimization_settings(16)
    assert settings_16gb['enable_attention_slicing'] == False, "Attention slicing should be disabled for 16GB"
    assert settings_16gb['enable_vae_slicing'] == False, "VAE slicing should be disabled for 16GB"
    assert settings_16gb['enable_cpu_offload'] == True, "CPU offload should be enabled"
    assert settings_16gb['batch_size'] == 2, "Batch size should be 2 for 16GB"
    assert settings_16gb['vae_tile_size'] == (256, 256), "VAE tile size should be (256, 256)"
    
    print("✓ Memory optimization settings passed")
    
    # Test optimization application (mocked)
    print("5. Testing RTX 4080 optimization application...")
    with patch('hardware_optimizer.TORCH_AVAILABLE', True), \
         patch('torch.backends.cudnn') as mock_cudnn, \
         patch('torch.backends.cuda.matmul') as mock_matmul, \
         patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.set_per_process_memory_fraction') as mock_memory_fraction, \
         patch('torch.set_num_threads') as mock_threads:
        
        result = optimizer.apply_rtx_4080_optimizations(rtx_4080_profile)
        
        assert result.success == True, "RTX 4080 optimization should succeed"
        assert "Enabled Tensor Cores (TF32)" in result.optimizations_applied, "Tensor cores should be enabled"
        assert "Set CUDA memory allocator optimization" in result.optimizations_applied, "Memory allocator should be optimized"
        assert result.settings is not None, "Settings should be included in result"
        assert len(result.errors) == 0, f"No errors expected, got: {result.errors}"
    
    print("✓ RTX 4080 optimization application passed")
    
    print("\n🎉 All RTX 4080 tests passed successfully!")
    return True

if __name__ == '__main__':
    try:
        test_rtx_4080_functionality()
        print("\n✅ RTX 4080 Hardware Optimizer implementation is working correctly!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)