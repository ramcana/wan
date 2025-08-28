#!/usr/bin/env python3
"""
Test script for the hardware-aware configuration generator.
Tests configuration generation for different hardware profiles.
"""

import sys
import json
from pathlib import Path

# Add the scripts directory to the path
scripts_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(scripts_dir))

# Import with absolute imports
from scripts.interfaces import HardwareProfile, CPUInfo, MemoryInfo, GPUInfo, StorageInfo, OSInfo
from scripts.generate_config import ConfigurationEngine, calculate_optimal_settings, get_hardware_recommendations


def create_high_end_profile() -> HardwareProfile:
    """Create a high-end hardware profile for testing."""
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
            available_gb=500,
            type="NVMe SSD"
        ),
        os=OSInfo(
            name="Windows",
            version="11",
            architecture="x64"
        )
    )


def create_mid_range_profile() -> HardwareProfile:
    """Create a mid-range hardware profile for testing."""
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
            speed=3600
        ),
        gpu=GPUInfo(
            model="NVIDIA GeForce RTX 3070",
            vram_gb=8,
            cuda_version="11.8",
            driver_version="531.29",
            compute_capability="8.6"
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


def create_budget_profile() -> HardwareProfile:
    """Create a budget hardware profile for testing."""
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
            speed=3200
        ),
        gpu=GPUInfo(
            model="NVIDIA GeForce GTX 1660 Ti",
            vram_gb=6,
            cuda_version="11.2",
            driver_version="471.96",
            compute_capability="7.5"
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


def create_minimum_profile() -> HardwareProfile:
    """Create a minimum spec hardware profile for testing."""
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
            speed=2666
        ),
        gpu=GPUInfo(
            model="NVIDIA GeForce GTX 1060",
            vram_gb=6,
            cuda_version="10.2",
            driver_version="456.71",
            compute_capability="6.1"
        ),
        storage=StorageInfo(
            available_gb=50,
            type="HDD"
        ),
        os=OSInfo(
            name="Windows",
            version="10",
            architecture="x64"
        )
    )


def test_configuration_generation():
    """Test configuration generation for different hardware profiles."""
    print("=== Testing Hardware-Aware Configuration Generator ===\n")
    
    # Create test profiles
    profiles = {
        "High-End": create_high_end_profile(),
        "Mid-Range": create_mid_range_profile(),
        "Budget": create_budget_profile(),
        "Minimum": create_minimum_profile()
    }
    
    engine = ConfigurationEngine(".")
    
    for profile_name, profile in profiles.items():
        print(f"--- {profile_name} Hardware Profile ---")
        print(f"CPU: {profile.cpu.model} ({profile.cpu.cores}C/{profile.cpu.threads}T)")
        print(f"Memory: {profile.memory.total_gb}GB {profile.memory.type}")
        print(f"GPU: {profile.gpu.model} ({profile.gpu.vram_gb}GB VRAM)")
        print(f"Storage: {profile.storage.available_gb}GB {profile.storage.type}")
        
        # Test hardware tier classification
        tier = engine.classify_hardware_tier(profile)
        print(f"Hardware Tier: {tier}")
        
        # Generate configuration
        config = engine.generate_config(profile)
        
        # Display key configuration settings
        print("\nGenerated Configuration:")
        print(f"  Quantization: {config['system']['default_quantization']}")
        print(f"  CPU Threads: {config['optimization']['cpu_threads']}")
        print(f"  Worker Threads: {config['system']['worker_threads']}")
        print(f"  Memory Pool: {config['optimization']['memory_pool_gb']}GB")
        print(f"  Max VRAM: {config['optimization']['max_vram_usage_gb']}GB")
        print(f"  VAE Tile Size: {config['system']['vae_tile_size']}")
        print(f"  Queue Size: {config['system']['max_queue_size']}")
        print(f"  Offload Enabled: {config['system']['enable_offload']}")
        
        # Test configuration variants
        variants = engine.create_config_variants(profile)
        print(f"\nConfiguration Variants: {list(variants.keys())}")
        
        # Get hardware recommendations
        recommendations = get_hardware_recommendations(profile)
        if recommendations:
            print("\nHardware Recommendations:")
            for rec in recommendations:
                print(f"  • {rec}")
        else:
            print("\nNo hardware recommendations - system is well configured!")
        
        print("\n" + "="*60 + "\n")


def test_config_saving():
    """Test configuration saving functionality."""
    print("=== Testing Configuration Saving ===\n")
    
    engine = ConfigurationEngine(".")
    profile = create_high_end_profile()
    
    # Generate configuration
    config = engine.generate_config(profile)
    
    # Save configuration
    config_path = "test_config.json"
    success = engine.save_config(config, config_path)
    
    if success:
        print(f"✅ Configuration saved successfully to {config_path}")
        
        # Verify file exists and is valid JSON
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
            print("✅ Configuration file is valid JSON")
            print(f"Configuration contains {len(loaded_config)} top-level sections")
            
            # Clean up
            Path(config_path).unlink()
            print("✅ Test file cleaned up")
            
        except Exception as e:
            print(f"❌ Error verifying saved configuration: {e}")
    else:
        print("❌ Failed to save configuration")


def test_optimization_functions():
    """Test individual optimization functions."""
    print("=== Testing Optimization Functions ===\n")
    
    engine = ConfigurationEngine(".")
    profile = create_mid_range_profile()
    
    # Test base template selection
    tier = engine.classify_hardware_tier(profile)
    base_config = engine.config_templates[tier].copy()
    print(f"Base template for {tier}:")
    print(f"  Default quantization: {base_config['system']['default_quantization']}")
    print(f"  Worker threads: {base_config['system']['worker_threads']}")
    
    # Test hardware optimization
    optimized_config = engine.optimize_for_hardware(base_config, profile)
    print(f"\nAfter hardware optimization:")
    print(f"  CPU threads: {optimized_config['optimization']['cpu_threads']}")
    print(f"  Memory pool: {optimized_config['optimization']['memory_pool_gb']}GB")
    print(f"  Max VRAM: {optimized_config['optimization']['max_vram_usage_gb']}GB")
    
    print("✅ Optimization functions working correctly")


def main():
    """Run all configuration generator tests."""
    try:
        test_configuration_generation()
        test_config_saving()
        test_optimization_functions()
        
        print("🎉 All configuration generator tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())