#!/usr/bin/env python3
"""
Test script for the comprehensive configuration manager.
Tests the integrated configuration management functionality.
"""

import sys
import json
import tempfile
from pathlib import Path

# Add the scripts directory to the path
scripts_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from scripts.interfaces import HardwareProfile, CPUInfo, MemoryInfo, GPUInfo, StorageInfo, OSInfo
from scripts.config_manager import ConfigurationManager, create_configuration_for_hardware, validate_configuration_file


def create_test_hardware_profile() -> HardwareProfile:
    """Create a test hardware profile."""
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


def test_configuration_creation():
    """Test configuration creation and management."""
    print("=== Testing Configuration Creation ===\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = ConfigurationManager(temp_dir)
        hardware_profile = create_test_hardware_profile()
        
        # Test different configuration variants
        variants = ["balanced", "performance", "memory_conservative", "quality_focused"]
        
        for variant in variants:
            print(f"--- Testing {variant.title()} Configuration ---")
            
            config = manager.create_optimized_configuration(hardware_profile, variant)
            
            print(f"✅ {variant.title()} configuration created")
            print(f"  Quantization: {config['system']['default_quantization']}")
            print(f"  CPU Threads: {config['optimization']['cpu_threads']}")
            print(f"  Memory Pool: {config['optimization']['memory_pool_gb']}GB")
            print(f"  Variant: {config['metadata']['variant']}")
            
            # Validate the created configuration
            validation_result = manager.config_validator.validate_configuration(config, hardware_profile)
            print(f"  Validation: {'✅ PASSED' if validation_result.success else '❌ FAILED'}")
            
            if validation_result.warnings:
                print(f"  Warnings: {len(validation_result.warnings)}")
            
            print()


def test_configuration_persistence():
    """Test configuration saving and loading."""
    print("=== Testing Configuration Persistence ===\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = ConfigurationManager(temp_dir)
        hardware_profile = create_test_hardware_profile()
        
        # Create and save configuration
        config = manager.create_optimized_configuration(hardware_profile, "balanced")
        success = manager.save_configuration(config)
        
        print(f"✅ Configuration saved: {success}")
        print(f"Config file exists: {manager.config_file.exists()}")
        
        # Load configuration
        loaded_config = manager.load_configuration()
        
        if loaded_config:
            print("✅ Configuration loaded successfully")
            print(f"  Loaded variant: {loaded_config.get('metadata', {}).get('variant', 'unknown')}")
            print(f"  CPU threads: {loaded_config['optimization']['cpu_threads']}")
        else:
            print("❌ Failed to load configuration")
        
        # Test configuration status
        status = manager.get_configuration_status(hardware_profile)
        print(f"\n--- Configuration Status ---")
        print(f"Config exists: {status['config_exists']}")
        print(f"Validation passed: {status['validation_result']['success'] if status['validation_result'] else 'N/A'}")
        print(f"Backups available: {status['backups_available']}")
        print(f"Recommendations: {len(status['recommendations'])}")


def test_configuration_optimization():
    """Test configuration optimization functionality."""
    print("\n=== Testing Configuration Optimization ===\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = ConfigurationManager(temp_dir)
        hardware_profile = create_test_hardware_profile()
        
        # Create a suboptimal configuration
        suboptimal_config = {
            "system": {
                "default_quantization": "int8",
                "enable_offload": True,
                "vae_tile_size": 128,
                "max_queue_size": 50,
                "worker_threads": 2
            },
            "optimization": {
                "cpu_threads": 32,
                "memory_pool_gb": 50,
                "max_vram_usage_gb": 15
            },
            "models": {
                "cache_models": False,
                "preload_models": False,
                "model_precision": "int8"
            }
        }
        
        # Save suboptimal configuration
        manager.save_configuration(suboptimal_config)
        
        print("--- Before Optimization ---")
        status_before = manager.get_configuration_status(hardware_profile)
        print(f"Validation passed: {status_before['validation_result']['success']}")
        print(f"Warnings: {status_before['validation_result']['warning_count']}")
        print(f"Recommendations: {len(status_before['recommendations'])}")
        
        # Optimize configuration
        success = manager.optimize_current_configuration(hardware_profile)
        print(f"\n✅ Optimization completed: {success}")
        
        # Check status after optimization
        print("\n--- After Optimization ---")
        status_after = manager.get_configuration_status(hardware_profile)
        print(f"Validation passed: {status_after['validation_result']['success']}")
        print(f"Warnings: {status_after['validation_result']['warning_count']}")
        print(f"Recommendations: {len(status_after['recommendations'])}")
        
        # Show optimized values
        optimized_config = manager.load_configuration()
        if optimized_config:
            print(f"\nOptimized values:")
            print(f"  CPU Threads: {optimized_config['optimization']['cpu_threads']}")
            print(f"  Memory Pool: {optimized_config['optimization']['memory_pool_gb']}GB")
            print(f"  VRAM Usage: {optimized_config['optimization']['max_vram_usage_gb']}GB")


def test_configuration_repair():
    """Test configuration repair functionality."""
    print("\n=== Testing Configuration Repair ===\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = ConfigurationManager(temp_dir)
        hardware_profile = create_test_hardware_profile()
        
        # Create a broken configuration
        broken_config = {
            "system": {
                "default_quantization": "invalid_quant",
                "enable_offload": "not_boolean",
                "vae_tile_size": 999,
                "max_queue_size": -5,
                "worker_threads": "not_number"
            },
            "optimization": {
                "cpu_threads": 1000,
                "memory_pool_gb": 500
            },
            "models": {
                "model_precision": "invalid_precision"
            }
        }
        
        # Save broken configuration
        manager.save_configuration(broken_config)
        
        print("--- Before Repair ---")
        validation_result = manager.validate_current_configuration(hardware_profile)
        print(f"Validation passed: {validation_result.success}")
        if validation_result.details and "errors" in validation_result.details:
            print(f"Errors: {len(validation_result.details['errors'])}")
            for error in validation_result.details["errors"][:3]:  # Show first 3 errors
                print(f"  ❌ {error}")
        
        # Repair configuration
        success = manager.repair_configuration(hardware_profile)
        print(f"\n✅ Repair completed: {success}")
        
        # Check status after repair
        print("\n--- After Repair ---")
        validation_result = manager.validate_current_configuration(hardware_profile)
        print(f"Validation passed: {validation_result.success}")
        
        if validation_result.success:
            repaired_config = manager.load_configuration()
            print(f"Repaired values:")
            print(f"  Quantization: {repaired_config['system']['default_quantization']}")
            print(f"  VAE Tile Size: {repaired_config['system']['vae_tile_size']}")
            print(f"  Worker Threads: {repaired_config['system']['worker_threads']}")


def test_configuration_report():
    """Test configuration report generation."""
    print("\n=== Testing Configuration Report ===\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = ConfigurationManager(temp_dir)
        hardware_profile = create_test_hardware_profile()
        
        # Create and save a configuration
        config = manager.create_optimized_configuration(hardware_profile, "performance")
        manager.save_configuration(config)
        
        # Create some backups
        for i in range(3):
            manager.config_validator.create_backup(str(manager.config_file), f"test_backup_{i}.json")
        
        # Generate report
        report = manager.create_configuration_report(hardware_profile)
        
        print("--- Configuration Report ---")
        print(f"Hardware: {report['hardware_profile']['cpu']}")
        print(f"Memory: {report['hardware_profile']['memory']}")
        print(f"GPU: {report['hardware_profile']['gpu']}")
        print(f"Storage: {report['hardware_profile']['storage']}")
        
        print(f"\nConfiguration Status:")
        print(f"  Exists: {report['configuration_status']['config_exists']}")
        print(f"  Valid: {report['configuration_status']['validation_result']['success']}")
        print(f"  Recommendations: {len(report['configuration_status']['recommendations'])}")
        
        print(f"\nBackup Info:")
        print(f"  Total backups: {report['backup_info']['backup_count']}")
        print(f"  Recent backups: {len(report['backup_info']['backups'])}")
        
        if "current_configuration" in report:
            print(f"\nCurrent Configuration:")
            current = report["current_configuration"]
            print(f"  Variant: {current['variant']}")
            print(f"  Quantization: {current['quantization']}")
            print(f"  CPU Threads: {current['cpu_threads']}")
            print(f"  Memory Pool: {current['memory_pool_gb']}GB")


def test_standalone_functions():
    """Test standalone utility functions."""
    print("\n=== Testing Standalone Functions ===\n")
    
    hardware_profile = create_test_hardware_profile()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test standalone configuration creation
        config = create_configuration_for_hardware(hardware_profile, temp_dir, "balanced")
        print("✅ Standalone configuration creation successful")
        print(f"  Variant: {config['metadata']['variant']}")
        print(f"  CPU Threads: {config['optimization']['cpu_threads']}")
        
        # Save config to file for validation test
        config_path = Path(temp_dir) / "test_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Test standalone validation
        validation_result = validate_configuration_file(str(config_path), hardware_profile)
        print(f"✅ Standalone validation: {'PASSED' if validation_result.success else 'FAILED'}")
        print(f"  Message: {validation_result.message}")


def main():
    """Run all configuration manager tests."""
    try:
        test_configuration_creation()
        test_configuration_persistence()
        test_configuration_optimization()
        test_configuration_repair()
        test_configuration_report()
        test_standalone_functions()
        
        print("\n🎉 All configuration manager tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
