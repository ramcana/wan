#!/usr/bin/env python3
"""
Test script for the configuration validator and optimization system.
Tests validation, optimization, and backup functionality.
"""

import sys
import json
import tempfile
from pathlib import Path

# Add the scripts directory to the path
scripts_dir = Path(__file__).parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from scripts.interfaces import HardwareProfile, CPUInfo, MemoryInfo, GPUInfo, StorageInfo, OSInfo
from scripts.config_validator import ConfigurationValidator
from scripts.generate_config import ConfigurationEngine


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


def create_valid_config() -> dict:
    """Create a valid configuration for testing."""
    return {
        "system": {
            "default_quantization": "fp16",
            "enable_offload": True,
            "vae_tile_size": 256,
            "max_queue_size": 10,
            "worker_threads": 8,
            "enable_gpu_acceleration": True
        },
        "optimization": {
            "cpu_threads": 12,
            "memory_pool_gb": 8,
            "max_vram_usage_gb": 6
        },
        "models": {
            "cache_models": True,
            "preload_models": False,
            "model_precision": "fp16"
        }
    }


def create_invalid_config() -> dict:
    """Create an invalid configuration for testing."""
    return {
        "system": {
            "default_quantization": "invalid_quant",  # Invalid quantization
            "enable_offload": "not_boolean",  # Should be boolean
            "vae_tile_size": 999,  # Invalid tile size
            "max_queue_size": -5,  # Negative value
            "worker_threads": "not_number"  # Should be number
        },
        "optimization": {
            "cpu_threads": 1000,  # Too high
            "memory_pool_gb": 500,  # Too high
            "max_vram_usage_gb": 50  # Too high for test GPU
        },
        "models": {
            "model_precision": "invalid_precision"  # Invalid precision
        }
        # Missing required sections
    }


def create_conflicting_config() -> dict:
    """Create a configuration with conflicting settings."""
    return {
        "system": {
            "default_quantization": "fp16",
            "enable_offload": True,
            "vae_tile_size": 256,
            "max_queue_size": 10,
            "worker_threads": 8,
            "enable_gpu_acceleration": False  # Disabled GPU
        },
        "optimization": {
            "cpu_threads": 12,
            "memory_pool_gb": 8,
            "max_vram_usage_gb": 6  # But VRAM usage set
        },
        "models": {
            "cache_models": False,
            "preload_models": True,  # Preload without cache
            "model_precision": "int8"  # Different from system quantization
        }
    }


def test_configuration_validation():
    """Test configuration validation functionality."""
    print("=== Testing Configuration Validation ===\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        validator = ConfigurationValidator(temp_dir)
        hardware_profile = create_test_hardware_profile()
        
        # Test valid configuration
        print("--- Testing Valid Configuration ---")
        valid_config = create_valid_config()
        result = validator.validate_configuration(valid_config, hardware_profile)
        
        print(f"Validation Result: {'✅ PASSED' if result.success else '❌ FAILED'}")
        print(f"Message: {result.message}")
        if result.warnings:
            print(f"Warnings: {len(result.warnings)}")
            for warning in result.warnings:
                print(f"  ⚠️  {warning}")
        
        # Test invalid configuration
        print("\n--- Testing Invalid Configuration ---")
        invalid_config = create_invalid_config()
        result = validator.validate_configuration(invalid_config, hardware_profile)
        
        print(f"Validation Result: {'✅ PASSED' if result.success else '❌ FAILED (Expected)'}")
        print(f"Message: {result.message}")
        if result.details and "errors" in result.details:
            print(f"Errors found: {len(result.details['errors'])}")
            for error in result.details["errors"]:
                print(f"  ❌ {error}")
        
        # Test conflicting configuration
        print("\n--- Testing Conflicting Configuration ---")
        conflicting_config = create_conflicting_config()
        result = validator.validate_configuration(conflicting_config, hardware_profile)
        
        print(f"Validation Result: {'✅ PASSED' if result.success else '❌ FAILED'}")
        if result.warnings:
            print(f"Conflicts found: {len(result.warnings)}")
            for warning in result.warnings:
                print(f"  ⚠️  {warning}")


def test_configuration_optimization():
    """Test configuration optimization functionality."""
    print("\n=== Testing Configuration Optimization ===\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        validator = ConfigurationValidator(temp_dir)
        hardware_profile = create_test_hardware_profile()
        
        # Create a suboptimal configuration
        suboptimal_config = {
            "system": {
                "default_quantization": "int8",
                "enable_offload": True,
                "vae_tile_size": 128,
                "max_queue_size": 50,  # Too high
                "worker_threads": 2   # Too low
            },
            "optimization": {
                "cpu_threads": 32,    # Too high for test system
                "memory_pool_gb": 30, # Too high
                "max_vram_usage_gb": 10  # Too high for test GPU
            },
            "models": {
                "cache_models": False,
                "preload_models": False,
                "model_precision": "int8"
            }
        }
        
        print("--- Original Configuration ---")
        print(f"CPU Threads: {suboptimal_config['optimization']['cpu_threads']}")
        print(f"Memory Pool: {suboptimal_config['optimization']['memory_pool_gb']}GB")
        print(f"VRAM Usage: {suboptimal_config['optimization']['max_vram_usage_gb']}GB")
        print(f"Worker Threads: {suboptimal_config['system']['worker_threads']}")
        print(f"Queue Size: {suboptimal_config['system']['max_queue_size']}")
        
        # Optimize configuration
        optimized_config = validator.optimize_configuration(suboptimal_config, hardware_profile)
        
        print("\n--- Optimized Configuration ---")
        print(f"CPU Threads: {optimized_config['optimization']['cpu_threads']}")
        print(f"Memory Pool: {optimized_config['optimization']['memory_pool_gb']}GB")
        print(f"VRAM Usage: {optimized_config['optimization']['max_vram_usage_gb']}GB")
        print(f"Worker Threads: {optimized_config['system']['worker_threads']}")
        print(f"Queue Size: {optimized_config['system']['max_queue_size']}")
        
        # Get optimization recommendations
        recommendations = validator.get_optimization_recommendations(optimized_config, hardware_profile)
        if recommendations:
            print("\n--- Optimization Recommendations ---")
            for rec in recommendations:
                print(f"  💡 {rec}")
        else:
            print("\n✅ Configuration is well optimized!")


def test_backup_and_restore():
    """Test configuration backup and restore functionality."""
    print("\n=== Testing Backup and Restore ===\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        validator = ConfigurationValidator(temp_dir)
        
        # Create a test configuration file
        config = create_valid_config()
        config_path = Path(temp_dir) / "test_config.json"
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Created test configuration: {config_path}")
        
        # Create backup
        backup_path = validator.create_backup(str(config_path))
        print(f"✅ Backup created: {backup_path}")
        
        # List backups
        backups = validator.list_backups()
        print(f"✅ Found {len(backups)} backup(s)")
        for backup in backups:
            print(f"  📁 {backup['name']} ({backup['size_bytes']} bytes)")
        
        # Modify original configuration
        modified_config = config.copy()
        modified_config["system"]["worker_threads"] = 999
        
        with open(config_path, 'w') as f:
            json.dump(modified_config, f, indent=2)
        
        print("✅ Modified original configuration")
        
        # Restore from backup
        success = validator.restore_backup(backup_path, str(config_path))
        if success:
            print("✅ Configuration restored from backup")
            
            # Verify restoration
            with open(config_path, 'r') as f:
                restored_config = json.load(f)
            
            if restored_config["system"]["worker_threads"] == 8:  # Original value
                print("✅ Restoration verified - original values restored")
            else:
                print("❌ Restoration failed - values not restored correctly")
        else:
            print("❌ Failed to restore configuration")
        
        # Test cleanup
        # Create multiple backups
        for i in range(3):
            validator.create_backup(str(config_path), f"test_backup_{i}.json")
        
        backups_before = len(validator.list_backups())
        deleted_count = validator.cleanup_old_backups(keep_count=2)
        backups_after = len(validator.list_backups())
        
        print(f"✅ Cleanup: {deleted_count} backups deleted ({backups_before} -> {backups_after})")


def test_integration_with_generator():
    """Test integration between configuration generator and validator."""
    print("\n=== Testing Generator-Validator Integration ===\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Generate configuration
        engine = ConfigurationEngine(temp_dir)
        validator = ConfigurationValidator(temp_dir)
        hardware_profile = create_test_hardware_profile()
        
        # Generate configuration
        generated_config = engine.generate_config(hardware_profile)
        print("✅ Configuration generated")
        
        # Validate generated configuration
        validation_result = validator.validate_configuration(generated_config, hardware_profile)
        print(f"✅ Generated config validation: {'PASSED' if validation_result.success else 'FAILED'}")
        
        if validation_result.warnings:
            print(f"Warnings: {len(validation_result.warnings)}")
            for warning in validation_result.warnings:
                print(f"  ⚠️  {warning}")
        
        # Optimize generated configuration
        optimized_config = validator.optimize_configuration(generated_config, hardware_profile)
        print("✅ Configuration optimized")
        
        # Validate optimized configuration
        opt_validation_result = validator.validate_configuration(optimized_config, hardware_profile)
        print(f"✅ Optimized config validation: {'PASSED' if opt_validation_result.success else 'FAILED'}")
        
        # Compare configurations
        print("\n--- Configuration Comparison ---")
        print(f"Original CPU threads: {generated_config['optimization']['cpu_threads']}")
        print(f"Optimized CPU threads: {optimized_config['optimization']['cpu_threads']}")
        print(f"Original memory pool: {generated_config['optimization']['memory_pool_gb']}GB")
        print(f"Optimized memory pool: {optimized_config['optimization']['memory_pool_gb']}GB")


def main():
    """Run all configuration validator tests."""
    try:
        test_configuration_validation()
        test_configuration_optimization()
        test_backup_and_restore()
        test_integration_with_generator()
        
        print("\n🎉 All configuration validator tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())