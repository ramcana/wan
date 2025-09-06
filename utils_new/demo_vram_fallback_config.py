#!/usr/bin/env python3
"""
Demo script for VRAM Fallback Configuration System

This script demonstrates the key features of task 3.3:
- Manual VRAM specification for detection failures
- Validation system for manual VRAM settings
- Persistent storage for VRAM configuration preferences
- GPU selection interface for multi-GPU systems
"""

import logging
import json
from vram_config_manager import VRAMConfigManager, GPUSelectionCriteria

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("=== VRAM Fallback Configuration System Demo ===\n")
    
    # Initialize VRAM config manager
    config_manager = VRAMConfigManager("demo_vram_configs")
    
    # 1. Manual VRAM specification for detection failures
    print("1. Manual VRAM Configuration for Detection Failures")
    print("-" * 50)
    
    # Example: RTX 4080 setup
    rtx_4080_config = {0: 16}  # GPU 0 with 16GB VRAM
    success, errors = config_manager.create_manual_vram_config(
        rtx_4080_config, 
        "rtx_4080_fallback", 
        "Fallback config for RTX 4080 when detection fails"
    )
    
    if success:
        print(f"✓ Created RTX 4080 fallback config: {rtx_4080_config}")
    else:
        print(f"✗ Failed to create config: {errors}")
    
    # Example: Multi-GPU setup
    multi_gpu_config = {0: 16, 1: 24}  # RTX 4080 + RTX 3090
    success, errors = config_manager.create_manual_vram_config(
        multi_gpu_config,
        "multi_gpu_fallback",
        "Fallback config for RTX 4080 + RTX 3090 setup"
    )
    
    if success:
        print(f"✓ Created multi-GPU fallback config: {multi_gpu_config}")
    else:
        print(f"✗ Failed to create config: {errors}")
    
    print()
    
    # 2. Validation system for manual VRAM settings
    print("2. Validation System for Manual VRAM Settings")
    print("-" * 50)
    
    # Test valid configurations
    valid_configs = [
        {0: 8},      # Single GPU with 8GB
        {0: 16, 1: 24},  # Dual GPU setup
        {0: 12, 1: 12, 2: 12, 3: 12}  # Quad GPU setup
    ]
    
    for config in valid_configs:
        is_valid, errors = config_manager.validate_manual_vram_config(config)
        status = "✓ Valid" if is_valid else f"✗ Invalid: {errors}"
        print(f"Config {config}: {status}")
    
    # Test invalid configurations
    invalid_configs = [
        {},           # Empty config
        {-1: 16},     # Negative GPU index
        {0: 0},       # Zero VRAM
        {0: 200},     # Too much VRAM
        {"invalid": 16}  # Non-integer GPU index
    ]
    
    print("\nTesting invalid configurations:")
    for config in invalid_configs:
        is_valid, errors = config_manager.validate_manual_vram_config(config)
        if not is_valid:
            print(f"Config {config}: ✓ Correctly rejected - {errors[0]}")
        else:
            print(f"Config {config}: ✗ Should have been rejected")
    
    print()
    
    # 3. Persistent storage for VRAM configuration preferences
    print("3. Persistent Storage for VRAM Configuration Preferences")
    print("-" * 50)
    
    # Create additional profiles
    profiles_to_create = [
        ("gaming_setup", "Gaming configuration with RTX 4080", {0: 16}),
        ("workstation", "Workstation with dual RTX 3090", {0: 24, 1: 24}),
        ("budget_setup", "Budget setup with GTX 1660", {0: 6})
    ]
    
    for name, desc, config in profiles_to_create:
        success, message = config_manager.create_profile(name, desc, config)
        print(f"Profile '{name}': {'✓' if success else '✗'} {message}")
    
    # List all profiles
    print("\nAll VRAM configuration profiles:")
    profiles = config_manager.list_profiles()
    for name, info in profiles.items():
        current = " (CURRENT)" if info['is_current'] else ""
        print(f"  - {name}: {info['description']} - {info['total_vram_gb']}GB total{current}")
    
    # Load a profile
    success, message = config_manager.load_profile("gaming_setup")
    print(f"\nLoading 'gaming_setup' profile: {'✓' if success else '✗'} {message}")
    
    print()
    
    # 4. GPU selection interface for multi-GPU systems
    print("4. GPU Selection Interface for Multi-GPU Systems")
    print("-" * 50)
    
    # Get fallback configuration options
    fallback_options = config_manager.get_fallback_config_options()
    
    print("Common GPU configurations:")
    for gpu_name, config in fallback_options['common_gpu_configs'].items():
        vram_gb = list(config.values())[0]
        print(f"  - {gpu_name}: {vram_gb}GB VRAM")
    
    print("\nMulti-GPU examples:")
    for setup_name, config in fallback_options['multi_gpu_examples'].items():
        total_vram = sum(config.values())
        gpu_count = len(config)
        print(f"  - {setup_name}: {gpu_count} GPUs, {total_vram}GB total")
    
    # Test GPU selection with criteria
    print("\nGPU Selection with Criteria:")
    
    # High-end gaming criteria
    gaming_criteria = GPUSelectionCriteria(
        min_vram_gb=12,
        preferred_models=["RTX 4080", "RTX 4090"],
        require_cuda=True
    )
    
    interface_data = config_manager.get_gpu_selection_interface(gaming_criteria)
    print(f"Gaming criteria (min 12GB VRAM, RTX 40xx series):")
    print(f"  - Available GPUs: {len(interface_data.get('available_gpus', []))}")
    print(f"  - Multi-GPU enabled: {interface_data.get('multi_gpu_enabled', False)}")
    
    # Show recommendations
    recommendations = interface_data.get('recommendations', [])
    if recommendations:
        print("  - Recommendations:")
        for rec in recommendations[:3]:  # Show first 3 recommendations
            print(f"    * {rec['type']}: {rec['message']}")
    
    print()
    
    # 5. Export/Import configuration
    print("5. Configuration Export/Import")
    print("-" * 50)
    
    # Export configuration
    export_path = "demo_vram_config_export.json"
    success, message = config_manager.export_config(export_path)
    print(f"Export config: {'✓' if success else '✗'} {message}")
    
    # Show export content
    if success:
        try:
            with open(export_path, 'r') as f:
                export_data = json.load(f)
            print(f"Exported {len(export_data.get('profiles', {}))} profiles")
        except Exception as e:
            print(f"Error reading export: {e}")
    
    # Test import (create new manager and import)
    new_manager = VRAMConfigManager("demo_import_test")
    success, message = new_manager.import_config(export_path)
    print(f"Import config: {'✓' if success else '✗'} {message}")
    
    if success:
        imported_profiles = new_manager.list_profiles()
        print(f"Imported {len(imported_profiles)} profiles successfully")
    
    print()
    
    # 6. System status
    print("6. System Status Summary")
    print("-" * 50)
    
    status = config_manager.get_system_status()
    print(f"Configuration valid: {status.get('config_valid', False)}")
    print(f"Total profiles: {status.get('profiles_count', 0)}")
    print(f"Current profile: {status.get('current_profile', 'None')}")
    
    detection_summary = status.get('detection_summary', {})
    print(f"Total GPUs detected: {detection_summary.get('total_gpus', 0)}")
    print(f"Available GPUs: {detection_summary.get('available_gpus', 0)}")
    
    print("\n=== Demo Complete ===")
    
    # Cleanup
    config_manager.cleanup()
    new_manager.cleanup()

if __name__ == "__main__":
    main()