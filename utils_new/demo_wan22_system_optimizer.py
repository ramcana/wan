"""
Demo script for WAN22 System Optimizer core framework.

This script demonstrates the basic functionality of the system optimizer
including hardware detection, system initialization, and monitoring.
"""

import json
from wan22_system_optimizer import WAN22SystemOptimizer


def main():
    """Demonstrate WAN22 System Optimizer functionality."""
    print("=== WAN22 System Optimizer Demo ===\n")
    
    # Initialize the optimizer
    print("1. Initializing WAN22 System Optimizer...")
    optimizer = WAN22SystemOptimizer(log_level="INFO")
    
    # Initialize the system
    print("\n2. Performing system initialization...")
    init_result = optimizer.initialize_system()
    
    if init_result.success:
        print("✓ System initialization completed successfully")
        print(f"  Applied {len(init_result.optimizations_applied)} optimizations:")
        for opt in init_result.optimizations_applied:
            print(f"    - {opt}")
    else:
        print("✗ System initialization failed")
        for error in init_result.errors:
            print(f"    Error: {error}")
        return
    
    # Display hardware profile
    print("\n3. Hardware Profile:")
    profile = optimizer.get_hardware_profile()
    if profile:
        print(f"  CPU: {profile.cpu_model}")
        print(f"  Cores: {profile.cpu_cores} cores, {profile.cpu_threads} threads")
        print(f"  Memory: {profile.total_memory_gb}GB")
        print(f"  GPU: {profile.gpu_model}")
        print(f"  VRAM: {profile.vram_gb}GB")
        print(f"  CUDA: {profile.cuda_version}")
        print(f"  Platform: {profile.platform_info}")
    else:
        print("  No hardware profile available")
    
    # Validate system
    print("\n4. Performing system validation...")
    validate_result = optimizer.validate_and_repair_system()
    
    if validate_result.success:
        print("✓ System validation completed successfully")
        if validate_result.optimizations_applied:
            for opt in validate_result.optimizations_applied:
                print(f"    - {opt}")
    else:
        print("⚠ System validation completed with issues")
        for warning in validate_result.warnings:
            print(f"    Warning: {warning}")
        for error in validate_result.errors:
            print(f"    Error: {error}")
    
    # Apply hardware optimizations
    print("\n5. Applying hardware-specific optimizations...")
    opt_result = optimizer.apply_hardware_optimizations()
    
    if opt_result.success:
        print("✓ Hardware optimizations applied successfully")
        for opt in opt_result.optimizations_applied:
            print(f"    - {opt}")
    else:
        print("⚠ Hardware optimizations completed with issues")
        for warning in opt_result.warnings:
            print(f"    Warning: {warning}")
        for error in opt_result.errors:
            print(f"    Error: {error}")
    
    # Monitor system health
    print("\n6. System Health Monitoring:")
    metrics = optimizer.monitor_system_health()
    
    print(f"  Timestamp: {metrics.timestamp}")
    print(f"  CPU Usage: {metrics.cpu_usage_percent}%")
    print(f"  Memory Usage: {metrics.memory_usage_gb}GB")
    print(f"  VRAM Usage: {metrics.vram_usage_mb}MB / {metrics.vram_total_mb}MB")
    if metrics.vram_total_mb > 0:
        vram_percent = (metrics.vram_usage_mb / metrics.vram_total_mb) * 100
        print(f"  VRAM Utilization: {vram_percent:.1f}%")
    
    # Show optimization history
    print("\n7. Optimization History:")
    history = optimizer.get_optimization_history()
    for i, entry in enumerate(history, 1):
        status = "✓" if entry['success'] else "✗"
        print(f"  {i}. {status} {entry['operation']} ({len(entry['optimizations_applied'])} optimizations)")
    
    # Save hardware profile
    print("\n8. Saving hardware profile...")
    if optimizer.save_profile_to_file("hardware_profile_demo.json"):
        print("✓ Hardware profile saved to hardware_profile_demo.json")
        
        # Display saved profile content
        try:
            with open("hardware_profile_demo.json", 'r') as f:
                saved_profile = json.load(f)
            print("  Saved profile summary:")
            print(f"    CPU: {saved_profile.get('cpu_model', 'Unknown')}")
            print(f"    GPU: {saved_profile.get('gpu_model', 'Unknown')}")
            print(f"    Detection Time: {saved_profile.get('detection_timestamp', 'Unknown')}")
        except Exception as e:
            print(f"  Could not read saved profile: {e}")
    else:
        print("✗ Failed to save hardware profile")
    
    print("\n=== Demo Complete ===")
    
    # Summary
    print(f"\nSummary:")
    print(f"- System initialized: {'Yes' if init_result.success else 'No'}")
    print(f"- Hardware detected: {'Yes' if profile else 'No'}")
    print(f"- Total optimizations applied: {len(init_result.optimizations_applied) + len(opt_result.optimizations_applied)}")
    print(f"- System ready for WAN22 operations: {'Yes' if init_result.success and validate_result.success else 'No'}")


if __name__ == "__main__":
    main()
