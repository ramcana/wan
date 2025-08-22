"""
Demo script for Threadripper PRO 5995WX optimizations
"""

import os
import sys
import logging

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hardware_optimizer import HardwareOptimizer, HardwareProfile

def main():
    """Demonstrate Threadripper PRO optimizations"""
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    print("=== Threadripper PRO 5995WX Optimization Demo ===\n")
    
    # Create hardware optimizer
    optimizer = HardwareOptimizer()
    
    # Detect current hardware
    print("1. Detecting hardware profile...")
    profile = optimizer.detect_hardware_profile()
    print(f"   CPU: {profile.cpu_model}")
    print(f"   CPU Cores: {profile.cpu_cores}")
    print(f"   Memory: {profile.total_memory_gb}GB")
    print(f"   GPU: {profile.gpu_model}")
    print(f"   VRAM: {profile.vram_gb}GB")
    print(f"   Threadripper PRO detected: {profile.is_threadripper_pro}")
    print()
    
    # Create a mock Threadripper PRO profile for demonstration
    print("2. Creating mock Threadripper PRO 5995WX profile...")
    threadripper_profile = HardwareProfile(
        cpu_model="AMD Ryzen Threadripper PRO 5995WX 64-Core Processor",
        cpu_cores=64,
        total_memory_gb=128,
        gpu_model="NVIDIA GeForce RTX 4080",
        vram_gb=16,
        cuda_version="12.1",
        driver_version="537.13",
        is_rtx_4080=True,
        is_threadripper_pro=True
    )
    print("   Mock profile created successfully")
    print()
    
    # Generate optimal settings
    print("3. Generating Threadripper PRO optimal settings...")
    settings = optimizer.generate_threadripper_pro_settings(threadripper_profile)
    print(f"   Tile size: {settings.tile_size}")
    print(f"   VAE tile size: {settings.vae_tile_size}")
    print(f"   Batch size: {settings.batch_size}")
    print(f"   CPU threads: {settings.num_threads}")
    print(f"   Parallel workers: {settings.parallel_workers}")
    print(f"   Preprocessing threads: {settings.preprocessing_threads}")
    print(f"   I/O threads: {settings.io_threads}")
    print(f"   NUMA nodes: {settings.numa_nodes}")
    print(f"   CPU affinity: {settings.cpu_affinity[:8] if settings.cpu_affinity else None}... (showing first 8)")
    print(f"   NUMA optimization: {settings.enable_numa_optimization}")
    print(f"   Memory fraction: {settings.memory_fraction}")
    print()
    
    # Configure parallel preprocessing
    print("4. Configuring parallel preprocessing...")
    parallel_config = optimizer.configure_parallel_preprocessing()
    print(f"   Preprocessing workers: {parallel_config['preprocessing_workers']}")
    print(f"   I/O workers: {parallel_config['io_workers']}")
    print(f"   Batch processing workers: {parallel_config['batch_processing_workers']}")
    print()
    
    # Apply optimizations
    print("5. Applying Threadripper PRO optimizations...")
    result = optimizer.apply_threadripper_pro_optimizations(threadripper_profile)
    
    if result.success:
        print("   ✓ Optimizations applied successfully!")
        print(f"   Applied {len(result.optimizations_applied)} optimizations:")
        for i, optimization in enumerate(result.optimizations_applied, 1):
            print(f"     {i}. {optimization}")
        
        if result.warnings:
            print(f"   Warnings ({len(result.warnings)}):")
            for warning in result.warnings:
                print(f"     - {warning}")
    else:
        print("   ✗ Optimization failed!")
        for error in result.errors:
            print(f"     Error: {error}")
    print()
    
    # Show environment variables set
    print("6. Environment variables set for optimization:")
    env_vars = ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 
                'TORCH_NUM_THREADS', 'PYTORCH_CUDA_ALLOC_CONF']
    
    for var in env_vars:
        if var in os.environ:
            print(f"   {var} = {os.environ[var]}")
    print()
    
    # Test hardware optimization routing
    print("7. Testing hardware optimization routing...")
    current_profile = optimizer.detect_hardware_profile()
    if current_profile.is_threadripper_pro:
        print("   Current system is Threadripper PRO - would apply Threadripper optimizations")
    elif current_profile.is_rtx_4080:
        print("   Current system has RTX 4080 - would apply RTX 4080 optimizations")
    else:
        print("   Current system would use default optimizations")
    
    # Save optimization profile
    print("8. Saving optimization profile...")
    profile_file = "threadripper_pro_optimization_profile.json"
    optimizer.hardware_profile = threadripper_profile
    optimizer.optimal_settings = settings
    
    if optimizer.save_optimization_profile(profile_file):
        print(f"   ✓ Profile saved to {profile_file}")
        
        # Test loading
        new_optimizer = HardwareOptimizer()
        if new_optimizer.load_optimization_profile(profile_file):
            print(f"   ✓ Profile loaded successfully")
            print(f"   Loaded CPU cores: {new_optimizer.hardware_profile.cpu_cores}")
            print(f"   Loaded thread count: {new_optimizer.optimal_settings.num_threads}")
        else:
            print("   ✗ Failed to load profile")
    else:
        print("   ✗ Failed to save profile")
    
    print("\n=== Demo completed ===")

if __name__ == "__main__":
    main()