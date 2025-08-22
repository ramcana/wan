"""
Demo script showing the OptimizationManager and resource management system in action.

This demonstrates the key features implemented for task 6:
- System resource analysis
- Memory optimization strategies (mixed precision, CPU offload)
- Chunked processing capabilities for memory-constrained systems
- Optimization recommendation engine based on available VRAM
"""

from optimization_manager import (
    OptimizationManager, ChunkedProcessor, SystemResources, ModelRequirements,
    get_gpu_memory_info, estimate_model_vram_usage
)
import json


def demo_system_analysis():
    """Demo system resource analysis"""
    print("=" * 60)
    print("SYSTEM RESOURCE ANALYSIS")
    print("=" * 60)
    
    manager = OptimizationManager()
    resources = manager.analyze_system_resources()
    
    print(f"GPU: {resources.gpu_name}")
    print(f"Total VRAM: {resources.total_vram_mb:,}MB")
    print(f"Available VRAM: {resources.available_vram_mb:,}MB")
    print(f"Total RAM: {resources.total_ram_mb:,}MB")
    print(f"Available RAM: {resources.available_ram_mb:,}MB")
    print(f"CPU Cores: {resources.cpu_cores}")
    print(f"Compute Capability: {resources.gpu_compute_capability}")
    print(f"Mixed Precision Support: {resources.supports_mixed_precision}")
    print(f"CPU Offload Support: {resources.supports_cpu_offload}")
    
    return resources


def demo_optimization_scenarios(system_resources):
    """Demo optimization recommendations for different scenarios"""
    print("\n" + "=" * 60)
    print("OPTIMIZATION SCENARIOS")
    print("=" * 60)
    
    manager = OptimizationManager()
    
    # Define different model sizes
    models = {
        "Small Model (6GB)": ModelRequirements(
            min_vram_mb=4096, recommended_vram_mb=6144, model_size_mb=4096,
            supports_mixed_precision=True, supports_cpu_offload=True,
            supports_chunked_processing=True, component_sizes={}
        ),
        "Medium Model (12GB)": ModelRequirements(
            min_vram_mb=8192, recommended_vram_mb=12288, model_size_mb=8192,
            supports_mixed_precision=True, supports_cpu_offload=True,
            supports_chunked_processing=True, component_sizes={}
        ),
        "Large Model (20GB)": ModelRequirements(
            min_vram_mb=16384, recommended_vram_mb=20480, model_size_mb=16384,
            supports_mixed_precision=True, supports_cpu_offload=True,
            supports_chunked_processing=True, component_sizes={}
        )
    }
    
    for model_name, model_req in models.items():
        print(f"\n--- {model_name} ---")
        plan = manager.recommend_optimizations(model_req, system_resources)
        
        print(f"Recommended VRAM: {model_req.recommended_vram_mb:,}MB")
        print(f"Mixed Precision: {plan.use_mixed_precision} ({plan.precision_type})")
        print(f"CPU Offload: {plan.enable_cpu_offload} ({plan.offload_strategy})")
        print(f"Chunked Processing: {plan.chunk_frames} (max {plan.max_chunk_size} frames)")
        print(f"VRAM Reduction: {plan.estimated_vram_reduction:.1%}")
        print(f"Performance Impact: {plan.estimated_performance_impact:+.1%}")
        
        if plan.optimization_steps:
            print("Optimization Steps:")
            for step in plan.optimization_steps:
                print(f"  • {step}")
        
        if plan.warnings:
            print("Warnings:")
            for warning in plan.warnings:
                print(f"  ⚠️ {warning}")


def demo_chunked_processing():
    """Demo chunked processing capabilities"""
    print("\n" + "=" * 60)
    print("CHUNKED PROCESSING DEMO")
    print("=" * 60)
    
    processor = ChunkedProcessor(chunk_size=4, overlap_frames=1)
    
    test_cases = [8, 16, 32, 100]
    
    for num_frames in test_cases:
        chunks = processor._calculate_chunks(num_frames)
        print(f"\n{num_frames} frames → {len(chunks)} chunks:")
        for i, (start, end) in enumerate(chunks):
            print(f"  Chunk {i+1}: frames {start}-{end-1} ({end-start} frames)")


def demo_memory_estimation():
    """Demo memory usage estimation"""
    print("\n" + "=" * 60)
    print("MEMORY USAGE ESTIMATION")
    print("=" * 60)
    
    manager = OptimizationManager()
    
    # Test different generation parameters
    test_configs = [
        {"width": 512, "height": 512, "num_frames": 16, "batch_size": 1},
        {"width": 1024, "height": 1024, "num_frames": 16, "batch_size": 1},
        {"width": 512, "height": 512, "num_frames": 32, "batch_size": 1},
        {"width": 512, "height": 512, "num_frames": 16, "batch_size": 2},
    ]
    
    model_req = ModelRequirements(
        min_vram_mb=8192, recommended_vram_mb=12288, model_size_mb=8192,
        supports_mixed_precision=True, supports_cpu_offload=True,
        supports_chunked_processing=True, component_sizes={}
    )
    
    for config in test_configs:
        estimate = manager.estimate_memory_usage(model_req, config)
        print(f"\nConfig: {config['width']}x{config['height']}, {config['num_frames']} frames, batch {config['batch_size']}")
        print(f"  Base model: {estimate['base_model_mb']:,}MB")
        print(f"  Intermediate tensors: {estimate['intermediate_tensors_mb']:,}MB")
        print(f"  Output tensors: {estimate['output_tensors_mb']:,}MB")
        print(f"  Overhead: {estimate['overhead_mb']:,}MB")
        print(f"  Total estimated: {estimate['total_estimated_mb']:,}MB")


def demo_optimization_recommendations():
    """Demo optimization recommendations"""
    print("\n" + "=" * 60)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 60)
    
    manager = OptimizationManager()
    
    generation_params = {
        "width": 1024,
        "height": 1024,
        "num_frames": 32,
        "batch_size": 1
    }
    
    recommendations = manager.get_optimization_recommendations(
        "/path/to/model", generation_params
    )
    
    print("Generation Parameters:")
    print(f"  Resolution: {generation_params['width']}x{generation_params['height']}")
    print(f"  Frames: {generation_params['num_frames']}")
    print(f"  Batch Size: {generation_params['batch_size']}")
    
    print("\nRecommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")


def demo_utility_functions():
    """Demo utility functions"""
    print("\n" + "=" * 60)
    print("UTILITY FUNCTIONS")
    print("=" * 60)
    
    # GPU memory info
    gpu_info = get_gpu_memory_info()
    print("Current GPU Memory:")
    print(f"  Total: {gpu_info['total']:,}MB")
    print(f"  Allocated: {gpu_info['allocated']:,}MB")
    print(f"  Reserved: {gpu_info['reserved']:,}MB")
    print(f"  Free: {gpu_info['free']:,}MB")
    
    # VRAM usage estimation
    print("\nVRAM Usage Estimation:")
    model_sizes = [4096, 8192, 16384]  # MB
    precisions = ["fp32", "fp16", "bf16"]
    
    for model_size in model_sizes:
        print(f"\n  {model_size}MB model:")
        for precision in precisions:
            estimated = estimate_model_vram_usage(model_size, precision, batch_size=1)
            print(f"    {precision}: {estimated:,}MB")


def main():
    """Run all demos"""
    print("🚀 OPTIMIZATION MANAGER DEMO")
    print("Demonstrating comprehensive optimization and resource management system")
    
    try:
        # System analysis
        resources = demo_system_analysis()
        
        # Optimization scenarios
        demo_optimization_scenarios(resources)
        
        # Chunked processing
        demo_chunked_processing()
        
        # Memory estimation
        demo_memory_estimation()
        
        # Optimization recommendations
        demo_optimization_recommendations()
        
        # Utility functions
        demo_utility_functions()
        
        print("\n" + "=" * 60)
        print("✅ DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nKey Features Demonstrated:")
        print("• System resource analysis and detection")
        print("• Automatic optimization recommendations")
        print("• Mixed precision and CPU offload strategies")
        print("• Chunked processing for memory-constrained systems")
        print("• Memory usage estimation and monitoring")
        print("• Comprehensive utility functions")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()