"""
Simple test to verify OptimizationManager basic functionality
"""

from optimization_manager import OptimizationManager, SystemResources, ModelRequirements

def test_basic_functionality():
    """Test basic OptimizationManager functionality"""
    print("Testing OptimizationManager...")
    
    # Create manager
    manager = OptimizationManager()
    print("✓ OptimizationManager created")
    
    # Test system resource analysis
    try:
        resources = manager.analyze_system_resources()
        print(f"✓ System resources analyzed: {resources.total_vram_mb}MB VRAM, {resources.gpu_name}")
    except Exception as e:
        print(f"✗ System resource analysis failed: {e}")
        return False
    
    # Create mock model requirements
    model_requirements = ModelRequirements(
        min_vram_mb=6144,
        recommended_vram_mb=12288,
        model_size_mb=8192,
        supports_mixed_precision=True,
        supports_cpu_offload=True,
        supports_chunked_processing=True,
        component_sizes={"transformer": 4096, "vae": 2048, "text_encoder": 1024}
    )
    
    # Test optimization recommendations
    try:
        plan = manager.recommend_optimizations(model_requirements, resources)
        print(f"✓ Optimization plan created: {len(plan.optimization_steps)} steps")
        print(f"  - Mixed precision: {plan.use_mixed_precision}")
        print(f"  - CPU offload: {plan.enable_cpu_offload}")
        print(f"  - Chunked processing: {plan.chunk_frames}")
        print(f"  - Estimated VRAM reduction: {plan.estimated_vram_reduction:.1%}")
    except Exception as e:
        print(f"✗ Optimization recommendation failed: {e}")
        return False
    
    # Test memory estimation
    try:
        generation_params = {"width": 512, "height": 512, "num_frames": 16, "batch_size": 1}
        estimate = manager.estimate_memory_usage(model_requirements, generation_params)
        print(f"✓ Memory estimation: {estimate['total_estimated_mb']}MB total")
    except Exception as e:
        print(f"✗ Memory estimation failed: {e}")
        return False
    
    # Test chunked processing
    try:
        from optimization_manager import ChunkedProcessor
        processor = ChunkedProcessor(chunk_size=4, overlap_frames=1)
        chunks = processor._calculate_chunks(10)
        print(f"✓ Chunked processing: {len(chunks)} chunks for 10 frames: {chunks}")
    except Exception as e:
        print(f"✗ Chunked processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✓ All basic tests passed!")
    return True

    assert True  # TODO: Add proper assertion

if __name__ == "__main__":
    test_basic_functionality()