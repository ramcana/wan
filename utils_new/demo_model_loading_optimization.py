#!/usr/bin/env python3
"""
Demo script for Model Loading Optimization System

This script demonstrates the comprehensive model loading optimization features
including progress tracking, caching, error handling, fallback options, and
hardware-based recommendations.

Features demonstrated:
- ModelLoadingManager with progress tracking and caching
- ModelFallbackSystem with intelligent fallback options
- Hardware-based model recommendations
- Input validation for image-to-video generation
- Integration with WAN22 system optimization
"""

import time
import json
from pathlib import Path

# Import the model loading optimization components
from model_loading_manager import (
    ModelLoadingManager, ModelLoadingPhase, LoadingParameters
)
from model_fallback_system import (
    ModelFallbackSystem, ModelType, QualityLevel, HardwareProfile
)


def demo_progress_callback(progress):
    """Demo progress callback to show loading progress"""
    bar_length = 30
    filled_length = int(bar_length * progress.progress_percent / 100)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    print(f"\r[{bar}] {progress.progress_percent:5.1f}% | {progress.phase.value:12} | {progress.current_step}", end='')
    
    if progress.phase == ModelLoadingPhase.COMPLETED:
        print()  # New line when completed
    elif progress.phase == ModelLoadingPhase.FAILED:
        print(f"\n❌ Failed: {progress.error_message}")


def demo_model_loading_manager():
    """Demonstrate ModelLoadingManager functionality"""
    print("=" * 60)
    print("🚀 MODEL LOADING MANAGER DEMO")
    print("=" * 60)
    
    # Initialize the manager
    manager = ModelLoadingManager(cache_dir="demo_cache", enable_logging=True)
    manager.add_progress_callback(demo_progress_callback)
    
    print("\n1. Testing Model Loading with Progress Tracking")
    print("-" * 50)
    
    # Simulate model loading (this will fail gracefully without actual models)
    test_models = [
        ("stabilityai/stable-diffusion-2-1", {"torch_dtype": "float16", "device_map": "auto"}),
        ("nonexistent/model", {"torch_dtype": "float16"}),
        ("Wan-AI/Wan2.2-TI2V-5B", {"torch_dtype": "bfloat16", "trust_remote_code": True})
    ]
    
    for model_path, kwargs in test_models:
        print(f"\n📦 Loading model: {model_path}")
        print(f"   Parameters: {kwargs}")
        
        result = manager.load_model(model_path, **kwargs)
        
        if result.success:
            print(f"✅ Success! Loaded in {result.loading_time:.2f}s")
            print(f"   Memory usage: {result.memory_usage_mb:.1f}MB")
            print(f"   Cache hit: {result.cache_hit}")
        else:
            print(f"❌ Failed: {result.error_message}")
            print(f"   Error code: {result.error_code}")
            print("   Suggestions:")
            for suggestion in result.suggestions[:3]:  # Show first 3 suggestions
                print(f"     • {suggestion}")
    
    print("\n2. Loading Statistics")
    print("-" * 50)
    stats = manager.get_loading_statistics()
    print(f"📊 Cached parameters: {stats['total_cached_parameters']}")
    print(f"📊 Cache hit rate: {stats['cache_hit_rate']:.1%}")
    if stats['average_loading_times']:
        print(f"📊 Average loading time: {stats['average_loading_times'].get('overall', 0):.1f}s")


def demo_model_fallback_system():
    """Demonstrate ModelFallbackSystem functionality"""
    print("\n" + "=" * 60)
    print("🔄 MODEL FALLBACK SYSTEM DEMO")
    print("=" * 60)
    
    # Initialize the fallback system
    fallback_system = ModelFallbackSystem()
    
    # Create hardware profiles for testing
    hardware_profiles = {
        "RTX 4080": HardwareProfile(
            gpu_model="RTX 4080",
            vram_gb=16,
            cpu_cores=16,
            ram_gb=32,
            supports_bf16=True,
            supports_int8=True
        ),
        "RTX 3080": HardwareProfile(
            gpu_model="RTX 3080",
            vram_gb=10,
            cpu_cores=8,
            ram_gb=16,
            supports_bf16=False,
            supports_int8=True
        ),
        "RTX 4090": HardwareProfile(
            gpu_model="RTX 4090",
            vram_gb=24,
            cpu_cores=16,
            ram_gb=64,
            supports_bf16=True,
            supports_int8=True,
            supports_fp8=True
        )
    }
    
    print("\n1. Hardware-Based Model Recommendations")
    print("-" * 50)
    
    for gpu_name, hw_profile in hardware_profiles.items():
        print(f"\n🖥️  Hardware: {gpu_name} ({hw_profile.vram_gb}GB VRAM)")
        
        recommendations = fallback_system.recommend_models(
            ModelType.TEXT_TO_VIDEO,
            hw_profile,
            QualityLevel.HIGH
        )
        
        print(f"   Found {len(recommendations)} recommendations:")
        for i, rec in enumerate(recommendations[:3], 1):  # Show top 3
            print(f"   {i}. {rec.model_info.name} (confidence: {rec.confidence_score:.2f})")
            print(f"      Quality: {rec.model_info.quality_level.value}")
            print(f"      VRAM: {rec.model_info.estimated_vram_gb}GB")
            print(f"      Performance: {rec.expected_performance}")
    
    print("\n2. Fallback Options for Failed Models")
    print("-" * 50)
    
    error_scenarios = [
        ("wan22-ti2v-5b", "CUDA_OUT_OF_MEMORY", hardware_profiles["RTX 3080"]),
        ("wan22-ti2v-5b", "TRUST_REMOTE_CODE_ERROR", hardware_profiles["RTX 4080"]),
        ("wan22-i2v-5b", "MODEL_NOT_FOUND", hardware_profiles["RTX 4090"])
    ]
    
    for failed_model, error_type, hw_profile in error_scenarios:
        print(f"\n❌ Scenario: {failed_model} failed with {error_type}")
        print(f"   Hardware: {hw_profile.gpu_model} ({hw_profile.vram_gb}GB VRAM)")
        
        fallbacks = fallback_system.get_fallback_options(failed_model, error_type, hw_profile)
        
        print(f"   Found {len(fallbacks)} fallback options:")
        for i, fallback in enumerate(fallbacks[:3], 1):  # Show top 3
            print(f"   {i}. {fallback.model_info.name}")
            print(f"      Reason: {fallback.reason}")
            print(f"      Quality Impact: {fallback.quality_impact}")
            print(f"      Performance: {fallback.performance_impact}")
    
    print("\n3. Input Validation for Generation")
    print("-" * 50)
    
    validation_tests = [
        ("wan22-ti2v-5b", {"width": 512, "height": 512, "num_frames": 16, "fps": 16}),
        ("wan22-ti2v-5b", {"width": 128, "height": 128, "num_frames": 4}),  # Invalid
        ("wan22-ti2v-5b", {"width": 2048, "height": 2048, "num_frames": 100}),  # Invalid
        ("wan22-i2v-5b", {"width": 768, "height": 768, "num_frames": 24, "image_path": "test.jpg"})
    ]
    
    for model_name, params in validation_tests:
        print(f"\n🔍 Validating: {model_name} with {params}")
        
        result = fallback_system.validate_generation_input(model_name, **params)
        
        if result.is_valid:
            print("   ✅ Validation passed")
        else:
            print("   ❌ Validation failed")
            if result.errors:
                print("   Errors:")
                for error in result.errors:
                    print(f"     • {error}")
            if result.corrected_parameters:
                print("   Suggested corrections:")
                for param, value in result.corrected_parameters.items():
                    print(f"     • {param}: {value}")


def demo_integration_scenario():
    """Demonstrate integration scenario with WAN22 system"""
    print("\n" + "=" * 60)
    print("🔧 INTEGRATION SCENARIO DEMO")
    print("=" * 60)
    
    print("\nScenario: User with RTX 4080 wants to generate a 2-second video")
    print("Original request: TI2V-5B model, 1280x720 resolution, 24 frames")
    
    # Initialize systems
    loading_manager = ModelLoadingManager(cache_dir="demo_cache")
    fallback_system = ModelFallbackSystem()
    
    # Hardware profile
    rtx4080 = HardwareProfile(
        gpu_model="RTX 4080",
        vram_gb=16,
        cpu_cores=16,
        ram_gb=32,
        supports_bf16=True,
        supports_int8=True
    )
    
    print("\n1. Input Validation")
    print("-" * 30)
    
    # Validate input parameters
    validation = fallback_system.validate_generation_input(
        "wan22-ti2v-5b",
        width=1280,
        height=720,
        num_frames=24,
        fps=12
    )
    
    if not validation.is_valid:
        print("❌ Input validation failed:")
        for error in validation.errors:
            print(f"   • {error}")
        
        print("\n🔧 Applying corrections:")
        corrected_width = validation.corrected_parameters.get('width', 1280)
        corrected_height = validation.corrected_parameters.get('height', 720)
        print(f"   • Resolution: {corrected_width}x{corrected_height}")
    else:
        print("✅ Input validation passed")
        corrected_width, corrected_height = 1280, 720
    
    print("\n2. Model Loading Attempt")
    print("-" * 30)
    
    # Simulate model loading failure
    print("🔄 Attempting to load wan22-ti2v-5b...")
    print("❌ Loading failed: CUDA out of memory")
    
    print("\n3. Fallback Options")
    print("-" * 30)
    
    fallbacks = fallback_system.get_fallback_options(
        "wan22-ti2v-5b",
        "CUDA_OUT_OF_MEMORY",
        rtx4080
    )
    
    print(f"Found {len(fallbacks)} fallback options:")
    for i, fallback in enumerate(fallbacks[:3], 1):
        print(f"{i}. {fallback.model_info.name}")
        print(f"   Quality Impact: {fallback.quality_impact}")
        print(f"   VRAM Required: {fallback.model_info.estimated_vram_gb}GB")
        print(f"   Notes: {fallback.compatibility_notes}")
    
    print("\n4. Recommendation")
    print("-" * 30)
    
    if fallbacks:
        recommended = fallbacks[0]
        print(f"🎯 Recommended: {recommended.model_info.name}")
        print(f"   Reason: {recommended.reason}")
        print(f"   Expected Performance: {recommended.performance_impact}")
        
        # Show optimization suggestions
        recommendations = fallback_system.recommend_models(
            ModelType.TEXT_TO_VIDEO,
            rtx4080,
            QualityLevel.HIGH
        )
        
        if recommendations:
            top_rec = recommendations[0]
            print(f"\n💡 Optimization suggestions:")
            for suggestion in top_rec.optimization_suggestions:
                print(f"   • {suggestion}")


def demo_caching_performance():
    """Demonstrate caching performance benefits"""
    print("\n" + "=" * 60)
    print("⚡ CACHING PERFORMANCE DEMO")
    print("=" * 60)
    
    manager = ModelLoadingManager(cache_dir="demo_cache")
    
    # Simulate multiple loading attempts with same parameters
    params = LoadingParameters(
        model_path="test/model",
        torch_dtype="float16",
        device_map="auto",
        trust_remote_code=True
    )
    
    print("\n1. First Load (Cache Miss)")
    print("-" * 30)
    cache_key = params.get_cache_key()
    print(f"Cache key: {cache_key}")
    
    # Simulate caching successful parameters
    manager._cache_parameters(cache_key, params, 180.5, 8192.0)
    print("✅ Parameters cached after successful load")
    print("   Loading time: 180.5s")
    print("   Memory usage: 8192.0MB")
    
    print("\n2. Subsequent Load (Cache Hit)")
    print("-" * 30)
    cached = manager._get_cached_parameters(cache_key)
    if cached:
        print("✅ Cache hit! Using cached parameters:")
        print(f"   Previous loading time: {cached['loading_time']}s")
        print(f"   Previous memory usage: {cached['memory_usage_mb']}MB")
        print(f"   Use count: {cached['use_count']}")
        print(f"   Last used: {cached['last_used']}")
    
    print("\n3. Cache Statistics")
    print("-" * 30)
    stats = manager.get_loading_statistics()
    print(f"📊 Total cached parameters: {stats['total_cached_parameters']}")
    if stats['memory_usage_stats']:
        mem_stats = stats['memory_usage_stats']
        print(f"📊 Average memory usage: {mem_stats['average_mb']:.1f}MB")
        print(f"📊 Max memory usage: {mem_stats['max_mb']:.1f}MB")


def main():
    """Main demo function"""
    print("🎬 WAN22 MODEL LOADING OPTIMIZATION DEMO")
    print("This demo showcases the comprehensive model loading optimization system")
    print("including progress tracking, caching, fallback options, and recommendations.")
    
    try:
        # Run all demo sections
        demo_model_loading_manager()
        demo_model_fallback_system()
        demo_integration_scenario()
        demo_caching_performance()
        
        print("\n" + "=" * 60)
        print("✅ DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nKey Features Demonstrated:")
        print("• Detailed progress tracking with time estimates")
        print("• Intelligent error handling with specific suggestions")
        print("• Parameter caching for faster subsequent loads")
        print("• Hardware-based model recommendations")
        print("• Intelligent fallback options for failed loads")
        print("• Comprehensive input validation")
        print("• Integration with WAN22 system optimization")
        
        print("\nNext Steps:")
        print("• Integrate with main WAN22 application")
        print("• Test with real hardware and models")
        print("• Fine-tune recommendations based on user feedback")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
