"""
Example usage of the Real Generation Pipeline
Demonstrates T2V, I2V, and TI2V generation with progress tracking
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.services.real_generation_pipeline import (
    RealGenerationPipeline, get_real_generation_pipeline, ProgressUpdate
)
from backend.core.model_integration_bridge import GenerationParams

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def progress_callback(update: ProgressUpdate):
    """Example progress callback"""
    print(f"[{update.stage.value.upper()}] {update.progress_percent}% - {update.message}")
    if update.current_step and update.total_steps:
        print(f"  Step {update.current_step}/{update.total_steps}")

async def example_t2v_generation():
    """Example T2V generation"""
    print("\n=== Text-to-Video Generation Example ===")
    
    # Get the pipeline instance
    pipeline = await get_real_generation_pipeline()
    
    # Create generation parameters
    params = GenerationParams(
        prompt="A majestic eagle soaring over snow-capped mountains at sunset",
        model_type="t2v-A14B",
        resolution="1280x720",
        steps=25,
        num_frames=16,
        fps=8.0,
        guidance_scale=7.5,
        negative_prompt="blurry, low quality, distorted"
    )
    
    # Generate video with progress tracking
    result = await pipeline.generate_t2v(
        params.prompt, params, progress_callback=progress_callback
    )
    
    if result.success:
        print(f"✅ T2V Generation successful!")
        print(f"   Output: {result.output_path}")
        print(f"   Time: {result.generation_time_seconds:.1f}s")
        print(f"   Peak VRAM: {result.peak_vram_usage_mb:.0f}MB")
        print(f"   Optimizations: {', '.join(result.optimizations_applied)}")
    else:
        print(f"❌ T2V Generation failed: {result.error_message}")
        if result.recovery_suggestions:
            print("   Suggestions:")
            for suggestion in result.recovery_suggestions:
                print(f"   - {suggestion}")

async def example_i2v_generation():
    """Example I2V generation"""
    print("\n=== Image-to-Video Generation Example ===")
    
    # For this example, we'll create a placeholder image path
    # In real usage, this would be a path to an actual image file
    image_path = "examples/sample_image.jpg"
    
    pipeline = await get_real_generation_pipeline()
    
    params = GenerationParams(
        prompt="Animate this landscape with gentle wind and moving clouds",
        model_type="i2v-A14B",
        resolution="1280x720",
        steps=20,
        num_frames=24,
        fps=12.0,
        guidance_scale=6.0
    )
    
    result = await pipeline.generate_i2v(
        image_path, params.prompt, params, progress_callback=progress_callback
    )
    
    if result.success:
        print(f"✅ I2V Generation successful!")
        print(f"   Output: {result.output_path}")
        print(f"   Time: {result.generation_time_seconds:.1f}s")
    else:
        print(f"❌ I2V Generation failed: {result.error_message}")
        print(f"   Category: {result.error_category}")

async def example_ti2v_generation():
    """Example TI2V generation"""
    print("\n=== Text+Image-to-Video Generation Example ===")
    
    image_path = "examples/start_image.jpg"
    
    pipeline = await get_real_generation_pipeline()
    
    params = GenerationParams(
        prompt="Transform this scene into a magical fantasy landscape with sparkles and ethereal lighting",
        model_type="ti2v-5B",
        resolution="1024x576",
        steps=30,
        num_frames=20,
        fps=10.0,
        guidance_scale=8.0,
        end_image_path="examples/end_image.jpg"  # Optional end frame
    )
    
    result = await pipeline.generate_ti2v(
        image_path, params.prompt, params, progress_callback=progress_callback
    )
    
    if result.success:
        print(f"✅ TI2V Generation successful!")
        print(f"   Output: {result.output_path}")
        print(f"   Time: {result.generation_time_seconds:.1f}s")
    else:
        print(f"❌ TI2V Generation failed: {result.error_message}")

async def example_pipeline_stats():
    """Example of getting pipeline statistics"""
    print("\n=== Pipeline Statistics ===")
    
    pipeline = await get_real_generation_pipeline()
    stats = pipeline.get_generation_stats()
    
    print(f"Total generations: {stats['total_generations']}")
    print(f"Total time: {stats['total_generation_time']:.1f}s")
    print(f"Average time: {stats['average_generation_time']:.1f}s")
    print(f"Cached pipelines: {stats['cached_pipelines']}")
    print(f"WAN Pipeline Loader available: {stats['wan_pipeline_loader_available']}")
    print(f"WebSocket manager available: {stats['websocket_manager_available']}")

async def example_optimization_config():
    """Example of using optimization parameters"""
    print("\n=== Optimization Configuration Example ===")
    
    pipeline = await get_real_generation_pipeline()
    
    # Parameters with optimization settings
    params = GenerationParams(
        prompt="A serene lake reflecting the aurora borealis",
        model_type="t2v-A14B",
        resolution="1920x1080",  # High resolution
        steps=50,  # High quality
        num_frames=32,  # Long video
        fps=24.0,
        # Optimization settings
        quantization_level="fp16",  # Use half precision
        enable_offload=True,  # Enable CPU offloading
        max_vram_usage_gb=6.0,  # Limit VRAM usage
        vae_tile_size=512  # Larger tiles for better quality
    )
    
    print(f"Generating with optimizations:")
    print(f"  Resolution: {params.resolution}")
    print(f"  Frames: {params.num_frames}")
    print(f"  Quantization: {params.quantization_level}")
    print(f"  Max VRAM: {params.max_vram_usage_gb}GB")
    
    result = await pipeline.generate_t2v(
        params.prompt, params, progress_callback=progress_callback
    )
    
    if result.success:
        print(f"✅ Optimized generation successful!")
        print(f"   Applied optimizations: {', '.join(result.optimizations_applied)}")
    else:
        print(f"❌ Optimized generation failed: {result.error_message}")

async def main():
    """Run all examples"""
    print("Real Generation Pipeline Examples")
    print("=" * 50)
    
    try:
        # Initialize pipeline
        pipeline = await get_real_generation_pipeline()
        print(f"Pipeline initialized: {pipeline.get_generation_stats()}")
        
        # Run examples (these will use mock generation since real models may not be available)
        await example_t2v_generation()
        await example_i2v_generation()
        await example_ti2v_generation()
        await example_optimization_config()
        await example_pipeline_stats()
        
        print("\n=== Examples completed ===")
        print("Note: These examples use mock generation since real models may not be available.")
        print("In a real deployment with models downloaded, actual video generation would occur.")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
