"""
Integration test for WAN Model Progress Tracking with RealGenerationPipeline

This test verifies that the enhanced WAN model progress tracking integrates
correctly with the existing RealGenerationPipeline infrastructure.

Requirements tested: 5.1, 5.2, 5.3, 5.4
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_real_generation_pipeline_with_wan_progress():
    """Test RealGenerationPipeline with WAN model progress tracking"""
    try:
        # Import the pipeline
        sys.path.append('backend/services')
        from backend.services.real_generation_pipeline import RealGenerationPipeline, GenerationParams
        
        # Create mock WAN model with progress tracking
        mock_wan_model = MagicMock()
        mock_wan_model.generate_video = AsyncMock()
        
        # Mock generation result
        from core.models.wan_models.wan_base_model import WANGenerationResult
        mock_result = WANGenerationResult(
            success=True,
            frames=[],
            generation_time=30.5,
            peak_memory_mb=8500.0,
            memory_used_mb=7200.0,
            applied_optimizations=['fp16', 'attention_slicing'],
            model_info={'model_type': 't2v-A14B', 'parameter_count': 14_000_000_000}
        )
        mock_wan_model.generate_video.return_value = mock_result
        
        # Create mock pipeline wrapper
        mock_pipeline_wrapper = MagicMock()
        mock_pipeline_wrapper.model = mock_wan_model
        
        # Create pipeline instance
        pipeline = RealGenerationPipeline()
        
        # Mock the pipeline loading to return our mock wrapper
        pipeline._load_pipeline = AsyncMock(return_value=mock_pipeline_wrapper)
        
        # Create generation parameters
        params = GenerationParams(
            prompt="A cat playing in a garden",
            model_type="t2v-A14B",
            num_frames=16,
            resolution="1280x720",
            steps=20,
            guidance_scale=7.5,
            seed=42
        )
        
        # Test T2V generation with progress tracking
        result = await pipeline.generate_t2v("A cat playing in a garden", params)
        
        # Verify the result
        assert result.success == True
        assert result.generation_time_seconds == 30.5
        assert result.peak_vram_usage_mb == 8500.0
        assert result.model_used == "t2v-A14B"
        assert 'fp16' in result.optimizations_applied
        
        # Verify WAN model was called with correct parameters
        mock_wan_model.generate_video.assert_called_once()
        call_args = mock_wan_model.generate_video.call_args[1]
        assert call_args['prompt'] == "A cat playing in a garden"
        assert call_args['num_frames'] == 16
        assert call_args['width'] == 1280
        assert call_args['height'] == 720
        assert call_args['num_inference_steps'] == 20
        assert 'task_id' in call_args
        
        logger.info("✓ RealGenerationPipeline with WAN progress tracking test passed")
        return True
        
    except Exception as e:
        logger.error(f"Integration test error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_progress_callback_integration():
    """Test progress callback integration between pipeline and WAN models"""
    try:
        from core.models.wan_models.wan_progress_tracker import create_wan_progress_callback, WANProgressTracker
        
        # Create progress tracker
        tracker = WANProgressTracker("t2v-A14B")
        
        # Mock the progress update method
        tracker.update_denoising_step_progress = AsyncMock()
        
        # Create progress callback
        callback = create_wan_progress_callback(tracker)
        
        # Test callback with different parameters
        await callback(5, 800, None, total_steps=20)
        await callback(10, 600, None, total_steps=20)
        await callback(15, 400, None, total_steps=20)
        
        # Verify callback was called correctly
        assert tracker.update_denoising_step_progress.call_count == 3
        
        # Check call arguments
        calls = tracker.update_denoising_step_progress.call_args_list
        assert calls[0][0] == (5, 20, 800)
        assert calls[1][0] == (10, 20, 600)
        assert calls[2][0] == (15, 20, 400)
        
        logger.info("✓ Progress callback integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"Progress callback integration test error: {e}")
        return False

async def test_websocket_progress_integration():
    """Test WebSocket progress integration with WAN models"""
    try:
        from core.models.wan_models.wan_progress_tracker import WANProgressTracker, WANInferenceStage
        
        # Create progress tracker
        tracker = WANProgressTracker("t2v-A14B")
        
        # Mock WebSocket integration
        mock_websocket = AsyncMock()
        tracker._websocket_integration = mock_websocket
        
        # Test progress updates
        await tracker.update_stage_progress(
            WANInferenceStage.TEXT_ENCODING, 25, "Encoding text prompts"
        )
        
        await tracker.update_denoising_step_progress(10, 50, 500)
        
        await tracker.update_attention_progress("temporal", 5, 14)
        
        # Verify WebSocket integration was called
        assert mock_websocket.update_stage_progress.call_count >= 3
        
        logger.info("✓ WebSocket progress integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"WebSocket progress integration test error: {e}")
        return False

def test_performance_profile_accuracy():
    """Test performance profile accuracy for different model types"""
    try:
        from core.models.wan_models.wan_progress_tracker import WANProgressTracker
        
        # Test T2V performance profile
        t2v_tracker = WANProgressTracker("t2v-A14B")
        t2v_profile = t2v_tracker.performance_profile
        
        assert t2v_profile.model_type == "t2v-A14B"
        assert t2v_profile.parameter_count == 14_000_000_000
        assert t2v_profile.base_vram_usage_gb == 10.5
        assert t2v_profile.denoising_step_time == 1.2
        
        # Test I2V performance profile
        i2v_tracker = WANProgressTracker("i2v-A14B")
        i2v_profile = i2v_tracker.performance_profile
        
        assert i2v_profile.model_type == "i2v-A14B"
        assert i2v_profile.parameter_count == 14_000_000_000
        assert i2v_profile.base_vram_usage_gb == 11.0  # Higher due to image conditioning
        assert i2v_profile.denoising_step_time == 1.4  # Slower due to image processing
        
        # Test TI2V performance profile
        ti2v_tracker = WANProgressTracker("ti2v-5B")
        ti2v_profile = ti2v_tracker.performance_profile
        
        assert ti2v_profile.model_type == "ti2v-5B"
        assert ti2v_profile.parameter_count == 5_000_000_000  # Smaller model
        assert ti2v_profile.base_vram_usage_gb == 6.5  # Lower VRAM usage
        assert ti2v_profile.denoising_step_time == 0.6  # Faster inference
        
        # Verify performance relationships
        assert i2v_profile.denoising_step_time > t2v_profile.denoising_step_time  # I2V slower than T2V
        assert ti2v_profile.denoising_step_time < t2v_profile.denoising_step_time  # TI2V faster than T2V
        assert ti2v_profile.base_vram_usage_gb < t2v_profile.base_vram_usage_gb  # TI2V uses less VRAM
        
        logger.info("✓ Performance profile accuracy test passed")
        return True
        
    except Exception as e:
        logger.error(f"Performance profile accuracy test error: {e}")
        return False

async def run_integration_tests():
    """Run all WAN progress tracking integration tests"""
    logger.info("Running WAN Model Progress Tracking Integration Tests...")
    
    success = True
    
    # Test RealGenerationPipeline integration
    if not await test_real_generation_pipeline_with_wan_progress():
        success = False
    
    # Test progress callback integration
    if not await test_progress_callback_integration():
        success = False
    
    # Test WebSocket integration
    if not await test_websocket_progress_integration():
        success = False
    
    # Test performance profiles
    if not test_performance_profile_accuracy():
        success = False
    
    if success:
        logger.info("✅ All WAN progress tracking integration tests passed!")
    else:
        logger.error("❌ Some WAN progress tracking integration tests failed!")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(run_integration_tests())
    exit(0 if success else 1)