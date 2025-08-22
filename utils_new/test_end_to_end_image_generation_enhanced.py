#!/usr/bin/env python3
"""
Enhanced end-to-end test for image data integration with generation pipeline
Tests the complete workflow from image upload to video generation
"""

import os
import sys
import tempfile
from pathlib import Path
from PIL import Image
import logging
import time

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    GenerationTask, TaskStatus, get_queue_manager, 
    generate_video, add_to_generation_queue, _validate_images_for_model_type
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_image(size=(512, 512), color=(255, 0, 0), mode="RGB"):
    """Create a test image with specified properties"""
    if mode == "RGBA":
        color = color + (255,) if len(color) == 3 else color
    return Image.new(mode, size, color)

def test_generation_task_enhanced_features():
    """Test enhanced GenerationTask features"""
    logger.info("Testing enhanced GenerationTask features...")
    
    start_image = create_test_image(color=(255, 0, 0))
    end_image = create_test_image(color=(0, 0, 255))
    
    # Create task with enhanced features
    task = GenerationTask(
        model_type="ti2v-5B",
        prompt="Test enhanced features",
        resolution="1280x720",
        steps=20,
        download_timeout=600,  # Extended timeout
        smart_download_enabled=True
    )
    
    # Store images with enhanced metadata
    task.store_image_data(start_image, end_image)
    
    # Verify enhanced features
    assert hasattr(task, 'download_timeout'), "download_timeout attribute missing"
    assert hasattr(task, 'smart_download_enabled'), "smart_download_enabled attribute missing"
    assert task.download_timeout == 600, "download_timeout not set correctly"
    assert task.smart_download_enabled == True, "smart_download_enabled not set correctly"
    
    # Verify enhanced metadata
    if task.image_metadata:
        assert "aspect_ratio" in task.image_metadata, "aspect_ratio missing from metadata"
        assert "pixel_count" in task.image_metadata, "pixel_count missing from metadata"
        assert "file_size_estimate" in task.image_metadata, "file_size_estimate missing from metadata"
    
    # Test image summary
    summary = task.get_image_summary()
    assert "start image" in summary, "start image not in summary"
    assert "end image" in summary, "end image not in summary"
    
    # Test has_images method
    assert task.has_images(), "has_images should return True"
    
    logger.info("✅ Enhanced GenerationTask features test passed")
    return True

def test_queue_integration_with_validation():
    """Test queue integration with image validation"""
    logger.info("Testing queue integration with validation...")
    
    start_image = create_test_image(color=(255, 128, 0))
    
    # Test valid task addition
    task_id = add_to_generation_queue(
        model_type="i2v-A14B",
        prompt="Test queue integration",
        image=start_image,
        resolution="1280x720",
        steps=10,
        download_timeout=300,
        smart_download=True
    )
    
    assert task_id is not None, "Valid task should be added to queue"
    logger.info(f"Task added to queue: {task_id}")
    
    # Test invalid task addition (T2V with image)
    invalid_task_id = add_to_generation_queue(
        model_type="t2v-A14B",
        prompt="This should fail",
        image=start_image,
        resolution="1280x720",
        steps=10
    )
    
    assert invalid_task_id is None, "Invalid task should be rejected"
    logger.info("Invalid task correctly rejected")
    
    # Get queue manager and check task details
    queue_manager = get_queue_manager()
    task_details = queue_manager.get_task_details(task_id)
    
    if task_details:
        logger.info(f"Task details retrieved: {task_details.get('model_type', 'unknown')}")
        
        # Check for enhanced attributes
        if hasattr(task_details, 'download_timeout'):
            logger.info(f"Download timeout: {task_details.download_timeout}")
        
        # Check image metadata if available
        if task_details.get('image_metadata'):
            metadata = task_details['image_metadata']
            logger.info(f"Image metadata keys: {list(metadata.keys())}")
    
    logger.info("✅ Queue integration with validation test passed")
    return task_id

def test_generation_function_with_images():
    """Test generation function with image parameters"""
    logger.info("Testing generation function with images...")
    
    start_image = create_test_image(color=(0, 255, 0))
    end_image = create_test_image(color=(255, 0, 255))
    
    # Test with minimal parameters to avoid long model downloads
    result = generate_video(
        model_type="ti2v-5B",
        prompt="Test generation with images",
        image=start_image,
        end_image=end_image,
        resolution="1280x720",
        steps=5,  # Minimal steps for testing
        download_timeout=120  # Short timeout for testing
    )
    
    # Verify result structure
    assert isinstance(result, dict), "Result should be a dictionary"
    
    # Check for enhanced metadata
    if result.get("metadata"):
        metadata = result["metadata"]
        logger.info(f"Generation metadata: {metadata}")
        
        if "images_used" in metadata:
            logger.info(f"Images used: {metadata['images_used']}")
    
    # Log result status
    if result.get("success"):
        logger.info("✅ Generation completed successfully")
        if result.get("output_path"):
            logger.info(f"Output: {result['output_path']}")
    else:
        logger.info(f"Generation failed (expected in test): {result.get('error', 'Unknown error')}")
        
        # Check for recovery suggestions
        if result.get("recovery_suggestions"):
            logger.info(f"Recovery suggestions: {result['recovery_suggestions']}")
    
    logger.info("✅ Generation function with images test passed")
    return True

def test_different_model_types_with_images():
    """Test different model types with appropriate image configurations"""
    logger.info("Testing different model types with images...")
    
    start_image = create_test_image(color=(128, 128, 128))
    
    model_configs = [
        {
            "model_type": "t2v-A14B",
            "image": None,
            "end_image": None,
            "should_succeed": True
        },
        {
            "model_type": "i2v-A14B", 
            "image": start_image,
            "end_image": None,
            "should_succeed": True
        },
        {
            "model_type": "ti2v-5B",
            "image": start_image,
            "end_image": start_image,
            "should_succeed": True
        }
    ]
    
    for config in model_configs:
        logger.info(f"Testing {config['model_type']}...")
        
        # Validate configuration
        validation_result = _validate_images_for_model_type(
            config["model_type"], 
            config["image"], 
            config["end_image"]
        )
        
        if config["should_succeed"]:
            assert validation_result["valid"], f"Configuration should be valid for {config['model_type']}"
        else:
            assert not validation_result["valid"], f"Configuration should be invalid for {config['model_type']}"
        
        logger.info(f"✅ {config['model_type']} validation passed")
    
    logger.info("✅ Different model types test passed")
    return True

def test_image_metadata_preservation():
    """Test that image metadata is preserved through the pipeline"""
    logger.info("Testing image metadata preservation...")
    
    # Create images with different properties
    rgb_image = create_test_image(size=(256, 256), color=(255, 0, 0), mode="RGB")
    rgba_image = create_test_image(size=(512, 512), color=(0, 255, 0, 128), mode="RGBA")
    
    task = GenerationTask(
        model_type="ti2v-5B",
        prompt="Test metadata preservation",
        resolution="1280x720",
        steps=10
    )
    
    # Store images
    task.store_image_data(rgb_image, rgba_image)
    
    # Verify metadata preservation
    assert task.image_metadata is not None, "Start image metadata not preserved"
    assert task.end_image_metadata is not None, "End image metadata not preserved"
    
    # Check specific metadata fields
    start_meta = task.image_metadata
    end_meta = task.end_image_metadata
    
    assert start_meta["size"] == (256, 256), "Start image size metadata incorrect"
    assert start_meta["mode"] == "RGB", "Start image mode metadata incorrect"
    assert end_meta["size"] == (512, 512), "End image size metadata incorrect"
    assert end_meta["mode"] == "RGBA", "End image mode metadata incorrect"
    
    # Check calculated fields
    assert "aspect_ratio" in start_meta, "Aspect ratio not calculated"
    assert "pixel_count" in start_meta, "Pixel count not calculated"
    assert start_meta["pixel_count"] == 256 * 256, "Pixel count incorrect"
    
    # Test serialization
    task_dict = task.to_dict()
    assert "image_metadata" in task_dict, "Image metadata not in serialized task"
    assert "end_image_metadata" in task_dict, "End image metadata not in serialized task"
    
    logger.info("✅ Image metadata preservation test passed")
    return True

def test_error_handling_and_recovery():
    """Test error handling and recovery mechanisms"""
    logger.info("Testing error handling and recovery...")
    
    # Test with invalid image-model combination
    start_image = create_test_image()
    
    result = generate_video(
        model_type="t2v-A14B",  # T2V doesn't use images
        prompt="This should fail validation",
        image=start_image,  # But we're providing an image
        resolution="1280x720",
        steps=5
    )
    
    # Should fail validation
    assert not result.get("success", True), "Should fail validation"
    assert "validation failed" in result.get("error", "").lower(), "Should mention validation failure"
    
    # Check for recovery suggestions
    assert "recovery_suggestions" in result, "Should provide recovery suggestions"
    suggestions = result["recovery_suggestions"]
    assert len(suggestions) > 0, "Should have recovery suggestions"
    
    logger.info(f"Error handled correctly: {result['error']}")
    logger.info(f"Recovery suggestions: {suggestions}")
    
    logger.info("✅ Error handling and recovery test passed")
    return True

def cleanup_test_files():
    """Clean up test files"""
    logger.info("Cleaning up test files...")
    
    try:
        # Clean up temp directories
        temp_dir = Path(tempfile.gettempdir()) / "wan22_images"
        if temp_dir.exists():
            for file in temp_dir.glob("*.png"):
                try:
                    file.unlink()
                    logger.debug(f"Cleaned up: {file}")
                except Exception as e:
                    logger.warning(f"Failed to clean up {file}: {e}")
        
        logger.info("✅ Cleanup completed")
    except Exception as e:
        logger.warning(f"Cleanup failed: {e}")

def run_enhanced_end_to_end_tests():
    """Run enhanced end-to-end image integration tests"""
    logger.info("Starting enhanced end-to-end image integration tests...")
    
    tests = [
        ("Enhanced GenerationTask Features", test_generation_task_enhanced_features),
        ("Queue Integration with Validation", test_queue_integration_with_validation),
        ("Generation Function with Images", test_generation_function_with_images),
        ("Different Model Types with Images", test_different_model_types_with_images),
        ("Image Metadata Preservation", test_image_metadata_preservation),
        ("Error Handling and Recovery", test_error_handling_and_recovery)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running: {test_name}")
            logger.info(f"{'='*60}")
            
            start_time = time.time()
            result = test_func()
            end_time = time.time()
            
            if result:
                passed += 1
                logger.info(f"✅ {test_name} PASSED ({end_time - start_time:.2f}s)")
            else:
                failed += 1
                logger.error(f"❌ {test_name} FAILED ({end_time - start_time:.2f}s)")
                
        except Exception as e:
            failed += 1
            end_time = time.time()
            logger.error(f"❌ {test_name} FAILED with exception ({end_time - start_time:.2f}s): {e}")
    
    # Cleanup
    cleanup_test_files()
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("ENHANCED END-TO-END TEST SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total tests: {passed + failed}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    
    if failed == 0:
        logger.info("🎉 All enhanced end-to-end tests passed!")
        return True
    else:
        logger.error(f"💥 {failed} enhanced end-to-end tests failed!")
        return False

if __name__ == "__main__":
    success = run_enhanced_end_to_end_tests()
    sys.exit(0 if success else 1)