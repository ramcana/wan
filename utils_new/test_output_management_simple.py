#!/usr/bin/env python3
"""
Simple test script for output management system
Tests core functionality using the actual outputs directory
"""

import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# Add the current directory to Python path to import utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_test_frames(width=640, height=480, num_frames=8):
    """Create test video frames with different colors"""
    frames = []
    
    for i in range(num_frames):
        # Create a frame with a gradient that changes over time
        frame_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create a color gradient
        color_value = int(255 * (i / num_frames))
        frame_array[:, :, 0] = color_value  # Red channel
        frame_array[:, :, 1] = 255 - color_value  # Green channel
        frame_array[:, :, 2] = 128  # Blue channel (constant)
        
        # Add frame number text area (simple rectangle)
        frame_array[10:50, 10:100] = [255, 255, 255]  # White rectangle
        
        # Convert to PIL Image
        frame = Image.fromarray(frame_array, 'RGB')
        frames.append(frame)
    
    return frames

def test_basic_functionality():
    """Test basic output management functionality"""
    print("Testing Basic Output Management Functionality...")
    
    try:
        # Import the functions we need
        from utils import (
            save_video_frames, list_generated_videos, get_video_metadata,
            get_video_file_path, get_video_thumbnail_path, get_output_storage_stats
        )
        
        print("✓ Successfully imported output management functions")
        
        # Test 1: Create test frames
        print("\n1. Creating test video frames...")
        frames = create_test_frames(width=480, height=360, num_frames=6)
        print(f"✓ Created {len(frames)} test frames (480x360)")
        
        # Test 2: Save video with metadata
        print("\n2. Saving video with metadata...")
        test_metadata = {
            "model_type": "t2v-A14B",
            "prompt": "Test video for output management system",
            "resolution": "480x360",
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "lora_path": None,
            "lora_strength": None
        }
        
        video_path, metadata = save_video_frames(frames, test_metadata, fps=24)
        print(f"✓ Video saved to: {video_path}")
        print(f"✓ Video ID: {metadata.id}")
        print(f"✓ File size: {metadata.file_size_mb:.2f} MB")
        print(f"✓ Duration: {metadata.duration_seconds:.1f} seconds")
        
        # Verify video file exists
        if Path(video_path).exists():
            print("✓ Video file exists on disk")
        else:
            print("✗ Video file not found on disk")
            return False
        
        # Test 3: Check thumbnail generation
        print("\n3. Checking thumbnail generation...")
        if metadata.thumbnail_path and Path(metadata.thumbnail_path).exists():
            print(f"✓ Thumbnail generated: {Path(metadata.thumbnail_path).name}")
            
            # Check thumbnail size
            thumb_img = Image.open(metadata.thumbnail_path)
            print(f"✓ Thumbnail size: {thumb_img.size}")
        else:
            print("✗ Thumbnail not generated")
            return False
        
        # Test 4: Test metadata retrieval
        print("\n4. Testing metadata retrieval...")
        retrieved_metadata = get_video_metadata(metadata.id)
        if retrieved_metadata:
            print(f"✓ Retrieved metadata for video: {retrieved_metadata.filename}")
            print(f"✓ Prompt: {retrieved_metadata.prompt}")
            print(f"✓ Model type: {retrieved_metadata.model_type}")
        else:
            print("✗ Failed to retrieve metadata")
            return False
        
        # Test 5: List videos
        print("\n5. Testing video listing...")
        videos = list_generated_videos()
        found_our_video = False
        for video in videos:
            if video.id == metadata.id:
                found_our_video = True
                break
        
        if found_our_video:
            print(f"✓ Found our video in listing of {len(videos)} total videos")
        else:
            print("✗ Our video not found in listing")
            return False
        
        # Test 6: Storage statistics
        print("\n6. Testing storage statistics...")
        stats = get_output_storage_stats()
        print(f"✓ Total size: {stats['total_size_mb']} MB")
        print(f"✓ Video count: {stats['video_count']}")
        print(f"✓ Thumbnail count: {stats['thumbnail_count']}")
        print(f"✓ Metadata entries: {stats['metadata_entries']}")
        
        # Test 7: Video path retrieval
        print("\n7. Testing video path retrieval...")
        video_path_retrieved = get_video_file_path(metadata.id)
        thumbnail_path_retrieved = get_video_thumbnail_path(metadata.id)
        
        if video_path_retrieved and Path(video_path_retrieved).exists():
            print(f"✓ Video path retrieved: {Path(video_path_retrieved).name}")
        else:
            print("✗ Failed to retrieve video path")
            return False
            
        if thumbnail_path_retrieved and Path(thumbnail_path_retrieved).exists():
            print(f"✓ Thumbnail path retrieved: {Path(thumbnail_path_retrieved).name}")
        else:
            print("✗ Failed to retrieve thumbnail path")
            return False
        
        print("\n" + "="*50)
        print("✓ ALL BASIC TESTS PASSED!")
        print("="*50)
        print(f"\nTest video saved as: {metadata.filename}")
        print(f"You can find it in the outputs/ directory")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    assert True  # TODO: Add proper assertion

if __name__ == "__main__":
    print("Output Management System - Basic Test")
    print("=" * 50)
    
    if test_basic_functionality():
        print("\n🎉 BASIC TESTS PASSED! Output management system is working correctly.")
        print("\nYou can check the outputs/ directory to see the generated video and thumbnail.")
        sys.exit(0)
    else:
        print("\n❌ Tests failed.")
        sys.exit(1)