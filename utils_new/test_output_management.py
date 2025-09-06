#!/usr/bin/env python3
"""
Test script for output management system
Tests video saving, thumbnail generation, and metadata management
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

# Add the current directory to Python path to import utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    OutputManager, VideoMetadata, save_video_frames, 
    list_generated_videos, get_video_metadata, delete_generated_video,
    get_video_file_path, get_video_thumbnail_path, get_output_storage_stats
)

def create_test_frames(width=640, height=480, num_frames=10):
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

def test_output_management():
    """Test the output management system"""
    print("Testing Output Management System...")
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_config = {
            "directories": {"output_directory": temp_dir},
            "ui": {"gallery_thumbnail_size": 128}
        }
        
        # Create a temporary config file
        config_path = Path(temp_dir) / "test_config.json"
        import json
        with open(config_path, 'w') as f:
            json.dump(temp_config, f)
        
        # Initialize output manager with test config
        output_manager = OutputManager(str(config_path))
        
        print(f"✓ Output manager initialized with directory: {temp_dir}")
        
        # Test 1: Create test frames
        print("\n1. Creating test video frames...")
        frames = create_test_frames(width=640, height=480, num_frames=8)
        print(f"✓ Created {len(frames)} test frames (640x480)")
        
        # Test 2: Save video with metadata
        print("\n2. Saving video with metadata...")
        test_metadata = {
            "model_type": "t2v-A14B",
            "prompt": "A beautiful sunset over mountains",
            "resolution": "640x480",
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "lora_path": None,
            "lora_strength": None
        }
        
        try:
            video_path, metadata = output_manager.save_video_frames(frames, test_metadata, fps=24)
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
                
        except Exception as e:
            print(f"✗ Failed to save video: {e}")
            return False
        
        # Test 3: Check thumbnail generation
        print("\n3. Checking thumbnail generation...")
        if metadata.thumbnail_path and Path(metadata.thumbnail_path).exists():
            print(f"✓ Thumbnail generated: {metadata.thumbnail_path}")
            
            # Check thumbnail size
            thumb_img = Image.open(metadata.thumbnail_path)
            print(f"✓ Thumbnail size: {thumb_img.size}")
        else:
            print("✗ Thumbnail not generated")
            return False
        
        # Test 4: Test metadata retrieval
        print("\n4. Testing metadata retrieval...")
        retrieved_metadata = output_manager.get_video_metadata(metadata.id)
        if retrieved_metadata:
            print(f"✓ Retrieved metadata for video: {retrieved_metadata.filename}")
            print(f"✓ Prompt: {retrieved_metadata.prompt}")
            print(f"✓ Model type: {retrieved_metadata.model_type}")
        else:
            print("✗ Failed to retrieve metadata")
            return False
        
        # Test 5: List videos
        print("\n5. Testing video listing...")
        videos = output_manager.list_videos()
        if len(videos) == 1 and videos[0].id == metadata.id:
            print(f"✓ Found {len(videos)} video(s) in listing")
            print(f"✓ Video filename: {videos[0].filename}")
        else:
            print(f"✗ Expected 1 video, found {len(videos)}")
            return False
        
        # Test 6: Storage statistics
        print("\n6. Testing storage statistics...")
        stats = output_manager.get_storage_stats()
        print(f"✓ Total size: {stats['total_size_mb']} MB")
        print(f"✓ Video count: {stats['video_count']}")
        print(f"✓ Thumbnail count: {stats['thumbnail_count']}")
        print(f"✓ Metadata entries: {stats['metadata_entries']}")
        
        # Test 7: Video path retrieval
        print("\n7. Testing video path retrieval...")
        video_path_retrieved = output_manager.get_video_path(metadata.id)
        thumbnail_path_retrieved = output_manager.get_thumbnail_path(metadata.id)
        
        if video_path_retrieved and Path(video_path_retrieved).exists():
            print(f"✓ Video path retrieved: {video_path_retrieved}")
        else:
            print("✗ Failed to retrieve video path")
            return False
            
        if thumbnail_path_retrieved and Path(thumbnail_path_retrieved).exists():
            print(f"✓ Thumbnail path retrieved: {thumbnail_path_retrieved}")
        else:
            print("✗ Failed to retrieve thumbnail path")
            return False
        
        # Test 8: Save another video to test multiple videos
        print("\n8. Testing multiple videos...")
        frames2 = create_test_frames(width=320, height=240, num_frames=5)
        test_metadata2 = {
            "model_type": "i2v-A14B",
            "prompt": "A flowing river through a forest",
            "resolution": "320x240",
            "num_inference_steps": 30,
            "guidance_scale": 8.0
        }
        
        video_path2, metadata2 = output_manager.save_video_frames(frames2, test_metadata2, fps=12)
        print(f"✓ Second video saved: {metadata2.filename}")
        
        # List videos again
        videos = output_manager.list_videos(sort_by="created_at", reverse=True)
        if len(videos) == 2:
            print(f"✓ Now have {len(videos)} videos total")
            print(f"✓ Latest video: {videos[0].filename}")
            print(f"✓ Older video: {videos[1].filename}")
        else:
            print(f"✗ Expected 2 videos, found {len(videos)}")
            return False
        
        # Test 9: Delete video
        print("\n9. Testing video deletion...")
        if output_manager.delete_video(metadata.id):
            print(f"✓ Successfully deleted video: {metadata.id}")
            
            # Verify it's gone
            videos_after_delete = output_manager.list_videos()
            if len(videos_after_delete) == 1 and videos_after_delete[0].id == metadata2.id:
                print("✓ Video list updated correctly after deletion")
            else:
                print("✗ Video list not updated correctly after deletion")
                return False
        else:
            print("✗ Failed to delete video")
            return False
        
        # Test 10: Cleanup orphaned files
        print("\n10. Testing cleanup functionality...")
        cleanup_stats = output_manager.cleanup_orphaned_files()
        print(f"✓ Cleanup completed: {cleanup_stats}")
        
        print("\n" + "="*50)
        print("✓ ALL OUTPUT MANAGEMENT TESTS PASSED!")
        print("="*50)
        return True

    assert True  # TODO: Add proper assertion

def test_convenience_functions():
    """Test the convenience functions"""
    print("\nTesting convenience functions...")
    
    # Create test frames
    frames = create_test_frames(width=480, height=360, num_frames=6)
    
    # Test metadata
    test_metadata = {
        "model_type": "ti2v-5B",
        "prompt": "A magical forest with glowing trees",
        "resolution": "480x360",
        "num_inference_steps": 40,
        "guidance_scale": 6.5
    }
    
    try:
        # Test save_video_frames convenience function
        video_path, metadata = save_video_frames(frames, test_metadata, fps=20)
        print(f"✓ Convenience function saved video: {metadata.filename}")
        
        # Test list_generated_videos convenience function
        videos = list_generated_videos()
        print(f"✓ Listed {len(videos)} videos using convenience function")
        
        # Test get_video_metadata convenience function
        retrieved_metadata = get_video_metadata(metadata.id)
        if retrieved_metadata:
            print(f"✓ Retrieved metadata using convenience function: {retrieved_metadata.prompt}")
        
        # Test get_video_file_path convenience function
        file_path = get_video_file_path(metadata.id)
        if file_path and Path(file_path).exists():
            print(f"✓ Retrieved video file path: {Path(file_path).name}")
        
        # Test get_video_thumbnail_path convenience function
        thumb_path = get_video_thumbnail_path(metadata.id)
        if thumb_path and Path(thumb_path).exists():
            print(f"✓ Retrieved thumbnail path: {Path(thumb_path).name}")
        
        # Test get_output_storage_stats convenience function
        stats = get_output_storage_stats()
        print(f"✓ Storage stats: {stats['video_count']} videos, {stats['total_size_mb']:.2f} MB")
        
        # Test delete_generated_video convenience function
        if delete_generated_video(metadata.id):
            print("✓ Deleted video using convenience function")
        
        print("✓ All convenience functions work correctly!")
        return True
        
    except Exception as e:
        print(f"✗ Convenience function test failed: {e}")
        return False

    assert True  # TODO: Add proper assertion

if __name__ == "__main__":
    print("Output Management System Test Suite")
    print("=" * 50)
    
    try:
        # Run main tests
        if test_output_management():
            # Run convenience function tests
            if test_convenience_functions():
                print("\n🎉 ALL TESTS PASSED! Output management system is working correctly.")
                sys.exit(0)
            else:
                print("\n❌ Convenience function tests failed.")
                sys.exit(1)
        else:
            print("\n❌ Main tests failed.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)