#!/usr/bin/env python3
"""
Minimal test script for output management system
Tests core functionality without heavy ML dependencies
"""

import os
import sys
import tempfile
import json
from pathlib import Path
from PIL import Image
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import uuid
import cv2

# Define minimal classes needed for testing
@dataclass
class VideoMetadata:
    """Metadata structure for generated videos"""
    id: str
    filename: str
    model_type: str
    prompt: str
    resolution: str
    width: int
    height: int
    num_frames: int
    fps: int
    duration_seconds: float
    file_size_mb: float
    num_inference_steps: int
    guidance_scale: float
    lora_path: Optional[str] = None
    lora_strength: Optional[float] = None
    input_image_path: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    thumbnail_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization"""
        return {
            "id": self.id,
            "filename": self.filename,
            "model_type": self.model_type,
            "prompt": self.prompt,
            "resolution": self.resolution,
            "width": self.width,
            "height": self.height,
            "num_frames": self.num_frames,
            "fps": self.fps,
            "duration_seconds": self.duration_seconds,
            "file_size_mb": self.file_size_mb,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "lora_path": self.lora_path,
            "lora_strength": self.lora_strength,
            "input_image_path": self.input_image_path,
            "created_at": self.created_at.isoformat(),
            "thumbnail_path": self.thumbnail_path
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VideoMetadata':
        """Create VideoMetadata from dictionary"""
        # Convert created_at string back to datetime
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        return cls(**data)


class OutputManager:
    """Manages video output files, thumbnails, and metadata"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.output_dir = Path(self.config["directories"]["output_directory"])
        self.thumbnails_dir = self.output_dir / "thumbnails"
        self.metadata_file = self.output_dir / "metadata.json"
        
        # Create directories if they don't exist
        self.output_dir.mkdir(exist_ok=True)
        self.thumbnails_dir.mkdir(exist_ok=True)
        
        # Load existing metadata
        self.metadata_cache = self._load_metadata()
        
        # Video encoding settings
        self.video_codec = 'libx264'
        self.video_bitrate = '5000k'
        self.default_fps = 24
        self.thumbnail_size = self.config.get("ui", {}).get("gallery_thumbnail_size", 256)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load system configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Failed to load config: {e}")
            # Return default config
            return {
                "directories": {"output_directory": "outputs"},
                "ui": {"gallery_thumbnail_size": 256}
            }
    
    def _load_metadata(self) -> Dict[str, VideoMetadata]:
        """Load video metadata from disk"""
        if not self.metadata_file.exists():
            return {}
        
        try:
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
            
            metadata_cache = {}
            for video_id, metadata_dict in data.items():
                try:
                    metadata_cache[video_id] = VideoMetadata.from_dict(metadata_dict)
                except Exception as e:
                    print(f"Failed to load metadata for video {video_id}: {e}")
            
            return metadata_cache
            
        except (json.JSONDecodeError, IOError) as e:
            print(f"Failed to load metadata: {e}")
            return {}
    
    def _save_metadata(self):
        """Save video metadata to disk"""
        try:
            data = {}
            for video_id, metadata in self.metadata_cache.items():
                data[video_id] = metadata.to_dict()
            
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except IOError as e:
            print(f"Failed to save metadata: {e}")
    
    def _generate_filename(self, model_type: str, prompt: str, resolution: str) -> str:
        """Generate a unique filename for the video"""
        # Create a safe filename from prompt (first 50 chars)
        safe_prompt = "".join(c for c in prompt[:50] if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_prompt = safe_prompt.replace(' ', '_')
        
        # Add timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create filename
        filename = f"{model_type}_{safe_prompt}_{resolution}_{timestamp}.mp4"
        
        return filename
    
    def save_video_frames(self, frames: List[Image.Image], metadata_dict: Dict[str, Any], 
                         fps: int = None) -> tuple:
        """Save video frames as MP4 file with metadata"""
        if not frames:
            raise ValueError("No frames provided for video saving")
        
        fps = fps or self.default_fps
        
        # Generate unique filename
        filename = self._generate_filename(
            metadata_dict.get("model_type", "unknown"),
            metadata_dict.get("prompt", "")[:50],
            metadata_dict.get("resolution", "unknown")
        )
        
        output_path = self.output_dir / filename
        
        print(f"Saving video with {len(frames)} frames to {output_path}")
        
        try:
            # Convert PIL images to numpy arrays
            frame_arrays = []
            for frame in frames:
                # Convert PIL image to RGB if not already
                if frame.mode != 'RGB':
                    frame = frame.convert('RGB')
                
                # Convert to numpy array (OpenCV uses BGR)
                frame_array = np.array(frame)
                frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                frame_arrays.append(frame_array)
            
            # Get video dimensions from first frame
            height, width = frame_arrays[0].shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            if not video_writer.isOpened():
                raise RuntimeError("Failed to open video writer")
            
            # Write frames
            for frame_array in frame_arrays:
                video_writer.write(frame_array)
            
            # Release video writer
            video_writer.release()
            
            # Verify file was created and get size
            if not output_path.exists():
                raise RuntimeError("Video file was not created")
            
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            
            # Create metadata object
            video_id = str(uuid.uuid4())
            duration_seconds = len(frames) / fps
            
            metadata = VideoMetadata(
                id=video_id,
                filename=filename,
                model_type=metadata_dict.get("model_type", "unknown"),
                prompt=metadata_dict.get("prompt", ""),
                resolution=metadata_dict.get("resolution", f"{width}x{height}"),
                width=width,
                height=height,
                num_frames=len(frames),
                fps=fps,
                duration_seconds=duration_seconds,
                file_size_mb=file_size_mb,
                num_inference_steps=metadata_dict.get("num_inference_steps", 50),
                guidance_scale=metadata_dict.get("guidance_scale", 7.5),
                lora_path=metadata_dict.get("lora_path"),
                lora_strength=metadata_dict.get("lora_strength"),
                input_image_path=metadata_dict.get("input_image_path")
            )
            
            # Generate thumbnail
            thumbnail_path = self._generate_thumbnail(frames[0], video_id)
            metadata.thumbnail_path = thumbnail_path
            
            # Cache metadata
            self.metadata_cache[video_id] = metadata
            self._save_metadata()
            
            print(f"Successfully saved video: {filename} ({file_size_mb:.2f} MB, {duration_seconds:.1f}s)")
            
            return str(output_path), metadata
            
        except Exception as e:
            print(f"Failed to save video: {e}")
            # Clean up partial file if it exists
            if output_path.exists():
                try:
                    output_path.unlink()
                except:
                    pass
            raise
    
    def _generate_thumbnail(self, first_frame: Image.Image, video_id: str) -> str:
        """Generate thumbnail from the first frame of the video"""
        try:
            # Create thumbnail filename
            thumbnail_filename = f"{video_id}_thumb.jpg"
            thumbnail_path = self.thumbnails_dir / thumbnail_filename
            
            # Resize image to thumbnail size while maintaining aspect ratio
            thumbnail = first_frame.copy()
            thumbnail.thumbnail((self.thumbnail_size, self.thumbnail_size), Image.Resampling.LANCZOS)
            
            # Create a square thumbnail with padding if needed
            square_thumbnail = Image.new('RGB', (self.thumbnail_size, self.thumbnail_size), (0, 0, 0))
            
            # Center the thumbnail
            x_offset = (self.thumbnail_size - thumbnail.width) // 2
            y_offset = (self.thumbnail_size - thumbnail.height) // 2
            square_thumbnail.paste(thumbnail, (x_offset, y_offset))
            
            # Save thumbnail
            square_thumbnail.save(thumbnail_path, 'JPEG', quality=85)
            
            print(f"Generated thumbnail: {thumbnail_filename}")
            return str(thumbnail_path)
            
        except Exception as e:
            print(f"Failed to generate thumbnail: {e}")
            return None
    
    def get_video_metadata(self, video_id: str) -> Optional[VideoMetadata]:
        """Get metadata for a specific video"""
        return self.metadata_cache.get(video_id)
    
    def list_videos(self, sort_by: str = "created_at", reverse: bool = True) -> List[VideoMetadata]:
        """List all videos with their metadata"""
        videos = list(self.metadata_cache.values())
        
        # Filter out videos whose files no longer exist
        existing_videos = []
        for video in videos:
            video_path = self.output_dir / video.filename
            if video_path.exists():
                existing_videos.append(video)
            else:
                print(f"Video file not found: {video.filename}")
        
        # Sort videos
        try:
            if sort_by == "created_at":
                existing_videos.sort(key=lambda x: x.created_at, reverse=reverse)
            elif sort_by == "filename":
                existing_videos.sort(key=lambda x: x.filename.lower(), reverse=reverse)
            elif sort_by == "file_size_mb":
                existing_videos.sort(key=lambda x: x.file_size_mb, reverse=reverse)
            elif sort_by == "duration_seconds":
                existing_videos.sort(key=lambda x: x.duration_seconds, reverse=reverse)
            elif sort_by == "resolution":
                existing_videos.sort(key=lambda x: (x.width * x.height), reverse=reverse)
            else:
                print(f"Unknown sort field: {sort_by}, using created_at")
                existing_videos.sort(key=lambda x: x.created_at, reverse=reverse)
        except Exception as e:
            print(f"Failed to sort videos: {e}")
        
        return existing_videos
    
    def delete_video(self, video_id: str) -> bool:
        """Delete a video and its associated files"""
        if video_id not in self.metadata_cache:
            print(f"Video not found: {video_id}")
            return False
        
        metadata = self.metadata_cache[video_id]
        
        try:
            # Delete video file
            video_path = self.output_dir / metadata.filename
            if video_path.exists():
                video_path.unlink()
                print(f"Deleted video file: {metadata.filename}")
            
            # Delete thumbnail
            if metadata.thumbnail_path:
                thumbnail_path = Path(metadata.thumbnail_path)
                if thumbnail_path.exists():
                    thumbnail_path.unlink()
                    print(f"Deleted thumbnail: {thumbnail_path.name}")
            
            # Remove from metadata cache
            del self.metadata_cache[video_id]
            self._save_metadata()
            
            print(f"Successfully deleted video: {video_id}")
            return True
            
        except Exception as e:
            print(f"Failed to delete video {video_id}: {e}")
            return False
    
    def get_video_path(self, video_id: str) -> Optional[str]:
        """Get the full path to a video file"""
        if video_id not in self.metadata_cache:
            return None
        
        metadata = self.metadata_cache[video_id]
        video_path = self.output_dir / metadata.filename
        
        if video_path.exists():
            return str(video_path)
        
        return None
    
    def get_thumbnail_path(self, video_id: str) -> Optional[str]:
        """Get the full path to a video's thumbnail"""
        if video_id not in self.metadata_cache:
            return None
        
        metadata = self.metadata_cache[video_id]
        if metadata.thumbnail_path and Path(metadata.thumbnail_path).exists():
            return metadata.thumbnail_path
        
        return None
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics for the output directory"""
        try:
            total_size = 0
            video_count = 0
            thumbnail_count = 0
            
            # Calculate video files size
            for video_path in self.output_dir.glob("*.mp4"):
                if video_path.is_file():
                    total_size += video_path.stat().st_size
                    video_count += 1
            
            # Calculate thumbnail files size
            for thumb_path in self.thumbnails_dir.glob("*.jpg"):
                if thumb_path.is_file():
                    total_size += thumb_path.stat().st_size
                    thumbnail_count += 1
            
            total_size_mb = total_size / (1024 * 1024)
            
            return {
                "total_size_mb": round(total_size_mb, 2),
                "video_count": video_count,
                "thumbnail_count": thumbnail_count,
                "output_directory": str(self.output_dir),
                "thumbnails_directory": str(self.thumbnails_dir),
                "metadata_entries": len(self.metadata_cache)
            }
            
        except Exception as e:
            print(f"Failed to get storage stats: {e}")
            return {
                "total_size_mb": 0,
                "video_count": 0,
                "thumbnail_count": 0,
                "output_directory": str(self.output_dir),
                "thumbnails_directory": str(self.thumbnails_dir),
                "metadata_entries": len(self.metadata_cache)
            }


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
        
        print("\n" + "="*50)
        print("✓ ALL OUTPUT MANAGEMENT TESTS PASSED!")
        print("="*50)
        return True


if __name__ == "__main__":
    print("Output Management System Test Suite")
    print("=" * 50)
    
    try:
        if test_output_management():
            print("\n🎉 ALL TESTS PASSED! Output management system is working correctly.")
            sys.exit(0)
        else:
            print("\n❌ Tests failed.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        import traceback
traceback.print_exc()
        sys.exit(1)