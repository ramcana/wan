#!/usr/bin/env python3
"""
Demo script for output management system
Creates a test video in the actual outputs directory
"""

import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import uuid
import cv2

# Define minimal classes needed for demo
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

class SimpleOutputManager:
    """Simple output manager for demo"""
    
    def __init__(self):
        self.output_dir = Path("outputs")
        self.thumbnails_dir = self.output_dir / "thumbnails"
        self.metadata_file = self.output_dir / "metadata.json"
        
        # Create directories if they don't exist
        self.output_dir.mkdir(exist_ok=True)
        self.thumbnails_dir.mkdir(exist_ok=True)
        
        # Load existing metadata
        self.metadata_cache = self._load_metadata()
        
        # Settings
        self.default_fps = 24
        self.thumbnail_size = 256
    
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
                    # Convert created_at string back to datetime
                    if isinstance(metadata_dict.get('created_at'), str):
                        metadata_dict['created_at'] = datetime.fromisoformat(metadata_dict['created_at'])
                    
                    metadata_cache[video_id] = VideoMetadata(**metadata_dict)
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
    
    def save_video_frames(self, frames: List[Image.Image], metadata_dict: Dict[str, Any], 
                         fps: int = None) -> tuple:
        """Save video frames as MP4 file with metadata"""
        if not frames:
            raise ValueError("No frames provided for video saving")
        
        fps = fps or self.default_fps
        
        # Generate unique filename
        safe_prompt = "".join(c for c in metadata_dict.get("prompt", "")[:30] if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_prompt = safe_prompt.replace(' ', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{metadata_dict.get('model_type', 'unknown')}_{safe_prompt}_{timestamp}.mp4"
        
        output_path = self.output_dir / filename
        
        print(f"Saving video with {len(frames)} frames to {output_path}")
        
        try:
            # Convert PIL images to numpy arrays
            frame_arrays = []
            for frame in frames:
                if frame.mode != 'RGB':
                    frame = frame.convert('RGB')
                
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
                resolution=f"{width}x{height}",
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
            if output_path.exists():
                try:
                    output_path.unlink()
                except:
                    pass
            raise
    
    def _generate_thumbnail(self, first_frame: Image.Image, video_id: str) -> str:
        """Generate thumbnail from the first frame of the video"""
        try:
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
    
    def list_videos(self) -> List[VideoMetadata]:
        """List all videos with their metadata"""
        videos = list(self.metadata_cache.values())
        videos.sort(key=lambda x: x.created_at, reverse=True)
        return videos
    
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
                "metadata_entries": len(self.metadata_cache)
            }
            
        except Exception as e:
            print(f"Failed to get storage stats: {e}")
            return {"total_size_mb": 0, "video_count": 0, "thumbnail_count": 0, "metadata_entries": 0}


def create_demo_frames(width=640, height=480, num_frames=12):
    """Create demo video frames with animated content"""
    frames = []
    
    for i in range(num_frames):
        # Create a frame with animated content
        frame_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create animated gradient
        progress = i / (num_frames - 1)
        
        # Background gradient
        for y in range(height):
            for x in range(width):
                r = int(255 * (x / width) * progress)
                g = int(255 * (y / height) * (1 - progress))
                b = int(128 + 127 * np.sin(progress * np.pi))
                frame_array[y, x] = [r, g, b]
        
        # Add animated circle
        center_x = int(width * (0.2 + 0.6 * progress))
        center_y = int(height * 0.5)
        radius = int(30 + 20 * np.sin(progress * 2 * np.pi))
        
        cv2.circle(frame_array, (center_x, center_y), radius, (255, 255, 255), -1)
        
        # Add frame counter
        cv2.putText(frame_array, f"Frame {i+1}/{num_frames}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Convert to PIL Image
        frame = Image.fromarray(cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB))
        frames.append(frame)
    
    return frames


def demo_output_management():
    """Demonstrate the output management system"""
    print("Output Management System Demo")
    print("=" * 50)
    
    # Initialize output manager
    output_manager = SimpleOutputManager()
    
    print("✓ Output manager initialized")
    print(f"✓ Output directory: {output_manager.output_dir}")
    print(f"✓ Thumbnails directory: {output_manager.thumbnails_dir}")
    
    # Show current stats
    print("\nCurrent storage statistics:")
    stats = output_manager.get_storage_stats()
    print(f"  Videos: {stats['video_count']}")
    print(f"  Thumbnails: {stats['thumbnail_count']}")
    print(f"  Total size: {stats['total_size_mb']} MB")
    print(f"  Metadata entries: {stats['metadata_entries']}")
    
    # Create demo video
    print("\nCreating demo video...")
    frames = create_demo_frames(width=480, height=360, num_frames=16)
    print(f"✓ Created {len(frames)} animated frames (480x360)")
    
    # Save video with metadata
    print("\nSaving demo video...")
    demo_metadata = {
        "model_type": "demo-output-test",
        "prompt": "Animated gradient with moving circle - Output Management Demo",
        "resolution": "480x360",
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "lora_path": None,
        "lora_strength": None
    }
    
    try:
        video_path, metadata = output_manager.save_video_frames(frames, demo_metadata, fps=8)
        
        print(f"✓ Demo video saved successfully!")
        print(f"  File: {metadata.filename}")
        print(f"  Path: {video_path}")
        print(f"  Size: {metadata.file_size_mb:.2f} MB")
        print(f"  Duration: {metadata.duration_seconds:.1f} seconds")
        print(f"  Frames: {metadata.num_frames}")
        print(f"  FPS: {metadata.fps}")
        
        if metadata.thumbnail_path:
            print(f"  Thumbnail: {Path(metadata.thumbnail_path).name}")
        
        # Show updated stats
        print("\nUpdated storage statistics:")
        stats = output_manager.get_storage_stats()
        print(f"  Videos: {stats['video_count']}")
        print(f"  Thumbnails: {stats['thumbnail_count']}")
        print(f"  Total size: {stats['total_size_mb']} MB")
        print(f"  Metadata entries: {stats['metadata_entries']}")
        
        # List all videos
        print("\nAll videos in output directory:")
        videos = output_manager.list_videos()
        for i, video in enumerate(videos, 1):
            print(f"  {i}. {video.filename}")
            print(f"     Prompt: {video.prompt[:60]}...")
            print(f"     Created: {video.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"     Size: {video.file_size_mb:.2f} MB")
            print()
        
        print("=" * 50)
        print("✅ DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"\nYou can find the demo video at: {video_path}")
        print(f"And its thumbnail at: {metadata.thumbnail_path}")
        print("\nThe output management system is working correctly!")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if demo_output_management():
        sys.exit(0)
    else:
        sys.exit(1)