"""
Thumbnail generation utilities for video files
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class ThumbnailGenerator:
    """Generate thumbnails for video files using ffmpeg"""
    
    def __init__(self, thumbnails_dir: str = "thumbnails"):
        self.thumbnails_dir = Path(thumbnails_dir)
        self.thumbnails_dir.mkdir(exist_ok=True)
        
    def generate_thumbnail(
        self, 
        video_path: str, 
        thumbnail_name: Optional[str] = None,
        timestamp: str = "00:00:01",
        width: int = 320,
        height: int = 180
    ) -> Optional[str]:
        """
        Generate a thumbnail for a video file
        
        Args:
            video_path: Path to the video file
            thumbnail_name: Name for the thumbnail file (without extension)
            timestamp: Timestamp to capture (format: HH:MM:SS)
            width: Thumbnail width in pixels
            height: Thumbnail height in pixels
            
        Returns:
            Path to the generated thumbnail or None if failed
        """
        try:
            if not os.path.exists(video_path):
                logger.error(f"Video file not found: {video_path}")
                return None
                
            # Generate thumbnail filename
            if thumbnail_name is None:
                video_name = Path(video_path).stem
                thumbnail_name = f"{video_name}_thumb"
                
            thumbnail_path = self.thumbnails_dir / f"{thumbnail_name}.jpg"
            
            # Skip if thumbnail already exists
            if thumbnail_path.exists():
                logger.info(f"Thumbnail already exists: {thumbnail_path}")
                return str(thumbnail_path)
            
            # Check if ffmpeg is available
            if not self._check_ffmpeg():
                logger.warning("ffmpeg not available, cannot generate thumbnails")
                return None
                
            # Generate thumbnail using ffmpeg
            cmd = [
                "ffmpeg",
                "-i", video_path,
                "-ss", timestamp,
                "-vframes", "1",
                "-vf", f"scale={width}:{height}",
                "-y",  # Overwrite output file
                str(thumbnail_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            
            if result.returncode == 0 and thumbnail_path.exists():
                logger.info(f"Generated thumbnail: {thumbnail_path}")
                return str(thumbnail_path)
            else:
                logger.error(f"Failed to generate thumbnail: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error(f"Thumbnail generation timed out for {video_path}")
            return None
        except Exception as e:
            logger.error(f"Error generating thumbnail for {video_path}: {str(e)}")
            return None
    
    def generate_thumbnail_async(
        self,
        video_path: str,
        thumbnail_name: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Generate thumbnail asynchronously (placeholder for future async implementation)
        Currently just calls the sync version
        """
        return self.generate_thumbnail(video_path, thumbnail_name, **kwargs)
    
    def delete_thumbnail(self, thumbnail_path: str) -> bool:
        """
        Delete a thumbnail file
        
        Args:
            thumbnail_path: Path to the thumbnail file
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            if os.path.exists(thumbnail_path):
                os.remove(thumbnail_path)
                logger.info(f"Deleted thumbnail: {thumbnail_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting thumbnail {thumbnail_path}: {str(e)}")
            return False
    
    def _check_ffmpeg(self) -> bool:
        """Check if ffmpeg is available in the system"""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

# Global thumbnail generator instance
thumbnail_generator = ThumbnailGenerator()

def generate_video_thumbnail(
    video_path: str,
    thumbnail_name: Optional[str] = None,
    **kwargs
) -> Optional[str]:
    """
    Convenience function to generate a video thumbnail
    
    Args:
        video_path: Path to the video file
        thumbnail_name: Name for the thumbnail file
        **kwargs: Additional arguments for thumbnail generation
        
    Returns:
        Path to the generated thumbnail or None if failed
    """
    return thumbnail_generator.generate_thumbnail(
        video_path, 
        thumbnail_name, 
        **kwargs
    )

def delete_video_thumbnail(thumbnail_path: str) -> bool:
    """
    Convenience function to delete a video thumbnail
    
    Args:
        thumbnail_path: Path to the thumbnail file
        
    Returns:
        True if deleted successfully, False otherwise
    """
    return thumbnail_generator.delete_thumbnail(thumbnail_path)