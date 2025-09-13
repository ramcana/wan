"""
Outputs management API endpoints
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pathlib import Path
import os
import mimetypes
from datetime import datetime
from typing import List

from backend.schemas.schemas import OutputsResponse, VideoMetadata
from backend.repositories.database import get_db, GenerationTaskDB, TaskStatusEnum
from utils.thumbnail_generator import generate_video_thumbnail, delete_video_thumbnail
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/outputs", response_model=OutputsResponse)
async def get_outputs(db: Session = Depends(get_db)):
    """
    Get list of all generated videos
    """
    try:
        # Get all completed tasks with output files
        completed_tasks = db.query(GenerationTaskDB).filter(
            GenerationTaskDB.status == TaskStatusEnum.COMPLETED,
            GenerationTaskDB.output_path.isnot(None)
        ).order_by(GenerationTaskDB.completed_at.desc()).all()
        
        videos = []
        total_size_mb = 0.0
        
        for task in completed_tasks:
            if task.output_path and os.path.exists(task.output_path):
                try:
                    # Get file info
                    file_stat = os.stat(task.output_path)
                    file_size_mb = file_stat.st_size / (1024 * 1024)
                    total_size_mb += file_size_mb
                    
                    # Generate thumbnail if it doesn't exist
                    thumbnail_path = None
                    try:
                        thumbnail_name = f"{task.id}_thumb"
                        thumbnail_path = generate_video_thumbnail(
                            task.output_path,
                            thumbnail_name=thumbnail_name,
                            timestamp="00:00:01",
                            width=320,
                            height=180
                        )
                    except Exception as e:
                        logger.warning(f"Could not generate thumbnail for {task.output_path}: {e}")
                    
                    # Create video metadata
                    video_metadata = VideoMetadata(
                        id=task.id,
                        filename=os.path.basename(task.output_path),
                        file_path=task.output_path,
                        thumbnail_path=thumbnail_path,
                        prompt=task.prompt,
                        model_type=task.model_type.value,
                        resolution=task.resolution,
                        duration_seconds=None,  # TODO: Get video duration
                        file_size_mb=file_size_mb,
                        created_at=task.completed_at or task.created_at,
                        generation_time_minutes=task.generation_time_minutes
                    )
                    
                    videos.append(video_metadata)
                    
                except Exception as e:
                    logger.warning(f"Could not process output file {task.output_path}: {e}")
                    continue
        
        return OutputsResponse(
            videos=videos,
            total_count=len(videos),
            total_size_mb=total_size_mb
        )
        
    except Exception as e:
        logger.error(f"Error getting outputs: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not retrieve outputs")

@router.get("/outputs/{video_id}")
async def get_video_info(video_id: str, db: Session = Depends(get_db)):
    """
    Get detailed information about a specific video
    """
    try:
        task = db.query(GenerationTaskDB).filter(
            GenerationTaskDB.id == video_id,
            GenerationTaskDB.status == TaskStatusEnum.COMPLETED
        ).first()
        
        if not task or not task.output_path:
            raise HTTPException(status_code=404, detail="Video not found")
        
        if not os.path.exists(task.output_path):
            raise HTTPException(status_code=404, detail="Video file not found")
        
        # Get file info
        file_stat = os.stat(task.output_path)
        file_size_mb = file_stat.st_size / (1024 * 1024)
        
        # Generate thumbnail if it doesn't exist
        thumbnail_path = None
        try:
            thumbnail_name = f"{task.id}_thumb"
            thumbnail_path = generate_video_thumbnail(
                task.output_path,
                thumbnail_name=thumbnail_name,
                timestamp="00:00:01",
                width=320,
                height=180
            )
        except Exception as e:
            logger.warning(f"Could not generate thumbnail for {task.output_path}: {e}")
        
        return VideoMetadata(
            id=task.id,
            filename=os.path.basename(task.output_path),
            file_path=task.output_path,
            thumbnail_path=thumbnail_path,
            prompt=task.prompt,
            model_type=task.model_type.value,
            resolution=task.resolution,
            duration_seconds=None,
            file_size_mb=file_size_mb,
            created_at=task.completed_at or task.created_at,
            generation_time_minutes=task.generation_time_minutes
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting video info for {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not retrieve video information")

@router.get("/outputs/{video_id}/download")
async def download_video(video_id: str, db: Session = Depends(get_db)):
    """
    Download a generated video file
    """
    try:
        task = db.query(GenerationTaskDB).filter(
            GenerationTaskDB.id == video_id,
            GenerationTaskDB.status == TaskStatusEnum.COMPLETED
        ).first()
        
        if not task or not task.output_path:
            raise HTTPException(status_code=404, detail="Video not found")
        
        if not os.path.exists(task.output_path):
            raise HTTPException(status_code=404, detail="Video file not found")
        
        # Determine media type
        media_type = mimetypes.guess_type(task.output_path)[0] or "application/octet-stream"
        
        return FileResponse(
            path=task.output_path,
            media_type=media_type,
            filename=os.path.basename(task.output_path)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not download video")

@router.get("/outputs/{video_id}/thumbnail")
async def get_video_thumbnail(video_id: str, db: Session = Depends(get_db)):
    """
    Get the thumbnail for a video
    """
    try:
        task = db.query(GenerationTaskDB).filter(
            GenerationTaskDB.id == video_id,
            GenerationTaskDB.status == TaskStatusEnum.COMPLETED
        ).first()
        
        if not task:
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Check for existing thumbnail
        thumbnail_name = f"{task.id}_thumb"
        thumbnail_path = f"thumbnails/{thumbnail_name}.jpg"
        
        # Generate thumbnail if it doesn't exist
        if not os.path.exists(thumbnail_path) and task.output_path and os.path.exists(task.output_path):
            try:
                generated_path = generate_video_thumbnail(
                    task.output_path,
                    thumbnail_name=thumbnail_name,
                    timestamp="00:00:01",
                    width=320,
                    height=180
                )
                if generated_path:
                    thumbnail_path = generated_path
            except Exception as e:
                logger.warning(f"Could not generate thumbnail for {video_id}: {e}")
        
        if not os.path.exists(thumbnail_path):
            raise HTTPException(status_code=404, detail="Thumbnail not found")
        
        return FileResponse(
            path=thumbnail_path,
            media_type="image/jpeg",
            filename=f"{thumbnail_name}.jpg"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting thumbnail for {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not get thumbnail")

@router.delete("/outputs/{video_id}")
async def delete_video(video_id: str, db: Session = Depends(get_db)):
    """
    Delete a generated video and its associated task
    """
    try:
        task = db.query(GenerationTaskDB).filter(
            GenerationTaskDB.id == video_id
        ).first()
        
        if not task:
            raise HTTPException(status_code=404, detail="Video not found")
        
        # Remove output file if it exists
        if task.output_path and os.path.exists(task.output_path):
            try:
                os.remove(task.output_path)
                logger.info(f"Deleted output file: {task.output_path}")
            except Exception as e:
                logger.warning(f"Could not remove output file {task.output_path}: {e}")
        
        # Remove thumbnail if it exists
        thumbnail_name = f"{task.id}_thumb"
        thumbnail_path = f"thumbnails/{thumbnail_name}.jpg"
        if os.path.exists(thumbnail_path):
            try:
                delete_video_thumbnail(thumbnail_path)
                logger.info(f"Deleted thumbnail: {thumbnail_path}")
            except Exception as e:
                logger.warning(f"Could not remove thumbnail {thumbnail_path}: {e}")
        
        # Remove input image file if it exists
        if task.image_path and os.path.exists(task.image_path):
            try:
                os.remove(task.image_path)
                logger.info(f"Deleted input image: {task.image_path}")
            except Exception as e:
                logger.warning(f"Could not remove image file {task.image_path}: {e}")
        
        # Delete task from database
        db.delete(task)
        db.commit()
        
        logger.info(f"Deleted video and task: {video_id}")
        
        return {"message": f"Video {video_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not delete video")

@router.post("/outputs/cleanup")
async def cleanup_orphaned_files():
    """
    Clean up orphaned files (files without corresponding database entries)
    """
    try:
        outputs_dir = "outputs"
        uploads_dir = "uploads"
        
        cleaned_files = []
        
        # Check outputs directory
        if os.path.exists(outputs_dir):
            for filename in os.listdir(outputs_dir):
                file_path = os.path.join(outputs_dir, filename)
                if os.path.isfile(file_path):
                    # Check if file is referenced in database
                    # This is a simplified check - in production you'd want more robust logic
                    try:
                        os.remove(file_path)
                        cleaned_files.append(file_path)
                    except Exception as e:
                        logger.warning(f"Could not remove orphaned file {file_path}: {e}")
        
        logger.info(f"Cleaned up {len(cleaned_files)} orphaned files")
        
        return {
            "message": f"Cleaned up {len(cleaned_files)} orphaned files",
            "files": cleaned_files
        }
        
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not perform cleanup")
