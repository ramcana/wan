"""
Queue management API endpoints
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List, Optional
from datetime import datetime
import os

from backend.schemas.schemas import QueueStatus, TaskInfo, TaskStatus
from backend.repositories.database import get_db, GenerationTaskDB, TaskStatusEnum
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/queue", response_model=QueueStatus)
async def get_queue_status(
    limit: Optional[int] = None,
    status_filter: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Get current queue status and all tasks
    Supports HTTP polling with optional filtering
    """
    try:
        # Build query with optional filtering
        query = db.query(GenerationTaskDB)
        
        # Apply status filter if provided
        if status_filter:
            try:
                status_enum = TaskStatusEnum(status_filter)
                query = query.filter(GenerationTaskDB.status == status_enum)
            except ValueError:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid status filter: {status_filter}"
                )
        
        # Order by creation time (newest first)
        query = query.order_by(desc(GenerationTaskDB.created_at))
        
        # Apply limit if provided
        if limit and limit > 0:
            query = query.limit(limit)
        
        tasks = query.all()
        
        # Get total counts (without filters for accurate statistics)
        all_tasks = db.query(GenerationTaskDB).all()
        total_tasks = len(all_tasks)
        pending_tasks = len([t for t in all_tasks if t.status == TaskStatusEnum.PENDING])
        processing_tasks = len([t for t in all_tasks if t.status == TaskStatusEnum.PROCESSING])
        completed_tasks = len([t for t in all_tasks if t.status == TaskStatusEnum.COMPLETED])
        failed_tasks = len([t for t in all_tasks if t.status == TaskStatusEnum.FAILED])
        cancelled_tasks = len([t for t in all_tasks if t.status == TaskStatusEnum.CANCELLED])
        
        # Convert to response format
        task_infos = []
        for task in tasks:
            task_info = TaskInfo(
                id=task.id,
                model_type=task.model_type.value,
                prompt=task.prompt,
                image_path=task.image_path,
                resolution=task.resolution,
                steps=task.steps,
                lora_path=task.lora_path,
                lora_strength=task.lora_strength,
                status=TaskStatus(task.status.value),
                progress=task.progress,
                created_at=task.created_at,
                started_at=task.started_at,
                completed_at=task.completed_at,
                output_path=task.output_path,
                error_message=task.error_message,
                estimated_time_minutes=task.estimated_time_minutes
            )
            task_infos.append(task_info)
        
        return QueueStatus(
            total_tasks=total_tasks,
            pending_tasks=pending_tasks,
            processing_tasks=processing_tasks,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            cancelled_tasks=cancelled_tasks,
            tasks=task_infos
        )
        
    except Exception as e:
        logger.error(f"Error getting queue status: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/queue/{task_id}/cancel")
async def cancel_task(task_id: str, db: Session = Depends(get_db)):
    """
    Cancel a pending or processing task
    """
    try:
        task = db.query(GenerationTaskDB).filter(GenerationTaskDB.id == task_id).first()
        
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        if task.status in [TaskStatusEnum.COMPLETED, TaskStatusEnum.FAILED, TaskStatusEnum.CANCELLED]:
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot cancel task with status: {task.status.value}"
            )
        
        # Update task status
        task.status = TaskStatusEnum.CANCELLED
        task.completed_at = datetime.utcnow()
        
        db.commit()
        
        logger.info(f"Cancelled task {task_id}")
        
        return {"message": f"Task {task_id} cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling task {task_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/queue/{task_id}")
async def delete_task(task_id: str, db: Session = Depends(get_db)):
    """
    Delete a task from the queue (only completed, failed, or cancelled tasks)
    """
    try:
        task = db.query(GenerationTaskDB).filter(GenerationTaskDB.id == task_id).first()
        
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        if task.status in [TaskStatusEnum.PENDING, TaskStatusEnum.PROCESSING]:
            raise HTTPException(
                status_code=400, 
                detail="Cannot delete active task. Cancel it first."
            )
        
        # Clean up associated files
        if task.image_path and os.path.exists(task.image_path):
            try:
                os.remove(task.image_path)
            except Exception as e:
                logger.warning(f"Could not remove image file {task.image_path}: {e}")
        
        if task.output_path and os.path.exists(task.output_path):
            try:
                os.remove(task.output_path)
            except Exception as e:
                logger.warning(f"Could not remove output file {task.output_path}: {e}")
        
        # Delete task from database
        db.delete(task)
        db.commit()
        
        logger.info(f"Deleted task {task_id}")
        
        return {"message": f"Task {task_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting task {task_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/queue/clear")
async def clear_completed_tasks(db: Session = Depends(get_db)):
    """
    Clear all completed, failed, and cancelled tasks from the queue
    """
    try:
        # Get tasks to clear
        tasks_to_clear = db.query(GenerationTaskDB).filter(
            GenerationTaskDB.status.in_([
                TaskStatusEnum.COMPLETED,
                TaskStatusEnum.FAILED,
                TaskStatusEnum.CANCELLED
            ])
        ).all()
        
        cleared_count = 0
        for task in tasks_to_clear:
            # Clean up associated files
            if task.image_path and os.path.exists(task.image_path):
                try:
                    os.remove(task.image_path)
                except Exception as e:
                    logger.warning(f"Could not remove image file {task.image_path}: {e}")
            
            db.delete(task)
            cleared_count += 1
        
        db.commit()
        
        logger.info(f"Cleared {cleared_count} completed tasks from queue")
        
        return {"message": f"Cleared {cleared_count} completed tasks"}
        
    except Exception as e:
        logger.error(f"Error clearing completed tasks: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/queue/poll")
async def poll_queue_updates(
    since: Optional[str] = None,
    active_only: bool = True,
    db: Session = Depends(get_db)
):
    """
    Optimized endpoint for HTTP polling (every 5 seconds)
    Returns only active tasks and recent updates to minimize bandwidth
    """
    try:
        query = db.query(GenerationTaskDB)
        
        # Filter for active tasks if requested
        if active_only:
            query = query.filter(
                GenerationTaskDB.status.in_([
                    TaskStatusEnum.PENDING,
                    TaskStatusEnum.PROCESSING
                ])
            )
        
        # Order by creation time
        query = query.order_by(desc(GenerationTaskDB.created_at))
        
        # Limit to recent tasks for polling efficiency
        tasks = query.limit(50).all()
        
        # Get summary counts
        pending_count = db.query(GenerationTaskDB).filter(
            GenerationTaskDB.status == TaskStatusEnum.PENDING
        ).count()
        
        processing_count = db.query(GenerationTaskDB).filter(
            GenerationTaskDB.status == TaskStatusEnum.PROCESSING
        ).count()
        
        # Convert to lightweight response format
        task_updates = []
        for task in tasks:
            task_update = {
                "id": task.id,
                "status": task.status.value,
                "progress": task.progress,
                "model_type": task.model_type.value,
                "prompt": task.prompt[:100] + "..." if len(task.prompt) > 100 else task.prompt,
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "error_message": task.error_message
            }
            task_updates.append(task_update)
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "pending_count": pending_count,
            "processing_count": processing_count,
            "active_tasks": task_updates,
            "has_active_tasks": pending_count > 0 or processing_count > 0
        }
        
    except Exception as e:
        logger.error(f"Error in queue polling: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")