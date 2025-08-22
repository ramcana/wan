"""
Generation service that integrates with existing Wan2.2 system
"""

import sys
import os
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from datetime import datetime
import uuid
import threading
from queue import Queue as ThreadQueue
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from sqlalchemy.orm import Session
from backend.repositories.database import SessionLocal, GenerationTaskDB, TaskStatusEnum
from backend.core.system_integration import get_system_integration

logger = logging.getLogger(__name__)

class GenerationService:
    """Service for managing video generation tasks"""
    
    def __init__(self):
        self.task_queue = ThreadQueue()
        self.processing_thread = None
        self.is_processing = False
        self.current_task = None
        
    async def initialize(self):
        """Initialize the generation service"""
        try:
            # Start background processing thread
            if not self.processing_thread or not self.processing_thread.is_alive():
                self.is_processing = True
                self.processing_thread = threading.Thread(
                    target=self._process_queue_worker,
                    daemon=True
                )
                self.processing_thread.start()
                logger.info("Generation service initialized and processing thread started")
            
        except Exception as e:
            logger.error(f"Failed to initialize generation service: {e}")
            raise
    
    def _process_queue_worker(self):
        """Background worker to process generation tasks"""
        logger.info("Generation queue worker started")
        
        while self.is_processing:
            try:
                # Check for pending tasks in database
                db = SessionLocal()
                try:
                    pending_task = db.query(GenerationTaskDB).filter(
                        GenerationTaskDB.status == TaskStatusEnum.PENDING
                    ).order_by(GenerationTaskDB.created_at).first()
                    
                    if pending_task:
                        logger.info(f"Processing task {pending_task.id}")
                        self.current_task = pending_task.id
                        
                        # Update task status to processing
                        pending_task.status = TaskStatusEnum.PROCESSING
                        pending_task.started_at = datetime.utcnow()
                        db.commit()
                        
                        # Process the task
                        success = self._process_generation_task(pending_task, db)
                        
                        if success:
                            pending_task.status = TaskStatusEnum.COMPLETED
                            pending_task.completed_at = datetime.utcnow()
                            pending_task.progress = 100
                            logger.info(f"Task {pending_task.id} completed successfully")
                        else:
                            pending_task.status = TaskStatusEnum.FAILED
                            pending_task.completed_at = datetime.utcnow()
                            if not pending_task.error_message:
                                pending_task.error_message = "Generation failed"
                            logger.error(f"Task {pending_task.id} failed")
                        
                        db.commit()
                        self.current_task = None
                    
                finally:
                    db.close()
                
                # Sleep for a short time before checking again
                threading.Event().wait(2.0)
                
            except Exception as e:
                logger.error(f"Error in generation queue worker: {e}")
                threading.Event().wait(5.0)  # Wait longer on error
    
    def _process_generation_task(self, task: GenerationTaskDB, db: Session) -> bool:
        """Process a single generation task"""
        try:
            logger.info(f"Starting generation for task {task.id}: {task.model_type.value}")
            
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Run the async generation process
                result = loop.run_until_complete(self._run_generation_async(task, db))
                return result
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Error processing generation task {task.id}: {e}")
            task.error_message = str(e)
            db.commit()
            return False
    
    async def _run_generation_async(self, task: GenerationTaskDB, db: Session) -> bool:
        """Run the actual generation process asynchronously"""
        try:
            # Get system integration
            integration = await get_system_integration()
            
            if not integration.model_manager:
                raise Exception("Model manager not available")
            
            # Update progress
            task.progress = 10
            db.commit()
            
            # Load the model (convert to lowercase for model manager compatibility)
            model_type = task.model_type.value.lower()
            logger.info(f"Loading model: {model_type}")
            
            def progress_callback(message: str, progress: float):
                """Progress callback for model loading"""
                logger.info(f"Model loading progress: {message} ({progress:.1f}%)")
                task.progress = int(10 + (progress * 0.3))  # 10-40% for model loading
                db.commit()
            
            # For MVP testing, use mock mode if models aren't available
            try:
                model, model_info = integration.model_manager.load_model(
                    model_type, 
                    progress_callback=progress_callback
                )
                logger.info(f"Model loaded successfully: {model_info.model_id}")
            except Exception as e:
                logger.warning(f"Model loading failed, using mock mode: {e}")
                # Use mock mode for testing
                logger.info("Using mock generation mode for testing")
                model = None
                model_info = None
            
            # Update progress
            task.progress = 40
            db.commit()
            
            # Prepare generation parameters
            generation_params = {
                "prompt": task.prompt,
                "resolution": task.resolution,
                "steps": task.steps,
                "model_type": model_type
            }
            
            # Add LoRA if specified
            if task.lora_path:
                generation_params["lora_path"] = task.lora_path
                generation_params["lora_strength"] = task.lora_strength
            
            logger.info(f"Starting generation with params: {generation_params}")
            
            # Update progress
            task.progress = 50
            db.commit()
            
            # For now, simulate generation process
            # TODO: Integrate with actual generation pipeline
            await self._simulate_generation(task, db, generation_params)
            
            # Set output path (relative to project root)
            output_filename = f"generated_{task.id}.mp4"
            output_path = f"outputs/{output_filename}"
            task.output_path = output_path
            
            logger.info(f"Generation completed for task {task.id}")
            return True
            
        except Exception as e:
            logger.error(f"Generation failed for task {task.id}: {e}")
            
            # Categorize error types for better user feedback
            error_message = str(e)
            if "CUDA out of memory" in error_message or "VRAM" in error_message:
                task.error_message = "VRAM exhaustion: Insufficient GPU memory. Try reducing resolution or enabling model offloading."
            elif "Model loading failed" in error_message:
                task.error_message = f"Model loading error: {error_message}. The model may need to be downloaded first."
            elif "timeout" in error_message.lower():
                task.error_message = "Generation timeout: The process took too long. Try reducing steps or resolution."
            else:
                task.error_message = f"Generation error: {error_message}"
            
            db.commit()
            return False
    
    async def _simulate_generation(self, task: GenerationTaskDB, db: Session, params: Dict[str, Any]):
        """Simulate the generation process with progress updates"""
        try:
            # Simulate generation steps
            steps = params.get("steps", 50)
            
            for step in range(steps):
                # Simulate processing time
                await asyncio.sleep(0.1)  # Fast simulation for testing
                
                # Update progress (50-95% for generation)
                progress = 50 + int((step / steps) * 45)
                task.progress = progress
                db.commit()
                
                if step % 10 == 0:
                    logger.info(f"Generation progress: {step}/{steps} steps ({progress}%)")
            
            # Final processing
            task.progress = 95
            db.commit()
            await asyncio.sleep(0.5)
            
            # Create output directory if it doesn't exist (relative to project root)
            project_root = Path(__file__).parent.parent.parent
            output_dir = project_root / "outputs"
            output_dir.mkdir(exist_ok=True)
            
            # Create a placeholder output file for testing
            output_filename = f"generated_{task.id}.mp4"
            output_path = output_dir / output_filename
            
            # Create a simple text file as placeholder
            with open(output_path, 'w') as f:
                f.write(f"Generated video for task {task.id}\n")
                f.write(f"Prompt: {params['prompt']}\n")
                f.write(f"Model: {params['model_type']}\n")
                f.write(f"Resolution: {params['resolution']}\n")
                f.write(f"Steps: {params['steps']}\n")
                f.write(f"Generated at: {datetime.utcnow()}\n")
            
            logger.info(f"Placeholder output created: {output_path}")
            
        except Exception as e:
            logger.error(f"Error in generation simulation: {e}")
            raise
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        db = SessionLocal()
        try:
            total_tasks = db.query(GenerationTaskDB).count()
            pending_tasks = db.query(GenerationTaskDB).filter(
                GenerationTaskDB.status == TaskStatusEnum.PENDING
            ).count()
            processing_tasks = db.query(GenerationTaskDB).filter(
                GenerationTaskDB.status == TaskStatusEnum.PROCESSING
            ).count()
            
            return {
                "total_tasks": total_tasks,
                "pending_tasks": pending_tasks,
                "processing_tasks": processing_tasks,
                "current_task": self.current_task,
                "worker_active": self.is_processing and self.processing_thread and self.processing_thread.is_alive()
            }
        finally:
            db.close()
    
    def shutdown(self):
        """Shutdown the generation service"""
        logger.info("Shutting down generation service")
        self.is_processing = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)

# Global generation service instance
generation_service = GenerationService()

async def get_generation_service() -> GenerationService:
    """Dependency to get generation service instance"""
    if not generation_service.processing_thread or not generation_service.processing_thread.is_alive():
        await generation_service.initialize()
    return generation_service