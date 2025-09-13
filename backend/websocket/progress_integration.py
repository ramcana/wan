"""
WebSocket Progress Integration for Real AI Model Generation
Provides detailed progress tracking and real-time updates for generation pipeline
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, Callable
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class GenerationStage(Enum):
    """Stages of video generation process"""
    INITIALIZING = "initializing"
    LOADING_MODEL = "loading_model"
    DOWNLOADING_MODEL = "downloading_model"
    PREPARING_INPUTS = "preparing_inputs"
    APPLYING_LORA = "applying_lora"
    GENERATING = "generating"
    POST_PROCESSING = "post_processing"
    SAVING = "saving"
    COMPLETED = "completed"
    FAILED = "failed"

class ProgressIntegration:
    """
    Integration class for sending detailed progress updates via WebSocket
    Connects the real generation pipeline with the WebSocket manager
    """
    
    def __init__(self, websocket_manager=None):
        """
        Initialize progress integration
        
        Args:
            websocket_manager: WebSocket connection manager instance
        """
        self.websocket_manager = websocket_manager
        self.logger = logging.getLogger(__name__ + ".ProgressIntegration")
        
        # Progress tracking state
        self._current_task_id: Optional[str] = None
        self._current_stage: Optional[GenerationStage] = None
        self._stage_start_time: Optional[float] = None
        self._generation_start_time: Optional[float] = None
        self._last_progress_update: Optional[float] = None
        
        # Stage progress tracking
        self._stage_progress: Dict[GenerationStage, int] = {}
        self._stage_messages: Dict[GenerationStage, str] = {}
        
        # VRAM monitoring
        self._vram_monitoring_active = False
        self._vram_monitoring_task: Optional[asyncio.Task] = None
        
        self.logger.info("ProgressIntegration initialized")
    
    async def initialize(self):
        """Initialize the progress integration system"""
        try:
            if self.websocket_manager is None:
                from backend.websocket.manager import get_connection_manager
                self.websocket_manager = get_connection_manager()
                self.logger.info("WebSocket manager initialized")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize progress integration: {e}")
            return False
    
    async def start_generation_tracking(self, task_id: str, model_type: str, 
                                      estimated_duration: Optional[float] = None):
        """
        Start tracking progress for a generation task
        
        Args:
            task_id: Unique task identifier
            model_type: Type of model being used (t2v, i2v, ti2v)
            estimated_duration: Estimated generation duration in seconds
        """
        try:
            self._current_task_id = task_id
            self._generation_start_time = time.time()
            self._last_progress_update = self._generation_start_time
            
            # Reset stage tracking
            self._stage_progress.clear()
            self._stage_messages.clear()
            
            # Send initial progress update
            await self._send_generation_start_notification(task_id, model_type, estimated_duration)
            
            # Start VRAM monitoring for this generation
            await self._start_vram_monitoring(task_id)
            
            self.logger.info(f"Started generation tracking for task {task_id} with model {model_type}")
            
        except Exception as e:
            self.logger.error(f"Failed to start generation tracking: {e}")
    
    async def update_stage_progress(self, stage: GenerationStage, progress: int, 
                                  message: str, **kwargs):
        """
        Update progress for a specific generation stage
        
        Args:
            stage: Current generation stage
            progress: Progress percentage (0-100)
            message: Descriptive message for current progress
            **kwargs: Additional metadata
        """
        try:
            if not self._current_task_id:
                self.logger.warning("No active task for progress update")
                return
            
            # Update stage tracking
            if self._current_stage != stage:
                self._current_stage = stage
                self._stage_start_time = time.time()
                await self._send_stage_change_notification(stage, message)
            
            self._stage_progress[stage] = progress
            self._stage_messages[stage] = message
            
            # Calculate estimated time remaining
            estimated_time_remaining = self._calculate_estimated_time_remaining(progress)
            
            # Send detailed progress update
            await self._send_detailed_progress_update(
                stage, progress, message, estimated_time_remaining, **kwargs
            )
            
            self._last_progress_update = time.time()
            
        except Exception as e:
            self.logger.error(f"Failed to update stage progress: {e}")
    
    async def update_model_loading_progress(self, model_type: str, progress: int, 
                                          status: str, **kwargs):
        """
        Update model loading progress
        
        Args:
            model_type: Type of model being loaded
            progress: Loading progress percentage
            status: Current loading status
            **kwargs: Additional metadata
        """
        try:
            if not self._current_task_id or not self.websocket_manager:
                return
            
            await self.websocket_manager.send_model_loading_progress(
                self._current_task_id, model_type, progress, status, **kwargs
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update model loading progress: {e}")
    
    async def update_generation_step_progress(self, current_step: int, total_steps: int, 
                                            stage_message: Optional[str] = None):
        """
        Update progress for individual generation steps
        
        Args:
            current_step: Current generation step
            total_steps: Total number of generation steps
            stage_message: Optional message for this stage
        """
        try:
            if not self._current_task_id:
                return
            
            # Calculate step progress percentage
            step_progress = int((current_step / total_steps) * 100) if total_steps > 0 else 0
            
            # Create message
            message = stage_message or f"Generating frame {current_step}/{total_steps}"
            
            # Send step progress update
            await self.update_stage_progress(
                GenerationStage.GENERATING,
                step_progress,
                message,
                current_step=current_step,
                total_steps=total_steps
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update generation step progress: {e}")
    
    async def complete_generation_tracking(self, success: bool, output_path: Optional[str] = None,
                                         error_message: Optional[str] = None):
        """
        Complete generation tracking and send final status
        
        Args:
            success: Whether generation completed successfully
            output_path: Path to generated output file
            error_message: Error message if generation failed
        """
        try:
            if not self._current_task_id:
                return
            
            # Calculate total generation time
            total_time = time.time() - self._generation_start_time if self._generation_start_time else 0
            
            # Determine final stage
            final_stage = GenerationStage.COMPLETED if success else GenerationStage.FAILED
            final_progress = 100 if success else 0
            
            # Create final message
            if success:
                final_message = f"Generation completed successfully in {total_time:.1f}s"
            else:
                final_message = f"Generation failed: {error_message or 'Unknown error'}"
            
            # Send final progress update
            await self.update_stage_progress(
                final_stage,
                final_progress,
                final_message,
                output_path=output_path,
                total_generation_time=total_time,
                error_message=error_message
            )
            
            # Stop VRAM monitoring
            await self._stop_vram_monitoring()
            
            # Reset tracking state
            self._current_task_id = None
            self._current_stage = None
            self._generation_start_time = None
            self._stage_start_time = None
            
            self.logger.info(f"Completed generation tracking: {final_message}")
            
        except Exception as e:
            self.logger.error(f"Failed to complete generation tracking: {e}")
    
    async def _send_generation_start_notification(self, task_id: str, model_type: str,
                                                estimated_duration: Optional[float]):
        """Send notification that generation has started"""
        try:
            if not self.websocket_manager:
                return
            
            await self.websocket_manager.send_generation_stage_notification(
                task_id,
                "generation_started",
                0,
                model_type=model_type,
                estimated_duration=estimated_duration,
                start_time=datetime.utcnow().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to send generation start notification: {e}")
    
    async def _send_stage_change_notification(self, stage: GenerationStage, message: str):
        """Send notification when generation stage changes"""
        try:
            if not self.websocket_manager or not self._current_task_id:
                return
            
            await self.websocket_manager.send_generation_stage_notification(
                self._current_task_id,
                stage.value,
                0,
                stage_message=message,
                stage_start_time=datetime.utcnow().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to send stage change notification: {e}")
    
    async def _send_detailed_progress_update(self, stage: GenerationStage, progress: int,
                                           message: str, estimated_time_remaining: Optional[float],
                                           **kwargs):
        """Send detailed progress update with all metadata"""
        try:
            if not self.websocket_manager or not self._current_task_id:
                return
            
            await self.websocket_manager.send_detailed_generation_progress(
                self._current_task_id,
                stage.value,
                progress,
                message,
                estimated_time_remaining=estimated_time_remaining,
                stage_start_time=self._stage_start_time,
                generation_start_time=self._generation_start_time,
                **kwargs
            )
            
        except Exception as e:
            self.logger.error(f"Failed to send detailed progress update: {e}")
    
    def _calculate_estimated_time_remaining(self, current_progress: int) -> Optional[float]:
        """Calculate estimated time remaining based on current progress"""
        try:
            if not self._generation_start_time or current_progress <= 0:
                return None
            
            elapsed_time = time.time() - self._generation_start_time
            
            # Estimate based on current progress
            if current_progress >= 100:
                return 0.0
            
            estimated_total_time = (elapsed_time / current_progress) * 100
            estimated_remaining = estimated_total_time - elapsed_time
            
            return max(0.0, estimated_remaining)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate estimated time remaining: {e}")
            return None
    
    async def _start_vram_monitoring(self, task_id: str):
        """Start real-time VRAM monitoring for the generation task"""
        try:
            if self._vram_monitoring_active:
                return
            
            self._vram_monitoring_active = True
            self._vram_monitoring_task = asyncio.create_task(
                self._vram_monitoring_loop(task_id)
            )
            
            self.logger.info(f"Started VRAM monitoring for task {task_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to start VRAM monitoring: {e}")
    
    async def _stop_vram_monitoring(self):
        """Stop VRAM monitoring"""
        try:
            self._vram_monitoring_active = False
            
            if self._vram_monitoring_task and not self._vram_monitoring_task.done():
                self._vram_monitoring_task.cancel()
                try:
                    await self._vram_monitoring_task
                except asyncio.CancelledError:
                    pass
            
            self._vram_monitoring_task = None
            self.logger.info("Stopped VRAM monitoring")
            
        except Exception as e:
            self.logger.error(f"Failed to stop VRAM monitoring: {e}")
    
    async def _vram_monitoring_loop(self, task_id: str):
        """Background loop for VRAM monitoring during generation"""
        try:
            while self._vram_monitoring_active:
                try:
                    # Get current VRAM usage
                    vram_stats = await self._get_current_vram_stats()
                    
                    if vram_stats and self.websocket_manager:
                        # Send VRAM monitoring update
                        await self.websocket_manager.send_vram_monitoring_update(
                            vram_stats,
                            task_id=task_id,
                            monitoring_type="generation_vram"
                        )
                    
                    # Update every 1 second during generation
                    await asyncio.sleep(1.0)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.warning(f"Error in VRAM monitoring loop: {e}")
                    await asyncio.sleep(2.0)
                    
        except Exception as e:
            self.logger.error(f"VRAM monitoring loop failed: {e}")
    
    async def _get_current_vram_stats(self) -> Optional[Dict[str, Any]]:
        """Get current VRAM statistics"""
        try:
            import torch
            if not torch.cuda.is_available():
                return None
            
            device = torch.cuda.current_device()
            allocated_bytes = torch.cuda.memory_allocated(device)
            reserved_bytes = torch.cuda.memory_reserved(device)
            total_bytes = torch.cuda.get_device_properties(device).total_memory
            
            allocated_mb = allocated_bytes / (1024 * 1024)
            reserved_mb = reserved_bytes / (1024 * 1024)
            total_mb = total_bytes / (1024 * 1024)
            free_mb = total_mb - allocated_mb
            
            allocated_percent = (allocated_mb / total_mb) * 100
            
            # Determine warning level
            warning_level = "normal"
            if allocated_percent > 90:
                warning_level = "critical"
            elif allocated_percent > 75:
                warning_level = "warning"
            
            return {
                "allocated_mb": round(allocated_mb, 1),
                "reserved_mb": round(reserved_mb, 1),
                "free_mb": round(free_mb, 1),
                "total_mb": round(total_mb, 1),
                "allocated_percent": round(allocated_percent, 1),
                "warning_level": warning_level,
                "device_name": torch.cuda.get_device_name(device),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to get VRAM stats: {e}")
            return None

# Global progress integration instance
_progress_integration: Optional[ProgressIntegration] = None

async def get_progress_integration() -> ProgressIntegration:
    """Get or create the global progress integration instance"""
    global _progress_integration
    
    if _progress_integration is None:
        _progress_integration = ProgressIntegration()
        await _progress_integration.initialize()
    
    return _progress_integration

def create_progress_callback(task_id: str) -> Callable:
    """
    Create a progress callback function for use with generation pipelines
    
    Args:
        task_id: Task ID for progress tracking
        
    Returns:
        Async callback function that can be used by generation pipelines
    """
    async def progress_callback(current_step: int, total_steps: int, **kwargs):
        """Progress callback for generation steps"""
        try:
            progress_integration = await get_progress_integration()
            await progress_integration.update_generation_step_progress(
                current_step, total_steps, kwargs.get('stage_message')
            )
        except Exception as e:
            logger.error(f"Error in progress callback: {e}")
    
    return progress_callback
