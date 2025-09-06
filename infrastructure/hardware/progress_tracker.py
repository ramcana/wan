"""
Progress Tracker for Wan2.2 Video Generation
Provides real-time progress tracking with detailed statistics during video generation
"""

import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from enum import Enum
import json

logger = logging.getLogger(__name__)

class GenerationPhase(Enum):
    """Phases of video generation"""
    INITIALIZATION = "initialization"
    MODEL_LOADING = "model_loading"
    PREPROCESSING = "preprocessing"
    GENERATION = "generation"
    POSTPROCESSING = "postprocessing"
    ENCODING = "encoding"
    COMPLETION = "completion"

@dataclass
class ProgressData:
    """Progress data structure for generation tracking"""
    current_step: int = 0
    total_steps: int = 0
    progress_percentage: float = 0.0
    elapsed_time: float = 0.0
    estimated_remaining: float = 0.0
    current_phase: str = GenerationPhase.INITIALIZATION.value
    frames_processed: int = 0
    processing_speed: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_utilization_percent: float = 0.0
    phase_start_time: float = field(default_factory=time.time)
    phase_progress: float = 0.0

@dataclass
class GenerationStats:
    """Comprehensive generation statistics"""
    start_time: datetime = field(default_factory=datetime.now)
    current_time: datetime = field(default_factory=datetime.now)
    elapsed_seconds: float = 0.0
    estimated_total_seconds: float = 0.0
    current_step: int = 0
    total_steps: int = 0
    current_phase: str = GenerationPhase.INITIALIZATION.value
    frames_processed: int = 0
    frames_per_second: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_utilization_percent: float = 0.0
    phase_durations: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

class ProgressTracker:
    """Real-time progress tracker for video generation"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.current_task: Optional[str] = None
        self.progress_data: Optional[ProgressData] = None
        self.generation_stats: Optional[GenerationStats] = None
        self.update_callbacks: List[Callable] = []
        self.is_tracking: bool = False
        self.update_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Configuration
        self.update_interval = self.config.get("progress_update_interval", 2.0)  # seconds
        self.enable_system_monitoring = self.config.get("enable_system_monitoring", True)
        
        logger.info("Progress tracker initialized")
    
    def add_update_callback(self, callback: Callable[[ProgressData], None]):
        """Add a callback function to receive progress updates"""
        self.update_callbacks.append(callback)
    
    def start_progress_tracking(self, task_id: str, total_steps: int = 100) -> None:
        """Initialize progress monitoring for a generation task"""
        logger.info(f"Starting progress tracking for task: {task_id}")
        
        self.current_task = task_id
        self.progress_data = ProgressData(
            total_steps=total_steps,
            current_phase=GenerationPhase.INITIALIZATION.value
        )
        self.generation_stats = GenerationStats(
            total_steps=total_steps
        )
        self.is_tracking = True
        self.stop_event.clear()
        
        # Start update thread
        self.update_thread = threading.Thread(
            target=self._update_loop,
            daemon=True
        )
        self.update_thread.start()
        
        self._notify_callbacks()
    
    def update_progress(self, step: int, phase: Optional[str] = None, 
                       frames_processed: Optional[int] = None,
                       additional_data: Optional[Dict[str, Any]] = None) -> None:
        """Update progress with current step and optional additional data"""
        if not self.is_tracking or not self.progress_data:
            return
        
        current_time = time.time()
        
        # Update basic progress
        self.progress_data.current_step = step
        self.progress_data.progress_percentage = (step / self.progress_data.total_steps) * 100
        self.progress_data.elapsed_time = current_time - self.generation_stats.start_time.timestamp()
        
        # Update phase if provided
        if phase and phase != self.progress_data.current_phase:
            self._update_phase(phase)
        
        # Update frames processed
        if frames_processed is not None:
            self.progress_data.frames_processed = frames_processed
            
            # Calculate processing speed
            if self.progress_data.elapsed_time > 0:
                self.progress_data.processing_speed = frames_processed / self.progress_data.elapsed_time
        
        # Calculate ETA
        if step > 0 and self.progress_data.elapsed_time > 0:
            time_per_step = self.progress_data.elapsed_time / step
            remaining_steps = self.progress_data.total_steps - step
            self.progress_data.estimated_remaining = remaining_steps * time_per_step
        
        # Update generation stats
        self._update_generation_stats(additional_data)
        
        # Notify callbacks
        self._notify_callbacks()
    
    def update_phase(self, phase: str, phase_progress: float = 0.0) -> None:
        """Update the current generation phase"""
        if not self.is_tracking or not self.progress_data:
            return
        
        self._update_phase(phase, phase_progress)
        self._notify_callbacks()
    
    def complete_progress_tracking(self, final_stats: Optional[Dict[str, Any]] = None) -> GenerationStats:
        """Complete progress tracking and return final statistics"""
        if not self.is_tracking:
            return self.generation_stats or GenerationStats()
        
        logger.info(f"Completing progress tracking for task: {self.current_task}")
        
        # Update final progress state before stopping tracking
        if self.progress_data:
            self.progress_data.progress_percentage = 100.0
            self.progress_data.current_step = self.progress_data.total_steps
            self.progress_data.current_phase = GenerationPhase.COMPLETION.value
            self.progress_data.estimated_remaining = 0.0
        
        # Update final statistics
        if self.generation_stats:
            self.generation_stats.current_time = datetime.now()
            self.generation_stats.elapsed_seconds = (
                self.generation_stats.current_time - self.generation_stats.start_time
            ).total_seconds()
            
            if final_stats:
                self.generation_stats.performance_metrics.update(final_stats)
        
        # Final callback notification before stopping
        self._notify_callbacks()
        
        # Now stop tracking
        self.is_tracking = False
        self.stop_event.set()
        
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=1.0)
        
        return self.generation_stats or GenerationStats()
    
    def get_progress_html(self) -> str:
        """Generate HTML for progress display"""
        if not self.progress_data or not self.generation_stats:
            return ""
        
        # Allow HTML generation even when not actively tracking (for final display)
        if not self.is_tracking and self.progress_data.progress_percentage < 100.0:
            return ""
        
        # Calculate display values
        elapsed_str = self._format_duration(self.progress_data.elapsed_time)
        eta_str = self._format_duration(self.progress_data.estimated_remaining)
        
        # Phase display
        phase_display = self.progress_data.current_phase.replace('_', ' ').title()
        
        # Progress bar HTML
        progress_html = f"""
        <div style="
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h4 style="margin: 0; color: #495057;">🎬 Generation Progress</h4>
                <span style="
                    background: #007bff;
                    color: white;
                    padding: 4px 12px;
                    border-radius: 12px;
                    font-size: 0.9em;
                    font-weight: bold;
                ">
                    {self.progress_data.progress_percentage:.1f}%
                </span>
            </div>
            
            <!-- Progress Bar -->
            <div style="
                background: #e9ecef;
                border-radius: 10px;
                height: 20px;
                margin-bottom: 15px;
                overflow: hidden;
                position: relative;
            ">
                <div style="
                    background: linear-gradient(90deg, #007bff, #0056b3);
                    height: 100%;
                    width: {self.progress_data.progress_percentage}%;
                    transition: width 0.3s ease;
                    border-radius: 10px;
                "></div>
                <div style="
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    color: #495057;
                    font-size: 0.8em;
                    font-weight: bold;
                    text-shadow: 1px 1px 2px rgba(255,255,255,0.8);
                ">
                    Step {self.progress_data.current_step} / {self.progress_data.total_steps}
                </div>
            </div>
            
            <!-- Statistics Grid -->
            <div style="
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 15px;
            ">
                <div style="text-align: center;">
                    <div style="font-size: 0.8em; color: #6c757d; margin-bottom: 5px;">Current Phase</div>
                    <div style="font-weight: bold; color: #495057;">{phase_display}</div>
                </div>
                
                <div style="text-align: center;">
                    <div style="font-size: 0.8em; color: #6c757d; margin-bottom: 5px;">Elapsed Time</div>
                    <div style="font-weight: bold; color: #495057;">{elapsed_str}</div>
                </div>
                
                <div style="text-align: center;">
                    <div style="font-size: 0.8em; color: #6c757d; margin-bottom: 5px;">Estimated Remaining</div>
                    <div style="font-weight: bold; color: #495057;">{eta_str}</div>
                </div>
                
                <div style="text-align: center;">
                    <div style="font-size: 0.8em; color: #6c757d; margin-bottom: 5px;">Frames Processed</div>
                    <div style="font-weight: bold; color: #495057;">{self.progress_data.frames_processed}</div>
                </div>
            </div>
            
            <!-- Performance Metrics -->
            <div style="
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 10px;
                padding-top: 15px;
                border-top: 1px solid #dee2e6;
            ">
                <div style="text-align: center;">
                    <div style="font-size: 0.7em; color: #6c757d;">Processing Speed</div>
                    <div style="font-size: 0.9em; font-weight: bold; color: #28a745;">
                        {self.progress_data.processing_speed:.2f} fps
                    </div>
                </div>
                
                <div style="text-align: center;">
                    <div style="font-size: 0.7em; color: #6c757d;">Memory Usage</div>
                    <div style="font-size: 0.9em; font-weight: bold; color: #17a2b8;">
                        {self.progress_data.memory_usage_mb:.0f} MB
                    </div>
                </div>
                
                <div style="text-align: center;">
                    <div style="font-size: 0.7em; color: #6c757d;">GPU Utilization</div>
                    <div style="font-size: 0.9em; font-weight: bold; color: #fd7e14;">
                        {self.progress_data.gpu_utilization_percent:.0f}%
                    </div>
                </div>
            </div>
        </div>
        """
        
        return progress_html
    
    def _update_phase(self, phase: str, phase_progress: float = 0.0) -> None:
        """Update the current phase and track phase duration"""
        if not self.progress_data or not self.generation_stats:
            return
        
        current_time = time.time()
        
        # Record duration of previous phase
        if self.progress_data.current_phase != GenerationPhase.INITIALIZATION.value:
            phase_duration = current_time - self.progress_data.phase_start_time
            self.generation_stats.phase_durations[self.progress_data.current_phase] = phase_duration
        
        # Update to new phase
        self.progress_data.current_phase = phase
        self.progress_data.phase_start_time = current_time
        self.progress_data.phase_progress = phase_progress
        
        logger.info(f"Generation phase updated to: {phase}")
    
    def _update_generation_stats(self, additional_data: Optional[Dict[str, Any]] = None) -> None:
        """Update comprehensive generation statistics"""
        if not self.generation_stats or not self.progress_data:
            return
        
        current_time = datetime.now()
        
        # Update basic stats
        self.generation_stats.current_time = current_time
        self.generation_stats.elapsed_seconds = (
            current_time - self.generation_stats.start_time
        ).total_seconds()
        self.generation_stats.current_step = self.progress_data.current_step
        self.generation_stats.current_phase = self.progress_data.current_phase
        self.generation_stats.frames_processed = self.progress_data.frames_processed
        self.generation_stats.frames_per_second = self.progress_data.processing_speed
        
        # Update system metrics if monitoring is enabled
        if self.enable_system_monitoring:
            self._update_system_metrics()
        
        # Add additional data if provided
        if additional_data:
            self.generation_stats.performance_metrics.update(additional_data)
    
    def _update_system_metrics(self) -> None:
        """Update system performance metrics"""
        try:
            import psutil
            import torch
            
            # Memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            self.progress_data.memory_usage_mb = memory_info.rss / 1024 / 1024
            self.generation_stats.memory_usage_mb = self.progress_data.memory_usage_mb
            
            # GPU utilization (if available)
            if torch.cuda.is_available():
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.progress_data.gpu_utilization_percent = utilization.gpu
                    self.generation_stats.gpu_utilization_percent = utilization.gpu
                except Exception:
                    pass  # GPU monitoring not available
                    
        except ImportError:
            pass  # System monitoring dependencies not available
    
    def _update_loop(self) -> None:
        """Background thread for periodic updates"""
        while not self.stop_event.wait(self.update_interval):
            if self.is_tracking:
                self._update_generation_stats()
                self._notify_callbacks()
    
    def _notify_callbacks(self) -> None:
        """Notify all registered callbacks of progress updates"""
        if not self.progress_data:
            return
        
        for callback in self.update_callbacks:
            try:
                callback(self.progress_data)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to human-readable string"""
        if seconds <= 0:
            return "0s"
        
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

# Global progress tracker instance
_progress_tracker = None

def get_progress_tracker(config: Optional[Dict[str, Any]] = None) -> ProgressTracker:
    """Get the global progress tracker instance"""
    global _progress_tracker
    if _progress_tracker is None or config is not None:
        _progress_tracker = ProgressTracker(config)
    return _progress_tracker

def create_progress_callback(tracker: ProgressTracker) -> Callable:
    """Create a progress callback function for use with generation functions"""
    def progress_callback(step: int, total_steps: int, **kwargs):
        """Progress callback for generation functions"""
        tracker.update_progress(
            step=step,
            phase=kwargs.get('phase'),
            frames_processed=kwargs.get('frames_processed'),
            additional_data=kwargs
        )
    
    return progress_callback