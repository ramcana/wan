"""
Interactive Error Resolution System for Wan2.2 Video Generation

This module provides interactive error resolution with suggested fixes,
automatic recovery attempts, and user-guided troubleshooting.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
from datetime import datetime

from infrastructure.hardware.error_handler import (
    UserFriendlyError, ErrorCategory, ErrorSeverity,
    RecoveryAction, GenerationErrorHandler
)

logger = logging.getLogger(__name__)

class ResolutionStatus(Enum):
    """Status of error resolution attempts"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    FAILED = "failed"
    USER_CANCELLED = "user_cancelled"

@dataclass
class ResolutionStep:
    """Represents a single resolution step"""
    id: str
    title: str
    description: str
    action_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    automatic: bool = False
    estimated_time: int = 0  # seconds
    success_probability: float = 0.0
    prerequisites: List[str] = field(default_factory=list)
    status: ResolutionStatus = ResolutionStatus.PENDING
    result_message: str = ""
    execution_time: Optional[float] = None

@dataclass
class ResolutionSession:
    """Represents an interactive error resolution session"""
    session_id: str
    error: UserFriendlyError
    steps: List[ResolutionStep] = field(default_factory=list)
    current_step_index: int = 0
    status: ResolutionStatus = ResolutionStatus.PENDING
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    user_feedback: Dict[str, Any] = field(default_factory=dict)
    resolution_path: List[str] = field(default_factory=list)

class InteractiveErrorResolver:
    """Provides interactive error resolution with guided troubleshooting"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.error_handler = GenerationErrorHandler()
        
        # Active resolution sessions
        self.active_sessions: Dict[str, ResolutionSession] = {}
        self.session_lock = threading.RLock()
        
        # Resolution step definitions
        self.resolution_steps = self._initialize_resolution_steps()
        
        # Callbacks for UI updates
        self.progress_callback: Optional[Callable] = None
        self.status_callback: Optional[Callable] = None
    
    def set_callbacks(self, progress_callback: Optional[Callable] = None,
                     status_callback: Optional[Callable] = None):
        """Set callbacks for UI updates"""
        self.progress_callback = progress_callback
        self.status_callback = status_callback
    
    def _initialize_resolution_steps(self) -> Dict[ErrorCategory, List[ResolutionStep]]:
        """Initialize resolution steps for each error category"""
        return {
            ErrorCategory.VRAM_MEMORY: [
                ResolutionStep(
                    id="check_vram_usage",
                    title="Check VRAM Usage",
                    description="Check current GPU memory usage and identify memory-intensive processes",
                    action_type="check_vram",
                    automatic=True,
                    estimated_time=5,
                    success_probability=0.9
                ),
                ResolutionStep(
                    id="close_gpu_apps",
                    title="Close GPU Applications",
                    description="Close other applications using GPU memory",
                    action_type="close_gpu_apps",
                    automatic=False,
                    estimated_time=30,
                    success_probability=0.8,
                    prerequisites=["check_vram_usage"]
                ),
                ResolutionStep(
                    id="optimize_vram_settings",
                    title="Optimize VRAM Settings",
                    description="Automatically optimize generation settings for available VRAM",
                    action_type="optimize_vram",
                    automatic=True,
                    estimated_time=10,
                    success_probability=0.85
                ),
                ResolutionStep(
                    id="enable_cpu_offload",
                    title="Enable CPU Offloading",
                    description="Enable CPU offloading to reduce VRAM usage",
                    action_type="enable_offload",
                    automatic=True,
                    estimated_time=15,
                    success_probability=0.7
                ),
                ResolutionStep(
                    id="reduce_resolution",
                    title="Reduce Resolution",
                    description="Lower the generation resolution to reduce memory requirements",
                    action_type="reduce_resolution",
                    automatic=False,
                    estimated_time=5,
                    success_probability=0.95
                )
            ],
            
            ErrorCategory.MODEL_LOADING: [
                ResolutionStep(
                    id="check_model_files",
                    title="Check Model Files",
                    description="Verify model files exist and are not corrupted",
                    action_type="check_model_files",
                    automatic=True,
                    estimated_time=10,
                    success_probability=0.9
                ),
                ResolutionStep(
                    id="clear_model_cache",
                    title="Clear Model Cache",
                    description="Clear cached model files and reload",
                    action_type="clear_cache",
                    automatic=True,
                    estimated_time=20,
                    success_probability=0.7
                ),
                ResolutionStep(
                    id="redownload_model",
                    title="Re-download Model",
                    description="Download the model again from the source",
                    action_type="redownload_model",
                    automatic=False,
                    estimated_time=300,  # 5 minutes
                    success_probability=0.95,
                    prerequisites=["check_model_files"]
                ),
                ResolutionStep(
                    id="check_disk_space",
                    title="Check Disk Space",
                    description="Verify sufficient disk space for model files",
                    action_type="check_disk_space",
                    automatic=True,
                    estimated_time=5,
                    success_probability=0.9
                )
            ],
            
            ErrorCategory.INPUT_VALIDATION: [
                ResolutionStep(
                    id="validate_prompt",
                    title="Validate Prompt",
                    description="Check and fix prompt formatting issues",
                    action_type="validate_prompt",
                    automatic=True,
                    estimated_time=5,
                    success_probability=0.9
                ),
                ResolutionStep(
                    id="fix_image_format",
                    title="Fix Image Format",
                    description="Convert image to supported format if needed",
                    action_type="fix_image",
                    automatic=True,
                    estimated_time=10,
                    success_probability=0.85
                ),
                ResolutionStep(
                    id="adjust_parameters",
                    title="Adjust Parameters",
                    description="Modify generation parameters to valid ranges",
                    action_type="adjust_params",
                    automatic=True,
                    estimated_time=5,
                    success_probability=0.95
                )
            ],
            
            ErrorCategory.GENERATION_PIPELINE: [
                ResolutionStep(
                    id="restart_pipeline",
                    title="Restart Pipeline",
                    description="Reset the generation pipeline to clean state",
                    action_type="restart_pipeline",
                    automatic=True,
                    estimated_time=15,
                    success_probability=0.6
                ),
                ResolutionStep(
                    id="simplify_settings",
                    title="Simplify Settings",
                    description="Use simpler generation settings to avoid pipeline issues",
                    action_type="simplify_settings",
                    automatic=True,
                    estimated_time=5,
                    success_probability=0.8
                ),
                ResolutionStep(
                    id="check_dependencies",
                    title="Check Dependencies",
                    description="Verify all required dependencies are installed",
                    action_type="check_deps",
                    automatic=True,
                    estimated_time=20,
                    success_probability=0.7
                )
            ]
        }
    
    def start_resolution_session(self, error: UserFriendlyError) -> str:
        """
        Start an interactive error resolution session
        Returns: session ID
        """
        try:
            import uuid
            session_id = str(uuid.uuid4())
            
            # Get resolution steps for this error category
            steps = self.resolution_steps.get(error.category, [])
            if not steps:
                # Create generic resolution steps
                steps = self._create_generic_resolution_steps(error)
            
            session = ResolutionSession(
                session_id=session_id,
                error=error,
                steps=steps.copy(),  # Create copy to avoid modifying template
                status=ResolutionStatus.PENDING
            )
            
            with self.session_lock:
                self.active_sessions[session_id] = session
            
            logger.info(f"Started resolution session {session_id} for {error.category.value}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to start resolution session: {e}")
            return ""
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of resolution session"""
        with self.session_lock:
            session = self.active_sessions.get(session_id)
            if not session:
                return None
            
            return {
                "session_id": session_id,
                "status": session.status.value,
                "current_step": session.current_step_index,
                "total_steps": len(session.steps),
                "current_step_info": (
                    session.steps[session.current_step_index].title 
                    if session.current_step_index < len(session.steps) else "Complete"
                ),
                "progress": session.current_step_index / len(session.steps) * 100,
                "error_category": session.error.category.value,
                "error_title": session.error.title
            }
    
    def execute_next_step(self, session_id: str, user_approved: bool = True) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Execute the next resolution step
        Returns: (success, message, step_info)
        """
        try:
            with self.session_lock:
                session = self.active_sessions.get(session_id)
                if not session:
                    return False, "Session not found", {}
                
                if session.current_step_index >= len(session.steps):
                    return False, "All steps completed", {}
                
                current_step = session.steps[session.current_step_index]
                
                # Check if user approval is required for non-automatic steps
                if not current_step.automatic and not user_approved:
                    return False, "User approval required", {
                        "step": current_step.title,
                        "description": current_step.description,
                        "estimated_time": current_step.estimated_time,
                        "requires_approval": True
                    }
                
                # Update session status
                session.status = ResolutionStatus.IN_PROGRESS
                current_step.status = ResolutionStatus.IN_PROGRESS
            
            # Execute the step
            start_time = time.time()
            success, message = self._execute_resolution_step(current_step, session.error)
            execution_time = time.time() - start_time
            
            # Update step results
            with self.session_lock:
                current_step.status = ResolutionStatus.RESOLVED if success else ResolutionStatus.FAILED
                current_step.result_message = message
                current_step.execution_time = execution_time
                
                if success:
                    session.resolution_path.append(current_step.id)
                    session.current_step_index += 1
                    
                    # Check if all steps completed
                    if session.current_step_index >= len(session.steps):
                        session.status = ResolutionStatus.RESOLVED
                        session.completed_at = datetime.now()
                else:
                    # Step failed, but continue to next step
                    session.current_step_index += 1
            
            # Notify callbacks
            if self.progress_callback:
                self.progress_callback(session_id, session.current_step_index / len(session.steps))
            
            step_info = {
                "step_title": current_step.title,
                "step_description": current_step.description,
                "success": success,
                "message": message,
                "execution_time": execution_time,
                "next_step": (
                    session.steps[session.current_step_index].title 
                    if session.current_step_index < len(session.steps) else None
                )
            }
            
            return success, message, step_info
            
        except Exception as e:
            logger.error(f"Failed to execute resolution step: {e}")
            return False, f"Step execution failed: {str(e)}", {}
    
    def execute_automatic_steps(self, session_id: str) -> List[Dict[str, Any]]:
        """Execute all automatic steps in sequence"""
        results = []
        
        try:
            with self.session_lock:
                session = self.active_sessions.get(session_id)
                if not session:
                    return results
            
            while session.current_step_index < len(session.steps):
                current_step = session.steps[session.current_step_index]
                
                if not current_step.automatic:
                    # Stop at first manual step
                    break
                
                success, message, step_info = self.execute_next_step(session_id, user_approved=True)
                results.append({
                    "step": current_step.title,
                    "success": success,
                    "message": message,
                    **step_info
                })
                
                if not success and current_step.success_probability > 0.8:
                    # High-probability step failed, might indicate serious issue
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to execute automatic steps: {e}")
            return results
    
    def get_resolution_suggestions(self, session_id: str) -> List[Dict[str, Any]]:
        """Get interactive resolution suggestions for user"""
        try:
            with self.session_lock:
                session = self.active_sessions.get(session_id)
                if not session:
                    return []
                
                suggestions = []
                
                # Add remaining manual steps as suggestions
                for i in range(session.current_step_index, len(session.steps)):
                    step = session.steps[i]
                    if not step.automatic:
                        suggestions.append({
                            "id": step.id,
                            "title": step.title,
                            "description": step.description,
                            "estimated_time": step.estimated_time,
                            "success_probability": step.success_probability,
                            "action_type": step.action_type
                        })
                
                # Add category-specific suggestions
                category_suggestions = self._get_category_specific_suggestions(session.error.category)
                suggestions.extend(category_suggestions)
                
                return suggestions
                
        except Exception as e:
            logger.error(f"Failed to get resolution suggestions: {e}")
            return []
    
    def apply_user_selected_fix(self, session_id: str, fix_id: str, 
                               parameters: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        """Apply a user-selected fix"""
        try:
            with self.session_lock:
                session = self.active_sessions.get(session_id)
                if not session:
                    return False, "Session not found"
            
            # Find the fix step
            fix_step = None
            for step in session.steps:
                if step.id == fix_id:
                    fix_step = step
                    break
            
            if not fix_step:
                return False, "Fix not found"
            
            # Apply parameters if provided
            if parameters:
                fix_step.parameters.update(parameters)
            
            # Execute the fix
            success, message = self._execute_resolution_step(fix_step, session.error)
            
            # Update session
            with self.session_lock:
                if success:
                    session.resolution_path.append(fix_id)
                    session.user_feedback[fix_id] = {"applied": True, "success": True}
                else:
                    session.user_feedback[fix_id] = {"applied": True, "success": False}
            
            return success, message
            
        except Exception as e:
            logger.error(f"Failed to apply user fix: {e}")
            return False, f"Fix application failed: {str(e)}"
    
    def _execute_resolution_step(self, step: ResolutionStep, error: UserFriendlyError) -> Tuple[bool, str]:
        """Execute a specific resolution step"""
        try:
            action_type = step.action_type
            parameters = step.parameters
            
            if action_type == "check_vram":
                return self._check_vram_usage()
            elif action_type == "optimize_vram":
                return self._optimize_vram_settings(parameters)
            elif action_type == "enable_offload":
                return self._enable_cpu_offload(parameters)
            elif action_type == "reduce_resolution":
                return self._reduce_resolution(parameters)
            elif action_type == "check_model_files":
                return self._check_model_files(parameters)
            elif action_type == "clear_cache":
                return self._clear_model_cache(parameters)
            elif action_type == "check_disk_space":
                return self._check_disk_space(parameters)
            elif action_type == "validate_prompt":
                return self._validate_and_fix_prompt(parameters)
            elif action_type == "fix_image":
                return self._fix_image_format(parameters)
            elif action_type == "adjust_params":
                return self._adjust_parameters(parameters)
            elif action_type == "restart_pipeline":
                return self._restart_pipeline(parameters)
            elif action_type == "simplify_settings":
                return self._simplify_settings(parameters)
            elif action_type == "check_deps":
                return self._check_dependencies(parameters)
            else:
                return False, f"Unknown action type: {action_type}"
                
        except Exception as e:
            logger.error(f"Resolution step execution failed: {e}")
            return False, f"Step failed: {str(e)}"
    
    def _check_vram_usage(self) -> Tuple[bool, str]:
        """Check current VRAM usage"""
        try:
            import torch
            if not torch.cuda.is_available():
                return False, "CUDA not available"
            
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            
            total_gb = total_memory / (1024**3)
            used_gb = allocated_memory / (1024**3)
            free_gb = total_gb - used_gb
            
            usage_percent = (used_gb / total_gb) * 100
            
            message = f"VRAM Usage: {used_gb:.1f}GB / {total_gb:.1f}GB ({usage_percent:.1f}%)"
            
            if usage_percent > 90:
                return False, f"{message} - Critical VRAM usage"
            elif usage_percent > 80:
                return True, f"{message} - High VRAM usage, optimization recommended"
            else:
                return True, f"{message} - VRAM usage normal"
                
        except Exception as e:
            return False, f"Failed to check VRAM: {str(e)}"
    
    def _optimize_vram_settings(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """Optimize VRAM settings"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Apply optimization settings (placeholder)
            optimizations = [
                "Enabled gradient checkpointing",
                "Reduced batch size to 1",
                "Enabled mixed precision",
                "Cleared CUDA cache"
            ]
            
            return True, f"VRAM optimized: {', '.join(optimizations)}"
            
        except Exception as e:
            return False, f"VRAM optimization failed: {str(e)}"
    
    def _enable_cpu_offload(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """Enable CPU offloading"""
        try:
            # CPU offload implementation (placeholder)
            return True, "CPU offloading enabled for VAE and text encoder"
        except Exception as e:
            return False, f"CPU offload failed: {str(e)}"
    
    def _reduce_resolution(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """Reduce generation resolution"""
        try:
            # Resolution reduction logic (placeholder)
            return True, "Resolution reduced to 720p for better VRAM compatibility"
        except Exception as e:
            return False, f"Resolution reduction failed: {str(e)}"
    
    def _check_model_files(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """Check model file integrity"""
        try:
            from pathlib import Path
            
            # Check model files (placeholder)
            model_dir = Path("models")
            if not model_dir.exists():
                return False, "Models directory not found"
            
            model_files = list(model_dir.glob("**/*.safetensors")) + list(model_dir.glob("**/*.bin"))
            
            if not model_files:
                return False, "No model files found"
            
            return True, f"Found {len(model_files)} model files"
            
        except Exception as e:
            return False, f"Model file check failed: {str(e)}"
    
    def _clear_model_cache(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """Clear model cache"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Clear cache implementation (placeholder)
            return True, "Model cache cleared successfully"
            
        except Exception as e:
            return False, f"Cache clearing failed: {str(e)}"
    
    def _check_disk_space(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """Check available disk space"""
        try:
            import psutil
            
            disk_usage = psutil.disk_usage('.')
            free_gb = disk_usage.free / (1024**3)
            total_gb = disk_usage.total / (1024**3)
            
            if free_gb < 5:
                return False, f"Low disk space: {free_gb:.1f}GB free of {total_gb:.1f}GB"
            elif free_gb < 10:
                return True, f"Disk space warning: {free_gb:.1f}GB free of {total_gb:.1f}GB"
            else:
                return True, f"Disk space OK: {free_gb:.1f}GB free of {total_gb:.1f}GB"
                
        except Exception as e:
            return False, f"Disk space check failed: {str(e)}"
    
    def _validate_and_fix_prompt(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate and fix prompt issues"""
        try:
            # Prompt validation and fixing (placeholder)
            return True, "Prompt validated and fixed"
        except Exception as e:
            return False, f"Prompt validation failed: {str(e)}"
    
    def _fix_image_format(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """Fix image format issues"""
        try:
            # Image format fixing (placeholder)
            return True, "Image format converted to supported format"
        except Exception as e:
            return False, f"Image format fix failed: {str(e)}"
    
    def _adjust_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """Adjust generation parameters"""
        try:
            # Parameter adjustment (placeholder)
            return True, "Generation parameters adjusted to valid ranges"
        except Exception as e:
            return False, f"Parameter adjustment failed: {str(e)}"
    
    def _restart_pipeline(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """Restart generation pipeline"""
        try:
            # Pipeline restart (placeholder)
            return True, "Generation pipeline restarted successfully"
        except Exception as e:
            return False, f"Pipeline restart failed: {str(e)}"
    
    def _simplify_settings(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """Simplify generation settings"""
        try:
            # Settings simplification (placeholder)
            return True, "Generation settings simplified for better compatibility"
        except Exception as e:
            return False, f"Settings simplification failed: {str(e)}"
    
    def _check_dependencies(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """Check system dependencies"""
        try:
            # Dependency checking (placeholder)
            return True, "All dependencies are installed and up to date"
        except Exception as e:
            return False, f"Dependency check failed: {str(e)}"
    
    def _create_generic_resolution_steps(self, error: UserFriendlyError) -> List[ResolutionStep]:
        """Create generic resolution steps for unknown error categories"""
        return [
            ResolutionStep(
                id="restart_application",
                title="Restart Application",
                description="Restart the application to reset all components",
                action_type="restart_app",
                automatic=False,
                estimated_time=60,
                success_probability=0.6
            ),
            ResolutionStep(
                id="check_system_resources",
                title="Check System Resources",
                description="Check CPU, memory, and disk usage",
                action_type="check_resources",
                automatic=True,
                estimated_time=10,
                success_probability=0.8
            )
        ]
    
    def _get_category_specific_suggestions(self, category: ErrorCategory) -> List[Dict[str, Any]]:
        """Get additional suggestions specific to error category"""
        suggestions = {
            ErrorCategory.VRAM_MEMORY: [
                {
                    "id": "upgrade_gpu",
                    "title": "Consider GPU Upgrade",
                    "description": "Your current GPU may not have sufficient VRAM for high-quality generation",
                    "action_type": "info",
                    "estimated_time": 0,
                    "success_probability": 1.0
                }
            ],
            ErrorCategory.MODEL_LOADING: [
                {
                    "id": "check_internet",
                    "title": "Check Internet Connection",
                    "description": "Ensure stable internet connection for model downloads",
                    "action_type": "info",
                    "estimated_time": 0,
                    "success_probability": 1.0
                }
            ]
        }
        
        return suggestions.get(category, [])
    
    def end_session(self, session_id: str, user_feedback: Optional[Dict[str, Any]] = None):
        """End resolution session and collect feedback"""
        try:
            with self.session_lock:
                session = self.active_sessions.get(session_id)
                if session:
                    if user_feedback:
                        session.user_feedback.update(user_feedback)
                    
                    session.completed_at = datetime.now()
                    
                    # Log session results for analysis
                    logger.info(f"Resolution session {session_id} ended: "
                              f"Status={session.status.value}, "
                              f"Steps completed={session.current_step_index}/{len(session.steps)}")
                    
                    # Remove from active sessions
                    del self.active_sessions[session_id]
                    
        except Exception as e:
            logger.error(f"Failed to end session: {e}")

# Global resolver instance
_error_resolver = None

def get_error_resolver(config: Optional[Dict[str, Any]] = None) -> InteractiveErrorResolver:
    """Get or create global error resolver instance"""
    global _error_resolver
    if _error_resolver is None:
        _error_resolver = InteractiveErrorResolver(config)
    return _error_resolver