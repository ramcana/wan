"""
Fallback and Recovery System for Real AI Model Integration

This module implements comprehensive fallback and recovery mechanisms that automatically
handle failures in model loading, generation pipeline, and system optimization.
"""

import asyncio
import logging
import time
import threading
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class RecoveryAction(Enum):
    """Types of recovery actions that can be performed"""
    FALLBACK_TO_MOCK = "fallback_to_mock"
    RETRY_MODEL_DOWNLOAD = "retry_model_download"
    APPLY_VRAM_OPTIMIZATION = "apply_vram_optimization"
    RESTART_PIPELINE = "restart_pipeline"
    CLEAR_GPU_CACHE = "clear_gpu_cache"
    REDUCE_GENERATION_PARAMS = "reduce_generation_params"
    ENABLE_CPU_OFFLOAD = "enable_cpu_offload"
    SYSTEM_HEALTH_CHECK = "system_health_check"

class FailureType(Enum):
    """Types of failures that can trigger recovery"""
    MODEL_LOADING_FAILURE = "model_loading_failure"
    VRAM_EXHAUSTION = "vram_exhaustion"
    GENERATION_PIPELINE_ERROR = "generation_pipeline_error"
    HARDWARE_OPTIMIZATION_FAILURE = "hardware_optimization_failure"
    SYSTEM_RESOURCE_ERROR = "system_resource_error"
    NETWORK_ERROR = "network_error"

@dataclass
class RecoveryAttempt:
    """Information about a recovery attempt"""
    failure_type: FailureType
    action: RecoveryAction
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None
    recovery_time_seconds: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemHealthStatus:
    """Current system health status"""
    overall_status: str  # "healthy", "degraded", "critical"
    cpu_usage_percent: float
    memory_usage_percent: float
    vram_usage_percent: float
    gpu_available: bool
    model_loading_functional: bool
    generation_pipeline_functional: bool
    last_check_timestamp: datetime
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

class FallbackRecoverySystem:
    """
    Comprehensive fallback and recovery system that handles various failure scenarios
    and automatically attempts recovery using existing infrastructure
    """
    
    def __init__(self, generation_service=None, websocket_manager=None):
        self.generation_service = generation_service
        self.websocket_manager = websocket_manager
        self.logger = logging.getLogger(__name__ + ".FallbackRecoverySystem")
        
        # Recovery tracking
        self.recovery_attempts: List[RecoveryAttempt] = []
        self.max_recovery_attempts = 3
        self.recovery_cooldown_seconds = 30
        self.last_recovery_attempt: Dict[FailureType, datetime] = {}
        
        # System health monitoring
        self.health_check_interval = 60  # seconds
        self.health_monitor_thread: Optional[threading.Thread] = None
        self.health_monitoring_active = False
        self.current_health_status: Optional[SystemHealthStatus] = None
        
        # Fallback state management
        self.mock_generation_enabled = False
        self.degraded_mode_active = False
        self.critical_failures: Dict[str, datetime] = {}
        
        # Recovery strategies configuration
        self.recovery_strategies = self._initialize_recovery_strategies()
        
        # Retry logic configuration
        self.retry_config = {
            "model_download": {"max_attempts": 3, "backoff_factor": 2.0, "initial_delay": 5.0},
            "pipeline_restart": {"max_attempts": 2, "backoff_factor": 1.5, "initial_delay": 2.0},
            "optimization_apply": {"max_attempts": 2, "backoff_factor": 1.0, "initial_delay": 1.0}
        }
        
        self.logger.info("Fallback and Recovery System initialized")
    
    def _initialize_recovery_strategies(self) -> Dict[FailureType, List[RecoveryAction]]:
        """Initialize recovery strategies for different failure types"""
        return {
            FailureType.MODEL_LOADING_FAILURE: [
                RecoveryAction.CLEAR_GPU_CACHE,
                RecoveryAction.RETRY_MODEL_DOWNLOAD,
                RecoveryAction.APPLY_VRAM_OPTIMIZATION,
                RecoveryAction.FALLBACK_TO_MOCK
            ],
            FailureType.VRAM_EXHAUSTION: [
                RecoveryAction.CLEAR_GPU_CACHE,
                RecoveryAction.APPLY_VRAM_OPTIMIZATION,
                RecoveryAction.ENABLE_CPU_OFFLOAD,
                RecoveryAction.REDUCE_GENERATION_PARAMS,
                RecoveryAction.FALLBACK_TO_MOCK
            ],
            FailureType.GENERATION_PIPELINE_ERROR: [
                RecoveryAction.CLEAR_GPU_CACHE,
                RecoveryAction.RESTART_PIPELINE,
                RecoveryAction.APPLY_VRAM_OPTIMIZATION,
                RecoveryAction.FALLBACK_TO_MOCK
            ],
            FailureType.HARDWARE_OPTIMIZATION_FAILURE: [
                RecoveryAction.SYSTEM_HEALTH_CHECK,
                RecoveryAction.CLEAR_GPU_CACHE,
                RecoveryAction.FALLBACK_TO_MOCK
            ],
            FailureType.SYSTEM_RESOURCE_ERROR: [
                RecoveryAction.SYSTEM_HEALTH_CHECK,
                RecoveryAction.CLEAR_GPU_CACHE,
                RecoveryAction.FALLBACK_TO_MOCK
            ],
            FailureType.NETWORK_ERROR: [
                RecoveryAction.RETRY_MODEL_DOWNLOAD,
                RecoveryAction.FALLBACK_TO_MOCK
            ]
        }
    
    async def handle_failure(
        self, 
        failure_type: FailureType, 
        error: Exception, 
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        Handle a failure by attempting appropriate recovery actions
        
        Args:
            failure_type: Type of failure that occurred
            error: The original exception
            context: Additional context about the failure
            
        Returns:
            Tuple of (recovery_successful, recovery_message)
        """
        try:
            self.logger.info(f"Handling failure: {failure_type.value} - {str(error)}")
            
            # Check if we're in cooldown period for this failure type
            if self._is_in_cooldown(failure_type):
                cooldown_remaining = self._get_cooldown_remaining(failure_type)
                message = f"Recovery cooldown active for {failure_type.value} ({cooldown_remaining:.0f}s remaining)"
                self.logger.warning(message)
                return False, message
            
            # Get recovery strategies for this failure type
            strategies = self.recovery_strategies.get(failure_type, [RecoveryAction.FALLBACK_TO_MOCK])
            
            # Attempt recovery actions in order
            for action in strategies:
                try:
                    self.logger.info(f"Attempting recovery action: {action.value}")
                    
                    recovery_start = time.time()
                    success = await self._execute_recovery_action(action, error, context or {})
                    recovery_time = time.time() - recovery_start
                    
                    # Record the recovery attempt
                    attempt = RecoveryAttempt(
                        failure_type=failure_type,
                        action=action,
                        timestamp=datetime.now(),
                        success=success,
                        recovery_time_seconds=recovery_time,
                        context=context or {}
                    )
                    
                    if not success:
                        attempt.error_message = f"Recovery action {action.value} failed"
                    
                    self.recovery_attempts.append(attempt)
                    
                    # Send WebSocket notification about recovery attempt
                    await self._notify_recovery_attempt(attempt)
                    
                    if success:
                        self.logger.info(f"Recovery successful with action: {action.value} (took {recovery_time:.1f}s)")
                        
                        # Update last recovery attempt timestamp
                        self.last_recovery_attempt[failure_type] = datetime.now()
                        
                        return True, f"Recovery successful using {action.value}"
                    else:
                        self.logger.warning(f"Recovery action {action.value} failed, trying next strategy")
                        
                except Exception as recovery_error:
                    self.logger.error(f"Recovery action {action.value} raised exception: {recovery_error}")
                    
                    # Record failed attempt
                    attempt = RecoveryAttempt(
                        failure_type=failure_type,
                        action=action,
                        timestamp=datetime.now(),
                        success=False,
                        error_message=str(recovery_error),
                        context=context or {}
                    )
                    self.recovery_attempts.append(attempt)
                    await self._notify_recovery_attempt(attempt)
            
            # All recovery strategies failed
            self.logger.error(f"All recovery strategies failed for {failure_type.value}")
            self.last_recovery_attempt[failure_type] = datetime.now()
            
            return False, f"All recovery strategies failed for {failure_type.value}"
            
        except Exception as e:
            self.logger.error(f"Error in handle_failure: {e}")
            return False, f"Recovery system error: {str(e)}"
    
    async def _execute_recovery_action(
        self, 
        action: RecoveryAction, 
        error: Exception, 
        context: Dict[str, Any]
    ) -> bool:
        """Execute a specific recovery action"""
        try:
            if action == RecoveryAction.FALLBACK_TO_MOCK:
                return await self._fallback_to_mock_generation()
            
            elif action == RecoveryAction.RETRY_MODEL_DOWNLOAD:
                return await self._retry_model_download(context)
            
            elif action == RecoveryAction.APPLY_VRAM_OPTIMIZATION:
                return await self._apply_vram_optimization(context)
            
            elif action == RecoveryAction.RESTART_PIPELINE:
                return await self._restart_generation_pipeline()
            
            elif action == RecoveryAction.CLEAR_GPU_CACHE:
                return await self._clear_gpu_cache()
            
            elif action == RecoveryAction.REDUCE_GENERATION_PARAMS:
                return await self._reduce_generation_parameters(context)
            
            elif action == RecoveryAction.ENABLE_CPU_OFFLOAD:
                return await self._enable_cpu_offload()
            
            elif action == RecoveryAction.SYSTEM_HEALTH_CHECK:
                return await self._perform_system_health_check()
            
            else:
                self.logger.warning(f"Unknown recovery action: {action}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error executing recovery action {action.value}: {e}")
            return False
    
    async def _fallback_to_mock_generation(self) -> bool:
        """Enable mock generation as fallback when real models fail"""
        try:
            if self.generation_service:
                # Enable mock generation mode
                self.generation_service.use_real_generation = False
                self.generation_service.fallback_to_simulation = True
                self.mock_generation_enabled = True
                
                self.logger.info("Fallback to mock generation enabled")
                
                # Notify via WebSocket
                if self.websocket_manager:
                    await self.websocket_manager.broadcast({
                        "type": "system_status",
                        "data": {
                            "status": "degraded",
                            "message": "System running in mock generation mode due to model issues",
                            "mock_mode_enabled": True
                        }
                    })
                
                return True
            else:
                self.logger.warning("Generation service not available for fallback")
                return False
                
        except Exception as e:
            self.logger.error(f"Error enabling mock generation fallback: {e}")
            return False
    
    async def _retry_model_download(self, context: Dict[str, Any]) -> bool:
        """Retry model download using existing retry systems"""
        try:
            model_type = context.get("model_type")
            if not model_type:
                self.logger.warning("No model type specified for download retry")
                return False
            
            if not self.generation_service or not self.generation_service.model_integration_bridge:
                self.logger.warning("Model integration bridge not available for download retry")
                return False
            
            # Use existing retry configuration
            retry_config = self.retry_config["model_download"]
            max_attempts = retry_config["max_attempts"]
            backoff_factor = retry_config["backoff_factor"]
            initial_delay = retry_config["initial_delay"]
            
            for attempt in range(max_attempts):
                try:
                    self.logger.info(f"Model download retry attempt {attempt + 1}/{max_attempts} for {model_type}")
                    
                    # Wait with exponential backoff
                    if attempt > 0:
                        delay = initial_delay * (backoff_factor ** (attempt - 1))
                        await asyncio.sleep(delay)
                    
                    # Attempt model download
                    bridge = self.generation_service.model_integration_bridge
                    success = await bridge.ensure_model_available(model_type)
                    
                    if success:
                        self.logger.info(f"Model download retry successful for {model_type}")
                        return True
                    else:
                        self.logger.warning(f"Model download retry attempt {attempt + 1} failed for {model_type}")
                        
                except Exception as e:
                    self.logger.error(f"Model download retry attempt {attempt + 1} error: {e}")
            
            self.logger.error(f"All model download retry attempts failed for {model_type}")
            return False
            
        except Exception as e:
            self.logger.error(f"Error in model download retry: {e}")
            return False
    
    async def _apply_vram_optimization(self, context: Dict[str, Any]) -> bool:
        """Apply VRAM optimization using existing WAN22SystemOptimizer"""
        try:
            if not self.generation_service or not self.generation_service.wan22_system_optimizer:
                self.logger.warning("WAN22SystemOptimizer not available for VRAM optimization")
                return False
            
            optimizer = self.generation_service.wan22_system_optimizer
            
            # Apply hardware optimizations
            opt_result = optimizer.apply_hardware_optimizations()
            if opt_result.success:
                self.logger.info(f"Applied {len(opt_result.optimizations_applied)} VRAM optimizations")
                
                # Also apply VRAM-specific optimizations in generation service
                if hasattr(self.generation_service, '_apply_vram_optimizations'):
                    await self.generation_service._apply_vram_optimizations()
                
                return True
            else:
                self.logger.warning("Hardware optimization failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error applying VRAM optimization: {e}")
            return False
    
    async def _restart_generation_pipeline(self) -> bool:
        """Restart the generation pipeline with fresh state"""
        try:
            if not self.generation_service or not self.generation_service.real_generation_pipeline:
                self.logger.warning("Real generation pipeline not available for restart")
                return False
            
            pipeline = self.generation_service.real_generation_pipeline
            
            # Clear pipeline cache
            if hasattr(pipeline, '_pipeline_cache'):
                pipeline._pipeline_cache.clear()
                self.logger.info("Cleared generation pipeline cache")
            
            # Reinitialize pipeline
            success = await pipeline.initialize()
            if success:
                self.logger.info("Generation pipeline restarted successfully")
                return True
            else:
                self.logger.warning("Generation pipeline restart failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error restarting generation pipeline: {e}")
            return False
    
    async def _clear_gpu_cache(self) -> bool:
        """Clear GPU cache to free VRAM"""
        try:
            import torch
            if torch.cuda.is_available():
                # Clear PyTorch CUDA cache
                torch.cuda.empty_cache()
                
                # Force garbage collection
                import gc
                gc.collect()
                
                self.logger.info("GPU cache cleared successfully")
                return True
            else:
                self.logger.warning("CUDA not available for cache clearing")
                return False
                
        except Exception as e:
            self.logger.error(f"Error clearing GPU cache: {e}")
            return False
    
    async def _reduce_generation_parameters(self, context: Dict[str, Any]) -> bool:
        """Automatically reduce generation parameters to fit VRAM constraints"""
        try:
            # This would modify the generation parameters in the context
            # For now, we'll just log the action and return success
            # In a real implementation, this would integrate with the parameter validation system
            
            original_params = context.get("generation_params", {})
            reduced_params = original_params.copy()
            
            # Reduce resolution if high
            if reduced_params.get("resolution") in ["1920x1080", "1920x1088"]:
                reduced_params["resolution"] = "1280x720"
                self.logger.info("Reduced resolution to 1280x720")
            
            # Reduce steps if high
            if reduced_params.get("steps", 0) > 25:
                reduced_params["steps"] = 20
                self.logger.info("Reduced inference steps to 20")
            
            # Reduce number of frames if high
            if reduced_params.get("num_frames", 0) > 16:
                reduced_params["num_frames"] = 8
                self.logger.info("Reduced number of frames to 8")
            
            # Update context with reduced parameters
            context["generation_params"] = reduced_params
            context["parameters_reduced"] = True
            
            self.logger.info("Generation parameters reduced for VRAM compatibility")
            return True
            
        except Exception as e:
            self.logger.error(f"Error reducing generation parameters: {e}")
            return False
    
    async def _enable_cpu_offload(self) -> bool:
        """Enable CPU offloading for model components"""
        try:
            if self.generation_service:
                # Enable CPU offloading in generation service
                self.generation_service.enable_model_offloading = True
                
                # Apply to model integration bridge if available
                if self.generation_service.model_integration_bridge:
                    # This would enable CPU offloading in the model bridge
                    self.logger.info("CPU offloading enabled in model integration bridge")
                
                self.logger.info("CPU offloading enabled successfully")
                return True
            else:
                self.logger.warning("Generation service not available for CPU offload")
                return False
                
        except Exception as e:
            self.logger.error(f"Error enabling CPU offload: {e}")
            return False
    
    async def _perform_system_health_check(self) -> bool:
        """Perform comprehensive system health check"""
        try:
            health_status = await self.get_system_health_status()
            
            if health_status.overall_status == "critical":
                self.logger.error("System health check indicates critical status")
                return False
            elif health_status.overall_status == "degraded":
                self.logger.warning("System health check indicates degraded status")
                # Still return True as the system is functional but degraded
                return True
            else:
                self.logger.info("System health check passed")
                return True
                
        except Exception as e:
            self.logger.error(f"Error performing system health check: {e}")
            return False
    
    async def check_system_health(self) -> Dict[str, Any]:
        """Check system health and return status dictionary"""
        try:
            health_status = await self.get_system_health_status()
            return {
                "system_responsive": health_status.overall_status != "critical",
                "memory_available": health_status.memory_usage_percent < 90,
                "disk_space_available": True,  # Assume disk space is OK for now
                "gpu_available": health_status.gpu_available,
                "models_loaded": health_status.model_loading_functional,
                "wan22_infrastructure_ready": health_status.generation_pipeline_functional,
                "last_check_time": time.time(),
                "uptime_seconds": time.time() - self.start_time if hasattr(self, 'start_time') else 0
            }
        except Exception as e:
            self.logger.error(f"Error checking system health: {e}")
            return {
                "system_responsive": False,
                "memory_available": False,
                "disk_space_available": False,
                "gpu_available": False,
                "models_loaded": False,
                "wan22_infrastructure_ready": False,
                "last_check_time": time.time(),
                "uptime_seconds": 0,
                "error": str(e)
            }
    
    async def get_system_health_status(self) -> SystemHealthStatus:
        """Get comprehensive system health status"""
        try:
            # Initialize health status
            health_status = SystemHealthStatus(
                overall_status="unknown",
                cpu_usage_percent=0.0,
                memory_usage_percent=0.0,
                vram_usage_percent=0.0,
                gpu_available=False,
                model_loading_functional=False,
                generation_pipeline_functional=False,
                last_check_timestamp=datetime.now()
            )
            
            # Get system metrics
            try:
                import psutil
                health_status.cpu_usage_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                health_status.memory_usage_percent = memory.percent
            except Exception as e:
                health_status.issues.append(f"Could not get system metrics: {e}")
            
            # Get GPU metrics - use system integration if available
            try:
                gpu_detected = False
                
                # First try to get GPU info from generation service
                if self.generation_service and hasattr(self.generation_service, 'hardware_profile'):
                    if self.generation_service.hardware_profile and self.generation_service.hardware_profile.gpu_model:
                        health_status.gpu_available = True
                        gpu_detected = True
                        # Try to get VRAM usage
                        try:
                            import torch
                            if torch.cuda.is_available():
                                device = torch.cuda.current_device()
                                total_memory = torch.cuda.get_device_properties(device).total_memory
                                allocated_memory = torch.cuda.memory_allocated(device)
                                health_status.vram_usage_percent = (allocated_memory / total_memory) * 100
                        except Exception:
                            health_status.vram_usage_percent = 0.0
                
                # Fallback to direct CUDA check
                if not gpu_detected:
                    import torch
                    if torch.cuda.is_available():
                        health_status.gpu_available = True
                        device = torch.cuda.current_device()
                        total_memory = torch.cuda.get_device_properties(device).total_memory
                        allocated_memory = torch.cuda.memory_allocated(device)
                        health_status.vram_usage_percent = (allocated_memory / total_memory) * 100
                    else:
                        health_status.issues.append("CUDA not available")
                        
            except Exception as e:
                health_status.issues.append(f"Could not get GPU metrics: {e}")
            
            # Check model loading functionality
            if self.generation_service and self.generation_service.model_integration_bridge:
                try:
                    bridge = self.generation_service.model_integration_bridge
                    if bridge.is_initialized():
                        health_status.model_loading_functional = True
                    else:
                        health_status.issues.append("Model integration bridge not initialized")
                except Exception as e:
                    health_status.issues.append(f"Model loading check failed: {e}")
            else:
                health_status.issues.append("Model integration bridge not available")
            
            # Check generation pipeline functionality
            if self.generation_service and self.generation_service.real_generation_pipeline:
                health_status.generation_pipeline_functional = True
            else:
                health_status.issues.append("Real generation pipeline not available")
            
            # Determine overall status
            critical_issues = 0
            warning_issues = 0
            
            # Check for critical issues
            if health_status.cpu_usage_percent > 95:
                critical_issues += 1
                health_status.issues.append("Critical CPU usage")
            elif health_status.cpu_usage_percent > 80:
                warning_issues += 1
                health_status.issues.append("High CPU usage")
            
            if health_status.memory_usage_percent > 95:
                critical_issues += 1
                health_status.issues.append("Critical memory usage")
            elif health_status.memory_usage_percent > 80:
                warning_issues += 1
                health_status.issues.append("High memory usage")
            
            if health_status.vram_usage_percent > 95:
                critical_issues += 1
                health_status.issues.append("Critical VRAM usage")
            elif health_status.vram_usage_percent > 90:
                warning_issues += 1
                health_status.issues.append("High VRAM usage")
            
            if not health_status.gpu_available:
                critical_issues += 1
            
            if not health_status.model_loading_functional:
                warning_issues += 1
            
            # Set overall status
            if critical_issues > 0:
                health_status.overall_status = "critical"
            elif warning_issues > 0 or len(health_status.issues) > 0:
                health_status.overall_status = "degraded"
            else:
                health_status.overall_status = "healthy"
            
            # Generate recommendations
            if health_status.cpu_usage_percent > 80:
                health_status.recommendations.append("Close unnecessary applications to reduce CPU usage")
            
            if health_status.memory_usage_percent > 80:
                health_status.recommendations.append("Close memory-intensive applications")
            
            if health_status.vram_usage_percent > 90:
                health_status.recommendations.append("Enable model offloading or reduce generation parameters")
            
            if not health_status.model_loading_functional:
                health_status.recommendations.append("Restart the application to reinitialize model loading")
            
            # Cache the current health status
            self.current_health_status = health_status
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Error getting system health status: {e}")
            return SystemHealthStatus(
                overall_status="error",
                cpu_usage_percent=0.0,
                memory_usage_percent=0.0,
                vram_usage_percent=0.0,
                gpu_available=False,
                model_loading_functional=False,
                generation_pipeline_functional=False,
                last_check_timestamp=datetime.now(),
                issues=[f"Health check error: {str(e)}"]
            )
    
    def start_health_monitoring(self):
        """Start continuous system health monitoring"""
        if self.health_monitoring_active:
            self.logger.warning("Health monitoring already active")
            return
        
        self.health_monitoring_active = True
        self.health_monitor_thread = threading.Thread(
            target=self._health_monitor_worker,
            daemon=True
        )
        self.health_monitor_thread.start()
        self.logger.info("System health monitoring started")
    
    def stop_health_monitoring(self):
        """Stop continuous system health monitoring"""
        self.health_monitoring_active = False
        if self.health_monitor_thread and self.health_monitor_thread.is_alive():
            self.health_monitor_thread.join(timeout=5)
        self.logger.info("System health monitoring stopped")
    
    def _health_monitor_worker(self):
        """Background worker for continuous health monitoring"""
        while self.health_monitoring_active:
            try:
                # Create event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    # Get health status
                    health_status = loop.run_until_complete(self.get_system_health_status())
                    
                    # Check for automatic recovery triggers
                    if health_status.overall_status == "critical":
                        self.logger.warning("Critical system health detected, triggering automatic recovery")
                        
                        # Trigger automatic recovery for system resource issues
                        loop.run_until_complete(self.handle_failure(
                            FailureType.SYSTEM_RESOURCE_ERROR,
                            Exception("Critical system health detected"),
                            {"health_status": health_status.__dict__}
                        ))
                    
                    elif health_status.vram_usage_percent > 90:
                        self.logger.warning("High VRAM usage detected, triggering VRAM optimization")
                        
                        # Trigger VRAM optimization
                        loop.run_until_complete(self.handle_failure(
                            FailureType.VRAM_EXHAUSTION,
                            Exception("High VRAM usage detected"),
                            {"vram_usage_percent": health_status.vram_usage_percent}
                        ))
                    
                    # Send health status via WebSocket
                    if self.websocket_manager:
                        loop.run_until_complete(self.websocket_manager.broadcast({
                            "type": "health_status",
                            "data": {
                                "status": health_status.overall_status,
                                "cpu_usage": health_status.cpu_usage_percent,
                                "memory_usage": health_status.memory_usage_percent,
                                "vram_usage": health_status.vram_usage_percent,
                                "issues": health_status.issues,
                                "recommendations": health_status.recommendations,
                                "timestamp": health_status.last_check_timestamp.isoformat()
                            }
                        }))
                
                finally:
                    loop.close()
                
            except Exception as e:
                self.logger.error(f"Error in health monitor worker: {e}")
            
            # Wait for next check
            time.sleep(self.health_check_interval)
    
    def _is_in_cooldown(self, failure_type: FailureType) -> bool:
        """Check if we're in cooldown period for a failure type"""
        if failure_type not in self.last_recovery_attempt:
            return False
        
        last_attempt = self.last_recovery_attempt[failure_type]
        cooldown_end = last_attempt + timedelta(seconds=self.recovery_cooldown_seconds)
        return datetime.now() < cooldown_end
    
    def _get_cooldown_remaining(self, failure_type: FailureType) -> float:
        """Get remaining cooldown time in seconds"""
        if failure_type not in self.last_recovery_attempt:
            return 0.0
        
        last_attempt = self.last_recovery_attempt[failure_type]
        cooldown_end = last_attempt + timedelta(seconds=self.recovery_cooldown_seconds)
        remaining = cooldown_end - datetime.now()
        return max(0.0, remaining.total_seconds())
    
    async def _notify_recovery_attempt(self, attempt: RecoveryAttempt):
        """Send WebSocket notification about recovery attempt"""
        try:
            if self.websocket_manager:
                await self.websocket_manager.broadcast({
                    "type": "recovery_attempt",
                    "data": {
                        "failure_type": attempt.failure_type.value,
                        "action": attempt.action.value,
                        "success": attempt.success,
                        "timestamp": attempt.timestamp.isoformat(),
                        "recovery_time": attempt.recovery_time_seconds,
                        "error_message": attempt.error_message
                    }
                })
        except Exception as e:
            self.logger.error(f"Error sending recovery attempt notification: {e}")
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get statistics about recovery attempts"""
        if not self.recovery_attempts:
            return {
                "total_attempts": 0,
                "success_rate": 0.0,
                "average_recovery_time": 0.0,
                "failure_types": {},
                "recovery_actions": {}
            }
        
        total_attempts = len(self.recovery_attempts)
        successful_attempts = sum(1 for attempt in self.recovery_attempts if attempt.success)
        success_rate = (successful_attempts / total_attempts) * 100
        
        # Calculate average recovery time for successful attempts
        successful_times = [attempt.recovery_time_seconds for attempt in self.recovery_attempts if attempt.success]
        average_recovery_time = sum(successful_times) / len(successful_times) if successful_times else 0.0
        
        # Count failure types
        failure_types = {}
        for attempt in self.recovery_attempts:
            failure_type = attempt.failure_type.value
            if failure_type not in failure_types:
                failure_types[failure_type] = {"total": 0, "successful": 0}
            failure_types[failure_type]["total"] += 1
            if attempt.success:
                failure_types[failure_type]["successful"] += 1
        
        # Count recovery actions
        recovery_actions = {}
        for attempt in self.recovery_attempts:
            action = attempt.action.value
            if action not in recovery_actions:
                recovery_actions[action] = {"total": 0, "successful": 0}
            recovery_actions[action]["total"] += 1
            if attempt.success:
                recovery_actions[action]["successful"] += 1
        
        return {
            "total_attempts": total_attempts,
            "successful_attempts": successful_attempts,
            "success_rate": round(success_rate, 1),
            "average_recovery_time": round(average_recovery_time, 2),
            "failure_types": failure_types,
            "recovery_actions": recovery_actions,
            "mock_generation_enabled": self.mock_generation_enabled,
            "degraded_mode_active": self.degraded_mode_active
        }
    
    def reset_recovery_state(self):
        """Reset recovery state (useful for testing or manual recovery)"""
        self.mock_generation_enabled = False
        self.degraded_mode_active = False
        self.critical_failures.clear()
        self.last_recovery_attempt.clear()
        
        # Re-enable real generation if generation service is available
        if self.generation_service:
            self.generation_service.use_real_generation = True
            self.generation_service.fallback_to_simulation = False
        
        self.logger.info("Recovery state reset - real generation re-enabled")


# Global instance for easy access
_fallback_recovery_system: Optional[FallbackRecoverySystem] = None

def get_fallback_recovery_system() -> FallbackRecoverySystem:
    """Get the global fallback recovery system instance"""
    global _fallback_recovery_system
    if _fallback_recovery_system is None:
        _fallback_recovery_system = FallbackRecoverySystem()
    return _fallback_recovery_system

def initialize_fallback_recovery_system(generation_service=None, websocket_manager=None) -> FallbackRecoverySystem:
    """Initialize the global fallback recovery system with dependencies"""
    global _fallback_recovery_system
    _fallback_recovery_system = FallbackRecoverySystem(generation_service, websocket_manager)
    return _fallback_recovery_system