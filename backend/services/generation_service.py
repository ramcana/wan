"""
Enhanced Generation service with real AI model integration
Integrates with existing Wan2.2 system using ModelIntegrationBridge and RealGenerationPipeline
"""

import sys
import os
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Tuple, List
from datetime import datetime, timedelta
import uuid
import threading
from queue import Queue as ThreadQueue
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from sqlalchemy.orm import Session
from backend.repositories.database import SessionLocal, GenerationTaskDB, TaskStatusEnum
from backend.core.system_integration import get_system_integration
from backend.core.fallback_recovery_system import (
    get_fallback_recovery_system, FailureType, FallbackRecoverySystem
)
from backend.core.performance_monitor import get_performance_monitor

# Enhanced Model Availability Components
from backend.core.model_availability_manager import (
    ModelAvailabilityManager, ModelAvailabilityStatus, DetailedModelStatus
)
from backend.core.enhanced_model_downloader import (
    EnhancedModelDownloader, DownloadStatus, DownloadProgress, DownloadResult
)
from backend.core.intelligent_fallback_manager import (
    IntelligentFallbackManager, FallbackType, ModelSuggestion, GenerationRequirements
)
from backend.core.model_health_monitor import (
    ModelHealthMonitor, HealthStatus, IntegrityResult, PerformanceHealth
)
from backend.core.model_usage_analytics import (
    ModelUsageAnalytics, UsageData, UsageEventType, UsageStatistics
)

logger = logging.getLogger(__name__)

class VRAMMonitor:
    """VRAM monitoring and management for generation tasks"""
    
    def __init__(self, total_vram_gb: float, optimal_usage_gb: float, system_optimizer=None):
        self.total_vram_gb = total_vram_gb
        self.optimal_usage_gb = optimal_usage_gb
        self.system_optimizer = system_optimizer
        self.warning_threshold = 0.9  # Warn at 90% of optimal usage
        self.critical_threshold = 0.95  # Critical at 95% of optimal usage
        
    def get_current_vram_usage(self) -> Dict[str, float]:
        """Get current VRAM usage statistics"""
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                allocated_bytes = torch.cuda.memory_allocated(device)
                reserved_bytes = torch.cuda.memory_reserved(device)
                total_bytes = torch.cuda.get_device_properties(device).total_memory
                
                allocated_gb = allocated_bytes / (1024**3)
                reserved_gb = reserved_bytes / (1024**3)
                total_gb = total_bytes / (1024**3)
                
                return {
                    "allocated_gb": allocated_gb,
                    "reserved_gb": reserved_gb,
                    "total_gb": total_gb,
                    "usage_percent": (allocated_gb / total_gb) * 100,
                    "optimal_usage_percent": (allocated_gb / self.optimal_usage_gb) * 100
                }
            else:
                return {"error": "CUDA not available"}
                
        except Exception as e:
            logger.warning(f"Failed to get VRAM usage: {e}")
            return {"error": str(e)}
    
    def check_vram_availability(self, required_gb: float) -> Tuple[bool, str]:
        """Check if sufficient VRAM is available for a task"""
        try:
            usage = self.get_current_vram_usage()
            if "error" in usage:
                return False, f"Cannot check VRAM: {usage['error']}"
            
            # Calculate actual available VRAM correctly
            total_gb = usage.get("total_gb", self.total_vram_gb)
            allocated_gb = usage["allocated_gb"]
            available_gb = total_gb - allocated_gb
            
            # Check against optimal usage to avoid overloading
            optimal_available = self.optimal_usage_gb - allocated_gb
            
            if required_gb <= available_gb:
                if allocated_gb + required_gb <= self.optimal_usage_gb:
                    return True, f"Sufficient VRAM available: {available_gb:.1f}GB free, {required_gb:.1f}GB required"
                else:
                    return True, f"VRAM available but will exceed optimal usage: {available_gb:.1f}GB free, {required_gb:.1f}GB required (optimal: {self.optimal_usage_gb:.1f}GB)"
            else:
                return False, f"Insufficient VRAM: {available_gb:.1f}GB free, {required_gb:.1f}GB required. Suggestions: Enable model offloading to CPU; Reduce VAE tile size"
                
        except Exception as e:
            return False, f"VRAM check failed: {str(e)}"
    
    def get_optimization_suggestions(self) -> List[str]:
        """Get VRAM optimization suggestions based on current usage"""
        suggestions = []
        
        try:
            usage = self.get_current_vram_usage()
            if "error" in usage:
                return ["Cannot analyze VRAM usage"]
            
            usage_percent = usage.get("optimal_usage_percent", 0)
            
            if usage_percent > self.critical_threshold * 100:
                suggestions.extend([
                    "Enable model offloading to CPU",
                    "Reduce VAE tile size",
                    "Use lower precision (fp16 or int8)",
                    "Reduce number of inference steps"
                ])
            elif usage_percent > self.warning_threshold * 100:
                suggestions.extend([
                    "Consider enabling model offloading",
                    "Monitor VRAM usage during generation",
                    "Reduce batch size if applicable"
                ])
            
            return suggestions
            
        except Exception as e:
            return [f"Error getting optimization suggestions: {str(e)}"]

class GenerationService:
    """Enhanced service for managing video generation tasks with real AI integration"""
    
    def __init__(self):
        self.task_queue = ThreadQueue()
        self.processing_thread = None
        self.is_processing = False
        self.current_task = None
        
        # Real AI integration components
        self.model_integration_bridge = None
        self.real_generation_pipeline = None
        self.error_handler = None
        self.websocket_manager = None
        
        # Hardware optimization integration
        self.wan22_system_optimizer = None
        self.hardware_profile = None
        self.optimization_applied = False
        self.vram_monitor = None
        
        # Generation mode (can be switched for testing)
        self.use_real_generation = True
        self.fallback_to_simulation = True
        
        # Fallback and recovery system
        self.fallback_recovery_system: Optional[FallbackRecoverySystem] = None
        
        # Performance monitoring
        self.performance_monitor = None
        
        # Enhanced Model Availability Components
        self.model_availability_manager: Optional[ModelAvailabilityManager] = None
        self.enhanced_model_downloader: Optional[EnhancedModelDownloader] = None
        self.intelligent_fallback_manager: Optional[IntelligentFallbackManager] = None
        self.model_health_monitor: Optional[ModelHealthMonitor] = None
        self.model_usage_analytics: Optional[ModelUsageAnalytics] = None
        
    async def initialize(self):
        """Initialize the enhanced generation service with real AI integration"""
        try:
            # Initialize hardware optimization integration first
            await self._initialize_hardware_optimization()
            
            # Initialize real AI integration components
            await self._initialize_real_ai_components()
            
            # Initialize error handling system
            await self._initialize_error_handling()
            
            # Initialize WebSocket manager for progress updates
            await self._initialize_websocket_manager()
            
            # Initialize VRAM monitoring
            await self._initialize_vram_monitoring()
            
            # Initialize fallback and recovery system
            await self._initialize_fallback_recovery_system()
            
            # Initialize performance monitoring
            await self._initialize_performance_monitoring()
            
            # Initialize enhanced model availability components
            await self._initialize_enhanced_model_availability()
            
            # Start background processing thread
            if not self.processing_thread or not self.processing_thread.is_alive():
                self.is_processing = True
                self.processing_thread = threading.Thread(
                    target=self._process_queue_worker,
                    daemon=True
                )
                self.processing_thread.start()
                logger.info("Enhanced generation service initialized with real AI integration and hardware optimization")
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced generation service: {e}")
            # Don't raise - allow service to start in fallback mode
            logger.warning("Starting generation service in fallback mode")
            self.use_real_generation = False
    
    async def _initialize_hardware_optimization(self):
        """Initialize hardware optimization integration with WAN22SystemOptimizer"""
        try:
            # Get WAN22SystemOptimizer from system integration
            from backend.core.system_integration import get_system_integration
            system_integration = await get_system_integration()
            self.wan22_system_optimizer = system_integration.get_wan22_system_optimizer()
            
            if self.wan22_system_optimizer:
                logger.info("WAN22SystemOptimizer integrated with generation service")
                
                # Get hardware profile for optimization decisions
                self.hardware_profile = self.wan22_system_optimizer.get_hardware_profile()
                if self.hardware_profile:
                    logger.info(f"Hardware profile loaded: {self.hardware_profile.gpu_model} with {self.hardware_profile.vram_gb}GB VRAM")
                    
                    # Apply hardware optimizations before model loading
                    await self._apply_hardware_optimizations_for_generation()
                else:
                    logger.warning("Hardware profile not available from system optimizer")
            else:
                logger.warning("WAN22SystemOptimizer not available - hardware optimization disabled")
                
        except Exception as e:
            logger.error(f"Failed to initialize hardware optimization: {e}")
            # Don't fail initialization - continue without hardware optimization
            self.wan22_system_optimizer = None
            self.hardware_profile = None
    
    async def _apply_hardware_optimizations_for_generation(self):
        """Apply hardware-specific optimizations for generation tasks"""
        try:
            if not self.wan22_system_optimizer or not self.hardware_profile:
                return
            
            optimizations_applied = []
            
            # RTX 4080 specific optimizations
            if "RTX 4080" in self.hardware_profile.gpu_model:
                # Set optimal VRAM usage for RTX 4080 (16GB)
                self.optimal_vram_usage_gb = min(14.0, self.hardware_profile.vram_gb * 0.85)  # Use 85% max
                optimizations_applied.append("RTX 4080 VRAM optimization")
                
                # Enable tensor core optimizations
                self.enable_tensor_cores = True
                optimizations_applied.append("RTX 4080 tensor core optimization")
                
                # Set optimal batch processing
                self.optimal_batch_size = 1  # RTX 4080 works best with single batch for video generation
                optimizations_applied.append("RTX 4080 batch size optimization")
            
            # High VRAM optimizations (12GB+)
            elif self.hardware_profile.vram_gb >= 12:
                self.optimal_vram_usage_gb = min(self.hardware_profile.vram_gb * 0.8, 12.0)
                optimizations_applied.append("High VRAM optimization")
            
            # Standard VRAM optimizations (8-12GB)
            elif self.hardware_profile.vram_gb >= 8:
                self.optimal_vram_usage_gb = self.hardware_profile.vram_gb * 0.75
                optimizations_applied.append("Standard VRAM optimization")
            
            # Low VRAM optimizations (<8GB)
            else:
                self.optimal_vram_usage_gb = self.hardware_profile.vram_gb * 0.6
                self.enable_model_offloading = True
                optimizations_applied.append("Low VRAM optimization with offloading")
            
            # Threadripper PRO CPU optimizations
            if "Threadripper PRO" in self.hardware_profile.cpu_model:
                # Enable multi-threading for preprocessing
                self.enable_cpu_multithreading = True
                self.cpu_worker_threads = min(self.hardware_profile.cpu_cores // 2, 16)
                optimizations_applied.append("Threadripper PRO multi-threading optimization")
            
            # High memory optimizations (64GB+)
            if self.hardware_profile.total_memory_gb >= 64:
                # Enable aggressive model caching
                self.enable_model_caching = True
                self.max_cached_models = 3
                optimizations_applied.append("High memory model caching optimization")
            
            self.optimization_applied = len(optimizations_applied) > 0
            
            if optimizations_applied:
                logger.info(f"Applied hardware optimizations: {', '.join(optimizations_applied)}")
            else:
                logger.info("No specific hardware optimizations applied for current configuration")
                
        except Exception as e:
            logger.error(f"Failed to apply hardware optimizations: {e}")
            # Set safe defaults
            self.optimal_vram_usage_gb = 6.0
            self.enable_model_offloading = True
    
    async def _initialize_vram_monitoring(self):
        """Initialize VRAM monitoring for generation tasks"""
        try:
            # Create VRAM monitor if hardware profile is available
            if self.hardware_profile and self.hardware_profile.vram_gb > 0:
                self.vram_monitor = VRAMMonitor(
                    total_vram_gb=self.hardware_profile.vram_gb,
                    optimal_usage_gb=getattr(self, 'optimal_vram_usage_gb', self.hardware_profile.vram_gb * 0.8),
                    system_optimizer=self.wan22_system_optimizer
                )
                logger.info(f"VRAM monitoring initialized: {self.hardware_profile.vram_gb}GB total, {self.vram_monitor.optimal_usage_gb}GB optimal")
            else:
                logger.warning("VRAM monitoring not available - hardware profile missing")
                
        except Exception as e:
            logger.error(f"Failed to initialize VRAM monitoring: {e}")
            self.vram_monitor = None
    
    async def _initialize_fallback_recovery_system(self):
        """Initialize fallback and recovery system for automatic error handling"""
        try:
            from backend.core.fallback_recovery_system import initialize_fallback_recovery_system
            
            # Initialize with this generation service and websocket manager
            self.fallback_recovery_system = initialize_fallback_recovery_system(
                generation_service=self,
                websocket_manager=self.websocket_manager
            )
            
            # Start health monitoring
            self.fallback_recovery_system.start_health_monitoring()
            
            logger.info("Fallback and recovery system initialized with health monitoring")
            
        except Exception as e:
            logger.error(f"Failed to initialize fallback and recovery system: {e}")
            self.fallback_recovery_system = None

    async def _initialize_performance_monitoring(self):
        """Initialize performance monitoring system"""
        try:
            # Get the global performance monitor instance
            self.performance_monitor = get_performance_monitor()
            
            # Start continuous monitoring
            self.performance_monitor.start_monitoring()
            
            logger.info("Performance monitoring initialized and started")
            
        except Exception as e:
            logger.error(f"Failed to initialize performance monitoring: {e}")
            self.performance_monitor = None

    async def _initialize_enhanced_model_availability(self):
        """Initialize enhanced model availability components"""
        try:
            logger.info("Initializing enhanced model availability components...")
            
            # Initialize Model Health Monitor
            self.model_health_monitor = ModelHealthMonitor()
            await self.model_health_monitor.initialize()
            logger.info("Model Health Monitor initialized")
            
            # Initialize Enhanced Model Downloader (wraps existing downloader if available)
            try:
                # Try to get existing model downloader from model integration bridge
                base_downloader = None
                if self.model_integration_bridge:
                    base_downloader = getattr(self.model_integration_bridge, 'model_downloader', None)
                
                self.enhanced_model_downloader = EnhancedModelDownloader(base_downloader)
                await self.enhanced_model_downloader.initialize()
                logger.info("Enhanced Model Downloader initialized")
            except Exception as e:
                logger.warning(f"Enhanced Model Downloader initialization failed: {e}")
                self.enhanced_model_downloader = None
            
            # Initialize Model Usage Analytics
            self.model_usage_analytics = ModelUsageAnalytics()
            await self.model_usage_analytics.initialize()
            logger.info("Model Usage Analytics initialized")
            
            # Initialize Model Availability Manager (requires other components)
            if self.model_health_monitor and self.enhanced_model_downloader:
                try:
                    # Get existing model manager if available
                    existing_model_manager = None
                    if self.model_integration_bridge:
                        existing_model_manager = getattr(self.model_integration_bridge, 'model_manager', None)
                    
                    self.model_availability_manager = ModelAvailabilityManager(
                        model_manager=existing_model_manager,
                        enhanced_downloader=self.enhanced_model_downloader,
                        health_monitor=self.model_health_monitor,
                        usage_analytics=self.model_usage_analytics
                    )
                    await self.model_availability_manager.initialize()
                    logger.info("Model Availability Manager initialized")
                except Exception as e:
                    logger.warning(f"Model Availability Manager initialization failed: {e}")
                    self.model_availability_manager = None
            
            # Initialize Intelligent Fallback Manager (requires availability manager)
            if self.model_availability_manager:
                try:
                    self.intelligent_fallback_manager = IntelligentFallbackManager(
                        availability_manager=self.model_availability_manager
                    )
                    await self.intelligent_fallback_manager.initialize()
                    logger.info("Intelligent Fallback Manager initialized")
                except Exception as e:
                    logger.warning(f"Intelligent Fallback Manager initialization failed: {e}")
                    self.intelligent_fallback_manager = None
            
            # Start health monitoring if available
            if self.model_health_monitor:
                await self.model_health_monitor.schedule_health_checks()
                logger.info("Model health monitoring started")
            
            logger.info("Enhanced model availability components initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced model availability components: {e}")
            # Don't fail the entire service - continue with basic functionality

    async def _initialize_real_ai_components(self):
        """Initialize real AI integration components"""
        try:
            # Initialize Model Integration Bridge
            from backend.core.model_integration_bridge import get_model_integration_bridge
            self.model_integration_bridge = await get_model_integration_bridge()
            
            if self.model_integration_bridge and self.model_integration_bridge.is_initialized():
                logger.info("Model Integration Bridge initialized successfully")
                
                # Integrate hardware optimization with model bridge
                if self.wan22_system_optimizer:
                    self.model_integration_bridge.set_hardware_optimizer(self.wan22_system_optimizer)
                    logger.info("Hardware optimizer integrated with model bridge")
            else:
                logger.warning("Model Integration Bridge initialization failed")
                self.use_real_generation = False
            
            # Initialize Real Generation Pipeline
            from backend.services.real_generation_pipeline import RealGenerationPipeline
            self.real_generation_pipeline = RealGenerationPipeline()
            
            pipeline_initialized = await self.real_generation_pipeline.initialize()
            if pipeline_initialized:
                logger.info("Real Generation Pipeline initialized successfully")
                
                # Integrate hardware optimization with pipeline
                if self.wan22_system_optimizer:
                    self.real_generation_pipeline.set_hardware_optimizer(self.wan22_system_optimizer)
                    logger.info("Hardware optimizer integrated with generation pipeline")
            else:
                logger.warning("Real Generation Pipeline initialization failed")
                self.use_real_generation = False
                
        except Exception as e:
            logger.error(f"Failed to initialize real AI components: {e}")
            self.use_real_generation = False
    
    async def _initialize_error_handling(self):
        """Initialize enhanced error handling system using integrated handler"""
        try:
            from backend.core.integrated_error_handler import get_integrated_error_handler
            self.error_handler = get_integrated_error_handler()
            logger.info("Integrated error handler initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize integrated error handler: {e}")
            # Fall back to basic error handler
            self.error_handler = self._create_fallback_error_handler()
    
    async def _initialize_websocket_manager(self):
        """Initialize WebSocket manager and progress integration for real-time updates"""
        try:
            from backend.websocket.manager import get_connection_manager
            self.websocket_manager = get_connection_manager()
            logger.info("WebSocket manager initialized for real-time updates")
            
            # Also initialize the enhanced progress integration system
            try:
                from backend.websocket.progress_integration import get_progress_integration
                self.progress_integration = await get_progress_integration()
                logger.info("Enhanced progress integration system initialized")
            except Exception as e:
                logger.warning(f"Could not initialize progress integration: {e}")
                self.progress_integration = None
                
        except Exception as e:
            logger.warning(f"Could not initialize WebSocket manager: {e}")
            self.websocket_manager = None
            self.progress_integration = None
    
    def _create_fallback_error_handler(self):
        """Create a fallback error handler if existing system is not available"""
        class FallbackErrorHandler:
            def handle_error(self, error, context):
                """Handle errors with basic categorization"""
                error_message = str(error)
                recovery_suggestions = []
                
                # Basic error categorization
                if "CUDA out of memory" in error_message or "VRAM" in error_message:
                    recovery_suggestions = [
                        "Try reducing the resolution",
                        "Enable model offloading",
                        "Reduce the number of inference steps",
                        "Close other GPU-intensive applications"
                    ]
                elif "model" in error_message.lower() and "download" in error_message.lower():
                    recovery_suggestions = [
                        "Check your internet connection",
                        "Verify available disk space",
                        "Try downloading the model manually"
                    ]
                elif "timeout" in error_message.lower():
                    recovery_suggestions = [
                        "Try reducing the number of frames",
                        "Reduce inference steps",
                        "Check system resources"
                    ]
                else:
                    recovery_suggestions = [
                        "Check the error logs for more details",
                        "Try restarting the generation",
                        "Verify your system configuration"
                    ]
                
                # Return error info object
                return type('ErrorInfo', (), {
                    'message': error_message,
                    'recovery_suggestions': recovery_suggestions,
                    'error_category': self._categorize_error(error_message)
                })()
            
            def _categorize_error(self, error_message):
                """Categorize error for better handling"""
                error_message_lower = error_message.lower()
                
                if "cuda" in error_message_lower or "memory" in error_message_lower:
                    return "gpu_memory_error"
                elif "model" in error_message_lower:
                    return "model_error"
                elif "timeout" in error_message_lower:
                    return "timeout_error"
                elif "network" in error_message_lower or "download" in error_message_lower:
                    return "network_error"
                else:
                    return "general_error"
        
        return FallbackErrorHandler()
    
    def _determine_failure_type(self, error: Exception, model_type: str) -> FailureType:
        """Determine the type of failure based on the error and context"""
        error_message = str(error).lower()
        
        if "model" in error_message and ("load" in error_message or "not found" in error_message):
            return FailureType.MODEL_LOADING_FAILURE
        elif "cuda out of memory" in error_message or "vram" in error_message:
            return FailureType.VRAM_EXHAUSTION
        elif "pipeline" in error_message or "generation" in error_message:
            return FailureType.GENERATION_PIPELINE_ERROR
        elif "optimization" in error_message or "hardware" in error_message:
            return FailureType.HARDWARE_OPTIMIZATION_FAILURE
        elif "network" in error_message or "download" in error_message:
            return FailureType.NETWORK_ERROR
        else:
            return FailureType.SYSTEM_RESOURCE_ERROR
    
    async def _run_mock_generation(self, task: GenerationTaskDB, db: Session, model_type: str) -> bool:
        """
        Run mock generation as fallback when real models fail
        This is a unified method that can be called by both the recovery system and manual fallback
        """
        try:
            logger.info(f"Running mock generation for task {task.id} (model: {model_type})")
            
            # Update task status to indicate mock generation
            task.progress = 5
            task.status = TaskStatusEnum.PROCESSING
            db.commit()
            
            # Send WebSocket notification about mock mode
            if self.websocket_manager:
                await self.websocket_manager.send_alert(
                    alert_type="generation_mode",
                    message="Using mock generation due to model issues",
                    severity="warning",
                    task_id=task.id,
                    mock_mode=True
                )
            
            # Prepare mock generation parameters
            mock_params = {
                "prompt": task.prompt,
                "model_type": model_type,
                "resolution": task.resolution,
                "steps": task.steps,
                "image_path": task.image_path,
                "lora_path": task.lora_path,
                "lora_strength": task.lora_strength
            }
            
            # Simulate generation process with progress updates
            await self._simulate_generation_with_recovery_context(task, db, mock_params)
            
            # Create mock output file
            output_filename = f"mock_generated_{task.id}_{model_type}.mp4"
            output_path = f"outputs/{output_filename}"
            
            # Ensure outputs directory exists
            outputs_dir = Path("outputs")
            outputs_dir.mkdir(exist_ok=True)
            
            # Create a simple mock video file (placeholder)
            mock_file_path = outputs_dir / output_filename
            with open(mock_file_path, 'w') as f:
                f.write(f"Mock video generated for task {task.id}\n")
                f.write(f"Model: {model_type}\n")
                f.write(f"Prompt: {task.prompt}\n")
                f.write(f"Resolution: {task.resolution}\n")
                f.write(f"Steps: {task.steps}\n")
                f.write(f"Generated at: {datetime.utcnow().isoformat()}\n")
            
            # Update task with completion
            task.output_path = output_path
            task.progress = 100
            task.status = TaskStatusEnum.COMPLETED
            task.completed_at = datetime.utcnow()
            db.commit()
            
            # Send completion notification
            if self.websocket_manager:
                await self.websocket_manager.send_alert(
                    alert_type="generation_completed",
                    message=f"Mock generation completed for {model_type}",
                    severity="info",
                    task_id=task.id,
                    output_path=output_path,
                    mock_mode=True
                )
            
            # Complete performance monitoring for mock generation
            if self.performance_monitor:
                self.performance_monitor.complete_task_monitoring(
                    str(task.id), success=True
                )
            
            logger.info(f"Mock generation completed for task {task.id}")
            return True
            
        except Exception as e:
            logger.error(f"Mock generation failed for task {task.id}: {e}")
            task.error_message = f"Mock generation error: {str(e)}"
            task.status = TaskStatusEnum.FAILED
            db.commit()
            return False
    
    async def _simulate_generation_with_recovery_context(self, task: GenerationTaskDB, db: Session, params: Dict[str, Any]):
        """Simulate generation process with recovery system context"""
        try:
            # Simulate different stages of generation
            stages = [
                (10, "Initializing mock generation"),
                (25, "Loading mock model"),
                (40, "Preparing mock inputs"),
                (60, "Generating mock frames"),
                (80, "Post-processing mock output"),
                (95, "Saving mock video"),
                (100, "Mock generation complete")
            ]
            
            for progress, message in stages:
                # Update progress in database
                task.progress = progress
                db.commit()
                
                # Send WebSocket update
                await self._send_websocket_progress_update(
                    task.id, progress, message, "mock_generation"
                )
                
                # Simulate processing time
                await asyncio.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error in mock generation simulation: {e}")
            raise
    
    def _estimate_vram_requirements(self, model_type: str, resolution: str) -> float:
        """Estimate VRAM requirements for a generation task"""
        try:
            # Base VRAM requirements by model type (in GB)
            base_requirements = {
                "t2v": 8.0,    # T2V-A14B base requirement
                "i2v": 9.0,    # I2V-A14B base requirement  
                "ti2v": 6.0    # TI2V-5B base requirement
            }
            
            base_vram = base_requirements.get(model_type, 8.0)
            
            # Resolution multipliers
            resolution_multipliers = {
                "1280x720": 1.0,
                "1280x704": 1.0,
                "1920x1080": 1.4,
                "1920x1088": 1.4
            }
            
            multiplier = resolution_multipliers.get(resolution, 1.2)  # Default multiplier for unknown resolutions
            
            estimated_vram = base_vram * multiplier
            
            # Add safety margin
            estimated_vram *= 1.1
            
            logger.debug(f"Estimated VRAM requirement: {estimated_vram:.1f}GB for {model_type} at {resolution}")
            return estimated_vram
            
        except Exception as e:
            logger.warning(f"Failed to estimate VRAM requirements: {e}")
            return 10.0  # Conservative fallback
    
    async def _apply_vram_optimizations(self):
        """Apply automatic VRAM optimizations"""
        try:
            if not self.vram_monitor:
                return
            
            optimizations_applied = []
            
            # Enable model offloading if not already enabled
            if not getattr(self, 'enable_model_offloading', False):
                self.enable_model_offloading = True
                optimizations_applied.append("Model offloading enabled")
            
            # Reduce VAE tile size for memory efficiency
            if not hasattr(self, 'vae_tile_size') or self.vae_tile_size > 256:
                self.vae_tile_size = 256
                optimizations_applied.append("VAE tile size reduced to 256")
            
            # Enable gradient checkpointing if available
            self.enable_gradient_checkpointing = True
            optimizations_applied.append("Gradient checkpointing enabled")
            
            if optimizations_applied:
                logger.info(f"Applied VRAM optimizations: {', '.join(optimizations_applied)}")
            
        except Exception as e:
            logger.error(f"Failed to apply VRAM optimizations: {e}")
    
    async def _apply_pre_generation_optimizations(self, model_type: str):
        """Apply hardware optimizations before generation starts"""
        try:
            if not self.wan22_system_optimizer:
                return
            
            # Monitor system health before generation
            health_metrics = self.wan22_system_optimizer.monitor_system_health()
            logger.info(f"System health before generation: CPU {health_metrics.cpu_usage_percent}%, "
                       f"Memory {health_metrics.memory_usage_gb}GB, VRAM {health_metrics.vram_usage_mb}MB")
            
            # Apply model-specific optimizations
            if model_type == "t2v" and self.hardware_profile:
                if "RTX 4080" in self.hardware_profile.gpu_model:
                    # RTX 4080 specific T2V optimizations
                    logger.info("Applying RTX 4080 T2V optimizations")
                    # These would be applied to the model loading process
                    
            elif model_type == "i2v" and self.hardware_profile:
                if "RTX 4080" in self.hardware_profile.gpu_model:
                    # RTX 4080 specific I2V optimizations
                    logger.info("Applying RTX 4080 I2V optimizations")
                    
            elif model_type == "ti2v" and self.hardware_profile:
                if "RTX 4080" in self.hardware_profile.gpu_model:
                    # RTX 4080 specific TI2V optimizations
                    logger.info("Applying RTX 4080 TI2V optimizations")
            
            # Clear GPU cache before generation
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.debug("GPU cache cleared before generation")
            except Exception as e:
                logger.warning(f"Failed to clear GPU cache: {e}")
                
        except Exception as e:
            logger.error(f"Failed to apply pre-generation optimizations: {e}")

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
        """Run the actual generation process using enhanced model availability system"""
        performance_metrics = None
        start_time = datetime.now()
        
        try:
            # Convert model type for compatibility
            model_type = task.model_type.value.lower()
            logger.info(f"Starting enhanced generation for model: {model_type}")
            
            # Track usage analytics for this generation request
            if self.model_usage_analytics:
                await self.model_usage_analytics.track_usage(
                    model_id=model_type,
                    event_type=UsageEventType.GENERATION_REQUEST,
                    generation_params={
                        "resolution": task.resolution,
                        "steps": task.steps,
                        "prompt": task.prompt[:100] if task.prompt else None  # Truncate for privacy
                    }
                )
            
            # Start performance monitoring
            if self.performance_monitor:
                performance_metrics = self.performance_monitor.start_task_monitoring(
                    task_id=str(task.id),
                    model_type=model_type,
                    resolution=getattr(task, 'resolution', '720p'),
                    steps=getattr(task, 'steps', 20)
                )
            
            # Update initial progress
            task.progress = 5
            db.commit()
            
            # Use enhanced model availability system if available
            if self.model_availability_manager:
                result = await self._run_enhanced_generation(task, db, model_type)
            else:
                # Fallback to existing generation logic
                logger.warning("Enhanced model availability not available, using legacy generation")
                result = await self._run_legacy_generation(task, db, model_type)
            
            # Track successful generation completion
            if result and self.model_usage_analytics:
                generation_time = (datetime.now() - start_time).total_seconds()
                await self.model_usage_analytics.track_usage(
                    model_id=model_type,
                    event_type=UsageEventType.GENERATION_COMPLETE,
                    duration_seconds=generation_time,
                    success=True
                )
            
            # Complete performance monitoring
            if self.performance_monitor and performance_metrics:
                self.performance_monitor.complete_task_monitoring(
                    str(task.id), success=result, error_category=None if result else "generation_failed"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Critical error in generation for task {task.id}: {e}")
            
            # Track failed generation in analytics
            if self.model_usage_analytics:
                generation_time = (datetime.now() - start_time).total_seconds()
                await self.model_usage_analytics.track_usage(
                    model_id=model_type if 'model_type' in locals() else "unknown",
                    event_type=UsageEventType.GENERATION_FAILED,
                    duration_seconds=generation_time,
                    success=False,
                    error_message=str(e)
                )
            
            # Use integrated error handler for critical errors
            if self.error_handler:
                error_info = self.error_handler.handle_error(e, {
                    'task_id': task.id,
                    'model_type': model_type if 'model_type' in locals() else "unknown",
                    'stage': 'generation'
                })
                logger.error(f"Error handled: {error_info.message}")
                
                # Send error details via WebSocket
                if self.websocket_manager:
                    await self.websocket_manager.send_alert(
                        alert_type="generation_error",
                        message=error_info.message,
                        severity="error",
                        task_id=task.id,
                        recovery_suggestions=getattr(error_info, 'recovery_suggestions', [])
                    )
            
            # Complete performance monitoring with error
            if self.performance_monitor and performance_metrics:
                self.performance_monitor.complete_task_monitoring(
                    str(task.id), success=False, error_category="critical_error"
                )
            
            db.commit()
            return False
    
    async def _run_enhanced_generation(self, task: GenerationTaskDB, db: Session, model_type: str) -> bool:
        """Run generation using enhanced model availability system with intelligent fallback"""
        try:
            logger.info(f"Running enhanced generation for model: {model_type}")
            
            # Step 1: Check model availability and health
            model_status = await self.model_availability_manager._check_single_model_availability(model_type)
            logger.info(f"Model {model_type} status: {model_status.availability_status}")
            
            # Step 2: Handle model availability based on status
            if model_status.availability_status == ModelAvailabilityStatus.AVAILABLE:
                # Model is available, check health before proceeding
                if self.model_health_monitor:
                    health_result = await self.model_health_monitor.check_model_integrity(model_type)
                    if not health_result.is_healthy:
                        logger.warning(f"Model {model_type} health check failed: {health_result.issues}")
                        # Try to repair the model
                        if self.enhanced_model_downloader:
                            repair_result = await self.enhanced_model_downloader.verify_and_repair_model(model_type)
                            if not repair_result.success:
                                logger.error(f"Model repair failed: {repair_result.error_message}")
                                return await self._handle_model_unavailable(task, db, model_type, "health_check_failed")
                
                # Model is healthy, proceed with generation
                return await self._run_real_generation_with_monitoring(task, db, model_type)
                
            elif model_status.availability_status == ModelAvailabilityStatus.DOWNLOADING:
                # Model is currently downloading, use intelligent fallback
                return await self._handle_downloading_model(task, db, model_type, model_status)
                
            elif model_status.availability_status in [ModelAvailabilityStatus.MISSING, ModelAvailabilityStatus.CORRUPTED]:
                # Model is missing or corrupted, try to download/repair
                return await self._handle_missing_or_corrupted_model(task, db, model_type, model_status)
                
            else:
                # Unknown status, use fallback
                logger.warning(f"Unknown model status: {model_status.availability_status}")
                return await self._handle_model_unavailable(task, db, model_type, "unknown_status")
                
        except Exception as e:
            logger.error(f"Enhanced generation failed for task {task.id}: {e}")
            
            # Use enhanced error recovery
            if self.fallback_recovery_system:
                failure_type = self._determine_failure_type(e, model_type)
                recovery_result = await self.fallback_recovery_system.attempt_recovery(
                    task, db, failure_type, str(e)
                )
                if recovery_result.success:
                    return True
            
            # Final fallback to mock generation
            return await self._run_mock_generation(task, db, model_type)

    async def _run_legacy_generation(self, task: GenerationTaskDB, db: Session, model_type: str) -> bool:
        """Fallback to legacy generation logic when enhanced system is not available"""
        logger.info(f"Running legacy generation for model: {model_type}")
        
        # Use existing real generation method as fallback
        if self.use_real_generation:
            return await self._run_real_generation(task, db, model_type)
        else:
            return await self._run_simulation_fallback(task, db, model_type)

    async def _run_real_generation_with_monitoring(self, task: GenerationTaskDB, db: Session, model_type: str) -> bool:
        """Run real generation with enhanced monitoring and analytics"""
        try:
            # Track model loading in analytics
            if self.model_usage_analytics:
                await self.model_usage_analytics.track_usage(
                    model_id=model_type,
                    event_type=UsageEventType.MODEL_LOAD
                )
            
            # Monitor model performance during generation
            performance_start = datetime.now()
            
            # Run the actual generation using existing real generation logic
            result = await self._run_real_generation(task, db, model_type)
            
            # Track performance metrics
            if result and self.model_health_monitor:
                generation_time = (datetime.now() - performance_start).total_seconds()
                generation_metrics = {
                    "generation_time": generation_time,
                    "success": True,
                    "resolution": task.resolution,
                    "steps": task.steps,
                    "prompt_length": len(task.prompt) if task.prompt else 0
                }
                await self.model_health_monitor.monitor_model_performance(
                    model_id=model_type,
                    generation_metrics=generation_metrics
                )
            
            return result
            
        except Exception as e:
            # Track performance failure
            if self.model_health_monitor:
                generation_time = (datetime.now() - performance_start).total_seconds()
                generation_metrics = {
                    "generation_time": generation_time,
                    "success": False,
                    "error_message": str(e),
                    "resolution": task.resolution,
                    "steps": task.steps
                }
                await self.model_health_monitor.monitor_model_performance(
                    model_id=model_type,
                    generation_metrics=generation_metrics
                )
            raise

    async def _handle_downloading_model(self, task: GenerationTaskDB, db: Session, model_type: str, model_status) -> bool:
        """Handle case where model is currently downloading"""
        try:
            if self.intelligent_fallback_manager:
                # Get fallback strategy for downloading model
                requirements = GenerationRequirements(
                    model_type=model_type,
                    quality="medium",
                    speed="medium",
                    resolution=task.resolution
                )
                
                fallback_strategy = await self.intelligent_fallback_manager.get_fallback_strategy(
                    failed_model=model_type,
                    requirements=requirements,
                    error_context={"reason": "model_downloading"}
                )
                
                if fallback_strategy.strategy_type == FallbackType.ALTERNATIVE_MODEL:
                    # Try alternative model
                    logger.info(f"Using alternative model: {fallback_strategy.alternative_model}")
                    return await self._run_real_generation_with_monitoring(task, db, fallback_strategy.alternative_model)
                    
                elif fallback_strategy.strategy_type == FallbackType.QUEUE_AND_WAIT:
                    # Queue the request using intelligent fallback manager
                    if self.intelligent_fallback_manager:
                        queue_result = await self.intelligent_fallback_manager.queue_request_for_downloading_model(
                            model_id=model_type,
                            request_data={
                                "task_id": str(task.id),
                                "prompt": task.prompt,
                                "resolution": task.resolution,
                                "steps": task.steps
                            }
                        )
                        if queue_result.get("success", False):
                            # Send notification about queuing
                            if self.websocket_manager:
                                await self.websocket_manager.send_alert(
                                    alert_type="generation_queued",
                                    message=f"Generation queued. Estimated wait time: {fallback_strategy.estimated_wait_time}",
                                    severity="info",
                                    task_id=task.id
                                )
                            return True
                
            # Default fallback to mock generation
            return await self._run_mock_generation(task, db, model_type)
            
        except Exception as e:
            logger.error(f"Error handling downloading model: {e}")
            return await self._run_mock_generation(task, db, model_type)

    async def _handle_missing_or_corrupted_model(self, task: GenerationTaskDB, db: Session, model_type: str, model_status) -> bool:
        """Handle case where model is missing or corrupted"""
        try:
            if self.enhanced_model_downloader:
                # Try to download/repair the model with retry logic
                logger.info(f"Attempting to download/repair model: {model_type}")
                
                # Update task status to indicate download attempt
                task.progress = 10
                task.status = TaskStatusEnum.PROCESSING
                db.commit()
                
                # Send notification about download attempt
                if self.websocket_manager:
                    await self.websocket_manager.send_alert(
                        alert_type="model_download_started",
                        message=f"Downloading required model: {model_type}",
                        severity="info",
                        task_id=task.id
                    )
                
                # Attempt download with retry logic
                download_result = await self.enhanced_model_downloader.download_with_retry(
                    model_id=model_type,
                    max_retries=3
                )
                
                if download_result.success:
                    logger.info(f"Model {model_type} downloaded successfully")
                    # Model is now available, proceed with generation
                    return await self._run_real_generation_with_monitoring(task, db, model_type)
                else:
                    logger.error(f"Model download failed: {download_result.error_message}")
                    # Download failed, use intelligent fallback
                    return await self._handle_model_unavailable(task, db, model_type, "download_failed")
            
            # No enhanced downloader available, use fallback
            return await self._handle_model_unavailable(task, db, model_type, "no_downloader")
            
        except Exception as e:
            logger.error(f"Error handling missing/corrupted model: {e}")
            return await self._handle_model_unavailable(task, db, model_type, "error_during_repair")

    async def _handle_model_unavailable(self, task: GenerationTaskDB, db: Session, model_type: str, reason: str) -> bool:
        """Handle case where model is unavailable and cannot be recovered"""
        try:
            if self.intelligent_fallback_manager:
                # Get intelligent fallback suggestions
                requirements = GenerationRequirements(
                    model_type=model_type,
                    quality="medium",
                    speed="medium",
                    resolution=task.resolution
                )
                
                suggestion = await self.intelligent_fallback_manager.suggest_alternative_model(
                    requested_model=model_type,
                    requirements=requirements
                )
                
                if suggestion and suggestion.compatibility_score > 0.7:
                    logger.info(f"Using suggested alternative model: {suggestion.suggested_model}")
                    
                    # Send notification about alternative model
                    if self.websocket_manager:
                        await self.websocket_manager.send_alert(
                            alert_type="alternative_model_used",
                            message=f"Using alternative model: {suggestion.suggested_model} (compatibility: {suggestion.compatibility_score:.1%})",
                            severity="warning",
                            task_id=task.id
                        )
                    
                    return await self._run_real_generation_with_monitoring(task, db, suggestion.suggested_model)
            
            # No good alternatives, fall back to mock generation with detailed explanation
            logger.warning(f"No suitable alternatives for {model_type}, using mock generation. Reason: {reason}")
            
            # Send detailed notification about fallback
            if self.websocket_manager:
                await self.websocket_manager.send_alert(
                    alert_type="fallback_to_mock",
                    message=f"Model {model_type} unavailable ({reason}). Using mock generation.",
                    severity="warning",
                    task_id=task.id,
                    recovery_suggestions=[
                        "Check your internet connection",
                        "Verify available disk space",
                        "Try a different model",
                        "Contact support if the issue persists"
                    ]
                )
            
            return await self._run_mock_generation(task, db, model_type)
            
        except Exception as e:
            logger.error(f"Error in model unavailable handler: {e}")
            return await self._run_mock_generation(task, db, model_type)

    async def _run_enhanced_generation(self, task: GenerationTaskDB, db: Session, model_type: str) -> bool:
        """Run generation using enhanced model availability system with intelligent fallback"""
        try:
            logger.info(f"Running enhanced generation for model: {model_type}")
            
            # Step 1: Check model availability and health
            model_status = await self.model_availability_manager._check_single_model_availability(model_type)
            logger.info(f"Model {model_type} status: {model_status.availability_status}")
            
            # Step 2: Handle model availability based on status
            if model_status.availability_status == ModelAvailabilityStatus.AVAILABLE:
                # Model is available, check health before proceeding
                if self.model_health_monitor:
                    health_result = await self.model_health_monitor.check_model_integrity(model_type)
                    if not health_result.is_healthy:
                        logger.warning(f"Model {model_type} health check failed: {health_result.issues}")
                        # Try to repair the model
                        if self.enhanced_model_downloader:
                            repair_result = await self.enhanced_model_downloader.verify_and_repair_model(model_type)
                            if not repair_result.success:
                                logger.error(f"Model repair failed: {repair_result.error_message}")
                                return await self._handle_model_unavailable(task, db, model_type, "health_check_failed")
                
                # Model is healthy, proceed with generation
                return await self._run_real_generation_with_monitoring(task, db, model_type)
                
            elif model_status.availability_status == ModelAvailabilityStatus.DOWNLOADING:
                # Model is currently downloading, use intelligent fallback
                return await self._handle_downloading_model(task, db, model_type, model_status)
                
            elif model_status.availability_status in [ModelAvailabilityStatus.MISSING, ModelAvailabilityStatus.CORRUPTED]:
                # Model is missing or corrupted, try to download/repair
                return await self._handle_missing_or_corrupted_model(task, db, model_type, model_status)
                
            else:
                # Unknown status, use fallback
                logger.warning(f"Unknown model status: {model_status.availability_status}")
                return await self._handle_model_unavailable(task, db, model_type, "unknown_status")
                
        except Exception as e:
            logger.error(f"Enhanced generation failed for task {task.id}: {e}")
            
            # Use enhanced error recovery
            if self.fallback_recovery_system:
                failure_type = self._determine_failure_type(e, model_type)
                recovery_result = await self.fallback_recovery_system.attempt_recovery(
                    task, db, failure_type, str(e)
                )
                if recovery_result.success:
                    return True
            
            # Final fallback to mock generation
            return await self._run_mock_generation(task, db, model_type)

    async def _run_legacy_generation(self, task: GenerationTaskDB, db: Session, model_type: str) -> bool:
        """Fallback to legacy generation logic when enhanced system is not available"""
        logger.info(f"Running legacy generation for model: {model_type}")
        
        # Use existing real generation method as fallback
        if self.use_real_generation:
            return await self._run_real_generation(task, db, model_type)
        else:
            return await self._run_simulation_fallback(task, db, model_type)

    async def _run_real_generation_with_monitoring(self, task: GenerationTaskDB, db: Session, model_type: str) -> bool:
        """Run real generation with enhanced monitoring and analytics"""
        try:
            # Track model loading in analytics
            if self.model_usage_analytics:
                await self.model_usage_analytics.track_usage(
                    model_id=model_type,
                    event_type=UsageEventType.MODEL_LOAD
                )
            
            # Monitor model performance during generation
            performance_start = datetime.now()
            
            # Run the actual generation using existing real generation logic
            result = await self._run_real_generation(task, db, model_type)
            
            # Track performance metrics
            if result and self.model_health_monitor:
                generation_time = (datetime.now() - performance_start).total_seconds()
                generation_metrics = {
                    "generation_time": generation_time,
                    "success": True,
                    "resolution": task.resolution,
                    "steps": task.steps,
                    "prompt_length": len(task.prompt) if task.prompt else 0
                }
                await self.model_health_monitor.monitor_model_performance(
                    model_id=model_type,
                    generation_metrics=generation_metrics
                )
            
            return result
            
        except Exception as e:
            # Track performance failure
            if self.model_health_monitor:
                generation_time = (datetime.now() - performance_start).total_seconds()
                generation_metrics = {
                    "generation_time": generation_time,
                    "success": False,
                    "error_message": str(e),
                    "resolution": task.resolution,
                    "steps": task.steps
                }
                await self.model_health_monitor.monitor_model_performance(
                    model_id=model_type,
                    generation_metrics=generation_metrics
                )
            raise

    async def _handle_downloading_model(self, task: GenerationTaskDB, db: Session, model_type: str, model_status) -> bool:
        """Handle case where model is currently downloading"""
        try:
            if self.intelligent_fallback_manager:
                # Get fallback strategy for downloading model
                requirements = GenerationRequirements(
                    model_type=model_type,
                    quality="medium",
                    speed="medium",
                    resolution=task.resolution
                )
                
                fallback_strategy = await self.intelligent_fallback_manager.get_fallback_strategy(
                    failed_model=model_type,
                    requirements=requirements,
                    error_context={"reason": "model_downloading"}
                )
                
                if fallback_strategy.strategy_type == FallbackType.ALTERNATIVE_MODEL:
                    # Try alternative model
                    logger.info(f"Using alternative model: {fallback_strategy.alternative_model}")
                    return await self._run_real_generation_with_monitoring(task, db, fallback_strategy.alternative_model)
                    
                elif fallback_strategy.strategy_type == FallbackType.QUEUE_AND_WAIT:
                    # Queue the request using intelligent fallback manager
                    if self.intelligent_fallback_manager:
                        queue_result = await self.intelligent_fallback_manager.queue_request_for_downloading_model(
                            model_id=model_type,
                            request_data={
                                "task_id": str(task.id),
                                "prompt": task.prompt,
                                "resolution": task.resolution,
                                "steps": task.steps
                            }
                        )
                        if queue_result.get("success", False):
                            # Send notification about queuing
                            if self.websocket_manager:
                                await self.websocket_manager.send_alert(
                                    alert_type="generation_queued",
                                    message=f"Generation queued. Estimated wait time: {fallback_strategy.estimated_wait_time}",
                                    severity="info",
                                    task_id=task.id
                                )
                            return True
                
            # Default fallback to mock generation
            return await self._run_mock_generation(task, db, model_type)
            
        except Exception as e:
            logger.error(f"Error handling downloading model: {e}")
            return await self._run_mock_generation(task, db, model_type)

    async def _handle_missing_or_corrupted_model(self, task: GenerationTaskDB, db: Session, model_type: str, model_status) -> bool:
        """Handle case where model is missing or corrupted"""
        try:
            if self.enhanced_model_downloader:
                # Try to download/repair the model with retry logic
                logger.info(f"Attempting to download/repair model: {model_type}")
                
                # Update task status to indicate download attempt
                task.progress = 10
                task.status = TaskStatusEnum.PROCESSING
                db.commit()
                
                # Send notification about download attempt
                if self.websocket_manager:
                    await self.websocket_manager.send_alert(
                        alert_type="model_download_started",
                        message=f"Downloading required model: {model_type}",
                        severity="info",
                        task_id=task.id
                    )
                
                # Attempt download with retry logic
                download_result = await self.enhanced_model_downloader.download_with_retry(
                    model_id=model_type,
                    download_url=f"https://example.com/models/{model_type}",  # This would be configured
                    max_retries=3
                )
                
                if download_result.success:
                    logger.info(f"Model {model_type} downloaded successfully")
                    # Model is now available, proceed with generation
                    return await self._run_real_generation_with_monitoring(task, db, model_type)
                else:
                    logger.error(f"Model download failed: {download_result.error_message}")
                    # Download failed, use intelligent fallback
                    return await self._handle_model_unavailable(task, db, model_type, "download_failed")
            
            # No enhanced downloader available, use fallback
            return await self._handle_model_unavailable(task, db, model_type, "no_downloader")
            
        except Exception as e:
            logger.error(f"Error handling missing/corrupted model: {e}")
            return await self._handle_model_unavailable(task, db, model_type, "error_during_repair")

    async def _handle_model_unavailable(self, task: GenerationTaskDB, db: Session, model_type: str, reason: str) -> bool:
        """Handle case where model is unavailable and cannot be recovered"""
        try:
            if self.intelligent_fallback_manager:
                # Get intelligent fallback suggestions
                requirements = GenerationRequirements(
                    model_type=model_type,
                    quality="medium",
                    speed="medium",
                    resolution=task.resolution
                )
                
                suggestion = await self.intelligent_fallback_manager.suggest_alternative_model(
                    requested_model=model_type,
                    requirements=requirements
                )
                
                if suggestion and suggestion.compatibility_score > 0.7:
                    logger.info(f"Using suggested alternative model: {suggestion.suggested_model}")
                    
                    # Send notification about alternative model
                    if self.websocket_manager:
                        await self.websocket_manager.send_alert(
                            alert_type="alternative_model_used",
                            message=f"Using alternative model: {suggestion.suggested_model} (compatibility: {suggestion.compatibility_score:.1%})",
                            severity="warning",
                            task_id=task.id
                        )
                    
                    return await self._run_real_generation_with_monitoring(task, db, suggestion.suggested_model)
            
            # No good alternatives, fall back to mock generation with detailed explanation
            logger.warning(f"No suitable alternatives for {model_type}, using mock generation. Reason: {reason}")
            
            # Send detailed notification about fallback
            if self.websocket_manager:
                await self.websocket_manager.send_alert(
                    alert_type="fallback_to_mock",
                    message=f"Model {model_type} unavailable ({reason}). Using mock generation.",
                    severity="warning",
                    task_id=task.id,
                    recovery_suggestions=[
                        "Check your internet connection",
                        "Verify available disk space",
                        "Try a different model",
                        "Contact support if the issue persists"
                    ]
                )
            
            return await self._run_mock_generation(task, db, model_type)
            
        except Exception as e:
            logger.error(f"Error in model unavailable handler: {e}")
            return await self._run_mock_generation(task, db, model_type)

    async def _run_real_generation(self, task: GenerationTaskDB, db: Session, model_type: str) -> bool:
        """Run real AI generation using ModelIntegrationBridge and RealGenerationPipeline with hardware optimization"""
        try:
            # Check VRAM availability before starting
            if self.vram_monitor:
                # Estimate VRAM requirements based on model type and resolution
                estimated_vram_gb = self._estimate_vram_requirements(model_type, task.resolution)
                vram_available, vram_message = self.vram_monitor.check_vram_availability(estimated_vram_gb)
                
                if not vram_available:
                    logger.warning(f"VRAM check failed: {vram_message}")
                    # Get optimization suggestions
                    suggestions = self.vram_monitor.get_optimization_suggestions()
                    if suggestions:
                        logger.info(f"VRAM optimization suggestions: {', '.join(suggestions[:3])}")
                    
                    # Apply automatic VRAM optimizations if available
                    await self._apply_vram_optimizations()
                else:
                    logger.info(f"VRAM check passed: {vram_message}")
            
            # Ensure model is available (will download if necessary)
            task.progress = 10
            db.commit()
            
            logger.info(f"Ensuring model {model_type} is available")
            model_available = await self.model_integration_bridge.ensure_model_available(model_type)
            
            if not model_available:
                error_msg = f"Model {model_type} is not available and could not be downloaded"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            # Apply hardware optimizations before model loading
            task.progress = 20
            db.commit()
            
            if self.wan22_system_optimizer:
                logger.info("Applying hardware optimizations before model loading")
                await self._apply_pre_generation_optimizations(model_type)
            
            # Load the model with optimization
            task.progress = 25
            db.commit()
            
            logger.info(f"Loading model with hardware optimization: {model_type}")
            model_loaded, load_message = await self.model_integration_bridge.load_model_with_optimization(model_type)
            
            if not model_loaded:
                error_msg = f"Failed to load model {model_type}: {load_message}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            logger.info(f"Model loaded successfully with optimization: {load_message}")
            
            # Monitor VRAM after model loading
            if self.vram_monitor:
                usage = self.vram_monitor.get_current_vram_usage()
                if "error" not in usage:
                    logger.info(f"VRAM usage after model loading: {usage['allocated_gb']:.1f}GB allocated, "
                               f"{usage['usage_percent']:.1f}% of total")
                    
                    # Check if we're approaching VRAM limits
                    if usage.get("optimal_usage_percent", 0) > 85:
                        logger.warning("High VRAM usage detected, monitoring for potential issues")
            
            # Prepare generation parameters
            task.progress = 30
            db.commit()
            
            generation_params = {
                "prompt": task.prompt,
                "resolution": task.resolution,
                "steps": task.steps,
                "image_path": task.image_path,
                "lora_path": task.lora_path,
                "lora_strength": task.lora_strength
            }
            
            # Send WebSocket notification about real generation start
            if self.websocket_manager:
                await self.websocket_manager.send_alert(
                    alert_type="generation_started",
                    message=f"Starting real AI generation with {model_type}",
                    severity="info",
                    task_id=task.id,
                    model_type=model_type
                )
            
            # Run the actual generation using RealGenerationPipeline
            task.progress = 40
            db.commit()
            
            logger.info(f"Starting real generation with optimized pipeline for {model_type}")
            
            # Create progress callback to update database during generation
            async def progress_callback(progress_update, message: str = ""):
                try:
                    # Handle both ProgressUpdate object and direct float
                    if hasattr(progress_update, 'progress_percent'):
                        # ProgressUpdate object from real generation pipeline
                        progress_percent = progress_update.progress_percent
                        message = progress_update.message or message
                    else:
                        # Direct float value (legacy support)
                        progress_percent = float(progress_update)
                    
                    # Map generation progress (0-100%) to task progress (40-95%)
                    task_progress = 40 + int((progress_percent / 100) * 55)
                    task.progress = min(task_progress, 95)
                    db.commit()
                    
                    # Send WebSocket update if available
                    if self.websocket_manager:
                        await self.websocket_manager.send_progress_update(
                            task_id=task.id,
                            progress=task.progress,
                            message=message or f"Generating... {progress_percent:.1f}%",
                            stage="generation"
                        )
                    
                    logger.debug(f"Generation progress: {progress_percent:.1f}% -> Task progress: {task.progress}%")
                except Exception as e:
                    logger.warning(f"Failed to update progress: {e}")
            
            generation_result = await self.real_generation_pipeline.generate_video_with_optimization(
                model_type=model_type,
                progress_callback=progress_callback,
                **generation_params
            )
            
            if generation_result.success:
                # Update task with successful result
                task.output_path = generation_result.output_path
                task.progress = 100
                task.status = TaskStatusEnum.COMPLETED
                task.completed_at = datetime.utcnow()
                db.commit()
                
                # Send completion notification
                if self.websocket_manager:
                    await self.websocket_manager.send_alert(
                        alert_type="generation_completed",
                        message=f"Real AI generation completed successfully",
                        severity="success",
                        task_id=task.id,
                        output_path=generation_result.output_path,
                        model_type=model_type
                    )
                
                logger.info(f"Real generation completed successfully for task {task.id}")
                return True
            else:
                error_msg = f"Real generation failed: {generation_result.error_message}"
                logger.error(error_msg)
                raise Exception(error_msg)
                
        except Exception as e:
            logger.error(f"Real generation failed for task {task.id}: {e}")
            task.error_message = str(e)
            task.status = TaskStatusEnum.FAILED
            db.commit()
            
            # Send error notification
            if self.websocket_manager:
                await self.websocket_manager.send_alert(
                    alert_type="generation_failed",
                    message=f"Real generation failed: {str(e)}",
                    severity="error",
                    task_id=task.id,
                    model_type=model_type
                )
            
            return False

    async def _run_real_generation_with_monitoring(self, task: GenerationTaskDB, db: Session, model_type: str) -> bool:
        """Run real generation with enhanced monitoring and analytics"""
        try:
            # Track model loading in analytics
            if self.model_usage_analytics:
                await self.model_usage_analytics.track_usage(
                    model_id=model_type,
                    event_type=UsageEventType.MODEL_LOAD
                )
            
            # Monitor model performance during generation
            performance_start = datetime.now()
            
            # Run the actual generation using existing real generation logic
            result = await self._run_real_generation(task, db, model_type)
            
            # Track performance metrics
            if result and self.model_health_monitor:
                generation_time = (datetime.now() - performance_start).total_seconds()
                generation_metrics = {
                    "generation_time": generation_time,
                    "success": True,
                    "resolution": task.resolution,
                    "steps": task.steps,
                    "prompt_length": len(task.prompt) if task.prompt else 0
                }
                await self.model_health_monitor.monitor_model_performance(
                    model_id=model_type,
                    generation_metrics=generation_metrics
                )
            
            return result
            
        except Exception as e:
            # Track performance failure
            if self.model_health_monitor:
                generation_time = (datetime.now() - performance_start).total_seconds()
                generation_metrics = {
                    "generation_time": generation_time,
                    "success": False,
                    "error_message": str(e),
                    "resolution": task.resolution,
                    "steps": task.steps
                }
                await self.model_health_monitor.monitor_model_performance(
                    model_id=model_type,
                    generation_metrics=generation_metrics
                )
            raise

    async def _run_enhanced_generation(self, task: GenerationTaskDB, db: Session, model_type: str) -> bool:
        """Run generation using enhanced model availability system with intelligent fallback"""
        try:
            logger.info(f"Running enhanced generation for model: {model_type}")
            
            # Step 1: Ensure model availability using ModelAvailabilityManager
            model_request_result = await self.model_availability_manager.handle_model_request(model_type)
            
            if model_request_result.success:
                logger.info(f"Model {model_type} is ready for generation")
                
                # Step 2: Perform health check before generation
                if self.model_health_monitor:
                    health_result = await self.model_health_monitor.check_model_integrity(model_type)
                    if not health_result.is_healthy:
                        logger.warning(f"Model {model_type} health check failed: {health_result.issues}")
                        
                        # Attempt automatic repair
                        if self.enhanced_model_downloader:
                            repair_result = await self.enhanced_model_downloader.verify_and_repair_model(model_type)
                            if not repair_result.success:
                                logger.error(f"Model repair failed: {repair_result.error_message}")
                                return await self._handle_model_unavailable_with_enhanced_recovery(
                                    task, db, model_type, {"error": "health_check_failed", "details": repair_result.error_message}
                                )
                            else:
                                logger.info(f"Model {model_type} repaired successfully")
                
                # Step 3: Model is healthy, proceed with generation
                return await self._run_real_generation_with_monitoring(task, db, model_type)
                
            else:
                # Model is not available, use intelligent fallback
                logger.warning(f"Model {model_type} not available: {model_request_result.error_message}")
                return await self._handle_model_unavailable_with_enhanced_recovery(
                    task, db, model_type, model_request_result
                )
                
        except Exception as e:
            logger.error(f"Enhanced generation failed for task {task.id}: {e}")
            
            # Use enhanced error recovery with detailed suggestions
            return await self._handle_generation_error_with_recovery(task, db, model_type, e)
            
            # Prepare generation parameters with hardware optimization
            task.progress = 40
            db.commit()
            
            from backend.core.model_integration_bridge import GenerationParams
            
            # Prepare generation parameters with hardware optimization
            generation_params = GenerationParams(
                prompt=task.prompt,
                model_type=model_type,
                resolution=task.resolution,
                steps=task.steps,
                image_path=task.image_path,
                lora_path=task.lora_path,
                lora_strength=task.lora_strength if task.lora_strength else 1.0,
                guidance_scale=7.5,
                fps=8.0,
                num_frames=16,
                # Hardware optimization parameters
                enable_offload=getattr(self, 'enable_model_offloading', False),
                vae_tile_size=getattr(self, 'vae_tile_size', 256),
                max_vram_usage_gb=getattr(self, 'optimal_vram_usage_gb', None),
                enable_gradient_checkpointing=getattr(self, 'enable_gradient_checkpointing', False),
                enable_tensor_cores=getattr(self, 'enable_tensor_cores', False),
                cpu_worker_threads=getattr(self, 'cpu_worker_threads', None)
            )
            
            # Set up progress callback for real-time updates with VRAM monitoring
            def progress_callback(stage, progress, message, **kwargs):
                """Enhanced progress callback with WebSocket support and VRAM monitoring"""
                try:
                    # Map generation progress to task progress (40-95%)
                    task_progress = 40 + int((progress / 100) * 55)
                    task.progress = min(task_progress, 95)
                    db.commit()
                    
                    # Monitor VRAM during generation
                    vram_info = {}
                    if self.vram_monitor:
                        usage = self.vram_monitor.get_current_vram_usage()
                        if "allocated_gb" in usage:
                            vram_info = {
                                "vram_usage_gb": usage["allocated_gb"],
                                "vram_usage_percent": usage["usage_percent"],
                                "vram_optimal_percent": usage.get("optimal_usage_percent", 0)
                            }
                            
                            # Check for VRAM warnings
                            if usage.get("optimal_usage_percent", 0) > 90:
                                logger.warning(f"High VRAM usage during generation: {usage['allocated_gb']:.1f}GB ({usage['usage_percent']:.1f}%)")
                    
                    # Send WebSocket update if available
                    if self.websocket_manager:
                        asyncio.create_task(self._send_websocket_progress_update(
                            task.id, task_progress, message, stage.value, vram_info
                        ))
                    
                    logger.info(f"Generation progress: {progress}% - {message}")
                    if vram_info:
                        logger.debug(f"VRAM usage: {vram_info['vram_usage_gb']:.1f}GB ({vram_info['vram_usage_percent']:.1f}%)")
                        
                except Exception as e:
                    logger.warning(f"Error updating progress: {e}")
            
            # Generate using real pipeline based on model type
            task.progress = 50
            db.commit()
            
            logger.info(f"Starting real generation with {model_type}")
            
            if model_type == "t2v":
                result = await self.real_generation_pipeline.generate_t2v(
                    task.prompt, generation_params, progress_callback
                )
            elif model_type == "i2v" and task.image_path:
                result = await self.real_generation_pipeline.generate_i2v(
                    task.image_path, task.prompt, generation_params, progress_callback
                )
            elif model_type == "ti2v" and task.image_path:
                result = await self.real_generation_pipeline.generate_ti2v(
                    task.image_path, task.prompt, generation_params, progress_callback
                )
            else:
                # Fallback to bridge generation
                logger.warning(f"Unsupported model type or missing image: {model_type}")
                result = await self.model_integration_bridge.generate_with_existing_pipeline(generation_params)
            
            # Process generation result
            if result.success:
                task.output_path = result.output_path
                task.generation_time_minutes = result.generation_time_seconds / 60.0  # Convert to minutes
                task.progress = 100
                db.commit()
                
                # Send completion WebSocket update
                if self.websocket_manager:
                    await self._send_websocket_completion_update(task.id, result)
                
                # Complete performance monitoring on success
                if self.performance_monitor:
                    completed_metrics = self.performance_monitor.complete_task_monitoring(
                        str(task.id), success=True
                    )
                    
                    # Update task with performance metrics
                    if completed_metrics:
                        task.model_used = completed_metrics.model_type
                        task.generation_time_seconds = completed_metrics.generation_time_seconds
                        task.peak_vram_usage_mb = completed_metrics.peak_vram_usage_mb
                        task.optimizations_applied = json.dumps(completed_metrics.optimizations_applied)
                
                logger.info(f"Real generation completed successfully for task {task.id}")
                return True
            else:
                error_msg = result.error_message or "Real generation failed"
                logger.error(f"Real generation failed for task {task.id}: {error_msg}")
                raise Exception(error_msg)
                
        except Exception as e:
            logger.error(f"Error in real generation for task {task.id}: {e}")
            raise  # Re-raise to be handled by caller
    
    async def _run_simulation_fallback(self, task: GenerationTaskDB, db: Session, model_type: str) -> bool:
        """Run simulation fallback when real generation is not available or fails"""
        try:
            logger.info(f"Running simulation fallback for task {task.id}")
            
            # Prepare simulation parameters
            simulation_params = {
                "prompt": task.prompt,
                "model_type": model_type,
                "resolution": task.resolution,
                "steps": task.steps,
                "image_path": task.image_path,
                "lora_path": task.lora_path,
                "lora_strength": task.lora_strength
            }
            
            # Run simulation with progress updates
            await self._simulate_generation(task, db, simulation_params)
            
            # Set output path
            output_filename = f"generated_{task.id}.mp4"
            output_path = f"outputs/{output_filename}"
            task.output_path = output_path
            task.progress = 100
            db.commit()
            
            # Complete performance monitoring for simulation fallback
            if self.performance_monitor:
                self.performance_monitor.complete_task_monitoring(
                    str(task.id), success=True
                )
            
            logger.info(f"Simulation fallback completed for task {task.id}")
            return True
            
        except Exception as e:
            logger.error(f"Simulation fallback failed for task {task.id}: {e}")
            task.error_message = f"Simulation fallback error: {str(e)}"
            db.commit()
            return False
    
    async def _send_websocket_progress_update(self, task_id: str, progress: int, message: str, stage: str, vram_info: Dict = None):
        """Send progress update via WebSocket with VRAM monitoring"""
        try:
            if self.websocket_manager:
                update_data = {
                    "alert_type": "generation_progress",
                    "message": message,
                    "severity": "info",
                    "task_id": task_id,
                    "progress": progress,
                    "stage": stage
                }
                
                # Add VRAM information if available
                if vram_info:
                    update_data.update(vram_info)
                    
                    # Adjust severity based on VRAM usage
                    if vram_info.get("vram_optimal_percent", 0) > 95:
                        update_data["severity"] = "error"
                        update_data["message"] += " (Critical VRAM usage)"
                    elif vram_info.get("vram_optimal_percent", 0) > 90:
                        update_data["severity"] = "warning"
                        update_data["message"] += " (High VRAM usage)"
                
                await self.websocket_manager.send_alert(**update_data)
        except Exception as e:
            logger.warning(f"Failed to send WebSocket progress update: {e}")
    
    async def _send_websocket_completion_update(self, task_id: str, result):
        """Send completion update via WebSocket"""
        try:
            if self.websocket_manager:
                await self.websocket_manager.send_alert(
                    alert_type="generation_completed",
                    message=f"Generation completed in {result.generation_time_seconds:.1f}s",
                    severity="success",
                    task_id=task_id,
                    output_path=result.output_path,
                    generation_time=result.generation_time_seconds,
                    model_used=result.model_used
                )
        except Exception as e:
            logger.warning(f"Failed to send WebSocket completion update: {e}")
    
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
        """Get enhanced queue status with real AI integration and hardware optimization information"""
        db = SessionLocal()
        try:
            total_tasks = db.query(GenerationTaskDB).count()
            pending_tasks = db.query(GenerationTaskDB).filter(
                GenerationTaskDB.status == TaskStatusEnum.PENDING
            ).count()
            processing_tasks = db.query(GenerationTaskDB).filter(
                GenerationTaskDB.status == TaskStatusEnum.PROCESSING
            ).count()
            completed_tasks = db.query(GenerationTaskDB).filter(
                GenerationTaskDB.status == TaskStatusEnum.COMPLETED
            ).count()
            failed_tasks = db.query(GenerationTaskDB).filter(
                GenerationTaskDB.status == TaskStatusEnum.FAILED
            ).count()
            
            # Get model status if bridge is available
            model_status = {}
            if self.model_integration_bridge:
                try:
                    model_status = self.model_integration_bridge.get_model_status_from_existing_system()
                except Exception as e:
                    logger.warning(f"Could not get model status: {e}")
            
            # Get hardware optimization status
            hardware_optimization_status = {
                "optimizer_available": self.wan22_system_optimizer is not None,
                "hardware_profile_loaded": self.hardware_profile is not None,
                "optimization_applied": self.optimization_applied,
                "vram_monitoring_enabled": self.vram_monitor is not None
            }
            
            # Add hardware profile information if available
            if self.hardware_profile:
                hardware_optimization_status.update({
                    "gpu_model": self.hardware_profile.gpu_model,
                    "vram_gb": self.hardware_profile.vram_gb,
                    "cpu_model": self.hardware_profile.cpu_model,
                    "cpu_cores": self.hardware_profile.cpu_cores,
                    "total_memory_gb": self.hardware_profile.total_memory_gb
                })
            
            # Add current VRAM usage if monitoring is available
            if self.vram_monitor:
                try:
                    vram_usage = self.vram_monitor.get_current_vram_usage()
                    if "allocated_gb" in vram_usage:
                        hardware_optimization_status.update({
                            "current_vram_usage_gb": vram_usage["allocated_gb"],
                            "current_vram_usage_percent": vram_usage["usage_percent"],
                            "optimal_vram_usage_gb": self.vram_monitor.optimal_usage_gb
                        })
                except Exception as e:
                    logger.warning(f"Could not get current VRAM usage: {e}")
            
            # Add optimization settings if applied
            if self.optimization_applied:
                optimization_settings = {}
                if hasattr(self, 'optimal_vram_usage_gb'):
                    optimization_settings["optimal_vram_usage_gb"] = self.optimal_vram_usage_gb
                if hasattr(self, 'enable_model_offloading'):
                    optimization_settings["model_offloading_enabled"] = self.enable_model_offloading
                if hasattr(self, 'enable_tensor_cores'):
                    optimization_settings["tensor_cores_enabled"] = self.enable_tensor_cores
                if hasattr(self, 'cpu_worker_threads'):
                    optimization_settings["cpu_worker_threads"] = self.cpu_worker_threads
                
                hardware_optimization_status["optimization_settings"] = optimization_settings
            
            return {
                "total_tasks": total_tasks,
                "pending_tasks": pending_tasks,
                "processing_tasks": processing_tasks,
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "current_task": self.current_task,
                "worker_active": self.is_processing and self.processing_thread and self.processing_thread.is_alive(),
                "real_generation_enabled": self.use_real_generation,
                "fallback_enabled": self.fallback_to_simulation,
                "model_bridge_available": self.model_integration_bridge is not None,
                "real_pipeline_available": self.real_generation_pipeline is not None,
                "model_status": model_status,
                "hardware_optimization": hardware_optimization_status
            }
        finally:
            db.close()
    
    def set_generation_mode(self, use_real: bool, fallback: bool = True):
        """Set generation mode (real vs simulation)"""
        self.use_real_generation = use_real
        self.fallback_to_simulation = fallback
        logger.info(f"Generation mode updated: real={use_real}, fallback={fallback}")
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get generation statistics with hardware optimization information"""
        stats = {
            "real_generation_enabled": self.use_real_generation,
            "fallback_enabled": self.fallback_to_simulation,
            "components_status": {
                "model_bridge": self.model_integration_bridge is not None,
                "real_pipeline": self.real_generation_pipeline is not None,
                "error_handler": self.error_handler is not None,
                "websocket_manager": self.websocket_manager is not None,
                "hardware_optimizer": self.wan22_system_optimizer is not None,
                "vram_monitor": self.vram_monitor is not None
            }
        }
        
        # Add hardware optimization stats
        if self.wan22_system_optimizer:
            try:
                # Get system health metrics
                health_metrics = self.wan22_system_optimizer.monitor_system_health()
                stats["system_health"] = {
                    "cpu_usage_percent": health_metrics.cpu_usage_percent,
                    "memory_usage_gb": health_metrics.memory_usage_gb,
                    "vram_usage_mb": health_metrics.vram_usage_mb,
                    "vram_total_mb": health_metrics.vram_total_mb,
                    "gpu_temperature": health_metrics.gpu_temperature,
                    "timestamp": health_metrics.timestamp
                }
                
                # Get optimization history summary
                optimization_history = self.wan22_system_optimizer.get_optimization_history()
                if optimization_history:
                    recent_optimizations = optimization_history[-3:]  # Last 3 operations
                    stats["recent_optimizations"] = [
                        {
                            "operation": opt["operation"],
                            "success": opt["success"],
                            "timestamp": opt["timestamp"],
                            "optimizations_count": len(opt.get("optimizations_applied", []))
                        }
                        for opt in recent_optimizations
                    ]
                
            except Exception as e:
                logger.warning(f"Could not get hardware optimization stats: {e}")
                stats["hardware_optimization_error"] = str(e)
        
        # Add VRAM monitoring stats
        if self.vram_monitor:
            try:
                vram_usage = self.vram_monitor.get_current_vram_usage()
                if "allocated_gb" in vram_usage:
                    stats["vram_monitoring"] = {
                        "current_usage_gb": vram_usage["allocated_gb"],
                        "current_usage_percent": vram_usage["usage_percent"],
                        "optimal_usage_gb": self.vram_monitor.optimal_usage_gb,
                        "total_vram_gb": self.vram_monitor.total_vram_gb,
                        "optimization_suggestions_count": len(self.vram_monitor.get_optimization_suggestions())
                    }
            except Exception as e:
                logger.warning(f"Could not get VRAM monitoring stats: {e}")
        
        # Add pipeline stats if available
        if self.real_generation_pipeline:
            try:
                stats["pipeline_stats"] = {
                    "generation_count": getattr(self.real_generation_pipeline, '_generation_count', 0),
                    "total_generation_time": getattr(self.real_generation_pipeline, '_total_generation_time', 0.0)
                }
            except Exception as e:
                logger.warning(f"Could not get pipeline stats: {e}")
        
        return stats
    
    async def submit_generation_task(
        self,
        prompt: str,
        model_type: str,
        resolution: str = "1280x720",
        steps: int = 50,
        image_path: Optional[str] = None,
        end_image_path: Optional[str] = None,
        lora_path: Optional[str] = None,
        lora_strength: float = 1.0
    ):
        """Submit a generation task to the enhanced generation service"""
        try:
            from backend.repositories.database import SessionLocal, GenerationTaskDB, TaskStatusEnum, ModelTypeEnum
            
            # Convert model type string to enum
            model_type_enum_map = {
                "T2V-A14B": ModelTypeEnum.T2V_A14B,
                "I2V-A14B": ModelTypeEnum.I2V_A14B,
                "TI2V-5B": ModelTypeEnum.TI2V_5B
            }
            
            model_type_enum = model_type_enum_map.get(model_type)
            if not model_type_enum:
                return type('TaskResult', (), {
                    'success': False,
                    'task_id': None,
                    'error_message': f"Invalid model type: {model_type}"
                })()
            
            # Create task in database
            db = SessionLocal()
            try:
                task_id = str(uuid.uuid4())
                task = GenerationTaskDB(
                    id=task_id,
                    prompt=prompt,
                    model_type=model_type_enum,
                    resolution=resolution,
                    steps=steps,
                    image_path=image_path,
                    end_image_path=end_image_path,
                    lora_path=lora_path,
                    lora_strength=lora_strength,
                    status=TaskStatusEnum.PENDING,
                    progress=0
                )
                
                db.add(task)
                db.commit()
                db.refresh(task)
                
                logger.info(f"Enhanced generation task created: {task_id}")
                
                return type('TaskResult', (), {
                    'success': True,
                    'task_id': task_id,
                    'error_message': None
                })()
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Failed to submit generation task: {e}")
            return type('TaskResult', (), {
                'success': False,
                'task_id': None,
                'error_message': str(e)
            })()
    
    def shutdown(self):
        """Shutdown the enhanced generation service with hardware optimization cleanup"""
        logger.info("Shutting down enhanced generation service with hardware optimization")
        self.is_processing = False
        
        # Stop processing thread
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        # Clean up hardware optimization components
        try:
            if self.wan22_system_optimizer:
                # Save final hardware profile and optimization history
                self.wan22_system_optimizer.save_profile_to_file("hardware_profile_final.json")
                
                # Get final system health metrics
                final_metrics = self.wan22_system_optimizer.monitor_system_health()
                logger.info(f"Final system metrics: CPU {final_metrics.cpu_usage_percent}%, "
                           f"Memory {final_metrics.memory_usage_gb}GB, VRAM {final_metrics.vram_usage_mb}MB")
                
                logger.info("Hardware optimization system cleaned up")
        except Exception as e:
            logger.warning(f"Error cleaning up hardware optimization: {e}")
        
        # Clean up VRAM monitoring
        try:
            if self.vram_monitor:
                # Log final VRAM usage
                final_vram = self.vram_monitor.get_current_vram_usage()
                if "allocated_gb" in final_vram:
                    logger.info(f"Final VRAM usage: {final_vram['allocated_gb']:.1f}GB ({final_vram['usage_percent']:.1f}%)")
                
                self.vram_monitor = None
                logger.info("VRAM monitoring cleaned up")
        except Exception as e:
            logger.warning(f"Error cleaning up VRAM monitoring: {e}")
        
        # Clear GPU cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU cache cleared during shutdown")
        except Exception as e:
            logger.warning(f"Error clearing GPU cache: {e}")
        
        # Clean up real AI components
        try:
            if self.real_generation_pipeline:
                # Clear pipeline cache if available
                if hasattr(self.real_generation_pipeline, '_pipeline_cache'):
                    self.real_generation_pipeline._pipeline_cache.clear()
                logger.info("Real generation pipeline cleaned up")
        except Exception as e:
            logger.warning(f"Error cleaning up real generation pipeline: {e}")
        
        try:
            if self.model_integration_bridge:
                # Clear model cache if available
                if hasattr(self.model_integration_bridge, '_model_cache'):
                    self.model_integration_bridge._model_cache.clear()
                logger.info("Model integration bridge cleaned up")
        except Exception as e:
            logger.warning(f"Error cleaning up model integration bridge: {e}")
        
        logger.info("Enhanced generation service shutdown complete with hardware optimization cleanup")

    async def _run_enhanced_generation(self, task: GenerationTaskDB, db: Session, model_type: str) -> bool:
        """Run generation using enhanced model availability system with intelligent fallback"""
        try:
            logger.info(f"Running enhanced generation for model: {model_type}")
            
            # Step 1: Ensure model availability using ModelAvailabilityManager
            model_request_result = await self.model_availability_manager.handle_model_request(model_type)
            
            if model_request_result.success:
                logger.info(f"Model {model_type} is ready for generation")
                
                # Step 2: Perform health check before generation
                if self.model_health_monitor:
                    health_result = await self.model_health_monitor.check_model_integrity(model_type)
                    if not health_result.is_healthy:
                        logger.warning(f"Model {model_type} health check failed: {health_result.issues}")
                        
                        # Attempt automatic repair
                        if self.enhanced_model_downloader:
                            repair_result = await self.enhanced_model_downloader.verify_and_repair_model(model_type)
                            if not repair_result.success:
                                logger.error(f"Model repair failed: {repair_result.error_message}")
                                return await self._handle_model_unavailable_with_enhanced_recovery(
                                    task, db, model_type, {"error": "health_check_failed", "details": repair_result.error_message}
                                )
                            else:
                                logger.info(f"Model {model_type} repaired successfully")
                
                # Step 3: Model is healthy, proceed with generation
                return await self._run_real_generation_with_monitoring(task, db, model_type)
                
            else:
                # Model is not available, use intelligent fallback
                logger.warning(f"Model {model_type} not available: {model_request_result.error_message}")
                return await self._handle_model_unavailable_with_enhanced_recovery(
                    task, db, model_type, model_request_result
                )
                
        except Exception as e:
            logger.error(f"Enhanced generation failed for task {task.id}: {e}")
            
            # Use enhanced error recovery with detailed suggestions
            return await self._handle_generation_error_with_recovery(task, db, model_type, e)

    async def _handle_model_unavailable_with_enhanced_recovery(self, task: GenerationTaskDB, db: Session, 
                                                             model_type: str, error_context: Dict[str, Any]) -> bool:
        """Handle model unavailability with enhanced recovery strategies"""
        try:
            logger.info(f"Handling model unavailability for {model_type} with enhanced recovery")
            
            # Step 1: Try intelligent fallback suggestions
            if self.intelligent_fallback_manager:
                requirements = GenerationRequirements(
                    model_type=model_type,
                    quality="medium",
                    speed="medium",
                    resolution=task.resolution
                )
                
                # Get fallback strategy
                fallback_strategy = await self.intelligent_fallback_manager.get_fallback_strategy(
                    failed_model=model_type,
                    requirements=requirements,
                    error_context=error_context
                )
                
                # Handle different fallback strategies
                if fallback_strategy.strategy_type == FallbackType.ALTERNATIVE_MODEL:
                    return await self._try_alternative_model(task, db, fallback_strategy.alternative_model)
                    
                elif fallback_strategy.strategy_type == FallbackType.DOWNLOAD_AND_RETRY:
                    return await self._try_download_and_retry(task, db, model_type)
                    
                elif fallback_strategy.strategy_type == FallbackType.QUEUE_AND_WAIT:
                    return await self._queue_generation_request(task, db, model_type, fallback_strategy)
            
            # Step 2: If no intelligent fallback available, use enhanced error recovery
            if self.fallback_recovery_system:
                failure_type = self._determine_failure_type_from_context(error_context, model_type)
                recovery_result = await self.fallback_recovery_system.attempt_recovery(
                    task, db, failure_type, error_context.get("error", "model_unavailable")
                )
                if recovery_result.success:
                    return True
            
            # Step 3: Final fallback to mock generation with detailed explanation
            return await self._run_mock_generation_with_enhanced_context(task, db, model_type, error_context)
            
        except Exception as e:
            logger.error(f"Enhanced recovery failed: {e}")
            return await self._run_mock_generation(task, db, model_type)

    async def _try_alternative_model(self, task: GenerationTaskDB, db: Session, alternative_model: str) -> bool:
        """Try using an alternative model suggested by intelligent fallback"""
        try:
            logger.info(f"Trying alternative model: {alternative_model}")
            
            # Send notification about alternative model usage
            if self.websocket_manager:
                await self.websocket_manager.send_alert(
                    alert_type="alternative_model_used",
                    message=f"Using alternative model: {alternative_model}",
                    severity="info",
                    task_id=task.id
                )
            
            # Track alternative model usage in analytics
            if self.model_usage_analytics:
                await self.model_usage_analytics.track_usage(
                    model_id=alternative_model,
                    event_type=UsageEventType.GENERATION_REQUEST,
                    generation_params={"alternative_for": task.model_type.value}
                )
            
            # Run generation with alternative model
            return await self._run_real_generation_with_monitoring(task, db, alternative_model)
            
        except Exception as e:
            logger.error(f"Alternative model {alternative_model} failed: {e}")
            return False

    async def _try_download_and_retry(self, task: GenerationTaskDB, db: Session, model_type: str) -> bool:
        """Try downloading the model and retrying generation"""
        try:
            logger.info(f"Attempting to download and retry for model: {model_type}")
            
            if not self.enhanced_model_downloader:
                logger.warning("Enhanced model downloader not available")
                return False
            
            # Update task progress to indicate download
            task.progress = 15
            db.commit()
            
            # Send notification about download attempt
            if self.websocket_manager:
                await self.websocket_manager.send_alert(
                    alert_type="model_download_started",
                    message=f"Downloading required model: {model_type}",
                    severity="info",
                    task_id=task.id
                )
            
            # Attempt download with enhanced retry logic
            download_result = await self.enhanced_model_downloader.download_with_retry(
                model_id=model_type,
                max_retries=3
            )
            
            if download_result.success:
                logger.info(f"Model {model_type} downloaded successfully, retrying generation")
                
                # Send success notification
                if self.websocket_manager:
                    await self.websocket_manager.send_alert(
                        alert_type="model_download_completed",
                        message=f"Model {model_type} downloaded successfully",
                        severity="success",
                        task_id=task.id
                    )
                
                # Retry generation with downloaded model
                return await self._run_real_generation_with_monitoring(task, db, model_type)
            else:
                logger.error(f"Model download failed: {download_result.error_message}")
                
                # Send failure notification
                if self.websocket_manager:
                    await self.websocket_manager.send_alert(
                        alert_type="model_download_failed",
                        message=f"Failed to download {model_type}: {download_result.error_message}",
                        severity="error",
                        task_id=task.id
                    )
                
                return False
                
        except Exception as e:
            logger.error(f"Download and retry failed: {e}")
            return False

    async def _queue_generation_request(self, task: GenerationTaskDB, db: Session, 
                                      model_type: str, fallback_strategy) -> bool:
        """Queue the generation request for when model becomes available"""
        try:
            logger.info(f"Queueing generation request for model: {model_type}")
            
            if not self.intelligent_fallback_manager:
                return False
            
            # Queue the request
            queue_result = await self.intelligent_fallback_manager.queue_request_for_downloading_model(
                model_id=model_type,
                request_data={
                    "task_id": str(task.id),
                    "prompt": task.prompt,
                    "resolution": task.resolution,
                    "steps": task.steps,
                    "image_path": task.image_path,
                    "lora_path": task.lora_path,
                    "lora_strength": task.lora_strength
                }
            )
            
            if queue_result.get("success", False):
                # Update task status to queued
                task.status = TaskStatusEnum.PROCESSING
                task.progress = 5
                db.commit()
                
                # Send notification about queuing
                if self.websocket_manager:
                    await self.websocket_manager.send_alert(
                        alert_type="generation_queued",
                        message=f"Generation queued. Estimated wait time: {fallback_strategy.estimated_wait_time}",
                        severity="info",
                        task_id=task.id
                    )
                
                logger.info(f"Generation request queued successfully for task {task.id}")
                return True
            else:
                logger.error(f"Failed to queue generation request: {queue_result}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to queue generation request: {e}")
            return False

    async def _handle_generation_error_with_recovery(self, task: GenerationTaskDB, db: Session, 
                                                   model_type: str, error: Exception) -> bool:
        """Handle generation errors with enhanced recovery and detailed suggestions"""
        try:
            logger.info(f"Handling generation error with enhanced recovery for task {task.id}")
            
            # Categorize the error for better recovery
            error_category = self._categorize_generation_error(error)
            
            # Track error in analytics
            if self.model_usage_analytics:
                await self.model_usage_analytics.track_usage(
                    model_id=model_type,
                    event_type=UsageEventType.GENERATION_FAILED,
                    success=False,
                    error_message=str(error)
                )
            
            # Get recovery suggestions based on error type
            recovery_suggestions = self._get_enhanced_recovery_suggestions(error_category, model_type)
            
            # Try enhanced error recovery
            if self.fallback_recovery_system:
                failure_type = self._determine_failure_type(error, model_type)
                recovery_result = await self.fallback_recovery_system.attempt_recovery(
                    task, db, failure_type, str(error)
                )
                
                if recovery_result.success:
                    logger.info(f"Enhanced error recovery successful for task {task.id}")
                    return True
            
            # Send detailed error notification with recovery suggestions
            if self.websocket_manager:
                await self.websocket_manager.send_alert(
                    alert_type="generation_error_with_recovery",
                    message=f"Generation failed: {str(error)}",
                    severity="error",
                    task_id=task.id,
                    error_category=error_category,
                    recovery_suggestions=recovery_suggestions
                )
            
            # Final fallback to mock generation with error context
            return await self._run_mock_generation_with_enhanced_context(
                task, db, model_type, {"error": str(error), "category": error_category}
            )
            
        except Exception as e:
            logger.error(f"Error recovery handling failed: {e}")
            return await self._run_mock_generation(task, db, model_type)

    async def _run_mock_generation_with_enhanced_context(self, task: GenerationTaskDB, db: Session, 
                                                       model_type: str, error_context: Dict[str, Any]) -> bool:
        """Run mock generation with enhanced error context and recovery suggestions"""
        try:
            logger.info(f"Running enhanced mock generation for task {task.id} with context: {error_context}")
            
            # Update task status with enhanced context
            task.progress = 5
            task.status = TaskStatusEnum.PROCESSING
            db.commit()
            
            # Send enhanced WebSocket notification about mock mode
            if self.websocket_manager:
                await self.websocket_manager.send_alert(
                    alert_type="enhanced_mock_generation",
                    message=f"Using mock generation due to: {error_context.get('error', 'model issues')}",
                    severity="warning",
                    task_id=task.id,
                    mock_mode=True,
                    error_context=error_context,
                    recovery_suggestions=self._get_enhanced_recovery_suggestions(
                        error_context.get('category', 'unknown'), model_type
                    )
                )
            
            # Run the mock generation with enhanced tracking
            result = await self._run_mock_generation(task, db, model_type)
            
            # Track enhanced mock generation in analytics
            if result and self.model_usage_analytics:
                await self.model_usage_analytics.track_usage(
                    model_id=model_type,
                    event_type=UsageEventType.GENERATION_COMPLETE,
                    success=True,
                    generation_params={
                        "mock_mode": True,
                        "error_context": error_context,
                        "resolution": task.resolution,
                        "steps": task.steps
                    }
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced mock generation failed: {e}")
            return await self._run_mock_generation(task, db, model_type)

    def _categorize_generation_error(self, error: Exception) -> str:
        """Categorize generation errors for better recovery handling"""
        error_message = str(error).lower()
        
        if "cuda out of memory" in error_message or "vram" in error_message:
            return "vram_exhaustion"
        elif "model" in error_message and ("load" in error_message or "not found" in error_message):
            return "model_loading_error"
        elif "download" in error_message or "network" in error_message:
            return "download_error"
        elif "corruption" in error_message or "integrity" in error_message:
            return "model_corruption"
        elif "timeout" in error_message:
            return "timeout_error"
        elif "permission" in error_message or "access" in error_message:
            return "permission_error"
        else:
            return "unknown_error"

    def _get_enhanced_recovery_suggestions(self, error_category: str, model_type: str) -> List[str]:
        """Get enhanced recovery suggestions based on error category and model type"""
        suggestions = []
        
        if error_category == "vram_exhaustion":
            suggestions.extend([
                "Try reducing the resolution (e.g., from 1920x1080 to 1280x720)",
                "Reduce the number of inference steps",
                "Enable model offloading to CPU",
                "Close other GPU-intensive applications",
                "Use a smaller model variant if available"
            ])
        elif error_category == "model_loading_error":
            suggestions.extend([
                "Check if the model files are complete and not corrupted",
                "Try re-downloading the model",
                "Verify available disk space",
                "Check model file permissions"
            ])
        elif error_category == "download_error":
            suggestions.extend([
                "Check your internet connection",
                "Verify firewall settings allow model downloads",
                "Try downloading during off-peak hours",
                "Check available disk space"
            ])
        elif error_category == "model_corruption":
            suggestions.extend([
                "Re-download the model to fix corruption",
                "Check disk health and available space",
                "Verify model checksums if available",
                "Try using an alternative model"
            ])
        elif error_category == "timeout_error":
            suggestions.extend([
                "Try reducing the number of frames or steps",
                "Check system resources (CPU, memory, GPU)",
                "Increase timeout settings if configurable",
                "Try generating during lower system load"
            ])
        elif error_category == "permission_error":
            suggestions.extend([
                "Check file and directory permissions",
                "Run the application with appropriate privileges",
                "Verify model directory is writable",
                "Check antivirus software interference"
            ])
        else:
            suggestions.extend([
                "Check the error logs for more details",
                "Try restarting the generation service",
                "Verify your system configuration",
                "Contact support if the issue persists"
            ])
        
        # Add model-specific suggestions
        if model_type in ["t2v", "i2v", "ti2v"]:
            suggestions.append(f"Try using a different {model_type.upper()} model variant")
        
        return suggestions[:5]  # Limit to top 5 suggestions

    def _determine_failure_type_from_context(self, error_context: Dict[str, Any], model_type: str) -> 'FailureType':
        """Determine failure type from error context for recovery system"""
        error = error_context.get("error", "").lower()
        
        if "health_check_failed" in error or "corruption" in error:
            return FailureType.MODEL_LOADING_FAILURE
        elif "download" in error or "network" in error:
            return FailureType.NETWORK_ERROR
        elif "vram" in error or "memory" in error:
            return FailureType.VRAM_EXHAUSTION
        else:
            return FailureType.SYSTEM_RESOURCE_ERROR


# Global generation service instance
generation_service = GenerationService()

async def get_generation_service() -> GenerationService:
    """Dependency to get enhanced generation service instance"""
    if not generation_service.processing_thread or not generation_service.processing_thread.is_alive():
        await generation_service.initialize()
    return generation_service

async def get_enhanced_generation_service() -> GenerationService:
    """Get generation service with guaranteed real AI integration"""
    service = await get_generation_service()
    
    # Ensure real AI components are initialized
    if not service.model_integration_bridge or not service.real_generation_pipeline:
        logger.warning("Real AI components not fully initialized, attempting re-initialization")
        await service._initialize_real_ai_components()
    
    return service