"""
Enhanced Generation Pipeline for Wan2.2 Video Generation
Integrates validation, orchestration, error handling, and retry mechanisms
"""

import logging
import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from enum import Enum
from datetime import datetime
from pathlib import Path
import json
import traceback

# Import existing components
from input_validation import ValidationResult, PromptValidator, ImageValidator, ConfigValidator
from generation_orchestrator import (
    GenerationOrchestrator, GenerationRequest, PreflightResult, 
    GenerationMode, ResourceEstimate, GenerationResult
)
from generation_mode_router import get_generation_mode_router
from infrastructure.hardware.error_handler import (
    ErrorCategory, ErrorSeverity, UserFriendlyError, RecoveryAction,
    GenerationErrorHandler
)
from resource_manager import get_resource_manager, ResourceStatus, OptimizationLevel
from core.services.utils import get_model_manager, GenerationTask, TaskStatus

logger = logging.getLogger(__name__)

class GenerationStage(Enum):
    """Stages of the generation pipeline"""
    VALIDATION = "validation"
    PREFLIGHT = "preflight"
    PREPARATION = "preparation"
    GENERATION = "generation"
    POST_PROCESSING = "post_processing"
    COMPLETION = "completion"

class RetryStrategy(Enum):
    """Retry strategies for failed generations"""
    NONE = "none"
    BASIC = "basic"
    OPTIMIZED = "optimized"
    AGGRESSIVE = "aggressive"

@dataclass
class GenerationContext:
    """Context information for a generation operation"""
    request: GenerationRequest
    task_id: str
    stage: GenerationStage = GenerationStage.VALIDATION
    attempt: int = 1
    max_attempts: int = 3
    retry_strategy: RetryStrategy = RetryStrategy.OPTIMIZED
    start_time: datetime = field(default_factory=datetime.now)
    stage_times: Dict[str, float] = field(default_factory=dict)
    errors: List[Exception] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PipelineResult:
    """Result of the complete generation pipeline"""
    success: bool
    output_path: Optional[str] = None
    context: Optional[GenerationContext] = None
    error: Optional[UserFriendlyError] = None
    generation_time: Optional[float] = None
    resource_usage: Optional[Dict[str, Any]] = None
    retry_count: int = 0
    final_parameters: Optional[GenerationRequest] = None

class EnhancedGenerationPipeline:
    """Enhanced generation pipeline with validation, orchestration, and retry mechanisms"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize core components
        self.orchestrator = GenerationOrchestrator(config)
        self.error_handler = GenerationErrorHandler()
        self.resource_manager = get_resource_manager(config)
        self.model_manager = get_model_manager()
        
        # Pipeline configuration
        self.max_retry_attempts = config.get("generation", {}).get("max_retry_attempts", 3)
        self.enable_auto_optimization = config.get("generation", {}).get("enable_auto_optimization", True)
        self.enable_preflight_checks = config.get("generation", {}).get("enable_preflight_checks", True)
        
        # Progress callbacks
        self.progress_callbacks: List[Callable] = []
        self.status_callbacks: List[Callable] = []
        
        logger.info("Enhanced generation pipeline initialized")
    
    def add_progress_callback(self, callback: Callable[[str, float], None]):
        """Add a progress callback function"""
        self.progress_callbacks.append(callback)
    
    def add_status_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add a status callback function"""
        self.status_callbacks.append(callback)
    
    def _notify_progress(self, stage: str, progress: float):
        """Notify all progress callbacks"""
        for callback in self.progress_callbacks:
            try:
                callback(stage, progress)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
    
    def _notify_status(self, status: str, data: Dict[str, Any]):
        """Notify all status callbacks"""
        for callback in self.status_callbacks:
            try:
                callback(status, data)
            except Exception as e:
                logger.warning(f"Status callback failed: {e}")
    
    async def generate_video(self, request: GenerationRequest, task_id: Optional[str] = None) -> PipelineResult:
        """Main entry point for video generation with full pipeline"""
        if task_id is None:
            task_id = f"gen_{int(time.time())}"
        
        context = GenerationContext(
            request=request,
            task_id=task_id,
            max_attempts=self.max_retry_attempts
        )
        
        logger.info(f"Starting generation pipeline for task {task_id}")
        self._notify_status("started", {"task_id": task_id, "model_type": request.model_type})
        
        try:
            # Execute pipeline stages with retry logic
            result = await self._execute_pipeline_with_retry(context)
            
            if result.success:
                logger.info(f"Generation completed successfully: {result.output_path}")
                self._notify_status("completed", {
                    "task_id": task_id,
                    "output_path": result.output_path,
                    "generation_time": result.generation_time
                })
            else:
                logger.error(f"Generation failed after {result.retry_count} attempts")
                self._notify_status("failed", {
                    "task_id": task_id,
                    "error": result.error.message if result.error else "Unknown error"
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            error = self.error_handler.handle_generation_error(e, context.request)
            
            return PipelineResult(
                success=False,
                context=context,
                error=error,
                retry_count=context.attempt - 1
            )
    
    async def _execute_pipeline_with_retry(self, context: GenerationContext) -> PipelineResult:
        """Execute the pipeline with retry logic"""
        last_error = None
        
        for attempt in range(1, context.max_attempts + 1):
            context.attempt = attempt
            logger.info(f"Pipeline attempt {attempt}/{context.max_attempts}")
            
            try:
                result = await self._execute_single_pipeline(context)
                if result.success:
                    result.retry_count = attempt - 1
                    return result
                
                # Store error for potential retry
                last_error = result.error
                
                # Determine if we should retry
                if attempt < context.max_attempts and self._should_retry(result.error, context):
                    logger.info(f"Retrying generation (attempt {attempt + 1})")
                    
                    # Apply retry optimizations
                    context.request = self._apply_retry_optimizations(context.request, result.error, attempt)
                    
                    # Wait before retry
                    await asyncio.sleep(min(2 ** attempt, 10))  # Exponential backoff
                    continue
                else:
                    break
                    
            except Exception as e:
                logger.error(f"Pipeline attempt {attempt} failed with exception: {e}")
                last_error = self.error_handler.handle_generation_error(e, context.request)
                
                if attempt < context.max_attempts:
                    await asyncio.sleep(min(2 ** attempt, 10))
                    continue
                else:
                    break
        
        # All attempts failed
        return PipelineResult(
            success=False,
            context=context,
            error=last_error,
            retry_count=context.max_attempts
        )
    
    async def _execute_single_pipeline(self, context: GenerationContext) -> PipelineResult:
        """Execute a single pipeline attempt"""
        start_time = time.time()
        
        try:
            # Stage 1: Input Validation
            context.stage = GenerationStage.VALIDATION
            self._notify_progress("validation", 10)
            
            validation_result = await self._validate_inputs(context)
            if not validation_result.success:
                return validation_result
            
            # Stage 2: Pre-flight Checks
            if self.enable_preflight_checks:
                context.stage = GenerationStage.PREFLIGHT
                self._notify_progress("preflight", 20)
                
                preflight_result = await self._run_preflight_checks(context)
                if not preflight_result.success:
                    return preflight_result
            
            # Stage 3: Preparation
            context.stage = GenerationStage.PREPARATION
            self._notify_progress("preparation", 30)
            
            preparation_result = await self._prepare_generation(context)
            if not preparation_result.success:
                return preparation_result
            
            # Stage 4: Generation
            context.stage = GenerationStage.GENERATION
            self._notify_progress("generation", 40)
            
            generation_result = await self._execute_generation(context)
            if not generation_result.success:
                return generation_result
            
            # Stage 5: Post-processing
            context.stage = GenerationStage.POST_PROCESSING
            self._notify_progress("post_processing", 90)
            
            post_result = await self._post_process_output(context, generation_result.output_path)
            if not post_result.success:
                return post_result
            
            # Stage 6: Completion
            context.stage = GenerationStage.COMPLETION
            self._notify_progress("completion", 100)
            
            total_time = time.time() - start_time
            
            return PipelineResult(
                success=True,
                output_path=generation_result.output_path,
                context=context,
                generation_time=total_time,
                final_parameters=context.request
            )
            
        except Exception as e:
            logger.error(f"Pipeline stage {context.stage.value} failed: {e}")
            error = self.error_handler.handle_generation_error(e, context.request)
            
            return PipelineResult(
                success=False,
                context=context,
                error=error
            )
    
    async def _validate_inputs(self, context: GenerationContext) -> PipelineResult:
        """Validate all inputs using the validation framework"""
        try:
            # Validate prompt
            prompt_validator = PromptValidator(self.config)
            prompt_result = prompt_validator.validate(context.request.prompt)
            
            if not prompt_result.is_valid:
                error_msg = "; ".join([issue.message for issue in prompt_result.get_errors()])
                error = UserFriendlyError(
                    category=ErrorCategory.INPUT_VALIDATION,
                    severity=ErrorSeverity.HIGH,
                    title="Invalid Prompt",
                    message=f"Prompt validation failed: {error_msg}",
                    recovery_suggestions=[
                        "Check prompt length and content",
                        "Remove special characters or problematic content",
                        "Try a simpler prompt"
                    ],
                    recovery_actions=[]
                )
                return PipelineResult(success=False, error=error, context=context)
            
            # Validate image if provided
            if context.request.image is not None:
                image_validator = ImageValidator(self.config)
                image_result = image_validator.validate(context.request.image)
                
                if not image_result.is_valid:
                    error_msg = "; ".join([issue.message for issue in image_result.get_errors()])
                    error = UserFriendlyError(
                        category=ErrorCategory.INPUT_VALIDATION,
                        severity=ErrorSeverity.HIGH,
                        title="Invalid Image",
                        message=f"Image validation failed: {error_msg}",
                        recovery_suggestions=[
                            "Check image format and size",
                            "Use a supported image format (PNG, JPG, WEBP)",
                            "Resize image to supported dimensions"
                        ],
                        recovery_actions=[]
                    )
                    return PipelineResult(success=False, error=error, context=context)
            
            # Validate configuration
            config_validator = ConfigValidator(self.config)
            config_data = {
                "resolution": context.request.resolution,
                "steps": context.request.steps,
                "guidance_scale": context.request.guidance_scale,
                "model_type": context.request.model_type
            }
            config_result = config_validator.validate(config_data)
            
            if not config_result.is_valid:
                error_msg = "; ".join([issue.message for issue in config_result.get_errors()])
                error = UserFriendlyError(
                    category=ErrorCategory.INPUT_VALIDATION,
                    severity=ErrorSeverity.HIGH,
                    title="Invalid Configuration",
                    message=f"Configuration validation failed: {error_msg}",
                    recovery_suggestions=[
                        "Check generation parameters",
                        "Use supported resolution and step values",
                        "Verify model type compatibility"
                    ],
                    recovery_actions=[]
                )
                return PipelineResult(success=False, error=error, context=context)
            
            logger.info("Input validation passed")
            return PipelineResult(success=True, context=context)
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            error = self.error_handler.handle_validation_error(e)
            return PipelineResult(success=False, error=error, context=context)
    
    async def _run_preflight_checks(self, context: GenerationContext) -> PipelineResult:
        """Run pre-flight checks using the orchestrator"""
        try:
            preflight_result = self.orchestrator.run_preflight_checks(context.request)
            
            if not preflight_result.can_proceed:
                issues = "; ".join(preflight_result.blocking_issues)
                error = UserFriendlyError(
                    category=ErrorCategory.SYSTEM_RESOURCE,
                    severity=ErrorSeverity.HIGH,
                    title="Pre-flight Check Failed",
                    message=f"System not ready for generation: {issues}",
                    recovery_suggestions=preflight_result.optimization_recommendations,
                    recovery_actions=[]
                )
                return PipelineResult(success=False, error=error, context=context)
            
            # Store preflight information in context
            context.metadata["preflight_result"] = preflight_result
            
            # Apply optimization recommendations if enabled
            if self.enable_auto_optimization and preflight_result.optimization_recommendations:
                logger.info(f"Applying {len(preflight_result.optimization_recommendations)} optimizations")
                context.request = self._apply_optimizations(context.request, preflight_result)
            
            logger.info("Pre-flight checks passed")
            return PipelineResult(success=True, context=context)
            
        except Exception as e:
            logger.error(f"Pre-flight checks failed: {e}")
            error = self.error_handler.handle_system_error(e)
            return PipelineResult(success=False, error=error, context=context)
    
    async def _prepare_generation(self, context: GenerationContext) -> PipelineResult:
        """Prepare the system for generation"""
        try:
            # Prepare generation environment
            success, message = self.orchestrator.prepare_generation(context.request)
            
            if not success:
                error = UserFriendlyError(
                    category=ErrorCategory.SYSTEM_RESOURCE,
                    severity=ErrorSeverity.HIGH,
                    title="Generation Preparation Failed",
                    message=message,
                    recovery_suggestions=[
                        "Free up system resources",
                        "Close other applications",
                        "Reduce generation parameters"
                    ],
                    recovery_actions=[]
                )
                return PipelineResult(success=False, error=error, context=context)
            
            logger.info("Generation preparation completed")
            return PipelineResult(success=True, context=context)
            
        except Exception as e:
            logger.error(f"Generation preparation failed: {e}")
            error = self.error_handler.handle_system_error(e)
            return PipelineResult(success=False, error=error, context=context)
    
    async def _execute_generation(self, context: GenerationContext) -> PipelineResult:
        """Execute the actual video generation"""
        try:
            # Use the generation mode router to validate and optimize the request
            router = get_generation_mode_router(self.config)
            if router:
                routing_result = router.route_request(context.request)
                if not routing_result.is_valid:
                    error_msg = "; ".join(routing_result.validation_issues)
                    error = UserFriendlyError(
                        category=ErrorCategory.INPUT_VALIDATION,
                        severity=ErrorSeverity.HIGH,
                        title="Mode Routing Failed",
                        message=f"Generation mode validation failed: {error_msg}",
                        recovery_suggestions=[
                            "Check input requirements for the selected mode",
                            "Verify prompt and image inputs are valid",
                            "Try a different generation mode"
                        ],
                        recovery_actions=[]
                    )
                    return PipelineResult(success=False, error=error, context=context)
                
                # Use optimized request from router
                context.request = routing_result.optimized_request
                logger.info(f"Using optimized request for {routing_result.mode.value}")
            
            # Create progress callback for generation
            def generation_progress(step: int, total_steps: int):
                progress = 40 + (step / total_steps) * 50  # 40-90% of total progress
                self._notify_progress("generation", progress)
            
            # Execute generation using the legacy function as fallback
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._execute_legacy_generation(context, generation_progress)
            )
            
            if result.get("success", False):
                logger.info(f"Generation completed: {result.get('output_path')}")
                return PipelineResult(
                    success=True,
                    output_path=result.get("output_path"),
                    context=context
                )
            else:
                error_msg = result.get("error", "Unknown generation error")
                recovery_suggestions = result.get("recovery_suggestions", [
                    "Try reducing generation steps",
                    "Use a lower resolution", 
                    "Simplify the prompt",
                    "Check available VRAM"
                ])
                
                error = UserFriendlyError(
                    category=ErrorCategory.GENERATION_PIPELINE,
                    severity=ErrorSeverity.HIGH,
                    title="Generation Failed",
                    message=error_msg,
                    recovery_suggestions=recovery_suggestions,
                    recovery_actions=[]
                )
                return PipelineResult(success=False, error=error, context=context)
            
        except Exception as e:
            logger.error(f"Generation execution failed: {e}")
            error = self.error_handler.handle_generation_error(e, context.request)
            return PipelineResult(success=False, error=error, context=context)
    
    def _execute_legacy_generation(self, context: GenerationContext, progress_callback) -> Dict[str, Any]:
        """Execute generation using the legacy generation engine"""
        try:
            # Import the generation engine
            from core.services.utils import get_generation_engine
            
            engine = get_generation_engine()
            
            # Convert resolution format if needed
            resolution = context.request.resolution
            if resolution == "720p":
                resolution = "1280x720"
            elif resolution == "1080p":
                resolution = "1920x1080"
            elif resolution == "480p":
                resolution = "854x480"
            
            # Execute generation based on model type
            if context.request.model_type in ["t2v-A14B", "text-to-video"]:
                result = engine.generate_t2v(
                    prompt=context.request.prompt,
                    resolution=resolution,
                    num_inference_steps=context.request.steps,
                    guidance_scale=context.request.guidance_scale,
                    seed=context.request.seed,
                    fps=context.request.fps,
                    duration=context.request.duration,
                    progress_callback=progress_callback
                )
            elif context.request.model_type in ["i2v-A14B", "image-to-video"]:
                result = engine.generate_i2v(
                    prompt=context.request.prompt or "",
                    image=context.request.image,
                    resolution=resolution,
                    num_inference_steps=context.request.steps,
                    guidance_scale=context.request.guidance_scale,
                    strength=context.request.strength,
                    seed=context.request.seed,
                    fps=context.request.fps,
                    duration=context.request.duration,
                    progress_callback=progress_callback
                )
            elif context.request.model_type in ["ti2v-5B", "text-image-to-video"]:
                result = engine.generate_ti2v(
                    prompt=context.request.prompt,
                    image=context.request.image,
                    resolution=resolution,
                    num_inference_steps=context.request.steps,
                    guidance_scale=context.request.guidance_scale,
                    strength=context.request.strength,
                    seed=context.request.seed,
                    fps=context.request.fps,
                    duration=context.request.duration,
                    progress_callback=progress_callback
                )
            else:
                return {
                    "success": False,
                    "error": f"Unsupported model type: {context.request.model_type}",
                    "recovery_suggestions": [
                        "Use a supported model type (t2v-A14B, i2v-A14B, ti2v-5B)",
                        "Check model type spelling and format"
                    ]
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Legacy generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "recovery_suggestions": [
                    "Check model availability",
                    "Verify system resources",
                    "Try reducing generation parameters"
                ]
            }
    
    async def _post_process_output(self, context: GenerationContext, output_path: str) -> PipelineResult:
        """Post-process the generated output"""
        try:
            # Verify output file exists and is valid
            if not Path(output_path).exists():
                error = UserFriendlyError(
                    category=ErrorCategory.FILE_SYSTEM,
                    severity=ErrorSeverity.HIGH,
                    title="Output File Missing",
                    message=f"Generated video file not found: {output_path}",
                    recovery_suggestions=[
                        "Check disk space",
                        "Verify output directory permissions",
                        "Try generating again"
                    ],
                    recovery_actions=[]
                )
                return PipelineResult(success=False, error=error, context=context)
            
            # Get file size for validation
            file_size = Path(output_path).stat().st_size
            if file_size < 1024:  # Less than 1KB is likely invalid
                error = UserFriendlyError(
                    category=ErrorCategory.GENERATION_PIPELINE,
                    severity=ErrorSeverity.HIGH,
                    title="Invalid Output",
                    message=f"Generated video file is too small ({file_size} bytes)",
                    recovery_suggestions=[
                        "Try generating with different parameters",
                        "Check for generation errors",
                        "Increase generation steps"
                    ],
                    recovery_actions=[]
                )
                return PipelineResult(success=False, error=error, context=context)
            
            # Store metadata
            context.metadata["output_file_size"] = file_size
            context.metadata["output_path"] = output_path
            
            logger.info(f"Post-processing completed: {output_path} ({file_size} bytes)")
            return PipelineResult(success=True, output_path=output_path, context=context)
            
        except Exception as e:
            logger.error(f"Post-processing failed: {e}")
            error = self.error_handler.handle_system_error(e)
            return PipelineResult(success=False, error=error, context=context)
    
    def _should_retry(self, error: Optional[UserFriendlyError], context: GenerationContext) -> bool:
        """Determine if generation should be retried based on error type"""
        if not error:
            return False
        
        # Don't retry validation errors
        if error.category == ErrorCategory.INPUT_VALIDATION:
            return False
        
        # Retry VRAM and system resource errors
        if error.category in [ErrorCategory.VRAM_MEMORY, ErrorCategory.SYSTEM_RESOURCE]:
            return True
        
        # Retry generation pipeline errors with optimization
        if error.category == ErrorCategory.GENERATION_PIPELINE:
            return True
        
        # Don't retry file system or configuration errors
        if error.category in [ErrorCategory.FILE_SYSTEM, ErrorCategory.CONFIGURATION]:
            return False
        
        return True
    
    def _apply_retry_optimizations(self, request: GenerationRequest, 
                                 error: Optional[UserFriendlyError], 
                                 attempt: int) -> GenerationRequest:
        """Apply optimizations for retry attempts"""
        optimized = GenerationRequest(
            model_type=request.model_type,
            prompt=request.prompt,
            image=request.image,
            resolution=request.resolution,
            steps=request.steps,
            guidance_scale=request.guidance_scale,
            strength=request.strength,
            seed=request.seed,
            fps=request.fps,
            duration=request.duration,
            lora_config=request.lora_config.copy(),
            optimization_settings=request.optimization_settings.copy()
        )
        
        if error and error.category == ErrorCategory.VRAM_MEMORY:
            # Reduce parameters for VRAM issues
            if attempt == 2:
                optimized.steps = max(20, optimized.steps - 10)
                if optimized.resolution == "1080p":
                    optimized.resolution = "720p"
            elif attempt >= 3:
                optimized.steps = max(15, optimized.steps - 20)
                optimized.resolution = "720p"
                optimized.lora_config = {}  # Remove LoRAs
        
        elif error and error.category == ErrorCategory.GENERATION_PIPELINE:
            # Adjust generation parameters
            if attempt == 2:
                optimized.guidance_scale = max(1.0, optimized.guidance_scale - 1.0)
            elif attempt >= 3:
                optimized.steps = max(20, optimized.steps - 15)
                optimized.guidance_scale = 7.5  # Reset to default
        
        return optimized
    
    def _apply_optimizations(self, request: GenerationRequest, 
                           preflight_result: PreflightResult) -> GenerationRequest:
        """Apply optimization recommendations from preflight checks"""
        # Use resource manager to optimize parameters
        return self.orchestrator.resource_manager.optimize_for_available_resources(request)
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            "pipeline_ready": True,
            "resource_status": self.orchestrator.get_generation_status(),
            "config": {
                "max_retry_attempts": self.max_retry_attempts,
                "enable_auto_optimization": self.enable_auto_optimization,
                "enable_preflight_checks": self.enable_preflight_checks
            }
        }


# Global pipeline instance
_pipeline_instance = None

def get_enhanced_pipeline(config: Dict[str, Any]) -> EnhancedGenerationPipeline:
    """Get the global enhanced pipeline instance"""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = EnhancedGenerationPipeline(config)
    return _pipeline_instance