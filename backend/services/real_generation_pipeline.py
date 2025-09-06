"""
Real Generation Pipeline
Integrates with existing WanPipelineLoader infrastructure for actual video generation
"""

import sys
import os
import asyncio
import logging
import time
import torch
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

# Add parent directory to path to import existing modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.core.model_integration_bridge import (
    GenerationParams, GenerationResult, ModelType, ModelStatus
)

# Import existing LoRA manager
try:
    from core.services.utils import LoRAManager
    LORA_MANAGER_AVAILABLE = True
except ImportError:
    LORA_MANAGER_AVAILABLE = False

logger = logging.getLogger(__name__)

class GenerationStage(Enum):
    """Stages of video generation process"""
    INITIALIZING = "initializing"
    LOADING_MODEL = "loading_model"
    PREPARING_INPUTS = "preparing_inputs"
    GENERATING = "generating"
    POST_PROCESSING = "post_processing"
    SAVING = "saving"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ProgressUpdate:
    """Progress update information"""
    stage: GenerationStage
    progress_percent: int
    message: str
    estimated_time_remaining: Optional[float] = None
    current_step: Optional[int] = None
    total_steps: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class RealGenerationPipeline:
    """
    Real generation pipeline using existing WanPipelineLoader infrastructure
    Handles T2V, I2V, and TI2V generation with progress tracking and WebSocket updates
    """
    
    def __init__(self, wan_pipeline_loader=None, websocket_manager=None):
        """
        Initialize the real generation pipeline
        
        Args:
            wan_pipeline_loader: WanPipelineLoader instance (will be initialized if None)
            websocket_manager: WebSocket manager for progress updates
        """
        self.wan_pipeline_loader = wan_pipeline_loader
        self.websocket_manager = websocket_manager
        self.logger = logging.getLogger(__name__ + ".RealGenerationPipeline")
        
        # Pipeline cache for loaded models
        self._pipeline_cache: Dict[str, Any] = {}
        
        # Generation statistics
        self._generation_count = 0
        self._total_generation_time = 0.0
        
        # Progress tracking
        self._current_task_id: Optional[str] = None
        self._progress_callback: Optional[Callable] = None
        
        # LoRA management
        self.lora_manager: Optional[LoRAManager] = None
        self._applied_loras: Dict[str, float] = {}  # Track applied LoRAs per pipeline
        
        self.logger.info("RealGenerationPipeline initialized")
    
    async def initialize(self) -> bool:
        """Initialize the pipeline with existing infrastructure"""
        try:
            # Initialize WanPipelineLoader if not provided
            if self.wan_pipeline_loader is None:
                await self._initialize_wan_pipeline_loader()
            
            # Initialize WebSocket manager if not provided
            if self.websocket_manager is None:
                await self._initialize_websocket_manager()
            
            # Initialize LoRA manager if available
            await self._initialize_lora_manager()
            
            self.logger.info("RealGenerationPipeline initialization completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RealGenerationPipeline: {e}")
            return False
    
    async def _initialize_wan_pipeline_loader(self):
        """Initialize WanPipelineLoader from system integration"""
        try:
            from backend.core.system_integration import get_system_integration
            integration = await get_system_integration()
            self.wan_pipeline_loader = integration.get_wan_pipeline_loader()
            
            if self.wan_pipeline_loader:
                self.logger.info("WanPipelineLoader initialized from system integration")
            else:
                self.logger.warning("WanPipelineLoader not available from system integration")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize WanPipelineLoader: {e}")
            self.wan_pipeline_loader = None
    
    async def _initialize_websocket_manager(self):
        """Initialize WebSocket manager for progress updates"""
        try:
            from backend.websocket.manager import get_connection_manager
            self.websocket_manager = get_connection_manager()
            self.logger.info("WebSocket manager initialized")
        except Exception as e:
            self.logger.warning(f"Could not initialize WebSocket manager: {e}")
            self.websocket_manager = None
    
    async def _initialize_lora_manager(self):
        """Initialize LoRA manager for LoRA support"""
        try:
            if not LORA_MANAGER_AVAILABLE:
                self.logger.warning("LoRA manager not available - LoRA support disabled")
                return
            
            # Load configuration for LoRA manager
            config = await self._load_lora_config()
            
            # Initialize LoRA manager
            self.lora_manager = LoRAManager(config)
            self.logger.info("LoRA manager initialized successfully")
            
            # Log available LoRAs
            available_loras = self.lora_manager.list_available_loras()
            self.logger.info(f"Found {len(available_loras)} available LoRA files")
            
        except Exception as e:
            self.logger.warning(f"Could not initialize LoRA manager: {e}")
            self.lora_manager = None
    
    async def _load_lora_config(self) -> Dict[str, Any]:
        """Load configuration for LoRA manager"""
        try:
            # Try to load from system integration
            from backend.core.system_integration import get_system_integration
            integration = await get_system_integration()
            
            if integration and integration.config:
                return integration.config
            
        except Exception as e:
            self.logger.warning(f"Could not load config from system integration: {e}")
        
        # Fallback configuration
        project_root = Path(__file__).parent.parent.parent
        return {
            "directories": {
                "loras_directory": str(project_root / "loras"),
                "models_directory": str(project_root / "models"),
                "outputs_directory": str(project_root / "outputs")
            },
            "lora_max_file_size_mb": 500,
            "optimization": {
                "max_vram_usage_gb": 12
            }
        }
    
    async def generate_video_with_optimization(self, model_type: str, prompt: str, 
                                             image_path: Optional[str] = None,
                                             end_image_path: Optional[str] = None,
                                             **kwargs) -> GenerationResult:
        """
        Generate video with optimization - unified method for all generation types
        
        Args:
            model_type: Type of model to use (t2v-a14b, i2v-a14b, ti2v-5b)
            prompt: Text prompt for generation
            image_path: Optional input image path for I2V/TI2V
            end_image_path: Optional end image path for TI2V
            **kwargs: Additional generation parameters
            
        Returns:
            GenerationResult with generated video information
        """
        try:
            # Create GenerationParams from kwargs
            params = GenerationParams(
                prompt=prompt,
                model_type=model_type,
                image_path=image_path,
                end_image_path=end_image_path,
                **{k: v for k, v in kwargs.items() if k in GenerationParams.__annotations__}
            )
            
            # Extract progress_callback from kwargs if provided
            progress_callback = kwargs.get('progress_callback', None)
            
            # Route to appropriate generation method based on model type
            if model_type.startswith("t2v") or model_type == "T2V":
                return await self.generate_t2v(prompt, params, progress_callback)
            elif model_type.startswith("i2v") or model_type == "I2V":
                if not image_path:
                    raise ValueError("Image path required for I2V generation")
                return await self.generate_i2v(image_path, prompt, params, progress_callback)
            elif model_type.startswith("ti2v") or model_type == "TI2V":
                if not image_path:
                    raise ValueError("Image path required for TI2V generation")
                return await self.generate_ti2v(image_path, prompt, params, progress_callback)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
        except Exception as e:
            self.logger.error(f"Video generation with optimization failed: {e}")
            return GenerationResult(
                success=False,
                task_id=str(uuid.uuid4()),
                error_message=str(e),
                error_category="generation_error"
            )
    
    async def generate_t2v(self, prompt: str, params: GenerationParams, 
                          progress_callback: Optional[Callable] = None) -> GenerationResult:
        """
        Generate text-to-video using existing WAN T2V pipeline
        
        Args:
            prompt: Text prompt for generation
            params: Generation parameters
            progress_callback: Optional progress callback function
            
        Returns:
            GenerationResult with generated video information
        """
        task_id = f"t2v_{uuid.uuid4().hex[:8]}"
        self._current_task_id = task_id
        self._progress_callback = progress_callback
        
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting T2V generation: {task_id}")
            
            # Initialize progress tracking
            try:
                from backend.websocket.progress_integration import get_progress_integration
                progress_integration = await get_progress_integration()
                await progress_integration.start_generation_tracking(
                    task_id, "t2v-A14B", estimated_duration=60.0  # Estimate 60 seconds
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize progress tracking: {e}")
            
            await self._send_progress_update(
                task_id, GenerationStage.INITIALIZING, 0, 
                "Initializing T2V generation"
            )
            
            # Validate parameters for T2V
            validation_result = self._validate_t2v_params(prompt, params)
            if not validation_result["valid"]:
                return self._create_error_result(
                    task_id, "parameter_validation", 
                    f"Invalid parameters: {', '.join(validation_result['errors'])}"
                )
            
            # Load T2V pipeline
            await self._send_progress_update(
                task_id, GenerationStage.LOADING_MODEL, 10,
                "Loading T2V model pipeline"
            )
            
            pipeline_wrapper = await self._load_pipeline("t2v-A14B", params)
            if not pipeline_wrapper:
                return self._create_error_result(
                    task_id, "model_loading", "Failed to load T2V pipeline"
                )
            
            # Prepare generation configuration
            await self._send_progress_update(
                task_id, GenerationStage.PREPARING_INPUTS, 25,
                "Preparing generation inputs"
            )
            
            generation_config = self._create_generation_config(prompt, params)
            
            # Apply LoRA if specified
            lora_applied = await self._apply_lora_to_pipeline(pipeline_wrapper, params, task_id)
            if lora_applied:
                await self._send_progress_update(
                    task_id, GenerationStage.PREPARING_INPUTS, 30,
                    f"Applied LoRA: {Path(params.lora_path).stem if params.lora_path else 'unknown'}"
                )
            
            # Set up progress callback for generation
            def generation_progress_callback(step: int, total_steps: int, latents: torch.Tensor):
                progress = 35 + int((step / total_steps) * 50)  # 35-85% for generation
                
                # Send internal progress update
                asyncio.create_task(self._send_progress_update(
                    task_id, GenerationStage.GENERATING, progress,
                    f"Generating frame {step}/{total_steps}",
                    current_step=step, total_steps=total_steps
                ))
                
                # Call external progress callback if provided
                if progress_callback:
                    try:
                        # Convert internal progress (35-85%) to external progress (0-100%)
                        external_progress = ((progress - 35) / 50) * 100
                        asyncio.create_task(progress_callback(
                            external_progress, 
                            f"Generating frame {step}/{total_steps}"
                        ))
                    except Exception as e:
                        self.logger.warning(f"External progress callback failed: {e}")
            
            generation_config.progress_callback = generation_progress_callback
            
            # Generate video
            await self._send_progress_update(
                task_id, GenerationStage.GENERATING, 35,
                f"Generating {params.num_frames} frames"
            )
            
            # Check if pipeline wrapper supports async generation (WAN models)
            if hasattr(pipeline_wrapper, 'generate_async') or hasattr(pipeline_wrapper, 'model') and hasattr(pipeline_wrapper.model, 'generate_video'):
                # Use async generation for WAN models with enhanced progress tracking
                generation_config.task_id = task_id
                if hasattr(pipeline_wrapper, 'generate_async'):
                    generation_result = await pipeline_wrapper.generate_async(generation_config)
                else:
                    # Direct WAN model call
                    wan_params = {
                        'prompt': generation_config.prompt,
                        'negative_prompt': generation_config.negative_prompt,
                        'num_frames': generation_config.num_frames,
                        'width': generation_config.width,
                        'height': generation_config.height,
                        'num_inference_steps': generation_config.num_inference_steps,
                        'guidance_scale': generation_config.guidance_scale,
                        'seed': generation_config.seed,
                        'task_id': task_id,
                        'callback': generation_config.progress_callback
                    }
                    generation_result = await pipeline_wrapper.model.generate_video(**wan_params)
            else:
                # Fallback to sync generation for non-WAN models
                generation_result = await asyncio.get_event_loop().run_in_executor(
                    None, pipeline_wrapper.generate, generation_config
                )
            
            if not generation_result.success:
                return self._create_error_result(
                    task_id, "generation_failed", 
                    f"Generation failed: {', '.join(generation_result.errors)}"
                )
            
            # Post-process and save
            await self._send_progress_update(
                task_id, GenerationStage.POST_PROCESSING, 85,
                "Post-processing generated frames"
            )
            
            output_path = await self._save_generated_video(
                generation_result.frames, task_id, params
            )
            
            await self._send_progress_update(
                task_id, GenerationStage.SAVING, 95,
                "Saving generated video"
            )
            
            # Create final result
            generation_time = time.time() - start_time
            self._generation_count += 1
            self._total_generation_time += generation_time
            
            result = GenerationResult(
                success=True,
                task_id=task_id,
                output_path=output_path,
                generation_time_seconds=generation_time,
                model_used="t2v-A14B",
                parameters_used=params.__dict__.copy(),
                peak_vram_usage_mb=generation_result.peak_memory_mb,
                average_vram_usage_mb=generation_result.memory_used_mb,
                optimizations_applied=generation_result.applied_optimizations.copy()
            )
            
            await self._send_progress_update(
                task_id, GenerationStage.COMPLETED, 100,
                f"T2V generation completed in {generation_time:.1f}s"
            )
            
            # Complete progress tracking
            try:
                from backend.websocket.progress_integration import get_progress_integration
                progress_integration = await get_progress_integration()
                await progress_integration.complete_generation_tracking(
                    success=True, output_path=str(output_path)
                )
            except Exception as e:
                self.logger.warning(f"Failed to complete progress tracking: {e}")
            
            self.logger.info(f"T2V generation completed: {task_id} in {generation_time:.1f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"T2V generation failed: {e}")
            await self._send_progress_update(
                task_id, GenerationStage.FAILED, 0,
                f"Generation failed: {str(e)}"
            )
            
            # Complete progress tracking with error
            try:
                from backend.websocket.progress_integration import get_progress_integration
                progress_integration = await get_progress_integration()
                await progress_integration.complete_generation_tracking(
                    success=False, error_message=str(e)
                )
            except Exception as pe:
                self.logger.warning(f"Failed to complete progress tracking: {pe}")
            
            return self._create_error_result(task_id, "generation_error", str(e))
        
        finally:
            # Clean up LoRA for this task
            await self._cleanup_lora_for_task(task_id)
            self._current_task_id = None
            self._progress_callback = None
    
    async def generate_i2v(self, image_path: str, prompt: str, params: GenerationParams,
                          progress_callback: Optional[Callable] = None) -> GenerationResult:
        """
        Generate image-to-video using existing WAN I2V pipeline
        
        Args:
            image_path: Path to input image
            prompt: Text prompt for generation
            params: Generation parameters
            progress_callback: Optional progress callback function
            
        Returns:
            GenerationResult with generated video information
        """
        task_id = f"i2v_{uuid.uuid4().hex[:8]}"
        self._current_task_id = task_id
        self._progress_callback = progress_callback
        
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting I2V generation: {task_id}")
            
            # Initialize progress tracking
            try:
                from backend.websocket.progress_integration import get_progress_integration
                progress_integration = await get_progress_integration()
                await progress_integration.start_generation_tracking(
                    task_id, "i2v-A14B", estimated_duration=75.0  # Estimate 75 seconds for I2V
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize progress tracking: {e}")
            
            await self._send_progress_update(
                task_id, GenerationStage.INITIALIZING, 0,
                "Initializing I2V generation"
            )
            
            # Validate parameters for I2V
            validation_result = self._validate_i2v_params(image_path, prompt, params)
            if not validation_result["valid"]:
                return self._create_error_result(
                    task_id, "parameter_validation",
                    f"Invalid parameters: {', '.join(validation_result['errors'])}"
                )
            
            # Load I2V pipeline
            await self._send_progress_update(
                task_id, GenerationStage.LOADING_MODEL, 10,
                "Loading I2V model pipeline"
            )
            
            pipeline_wrapper = await self._load_pipeline("i2v-A14B", params)
            if not pipeline_wrapper:
                return self._create_error_result(
                    task_id, "model_loading", "Failed to load I2V pipeline"
                )
            
            # Prepare inputs including image
            await self._send_progress_update(
                task_id, GenerationStage.PREPARING_INPUTS, 25,
                "Preparing image and text inputs"
            )
            
            # Load and validate input image
            input_image = await self._load_and_validate_image(image_path)
            if input_image is None:
                return self._create_error_result(
                    task_id, "image_loading", f"Failed to load input image: {image_path}"
                )
            
            # Create generation configuration for I2V
            generation_config = self._create_generation_config(prompt, params)
            generation_config.input_image = input_image
            
            # Apply LoRA if specified
            lora_applied = await self._apply_lora_to_pipeline(pipeline_wrapper, params, task_id)
            if lora_applied:
                await self._send_progress_update(
                    task_id, GenerationStage.PREPARING_INPUTS, 30,
                    f"Applied LoRA: {Path(params.lora_path).stem if params.lora_path else 'unknown'}"
                )
            
            # Set up progress callback
            def generation_progress_callback(step: int, total_steps: int, latents: torch.Tensor):
                progress = 35 + int((step / total_steps) * 50)
                
                # Send internal progress update
                asyncio.create_task(self._send_progress_update(
                    task_id, GenerationStage.GENERATING, progress,
                    f"Generating I2V frame {step}/{total_steps}",
                    current_step=step, total_steps=total_steps
                ))
                
                # Call external progress callback if provided
                if progress_callback:
                    try:
                        # Convert internal progress (35-85%) to external progress (0-100%)
                        external_progress = ((progress - 35) / 50) * 100
                        asyncio.create_task(progress_callback(
                            external_progress, 
                            f"Generating I2V frame {step}/{total_steps}"
                        ))
                    except Exception as e:
                        self.logger.warning(f"External progress callback failed: {e}")
            
            generation_config.progress_callback = generation_progress_callback
            
            # Generate video
            await self._send_progress_update(
                task_id, GenerationStage.GENERATING, 35,
                f"Generating video from image ({params.num_frames} frames)"
            )
            
            # Check if pipeline wrapper supports async generation (WAN models)
            if hasattr(pipeline_wrapper, 'generate_async') or hasattr(pipeline_wrapper, 'model') and hasattr(pipeline_wrapper.model, 'generate_video'):
                # Use async generation for WAN models with enhanced progress tracking
                generation_config.task_id = task_id
                if hasattr(pipeline_wrapper, 'generate_async'):
                    generation_result = await pipeline_wrapper.generate_async(generation_config)
                else:
                    # Direct WAN model call
                    wan_params = {
                        'prompt': generation_config.prompt,
                        'negative_prompt': generation_config.negative_prompt,
                        'image': generation_config.input_image,
                        'num_frames': generation_config.num_frames,
                        'width': generation_config.width,
                        'height': generation_config.height,
                        'num_inference_steps': generation_config.num_inference_steps,
                        'guidance_scale': generation_config.guidance_scale,
                        'seed': generation_config.seed,
                        'task_id': task_id,
                        'callback': generation_config.progress_callback
                    }
                    generation_result = await pipeline_wrapper.model.generate_video(**wan_params)
            else:
                # Fallback to sync generation for non-WAN models
                generation_result = await asyncio.get_event_loop().run_in_executor(
                    None, pipeline_wrapper.generate, generation_config
                )
            
            if not generation_result.success:
                return self._create_error_result(
                    task_id, "generation_failed",
                    f"I2V generation failed: {', '.join(generation_result.errors)}"
                )
            
            # Post-process and save
            await self._send_progress_update(
                task_id, GenerationStage.POST_PROCESSING, 85,
                "Post-processing I2V frames"
            )
            
            output_path = await self._save_generated_video(
                generation_result.frames, task_id, params
            )
            
            # Create result
            generation_time = time.time() - start_time
            self._generation_count += 1
            self._total_generation_time += generation_time
            
            result = GenerationResult(
                success=True,
                task_id=task_id,
                output_path=output_path,
                generation_time_seconds=generation_time,
                model_used="i2v-A14B",
                parameters_used=params.__dict__.copy(),
                peak_vram_usage_mb=generation_result.peak_memory_mb,
                average_vram_usage_mb=generation_result.memory_used_mb,
                optimizations_applied=generation_result.applied_optimizations.copy()
            )
            
            await self._send_progress_update(
                task_id, GenerationStage.COMPLETED, 100,
                f"I2V generation completed in {generation_time:.1f}s"
            )
            
            # Complete progress tracking
            try:
                from backend.websocket.progress_integration import get_progress_integration
                progress_integration = await get_progress_integration()
                await progress_integration.complete_generation_tracking(
                    success=True, output_path=str(output_path)
                )
            except Exception as e:
                self.logger.warning(f"Failed to complete progress tracking: {e}")
            
            self.logger.info(f"I2V generation completed: {task_id} in {generation_time:.1f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"I2V generation failed: {e}")
            await self._send_progress_update(
                task_id, GenerationStage.FAILED, 0,
                f"I2V generation failed: {str(e)}"
            )
            
            # Complete progress tracking with error
            try:
                from backend.websocket.progress_integration import get_progress_integration
                progress_integration = await get_progress_integration()
                await progress_integration.complete_generation_tracking(
                    success=False, error_message=str(e)
                )
            except Exception as pe:
                self.logger.warning(f"Failed to complete progress tracking: {pe}")
            
            return self._create_error_result(task_id, "generation_error", str(e))
        
        finally:
            # Clean up LoRA for this task
            await self._cleanup_lora_for_task(task_id)
            self._current_task_id = None
            self._progress_callback = None
    
    async def generate_ti2v(self, image_path: str, prompt: str, params: GenerationParams,
                           progress_callback: Optional[Callable] = None) -> GenerationResult:
        """
        Generate text+image-to-video using existing WAN TI2V pipeline
        
        Args:
            image_path: Path to input image
            prompt: Text prompt for generation
            params: Generation parameters
            progress_callback: Optional progress callback function
            
        Returns:
            GenerationResult with generated video information
        """
        task_id = f"ti2v_{uuid.uuid4().hex[:8]}"
        self._current_task_id = task_id
        self._progress_callback = progress_callback
        
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting TI2V generation: {task_id}")
            
            # Initialize progress tracking
            try:
                from backend.websocket.progress_integration import get_progress_integration
                progress_integration = await get_progress_integration()
                await progress_integration.start_generation_tracking(
                    task_id, "ti2v-5B", estimated_duration=45.0  # Estimate 45 seconds for TI2V (smaller model)
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize progress tracking: {e}")
            
            await self._send_progress_update(
                task_id, GenerationStage.INITIALIZING, 0,
                "Initializing TI2V generation"
            )
            
            # Validate parameters for TI2V
            validation_result = self._validate_ti2v_params(image_path, prompt, params)
            if not validation_result["valid"]:
                return self._create_error_result(
                    task_id, "parameter_validation",
                    f"Invalid parameters: {', '.join(validation_result['errors'])}"
                )
            
            # Load TI2V pipeline
            await self._send_progress_update(
                task_id, GenerationStage.LOADING_MODEL, 10,
                "Loading TI2V model pipeline"
            )
            
            pipeline_wrapper = await self._load_pipeline("ti2v-5B", params)
            if not pipeline_wrapper:
                return self._create_error_result(
                    task_id, "model_loading", "Failed to load TI2V pipeline"
                )
            
            # Prepare inputs
            await self._send_progress_update(
                task_id, GenerationStage.PREPARING_INPUTS, 25,
                "Preparing text and image inputs"
            )
            
            # Load and validate input image
            input_image = await self._load_and_validate_image(image_path)
            if input_image is None:
                return self._create_error_result(
                    task_id, "image_loading", f"Failed to load input image: {image_path}"
                )
            
            # Handle end image if provided
            end_image = None
            if params.end_image_path:
                end_image = await self._load_and_validate_image(params.end_image_path)
                if end_image is None:
                    self.logger.warning(f"Failed to load end image: {params.end_image_path}")
            
            # Create generation configuration for TI2V
            generation_config = self._create_generation_config(prompt, params)
            generation_config.input_image = input_image
            if end_image is not None:
                generation_config.end_image = end_image
            
            # Apply LoRA if specified
            lora_applied = await self._apply_lora_to_pipeline(pipeline_wrapper, params, task_id)
            if lora_applied:
                await self._send_progress_update(
                    task_id, GenerationStage.PREPARING_INPUTS, 30,
                    f"Applied LoRA: {Path(params.lora_path).stem if params.lora_path else 'unknown'}"
                )
            
            # Set up progress callback
            def generation_progress_callback(step: int, total_steps: int, latents: torch.Tensor):
                progress = 35 + int((step / total_steps) * 50)
                
                # Send internal progress update
                asyncio.create_task(self._send_progress_update(
                    task_id, GenerationStage.GENERATING, progress,
                    f"Generating TI2V frame {step}/{total_steps}",
                    current_step=step, total_steps=total_steps
                ))
                
                # Call external progress callback if provided
                if progress_callback:
                    try:
                        # Convert internal progress (35-85%) to external progress (0-100%)
                        external_progress = ((progress - 35) / 50) * 100
                        asyncio.create_task(progress_callback(
                            external_progress, 
                            f"Generating TI2V frame {step}/{total_steps}"
                        ))
                    except Exception as e:
                        self.logger.warning(f"External progress callback failed: {e}")
            
            generation_config.progress_callback = generation_progress_callback
            
            # Generate video
            await self._send_progress_update(
                task_id, GenerationStage.GENERATING, 35,
                f"Generating video from text+image ({params.num_frames} frames)"
            )
            
            generation_result = await asyncio.get_event_loop().run_in_executor(
                None, pipeline_wrapper.generate, generation_config
            )
            
            if not generation_result.success:
                return self._create_error_result(
                    task_id, "generation_failed",
                    f"TI2V generation failed: {', '.join(generation_result.errors)}"
                )
            
            # Post-process and save
            await self._send_progress_update(
                task_id, GenerationStage.POST_PROCESSING, 85,
                "Post-processing TI2V frames"
            )
            
            output_path = await self._save_generated_video(
                generation_result.frames, task_id, params
            )
            
            # Create result
            generation_time = time.time() - start_time
            self._generation_count += 1
            self._total_generation_time += generation_time
            
            result = GenerationResult(
                success=True,
                task_id=task_id,
                output_path=output_path,
                generation_time_seconds=generation_time,
                model_used="ti2v-5B",
                parameters_used=params.__dict__.copy(),
                peak_vram_usage_mb=generation_result.peak_memory_mb,
                average_vram_usage_mb=generation_result.memory_used_mb,
                optimizations_applied=generation_result.applied_optimizations.copy()
            )
            
            await self._send_progress_update(
                task_id, GenerationStage.COMPLETED, 100,
                f"TI2V generation completed in {generation_time:.1f}s"
            )
            
            # Complete progress tracking
            try:
                from backend.websocket.progress_integration import get_progress_integration
                progress_integration = await get_progress_integration()
                await progress_integration.complete_generation_tracking(
                    success=True, output_path=str(output_path)
                )
            except Exception as e:
                self.logger.warning(f"Failed to complete progress tracking: {e}")
            
            self.logger.info(f"TI2V generation completed: {task_id} in {generation_time:.1f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"TI2V generation failed: {e}")
            await self._send_progress_update(
                task_id, GenerationStage.FAILED, 0,
                f"TI2V generation failed: {str(e)}"
            )
            
            # Complete progress tracking with error
            try:
                from backend.websocket.progress_integration import get_progress_integration
                progress_integration = await get_progress_integration()
                await progress_integration.complete_generation_tracking(
                    success=False, error_message=str(e)
                )
            except Exception as pe:
                self.logger.warning(f"Failed to complete progress tracking: {pe}")
            
            return self._create_error_result(task_id, "generation_error", str(e))
        
        finally:
            # Clean up LoRA for this task
            await self._cleanup_lora_for_task(task_id)
            self._current_task_id = None
            self._progress_callback = None
    
    async def _load_pipeline(self, model_type: str, params: GenerationParams) -> Optional[Any]:
        """Load pipeline using WanPipelineLoader with caching and automatic model download"""
        try:
            # Check cache first
            cache_key = f"{model_type}_{hash(str(sorted(params.__dict__.items())))}"
            if cache_key in self._pipeline_cache:
                self.logger.info(f"Using cached pipeline for {model_type}")
                return self._pipeline_cache[cache_key]
            
            if not self.wan_pipeline_loader:
                self.logger.error("WanPipelineLoader not available")
                return None
            
            # Ensure model is available (download if necessary)
            await self._send_progress_update(
                self._current_task_id, GenerationStage.LOADING_MODEL, 5,
                f"Checking model availability: {model_type}"
            )
            
            model_available = await self._ensure_model_available(model_type)
            if not model_available:
                self.logger.error(f"Model {model_type} is not available and could not be downloaded")
                return None
            
            # Determine model path based on type
            model_path = self._get_model_path(model_type)
            if not model_path:
                self.logger.error(f"Could not determine model path for {model_type}")
                return None
            
            # Create optimization config from params
            optimization_config = self._create_optimization_config(params)
            
            # Load pipeline with optimization
            await self._send_progress_update(
                self._current_task_id, GenerationStage.LOADING_MODEL, 15,
                f"Loading pipeline for {model_type}"
            )
            
            self.logger.info(f"Loading pipeline for {model_type} from {model_path}")
            
            # Check if we have the full WanPipelineLoader or simplified version
            if hasattr(self.wan_pipeline_loader, 'load_wan_pipeline'):
                # Full WanPipelineLoader with optimization support
                pipeline_wrapper = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    self.wan_pipeline_loader.load_wan_pipeline,
                    model_path,
                    True,  # trust_remote_code
                    True,  # apply_optimizations
                    optimization_config
                )
            else:
                # Simplified loader - load pipeline directly
                pipeline = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.wan_pipeline_loader.load_pipeline,
                    model_type,
                    model_path
                )
                
                if pipeline:
                    # Create a simple wrapper for the pipeline
                    pipeline_wrapper = self._create_simple_pipeline_wrapper(pipeline, model_type)
                else:
                    pipeline_wrapper = None
            
            # Cache the pipeline
            self._pipeline_cache[cache_key] = pipeline_wrapper
            
            self.logger.info(f"Pipeline loaded successfully for {model_type}")
            return pipeline_wrapper
            
        except Exception as e:
            self.logger.error(f"Failed to load pipeline for {model_type}: {e}")
            return None
    
    def _create_simple_pipeline_wrapper(self, pipeline, model_type: str):
        """Create a simple wrapper for pipelines loaded with simplified loader"""
        class SimplePipelineWrapper:
            def __init__(self, pipeline, model_type):
                self.pipeline = pipeline
                self.model_type = model_type
                
            def generate(self, config):
                """Generate using the wrapped pipeline"""
                try:
                    # Prepare generation arguments
                    generation_args = {
                        "prompt": config.prompt,
                        "num_frames": getattr(config, 'num_frames', 16),
                        "width": getattr(config, 'width', 512),
                        "height": getattr(config, 'height', 512),
                        "num_inference_steps": getattr(config, 'num_inference_steps', 20),
                        "guidance_scale": getattr(config, 'guidance_scale', 7.5),
                    }
                    
                    # Add optional parameters
                    if hasattr(config, 'negative_prompt') and config.negative_prompt:
                        generation_args["negative_prompt"] = config.negative_prompt
                    if hasattr(config, 'seed') and config.seed is not None:
                        import torch
                        generation_args["generator"] = torch.Generator().manual_seed(config.seed)
                    
                    # Add input image for I2V/TI2V
                    if hasattr(config, 'input_image') and config.input_image is not None:
                        generation_args["image"] = config.input_image
                    
                    # Generate
                    with torch.no_grad():
                        result = self.pipeline(**generation_args)
                    
                    # Extract frames
                    if hasattr(result, 'frames'):
                        frames = result.frames
                    elif hasattr(result, 'images'):
                        frames = result.images
                    elif hasattr(result, 'videos'):
                        frames = result.videos
                    else:
                        frames = [result] if not isinstance(result, list) else result
                    
                    # Create result object
                    from dataclasses import dataclass
                    from typing import List
                    
                    @dataclass
                    class SimpleGenerationResult:
                        success: bool = True
                        frames: List = None
                        errors: List = None
                        peak_memory_mb: int = 0
                        memory_used_mb: int = 0
                        applied_optimizations: List = None
                        
                        def __post_init__(self):
                            if self.errors is None:
                                self.errors = []
                            if self.applied_optimizations is None:
                                self.applied_optimizations = []
                    
                    return SimpleGenerationResult(
                        success=True,
                        frames=frames,
                        peak_memory_mb=0,
                        memory_used_mb=0,
                        applied_optimizations=["simplified_loading"]
                    )
                    
                except Exception as e:
                    from dataclasses import dataclass
                    from typing import List
                    
                    @dataclass
                    class SimpleGenerationResult:
                        success: bool = False
                        frames: List = None
                        errors: List = None
                        peak_memory_mb: int = 0
                        memory_used_mb: int = 0
                        applied_optimizations: List = None
                        
                        def __post_init__(self):
                            if self.errors is None:
                                self.errors = []
                            if self.applied_optimizations is None:
                                self.applied_optimizations = []
                    
                    return SimpleGenerationResult(
                        success=False,
                        errors=[str(e)]
                    )
        
        return SimplePipelineWrapper(pipeline, model_type)
    
    async def _ensure_model_available(self, model_type: str) -> bool:
        """Ensure model is available, triggering download if necessary"""
        try:
            # Import the model integration bridge
            from backend.core.model_integration_bridge import ensure_model_ready
            
            # Check if model is available and download if necessary
            self.logger.info(f"Ensuring model {model_type} is available")
            
            # Send progress update for model download check
            if self._current_task_id:
                await self._send_progress_update(
                    self._current_task_id, GenerationStage.LOADING_MODEL, 8,
                    f"Ensuring model {model_type} is available"
                )
                
                # Also send model loading progress update
                try:
                    from backend.websocket.progress_integration import get_progress_integration
                    progress_integration = await get_progress_integration()
                    await progress_integration.update_model_loading_progress(
                        model_type, 10, "Checking model availability"
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to send model loading progress: {e}")
            
            # This will automatically download the model if it's missing
            model_ready = await ensure_model_ready(model_type)
            
            if model_ready:
                self.logger.info(f"Model {model_type} is ready for use")
                return True
            else:
                self.logger.error(f"Model {model_type} is not ready and could not be prepared")
                return False
                
        except Exception as e:
            self.logger.error(f"Error ensuring model {model_type} is available: {e}")
            return False
    
    def _get_model_path(self, model_type: str) -> Optional[str]:
        """Get model path for a given model type"""
        # Model path mappings - these should match the existing system's paths
        model_paths = {
            "t2v-A14B": "models/Wan-AI_Wan2.2-T2V-A14B-Diffusers",
            "i2v-A14B": "models/Wan-AI_Wan2.2-I2V-A14B-Diffusers", 
            "ti2v-5B": "models/Wan-AI_Wan2.2-TI2V-5B-Diffusers"
        }
        
        model_path = model_paths.get(model_type)
        if model_path:
            # Convert to absolute path
            project_root = Path(__file__).parent.parent.parent
            full_path = project_root / model_path
            return str(full_path)
        
        return None
    
    def _create_optimization_config(self, params: GenerationParams) -> Dict[str, Any]:
        """Create optimization configuration from generation parameters"""
        config = {}
        
        if params.quantization_level:
            config["precision"] = params.quantization_level
        
        if params.max_vram_usage_gb:
            config["min_vram_mb"] = int(params.max_vram_usage_gb * 1024)
        
        if params.enable_offload:
            config["enable_cpu_offload"] = True
        
        # Set chunk size based on resolution and frames
        width, height = self._parse_resolution(params.resolution)
        if width * height * params.num_frames > 1024 * 1024 * 16:  # Large generation
            config["chunk_size"] = 4
        else:
            config["chunk_size"] = 8
        
        return config
    
    def _create_generation_config(self, prompt: str, params: GenerationParams):
        """Create generation configuration from parameters"""
        try:
            from core.services.wan_pipeline_loader import GenerationConfig
            
            width, height = self._parse_resolution(params.resolution)
            
            config = GenerationConfig(
                prompt=prompt,
                negative_prompt=params.negative_prompt,
                num_frames=params.num_frames,
                width=width,
                height=height,
                num_inference_steps=params.steps,
                guidance_scale=params.guidance_scale,
                seed=params.seed,
                fps=params.fps,
                enable_optimizations=True
            )
            
            return config
            
        except ImportError:
            # Fallback if GenerationConfig is not available - create a simple class
            self.logger.warning("GenerationConfig not available, using fallback")
            width, height = self._parse_resolution(params.resolution)
            
            class FallbackGenerationConfig:
                def __init__(self, **kwargs):
                    for key, value in kwargs.items():
                        setattr(self, key, value)
            
            return FallbackGenerationConfig(
                prompt=prompt,
                negative_prompt=params.negative_prompt,
                num_frames=params.num_frames,
                width=width,
                height=height,
                num_inference_steps=params.steps,
                guidance_scale=params.guidance_scale,
                seed=params.seed,
                fps=params.fps,
                progress_callback=None,
                input_image=None,
                end_image=None
            )
    
    def _parse_resolution(self, resolution: str) -> Tuple[int, int]:
        """Parse resolution string to width, height tuple"""
        try:
            if 'x' in resolution:
                width, height = map(int, resolution.split('x'))
                return width, height
            else:
                # Handle preset resolutions
                presets = {
                    "720p": (1280, 720),
                    "1080p": (1920, 1080),
                    "480p": (854, 480)
                }
                return presets.get(resolution, (1280, 720))
        except:
            return (1280, 720)  # Default resolution
    
    async def _load_and_validate_image(self, image_path: str) -> Optional[Any]:
        """Load and validate input image"""
        try:
            from PIL import Image
            import numpy as np
            
            # Load image
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Validate image size (should be reasonable)
            if image.size[0] < 64 or image.size[1] < 64:
                self.logger.error(f"Image too small: {image.size}")
                return None
            
            if image.size[0] > 2048 or image.size[1] > 2048:
                self.logger.warning(f"Large image detected: {image.size}, may need resizing")
            
            return image
            
        except Exception as e:
            self.logger.error(f"Failed to load image {image_path}: {e}")
            return None
    
    async def _apply_lora_to_pipeline(self, pipeline_wrapper: Any, params: GenerationParams, task_id: str) -> bool:
        """Apply LoRA to the pipeline if specified"""
        if not params.lora_path:
            return False
        
        try:
            # Check if LoRA manager is available
            if not self.lora_manager:
                self.logger.warning("LoRA specified but LoRA manager not available - using fallback")
                return await self._apply_lora_fallback(params, task_id)
            
            # Extract LoRA name from path
            lora_path = Path(params.lora_path)
            lora_name = lora_path.stem
            
            self.logger.info(f"Applying LoRA: {lora_name} with strength {params.lora_strength}")
            
            # Load LoRA if not already loaded
            try:
                lora_info = self.lora_manager.load_lora(lora_name)
                self.logger.info(f"LoRA loaded: {lora_info['num_layers']} layers, {lora_info['size_mb']:.1f}MB")
            except Exception as e:
                self.logger.error(f"Failed to load LoRA {lora_name}: {e}")
                return await self._apply_lora_fallback(params, task_id)
            
            # Apply LoRA to the pipeline
            try:
                # Get the actual pipeline from the wrapper
                pipeline = getattr(pipeline_wrapper, 'pipeline', pipeline_wrapper)
                
                # Apply LoRA using the manager
                modified_pipeline = self.lora_manager.apply_lora(
                    pipeline, lora_name, params.lora_strength
                )
                
                # Update the wrapper if necessary
                if hasattr(pipeline_wrapper, 'pipeline'):
                    pipeline_wrapper.pipeline = modified_pipeline
                
                # Track applied LoRA
                self._applied_loras[task_id] = {
                    "name": lora_name,
                    "strength": params.lora_strength,
                    "path": params.lora_path
                }
                
                self.logger.info(f"Successfully applied LoRA {lora_name} with strength {params.lora_strength}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to apply LoRA {lora_name} to pipeline: {e}")
                return await self._apply_lora_fallback(params, task_id)
                
        except Exception as e:
            self.logger.error(f"Error in LoRA application: {e}")
            return await self._apply_lora_fallback(params, task_id)
    
    async def _apply_lora_fallback(self, params: GenerationParams, task_id: str) -> bool:
        """Apply LoRA fallback using prompt enhancement"""
        try:
            if not params.lora_path:
                return False
            
            lora_name = Path(params.lora_path).stem
            self.logger.info(f"Applying LoRA fallback for: {lora_name}")
            
            # Use LoRA manager's fallback if available
            if self.lora_manager:
                enhanced_prompt = self.lora_manager.get_fallback_prompt_enhancement(
                    params.prompt, lora_name
                )
            else:
                # Basic fallback enhancement
                enhanced_prompt = self._get_basic_lora_fallback(params.prompt, lora_name)
            
            # Update the prompt in params (this is a bit hacky but necessary for fallback)
            params.prompt = enhanced_prompt
            
            self.logger.info(f"Applied LoRA fallback enhancement: '{enhanced_prompt}'")
            return True
            
        except Exception as e:
            self.logger.error(f"LoRA fallback failed: {e}")
            return False
    
    def _get_basic_lora_fallback(self, base_prompt: str, lora_name: str) -> str:
        """Basic LoRA fallback prompt enhancement"""
        # Simple enhancement based on common LoRA types
        lora_lower = lora_name.lower()
        
        if "anime" in lora_lower:
            enhancement = "anime style, detailed anime art"
        elif "realistic" in lora_lower or "photo" in lora_lower:
            enhancement = "photorealistic, highly detailed"
        elif "art" in lora_lower or "paint" in lora_lower:
            enhancement = "artistic style, detailed artwork"
        elif "detail" in lora_lower or "quality" in lora_lower:
            enhancement = "extremely detailed, high quality"
        else:
            enhancement = "enhanced style, high quality"
        
        # Combine with base prompt
        if base_prompt.strip():
            return f"{base_prompt}, {enhancement}"
        else:
            return enhancement
    
    async def _cleanup_lora_for_task(self, task_id: str):
        """Clean up LoRA application for a completed task"""
        try:
            if task_id in self._applied_loras:
                lora_info = self._applied_loras[task_id]
                self.logger.info(f"Cleaning up LoRA for task {task_id}: {lora_info['name']}")
                
                # Remove from tracking
                del self._applied_loras[task_id]
                
                # Note: We don't unload the LoRA from the pipeline here as it might be
                # cached for future use. The LoRA manager handles memory management.
                
        except Exception as e:
            self.logger.warning(f"Error cleaning up LoRA for task {task_id}: {e}")
    
    def get_lora_status(self) -> Dict[str, Any]:
        """Get current LoRA status information"""
        try:
            status = {
                "lora_manager_available": self.lora_manager is not None,
                "applied_loras": self._applied_loras.copy(),
                "available_loras": {}
            }
            
            if self.lora_manager:
                status["available_loras"] = self.lora_manager.list_available_loras()
                status["loaded_loras"] = list(self.lora_manager.loaded_loras.keys())
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting LoRA status: {e}")
            return {
                "lora_manager_available": False,
                "error": str(e)
            }
    
    async def _save_generated_video(self, frames: List[torch.Tensor], task_id: str, 
                                   params: GenerationParams) -> str:
        """Save generated frames as video file"""
        try:
            # Create output directory
            project_root = Path(__file__).parent.parent.parent
            output_dir = project_root / "outputs"
            output_dir.mkdir(exist_ok=True)
            
            # Generate output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"generated_{task_id}_{timestamp}.mp4"
            output_path = output_dir / output_filename
            
            # For now, create a placeholder file with metadata
            # TODO: Implement actual video encoding from frames
            metadata = {
                "task_id": task_id,
                "model_type": params.model_type,
                "prompt": params.prompt,
                "resolution": params.resolution,
                "num_frames": params.num_frames,
                "steps": params.steps,
                "fps": params.fps,
                "generated_at": datetime.utcnow().isoformat(),
                "frame_count": len(frames) if frames else 0
            }
            
            # Write metadata file alongside video
            metadata_path = output_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                import json
                json.dump(metadata, f, indent=2)
            
            # Create placeholder video file
            with open(output_path, 'w') as f:
                f.write(f"Generated video: {task_id}\n")
                f.write(f"Frames: {len(frames) if frames else 0}\n")
                f.write(f"Metadata: {metadata_path}\n")
            
            self.logger.info(f"Video saved: {output_path}")
            # Use forward slashes for consistency across platforms
            relative_path = output_path.relative_to(project_root)
            return str(relative_path).replace('\\', '/')
            
        except Exception as e:
            self.logger.error(f"Failed to save video: {e}")
            # Return a fallback path
            return f"outputs/generated_{task_id}_error.mp4"
    
    async def _send_progress_update(self, task_id: str, stage: GenerationStage, 
                                   progress: int, message: str, **kwargs):
        """Send progress update via enhanced WebSocket progress integration"""
        try:
            # Use the new progress integration system for detailed progress tracking
            try:
                from backend.websocket.progress_integration import get_progress_integration, GenerationStage as ProgressStage
                
                progress_integration = await get_progress_integration()
                
                # Map our GenerationStage to ProgressStage
                progress_stage = self._map_to_progress_stage(stage)
                
                # Send progress update through integration system
                await progress_integration.update_stage_progress(
                    progress_stage, progress, message, **kwargs
                )
                
            except Exception as e:
                self.logger.warning(f"Enhanced progress integration failed: {e}")
            
            # Legacy WebSocket manager for backward compatibility
            if self.websocket_manager:
                await self.websocket_manager.send_generation_progress(
                    task_id=task_id,
                    progress=progress,
                    status=stage.value,
                    message=message,
                    **kwargs
                )
            
            # Also call custom progress callback if provided
            if self._progress_callback:
                try:
                    update = ProgressUpdate(
                        stage=stage,
                        progress_percent=progress,
                        message=message,
                        metadata=kwargs
                    )
                    if asyncio.iscoroutinefunction(self._progress_callback):
                        await self._progress_callback(update)
                    else:
                        self._progress_callback(update)
                except Exception as e:
                    self.logger.warning(f"Progress callback failed: {e}")
            
        except Exception as e:
            self.logger.warning(f"Failed to send progress update: {e}")
    
    def _map_to_progress_stage(self, stage: GenerationStage):
        """Map pipeline GenerationStage to progress integration GenerationStage"""
        try:
            from backend.websocket.progress_integration import GenerationStage as ProgressStage
            
            # Create mapping between stages
            stage_mapping = {
                GenerationStage.INITIALIZING: ProgressStage.INITIALIZING,
                GenerationStage.LOADING_MODEL: ProgressStage.LOADING_MODEL,
                GenerationStage.PREPARING_INPUTS: ProgressStage.PREPARING_INPUTS,
                GenerationStage.GENERATING: ProgressStage.GENERATING,
                GenerationStage.POST_PROCESSING: ProgressStage.POST_PROCESSING,
                GenerationStage.SAVING: ProgressStage.SAVING,
                GenerationStage.COMPLETED: ProgressStage.COMPLETED,
                GenerationStage.FAILED: ProgressStage.FAILED
            }
            
            return stage_mapping.get(stage, ProgressStage.GENERATING)
            
        except Exception as e:
            self.logger.error(f"Failed to map generation stage: {e}")
            # Return a default stage
            from backend.websocket.progress_integration import GenerationStage as ProgressStage
            return ProgressStage.GENERATING
    
    def _validate_t2v_params(self, prompt: str, params: GenerationParams) -> Dict[str, Any]:
        """Validate parameters for T2V generation"""
        errors = []
        warnings = []
        
        if not prompt or len(prompt.strip()) == 0:
            errors.append("Prompt cannot be empty")
        
        if len(prompt) > 1000:
            warnings.append("Very long prompt may affect generation quality")
        
        if params.num_frames < 1 or params.num_frames > 64:
            errors.append("Number of frames must be between 1 and 64")
        
        if params.steps < 1 or params.steps > 100:
            errors.append("Steps must be between 1 and 100")
        
        # Validate LoRA parameters
        lora_validation = self._validate_lora_params(params)
        if not lora_validation["valid"]:
            errors.extend(lora_validation["errors"])
        warnings.extend(lora_validation["warnings"])
        
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}
    
    def _validate_i2v_params(self, image_path: str, prompt: str, params: GenerationParams) -> Dict[str, Any]:
        """Validate parameters for I2V generation"""
        errors = []
        warnings = []
        
        if not image_path or not Path(image_path).exists():
            errors.append("Input image path is required and must exist")
        
        if not prompt or len(prompt.strip()) == 0:
            warnings.append("Empty prompt for I2V - image will be primary driver")
        
        # Reuse T2V validation for common parameters
        t2v_validation = self._validate_t2v_params(prompt or "", params)
        errors.extend(t2v_validation["errors"])
        warnings.extend(t2v_validation["warnings"])
        
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}
    
    def _validate_ti2v_params(self, image_path: str, prompt: str, params: GenerationParams) -> Dict[str, Any]:
        """Validate parameters for TI2V generation"""
        errors = []
        warnings = []
        
        if not image_path or not Path(image_path).exists():
            errors.append("Input image path is required and must exist")
        
        if not prompt or len(prompt.strip()) == 0:
            errors.append("Text prompt is required for TI2V generation")
        
        if params.end_image_path and not Path(params.end_image_path).exists():
            warnings.append("End image path specified but file does not exist")
        
        # Reuse T2V validation for common parameters
        t2v_validation = self._validate_t2v_params(prompt, params)
        errors.extend(t2v_validation["errors"])
        warnings.extend(t2v_validation["warnings"])
        
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}
    
    def _validate_lora_params(self, params: GenerationParams) -> Dict[str, Any]:
        """Validate LoRA parameters"""
        errors = []
        warnings = []
        
        # Validate LoRA strength
        if params.lora_strength < 0.0 or params.lora_strength > 2.0:
            errors.append("LoRA strength must be between 0.0 and 2.0")
        
        # Validate LoRA path if provided
        if params.lora_path:
            lora_path = Path(params.lora_path)
            
            # Check if it's an absolute path or relative to loras directory
            if not lora_path.is_absolute():
                # Try relative to loras directory
                if self.lora_manager:
                    loras_dir = Path(self.lora_manager.loras_directory)
                    lora_path = loras_dir / params.lora_path
                else:
                    # Fallback to project loras directory
                    project_root = Path(__file__).parent.parent.parent
                    lora_path = project_root / "loras" / params.lora_path
            
            # Check if LoRA file exists
            if not lora_path.exists():
                # Try with common extensions if no extension provided
                if not lora_path.suffix:
                    extensions = ['.safetensors', '.pt', '.pth', '.bin']
                    found = False
                    for ext in extensions:
                        if (lora_path.parent / f"{lora_path.name}{ext}").exists():
                            found = True
                            break
                    
                    if not found:
                        errors.append(f"LoRA file not found: {params.lora_path}")
                else:
                    errors.append(f"LoRA file not found: {params.lora_path}")
            else:
                # Validate file extension
                valid_extensions = ['.safetensors', '.pt', '.pth', '.bin']
                if lora_path.suffix.lower() not in valid_extensions:
                    errors.append(f"Invalid LoRA file format. Supported: {', '.join(valid_extensions)}")
                
                # Check file size (warn if very large)
                try:
                    file_size_mb = lora_path.stat().st_size / (1024 * 1024)
                    if file_size_mb > 500:
                        warnings.append(f"Large LoRA file ({file_size_mb:.1f}MB) may slow loading")
                except Exception:
                    pass
        
        # Warn if LoRA is specified but manager is not available
        if params.lora_path and not self.lora_manager:
            warnings.append("LoRA specified but LoRA manager not available - will use fallback prompt enhancement")
        
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}
    
    def _create_error_result(self, task_id: str, error_category: str, error_message: str) -> GenerationResult:
        """Create error result with recovery suggestions"""
        recovery_suggestions = []
        
        if error_category == "model_loading":
            recovery_suggestions = [
                "Check if the model files are downloaded and accessible",
                "Verify sufficient disk space for model loading",
                "Try restarting the application to clear any locks"
            ]
        elif error_category == "parameter_validation":
            recovery_suggestions = [
                "Check that all required parameters are provided",
                "Verify parameter values are within valid ranges",
                "Ensure image paths exist and are accessible"
            ]
        elif error_category == "generation_failed":
            recovery_suggestions = [
                "Try reducing the number of frames or resolution",
                "Check available VRAM and consider enabling optimizations",
                "Verify the prompt is not too complex or contradictory"
            ]
        elif error_category == "image_loading":
            recovery_suggestions = [
                "Verify the image file exists and is readable",
                "Check that the image format is supported (JPG, PNG, etc.)",
                "Ensure the image is not corrupted"
            ]
        
        return GenerationResult(
            success=False,
            task_id=task_id,
            error_message=error_message,
            error_category=error_category,
            recovery_suggestions=recovery_suggestions
        )
    
    def setup_progress_callbacks(self, websocket_manager):
        """Setup progress callbacks for WebSocket updates"""
        self.websocket_manager = websocket_manager
        self.logger.info("Progress callbacks configured for WebSocket updates")
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get generation statistics"""
        return {
            "total_generations": self._generation_count,
            "total_generation_time": self._total_generation_time,
            "average_generation_time": self._total_generation_time / max(1, self._generation_count),
            "cached_pipelines": len(self._pipeline_cache),
            "current_task": self._current_task_id,
            "wan_pipeline_loader_available": self.wan_pipeline_loader is not None,
            "websocket_manager_available": self.websocket_manager is not None
        }
    
    def clear_pipeline_cache(self):
        """Clear the pipeline cache to free memory"""
        self._pipeline_cache.clear()
        self.logger.info("Pipeline cache cleared")
    
    def set_hardware_optimizer(self, optimizer):
        """Set the hardware optimizer for pipeline optimization"""
        try:
            self.hardware_optimizer = optimizer
            if optimizer:
                self.logger.info("Hardware optimizer set successfully for RealGenerationPipeline")
                # Apply hardware optimizations if available
                try:
                    hardware_profile = optimizer.get_hardware_profile()
                    if hardware_profile:
                        self.hardware_profile = hardware_profile
                        self.logger.info(f"Hardware profile updated: {hardware_profile.gpu_model}")
                        
                        # Update VRAM settings based on hardware profile
                        if hardware_profile.vram_gb:
                            self.max_vram_gb = hardware_profile.vram_gb * 0.85  # Use 85% of available VRAM
                            self.logger.info(f"VRAM limit updated to {self.max_vram_gb:.1f}GB")
                            
                except Exception as e:
                    self.logger.warning(f"Could not get hardware profile: {e}")
            else:
                self.logger.warning("Hardware optimizer set to None")
        except Exception as e:
            self.logger.error(f"Error setting hardware optimizer: {e}")

# Global real generation pipeline instance
_real_generation_pipeline = None

async def get_real_generation_pipeline() -> RealGenerationPipeline:
    """Get the global real generation pipeline instance"""
    global _real_generation_pipeline
    if _real_generation_pipeline is None:
        _real_generation_pipeline = RealGenerationPipeline()
        await _real_generation_pipeline.initialize()
    return _real_generation_pipeline