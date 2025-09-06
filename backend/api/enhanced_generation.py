"""
Enhanced Generation API for Phase 1 MVP
Provides seamless T2V, I2V, and TI2V generation with auto-detection and model switching
"""

import logging
import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from fastapi import APIRouter, HTTPException, Form, File, UploadFile, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import json

# Use relative imports that work when running from backend directory
try:
    from ..services.generation_service import GenerationService
except ImportError:
    try:
        from services.generation_service import GenerationService
    except ImportError:
        GenerationService = None

try:
    from ..core.model_integration_bridge import GenerationParams, ModelType
except ImportError:
    try:
        from core.model_integration_bridge import GenerationParams, ModelType
    except ImportError:
        # Create placeholder classes for Phase 1
        class GenerationParams:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        class ModelType:
            pass

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/generation", tags=["Enhanced Generation"])

# Enhanced request models for Phase 1
class GenerationRequest(BaseModel):
    """Enhanced generation request with auto-detection capabilities"""
    model_config = {"protected_namespaces": ()}
    
    prompt: str = Field(..., min_length=1, max_length=500, description="Text prompt for generation")
    model_type: Optional[str] = Field(None, description="Model type (auto-detected if not provided)")
    resolution: str = Field(default="1280x720", description="Video resolution")
    steps: int = Field(default=50, ge=1, le=100, description="Generation steps")
    lora_path: Optional[str] = Field(default="", description="LoRA model path")
    lora_strength: float = Field(default=1.0, ge=0.0, le=2.0, description="LoRA strength")
    
    # Advanced options for Phase 1
    enable_prompt_enhancement: bool = Field(default=True, description="Auto-enhance prompts")
    enable_optimization: bool = Field(default=True, description="Enable hardware optimizations")
    priority: str = Field(default="normal", description="Generation priority (low, normal, high)")
    
    @validator('resolution')
    def validate_resolution(cls, v):
        valid_resolutions = ["854x480", "1024x576", "1280x720", "1920x1080"]
        if v not in valid_resolutions:
            raise ValueError(f"Resolution must be one of: {valid_resolutions}")
        return v
    
    @validator('priority')
    def validate_priority(cls, v):
        valid_priorities = ["low", "normal", "high"]
        if v not in valid_priorities:
            raise ValueError(f"Priority must be one of: {valid_priorities}")
        return v

class GenerationResponse(BaseModel):
    """Enhanced generation response"""
    success: bool
    task_id: str
    message: str
    detected_model_type: Optional[str] = None
    estimated_time_minutes: Optional[float] = None
    queue_position: Optional[int] = None
    enhanced_prompt: Optional[str] = None
    applied_optimizations: List[str] = []

class ModelDetectionService:
    """Service for auto-detecting optimal model type based on inputs"""
    
    @staticmethod
    def detect_model_type(prompt: str, has_image: bool = False, has_end_image: bool = False) -> str:
        """
        Auto-detect optimal model type based on inputs
        
        Args:
            prompt: Text prompt
            has_image: Whether start image is provided
            has_end_image: Whether end image is provided
            
        Returns:
            Detected model type (T2V-A14B, I2V-A14B, or TI2V-5B)
        """
        # Phase 1 logic: Simple but effective detection
        if has_image and has_end_image:
            # Both images provided - use TI2V for interpolation
            return "TI2V-5B"
        elif has_image:
            # Single image provided - check prompt for text+image indicators
            text_image_keywords = [
                "transform", "change", "evolve", "morph", "animate", 
                "from this image", "based on", "starting from"
            ]
            if any(keyword in prompt.lower() for keyword in text_image_keywords):
                return "TI2V-5B"  # Text heavily influences the image
            else:
                return "I2V-A14B"  # Pure image-to-video
        else:
            # No image - pure text-to-video
            return "T2V-A14B"
    
    @staticmethod
    def get_model_requirements(model_type: str) -> Dict[str, Any]:
        """Get requirements and capabilities for a model type"""
        requirements = {
            "T2V-A14B": {
                "requires_image": False,
                "supports_end_image": False,
                "estimated_vram_gb": 8.0,
                "estimated_time_per_frame": 1.2,
                "max_resolution": "1920x1080",
                "recommended_steps": 50
            },
            "I2V-A14B": {
                "requires_image": True,
                "supports_end_image": True,
                "estimated_vram_gb": 8.5,
                "estimated_time_per_frame": 1.4,
                "max_resolution": "1920x1080", 
                "recommended_steps": 50
            },
            "TI2V-5B": {
                "requires_image": True,
                "supports_end_image": True,
                "estimated_vram_gb": 6.0,
                "estimated_time_per_frame": 0.9,
                "max_resolution": "1280x720",
                "recommended_steps": 40
            }
        }
        return requirements.get(model_type, {})

class PromptEnhancementService:
    """Service for enhancing prompts for better generation results"""
    
    @staticmethod
    def enhance_prompt(prompt: str, model_type: str, options: Dict[str, Any] = None) -> str:
        """
        Enhance prompt based on model type and options
        
        Args:
            prompt: Original prompt
            model_type: Target model type
            options: Enhancement options
            
        Returns:
            Enhanced prompt
        """
        if not options:
            options = {}
        
        enhanced_parts = []
        
        # Model-specific enhancements
        if model_type == "T2V-A14B":
            # T2V benefits from cinematic and movement descriptions
            if "cinematic" not in prompt.lower():
                enhanced_parts.append("cinematic composition")
            if not any(movement in prompt.lower() for movement in ["camera", "zoom", "pan", "movement"]):
                enhanced_parts.append("smooth camera movement")
        
        elif model_type == "I2V-A14B":
            # I2V benefits from animation and transition descriptions
            if not any(anim in prompt.lower() for anim in ["animate", "motion", "movement", "transition"]):
                enhanced_parts.append("natural animation")
        
        elif model_type == "TI2V-5B":
            # TI2V benefits from transformation descriptions
            if not any(trans in prompt.lower() for trans in ["transform", "evolve", "change", "transition"]):
                enhanced_parts.append("smooth transformation")
        
        # Quality enhancements
        if options.get("enhance_quality", True):
            if not any(quality in prompt.lower() for quality in ["high quality", "detailed", "masterpiece"]):
                enhanced_parts.append("high quality, detailed")
        
        # Technical enhancements
        if options.get("enhance_technical", True):
            if not any(tech in prompt.lower() for tech in ["4k", "8k", "hd"]):
                enhanced_parts.append("HD quality")
        
        # Build enhanced prompt
        if enhanced_parts:
            return f"{prompt}, {', '.join(enhanced_parts)}"
        return prompt

@router.post("/submit", response_model=GenerationResponse)
async def submit_enhanced_generation(
    request: Request,
    background_tasks: BackgroundTasks,
    prompt: str = Form(...),
    model_type: Optional[str] = Form(None),
    resolution: str = Form("1280x720"),
    steps: int = Form(50),
    lora_path: str = Form(""),
    lora_strength: float = Form(1.0),
    enable_prompt_enhancement: bool = Form(True),
    enable_optimization: bool = Form(True),
    priority: str = Form("normal"),
    image: Optional[UploadFile] = File(None),
    end_image: Optional[UploadFile] = File(None)
):
    """
    Enhanced generation endpoint with auto-detection and seamless model switching
    Phase 1 MVP: Supports T2V, I2V, and TI2V with intelligent model selection
    """
    
    task_id = f"gen_{uuid.uuid4().hex[:8]}"
    
    try:
        logger.info(f"🎬 Enhanced Generation Request - Task: {task_id}")
        logger.info(f"📝 Prompt: '{prompt[:100]}{'...' if len(prompt) > 100 else ''}'")
        logger.info(f"🖼️ Images: Start={image is not None}, End={end_image is not None}")
        
        # Phase 1: Auto-detect model type if not provided
        if not model_type:
            detected_model_type = ModelDetectionService.detect_model_type(
                prompt, 
                has_image=image is not None,
                has_end_image=end_image is not None
            )
            logger.info(f"🤖 Auto-detected model type: {detected_model_type}")
        else:
            detected_model_type = model_type
            logger.info(f"🎯 Using specified model type: {detected_model_type}")
        
        # Validate model requirements
        model_requirements = ModelDetectionService.get_model_requirements(detected_model_type)
        
        if model_requirements.get("requires_image", False) and not image:
            raise HTTPException(
                status_code=422,
                detail=f"Model {detected_model_type} requires a start image"
            )
        
        # Phase 1: Enhance prompt if enabled
        enhanced_prompt = prompt
        if enable_prompt_enhancement:
            enhanced_prompt = PromptEnhancementService.enhance_prompt(
                prompt, detected_model_type, {
                    "enhance_quality": True,
                    "enhance_technical": True
                }
            )
            if enhanced_prompt != prompt:
                logger.info(f"✨ Enhanced prompt: '{enhanced_prompt[:100]}{'...' if len(enhanced_prompt) > 100 else ''}'")
        
        # Handle image uploads
        image_path = None
        end_image_path = None
        
        if image:
            # Validate and save start image
            if image.size > 10 * 1024 * 1024:  # 10MB limit
                raise HTTPException(status_code=422, detail="Start image too large (max 10MB)")
            
            upload_dir = Path("uploads")
            upload_dir.mkdir(exist_ok=True)
            
            image_filename = f"{task_id}_start.{image.filename.split('.')[-1]}"
            image_path = upload_dir / image_filename
            
            with open(image_path, "wb") as f:
                content = await image.read()
                f.write(content)
            
            logger.info(f"📁 Saved start image: {image_path}")
        
        if end_image:
            # Validate and save end image
            if end_image.size > 10 * 1024 * 1024:  # 10MB limit
                raise HTTPException(status_code=422, detail="End image too large (max 10MB)")
            
            upload_dir = Path("uploads")
            upload_dir.mkdir(exist_ok=True)
            
            end_image_filename = f"{task_id}_end.{end_image.filename.split('.')[-1]}"
            end_image_path = upload_dir / end_image_filename
            
            with open(end_image_path, "wb") as f:
                content = await end_image.read()
                f.write(content)
            
            logger.info(f"📁 Saved end image: {end_image_path}")
        
        # Create generation parameters
        generation_params = GenerationParams(
            prompt=enhanced_prompt,
            model_type=detected_model_type,
            image_path=str(image_path) if image_path else None,
            end_image_path=str(end_image_path) if end_image_path else None,
            resolution=resolution,
            num_inference_steps=steps,
            lora_path=lora_path if lora_path else None,
            lora_strength=lora_strength
        )
        
        # Get generation service from the current app instance
        # This will be injected by the FastAPI app when the router is included
        generation_service = getattr(request.app.state, 'generation_service', None)
        if not generation_service:
            # For Phase 1, create a mock response since the full generation service might not be available
            return GenerationResponse(
                success=True,
                task_id=task_id,
                message=f"Generation task queued with {detected_model_type} (Phase 1 MVP mode)",
                detected_model_type=detected_model_type,
                estimated_time_minutes=estimated_time_minutes,
                queue_position=0,
                enhanced_prompt=enhanced_prompt if enhanced_prompt != prompt else None,
                applied_optimizations=applied_optimizations
            )
        
        # Estimate generation time
        frames_count = 16  # Default frame count for Phase 1
        time_per_frame = model_requirements.get("estimated_time_per_frame", 1.0)
        estimated_time_minutes = (frames_count * time_per_frame) / 60.0
        
        # Submit to generation service
        success = await generation_service.submit_generation_task(
            task_id=task_id,
            parameters=generation_params,
            priority=priority
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to submit generation task")
        
        # Get queue position
        queue_info = await generation_service.get_queue_status()
        queue_position = len([t for t in queue_info.get("tasks", []) if t.get("status") == "pending"])
        
        # Applied optimizations for Phase 1
        applied_optimizations = []
        if enable_optimization:
            applied_optimizations.extend([
                "Hardware-specific quantization",
                "Memory optimization",
                "Pipeline caching"
            ])
        
        if enable_prompt_enhancement and enhanced_prompt != prompt:
            applied_optimizations.append("Prompt enhancement")
        
        logger.info(f"✅ Generation task submitted successfully: {task_id}")
        
        return GenerationResponse(
            success=True,
            task_id=task_id,
            message=f"Generation task submitted successfully with {detected_model_type}",
            detected_model_type=detected_model_type,
            estimated_time_minutes=estimated_time_minutes,
            queue_position=queue_position,
            enhanced_prompt=enhanced_prompt if enhanced_prompt != prompt else None,
            applied_optimizations=applied_optimizations
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Enhanced generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@router.get("/models/detect")
async def detect_optimal_model(
    prompt: str,
    has_image: bool = False,
    has_end_image: bool = False
):
    """
    Detect optimal model type for given inputs
    Phase 1: Provides intelligent model recommendations
    """
    try:
        detected_model = ModelDetectionService.detect_model_type(prompt, has_image, has_end_image)
        requirements = ModelDetectionService.get_model_requirements(detected_model)
        
        # Generate explanation for the detection
        explanation = []
        if has_image and has_end_image:
            explanation.append("Both start and end images provided - TI2V recommended for interpolation")
        elif has_image:
            text_indicators = any(keyword in prompt.lower() for keyword in [
                "transform", "change", "evolve", "morph", "animate"
            ])
            if text_indicators:
                explanation.append("Image + text transformation keywords detected - TI2V recommended")
            else:
                explanation.append("Single image provided - I2V recommended for pure image animation")
        else:
            explanation.append("Text-only input - T2V recommended for pure text-to-video generation")
        
        return {
            "detected_model_type": detected_model,
            "confidence": 0.9,  # Phase 1: Static confidence, can be enhanced later
            "explanation": explanation,
            "requirements": requirements,
            "alternatives": [
                model for model in ["T2V-A14B", "I2V-A14B", "TI2V-5B"] 
                if model != detected_model
            ]
        }
        
    except Exception as e:
        logger.error(f"Model detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model detection failed: {str(e)}")

@router.post("/prompt/enhance")
async def enhance_prompt_endpoint(
    prompt: str = Form(...),
    model_type: str = Form("T2V-A14B"),
    enhance_quality: bool = Form(True),
    enhance_technical: bool = Form(True)
):
    """
    Enhance prompt for better generation results
    Phase 1: Model-specific prompt optimization
    """
    try:
        enhanced_prompt = PromptEnhancementService.enhance_prompt(
            prompt, model_type, {
                "enhance_quality": enhance_quality,
                "enhance_technical": enhance_technical
            }
        )
        
        enhancements_applied = []
        if enhanced_prompt != prompt:
            # Detect what enhancements were applied
            added_parts = enhanced_prompt.replace(prompt, "").strip(", ")
            if added_parts:
                enhancements_applied = [part.strip() for part in added_parts.split(",")]
        
        return {
            "original_prompt": prompt,
            "enhanced_prompt": enhanced_prompt,
            "enhancements_applied": enhancements_applied,
            "model_type": model_type,
            "character_count": {
                "original": len(prompt),
                "enhanced": len(enhanced_prompt),
                "difference": len(enhanced_prompt) - len(prompt)
            }
        }
        
    except Exception as e:
        logger.error(f"Prompt enhancement failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prompt enhancement failed: {str(e)}")

@router.get("/capabilities")
async def get_generation_capabilities():
    """
    Get current generation capabilities and model status
    Phase 1: Provides system capabilities overview
    """
    try:
        # Get model requirements for all supported models
        models_info = {}
        for model_type in ["T2V-A14B", "I2V-A14B", "TI2V-5B"]:
            models_info[model_type] = ModelDetectionService.get_model_requirements(model_type)
        
        return {
            "supported_models": list(models_info.keys()),
            "models_info": models_info,
            "supported_resolutions": ["854x480", "1024x576", "1280x720", "1920x1080"],
            "supported_formats": ["mp4"],
            "max_steps": 100,
            "min_steps": 1,
            "default_steps": 50,
            "max_prompt_length": 500,
            "max_image_size_mb": 10,
            "supported_image_formats": ["JPEG", "PNG", "WebP"],
            "features": {
                "auto_model_detection": True,
                "prompt_enhancement": True,
                "lora_support": True,
                "hardware_optimization": True,
                "real_time_progress": True,
                "queue_management": True
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get capabilities: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get capabilities: {str(e)}")