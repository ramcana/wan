"""
Generation API endpoints
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from sqlalchemy.orm import Session
from typing import Optional
import uuid
import os
import shutil
from datetime import datetime
from PIL import Image
import io

from backend.schemas.schemas import GenerationRequest, GenerationResponse, TaskStatus, ModelType
from backend.repositories.database import get_db, GenerationTaskDB, TaskStatusEnum, ModelTypeEnum
from backend.services.generation_service import get_generation_service, GenerationService
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Supported image formats
SUPPORTED_IMAGE_FORMATS = {
    'image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/bmp', 'image/tiff'
}

# Maximum image file size (10MB)
MAX_IMAGE_SIZE = 10 * 1024 * 1024

async def validate_and_process_image(image: UploadFile, task_id: str) -> str:
    """
    Validate and process uploaded image for I2V/TI2V generation
    """
    try:
        # Validate content type
        if not image.content_type or image.content_type not in SUPPORTED_IMAGE_FORMATS:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported image format. Supported formats: {', '.join(SUPPORTED_IMAGE_FORMATS)}"
            )
        
        # Read image content
        image_content = await image.read()
        
        # Validate file size
        if len(image_content) > MAX_IMAGE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Image file too large. Maximum size: {MAX_IMAGE_SIZE // (1024*1024)}MB"
            )
        
        # Validate image can be opened and processed
        try:
            pil_image = Image.open(io.BytesIO(image_content))
            
            # Validate image dimensions
            width, height = pil_image.size
            if width < 64 or height < 64:
                raise HTTPException(
                    status_code=400,
                    detail="Image too small. Minimum dimensions: 64x64 pixels"
                )
            
            if width > 4096 or height > 4096:
                raise HTTPException(
                    status_code=400,
                    detail="Image too large. Maximum dimensions: 4096x4096 pixels"
                )
            
            # Convert to RGB if necessary (for JPEG compatibility)
            if pil_image.mode in ('RGBA', 'LA', 'P'):
                pil_image = pil_image.convert('RGB')
            
            logger.info(f"Image validated: {width}x{height}, format: {pil_image.format}, mode: {pil_image.mode}")
            
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}"
            )
        
        # Create uploads directory if it doesn't exist
        uploads_dir = "uploads"
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Generate safe filename
        original_filename = image.filename or "image"
        file_extension = os.path.splitext(original_filename)[1].lower()
        if not file_extension:
            file_extension = '.jpg'  # Default extension
        
        safe_filename = f"{task_id}_input{file_extension}"
        image_path = os.path.join(uploads_dir, safe_filename)
        
        # Save processed image
        pil_image.save(image_path, format='JPEG' if file_extension in ['.jpg', '.jpeg'] else 'PNG', quality=95)
        
        logger.info(f"Image saved to: {image_path}")
        return image_path
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process image: {str(e)}"
        )

@router.post("/generate", response_model=GenerationResponse)
async def generate_video(
    model_type: ModelType = Form(...),
    prompt: str = Form(..., min_length=1, max_length=500),
    resolution: str = Form(default="1280x720"),
    steps: int = Form(default=50, ge=1, le=100),
    lora_path: Optional[str] = Form(None),
    lora_strength: float = Form(default=1.0, ge=0.0, le=2.0),
    image: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    generation_service: GenerationService = Depends(get_generation_service)
):
    """
    Generate a video based on the provided parameters
    """
    try:
        # Validate inputs
        if not prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        # Validate resolution format
        if not resolution.count('x') == 1:
            raise HTTPException(status_code=400, detail="Invalid resolution format")
        
        try:
            width, height = map(int, resolution.split('x'))
            if width <= 0 or height <= 0:
                raise ValueError()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid resolution values")
        
        # Validate image requirements based on model type
        if model_type == ModelType.T2V_A14B:
            # T2V mode should not have an image
            if image:
                raise HTTPException(
                    status_code=400,
                    detail="T2V mode does not accept image input. Use I2V or TI2V for image-based generation."
                )
        elif model_type in [ModelType.I2V_A14B, ModelType.TI2V_5B]:
            # I2V and TI2V modes require an image
            if not image:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Image is required for {model_type} mode"
                )
        
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Handle image upload if provided
        image_path = None
        if image:
            # Validate and process image
            image_path = await validate_and_process_image(image, task_id)
        
        # Estimate generation time based on resolution and model
        estimated_time = estimate_generation_time(model_type, resolution)
        
        # Create task in database
        db_task = GenerationTaskDB(
            id=task_id,
            model_type=ModelTypeEnum(model_type.value),
            prompt=prompt.strip(),
            image_path=image_path,
            resolution=resolution,
            steps=steps,
            lora_path=lora_path,
            lora_strength=lora_strength,
            status=TaskStatusEnum.PENDING,
            progress=0,
            created_at=datetime.utcnow(),
            estimated_time_minutes=estimated_time
        )
        
        db.add(db_task)
        db.commit()
        db.refresh(db_task)
        
        logger.info(f"Created generation task {task_id} for {model_type} with prompt: {prompt[:50]}...")
        
        # Initialize generation service (this will start the background worker)
        await generation_service.initialize()
        
        logger.info(f"Task {task_id} added to generation queue")
        
        return GenerationResponse(
            task_id=task_id,
            status=TaskStatus.PENDING,
            message="Task created successfully and added to queue",
            estimated_time_minutes=estimated_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating generation task: {str(e)}")
        # Standardized error response
        error_detail = {
            "error": "generation_task_creation_failed",
            "message": "Failed to create generation task",
            "details": str(e),
            "suggestions": [
                "Check if the system has sufficient resources",
                "Verify that the model type is supported",
                "Try again with different parameters"
            ]
        }
        raise HTTPException(status_code=500, detail=error_detail)

def estimate_generation_time(model_type: ModelType, resolution: str) -> int:
    """
    Estimate generation time based on model type and resolution
    """
    base_times = {
        ModelType.T2V_A14B: 8,  # minutes for 720p
        ModelType.I2V_A14B: 7,  # minutes for 720p
        ModelType.TI2V_5B: 15,  # minutes for 720p
    }
    
    base_time = base_times.get(model_type, 10)
    
    # Adjust for resolution
    width, height = map(int, resolution.split('x'))
    pixels = width * height
    
    # 720p baseline (1280x720 = 921,600 pixels)
    baseline_pixels = 1280 * 720
    
    # Scale time based on pixel count
    scale_factor = pixels / baseline_pixels
    
    return max(1, int(base_time * scale_factor))

@router.get("/generate/{task_id}")
async def get_generation_task(
    task_id: str,
    db: Session = Depends(get_db)
):
    """
    Get information about a specific generation task
    """
    task = db.query(GenerationTaskDB).filter(GenerationTaskDB.id == task_id).first()
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return {
        "id": task.id,
        "model_type": task.model_type.value,
        "prompt": task.prompt,
        "resolution": task.resolution,
        "status": task.status.value,
        "progress": task.progress,
        "created_at": task.created_at,
        "started_at": task.started_at,
        "completed_at": task.completed_at,
        "output_path": task.output_path,
        "error_message": task.error_message,
        "estimated_time_minutes": task.estimated_time_minutes
    }