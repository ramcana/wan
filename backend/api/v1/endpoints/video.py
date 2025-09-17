from fastapi import (
    APIRouter, Depends, HTTPException, File, UploadFile, BackgroundTasks,
    Request
)
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from backend.schemas.validation import VideoRequest
from backend.services.input_validation_service import InputValidationService
from backend.middleware.auth_middleware import AuthMiddleware
from backend.middleware.rate_limit_middleware import RateLimitMiddleware
from backend.database import get_db
from backend.models.auth import User
from backend.core.security_config import security_settings
from backend.services.auth_service import AuthService
from backend.services.rate_limit_service import RateLimitService
import uuid


# Initialize services
auth_service = AuthService(secret_key=security_settings.SECRET_KEY)
rate_limit_service = RateLimitService()
auth_middleware = AuthMiddleware(auth_service=auth_service)
rate_limit_middleware = RateLimitMiddleware(
    rate_limit_service=rate_limit_service
)

router = APIRouter()


@router.post("/generate")
async def generate_video(
    request: Request,
    video_request: VideoRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(auth_middleware.get_current_user),
    db: Session = Depends(get_db),
    _rate_limit: None = Depends(rate_limit_middleware.check_rate_limit)
):
    """Generate video with security validation"""
    
    # Initialize input validator for this request
    input_validator = InputValidationService()
    
    # Validate content policy
    is_valid, violations = input_validator.validate_content_policy(
        video_request.prompt
    )
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Content policy violation",
                "violations": violations
            }
        )
    
    # Sanitize inputs
    sanitized_prompt = input_validator.sanitize_text(
        video_request.prompt, 500
    )
    sanitized_negative_prompt = input_validator.sanitize_text(
        video_request.negative_prompt or "", 300
    )
    
    # For demonstration, we'll just use the sanitized prompts
    # In a real implementation, these would be used in the video generation
    _ = sanitized_prompt
    _ = sanitized_negative_prompt
    
    # Create video generation task
    task_id = str(uuid.uuid4())
    
    # For now, we'll just return a mock response since we don't have
    # the actual video service implemented
    # In a real implementation, you would queue the generation task
    
    # Add rate limit headers to response
    response_headers = getattr(request.state, 'rate_limit_headers', {})
    
    return JSONResponse(
        content={
            "task_id": task_id,
            "status": "queued",
            "message": "Video generation started"
        },
        headers=response_headers
    )


@router.post("/upload-image")
async def upload_image(
    request: Request,
    file: UploadFile = File(...),
    current_user: User = Depends(auth_middleware.get_current_user),
    db: Session = Depends(get_db),
    _rate_limit: None = Depends(rate_limit_middleware.check_rate_limit)
):
    """Upload image with security validation"""
    
    # Initialize input validator for this request
    input_validator = InputValidationService()
    
    # Read file content
    file_content = await file.read()
    
    # Validate file upload
    # Handle potential None values for filename and content_type
    filename = file.filename or "unknown_file"
    content_type = file.content_type or "application/octet-stream"
    
    is_valid, errors = input_validator.validate_file_upload(
        file_content, filename, content_type
    )
    
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "File validation failed",
                "errors": errors
            }
        )
    
    # Sanitize filename
    safe_filename = input_validator.sanitize_filename(filename)
    
    # Store file securely
    file_id = str(uuid.uuid4())
    storage_path = f"uploads/{current_user.id}/{file_id}_{safe_filename}"
    
    # For demonstration, we'll just use the storage path
    # In a real implementation, this would be used to store the file
    _ = storage_path
    
    # For now, we'll just return a mock response since we don't have
    # the actual video service implemented
    # In a real implementation, you would store the file
    
    response_headers = getattr(request.state, 'rate_limit_headers', {})
    
    return JSONResponse(
        content={
            "file_id": file_id,
            "filename": safe_filename,
            "size": len(file_content),
            "status": "uploaded"
        },
        headers=response_headers
    )


@router.get("/task/{task_id}")
async def get_task_status(
    task_id: str,
    current_user: User = Depends(auth_middleware.get_current_user),
    db: Session = Depends(get_db)
):
    """Get video generation task status"""
    
    # Validate task_id format
    try:
        uuid.UUID(task_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task ID format")
    
    # For now, we'll just return a mock response since we don't have
    # the actual video service implemented
    # In a real implementation, you would get the task status
    
    return {
        "task_id": task_id,
        "status": "completed",
        "result_url": f"/outputs/{task_id}.mp4"
    }