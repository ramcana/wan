#!/usr/bin/env python3
"""
WAN22 FastAPI Backend
Main FastAPI application entry point
"""

import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
import logging
from datetime import datetime
from typing import Dict, List, Optional
import psutil
import platform

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic models for request validation
class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Text prompt for enhancement")
    options: Optional[Dict] = Field(default=None, description="Enhancement options")

class GenerationRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    prompt: str = Field(..., min_length=1, description="Text prompt for video generation")
    model_type: str = Field(..., description="Model type (T2V-A14B, I2V-A14B, TI2V-5B)")
    resolution: str = Field(default="720p", description="Video resolution")
    steps: int = Field(default=50, ge=1, le=100, description="Generation steps")
    lora_path: Optional[str] = Field(default="", description="LoRA model path")
    lora_strength: float = Field(default=1.0, ge=0.0, le=2.0, description="LoRA strength")

# In-memory task storage (in production, use a proper database)
task_queue: Dict[str, Dict] = {}

# Create FastAPI app
app = FastAPI(
    title="WAN22 Video Generation API",
    description="AI Video Generation System API",
    version="2.2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    if request.method == "POST" and "generation/submit" in str(request.url):
        logger.info(f"Generation request headers: {dict(request.headers)}")
    response = await call_next(request)
    return response

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "wan22-backend"}

@app.get("/api/health")
async def api_health_check():
    """API health check endpoint"""
    return {"status": "healthy", "api_version": "2.2.0"}

# Prompt enhancement endpoint
@app.post("/api/v1/prompt/enhance")
async def enhance_prompt(request: dict):
    """Enhance image generation prompt with AI improvements"""
    try:
        logger.info(f"Received prompt enhancement request: {request}")
        
        # Handle nested structure from frontend
        if "prompt" in request and isinstance(request["prompt"], dict):
            # Frontend sends nested structure: {"prompt": {"prompt": "text", "options": {...}}}
            inner_request = request["prompt"]
            prompt = inner_request.get("prompt", "").strip()
            options = inner_request.get("options", {}) or {}
        else:
            # Direct structure: {"prompt": "text", "options": {...}}
            prompt = request.get("prompt", "").strip()
            options = request.get("options", {}) or {}
        
        if not prompt:
            logger.warning("Empty prompt received")
            raise HTTPException(status_code=422, detail="Prompt cannot be empty")
    except Exception as e:
        logger.error(f"Error processing prompt enhancement request: {e}")
        raise HTTPException(status_code=422, detail=f"Invalid request: {str(e)}")
    
    # Basic prompt enhancement logic
    enhanced_parts = []
    enhancements_applied = []
    
    # Add cinematic quality if enabled
    if options.get("apply_cinematic", True):
        if "cinematic" not in prompt.lower():
            enhanced_parts.append("cinematic lighting")
            enhancements_applied.append("Cinematic lighting")
    
    # Add style improvements if enabled  
    if options.get("apply_style", True):
        if not any(style in prompt.lower() for style in ["detailed", "high quality", "masterpiece"]):
            enhanced_parts.append("highly detailed, masterpiece quality")
            enhancements_applied.append("Quality enhancement")
    
    # Add technical improvements
    if not any(tech in prompt.lower() for tech in ["4k", "8k", "hd", "resolution"]):
        enhanced_parts.append("8K resolution")
        enhancements_applied.append("Resolution enhancement")
    
    # Detect and enhance VACE (Visual Aesthetic Composition Enhancement)
    vace_detected = options.get("apply_vace") is not False
    if vace_detected:
        enhanced_parts.append("perfect composition, rule of thirds")
        enhancements_applied.append("VACE composition")
    
    # Build enhanced prompt
    if enhanced_parts:
        enhanced_prompt = f"{prompt}, {', '.join(enhanced_parts)}"
    else:
        enhanced_prompt = prompt
    
    return {
        "original_prompt": prompt,
        "enhanced_prompt": enhanced_prompt,
        "enhancements_applied": enhancements_applied,
        "character_count": {
            "original": len(prompt),
            "enhanced": len(enhanced_prompt),
            "difference": len(enhanced_prompt) - len(prompt)
        },
        "vace_detected": vace_detected,
        "detected_style": "photorealistic" if "photo" in prompt.lower() else "artistic",
        "confidence": 0.9
    }

# Prompt preview endpoint
@app.post("/api/v1/prompt/preview")
async def preview_prompt(request: dict):
    """Preview prompt enhancement without applying changes"""
    prompt = request.get("prompt", "").strip()
    
    if not prompt:
        raise HTTPException(status_code=422, detail="Prompt cannot be empty")
    
    # Generate preview enhancement
    enhanced_parts = []
    suggested_enhancements = []
    
    # Detect potential improvements
    if "cinematic" not in prompt.lower():
        enhanced_parts.append("cinematic lighting")
        suggested_enhancements.append("Add cinematic lighting for better visual appeal")
    
    if not any(style in prompt.lower() for style in ["detailed", "high quality", "masterpiece"]):
        enhanced_parts.append("highly detailed, masterpiece quality")
        suggested_enhancements.append("Enhance quality descriptors")
    
    if not any(tech in prompt.lower() for tech in ["4k", "8k", "hd", "resolution"]):
        enhanced_parts.append("8K resolution")
        suggested_enhancements.append("Add resolution enhancement")
    
    # Build preview
    if enhanced_parts:
        preview_enhanced = f"{prompt}, {', '.join(enhanced_parts)}"
    else:
        preview_enhanced = prompt
    
    return {
        "original_prompt": prompt,
        "preview_enhanced": preview_enhanced,
        "suggested_enhancements": suggested_enhancements,
        "detected_style": "photorealistic" if "photo" in prompt.lower() else "artistic",
        "vace_detected": "composition" in prompt.lower() or "rule of thirds" in prompt.lower(),
        "character_count": {
            "original": len(prompt),
            "preview": len(preview_enhanced),
            "difference": len(preview_enhanced) - len(prompt)
        },
        "quality_score": 0.85
    }

# Prompt validation endpoint
@app.post("/api/v1/prompt/validate")
async def validate_prompt(request: dict):
    """Validate prompt for generation"""
    prompt = request.get("prompt", "").strip()
    
    is_valid = True
    message = "Prompt is valid"
    suggestions = []
    
    if not prompt:
        is_valid = False
        message = "Prompt cannot be empty"
    elif len(prompt) < 3:
        is_valid = False
        message = "Prompt is too short"
        suggestions.append("Add more descriptive details")
    elif len(prompt) > 500:
        is_valid = False
        message = "Prompt is too long (max 500 characters)"
        suggestions.append("Shorten the prompt while keeping key details")
    else:
        # Check for common issues
        if not any(char.isalpha() for char in prompt):
            is_valid = False
            message = "Prompt should contain descriptive text"
        elif prompt.count(',') > 20:
            suggestions.append("Consider reducing the number of comma-separated elements")
        elif len(prompt.split()) < 3:
            suggestions.append("Add more descriptive words for better results")
    
    return {
        "is_valid": is_valid,
        "message": message,
        "character_count": len(prompt),
        "suggestions": suggestions
    }

# Prompt styles endpoint
@app.get("/api/v1/prompt/styles")
async def get_prompt_styles():
    """Get available prompt styles"""
    styles = [
        {
            "name": "photorealistic",
            "display_name": "Photorealistic",
            "description": "Realistic photography style with natural lighting"
        },
        {
            "name": "artistic",
            "display_name": "Artistic",
            "description": "Creative artistic style with enhanced visual elements"
        },
        {
            "name": "cinematic",
            "display_name": "Cinematic",
            "description": "Movie-like composition with dramatic lighting"
        },
        {
            "name": "anime",
            "display_name": "Anime",
            "description": "Japanese animation style"
        },
        {
            "name": "fantasy",
            "display_name": "Fantasy",
            "description": "Magical and fantastical elements"
        }
    ]
    
    return {
        "styles": styles,
        "total_count": len(styles)
    }

# Basic generation endpoint (placeholder)
@app.post("/api/v1/generate")
async def generate_video():
    """Generate video endpoint (placeholder)"""
    return {"message": "Video generation endpoint - implementation pending"}

# Queue endpoint
@app.get("/api/v1/queue")
async def get_queue():
    """Get generation queue"""
    queue_items = []
    for task_id, task_data in task_queue.items():
        queue_items.append({
            "id": task_id,
            "model_type": task_data["parameters"]["model_type"],
            "prompt": task_data["parameters"]["prompt"],
            "image_path": task_data["parameters"].get("image_path"),
            "resolution": task_data["parameters"]["resolution"],
            "steps": task_data["parameters"]["steps"],
            "lora_path": task_data["parameters"].get("lora_path"),
            "lora_strength": task_data["parameters"]["lora_strength"],
            "status": task_data["status"],
            "progress": task_data.get("progress", 0),
            "created_at": task_data["created_at"],
            "started_at": task_data.get("started_at"),
            "completed_at": task_data.get("completed_at"),
            "output_path": f"outputs/{task_id}.mp4" if task_data["status"] == "completed" else None,
            "error_message": task_data.get("error_message"),
            "estimated_time_minutes": 10 if task_data["status"] == "queued" else None
        })
    
    # Sort by creation time (newest first)
    queue_items.sort(key=lambda x: x["created_at"], reverse=True)
    
    return {
        "tasks": queue_items,
        "total_tasks": len(queue_items),
        "pending_tasks": len([t for t in queue_items if t["status"] == "queued"]),
        "processing_tasks": len([t for t in queue_items if t["status"] == "processing"]),
        "completed_tasks": len([t for t in queue_items if t["status"] == "completed"]),
        "failed_tasks": len([t for t in queue_items if t["status"] == "failed"])
    }

# Generation endpoint with image upload support
@app.post("/api/v1/generation/submit")
async def submit_generation(
    prompt: str = Form(None),
    model_type: str = Form(None),
    resolution: str = Form("1280x720"),
    steps: int = Form(50),
    lora_path: str = Form(""),
    lora_strength: float = Form(1.0),
    image: UploadFile = File(None),
    end_image: UploadFile = File(None)
):
    """Submit video generation request with optional image upload"""
    
    logger.info(f"Generation endpoint reached - prompt: '{prompt}', model_type: '{model_type}', resolution: '{resolution}', steps: {steps}")
    logger.info(f"Prompt type: {type(prompt)}, Model type: {type(model_type)}")
    
    # Validate required fields
    if not prompt:
        logger.warning("Missing prompt in generation request")
        raise HTTPException(status_code=422, detail="Prompt is required")
    
    if not model_type:
        logger.warning("Missing model_type in generation request")
        raise HTTPException(status_code=422, detail="Model type is required")
    
    # Validate prompt content
    if not prompt.strip():
        logger.warning("Empty prompt received in generation request")
        raise HTTPException(status_code=422, detail="Prompt cannot be empty")
    
    # Validate model type
    valid_models = ["T2V-A14B", "I2V-A14B", "TI2V-5B"]
    if model_type not in valid_models:
        raise HTTPException(status_code=422, detail=f"Invalid model type. Must be one of: {valid_models}")
    
    # Validate steps range
    if steps < 1 or steps > 100:
        raise HTTPException(status_code=422, detail="Steps must be between 1 and 100")
    
    # Validate resolution
    valid_resolutions = ["854x480", "1024x576", "1280x720", "1920x1080"]
    if resolution not in valid_resolutions:
        raise HTTPException(status_code=422, detail=f"Invalid resolution. Must be one of: {valid_resolutions}")
    
    # Validate LoRA strength
    if lora_strength < 0.0 or lora_strength > 2.0:
        raise HTTPException(status_code=422, detail="LoRA strength must be between 0.0 and 2.0")
    
    # Check image requirements
    image_required_models = ["I2V-A14B", "TI2V-5B"]
    if model_type in image_required_models and not image:
        raise HTTPException(status_code=422, detail=f"Image is required for {model_type} model")
    
    # Validate and save start image if provided
    image_path = None
    if image:
        # Check file type
        allowed_types = ["image/jpeg", "image/png", "image/webp"]
        if image.content_type not in allowed_types:
            raise HTTPException(status_code=422, detail="Invalid image format. Only JPEG, PNG, and WebP are supported")
        
        # Check file size (10MB limit)
        content = await image.read()
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=422, detail="Image file too large. Maximum size is 10MB")
        
        # Save uploaded image (in real implementation, save to proper location)
        import os
        import uuid
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        file_extension = image.filename.split('.')[-1] if '.' in image.filename else 'jpg'
        image_filename = f"{uuid.uuid4()}.{file_extension}"
        image_path = os.path.join(upload_dir, image_filename)
        
        with open(image_path, "wb") as f:
            f.write(content)
    
    # Validate and save end image if provided
    end_image_path = None
    if end_image:
        # Check file type
        allowed_types = ["image/jpeg", "image/png", "image/webp"]
        if end_image.content_type not in allowed_types:
            raise HTTPException(status_code=422, detail="Invalid end image format. Only JPEG, PNG, and WebP are supported")
        
        # Check file size (10MB limit)
        end_content = await end_image.read()
        if len(end_content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=422, detail="End image file too large. Maximum size is 10MB")
        
        # Save uploaded end image
        import os
        import uuid
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        end_file_extension = end_image.filename.split('.')[-1] if '.' in end_image.filename else 'jpg'
        end_image_filename = f"{uuid.uuid4()}_end.{end_file_extension}"
        end_image_path = os.path.join(upload_dir, end_image_filename)
        
        with open(end_image_path, "wb") as f:
            f.write(end_content)
    
    # Generate task ID
    import uuid
    task_id = str(uuid.uuid4())
    
    # Store task in queue
    task_data = {
        "task_id": task_id,
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "progress": 0,
        "estimated_time": "5-10 minutes",
        "parameters": {
            "model_type": model_type,
            "prompt": prompt,
            "resolution": resolution,
            "steps": steps,
            "image_path": image_path,
            "end_image_path": end_image_path,
            "lora_path": lora_path if lora_path else None,
            "lora_strength": lora_strength
        }
    }
    
    task_queue[task_id] = task_data
    
    # Return generation response
    response = {
        "task_id": task_id,
        "status": "queued",
        "message": "Generation request submitted successfully",
        "parameters": {
            "model_type": model_type,
            "prompt": prompt,
            "resolution": resolution,
            "steps": steps,
            "image_path": image_path,
            "end_image_path": end_image_path,
            "lora_path": lora_path if lora_path else None,
            "lora_strength": lora_strength
        }
    }
    
    return response

# System stats endpoint
@app.get("/api/v1/system/stats")
async def get_system_stats():
    """Get system statistics"""
    try:
        # CPU stats
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory stats
        memory = psutil.virtual_memory()
        ram_used_gb = memory.used / (1024**3)
        ram_total_gb = memory.total / (1024**3)
        ram_percent = memory.percent
        
        # GPU stats (basic fallback if no GPU libraries available)
        gpu_percent = 0
        vram_used_mb = 0
        vram_total_mb = 1024  # Default 1GB
        vram_percent = 0
        
        # Try to get GPU stats if available
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                gpu_percent = gpu.load * 100
                vram_used_mb = gpu.memoryUsed
                vram_total_mb = gpu.memoryTotal
                vram_percent = (gpu.memoryUsed / gpu.memoryTotal) * 100
        except ImportError:
            # GPUtil not available, use fallback values
            pass
        except Exception:
            # GPU detection failed, use fallback values
            pass
        
        return {
            "cpu_percent": round(cpu_percent, 1),
            "ram_used_gb": round(ram_used_gb, 2),
            "ram_total_gb": round(ram_total_gb, 2),
            "ram_percent": round(ram_percent, 1),
            "gpu_percent": round(gpu_percent, 1),
            "vram_used_mb": round(vram_used_mb, 1),
            "vram_total_mb": round(vram_total_mb, 1),
            "vram_percent": round(vram_percent, 1),
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "platform": platform.system(),
                "architecture": platform.architecture()[0],
                "processor": platform.processor(),
                "python_version": platform.python_version()
            }
        }
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        # Return fallback stats if system monitoring fails
        return {
            "cpu_percent": 0.0,
            "ram_used_gb": 0.0,
            "ram_total_gb": 1.0,
            "ram_percent": 0.0,
            "gpu_percent": 0.0,
            "vram_used_mb": 0.0,
            "vram_total_mb": 1024.0,
            "vram_percent": 0.0,
            "timestamp": datetime.now().isoformat(),
            "error": "System monitoring unavailable"
        }

# Outputs endpoint
@app.get("/api/v1/outputs")
async def get_outputs():
    """Get generated outputs"""
    outputs_dir = "outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    
    videos = []
    completed_tasks = [task for task in task_queue.values() if task["status"] == "completed"]
    
    for task in completed_tasks:
        task_id = task["task_id"]
        video_path = f"{outputs_dir}/{task_id}.mp4"
        thumbnail_path = f"{outputs_dir}/thumbnails/{task_id}_thumb.jpg"
        
        # Check if video file exists
        if os.path.exists(video_path):
            file_size = os.path.getsize(video_path) / (1024 * 1024)  # Size in MB
            
            videos.append({
                "id": task_id,
                "filename": f"{task_id}.mp4",
                "prompt": task["parameters"]["prompt"],
                "model_type": task["parameters"]["model_type"],
                "resolution": task["parameters"]["resolution"],
                "file_size_mb": round(file_size, 2),
                "created_at": task["created_at"],
                "duration": "10.0",  # Default duration
                "thumbnail_url": f"/api/v1/outputs/{task_id}/thumbnail",
                "download_url": f"/api/v1/outputs/{task_id}/download",
                "video_url": f"/api/v1/outputs/{task_id}/video"
            })
    
    return {
        "videos": videos,
        "total_count": len(videos),
        "total_size_mb": sum(v["file_size_mb"] for v in videos)
    }

# Get specific video info
@app.get("/api/v1/outputs/{video_id}")
async def get_video_info(video_id: str):
    """Get specific video information"""
    if video_id not in task_queue:
        raise HTTPException(status_code=404, detail="Video not found")
    
    task = task_queue[video_id]
    video_path = f"outputs/{video_id}.mp4"
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    file_size = os.path.getsize(video_path) / (1024 * 1024)
    
    return {
        "id": video_id,
        "filename": f"{video_id}.mp4",
        "prompt": task["parameters"]["prompt"],
        "model_type": task["parameters"]["model_type"],
        "resolution": task["parameters"]["resolution"],
        "file_size_mb": round(file_size, 2),
        "created_at": task["created_at"],
        "duration": "10.0",
        "thumbnail_url": f"/api/v1/outputs/{video_id}/thumbnail",
        "download_url": f"/api/v1/outputs/{video_id}/download",
        "video_url": f"/api/v1/outputs/{video_id}/video"
    }

# Serve video files
@app.get("/api/v1/outputs/{video_id}/video")
async def get_video_file(video_id: str):
    """Serve video file"""
    video_path = f"outputs/{video_id}.mp4"
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    from fastapi.responses import FileResponse
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"{video_id}.mp4"
    )

# Serve thumbnail files
@app.get("/api/v1/outputs/{video_id}/thumbnail")
async def get_thumbnail_file(video_id: str):
    """Serve thumbnail file"""
    thumbnail_path = f"outputs/thumbnails/{video_id}_thumb.jpg"
    if not os.path.exists(thumbnail_path):
        # Return a default thumbnail or generate one
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    
    from fastapi.responses import FileResponse
    return FileResponse(
        thumbnail_path,
        media_type="image/jpeg",
        filename=f"{video_id}_thumb.jpg"
    )

# Download video
@app.get("/api/v1/outputs/{video_id}/download")
async def download_video(video_id: str):
    """Download video file"""
    video_path = f"outputs/{video_id}.mp4"
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    from fastapi.responses import FileResponse
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"wan22_video_{video_id}.mp4",
        headers={"Content-Disposition": "attachment"}
    )

# Delete video
@app.delete("/api/v1/outputs/{video_id}")
async def delete_video(video_id: str):
    """Delete video and associated files"""
    if video_id not in task_queue:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Remove files
    video_path = f"outputs/{video_id}.mp4"
    thumbnail_path = f"outputs/thumbnails/{video_id}_thumb.jpg"
    
    if os.path.exists(video_path):
        os.remove(video_path)
    if os.path.exists(thumbnail_path):
        os.remove(thumbnail_path)
    
    # Remove from task queue
    del task_queue[video_id]
    
    return {"message": "Video deleted successfully"}

# Simulate video generation completion (for testing)
@app.post("/api/v1/simulate/complete/{task_id}")
async def simulate_completion(task_id: str):
    """Simulate video generation completion for testing"""
    if task_id not in task_queue:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Update task status
    task_queue[task_id]["status"] = "completed"
    task_queue[task_id]["progress"] = 100
    
    # Create dummy video file for demonstration
    outputs_dir = "outputs"
    thumbnails_dir = "outputs/thumbnails"
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(thumbnails_dir, exist_ok=True)
    
    video_path = f"{outputs_dir}/{task_id}.mp4"
    thumbnail_path = f"{thumbnails_dir}/{task_id}_thumb.jpg"
    
    # Create a small dummy video file (1KB)
    with open(video_path, "wb") as f:
        f.write(b"dummy video content for testing" * 32)  # ~1KB file
    
    # Create a small dummy thumbnail (if one exists in outputs/thumbnails/)
    existing_thumb = "outputs/thumbnails/0846777d-d15a-40d5-ae78-ab1b13244649_thumb.jpg"
    if os.path.exists(existing_thumb):
        import shutil
        shutil.copy2(existing_thumb, thumbnail_path)
    else:
        # Create dummy thumbnail
        with open(thumbnail_path, "wb") as f:
            f.write(b"dummy thumbnail content" * 10)
    
    return {
        "message": "Video generation completed",
        "task_id": task_id,
        "video_url": f"/api/v1/outputs/{task_id}/video",
        "thumbnail_url": f"/api/v1/outputs/{task_id}/thumbnail"
    }

if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
