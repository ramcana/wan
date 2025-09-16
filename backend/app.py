#!/usr/bin/env python3
"""
WAN22 FastAPI Backend
Main FastAPI application entry point
"""

import os
import sys
import asyncio
import threading
import time
from pathlib import Path
from fastapi import FastAPI, HTTPException, Form, File, UploadFile, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
import logging
from datetime import datetime
from typing import Dict, List, Optional, Set
import psutil
import platform
import json

# Add current directory to Python path for relative imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import fallback recovery system
from backend.core.fallback_recovery_system import (
    get_fallback_recovery_system, FailureType, RecoveryAction
)

# Import CORS validator
from backend.core.cors_validator import (
    validate_cors_configuration, get_cors_error_info, generate_cors_error_response
)
from backend.repositories.database import (
    SessionLocal,
    GenerationTaskDB,
    TaskStatusEnum,
    ModelTypeEnum,
)

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

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: dict):
        if not self.active_connections:
            return
        
        message_str = json.dumps(message)
        disconnected = set()
        
        for connection in self.active_connections.copy():
            try:
                await connection.send_text(message_str)
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                disconnected.add(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

manager = ConnectionManager()

# Create FastAPI app
app = FastAPI(
    title="WAN22 Video Generation API",
    description="AI Video Generation System API",
    version="2.2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database and services on startup"""
    try:
        from repositories.database import create_tables
        create_tables()  # Create tables if they don't exist
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        # Don't raise - allow server to start even if database fails
        logger.warning("Server starting without database functionality")
    
    # Initialize performance monitoring
    try:
        from backend.core.performance_monitoring_system import initialize_performance_monitoring
        await initialize_performance_monitoring()
        logger.info("Performance monitoring system initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize performance monitoring: {e}")
        # Don't raise - performance monitoring is optional
    
    # Validate CORS configuration
    try:
        is_valid, errors = validate_cors_configuration(app)
        if is_valid:
            logger.info("CORS configuration validation passed")
        else:
            logger.warning(f"CORS configuration issues detected: {errors}")
    except Exception as e:
        logger.warning(f"Failed to validate CORS configuration: {e}")
    
    # Initialize generation service (singleton pattern)
    try:
        from backend.services.generation_service import generation_service
        await generation_service.initialize()
        app.state.generation_service = generation_service
        logger.info("Generation service initialized and background processing started")
    except Exception as e:
        logger.error(f"Failed to initialize generation service: {e}")
        # Don't raise - we want the server to start even if generation service fails


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup services on shutdown"""
    try:
        from backend.core.performance_monitoring_system import shutdown_performance_monitoring
        await shutdown_performance_monitoring()
        logger.info("Performance monitoring system shutdown successfully")
    except Exception as e:
        logger.warning(f"Failed to shutdown performance monitoring: {e}")
    
    # Shutdown generation service
    try:
        if hasattr(app.state, 'generation_service'):
            app.state.generation_service.is_processing = False
            logger.info("Generation service shutdown successfully")
    except Exception as e:
        logger.warning(f"Failed to shutdown generation service: {e}")


# Include API routers
try:
    from api.performance import router as performance_router
    app.include_router(performance_router)
    logger.info("Performance API router included")
except ImportError as e:
    logger.warning(f"Could not import performance router: {e}")

try:
    from api.performance_dashboard import router as performance_dashboard_router
    app.include_router(performance_dashboard_router)
    logger.info("Performance dashboard API router included")
except ImportError as e:
    logger.warning(f"Could not import performance dashboard router: {e}")

try:
    from api.enhanced_model_configuration import router as config_router
    app.include_router(config_router)
    logger.info("Enhanced model configuration API router included")
except ImportError as e:
    logger.warning(f"Could not import configuration router: {e}")

try:
    from api.enhanced_model_management import router as model_mgmt_router
    app.include_router(model_mgmt_router)
    logger.info("Enhanced model management API router included")
except ImportError as e:
    logger.warning(f"Could not import model management router: {e}")

try:
    from api.enhanced_generation import router as enhanced_gen_router
    app.include_router(enhanced_gen_router)
    logger.info("Enhanced generation API router included")
except ImportError as e:
    logger.warning(f"Could not import enhanced generation router: {e}")

try:
    from api.wan_model_info import router as wan_model_info_router
    app.include_router(wan_model_info_router)
    logger.info("WAN Model Information API router included")
except ImportError as e:
    logger.warning(f"Could not import WAN model info router: {e}")

try:
    from api.wan_model_dashboard import router as wan_dashboard_router
    app.include_router(wan_dashboard_router)
    logger.info("WAN Model Dashboard API router included")
except ImportError as e:
    logger.warning(f"Could not import WAN dashboard router: {e}")

try:
    from api.v1.endpoints.generation import router as v1_generation_router
    app.include_router(v1_generation_router, prefix="/api/v1")
    logger.info("V1 Generation API router included")
except ImportError as e:
    logger.warning(f"Could not import V1 generation router: {e}")

try:
    from api.v1.endpoints.queue import router as v1_queue_router
    app.include_router(v1_queue_router, prefix="/api/v1")
    logger.info("V1 Queue API router included")
except ImportError as e:
    logger.warning(f"Could not import V1 queue router: {e}")

try:
    from backend.api.v2.router import router as v2_router
    app.include_router(v2_router)
    logger.info("V2 API router included")
except ImportError as e:
    logger.warning(f"Could not import V2 router: {e}")

# Configure CORS with enhanced validation
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request logging middleware with CORS error detection
@app.middleware("http")
async def log_requests_and_cors_errors(request, call_next):
    origin = request.headers.get('origin', 'unknown')
    logger.info(f"Request: {request.method} {request.url} from origin: {origin}")
    
    if request.method == "POST" and "generation/submit" in str(request.url):
        logger.info(f"Generation request headers: {dict(request.headers)}")
    
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        # Check if this is a CORS-related error
        cors_error_info = get_cors_error_info(request, e)
        if cors_error_info:
            logger.error(f"CORS error detected: {cors_error_info}")
            # Generate helpful CORS error response
            cors_response = generate_cors_error_response(origin, request.method)
            raise HTTPException(status_code=400, detail=cors_response)
        else:
            # Re-raise non-CORS errors
            raise

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "wan22-backend"}

@app.get("/api/health")
async def api_health_check():
    """API health check endpoint"""
    return {"status": "healthy", "api_version": "2.2.0"}

@app.get("/api/v1/system/health")
async def system_health_check(request: Request):
    """Enhanced system health check endpoint with port and connectivity information"""
    # Get the current server port from environment or default
    server_port = int(os.environ.get("PORT", "9000"))
    
    # Try to detect actual running port from the request
    actual_port = server_port
    try:
        # Extract port from the request URL if available
        if hasattr(request, 'url') and request.url.port:
            actual_port = request.url.port
        elif hasattr(request, 'headers'):
            # Try to get port from Host header
            host_header = request.headers.get('host', '')
            if ':' in host_header:
                try:
                    actual_port = int(host_header.split(':')[1])
                except (ValueError, IndexError):
                    pass
    except Exception as e:
        logger.debug(f"Could not detect actual port from request: {e}")
        # Fall back to environment/default port
        pass
    
    return {
        "status": "ok",
        "port": actual_port,
        "timestamp": datetime.now().isoformat() + 'Z',
        "api_version": "2.2.0",
        "system": "operational",
        "service": "wan22-backend",
        "endpoints": {
            "health": "/api/v1/system/health",
            "docs": "/docs",
            "websocket": "/ws",
            "api_base": "/api/v1"
        },
        "connectivity": {
            "cors_enabled": True,
            "allowed_origins": ["http://localhost:3000"],
            "websocket_available": True,
            "request_origin": request.headers.get('origin', 'unknown'),
            "host_header": request.headers.get('host', 'unknown')
        },
        "server_info": {
            "configured_port": server_port,
            "detected_port": actual_port,
            "environment": os.environ.get("NODE_ENV", "development")
        }
    }

@app.get("/api/v1/system/cors/validate")
async def validate_cors_config(request: Request):
    """Validate CORS configuration and provide diagnostics"""
    try:
        # Validate current CORS configuration
        is_valid, errors = validate_cors_configuration(app)
        
        # Get request origin for diagnostics
        origin = request.headers.get('origin', 'unknown')
        method = request.method
        
        # Check if current request would be allowed
        allowed_origins = ["http://localhost:3000"]  # Current configuration
        origin_allowed = origin in allowed_origins or origin == 'unknown'
        
        return {
            "cors_valid": is_valid,
            "errors": errors,
            "current_request": {
                "origin": origin,
                "method": method,
                "origin_allowed": origin_allowed,
                "headers": dict(request.headers)
            },
            "configuration": {
                "allowed_origins": allowed_origins,
                "allowed_methods": ["*"],
                "allowed_headers": ["*"],
                "allow_credentials": True
            },
            "diagnostics": {
                "preflight_supported": True,
                "credentials_supported": True,
                "wildcard_methods": True,
                "wildcard_headers": True
            }
        }
    except Exception as e:
        logger.error(f"Error validating CORS configuration: {e}")
        return {
            "cors_valid": False,
            "errors": [f"Validation error: {str(e)}"],
            "current_request": {
                "origin": request.headers.get('origin', 'unknown'),
                "method": request.method
            }
        }

@app.options("/api/v1/system/cors/test")
async def cors_preflight_test(request: Request):
    """Test endpoint for CORS preflight requests"""
    origin = request.headers.get('origin', 'unknown')
    method = request.headers.get('access-control-request-method', 'unknown')
    headers = request.headers.get('access-control-request-headers', 'unknown')
    
    logger.info(f"CORS preflight test - Origin: {origin}, Method: {method}, Headers: {headers}")
    
    return {
        "message": "CORS preflight test successful",
        "origin": origin,
        "requested_method": method,
        "requested_headers": headers,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/v1/system/cors/test")
async def cors_post_test(request: Request):
    """Test endpoint for CORS POST requests with custom headers"""
    origin = request.headers.get('origin', 'unknown')
    content_type = request.headers.get('content-type', 'unknown')
    
    logger.info(f"CORS POST test - Origin: {origin}, Content-Type: {content_type}")
    
    try:
        body = await request.json() if content_type == 'application/json' else {}
    except:
        body = {}
    
    return {
        "message": "CORS POST test successful",
        "origin": origin,
        "content_type": content_type,
        "body_received": bool(body),
        "timestamp": datetime.now().isoformat()
    }

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

# Generation endpoint is handled by the v1 router in backend/api/v1/endpoints/generation.py

# Queue endpoint
@app.get("/api/v1/queue")
async def get_queue():
    """Get generation queue"""
    db = SessionLocal()
    try:
        tasks = (
            db.query(GenerationTaskDB)
            .order_by(GenerationTaskDB.created_at.desc())
            .all()
        )

        queue_items: List[Dict[str, object]] = []

        for task in tasks:
            status_value = task.status.value if task.status else TaskStatusEnum.PENDING.value

            task_item: Dict[str, object] = {
                "id": task.id,
                "model_type": task.model_type.value if task.model_type else None,
                "prompt": task.prompt,
                "resolution": task.resolution,
                "steps": task.steps,
                "lora_strength": task.lora_strength,
                "status": status_value,
                "progress": task.progress or 0,
                "created_at": task.created_at.isoformat() if task.created_at else None,
            }

            if task.image_path:
                task_item["image_path"] = task.image_path

            if task.lora_path:
                task_item["lora_path"] = task.lora_path

            if task.started_at:
                task_item["started_at"] = task.started_at.isoformat()

            if task.completed_at:
                task_item["completed_at"] = task.completed_at.isoformat()

            if task.output_path:
                task_item["output_path"] = task.output_path

            if task.error_message:
                task_item["error_message"] = task.error_message

            if task.estimated_time_minutes is not None:
                task_item["estimated_time_minutes"] = task.estimated_time_minutes
            elif status_value == TaskStatusEnum.PENDING.value:
                task_item["estimated_time_minutes"] = 10

            queue_items.append(task_item)

        return {
            "tasks": queue_items,
            "total_tasks": len(tasks),
            "pending_tasks": sum(1 for t in tasks if t.status == TaskStatusEnum.PENDING),
            "processing_tasks": sum(1 for t in tasks if t.status == TaskStatusEnum.PROCESSING),
            "completed_tasks": sum(1 for t in tasks if t.status == TaskStatusEnum.COMPLETED),
            "failed_tasks": sum(1 for t in tasks if t.status == TaskStatusEnum.FAILED),
        }
    finally:
        db.close()

# Enhanced generation endpoint with real AI integration
@app.post("/api/v1/generation/submit")
async def submit_generation(
    request: Request,
    prompt: str = Form(None),
    model_type: str = Form(None),
    resolution: str = Form("1280x720"),
    steps: int = Form(50),
    lora_path: str = Form(""),
    lora_strength: float = Form(1.0),
    image: UploadFile = File(None),
    end_image: UploadFile = File(None)
):
    """Submit video generation request with enhanced real AI integration"""
    
    # Handle both JSON and form data requests
    content_type = request.headers.get("content-type", "")
    
    if "application/json" in content_type:
        # Handle JSON request
        try:
            json_data = await request.json()
            prompt = json_data.get("prompt")
            model_type = json_data.get("model_type") or json_data.get("modelType")
            resolution = json_data.get("resolution", "1280x720")
            steps = json_data.get("steps", 50)
            lora_path = json_data.get("lora_path", "") or json_data.get("loraPath", "")
            lora_strength = json_data.get("lora_strength", 1.0) or json_data.get("loraStrength", 1.0)
            # Note: JSON requests don't support file uploads
            image = None
            end_image = None
        except Exception as e:
            logger.error(f"Error parsing JSON request: {e}")
            raise HTTPException(status_code=400, detail="Invalid JSON request")
    
    logger.info("=" * 60)
    logger.info("🎬 NEW ENHANCED VIDEO GENERATION REQUEST")
    logger.info("=" * 60)
    logger.info(f"📝 Prompt: '{prompt}'")
    logger.info(f"🤖 Model: {model_type}")
    logger.info(f"📐 Resolution: {resolution}")
    logger.info(f"🔄 Steps: {steps}")
    logger.info(f"🎨 LoRA: {lora_path if lora_path else 'None'}")
    logger.info(f"💪 LoRA Strength: {lora_strength}")
    logger.info(f"🖼️  Image Upload: {'Yes' if image else 'No'}")
    logger.info("=" * 60)
    
    try:
        # Use the pre-initialized generation service
        if not hasattr(app.state, 'generation_service'):
            logger.error("Generation service not initialized - this should not happen")
            raise HTTPException(status_code=500, detail="Generation service not available")
        
        generation_service = app.state.generation_service
        
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
            raise HTTPException(status_code=422, detail=f"Start image is required for {model_type} model")
        
        # Check end image for interpolation models (optional but recommended for TI2V)
        if model_type == "TI2V-5B" and end_image:
            logger.info(f"End image provided for {model_type} - will enable interpolation mode")
        
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
            
            # Save uploaded image
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
        
        # Check model availability and VRAM requirements using enhanced service
        model_type_normalized = model_type.lower().replace('-', '_')
        
        # Pre-flight checks using enhanced generation service
        if generation_service.vram_monitor:
            estimated_vram = generation_service._estimate_vram_requirements(model_type_normalized, resolution)
            vram_available, vram_message = generation_service.vram_monitor.check_vram_availability(estimated_vram)
            
            if not vram_available:
                logger.warning(f"VRAM check failed: {vram_message}")
                # Get optimization suggestions
                suggestions = generation_service.vram_monitor.get_optimization_suggestions()
                suggestion_text = "; ".join(suggestions[:2]) if suggestions else "Try reducing resolution or steps"
                raise HTTPException(
                    status_code=422, 
                    detail=f"Insufficient VRAM for generation. {vram_message}. Suggestions: {suggestion_text}"
                )
            else:
                logger.info(f"VRAM check passed: {vram_message}")
        
        # Submit task to enhanced generation service
        task_result = await generation_service.submit_generation_task(
            prompt=prompt,
            model_type=model_type,
            resolution=resolution,
            steps=steps,
            image_path=image_path,
            end_image_path=end_image_path,
            lora_path=lora_path if lora_path else None,
            lora_strength=lora_strength
        )
        
        if not task_result.success:
            logger.error(f"Failed to submit generation task: {task_result.error_message}")
            raise HTTPException(status_code=500, detail=task_result.error_message)
        
        # Return enhanced response with real AI integration status
        response = {
            "task_id": task_result.task_id,
            "status": "pending",
            "message": "Generation request submitted to enhanced AI pipeline",
            "real_ai_enabled": generation_service.use_real_generation,
            "hardware_optimized": generation_service.optimization_applied,
            "estimated_vram_gb": estimated_vram if generation_service.vram_monitor else None,
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
        
        logger.info(f"✅ Enhanced generation task submitted: {task_result.task_id}")
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error in enhanced generation submit: {e}")
        # Fallback to basic task creation if enhanced service fails
        import uuid
        task_id = str(uuid.uuid4())
        
        task_data = {
            "task_id": task_id,
            "status": "pending",
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

        db = SessionLocal()
        try:
            try:
                model_enum = ModelTypeEnum(model_type) if model_type else ModelTypeEnum.T2V_A14B
            except ValueError:
                model_enum = ModelTypeEnum.T2V_A14B

            fallback_task = GenerationTaskDB(
                id=task_id,
                model_type=model_enum,
                prompt=prompt or "",
                image_path=image_path,
                end_image_path=end_image_path,
                resolution=resolution,
                steps=steps,
                lora_path=lora_path if lora_path else None,
                lora_strength=lora_strength,
                status=TaskStatusEnum.PENDING,
                progress=0,
                estimated_time_minutes=10,
            )

            db.add(fallback_task)
            db.commit()
        except Exception as db_error:
            db.rollback()
            logger.error(f"Failed to persist fallback task {task_id}: {db_error}")
        finally:
            db.close()

        return {
            "task_id": task_id,
            "status": "pending",
            "message": "Generation request submitted (fallback mode)",
            "real_ai_enabled": False,
            "hardware_optimized": False,
            "error": f"Enhanced service unavailable: {str(e)}",
            "parameters": task_data["parameters"]
        }

# System stats endpoint
@app.get("/api/v1/system/stats")
async def get_system_stats():
    """Get enhanced system statistics with real model status information"""
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
        
        # Get real model status information
        model_status = {}
        real_ai_enabled = False
        hardware_optimized = False
        generation_service_status = "unavailable"
        
        try:
            # Check if enhanced generation service is available
            if hasattr(app.state, 'generation_service'):
                generation_service = app.state.generation_service
                generation_service_status = "available"
                real_ai_enabled = generation_service.use_real_generation
                hardware_optimized = generation_service.optimization_applied
                
                # Get model status from integration bridge
                if generation_service.model_integration_bridge:
                    try:
                        status_dict = generation_service.model_integration_bridge.get_model_status_from_existing_system()
                        for model_type, status in status_dict.items():
                            model_status[model_type] = {
                                "status": status.status.value,
                                "is_loaded": status.is_loaded,
                                "is_cached": status.is_cached,
                                "size_mb": status.size_mb,
                                "hardware_compatible": status.hardware_compatible,
                                "optimization_applied": status.optimization_applied,
                                "estimated_vram_usage_mb": status.estimated_vram_usage_mb
                            }
                    except Exception as e:
                        logger.warning(f"Failed to get model status from integration bridge: {e}")
            else:
                # Try to get model status directly from model integration bridge
                try:
                    from backend.core.model_integration_bridge import get_model_integration_bridge
                    bridge = await get_model_integration_bridge()
                    if bridge:
                        status_dict = bridge.get_model_status_from_existing_system()
                        for model_type, status in status_dict.items():
                            model_status[model_type] = {
                                "status": status.status.value,
                                "is_loaded": status.is_loaded,
                                "is_cached": status.is_cached,
                                "size_mb": status.size_mb,
                                "hardware_compatible": status.hardware_compatible,
                                "optimization_applied": status.optimization_applied,
                                "estimated_vram_usage_mb": status.estimated_vram_usage_mb
                            }
                        generation_service_status = "bridge_available"
                except Exception as e:
                    logger.warning(f"Failed to get model status directly: {e}")
                    
        except Exception as e:
            logger.warning(f"Failed to get enhanced model status: {e}")
        
        # Get hardware optimization status
        hardware_info = {
            "platform": platform.system(),
            "architecture": platform.architecture()[0],
            "processor": platform.processor(),
            "python_version": platform.python_version()
        }
        
        # Add hardware optimization details if available
        try:
            if hasattr(app.state, 'generation_service') and app.state.generation_service.hardware_profile:
                profile = app.state.generation_service.hardware_profile
                hardware_info.update({
                    "gpu_model": profile.gpu_model,
                    "vram_gb": profile.vram_gb,
                    "cpu_model": profile.cpu_model,
                    "cpu_cores": profile.cpu_cores,
                    "total_memory_gb": profile.total_memory_gb,
                    "optimization_profile": "custom" if hardware_optimized else "default"
                })
        except Exception as e:
            logger.debug(f"Hardware profile not available: {e}")
        
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
            "system_info": hardware_info,
            # Enhanced real AI integration status
            "real_ai_integration": {
                "enabled": real_ai_enabled,
                "generation_service_status": generation_service_status,
                "hardware_optimized": hardware_optimized,
                "model_status": model_status
            }
        }
    except Exception as e:
        logger.error(f"Failed to get enhanced system stats: {e}")
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
            "system_info": {
                "platform": platform.system(),
                "architecture": platform.architecture()[0],
                "processor": platform.processor(),
                "python_version": platform.python_version()
            },
            "real_ai_integration": {
                "enabled": False,
                "generation_service_status": "error",
                "hardware_optimized": False,
                "model_status": {},
                "error": str(e)
            }
        }

# Outputs endpoint
@app.get("/api/v1/outputs")
async def get_outputs():
    """Get generated outputs"""
    outputs_dir = "outputs"
    os.makedirs(outputs_dir, exist_ok=True)

    db = SessionLocal()
    try:
        completed_tasks = (
            db.query(GenerationTaskDB)
            .filter(GenerationTaskDB.status == TaskStatusEnum.COMPLETED)
            .order_by(GenerationTaskDB.completed_at.desc())
            .all()
        )

        videos: List[Dict[str, object]] = []
        total_size = 0.0

        for task in completed_tasks:
            video_path = task.output_path or os.path.join(outputs_dir, f"{task.id}.mp4")
            thumbnail_path = os.path.join(outputs_dir, "thumbnails", f"{task.id}_thumb.jpg")

            if not os.path.exists(video_path):
                continue

            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            total_size += file_size_mb

            videos.append({
                "id": task.id,
                "filename": os.path.basename(video_path),
                "prompt": task.prompt,
                "model_type": task.model_type.value if task.model_type else None,
                "resolution": task.resolution,
                "file_size_mb": round(file_size_mb, 2),
                "created_at": (task.completed_at or task.created_at).isoformat()
                if (task.completed_at or task.created_at)
                else None,
                "duration": "10.0",
                "thumbnail_url": f"/api/v1/outputs/{task.id}/thumbnail",
                "download_url": f"/api/v1/outputs/{task.id}/download",
                "video_url": f"/api/v1/outputs/{task.id}/video",
                "thumbnail_path": thumbnail_path if os.path.exists(thumbnail_path) else None,
                "output_path": video_path,
            })

        return {
            "videos": videos,
            "total_count": len(videos),
            "total_size_mb": round(total_size, 2),
        }
    finally:
        db.close()

# Get specific video info
@app.get("/api/v1/outputs/{video_id}")
async def get_video_info(video_id: str):
    """Get specific video information"""
    db = SessionLocal()
    try:
        task = (
            db.query(GenerationTaskDB)
            .filter(
                GenerationTaskDB.id == video_id,
                GenerationTaskDB.status == TaskStatusEnum.COMPLETED,
            )
            .first()
        )

        if not task:
            raise HTTPException(status_code=404, detail="Video not found")

        video_path = task.output_path or os.path.join("outputs", f"{video_id}.mp4")

        if not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail="Video file not found")

        file_size = os.path.getsize(video_path) / (1024 * 1024)

        created_at = task.completed_at or task.created_at

        return {
            "id": video_id,
            "filename": os.path.basename(video_path),
            "prompt": task.prompt,
            "model_type": task.model_type.value if task.model_type else None,
            "resolution": task.resolution,
            "file_size_mb": round(file_size, 2),
            "created_at": created_at.isoformat() if created_at else None,
            "duration": "10.0",
            "thumbnail_url": f"/api/v1/outputs/{video_id}/thumbnail",
            "download_url": f"/api/v1/outputs/{video_id}/download",
            "video_url": f"/api/v1/outputs/{video_id}/video",
        }
    finally:
        db.close()

# Serve video files
@app.get("/api/v1/outputs/{video_id}/video")
async def get_video_file(video_id: str):
    """Serve video file"""
    db = SessionLocal()
    try:
        task = db.query(GenerationTaskDB).filter(GenerationTaskDB.id == video_id).first()
    finally:
        db.close()

    video_path = (
        task.output_path if task and task.output_path else os.path.join("outputs", f"{video_id}.mp4")
    )

    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")

    from fastapi.responses import FileResponse

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=os.path.basename(video_path)
    )

# Serve thumbnail files
@app.get("/api/v1/outputs/{video_id}/thumbnail")
async def get_thumbnail_file(video_id: str):
    """Serve thumbnail file"""
    db = SessionLocal()
    try:
        task = db.query(GenerationTaskDB).filter(GenerationTaskDB.id == video_id).first()
    finally:
        db.close()

    if not task:
        raise HTTPException(status_code=404, detail="Video not found")

    thumbnail_path = os.path.join("outputs", "thumbnails", f"{video_id}_thumb.jpg")

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
    db = SessionLocal()
    try:
        task = db.query(GenerationTaskDB).filter(GenerationTaskDB.id == video_id).first()
    finally:
        db.close()

    video_path = (
        task.output_path if task and task.output_path else os.path.join("outputs", f"{video_id}.mp4")
    )

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
    db = SessionLocal()
    try:
        task = db.query(GenerationTaskDB).filter(GenerationTaskDB.id == video_id).first()

        if not task:
            raise HTTPException(status_code=404, detail="Video not found")

        video_path = task.output_path or os.path.join("outputs", f"{video_id}.mp4")
        thumbnail_path = os.path.join("outputs", "thumbnails", f"{video_id}_thumb.jpg")

        if os.path.exists(video_path):
            os.remove(video_path)

        if os.path.exists(thumbnail_path):
            os.remove(thumbnail_path)

        if task.image_path and os.path.exists(task.image_path):
            try:
                os.remove(task.image_path)
            except Exception as exc:
                logger.warning(f"Could not remove input image for {video_id}: {exc}")

        db.delete(task)
        db.commit()

        return {"message": "Video deleted successfully"}
    finally:
        db.close()

# Simulate video generation completion (for testing)
@app.post("/api/v1/simulate/complete/{task_id}")
async def simulate_completion(task_id: str):
    """Simulate video generation completion for testing"""
    db = SessionLocal()
    try:
        task = db.query(GenerationTaskDB).filter(GenerationTaskDB.id == task_id).first()

        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        outputs_dir = "outputs"
        thumbnails_dir = os.path.join(outputs_dir, "thumbnails")
        os.makedirs(outputs_dir, exist_ok=True)
        os.makedirs(thumbnails_dir, exist_ok=True)

        video_path = os.path.join(outputs_dir, f"{task_id}.mp4")
        thumbnail_path = os.path.join(thumbnails_dir, f"{task_id}_thumb.jpg")

        with open(video_path, "wb") as f:
            f.write(b"dummy video content for testing" * 32)

        existing_thumb = os.path.join(
            "outputs",
            "thumbnails",
            "0846777d-d15a-40d5-ae78-ab1b13244649_thumb.jpg",
        )
        if os.path.exists(existing_thumb):
            import shutil

            shutil.copy2(existing_thumb, thumbnail_path)
        else:
            with open(thumbnail_path, "wb") as f:
                f.write(b"dummy thumbnail content" * 10)

        task.status = TaskStatusEnum.COMPLETED
        task.progress = 100
        task.completed_at = datetime.utcnow()
        task.output_path = video_path

        db.commit()

    finally:
        db.close()

    await manager.broadcast({
        "type": "task_update",
        "task_id": task_id,
        "status": "completed",
        "progress": 100,
        "message": "Video generation completed!",
        "output_path": f"outputs/{task_id}.mp4"
    })

    return {
        "message": "Video generation completed",
        "task_id": task_id,
        "video_url": f"/api/v1/outputs/{task_id}/video",
        "thumbnail_url": f"/api/v1/outputs/{task_id}/thumbnail"
    }

# LoRA Management Endpoints
@app.get("/api/v1/lora/list")
async def get_lora_list():
    """Get list of available LoRA files"""
    loras_dir = "loras"
    os.makedirs(loras_dir, exist_ok=True)
    
    loras = []
    
    # Scan for LoRA files
    for file_path in Path(loras_dir).glob("*"):
        if file_path.is_file() and file_path.suffix.lower() in ['.safetensors', '.pt', '.pth', '.bin']:
            file_size = file_path.stat().st_size / (1024 * 1024)  # Size in MB
            modified_time = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            
            loras.append({
                "name": file_path.stem,
                "filename": file_path.name,
                "path": str(file_path),
                "size_mb": round(file_size, 2),
                "modified_time": modified_time,
                "is_loaded": False,
                "is_applied": False,
                "current_strength": 0.0
            })
    
    # Sort by modification time (newest first)
    loras.sort(key=lambda x: x["modified_time"], reverse=True)
    
    return {
        "loras": loras,
        "total_count": len(loras),
        "total_size_mb": sum(lora["size_mb"] for lora in loras)
    }

@app.post("/api/v1/lora/upload")
async def upload_lora(file: UploadFile = File(...)):
    """Upload a new LoRA file"""
    # Validate file type
    allowed_extensions = ['.safetensors', '.pt', '.pth', '.bin']
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=422, 
            detail=f"Invalid file type. Only {', '.join(allowed_extensions)} files are supported"
        )
    
    # Check file size (500MB limit)
    content = await file.read()
    if len(content) > 500 * 1024 * 1024:
        raise HTTPException(status_code=422, detail="File too large. Maximum size is 500MB")
    
    # Save file
    loras_dir = "loras"
    os.makedirs(loras_dir, exist_ok=True)
    
    file_path = Path(loras_dir) / file.filename
    
    # Check if file already exists
    if file_path.exists():
        raise HTTPException(status_code=409, detail="File already exists")
    
    with open(file_path, "wb") as f:
        f.write(content)
    
    file_size_mb = len(content) / (1024 * 1024)
    
    return {
        "success": True,
        "message": "LoRA file uploaded successfully",
        "lora_name": file_path.stem,
        "file_path": str(file_path),
        "size_mb": round(file_size_mb, 2)
    }

@app.get("/api/v1/lora/{lora_name}/status")
async def get_lora_status(lora_name: str):
    """Get status of a specific LoRA file"""
    loras_dir = "loras"
    
    # Find the LoRA file
    lora_file = None
    for ext in ['.safetensors', '.pt', '.pth', '.bin']:
        potential_path = Path(loras_dir) / f"{lora_name}{ext}"
        if potential_path.exists():
            lora_file = potential_path
            break
    
    if not lora_file:
        return {
            "name": lora_name,
            "exists": False,
            "is_loaded": False,
            "is_applied": False,
            "current_strength": 0.0
        }
    
    file_size_mb = lora_file.stat().st_size / (1024 * 1024)
    modified_time = datetime.fromtimestamp(lora_file.stat().st_mtime).isoformat()
    
    return {
        "name": lora_name,
        "exists": True,
        "path": str(lora_file),
        "size_mb": round(file_size_mb, 2),
        "is_loaded": False,  # In a real implementation, check if loaded in memory
        "is_applied": False,  # In a real implementation, check if applied to current generation
        "current_strength": 0.0,
        "modified_time": modified_time
    }

@app.delete("/api/v1/lora/{lora_name}")
async def delete_lora(lora_name: str):
    """Delete a LoRA file"""
    loras_dir = "loras"
    
    # Find and delete the LoRA file
    deleted = False
    for ext in ['.safetensors', '.pt', '.pth', '.bin']:
        potential_path = Path(loras_dir) / f"{lora_name}{ext}"
        if potential_path.exists():
            potential_path.unlink()
            deleted = True
            break
    
    if not deleted:
        raise HTTPException(status_code=404, detail="LoRA file not found")
    
    return {"message": f"LoRA '{lora_name}' deleted successfully"}

@app.post("/api/v1/lora/{lora_name}/preview")
async def preview_lora_effect(lora_name: str, request: dict):
    """Preview the effect of applying a LoRA to a prompt"""
    base_prompt = request.get("prompt", "").strip()
    
    if not base_prompt:
        raise HTTPException(status_code=422, detail="Prompt is required")
    
    # Check if LoRA exists
    loras_dir = "loras"
    lora_exists = False
    for ext in ['.safetensors', '.pt', '.pth', '.bin']:
        if Path(loras_dir, f"{lora_name}{ext}").exists():
            lora_exists = True
            break
    
    if not lora_exists:
        raise HTTPException(status_code=404, detail="LoRA file not found")
    
    # Generate style indicators based on LoRA name (simplified)
    style_indicators = []
    lora_lower = lora_name.lower()
    
    if "anime" in lora_lower:
        style_indicators.extend(["anime style", "cel shading", "vibrant colors"])
    elif "realistic" in lora_lower or "photo" in lora_lower:
        style_indicators.extend(["photorealistic", "detailed textures", "natural lighting"])
    elif "art" in lora_lower or "paint" in lora_lower:
        style_indicators.extend(["artistic style", "painterly", "creative composition"])
    else:
        style_indicators.extend(["enhanced style", "improved quality"])
    
    # Create enhanced prompt
    enhanced_prompt = f"{base_prompt}, {', '.join(style_indicators[:2])}"
    
    return {
        "lora_name": lora_name,
        "base_prompt": base_prompt,
        "enhanced_prompt": enhanced_prompt,
        "style_indicators": style_indicators,
        "preview_note": f"This preview shows how '{lora_name}' might enhance your prompt"
    }

# Enhanced Model Management Endpoints
@app.get("/api/v1/models/status")
async def get_all_model_status():
    """Get status of all supported models using existing ModelManager"""
    try:
        from api.model_management import get_model_management_api
        api = await get_model_management_api()
        return await api.get_all_model_status()
        
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")

@app.get("/api/v1/models/status/{model_type}")
async def get_model_status(model_type: str):
    """Get status of a specific model using existing ModelManager"""
    try:
        from api.model_management import get_model_management_api
        api = await get_model_management_api()
        return await api.get_model_status(model_type)
        
    except Exception as e:
        logger.error(f"Error getting model status for {model_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")

@app.post("/api/v1/models/download")
async def download_model(request: dict):
    """Trigger model download using existing ModelDownloader"""
    try:
        model_type = request.get("model_type")
        force_redownload = request.get("force_redownload", False)
        
        if not model_type:
            raise HTTPException(status_code=422, detail="model_type is required")
        
        from api.model_management import get_model_management_api
        api = await get_model_management_api()
        return await api.trigger_model_download(model_type, force_redownload)
        
    except Exception as e:
        logger.error(f"Error starting model download: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start download: {str(e)}")

@app.get("/api/v1/models/download/progress")
async def get_all_download_progress():
    """Get download progress for all models"""
    try:
        from backend.core.model_integration_bridge import get_all_model_download_progress
        progress_dict = await get_all_model_download_progress()
        
        response = {}
        for model_type, progress in progress_dict.items():
            response[model_type] = {
                "model_type": model_type,
                "status": progress.get("status", "unknown"),
                "progress": progress.get("progress", 0.0),
                "speed_mbps": progress.get("speed_mbps", 0.0),
                "eta_seconds": progress.get("eta_seconds", 0.0),
                "error": progress.get("error"),
                "last_update": progress.get("last_update")
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting download progress: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get download progress: {str(e)}")

@app.get("/api/v1/models/download/progress/{model_type}")
async def get_model_download_progress_endpoint(model_type: str):
    """Get download progress for a specific model"""
    try:
        from backend.core.model_integration_bridge import get_model_download_progress
        progress = await get_model_download_progress(model_type)
        
        if progress is None:
            return None
        
        return {
            "model_type": model_type,
            "status": progress.get("status", "unknown"),
            "progress": progress.get("progress", 0.0),
            "speed_mbps": progress.get("speed_mbps", 0.0),
            "eta_seconds": progress.get("eta_seconds", 0.0),
            "error": progress.get("error"),
            "last_update": progress.get("last_update")
        }
        
    except Exception as e:
        logger.error(f"Error getting download progress for {model_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get download progress: {str(e)}")

@app.post("/api/v1/models/verify/{model_type}")
async def verify_model_integrity_endpoint(model_type: str):
    """Verify model integrity and attempt recovery if needed"""
    try:
        from backend.core.model_integration_bridge import verify_model_integrity
        is_valid = await verify_model_integrity(model_type)
        
        return {
            "model_type": model_type,
            "is_valid": is_valid,
            "details": f"Model {model_type} integrity check {'passed' if is_valid else 'failed'}"
        }
        
    except Exception as e:
        logger.error(f"Error verifying model integrity for {model_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to verify model integrity: {str(e)}")

@app.get("/api/v1/models/integration/status")
async def get_integration_status():
    """Get model integration system status"""
    try:
        from backend.core.model_integration_bridge import get_model_integration_bridge
        bridge = await get_model_integration_bridge()
        status = bridge.get_integration_status()
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting integration status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get integration status: {str(e)}")

async def _download_model_background(model_type: str):
    """Background task to download a model"""
    try:
        logger.info(f"Starting background download for model {model_type}")
        from backend.core.model_integration_bridge import ensure_model_ready
        success = await ensure_model_ready(model_type)
        
        if success:
            logger.info(f"Background download completed successfully for model {model_type}")
            # Broadcast success via WebSocket
            await manager.broadcast({
                "type": "model_download_complete",
                "model_type": model_type,
                "status": "success",
                "message": f"Model {model_type} downloaded successfully"
            })
        else:
            logger.error(f"Background download failed for model {model_type}")
            # Broadcast failure via WebSocket
            await manager.broadcast({
                "type": "model_download_complete",
                "model_type": model_type,
                "status": "failed",
                "message": f"Model {model_type} download failed"
            })
            
    except Exception as e:
        logger.error(f"Error in background download for model {model_type}: {e}")
        # Broadcast error via WebSocket
        await manager.broadcast({
            "type": "model_download_complete",
            "model_type": model_type,
            "status": "error",
            "message": f"Model {model_type} download error: {str(e)}"
        })

# Model validation endpoint
@app.post("/api/v1/models/validate/{model_type}")
async def validate_model_integrity_endpoint(model_type: str):
    """Validate model integrity using existing integrity checking"""
    try:
        from api.model_management import get_model_management_api
        api = await get_model_management_api()
        return await api.validate_model_integrity(model_type)
        
    except Exception as e:
        logger.error(f"Error validating model integrity for {model_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to validate model integrity: {str(e)}")

# System optimization status endpoint
@app.get("/api/v1/system/optimization/status")
async def get_system_optimization_status():
    """Get system optimization status using existing WAN22SystemOptimizer"""
    try:
        from api.model_management import get_model_management_api
        api = await get_model_management_api()
        return await api.get_system_optimization_status()
        
    except Exception as e:
        logger.error(f"Error getting system optimization status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system optimization status: {str(e)}")

# Hardware profile endpoint
@app.get("/api/v1/system/hardware/profile")
async def get_hardware_profile():
    """Get detected hardware profile information"""
    try:
        from backend.core.system_integration import get_system_integration
        integration = await get_system_integration()
        
        # Get hardware profile from WAN22SystemOptimizer
        optimizer = integration.get_wan22_system_optimizer()
        if optimizer:
            profile = optimizer.get_hardware_profile()
            if profile:
                return {
                    "cpu_model": profile.cpu_model,
                    "cpu_cores": profile.cpu_cores,
                    "total_memory_gb": profile.total_memory_gb,
                    "gpu_model": profile.gpu_model,
                    "vram_gb": profile.vram_gb,
                    "architecture_type": getattr(profile, 'architecture_type', 'unknown'),
                    "optimization_recommendations": optimizer.get_optimization_recommendations() if hasattr(optimizer, 'get_optimization_recommendations') else []
                }
        
        # Fallback hardware detection
        import platform
        import psutil
        
        hardware_info = {
            "cpu_model": platform.processor() or "Unknown",
            "cpu_cores": psutil.cpu_count(),
            "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "gpu_model": "Unknown",
            "vram_gb": 0.0,
            "architecture_type": platform.architecture()[0],
            "optimization_recommendations": []
        }
        
        # Try to get GPU info
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                hardware_info["gpu_model"] = torch.cuda.get_device_name(device)
                hardware_info["vram_gb"] = round(torch.cuda.get_device_properties(device).total_memory / (1024**3), 2)
        except ImportError:
            pass
        
        return hardware_info
        
    except Exception as e:
        logger.error(f"Error getting hardware profile: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get hardware profile: {str(e)}")

# Model management summary endpoint
@app.get("/api/v1/models/summary")
async def get_models_summary():
    """Get a summary of all models and system status"""
    try:
        from api.model_management import get_model_management_api
        api = await get_model_management_api()
        
        # Get all model status
        all_models = await api.get_all_model_status()
        
        # Get system optimization status
        system_status = await api.get_system_optimization_status()
        
        # Calculate summary statistics
        models = all_models.get("models", {})
        total_models = len(models)
        available_models = sum(1 for model in models.values() if model.get("is_available", False))
        loaded_models = sum(1 for model in models.values() if model.get("is_loaded", False))
        total_size_gb = sum(model.get("size_mb", 0) for model in models.values()) / 1024
        
        return {
            "models_summary": {
                "total_models": total_models,
                "available_models": available_models,
                "loaded_models": loaded_models,
                "missing_models": total_models - available_models,
                "total_size_gb": round(total_size_gb, 2)
            },
            "system_summary": {
                "integration_initialized": system_status.get("system_integration", {}).get("initialized", False),
                "optimization_applied": len(system_status.get("optimization_settings", {})) > 0,
                "hardware_detected": system_status.get("hardware_profile") is not None
            },
            "models": models,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting models summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get models summary: {str(e)}")

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and listen for client messages
            data = await websocket.receive_text()
            logger.info(f"Received WebSocket message: {data}")
            
            # Echo back for testing
            await manager.send_personal_message(f"Echo: {data}", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# Background task processor
async def process_generation_task(task_id: str):
    """Process a generation task with real-time updates"""
    db = SessionLocal()
    task: Optional[GenerationTaskDB] = None

    try:
        task = db.query(GenerationTaskDB).filter(GenerationTaskDB.id == task_id).first()

        if not task:
            logger.error(f"Task {task_id} not found in database")
            return

        task.status = TaskStatusEnum.PROCESSING
        task.started_at = datetime.utcnow()
        task.progress = 0
        task.error_message = None
        db.commit()
        db.refresh(task)

        await manager.broadcast({
            "type": "task_update",
            "task_id": task_id,
            "status": "processing",
            "progress": 0,
            "message": "Loading AI models...",
        })

        model_label = task.model_type.value if task.model_type else "unknown"
        logger.info(f"Starting fallback generation for task {task_id}")
        logger.info(f"Model: {model_label}")
        logger.info(f"Prompt: {task.prompt}")
        logger.info(f"Resolution: {task.resolution}")
        logger.info(f"Steps: {task.steps}")

        try:
            sys.path.append(str(Path(__file__).parent.parent))
            from utils import generate_video_from_prompt  # noqa: F401

            logger.info("🤖 Loading AI models (fallback simulation)...")
            await manager.broadcast({
                "type": "task_update",
                "task_id": task_id,
                "status": "processing",
                "progress": 10,
                "message": "AI models loaded, starting generation...",
            })

            logger.warning("⚠️  Real AI generation not yet integrated - using simulation")

        except ImportError as import_error:
            logger.warning(f"Real AI system not available: {import_error}")
            logger.info("🎭 Using simulation mode for demo")

        total_steps = task.steps or 50

        for step in range(1, total_steps + 1):
            await asyncio.sleep(0.5)

            progress = int(10 + (step / total_steps) * 85)
            task.progress = progress
            db.commit()
            db.refresh(task)

            if task.status == TaskStatusEnum.CANCELLED:
                logger.info(f"Task {task_id} was cancelled during processing")
                await manager.broadcast({
                    "type": "task_update",
                    "task_id": task_id,
                    "status": "cancelled",
                    "progress": task.progress,
                    "message": "Task cancelled by user",
                })
                return

            if step <= total_steps * 0.2:
                message = f"Initializing generation pipeline... ({step}/{total_steps})"
            elif step <= total_steps * 0.5:
                message = f"Processing prompt and generating frames... ({step}/{total_steps})"
            elif step <= total_steps * 0.8:
                message = f"Rendering video sequences... ({step}/{total_steps})"
            else:
                message = f"Finalizing video output... ({step}/{total_steps})"

            await manager.broadcast({
                "type": "task_update",
                "task_id": task_id,
                "status": "processing",
                "progress": progress,
                "message": message,
            })

        await manager.broadcast({
            "type": "task_update",
            "task_id": task_id,
            "status": "processing",
            "progress": 95,
            "message": "Post-processing and saving video...",
        })

        outputs_dir = "outputs"
        thumbnails_dir = os.path.join(outputs_dir, "thumbnails")
        os.makedirs(outputs_dir, exist_ok=True)
        os.makedirs(thumbnails_dir, exist_ok=True)

        video_path = os.path.join(outputs_dir, f"{task_id}.mp4")
        thumbnail_path = os.path.join(thumbnails_dir, f"{task_id}_thumb.jpg")

        with open(video_path, "wb") as file_handle:
            file_handle.write(b"dummy video content for demo" * 100)

        with open(thumbnail_path, "wb") as thumb_handle:
            thumb_handle.write(b"dummy thumbnail content" * 50)

        task.status = TaskStatusEnum.COMPLETED
        task.completed_at = datetime.utcnow()
        task.progress = 100
        task.output_path = video_path
        db.commit()

        await manager.broadcast({
            "type": "task_update",
            "task_id": task_id,
            "status": "completed",
            "progress": 100,
            "message": "Video generation completed successfully!",
            "output_path": f"outputs/{task_id}.mp4",
        })

        await manager.broadcast({"type": "queue_update", "message": "Queue updated"})

        logger.info(f"✅ Task {task_id} completed successfully")
        logger.info(f"📁 Output saved to: {video_path}")

    except Exception as error:
        logger.error(f"❌ Error processing task {task_id}: {error}")

        if db.is_active:
            db.rollback()

        if task:
            task.status = TaskStatusEnum.FAILED
            task.error_message = str(error)
            task.progress = 0
            db.commit()

            await manager.broadcast({
                "type": "task_update",
                "task_id": task_id,
                "status": "failed",
                "progress": 0,
                "message": f"Generation failed: {error}",
            })

    finally:
        db.close()

# Start background task processing
def start_background_task(task_id: str):
    """Start background task processing"""
    asyncio.create_task(process_generation_task(task_id))


# Add task cancellation endpoint
@app.post("/api/v1/queue/{task_id}/cancel")
async def cancel_task(task_id: str):
    """Cancel a pending or processing task"""
    db = SessionLocal()
    try:
        task = db.query(GenerationTaskDB).filter(GenerationTaskDB.id == task_id).first()

        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        if task.status in [
            TaskStatusEnum.COMPLETED,
            TaskStatusEnum.FAILED,
            TaskStatusEnum.CANCELLED,
        ]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel task with status: {task.status.value}",
            )

        task.status = TaskStatusEnum.CANCELLED
        task.progress = 0
        task.completed_at = datetime.utcnow()
        db.commit()

    finally:
        db.close()

    await manager.broadcast({
        "type": "task_update",
        "task_id": task_id,
        "status": "cancelled",
        "progress": 0,
        "message": "Task cancelled by user",
    })

    return {"message": "Task cancelled successfully", "task_id": task_id}

# Enhanced Model Management Endpoints

@app.get("/api/v1/models/status/detailed")
async def get_detailed_model_status():
    """Get comprehensive model status with enhanced information"""
    try:
        from api.enhanced_model_management import get_enhanced_model_management_api
        api = await get_enhanced_model_management_api()
        return await api.get_detailed_model_status()
    except Exception as e:
        logger.error(f"Error getting detailed model status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get detailed model status: {str(e)}")

@app.post("/api/v1/models/download/manage")
async def manage_model_download(request: dict):
    """Manage download operations (pause, resume, cancel, priority)"""
    try:
        from api.enhanced_model_management import get_enhanced_model_management_api, DownloadControlRequest
        api = await get_enhanced_model_management_api()
        
        # Validate request
        control_request = DownloadControlRequest(**request)
        return await api.manage_download(control_request)
    except Exception as e:
        logger.error(f"Error managing model download: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to manage download: {str(e)}")

@app.get("/api/v1/models/health")
async def get_model_health_monitoring():
    """Get comprehensive health monitoring data for all models"""
    try:
        from api.enhanced_model_management import get_enhanced_model_management_api
        api = await get_enhanced_model_management_api()
        return await api.get_health_monitoring_data()
    except Exception as e:
        logger.error(f"Error getting model health data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get health data: {str(e)}")

@app.get("/api/v1/models/analytics")
async def get_model_usage_analytics(time_period_days: int = 30):
    """Get usage analytics and statistics for all models"""
    try:
        from api.enhanced_model_management import get_enhanced_model_management_api
        api = await get_enhanced_model_management_api()
        return await api.get_usage_analytics(time_period_days)
    except Exception as e:
        logger.error(f"Error getting model analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")

@app.post("/api/v1/models/cleanup")
async def manage_storage_cleanup(request: dict):
    """Manage storage cleanup operations"""
    try:
        from api.enhanced_model_management import get_enhanced_model_management_api, CleanupRequest
        api = await get_enhanced_model_management_api()
        
        # Validate request
        cleanup_request = CleanupRequest(**request)
        return await api.manage_storage_cleanup(cleanup_request)
    except Exception as e:
        logger.error(f"Error managing storage cleanup: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to manage cleanup: {str(e)}")

@app.post("/api/v1/models/fallback/suggest")
async def suggest_fallback_alternatives(request: dict):
    """Suggest alternative models and fallback strategies"""
    try:
        from api.enhanced_model_management import get_enhanced_model_management_api, FallbackSuggestionRequest
        api = await get_enhanced_model_management_api()
        
        # Validate request
        suggestion_request = FallbackSuggestionRequest(**request)
        return await api.suggest_fallback_alternatives(suggestion_request)
    except Exception as e:
        logger.error(f"Error suggesting fallback alternatives: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to suggest alternatives: {str(e)}")

# Fallback and Recovery System Endpoints

@app.get("/api/v1/recovery/status")
async def get_recovery_system_status():
    """Get current status of the fallback and recovery system"""
    try:
        recovery_system = get_fallback_recovery_system()
        
        # Get recovery statistics
        stats = recovery_system.get_recovery_statistics()
        
        # Get current health status
        health_status = None
        if recovery_system.current_health_status:
            health = recovery_system.current_health_status
            health_status = {
                "overall_status": health.overall_status,
                "cpu_usage_percent": health.cpu_usage_percent,
                "memory_usage_percent": health.memory_usage_percent,
                "vram_usage_percent": health.vram_usage_percent,
                "gpu_available": health.gpu_available,
                "model_loading_functional": health.model_loading_functional,
                "generation_pipeline_functional": health.generation_pipeline_functional,
                "issues": health.issues,
                "recommendations": health.recommendations,
                "last_check": health.last_check_timestamp.isoformat()
            }
        
        return {
            "recovery_system_active": True,
            "health_monitoring_active": recovery_system.health_monitoring_active,
            "mock_generation_enabled": recovery_system.mock_generation_enabled,
            "degraded_mode_active": recovery_system.degraded_mode_active,
            "recovery_statistics": stats,
            "current_health_status": health_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting recovery system status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recovery system status: {str(e)}")

@app.get("/api/v1/recovery/health")
async def get_system_health():
    """Get comprehensive system health status"""
    try:
        recovery_system = get_fallback_recovery_system()
        health_status = await recovery_system.get_system_health_status()
        
        return {
            "overall_status": health_status.overall_status,
            "cpu_usage_percent": health_status.cpu_usage_percent,
            "memory_usage_percent": health_status.memory_usage_percent,
            "vram_usage_percent": health_status.vram_usage_percent,
            "gpu_available": health_status.gpu_available,
            "model_loading_functional": health_status.model_loading_functional,
            "generation_pipeline_functional": health_status.generation_pipeline_functional,
            "issues": health_status.issues,
            "recommendations": health_status.recommendations,
            "last_check_timestamp": health_status.last_check_timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system health: {str(e)}")

@app.post("/api/v1/recovery/trigger")
async def trigger_manual_recovery(request: dict):
    """Manually trigger recovery for a specific failure type"""
    try:
        failure_type = request.get("failure_type")
        context = request.get("context", {})
        
        if not failure_type:
            raise HTTPException(status_code=400, detail="failure_type is required")
        
        # Validate failure type
        try:
            failure_enum = FailureType(failure_type)
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid failure type: {failure_type}. Valid types: {[ft.value for ft in FailureType]}"
            )
        
        recovery_system = get_fallback_recovery_system()
        
        # Create a mock exception for manual recovery
        mock_error = Exception(f"Manual recovery triggered for {failure_type}")
        
        # Attempt recovery
        success, message = await recovery_system.handle_failure(
            failure_enum, 
            mock_error, 
            {**context, "manual_trigger": True, "timestamp": datetime.now().isoformat()}
        )
        
        return {
            "recovery_triggered": True,
            "failure_type": failure_type,
            "success": success,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering manual recovery: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger recovery: {str(e)}")

@app.post("/api/v1/recovery/reset")
async def reset_recovery_state():
    """Reset recovery state and re-enable real generation"""
    try:
        recovery_system = get_fallback_recovery_system()
        recovery_system.reset_recovery_state()
        
        return {
            "recovery_state_reset": True,
            "mock_generation_disabled": True,
            "real_generation_enabled": True,
            "message": "Recovery state has been reset and real generation re-enabled",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error resetting recovery state: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset recovery state: {str(e)}")

@app.get("/api/v1/recovery/actions")
async def get_available_recovery_actions():
    """Get list of available recovery actions and failure types"""
    try:
        return {
            "failure_types": [ft.value for ft in FailureType],
            "recovery_actions": [ra.value for ra in RecoveryAction],
            "failure_type_descriptions": {
                FailureType.MODEL_LOADING_FAILURE.value: "Issues with loading AI models",
                FailureType.VRAM_EXHAUSTION.value: "GPU memory exhaustion errors",
                FailureType.GENERATION_PIPELINE_ERROR.value: "Errors in the generation pipeline",
                FailureType.HARDWARE_OPTIMIZATION_FAILURE.value: "Hardware optimization failures",
                FailureType.SYSTEM_RESOURCE_ERROR.value: "System resource issues",
                FailureType.NETWORK_ERROR.value: "Network connectivity problems"
            },
            "recovery_action_descriptions": {
                RecoveryAction.FALLBACK_TO_MOCK.value: "Switch to mock generation mode",
                RecoveryAction.RETRY_MODEL_DOWNLOAD.value: "Retry downloading models",
                RecoveryAction.APPLY_VRAM_OPTIMIZATION.value: "Apply VRAM optimization settings",
                RecoveryAction.RESTART_PIPELINE.value: "Restart the generation pipeline",
                RecoveryAction.CLEAR_GPU_CACHE.value: "Clear GPU memory cache",
                RecoveryAction.REDUCE_GENERATION_PARAMS.value: "Reduce generation parameters",
                RecoveryAction.ENABLE_CPU_OFFLOAD.value: "Enable CPU offloading",
                RecoveryAction.SYSTEM_HEALTH_CHECK.value: "Perform system health check"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting available recovery actions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recovery actions: {str(e)}")

if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
