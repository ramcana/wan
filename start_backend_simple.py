#!/usr/bin/env python3
"""
Simple backend startup script that bypasses database initialization
"""

import os
import sys
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create simple FastAPI app
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
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "wan22-backend"}

@app.get("/api/health")
async def api_health_check():
    """API health check endpoint"""
    return {"status": "healthy", "api_version": "2.2.0"}

@app.get("/api/v1/system/health")
async def system_health_check():
    """System health check endpoint"""
    return {
        "status": "ok",
        "port": 9000,
        "api_version": "2.2.0",
        "system": "operational",
        "service": "wan22-backend-simple"
    }

# Simple prompt enhancement endpoint
@app.post("/api/v1/prompt/enhance")
async def enhance_prompt(request: dict):
    """Simple prompt enhancement"""
    prompt = request.get("prompt", "")
    if isinstance(prompt, dict):
        prompt = prompt.get("prompt", "")
    
    enhanced_prompt = f"{prompt}, high quality, detailed"
    
    return {
        "original_prompt": prompt,
        "enhanced_prompt": enhanced_prompt,
        "enhancements_applied": ["Quality enhancement"],
        "character_count": {
            "original": len(prompt),
            "enhanced": len(enhanced_prompt)
        }
    }

# Simple generation endpoint
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
    """Simple generation endpoint (mock) - handles both JSON and FormData"""
    
    # Handle both JSON and form data requests like the full backend
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
        except Exception as e:
            logger.error(f"Error parsing JSON request: {e}")
            return {"error": "Invalid JSON request"}
    
    logger.info(f"Generation request - Prompt: '{prompt}', Model: {model_type}")
    
    # Mock generation response
    return {
        "task_id": "mock_task_123",
        "status": "queued",
        "prompt": prompt,
        "model_type": model_type,
        "resolution": resolution,
        "steps": steps,
        "estimated_time": 30,
        "queue_position": 1,
        "message": "Generation request received (mock mode)"
    }

# Simple queue endpoint
@app.get("/api/v1/queue")
async def get_queue():
    """Get generation queue (empty for now)"""
    return {
        "tasks": [],
        "total_tasks": 0,
        "pending_tasks": 0,
        "processing_tasks": 0,
        "completed_tasks": 0,
        "failed_tasks": 0
    }

if __name__ == "__main__":
    logger.info("Starting simple WAN22 backend...")
    uvicorn.run(
        "start_backend_simple:app",
        host="0.0.0.0",
        port=9000,
        reload=True
    )
