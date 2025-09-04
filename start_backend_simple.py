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

from fastapi import FastAPI
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