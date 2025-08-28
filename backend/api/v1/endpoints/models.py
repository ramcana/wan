"""
Model management endpoints
Provides model status, download progress, and integrity verification
"""

import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from backend.core.model_integration_bridge import (
    get_model_integration_bridge,
    check_model_availability,
    ensure_model_ready,
    get_model_download_progress,
    get_all_model_download_progress,
    verify_model_integrity,
    ModelIntegrationStatus
)

logger = logging.getLogger(__name__)

router = APIRouter()

class ModelStatusResponse(BaseModel):
    """Response model for model status"""
    model_id: str
    model_type: str
    status: str
    is_cached: bool
    is_loaded: bool
    is_valid: bool
    size_mb: float
    download_progress: Optional[float] = None
    optimization_applied: bool = False
    hardware_compatible: bool = True
    estimated_vram_usage_mb: float = 0.0
    error_message: Optional[str] = None
    download_speed_mbps: Optional[float] = None
    download_eta_seconds: Optional[float] = None
    integrity_verified: bool = False

class ModelDownloadRequest(BaseModel):
    """Request model for model download"""
    model_type: str
    force_redownload: bool = False

class ModelDownloadProgressResponse(BaseModel):
    """Response model for download progress"""
    model_type: str
    status: str
    progress: float
    speed_mbps: float = 0.0
    eta_seconds: float = 0.0
    error: Optional[str] = None
    last_update: Optional[float] = None

class ModelIntegrityResponse(BaseModel):
    """Response model for model integrity check"""
    model_type: str
    is_valid: bool
    issues_found: int = 0
    recovery_attempted: bool = False
    recovery_successful: bool = False
    details: str = ""

@router.get("/status", response_model=Dict[str, ModelStatusResponse])
async def get_all_model_status():
    """Get status of all supported models"""
    try:
        bridge = await get_model_integration_bridge()
        status_dict = bridge.get_model_status_from_existing_system()
        
        response = {}
        for model_type, status in status_dict.items():
            response[model_type] = ModelStatusResponse(
                model_id=status.model_id,
                model_type=status.model_type.value,
                status=status.status.value,
                is_cached=status.is_cached,
                is_loaded=status.is_loaded,
                is_valid=status.is_valid,
                size_mb=status.size_mb,
                download_progress=status.download_progress,
                optimization_applied=status.optimization_applied,
                hardware_compatible=status.hardware_compatible,
                estimated_vram_usage_mb=status.estimated_vram_usage_mb,
                error_message=status.error_message,
                download_speed_mbps=status.download_speed_mbps,
                download_eta_seconds=status.download_eta_seconds,
                integrity_verified=status.integrity_verified
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")

@router.get("/status/{model_type}", response_model=ModelStatusResponse)
async def get_model_status(model_type: str):
    """Get status of a specific model"""
    try:
        status = await check_model_availability(model_type)
        
        return ModelStatusResponse(
            model_id=status.model_id,
            model_type=status.model_type.value,
            status=status.status.value,
            is_cached=status.is_cached,
            is_loaded=status.is_loaded,
            is_valid=status.is_valid,
            size_mb=status.size_mb,
            download_progress=status.download_progress,
            optimization_applied=status.optimization_applied,
            hardware_compatible=status.hardware_compatible,
            estimated_vram_usage_mb=status.estimated_vram_usage_mb,
            error_message=status.error_message,
            download_speed_mbps=status.download_speed_mbps,
            download_eta_seconds=status.download_eta_seconds,
            integrity_verified=status.integrity_verified
        )
        
    except Exception as e:
        logger.error(f"Error getting model status for {model_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")

@router.post("/download", response_model=Dict[str, Any])
async def download_model(request: ModelDownloadRequest, background_tasks: BackgroundTasks):
    """Trigger model download"""
    try:
        model_type = request.model_type
        
        # Check if model is already available
        if not request.force_redownload:
            status = await check_model_availability(model_type)
            if status.status.value in ["available", "loaded"]:
                return {
                    "message": f"Model {model_type} is already available",
                    "model_type": model_type,
                    "status": "already_available"
                }
        
        # Start download in background
        background_tasks.add_task(_download_model_background, model_type)
        
        return {
            "message": f"Download started for model {model_type}",
            "model_type": model_type,
            "status": "download_started"
        }
        
    except Exception as e:
        logger.error(f"Error starting model download for {request.model_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start download: {str(e)}")

@router.get("/download/progress", response_model=Dict[str, ModelDownloadProgressResponse])
async def get_all_download_progress():
    """Get download progress for all models"""
    try:
        progress_dict = await get_all_model_download_progress()
        
        response = {}
        for model_type, progress in progress_dict.items():
            response[model_type] = ModelDownloadProgressResponse(
                model_type=model_type,
                status=progress.get("status", "unknown"),
                progress=progress.get("progress", 0.0),
                speed_mbps=progress.get("speed_mbps", 0.0),
                eta_seconds=progress.get("eta_seconds", 0.0),
                error=progress.get("error"),
                last_update=progress.get("last_update")
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting download progress: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get download progress: {str(e)}")

@router.get("/download/progress/{model_type}", response_model=Optional[ModelDownloadProgressResponse])
async def get_model_download_progress_endpoint(model_type: str):
    """Get download progress for a specific model"""
    try:
        progress = await get_model_download_progress(model_type)
        
        if progress is None:
            return None
        
        return ModelDownloadProgressResponse(
            model_type=model_type,
            status=progress.get("status", "unknown"),
            progress=progress.get("progress", 0.0),
            speed_mbps=progress.get("speed_mbps", 0.0),
            eta_seconds=progress.get("eta_seconds", 0.0),
            error=progress.get("error"),
            last_update=progress.get("last_update")
        )
        
    except Exception as e:
        logger.error(f"Error getting download progress for {model_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get download progress: {str(e)}")

@router.post("/verify/{model_type}", response_model=ModelIntegrityResponse)
async def verify_model_integrity_endpoint(model_type: str):
    """Verify model integrity and attempt recovery if needed"""
    try:
        is_valid = await verify_model_integrity(model_type)
        
        return ModelIntegrityResponse(
            model_type=model_type,
            is_valid=is_valid,
            details=f"Model {model_type} integrity check {'passed' if is_valid else 'failed'}"
        )
        
    except Exception as e:
        logger.error(f"Error verifying model integrity for {model_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to verify model integrity: {str(e)}")

@router.get("/integration/status", response_model=Dict[str, Any])
async def get_integration_status():
    """Get model integration system status"""
    try:
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
        success = await ensure_model_ready(model_type)
        
        if success:
            logger.info(f"Background download completed successfully for model {model_type}")
        else:
            logger.error(f"Background download failed for model {model_type}")
            
    except Exception as e:
        logger.error(f"Error in background download for model {model_type}: {e}")