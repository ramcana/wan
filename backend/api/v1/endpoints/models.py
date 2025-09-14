"""
Model management endpoints
Provides model status, download progress, and integrity verification
"""

import logging
import time
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Response
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
from backend.services.model_health_service import get_model_health_service
from backend.core.model_orchestrator.metrics import get_metrics_collector
from backend.core.model_orchestrator.logging_config import set_correlation_id, generate_correlation_id

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

class ModelHealthInfo(BaseModel):
    """Enhanced health information for a single model"""
    model_id: str
    variant: Optional[str] = None
    status: str
    local_path: Optional[str] = None
    missing_files: List[str] = []
    bytes_needed: int = 0
    last_verified: Optional[float] = None
    error_message: Optional[str] = None
    gpu_health: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    component_details: Optional[Dict[str, Any]] = None

class OrchestratorHealthResponse(BaseModel):
    """Enhanced health response for the model orchestrator"""
    status: str
    timestamp: float
    models: Dict[str, ModelHealthInfo]
    total_models: int
    healthy_models: int
    missing_models: int
    partial_models: int
    corrupt_models: int
    total_bytes_needed: int
    response_time_ms: float
    correlation_id: str
    system_metrics: Optional[Dict[str, Any]] = None
    gpu_system_health: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

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

@router.get("/health", response_model=OrchestratorHealthResponse)
async def get_models_health(
    dry_run: bool = Query(True, description="Prevent any side effects, only check status"),
    include_gpu_checks: bool = Query(False, description="Include GPU-based health checks"),
    include_detailed_diagnostics: bool = Query(False, description="Include detailed component diagnostics")
):
    """
    Get comprehensive health status of all models in the orchestrator.
    
    This endpoint provides detailed model availability and health information with
    optional GPU validation and performance metrics.
    
    Args:
        dry_run: If True (default), only check status without triggering downloads
        include_gpu_checks: Include GPU-based smoke tests for model validation
        include_detailed_diagnostics: Include detailed component and file diagnostics
        
    Returns:
        Enhanced OrchestratorHealthResponse with observability metrics
    """
    # Set correlation ID for request tracking
    correlation_id = generate_correlation_id()
    set_correlation_id(correlation_id)
    
    try:
        health_service = get_model_health_service()
        if not health_service:
            raise HTTPException(
                status_code=503, 
                detail="Model health service not initialized"
            )
        
        health_response = await health_service.get_health_status(
            dry_run=dry_run,
            include_gpu_checks=include_gpu_checks,
            include_detailed_diagnostics=include_detailed_diagnostics
        )
        
        # Convert to Pydantic models for proper serialization
        models_dict = {}
        for model_id, health_info in health_response.models.items():
            models_dict[model_id] = ModelHealthInfo(
                model_id=health_info.model_id,
                variant=health_info.variant,
                status=health_info.status,
                local_path=health_info.local_path,
                missing_files=health_info.missing_files,
                bytes_needed=health_info.bytes_needed,
                last_verified=health_info.last_verified,
                error_message=health_info.error_message,
                gpu_health=health_info.gpu_health,
                performance_metrics=health_info.performance_metrics,
                component_details=health_info.component_details
            )
        
        return OrchestratorHealthResponse(
            status=health_response.status,
            timestamp=health_response.timestamp,
            models=models_dict,
            total_models=health_response.total_models,
            healthy_models=health_response.healthy_models,
            missing_models=health_response.missing_models,
            partial_models=health_response.partial_models,
            corrupt_models=health_response.corrupt_models,
            total_bytes_needed=health_response.total_bytes_needed,
            response_time_ms=health_response.response_time_ms,
            correlation_id=health_response.correlation_id,
            system_metrics=health_response.system_metrics,
            gpu_system_health=health_response.gpu_system_health,
            error_message=health_response.error_message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting models health: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get models health: {str(e)}"
        )

@router.get("/health/{model_id}", response_model=ModelHealthInfo)
async def get_model_health(
    model_id: str,
    variant: Optional[str] = Query(None, description="Model variant (e.g., fp16, bf16)"),
    include_gpu_check: bool = Query(False, description="Include GPU-based health check"),
    include_detailed_diagnostics: bool = Query(False, description="Include detailed component diagnostics")
):
    """
    Get comprehensive health status of a specific model.
    
    Args:
        model_id: The model identifier
        variant: Optional model variant
        include_gpu_check: Include GPU-based smoke test validation
        include_detailed_diagnostics: Include detailed component and file diagnostics
        
    Returns:
        Enhanced ModelHealthInfo with detailed status for the specified model
    """
    # Set correlation ID for request tracking
    correlation_id = generate_correlation_id()
    set_correlation_id(correlation_id)
    
    try:
        health_service = get_model_health_service()
        if not health_service:
            raise HTTPException(
                status_code=503, 
                detail="Model health service not initialized"
            )
        
        health_info = await health_service.get_model_health(
            model_id, 
            variant,
            include_gpu_check=include_gpu_check,
            include_detailed_diagnostics=include_detailed_diagnostics
        )
        
        return ModelHealthInfo(
            model_id=health_info.model_id,
            variant=health_info.variant,
            status=health_info.status,
            local_path=health_info.local_path,
            missing_files=health_info.missing_files,
            bytes_needed=health_info.bytes_needed,
            last_verified=health_info.last_verified,
            error_message=health_info.error_message,
            gpu_health=health_info.gpu_health,
            performance_metrics=health_info.performance_metrics,
            component_details=health_info.component_details
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting health for model {model_id}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get model health: {str(e)}"
        )

@router.get("/metrics", response_model=Dict[str, Any])
async def get_orchestrator_metrics():
    """
    Get comprehensive metrics for the Model Orchestrator.
    
    Returns Prometheus-compatible metrics including download statistics,
    error rates, storage usage, and performance data.
    """
    try:
        metrics_collector = get_metrics_collector()
        health_service = get_model_health_service()
        
        response = {
            "prometheus_metrics": metrics_collector.get_metrics_dict(),
            "timestamp": time.time()
        }
        
        if health_service:
            response["health_service_metrics"] = await health_service.get_metrics_summary()
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting orchestrator metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get metrics: {str(e)}"
        )

@router.get("/metrics/prometheus")
async def get_prometheus_metrics(response: Response):
    """
    Get metrics in Prometheus text format.
    
    Returns metrics in the standard Prometheus exposition format
    for integration with monitoring systems.
    """
    try:
        metrics_collector = get_metrics_collector()
        metrics_text = metrics_collector.get_metrics_text()
        
        response.headers["Content-Type"] = "text/plain; version=0.0.4; charset=utf-8"
        return metrics_text
        
    except Exception as e:
        logger.error(f"Error getting Prometheus metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get Prometheus metrics: {str(e)}"
        )

@router.get("/diagnostics/{model_id}", response_model=Dict[str, Any])
async def get_model_diagnostics(
    model_id: str,
    variant: Optional[str] = Query(None, description="Model variant")
):
    """
    Get detailed diagnostics for a specific model.
    
    Provides comprehensive information including file system details,
    component validation, and performance history.
    
    Args:
        model_id: The model identifier
        variant: Optional model variant
        
    Returns:
        Detailed diagnostic information
    """
    # Set correlation ID for request tracking
    correlation_id = generate_correlation_id()
    set_correlation_id(correlation_id)
    
    try:
        health_service = get_model_health_service()
        if not health_service:
            raise HTTPException(
                status_code=503,
                detail="Model health service not initialized"
            )
        
        diagnostics = await health_service.get_detailed_model_diagnostics(model_id, variant)
        diagnostics["correlation_id"] = correlation_id
        
        return diagnostics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting diagnostics for model {model_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model diagnostics: {str(e)}"
        )

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
