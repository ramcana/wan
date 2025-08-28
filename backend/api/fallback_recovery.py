"""
API endpoints for fallback and recovery system management
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

from backend.core.fallback_recovery_system import (
    get_fallback_recovery_system, FailureType, RecoveryAction, FallbackRecoverySystem
)
from backend.core.system_integration import get_system_integration, SystemIntegration

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/recovery", tags=["recovery"])

@router.get("/status")
async def get_recovery_system_status(
    recovery_system: FallbackRecoverySystem = Depends(get_fallback_recovery_system)
) -> Dict[str, Any]:
    """Get current status of the fallback and recovery system"""
    try:
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

@router.get("/health")
async def get_system_health(
    recovery_system: FallbackRecoverySystem = Depends(get_fallback_recovery_system)
) -> Dict[str, Any]:
    """Get comprehensive system health status"""
    try:
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

@router.get("/recovery-attempts")
async def get_recovery_attempts(
    limit: int = 50,
    recovery_system: FallbackRecoverySystem = Depends(get_fallback_recovery_system)
) -> Dict[str, Any]:
    """Get recent recovery attempts"""
    try:
        # Get recent recovery attempts (limited)
        recent_attempts = recovery_system.recovery_attempts[-limit:] if recovery_system.recovery_attempts else []
        
        attempts_data = []
        for attempt in recent_attempts:
            attempts_data.append({
                "failure_type": attempt.failure_type.value,
                "action": attempt.action.value,
                "timestamp": attempt.timestamp.isoformat(),
                "success": attempt.success,
                "recovery_time_seconds": attempt.recovery_time_seconds,
                "error_message": attempt.error_message,
                "context": attempt.context
            })
        
        return {
            "recovery_attempts": attempts_data,
            "total_attempts": len(recovery_system.recovery_attempts),
            "showing_recent": len(attempts_data),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting recovery attempts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recovery attempts: {str(e)}")

@router.post("/trigger-recovery")
async def trigger_manual_recovery(
    failure_type: str,
    context: Optional[Dict[str, Any]] = None,
    recovery_system: FallbackRecoverySystem = Depends(get_fallback_recovery_system)
) -> Dict[str, Any]:
    """Manually trigger recovery for a specific failure type"""
    try:
        # Validate failure type
        try:
            failure_enum = FailureType(failure_type)
        except ValueError:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid failure type: {failure_type}. Valid types: {[ft.value for ft in FailureType]}"
            )
        
        # Create a mock exception for manual recovery
        mock_error = Exception(f"Manual recovery triggered for {failure_type}")
        
        # Attempt recovery
        success, message = await recovery_system.handle_failure(
            failure_enum, 
            mock_error, 
            context or {"manual_trigger": True, "timestamp": datetime.now().isoformat()}
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

@router.post("/reset-recovery-state")
async def reset_recovery_state(
    recovery_system: FallbackRecoverySystem = Depends(get_fallback_recovery_system)
) -> Dict[str, Any]:
    """Reset recovery state and re-enable real generation"""
    try:
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

@router.post("/enable-mock-generation")
async def enable_mock_generation(
    recovery_system: FallbackRecoverySystem = Depends(get_fallback_recovery_system)
) -> Dict[str, Any]:
    """Manually enable mock generation mode"""
    try:
        # Trigger fallback to mock generation
        success = await recovery_system._fallback_to_mock_generation()
        
        return {
            "mock_generation_enabled": success,
            "message": "Mock generation mode enabled" if success else "Failed to enable mock generation",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error enabling mock generation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to enable mock generation: {str(e)}")

@router.get("/available-actions")
async def get_available_recovery_actions() -> Dict[str, Any]:
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

@router.get("/health-monitoring/start")
async def start_health_monitoring(
    recovery_system: FallbackRecoverySystem = Depends(get_fallback_recovery_system)
) -> Dict[str, Any]:
    """Start continuous health monitoring"""
    try:
        if recovery_system.health_monitoring_active:
            return {
                "health_monitoring_started": False,
                "message": "Health monitoring is already active",
                "timestamp": datetime.now().isoformat()
            }
        
        recovery_system.start_health_monitoring()
        
        return {
            "health_monitoring_started": True,
            "message": "Health monitoring started successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error starting health monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start health monitoring: {str(e)}")

@router.get("/health-monitoring/stop")
async def stop_health_monitoring(
    recovery_system: FallbackRecoverySystem = Depends(get_fallback_recovery_system)
) -> Dict[str, Any]:
    """Stop continuous health monitoring"""
    try:
        if not recovery_system.health_monitoring_active:
            return {
                "health_monitoring_stopped": False,
                "message": "Health monitoring is not active",
                "timestamp": datetime.now().isoformat()
            }
        
        recovery_system.stop_health_monitoring()
        
        return {
            "health_monitoring_stopped": True,
            "message": "Health monitoring stopped successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error stopping health monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop health monitoring: {str(e)}")

@router.get("/system-integration-health")
async def get_system_integration_health(
    integration: SystemIntegration = Depends(get_system_integration)
) -> Dict[str, Any]:
    """Get system integration health with recovery context"""
    try:
        health_info = await integration.get_system_health_with_recovery_context()
        return health_info
        
    except Exception as e:
        logger.error(f"Error getting system integration health: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system integration health: {str(e)}")