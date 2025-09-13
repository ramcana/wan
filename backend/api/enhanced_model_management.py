"""
Enhanced Model Management API Endpoints
Provides comprehensive model status, download management, health monitoring,
analytics, cleanup, and fallback suggestion endpoints.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import HTTPException
from pydantic import BaseModel, Field

# Import enhanced model components
from backend.core.model_availability_manager import (
    ModelAvailabilityManager, ModelAvailabilityStatus, ModelPriority
)
from backend.core.enhanced_model_downloader import (
    EnhancedModelDownloader, DownloadStatus, DownloadProgress
)
from backend.core.model_health_monitor import (
    ModelHealthMonitor, HealthStatus, IntegrityResult
)
from backend.core.intelligent_fallback_manager import (
    IntelligentFallbackManager, FallbackType, GenerationRequirements
)
from backend.core.model_usage_analytics import (
    ModelUsageAnalytics, UsageStatistics, CleanupRecommendations
)

logger = logging.getLogger(__name__)


# Pydantic models for request/response validation
class DownloadControlRequest(BaseModel):
    """Request model for download control operations"""
    model_id: str = Field(..., description="Model identifier")
    action: str = Field(..., description="Action: pause, resume, cancel, priority")
    priority: Optional[str] = Field(None, description="Priority level for priority action")
    bandwidth_limit_mbps: Optional[float] = Field(None, description="Bandwidth limit in Mbps")


class CleanupRequest(BaseModel):
    """Request model for storage cleanup operations"""
    target_space_gb: Optional[float] = Field(None, description="Target free space in GB")
    keep_recent_days: Optional[int] = Field(30, description="Keep models used in last N days")
    dry_run: bool = Field(True, description="Preview cleanup without executing")


class FallbackSuggestionRequest(BaseModel):
    """Request model for fallback suggestions"""
    requested_model: str = Field(..., description="Originally requested model")
    quality: str = Field("medium", description="Quality requirement: low, medium, high")
    speed: str = Field("medium", description="Speed requirement: fast, medium, slow")
    resolution: str = Field("1280x720", description="Target resolution")
    max_wait_minutes: Optional[int] = Field(None, description="Maximum acceptable wait time")


class EnhancedModelManagementAPI:
    """Enhanced model management API implementation"""
    
    def __init__(self):
        self.availability_manager: Optional[ModelAvailabilityManager] = None
        self.enhanced_downloader: Optional[EnhancedModelDownloader] = None
        self.health_monitor: Optional[ModelHealthMonitor] = None
        self.fallback_manager: Optional[IntelligentFallbackManager] = None
        self.usage_analytics: Optional[ModelUsageAnalytics] = None
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize all enhanced model management components"""
        try:
            # Initialize components in dependency order
            self.enhanced_downloader = EnhancedModelDownloader()
            await self.enhanced_downloader.initialize()
            
            self.health_monitor = ModelHealthMonitor()
            await self.health_monitor.initialize()
            
            self.usage_analytics = ModelUsageAnalytics()
            await self.usage_analytics.initialize()
            
            self.availability_manager = ModelAvailabilityManager(
                downloader=self.enhanced_downloader,
                health_monitor=self.health_monitor
            )
            await self.availability_manager.initialize()
            
            self.fallback_manager = IntelligentFallbackManager(
                availability_manager=self.availability_manager
            )
            await self.fallback_manager.initialize()
            
            self._initialized = True
            logger.info("Enhanced Model Management API initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced Model Management API: {e}")
            return False
    
    async def get_detailed_model_status(self) -> Dict[str, Any]:
        """Get comprehensive model status with enhanced information"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get comprehensive status from availability manager
            detailed_status = await self.availability_manager.get_comprehensive_model_status()
            
            # Add system-wide statistics
            system_stats = {
                "total_models": len(detailed_status),
                "available_models": sum(1 for status in detailed_status.values() 
                                      if status.availability_status == ModelAvailabilityStatus.AVAILABLE),
                "downloading_models": sum(1 for status in detailed_status.values() 
                                        if status.availability_status == ModelAvailabilityStatus.DOWNLOADING),
                "corrupted_models": sum(1 for status in detailed_status.values() 
                                      if status.availability_status == ModelAvailabilityStatus.CORRUPTED),
                "total_size_gb": sum(status.size_mb for status in detailed_status.values()) / 1024,
                "last_updated": datetime.now().isoformat()
            }
            
            # Convert to serializable format
            models_dict = {}
            for model_id, status in detailed_status.items():
                models_dict[model_id] = {
                    "model_id": status.model_id,
                    "availability_status": status.availability_status.value,
                    "is_available": status.is_available,
                    "is_loaded": status.is_loaded,
                    "size_mb": status.size_mb,
                    "download_progress": status.download_progress,
                    "missing_files": status.missing_files,
                    "integrity_score": status.integrity_score,
                    "last_health_check": status.last_health_check.isoformat() if status.last_health_check else None,
                    "performance_score": status.performance_score,
                    "corruption_detected": status.corruption_detected,
                    "usage_frequency": status.usage_frequency,
                    "last_used": status.last_used.isoformat() if status.last_used else None,
                    "average_generation_time": status.average_generation_time,
                    "can_pause_download": status.can_pause_download,
                    "can_resume_download": status.can_resume_download,
                    "estimated_download_time": str(status.estimated_download_time) if status.estimated_download_time else None,
                    "current_version": status.current_version,
                    "latest_version": status.latest_version,
                    "update_available": status.update_available
                }
            
            return {
                "models": models_dict,
                "system_statistics": system_stats,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting detailed model status: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")
    
    async def manage_download(self, request: DownloadControlRequest) -> Dict[str, Any]:
        """Manage download operations (pause, resume, cancel, priority)"""
        if not self._initialized:
            await self.initialize()
        
        try:
            model_id = request.model_id
            action = request.action.lower()
            
            if action == "pause":
                success = await self.enhanced_downloader.pause_download(model_id)
                message = f"Download paused for {model_id}" if success else f"Failed to pause download for {model_id}"
                
            elif action == "resume":
                success = await self.enhanced_downloader.resume_download(model_id)
                message = f"Download resumed for {model_id}" if success else f"Failed to resume download for {model_id}"
                
            elif action == "cancel":
                success = await self.enhanced_downloader.cancel_download(model_id)
                message = f"Download cancelled for {model_id}" if success else f"Failed to cancel download for {model_id}"
                
            elif action == "priority":
                if not request.priority:
                    raise HTTPException(status_code=422, detail="Priority level required for priority action")
                
                try:
                    priority = ModelPriority(request.priority.upper())
                    success = await self.availability_manager.set_download_priority(model_id, priority)
                    message = f"Priority set to {priority.value} for {model_id}" if success else f"Failed to set priority for {model_id}"
                except ValueError:
                    raise HTTPException(status_code=422, detail=f"Invalid priority level: {request.priority}")
                
            elif action == "bandwidth":
                if request.bandwidth_limit_mbps is None:
                    raise HTTPException(status_code=422, detail="Bandwidth limit required for bandwidth action")
                
                success = await self.enhanced_downloader.set_bandwidth_limit(request.bandwidth_limit_mbps)
                message = f"Bandwidth limit set to {request.bandwidth_limit_mbps} Mbps" if success else "Failed to set bandwidth limit"
                
            else:
                raise HTTPException(status_code=422, detail=f"Invalid action: {action}")
            
            # Get updated download progress
            progress = await self.enhanced_downloader.get_download_progress(model_id)
            
            return {
                "success": success,
                "message": message,
                "model_id": model_id,
                "action": action,
                "current_status": progress.status.value if progress else "unknown",
                "progress_percent": progress.progress_percent if progress else 0.0,
                "timestamp": datetime.now().isoformat()
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error managing download: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to manage download: {str(e)}")
    
    async def get_health_monitoring_data(self) -> Dict[str, Any]:
        """Get comprehensive health monitoring data for all models"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get system health report
            health_report = await self.health_monitor.get_health_report()
            
            # Get individual model health data
            model_health = {}
            model_types = ["T2V-A14B", "I2V-A14B", "TI2V-5B"]
            
            for model_type in model_types:
                try:
                    integrity_result = await self.health_monitor.check_model_integrity(model_type)
                    model_health[model_type] = {
                        "model_id": model_type,
                        "health_status": integrity_result.health_status.value,
                        "is_healthy": integrity_result.is_healthy,
                        "integrity_score": integrity_result.integrity_score,
                        "issues": integrity_result.issues,
                        "corruption_types": [ct.value for ct in integrity_result.corruption_types],
                        "last_check": integrity_result.last_check.isoformat() if integrity_result.last_check else None,
                        "repair_suggestions": integrity_result.repair_suggestions,
                        "can_auto_repair": integrity_result.can_auto_repair
                    }
                except Exception as e:
                    logger.warning(f"Failed to get health data for {model_type}: {e}")
                    model_health[model_type] = {
                        "model_id": model_type,
                        "health_status": "unknown",
                        "is_healthy": False,
                        "error": str(e)
                    }
            
            return {
                "system_health": {
                    "overall_health_score": health_report.overall_health_score,
                    "models_healthy": health_report.models_healthy,
                    "models_degraded": health_report.models_degraded,
                    "models_corrupted": health_report.models_corrupted,
                    "storage_usage_percent": health_report.storage_usage_percent,
                    "last_updated": health_report.last_updated.isoformat()
                },
                "model_health": model_health,
                "recommendations": [
                    {
                        "type": getattr(rec, 'recommendation_type', {}).get('value', 'unknown') if hasattr(rec, 'recommendation_type') else 'unknown',
                        "priority": getattr(rec, 'priority', {}).get('value', 'medium') if hasattr(rec, 'priority') else 'medium',
                        "message": getattr(rec, 'message', str(rec)),
                        "action": getattr(rec, 'action', 'none'),
                        "model_id": getattr(rec, 'model_id', 'unknown')
                    }
                    for rec in (health_report.recommendations if hasattr(health_report, 'recommendations') else [])
                ],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting health monitoring data: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get health data: {str(e)}")
    
    async def get_usage_analytics(self, time_period_days: int = 30) -> Dict[str, Any]:
        """Get usage analytics and statistics for all models"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Calculate time period
            end_time = datetime.now()
            start_time = end_time - timedelta(days=time_period_days)
            
            # Get usage statistics for all models
            model_analytics = {}
            model_types = ["T2V-A14B", "I2V-A14B", "TI2V-5B"]
            
            for model_type in model_types:
                try:
                    stats = await self.usage_analytics.get_usage_statistics(
                        model_id=model_type,
                        start_time=start_time,
                        end_time=end_time
                    )
                    
                    model_analytics[model_type] = {
                        "model_id": stats.model_id,
                        "total_uses": stats.total_uses,
                        "uses_per_day": stats.uses_per_day,
                        "average_generation_time": stats.average_generation_time,
                        "success_rate": stats.success_rate,
                        "last_30_days_usage": [
                            {
                                "date": usage.date.isoformat(),
                                "uses": usage.uses,
                                "avg_time": usage.avg_generation_time,
                                "success_rate": usage.success_rate
                            }
                            for usage in stats.last_30_days_usage
                        ],
                        "peak_usage_hours": stats.peak_usage_hours
                    }
                except Exception as e:
                    logger.warning(f"Failed to get analytics for {model_type}: {e}")
                    model_analytics[model_type] = {
                        "model_id": model_type,
                        "total_uses": 0,
                        "uses_per_day": 0.0,
                        "error": str(e)
                    }
            
            # Get system-wide analytics
            total_uses = sum(stats.get("total_uses", 0) for stats in model_analytics.values())
            most_used_model = max(model_analytics.keys(), 
                                key=lambda k: model_analytics[k].get("total_uses", 0)) if model_analytics else None
            
            return {
                "time_period": {
                    "start_date": start_time.isoformat(),
                    "end_date": end_time.isoformat(),
                    "days": time_period_days
                },
                "system_analytics": {
                    "total_uses": total_uses,
                    "average_uses_per_day": total_uses / time_period_days if time_period_days > 0 else 0,
                    "most_used_model": most_used_model,
                    "active_models": len([m for m in model_analytics.values() if m.get("total_uses", 0) > 0])
                },
                "model_analytics": model_analytics,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting usage analytics: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")
    
    async def manage_storage_cleanup(self, request: CleanupRequest) -> Dict[str, Any]:
        """Manage storage cleanup operations"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get cleanup recommendations
            recommendations = await self.usage_analytics.recommend_model_cleanup(
                target_space_gb=request.target_space_gb,
                keep_recent_days=request.keep_recent_days
            )
            
            cleanup_actions = []
            total_space_freed_gb = 0.0
            
            if not request.dry_run:
                # Execute cleanup actions
                for rec in recommendations.cleanup_actions:
                    try:
                        if rec.action_type == "remove_model":
                            success = await self.availability_manager.remove_model(rec.model_id)
                            if success:
                                cleanup_actions.append({
                                    "action": "removed_model",
                                    "model_id": rec.model_id,
                                    "space_freed_gb": rec.space_freed_gb,
                                    "success": True
                                })
                                total_space_freed_gb += rec.space_freed_gb
                            else:
                                cleanup_actions.append({
                                    "action": "remove_model_failed",
                                    "model_id": rec.model_id,
                                    "success": False,
                                    "error": "Failed to remove model"
                                })
                        
                        elif rec.action_type == "clear_cache":
                            success = await self.availability_manager.clear_model_cache(rec.model_id)
                            if success:
                                cleanup_actions.append({
                                    "action": "cleared_cache",
                                    "model_id": rec.model_id,
                                    "space_freed_gb": rec.space_freed_gb,
                                    "success": True
                                })
                                total_space_freed_gb += rec.space_freed_gb
                    
                    except Exception as e:
                        logger.warning(f"Failed to execute cleanup action for {rec.model_id}: {e}")
                        cleanup_actions.append({
                            "action": "failed",
                            "model_id": rec.model_id,
                            "success": False,
                            "error": str(e)
                        })
            
            return {
                "dry_run": request.dry_run,
                "recommendations": {
                    "total_space_available_gb": recommendations.total_space_available_gb,
                    "target_space_gb": recommendations.target_space_gb,
                    "space_to_free_gb": recommendations.space_to_free_gb,
                    "cleanup_actions": [
                        {
                            "action_type": action.action_type,
                            "model_id": action.model_id,
                            "space_freed_gb": action.space_freed_gb,
                            "reason": action.reason,
                            "last_used": action.last_used.isoformat() if action.last_used else None
                        }
                        for action in recommendations.cleanup_actions
                    ]
                },
                "executed_actions": cleanup_actions if not request.dry_run else [],
                "total_space_freed_gb": total_space_freed_gb,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error managing storage cleanup: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to manage cleanup: {str(e)}")
    
    async def suggest_fallback_alternatives(self, request: FallbackSuggestionRequest) -> Dict[str, Any]:
        """Suggest alternative models and fallback strategies"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Create generation requirements
            requirements = GenerationRequirements(
                model_type=request.requested_model,
                quality=request.quality,
                speed=request.speed,
                resolution=request.resolution,
                max_wait_time=timedelta(minutes=request.max_wait_minutes) if request.max_wait_minutes else None
            )
            
            # Get model suggestion
            suggestion = await self.fallback_manager.suggest_alternative_model(
                requested_model=request.requested_model,
                requirements=requirements
            )
            
            # Get fallback strategy
            fallback_strategy = await self.fallback_manager.get_fallback_strategy(
                failed_model=request.requested_model,
                error_context={"reason": "model_unavailable"}
            )
            
            # Get wait time estimate if applicable
            wait_time = None
            if fallback_strategy.strategy_type in [FallbackType.QUEUE_AND_WAIT, FallbackType.DOWNLOAD_AND_RETRY]:
                wait_time = await self.fallback_manager.estimate_wait_time(request.requested_model)
            
            return {
                "requested_model": request.requested_model,
                "alternative_suggestion": {
                    "suggested_model": suggestion.suggested_model,
                    "compatibility_score": suggestion.compatibility_score,
                    "performance_difference": suggestion.performance_difference,
                    "availability_status": suggestion.availability_status.value if hasattr(suggestion.availability_status, 'value') else str(suggestion.availability_status),
                    "reason": suggestion.reason,
                    "estimated_quality_difference": suggestion.estimated_quality_difference
                } if suggestion else None,
                "fallback_strategy": {
                    "strategy_type": fallback_strategy.strategy_type.value if hasattr(fallback_strategy.strategy_type, 'value') else str(fallback_strategy.strategy_type),
                    "recommended_action": fallback_strategy.recommended_action,
                    "alternative_model": fallback_strategy.alternative_model,
                    "estimated_wait_time": str(fallback_strategy.estimated_wait_time) if fallback_strategy.estimated_wait_time else None,
                    "user_message": fallback_strategy.user_message,
                    "can_queue_request": fallback_strategy.can_queue_request
                },
                "wait_time_estimate": {
                    "estimated_minutes": wait_time.total_seconds() / 60 if wait_time else None,
                    "confidence": "medium"  # Could be enhanced with actual confidence scoring
                } if wait_time else None,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error suggesting fallback alternatives: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to suggest alternatives: {str(e)}")


# Global instance
_enhanced_api = None

async def get_enhanced_model_management_api() -> EnhancedModelManagementAPI:
    """Get the global enhanced model management API instance"""
    global _enhanced_api
    if _enhanced_api is None:
        _enhanced_api = EnhancedModelManagementAPI()
        await _enhanced_api.initialize()
    return _enhanced_api


# FastAPI Router
from fastapi import APIRouter

router = APIRouter()

@router.get("/model-management/health")
async def health():
    """Health check endpoint for model management"""
    return {"ok": True}

@router.get("/model-management/status")
async def get_model_status():
    """Get detailed model status"""
    api = await get_enhanced_model_management_api()
    return await api.get_detailed_model_status()

@router.post("/model-management/download/control")
async def control_download(request: DownloadControlRequest):
    """Control download operations (pause, resume, cancel, priority)"""
    api = await get_enhanced_model_management_api()
    return await api.manage_download(request)

@router.get("/model-management/health-monitoring")
async def get_health_monitoring():
    """Get health monitoring data"""
    api = await get_enhanced_model_management_api()
    return await api.get_health_monitoring_data()

@router.get("/model-management/analytics")
async def get_analytics(time_period_days: int = 30):
    """Get usage analytics"""
    api = await get_enhanced_model_management_api()
    return await api.get_usage_analytics(time_period_days)

@router.post("/model-management/cleanup")
async def manage_cleanup(request: CleanupRequest):
    """Manage storage cleanup"""
    api = await get_enhanced_model_management_api()
    return await api.manage_storage_cleanup(request)

@router.post("/model-management/fallback-suggestions")
async def get_fallback_suggestions(request: FallbackSuggestionRequest):
    """Get fallback model suggestions"""
    api = await get_enhanced_model_management_api()
    return await api.suggest_fallback_alternatives(request)
