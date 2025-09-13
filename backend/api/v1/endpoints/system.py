"""
System monitoring and optimization API endpoints
Fixed version - git operations working correctly
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import psutil
import logging
import asyncio
from typing import List, Dict, Any, Optional

from backend.schemas.schemas import (
    SystemStats, OptimizationSettings, HealthResponse, 
    QuantizationLevel, ErrorResponse
)
from backend.repositories.database import get_db, SystemStatsDB
from backend.core.system_integration import get_system_integration, SystemIntegration

logger = logging.getLogger(__name__)

router = APIRouter()

# Global optimization settings (in production, this would be stored in database/config)
_current_optimization_settings = OptimizationSettings()
_resource_constraints = {
    "max_concurrent_generations": 2,
    "vram_warning_threshold": 0.85,
    "vram_critical_threshold": 0.95,
    "cpu_warning_threshold": 0.90,
    "ram_warning_threshold": 0.90
}

@router.get("/system/stats", response_model=SystemStats)
async def get_system_stats(
    integration: SystemIntegration = Depends(get_system_integration)
):
    """
    Get current system resource statistics using enhanced monitoring
    Requirement 7.1: Display real-time charts and graphs for CPU, RAM, GPU, and VRAM usage
    """
    try:
        # Try to get enhanced stats from existing system
        enhanced_stats = await integration.get_enhanced_system_stats()
        
        if enhanced_stats:
            # Use enhanced stats if available
            return SystemStats(
                cpu_percent=enhanced_stats.get("cpu_percent", 0.0),
                ram_used_gb=enhanced_stats.get("ram_used_gb", 0.0),
                ram_total_gb=enhanced_stats.get("ram_total_gb", 0.0),
                ram_percent=enhanced_stats.get("ram_percent", 0.0),
                gpu_percent=enhanced_stats.get("gpu_percent", 0.0),
                vram_used_mb=enhanced_stats.get("vram_used_mb", 0.0),
                vram_total_mb=enhanced_stats.get("vram_total_mb", 0.0),
                vram_percent=enhanced_stats.get("vram_percent", 0.0),
                timestamp=datetime.utcnow()
            )
        
        # Fallback to basic stats collection
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Get RAM usage
        ram = psutil.virtual_memory()
        ram_used_gb = ram.used / (1024**3)
        ram_total_gb = ram.total / (1024**3)
        ram_percent = ram.percent
        
        # Get GPU usage (if available)
        gpu_percent = 0.0
        vram_used_mb = 0.0
        vram_total_mb = 0.0
        vram_percent = 0.0
        
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                gpu_percent = gpu.load * 100
                vram_used_mb = gpu.memoryUsed
                vram_total_mb = gpu.memoryTotal
                vram_percent = (vram_used_mb / vram_total_mb) * 100 if vram_total_mb > 0 else 0
        except ImportError:
            logger.warning("GPUtil not available, trying PyTorch for GPU stats")
            try:
                import torch
                if torch.cuda.is_available():
                    device = torch.cuda.current_device()
                    total_memory = torch.cuda.get_device_properties(device).total_memory
                    allocated_memory = torch.cuda.memory_allocated(device)
                    
                    vram_used_mb = allocated_memory / (1024**2)
                    vram_total_mb = total_memory / (1024**2)
                    vram_percent = (allocated_memory / total_memory) * 100 if total_memory > 0 else 0
            except Exception as torch_error:
                logger.warning(f"Could not get GPU stats via PyTorch: {torch_error}")
        except Exception as e:
            logger.warning(f"Could not get GPU stats: {e}")
        
        return SystemStats(
            cpu_percent=cpu_percent,
            ram_used_gb=ram_used_gb,
            ram_total_gb=ram_total_gb,
            ram_percent=ram_percent,
            gpu_percent=gpu_percent,
            vram_used_mb=vram_used_mb,
            vram_total_mb=vram_total_mb,
            vram_percent=vram_percent,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Error getting system stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not retrieve system statistics")

@router.get("/system/stats/history")
async def get_system_stats_history(
    hours: float = 24,
    db: Session = Depends(get_db)
):
    """
    Get historical system stats for monitoring dashboard
    Requirement 7.5: Show historical data with interactive time range selection
    """
    try:
        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        # Query historical stats
        stats_query = db.query(SystemStatsDB).filter(
            SystemStatsDB.timestamp >= start_time,
            SystemStatsDB.timestamp <= end_time
        ).order_by(SystemStatsDB.timestamp.desc()).limit(1000)
        
        historical_stats = stats_query.all()
        
        # Convert to response format
        stats_data = []
        for stat in historical_stats:
            stats_data.append({
                "cpu_percent": stat.cpu_percent,
                "ram_used_gb": stat.ram_used_gb,
                "ram_total_gb": stat.ram_total_gb,
                "ram_percent": stat.ram_percent,
                "gpu_percent": stat.gpu_percent,
                "vram_used_mb": stat.vram_used_mb,
                "vram_total_mb": stat.vram_total_mb,
                "vram_percent": stat.vram_percent,
                "timestamp": stat.timestamp
            })
        
        return {
            "stats": stats_data,
            "total_count": len(stats_data),
            "time_range": {
                "start": start_time,
                "end": end_time,
                "hours": hours
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting historical stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not retrieve historical statistics")

@router.post("/system/stats/save")
async def save_system_stats(
    db: Session = Depends(get_db),
    integration: SystemIntegration = Depends(get_system_integration)
):
    """
    Save current system stats to database for historical tracking
    """
    try:
        # Get current stats
        stats = await get_system_stats(integration)
        
        # Save to database
        db_stats = SystemStatsDB(
            cpu_percent=stats.cpu_percent,
            ram_used_gb=stats.ram_used_gb,
            ram_total_gb=stats.ram_total_gb,
            ram_percent=stats.ram_percent,
            gpu_percent=stats.gpu_percent,
            vram_used_mb=stats.vram_used_mb,
            vram_total_mb=stats.vram_total_mb,
            vram_percent=stats.vram_percent,
            timestamp=stats.timestamp
        )
        
        db.add(db_stats)
        db.commit()
        
        return {"message": "System stats saved successfully"}
        
    except Exception as e:
        logger.error(f"Error saving system stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not save system statistics")

@router.get("/system/optimization", response_model=OptimizationSettings)
async def get_optimization_settings(
    integration: SystemIntegration = Depends(get_system_integration)
):
    """
    Get current optimization settings from configuration
    Requirement 4.1: Provide modern settings panel with quantization options (fp16, bf16, int8)
    """
    try:
        # Return current global settings (in production, load from database/config)
        return _current_optimization_settings
        
    except Exception as e:
        logger.error(f"Error getting optimization settings: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not retrieve optimization settings")

@router.post("/system/optimization")
async def update_optimization_settings(
    settings: OptimizationSettings,
    integration: SystemIntegration = Depends(get_system_integration)
):
    """
    Update optimization settings
    Requirement 4.1: Provide quantization options and VRAM management
    Requirement 4.2: Provide real-time VRAM usage estimates and recommendations
    """
    global _current_optimization_settings
    
    try:
        # Validate settings
        if settings.vae_tile_size < 128 or settings.vae_tile_size > 512:
            raise HTTPException(
                status_code=400, 
                detail="VAE tile size must be between 128 and 512"
            )
        
        if settings.max_vram_usage_gb < 4.0 or settings.max_vram_usage_gb > 24.0:
            raise HTTPException(
                status_code=400, 
                detail="Max VRAM usage must be between 4.0 and 24.0 GB"
            )
        
        # Get current VRAM usage to provide recommendations
        current_stats = await get_system_stats(integration)
        vram_usage_gb = current_stats.vram_used_mb / 1024
        
        recommendations = []
        
        # Provide VRAM usage recommendations
        if settings.max_vram_usage_gb < vram_usage_gb + 2:
            recommendations.append("Consider increasing max VRAM usage or enabling model offloading")
        
        if settings.quantization == QuantizationLevel.FP16 and current_stats.vram_total_mb < 16000:
            recommendations.append("Consider using bf16 or int8 quantization for better VRAM efficiency")
        
        if not settings.enable_offload and current_stats.vram_total_mb < 12000:
            recommendations.append("Enable model offloading to reduce VRAM usage")
        
        # Update global settings (in production, save to database/config)
        _current_optimization_settings = settings
        
        logger.info(f"Updated optimization settings: {settings}")
        
        response = {
            "message": "Optimization settings updated successfully",
            "current_vram_usage_gb": round(vram_usage_gb, 2),
            "estimated_vram_savings_gb": _calculate_vram_savings(settings),
            "recommendations": recommendations
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating optimization settings: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not update optimization settings")

def _calculate_vram_savings(settings: OptimizationSettings) -> float:
    """Calculate estimated VRAM savings from optimization settings"""
    savings = 0.0
    
    # Quantization savings (rough estimates)
    if settings.quantization == QuantizationLevel.BF16:
        savings += 2.0  # ~2GB savings vs FP32
    elif settings.quantization == QuantizationLevel.INT8:
        savings += 4.0  # ~4GB savings vs FP32
    
    # Offloading savings
    if settings.enable_offload:
        savings += 3.0  # ~3GB savings from CPU offloading
    
    # VAE tile size savings
    if settings.vae_tile_size <= 256:
        savings += 1.0  # ~1GB savings from smaller tiles
    
    return round(savings, 1)

@router.get("/system/health", response_model=HealthResponse)
async def get_system_health(
    integration: SystemIntegration = Depends(get_system_integration)
):
    """
    Get system health status with resource constraint checking
    Requirement 7.4: Display prominent warnings when VRAM usage approaches 90%
    """
    try:
        # Get basic system info
        stats = await get_system_stats(integration)
        
        # Determine health status based on resource constraints
        status = "healthy"
        issues = []
        warnings = []
        
        # Check CPU usage
        if stats.cpu_percent > _resource_constraints["cpu_warning_threshold"] * 100:
            issues.append("High CPU usage")
            status = "warning"
        
        # Check RAM usage
        if stats.ram_percent > _resource_constraints["ram_warning_threshold"] * 100:
            issues.append("High RAM usage")
            status = "warning"
        
        # Check VRAM usage with different thresholds
        vram_percent_decimal = stats.vram_percent / 100
        if vram_percent_decimal > _resource_constraints["vram_critical_threshold"]:
            issues.append("Critical VRAM usage - generation may fail")
            status = "critical"
        elif vram_percent_decimal > _resource_constraints["vram_warning_threshold"]:
            warnings.append("High VRAM usage - consider optimization")
            if status == "healthy":
                status = "warning"
        
        # Generate recommendations based on issues
        recommendations = []
        if "High VRAM usage" in issues or any("VRAM" in w for w in warnings):
            recommendations.extend([
                "Enable model offloading to reduce VRAM usage",
                "Use int8 quantization for better memory efficiency",
                "Reduce VAE tile size to 256 or lower"
            ])
        
        if "High CPU usage" in issues:
            recommendations.append("Reduce concurrent generation tasks")
        
        if "High RAM usage" in issues:
            recommendations.append("Close unnecessary applications")
        
        message = "System is running normally"
        if issues:
            message = f"Issues detected: {', '.join(issues)}"
        elif warnings:
            message = f"Warnings: {', '.join(warnings)}"
        
        return HealthResponse(
            status=status,
            message=message,
            timestamp=datetime.utcnow(),
            system_info={
                "cpu_percent": stats.cpu_percent,
                "ram_percent": stats.ram_percent,
                "vram_percent": stats.vram_percent,
                "issues": issues,
                "warnings": warnings,
                "recommendations": recommendations,
                "resource_constraints": _resource_constraints
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting system health: {str(e)}")
        return HealthResponse(
            status="error",
            message=f"Could not determine system health: {str(e)}",
            timestamp=datetime.utcnow()
        )

@router.get("/system/constraints")
async def get_resource_constraints():
    """
    Get current resource constraint settings for graceful degradation
    """
    return {
        "constraints": _resource_constraints,
        "description": {
            "max_concurrent_generations": "Maximum number of simultaneous video generations",
            "vram_warning_threshold": "VRAM usage percentage that triggers warnings",
            "vram_critical_threshold": "VRAM usage percentage that prevents new generations",
            "cpu_warning_threshold": "CPU usage percentage that triggers warnings",
            "ram_warning_threshold": "RAM usage percentage that triggers warnings"
        }
    }

@router.post("/system/constraints")
async def update_resource_constraints(constraints: Dict[str, float]):
    """
    Update resource constraint settings for graceful degradation
    """
    global _resource_constraints
    
    try:
        # Validate constraint values
        valid_keys = set(_resource_constraints.keys())
        provided_keys = set(constraints.keys())
        
        if not provided_keys.issubset(valid_keys):
            invalid_keys = provided_keys - valid_keys
            raise HTTPException(
                status_code=400,
                detail=f"Invalid constraint keys: {list(invalid_keys)}"
            )
        
        # Validate ranges
        for key, value in constraints.items():
            if key == "max_concurrent_generations":
                if not (1 <= value <= 10):
                    raise HTTPException(
                        status_code=400,
                        detail="max_concurrent_generations must be between 1 and 10"
                    )
            else:  # Percentage thresholds
                if not (0.1 <= value <= 1.0):
                    raise HTTPException(
                        status_code=400,
                        detail=f"{key} must be between 0.1 and 1.0"
                    )
        
        # Update constraints
        _resource_constraints.update(constraints)
        
        logger.info(f"Updated resource constraints: {constraints}")
        
        return {
            "message": "Resource constraints updated successfully",
            "updated_constraints": _resource_constraints
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating resource constraints: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not update resource constraints")

@router.get("/system/resource-check")
async def check_resource_availability(
    integration: SystemIntegration = Depends(get_system_integration)
):
    """
    Check if system resources are available for new generation tasks
    Used for graceful degradation behavior
    """
    try:
        stats = await get_system_stats(integration)
        
        # Check resource availability
        vram_available = (stats.vram_percent / 100) < _resource_constraints["vram_critical_threshold"]
        cpu_available = (stats.cpu_percent / 100) < _resource_constraints["cpu_warning_threshold"]
        ram_available = (stats.ram_percent / 100) < _resource_constraints["ram_warning_threshold"]
        
        # Overall availability
        can_start_generation = vram_available and cpu_available and ram_available
        
        # Generate specific messages
        blocking_issues = []
        if not vram_available:
            blocking_issues.append(f"VRAM usage too high ({stats.vram_percent:.1f}%)")
        if not cpu_available:
            blocking_issues.append(f"CPU usage too high ({stats.cpu_percent:.1f}%)")
        if not ram_available:
            blocking_issues.append(f"RAM usage too high ({stats.ram_percent:.1f}%)")
        
        return {
            "can_start_generation": can_start_generation,
            "resource_status": {
                "vram_available": vram_available,
                "cpu_available": cpu_available,
                "ram_available": ram_available
            },
            "current_usage": {
                "vram_percent": stats.vram_percent,
                "cpu_percent": stats.cpu_percent,
                "ram_percent": stats.ram_percent
            },
            "blocking_issues": blocking_issues,
            "message": "Resources available for generation" if can_start_generation 
                      else f"Cannot start generation: {', '.join(blocking_issues)}"
        }
        
    except Exception as e:
        logger.error(f"Error checking resource availability: {str(e)}")
        raise HTTPException(status_code=500, detail="Could not check resource availability")
