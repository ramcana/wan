"""
Performance monitoring API endpoints.
Provides access to performance metrics, analysis, and optimization recommendations.
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta

from core.performance_monitor import get_performance_monitor, PerformanceAnalysis
from core.system_integration import SystemIntegration

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/performance", tags=["performance"])

@router.get("/status")
async def get_system_status() -> Dict[str, Any]:
    """Get current system performance status."""
    try:
        monitor = get_performance_monitor()
        status = monitor.get_current_system_status()
        
        return {
            "success": True,
            "data": status
        }
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analysis")
async def get_performance_analysis(
    time_window_hours: int = Query(24, ge=1, le=168, description="Time window in hours (1-168)")
) -> Dict[str, Any]:
    """Get performance analysis for the specified time window."""
    try:
        monitor = get_performance_monitor()
        analysis = monitor.get_performance_analysis(time_window_hours)
        
        return {
            "success": True,
            "data": {
                "time_window_hours": time_window_hours,
                "analysis": {
                    "average_generation_time": analysis.average_generation_time,
                    "success_rate": analysis.success_rate,
                    "resource_efficiency": analysis.resource_efficiency,
                    "bottleneck_analysis": analysis.bottleneck_analysis,
                    "optimization_recommendations": analysis.optimization_recommendations,
                    "performance_trends": analysis.performance_trends
                }
            }
        }
    except Exception as e:
        logger.error(f"Failed to get performance analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def get_performance_metrics(
    time_window_hours: int = Query(24, ge=1, le=168),
    model_type: Optional[str] = Query(None, description="Filter by model type"),
    resolution: Optional[str] = Query(None, description="Filter by resolution"),
    success_only: bool = Query(False, description="Include only successful generations")
) -> Dict[str, Any]:
    """Get performance metrics with optional filtering."""
    try:
        monitor = get_performance_monitor()
        
        # Get metrics from history
        cutoff_time = datetime.now().timestamp() - (time_window_hours * 3600)
        metrics = [
            m for m in monitor.metrics_history
            if m.start_time >= cutoff_time
        ]
        
        # Apply filters
        if model_type:
            metrics = [m for m in metrics if m.model_type == model_type]
        
        if resolution:
            metrics = [m for m in metrics if m.resolution == resolution]
        
        if success_only:
            metrics = [m for m in metrics if m.success]
        
        # Convert to dict format
        metrics_data = []
        for m in metrics:
            metrics_data.append({
                "task_id": m.task_id,
                "model_type": m.model_type,
                "resolution": m.resolution,
                "steps": m.steps,
                "start_time": m.start_time,
                "end_time": m.end_time,
                "generation_time_seconds": m.generation_time_seconds,
                "model_load_time_seconds": m.model_load_time_seconds,
                "peak_vram_usage_mb": m.peak_vram_usage_mb,
                "average_vram_usage_mb": m.average_vram_usage_mb,
                "peak_ram_usage_mb": m.peak_ram_usage_mb,
                "average_cpu_usage_percent": m.average_cpu_usage_percent,
                "optimizations_applied": m.optimizations_applied,
                "quantization_used": m.quantization_used,
                "offload_used": m.offload_used,
                "success": m.success,
                "error_category": m.error_category
            })
        
        return {
            "success": True,
            "data": {
                "time_window_hours": time_window_hours,
                "total_metrics": len(metrics_data),
                "filters_applied": {
                    "model_type": model_type,
                    "resolution": resolution,
                    "success_only": success_only
                },
                "metrics": metrics_data
            }
        }
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recommendations")
async def get_optimization_recommendations(
    time_window_hours: int = Query(24, ge=1, le=168)
) -> Dict[str, Any]:
    """Get optimization recommendations based on recent performance."""
    try:
        monitor = get_performance_monitor()
        analysis = monitor.get_performance_analysis(time_window_hours)
        
        # Categorize recommendations
        categorized_recommendations = {
            "hardware": [],
            "configuration": [],
            "workflow": [],
            "general": []
        }
        
        for rec in analysis.optimization_recommendations:
            rec_lower = rec.lower()
            if any(keyword in rec_lower for keyword in ["gpu", "vram", "ram", "cpu", "memory", "hardware"]):
                categorized_recommendations["hardware"].append(rec)
            elif any(keyword in rec_lower for keyword in ["quantization", "offloading", "batch", "resolution"]):
                categorized_recommendations["configuration"].append(rec)
            elif any(keyword in rec_lower for keyword in ["model", "loading", "caching"]):
                categorized_recommendations["workflow"].append(rec)
            else:
                categorized_recommendations["general"].append(rec)
        
        return {
            "success": True,
            "data": {
                "time_window_hours": time_window_hours,
                "total_recommendations": len(analysis.optimization_recommendations),
                "categorized_recommendations": categorized_recommendations,
                "bottleneck_analysis": analysis.bottleneck_analysis,
                "resource_efficiency": analysis.resource_efficiency
            }
        }
    except Exception as e:
        logger.error(f"Failed to get optimization recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/benchmarks")
async def get_performance_benchmarks() -> Dict[str, Any]:
    """Get performance benchmarks and targets."""
    try:
        monitor = get_performance_monitor()
        
        benchmarks = {
            "generation_time_targets": {
                "720p": {
                    "target_seconds": 300,  # 5 minutes
                    "acceptable_seconds": 450,  # 7.5 minutes
                    "description": "720p video generation time targets"
                },
                "1080p": {
                    "target_seconds": 900,  # 15 minutes
                    "acceptable_seconds": 1350,  # 22.5 minutes
                    "description": "1080p video generation time targets"
                }
            },
            "resource_usage_targets": {
                "vram_usage_percent": {
                    "optimal": 70,
                    "acceptable": 85,
                    "maximum": 95,
                    "description": "VRAM usage percentage targets"
                },
                "ram_usage_percent": {
                    "optimal": 60,
                    "acceptable": 75,
                    "maximum": 90,
                    "description": "System RAM usage percentage targets"
                },
                "cpu_usage_percent": {
                    "optimal": 70,
                    "acceptable": 85,
                    "maximum": 95,
                    "description": "CPU usage percentage targets"
                }
            },
            "quality_targets": {
                "success_rate": {
                    "target": 0.98,
                    "acceptable": 0.95,
                    "minimum": 0.90,
                    "description": "Generation success rate targets"
                },
                "resource_efficiency": {
                    "target": 0.85,
                    "acceptable": 0.70,
                    "minimum": 0.50,
                    "description": "Overall resource efficiency targets"
                }
            }
        }
        
        # Get current performance against benchmarks
        analysis = monitor.get_performance_analysis(24)  # Last 24 hours
        
        current_performance = {
            "average_generation_time": analysis.average_generation_time,
            "success_rate": analysis.success_rate,
            "resource_efficiency": analysis.resource_efficiency,
            "meets_targets": {
                "generation_time": analysis.average_generation_time <= 300,  # Assuming 720p
                "success_rate": analysis.success_rate >= 0.95,
                "resource_efficiency": analysis.resource_efficiency >= 0.70
            }
        }
        
        return {
            "success": True,
            "data": {
                "benchmarks": benchmarks,
                "current_performance": current_performance,
                "analysis_time_window_hours": 24
            }
        }
    except Exception as e:
        logger.error(f"Failed to get performance benchmarks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/export")
async def export_performance_data(
    background_tasks: BackgroundTasks,
    time_window_hours: int = Query(24, ge=1, le=168),
    format: str = Query("json", regex="^(json|csv)$")
) -> Dict[str, Any]:
    """Export performance data to file."""
    try:
        monitor = get_performance_monitor()
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_export_{timestamp}.{format}"
        filepath = f"exports/{filename}"
        
        # Schedule export in background
        if format == "json":
            background_tasks.add_task(
                monitor.export_metrics,
                filepath,
                time_window_hours
            )
        else:  # CSV format
            background_tasks.add_task(
                _export_metrics_csv,
                monitor,
                filepath,
                time_window_hours
            )
        
        return {
            "success": True,
            "data": {
                "message": "Export started",
                "filename": filename,
                "filepath": filepath,
                "format": format,
                "time_window_hours": time_window_hours
            }
        }
    except Exception as e:
        logger.error(f"Failed to start performance data export: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/optimize")
async def apply_optimization_recommendations(
    recommendations: List[str],
    auto_apply: bool = Query(False, description="Automatically apply safe optimizations")
) -> Dict[str, Any]:
    """Apply optimization recommendations to the system."""
    try:
        # This would integrate with the system configuration to apply optimizations
        # For now, we'll return what would be applied
        
        applied_optimizations = []
        skipped_optimizations = []
        
        for rec in recommendations:
            rec_lower = rec.lower()
            
            # Safe optimizations that can be auto-applied
            if auto_apply and any(keyword in rec_lower for keyword in [
                "enable model caching",
                "enable quantization",
                "enable offloading"
            ]):
                applied_optimizations.append(rec)
            else:
                skipped_optimizations.append(rec)
        
        return {
            "success": True,
            "data": {
                "message": "Optimization recommendations processed",
                "applied_optimizations": applied_optimizations,
                "skipped_optimizations": skipped_optimizations,
                "auto_apply_enabled": auto_apply,
                "note": "Manual configuration changes may be required for some optimizations"
            }
        }
    except Exception as e:
        logger.error(f"Failed to apply optimization recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _export_metrics_csv(monitor, filepath: str, time_window_hours: int):
    """Export metrics to CSV format (background task)."""
    import csv
    from pathlib import Path
    
    # Ensure export directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Get metrics
    cutoff_time = datetime.now().timestamp() - (time_window_hours * 3600)
    metrics = [
        m for m in monitor.metrics_history
        if m.start_time >= cutoff_time
    ]
    
    # Write CSV
    with open(filepath, 'w', newline='') as csvfile:
        fieldnames = [
            'task_id', 'model_type', 'resolution', 'steps',
            'start_time', 'end_time', 'generation_time_seconds',
            'model_load_time_seconds', 'peak_vram_usage_mb',
            'average_vram_usage_mb', 'peak_ram_usage_mb',
            'average_cpu_usage_percent', 'optimizations_applied',
            'quantization_used', 'offload_used', 'success',
            'error_category'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for m in metrics:
            writer.writerow({
                'task_id': m.task_id,
                'model_type': m.model_type,
                'resolution': m.resolution,
                'steps': m.steps,
                'start_time': m.start_time,
                'end_time': m.end_time,
                'generation_time_seconds': m.generation_time_seconds,
                'model_load_time_seconds': m.model_load_time_seconds,
                'peak_vram_usage_mb': m.peak_vram_usage_mb,
                'average_vram_usage_mb': m.average_vram_usage_mb,
                'peak_ram_usage_mb': m.peak_ram_usage_mb,
                'average_cpu_usage_percent': m.average_cpu_usage_percent,
                'optimizations_applied': ','.join(m.optimizations_applied or []),
                'quantization_used': m.quantization_used,
                'offload_used': m.offload_used,
                'success': m.success,
                'error_category': m.error_category
            })
    
    logger.info(f"Exported {len(metrics)} metrics to CSV: {filepath}")