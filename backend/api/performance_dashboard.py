"""
Performance Dashboard API Endpoints

Provides REST API endpoints for accessing performance monitoring data
and dashboard visualization support.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

from ..core.performance_monitoring_system import (
    get_performance_monitor,
    PerformanceMonitoringSystem,
    PerformanceReport,
    PerformanceMetricType
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/performance", tags=["performance"])


def get_monitor() -> PerformanceMonitoringSystem:
    """Dependency to get performance monitor instance"""
    return get_performance_monitor()


@router.get("/dashboard")
async def get_dashboard_data(
    force_refresh: bool = Query(False, description="Force refresh cached data"),
    monitor: PerformanceMonitoringSystem = Depends(get_monitor)
) -> Dict[str, Any]:
    """
    Get comprehensive performance dashboard data
    
    Returns real-time performance metrics, resource usage, trends,
    and optimization recommendations for dashboard display.
    """
    try:
        dashboard_data = monitor.get_dashboard_data(force_refresh=force_refresh)
        return {
            "success": True,
            "data": dashboard_data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")


@router.get("/report")
async def get_performance_report(
    hours_back: int = Query(24, ge=1, le=168, description="Hours of data to include in report"),
    monitor: PerformanceMonitoringSystem = Depends(get_monitor)
) -> Dict[str, Any]:
    """
    Get detailed performance report
    
    Provides comprehensive analysis of system performance including
    operation statistics, bottlenecks, and recommendations.
    """
    try:
        report = monitor.get_performance_report(hours_back)
        
        return {
            "success": True,
            "report": {
                "report_period": {
                    "start": report.report_period[0].isoformat(),
                    "end": report.report_period[1].isoformat()
                },
                "summary": {
                    "total_operations": report.total_operations,
                    "success_rate": report.success_rate,
                    "average_duration": report.average_duration,
                    "median_duration": report.median_duration,
                    "p95_duration": report.p95_duration
                },
                "operations_by_type": report.operations_by_type,
                "resource_usage_summary": report.resource_usage_summary,
                "bottlenecks_identified": report.bottlenecks_identified,
                "optimization_recommendations": report.optimization_recommendations
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to generate performance report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate performance report: {str(e)}")


@router.get("/metrics/{metric_type}")
async def get_metrics_by_type(
    metric_type: str,
    hours_back: int = Query(24, ge=1, le=168, description="Hours of data to retrieve"),
    monitor: PerformanceMonitoringSystem = Depends(get_monitor)
) -> Dict[str, Any]:
    """
    Get performance metrics filtered by type
    
    Returns detailed metrics for specific operation types like downloads,
    health checks, fallback strategies, etc.
    """
    try:
        # Validate metric type
        try:
            metric_enum = PerformanceMetricType(metric_type)
        except ValueError:
            valid_types = [t.value for t in PerformanceMetricType]
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid metric type. Valid types: {valid_types}"
            )
        
        metrics = monitor.tracker.get_metrics_by_type(metric_enum, hours_back)
        
        # Convert metrics to serializable format
        metrics_data = []
        for metric in metrics:
            metrics_data.append({
                "operation_name": metric.operation_name,
                "start_time": metric.start_time.isoformat(),
                "end_time": metric.end_time.isoformat() if metric.end_time else None,
                "duration_seconds": metric.duration_seconds,
                "success": metric.success,
                "error_message": metric.error_message,
                "metadata": metric.metadata,
                "resource_usage": metric.resource_usage
            })
        
        return {
            "success": True,
            "metric_type": metric_type,
            "total_metrics": len(metrics_data),
            "metrics": metrics_data,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get metrics by type {metric_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@router.get("/resources/current")
async def get_current_resources(
    monitor: PerformanceMonitoringSystem = Depends(get_monitor)
) -> Dict[str, Any]:
    """
    Get current system resource usage
    
    Returns real-time CPU, memory, disk, and GPU usage information.
    """
    try:
        current_usage = monitor.resource_monitor.get_current_usage()
        
        return {
            "success": True,
            "resources": {
                "timestamp": current_usage.timestamp.isoformat(),
                "cpu_percent": current_usage.cpu_percent,
                "memory_percent": current_usage.memory_percent,
                "memory_used_mb": current_usage.memory_used_mb,
                "disk_usage_percent": current_usage.disk_usage_percent,
                "disk_free_gb": current_usage.disk_free_gb,
                "network_bytes_sent": current_usage.network_bytes_sent,
                "network_bytes_recv": current_usage.network_bytes_recv,
                "gpu_memory_used_mb": current_usage.gpu_memory_used_mb,
                "gpu_utilization_percent": current_usage.gpu_utilization_percent
            }
        }
    except Exception as e:
        logger.error(f"Failed to get current resources: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get current resources: {str(e)}")


@router.get("/resources/history")
async def get_resource_history(
    hours_back: int = Query(24, ge=1, le=168, description="Hours of history to retrieve"),
    sample_limit: int = Query(100, ge=10, le=1000, description="Maximum number of samples to return"),
    monitor: PerformanceMonitoringSystem = Depends(get_monitor)
) -> Dict[str, Any]:
    """
    Get historical system resource usage
    
    Returns time-series data for system resource usage over the specified period.
    """
    try:
        history = monitor.resource_monitor.get_resource_history(hours_back)
        
        # Downsample if we have too many points
        if len(history) > sample_limit:
            step = len(history) // sample_limit
            history = history[::step]
        
        # Convert to serializable format
        history_data = []
        for snapshot in history:
            history_data.append({
                "timestamp": snapshot.timestamp.isoformat(),
                "cpu_percent": snapshot.cpu_percent,
                "memory_percent": snapshot.memory_percent,
                "memory_used_mb": snapshot.memory_used_mb,
                "disk_usage_percent": snapshot.disk_usage_percent,
                "disk_free_gb": snapshot.disk_free_gb,
                "network_bytes_sent": snapshot.network_bytes_sent,
                "network_bytes_recv": snapshot.network_bytes_recv,
                "gpu_memory_used_mb": snapshot.gpu_memory_used_mb,
                "gpu_utilization_percent": snapshot.gpu_utilization_percent
            })
        
        return {
            "success": True,
            "total_samples": len(history_data),
            "time_range_hours": hours_back,
            "history": history_data,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get resource history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get resource history: {str(e)}")


@router.get("/operations/active")
async def get_active_operations(
    monitor: PerformanceMonitoringSystem = Depends(get_monitor)
) -> Dict[str, Any]:
    """
    Get currently active operations being tracked
    
    Returns information about operations that are currently in progress.
    """
    try:
        active_ops = []
        for op_id, metric in monitor.tracker.active_operations.items():
            duration_so_far = (datetime.now() - metric.start_time).total_seconds()
            active_ops.append({
                "operation_id": op_id,
                "metric_type": metric.metric_type.value,
                "operation_name": metric.operation_name,
                "start_time": metric.start_time.isoformat(),
                "duration_so_far": duration_so_far,
                "metadata": metric.metadata
            })
        
        return {
            "success": True,
            "active_operations_count": len(active_ops),
            "active_operations": active_ops,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get active operations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get active operations: {str(e)}")


@router.get("/benchmarks")
async def get_performance_benchmarks(
    operation_type: Optional[str] = Query(None, description="Filter by operation type"),
    hours_back: int = Query(24, ge=1, le=168, description="Hours of data for benchmarks"),
    monitor: PerformanceMonitoringSystem = Depends(get_monitor)
) -> Dict[str, Any]:
    """
    Get performance benchmarks and statistics
    
    Returns statistical analysis of operation performance including
    percentiles, trends, and comparative metrics.
    """
    try:
        if operation_type:
            try:
                metric_enum = PerformanceMetricType(operation_type)
                metrics = monitor.tracker.get_metrics_by_type(metric_enum, hours_back)
            except ValueError:
                valid_types = [t.value for t in PerformanceMetricType]
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid operation type. Valid types: {valid_types}"
                )
        else:
            metrics = monitor.tracker.get_all_metrics(hours_back)
        
        if not metrics:
            return {
                "success": True,
                "benchmarks": {},
                "message": "No metrics available for the specified period"
            }
        
        # Calculate benchmarks
        successful_metrics = [m for m in metrics if m.success and m.duration_seconds is not None]
        durations = [m.duration_seconds for m in successful_metrics]
        
        if not durations:
            return {
                "success": True,
                "benchmarks": {},
                "message": "No successful operations with duration data"
            }
        
        durations.sort()
        n = len(durations)
        
        benchmarks = {
            "total_operations": len(metrics),
            "successful_operations": len(successful_metrics),
            "success_rate": len(successful_metrics) / len(metrics),
            "duration_statistics": {
                "min_seconds": min(durations),
                "max_seconds": max(durations),
                "mean_seconds": sum(durations) / n,
                "median_seconds": durations[n // 2],
                "p75_seconds": durations[int(n * 0.75)],
                "p90_seconds": durations[int(n * 0.90)],
                "p95_seconds": durations[int(n * 0.95)],
                "p99_seconds": durations[int(n * 0.99)]
            },
            "operation_breakdown": {}
        }
        
        # Breakdown by operation type
        from collections import defaultdict
        type_stats = defaultdict(list)
        for metric in successful_metrics:
            type_stats[metric.metric_type.value].append(metric.duration_seconds)
        
        for op_type, type_durations in type_stats.items():
            type_durations.sort()
            tn = len(type_durations)
            benchmarks["operation_breakdown"][op_type] = {
                "count": tn,
                "mean_seconds": sum(type_durations) / tn,
                "median_seconds": type_durations[tn // 2],
                "p95_seconds": type_durations[int(tn * 0.95)] if tn > 20 else type_durations[-1]
            }
        
        return {
            "success": True,
            "benchmarks": benchmarks,
            "time_range_hours": hours_back,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get performance benchmarks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance benchmarks: {str(e)}")


@router.post("/optimize")
async def trigger_optimization_analysis(
    hours_back: int = Query(24, ge=1, le=168, description="Hours of data to analyze"),
    monitor: PerformanceMonitoringSystem = Depends(get_monitor)
) -> Dict[str, Any]:
    """
    Trigger comprehensive optimization analysis
    
    Performs deep analysis of performance data and generates
    detailed optimization recommendations.
    """
    try:
        # Generate comprehensive report
        report = monitor.get_performance_report(hours_back)
        
        # Additional optimization analysis
        optimization_analysis = {
            "performance_score": _calculate_performance_score(report),
            "critical_issues": _identify_critical_issues(report),
            "quick_wins": _identify_quick_wins(report),
            "long_term_improvements": _identify_long_term_improvements(report),
            "resource_optimization": _analyze_resource_optimization(report)
        }
        
        return {
            "success": True,
            "analysis": optimization_analysis,
            "recommendations": report.optimization_recommendations,
            "bottlenecks": report.bottlenecks_identified,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to perform optimization analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to perform optimization analysis: {str(e)}")


def _calculate_performance_score(report: PerformanceReport) -> float:
    """Calculate overall performance score (0-100)"""
    score = 100.0
    
    # Deduct for low success rate
    if report.success_rate < 0.95:
        score -= (0.95 - report.success_rate) * 100
    
    # Deduct for slow operations
    if report.p95_duration > 60:  # More than 1 minute
        score -= min(30, (report.p95_duration - 60) / 10)
    
    # Deduct for bottlenecks
    score -= len(report.bottlenecks_identified) * 5
    
    return max(0.0, score)


def _identify_critical_issues(report: PerformanceReport) -> List[str]:
    """Identify critical performance issues"""
    critical_issues = []
    
    if report.success_rate < 0.9:
        critical_issues.append("Low success rate indicates system instability")
    
    if report.p95_duration > 300:  # 5 minutes
        critical_issues.append("Very slow operations affecting user experience")
    
    if "High memory usage detected" in report.bottlenecks_identified:
        critical_issues.append("Memory pressure may cause system failures")
    
    if "Low disk space detected" in report.bottlenecks_identified:
        critical_issues.append("Disk space critically low")
    
    return critical_issues


def _identify_quick_wins(report: PerformanceReport) -> List[str]:
    """Identify quick optimization wins"""
    quick_wins = []
    
    if report.operations_by_type.get('health_check', 0) > 100:
        quick_wins.append("Reduce health check frequency for immediate performance gain")
    
    if report.p95_duration > 120:  # 2 minutes
        quick_wins.append("Implement operation timeouts to prevent hanging operations")
    
    if "High CPU usage detected" in report.bottlenecks_identified:
        quick_wins.append("Reduce concurrent operations during peak usage")
    
    return quick_wins


def _identify_long_term_improvements(report: PerformanceReport) -> List[str]:
    """Identify long-term improvement opportunities"""
    improvements = []
    
    if report.operations_by_type.get('download_operation', 0) > 50:
        improvements.append("Implement intelligent caching to reduce download operations")
    
    if report.average_duration > 30:
        improvements.append("Redesign algorithms for better performance characteristics")
    
    improvements.append("Implement predictive resource scaling")
    improvements.append("Add machine learning-based optimization")
    
    return improvements


def _analyze_resource_optimization(report: PerformanceReport) -> Dict[str, str]:
    """Analyze resource optimization opportunities"""
    optimization = {}
    
    cpu_avg = report.resource_usage_summary.get('avg_cpu_percent', 0)
    memory_avg = report.resource_usage_summary.get('avg_memory_percent', 0)
    
    if cpu_avg > 70:
        optimization['cpu'] = "High CPU usage - consider load balancing or optimization"
    elif cpu_avg < 20:
        optimization['cpu'] = "Low CPU usage - can handle more concurrent operations"
    else:
        optimization['cpu'] = "CPU usage is optimal"
    
    if memory_avg > 80:
        optimization['memory'] = "High memory usage - implement memory cleanup strategies"
    elif memory_avg < 30:
        optimization['memory'] = "Low memory usage - can increase cache sizes for better performance"
    else:
        optimization['memory'] = "Memory usage is optimal"
    
    return optimization