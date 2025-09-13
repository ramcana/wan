from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
import asyncio
from backend.monitoring.analytics_collector import AnalyticsCollector
from backend.monitoring.performance_tracker import PerformanceTracker
from backend.monitoring.error_aggregator import ErrorAggregator

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])

# Analytics collector instances
analytics_collector = AnalyticsCollector()
performance_tracker = PerformanceTracker()
error_aggregator = ErrorAggregator()

class PerformanceMetric(BaseModel):
    name: str
    value: float
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

class ErrorReport(BaseModel):
    id: str
    message: str
    stack: Optional[str] = None
    timestamp: datetime
    url: str
    userAgent: str
    userId: Optional[str] = None
    sessionId: str
    context: Dict[str, Any]
    severity: str
    tags: List[str]
    breadcrumbs: List[Dict[str, Any]]

class UserJourneyEvent(BaseModel):
    id: str
    type: str
    timestamp: datetime
    sessionId: str
    userId: Optional[str] = None
    page: str
    action: Optional[str] = None
    data: Dict[str, Any]
    duration: Optional[float] = None

class AnalyticsQuery(BaseModel):
    startDate: datetime
    endDate: datetime
    metrics: Optional[List[str]] = None
    filters: Optional[Dict[str, Any]] = None
    groupBy: Optional[str] = None

@router.post("/performance")
async def record_performance_metric(
    metric: PerformanceMetric,
    background_tasks: BackgroundTasks
):
    """Record a performance metric"""
    try:
        # Store metric asynchronously
        background_tasks.add_task(
            performance_tracker.record_metric,
            metric.name,
            metric.value,
            metric.timestamp,
            metric.metadata
        )
        
        return {"status": "recorded", "metric": metric.name}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record metric: {str(e)}")

@router.post("/errors")
async def record_error_reports(
    reports: Dict[str, List[ErrorReport]],
    background_tasks: BackgroundTasks
):
    """Record multiple error reports"""
    try:
        error_reports = reports.get("reports", [])
        
        # Process errors asynchronously
        background_tasks.add_task(
            error_aggregator.process_error_batch,
            error_reports
        )
        
        return {
            "status": "recorded",
            "count": len(error_reports)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record errors: {str(e)}")

@router.post("/journey")
async def record_journey_event(
    event: UserJourneyEvent,
    background_tasks: BackgroundTasks
):
    """Record a user journey event"""
    try:
        # Store event asynchronously
        background_tasks.add_task(
            analytics_collector.record_journey_event,
            event
        )
        
        return {"status": "recorded", "event": event.id}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record event: {str(e)}")

@router.get("/performance/summary")
async def get_performance_summary(
    hours: int = 24,
    metrics: Optional[str] = None
):
    """Get performance metrics summary"""
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        metric_names = metrics.split(",") if metrics else None
        
        summary = await performance_tracker.get_summary(
            start_time, end_time, metric_names
        )
        
        return {
            "timeRange": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "summary": summary
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance summary: {str(e)}")

@router.get("/errors/summary")
async def get_error_summary(
    hours: int = 24,
    severity: Optional[str] = None
):
    """Get error summary and trends"""
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        summary = await error_aggregator.get_error_summary(
            start_time, end_time, severity
        )
        
        return {
            "timeRange": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "summary": summary
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get error summary: {str(e)}")

@router.get("/journey/funnels")
async def get_funnel_analysis(
    funnel: str,
    hours: int = 24
):
    """Get funnel analysis for user journeys"""
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        funnel_data = await analytics_collector.get_funnel_analysis(
            funnel, start_time, end_time
        )
        
        return {
            "funnel": funnel,
            "timeRange": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "data": funnel_data
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get funnel analysis: {str(e)}")

@router.get("/dashboard")
async def get_analytics_dashboard(
    hours: int = 24
):
    """Get comprehensive analytics dashboard data"""
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        # Gather all analytics data concurrently
        performance_summary, error_summary, user_metrics, system_health = await asyncio.gather(
            performance_tracker.get_summary(start_time, end_time),
            error_aggregator.get_error_summary(start_time, end_time),
            analytics_collector.get_user_metrics(start_time, end_time),
            analytics_collector.get_system_health(start_time, end_time)
        )
        
        return {
            "timeRange": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "performance": performance_summary,
            "errors": error_summary,
            "users": user_metrics,
            "system": system_health,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")

@router.get("/performance/trends")
async def get_performance_trends(
    metric: str,
    hours: int = 24,
    interval: str = "hour"
):
    """Get performance trends over time"""
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        trends = await performance_tracker.get_trends(
            metric, start_time, end_time, interval
        )
        
        return {
            "metric": metric,
            "interval": interval,
            "timeRange": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "trends": trends
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance trends: {str(e)}")

@router.get("/errors/trends")
async def get_error_trends(
    hours: int = 24,
    interval: str = "hour"
):
    """Get error trends over time"""
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        trends = await error_aggregator.get_error_trends(
            start_time, end_time, interval
        )
        
        return {
            "interval": interval,
            "timeRange": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "trends": trends
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get error trends: {str(e)}")

@router.get("/users/behavior")
async def get_user_behavior_analysis(
    hours: int = 24,
    userId: Optional[str] = None
):
    """Get user behavior analysis"""
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        behavior_data = await analytics_collector.get_user_behavior_analysis(
            start_time, end_time, userId
        )
        
        return {
            "timeRange": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "userId": userId,
            "behavior": behavior_data
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user behavior analysis: {str(e)}")

@router.post("/query")
async def custom_analytics_query(query: AnalyticsQuery):
    """Execute custom analytics query"""
    try:
        results = await analytics_collector.execute_custom_query(
            start_date=query.startDate,
            end_date=query.endDate,
            metrics=query.metrics,
            filters=query.filters,
            group_by=query.groupBy
        )
        
        return {
            "query": query.dict(),
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute query: {str(e)}")

@router.get("/alerts")
async def get_active_alerts():
    """Get active performance and error alerts"""
    try:
        performance_alerts = await performance_tracker.get_active_alerts()
        error_alerts = await error_aggregator.get_active_alerts()
        
        return {
            "performance": performance_alerts,
            "errors": error_alerts,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")

@router.post("/alerts/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Acknowledge an alert"""
    try:
        # Try both performance and error alert systems
        acknowledged = (
            await performance_tracker.acknowledge_alert(alert_id) or
            await error_aggregator.acknowledge_alert(alert_id)
        )
        
        if not acknowledged:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        return {"status": "acknowledged", "alertId": alert_id}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to acknowledge alert: {str(e)}")

@router.get("/health")
async def get_analytics_health():
    """Get analytics system health status"""
    try:
        health_status = {
            "performance_tracker": await performance_tracker.health_check(),
            "error_aggregator": await error_aggregator.health_check(),
            "analytics_collector": await analytics_collector.health_check(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        overall_healthy = all(
            status["status"] == "healthy" 
            for status in health_status.values() 
            if isinstance(status, dict) and "status" in status
        )
        
        return {
            "status": "healthy" if overall_healthy else "degraded",
            "components": health_status
        }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# Real-time analytics endpoints
@router.get("/realtime/metrics")
async def get_realtime_metrics():
    """Get real-time performance metrics"""
    try:
        metrics = await performance_tracker.get_realtime_metrics()
        return {
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get real-time metrics: {str(e)}")

@router.get("/realtime/errors")
async def get_realtime_errors():
    """Get recent errors in real-time"""
    try:
        errors = await error_aggregator.get_recent_errors(limit=50)
        return {
            "errors": errors,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get real-time errors: {str(e)}")

@router.get("/export")
async def export_analytics_data(
    format: str = "json",
    hours: int = 24,
    include_performance: bool = True,
    include_errors: bool = True,
    include_journey: bool = True
):
    """Export analytics data in various formats"""
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        export_data = {}
        
        if include_performance:
            export_data["performance"] = await performance_tracker.export_data(start_time, end_time)
        
        if include_errors:
            export_data["errors"] = await error_aggregator.export_data(start_time, end_time)
        
        if include_journey:
            export_data["journey"] = await analytics_collector.export_journey_data(start_time, end_time)
        
        export_data["metadata"] = {
            "exportTime": datetime.utcnow().isoformat(),
            "timeRange": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "format": format
        }
        
        if format == "csv":
            # Convert to CSV format (simplified)
            return {"message": "CSV export not implemented yet", "data": export_data}
        
        return export_data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export data: {str(e)}")
