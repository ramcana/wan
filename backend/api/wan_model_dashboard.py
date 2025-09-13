"""
WAN Model Dashboard Integration
Provides dashboard-specific endpoints and real-time data for WAN model monitoring
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio
import json

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, Depends
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Import WAN model info API
try:
    from api.wan_model_info import get_wan_model_info_api, WANModelInfoAPI
    from websocket.manager import get_connection_manager
    WAN_API_AVAILABLE = True
except ImportError as e:
    logging.warning(f"WAN model info API not available: {e}")
    WAN_API_AVAILABLE = False

logger = logging.getLogger(__name__)

class DashboardMetrics(BaseModel):
    """Dashboard metrics summary"""
    timestamp: datetime
    total_models: int
    healthy_models: int
    active_models: int
    total_generations_24h: int
    average_quality_score: float
    average_performance_score: float
    system_load_percent: float
    memory_usage_percent: float
    alerts_count: int
    recommendations_count: int

class ModelStatusSummary(BaseModel):
    """Model status summary for dashboard"""
    model_id: str
    model_type: str
    status: str
    health_score: float
    performance_score: float
    last_used: Optional[datetime]
    generations_24h: int
    average_generation_time: float
    memory_usage_mb: float
    gpu_utilization_percent: float
    issues_count: int

class SystemAlert(BaseModel):
    """System alert for dashboard"""
    id: str
    severity: str  # critical, warning, info
    model_id: Optional[str]
    title: str
    message: str
    timestamp: datetime
    acknowledged: bool
    action_required: bool
    suggested_action: Optional[str]

class WANModelDashboard:
    """WAN Model Dashboard Integration"""
    
    def __init__(self):
        self.wan_api: Optional[WANModelInfoAPI] = None
        self.websocket_manager = None
        self._initialized = False
        self._dashboard_cache: Dict[str, Any] = {}
        self._cache_ttl = 30  # 30 seconds for dashboard data
        self._alerts: List[SystemAlert] = []
        self._metrics_history: List[DashboardMetrics] = []
        self._max_history = 100  # Keep last 100 metrics entries
    
    async def initialize(self) -> bool:
        """Initialize the dashboard"""
        try:
            if WAN_API_AVAILABLE:
                self.wan_api = await get_wan_model_info_api()
                self.websocket_manager = get_connection_manager()
            
            self._initialized = True
            logger.info("WAN Model Dashboard initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize WAN Model Dashboard: {e}")
            return False
    
    async def get_dashboard_overview(self) -> Dict[str, Any]:
        """Get comprehensive dashboard overview"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Check cache
            cache_key = "dashboard_overview"
            if self._is_cache_valid(cache_key):
                return self._dashboard_cache[cache_key]
            
            # Get fresh data
            if self.wan_api:
                dashboard_data = await self.wan_api.get_wan_model_dashboard_data()
            else:
                dashboard_data = await self._get_fallback_dashboard_data()
            
            # Process and enhance data
            overview = await self._process_dashboard_data(dashboard_data)
            
            # Cache the result
            self._dashboard_cache[cache_key] = {
                "data": overview,
                "timestamp": datetime.now()
            }
            
            return overview
            
        except Exception as e:
            logger.error(f"Error getting dashboard overview: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get dashboard overview: {str(e)}")
    
    async def get_dashboard_metrics(self) -> DashboardMetrics:
        """Get current dashboard metrics"""
        if not self._initialized:
            await self.initialize()
        
        try:
            overview = await self.get_dashboard_overview()
            
            # Calculate metrics
            models_data = overview.get("models", {})
            total_models = len(models_data)
            healthy_models = sum(1 for data in models_data.values() 
                               if isinstance(data, dict) and data.get("health", {}).get("health_status") == "healthy")
            
            # Calculate averages
            quality_scores = []
            performance_scores = []
            total_generations = 0
            system_memory = 0
            
            for model_data in models_data.values():
                if isinstance(model_data, dict):
                    if "performance" in model_data:
                        quality_scores.append(model_data["performance"].quality_score)
                        performance_scores.append(model_data["performance"].stability_score)
                    if "health" in model_data:
                        system_memory += model_data["health"].memory_usage_mb
                    # Simulate generation count
                    total_generations += 25  # Placeholder
            
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            avg_performance = sum(performance_scores) / len(performance_scores) if performance_scores else 0.0
            
            metrics = DashboardMetrics(
                timestamp=datetime.now(),
                total_models=total_models,
                healthy_models=healthy_models,
                active_models=healthy_models,  # Assume healthy models are active
                total_generations_24h=total_generations,
                average_quality_score=avg_quality,
                average_performance_score=avg_performance,
                system_load_percent=min(75.0, (system_memory / 16000) * 100),  # Estimate based on memory
                memory_usage_percent=min(85.0, (system_memory / 12000) * 100),
                alerts_count=len(overview.get("alerts", [])),
                recommendations_count=len(overview.get("recommendations", []))
            )
            
            # Store in history
            self._metrics_history.append(metrics)
            if len(self._metrics_history) > self._max_history:
                self._metrics_history.pop(0)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting dashboard metrics: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get dashboard metrics: {str(e)}")
    
    async def get_model_status_summaries(self) -> List[ModelStatusSummary]:
        """Get model status summaries for dashboard"""
        if not self._initialized:
            await self.initialize()
        
        try:
            overview = await self.get_dashboard_overview()
            summaries = []
            
            for model_type, model_data in overview.get("models", {}).items():
                if isinstance(model_data, dict) and "health" in model_data and "performance" in model_data:
                    health = model_data["health"]
                    performance = model_data["performance"]
                    
                    summary = ModelStatusSummary(
                        model_id=f"WAN2.2-{model_type}",
                        model_type=model_type,
                        status=health.health_status,
                        health_score=health.integrity_score,
                        performance_score=performance.quality_score,
                        last_used=datetime.now() - timedelta(hours=2),  # Placeholder
                        generations_24h=25,  # Placeholder
                        average_generation_time=performance.generation_time_avg_seconds,
                        memory_usage_mb=health.memory_usage_mb,
                        gpu_utilization_percent=health.gpu_utilization_percent,
                        issues_count=len(health.issues)
                    )
                    summaries.append(summary)
            
            return summaries
            
        except Exception as e:
            logger.error(f"Error getting model status summaries: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get model summaries: {str(e)}")
    
    async def get_system_alerts(self, severity: Optional[str] = None, acknowledged: Optional[bool] = None) -> List[SystemAlert]:
        """Get system alerts with optional filtering"""
        try:
            # Get alerts from dashboard data
            overview = await self.get_dashboard_overview()
            alerts = []
            
            # Convert dashboard alerts to SystemAlert objects
            for i, alert in enumerate(overview.get("alerts", [])):
                system_alert = SystemAlert(
                    id=f"alert_{i}_{int(datetime.now().timestamp())}",
                    severity=alert.get("severity", "warning"),
                    model_id=alert.get("model"),
                    title=f"Model {alert.get('model', 'System')} Alert",
                    message=alert.get("message", "Unknown alert"),
                    timestamp=datetime.now(),
                    acknowledged=False,
                    action_required=True,
                    suggested_action=alert.get("action")
                )
                alerts.append(system_alert)
            
            # Add stored alerts
            alerts.extend(self._alerts)
            
            # Apply filters
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            if acknowledged is not None:
                alerts = [a for a in alerts if a.acknowledged == acknowledged]
            
            # Sort by timestamp (newest first)
            alerts.sort(key=lambda x: x.timestamp, reverse=True)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error getting system alerts: {e}")
            return []
    
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge a system alert"""
        try:
            for alert in self._alerts:
                if alert.id == alert_id:
                    alert.acknowledged = True
                    logger.info(f"Alert {alert_id} acknowledged")
                    return True
            
            logger.warning(f"Alert {alert_id} not found")
            return False
            
        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {e}")
            return False
    
    async def get_metrics_history(self, hours: int = 24) -> List[DashboardMetrics]:
        """Get metrics history for the specified number of hours"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            filtered_metrics = [
                m for m in self._metrics_history 
                if m.timestamp >= cutoff_time
            ]
            return filtered_metrics
            
        except Exception as e:
            logger.error(f"Error getting metrics history: {e}")
            return []
    
    async def get_dashboard_html(self) -> str:
        """Get HTML dashboard page"""
        try:
            # Get current metrics
            metrics = await self.get_dashboard_metrics()
            model_summaries = await self.get_model_status_summaries()
            alerts = await self.get_system_alerts(acknowledged=False)
            
            # Generate HTML
            html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WAN Model Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .dashboard {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }}
        .metric-card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
        .metric-label {{ color: #666; margin-top: 5px; }}
        .models-section {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        .model-card {{ border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 10px; }}
        .model-status {{ display: inline-block; padding: 3px 8px; border-radius: 3px; color: white; font-size: 0.8em; }}
        .status-healthy {{ background-color: #28a745; }}
        .status-degraded {{ background-color: #ffc107; }}
        .status-critical {{ background-color: #dc3545; }}
        .alerts-section {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .alert {{ border-left: 4px solid #ffc107; padding: 10px; margin-bottom: 10px; background-color: #fff3cd; }}
        .alert.critical {{ border-left-color: #dc3545; background-color: #f8d7da; }}
        .refresh-btn {{ background: #667eea; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }}
        .timestamp {{ color: #666; font-size: 0.9em; }}
    </style>
    <script>
        function refreshDashboard() {{
            location.reload();
        }}
        
        // Auto-refresh every 30 seconds
        setInterval(refreshDashboard, 30000);
    </script>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>🎬 WAN Model Dashboard</h1>
            <p>Real-time monitoring and status of WAN video generation models</p>
            <button class="refresh-btn" onclick="refreshDashboard()">🔄 Refresh</button>
            <div class="timestamp">Last updated: {metrics.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{metrics.total_models}</div>
                <div class="metric-label">Total Models</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.healthy_models}</div>
                <div class="metric-label">Healthy Models</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.total_generations_24h}</div>
                <div class="metric-label">Generations (24h)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.average_quality_score:.2f}</div>
                <div class="metric-label">Avg Quality Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.system_load_percent:.1f}%</div>
                <div class="metric-label">System Load</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.alerts_count}</div>
                <div class="metric-label">Active Alerts</div>
            </div>
        </div>
        
        <div class="models-section">
            <h2>📊 Model Status</h2>
            {self._generate_model_cards_html(model_summaries)}
        </div>
        
        <div class="alerts-section">
            <h2>🚨 System Alerts</h2>
            {self._generate_alerts_html(alerts)}
        </div>
    </div>
</body>
</html>
            """
            
            return html_content
            
        except Exception as e:
            logger.error(f"Error generating dashboard HTML: {e}")
            return f"<html><body><h1>Dashboard Error</h1><p>{str(e)}</p></body></html>"
    
    def _generate_model_cards_html(self, model_summaries: List[ModelStatusSummary]) -> str:
        """Generate HTML for model cards"""
        if not model_summaries:
            return "<p>No model data available</p>"
        
        html = ""
        for model in model_summaries:
            status_class = f"status-{model.status}" if model.status in ["healthy", "degraded", "critical"] else "status-degraded"
            html += f"""
            <div class="model-card">
                <h3>{model.model_type} <span class="model-status {status_class}">{model.status.upper()}</span></h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
                    <div><strong>Health Score:</strong> {model.health_score:.2f}</div>
                    <div><strong>Performance:</strong> {model.performance_score:.2f}</div>
                    <div><strong>Avg Gen Time:</strong> {model.average_generation_time:.1f}s</div>
                    <div><strong>Memory Usage:</strong> {model.memory_usage_mb:.0f}MB</div>
                    <div><strong>GPU Utilization:</strong> {model.gpu_utilization_percent:.1f}%</div>
                    <div><strong>Issues:</strong> {model.issues_count}</div>
                </div>
            </div>
            """
        
        return html
    
    def _generate_alerts_html(self, alerts: List[SystemAlert]) -> str:
        """Generate HTML for alerts"""
        if not alerts:
            return "<p>✅ No active alerts</p>"
        
        html = ""
        for alert in alerts:
            alert_class = "critical" if alert.severity == "critical" else ""
            html += f"""
            <div class="alert {alert_class}">
                <strong>{alert.title}</strong>
                <p>{alert.message}</p>
                <small>Model: {alert.model_id or 'System'} | {alert.timestamp.strftime('%H:%M:%S')}</small>
                {f'<br><em>Suggested action: {alert.suggested_action}</em>' if alert.suggested_action else ''}
            </div>
            """
        
        return html
    
    async def _process_dashboard_data(self, dashboard_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and enhance dashboard data"""
        # Add computed fields and enhancements
        processed_data = dashboard_data.copy()
        
        # Add system health summary
        models_data = processed_data.get("models", {})
        healthy_count = sum(1 for data in models_data.values() 
                          if isinstance(data, dict) and data.get("health", {}).get("health_status") == "healthy")
        total_count = len(models_data)
        
        processed_data["system_health"] = {
            "overall_status": "healthy" if healthy_count == total_count else "degraded",
            "health_percentage": (healthy_count / total_count * 100) if total_count > 0 else 0
        }
        
        return processed_data
    
    async def _get_fallback_dashboard_data(self) -> Dict[str, Any]:
        """Get fallback dashboard data when WAN API is not available"""
        return {
            "timestamp": datetime.now().isoformat(),
            "models": {
                "T2V-A14B": {"error": "WAN API not available"},
                "I2V-A14B": {"error": "WAN API not available"},
                "TI2V-5B": {"error": "WAN API not available"}
            },
            "system_overview": {
                "total_models": 3,
                "healthy_models": 0,
                "system_status": "unavailable"
            },
            "alerts": [
                {
                    "severity": "warning",
                    "model": "System",
                    "message": "WAN Model API is not available",
                    "action": "Check WAN model components installation"
                }
            ],
            "recommendations": []
        }
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid"""
        if cache_key not in self._dashboard_cache:
            return False
        
        cache_entry = self._dashboard_cache[cache_key]
        cache_time = cache_entry.get("timestamp")
        if not cache_time:
            return False
        
        return (datetime.now() - cache_time).total_seconds() < self._cache_ttl

# Global instance
_wan_dashboard = None

async def get_wan_dashboard() -> WANModelDashboard:
    """Get the global WAN dashboard instance"""
    global _wan_dashboard
    if _wan_dashboard is None:
        _wan_dashboard = WANModelDashboard()
        await _wan_dashboard.initialize()
    return _wan_dashboard

# Create FastAPI router
router = APIRouter(prefix="/api/v1/dashboard", tags=["WAN Model Dashboard"])

@router.get("/overview")
async def get_dashboard_overview(dashboard: WANModelDashboard = Depends(get_wan_dashboard)):
    """Get comprehensive dashboard overview"""
    return await dashboard.get_dashboard_overview()

@router.get("/metrics", response_model=DashboardMetrics)
async def get_dashboard_metrics(dashboard: WANModelDashboard = Depends(get_wan_dashboard)):
    """Get current dashboard metrics"""
    return await dashboard.get_dashboard_metrics()

@router.get("/models", response_model=List[ModelStatusSummary])
async def get_model_summaries(dashboard: WANModelDashboard = Depends(get_wan_dashboard)):
    """Get model status summaries"""
    return await dashboard.get_model_status_summaries()

@router.get("/alerts", response_model=List[SystemAlert])
async def get_system_alerts(
    severity: Optional[str] = None,
    acknowledged: Optional[bool] = None,
    dashboard: WANModelDashboard = Depends(get_wan_dashboard)
):
    """Get system alerts with optional filtering"""
    return await dashboard.get_system_alerts(severity, acknowledged)

@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    dashboard: WANModelDashboard = Depends(get_wan_dashboard)
):
    """Acknowledge a system alert"""
    success = await dashboard.acknowledge_alert(alert_id)
    if success:
        return {"message": f"Alert {alert_id} acknowledged"}
    else:
        raise HTTPException(status_code=404, detail="Alert not found")

@router.get("/metrics/history", response_model=List[DashboardMetrics])
async def get_metrics_history(
    hours: int = 24,
    dashboard: WANModelDashboard = Depends(get_wan_dashboard)
):
    """Get metrics history"""
    return await dashboard.get_metrics_history(hours)

@router.get("/html", response_class=HTMLResponse)
async def get_dashboard_html(dashboard: WANModelDashboard = Depends(get_wan_dashboard)):
    """Get HTML dashboard page"""
    return await dashboard.get_dashboard_html()

@router.websocket("/ws")
async def dashboard_websocket(websocket: WebSocket, dashboard: WANModelDashboard = Depends(get_wan_dashboard)):
    """WebSocket endpoint for real-time dashboard updates"""
    await websocket.accept()
    
    try:
        while True:
            # Send current metrics every 10 seconds
            metrics = await dashboard.get_dashboard_metrics()
            await websocket.send_json({
                "type": "metrics_update",
                "data": metrics.dict(),
                "timestamp": datetime.now().isoformat()
            })
            
            # Wait for 10 seconds
            await asyncio.sleep(10)
            
    except WebSocketDisconnect:
        logger.info("Dashboard WebSocket client disconnected")
    except Exception as e:
        logger.error(f"Dashboard WebSocket error: {e}")
        await websocket.close()