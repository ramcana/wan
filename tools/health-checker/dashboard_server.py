"""
Simple health monitoring dashboard server
"""

import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import logging

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import HTMLResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from health_checker import ProjectHealthChecker
from health_reporter import HealthReporter
from health_analytics import HealthAnalytics
from health_models import HealthConfig


class HealthDashboard:
    """
    Real-time health monitoring dashboard
    """
    
    def __init__(self, config: Optional[HealthConfig] = None):
        self.config = config or HealthConfig()
        self.logger = logging.getLogger(__name__)
        
        self.health_checker = ProjectHealthChecker(self.config)
        self.health_reporter = HealthReporter(self.config)
        self.health_analytics = HealthAnalytics()
        
        # WebSocket connections for real-time updates
        self.active_connections = []
        
        # Dashboard data cache
        self.dashboard_data = {}
        self.last_update = None
        
        if FASTAPI_AVAILABLE:
            self.app = self._create_fastapi_app()
        else:
            self.app = None
            self.logger.warning("FastAPI not available. Dashboard server disabled.")
    
    def _create_fastapi_app(self) -> FastAPI:
        """Create FastAPI application for dashboard"""
        app = FastAPI(title="Project Health Dashboard", version="1.0.0")
        
        # Serve static files (dashboard HTML/CSS/JS)
        dashboard_static = Path(__file__).parent / "dashboard_static"
        if dashboard_static.exists():
            app.mount("/static", StaticFiles(directory=str(dashboard_static)), name="static")
        
        @app.get("/", response_class=HTMLResponse)
        async def dashboard_home():
            return self._get_dashboard_html()
        
        @app.get("/api/health")
        async def get_health_data():
            """Get current health data"""
            return await self._get_current_health_data()
        
        @app.get("/api/trends/{days}")
        async def get_trends(days: int = 30):
            """Get health trends for specified days"""
            return self.health_analytics.analyze_health_trends(days)
        
        @app.get("/api/component/{component}")
        async def get_component_data(component: str):
            """Get detailed component data"""
            return self.health_analytics.analyze_component_performance(component)
        
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self._handle_websocket(websocket)
        
        return app
    
    async def _handle_websocket(self, websocket: WebSocket):
        """Handle WebSocket connections for real-time updates"""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        try:
            while True:
                # Send periodic updates
                await asyncio.sleep(30)  # Update every 30 seconds
                
                if self._should_update_data():
                    await self._update_dashboard_data()
                    await self._broadcast_update()
                
        except WebSocketDisconnect:
            self.active_connections.remove(websocket)
    
    async def _get_current_health_data(self) -> Dict[str, Any]:
        """Get current health data for API"""
        if self._should_update_data():
            await self._update_dashboard_data()
        
        return self.dashboard_data
    
    def _should_update_data(self) -> bool:
        """Check if dashboard data should be updated"""
        if not self.last_update:
            return True
        
        # Update every 5 minutes
        return (datetime.now() - self.last_update).total_seconds() > 300
    
    async def _update_dashboard_data(self):
        """Update dashboard data cache"""
        try:
            # Run health check
            health_report = await self.health_checker.run_health_check()
            
            # Generate dashboard data
            self.dashboard_data = {
                "timestamp": datetime.now().isoformat(),
                "health_report": self.health_reporter.generate_dashboard_data(health_report),
                "analytics": self.health_analytics.generate_health_insights(health_report),
                "metrics": self.health_analytics.calculate_health_metrics(health_report)
            }
            
            self.last_update = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Failed to update dashboard data: {e}")
            self.dashboard_data = {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _broadcast_update(self):
        """Broadcast updates to all connected WebSocket clients"""
        if not self.active_connections:
            return
        
        message = json.dumps({
            "type": "health_update",
            "data": self.dashboard_data
        })
        
        # Send to all connected clients
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.active_connections.remove(connection)
    
    def _get_dashboard_html(self) -> str:
        """Get dashboard HTML content"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Project Health Dashboard</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-value { font-size: 2em; font-weight: bold; margin: 10px 0; }
        .metric-value.healthy { color: #27ae60; }
        .metric-value.warning { color: #f39c12; }
        .metric-value.critical { color: #e74c3c; }
        .chart-container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
        .issues-list { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .issue-item { padding: 10px; margin: 5px 0; border-left: 4px solid #ddd; background: #f8f9fa; }
        .issue-item.critical { border-left-color: #e74c3c; }
        .issue-item.high { border-left-color: #f39c12; }
        .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
        .status-indicator.healthy { background: #27ae60; }
        .status-indicator.warning { background: #f39c12; }
        .status-indicator.critical { background: #e74c3c; }
        .last-updated { color: #666; font-size: 0.9em; }
        .loading { text-align: center; padding: 40px; color: #666; }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Project Health Dashboard</h1>
            <p class="last-updated" id="lastUpdated">Loading...</p>
        </div>
        
        <div class="metrics-grid" id="metricsGrid">
            <div class="loading">Loading health data...</div>
        </div>
        
        <div class="chart-container">
            <canvas id="healthChart" width="400" height="200"></canvas>
        </div>
        
        <div class="issues-list" id="issuesList">
            <h3>Recent Issues</h3>
            <div class="loading">Loading issues...</div>
        </div>
    </div>
    
    <script>
        let healthChart;
        let ws;
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            initializeWebSocket();
            loadInitialData();
        });
        
        function initializeWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
            
            ws.onmessage = function(event) {
                const message = JSON.parse(event.data);
                if (message.type === 'health_update') {
                    updateDashboard(message.data);
                }
            };
            
            ws.onclose = function() {
                setTimeout(initializeWebSocket, 5000); // Reconnect after 5 seconds
            };
        }
        
        async function loadInitialData() {
            try {
                const response = await fetch('/api/health');
                const data = await response.json();
                updateDashboard(data);
            } catch (error) {
                console.error('Failed to load initial data:', error);
            }
        }
        
        function updateDashboard(data) {
            if (data.error) {
                document.getElementById('metricsGrid').innerHTML = `<div class="loading">Error: ${data.error}</div>`;
                return;
            }
            
            updateMetrics(data.health_report);
            updateChart(data.health_report);
            updateIssues(data.analytics);
            updateTimestamp(data.timestamp);
        }
        
        function updateMetrics(healthReport) {
            const metricsGrid = document.getElementById('metricsGrid');
            
            const overallScore = healthReport.overall_score;
            const status = healthReport.status;
            const statusClass = getStatusClass(overallScore);
            
            let html = `
                <div class="metric-card">
                    <h3>Overall Health</h3>
                    <div class="metric-value ${statusClass}">${overallScore.toFixed(1)}</div>
                    <div><span class="status-indicator ${statusClass}"></span>${status}</div>
                </div>
            `;
            
            // Component metrics
            for (const [name, component] of Object.entries(healthReport.components)) {
                const componentStatusClass = getStatusClass(component.score);
                html += `
                    <div class="metric-card">
                        <h3>${name.replace('_', ' ').replace(/\\b\\w/g, l => l.toUpperCase())}</h3>
                        <div class="metric-value ${componentStatusClass}">${component.score.toFixed(1)}</div>
                        <div><span class="status-indicator ${componentStatusClass}"></span>${component.status}</div>
                        <div>Issues: ${component.issues_count}</div>
                    </div>
                `;
            }
            
            metricsGrid.innerHTML = html;
        }
        
        function updateChart(healthReport) {
            const ctx = document.getElementById('healthChart').getContext('2d');
            
            if (healthChart) {
                healthChart.destroy();
            }
            
            const recentScores = healthReport.trends?.recent_scores || [];
            
            healthChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: recentScores.map(item => new Date(item.timestamp).toLocaleTimeString()),
                    datasets: [{
                        label: 'Health Score',
                        data: recentScores.map(item => item.score),
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Health Score Trend'
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            });
        }
        
        function updateIssues(analytics) {
            const issuesList = document.getElementById('issuesList');
            
            let html = '<h3>Recent Issues</h3>';
            
            if (analytics.critical_insights && analytics.critical_insights.length > 0) {
                html += '<h4>Critical Issues</h4>';
                analytics.critical_insights.forEach(insight => {
                    html += `<div class="issue-item critical">${insight.message}</div>`;
                });
            }
            
            if (analytics.improvement_opportunities && analytics.improvement_opportunities.length > 0) {
                html += '<h4>Improvement Opportunities</h4>';
                analytics.improvement_opportunities.forEach(insight => {
                    html += `<div class="issue-item high">${insight.message}</div>`;
                });
            }
            
            if (!analytics.critical_insights?.length && !analytics.improvement_opportunities?.length) {
                html += '<div class="loading">No critical issues found</div>';
            }
            
            issuesList.innerHTML = html;
        }
        
        function updateTimestamp(timestamp) {
            const lastUpdated = document.getElementById('lastUpdated');
            lastUpdated.textContent = `Last updated: ${new Date(timestamp).toLocaleString()}`;
        }
        
        function getStatusClass(score) {
            if (score >= 75) return 'healthy';
            if (score >= 50) return 'warning';
            return 'critical';
        }
    </script>
</body>
</html>
        """
    
    def run_server(self, host: str = "127.0.0.1", port: int = 8080):
        """Run the dashboard server"""
        if not FASTAPI_AVAILABLE:
            self.logger.error("FastAPI not available. Cannot run dashboard server.")
            return
        
        try:
            import uvicorn
            self.logger.info(f"Starting health dashboard server at http://{host}:{port}")
            uvicorn.run(self.app, host=host, port=port)
        except ImportError:
            self.logger.error("uvicorn not available. Cannot run dashboard server.")
        except Exception as e:
            self.logger.error(f"Failed to start dashboard server: {e}")


# CLI interface for dashboard
def main():
    """Main entry point for dashboard CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Project Health Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--config", help="Path to health config file")
    
    args = parser.parse_args()
    
    # Load config if provided
    config = HealthConfig()
    if args.config:
        # Load custom config (implementation would depend on config format)
        pass
    
    dashboard = HealthDashboard(config)
    dashboard.run_server(args.host, args.port)


if __name__ == "__main__":
    main()