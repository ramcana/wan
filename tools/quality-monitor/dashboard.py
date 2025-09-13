"""
Real-time quality monitoring dashboard.
"""

import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

try:
    from tools.quality-monitor.models import QualityDashboard, QualityMetric, QualityTrend, QualityAlert, QualityRecommendation
    from tools.quality-monitor.metrics_collector import MetricsCollector
    from tools.quality-monitor.trend_analyzer import TrendAnalyzer
    from tools.quality-monitor.alert_system import AlertSystem
    from tools.quality-monitor.recommendation_engine import RecommendationEngine
except ImportError:
    from models import QualityDashboard, QualityMetric, QualityTrend, QualityAlert, QualityRecommendation
    from metrics_collector import MetricsCollector
    from trend_analyzer import TrendAnalyzer
    from alert_system import AlertSystem
    from recommendation_engine import RecommendationEngine


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP handler for the quality monitoring dashboard."""
    
    def __init__(self, dashboard_manager, *args, **kwargs):
        self.dashboard_manager = dashboard_manager
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        
        if path == '/':
            self._serve_dashboard()
        elif path == '/api/dashboard':
            self._serve_dashboard_data()
        elif path == '/api/metrics':
            self._serve_metrics()
        elif path == '/api/trends':
            self._serve_trends()
        elif path == '/api/alerts':
            self._serve_alerts()
        elif path == '/api/recommendations':
            self._serve_recommendations()
        elif path.startswith('/static/'):
            self._serve_static_file(path)
        else:
            self._send_404()
    
    def do_POST(self):
        """Handle POST requests."""
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        
        if path == '/api/alerts/resolve':
            self._resolve_alert()
        elif path == '/api/refresh':
            self._refresh_data()
        else:
            self._send_404()
    
    def _serve_dashboard(self):
        """Serve the main dashboard HTML."""
        html_content = self._generate_dashboard_html()
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode())
    
    def _serve_dashboard_data(self):
        """Serve complete dashboard data as JSON."""
        dashboard_data = self.dashboard_manager.get_dashboard_data()
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(dashboard_data.to_json().encode())
    
    def _serve_metrics(self):
        """Serve current metrics data."""
        metrics = self.dashboard_manager.get_current_metrics()
        data = {'metrics': [m.to_dict() for m in metrics]}
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def _serve_trends(self):
        """Serve trend analysis data."""
        trends = self.dashboard_manager.get_current_trends()
        data = {'trends': [t.to_dict() for t in trends]}
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def _serve_alerts(self):
        """Serve active alerts data."""
        alerts = self.dashboard_manager.get_active_alerts()
        data = {'alerts': [a.to_dict() for a in alerts]}
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def _serve_recommendations(self):
        """Serve recommendations data."""
        recommendations = self.dashboard_manager.get_active_recommendations()
        data = {'recommendations': [r.to_dict() for r in recommendations]}
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def _resolve_alert(self):
        """Resolve an alert."""
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            data = json.loads(post_data.decode())
            alert_id = data.get('alert_id')
            
            if alert_id:
                success = self.dashboard_manager.resolve_alert(alert_id)
                response = {'success': success}
            else:
                response = {'success': False, 'error': 'Missing alert_id'}
        
        except Exception as e:
            response = {'success': False, 'error': str(e)}
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
    
    def _refresh_data(self):
        """Refresh dashboard data."""
        try:
            self.dashboard_manager.refresh_data()
            response = {'success': True}
        except Exception as e:
            response = {'success': False, 'error': str(e)}
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
    
    def _serve_static_file(self, path):
        """Serve static files (CSS, JS)."""
        # For simplicity, we'll embed CSS/JS in the HTML
        self._send_404()
    
    def _send_404(self):
        """Send 404 response."""
        self.send_response(404)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b'<html><body><h1>404 Not Found</h1></body></html>')
    
    def _generate_dashboard_html(self):
        """Generate the dashboard HTML."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quality Monitoring Dashboard</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .header h1 {
            margin: 0;
            color: #333;
        }
        .last-updated {
            color: #666;
            font-size: 14px;
            margin-top: 5px;
        }
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .card h2 {
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }
        .metric:last-child {
            border-bottom: none;
        }
        .metric-name {
            font-weight: 500;
        }
        .metric-value {
            font-size: 18px;
            font-weight: bold;
        }
        .metric-good { color: #28a745; }
        .metric-warning { color: #ffc107; }
        .metric-danger { color: #dc3545; }
        .alert {
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
            border-left: 4px solid;
        }
        .alert-critical {
            background-color: #f8d7da;
            border-color: #dc3545;
            color: #721c24;
        }
        .alert-high {
            background-color: #fff3cd;
            border-color: #ffc107;
            color: #856404;
        }
        .alert-medium {
            background-color: #d1ecf1;
            border-color: #17a2b8;
            color: #0c5460;
        }
        .alert-low {
            background-color: #d4edda;
            border-color: #28a745;
            color: #155724;
        }
        .alert-title {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .alert-description {
            font-size: 14px;
            margin-bottom: 10px;
        }
        .alert-actions {
            display: flex;
            gap: 10px;
        }
        .btn {
            padding: 5px 10px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        }
        .btn-primary {
            background-color: #007bff;
            color: white;
        }
        .btn-success {
            background-color: #28a745;
            color: white;
        }
        .recommendation {
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
            background-color: #f8f9fa;
            border-left: 4px solid #6c757d;
        }
        .recommendation-title {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .recommendation-description {
            font-size: 14px;
            margin-bottom: 10px;
        }
        .recommendation-impact {
            font-size: 12px;
            color: #666;
        }
        .trend-indicator {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        }
        .trend-improving {
            background-color: #d4edda;
            color: #155724;
        }
        .trend-stable {
            background-color: #e2e3e5;
            color: #383d41;
        }
        .trend-degrading {
            background-color: #f8d7da;
            color: #721c24;
        }
        .refresh-btn {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 20px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .loading {
            opacity: 0.6;
            pointer-events: none;
        }
    </style>
</head>
<body>
    <button class="refresh-btn" onclick="refreshData()">Refresh</button>
    
    <div class="header">
        <h1>Quality Monitoring Dashboard</h1>
        <div class="last-updated" id="lastUpdated">Loading...</div>
    </div>
    
    <div class="dashboard-grid">
        <div class="card">
            <h2>Quality Metrics</h2>
            <div id="metrics">Loading...</div>
        </div>
        
        <div class="card">
            <h2>Quality Trends</h2>
            <div id="trends">Loading...</div>
        </div>
        
        <div class="card">
            <h2>Active Alerts</h2>
            <div id="alerts">Loading...</div>
        </div>
        
        <div class="card">
            <h2>Recommendations</h2>
            <div id="recommendations">Loading...</div>
        </div>
    </div>

    <script>
        let dashboardData = null;
        
        function formatMetricName(name) {
            return name.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase());
        }
        
        function getMetricClass(metricType, value) {
            const thresholds = {
                'test_coverage': { good: 80, warning: 60 },
                'documentation_coverage': { good: 80, warning: 60 },
                'type_hint_coverage': { good: 80, warning: 60 },
                'code_complexity': { good: 5, warning: 10 },
                'duplicate_code': { good: 5, warning: 10 },
                'style_violations': { good: 10, warning: 50 }
            };
            
            const threshold = thresholds[metricType];
            if (!threshold) return 'metric-good';
            
            if (metricType.includes('coverage')) {
                // Higher is better
                if (value >= threshold.good) return 'metric-good';
                if (value >= threshold.warning) return 'metric-warning';
                return 'metric-danger';
            } else {
                // Lower is better
                if (value <= threshold.good) return 'metric-good';
                if (value <= threshold.warning) return 'metric-warning';
                return 'metric-danger';
            }
        }
        
        function renderMetrics(metrics) {
            const container = document.getElementById('metrics');
            if (!metrics || metrics.length === 0) {
                container.innerHTML = '<p>No metrics available</p>';
                return;
            }
            
            const html = metrics.map(metric => `
                <div class="metric">
                    <span class="metric-name">${formatMetricName(metric.metric_type)}</span>
                    <span class="metric-value ${getMetricClass(metric.metric_type, metric.value)}">
                        ${metric.value.toFixed(1)}${metric.metric_type.includes('coverage') ? '%' : ''}
                    </span>
                </div>
            `).join('');
            
            container.innerHTML = html;
        }
        
        function renderTrends(trends) {
            const container = document.getElementById('trends');
            if (!trends || trends.length === 0) {
                container.innerHTML = '<p>No trend data available</p>';
                return;
            }
            
            const html = trends.map(trend => `
                <div class="metric">
                    <span class="metric-name">${formatMetricName(trend.metric_type)}</span>
                    <span class="trend-indicator trend-${trend.direction}">
                        ${trend.direction.toUpperCase()}
                        ${trend.change_rate > 0 ? '+' : ''}${trend.change_rate.toFixed(1)}%/day
                    </span>
                </div>
            `).join('');
            
            container.innerHTML = html;
        }
        
        function renderAlerts(alerts) {
            const container = document.getElementById('alerts');
            if (!alerts || alerts.length === 0) {
                container.innerHTML = '<p>No active alerts</p>';
                return;
            }
            
            const html = alerts.map(alert => `
                <div class="alert alert-${alert.severity}">
                    <div class="alert-title">${alert.message}</div>
                    <div class="alert-description">${alert.description}</div>
                    <div class="alert-actions">
                        <button class="btn btn-success" onclick="resolveAlert('${alert.id}')">
                            Resolve
                        </button>
                    </div>
                </div>
            `).join('');
            
            container.innerHTML = html;
        }
        
        function renderRecommendations(recommendations) {
            const container = document.getElementById('recommendations');
            if (!recommendations || recommendations.length === 0) {
                container.innerHTML = '<p>No recommendations available</p>';
                return;
            }
            
            const html = recommendations.slice(0, 5).map(rec => `
                <div class="recommendation">
                    <div class="recommendation-title">${rec.title}</div>
                    <div class="recommendation-description">${rec.description}</div>
                    <div class="recommendation-impact">
                        Impact: ${rec.estimated_impact.toFixed(1)}% | Effort: ${rec.estimated_effort}
                    </div>
                </div>
            `).join('');
            
            container.innerHTML = html;
        }
        
        function loadDashboardData() {
            fetch('/api/dashboard')
                .then(response => response.json())
                .then(data => {
                    dashboardData = data;
                    renderMetrics(data.metrics);
                    renderTrends(data.trends);
                    renderAlerts(data.alerts);
                    renderRecommendations(data.recommendations);
                    
                    document.getElementById('lastUpdated').textContent = 
                        `Last updated: ${new Date(data.last_updated).toLocaleString()}`;
                })
                .catch(error => {
                    console.error('Error loading dashboard data:', error);
                });
        }
        
        function refreshData() {
            document.body.classList.add('loading');
            
            fetch('/api/refresh', { method: 'POST' })
                .then(response => response.json())
                .then(result => {
                    if (result.success) {
                        loadDashboardData();
                    } else {
                        alert('Error refreshing data: ' + result.error);
                    }
                })
                .catch(error => {
                    console.error('Error refreshing data:', error);
                    alert('Error refreshing data');
                })
                .finally(() => {
                    document.body.classList.remove('loading');
                });
        }
        
        function resolveAlert(alertId) {
            fetch('/api/alerts/resolve', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ alert_id: alertId })
            })
            .then(response => response.json())
            .then(result => {
                if (result.success) {
                    loadDashboardData();
                } else {
                    alert('Error resolving alert: ' + result.error);
                }
            })
            .catch(error => {
                console.error('Error resolving alert:', error);
                alert('Error resolving alert');
            });
        }
        
        // Load initial data
        loadDashboardData();
        
        // Auto-refresh every 5 minutes
        setInterval(loadDashboardData, 5 * 60 * 1000);
    </script>
</body>
</html>
        """


class DashboardManager:
    """Manages the quality monitoring dashboard."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = project_root
        self.metrics_collector = MetricsCollector(project_root)
        self.trend_analyzer = TrendAnalyzer()
        self.alert_system = AlertSystem()
        self.recommendation_engine = RecommendationEngine()
        
        self.current_dashboard_data: Optional[QualityDashboard] = None
        self.last_refresh = datetime.min
        self.refresh_interval = timedelta(minutes=15)  # Refresh every 15 minutes
        
        # Start background refresh thread
        self.refresh_thread = threading.Thread(target=self._background_refresh, daemon=True)
        self.refresh_thread.start()
    
    def _background_refresh(self):
        """Background thread for periodic data refresh."""
        while True:
            try:
                if datetime.now() - self.last_refresh > self.refresh_interval:
                    self.refresh_data()
                time.sleep(60)  # Check every minute
            except Exception as e:
                print(f"Error in background refresh: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    def refresh_data(self):
        """Refresh all dashboard data."""
        try:
            # Collect current metrics
            metrics = self.metrics_collector.collect_all_metrics()
            
            # Store metrics for trend analysis
            self.trend_analyzer.store_metrics(metrics)
            
            # Analyze trends
            trends = self.trend_analyzer.analyze_all_trends()
            
            # Check for alerts
            metric_alerts = self.alert_system.check_metric_alerts(metrics)
            trend_alerts = self.alert_system.check_trend_alerts(trends)
            
            # Send notifications for new alerts
            new_alerts = metric_alerts + trend_alerts
            if new_alerts:
                self.alert_system.send_notifications(new_alerts)
            
            # Generate recommendations
            metric_recommendations = self.recommendation_engine.generate_metric_recommendations(metrics)
            trend_recommendations = self.recommendation_engine.generate_trend_recommendations(trends)
            proactive_recommendations = self.recommendation_engine.generate_proactive_recommendations(metrics, trends)
            
            all_recommendations = metric_recommendations + trend_recommendations + proactive_recommendations
            
            # Get active alerts and recommendations
            active_alerts = self.alert_system.get_active_alerts()
            active_recommendations = self.recommendation_engine.get_active_recommendations()
            
            # Create dashboard data
            self.current_dashboard_data = QualityDashboard(
                metrics=metrics,
                trends=trends,
                alerts=active_alerts,
                recommendations=active_recommendations
            )
            
            self.last_refresh = datetime.now()
            
        except Exception as e:
            print(f"Error refreshing dashboard data: {e}")
            raise
    
    def get_dashboard_data(self) -> QualityDashboard:
        """Get current dashboard data."""
        if self.current_dashboard_data is None:
            self.refresh_data()
        
        return self.current_dashboard_data
    
    def get_current_metrics(self) -> List[QualityMetric]:
        """Get current quality metrics."""
        dashboard_data = self.get_dashboard_data()
        return dashboard_data.metrics
    
    def get_current_trends(self) -> List[QualityTrend]:
        """Get current quality trends."""
        dashboard_data = self.get_dashboard_data()
        return dashboard_data.trends
    
    def get_active_alerts(self) -> List[QualityAlert]:
        """Get active alerts."""
        dashboard_data = self.get_dashboard_data()
        return dashboard_data.alerts
    
    def get_active_recommendations(self) -> List[QualityRecommendation]:
        """Get active recommendations."""
        dashboard_data = self.get_dashboard_data()
        return dashboard_data.recommendations
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        return self.alert_system.resolve_alert(alert_id)
    
    def start_server(self, host: str = "localhost", port: int = 8080):
        """Start the dashboard web server."""
        def handler(*args, **kwargs):
            return DashboardHandler(self, *args, **kwargs)
        
        server = HTTPServer((host, port), handler)
        print(f"Quality monitoring dashboard started at http://{host}:{port}")
        
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down dashboard server...")
            server.shutdown()
    
    def get_dashboard_summary(self) -> Dict[str, any]:
        """Get a summary of dashboard status."""
        dashboard_data = self.get_dashboard_data()
        
        return {
            'metrics_count': len(dashboard_data.metrics),
            'trends_count': len(dashboard_data.trends),
            'active_alerts': len(dashboard_data.alerts),
            'recommendations_count': len(dashboard_data.recommendations),
            'last_updated': dashboard_data.last_updated.isoformat(),
            'next_refresh': (self.last_refresh + self.refresh_interval).isoformat()
        }
