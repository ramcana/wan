"""
Quality monitoring and alerting system.

This package provides comprehensive quality monitoring capabilities including:
- Real-time quality metrics collection
- Trend analysis for code quality metrics over time
- Automated alerting for quality regressions and maintenance needs
- Proactive recommendations for quality improvements
- Web-based monitoring dashboard

Usage:
    from tools.quality_monitor import QualityMonitor
    
    monitor = QualityMonitor()
    dashboard_data = monitor.get_dashboard_data()
"""

from tools..models import (
    QualityMetric, QualityTrend, QualityAlert, QualityRecommendation,
    QualityThreshold, QualityDashboard, MetricType, AlertSeverity, TrendDirection
)
from tools..metrics_collector import MetricsCollector
from tools..trend_analyzer import TrendAnalyzer
from tools..alert_system import AlertSystem
from tools..recommendation_engine import RecommendationEngine
from tools..dashboard import DashboardManager


class QualityMonitor:
    """Main quality monitoring system interface."""
    
    def __init__(self, project_root: str = "."):
        """Initialize the quality monitoring system.
        
        Args:
            project_root: Root directory of the project to monitor
        """
        self.project_root = project_root
        self.dashboard_manager = DashboardManager(project_root)
    
    def collect_metrics(self):
        """Collect current quality metrics."""
        return self.dashboard_manager.metrics_collector.collect_all_metrics()
    
    def analyze_trends(self, days: int = 30):
        """Analyze quality trends over the specified period."""
        return self.dashboard_manager.trend_analyzer.analyze_all_trends(days)
    
    def check_alerts(self):
        """Check for quality alerts and return active alerts."""
        return self.dashboard_manager.alert_system.get_active_alerts()
    
    def get_recommendations(self):
        """Get quality improvement recommendations."""
        return self.dashboard_manager.recommendation_engine.get_active_recommendations()
    
    def get_dashboard_data(self):
        """Get complete dashboard data."""
        return self.dashboard_manager.get_dashboard_data()
    
    def start_dashboard(self, host: str = "localhost", port: int = 8080):
        """Start the web-based monitoring dashboard."""
        self.dashboard_manager.start_server(host, port)
    
    def refresh_data(self):
        """Manually refresh all monitoring data."""
        self.dashboard_manager.refresh_data()


__all__ = [
    'QualityMonitor',
    'QualityMetric',
    'QualityTrend', 
    'QualityAlert',
    'QualityRecommendation',
    'QualityThreshold',
    'QualityDashboard',
    'MetricType',
    'AlertSeverity',
    'TrendDirection',
    'MetricsCollector',
    'TrendAnalyzer',
    'AlertSystem',
    'RecommendationEngine',
    'DashboardManager'
]
