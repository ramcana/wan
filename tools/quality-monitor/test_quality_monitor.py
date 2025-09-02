"""
Tests for the quality monitoring and alerting system.
"""

import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from .models import QualityMetric, MetricType, AlertSeverity
from .metrics_collector import MetricsCollector
from .trend_analyzer import TrendAnalyzer
from .alert_system import AlertSystem
from .recommendation_engine import RecommendationEngine
from .dashboard import DashboardManager


class TestQualityMonitor(unittest.TestCase):
    """Test cases for the quality monitoring system."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.test_project = Path(self.test_dir)
        
        # Create a simple Python file for testing
        test_file = self.test_project / "test_module.py"
        test_file.write_text('''
"""Test module for quality monitoring."""

def simple_function(x: int) -> int:
    """A simple function with documentation and type hints."""
    if x > 0:
        return x * 2
    else:
        return 0

def complex_function(a, b, c):
    # Complex function without documentation or type hints
    if a > 0:
        if b > 0:
            if c > 0:
                return a + b + c
            else:
                return a + b
        else:
            if c > 0:
                return a + c
            else:
                return a
    else:
        if b > 0:
            if c > 0:
                return b + c
            else:
                return b
        else:
            if c > 0:
                return c
            else:
                return 0

class TestClass:
    """A test class with documentation."""
    
    def method_with_docs(self):
        """Method with documentation."""
        pass
    
    def method_without_docs(self):
        pass
''')
        
        # Create a test file
        test_test_file = self.test_project / "test_test_module.py"
        test_test_file.write_text('''
"""Tests for test module."""

import unittest
from test_module import simple_function, complex_function

class TestModule(unittest.TestCase):
    
    def test_simple_function(self):
        """Test simple function."""
        self.assertEqual(simple_function(5), 10)
        self.assertEqual(simple_function(-1), 0)
    
    def test_complex_function(self):
        """Test complex function."""
        self.assertEqual(complex_function(1, 2, 3), 6)
''')
    
    def test_metrics_collection(self):
        """Test quality metrics collection."""
        collector = MetricsCollector(str(self.test_project))
        metrics = collector.collect_all_metrics()
        
        # Should collect multiple metrics
        self.assertGreater(len(metrics), 0)
        
        # Check that we have expected metric types
        metric_types = {m.metric_type for m in metrics}
        expected_types = {
            MetricType.CODE_COMPLEXITY,
            MetricType.DOCUMENTATION_COVERAGE,
            MetricType.TYPE_HINT_COVERAGE
        }
        
        # Should have at least some of the expected types
        self.assertTrue(expected_types.intersection(metric_types))
        
        # All metrics should have valid values
        for metric in metrics:
            self.assertIsInstance(metric.value, (int, float))
            self.assertGreaterEqual(metric.value, 0)
            self.assertIsInstance(metric.timestamp, datetime)
    
    def test_trend_analysis(self):
        """Test trend analysis functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer = TrendAnalyzer(temp_dir)
            
            # Create some test metrics over time
            base_time = datetime.now() - timedelta(days=10)
            
            for i in range(5):
                metrics = [
                    QualityMetric(
                        metric_type=MetricType.TEST_COVERAGE,
                        value=70.0 + i * 2,  # Improving trend
                        timestamp=base_time + timedelta(days=i * 2)
                    ),
                    QualityMetric(
                        metric_type=MetricType.CODE_COMPLEXITY,
                        value=8.0 + i * 0.5,  # Degrading trend
                        timestamp=base_time + timedelta(days=i * 2)
                    )
                ]
                analyzer.store_metrics(metrics)
            
            # Analyze trends
            trends = analyzer.analyze_all_trends(days=15)
            
            self.assertGreater(len(trends), 0)
            
            # Check trend analysis results
            for trend in trends:
                self.assertIn(trend.direction.value, ['improving', 'stable', 'degrading', 'unknown'])
                self.assertIsInstance(trend.change_rate, (int, float))
                self.assertGreaterEqual(trend.confidence, 0.0)
                self.assertLessEqual(trend.confidence, 1.0)
    
    def test_alert_system(self):
        """Test alert system functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "alerts.json"
            alert_system = AlertSystem(str(config_file))
            
            # Create test metrics that should trigger alerts
            metrics = [
                QualityMetric(
                    metric_type=MetricType.TEST_COVERAGE,
                    value=30.0,  # Below critical threshold
                    timestamp=datetime.now()
                ),
                QualityMetric(
                    metric_type=MetricType.CODE_COMPLEXITY,
                    value=20.0,  # Above critical threshold
                    timestamp=datetime.now()
                )
            ]
            
            # Check for alerts
            new_alerts = alert_system.check_metric_alerts(metrics)
            
            # Should generate alerts for both metrics
            self.assertGreater(len(new_alerts), 0)
            
            # Check alert properties
            for alert in new_alerts:
                self.assertIn(alert.severity, [AlertSeverity.CRITICAL, AlertSeverity.HIGH, AlertSeverity.MEDIUM, AlertSeverity.LOW])
                self.assertIsInstance(alert.message, str)
                self.assertIsInstance(alert.description, str)
                self.assertGreater(len(alert.recommendations), 0)
    
    def test_recommendation_engine(self):
        """Test recommendation engine functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            rec_engine = RecommendationEngine(temp_dir)
            
            # Create test metrics that need improvement
            metrics = [
                QualityMetric(
                    metric_type=MetricType.TEST_COVERAGE,
                    value=40.0,  # Low coverage
                    timestamp=datetime.now()
                ),
                QualityMetric(
                    metric_type=MetricType.DOCUMENTATION_COVERAGE,
                    value=30.0,  # Low documentation
                    timestamp=datetime.now()
                )
            ]
            
            # Generate recommendations
            recommendations = rec_engine.generate_metric_recommendations(metrics)
            
            # Should generate recommendations
            self.assertGreater(len(recommendations), 0)
            
            # Check recommendation properties
            for rec in recommendations:
                self.assertIsInstance(rec.title, str)
                self.assertIsInstance(rec.description, str)
                self.assertGreater(rec.estimated_impact, 0)
                self.assertIn(rec.estimated_effort, ['low', 'medium', 'high'])
                self.assertGreater(len(rec.actions), 0)
    
    def test_dashboard_manager(self):
        """Test dashboard manager functionality."""
        dashboard = DashboardManager(str(self.test_project))
        
        # Test data refresh
        dashboard.refresh_data()
        
        # Get dashboard data
        dashboard_data = dashboard.get_dashboard_data()
        
        # Should have collected data
        self.assertIsNotNone(dashboard_data)
        self.assertGreater(len(dashboard_data.metrics), 0)
        self.assertIsInstance(dashboard_data.last_updated, datetime)
        
        # Test individual data access
        metrics = dashboard.get_current_metrics()
        self.assertGreater(len(metrics), 0)
        
        trends = dashboard.get_current_trends()
        # Trends might be empty for new data
        self.assertIsInstance(trends, list)
        
        alerts = dashboard.get_active_alerts()
        self.assertIsInstance(alerts, list)
        
        recommendations = dashboard.get_active_recommendations()
        self.assertIsInstance(recommendations, list)
    
    def test_integration(self):
        """Test full system integration."""
        from tools.quality_monitor import QualityMonitor
        
        monitor = QualityMonitor(str(self.test_project))
        
        # Test basic functionality
        metrics = monitor.collect_metrics()
        self.assertGreater(len(metrics), 0)
        
        trends = monitor.analyze_trends(days=30)
        self.assertIsInstance(trends, list)
        
        alerts = monitor.check_alerts()
        self.assertIsInstance(alerts, list)
        
        recommendations = monitor.get_recommendations()
        self.assertIsInstance(recommendations, list)
        
        dashboard_data = monitor.get_dashboard_data()
        self.assertIsNotNone(dashboard_data)
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)


def run_basic_functionality_test():
    """Run a basic functionality test."""
    print("Running basic functionality test...")
    
    try:
        # Test metrics collection
        collector = MetricsCollector(".")
        metrics = collector.collect_all_metrics()
        print(f"✓ Collected {len(metrics)} metrics")
        
        # Test trend analysis
        analyzer = TrendAnalyzer()
        trends = analyzer.analyze_all_trends(days=30)
        print(f"✓ Analyzed {len(trends)} trends")
        
        # Test alert system
        alert_system = AlertSystem()
        alerts = alert_system.get_active_alerts()
        print(f"✓ Found {len(alerts)} active alerts")
        
        # Test recommendation engine
        rec_engine = RecommendationEngine()
        recommendations = rec_engine.get_active_recommendations()
        print(f"✓ Found {len(recommendations)} recommendations")
        
        # Test dashboard manager
        dashboard = DashboardManager(".")
        dashboard_data = dashboard.get_dashboard_data()
        print(f"✓ Dashboard data collected successfully")
        
        print("\n✅ All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


if __name__ == '__main__':
    # Run basic functionality test
    if run_basic_functionality_test():
        print("\nRunning unit tests...")
        unittest.main(verbosity=2)
    else:
        print("Basic functionality test failed. Skipping unit tests.")
        exit(1)