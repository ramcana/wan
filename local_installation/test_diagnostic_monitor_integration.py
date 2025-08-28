#!/usr/bin/env python3
"""
Integration tests for diagnostic monitor with the reliability system
Tests integration with existing components and error handling
"""

import unittest
import time
import tempfile
import json
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Import diagnostic monitor and related components
from scripts.diagnostic_monitor import DiagnosticMonitor, AlertLevel
from scripts.reliability_manager import ReliabilityManager
from scripts.error_handler import ErrorHandler, ErrorContext


class TestDiagnosticMonitorIntegration(unittest.TestCase):
    """Integration tests for diagnostic monitor with reliability system"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        config_data = {
            "monitoring": {
                "interval": 1,
                "thresholds": {
                    "cpu_warning": 70.0,
                    "memory_warning": 75.0
                }
            }
        }
        json.dump(config_data, self.temp_config)
        self.temp_config.close()
        
        # Initialize components
        self.monitor = DiagnosticMonitor(config_path=self.temp_config.name)
        self.reliability_manager = ReliabilityManager()
        self.error_handler = ErrorHandler()
        
        # Mock callbacks
        self.alert_callback = Mock()
        self.monitor.add_alert_callback(self.alert_callback)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if self.monitor.is_monitoring:
            self.monitor.stop_monitoring()
        
        try:
            os.unlink(self.temp_config.name)
        except OSError:
            pass
    
    def test_monitor_with_reliability_manager(self):
        """Test diagnostic monitor integration with reliability manager"""
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Simulate reliability manager operations
        component = Mock()
        component.__class__.__name__ = "TestComponent"
        
        # Wrap component with reliability manager
        wrapped_component = self.reliability_manager.wrap_component(component, "test_component")
        
        # Record some errors through the monitor
        self.monitor.record_error("test_component", "Test integration error")
        
        # Let monitoring run briefly
        time.sleep(2)
        
        # Check component health
        health = self.monitor.check_component_health("test_component")
        
        # Verify integration
        self.assertIsNotNone(health)
        self.assertEqual(health.component_name, "test_component")
        
        # Generate health report
        report = self.monitor.generate_health_report()
        self.assertIsNotNone(report)
        self.assertGreater(len(report.component_health), 0)
    
    def test_monitor_with_error_handler(self):
        """Test diagnostic monitor integration with error handler"""
        # Create error context
        error_context = ErrorContext(
            phase="test_phase",
            task="test_task",
            component="test_component",
            method="test_method",
            error_message="Test error message",
            stack_trace="Test stack trace"
        )
        
        # Record error through monitor
        self.monitor.record_error("test_component", "Test error from error handler")
        
        # Start monitoring
        self.monitor.start_monitoring()
        time.sleep(1)
        
        # Generate health report
        report = self.monitor.generate_health_report()
        
        # Verify error was recorded
        self.assertGreater(len(self.monitor.error_history), 0)
        
        # Check if error patterns were detected
        if report.error_patterns:
            pattern = report.error_patterns[0]
            self.assertIn("test_component", pattern.pattern_type)
    
    def test_alert_integration_with_recovery(self):
        """Test alert integration with recovery mechanisms"""
        # Mock high resource usage to trigger alerts
        with patch('scripts.diagnostic_monitor.psutil.cpu_percent', return_value=85.0):
            with patch('scripts.diagnostic_monitor.psutil.virtual_memory') as mock_memory:
                mock_memory.return_value.percent = 80.0
                mock_memory.return_value.used = 12 * 1024**3
                mock_memory.return_value.total = 16 * 1024**3
                
                # Start monitoring
                self.monitor.start_monitoring()
                time.sleep(2)
                
                # Check if alerts were generated
                self.assertGreater(self.alert_callback.call_count, 0)
                
                # Get the alert
                alert_call = self.alert_callback.call_args[0][0]
                self.assertEqual(alert_call.level, AlertLevel.WARNING)
                self.assertIn("CPU", alert_call.message)
    
    def test_predictive_analysis_with_reliability_data(self):
        """Test predictive analysis using reliability system data"""
        # Add historical data that shows degrading performance
        base_time = datetime.now()
        
        for i in range(10):
            # Simulate increasing resource usage over time
            from scripts.diagnostic_monitor import ResourceMetrics
            metrics = ResourceMetrics(
                timestamp=base_time + timedelta(seconds=i * 5),
                cpu_percent=50.0 + i * 3,  # Increasing CPU usage
                memory_percent=40.0 + i * 2,  # Increasing memory usage
                memory_used_gb=6.0 + i * 0.5,
                memory_total_gb=16.0,
                disk_usage_percent=60.0,
                disk_free_gb=400.0
            )
            self.monitor.metrics_history.append(metrics)
        
        # Detect potential issues
        issues = self.monitor.detect_potential_issues()
        
        # Verify predictive analysis
        self.assertIsInstance(issues, list)
        
        # Check for CPU exhaustion prediction
        cpu_issues = [i for i in issues if i.issue_type == "cpu_exhaustion"]
        if cpu_issues:
            issue = cpu_issues[0]
            self.assertGreater(issue.probability, 0.0)
            self.assertIn("system", issue.affected_components)
    
    def test_component_health_monitoring_integration(self):
        """Test component health monitoring with actual reliability components"""
        # Test monitoring of reliability manager components
        components_to_test = [
            "model_downloader",
            "dependency_manager", 
            "config_validator",
            "error_handler",
            "reliability_manager"
        ]
        
        for component in components_to_test:
            health = self.monitor.check_component_health(component)
            
            self.assertIsNotNone(health)
            self.assertEqual(health.component_name, component)
            self.assertIn(health.status.value, ["healthy", "degraded", "unhealthy", "unknown"])
            self.assertGreaterEqual(health.response_time_ms, 0.0)
            self.assertGreaterEqual(health.performance_score, 0.0)
            self.assertLessEqual(health.performance_score, 100.0)
    
    def test_monitoring_during_error_recovery(self):
        """Test monitoring behavior during error recovery scenarios"""
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Simulate multiple errors in quick succession (like during recovery)
        error_components = ["model_downloader", "dependency_manager"]
        
        for i in range(10):
            component = error_components[i % len(error_components)]
            self.monitor.record_error(component, f"Recovery error {i}")
            time.sleep(0.1)  # Small delay between errors
        
        # Let monitoring process the errors
        time.sleep(2)
        
        # Generate health report
        report = self.monitor.generate_health_report()
        
        # Verify error patterns were detected
        self.assertGreater(len(report.error_patterns), 0)
        
        # Check if recommendations include recovery-related actions
        if report.recommendations:
            recommendations_text = " ".join(report.recommendations).lower()
            # Should suggest some form of intervention
            self.assertTrue(
                any(keyword in recommendations_text for keyword in 
                    ["restart", "investigate", "attention", "critical"])
            )
    
    def test_health_report_completeness(self):
        """Test that health reports contain all expected information"""
        # Add some test data
        self.monitor.record_error("test_component", "Test error")
        
        # Start monitoring briefly
        self.monitor.start_monitoring()
        time.sleep(1)
        
        # Generate comprehensive health report
        report = self.monitor.generate_health_report()
        
        # Verify all sections are present
        self.assertIsNotNone(report.timestamp)
        self.assertIsNotNone(report.overall_health)
        self.assertIsNotNone(report.resource_metrics)
        self.assertIsInstance(report.component_health, list)
        self.assertIsInstance(report.active_alerts, list)
        self.assertIsInstance(report.error_patterns, list)
        self.assertIsInstance(report.potential_issues, list)
        self.assertIsInstance(report.performance_trends, dict)
        self.assertIsInstance(report.recommendations, list)
        
        # Verify report can be serialized (important for logging/storage)
        report_dict = report.to_dict()
        self.assertIsInstance(report_dict, dict)
        
        # Verify JSON serialization works
        json_str = json.dumps(report_dict, default=str)
        self.assertIsInstance(json_str, str)
        self.assertGreater(len(json_str), 0)
    
    def test_monitoring_performance_impact(self):
        """Test that monitoring has minimal performance impact"""
        # Measure time without monitoring
        start_time = time.time()
        
        # Simulate some work
        for _ in range(1000):
            sum(i * i for i in range(100))
        
        baseline_time = time.time() - start_time
        
        # Measure time with monitoring
        self.monitor.start_monitoring()
        
        start_time = time.time()
        
        # Simulate same work
        for _ in range(1000):
            sum(i * i for i in range(100))
        
        monitored_time = time.time() - start_time
        
        # Verify monitoring overhead is reasonable (less than 50% overhead)
        overhead_ratio = monitored_time / baseline_time
        self.assertLess(overhead_ratio, 1.5, 
                       f"Monitoring overhead too high: {overhead_ratio:.2f}x")
    
    def test_concurrent_monitoring_operations(self):
        """Test monitoring under concurrent operations"""
        import threading
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Define concurrent operations
        def record_errors():
            for i in range(20):
                self.monitor.record_error("concurrent_component", f"Concurrent error {i}")
                time.sleep(0.05)
        
        def check_health():
            for _ in range(10):
                self.monitor.check_component_health("concurrent_component")
                time.sleep(0.1)
        
        def generate_reports():
            for _ in range(5):
                self.monitor.generate_health_report()
                time.sleep(0.2)
        
        # Start concurrent threads
        threads = [
            threading.Thread(target=record_errors),
            threading.Thread(target=check_health),
            threading.Thread(target=generate_reports)
        ]
        
        for thread in threads:
            thread.start()
        
        # Let them run concurrently
        time.sleep(2)
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=5)
        
        # Verify system is still functional
        final_report = self.monitor.generate_health_report()
        self.assertIsNotNone(final_report)
        
        # Verify no deadlocks or crashes occurred
        self.assertTrue(self.monitor.is_monitoring)


if __name__ == "__main__":
    # Run integration tests
    unittest.main(verbosity=2)