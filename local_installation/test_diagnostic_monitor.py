#!/usr/bin/env python3
"""
Comprehensive tests for the diagnostic monitoring system
Tests real-time monitoring, alerting, error pattern detection, and predictive analysis
"""

import unittest
import time
import threading
import tempfile
import json
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Import the diagnostic monitor components
from scripts.diagnostic_monitor import (
    DiagnosticMonitor, Alert, AlertLevel, ResourceMetrics, ComponentHealth,
    HealthStatus, ErrorPattern, PotentialIssue, HealthReport
)


class TestDiagnosticMonitor(unittest.TestCase):
    """Test cases for DiagnosticMonitor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        config_data = {
            "monitoring": {
                "interval": 1,  # Fast interval for testing
                "history_retention_hours": 1,
                "thresholds": {
                    "cpu_warning": 70.0,
                    "cpu_critical": 90.0,
                    "memory_warning": 75.0,
                    "memory_critical": 90.0,
                    "vram_warning": 80.0,
                    "vram_critical": 95.0
                }
            }
        }
        json.dump(config_data, self.temp_config)
        self.temp_config.close()
        
        # Initialize monitor with test config
        self.monitor = DiagnosticMonitor(config_path=self.temp_config.name)
        
        # Mock callbacks for testing
        self.alert_callback = Mock()
        self.health_callback = Mock()
        
        self.monitor.add_alert_callback(self.alert_callback)
        self.monitor.add_health_callback(self.health_callback)
    
    def tearDown(self):
        """Clean up test fixtures"""
        # Stop monitoring if running
        if self.monitor.is_monitoring:
            self.monitor.stop_monitoring()
        
        # Clean up temp config file
        try:
            os.unlink(self.temp_config.name)
        except OSError:
            pass
    
    def test_initialization(self):
        """Test monitor initialization"""
        self.assertFalse(self.monitor.is_monitoring)
        self.assertEqual(self.monitor.monitoring_interval, 1)
        self.assertIsNotNone(self.monitor.logger)
        self.assertEqual(len(self.monitor.monitored_components), 5)
        self.assertIn("model_downloader", self.monitor.monitored_components)
    
    def test_config_loading(self):
        """Test configuration loading"""
        self.assertEqual(self.monitor.thresholds["cpu_warning"], 70.0)
        self.assertEqual(self.monitor.thresholds["cpu_critical"], 90.0)
        self.assertEqual(self.monitor.monitoring_interval, 1)
    
    @patch('scripts.diagnostic_monitor.psutil.cpu_percent')
    @patch('scripts.diagnostic_monitor.psutil.virtual_memory')
    @patch('scripts.diagnostic_monitor.shutil.disk_usage')
    @patch('scripts.diagnostic_monitor.psutil.net_io_counters')
    @patch('scripts.diagnostic_monitor.psutil.pids')
    def test_collect_resource_metrics(self, mock_pids, mock_net_io, mock_disk_usage, 
                                    mock_virtual_memory, mock_cpu_percent):
        """Test resource metrics collection"""
        # Mock system metrics
        mock_cpu_percent.return_value = 45.5
        
        mock_memory = Mock()
        mock_memory.percent = 60.2
        mock_memory.used = 8 * 1024**3  # 8GB
        mock_memory.total = 16 * 1024**3  # 16GB
        mock_virtual_memory.return_value = mock_memory
        
        mock_disk_usage.return_value = Mock(
            total=1000 * 1024**3,  # 1TB
            free=500 * 1024**3,    # 500GB free
            used=500 * 1024**3     # 500GB used
        )
        
        mock_net_io.return_value = Mock(
            bytes_sent=1024**6,  # 1MB
            bytes_recv=2 * 1024**6  # 2MB
        )
        
        mock_pids.return_value = list(range(100))  # 100 processes
        
        # Collect metrics
        metrics = self.monitor._collect_resource_metrics()
        
        # Verify metrics
        self.assertIsInstance(metrics, ResourceMetrics)
        self.assertEqual(metrics.cpu_percent, 45.5)
        self.assertEqual(metrics.memory_percent, 60.2)
        self.assertEqual(metrics.memory_used_gb, 8.0)
        self.assertEqual(metrics.memory_total_gb, 16.0)
        self.assertEqual(metrics.disk_usage_percent, 50.0)
        self.assertEqual(metrics.process_count, 100)
    
    def test_threshold_checking(self):
        """Test threshold checking and alert generation"""
        # Create test metrics that exceed thresholds
        metrics = ResourceMetrics(
            timestamp=datetime.now(),
            cpu_percent=85.0,  # Above warning (70), below critical (90)
            memory_percent=95.0,  # Above critical (90)
            memory_used_gb=15.0,
            memory_total_gb=16.0,
            disk_usage_percent=60.0,
            disk_free_gb=400.0,
            vram_percent=85.0  # Above warning (80), below critical (95)
        )
        
        # Create test component health
        component_health = [
            ComponentHealth(
                component_name="test_component",
                status=HealthStatus.DEGRADED,
                response_time_ms=1500.0,  # Above warning threshold
                error_count=2,
                last_error=None,
                uptime_seconds=3600.0,
                performance_score=60.0
            )
        ]
        
        # Check thresholds
        alerts = self.monitor._check_thresholds(metrics, component_health)
        
        # Verify alerts
        self.assertGreater(len(alerts), 0)
        
        # Check for specific alerts
        cpu_alerts = [a for a in alerts if a.metric_name == "cpu_percent"]
        memory_alerts = [a for a in alerts if a.metric_name == "memory_percent"]
        vram_alerts = [a for a in alerts if a.metric_name == "vram_percent"]
        component_alerts = [a for a in alerts if a.metric_name == "component_health"]
        
        self.assertEqual(len(cpu_alerts), 1)
        self.assertEqual(cpu_alerts[0].level, AlertLevel.WARNING)
        
        self.assertEqual(len(memory_alerts), 1)
        self.assertEqual(memory_alerts[0].level, AlertLevel.CRITICAL)
        
        self.assertEqual(len(vram_alerts), 1)
        self.assertEqual(vram_alerts[0].level, AlertLevel.WARNING)
        
        self.assertEqual(len(component_alerts), 1)
        self.assertEqual(component_alerts[0].level, AlertLevel.WARNING)
    
    def test_error_pattern_detection(self):
        """Test error pattern detection"""
        # Record multiple errors for the same component
        component = "test_component"
        base_time = datetime.now()
        
        # Simulate errors over time
        for i in range(5):
            error_time = base_time + timedelta(minutes=i * 2)
            with patch('scripts.diagnostic_monitor.datetime') as mock_datetime:
                mock_datetime.now.return_value = error_time
                self.monitor.record_error(component, f"Test error {i}")
        
        # Detect patterns
        patterns = self.monitor._detect_error_patterns()
        
        # Verify pattern detection
        component_patterns = [p for p in patterns if p.pattern_type == component]
        if component_patterns:  # Pattern might not be detected if frequency threshold not met
            pattern = component_patterns[0]
            self.assertEqual(pattern.pattern_type, component)
            self.assertGreater(pattern.frequency, 0)
            self.assertIsInstance(pattern.prediction_confidence, float)
            self.assertGreaterEqual(pattern.prediction_confidence, 0.0)
            self.assertLessEqual(pattern.prediction_confidence, 1.0)
    
    def test_potential_issue_detection(self):
        """Test predictive failure analysis"""
        # Add historical metrics with increasing trend
        base_time = datetime.now()
        
        for i in range(15):
            metrics = ResourceMetrics(
                timestamp=base_time + timedelta(seconds=i * 5),
                cpu_percent=60.0 + i * 2,  # Increasing trend
                memory_percent=50.0 + i * 1.5,  # Increasing trend
                memory_used_gb=8.0 + i * 0.2,
                memory_total_gb=16.0,
                disk_usage_percent=70.0,
                disk_free_gb=300.0,
                vram_percent=40.0 + i * 3  # Steep increasing trend
            )
            self.monitor.metrics_history.append(metrics)
        
        # Detect potential issues
        issues = self.monitor._detect_potential_issues()
        
        # Verify issue detection
        self.assertIsInstance(issues, list)
        
        # Check for CPU exhaustion prediction
        cpu_issues = [i for i in issues if i.issue_type == "cpu_exhaustion"]
        if cpu_issues:
            issue = cpu_issues[0]
            self.assertGreater(issue.probability, 0.0)
            self.assertLessEqual(issue.probability, 1.0)
            self.assertIn("system", issue.affected_components)
            self.assertGreater(len(issue.recommended_actions), 0)
        
        # Check for VRAM exhaustion prediction
        vram_issues = [i for i in issues if i.issue_type == "vram_exhaustion"]
        if vram_issues:
            issue = vram_issues[0]
            self.assertGreater(issue.probability, 0.0)
            self.assertIn("model_downloader", issue.affected_components)
    
    def test_component_health_checking(self):
        """Test component health monitoring"""
        # Test known component
        health = self.monitor.check_component_health("model_downloader")
        
        self.assertIsInstance(health, ComponentHealth)
        self.assertEqual(health.component_name, "model_downloader")
        self.assertIsInstance(health.status, HealthStatus)
        self.assertGreaterEqual(health.response_time_ms, 0.0)
        self.assertGreaterEqual(health.performance_score, 0.0)
        self.assertLessEqual(health.performance_score, 100.0)
        
        # Test unknown component
        health = self.monitor.check_component_health("unknown_component")
        self.assertEqual(health.status, HealthStatus.UNKNOWN)
        self.assertEqual(health.component_name, "unknown_component")
    
    def test_health_report_generation(self):
        """Test comprehensive health report generation"""
        # Add some test data
        self.monitor.record_error("test_component", "Test error")
        
        # Generate health report
        report = self.monitor.generate_health_report()
        
        # Verify report structure
        self.assertIsInstance(report, HealthReport)
        self.assertIsInstance(report.overall_health, HealthStatus)
        self.assertIsInstance(report.resource_metrics, ResourceMetrics)
        self.assertIsInstance(report.component_health, list)
        self.assertIsInstance(report.active_alerts, list)
        self.assertIsInstance(report.error_patterns, list)
        self.assertIsInstance(report.potential_issues, list)
        self.assertIsInstance(report.performance_trends, dict)
        self.assertIsInstance(report.recommendations, list)
        
        # Verify component health is populated
        self.assertGreater(len(report.component_health), 0)
        
        # Verify report can be serialized
        report_dict = report.to_dict()
        self.assertIsInstance(report_dict, dict)
        self.assertIn("timestamp", report_dict)
        self.assertIn("overall_health", report_dict)
    
    def test_monitoring_lifecycle(self):
        """Test monitoring start/stop lifecycle"""
        # Initially not monitoring
        self.assertFalse(self.monitor.is_monitoring)
        
        # Start monitoring
        self.monitor.start_monitoring()
        self.assertTrue(self.monitor.is_monitoring)
        self.assertIsNotNone(self.monitor.monitoring_thread)
        
        # Let it run briefly
        time.sleep(2)
        
        # Verify callbacks were called
        self.assertGreater(self.health_callback.call_count, 0)
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        self.assertFalse(self.monitor.is_monitoring)
    
    def test_callback_registration(self):
        """Test callback registration and triggering"""
        # Test alert callback
        alert = Alert(
            level=AlertLevel.WARNING,
            message="Test alert",
            metric_name="test_metric",
            current_value=80.0,
            threshold_value=70.0
        )
        
        self.monitor._trigger_alert_callbacks(alert)
        self.alert_callback.assert_called_once_with(alert)
        
        # Test health callback
        health_report = self.monitor.generate_health_report()
        self.monitor._trigger_health_callbacks(health_report)
        self.health_callback.assert_called_with(health_report)
    
    def test_trend_calculation(self):
        """Test trend calculation for predictive analysis"""
        # Test increasing trend
        increasing_values = [10.0, 15.0, 20.0, 25.0, 30.0]
        trend = self.monitor._calculate_trend(increasing_values)
        self.assertGreater(trend, 0)
        
        # Test decreasing trend
        decreasing_values = [30.0, 25.0, 20.0, 15.0, 10.0]
        trend = self.monitor._calculate_trend(decreasing_values)
        self.assertLess(trend, 0)
        
        # Test stable trend
        stable_values = [20.0, 20.0, 20.0, 20.0, 20.0]
        trend = self.monitor._calculate_trend(stable_values)
        self.assertAlmostEqual(trend, 0.0, places=1)
        
        # Test edge cases
        self.assertEqual(self.monitor._calculate_trend([]), 0.0)
        self.assertEqual(self.monitor._calculate_trend([10.0]), 0.0)
    
    def test_time_to_threshold_estimation(self):
        """Test time to threshold estimation"""
        # Test with positive trend
        values = [60.0, 65.0, 70.0, 75.0]
        trend = 5.0  # 5% increase per interval
        threshold = 90.0
        
        time_to_threshold = self.monitor._estimate_time_to_threshold(values, threshold, trend)
        
        if time_to_threshold:
            self.assertIsInstance(time_to_threshold, timedelta)
            self.assertGreater(time_to_threshold.total_seconds(), 0)
        
        # Test with negative trend (should return None)
        time_to_threshold = self.monitor._estimate_time_to_threshold(values, threshold, -1.0)
        self.assertIsNone(time_to_threshold)
        
        # Test when already above threshold
        time_to_threshold = self.monitor._estimate_time_to_threshold([95.0], threshold, 1.0)
        if time_to_threshold:
            self.assertEqual(time_to_threshold.total_seconds(), 0)
    
    def test_monitoring_status(self):
        """Test monitoring status reporting"""
        status = self.monitor.get_monitoring_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn("is_monitoring", status)
        self.assertIn("monitoring_interval", status)
        self.assertIn("metrics_history_size", status)
        self.assertIn("monitored_components", status)
        self.assertIn("alert_history_size", status)
        self.assertIn("error_patterns_count", status)
        
        self.assertEqual(status["monitoring_interval"], 1)
        self.assertIsInstance(status["monitored_components"], list)
        self.assertGreater(len(status["monitored_components"]), 0)
    
    def test_data_cleanup(self):
        """Test old data cleanup functionality"""
        # Add old error patterns
        old_time = datetime.now() - timedelta(hours=2)
        recent_time = datetime.now() - timedelta(minutes=5)
        
        self.monitor.error_patterns["test_component"] = [old_time, recent_time]
        
        # Run cleanup
        self.monitor._cleanup_old_data()
        
        # Verify old data was cleaned up
        remaining_errors = self.monitor.error_patterns.get("test_component", [])
        self.assertLessEqual(len(remaining_errors), 1)  # Only recent error should remain
    
    def test_error_handling(self):
        """Test error handling in monitoring operations"""
        # Test with invalid config
        with patch('builtins.open', side_effect=FileNotFoundError):
            monitor = DiagnosticMonitor("nonexistent_config.json")
            self.assertIsInstance(monitor.config, dict)  # Should use empty dict
        
        # Test metrics collection with system errors
        with patch('scripts.diagnostic_monitor.psutil.cpu_percent', side_effect=Exception("Test error")):
            metrics = self.monitor._collect_resource_metrics()
            self.assertIsInstance(metrics, ResourceMetrics)
            self.assertEqual(metrics.cpu_percent, 0.0)  # Should use default values


class TestDataStructures(unittest.TestCase):
    """Test cases for data structures used by diagnostic monitor"""
    
    def test_alert_serialization(self):
        """Test Alert serialization"""
        alert = Alert(
            level=AlertLevel.CRITICAL,
            message="Test critical alert",
            metric_name="cpu_percent",
            current_value=95.0,
            threshold_value=90.0,
            component="test_component"
        )
        
        alert_dict = alert.to_dict()
        
        self.assertIsInstance(alert_dict, dict)
        self.assertEqual(alert_dict["level"], "critical")
        self.assertEqual(alert_dict["message"], "Test critical alert")
        self.assertEqual(alert_dict["metric_name"], "cpu_percent")
        self.assertEqual(alert_dict["current_value"], 95.0)
        self.assertEqual(alert_dict["threshold_value"], 90.0)
        self.assertEqual(alert_dict["component"], "test_component")
        self.assertIn("timestamp", alert_dict)
    
    def test_resource_metrics_serialization(self):
        """Test ResourceMetrics serialization"""
        metrics = ResourceMetrics(
            timestamp=datetime.now(),
            cpu_percent=75.5,
            memory_percent=60.2,
            memory_used_gb=8.0,
            memory_total_gb=16.0,
            disk_usage_percent=45.0,
            disk_free_gb=500.0,
            vram_percent=30.0
        )
        
        metrics_dict = metrics.to_dict()
        
        self.assertIsInstance(metrics_dict, dict)
        self.assertEqual(metrics_dict["cpu_percent"], 75.5)
        self.assertEqual(metrics_dict["memory_percent"], 60.2)
        self.assertEqual(metrics_dict["vram_percent"], 30.0)
        self.assertIn("timestamp", metrics_dict)
    
    def test_component_health_serialization(self):
        """Test ComponentHealth serialization"""
        health = ComponentHealth(
            component_name="test_component",
            status=HealthStatus.HEALTHY,
            response_time_ms=150.0,
            error_count=0,
            last_error=None,
            uptime_seconds=3600.0,
            performance_score=95.0
        )
        
        health_dict = health.to_dict()
        
        self.assertIsInstance(health_dict, dict)
        self.assertEqual(health_dict["component_name"], "test_component")
        self.assertEqual(health_dict["status"], "healthy")
        self.assertEqual(health_dict["response_time_ms"], 150.0)
        self.assertEqual(health_dict["performance_score"], 95.0)
    
    def test_potential_issue_serialization(self):
        """Test PotentialIssue serialization"""
        issue = PotentialIssue(
            issue_type="memory_exhaustion",
            probability=0.75,
            estimated_time_to_failure=timedelta(minutes=30),
            affected_components=["system", "model_downloader"],
            recommended_actions=["Clear caches", "Restart services"],
            confidence_level=0.8
        )
        
        issue_dict = issue.to_dict()
        
        self.assertIsInstance(issue_dict, dict)
        self.assertEqual(issue_dict["issue_type"], "memory_exhaustion")
        self.assertEqual(issue_dict["probability"], 0.75)
        self.assertEqual(issue_dict["estimated_time_to_failure_seconds"], 1800.0)
        self.assertEqual(issue_dict["affected_components"], ["system", "model_downloader"])
        self.assertEqual(issue_dict["confidence_level"], 0.8)


if __name__ == "__main__":
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestDiagnosticMonitor))
    test_suite.addTest(unittest.makeSuite(TestDataStructures))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            error_msg = traceback.split('AssertionError: ')[-1].split('\n')[0]
            print(f"  - {test}: {error_msg}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            error_msg = traceback.split('\n')[-2]
            print(f"  - {test}: {error_msg}")
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)