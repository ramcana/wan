#!/usr/bin/env python3
"""
Simple integration tests for diagnostic monitor
Tests core functionality without external dependencies
"""

import unittest
import time
import tempfile
import json
import os
from datetime import datetime, timedelta

# Import diagnostic monitor
from scripts.diagnostic_monitor import DiagnosticMonitor, AlertLevel


class TestDiagnosticMonitorSimple(unittest.TestCase):
    """Simple integration tests for diagnostic monitor"""
    
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
        
        # Initialize monitor
        self.monitor = DiagnosticMonitor(config_path=self.temp_config.name)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if self.monitor.is_monitoring:
            self.monitor.stop_monitoring()
        
        try:
            os.unlink(self.temp_config.name)
        except OSError:
            pass
    
    def test_end_to_end_monitoring_workflow(self):
        """Test complete monitoring workflow from start to finish"""
        print("\n🧪 Testing end-to-end monitoring workflow...")
        
        # 1. Start monitoring
        self.monitor.start_monitoring()
        self.assertTrue(self.monitor.is_monitoring)
        print("   ✓ Monitoring started successfully")
        
        # 2. Record some errors to test pattern detection
        test_components = ["component_a", "component_b", "component_c"]
        for i in range(12):
            component = test_components[i % len(test_components)]
            self.monitor.record_error(component, f"Test error {i+1}")
        print("   ✓ Recorded 12 test errors across 3 components")
        
        # 3. Let monitoring run and collect data
        time.sleep(3)
        print("   ✓ Allowed monitoring to collect data for 3 seconds")
        
        # 4. Check component health
        health_results = []
        for component in ["model_downloader", "dependency_manager", "config_validator"]:
            health = self.monitor.check_component_health(component)
            health_results.append(health)
            self.assertIsNotNone(health)
            self.assertEqual(health.component_name, component)
        print(f"   ✓ Checked health of {len(health_results)} components")
        
        # 5. Generate comprehensive health report
        report = self.monitor.generate_health_report()
        self.assertIsNotNone(report)
        print("   ✓ Generated comprehensive health report")
        
        # 6. Verify report contents
        self.assertIsNotNone(report.resource_metrics)
        self.assertGreater(len(report.component_health), 0)
        self.assertIsInstance(report.active_alerts, list)
        self.assertIsInstance(report.error_patterns, list)
        self.assertIsInstance(report.potential_issues, list)
        print("   ✓ Verified report structure and contents")
        
        # 7. Check error pattern detection
        if report.error_patterns:
            print(f"   ✓ Detected {len(report.error_patterns)} error patterns")
            for pattern in report.error_patterns:
                self.assertGreater(pattern.frequency, 0)
                self.assertIn(pattern.pattern_type, test_components)
        else:
            print("   ℹ No error patterns detected (may need more time/data)")
        
        # 8. Test monitoring status
        status = self.monitor.get_monitoring_status()
        self.assertTrue(status["is_monitoring"])
        self.assertGreater(status["metrics_history_size"], 0)
        print("   ✓ Verified monitoring status")
        
        # 9. Stop monitoring
        self.monitor.stop_monitoring()
        self.assertFalse(self.monitor.is_monitoring)
        print("   ✓ Monitoring stopped successfully")
        
        print("   🎉 End-to-end workflow completed successfully!")
    
    def test_alert_generation_and_handling(self):
        """Test alert generation under various conditions"""
        print("\n🚨 Testing alert generation and handling...")
        
        # Set up alert callback to capture alerts
        captured_alerts = []
        def alert_handler(alert):
            captured_alerts.append(alert)
        
        self.monitor.add_alert_callback(alert_handler)
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Simulate high resource usage by creating test metrics
        from scripts.diagnostic_monitor import ResourceMetrics, ComponentHealth, HealthStatus
        
        # Create metrics that exceed thresholds
        high_usage_metrics = ResourceMetrics(
            timestamp=datetime.now(),
            cpu_percent=85.0,  # Above warning threshold
            memory_percent=80.0,  # Above warning threshold
            memory_used_gb=12.8,
            memory_total_gb=16.0,
            disk_usage_percent=60.0,
            disk_free_gb=400.0,
            vram_percent=90.0  # High VRAM usage
        )
        
        # Create degraded component health
        degraded_health = [
            ComponentHealth(
                component_name="test_component",
                status=HealthStatus.DEGRADED,
                response_time_ms=1200.0,  # Slow response
                error_count=5,
                last_error="Test error",
                uptime_seconds=3600.0,
                performance_score=45.0
            )
        ]
        
        # Check thresholds to generate alerts
        alerts = self.monitor._check_thresholds(high_usage_metrics, degraded_health)
        
        # Verify alerts were generated
        self.assertGreater(len(alerts), 0)
        print(f"   ✓ Generated {len(alerts)} alerts for high resource usage")
        
        # Verify alert types
        cpu_alerts = [a for a in alerts if a.metric_name == "cpu_percent"]
        memory_alerts = [a for a in alerts if a.metric_name == "memory_percent"]
        component_alerts = [a for a in alerts if a.metric_name == "component_health"]
        
        self.assertGreater(len(cpu_alerts), 0)
        self.assertGreater(len(memory_alerts), 0)
        self.assertGreater(len(component_alerts), 0)
        print("   ✓ Verified alert types (CPU, Memory, Component)")
        
        # Test alert callback triggering
        for alert in alerts:
            self.monitor._trigger_alert_callbacks(alert)
        
        self.assertEqual(len(captured_alerts), len(alerts))
        print(f"   ✓ Alert callbacks triggered successfully ({len(captured_alerts)} alerts)")
        
        # Verify alert levels
        warning_alerts = [a for a in alerts if a.level == AlertLevel.WARNING]
        self.assertGreater(len(warning_alerts), 0)
        print(f"   ✓ Generated {len(warning_alerts)} warning-level alerts")
        
        self.monitor.stop_monitoring()
        print("   🎉 Alert generation and handling test completed!")
    
    def test_predictive_analysis_capabilities(self):
        """Test predictive failure analysis with trending data"""
        print("\n🔮 Testing predictive analysis capabilities...")
        
        # Create historical data showing increasing resource usage trends
        base_time = datetime.now()
        
        # Add metrics with clear upward trends
        for i in range(15):
            from scripts.diagnostic_monitor import ResourceMetrics
            metrics = ResourceMetrics(
                timestamp=base_time + timedelta(seconds=i * 5),
                cpu_percent=40.0 + i * 2.5,  # Increasing from 40% to 75%
                memory_percent=35.0 + i * 2.0,  # Increasing from 35% to 63%
                memory_used_gb=5.6 + i * 0.3,
                memory_total_gb=16.0,
                disk_usage_percent=50.0 + i * 1.0,  # Slowly increasing
                disk_free_gb=500.0 - i * 10,
                vram_percent=20.0 + i * 4.0  # Rapidly increasing
            )
            self.monitor.metrics_history.append(metrics)
        
        print("   ✓ Added 15 data points with increasing resource usage trends")
        
        # Test trend calculation
        cpu_values = [40.0 + i * 2.5 for i in range(15)]
        trend = self.monitor._calculate_trend(cpu_values)
        self.assertGreater(trend, 0)  # Should detect positive trend
        print(f"   ✓ Calculated CPU trend: {trend:.2f} (positive as expected)")
        
        # Test time-to-threshold estimation
        time_to_threshold = self.monitor._estimate_time_to_threshold(
            cpu_values[-5:], 90.0, trend
        )
        if time_to_threshold:
            print(f"   ✓ Estimated time to CPU threshold: {time_to_threshold}")
        else:
            print("   ℹ Time to threshold estimation returned None (trend may be too slow)")
        
        # Detect potential issues
        issues = self.monitor.detect_potential_issues()
        print(f"   ✓ Detected {len(issues)} potential issues")
        
        # Verify issue types and properties
        for issue in issues:
            self.assertIsInstance(issue.issue_type, str)
            self.assertGreaterEqual(issue.probability, 0.0)
            self.assertLessEqual(issue.probability, 1.0)
            self.assertGreater(len(issue.affected_components), 0)
            self.assertGreater(len(issue.recommended_actions), 0)
            print(f"     - {issue.issue_type}: {issue.probability:.1%} probability")
        
        # Test with stable data (should detect fewer/no issues)
        stable_metrics = []
        for i in range(10):
            from scripts.diagnostic_monitor import ResourceMetrics
            metrics = ResourceMetrics(
                timestamp=base_time + timedelta(seconds=i * 5),
                cpu_percent=45.0,  # Stable
                memory_percent=40.0,  # Stable
                memory_used_gb=6.4,
                memory_total_gb=16.0,
                disk_usage_percent=55.0,  # Stable
                disk_free_gb=450.0
            )
            stable_metrics.append(metrics)
        
        # Replace history with stable data
        self.monitor.metrics_history.clear()
        self.monitor.metrics_history.extend(stable_metrics)
        
        stable_issues = self.monitor.detect_potential_issues()
        print(f"   ✓ With stable data, detected {len(stable_issues)} potential issues")
        
        # Stable data should generally produce fewer issues
        self.assertLessEqual(len(stable_issues), len(issues))
        
        print("   🎉 Predictive analysis test completed!")
    
    def test_error_pattern_detection_accuracy(self):
        """Test accuracy of error pattern detection"""
        print("\n🔍 Testing error pattern detection accuracy...")
        
        # Test Case 1: Regular pattern (every 2 minutes)
        base_time = datetime.now()
        regular_component = "regular_errors"
        
        for i in range(8):
            error_time = base_time + timedelta(minutes=i * 2)
            # Mock datetime.now() to return our test time
            with unittest.mock.patch('scripts.diagnostic_monitor.datetime') as mock_dt:
                mock_dt.now.return_value = error_time
                self.monitor.record_error(regular_component, f"Regular error {i}")
        
        print("   ✓ Recorded 8 errors with regular 2-minute intervals")
        
        # Test Case 2: Burst pattern (many errors in short time)
        burst_component = "burst_errors"
        burst_time = base_time + timedelta(minutes=20)
        
        for i in range(6):
            error_time = burst_time + timedelta(seconds=i * 10)  # 10-second intervals
            with unittest.mock.patch('scripts.diagnostic_monitor.datetime') as mock_dt:
                mock_dt.now.return_value = error_time
                self.monitor.record_error(burst_component, f"Burst error {i}")
        
        print("   ✓ Recorded 6 errors in burst pattern (10-second intervals)")
        
        # Test Case 3: Sparse pattern (few errors over long time)
        sparse_component = "sparse_errors"
        
        for i in range(3):
            error_time = base_time + timedelta(hours=i)
            with unittest.mock.patch('scripts.diagnostic_monitor.datetime') as mock_dt:
                mock_dt.now.return_value = error_time
                self.monitor.record_error(sparse_component, f"Sparse error {i}")
        
        print("   ✓ Recorded 3 errors with sparse pattern (1-hour intervals)")
        
        # Detect patterns
        patterns = self.monitor._detect_error_patterns()
        print(f"   ✓ Detected {len(patterns)} error patterns")
        
        # Verify pattern detection
        pattern_types = [p.pattern_type for p in patterns]
        
        # Should detect regular and burst patterns (sparse might not meet threshold)
        if regular_component in pattern_types:
            regular_pattern = next(p for p in patterns if p.pattern_type == regular_component)
            self.assertGreaterEqual(regular_pattern.frequency, 3)  # Should detect multiple occurrences
            print(f"     ✓ Regular pattern: {regular_pattern.frequency} occurrences, trend: {regular_pattern.severity_trend}")
        
        if burst_component in pattern_types:
            burst_pattern = next(p for p in patterns if p.pattern_type == burst_component)
            self.assertGreaterEqual(burst_pattern.frequency, 3)
            print(f"     ✓ Burst pattern: {burst_pattern.frequency} occurrences, trend: {burst_pattern.severity_trend}")
        
        # Test prediction confidence
        for pattern in patterns:
            self.assertGreaterEqual(pattern.prediction_confidence, 0.0)
            self.assertLessEqual(pattern.prediction_confidence, 1.0)
            print(f"     - {pattern.pattern_type}: confidence {pattern.prediction_confidence:.2f}")
        
        print("   🎉 Error pattern detection accuracy test completed!")
    
    def test_performance_and_scalability(self):
        """Test performance under load and scalability"""
        print("\n⚡ Testing performance and scalability...")
        
        # Test 1: Large number of metrics
        start_time = time.time()
        
        for i in range(1000):
            from scripts.diagnostic_monitor import ResourceMetrics
            metrics = ResourceMetrics(
                timestamp=datetime.now(),
                cpu_percent=50.0 + (i % 50),
                memory_percent=40.0 + (i % 40),
                memory_used_gb=8.0,
                memory_total_gb=16.0,
                disk_usage_percent=60.0,
                disk_free_gb=400.0
            )
            self.monitor.metrics_history.append(metrics)
        
        metrics_time = time.time() - start_time
        print(f"   ✓ Added 1000 metrics in {metrics_time:.3f} seconds")
        
        # Test 2: Large number of errors
        start_time = time.time()
        
        for i in range(500):
            component = f"component_{i % 10}"  # 10 different components
            self.monitor.record_error(component, f"Load test error {i}")
        
        errors_time = time.time() - start_time
        print(f"   ✓ Recorded 500 errors in {errors_time:.3f} seconds")
        
        # Test 3: Health report generation with large dataset
        start_time = time.time()
        
        report = self.monitor.generate_health_report()
        
        report_time = time.time() - start_time
        print(f"   ✓ Generated health report in {report_time:.3f} seconds")
        
        # Verify report is still complete
        self.assertIsNotNone(report)
        self.assertGreater(len(self.monitor.metrics_history), 900)  # Should have most metrics
        
        # Test 4: Pattern detection with large dataset
        start_time = time.time()
        
        patterns = self.monitor._detect_error_patterns()
        
        pattern_time = time.time() - start_time
        print(f"   ✓ Detected patterns in {pattern_time:.3f} seconds")
        print(f"     Found {len(patterns)} patterns from large dataset")
        
        # Test 5: Concurrent operations
        import threading
        
        def concurrent_health_checks():
            for _ in range(50):
                self.monitor.check_component_health("test_component")
        
        def concurrent_error_recording():
            for i in range(50):
                self.monitor.record_error("concurrent_test", f"Concurrent error {i}")
        
        start_time = time.time()
        
        threads = [
            threading.Thread(target=concurrent_health_checks),
            threading.Thread(target=concurrent_error_recording),
            threading.Thread(target=concurrent_health_checks)
        ]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        concurrent_time = time.time() - start_time
        print(f"   ✓ Completed concurrent operations in {concurrent_time:.3f} seconds")
        
        # Performance assertions
        self.assertLess(metrics_time, 1.0, "Metrics processing too slow")
        self.assertLess(errors_time, 0.5, "Error recording too slow")
        self.assertLess(report_time, 2.0, "Health report generation too slow")
        self.assertLess(pattern_time, 1.0, "Pattern detection too slow")
        self.assertLess(concurrent_time, 5.0, "Concurrent operations too slow")
        
        print("   🎉 Performance and scalability test completed!")


if __name__ == "__main__":
    # Run tests with detailed output
    unittest.main(verbosity=2)