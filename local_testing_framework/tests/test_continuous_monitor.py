"""
Tests for continuous monitoring system
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import time
import threading
from datetime import datetime, timedelta

from local_testing_framework.continuous_monitor import (
    ContinuousMonitor, Alert, AlertLevel, ProgressInfo, 
    DiagnosticSnapshot, RecoveryAction, MonitoringSession
)
from local_testing_framework.models.test_results import ResourceMetrics


class TestContinuousMonitor(unittest.TestCase):
    """Test cases for ContinuousMonitor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.monitor = ContinuousMonitor()
        self.monitor.refresh_interval = 0.1  # Fast interval for testing
        self.monitor.stability_check_interval = 0.2  # Fast stability checks
    
    def tearDown(self):
        """Clean up after tests"""
        self.monitor.cleanup_resources()
    
    def test_initialization(self):
        """Test monitor initialization"""
        self.assertIsNotNone(self.monitor.config)
        self.assertEqual(self.monitor.refresh_interval, 0.1)
        self.assertIsNone(self.monitor.current_session)
        self.assertFalse(self.monitor.stop_monitoring.is_set())
    
    @patch('local_testing_framework.continuous_monitor.psutil.cpu_percent')
    @patch('local_testing_framework.continuous_monitor.psutil.virtual_memory')
    def test_collect_resource_metrics(self, mock_memory, mock_cpu):
        """Test resource metrics collection"""
        # Mock system metrics
        mock_cpu.return_value = 45.5
        mock_memory.return_value = Mock(
            percent=60.2,
            used=8 * 1024**3,  # 8GB
            total=16 * 1024**3  # 16GB
        )
        
        metrics = self.monitor._collect_resource_metrics()
        
        self.assertIsInstance(metrics, ResourceMetrics)
        self.assertEqual(metrics.cpu_percent, 45.5)
        self.assertEqual(metrics.memory_percent, 60.2)
        self.assertEqual(metrics.memory_used_gb, 8.0)
        self.assertEqual(metrics.memory_total_gb, 16.0)
    
    def test_check_thresholds(self):
        """Test threshold checking and alert generation"""
        # Create metrics that exceed thresholds
        high_metrics = ResourceMetrics(
            cpu_percent=85.0,  # Above 80% threshold
            memory_percent=85.0,  # Above 80% threshold
            memory_used_gb=12.0,
            memory_total_gb=16.0,
            gpu_percent=50.0,
            vram_used_mb=11000,  # 11GB
            vram_total_mb=12000,  # 12GB total
            vram_percent=91.7  # Above 90% threshold
        )
        
        alerts = self.monitor._check_thresholds(high_metrics)
        
        # Should generate 3 alerts (CPU, Memory, VRAM)
        self.assertEqual(len(alerts), 3)
        
        # Check alert types
        alert_metrics = [alert.metric_name for alert in alerts]
        self.assertIn("cpu_percent", alert_metrics)
        self.assertIn("memory_percent", alert_metrics)
        self.assertIn("vram_percent", alert_metrics)
        
        # Check alert levels
        for alert in alerts:
            self.assertEqual(alert.level, AlertLevel.WARNING)
    
    def test_start_stop_monitoring_session(self):
        """Test starting and stopping monitoring sessions"""
        session_id = "test_session_001"
        
        # Start monitoring
        session = self.monitor.start_monitoring(session_id)
        
        self.assertIsNotNone(session)
        self.assertEqual(session.session_id, session_id)
        self.assertTrue(session.is_active)
        self.assertIsNotNone(self.monitor.monitoring_thread)
        self.assertTrue(self.monitor.monitoring_thread.is_alive())
        
        # Let it run briefly
        time.sleep(0.3)
        
        # Stop monitoring
        stopped_session = self.monitor.stop_monitoring_session()
        
        self.assertIsNotNone(stopped_session)
        self.assertEqual(stopped_session.session_id, session_id)
        self.assertFalse(stopped_session.is_active)
        self.assertIsNotNone(stopped_session.end_time)
        
        # Should have collected some metrics
        self.assertGreater(len(stopped_session.metrics_history), 0)
    
    def test_progress_tracking(self):
        """Test progress tracking with ETA calculation"""
        session_id = "progress_test"
        self.monitor.start_monitoring(session_id)
        
        start_time = datetime.now()
        
        # Update progress
        progress = self.monitor.update_progress(25, 100, start_time)
        
        self.assertEqual(progress.current_step, 25)
        self.assertEqual(progress.total_steps, 100)
        self.assertEqual(progress.percentage, 25.0)
        self.assertIsNotNone(progress.eta_seconds)
        
        self.monitor.stop_monitoring_session()
    
    def test_alert_callbacks(self):
        """Test alert callback system"""
        callback_called = threading.Event()
        received_alert = None
        
        def alert_callback(alert: Alert):
            nonlocal received_alert
            received_alert = alert
            callback_called.set()
        
        self.monitor.add_alert_callback(alert_callback)
        
        # Create a test alert
        test_alert = Alert(
            level=AlertLevel.WARNING,
            message="Test alert",
            metric_name="test_metric",
            current_value=85.0,
            threshold_value=80.0
        )
        
        # Trigger callback
        self.monitor._trigger_alert_callbacks(test_alert)
        
        # Wait for callback
        self.assertTrue(callback_called.wait(timeout=1.0))
        self.assertIsNotNone(received_alert)
        self.assertEqual(received_alert.message, "Test alert")
    
    def test_progress_callbacks(self):
        """Test progress callback system"""
        callback_called = threading.Event()
        received_progress = None
        
        def progress_callback(progress: ProgressInfo):
            nonlocal received_progress
            received_progress = progress
            callback_called.set()
        
        self.monitor.add_progress_callback(progress_callback)
        
        # Start session and update progress
        self.monitor.start_monitoring("callback_test")
        progress = self.monitor.update_progress(50, 100)
        
        # Wait for callback
        self.assertTrue(callback_called.wait(timeout=1.0))
        self.assertIsNotNone(received_progress)
        self.assertEqual(received_progress.percentage, 50.0)
        
        self.monitor.stop_monitoring_session()
    
    @patch('local_testing_framework.continuous_monitor.torch')
    def test_capture_diagnostic_snapshot(self, mock_torch):
        """Test diagnostic snapshot capture"""
        # Mock torch CUDA functions
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 8 * 1024**3  # 8GB
        mock_torch.cuda.memory_reserved.return_value = 10 * 1024**3  # 10GB
        mock_torch.cuda.max_memory_allocated.return_value = 9 * 1024**3
        mock_torch.cuda.max_memory_reserved.return_value = 11 * 1024**3
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.current_device.return_value = 0
        
        snapshot = self.monitor._capture_diagnostic_snapshot()
        
        self.assertIsInstance(snapshot, DiagnosticSnapshot)
        self.assertIsInstance(snapshot.gpu_memory_state, dict)
        self.assertIsInstance(snapshot.system_processes, list)
        self.assertIsInstance(snapshot.system_logs, list)
        self.assertIsInstance(snapshot.disk_usage, dict)
        self.assertIsInstance(snapshot.network_stats, dict)
        
        # Check GPU memory state
        self.assertEqual(snapshot.gpu_memory_state["allocated_mb"], 8 * 1024)
        self.assertEqual(snapshot.gpu_memory_state["device_count"], 1)
    
    @patch('local_testing_framework.continuous_monitor.torch')
    def test_gpu_cache_recovery(self, mock_torch):
        """Test GPU cache recovery action"""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.empty_cache = Mock()
        
        recovery = self.monitor._attempt_gpu_cache_recovery()
        
        self.assertIsInstance(recovery, RecoveryAction)
        self.assertEqual(recovery.action_name, "clear_gpu_cache")
        self.assertTrue(recovery.success)
        mock_torch.cuda.empty_cache.assert_called_once()
    
    def test_memory_cleanup_recovery(self):
        """Test memory cleanup recovery action"""
        recovery = self.monitor._attempt_memory_cleanup()
        
        self.assertIsInstance(recovery, RecoveryAction)
        self.assertEqual(recovery.action_name, "memory_cleanup")
        self.assertTrue(recovery.success)
        self.assertIn("Collected", recovery.message)
    
    def test_service_restart_recovery(self):
        """Test service restart recovery action"""
        recovery = self.monitor._attempt_service_restart()
        
        self.assertIsInstance(recovery, RecoveryAction)
        self.assertEqual(recovery.action_name, "service_restart")
        self.assertTrue(recovery.success)
    
    @patch('local_testing_framework.continuous_monitor.ContinuousMonitor._collect_resource_metrics')
    def test_system_stability_check(self, mock_collect):
        """Test system stability checking"""
        # Mock critical resource usage
        critical_metrics = ResourceMetrics(
            cpu_percent=96.0,  # Critical
            memory_percent=96.0,  # Critical
            memory_used_gb=15.0,
            memory_total_gb=16.0,
            gpu_percent=50.0,
            vram_used_mb=11500,
            vram_total_mb=12000,
            vram_percent=96.0  # Critical
        )
        mock_collect.return_value = critical_metrics
        
        # Start session to capture alerts
        self.monitor.start_monitoring("stability_test")
        
        alerts = self.monitor._check_system_stability()
        
        # Should generate critical alerts
        self.assertGreater(len(alerts), 0)
        
        # Check for critical alerts
        critical_alerts = [a for a in alerts if a.level == AlertLevel.CRITICAL]
        self.assertGreater(len(critical_alerts), 0)
        
        # Should have triggered recovery actions
        session = self.monitor.current_session
        self.assertGreater(len(session.recovery_actions), 0)
        self.assertGreater(len(session.diagnostic_snapshots), 0)
        
        self.monitor.stop_monitoring_session()
    
    def test_force_diagnostic_snapshot(self):
        """Test forced diagnostic snapshot capture"""
        self.monitor.start_monitoring("snapshot_test")
        
        snapshot = self.monitor.force_diagnostic_snapshot()
        
        self.assertIsInstance(snapshot, DiagnosticSnapshot)
        
        # Should be added to session
        session = self.monitor.current_session
        self.assertIn(snapshot, session.diagnostic_snapshots)
        
        self.monitor.stop_monitoring_session()
    
    def test_trigger_recovery_procedures(self):
        """Test manual recovery procedure triggering"""
        self.monitor.start_monitoring("recovery_test")
        
        actions = self.monitor.trigger_recovery_procedures()
        
        self.assertEqual(len(actions), 3)  # GPU cache, memory cleanup, service restart
        
        # Check action types
        action_names = [action.action_name for action in actions]
        self.assertIn("clear_gpu_cache", action_names)
        self.assertIn("memory_cleanup", action_names)
        self.assertIn("service_restart", action_names)
        
        # Should be added to session
        session = self.monitor.current_session
        self.assertEqual(len(session.recovery_actions), 3)
        
        self.monitor.stop_monitoring_session()
    
    def test_generate_monitoring_report(self):
        """Test comprehensive monitoring report generation"""
        self.monitor.start_monitoring("report_test")
        
        # Let it collect some data
        time.sleep(0.3)
        
        # Add some test data
        test_alert = Alert(
            level=AlertLevel.WARNING,
            message="Test threshold violation",
            metric_name="cpu_percent",
            current_value=85.0,
            threshold_value=80.0
        )
        self.monitor.current_session.alerts_history.append(test_alert)
        
        # Force a diagnostic snapshot
        self.monitor.force_diagnostic_snapshot()
        
        # Trigger recovery
        self.monitor.trigger_recovery_procedures()
        
        report = self.monitor.generate_monitoring_report()
        
        self.assertIsInstance(report, dict)
        self.assertIn("session_info", report)
        self.assertIn("timeline", report)
        self.assertIn("threshold_violations", report)
        self.assertIn("stability_events", report)
        self.assertIn("recovery_summary", report)
        
        # Check timeline has data
        self.assertGreater(len(report["timeline"]), 0)
        
        # Check threshold violations
        self.assertEqual(len(report["threshold_violations"]), 1)
        
        # Check stability events
        self.assertEqual(len(report["stability_events"]), 1)
        
        # Check recovery summary
        self.assertGreater(len(report["recovery_summary"]), 0)
        
        self.monitor.stop_monitoring_session()
    
    def test_get_session_summary(self):
        """Test session summary generation"""
        self.monitor.start_monitoring("summary_test")
        
        # Let it collect some data
        time.sleep(0.2)
        
        summary = self.monitor.get_session_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn("session_id", summary)
        self.assertIn("start_time", summary)
        self.assertIn("is_active", summary)
        self.assertIn("metrics_count", summary)
        self.assertIn("latest_metrics", summary)
        self.assertIn("averages", summary)
        
        self.monitor.stop_monitoring_session()
    
    def test_get_current_metrics_without_session(self):
        """Test getting current metrics without active session"""
        metrics = self.monitor.get_current_metrics()
        
        self.assertIsInstance(metrics, ResourceMetrics)
        self.assertGreaterEqual(metrics.cpu_percent, 0.0)
        self.assertGreaterEqual(metrics.memory_percent, 0.0)


if __name__ == '__main__':
    unittest.main()