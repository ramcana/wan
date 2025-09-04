"""
Test suite for Performance Monitoring System

Tests performance monitoring, optimization effectiveness measurement,
and regression detection capabilities.
"""

import unittest
import time
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from performance_monitor import (
    PerformanceMonitor, PerformanceMetrics, OptimizationEffectiveness,
    RegressionAlert, get_performance_monitor
)


class TestPerformanceMonitor(unittest.TestCase):
    """Test PerformanceMonitor functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.temp_file.close()
        
        self.monitor = PerformanceMonitor(
            metrics_file=self.temp_file.name,
            max_metrics_history=100,
            regression_threshold=0.2
        )
    
    def tearDown(self):
        """Clean up test environment"""
        Path(self.temp_file.name).unlink(missing_ok=True)
    
    def test_operation_monitoring(self):
        """Test basic operation monitoring"""
        operation_id = "test_op_1"
        
        # Start monitoring
        self.monitor.start_operation(
            operation_id=operation_id,
            operation_type="generation",
            metadata={"model": "wan_t2v", "prompt": "test"}
        )
        
        # Verify operation is tracked
        self.assertIn(operation_id, self.monitor.active_operations)
        
        # Simulate some work
        time.sleep(0.1)
        
        # End monitoring
        metrics = self.monitor.end_operation(operation_id, success=True)
        
        # Verify metrics
        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertEqual(metrics.operation_id, operation_id)
        self.assertEqual(metrics.operation_type, "generation")
        self.assertTrue(metrics.success)
        self.assertGreater(metrics.duration, 0.05)  # At least 50ms
        self.assertNotIn(operation_id, self.monitor.active_operations)
        
        # Verify metrics are stored
        self.assertEqual(len(self.monitor.metrics_history), 1)

        assert True  # TODO: Add proper assertion
    
    def test_optimization_effectiveness_measurement(self):
        """Test optimization effectiveness measurement"""
        # Create baseline metrics
        baseline_metrics = PerformanceMetrics(
            operation_id="baseline",
            operation_type="generation",
            start_time=time.time(),
            end_time=time.time() + 10.0,
            duration=10.0,
            memory_peak_mb=8192,
            memory_allocated_mb=6144,
            gpu_utilization=90.0,
            cpu_utilization=50.0,
            success=True
        )
        
        # Create optimized metrics (50% faster, 25% less memory)
        optimized_metrics = PerformanceMetrics(
            operation_id="optimized",
            operation_type="generation",
            start_time=time.time(),
            end_time=time.time() + 5.0,
            duration=5.0,
            memory_peak_mb=6144,
            memory_allocated_mb=4096,
            gpu_utilization=85.0,
            cpu_utilization=45.0,
            success=True
        )
        
        # Measure effectiveness
        effectiveness = self.monitor.measure_optimization_effectiveness(
            optimization_name="cpu_offload",
            baseline_metrics=baseline_metrics,
            optimized_metrics=optimized_metrics
        )
        
        # Verify effectiveness calculation
        self.assertEqual(effectiveness.optimization_name, "cpu_offload")
        self.assertAlmostEqual(effectiveness.performance_improvement, 50.0, places=1)
        self.assertEqual(effectiveness.memory_reduction_mb, 2048)
        self.assertAlmostEqual(effectiveness.memory_reduction_percent, 25.0, places=1)
        self.assertEqual(effectiveness.stability_score, 1.0)
        self.assertGreater(effectiveness.recommendation_score, 0.5)
        
        # Verify storage
        self.assertIn("cpu_offload", self.monitor.optimization_effectiveness)

        assert True  # TODO: Add proper assertion
    
    def test_performance_summary(self):
        """Test performance summary generation"""
        # Add some test metrics
        for i in range(5):
            metrics = PerformanceMetrics(
                operation_id=f"test_{i}",
                operation_type="generation",
                start_time=time.time(),
                end_time=time.time() + (i + 1),
                duration=float(i + 1),
                memory_peak_mb=1000 + i * 100,
                memory_allocated_mb=800 + i * 80,
                gpu_utilization=80.0 + i,
                cpu_utilization=40.0 + i,
                success=True
            )
            self.monitor.metrics_history.append(metrics)
        
        # Get summary
        summary = self.monitor.get_performance_summary()
        
        # Verify summary structure
        self.assertIn("total_operations", summary)
        self.assertIn("success_rate", summary)
        self.assertIn("duration_stats", summary)
        self.assertIn("memory_stats", summary)
        
        # Verify statistics
        self.assertEqual(summary["total_operations"], 5)
        self.assertEqual(summary["success_rate"], 1.0)
        self.assertEqual(summary["duration_stats"]["mean"], 3.0)  # (1+2+3+4+5)/5
        self.assertEqual(summary["memory_stats"]["mean_peak_mb"], 1200)  # (1000+1100+1200+1300+1400)/5

        assert True  # TODO: Add proper assertion
    
    def test_regression_detection(self):
        """Test performance regression detection"""
        # Set up baseline
        self.monitor.baselines["generation"] = {
            "duration": 5.0,
            "memory_peak": 1000,
            "sample_size": 10,
            "updated_at": time.time()
        }
        
        # Add metrics showing regression (duration increased by 30%)
        for i in range(5):
            metrics = PerformanceMetrics(
                operation_id=f"regressed_{i}",
                operation_type="generation",
                start_time=time.time(),
                end_time=time.time() + 6.5,  # 30% slower than baseline
                duration=6.5,
                memory_peak_mb=1000,
                memory_allocated_mb=800,
                gpu_utilization=80.0,
                cpu_utilization=40.0,
                success=True
            )
            self.monitor.metrics_history.append(metrics)
        
        # Detect regressions
        alerts = self.monitor.detect_performance_regressions()
        
        # Verify regression detected
        self.assertGreater(len(alerts), 0)
        duration_alert = next((a for a in alerts if a.metric_name == "duration"), None)
        self.assertIsNotNone(duration_alert)
        self.assertAlmostEqual(duration_alert.regression_percent, 30.0, places=1)
        self.assertIn(duration_alert.severity, ["low", "medium", "high", "critical"])

        assert True  # TODO: Add proper assertion
    
    def test_baseline_updates(self):
        """Test baseline update functionality"""
        # Add successful metrics
        for i in range(15):  # Need at least 10 for baseline
            metrics = PerformanceMetrics(
                operation_id=f"baseline_{i}",
                operation_type="generation",
                start_time=time.time(),
                end_time=time.time() + (i % 3 + 2),  # Duration 2-4 seconds
                duration=float(i % 3 + 2),
                memory_peak_mb=1000 + (i % 2) * 100,  # Memory 1000-1100 MB
                memory_allocated_mb=800,
                gpu_utilization=80.0,
                cpu_utilization=40.0,
                success=True
            )
            self.monitor.metrics_history.append(metrics)
        
        # Update baselines
        self.monitor.update_baselines()
        
        # Verify baseline created
        self.assertIn("generation", self.monitor.baselines)
        baseline = self.monitor.baselines["generation"]
        self.assertIn("duration", baseline)
        self.assertIn("memory_peak", baseline)
        self.assertEqual(baseline["sample_size"], 15)
        
        # Verify baseline values are reasonable (median of 2,3,4 repeated)
        self.assertEqual(baseline["duration"], 3.0)  # Median of [2,3,4,2,3,4,...]

        assert True  # TODO: Add proper assertion
    
    @patch('performance_monitor.torch.cuda.is_available')
    @patch('performance_monitor.torch.cuda.memory_allocated')
    def test_memory_monitoring(self, mock_memory_allocated, mock_cuda_available):
        """Test GPU memory monitoring"""
        mock_cuda_available.return_value = True
        mock_memory_allocated.return_value = 1024 * 1024 * 1024  # 1GB in bytes
        
        memory_usage = self.monitor._get_memory_usage()
        self.assertEqual(memory_usage, 1024)  # Should return MB

        assert True  # TODO: Add proper assertion
    
    @patch('performance_monitor.GPUtil.getGPUs')
    def test_gpu_utilization_monitoring(self, mock_get_gpus):
        """Test GPU utilization monitoring"""
        mock_gpu = Mock()
        mock_gpu.load = 0.75  # 75% utilization
        mock_get_gpus.return_value = [mock_gpu]
        
        utilization = self.monitor._get_gpu_utilization()
        self.assertEqual(utilization, 75.0)

        assert True  # TODO: Add proper assertion
    
    def test_metrics_persistence(self):
        """Test saving and loading metrics"""
        # Add some metrics
        metrics = PerformanceMetrics(
            operation_id="persist_test",
            operation_type="generation",
            start_time=time.time(),
            end_time=time.time() + 5.0,
            duration=5.0,
            memory_peak_mb=1024,
            memory_allocated_mb=512,
            gpu_utilization=80.0,
            cpu_utilization=40.0,
            success=True,
            metadata={"test": "data"}
        )
        self.monitor.metrics_history.append(metrics)
        
        # Add baseline
        self.monitor.baselines["test_type"] = {
            "duration": 3.0,
            "memory_peak": 800,
            "sample_size": 5,
            "updated_at": time.time()
        }
        
        # Save metrics
        self.monitor._save_metrics()
        
        # Create new monitor and load
        new_monitor = PerformanceMonitor(
            metrics_file=self.temp_file.name,
            max_metrics_history=100
        )
        
        # Verify data loaded
        self.assertEqual(len(new_monitor.metrics_history), 1)
        loaded_metrics = new_monitor.metrics_history[0]
        self.assertEqual(loaded_metrics.operation_id, "persist_test")
        self.assertEqual(loaded_metrics.metadata["test"], "data")
        
        self.assertIn("test_type", new_monitor.baselines)
        self.assertEqual(new_monitor.baselines["test_type"]["duration"], 3.0)

        assert True  # TODO: Add proper assertion
    
    def test_error_handling(self):
        """Test error handling in monitoring"""
        operation_id = "error_test"
        
        # Start operation
        self.monitor.start_operation(operation_id, "generation")
        
        # End with error
        metrics = self.monitor.end_operation(
            operation_id, 
            success=False, 
            error_message="Test error"
        )
        
        # Verify error recorded
        self.assertFalse(metrics.success)
        self.assertEqual(metrics.error_message, "Test error")
        
        # Verify failed operations don't affect baselines
        initial_baseline_count = len(self.monitor.baselines)
        self.monitor.update_baselines()
        self.assertEqual(len(self.monitor.baselines), initial_baseline_count)


        assert True  # TODO: Add proper assertion

class TestOptimizationEffectiveness(unittest.TestCase):
    """Test OptimizationEffectiveness data class"""
    
    def test_effectiveness_serialization(self):
        """Test effectiveness serialization"""
        effectiveness = OptimizationEffectiveness(
            optimization_name="test_opt",
            baseline_duration=10.0,
            optimized_duration=7.0,
            performance_improvement=30.0,
            memory_reduction_mb=512,
            memory_reduction_percent=20.0,
            stability_score=0.95,
            recommendation_score=0.8
        )
        
        # Test serialization
        data = effectiveness.to_dict()
        self.assertEqual(data["optimization_name"], "test_opt")
        self.assertEqual(data["performance_improvement"], 30.0)
        
        # Test deserialization
        new_effectiveness = OptimizationEffectiveness(**data)
        self.assertEqual(new_effectiveness.optimization_name, "test_opt")
        self.assertEqual(new_effectiveness.performance_improvement, 30.0)


        assert True  # TODO: Add proper assertion

class TestRegressionAlert(unittest.TestCase):
    """Test RegressionAlert data class"""
    
    def test_alert_serialization(self):
        """Test alert serialization"""
        from datetime import datetime
        
        alert = RegressionAlert(
            alert_id="test_alert",
            metric_name="duration",
            current_value=8.0,
            baseline_value=5.0,
            regression_percent=60.0,
            severity="high",
            timestamp=datetime.now(),
            context={"operation": "test"}
        )
        
        # Test serialization
        data = alert.to_dict()
        self.assertEqual(data["alert_id"], "test_alert")
        self.assertEqual(data["regression_percent"], 60.0)
        self.assertEqual(data["context"]["operation"], "test")


        assert True  # TODO: Add proper assertion

class TestGlobalPerformanceMonitor(unittest.TestCase):
    """Test global performance monitor access"""
    
    def test_global_monitor_singleton(self):
        """Test global monitor singleton behavior"""
        monitor1 = get_performance_monitor()
        monitor2 = get_performance_monitor()
        
        # Should return same instance
        self.assertIs(monitor1, monitor2)


        assert True  # TODO: Add proper assertion

if __name__ == '__main__':
    unittest.main()