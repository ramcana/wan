"""
Unit tests for PerformanceTester components
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import time

from ..performance_tester import (
    MetricsCollector, BenchmarkRunner, PerformanceTester,
    OptimizationValidator, PerformanceTargetValidator,
    FrameworkOverheadMonitor, OptimizationRecommendationSystem
)
from ..models.test_results import ValidationStatus
from ..models.configuration import LocalTestConfiguration, PerformanceTargets


class TestMetricsCollector(unittest.TestCase):
    """Test cases for MetricsCollector"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.collector = MetricsCollector()
    
    def test_metrics_collector_initialization(self):
        """Test MetricsCollector initialization"""
        self.assertEqual(len(self.collector.metrics), 0)
        self.assertFalse(self.collector.monitoring)
        self.assertIsNone(self.collector.monitor_thread)
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_start_stop_monitoring(self, mock_memory, mock_cpu):
        """Test starting and stopping metrics collection"""
        # Mock system metrics
        mock_cpu.return_value = 25.0
        mock_memory.return_value = Mock(
            used=4 * 1024**3,  # 4GB
            percent=50.0,
            available=4 * 1024**3
        )
        
        # Start monitoring
        self.collector.start_monitoring(interval=0.1)
        self.assertTrue(self.collector.monitoring)
        
        # Let it collect a few samples
        time.sleep(0.3)
        
        # Stop monitoring
        self.collector.stop_monitoring()
        self.assertFalse(self.collector.monitoring)
        
        # Should have collected some metrics
        self.assertGreater(len(self.collector.metrics), 0)
    
    def test_get_summary_stats_empty(self):
        """Test summary stats with no metrics"""
        stats = self.collector.get_summary_stats()
        self.assertEqual(stats, {})
    
    def test_get_summary_stats_with_data(self):
        """Test summary stats calculation"""
        # Add mock metrics
        self.collector.metrics = [
            {
                "cpu_percent": 20.0,
                "memory_used_gb": 4.0,
                "gpu_memory_used_gb": 2.0
            },
            {
                "cpu_percent": 30.0,
                "memory_used_gb": 5.0,
                "gpu_memory_used_gb": 3.0
            }
        ]
        
        stats = self.collector.get_summary_stats()
        
        self.assertEqual(stats["duration_seconds"], 2)
        self.assertEqual(stats["cpu_avg"], 25.0)
        self.assertEqual(stats["cpu_peak"], 30.0)
        self.assertEqual(stats["memory_avg_gb"], 4.5)
        self.assertEqual(stats["memory_peak_gb"], 5.0)


class TestBenchmarkRunner(unittest.TestCase):
    """Test cases for BenchmarkRunner"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.runner = BenchmarkRunner()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('subprocess.run')
    def test_run_performance_profiler_benchmark_success(self, mock_run):
        """Test successful performance profiler benchmark"""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Benchmark completed successfully",
            stderr=""
        )
        
        result = self.runner.run_performance_profiler_benchmark()
        
        self.assertTrue(result["success"])
        self.assertEqual(result["return_code"], 0)
        self.assertTrue(result["profiler_available"])
    
    @patch('subprocess.run')
    def test_run_performance_profiler_benchmark_not_found(self, mock_run):
        """Test performance profiler benchmark when file not found"""
        mock_run.side_effect = FileNotFoundError()
        
        result = self.runner.run_performance_profiler_benchmark()
        
        self.assertFalse(result["success"])
        self.assertFalse(result["profiler_available"])
        self.assertIn("not found", result["error"])
    
    @patch('subprocess.run')
    def test_run_performance_profiler_benchmark_timeout(self, mock_run):
        """Test performance profiler benchmark timeout"""
        import subprocess
mock_run.side_effect = subprocess.TimeoutExpired("cmd", 1800)
        
        result = self.runner.run_performance_profiler_benchmark()
        
        self.assertFalse(result["success"])
        self.assertTrue(result["profiler_available"])
        self.assertIn("timed out", result["error"])


class TestOptimizationValidator(unittest.TestCase):
    """Test cases for OptimizationValidator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validator = OptimizationValidator()
    
    @patch('subprocess.run')
    def test_test_vram_optimization_success(self, mock_run):
        """Test successful VRAM optimization test"""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="VRAM optimization test passed",
            stderr=""
        )
        
        result = self.validator.test_vram_optimization()
        
        self.assertEqual(result.status, ValidationStatus.PASSED)
        self.assertEqual(result.component, "vram_optimization")
    
    @patch('subprocess.run')
    def test_test_vram_optimization_failure(self, mock_run):
        """Test failed VRAM optimization test"""
        mock_run.return_value = Mock(
            returncode=1,
            stdout="",
            stderr="VRAM optimization failed"
        )
        
        result = self.validator.test_vram_optimization()
        
        self.assertEqual(result.status, ValidationStatus.FAILED)
        self.assertEqual(result.component, "vram_optimization")
        self.assertTrue(len(result.remediation_steps) > 0)
    
    def test_validate_vram_reduction_success(self):
        """Test successful VRAM reduction validation"""
        baseline_vram = 10.0  # 10GB
        optimized_vram = 2.0  # 2GB (80% reduction)
        
        result = self.validator.validate_vram_reduction(baseline_vram, optimized_vram)
        
        self.assertEqual(result.status, ValidationStatus.PASSED)
        self.assertEqual(result.component, "vram_reduction")
        self.assertEqual(result.details["reduction_percent"], 80.0)
    
    def test_validate_vram_reduction_insufficient(self):
        """Test insufficient VRAM reduction validation"""
        baseline_vram = 10.0  # 10GB
        optimized_vram = 6.0   # 6GB (40% reduction, less than 80% target)
        
        result = self.validator.validate_vram_reduction(baseline_vram, optimized_vram)
        
        self.assertEqual(result.status, ValidationStatus.FAILED)
        self.assertEqual(result.component, "vram_reduction")
        self.assertEqual(result.details["reduction_percent"], 40.0)
    
    def test_validate_vram_reduction_invalid_baseline(self):
        """Test VRAM reduction validation with invalid baseline"""
        result = self.validator.validate_vram_reduction(0.0, 2.0)
        
        self.assertEqual(result.status, ValidationStatus.FAILED)
        self.assertIn("Invalid baseline", result.message)


class TestPerformanceTargetValidator(unittest.TestCase):
    """Test cases for PerformanceTargetValidator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.validator = PerformanceTargetValidator()
    
    def test_validate_720p_target_success(self):
        """Test successful 720p target validation"""
        result = self.validator.validate_720p_target(8.0, 10.0)  # 8min, 10GB VRAM
        
        self.assertEqual(result.status, ValidationStatus.PASSED)
        self.assertEqual(result.component, "720p_performance")
    
    def test_validate_720p_target_failure(self):
        """Test failed 720p target validation"""
        result = self.validator.validate_720p_target(12.0, 15.0)  # 12min (>9min), 15GB (>12GB)
        
        self.assertEqual(result.status, ValidationStatus.FAILED)
        self.assertEqual(result.component, "720p_performance")
        self.assertTrue(len(result.remediation_steps) > 0)
    
    def test_validate_1080p_target_success(self):
        """Test successful 1080p target validation"""
        result = self.validator.validate_1080p_target(15.0, 11.0)  # 15min, 11GB VRAM
        
        self.assertEqual(result.status, ValidationStatus.PASSED)
        self.assertEqual(result.component, "1080p_performance")
    
    def test_validate_all_targets(self):
        """Test validation of all targets from test results"""
        test_results = {
            "720p": {
                "duration_minutes": 8.0,
                "metrics": {"gpu_memory_peak_gb": 10.0}
            },
            "1080p": {
                "duration_minutes": 16.0,
                "metrics": {"gpu_memory_peak_gb": 11.5}
            }
        }
        
        validations = self.validator.validate_all_targets(test_results)
        
        self.assertEqual(len(validations), 2)
        self.assertEqual(validations[0].component, "720p_performance")
        self.assertEqual(validations[1].component, "1080p_performance")


class TestOptimizationRecommendationSystem(unittest.TestCase):
    """Test cases for OptimizationRecommendationSystem"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.system = OptimizationRecommendationSystem()
    
    def test_analyze_performance_issues(self):
        """Test performance issue analysis"""
        test_results = {
            "tests": {
                "720p": {
                    "duration_minutes": 10.0,  # Above 9min target
                    "metrics": {"gpu_memory_peak_gb": 11.0}  # High VRAM usage
                }
            }
        }
        
        recommendations = self.system.analyze_performance_issues(test_results)
        
        self.assertIn("vram_optimizations", recommendations)
        self.assertIn("speed_optimizations", recommendations)
        self.assertIn("configuration_changes", recommendations)
        self.assertTrue(len(recommendations["vram_optimizations"]) > 0)
        self.assertTrue(len(recommendations["speed_optimizations"]) > 0)
    
    def test_generate_optimization_config(self):
        """Test optimization configuration generation"""
        recommendations = {
            "vram_optimizations": ["Enable attention slicing"],
            "speed_optimizations": ["Use faster schedulers"]
        }
        
        config = self.system.generate_optimization_config(recommendations)
        
        self.assertIn("optimization", config)
        self.assertIn("performance", config)
        self.assertIn("system", config)
        self.assertTrue(config["optimization"]["enable_attention_slicing"])
        self.assertTrue(config["optimization"]["enable_vae_tiling"])
    
    def test_create_detailed_performance_report(self):
        """Test detailed performance report creation"""
        test_results = {
            "test_session_id": "test_123",
            "overall_status": "passed",
            "total_duration_minutes": 25.0,
            "tests": {
                "720p": {
                    "duration_minutes": 8.0,
                    "success": True,
                    "metrics": {"gpu_memory_peak_gb": 10.0}
                }
            },
            "validations": {
                "720p": {
                    "status": "passed",
                    "message": "Target met"
                }
            }
        }
        
        recommendations = {
            "vram_optimizations": ["Enable attention slicing"],
            "speed_optimizations": ["Use xformers"]
        }
        
        report = self.system.create_detailed_performance_report(test_results, recommendations)
        
        self.assertIn("report_metadata", report)
        self.assertIn("executive_summary", report)
        self.assertIn("performance_results", report)
        self.assertIn("optimization_analysis", report)
        self.assertIn("technical_details", report)
        self.assertIn("next_steps", report)
        
        # Check specific content
        self.assertEqual(report["report_metadata"]["test_session_id"], "test_123")
        self.assertIn("720p", report["performance_results"])


class TestPerformanceTester(unittest.TestCase):
    """Test cases for main PerformanceTester class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.tester = PerformanceTester()
    
    def test_validate_performance_targets_success(self):
        """Test successful performance target validation"""
        benchmark_results = {
            "resolution": "720p",
            "duration_minutes": 8.0,
            "success": True,
            "metrics": {"gpu_memory_peak_gb": 10.0}
        }
        
        result = self.tester.validate_performance_targets(benchmark_results)
        
        self.assertEqual(result.status, ValidationStatus.PASSED)
        self.assertEqual(result.component, "performance_targets")
    
    def test_validate_performance_targets_failure(self):
        """Test failed performance target validation"""
        benchmark_results = {
            "resolution": "720p",
            "duration_minutes": 12.0,  # Above 9min target
            "success": True,
            "metrics": {"gpu_memory_peak_gb": 15.0}  # Above 12GB limit
        }
        
        result = self.tester.validate_performance_targets(benchmark_results)
        
        self.assertEqual(result.status, ValidationStatus.FAILED)
        self.assertEqual(result.component, "performance_targets")
        self.assertTrue(len(result.remediation_steps) > 0)
    
    def test_validate_performance_targets_benchmark_failed(self):
        """Test performance target validation when benchmark failed"""
        benchmark_results = {
            "resolution": "720p",
            "success": False,
            "error": "Benchmark failed"
        }
        
        result = self.tester.validate_performance_targets(benchmark_results)
        
        self.assertEqual(result.status, ValidationStatus.FAILED)
        self.assertIn("Benchmark failed", result.message)
    
    def test_generate_performance_report(self):
        """Test performance report generation"""
        test_results = {
            "test_session_id": "test_123",
            "start_time": "2023-01-01T12:00:00",
            "total_duration_minutes": 25.0,
            "overall_status": "passed",
            "tests": {
                "720p": {
                    "success": True,
                    "duration_minutes": 8.0,
                    "metrics": {
                        "cpu_peak": 75.0,
                        "memory_peak_gb": 8.0,
                        "gpu_memory_peak_gb": 10.0
                    }
                }
            },
            "validations": {
                "720p": {
                    "status": "passed",
                    "message": "Target met"
                }
            }
        }
        
        report = self.tester.generate_performance_report(test_results)
        
        self.assertIn("Performance Test Report", report)
        self.assertIn("test_123", report)
        self.assertIn("720P", report)
        self.assertIn("8.0 minutes", report)


if __name__ == '__main__':
    unittest.main()