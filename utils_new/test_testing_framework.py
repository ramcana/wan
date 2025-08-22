#!/usr/bin/env python3
"""
Test for the Testing and Validation Framework
Validates that all testing components work correctly
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import time

# Import testing framework components
from smoke_test_runner import SmokeTestRunner, SmokeTestResult
from integration_test_suite import IntegrationTestSuite
from performance_benchmark_suite import PerformanceBenchmarkSuite
from test_coverage_validator import TestCoverageValidator
from comprehensive_test_runner import ComprehensiveTestRunner

class TestTestingFramework(unittest.TestCase):
    """Test cases for the testing and validation framework"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_smoke_test_runner_initialization(self):
        """Test SmokeTestRunner initialization"""
        runner = SmokeTestRunner()
        
        self.assertIsNotNone(runner)
        self.assertIsNotNone(runner.test_config)
        self.assertTrue(runner.test_results_dir.exists())
    
    def test_smoke_test_execution(self):
        """Test smoke test execution with mock pipeline"""
        runner = SmokeTestRunner()
        
        # Create mock pipeline
        class MockPipeline:
            def generate(self, prompt, num_frames=8, height=256, width=256):
                time.sleep(0.01)  # Quick test
                return np.random.rand(num_frames, height, width, 3).astype(np.float32)
        
        mock_pipeline = MockPipeline()
        
        # Run smoke test
        result = runner.run_pipeline_smoke_test(mock_pipeline, "test prompt")
        
        self.assertIsInstance(result, SmokeTestResult)
        self.assertTrue(result.success)
        self.assertGreater(result.generation_time, 0)
        # Note: Default test uses 320x320, not 256x256
        self.assertEqual(result.output_shape, (8, 320, 320, 3))
    
    def test_output_format_validation(self):
        """Test output format validation"""
        runner = SmokeTestRunner()
        
        # Test with numpy array
        test_output = np.random.rand(8, 256, 256, 3).astype(np.float32)
        validation = runner.validate_output_format(test_output)
        
        self.assertTrue(validation.is_valid)
        self.assertEqual(validation.expected_format, "numpy_array")
        self.assertEqual(validation.actual_format, "ndarray")
    
    def test_memory_usage_testing(self):
        """Test memory usage testing"""
        runner = SmokeTestRunner()
        
        # Create mock pipeline
        class MockPipeline:
            def generate(self, prompt, num_frames=4, height=256, width=256):
                # Simulate memory usage
                temp_data = np.random.rand(num_frames, height, width, 3)
                time.sleep(0.01)
                return temp_data
        
        mock_pipeline = MockPipeline()
        
        # Run memory test
        result = runner.test_memory_usage(mock_pipeline)
        
        self.assertGreater(result.peak_memory_mb, 0)
        self.assertGreaterEqual(result.memory_increase_mb, 0)
        self.assertFalse(result.memory_leaks_detected)  # Should be clean for mock
    
    def test_performance_benchmarking(self):
        """Test performance benchmarking"""
        runner = SmokeTestRunner()
        
        # Create mock pipeline
        class MockPipeline:
            def generate(self, prompt, num_frames=8, height=256, width=256):
                time.sleep(0.01)  # Consistent timing
                return np.random.rand(num_frames, height, width, 3).astype(np.float32)
        
        mock_pipeline = MockPipeline()
        
        # Run benchmark
        result = runner.benchmark_generation_speed(mock_pipeline)
        
        self.assertGreater(result.generation_time, 0)
        self.assertGreater(result.frames_per_second, 0)
        self.assertIn(result.performance_category, ["excellent", "good", "acceptable", "poor"])
    
    def test_integration_test_suite_initialization(self):
        """Test IntegrationTestSuite initialization"""
        suite = IntegrationTestSuite()
        
        self.assertIsNotNone(suite)
        self.assertIsNotNone(suite.test_config)
        self.assertTrue(suite.artifacts_dir.exists())
    
    def test_integration_test_execution(self):
        """Test integration test execution"""
        suite = IntegrationTestSuite()
        
        # Run a single integration test
        result = suite.test_architecture_detection_integration()
        
        self.assertIsNotNone(result)
        self.assertEqual(result.test_name, "architecture_detection_integration")
        self.assertGreater(result.execution_time, 0)
        self.assertIn("ArchitectureDetector", result.components_tested)
    
    def test_performance_benchmark_suite_initialization(self):
        """Test PerformanceBenchmarkSuite initialization"""
        suite = PerformanceBenchmarkSuite()
        
        self.assertIsNotNone(suite)
        self.assertIsNotNone(suite.config)
        self.assertTrue(suite.results_dir.exists())
    
    def test_performance_benchmark_execution(self):
        """Test performance benchmark execution"""
        suite = PerformanceBenchmarkSuite()
        
        # Run a single benchmark
        benchmarks = suite._benchmark_model_detection()
        
        self.assertGreater(len(benchmarks), 0)
        
        for benchmark in benchmarks:
            self.assertGreater(benchmark.execution_time, 0)
            self.assertGreaterEqual(benchmark.throughput, 0)
            self.assertIn("model_detection", benchmark.test_name)
    
    def test_coverage_validator_initialization(self):
        """Test TestCoverageValidator initialization"""
        validator = TestCoverageValidator()
        
        self.assertIsNotNone(validator)
        self.assertIsNotNone(validator.core_components)
        self.assertIsNotNone(validator.required_test_scenarios)
    
    def test_coverage_analysis_execution(self):
        """Test coverage analysis execution"""
        validator = TestCoverageValidator()
        
        # Run coverage analysis
        report = validator.analyze_coverage()
        
        self.assertIsNotNone(report)
        self.assertGreaterEqual(report.total_modules, 0)
        self.assertGreaterEqual(report.total_functions, 0)
        self.assertGreaterEqual(report.overall_coverage_percentage, 0)
    
    def test_comprehensive_test_runner_initialization(self):
        """Test ComprehensiveTestRunner initialization"""
        runner = ComprehensiveTestRunner()
        
        self.assertIsNotNone(runner)
        self.assertIsNotNone(runner.config)
        self.assertIsNotNone(runner.session_id)
        self.assertTrue(runner.results_dir.exists())
    
    def test_comprehensive_test_execution_smoke_only(self):
        """Test comprehensive test execution with smoke tests only"""
        config = {
            "test_execution": {
                "run_smoke_tests": True,
                "run_integration_tests": False,
                "run_performance_benchmarks": False,
                "run_coverage_analysis": False,
                "fail_fast": False,
                "timeout_minutes": 5
            }
        }
        
        runner = ComprehensiveTestRunner(config)
        
        # Run tests
        report = runner.run_comprehensive_tests()
        
        self.assertIsNotNone(report)
        self.assertGreater(report.total_execution_time, 0)
        self.assertEqual(len(report.test_results), 1)
        self.assertEqual(report.test_results[0].test_type, "smoke_tests")
    
    def test_mock_pipeline_creation(self):
        """Test mock pipeline creation"""
        runner = ComprehensiveTestRunner()
        
        mock_pipeline = runner._create_mock_pipeline()
        
        self.assertIsNotNone(mock_pipeline)
        self.assertTrue(hasattr(mock_pipeline, 'generate'))
        self.assertTrue(callable(mock_pipeline.generate))
        
        # Test mock pipeline functionality
        output = mock_pipeline.generate("test prompt", num_frames=4, height=128, width=128)
        self.assertEqual(output.shape, (4, 128, 128, 3))
    
    def test_error_handling_in_smoke_tests(self):
        """Test error handling in smoke tests"""
        runner = SmokeTestRunner()
        
        # Create failing mock pipeline
        class FailingPipeline:
            def generate(self, prompt, num_frames=8, height=256, width=256):
                raise Exception("Mock pipeline failure")
        
        failing_pipeline = FailingPipeline()
        
        # Run smoke test (should handle error gracefully)
        result = runner.run_pipeline_smoke_test(failing_pipeline)
        
        self.assertFalse(result.success)
        self.assertGreater(len(result.errors), 0)
        self.assertIn("Generation failed", result.errors[0])
    
    def test_test_result_serialization(self):
        """Test that test results can be serialized to JSON"""
        runner = SmokeTestRunner()
        
        # Create mock pipeline
        class MockPipeline:
            def generate(self, prompt, num_frames=8, height=256, width=256):
                return np.random.rand(num_frames, height, width, 3).astype(np.float32)
        
        mock_pipeline = MockPipeline()
        result = runner.run_pipeline_smoke_test(mock_pipeline)
        
        # Test that result can be converted to dict (for JSON serialization)
        result_dict = result.__dict__.copy()
        
        # Handle non-serializable types
        for key, value in result_dict.items():
            if isinstance(value, tuple):
                result_dict[key] = list(value)
        
        # Should not raise exception
        import json
        json_str = json.dumps(result_dict, default=str)
        self.assertIsInstance(json_str, str)
        self.assertGreater(len(json_str), 0)


def run_framework_validation():
    """Run validation of the testing framework"""
    print("Testing Framework Validation")
    print("=" * 50)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n" + "=" * 50)
    print("Testing Framework Validation Completed")


if __name__ == "__main__":
    run_framework_validation()