#!/usr/bin/env python3
"""
Integration tests for component interactions and data flow
Tests how different components work together in realistic scenarios.
"""

import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from local_testing_framework.test_manager import LocalTestManager
from local_testing_framework.environment_validator import EnvironmentValidator
from local_testing_framework.performance_tester import PerformanceTester
from local_testing_framework.integration_tester import IntegrationTester
from local_testing_framework.diagnostic_tool import DiagnosticTool
from local_testing_framework.report_generator import ReportGenerator
from local_testing_framework.sample_manager import SampleManager
from local_testing_framework.models.test_results import (
    TestResults, TestStatus, ValidationStatus, EnvironmentValidationResults,
    PerformanceTestResults, ValidationResult
)
from local_testing_framework.models.configuration import TestConfiguration


class TestComponentIntegration(unittest.TestCase):
    """Test integration between different components"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.json")
        
        # Create minimal test config
        test_config = {
            "system": {"gpu_enabled": True},
            "directories": {"models": "models", "outputs": "outputs"},
            "optimization": {"enable_attention_slicing": True},
            "performance": {"stats_refresh_interval": 5}
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(test_config, f)
        
        # Change to temp directory
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        os.chdir(self.original_cwd)
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_environment_validator_to_report_generator_flow(self):
        """Test data flow from environment validation to report generation"""
        # Create environment validator
        validator = EnvironmentValidator()
        
        # Run validation
        env_results = validator.validate_full_environment()
        
        # Create test results container
        test_results = TestResults(
            session_id="integration_test_001",
            start_time=datetime.now(),
            environment_results=env_results,
            overall_status=TestStatus.PASSED
        )
        
        # Generate report
        report_generator = ReportGenerator(output_dir=self.temp_dir)
        html_report = report_generator.generate_html_report(test_results)
        json_report = report_generator.generate_json_report(test_results)
        
        # Verify integration
        self.assertIsNotNone(html_report)
        self.assertIsNotNone(json_report)
        self.assertIn("integration_test_001", html_report.content)
        self.assertEqual(json_report.data["test_results"]["session_id"], "integration_test_001")
    
    @patch('local_testing_framework.performance_tester.subprocess.run')
    def test_performance_tester_to_diagnostic_tool_flow(self, mock_subprocess):
        """Test integration between performance testing and diagnostics"""
        # Mock subprocess for performance profiler
        mock_subprocess.return_value = Mock(
            returncode=0,
            stdout="Performance test completed",
            stderr=""
        )
        
        # Create performance tester
        perf_tester = PerformanceTester(self.config_path)
        
        # Create diagnostic tool
        diagnostic_tool = DiagnosticTool(self.config_path)
        
        # Simulate performance test failure scenario
        with patch.object(perf_tester, 'run_performance_tests') as mock_perf:
            # Mock failed performance results
            mock_perf.return_value = Mock(
                overall_status=TestStatus.FAILED,
                recommendations=["Enable attention slicing", "Reduce batch size"]
            )
            
            perf_results = perf_tester.run_performance_tests()
            
            # Run diagnostics based on performance failure
            diagnostic_results = diagnostic_tool.run_comprehensive_diagnosis()
            
            # Verify integration
            self.assertEqual(perf_results.overall_status, TestStatus.FAILED)
            self.assertIsNotNone(diagnostic_results)
            self.assertIn(diagnostic_results.overall_status, ["healthy", "warning", "critical", "error"])
    
    def test_sample_manager_to_integration_tester_flow(self):
        """Test flow from sample generation to integration testing"""
        # Create sample manager
        sample_manager = SampleManager()
        sample_manager.output_dir = Path(self.temp_dir) / "samples"
        sample_manager.output_dir.mkdir(exist_ok=True)
        
        # Generate sample data
        sample_files = sample_manager.generate_sample_input_files(count=2, resolutions=["720p"])
        
        # Verify samples were created
        self.assertEqual(len(sample_files), 2)
        for sample_file in sample_files:
            self.assertTrue(sample_file.exists())
        
        # Create integration tester
        integration_tester = IntegrationTester(self.config_path)
        
        # Use generated samples for testing (mock the actual test execution)
        with patch.object(integration_tester, '_execute_generation_test') as mock_execute:
            mock_execute.return_value = {
                'success': True,
                'stdout': 'Generation completed',
                'stderr': '',
                'duration_seconds': 30.0
            }
            
            # Load and use sample data
            with open(sample_files[0], 'r') as f:
                sample_data = json.load(f)
            
            test_config = {
                'model_type': 't2v-A14B',
                'prompt': sample_data['input'],
                'resolution': sample_data['resolution'],
                'expected_time_limit': 600
            }
            
            result = integration_tester._execute_generation_test(test_config)
            
            # Verify integration
            self.assertTrue(result['success'])
            self.assertEqual(result['duration_seconds'], 30.0)
    
    def test_test_manager_component_orchestration(self):
        """Test TestManager orchestrating multiple components"""
        # Create test manager
        test_manager = LocalTestManager(self.config_path)
        
        # Mock all component methods to avoid actual execution
        with patch.object(test_manager.environment_validator, 'validate_full_environment') as mock_env, \
             patch.object(test_manager.performance_tester, 'run_performance_tests') as mock_perf, \
             patch.object(test_manager.integration_tester, 'run_integration_tests') as mock_int, \
             patch.object(test_manager.diagnostic_tool, 'run_diagnostics') as mock_diag, \
             patch.object(test_manager.report_generator, 'generate_html_report') as mock_report:
            
            # Mock return values
            mock_env.return_value = Mock(overall_status=ValidationStatus.PASSED)
            mock_perf.return_value = Mock(overall_status=TestStatus.PASSED)
            mock_int.return_value = Mock(overall_status=TestStatus.PASSED)
            mock_diag.return_value = {"status": "healthy"}
            mock_report.return_value = Mock(content="<html>Test Report</html>")
            
            # Run full test suite
            results = test_manager.run_full_test_suite()
            
            # Verify orchestration
            self.assertIsNotNone(results)
            mock_env.assert_called_once()
            mock_perf.assert_called_once()
            mock_int.assert_called_once()
    
    def test_error_propagation_between_components(self):
        """Test how errors propagate between components"""
        # Create test manager
        test_manager = LocalTestManager(self.config_path)
        
        # Mock environment validator to raise an exception
        with patch.object(test_manager.environment_validator, 'validate_full_environment') as mock_env:
            mock_env.side_effect = Exception("Environment validation failed")
            
            # Run environment validation and check error handling
            try:
                test_manager.run_environment_validation()
                self.fail("Expected exception was not raised")
            except Exception as e:
                self.assertIn("Environment validation failed", str(e))
    
    def test_configuration_sharing_between_components(self):
        """Test that configuration is properly shared between components"""
        # Create test manager
        test_manager = LocalTestManager(self.config_path)
        
        # Verify all components have access to configuration
        self.assertIsNotNone(test_manager.configuration)
        self.assertEqual(test_manager.environment_validator.config.config_path, self.config_path)
        self.assertEqual(test_manager.performance_tester.config_path, self.config_path)
        self.assertEqual(test_manager.integration_tester.config_path, self.config_path)
        self.assertEqual(test_manager.diagnostic_tool.config_path, self.config_path)
    
    def test_data_consistency_across_components(self):
        """Test data consistency when passed between components"""
        # Create components
        validator = EnvironmentValidator()
        report_generator = ReportGenerator(output_dir=self.temp_dir)
        
        # Generate validation results
        env_results = validator.validate_full_environment()
        
        # Create test results
        test_results = TestResults(
            session_id="consistency_test",
            start_time=datetime.now(),
            environment_results=env_results,
            overall_status=TestStatus.PASSED
        )
        
        # Generate reports
        json_report = report_generator.generate_json_report(test_results)
        
        # Verify data consistency
        original_session_id = test_results.session_id
        report_session_id = json_report.data["test_results"]["session_id"]
        self.assertEqual(original_session_id, report_session_id)
        
        # Verify environment results are preserved
        self.assertIn("environment_results", json_report.data["test_results"])
        env_data = json_report.data["test_results"]["environment_results"]
        self.assertEqual(env_data["overall_status"], env_results.overall_status.value)


class TestCrossComponentDataFlow(unittest.TestCase):
    """Test data flow across multiple components in realistic scenarios"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.json")
        
        # Create test config
        test_config = {
            "system": {"gpu_enabled": True},
            "directories": {"models": "models", "outputs": "outputs"},
            "optimization": {"enable_attention_slicing": True},
            "performance": {"stats_refresh_interval": 5}
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(test_config, f)
        
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        os.chdir(self.original_cwd)
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_testing_pipeline_data_flow(self):
        """Test complete data flow through the entire testing pipeline"""
        # Create test manager
        test_manager = LocalTestManager(self.config_path)
        
        # Mock all external dependencies
        with patch('subprocess.run') as mock_subprocess, \
             patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('torch.cuda.is_available', return_value=True):
            
            # Mock system memory
            mock_memory.return_value = Mock(
                percent=60.0,
                used=8 * (1024**3),
                total=16 * (1024**3),
                available=8 * (1024**3)
            )
            
            # Mock subprocess calls
            mock_subprocess.return_value = Mock(
                returncode=0,
                stdout="Test completed successfully",
                stderr=""
            )
            
            # Run environment validation
            env_results = test_manager.run_environment_validation()
            
            # Verify environment results structure
            self.assertIsNotNone(env_results)
            self.assertIsNotNone(env_results.python_version)
            self.assertIsNotNone(env_results.dependencies)
            self.assertIsNotNone(env_results.cuda_availability)
            
            # Create comprehensive test results
            test_results = TestResults(
                session_id="pipeline_test",
                start_time=datetime.now(),
                environment_results=env_results,
                overall_status=TestStatus.PASSED
            )
            
            # Generate reports and verify data flow
            html_report = test_manager.generate_reports(test_results, "html")
            json_report = test_manager.generate_reports(test_results, "json")
            
            # Verify reports contain environment data
            self.assertIsNotNone(html_report)
            self.assertIsNotNone(json_report)
    
    def test_error_recovery_data_flow(self):
        """Test data flow during error recovery scenarios"""
        # Create diagnostic tool
        diagnostic_tool = DiagnosticTool(self.config_path)
        
        # Simulate error scenario
        with patch('psutil.cpu_percent', return_value=95.0), \
             patch('psutil.virtual_memory') as mock_memory:
            
            # Mock high memory usage
            mock_memory.return_value = Mock(
                percent=95.0,
                used=15 * (1024**3),
                total=16 * (1024**3),
                available=1 * (1024**3)
            )
            
            # Run diagnostics
            diagnostic_results = diagnostic_tool.run_comprehensive_diagnosis()
            
            # Verify diagnostic results contain resource information
            self.assertIsNotNone(diagnostic_results)
            self.assertIn(diagnostic_results.overall_status, ["healthy", "warning", "critical", "error"])
    
    def test_monitoring_data_aggregation(self):
        """Test data aggregation across monitoring sessions"""
        from local_testing_framework.continuous_monitor import ContinuousMonitor
        
        # Create monitor
        monitor = ContinuousMonitor()
        monitor.refresh_interval = 0.1  # Fast for testing
        
        # Mock resource collection
        with patch.object(monitor, '_collect_resource_metrics') as mock_collect:
            mock_metrics = Mock()
            mock_metrics.cpu_percent = 50.0
            mock_metrics.memory_percent = 60.0
            mock_metrics.memory_used_gb = 8.0
            mock_metrics.memory_total_gb = 16.0
            mock_collect.return_value = mock_metrics
            
            # Start monitoring session
            session = monitor.start_monitoring("data_aggregation_test")
            
            # Let it collect some data
            import time
            time.sleep(0.3)
            
            # Stop monitoring
            stopped_session = monitor.stop_monitoring_session()
            
            # Verify data aggregation
            self.assertIsNotNone(stopped_session)
            self.assertGreater(len(stopped_session.metrics_history), 0)
            
            # Generate monitoring report
            report = monitor.generate_monitoring_report()
            
            # Verify report contains aggregated data
            self.assertIn("session_info", report)
            self.assertIn("timeline", report)
            self.assertGreater(len(report["timeline"]), 0)


class TestComponentCompatibility(unittest.TestCase):
    """Test compatibility between different component versions and configurations"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        os.chdir(self.original_cwd)
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_configuration_compatibility(self):
        """Test component compatibility with different configuration formats"""
        # Test with minimal configuration
        minimal_config = {"system": {}}
        minimal_config_path = os.path.join(self.temp_dir, "minimal_config.json")
        
        with open(minimal_config_path, 'w') as f:
            json.dump(minimal_config, f)
        
        # Create components with minimal config
        try:
            validator = EnvironmentValidator()
            perf_tester = PerformanceTester(minimal_config_path)
            diagnostic_tool = DiagnosticTool(minimal_config_path)
            
            # Verify components can handle minimal configuration
            self.assertIsNotNone(validator)
            self.assertIsNotNone(perf_tester)
            self.assertIsNotNone(diagnostic_tool)
            
        except Exception as e:
            self.fail(f"Components should handle minimal configuration: {e}")
    
    def test_result_format_compatibility(self):
        """Test compatibility of result formats between components"""
        # Create environment validator
        validator = EnvironmentValidator()
        
        # Generate results
        env_results = validator.validate_full_environment()
        
        # Test serialization/deserialization compatibility
        try:
            # Convert to dict (simulating JSON serialization)
            results_dict = env_results.to_dict()
            
            # Verify essential fields are present
            self.assertIn("overall_status", results_dict)
            self.assertIn("python_version", results_dict)
            self.assertIn("dependencies", results_dict)
            
            # Verify status values are serializable
            self.assertIsInstance(results_dict["overall_status"], str)
            
        except Exception as e:
            self.fail(f"Result format should be compatible: {e}")
    
    def test_cross_platform_compatibility(self):
        """Test component behavior across different platforms"""
        # Create environment validator
        validator = EnvironmentValidator()
        
        # Test platform detection
        platform_info = validator._detect_platform()
        
        # Verify platform info contains expected fields
        self.assertIn("system", platform_info)
        self.assertIn("python_version", platform_info)
        
        # Verify platform-specific command generation
        commands = validator._generate_env_setup_commands(["TEST_VAR"])
        
        # Should generate appropriate commands for current platform
        self.assertIsInstance(commands, list)
        self.assertGreater(len(commands), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)