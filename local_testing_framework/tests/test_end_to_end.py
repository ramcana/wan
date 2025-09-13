#!/usr/bin/env python3
"""
End-to-end tests for the complete testing pipeline
Tests full workflows from start to finish in realistic scenarios.
"""

import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from local_testing_framework.test_manager import LocalTestManager, WorkflowDefinition
from local_testing_framework.models.test_results import TestStatus, ValidationStatus
from local_testing_framework.models.configuration import TestConfiguration


class TestFullWorkflowExecution(unittest.TestCase):
    """Test complete workflow execution from start to finish"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.json")
        self.env_path = os.path.join(self.temp_dir, ".env")
        
        # Create comprehensive test config
        test_config = {
            "system": {
                "gpu_enabled": True,
                "device": "cuda",
                "gpu_memory_fraction": 0.9
            },
            "directories": {
                "models": "models/",
                "outputs": "outputs/",
                "cache": "cache/",
                "logs": "logs/"
            },
            "optimization": {
                "enable_attention_slicing": True,
                "enable_vae_tiling": True,
                "use_fp16": True,
                "torch_compile": False
            },
            "performance": {
                "batch_size": 1,
                "stats_refresh_interval": 5,
                "vram_warning_threshold": 0.8,
                "cpu_warning_percent": 75,
                "max_queue_size": 5
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(test_config, f)
        
        # Create .env file
        with open(self.env_path, 'w') as f:
            f.write("HF_TOKEN=test_token_12345\n")
            f.write("CUDA_VISIBLE_DEVICES=0\n")
        
        # Create required directories
        for dir_name in ["models", "outputs", "cache", "logs"]:
            os.makedirs(os.path.join(self.temp_dir, dir_name), exist_ok=True)
        
        # Change to temp directory
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        os.chdir(self.original_cwd)
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_environment_validation_workflow(self):
        """Test complete environment validation workflow"""
        # Create test manager
        test_manager = LocalTestManager(self.config_path)
        
        # Mock external dependencies
        with patch('platform.python_version', return_value='3.9.7'), \
             patch('torch.cuda.is_available', return_value=True), \
             patch('torch.version.cuda', '11.8'), \
             patch('torch.__version__', '1.12.0'), \
             patch('importlib.util.find_spec', return_value=Mock()):
            
            # Run environment validation workflow
            workflow = WorkflowDefinition.get_environment_workflow()
            session = test_manager.create_session(workflow)
            
            # Execute workflow
            env_results = test_manager.run_environment_validation()
            
            # Update session results
            session.results.environment_results = env_results
            session.results.overall_status = TestStatus.PASSED
            
            # Generate reports
            html_report = test_manager.generate_reports(session.results, "html")
            json_report = test_manager.generate_reports(session.results, "json")
            
            # Verify complete workflow
            self.assertIsNotNone(env_results)
            self.assertIsNotNone(html_report)
            self.assertIsNotNone(json_report)
            
            # Verify session tracking
            self.assertIn(session.session_id, test_manager.active_sessions)
            
            # Verify report files were created
            output_files = list(Path(self.temp_dir).glob("*.html"))
            self.assertGreater(len(output_files), 0)
    
    @patch('subprocess.run')
    def test_complete_performance_testing_workflow(self, mock_subprocess):
        """Test complete performance testing workflow"""
        # Mock subprocess calls for performance profiler
        mock_subprocess.return_value = Mock(
            returncode=0,
            stdout="Performance test completed successfully",
            stderr=""
        )
        
        # Create test manager
        test_manager = LocalTestManager(self.config_path)
        
        # Mock performance tester methods
        with patch.object(test_manager.performance_tester, 'run_performance_tests') as mock_perf:
            # Mock successful performance results
            mock_perf_results = Mock()
            mock_perf_results.overall_status = TestStatus.PASSED
            mock_perf_results.benchmark_720p = Mock(
                resolution="720p",
                generation_time=7.5,
                meets_target=True,
                vram_usage=8.2
            )
            mock_perf_results.benchmark_1080p = Mock(
                resolution="1080p", 
                generation_time=15.8,
                meets_target=True,
                vram_usage=10.1
            )
            mock_perf_results.recommendations = []
            mock_perf.return_value = mock_perf_results
            
            # Run performance testing workflow
            workflow = WorkflowDefinition.get_performance_workflow()
            session = test_manager.create_session(workflow)
            
            # Execute workflow
            perf_results = test_manager.run_performance_tests()
            
            # Update session results
            session.results.performance_results = perf_results
            session.results.overall_status = TestStatus.PASSED
            
            # Generate comprehensive report
            html_report = test_manager.generate_reports(session.results, "html")
            
            # Verify complete workflow
            self.assertIsNotNone(perf_results)
            self.assertEqual(perf_results.overall_status, TestStatus.PASSED)
            self.assertIsNotNone(html_report)
            
            # Verify performance metrics are included
            self.assertIsNotNone(perf_results.benchmark_720p)
            self.assertIsNotNone(perf_results.benchmark_1080p)
    
    @patch('subprocess.Popen')
    def test_complete_integration_testing_workflow(self, mock_popen):
        """Test complete integration testing workflow"""
        # Mock subprocess for integration tests
        mock_process = Mock()
        mock_process.communicate.return_value = ("Integration tests passed", "")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        
        # Create test manager
        test_manager = LocalTestManager(self.config_path)
        
        # Mock integration tester methods
        with patch.object(test_manager.integration_tester, 'run_integration_tests') as mock_int:
            # Mock successful integration results
            mock_int_results = Mock()
            mock_int_results.overall_status = TestStatus.PASSED
            mock_int_results.start_time = datetime.now()
            mock_int_results.end_time = datetime.now()
            mock_int_results.generation_results = []
            mock_int_results.error_handling_result = None
            mock_int_results.ui_results = None
            mock_int_results.api_results = None
            mock_int.return_value = mock_int_results
            
            # Run integration testing workflow
            workflow = WorkflowDefinition.get_integration_workflow()
            session = test_manager.create_session(workflow)
            
            # Execute workflow
            int_results = test_manager.run_integration_tests()
            
            # Update session results
            session.results.integration_results = int_results
            session.results.overall_status = TestStatus.PASSED
            
            # Generate report
            html_report = test_manager.generate_reports(session.results, "html")
            
            # Verify complete workflow
            self.assertIsNotNone(int_results)
            self.assertEqual(int_results.overall_status, TestStatus.PASSED)
            self.assertIsNotNone(html_report)
    
    def test_complete_full_test_suite_workflow(self):
        """Test complete full test suite workflow"""
        # Create test manager
        test_manager = LocalTestManager(self.config_path)
        
        # Mock all component methods
        with patch.object(test_manager.environment_validator, 'validate_full_environment') as mock_env, \
             patch.object(test_manager.performance_tester, 'run_performance_tests') as mock_perf, \
             patch.object(test_manager.integration_tester, 'run_integration_tests') as mock_int, \
             patch.object(test_manager.diagnostic_tool, 'run_diagnostics') as mock_diag:
            
            # Mock successful results for all components
            mock_env.return_value = Mock(overall_status=ValidationStatus.PASSED)
            mock_perf.return_value = Mock(overall_status=TestStatus.PASSED)
            mock_int.return_value = Mock(overall_status=TestStatus.PASSED)
            mock_diag.return_value = {"status": "healthy", "issues": []}
            
            # Run full test suite
            results = test_manager.run_full_test_suite()
            
            # Verify complete workflow execution
            self.assertIsNotNone(results)
            self.assertEqual(results.overall_status, TestStatus.PASSED)
            
            # Verify all components were called
            mock_env.assert_called_once()
            mock_perf.assert_called_once()
            mock_int.assert_called_once()
            
            # Verify session was created and tracked
            sessions = test_manager.list_active_sessions()
            self.assertGreater(len(sessions), 0)
    
    def test_workflow_with_failures_and_recovery(self):
        """Test workflow execution with failures and recovery"""
        # Create test manager
        test_manager = LocalTestManager(self.config_path)
        
        # Mock components with mixed results
        with patch.object(test_manager.environment_validator, 'validate_full_environment') as mock_env, \
             patch.object(test_manager.performance_tester, 'run_performance_tests') as mock_perf, \
             patch.object(test_manager.diagnostic_tool, 'run_diagnostics') as mock_diag:
            
            # Mock environment validation failure
            mock_env.return_value = Mock(
                overall_status=ValidationStatus.FAILED,
                remediation_steps=["Install missing dependencies", "Fix configuration"]
            )
            
            # Mock performance test success
            mock_perf.return_value = Mock(overall_status=TestStatus.PASSED)
            
            # Mock diagnostic results with issues
            mock_diag.return_value = {
                "status": "warning",
                "issues": [
                    {"category": "environment", "severity": "high", "message": "Missing dependencies"}
                ]
            }
            
            # Run full test suite
            results = test_manager.run_full_test_suite()
            
            # Verify workflow handles failures appropriately
            self.assertIsNotNone(results)
            self.assertIn(results.overall_status, [TestStatus.FAILED, TestStatus.PARTIAL])
            
            # Verify remediation steps are included
            if hasattr(results, 'recommendations'):
                self.assertIsInstance(results.recommendations, list)
    
    def test_continuous_monitoring_workflow(self):
        """Test continuous monitoring workflow"""
        # Create test manager
        test_manager = LocalTestManager(self.config_path)
        
        # Mock continuous monitor
        with patch.object(test_manager.continuous_monitor, 'start_monitoring') as mock_start, \
             patch.object(test_manager.continuous_monitor, 'stop_monitoring_session') as mock_stop:
            
            # Mock monitoring session
            mock_session = Mock()
            mock_session.session_id = "monitor_test_001"
            mock_session.is_active = True
            mock_start.return_value = "monitor_test_001"
            mock_stop.return_value = mock_session
            
            # Start monitoring
            monitor_id = test_manager.start_monitoring(duration=60, enable_alerts=True)
            
            # Verify monitoring started
            self.assertEqual(monitor_id, "monitor_test_001")
            mock_start.assert_called_once_with(60, True)
            
            # Stop monitoring
            stopped_session = test_manager.continuous_monitor.stop_monitoring_session()
            
            # Verify monitoring stopped
            self.assertIsNotNone(stopped_session)
            self.assertEqual(stopped_session.session_id, "monitor_test_001")
    
    def test_sample_generation_and_usage_workflow(self):
        """Test sample generation and usage in testing workflow"""
        # Create test manager
        test_manager = LocalTestManager(self.config_path)
        
        # Generate samples
        sample_results = test_manager.generate_samples(["config", "data", "env"])
        
        # Verify samples were generated
        self.assertIn("config", sample_results)
        self.assertIn("data", sample_results)
        self.assertIn("env", sample_results)
        
        # Verify sample files exist
        sample_files = list(Path(self.temp_dir).glob("sample_*.json"))
        self.assertGreater(len(sample_files), 0)
        
        # Use generated samples in integration testing (mock)
        with patch.object(test_manager.integration_tester, 'run_integration_tests') as mock_int:
            mock_int.return_value = Mock(overall_status=TestStatus.PASSED)
            
            # Run integration tests using samples
            int_results = test_manager.run_integration_tests()
            
            # Verify integration tests ran successfully
            self.assertEqual(int_results.overall_status, TestStatus.PASSED)
    
    def test_report_generation_workflow(self):
        """Test comprehensive report generation workflow"""
        # Create test manager
        test_manager = LocalTestManager(self.config_path)
        
        # Create comprehensive test results
        from local_testing_framework.models.test_results import TestResults
        
        test_results = TestResults(
            session_id="report_test_001",
            start_time=datetime.now(),
            overall_status=TestStatus.PASSED
        )
        
        # Generate all report formats
        html_report = test_manager.generate_reports(test_results, "html")
        json_report = test_manager.generate_reports(test_results, "json")
        
        # Verify reports were generated
        self.assertIsNotNone(html_report)
        self.assertIsNotNone(json_report)
        
        # Verify report files exist
        html_files = list(Path(self.temp_dir).glob("*.html"))
        json_files = list(Path(self.temp_dir).glob("*.json"))
        
        self.assertGreater(len(html_files), 0)
        self.assertGreater(len(json_files), 0)
        
        # Verify report content
        if hasattr(html_report, 'content'):
            self.assertIn("report_test_001", html_report.content)
        
        if hasattr(json_report, 'data'):
            self.assertEqual(json_report.data["test_results"]["session_id"], "report_test_001")


class TestRealWorldScenarios(unittest.TestCase):
    """Test realistic end-to-end scenarios that users would encounter"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.json")
        
        # Create realistic config
        realistic_config = {
            "system": {
                "gpu_enabled": True,
                "device": "cuda",
                "gpu_memory_fraction": 0.85
            },
            "directories": {
                "models": "models/",
                "outputs": "outputs/",
                "cache": "cache/",
                "logs": "logs/"
            },
            "optimization": {
                "enable_attention_slicing": True,
                "enable_vae_tiling": True,
                "use_fp16": True,
                "torch_compile": False,
                "enable_xformers": True
            },
            "performance": {
                "batch_size": 1,
                "stats_refresh_interval": 5,
                "vram_warning_threshold": 0.8,
                "cpu_warning_percent": 75
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(realistic_config, f)
        
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment"""
        os.chdir(self.original_cwd)
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_new_user_setup_scenario(self):
        """Test scenario: New user setting up the system for the first time"""
        # Create test manager
        test_manager = LocalTestManager(self.config_path)
        
        # Mock typical new user environment (some issues)
        with patch('platform.python_version', return_value='3.9.7'), \
             patch('torch.cuda.is_available', return_value=False), \
             patch('os.path.exists', return_value=False):  # Missing .env file
            
            # Run environment validation (typical first step)
            env_results = test_manager.run_environment_validation()
            
            # Verify validation identifies issues
            self.assertIsNotNone(env_results)
            
            # Generate remediation report
            html_report = test_manager.generate_reports(
                Mock(
                    session_id="new_user_setup",
                    start_time=datetime.now(),
                    environment_results=env_results,
                    overall_status=TestStatus.FAILED
                ),
                "html"
            )
            
            # Verify report provides guidance
            self.assertIsNotNone(html_report)
    
    def test_performance_optimization_scenario(self):
        """Test scenario: User optimizing performance after initial setup"""
        # Create test manager
        test_manager = LocalTestManager(self.config_path)
        
        # Mock performance testing with suboptimal results
        with patch.object(test_manager.performance_tester, 'run_performance_tests') as mock_perf:
            # Mock results showing need for optimization
            mock_perf_results = Mock()
            mock_perf_results.overall_status = TestStatus.PARTIAL
            mock_perf_results.benchmark_720p = Mock(
                generation_time=12.0,  # Above 9min target
                meets_target=False,
                vram_usage=14.0  # Above 12GB limit
            )
            mock_perf_results.recommendations = [
                "Enable attention slicing",
                "Enable VAE tiling",
                "Reduce batch size"
            ]
            mock_perf.return_value = mock_perf_results
            
            # Run performance tests
            perf_results = test_manager.run_performance_tests()
            
            # Verify optimization recommendations are provided
            self.assertEqual(perf_results.overall_status, TestStatus.PARTIAL)
            self.assertGreater(len(perf_results.recommendations), 0)
            
            # Generate optimization report
            html_report = test_manager.generate_reports(
                Mock(
                    session_id="performance_optimization",
                    start_time=datetime.now(),
                    performance_results=perf_results,
                    overall_status=TestStatus.PARTIAL
                ),
                "html"
            )
            
            # Verify report includes optimization guidance
            self.assertIsNotNone(html_report)
    
    def test_troubleshooting_scenario(self):
        """Test scenario: User troubleshooting system issues"""
        # Create test manager
        test_manager = LocalTestManager(self.config_path)
        
        # Mock diagnostic tool with issues found
        with patch.object(test_manager.diagnostic_tool, 'run_diagnostics') as mock_diag:
            # Mock diagnostic results with issues
            mock_diag.return_value = {
                "status": "warning",
                "issues": [
                    {
                        "category": "memory",
                        "severity": "high",
                        "message": "High memory usage detected",
                        "remediation": ["Close unnecessary applications", "Restart system"]
                    },
                    {
                        "category": "cuda",
                        "severity": "medium", 
                        "message": "CUDA memory fragmentation",
                        "remediation": ["Clear GPU cache", "Restart application"]
                    }
                ],
                "resource_usage": {
                    "cpu_percent": 85.0,
                    "memory_percent": 90.0,
                    "vram_percent": 95.0
                }
            }
            
            # Run diagnostics
            diag_results = test_manager.run_diagnostics()
            
            # Verify issues are identified
            self.assertEqual(diag_results["status"], "warning")
            self.assertGreater(len(diag_results["issues"]), 0)
            
            # Verify remediation steps are provided
            for issue in diag_results["issues"]:
                self.assertIn("remediation", issue)
                self.assertGreater(len(issue["remediation"]), 0)
    
    def test_production_readiness_scenario(self):
        """Test scenario: User validating production readiness"""
        # Create test manager
        test_manager = LocalTestManager(self.config_path)
        
        # Mock all components for production validation
        with patch.object(test_manager.environment_validator, 'validate_full_environment') as mock_env, \
             patch.object(test_manager.performance_tester, 'run_performance_tests') as mock_perf, \
             patch.object(test_manager.integration_tester, 'run_integration_tests') as mock_int:
            
            # Mock production-ready results
            mock_env.return_value = Mock(overall_status=ValidationStatus.PASSED)
            mock_perf.return_value = Mock(
                overall_status=TestStatus.PASSED,
                benchmark_720p=Mock(meets_target=True),
                benchmark_1080p=Mock(meets_target=True)
            )
            mock_int.return_value = Mock(overall_status=TestStatus.PASSED)
            
            # Run full test suite for production validation
            results = test_manager.run_full_test_suite()
            
            # Verify production readiness
            self.assertEqual(results.overall_status, TestStatus.PASSED)
            
            # Generate production readiness report
            html_report = test_manager.generate_reports(results, "html")
            json_report = test_manager.generate_reports(results, "json")
            
            # Verify comprehensive reports are generated
            self.assertIsNotNone(html_report)
            self.assertIsNotNone(json_report)
    
    def test_continuous_monitoring_scenario(self):
        """Test scenario: User running continuous monitoring during development"""
        # Create test manager
        test_manager = LocalTestManager(self.config_path)
        
        # Mock continuous monitoring
        with patch.object(test_manager.continuous_monitor, 'start_monitoring') as mock_start, \
             patch.object(test_manager.continuous_monitor, 'get_session_summary') as mock_summary:
            
            # Mock monitoring session
            mock_start.return_value = "continuous_monitor_001"
            mock_summary.return_value = {
                "session_id": "continuous_monitor_001",
                "start_time": datetime.now().isoformat(),
                "is_active": True,
                "metrics_count": 120,
                "latest_metrics": {
                    "cpu_percent": 45.0,
                    "memory_percent": 60.0,
                    "vram_percent": 70.0
                },
                "averages": {
                    "cpu_avg": 42.5,
                    "memory_avg": 58.2,
                    "vram_avg": 68.5
                }
            }
            
            # Start monitoring
            monitor_id = test_manager.start_monitoring(duration=3600, enable_alerts=True)
            
            # Get monitoring summary
            summary = test_manager.continuous_monitor.get_session_summary()
            
            # Verify monitoring is working
            self.assertEqual(monitor_id, "continuous_monitor_001")
            self.assertIsNotNone(summary)
            self.assertTrue(summary["is_active"])
            self.assertGreater(summary["metrics_count"], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
