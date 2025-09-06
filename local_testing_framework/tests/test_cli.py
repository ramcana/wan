"""
Tests for the CLI interface
"""

import pytest
import argparse
from unittest.mock import Mock, patch, MagicMock
from io import StringIO
import sys

from local_testing_framework.cli.main import (
    create_parser, validate_env_command, test_performance_command,
    test_integration_command, diagnose_command, generate_samples_command,
    run_all_command, monitor_command, main
)
from local_testing_framework.models.test_results import (
    TestStatus, ValidationStatus, EnvironmentValidationResults,
    PerformanceTestResults, IntegrationTestResults, ValidationResult,
    BenchmarkResult, OptimizationResult
)


class TestCLIParser:
    """Test CLI argument parser"""
    
    def test_create_parser(self):
        """Test parser creation"""
        parser = create_parser()
        
        assert isinstance(parser, argparse.ArgumentParser)
        assert parser.prog == 'local-testing-framework'
    
    def test_validate_env_parser(self):
        """Test validate-env command parser"""
        parser = create_parser()
        
        # Test basic command
        args = parser.parse_args(['validate-env'])
        assert args.command == 'validate-env'
        assert args.func == validate_env_command
        assert not args.report
        
        # Test with report flag
        args = parser.parse_args(['validate-env', '--report'])
        assert args.report
    
    def test_test_performance_parser(self):
        """Test test-performance command parser"""
        parser = create_parser()
        
        # Test basic command
        args = parser.parse_args(['test-performance'])
        assert args.command == 'test-performance'
        assert args.func == test_performance_command
        assert args.resolution == 'both'
        assert not args.benchmark
        
        # Test with options
        args = parser.parse_args(['test-performance', '--resolution', '720p', '--benchmark'])
        assert args.resolution == '720p'
        assert args.benchmark
    
    def test_test_integration_parser(self):
        """Test test-integration command parser"""
        parser = create_parser()
        
        # Test basic command
        args = parser.parse_args(['test-integration'])
        assert args.command == 'test-integration'
        assert args.func == test_integration_command
        
        # Test with flags
        args = parser.parse_args(['test-integration', '--ui', '--api', '--full'])
        assert args.ui
        assert args.api
        assert args.full
    
    def test_diagnose_parser(self):
        """Test diagnose command parser"""
        parser = create_parser()
        
        # Test basic command
        args = parser.parse_args(['diagnose'])
        assert args.command == 'diagnose'
        assert args.func == diagnose_command
        
        # Test with flags
        args = parser.parse_args(['diagnose', '--system', '--cuda', '--memory'])
        assert args.system
        assert args.cuda
        assert args.memory
    
    def test_generate_samples_parser(self):
        """Test generate-samples command parser"""
        parser = create_parser()
        
        # Test basic command
        args = parser.parse_args(['generate-samples'])
        assert args.command == 'generate-samples'
        assert args.func == generate_samples_command
        
        # Test with flags
        args = parser.parse_args(['generate-samples', '--config', '--data', '--env', '--all'])
        assert args.config_samples
        assert args.data_samples
        assert args.env_samples
        assert args.all_samples
    
    def test_run_all_parser(self):
        """Test run-all command parser"""
        parser = create_parser()
        
        # Test basic command
        args = parser.parse_args(['run-all'])
        assert args.command == 'run-all'
        assert args.func == run_all_command
        
        # Test with report format
        args = parser.parse_args(['run-all', '--report-format', 'html'])
        assert args.report_format == 'html'
    
    def test_monitor_parser(self):
        """Test monitor command parser"""
        parser = create_parser()
        
        # Test basic command
        args = parser.parse_args(['monitor'])
        assert args.command == 'monitor'
        assert args.func == monitor_command
        assert args.duration == 3600
        assert not args.alerts
        
        # Test with options
        args = parser.parse_args(['monitor', '--duration', '1800', '--alerts'])
        assert args.duration == 1800
        assert args.alerts


class TestCLICommands:
    """Test CLI command implementations"""
    
    @pytest.fixture
    def mock_args(self):
        """Create mock arguments"""
        args = Mock()
        args.config = 'test_config.json'
        args.verbose = False
        return args
    
    @patch('local_testing_framework.cli.main.LocalTestManager')
    def test_validate_env_command_success(self, mock_manager_class, mock_args):
        """Test successful environment validation command"""
        # Setup mock
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        # Create mock validation results
        mock_result = Mock()
        mock_result.python_version = ValidationResult("python", ValidationStatus.PASSED, "Python 3.9.7")
        mock_result.cuda_availability = ValidationResult("cuda", ValidationStatus.PASSED, "CUDA 11.8")
        mock_result.dependencies = ValidationResult("deps", ValidationStatus.PASSED, "All packages installed")
        mock_result.configuration = ValidationResult("config", ValidationStatus.PASSED, "Config valid")
        mock_result.environment_variables = ValidationResult("env", ValidationStatus.PASSED, "Env vars set")
        mock_result.overall_status.value = "passed"
        mock_result.remediation_steps = []
        mock_result.to_dict.return_value = {"status": "passed"}
        
        mock_manager.run_environment_validation.return_value = mock_result
        mock_args.report = False
        
        # Capture output
        with patch('builtins.print') as mock_print:
            result = validate_env_command(mock_args)
        
        assert result == 0
        mock_manager.run_environment_validation.assert_called_once()
        mock_print.assert_called()
    
    @patch('local_testing_framework.cli.main.LocalTestManager')
    def test_test_performance_command_success(self, mock_manager_class, mock_args):
        """Test successful performance testing command"""
        # Setup mock
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        # Create mock performance results
        mock_result = Mock()
        mock_result.benchmark_720p = BenchmarkResult(
            resolution="720p", generation_time=7.5, target_time=9.0,
            meets_target=True, vram_usage=8.2, cpu_usage=65.0,
            memory_usage=45.0, optimization_level="high"
        )
        mock_result.benchmark_1080p = None
        mock_result.vram_optimization = OptimizationResult(
            baseline_vram_mb=12000, optimized_vram_mb=2400,
            reduction_percent=80.0, target_reduction_percent=80.0,
            meets_target=True, optimizations_applied=["attention_slicing"]
        )
        mock_result.overall_status = TestStatus.PASSED
        mock_result.recommendations = []
        
        mock_manager.run_performance_tests.return_value = mock_result
        mock_args.benchmark = False
        
        # Capture output
        with patch('builtins.print') as mock_print:
            result = test_performance_command(mock_args)
        
        assert result == 0
        mock_manager.run_performance_tests.assert_called_once()
        mock_print.assert_called()
    
    @patch('local_testing_framework.cli.main.LocalTestManager')
    def test_test_integration_command_success(self, mock_manager_class, mock_args):
        """Test successful integration testing command"""
        # Setup mock
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        # Create mock integration results
        from datetime import datetime
        mock_result = Mock()
        mock_result.overall_status = TestStatus.PASSED
        mock_result.start_time = datetime.now()
        mock_result.end_time = datetime.now()
        mock_result.error_handling_result = None
        mock_result.ui_results = None
        mock_result.api_results = None
        mock_result.resource_monitoring_result = None
        
        mock_manager.run_integration_tests.return_value = mock_result
        
        # Capture output
        with patch('builtins.print') as mock_print:
            result = test_integration_command(mock_args)
        
        assert result == 0
        mock_manager.run_integration_tests.assert_called_once()
        mock_print.assert_called()
    
    @patch('local_testing_framework.cli.main.LocalTestManager')
    def test_diagnose_command_success(self, mock_manager_class, mock_args):
        """Test successful diagnostic command"""
        # Setup mock
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        # Create mock diagnostic results
        mock_result = {
            "system_status": "healthy",
            "issues": [],
            "resource_usage": {
                "cpu_percent": 25.5,
                "memory_percent": 45.2,
                "vram_percent": 60.1
            }
        }
        
        mock_manager.run_diagnostics.return_value = mock_result
        
        # Capture output
        with patch('builtins.print') as mock_print:
            result = diagnose_command(mock_args)
        
        assert result == 0
        mock_manager.run_diagnostics.assert_called_once()
        mock_print.assert_called()
    
    @patch('local_testing_framework.cli.main.LocalTestManager')
    def test_generate_samples_command_success(self, mock_manager_class, mock_args):
        """Test successful sample generation command"""
        # Setup mock
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        # Create mock sample results
        mock_result = {
            "config": {"path": "sample_config.json"},
            "data": [{"input": "test prompt", "resolution": "720p"}],
            "env": {"path": "sample.env"}
        }
        
        mock_manager.generate_samples.return_value = mock_result
        mock_args.config_samples = True
        mock_args.data_samples = True
        mock_args.env_samples = True
        mock_args.all_samples = False
        
        # Capture output
        with patch('builtins.print') as mock_print:
            result = generate_samples_command(mock_args)
        
        assert result == 0
        mock_manager.generate_samples.assert_called_once_with(["config", "data", "env"])
        mock_print.assert_called()
    
    @patch('local_testing_framework.cli.main.LocalTestManager')
    def test_run_all_command_success(self, mock_manager_class, mock_args):
        """Test successful run-all command"""
        # Setup mock
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        # Create mock test results
        from datetime import datetime
        mock_result = Mock()
        mock_result.overall_status = TestStatus.PASSED
        mock_result.start_time = datetime.now()
        mock_result.end_time = datetime.now()
        mock_result.environment_results = None
        mock_result.performance_results = None
        mock_result.integration_results = None
        mock_result.recommendations = []
        
        mock_manager.run_full_test_suite.return_value = mock_result
        mock_args.report_format = None
        
        # Capture output
        with patch('builtins.print') as mock_print:
            result = run_all_command(mock_args)
        
        assert result == 0
        mock_manager.run_full_test_suite.assert_called_once()
        mock_print.assert_called()
    
    @patch('local_testing_framework.cli.main.LocalTestManager')
    def test_monitor_command_success(self, mock_manager_class, mock_args):
        """Test successful monitor command"""
        # Setup mock
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        mock_manager.start_monitoring.return_value = "monitor-123"
        mock_args.duration = 3600
        mock_args.alerts = True
        
        # Capture output
        with patch('builtins.print') as mock_print:
            result = monitor_command(mock_args)
        
        assert result == 0
        mock_manager.start_monitoring.assert_called_once_with(3600, True)
        mock_print.assert_called()
    
    @patch('local_testing_framework.cli.main.LocalTestManager')
    def test_command_error_handling(self, mock_manager_class, mock_args):
        """Test command error handling"""
        # Setup mock to raise exception
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        mock_manager.run_environment_validation.side_effect = Exception("Test error")
        
        mock_args.report = False
        
        # Capture output
        with patch('builtins.print') as mock_print:
            result = validate_env_command(mock_args)
        
        assert result == 1
        mock_print.assert_called()


class TestMainFunction:
    """Test main function"""
    
    @patch('local_testing_framework.cli.main.create_parser')
    def test_main_no_command(self, mock_create_parser):
        """Test main function with no command"""
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.command = None
        
        mock_parser.parse_args.return_value = mock_args
        mock_create_parser.return_value = mock_parser
        
        with patch('sys.argv', ['local-testing-framework']):
            result = main()
        
        assert result == 1
        mock_parser.print_help.assert_called_once()
    
    @patch('local_testing_framework.cli.main.create_parser')
    def test_main_with_command(self, mock_create_parser):
        """Test main function with valid command"""
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.command = 'validate-env'
        mock_args.verbose = False
        mock_args.func = Mock(return_value=0)
        
        mock_parser.parse_args.return_value = mock_args
        mock_create_parser.return_value = mock_parser
        
        with patch('local_testing_framework.cli.main.setup_logging'):
            result = main()
        
        assert result == 0
        mock_args.func.assert_called_once_with(mock_args)
    
    @patch('local_testing_framework.cli.main.create_parser')
    def test_main_keyboard_interrupt(self, mock_create_parser):
        """Test main function with keyboard interrupt"""
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.command = 'validate-env'
        mock_args.verbose = False
        mock_args.func = Mock(side_effect=KeyboardInterrupt())
        
        mock_parser.parse_args.return_value = mock_args
        mock_create_parser.return_value = mock_parser
        
        with patch('local_testing_framework.cli.main.setup_logging'), \
             patch('builtins.print') as mock_print:
            result = main()
        
        assert result == 130
        mock_print.assert_called()
    
    @patch('local_testing_framework.cli.main.create_parser')
    def test_main_unexpected_error(self, mock_create_parser):
        """Test main function with unexpected error"""
        mock_parser = Mock()
        mock_args = Mock()
        mock_args.command = 'validate-env'
        mock_args.verbose = False
        mock_args.func = Mock(side_effect=Exception("Unexpected error"))
        
        mock_parser.parse_args.return_value = mock_args
        mock_create_parser.return_value = mock_parser
        
        with patch('local_testing_framework.cli.main.setup_logging'), \
             patch('builtins.print') as mock_print:
            result = main()
        
        assert result == 1
        mock_print.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])