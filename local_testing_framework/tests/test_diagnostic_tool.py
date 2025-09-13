"""
Unit tests for DiagnosticTool
"""

import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Import the modules to test
from local_testing_framework.diagnostic_tool import (
    DiagnosticTool, SystemAnalyzer, ErrorLogAnalyzer, RecoveryManager,
    DiagnosticReportGenerator, DiagnosticIssue, DiagnosticCategory,
    SystemAnalysis, DiagnosticResults
)
from local_testing_framework.models.test_results import ResourceMetrics, ValidationStatus
from error_handler import ErrorInfo, ErrorCategory


class TestSystemAnalyzer(unittest.TestCase):
    """Test SystemAnalyzer functionality"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.json"
        
        # Create test config
        test_config = {
            "system": {"test": True},
            "directories": {"models": "models"},
            "optimization": {"enable_attention_slicing": False},
            "performance": {"cpu_warning_percent": 80}
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(test_config, f)
        
        self.analyzer = SystemAnalyzer(str(self.config_path))
    
    def test_load_config(self):
        """Test configuration loading"""
        self.assertIsInstance(self.analyzer.config, dict)
        self.assertIn("system", self.analyzer.config)
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_analyze_system_resources(self, mock_disk, mock_memory, mock_cpu):
        """Test system resource analysis"""
        # Mock system metrics
        mock_cpu.return_value = 50.0
        mock_memory.return_value = Mock(percent=60.0, used=8*1024**3, total=16*1024**3, available=8*1024**3)
        mock_disk.return_value = Mock(free=100*1024**3)
        
        metrics, issues = self.analyzer.analyze_system_resources()
        
        self.assertIsInstance(metrics, ResourceMetrics)
        self.assertEqual(metrics.cpu_percent, 50.0)
        self.assertEqual(metrics.memory_percent, 60.0)
        self.assertIsInstance(issues, list)
    
    def test_analyze_configuration(self):
        """Test configuration analysis"""
        issues = self.analyzer.analyze_configuration()
        
        self.assertIsInstance(issues, list)
        # Should not have issues since we created a valid config
        config_issues = [issue for issue in issues if "Missing Configuration Section" in issue.title]
        self.assertEqual(len(config_issues), 0)
    
    @patch('importlib.import_module')
    def test_analyze_dependencies(self, mock_import):
        """Test dependency analysis"""
        # Mock successful imports
        mock_import.return_value = Mock()
        
        issues = self.analyzer.analyze_dependencies()
        
        self.assertIsInstance(issues, list)


class TestErrorLogAnalyzer(unittest.TestCase):
    """Test ErrorLogAnalyzer functionality"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = Path(self.temp_dir) / "test_errors.log"
        
        # Create test log content
        log_content = """2025-07-30 10:00:00,000 - wan22_errors - ERROR - test_function:10 - Test error message
Traceback (most recent call last):
  File "test.py", line 10, in test_function
    raise ValueError("Test error")
ValueError: Test error

2025-07-30 10:01:00,000 - wan22_errors - CRITICAL - another_function:20 - CUDA out of memory
RuntimeError: CUDA out of memory
"""
        
        with open(self.log_file, 'w') as f:
            f.write(log_content)
        
        self.analyzer = ErrorLogAnalyzer(str(self.log_file))
    
    def test_analyze_error_logs(self):
        """Test error log analysis"""
        analysis = self.analyzer.analyze_error_logs(hours_back=24)
        
        self.assertEqual(analysis["status"], "analyzed")
        self.assertGreater(analysis["total_entries"], 0)
        self.assertIn("error_patterns", analysis)
        self.assertIn("recommendations", analysis)
    
    def test_parse_log_entries(self):
        """Test log entry parsing"""
        with open(self.log_file, 'r') as f:
            content = f.read()
        
        entries = self.analyzer._parse_log_entries(content)
        
        self.assertGreater(len(entries), 0)
        self.assertIn("timestamp", entries[0])
        self.assertIn("error_type", entries[0])


class TestRecoveryManager(unittest.TestCase):
    """Test RecoveryManager functionality"""
    
    def setUp(self):
        self.recovery_manager = RecoveryManager()
    
    def test_attempt_automatic_recovery(self):
        """Test automatic recovery attempt"""
        issue = DiagnosticIssue(
            category=DiagnosticCategory.MEMORY_ERROR,
            severity="warning",
            title="High Memory Usage",
            description="Memory usage is high",
            affected_components=["system"],
            symptoms=["Slow performance"],
            root_cause="High memory utilization",
            remediation_steps=["Close applications"],
            auto_recoverable=True
        )
        
        result = self.recovery_manager.attempt_automatic_recovery(issue)
        
        self.assertIsInstance(result, dict)
        self.assertIn("success", result)
        self.assertIn("actions_taken", result)
    
    def test_recovery_history(self):
        """Test recovery history tracking"""
        initial_count = len(self.recovery_manager.get_recovery_history())
        
        issue = DiagnosticIssue(
            category=DiagnosticCategory.MEMORY_ERROR,
            severity="warning",
            title="Test Issue",
            description="Test description",
            affected_components=["test"],
            symptoms=["test"],
            root_cause="test",
            remediation_steps=["test"],
            auto_recoverable=True
        )
        
        self.recovery_manager.attempt_automatic_recovery(issue)
        
        final_count = len(self.recovery_manager.get_recovery_history())
        self.assertEqual(final_count, initial_count + 1)


class TestDiagnosticReportGenerator(unittest.TestCase):
    """Test DiagnosticReportGenerator functionality"""
    
    def setUp(self):
        self.generator = DiagnosticReportGenerator()
        
        # Create test diagnostic results
        self.test_results = DiagnosticResults(
            session_id="test_session",
            start_time=datetime.now(),
            end_time=datetime.now(),
            overall_status="warning"
        )
        
        # Add test issue
        test_issue = DiagnosticIssue(
            category=DiagnosticCategory.MEMORY_ERROR,
            severity="warning",
            title="Test Issue",
            description="Test description",
            affected_components=["test"],
            symptoms=["test symptom"],
            root_cause="test cause",
            remediation_steps=["test step"],
            auto_recoverable=True
        )
        
        self.test_results.issues_found = [test_issue]
    
    def test_generate_comprehensive_report(self):
        """Test comprehensive report generation"""
        report = self.generator.generate_comprehensive_report(self.test_results)
        
        self.assertIsInstance(report, dict)
        self.assertIn("report_metadata", report)
        self.assertIn("executive_summary", report)
        self.assertIn("issues_analysis", report)
        self.assertIn("recommendations", report)
    
    def test_save_report_to_file(self):
        """Test saving report to file"""
        report = {"test": "data"}
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            output_path = f.name
        
        result_path = self.generator.save_report_to_file(report, output_path)
        
        self.assertEqual(result_path, output_path)
        self.assertTrue(Path(output_path).exists())
        
        # Verify content
        with open(output_path, 'r') as f:
            saved_data = json.load(f)
        
        self.assertEqual(saved_data, report)


class TestDiagnosticTool(unittest.TestCase):
    """Test DiagnosticTool main functionality"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.json"
        
        # Create test config
        test_config = {
            "system": {"test": True},
            "directories": {"models": "models"},
            "optimization": {"enable_attention_slicing": False},
            "performance": {"cpu_warning_percent": 80}
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(test_config, f)
        
        self.diagnostic_tool = DiagnosticTool(str(self.config_path))
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_run_comprehensive_diagnosis(self, mock_disk, mock_memory, mock_cpu):
        """Test comprehensive diagnosis"""
        # Mock system metrics
        mock_cpu.return_value = 50.0
        mock_memory.return_value = Mock(percent=60.0, used=8*1024**3, total=16*1024**3, available=8*1024**3)
        mock_disk.return_value = Mock(free=100*1024**3)
        
        results = self.diagnostic_tool.run_comprehensive_diagnosis()
        
        self.assertIsInstance(results, DiagnosticResults)
        self.assertIsNotNone(results.session_id)
        self.assertIsNotNone(results.start_time)
        self.assertIsNotNone(results.end_time)
        self.assertIn(results.overall_status, ["healthy", "warning", "critical", "error"])
    
    @patch('builtins.__import__')
    def test_run_specialized_diagnosis(self, mock_import):
        """Test specialized diagnosis"""
        # Mock imports to avoid dependency issues in tests
        def side_effect(name, *args, **kwargs):
            if name == 'requests':
                mock_requests = Mock()
                mock_response = Mock()
                mock_response.status_code = 200
                mock_requests.get.return_value = mock_response
                mock_requests.exceptions = Mock()
                mock_requests.exceptions.RequestException = Exception
                return mock_requests
            elif name == 'torch':
                mock_torch = Mock()
                mock_torch.cuda.is_available.return_value = False
                return mock_torch
            else:
                # Call the real import for other modules
                return __import__(name, *args, **kwargs)
        
        mock_import.side_effect = side_effect
        
        # Test CUDA diagnosis
        cuda_issues = self.diagnostic_tool.run_specialized_diagnosis("cuda")
        self.assertIsInstance(cuda_issues, list)
        
        # Test memory diagnosis
        memory_issues = self.diagnostic_tool.run_specialized_diagnosis("memory")
        self.assertIsInstance(memory_issues, list)
        
        # Test model download diagnosis
        download_issues = self.diagnostic_tool.run_specialized_diagnosis("model_download")
        self.assertIsInstance(download_issues, list)
    
    def test_integrate_with_error_handler(self):
        """Test integration with error handler"""
        # Create test error info
        error_info = ErrorInfo(
            category=ErrorCategory.VRAM_ERROR,
            error_type="RuntimeError",
            message="CUDA out of memory",
            user_message="GPU memory is full",
            recovery_suggestions=["Clear GPU cache"],
            timestamp=datetime.now(),
            function_name="test_function",
            traceback_info="test traceback",
            system_info={"test": "info"},
            is_recoverable=True
        )
        
        results = self.diagnostic_tool.integrate_with_error_handler(error_info)
        
        self.assertIsInstance(results, DiagnosticResults)
        self.assertGreater(len(results.issues_found), 0)
        self.assertEqual(results.issues_found[0].category, DiagnosticCategory.CUDA_ERROR)


if __name__ == '__main__':
    unittest.main()
