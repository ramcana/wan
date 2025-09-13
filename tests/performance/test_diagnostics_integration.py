"""
Integration tests for the diagnostics system.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from scripts.startup_manager.diagnostics import (
    SystemDiagnostics,
    LogAnalyzer,
    DiagnosticMode,
    SystemInfo,
    DiagnosticResult,
    LogAnalysisResult
)


class TestSystemDiagnosticsIntegration:
    """Integration tests for SystemDiagnostics."""
    
    def setup_method(self):
        """Setup test environment."""
        self.diagnostics = SystemDiagnostics()
    
    def test_collect_system_info_integration(self):
        """Test collecting real system information."""
        system_info = self.diagnostics.collect_system_info()
        
        assert isinstance(system_info, SystemInfo)
        assert system_info.os_name is not None
        assert system_info.python_version is not None
        assert system_info.memory_total > 0
        assert isinstance(system_info.network_interfaces, list)
        assert isinstance(system_info.environment_variables, dict)
        assert isinstance(system_info.installed_packages, list)
    
    def test_run_diagnostic_checks_integration(self):
        """Test running real diagnostic checks."""
        results = self.diagnostics.run_diagnostic_checks()
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Check that all results have required fields
        for result in results:
            assert isinstance(result, DiagnosticResult)
            assert result.check_name is not None
            assert result.status in ["pass", "fail", "warning"]
            assert result.message is not None
    
    def test_python_version_check_integration(self):
        """Test Python version check with real system."""
        result = self.diagnostics._check_python_version()
        
        assert isinstance(result, DiagnosticResult)
        assert result.check_name == "Python Version"
        assert result.status in ["pass", "warning", "fail"]
        # Should pass on most modern systems
        assert result.status in ["pass", "warning"]
    
    def test_virtual_environment_check_integration(self):
        """Test virtual environment check."""
        result = self.diagnostics._check_virtual_environment()
        
        assert isinstance(result, DiagnosticResult)
        assert result.check_name == "Virtual Environment"
        assert result.status in ["pass", "warning"]
    
    def test_disk_space_check_integration(self):
        """Test disk space check with real system."""
        result = self.diagnostics._check_disk_space()
        
        assert isinstance(result, DiagnosticResult)
        assert result.check_name == "Disk Space"
        assert result.status in ["pass", "warning", "fail"]
    
    def test_memory_usage_check_integration(self):
        """Test memory usage check with real system."""
        result = self.diagnostics._check_memory_usage()
        
        assert isinstance(result, DiagnosticResult)
        assert result.check_name == "Memory Usage"
        assert result.status in ["pass", "warning", "fail"]
    
    def test_file_permissions_check_integration(self):
        """Test file permissions check with real system."""
        result = self.diagnostics._check_file_permissions()
        
        assert isinstance(result, DiagnosticResult)
        assert result.check_name == "File Permissions"
        # Should pass in most cases
        assert result.status in ["pass", "warning"]


class TestLogAnalyzerIntegration:
    """Integration tests for LogAnalyzer."""
    
    def setup_method(self):
        """Setup test environment."""
        self.analyzer = LogAnalyzer()
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir) / "logs"
        self.log_dir.mkdir(exist_ok=True)
    
    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_analyze_empty_logs(self):
        """Test analyzing empty log directory."""
        result = self.analyzer.analyze_logs(self.log_dir)
        
        assert isinstance(result, LogAnalysisResult)
        assert result.total_entries == 0
        assert result.error_count == 0
        assert result.warning_count == 0
        assert len(result.common_errors) == 0
        assert len(result.suggestions) > 0
    
    def test_analyze_text_logs_with_errors(self):
        """Test analyzing text logs containing errors."""
        # Create test log file with various error types
        log_file = self.log_dir / "test.log"
        log_content = """
2023-01-01 12:00:00 | INFO     | StartupManager | Starting application
2023-01-01 12:00:01 | ERROR    | StartupManager | Address already in use
2023-01-01 12:00:02 | WARNING  | StartupManager | High memory usage detected
2023-01-01 12:00:03 | ERROR    | StartupManager | Permission denied
2023-01-01 12:00:04 | INFO     | StartupManager | Retrying operation
2023-01-01 12:00:05 | ERROR    | StartupManager | ModuleNotFoundError: No module named 'test'
        """
        log_file.write_text(log_content.strip())
        
        result = self.analyzer.analyze_logs(self.log_dir)
        
        assert result.total_entries == 6
        assert result.error_count == 3
        assert result.warning_count == 1
        assert len(result.common_errors) > 0
        
        # Check that error categories were detected
        error_categories = [error["category"] for error in result.common_errors]
        assert "port_conflict" in error_categories
        assert "permission_denied" in error_categories
        assert "module_not_found" in error_categories
    
    def test_analyze_json_logs_with_performance_data(self):
        """Test analyzing JSON logs with performance metrics."""
        # Create test JSON log file
        json_file = self.log_dir / "test.json"
        
        log_entries = [
            {
                "timestamp": "2023-01-01T12:00:00",
                "level": "INFO",
                "logger_name": "StartupManager",
                "message": "PERFORMANCE: startup completed in 2.50s",
                "extra_data": {
                    "metric_type": "performance",
                    "operation": "startup",
                    "duration": 2.5
                }
            },
            {
                "timestamp": "2023-01-01T12:00:01",
                "level": "ERROR",
                "logger_name": "StartupManager",
                "message": "Connection refused",
                "extra_data": {"error_context": {"step": "network_check"}}
            },
            {
                "timestamp": "2023-01-01T12:00:02",
                "level": "INFO",
                "logger_name": "StartupManager",
                "message": "PERFORMANCE: validation completed in 1.20s",
                "extra_data": {
                    "metric_type": "performance",
                    "operation": "validation",
                    "duration": 1.2
                }
            }
        ]
        
        with open(json_file, 'w') as f:
            for entry in log_entries:
                f.write(json.dumps(entry) + '\n')
        
        result = self.analyzer.analyze_logs(self.log_dir)
        
        assert result.total_entries == 3
        assert result.error_count == 1
        assert len(result.performance_metrics) > 0
        
        # Check performance metrics
        assert "startup" in result.performance_metrics
        assert "validation" in result.performance_metrics
        assert result.performance_metrics["startup"]["avg_duration"] == 2.5
        assert result.performance_metrics["validation"]["avg_duration"] == 1.2
    
    def test_analyze_mixed_log_formats(self):
        """Test analyzing both text and JSON logs together."""
        # Create text log
        text_log = self.log_dir / "text.log"
        text_log.write_text("2023-01-01 12:00:00 | ERROR | Test | Address already in use")
        
        # Create JSON log
        json_log = self.log_dir / "json.json"
        json_entry = {
            "timestamp": "2023-01-01T12:00:01",
            "level": "WARNING",
            "logger_name": "Test",
            "message": "High memory usage"
        }
        json_log.write_text(json.dumps(json_entry))
        
        result = self.analyzer.analyze_logs(self.log_dir)
        
        assert result.total_entries == 2
        assert result.error_count == 1
        assert result.warning_count == 1
        assert len(result.common_errors) > 0


class TestDiagnosticModeIntegration:
    """Integration tests for DiagnosticMode."""
    
    def setup_method(self):
        """Setup test environment."""
        self.diagnostic_mode = DiagnosticMode()
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir) / "logs"
        self.log_dir.mkdir(exist_ok=True)
    
    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_run_full_diagnostics_integration(self):
        """Test running full diagnostics with real system."""
        # Create some test logs
        log_file = self.log_dir / "test.log"
        log_file.write_text("2023-01-01 12:00:00 | INFO | Test | Application started")
        
        result = self.diagnostic_mode.run_full_diagnostics(self.log_dir)
        
        assert isinstance(result, dict)
        assert "timestamp" in result
        assert "system_info" in result
        assert "diagnostic_checks" in result
        assert "log_analysis" in result
        assert "summary" in result
        
        # Check system info
        assert result["system_info"] is not None
        assert "os_name" in result["system_info"]
        assert "python_version" in result["system_info"]
        
        # Check diagnostic checks
        assert isinstance(result["diagnostic_checks"], list)
        assert len(result["diagnostic_checks"]) > 0
        
        # Check log analysis
        assert result["log_analysis"] is not None
        assert "total_entries" in result["log_analysis"]
        
        # Check summary
        assert result["summary"] is not None
        assert "overall_status" in result["summary"]
        assert result["summary"]["overall_status"] in ["healthy", "warnings_present", "issues_detected"]
    
    def test_save_diagnostic_report_integration(self):
        """Test saving diagnostic report to file."""
        # Create minimal diagnostic data
        diagnostic_data = {
            "timestamp": "2023-01-01T12:00:00",
            "system_info": {"os_name": "Windows"},
            "diagnostic_checks": [],
            "log_analysis": {"total_entries": 0},
            "summary": {"overall_status": "healthy"}
        }
        
        output_file = Path(self.temp_dir) / "test_report.json"
        saved_path = self.diagnostic_mode.save_diagnostic_report(diagnostic_data, output_file)
        
        assert saved_path == output_file
        assert output_file.exists()
        
        # Verify file content
        with open(output_file, 'r') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == diagnostic_data
    
    def test_diagnostic_mode_with_errors(self):
        """Test diagnostic mode handling of system errors."""
        # Create logs with errors
        error_log = self.log_dir / "errors.log"
        error_content = """
2023-01-01 12:00:00 | ERROR | Test | Address already in use
2023-01-01 12:00:01 | ERROR | Test | Permission denied
2023-01-01 12:00:02 | WARNING | Test | High memory usage
        """
        error_log.write_text(error_content.strip())
        
        result = self.diagnostic_mode.run_full_diagnostics(self.log_dir)
        
        # Should detect errors in logs
        log_analysis = result["log_analysis"]
        assert log_analysis["error_count"] > 0
        assert log_analysis["warning_count"] > 0
        assert len(log_analysis["common_errors"]) > 0
        
        # Summary should reflect issues
        summary = result["summary"]
        assert summary["overall_status"] in ["warnings_present", "issues_detected"]
        assert len(summary["recommendations"]) > 0


class TestDiagnosticsEndToEnd:
    """End-to-end integration tests for the entire diagnostics system."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir) / "logs"
        self.log_dir.mkdir(exist_ok=True)
    
    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_diagnostic_workflow(self):
        """Test complete diagnostic workflow from start to finish."""
        # Create realistic log scenario
        self._create_realistic_logs()
        
        # Run diagnostics
        diagnostic_mode = DiagnosticMode()
        result = diagnostic_mode.run_full_diagnostics(self.log_dir)
        
        # Save report
        report_file = Path(self.temp_dir) / "diagnostic_report.json"
        saved_path = diagnostic_mode.save_diagnostic_report(result, report_file)
        
        # Verify complete workflow
        assert saved_path.exists()
        assert result["system_info"] is not None
        assert len(result["diagnostic_checks"]) > 0
        assert result["log_analysis"]["total_entries"] > 0
        assert result["summary"]["overall_status"] is not None
        
        # Verify report can be loaded
        with open(saved_path, 'r') as f:
            loaded_report = json.load(f)
        
        assert loaded_report == result
    
    def test_diagnostic_performance(self):
        """Test diagnostic performance with larger datasets."""
        # Create larger log files
        self._create_large_logs()
        
        import time
        start_time = time.time()
        
        diagnostic_mode = DiagnosticMode()
        result = diagnostic_mode.run_full_diagnostics(self.log_dir)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time (adjust as needed)
        assert duration < 30  # 30 seconds max
        assert result["log_analysis"]["total_entries"] > 100
    
    def _create_realistic_logs(self):
        """Create realistic log files for testing."""
        # Text log with mixed content
        text_log = self.log_dir / "startup_20230101.log"
        text_content = """
2023-01-01 12:00:00 | INFO     | StartupManager | Starting application
2023-01-01 12:00:01 | DEBUG    | EnvironmentValidator | Checking Python version
2023-01-01 12:00:02 | INFO     | EnvironmentValidator | Python 3.11.4 detected
2023-01-01 12:00:03 | WARNING  | PortManager | Port 8000 is already in use
2023-01-01 12:00:04 | INFO     | PortManager | Using alternative port 8001
2023-01-01 12:00:05 | ERROR    | ProcessManager | Failed to start backend: Permission denied
2023-01-01 12:00:06 | INFO     | RecoveryEngine | Attempting recovery
2023-01-01 12:00:07 | INFO     | ProcessManager | Backend started successfully on port 8001
2023-01-01 12:00:08 | INFO     | ProcessManager | Frontend started on port 3000
2023-01-01 12:00:09 | INFO     | StartupManager | Application startup completed
        """
        text_log.write_text(text_content.strip())
        
        # JSON log with performance data
        json_log = self.log_dir / "startup_20230101.json"
        json_entries = [
            {
                "timestamp": "2023-01-01T12:00:00",
                "level": "INFO",
                "logger_name": "StartupManager",
                "message": "PERFORMANCE: environment_validation completed in 0.50s",
                "extra_data": {
                    "metric_type": "performance",
                    "operation": "environment_validation",
                    "duration": 0.5
                }
            },
            {
                "timestamp": "2023-01-01T12:00:05",
                "level": "ERROR",
                "logger_name": "ProcessManager",
                "message": "Failed to start backend: Permission denied",
                "extra_data": {
                    "error_context": {"operation": "start_backend", "port": 8000},
                    "error_type": "PermissionError"
                }
            },
            {
                "timestamp": "2023-01-01T12:00:09",
                "level": "INFO",
                "logger_name": "StartupManager",
                "message": "PERFORMANCE: total_startup completed in 9.00s",
                "extra_data": {
                    "metric_type": "performance",
                    "operation": "total_startup",
                    "duration": 9.0
                }
            }
        ]
        
        with open(json_log, 'w') as f:
            for entry in json_entries:
                f.write(json.dumps(entry) + '\n')
    
    def _create_large_logs(self):
        """Create larger log files for performance testing."""
        text_log = self.log_dir / "large.log"
        
        # Generate 200 log entries
        with open(text_log, 'w') as f:
            for i in range(200):
                level = "INFO" if i % 10 != 0 else ("ERROR" if i % 20 == 0 else "WARNING")
                f.write(f"2023-01-01 12:{i//60:02d}:{i%60:02d} | {level:8s} | Test | Log entry {i}\n")


if __name__ == "__main__":
    pytest.main([__file__])
