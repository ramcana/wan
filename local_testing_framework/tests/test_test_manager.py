"""
Tests for the LocalTestManager and related classes
"""

import pytest
import uuid
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from local_testing_framework.test_manager import (
    LocalTestManager, SessionManager, WorkflowDefinition
)
from local_testing_framework.models.test_results import (
    TestResults, TestStatus, EnvironmentValidationResults,
    PerformanceTestResults, IntegrationTestResults, ValidationStatus
)
from local_testing_framework.models.configuration import TestConfiguration


class TestSessionManager:
    """Test SessionManager class"""
    
    def test_session_initialization(self):
        """Test session initialization"""
        session_id = str(uuid.uuid4())
        start_time = datetime.now()
        config = TestConfiguration()
        results = TestResults(session_id=session_id, start_time=start_time)
        
        session = SessionManager(
            session_id=session_id,
            start_time=start_time,
            configuration=config,
            results=results
        )
        
        assert session.session_id == session_id
        assert session.start_time == start_time
        assert session.configuration == config
        assert session.results == results
        assert session.status == TestStatus.ERROR
        assert session.current_phase is None
        assert len(session.error_log) == 0
    
    def test_log_error(self):
        """Test error logging"""
        session_id = str(uuid.uuid4())
        start_time = datetime.now()
        config = TestConfiguration()
        results = TestResults(session_id=session_id, start_time=start_time)
        
        session = SessionManager(
            session_id=session_id,
            start_time=start_time,
            configuration=config,
            results=results
        )
        
        error_message = "Test error message"
        session.log_error(error_message)
        
        assert len(session.error_log) == 1
        assert error_message in session.error_log[0]
    
    def test_update_status(self):
        """Test status updates"""
        session_id = str(uuid.uuid4())
        start_time = datetime.now()
        config = TestConfiguration()
        results = TestResults(session_id=session_id, start_time=start_time)
        
        session = SessionManager(
            session_id=session_id,
            start_time=start_time,
            configuration=config,
            results=results
        )
        
        session.update_status(TestStatus.PARTIAL, "testing_phase")
        
        assert session.status == TestStatus.PARTIAL
        assert session.current_phase == "testing_phase"


class TestWorkflowDefinition:
    """Test WorkflowDefinition class"""
    
    def test_full_workflow(self):
        """Test full workflow creation"""
        workflow = WorkflowDefinition.get_full_workflow()
        
        assert workflow.name == "full"
        assert "environment_validation" in workflow.phases
        assert "performance_testing" in workflow.phases
        assert "integration_testing" in workflow.phases
        assert "report_generation" in workflow.phases
        assert "environment_validator" in workflow.required_components
    
    def test_environment_workflow(self):
        """Test environment workflow creation"""
        workflow = WorkflowDefinition.get_environment_workflow()
        
        assert workflow.name == "environment"
        assert workflow.phases == ["environment_validation"]
        assert workflow.required_components == ["environment_validator"]
    
    def test_performance_workflow(self):
        """Test performance workflow creation"""
        workflow = WorkflowDefinition.get_performance_workflow()
        
        assert workflow.name == "performance"
        assert workflow.phases == ["performance_testing"]
        assert workflow.required_components == ["performance_tester"]
    
    def test_integration_workflow(self):
        """Test integration workflow creation"""
        workflow = WorkflowDefinition.get_integration_workflow()
        
        assert workflow.name == "integration"
        assert workflow.phases == ["integration_testing"]
        assert workflow.required_components == ["integration_tester"]


class TestLocalTestManager:
    """Test LocalTestManager class"""
    
    @pytest.fixture
    def mock_components(self):
        """Mock all component dependencies"""
        with patch('local_testing_framework.test_manager.EnvironmentValidator') as mock_env, \
             patch('local_testing_framework.test_manager.PerformanceTester') as mock_perf, \
             patch('local_testing_framework.test_manager.IntegrationTester') as mock_int, \
             patch('local_testing_framework.test_manager.DiagnosticTool') as mock_diag, \
             patch('local_testing_framework.test_manager.ReportGenerator') as mock_report, \
             patch('local_testing_framework.test_manager.SampleManager') as mock_sample, \
             patch('local_testing_framework.test_manager.ContinuousMonitor') as mock_monitor:
            
            yield {
                'env': mock_env,
                'perf': mock_perf,
                'int': mock_int,
                'diag': mock_diag,
                'report': mock_report,
                'sample': mock_sample,
                'monitor': mock_monitor
            }
    
    def test_manager_initialization(self, mock_components):
        """Test manager initialization"""
        manager = LocalTestManager("test_config.json")
        
        assert manager.config_path == "test_config.json"
        assert isinstance(manager.configuration, TestConfiguration)
        assert len(manager.active_sessions) == 0
        
        # Verify components were initialized with TestConfiguration objects
        mock_components['env'].assert_called_once()
        mock_components['perf'].assert_called_once_with("test_config.json")
        mock_components['int'].assert_called_once_with("test_config.json")
    
    def test_create_session(self, mock_components):
        """Test session creation"""
        manager = LocalTestManager()
        workflow = WorkflowDefinition.get_full_workflow()
        
        session = manager.create_session(workflow)
        
        assert session.session_id in manager.active_sessions
        assert isinstance(session.results, TestResults)
        assert session.configuration == manager.configuration
    
    @patch('local_testing_framework.test_manager.logging')
    def test_run_environment_validation(self, mock_logging, mock_components):
        """Test environment validation execution"""
        manager = LocalTestManager()
        
        # Mock the environment validator
        mock_env_results = Mock(spec=EnvironmentValidationResults)
        manager.environment_validator.validate_full_environment.return_value = mock_env_results
        
        results = manager.run_environment_validation()
        
        assert results == mock_env_results
        manager.environment_validator.validate_full_environment.assert_called_once()
    
    @patch('local_testing_framework.test_manager.logging')
    def test_run_performance_tests(self, mock_logging, mock_components):
        """Test performance testing execution"""
        manager = LocalTestManager()
        
        # Mock the performance tester
        mock_perf_results = Mock(spec=PerformanceTestResults)
        manager.performance_tester.run_performance_tests.return_value = mock_perf_results
        
        results = manager.run_performance_tests()
        
        assert results == mock_perf_results
        manager.performance_tester.run_performance_tests.assert_called_once()
    
    @patch('local_testing_framework.test_manager.logging')
    def test_run_integration_tests(self, mock_logging, mock_components):
        """Test integration testing execution"""
        manager = LocalTestManager()
        
        # Mock the integration tester
        mock_int_results = Mock(spec=IntegrationTestResults)
        manager.integration_tester.run_integration_tests.return_value = mock_int_results
        
        results = manager.run_integration_tests()
        
        assert results == mock_int_results
        manager.integration_tester.run_integration_tests.assert_called_once()
    
    @patch('local_testing_framework.test_manager.logging')
    def test_run_diagnostics(self, mock_logging, mock_components):
        """Test diagnostic execution"""
        manager = LocalTestManager()
        
        # Mock the diagnostic tool
        mock_diag_results = {"status": "healthy", "issues": []}
        manager.diagnostic_tool.run_diagnostics.return_value = mock_diag_results
        
        results = manager.run_diagnostics()
        
        assert results == mock_diag_results
        manager.diagnostic_tool.run_diagnostics.assert_called_once()
    
    @patch('local_testing_framework.test_manager.logging')
    def test_generate_reports_html(self, mock_logging, mock_components):
        """Test HTML report generation"""
        manager = LocalTestManager()
        
        # Create test results
        test_results = TestResults(
            session_id="test-session",
            start_time=datetime.now()
        )
        
        # Mock the report generator
        mock_html_report = "<html>Test Report</html>"
        manager.report_generator.generate_html_report.return_value = mock_html_report
        
        result = manager.generate_reports(test_results, "html")
        
        assert result == mock_html_report
        manager.report_generator.generate_html_report.assert_called_once_with(test_results)
    
    @patch('local_testing_framework.test_manager.logging')
    def test_generate_reports_json(self, mock_logging, mock_components):
        """Test JSON report generation"""
        manager = LocalTestManager()
        
        # Create test results
        test_results = TestResults(
            session_id="test-session",
            start_time=datetime.now()
        )
        
        # Mock the report generator
        mock_json_report = '{"status": "test"}'
        manager.report_generator.generate_json_report.return_value = mock_json_report
        
        result = manager.generate_reports(test_results, "json")
        
        assert result == mock_json_report
        manager.report_generator.generate_json_report.assert_called_once_with(test_results)
    
    @patch('local_testing_framework.test_manager.logging')
    def test_generate_samples(self, mock_logging, mock_components):
        """Test sample generation"""
        manager = LocalTestManager()
        
        # Mock the sample manager
        manager.sample_manager.generate_config_json_template.return_value = {"config": "test"}
        manager.sample_manager.generate_sample_input_files.return_value = [{"input": "test"}]
        manager.sample_manager.generate_env_template.return_value = {"env": "test"}
        
        results = manager.generate_samples(["config", "data", "env"])
        
        assert "config" in results
        assert "data" in results
        assert "env" in results
        
        manager.sample_manager.generate_config_json_template.assert_called_once()
        manager.sample_manager.generate_sample_input_files.assert_called_once_with(5, ["720p", "1080p"])
        manager.sample_manager.generate_env_template.assert_called_once()
    
    @patch('local_testing_framework.test_manager.logging')
    def test_start_monitoring(self, mock_logging, mock_components):
        """Test continuous monitoring start"""
        manager = LocalTestManager()
        
        # Mock the continuous monitor
        mock_monitor_id = "monitor-123"
        manager.continuous_monitor.start_monitoring.return_value = mock_monitor_id
        
        result = manager.start_monitoring(3600, True)
        
        assert result == mock_monitor_id
        manager.continuous_monitor.start_monitoring.assert_called_once_with(3600, True)
    
    def test_get_session_status(self, mock_components):
        """Test session status retrieval"""
        manager = LocalTestManager()
        workflow = WorkflowDefinition.get_full_workflow()
        
        session = manager.create_session(workflow)
        session.update_status(TestStatus.PARTIAL, "testing")
        
        status = manager.get_session_status(session.session_id)
        
        assert status is not None
        assert status["session_id"] == session.session_id
        assert status["status"] == TestStatus.PARTIAL.value
        assert status["current_phase"] == "testing"
    
    def test_list_active_sessions(self, mock_components):
        """Test active sessions listing"""
        manager = LocalTestManager()
        workflow = WorkflowDefinition.get_full_workflow()
        
        # Create multiple sessions
        session1 = manager.create_session(workflow)
        session2 = manager.create_session(workflow)
        
        sessions = manager.list_active_sessions()
        
        assert len(sessions) == 2
        session_ids = [s["session_id"] for s in sessions]
        assert session1.session_id in session_ids
        assert session2.session_id in session_ids
    
    def test_determine_overall_status_all_passed(self, mock_components):
        """Test overall status determination when all tests pass"""
        manager = LocalTestManager()
        
        # Create test results with all passed
        results = TestResults(
            session_id="test",
            start_time=datetime.now()
        )
        
        # Mock environment results
        env_results = Mock()
        env_results.overall_status = Mock()
        env_results.overall_status.value = "passed"
        results.environment_results = env_results
        
        # Mock performance results
        perf_results = Mock()
        perf_results.overall_status = TestStatus.PASSED
        results.performance_results = perf_results
        
        # Mock integration results
        int_results = Mock()
        int_results.overall_status = TestStatus.PASSED
        results.integration_results = int_results
        
        status = manager._determine_overall_status(results)
        
        assert status == TestStatus.PASSED
    
    def test_determine_overall_status_with_failures(self, mock_components):
        """Test overall status determination with failures"""
        manager = LocalTestManager()
        
        # Create test results with failures
        results = TestResults(
            session_id="test",
            start_time=datetime.now()
        )
        
        # Mock environment results
        env_results = Mock()
        env_results.overall_status = Mock()
        env_results.overall_status.value = "failed"
        results.environment_results = env_results
        
        # Mock performance results
        perf_results = Mock()
        perf_results.overall_status = TestStatus.PASSED
        results.performance_results = perf_results
        
        status = manager._determine_overall_status(results)
        
        assert status == TestStatus.FAILED


if __name__ == "__main__":
    pytest.main([__file__])