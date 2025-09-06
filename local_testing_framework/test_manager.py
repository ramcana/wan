"""
Central test orchestrator for the local testing framework
"""

import uuid
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from .models.test_results import (
    TestResults, TestStatus, EnvironmentValidationResults, 
    PerformanceTestResults, IntegrationTestResults
)
from .models.configuration import LocalTestConfiguration, TestMode
from .environment_validator import EnvironmentValidator
from .performance_tester import PerformanceTester
from .integration_tester import IntegrationTester
from .diagnostic_tool import DiagnosticTool
from .report_generator import ReportGenerator
from .sample_manager import SampleManager
from .continuous_monitor import ContinuousMonitor
from .production_validator import ProductionValidator


@dataclass
class SessionManager:
    """Manages individual test sessions"""
    session_id: str
    start_time: datetime
    configuration: LocalTestConfiguration
    results: TestResults
    status: TestStatus = TestStatus.ERROR
    current_phase: Optional[str] = None
    error_log: List[str] = field(default_factory=list)
    
    def log_error(self, error: str) -> None:
        """Log an error for this session"""
        self.error_log.append(f"{datetime.now().isoformat()}: {error}")
        logging.error(f"Session {self.session_id}: {error}")
    
    def update_status(self, status: TestStatus, phase: Optional[str] = None) -> None:
        """Update session status and current phase"""
        self.status = status
        if phase:
            self.current_phase = phase
        logging.info(f"Session {self.session_id}: Status updated to {status.value}, Phase: {phase}")


@dataclass
class WorkflowDefinition:
    """Defines test execution workflows"""
    name: str
    phases: List[str]
    required_components: List[str]
    description: str
    
    @classmethod
    def get_full_workflow(cls) -> 'WorkflowDefinition':
        """Get the full test workflow"""
        return cls(
            name="full",
            phases=[
                "environment_validation",
                "performance_testing", 
                "integration_testing",
                "report_generation"
            ],
            required_components=[
                "environment_validator",
                "performance_tester",
                "integration_tester", 
                "report_generator"
            ],
            description="Complete test suite including all validation phases"
        )
    
    @classmethod
    def get_environment_workflow(cls) -> 'WorkflowDefinition':
        """Get environment validation workflow"""
        return cls(
            name="environment",
            phases=["environment_validation"],
            required_components=["environment_validator"],
            description="Environment validation and setup verification"
        )
    
    @classmethod
    def get_performance_workflow(cls) -> 'WorkflowDefinition':
        """Get performance testing workflow"""
        return cls(
            name="performance",
            phases=["performance_testing"],
            required_components=["performance_tester"],
            description="Performance benchmarking and optimization validation"
        )
    
    @classmethod
    def get_integration_workflow(cls) -> 'WorkflowDefinition':
        """Get integration testing workflow"""
        return cls(
            name="integration",
            phases=["integration_testing"],
            required_components=["integration_tester"],
            description="Integration testing and component validation"
        )


class LocalTestManager:
    """Main coordinator for all testing activities"""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the test manager"""
        self.config_path = config_path
        self.configuration = LocalTestConfiguration(config_path=config_path)
        self.active_sessions: Dict[str, SessionManager] = {}
        
        # Initialize components
        self.environment_validator = EnvironmentValidator(self.configuration)
        self.performance_tester = PerformanceTester(config_path)
        self.integration_tester = IntegrationTester(config_path)
        self.diagnostic_tool = DiagnosticTool(config_path)
        self.report_generator = ReportGenerator()
        self.sample_manager = SampleManager(self.configuration)
        self.continuous_monitor = ContinuousMonitor(config_path)
        self.production_validator = ProductionValidator(config_path)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def create_session(self, workflow: WorkflowDefinition) -> SessionManager:
        """Create a new test session"""
        session_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        results = TestResults(
            session_id=session_id,
            start_time=start_time
        )
        
        session = SessionManager(
            session_id=session_id,
            start_time=start_time,
            configuration=self.configuration,
            results=results
        )
        
        self.active_sessions[session_id] = session
        self.logger.info(f"Created new test session: {session_id} for workflow: {workflow.name}")
        
        return session
    
    def run_full_test_suite(self) -> TestResults:
        """Run the complete test suite"""
        workflow = WorkflowDefinition.get_full_workflow()
        session = self.create_session(workflow)
        
        try:
            session.update_status(TestStatus.PARTIAL, "starting")
            
            # Run environment validation
            session.update_status(TestStatus.PARTIAL, "environment_validation")
            env_results = self.run_environment_validation()
            session.results.environment_results = env_results
            
            # Run performance tests
            session.update_status(TestStatus.PARTIAL, "performance_testing")
            perf_results = self.run_performance_tests()
            session.results.performance_results = perf_results
            
            # Run integration tests
            session.update_status(TestStatus.PARTIAL, "integration_testing")
            int_results = self.run_integration_tests()
            session.results.integration_results = int_results
            
            # Generate final status
            session.results.end_time = datetime.now()
            session.results.overall_status = self._determine_overall_status(session.results)
            session.update_status(session.results.overall_status, "completed")
            
            self.logger.info(f"Full test suite completed for session {session.session_id}")
            return session.results
            
        except Exception as e:
            error_msg = f"Error during full test suite: {str(e)}"
            session.log_error(error_msg)
            session.results.overall_status = TestStatus.ERROR
            session.update_status(TestStatus.ERROR, "error")
            raise
        
        finally:
            # Clean up session
            if session.session_id in self.active_sessions:
                del self.active_sessions[session.session_id]
    
    def run_environment_validation(self) -> EnvironmentValidationResults:
        """Run environment validation"""
        workflow = WorkflowDefinition.get_environment_workflow()
        session = self.create_session(workflow)
        
        try:
            session.update_status(TestStatus.PARTIAL, "environment_validation")
            results = self.environment_validator.validate_full_environment()
            session.update_status(TestStatus.PASSED, "completed")
            return results
            
        except Exception as e:
            error_msg = f"Error during environment validation: {str(e)}"
            session.log_error(error_msg)
            session.update_status(TestStatus.ERROR, "error")
            raise
        
        finally:
            if session.session_id in self.active_sessions:
                del self.active_sessions[session.session_id]
    
    def run_performance_tests(self) -> PerformanceTestResults:
        """Run performance tests"""
        workflow = WorkflowDefinition.get_performance_workflow()
        session = self.create_session(workflow)
        
        try:
            session.update_status(TestStatus.PARTIAL, "performance_testing")
            results = self.performance_tester.run_performance_tests()
            session.update_status(TestStatus.PASSED, "completed")
            return results
            
        except Exception as e:
            error_msg = f"Error during performance testing: {str(e)}"
            session.log_error(error_msg)
            session.update_status(TestStatus.ERROR, "error")
            raise
        
        finally:
            if session.session_id in self.active_sessions:
                del self.active_sessions[session.session_id]
    
    def run_integration_tests(self) -> IntegrationTestResults:
        """Run integration tests"""
        workflow = WorkflowDefinition.get_integration_workflow()
        session = self.create_session(workflow)
        
        try:
            session.update_status(TestStatus.PARTIAL, "integration_testing")
            results = self.integration_tester.run_integration_tests()
            session.update_status(TestStatus.PASSED, "completed")
            return results
            
        except Exception as e:
            error_msg = f"Error during integration testing: {str(e)}"
            session.log_error(error_msg)
            session.update_status(TestStatus.ERROR, "error")
            raise
        
        finally:
            if session.session_id in self.active_sessions:
                del self.active_sessions[session.session_id]
    
    def run_diagnostics(self) -> Dict[str, Any]:
        """Run diagnostic analysis"""
        try:
            self.logger.info("Running diagnostic analysis")
            return self.diagnostic_tool.run_diagnostics()
            
        except Exception as e:
            error_msg = f"Error during diagnostics: {str(e)}"
            self.logger.error(error_msg)
            raise
    
    def generate_reports(self, results: TestResults, format_type: str = "html") -> str:
        """Generate test reports"""
        try:
            self.logger.info(f"Generating {format_type} report for session {results.session_id}")
            
            if format_type.lower() == "html":
                return self.report_generator.generate_html_report(results)
            elif format_type.lower() == "json":
                return self.report_generator.generate_json_report(results)
            else:
                raise ValueError(f"Unsupported report format: {format_type}")
                
        except Exception as e:
            error_msg = f"Error generating reports: {str(e)}"
            self.logger.error(error_msg)
            raise
    
    def generate_samples(self, sample_types: List[str] = None) -> Dict[str, Any]:
        """Generate sample data and configurations"""
        try:
            self.logger.info("Generating sample data and configurations")
            
            if sample_types is None:
                sample_types = ["config", "data", "env"]
            
            results = {}
            
            if "config" in sample_types:
                results["config"] = self.sample_manager.generate_config_json_template()
            
            if "data" in sample_types:
                results["data"] = self.sample_manager.generate_sample_input_files(5, ["720p", "1080p"])
            
            if "env" in sample_types:
                results["env"] = self.sample_manager.generate_env_template()
            
            return results
            
        except Exception as e:
            error_msg = f"Error generating samples: {str(e)}"
            self.logger.error(error_msg)
            raise
    
    def run_production_validation(self) -> 'ProductionReadinessResults':
        """Run production readiness validation"""
        try:
            self.logger.info("Running production readiness validation")
            return self.production_validator.validate_production_readiness()
            
        except Exception as e:
            error_msg = f"Error during production validation: {str(e)}"
            self.logger.error(error_msg)
            raise
    
    def start_monitoring(self, duration: int = 3600, enable_alerts: bool = True) -> str:
        """Start continuous monitoring"""
        try:
            self.logger.info(f"Starting continuous monitoring for {duration} seconds")
            monitor_id = self.continuous_monitor.start_monitoring(duration, enable_alerts)
            return monitor_id
            
        except Exception as e:
            error_msg = f"Error starting monitoring: {str(e)}"
            self.logger.error(error_msg)
            raise
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific session"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            return {
                "session_id": session.session_id,
                "status": session.status.value,
                "current_phase": session.current_phase,
                "start_time": session.start_time.isoformat(),
                "error_count": len(session.error_log)
            }
        return None
    
    def list_active_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions"""
        return [
            self.get_session_status(session_id) 
            for session_id in self.active_sessions.keys()
        ]
    
    def _determine_overall_status(self, results: TestResults) -> TestStatus:
        """Determine overall test status based on individual results"""
        statuses = []
        
        if results.environment_results:
            if results.environment_results.overall_status.value == "passed":
                statuses.append(TestStatus.PASSED)
            elif results.environment_results.overall_status.value == "failed":
                statuses.append(TestStatus.FAILED)
            else:
                statuses.append(TestStatus.PARTIAL)
        
        if results.performance_results:
            statuses.append(results.performance_results.overall_status)
        
        if results.integration_results:
            statuses.append(results.integration_results.overall_status)
        
        # Determine overall status
        if not statuses:
            return TestStatus.ERROR
        
        if all(status == TestStatus.PASSED for status in statuses):
            return TestStatus.PASSED
        elif any(status == TestStatus.ERROR for status in statuses):
            return TestStatus.ERROR
        elif any(status == TestStatus.FAILED for status in statuses):
            return TestStatus.FAILED
        else:
            return TestStatus.PARTIAL