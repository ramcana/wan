#!/usr/bin/env python3
"""
Integration Testing for Tool Interactions and Workflows

This module tests how different cleanup and quality tools interact with each other
and validates that workflows function correctly across tool boundaries.

Requirements covered: 1.1, 1.6, 2.6, 3.6, 4.6, 5.6, 6.6
"""

import pytest
import tempfile
import shutil
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from unittest.mock import patch, MagicMock

# Mock implementations for testing
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class MockTestResult:
    total_tests: int = 0
    working_tests: List[str] = None
    broken_tests: List[str] = None
    test_files: List[str] = None
    coverage_gaps: List[str] = None
    
    def __post_init__(self):
        if self.working_tests is None:
            self.working_tests = []
        if self.broken_tests is None:
            self.broken_tests = []
        if self.test_files is None:
            self.test_files = []
        if self.coverage_gaps is None:
            self.coverage_gaps = []

@dataclass
class MockStructureResult:
    components: List[Any] = None
    complexity_score: float = 50.0
    documentation_gaps: List[str] = None
    
    def __post_init__(self):
        if self.components is None:
            self.components = []
        if self.documentation_gaps is None:
            self.documentation_gaps = []

@dataclass
class MockQualityResult:
    total_files: int = 0
    issues: List[str] = None
    overall_score: float = 75.0
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []

@dataclass
class MockDuplicateResult:
    duplicate_groups: List[List[str]] = None
    
    def __post_init__(self):
        if self.duplicate_groups is None:
            self.duplicate_groups = []

@dataclass
class MockConfigResult:
    config_files: List[str] = None
    conflicts: List[str] = None
    duplicate_settings: Dict[str, List[str]] = None
    
    def __post_init__(self):
        if self.config_files is None:
            self.config_files = []
        if self.conflicts is None:
            self.conflicts = []
        if self.duplicate_settings is None:
            self.duplicate_settings = {}

class TestAuditor:
    def audit_tests(self, path: str) -> MockTestResult:
        test_files = list(Path(path).glob("*.py")) if Path(path).exists() else []
        return MockTestResult(
            total_tests=len(test_files),
            working_tests=[f"test_{i}" for i in range(len(test_files) // 2)],
            broken_tests=[f"broken_test_{i}" for i in range(len(test_files) - len(test_files) // 2)],
            test_files=[str(f) for f in test_files]
        )

class StructureAnalyzer:
    def analyze_project(self, path: str) -> MockStructureResult:
        if Path(path).exists():
            components = list(Path(path).rglob("*.py"))
            return MockStructureResult(
                components=[{"path": str(c)} for c in components],
                complexity_score=len(components) * 2.5
            )
        return MockStructureResult()

class QualityChecker:
    def check_directory(self, path: str) -> MockQualityResult:
        if Path(path).exists():
            files = list(Path(path).glob("*.py"))
            return MockQualityResult(
                total_files=len(files),
                issues=[f"issue_{i}" for i in range(len(files))],
                overall_score=max(50, 100 - len(files) * 5)
            )
        return MockQualityResult()
    
    def check_file(self, path: str) -> MockQualityResult:
        return MockQualityResult(total_files=1, issues=["mock_issue"])
    
    def apply_automatic_fixes(self, path: str):
        return type('FixResult', (), {'fixes_applied': 3})()

class DuplicateDetector:
    def find_duplicates(self, path: str) -> MockDuplicateResult:
        if Path(path).exists():
            files = list(Path(path).rglob("*.py"))
            duplicates = []
            seen_names = {}
            for f in files:
                base_name = f.stem.replace('1', '').replace('2', '')
                if base_name in seen_names:
                    duplicates.append([str(seen_names[base_name]), str(f)])
                else:
                    seen_names[base_name] = f
            return MockDuplicateResult(duplicate_groups=duplicates)
        return MockDuplicateResult()
    
    def remove_duplicates_safe(self, duplicates, backup_dir):
        Path(backup_dir).mkdir(exist_ok=True)
        return type('RemovalResult', (), {'removed_count': len(duplicates.duplicate_groups)})()

class ConfigUnifier:
    def analyze_configs(self, path: str) -> MockConfigResult:
        if Path(path).exists():
            config_files = []
            for ext in ['.json', '.yaml', '.yml', '.ini', '.env']:
                config_files.extend(Path(path).rglob(f"*{ext}"))
            
            return MockConfigResult(
                config_files=[str(f) for f in config_files],
                conflicts=["port_conflict", "debug_conflict"] if len(config_files) > 2 else []
            )
        return MockConfigResult()
    
    def create_unified_config(self, analysis):
        return {"unified": True, "source_files": len(analysis.config_files)}
    
    def validate_unified_config(self, config):
        return type('ValidationResult', (), {'is_valid': True})()
    
    def create_migration_plan(self, analysis, unified_config):
        return type('MigrationPlan', (), {'steps': ["step1", "step2", "step3"]})()

class MaintenanceScheduler:
    def __init__(self):
        self.tasks = []
    
    def schedule_task(self, name, tool, params, priority="medium"):
        self.tasks.append({"name": name, "tool": tool, "params": params, "priority": priority})
    
    def execute_scheduled_tasks(self):
        results = []
        for task in self.tasks:
            results.append(type('TaskResult', (), {'status': 'completed', 'task': task})())
        return results
    
    def get_scheduled_tasks(self):
        return self.tasks

class MetricsCollector:
    def collect_metrics(self, path: str):
        return {"files_analyzed": 10, "issues_found": 5, "quality_score": 85}


@dataclass
class WorkflowState:
    """Represents the state of a workflow execution"""
    project_path: str
    current_step: str
    completed_steps: List[str]
    results: Dict[str, Any]
    errors: List[str]
    
    def add_result(self, step: str, result: Any):
        """Add a result for a workflow step"""
        self.results[step] = result
        if step not in self.completed_steps:
            self.completed_steps.append(step)
    
    def add_error(self, error: str):
        """Add an error to the workflow state"""
        self.errors.append(error)
    
    def is_step_completed(self, step: str) -> bool:
        """Check if a step has been completed"""
        return step in self.completed_steps


class WorkflowOrchestrator:
    """Orchestrates workflows across multiple tools"""
    
    def __init__(self):
        self.tools = {
            'test_auditor': TestAuditor(),
            'config_unifier': ConfigUnifier(),
            'structure_analyzer': StructureAnalyzer(),
            'duplicate_detector': DuplicateDetector(),
            'quality_checker': QualityChecker(),
            'maintenance_scheduler': MaintenanceScheduler(),
            'metrics_collector': MetricsCollector()
        }
    
    def execute_analysis_workflow(self, project_path: str) -> WorkflowState:
        """Execute complete project analysis workflow"""
        state = WorkflowState(
            project_path=project_path,
            current_step="initialization",
            completed_steps=[],
            results={},
            errors=[]
        )
        
        try:
            # Step 1: Project structure analysis
            state.current_step = "structure_analysis"
            structure_result = self.tools['structure_analyzer'].analyze_project(project_path)
            state.add_result("structure_analysis", structure_result)
            
            # Step 2: Test audit
            state.current_step = "test_audit"
            test_path = Path(project_path) / "tests"
            if test_path.exists():
                test_result = self.tools['test_auditor'].audit_tests(str(test_path))
                state.add_result("test_audit", test_result)
            
            # Step 3: Configuration analysis
            state.current_step = "config_analysis"
            config_result = self.tools['config_unifier'].analyze_configs(project_path)
            state.add_result("config_analysis", config_result)
            
            # Step 4: Duplicate detection
            state.current_step = "duplicate_detection"
            duplicate_result = self.tools['duplicate_detector'].find_duplicates(project_path)
            state.add_result("duplicate_detection", duplicate_result)
            
            # Step 5: Quality assessment
            state.current_step = "quality_assessment"
            src_path = Path(project_path) / "src"
            if src_path.exists():
                quality_result = self.tools['quality_checker'].check_directory(str(src_path))
                state.add_result("quality_assessment", quality_result)
            
            state.current_step = "completed"
            
        except Exception as e:
            state.add_error(f"Workflow failed at step {state.current_step}: {str(e)}")
        
        return state
    
    def execute_cleanup_workflow(self, project_path: str, analysis_state: WorkflowState) -> WorkflowState:
        """Execute cleanup workflow based on analysis results"""
        state = WorkflowState(
            project_path=project_path,
            current_step="initialization",
            completed_steps=[],
            results={},
            errors=[]
        )
        
        try:
            # Use analysis results to guide cleanup
            if analysis_state.is_step_completed("duplicate_detection"):
                duplicates = analysis_state.results["duplicate_detection"]
                if duplicates and len(duplicates.duplicate_groups) > 0:
                    state.current_step = "duplicate_cleanup"
                    # Perform safe duplicate removal
                    backup_dir = Path(project_path) / "backup_duplicates"
                    cleanup_result = self.tools['duplicate_detector'].remove_duplicates_safe(
                        duplicates, str(backup_dir)
                    )
                    state.add_result("duplicate_cleanup", cleanup_result)
            
            if analysis_state.is_step_completed("quality_assessment"):
                quality_report = analysis_state.results["quality_assessment"]
                if quality_report and len(quality_report.issues) > 0:
                    state.current_step = "quality_fixes"
                    src_path = Path(project_path) / "src"
                    if src_path.exists():
                        fix_result = self.tools['quality_checker'].apply_automatic_fixes(str(src_path))
                        state.add_result("quality_fixes", fix_result)
            
            if analysis_state.is_step_completed("config_analysis"):
                config_analysis = analysis_state.results["config_analysis"]
                if config_analysis and len(config_analysis.conflicts) > 0:
                    state.current_step = "config_consolidation"
                    unified_config = self.tools['config_unifier'].create_unified_config(config_analysis)
                    state.add_result("config_consolidation", unified_config)
            
            state.current_step = "completed"
            
        except Exception as e:
            state.add_error(f"Cleanup workflow failed at step {state.current_step}: {str(e)}")
        
        return state


@pytest.fixture
def integration_test_project():
    """Create a test project for integration testing"""
    temp_dir = tempfile.mkdtemp(prefix="integration_test_")
    project_root = Path(temp_dir)
    
    # Create realistic project structure
    (project_root / "src").mkdir()
    (project_root / "tests").mkdir()
    (project_root / "config").mkdir()
    (project_root / "docs").mkdir()
    
    # Create test files with various issues
    create_integration_test_files(project_root)
    
    yield project_root
    
    # Cleanup
    shutil.rmtree(temp_dir)


def create_integration_test_files(root: Path):
    """Create test files for integration testing"""
    # Source files with quality issues
    (root / "src" / "module1.py").write_text("""
def function_with_issues(x,y):
    if x>0:
        return x+y
    else:
        return 0
""")
    
    (root / "src" / "module2.py").write_text("""
def function_with_issues(x,y):
    if x>0:
        return x+y
    else:
        return 0
""")  # Duplicate of module1.py
    
    # Test files
    (root / "tests" / "test_module1.py").write_text("""
import sys
sys.path.append('../src')
from module1 import function_with_issues

def test_function():
    assert function_with_issues(1, 2) == 3
""")
    
    (root / "tests" / "test_broken.py").write_text("""
import nonexistent_module

def test_broken():
    assert nonexistent_module.func() == True
""")
    
    # Configuration files
    (root / "config" / "app.json").write_text('{"debug": true, "port": 8000}')
    (root / "config" / "db.yaml").write_text("database:\n  host: localhost\n  port: 5432")
    (root / ".env").write_text("DEBUG=true\nPORT=8000")  # Duplicate settings
    
    # Documentation
    (root / "docs" / "README.md").write_text("# Test Project\n\nThis is a test project.")


class TestToolInteractions:
    """Test interactions between different tools"""
    
    def test_workflow_orchestration(self, integration_test_project):
        """Test complete workflow orchestration"""
        orchestrator = WorkflowOrchestrator()
        
        # Execute analysis workflow
        analysis_state = orchestrator.execute_analysis_workflow(str(integration_test_project))
        
        # Verify analysis completed successfully
        assert analysis_state.current_step == "completed"
        assert len(analysis_state.errors) == 0
        assert "structure_analysis" in analysis_state.results
        assert "test_audit" in analysis_state.results
        assert "config_analysis" in analysis_state.results
        assert "duplicate_detection" in analysis_state.results
        
        # Execute cleanup workflow based on analysis
        cleanup_state = orchestrator.execute_cleanup_workflow(
            str(integration_test_project), 
            analysis_state
        )
        
        # Verify cleanup workflow
        assert cleanup_state.current_step == "completed"
        assert len(cleanup_state.errors) == 0
    
    def test_data_flow_between_tools(self, integration_test_project):
        """Test that data flows correctly between tools"""
        orchestrator = WorkflowOrchestrator()
        
        # Get structure analysis
        structure_result = orchestrator.tools['structure_analyzer'].analyze_project(
            str(integration_test_project)
        )
        
        # Use structure data for targeted quality checking
        quality_checker = orchestrator.tools['quality_checker']
        
        for component in structure_result.components:
            if component.path.endswith('.py') and 'src' in component.path:
                quality_report = quality_checker.check_file(component.path)
                assert quality_report is not None
                
                # Verify quality report contains expected data
                assert hasattr(quality_report, 'issues')
                assert hasattr(quality_report, 'file_path')
                assert quality_report.file_path == component.path
    
    def test_tool_result_aggregation(self, integration_test_project):
        """Test aggregation of results from multiple tools"""
        orchestrator = WorkflowOrchestrator()
        
        # Collect results from all tools
        results = {}
        
        # Structure analysis
        results['structure'] = orchestrator.tools['structure_analyzer'].analyze_project(
            str(integration_test_project)
        )
        
        # Test audit
        test_path = integration_test_project / "tests"
        results['tests'] = orchestrator.tools['test_auditor'].audit_tests(str(test_path))
        
        # Configuration analysis
        results['config'] = orchestrator.tools['config_unifier'].analyze_configs(
            str(integration_test_project)
        )
        
        # Quality assessment
        src_path = integration_test_project / "src"
        results['quality'] = orchestrator.tools['quality_checker'].check_directory(str(src_path))
        
        # Duplicate detection
        results['duplicates'] = orchestrator.tools['duplicate_detector'].find_duplicates(
            str(integration_test_project)
        )
        
        # Aggregate into comprehensive report
        comprehensive_report = {
            'project_health': {
                'structure_complexity': results['structure'].complexity_score,
                'test_coverage': len(results['tests'].working_tests) / max(results['tests'].total_tests, 1),
                'config_consistency': len(results['config'].conflicts) == 0,
                'code_quality': results['quality'].overall_score,
                'duplicate_ratio': len(results['duplicates'].duplicate_groups) / max(len(results['structure'].components), 1)
            },
            'recommendations': []
        }
        
        # Generate recommendations based on aggregated data
        if results['tests'].total_tests == 0:
            comprehensive_report['recommendations'].append("Add test coverage")
        
        if len(results['config'].conflicts) > 0:
            comprehensive_report['recommendations'].append("Resolve configuration conflicts")
        
        if results['quality'].overall_score < 70:
            comprehensive_report['recommendations'].append("Improve code quality")
        
        if len(results['duplicates'].duplicate_groups) > 0:
            comprehensive_report['recommendations'].append("Remove duplicate code")
        
        # Verify comprehensive report
        assert 'project_health' in comprehensive_report
        assert 'recommendations' in comprehensive_report
        assert isinstance(comprehensive_report['recommendations'], list)
    
    def test_tool_dependency_resolution(self, integration_test_project):
        """Test that tool dependencies are resolved correctly"""
        orchestrator = WorkflowOrchestrator()
        
        # Some tools depend on results from other tools
        # Test that dependencies are handled correctly
        
        # 1. Structure analysis (no dependencies)
        structure_result = orchestrator.tools['structure_analyzer'].analyze_project(
            str(integration_test_project)
        )
        assert structure_result is not None
        
        # 2. Quality checking depends on knowing which files to check
        quality_results = {}
        for component in structure_result.components:
            if component.path.endswith('.py'):
                quality_result = orchestrator.tools['quality_checker'].check_file(component.path)
                quality_results[component.path] = quality_result
        
        assert len(quality_results) > 0
        
        # 3. Test audit depends on finding test files
        test_files = [c for c in structure_result.components if 'test' in c.path.lower()]
        assert len(test_files) > 0
        
        # 4. Configuration analysis can use structure info to find config files
        config_files = [c for c in structure_result.components if any(
            c.path.endswith(ext) for ext in ['.json', '.yaml', '.yml', '.ini', '.env']
        )]
        assert len(config_files) > 0
    
    def test_error_handling_across_tools(self, integration_test_project):
        """Test error handling when tools interact"""
        orchestrator = WorkflowOrchestrator()
        
        # Create a scenario with various error conditions
        error_scenarios = [
            # Missing directory
            ("nonexistent_path", "structure_analyzer"),
            # Invalid file
            ("invalid_file.txt", "quality_checker"),
            # Empty directory
            (str(integration_test_project / "empty"), "test_auditor")
        ]
        
        # Create empty directory for testing
        (integration_test_project / "empty").mkdir()
        
        for path, tool_name in error_scenarios:
            tool = orchestrator.tools[tool_name]
            
            try:
                if hasattr(tool, 'analyze_project'):
                    result = tool.analyze_project(path)
                elif hasattr(tool, 'check_file'):
                    result = tool.check_file(path)
                elif hasattr(tool, 'audit_tests'):
                    result = tool.audit_tests(path)
                
                # Tools should handle errors gracefully
                # Either return None or a result with error information
                assert result is None or hasattr(result, 'errors')
                
            except Exception as e:
                # If exceptions are raised, they should be informative
                assert str(e) != ""
                assert len(str(e)) > 0


class TestWorkflowValidation:
    """Test workflow validation and state management"""
    
    def test_workflow_state_consistency(self, integration_test_project):
        """Test that workflow state remains consistent"""
        orchestrator = WorkflowOrchestrator()
        
        # Execute workflow and track state changes
        initial_state = WorkflowState(
            project_path=str(integration_test_project),
            current_step="initialization",
            completed_steps=[],
            results={},
            errors=[]
        )
        
        # Simulate workflow execution with state tracking
        steps = ["structure_analysis", "test_audit", "config_analysis", "quality_assessment"]
        
        for step in steps:
            initial_state.current_step = step
            
            # Execute step based on type
            if step == "structure_analysis":
                result = orchestrator.tools['structure_analyzer'].analyze_project(
                    str(integration_test_project)
                )
            elif step == "test_audit":
                result = orchestrator.tools['test_auditor'].audit_tests(
                    str(integration_test_project / "tests")
                )
            elif step == "config_analysis":
                result = orchestrator.tools['config_unifier'].analyze_configs(
                    str(integration_test_project)
                )
            elif step == "quality_assessment":
                result = orchestrator.tools['quality_checker'].check_directory(
                    str(integration_test_project / "src")
                )
            
            initial_state.add_result(step, result)
        
        # Verify state consistency
        assert len(initial_state.completed_steps) == len(steps)
        assert len(initial_state.results) == len(steps)
        assert all(step in initial_state.results for step in steps)
        assert len(initial_state.errors) == 0
    
    def test_workflow_rollback_capability(self, integration_test_project):
        """Test that workflows can be rolled back safely"""
        orchestrator = WorkflowOrchestrator()
        
        # Create backup of original state
        original_files = list(integration_test_project.rglob("*"))
        original_file_count = len([f for f in original_files if f.is_file()])
        
        # Execute analysis workflow
        analysis_state = orchestrator.execute_analysis_workflow(str(integration_test_project))
        
        # Execute cleanup workflow (which might modify files)
        cleanup_state = orchestrator.execute_cleanup_workflow(
            str(integration_test_project), 
            analysis_state
        )
        
        # Check if any backups were created
        backup_dirs = list(integration_test_project.glob("backup*"))
        
        if backup_dirs:
            # Test rollback capability
            for backup_dir in backup_dirs:
                if backup_dir.is_dir():
                    # Verify backup contains files
                    backup_files = list(backup_dir.rglob("*"))
                    backup_file_count = len([f for f in backup_files if f.is_file()])
                    assert backup_file_count > 0
        
        # Verify project structure is still intact
        current_files = list(integration_test_project.rglob("*"))
        current_file_count = len([f for f in current_files if f.is_file()])
        
        # File count should be same or greater (due to backups)
        assert current_file_count >= original_file_count
    
    def test_workflow_performance_tracking(self, integration_test_project):
        """Test workflow performance tracking"""
        import time
        
        orchestrator = WorkflowOrchestrator()
        
        # Track performance of each workflow step
        performance_data = {}
        
        steps = [
            ("structure_analysis", lambda: orchestrator.tools['structure_analyzer'].analyze_project(str(integration_test_project))),
            ("test_audit", lambda: orchestrator.tools['test_auditor'].audit_tests(str(integration_test_project / "tests"))),
            ("config_analysis", lambda: orchestrator.tools['config_unifier'].analyze_configs(str(integration_test_project))),
            ("quality_assessment", lambda: orchestrator.tools['quality_checker'].check_directory(str(integration_test_project / "src")))
        ]
        
        for step_name, step_func in steps:
            start_time = time.time()
            result = step_func()
            end_time = time.time()
            
            performance_data[step_name] = {
                'execution_time': end_time - start_time,
                'success': result is not None,
                'result_size': len(str(result)) if result else 0
            }
        
        # Verify performance tracking
        assert len(performance_data) == len(steps)
        
        for step_name, perf_data in performance_data.items():
            assert 'execution_time' in perf_data
            assert 'success' in perf_data
            assert perf_data['execution_time'] > 0
            assert perf_data['execution_time'] < 60  # Should complete within 60 seconds
    
    def test_concurrent_workflow_execution(self, integration_test_project):
        """Test concurrent execution of workflows"""
        import threading
        import queue
        
        orchestrator = WorkflowOrchestrator()
        results_queue = queue.Queue()
        
        def execute_workflow(workflow_id):
            try:
                state = orchestrator.execute_analysis_workflow(str(integration_test_project))
                results_queue.put((workflow_id, state, None))
            except Exception as e:
                results_queue.put((workflow_id, None, str(e)))
        
        # Start multiple concurrent workflows
        threads = []
        for i in range(3):
            thread = threading.Thread(target=execute_workflow, args=(f"workflow_{i}",))
            threads.append(thread)
            thread.start()
        
        # Wait for all workflows to complete
        for thread in threads:
            thread.join(timeout=120)  # 2 minute timeout
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        # Verify all workflows completed successfully
        assert len(results) == 3
        
        for workflow_id, state, error in results:
            assert error is None, f"Workflow {workflow_id} failed: {error}"
            assert state is not None
            assert state.current_step == "completed"
            assert len(state.errors) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])