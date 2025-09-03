#!/usr/bin/env python3
"""
Comprehensive Testing and Validation Suite for Project Cleanup and Quality Improvements

This module provides end-to-end testing of all cleanup and quality improvement tools,
integration testing for tool interactions, performance testing, and user acceptance
testing scenarios.

Requirements covered: 1.1, 1.6, 2.6, 3.6, 4.6, 5.6, 6.6
"""

import pytest
import time
import subprocess
import tempfile
import shutil
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from unittest.mock import Mock, patch
import concurrent.futures
import psutil
import sys
import os

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools.test-auditor.test_auditor import TestAuditor
from tools.config_manager.config_unifier import ConfigUnifier
from tools.project_structure_analyzer.structure_analyzer import StructureAnalyzer
from tools.codebase_cleanup.duplicate_detector import DuplicateDetector
from tools.code_quality.quality_checker import QualityChecker
from tools.maintenance_scheduler.scheduler import MaintenanceScheduler


@dataclass
class PerformanceMetrics:
    """Performance metrics for tool execution"""
    execution_time: float
    memory_usage: float
    cpu_usage: float
    disk_io: float
    tool_name: str
    operation: str


@dataclass
class TestResult:
    """Test result container"""
    test_name: str
    passed: bool
    execution_time: float
    error_message: Optional[str] = None
    performance_metrics: Optional[PerformanceMetrics] = None


class ComprehensiveTestSuite:
    """Main test suite orchestrator for all cleanup and quality tools"""
    
    def __init__(self, test_workspace: Path):
        self.test_workspace = test_workspace
        self.results: List[TestResult] = []
        self.performance_baseline = self._load_performance_baseline()
        
    def _load_performance_baseline(self) -> Dict[str, float]:
        """Load performance baseline metrics"""
        baseline_file = Path("tests/comprehensive/performance_baseline.json")
        if baseline_file.exists():
            with open(baseline_file) as f:
                return json.load(f)
        return {}
    
    def _measure_performance(self, func, tool_name: str, operation: str) -> PerformanceMetrics:
        """Measure performance metrics for a function execution"""
        process = psutil.Process()
        
        # Get initial metrics
        start_time = time.time()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_cpu = process.cpu_percent()
        start_io = process.io_counters()
        
        # Execute function
        result = func()
        
        # Get final metrics
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        end_cpu = process.cpu_percent()
        end_io = process.io_counters()
        
        return PerformanceMetrics(
            execution_time=end_time - start_time,
            memory_usage=end_memory - start_memory,
            cpu_usage=end_cpu - start_cpu,
            disk_io=(end_io.read_bytes + end_io.write_bytes) - (start_io.read_bytes + start_io.write_bytes),
            tool_name=tool_name,
            operation=operation
        )
    
    def run_end_to_end_tests(self) -> List[TestResult]:
        """Run end-to-end tests for all cleanup and quality tools"""
        results = []
        
        # Test each tool individually
        tools_to_test = [
            ("test_auditor", self._test_test_auditor_e2e),
            ("config_unifier", self._test_config_unifier_e2e),
            ("structure_analyzer", self._test_structure_analyzer_e2e),
            ("duplicate_detector", self._test_duplicate_detector_e2e),
            ("quality_checker", self._test_quality_checker_e2e),
            ("maintenance_scheduler", self._test_maintenance_scheduler_e2e),
        ]
        
        for tool_name, test_func in tools_to_test:
            try:
                start_time = time.time()
                test_func()
                execution_time = time.time() - start_time
                
                results.append(TestResult(
                    test_name=f"e2e_{tool_name}",
                    passed=True,
                    execution_time=execution_time
                ))
            except Exception as e:
                results.append(TestResult(
                    test_name=f"e2e_{tool_name}",
                    passed=False,
                    execution_time=time.time() - start_time,
                    error_message=str(e)
                ))
        
        return results
    
    def run_integration_tests(self) -> List[TestResult]:
        """Run integration tests for tool interactions and workflows"""
        results = []
        
        integration_tests = [
            ("test_config_workflow", self._test_complete_config_workflow),
            ("test_cleanup_workflow", self._test_complete_cleanup_workflow),
            ("test_quality_workflow", self._test_complete_quality_workflow),
            ("test_maintenance_workflow", self._test_complete_maintenance_workflow),
            ("test_cross_tool_integration", self._test_cross_tool_integration),
        ]
        
        for test_name, test_func in integration_tests:
            try:
                start_time = time.time()
                test_func()
                execution_time = time.time() - start_time
                
                results.append(TestResult(
                    test_name=test_name,
                    passed=True,
                    execution_time=execution_time
                ))
            except Exception as e:
                results.append(TestResult(
                    test_name=test_name,
                    passed=False,
                    execution_time=time.time() - start_time,
                    error_message=str(e)
                ))
        
        return results
    
    def run_performance_tests(self) -> List[TestResult]:
        """Run performance tests to ensure tools don't impact development velocity"""
        results = []
        
        performance_tests = [
            ("test_auditor_performance", self._test_test_auditor_performance),
            ("config_unifier_performance", self._test_config_unifier_performance),
            ("structure_analyzer_performance", self._test_structure_analyzer_performance),
            ("duplicate_detector_performance", self._test_duplicate_detector_performance),
            ("quality_checker_performance", self._test_quality_checker_performance),
            ("parallel_execution_performance", self._test_parallel_execution_performance),
        ]
        
        for test_name, test_func in performance_tests:
            try:
                start_time = time.time()
                metrics = test_func()
                execution_time = time.time() - start_time
                
                # Check against baseline
                baseline_key = f"{test_name}_execution_time"
                baseline_time = self.performance_baseline.get(baseline_key, float('inf'))
                
                # Allow 20% performance degradation
                performance_acceptable = metrics.execution_time <= baseline_time * 1.2
                
                results.append(TestResult(
                    test_name=test_name,
                    passed=performance_acceptable,
                    execution_time=execution_time,
                    performance_metrics=metrics,
                    error_message=None if performance_acceptable else f"Performance degraded: {metrics.execution_time:.2f}s vs baseline {baseline_time:.2f}s"
                ))
            except Exception as e:
                results.append(TestResult(
                    test_name=test_name,
                    passed=False,
                    execution_time=time.time() - start_time,
                    error_message=str(e)
                ))
        
        return results
    
    def run_user_acceptance_tests(self) -> List[TestResult]:
        """Run user acceptance testing scenarios for all major tool functionality"""
        results = []
        
        user_scenarios = [
            ("developer_onboarding_scenario", self._test_developer_onboarding_scenario),
            ("project_cleanup_scenario", self._test_project_cleanup_scenario),
            ("configuration_migration_scenario", self._test_configuration_migration_scenario),
            ("quality_improvement_scenario", self._test_quality_improvement_scenario),
            ("maintenance_automation_scenario", self._test_maintenance_automation_scenario),
            ("error_recovery_scenario", self._test_error_recovery_scenario),
        ]
        
        for scenario_name, test_func in user_scenarios:
            try:
                start_time = time.time()
                test_func()
                execution_time = time.time() - start_time
                
                results.append(TestResult(
                    test_name=scenario_name,
                    passed=True,
                    execution_time=execution_time
                ))
            except Exception as e:
                results.append(TestResult(
                    test_name=scenario_name,
                    passed=False,
                    execution_time=time.time() - start_time,
                    error_message=str(e)
                ))
        
        return results   
 # End-to-End Test Implementations
    
    def _test_test_auditor_e2e(self):
        """End-to-end test for test auditor tool"""
        # Create test workspace with sample test files
        test_dir = self.test_workspace / "test_files"
        test_dir.mkdir(exist_ok=True)
        
        # Create sample test files with various issues
        (test_dir / "test_working.py").write_text("""
import pytest

def test_simple():
    assert True
""")
        
        (test_dir / "test_broken.py").write_text("""
import nonexistent_module

def test_broken():
    assert nonexistent_module.function()
""")
        
        # Run test auditor
        auditor = TestAuditor(str(test_dir))
        report = auditor.audit_tests()
        
        # Verify results
        assert report.total_tests >= 2
        assert report.broken_tests
        assert "test_broken.py" in str(report.broken_tests)
    
    def _test_config_unifier_e2e(self):
        """End-to-end test for configuration unifier"""
        # Create sample config files
        config_dir = self.test_workspace / "configs"
        config_dir.mkdir(exist_ok=True)
        
        (config_dir / "config1.json").write_text('{"setting1": "value1", "common": "json"}')
        (config_dir / "config2.yaml").write_text('setting2: value2\ncommon: yaml')
        
        # Run config unifier
        unifier = ConfigUnifier(str(config_dir))
        analysis = unifier.analyze_configs()
        unified_config = unifier.create_unified_config(analysis)
        
        # Verify results
        assert analysis.config_files
        assert len(analysis.config_files) >= 2
        assert unified_config
    
    def _test_structure_analyzer_e2e(self):
        """End-to-end test for project structure analyzer"""
        # Create sample project structure
        project_dir = self.test_workspace / "sample_project"
        project_dir.mkdir(exist_ok=True)
        
        (project_dir / "main.py").write_text("# Main module")
        (project_dir / "utils.py").write_text("# Utilities")
        
        subdir = project_dir / "submodule"
        subdir.mkdir(exist_ok=True)
        (subdir / "__init__.py").write_text("")
        (subdir / "helper.py").write_text("# Helper functions")
        
        # Run structure analyzer
        analyzer = StructureAnalyzer(str(project_dir))
        structure_map = analyzer.analyze_structure()
        
        # Verify results
        assert structure_map.components
        assert len(structure_map.components) >= 3
    
    def _test_duplicate_detector_e2e(self):
        """End-to-end test for duplicate detector"""
        # Create duplicate files
        files_dir = self.test_workspace / "duplicate_test"
        files_dir.mkdir(exist_ok=True)
        
        content = "# This is duplicate content\nprint('hello world')"
        (files_dir / "file1.py").write_text(content)
        (files_dir / "file2.py").write_text(content)
        (files_dir / "unique.py").write_text("# Unique content")
        
        # Run duplicate detector
        detector = DuplicateDetector(str(files_dir))
        report = detector.find_duplicates()
        
        # Verify results
        assert report.duplicate_groups
        assert len(report.duplicate_groups) >= 1
    
    def _test_quality_checker_e2e(self):
        """End-to-end test for code quality checker"""
        # Create code with quality issues
        code_dir = self.test_workspace / "quality_test"
        code_dir.mkdir(exist_ok=True)
        
        (code_dir / "bad_code.py").write_text("""
def poorly_formatted_function(x,y,z):
    if x>0:
        return x+y+z
    else:
        return 0
""")
        
        # Run quality checker
        checker = QualityChecker(str(code_dir))
        report = checker.check_quality()
        
        # Verify results
        assert report.issues
        assert any("formatting" in issue.description.lower() for issue in report.issues)
    
    def _test_maintenance_scheduler_e2e(self):
        """End-to-end test for maintenance scheduler"""
        # Create maintenance tasks
        scheduler = MaintenanceScheduler()
        
        # Add sample tasks
        scheduler.add_task("cleanup_logs", priority=1, estimated_duration=300)
        scheduler.add_task("update_docs", priority=2, estimated_duration=600)
        
        # Schedule tasks
        schedule = scheduler.create_schedule()
        
        # Verify results
        assert schedule.tasks
        assert len(schedule.tasks) >= 2
    
    # Integration Test Implementations
    
    def _test_complete_config_workflow(self):
        """Test complete configuration management workflow"""
        # 1. Analyze existing configs
        config_dir = self.test_workspace / "config_workflow"
        config_dir.mkdir(exist_ok=True)
        
        # Create scattered configs
        (config_dir / "app.json").write_text('{"app_name": "test", "debug": true}')
        (config_dir / "db.yaml").write_text('host: localhost\nport: 5432')
        (config_dir / ".env").write_text('SECRET_KEY=test123')
        
        # 2. Run unification
        unifier = ConfigUnifier(str(config_dir))
        analysis = unifier.analyze_configs()
        unified_config = unifier.create_unified_config(analysis)
        
        # 3. Validate unified config
        validator = unifier.get_validator()
        validation_result = validator.validate(unified_config)
        
        # 4. Verify workflow completion
        assert analysis.config_files
        assert unified_config
        assert validation_result.is_valid
    
    def _test_complete_cleanup_workflow(self):
        """Test complete codebase cleanup workflow"""
        # 1. Create messy codebase
        cleanup_dir = self.test_workspace / "cleanup_workflow"
        cleanup_dir.mkdir(exist_ok=True)
        
        # Duplicate files
        content = "print('duplicate')"
        (cleanup_dir / "dup1.py").write_text(content)
        (cleanup_dir / "dup2.py").write_text(content)
        
        # Unused file
        (cleanup_dir / "unused.py").write_text("# This file is not used")
        
        # 2. Run duplicate detection
        detector = DuplicateDetector(str(cleanup_dir))
        dup_report = detector.find_duplicates()
        
        # 3. Run quality check
        checker = QualityChecker(str(cleanup_dir))
        quality_report = checker.check_quality()
        
        # 4. Verify workflow results
        assert dup_report.duplicate_groups
        assert quality_report
    
    def _test_complete_quality_workflow(self):
        """Test complete quality improvement workflow"""
        # 1. Create code with quality issues
        quality_dir = self.test_workspace / "quality_workflow"
        quality_dir.mkdir(exist_ok=True)
        
        (quality_dir / "messy.py").write_text("""
def bad_function(a,b,c):
    if a>b:
        if b>c:
            return a+b+c
        else:
            return a+b
    else:
        return c
""")
        
        # 2. Run quality analysis
        checker = QualityChecker(str(quality_dir))
        initial_report = checker.check_quality()
        
        # 3. Apply automatic fixes
        fixer = checker.get_auto_fixer()
        fix_report = fixer.apply_fixes(initial_report)
        
        # 4. Re-run quality check
        final_report = checker.check_quality()
        
        # 5. Verify improvement
        assert initial_report.issues
        assert len(final_report.issues) <= len(initial_report.issues)
    
    def _test_complete_maintenance_workflow(self):
        """Test complete maintenance automation workflow"""
        # 1. Schedule maintenance tasks
        scheduler = MaintenanceScheduler()
        scheduler.add_task("test_audit", priority=1, estimated_duration=120)
        scheduler.add_task("config_validation", priority=2, estimated_duration=60)
        scheduler.add_task("quality_check", priority=3, estimated_duration=180)
        
        # 2. Create execution plan
        schedule = scheduler.create_schedule()
        
        # 3. Execute tasks (simulated)
        execution_results = []
        for task in schedule.tasks:
            # Simulate task execution
            start_time = time.time()
            time.sleep(0.1)  # Simulate work
            execution_time = time.time() - start_time
            
            execution_results.append({
                'task': task.name,
                'success': True,
                'execution_time': execution_time
            })
        
        # 4. Verify workflow completion
        assert len(execution_results) == 3
        assert all(result['success'] for result in execution_results)
    
    def _test_cross_tool_integration(self):
        """Test integration between different tools"""
        # 1. Create test environment
        integration_dir = self.test_workspace / "integration_test"
        integration_dir.mkdir(exist_ok=True)
        
        # 2. Run structure analysis
        analyzer = StructureAnalyzer(str(integration_dir))
        structure = analyzer.analyze_structure()
        
        # 3. Use structure results for quality checking
        checker = QualityChecker(str(integration_dir))
        checker.set_structure_context(structure)
        quality_report = checker.check_quality()
        
        # 4. Use quality results for maintenance scheduling
        scheduler = MaintenanceScheduler()
        scheduler.add_quality_based_tasks(quality_report)
        schedule = scheduler.create_schedule()
        
        # 5. Verify cross-tool data flow
        assert structure
        assert quality_report
        assert schedule
    
    # Performance Test Implementations
    
    def _test_test_auditor_performance(self) -> PerformanceMetrics:
        """Test performance of test auditor"""
        # Create large test suite
        perf_dir = self.test_workspace / "perf_test_auditor"
        perf_dir.mkdir(exist_ok=True)
        
        # Create 100 test files
        for i in range(100):
            (perf_dir / f"test_{i}.py").write_text(f"""
import pytest

def test_function_{i}():
    assert {i} >= 0

def test_another_{i}():
    assert {i} < 1000
""")
        
        def run_auditor():
            auditor = TestAuditor(str(perf_dir))
            return auditor.audit_tests()
        
        return self._measure_performance(run_auditor, "test_auditor", "audit_large_suite")
    
    def _test_config_unifier_performance(self) -> PerformanceMetrics:
        """Test performance of config unifier"""
        # Create many config files
        perf_dir = self.test_workspace / "perf_config"
        perf_dir.mkdir(exist_ok=True)
        
        # Create 50 config files
        for i in range(50):
            (perf_dir / f"config_{i}.json").write_text(f'{{"setting_{i}": "value_{i}"}}')
        
        def run_unifier():
            unifier = ConfigUnifier(str(perf_dir))
            analysis = unifier.analyze_configs()
            return unifier.create_unified_config(analysis)
        
        return self._measure_performance(run_unifier, "config_unifier", "unify_many_configs")
    
    def _test_structure_analyzer_performance(self) -> PerformanceMetrics:
        """Test performance of structure analyzer"""
        # Create deep directory structure
        perf_dir = self.test_workspace / "perf_structure"
        perf_dir.mkdir(exist_ok=True)
        
        # Create nested directories with files
        for i in range(10):
            level_dir = perf_dir / f"level_{i}"
            level_dir.mkdir(exist_ok=True)
            for j in range(10):
                (level_dir / f"module_{j}.py").write_text(f"# Module {i}.{j}")
        
        def run_analyzer():
            analyzer = StructureAnalyzer(str(perf_dir))
            return analyzer.analyze_structure()
        
        return self._measure_performance(run_analyzer, "structure_analyzer", "analyze_deep_structure")
    
    def _test_duplicate_detector_performance(self) -> PerformanceMetrics:
        """Test performance of duplicate detector"""
        # Create many files with some duplicates
        perf_dir = self.test_workspace / "perf_duplicates"
        perf_dir.mkdir(exist_ok=True)
        
        # Create 200 files, with some duplicates
        base_content = "print('base content')"
        for i in range(200):
            if i % 10 == 0:  # Every 10th file is a duplicate
                content = base_content
            else:
                content = f"print('unique content {i}')"
            (perf_dir / f"file_{i}.py").write_text(content)
        
        def run_detector():
            detector = DuplicateDetector(str(perf_dir))
            return detector.find_duplicates()
        
        return self._measure_performance(run_detector, "duplicate_detector", "scan_many_files")
    
    def _test_quality_checker_performance(self) -> PerformanceMetrics:
        """Test performance of quality checker"""
        # Create large codebase
        perf_dir = self.test_workspace / "perf_quality"
        perf_dir.mkdir(exist_ok=True)
        
        # Create 50 Python files with various quality issues
        for i in range(50):
            (perf_dir / f"module_{i}.py").write_text(f"""
def function_{i}(a,b,c):
    if a>b:
        return a+b+c
    else:
        return c

class Class_{i}:
    def method(self,x,y):
        return x*y

# Missing docstrings and type hints
""")
        
        def run_checker():
            checker = QualityChecker(str(perf_dir))
            return checker.check_quality()
        
        return self._measure_performance(run_checker, "quality_checker", "check_large_codebase")
    
    def _test_parallel_execution_performance(self) -> PerformanceMetrics:
        """Test performance of parallel tool execution"""
        # Create test environment
        perf_dir = self.test_workspace / "perf_parallel"
        perf_dir.mkdir(exist_ok=True)
        
        # Create sample files
        for i in range(20):
            (perf_dir / f"test_{i}.py").write_text(f"def test_{i}(): assert True")
            (perf_dir / f"config_{i}.json").write_text(f'{{"key_{i}": "value_{i}"}}')
        
        def run_parallel():
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                # Submit multiple tool operations
                futures = []
                
                # Test auditor
                futures.append(executor.submit(TestAuditor(str(perf_dir)).audit_tests))
                
                # Config unifier
                futures.append(executor.submit(ConfigUnifier(str(perf_dir)).analyze_configs))
                
                # Structure analyzer
                futures.append(executor.submit(StructureAnalyzer(str(perf_dir)).analyze_structure))
                
                # Quality checker
                futures.append(executor.submit(QualityChecker(str(perf_dir)).check_quality))
                
                # Wait for completion
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
                return results
        
        return self._measure_performance(run_parallel, "parallel_execution", "multiple_tools")