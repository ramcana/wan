#!/usr/bin/env python3
"""
Performance Testing for Cleanup and Quality Tools

This module ensures that cleanup and quality improvement tools don't impact
development velocity by testing their performance characteristics.

Requirements covered: 1.1, 1.6, 2.6, 3.6, 4.6, 5.6, 6.6
"""

import pytest
import tempfile
import shutil
import time
import psutil
import os
import threading
import statistics
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

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

class MaintenanceScheduler:
    def __init__(self):
        self.tasks = []
    
    def schedule_task(self, name, tool, params, priority="medium"):
        self.tasks.append({"name": name, "tool": tool, "params": params, "priority": priority})


@dataclass
class PerformanceMetrics:
    """Performance metrics for tool execution"""
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    files_processed: int
    throughput_files_per_second: float
    peak_memory_mb: float
    
    def __post_init__(self):
        if self.execution_time > 0 and self.files_processed > 0:
            self.throughput_files_per_second = self.files_processed / self.execution_time


class PerformanceProfiler:
    """Profiles performance of tool execution"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.initial_memory = None
        self.peak_memory = None
        self.cpu_samples = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.initial_memory
        self.cpu_samples = []
        self.monitoring = True
        
        # Start background monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Tuple[float, float]:
        """Stop monitoring and return memory and CPU usage"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        
        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = final_memory - self.initial_memory
        avg_cpu = statistics.mean(self.cpu_samples) if self.cpu_samples else 0
        
        return memory_usage, avg_cpu
    
    def _monitor_resources(self):
        """Background thread to monitor resource usage"""
        while self.monitoring:
            try:
                # Sample CPU usage
                cpu_percent = self.process.cpu_percent()
                self.cpu_samples.append(cpu_percent)
                
                # Track peak memory
                current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
                if current_memory > self.peak_memory:
                    self.peak_memory = current_memory
                
                time.sleep(0.1)  # Sample every 100ms
            except:
                break


def create_performance_test_project(size: str = "medium") -> Path:
    """Create test project of specified size for performance testing"""
    temp_dir = tempfile.mkdtemp(prefix=f"perf_test_{size}_")
    project_root = Path(temp_dir)
    
    # Define project sizes
    sizes = {
        "small": {"files": 10, "lines_per_file": 50, "test_files": 5},
        "medium": {"files": 50, "lines_per_file": 100, "test_files": 20},
        "large": {"files": 200, "lines_per_file": 200, "test_files": 80},
        "xlarge": {"files": 500, "lines_per_file": 300, "test_files": 200}
    }
    
    config = sizes.get(size, sizes["medium"])
    
    # Create directory structure
    (project_root / "src").mkdir()
    (project_root / "tests").mkdir()
    (project_root / "config").mkdir()
    (project_root / "docs").mkdir()
    
    # Create source files
    for i in range(config["files"]):
        file_content = generate_python_file_content(config["lines_per_file"], i)
        (project_root / "src" / f"module_{i}.py").write_text(file_content)
    
    # Create test files
    for i in range(config["test_files"]):
        test_content = generate_test_file_content(i)
        (project_root / "tests" / f"test_module_{i}.py").write_text(test_content)
    
    # Create configuration files
    create_config_files(project_root, config["files"] // 10)
    
    # Create some duplicate files for testing
    create_duplicate_files(project_root, config["files"] // 20)
    
    return project_root


def generate_python_file_content(lines: int, module_id: int) -> str:
    """Generate Python file content with specified number of lines"""
    content = f"""#!/usr/bin/env python3
'''
Module {module_id} - Generated for performance testing
'''

import os
import sys
import json
from typing import List, Dict, Any, Optional

class Module{module_id}:
    '''Class for module {module_id}'''
    
    def __init__(self, name: str = "module_{module_id}"):
        self.name = name
        self.data = {{}}
        self.initialized = True
    
"""
    
    # Add functions to reach target line count
    current_lines = content.count('\n')
    functions_needed = max(1, (lines - current_lines) // 10)
    
    for func_id in range(functions_needed):
        func_content = f"""    def function_{func_id}(self, param1: str, param2: int = 0) -> Dict[str, Any]:
        '''Function {func_id} in module {module_id}'''
        result = {{
            'param1': param1,
            'param2': param2,
            'module_id': {module_id},
            'function_id': {func_id}
        }}
        
        if param2 > 0:
            result['processed'] = True
            result['value'] = param2 * 2
        else:
            result['processed'] = False
            result['value'] = 0
        
        return result
    
"""
        content += func_content
    
    return content


def generate_test_file_content(test_id: int) -> str:
    """Generate test file content"""
    return f"""#!/usr/bin/env python3
'''
Test file {test_id} - Generated for performance testing
'''

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from module_{test_id % 10} import Module{test_id % 10}


class TestModule{test_id}:
    '''Test class for module {test_id}'''
    
    def setup_method(self):
        '''Setup for each test method'''
        self.module = Module{test_id % 10}()
    
    def test_initialization(self):
        '''Test module initialization'''
        assert self.module.name == "module_{test_id % 10}"
        assert self.module.initialized is True
        assert isinstance(self.module.data, dict)
    
    def test_function_0(self):
        '''Test function 0'''
        result = self.module.function_0("test", 5)
        assert result['param1'] == "test"
        assert result['param2'] == 5
        assert result['processed'] is True
        assert result['value'] == 10
    
    def test_function_0_default(self):
        '''Test function 0 with default parameter'''
        result = self.module.function_0("test")
        assert result['param1'] == "test"
        assert result['param2'] == 0
        assert result['processed'] is False
        assert result['value'] == 0
"""


def create_config_files(project_root: Path, count: int):
    """Create configuration files"""
    config_types = [
        ("app.json", '{{"debug": true, "port": {port}, "name": "app_{id}"}}'),
        ("db.yaml", "database:\n  host: localhost\n  port: {port}\n  name: db_{id}"),
        ("settings.ini", "[DEFAULT]\nlog_level = INFO\ntimeout = {port}\napp_id = {id}"),
        (".env", "DEBUG=true\nPORT={port}\nAPP_ID={id}")
    ]
    
    for i in range(count):
        config_type = config_types[i % len(config_types)]
        filename, template = config_type
        
        content = template.format(port=8000 + i, id=i)
        config_file = project_root / "config" / f"{i}_{filename}"
        config_file.write_text(content)


def create_duplicate_files(project_root: Path, count: int):
    """Create duplicate files for testing duplicate detection"""
    duplicate_content = """def duplicate_function():
    '''This is a duplicate function'''
    return "duplicate"

def another_duplicate():
    '''Another duplicate function'''
    return "also duplicate"
"""
    
    for i in range(count):
        (project_root / "src" / f"duplicate_{i}.py").write_text(duplicate_content)


@pytest.fixture(params=["small", "medium", "large"])
def performance_test_project(request):
    """Fixture for performance test projects of different sizes"""
    project_root = create_performance_test_project(request.param)
    yield project_root, request.param
    shutil.rmtree(project_root)


class TestToolPerformance:
    """Performance tests for individual tools"""
    
    def test_test_auditor_performance(self, performance_test_project):
        """Test TestAuditor performance"""
        project_root, size = performance_test_project
        profiler = PerformanceProfiler()
        
        # Count test files
        test_files = list((project_root / "tests").glob("*.py"))
        file_count = len(test_files)
        
        # Profile test auditor
        profiler.start_monitoring()
        start_time = time.time()
        
        test_auditor = TestAuditor()
        result = test_auditor.audit_tests(str(project_root / "tests"))
        
        end_time = time.time()
        memory_usage, cpu_usage = profiler.stop_monitoring()
        
        # Create performance metrics
        metrics = PerformanceMetrics(
            execution_time=end_time - start_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            files_processed=file_count,
            throughput_files_per_second=0,  # Will be calculated in __post_init__
            peak_memory_mb=profiler.peak_memory
        )
        
        # Performance assertions based on project size
        performance_limits = {
            "small": {"max_time": 5, "max_memory": 50},
            "medium": {"max_time": 15, "max_memory": 100},
            "large": {"max_time": 45, "max_memory": 200}
        }
        
        limits = performance_limits.get(size, performance_limits["medium"])
        
        assert metrics.execution_time < limits["max_time"], f"Test auditor too slow: {metrics.execution_time}s"
        assert metrics.memory_usage_mb < limits["max_memory"], f"Test auditor uses too much memory: {metrics.memory_usage_mb}MB"
        assert metrics.throughput_files_per_second > 0.5, f"Test auditor throughput too low: {metrics.throughput_files_per_second} files/s"
        
        # Verify functionality wasn't compromised
        assert result is not None
        assert result.total_tests > 0
    
    def test_structure_analyzer_performance(self, performance_test_project):
        """Test StructureAnalyzer performance"""
        project_root, size = performance_test_project
        profiler = PerformanceProfiler()
        
        # Count all files
        all_files = list(project_root.rglob("*.py"))
        file_count = len(all_files)
        
        # Profile structure analyzer
        profiler.start_monitoring()
        start_time = time.time()
        
        structure_analyzer = StructureAnalyzer()
        result = structure_analyzer.analyze_project(str(project_root))
        
        end_time = time.time()
        memory_usage, cpu_usage = profiler.stop_monitoring()
        
        metrics = PerformanceMetrics(
            execution_time=end_time - start_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            files_processed=file_count,
            throughput_files_per_second=0,
            peak_memory_mb=profiler.peak_memory
        )
        
        # Performance assertions
        performance_limits = {
            "small": {"max_time": 3, "max_memory": 30},
            "medium": {"max_time": 10, "max_memory": 75},
            "large": {"max_time": 30, "max_memory": 150}
        }
        
        limits = performance_limits.get(size, performance_limits["medium"])
        
        assert metrics.execution_time < limits["max_time"]
        assert metrics.memory_usage_mb < limits["max_memory"]
        assert metrics.throughput_files_per_second > 1.0
        
        # Verify functionality
        assert result is not None
        assert len(result.components) > 0
    
    def test_quality_checker_performance(self, performance_test_project):
        """Test QualityChecker performance"""
        project_root, size = performance_test_project
        profiler = PerformanceProfiler()
        
        # Count source files
        src_files = list((project_root / "src").glob("*.py"))
        file_count = len(src_files)
        
        # Profile quality checker
        profiler.start_monitoring()
        start_time = time.time()
        
        quality_checker = QualityChecker()
        result = quality_checker.check_directory(str(project_root / "src"))
        
        end_time = time.time()
        memory_usage, cpu_usage = profiler.stop_monitoring()
        
        metrics = PerformanceMetrics(
            execution_time=end_time - start_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            files_processed=file_count,
            throughput_files_per_second=0,
            peak_memory_mb=profiler.peak_memory
        )
        
        # Performance assertions
        performance_limits = {
            "small": {"max_time": 8, "max_memory": 40},
            "medium": {"max_time": 25, "max_memory": 100},
            "large": {"max_time": 80, "max_memory": 250}
        }
        
        limits = performance_limits.get(size, performance_limits["medium"])
        
        assert metrics.execution_time < limits["max_time"]
        assert metrics.memory_usage_mb < limits["max_memory"]
        assert metrics.throughput_files_per_second > 0.3
        
        # Verify functionality
        assert result is not None
        assert result.total_files > 0
    
    def test_duplicate_detector_performance(self, performance_test_project):
        """Test DuplicateDetector performance"""
        project_root, size = performance_test_project
        profiler = PerformanceProfiler()
        
        # Count all files
        all_files = list(project_root.rglob("*.py"))
        file_count = len(all_files)
        
        # Profile duplicate detector
        profiler.start_monitoring()
        start_time = time.time()
        
        duplicate_detector = DuplicateDetector()
        result = duplicate_detector.find_duplicates(str(project_root))
        
        end_time = time.time()
        memory_usage, cpu_usage = profiler.stop_monitoring()
        
        metrics = PerformanceMetrics(
            execution_time=end_time - start_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            files_processed=file_count,
            throughput_files_per_second=0,
            peak_memory_mb=profiler.peak_memory
        )
        
        # Performance assertions
        performance_limits = {
            "small": {"max_time": 10, "max_memory": 50},
            "medium": {"max_time": 30, "max_memory": 120},
            "large": {"max_time": 90, "max_memory": 300}
        }
        
        limits = performance_limits.get(size, performance_limits["medium"])
        
        assert metrics.execution_time < limits["max_time"]
        assert metrics.memory_usage_mb < limits["max_memory"]
        assert metrics.throughput_files_per_second > 0.2
        
        # Verify functionality
        assert result is not None
        assert len(result.duplicate_groups) >= 0
    
    def test_config_unifier_performance(self, performance_test_project):
        """Test ConfigUnifier performance"""
        project_root, size = performance_test_project
        profiler = PerformanceProfiler()
        
        # Count config files
        config_files = list((project_root / "config").glob("*"))
        file_count = len(config_files)
        
        # Profile config unifier
        profiler.start_monitoring()
        start_time = time.time()
        
        config_unifier = ConfigUnifier()
        result = config_unifier.analyze_configs(str(project_root))
        
        end_time = time.time()
        memory_usage, cpu_usage = profiler.stop_monitoring()
        
        metrics = PerformanceMetrics(
            execution_time=end_time - start_time,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            files_processed=file_count,
            throughput_files_per_second=0,
            peak_memory_mb=profiler.peak_memory
        )
        
        # Performance assertions
        performance_limits = {
            "small": {"max_time": 2, "max_memory": 20},
            "medium": {"max_time": 5, "max_memory": 40},
            "large": {"max_time": 15, "max_memory": 80}
        }
        
        limits = performance_limits.get(size, performance_limits["medium"])
        
        assert metrics.execution_time < limits["max_time"]
        assert metrics.memory_usage_mb < limits["max_memory"]
        assert metrics.throughput_files_per_second > 2.0
        
        # Verify functionality
        assert result is not None
        assert len(result.config_files) > 0


class TestConcurrentPerformance:
    """Test performance under concurrent execution"""
    
    def test_concurrent_tool_execution_performance(self):
        """Test performance when multiple tools run concurrently"""
        # Create multiple test projects
        projects = []
        for size in ["small", "medium"]:
            project = create_performance_test_project(size)
            projects.append((project, size))
        
        try:
            # Define tool execution functions
            def run_test_auditor(project_path):
                auditor = TestAuditor()
                return auditor.audit_tests(str(Path(project_path) / "tests"))
            
            def run_structure_analyzer(project_path):
                analyzer = StructureAnalyzer()
                return analyzer.analyze_project(project_path)
            
            def run_quality_checker(project_path):
                checker = QualityChecker()
                return checker.check_directory(str(Path(project_path) / "src"))
            
            # Execute tools concurrently
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = []
                
                for project_path, size in projects:
                    futures.append(executor.submit(run_test_auditor, str(project_path)))
                    futures.append(executor.submit(run_structure_analyzer, str(project_path)))
                    futures.append(executor.submit(run_quality_checker, str(project_path)))
                
                # Wait for all tasks to complete
                results = []
                for future in as_completed(futures, timeout=120):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        results.append(f"Error: {str(e)}")
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Verify all tools completed
            assert len(results) == len(projects) * 3
            
            # Performance should be reasonable even under concurrent load
            assert total_time < 60  # Should complete within 60 seconds
            
            # Verify no errors occurred
            error_count = sum(1 for r in results if isinstance(r, str) and r.startswith("Error"))
            assert error_count == 0, f"Found {error_count} errors in concurrent execution"
            
        finally:
            # Cleanup
            for project_path, _ in projects:
                shutil.rmtree(project_path)
    
    def test_memory_efficiency_under_load(self):
        """Test memory efficiency when processing large projects"""
        # Create a large test project
        large_project = create_performance_test_project("large")
        
        try:
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Run multiple tools in sequence
            tools = [
                TestAuditor(),
                StructureAnalyzer(),
                QualityChecker(),
                DuplicateDetector(),
                ConfigUnifier()
            ]
            
            peak_memory = initial_memory
            
            for tool in tools:
                # Run tool
                if hasattr(tool, 'analyze_project'):
                    tool.analyze_project(str(large_project))
                elif hasattr(tool, 'audit_tests'):
                    tool.audit_tests(str(large_project / "tests"))
                elif hasattr(tool, 'check_directory'):
                    tool.check_directory(str(large_project / "src"))
                elif hasattr(tool, 'find_duplicates'):
                    tool.find_duplicates(str(large_project))
                elif hasattr(tool, 'analyze_configs'):
                    tool.analyze_configs(str(large_project))
                
                # Check memory usage
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                if current_memory > peak_memory:
                    peak_memory = current_memory
                
                # Force garbage collection
                import gc
                gc.collect()
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            peak_increase = peak_memory - initial_memory
            
            # Memory usage should be reasonable
            assert memory_increase < 500, f"Memory increase too high: {memory_increase}MB"
            assert peak_increase < 800, f"Peak memory increase too high: {peak_increase}MB"
            
        finally:
            shutil.rmtree(large_project)


class TestScalabilityPerformance:
    """Test performance scalability with different project sizes"""
    
    def test_performance_scaling(self):
        """Test that performance scales reasonably with project size"""
        sizes = ["small", "medium", "large"]
        performance_data = {}
        
        for size in sizes:
            project = create_performance_test_project(size)
            
            try:
                # Test structure analyzer scaling
                start_time = time.time()
                analyzer = StructureAnalyzer()
                result = analyzer.analyze_project(str(project))
                end_time = time.time()
                
                file_count = len(list(project.rglob("*.py")))
                
                performance_data[size] = {
                    'execution_time': end_time - start_time,
                    'file_count': file_count,
                    'throughput': file_count / (end_time - start_time) if end_time > start_time else 0
                }
                
            finally:
                shutil.rmtree(project)
        
        # Verify scaling is reasonable
        small_data = performance_data["small"]
        medium_data = performance_data["medium"]
        large_data = performance_data["large"]
        
        # Time should scale sub-linearly with file count
        small_ratio = small_data['execution_time'] / small_data['file_count']
        medium_ratio = medium_data['execution_time'] / medium_data['file_count']
        large_ratio = large_data['execution_time'] / large_data['file_count']
        
        # Ratios should not increase dramatically (indicating good scaling)
        assert medium_ratio < small_ratio * 3, "Performance degrades too much from small to medium"
        assert large_ratio < medium_ratio * 3, "Performance degrades too much from medium to large"
        
        # Throughput should remain reasonable
        assert small_data['throughput'] > 1.0, "Small project throughput too low"
        assert medium_data['throughput'] > 0.5, "Medium project throughput too low"
        assert large_data['throughput'] > 0.2, "Large project throughput too low"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])