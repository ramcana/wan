"""
System performance testing and benchmarking for project health system.

This module implements comprehensive performance tests to validate that
the project health system meets performance requirements under various conditions.

Requirements: 1.1, 4.1
"""

import asyncio
import time
import psutil
import tempfile
import shutil
import json
import threading
from pathlib import Path
from typing import Dict, List, Any
import pytest
import yaml
import concurrent.futures

from tools.test-runner.orchestrator import TestSuiteOrchestrator
from tools.doc_generator.documentation_generator import DocumentationGenerator
from tools.config_manager.config_unifier import ConfigurationUnifier
from tools.health_checker.health_checker import ProjectHealthChecker


class PerformanceMetrics:
    """Utility class for collecting performance metrics."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.peak_memory = 0
        self.cpu_usage = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.end_time = time.time()
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_resources(self):
        """Monitor CPU and memory usage."""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                # Monitor memory usage
                memory_info = process.memory_info()
                self.peak_memory = max(self.peak_memory, memory_info.rss)
                
                # Monitor CPU usage
                cpu_percent = process.cpu_percent()
                self.cpu_usage.append(cpu_percent)
                
                time.sleep(0.1)  # Sample every 100ms
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected performance metrics."""
        duration = self.end_time - self.start_time if self.end_time else 0
        avg_cpu = sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0
        
        return {
            "duration_seconds": duration,
            "peak_memory_mb": self.peak_memory / (1024 * 1024),
            "average_cpu_percent": avg_cpu,
            "max_cpu_percent": max(self.cpu_usage) if self.cpu_usage else 0
        }


class TestSystemPerformanceBenchmarks:
    """Performance benchmarks for project health system."""

    @pytest.fixture
    def large_test_project(self):
        """Create a large test project for performance testing."""
        temp_dir = tempfile.mkdtemp()
        project_dir = Path(temp_dir) / "large_project"
        project_dir.mkdir()
        
        # Create extensive project structure
        (project_dir / "tests" / "unit").mkdir(parents=True)
        (project_dir / "tests" / "integration").mkdir(parents=True)
        (project_dir / "tests" / "performance").mkdir(parents=True)
        (project_dir / "tests" / "e2e").mkdir(parents=True)
        (project_dir / "docs" / "user-guide").mkdir(parents=True)
        (project_dir / "docs" / "developer-guide").mkdir(parents=True)
        (project_dir / "docs" / "api").mkdir(parents=True)
        (project_dir / "config" / "environments").mkdir(parents=True)
        (project_dir / "src" / "modules").mkdir(parents=True)
        
        # Create many test files
        for i in range(50):
            test_content = f"""
import pytest

def test_function_{i}_1():
    assert True

def test_function_{i}_2():
    result = {i} * 2
    assert result == {i * 2}

def test_function_{i}_3():
    data = [{j} for j in range(10)]
    assert len(data) == 10

class TestClass{i}:
    def test_method_1(self):
        assert {i} > 0
    
    def test_method_2(self):
        assert {i} + 1 == {i + 1}
"""
            (project_dir / "tests" / "unit" / f"test_module_{i}.py").write_text(test_content)
        
        # Create integration tests
        for i in range(20):
            integration_content = f"""
import pytest
import asyncio

@pytest.mark.asyncio
async def test_integration_{i}():
    await asyncio.sleep(0.01)  # Simulate async operation
    assert True

def test_integration_sync_{i}():
    # Simulate some processing
    result = sum(range({i * 10}))
    assert result >= 0
"""
            (project_dir / "tests" / "integration" / f"test_integration_{i}.py").write_text(integration_content)
        
        # Create extensive documentation
        for i in range(30):
            doc_content = f"""
# Module {i} Documentation

This is documentation for module {i}.

## Overview

Module {i} provides functionality for:
- Feature A
- Feature B
- Feature C

## API Reference

### function_{i}_1()
Description of function {i}_1.

### function_{i}_2()
Description of function {i}_2.

## Examples

```python
result = function_{i}_1()
print(result)
```

## See Also

- [Module {i-1} Documentation](module_{i-1}.md)
- [Module {i+1} Documentation](module_{i+1}.md)
"""
            (project_dir / "docs" / "user-guide" / f"module_{i}.md").write_text(doc_content)
        
        # Create multiple configuration files
        for env in ["development", "staging", "production", "testing"]:
            config_content = {
                "system": {
                    "name": "large_project",
                    "version": "1.0.0",
                    "environment": env,
                    "debug": env == "development"
                },
                "api": {
                    "host": "localhost",
                    "port": 8000 + hash(env) % 1000,
                    "timeout": 30,
                    "rate_limit": 1000
                },
                "database": {
                    "host": f"db-{env}.example.com",
                    "port": 5432,
                    "name": f"large_project_{env}",
                    "pool_size": 10
                },
                "features": {f"feature_{i}": i % 2 == 0 for i in range(20)}
            }
            
            config_file = project_dir / "config" / "environments" / f"{env}.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(config_content, f)
        
        # Create source files
        for i in range(100):
            src_content = f"""
class Module{i}:
    def __init__(self):
        self.value = {i}
    
    def process(self, data):
        return [x * self.value for x in data]
    
    def calculate(self, x, y):
        return x + y + self.value

def function_{i}(param):
    return param * {i}

def async_function_{i}(param):
    import asyncio
    return asyncio.create_task(asyncio.sleep(0.001))
"""
            (project_dir / "src" / "modules" / f"module_{i}.py").write_text(src_content)
        
        yield project_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_test_suite_execution_performance(self, large_test_project):
        """Benchmark test suite execution performance."""
        
        metrics = PerformanceMetrics()
        metrics.start_monitoring()
        
        # Execute full test suite
        test_orchestrator = TestSuiteOrchestrator(
            project_root=large_test_project,
            config_path=None
        )
        
        test_results = await test_orchestrator.run_full_suite()
        
        metrics.stop_monitoring()
        performance_data = metrics.get_metrics()
        
        # Performance assertions
        assert performance_data["duration_seconds"] < 300  # Should complete within 5 minutes
        assert performance_data["peak_memory_mb"] < 1024   # Should use less than 1GB RAM
        
        # Verify test results
        assert test_results is not None
        assert test_results.overall_summary.total_tests > 0
        
        # Log performance metrics
        print(f"Test Suite Performance: {performance_data}")
        
        # Save performance report
        perf_report = {
            "test_type": "test_suite_execution",
            "project_size": {
                "test_files": len(list(large_test_project.glob("tests/**/*.py"))),
                "total_tests": test_results.overall_summary.total_tests
            },
            "performance": performance_data,
            "requirements_met": {
                "duration_under_5min": performance_data["duration_seconds"] < 300,
                "memory_under_1gb": performance_data["peak_memory_mb"] < 1024
            }
        }
        
        report_file = large_test_project / "test_suite_performance.json"
        with open(report_file, 'w') as f:
            json.dump(perf_report, f, indent=2)
        
        assert perf_report["requirements_met"]["duration_under_5min"]
        assert perf_report["requirements_met"]["memory_under_1gb"]

    def test_documentation_generation_performance(self, large_test_project):
        """Benchmark documentation generation performance."""
        
        metrics = PerformanceMetrics()
        metrics.start_monitoring()
        
        # Generate documentation
        doc_generator = DocumentationGenerator(
            source_dirs=[large_test_project / "docs"],
            output_dir=large_test_project / "docs" / "_build"
        )
        
        doc_generator.consolidate_existing_docs()
        doc_generator.generate_search_index()
        validation_report = doc_generator.validate_links()
        
        metrics.stop_monitoring()
        performance_data = metrics.get_metrics()
        
        # Performance assertions
        assert performance_data["duration_seconds"] < 60   # Should complete within 1 minute
        assert performance_data["peak_memory_mb"] < 512    # Should use less than 512MB RAM
        
        # Verify documentation generation
        assert validation_report is not None
        
        # Log performance metrics
        print(f"Documentation Generation Performance: {performance_data}")
        
        # Save performance report
        perf_report = {
            "test_type": "documentation_generation",
            "project_size": {
                "doc_files": len(list(large_test_project.glob("docs/**/*.md"))),
                "total_pages": len(list(large_test_project.glob("docs/**/*.md")))
            },
            "performance": performance_data,
            "requirements_met": {
                "duration_under_1min": performance_data["duration_seconds"] < 60,
                "memory_under_512mb": performance_data["peak_memory_mb"] < 512
            }
        }
        
        report_file = large_test_project / "doc_generation_performance.json"
        with open(report_file, 'w') as f:
            json.dump(perf_report, f, indent=2)
        
        assert perf_report["requirements_met"]["duration_under_1min"]
        assert perf_report["requirements_met"]["memory_under_512mb"]

    def test_configuration_unification_performance(self, large_test_project):
        """Benchmark configuration unification performance."""
        
        metrics = PerformanceMetrics()
        metrics.start_monitoring()
        
        # Unify configurations
        config_unifier = ConfigurationUnifier(
            config_sources=[large_test_project / "config"]
        )
        
        migration_report = config_unifier.migrate_existing_configs()
        
        # Validate all configurations
        config_files = list((large_test_project / "config").rglob("*.yaml"))
        for config_file in config_files:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            config_unifier.validate_configuration(config_data)
        
        metrics.stop_monitoring()
        performance_data = metrics.get_metrics()
        
        # Performance assertions
        assert performance_data["duration_seconds"] < 30   # Should complete within 30 seconds
        assert performance_data["peak_memory_mb"] < 256    # Should use less than 256MB RAM
        
        # Verify configuration processing
        assert migration_report is not None
        
        # Log performance metrics
        print(f"Configuration Unification Performance: {performance_data}")
        
        # Save performance report
        perf_report = {
            "test_type": "configuration_unification",
            "project_size": {
                "config_files": len(config_files),
                "total_configs": len(config_files)
            },
            "performance": performance_data,
            "requirements_met": {
                "duration_under_30sec": performance_data["duration_seconds"] < 30,
                "memory_under_256mb": performance_data["peak_memory_mb"] < 256
            }
        }
        
        report_file = large_test_project / "config_unification_performance.json"
        with open(report_file, 'w') as f:
            json.dump(perf_report, f, indent=2)
        
        assert perf_report["requirements_met"]["duration_under_30sec"]
        assert perf_report["requirements_met"]["memory_under_256mb"]

    @pytest.mark.asyncio
    async def test_health_check_performance(self, large_test_project):
        """Benchmark health check execution performance."""
        
        metrics = PerformanceMetrics()
        metrics.start_monitoring()
        
        # Execute health check
        health_checker = ProjectHealthChecker(project_root=large_test_project)
        health_report = await health_checker.run_health_check()
        
        metrics.stop_monitoring()
        performance_data = metrics.get_metrics()
        
        # Performance assertions
        assert performance_data["duration_seconds"] < 30   # Should complete within 30 seconds
        assert performance_data["peak_memory_mb"] < 512    # Should use less than 512MB RAM
        
        # Verify health check results
        assert health_report is not None
        assert health_report.overall_score >= 0
        assert health_report.overall_score <= 100
        
        # Log performance metrics
        print(f"Health Check Performance: {performance_data}")
        
        # Save performance report
        perf_report = {
            "test_type": "health_check_execution",
            "project_size": {
                "total_files": len(list(large_test_project.rglob("*"))),
                "components_checked": len(health_report.component_scores)
            },
            "performance": performance_data,
            "health_score": health_report.overall_score,
            "requirements_met": {
                "duration_under_30sec": performance_data["duration_seconds"] < 30,
                "memory_under_512mb": performance_data["peak_memory_mb"] < 512
            }
        }
        
        report_file = large_test_project / "health_check_performance.json"
        with open(report_file, 'w') as f:
            json.dump(perf_report, f, indent=2)
        
        assert perf_report["requirements_met"]["duration_under_30sec"]
        assert perf_report["requirements_met"]["memory_under_512mb"]

    @pytest.mark.asyncio
    async def test_concurrent_operations_performance(self, large_test_project):
        """Benchmark performance of concurrent operations."""
        
        metrics = PerformanceMetrics()
        metrics.start_monitoring()
        
        # Define concurrent operations
        async def run_tests():
            orchestrator = TestSuiteOrchestrator(
                project_root=large_test_project,
                config_path=None
            )
            return await orchestrator.run_category("unit")
        
        async def generate_docs():
            doc_generator = DocumentationGenerator(
                source_dirs=[large_test_project / "docs"],
                output_dir=large_test_project / "docs" / "_build"
            )
            doc_generator.consolidate_existing_docs()
            return "docs_generated"
        
        async def check_health():
            health_checker = ProjectHealthChecker(project_root=large_test_project)
            return await health_checker.run_health_check()
        
        # Execute operations concurrently
        results = await asyncio.gather(
            run_tests(),
            generate_docs(),
            check_health(),
            return_exceptions=True
        )
        
        metrics.stop_monitoring()
        performance_data = metrics.get_metrics()
        
        # Performance assertions
        assert performance_data["duration_seconds"] < 180  # Should complete within 3 minutes
        assert performance_data["peak_memory_mb"] < 2048   # Should use less than 2GB RAM
        
        # Verify all operations completed successfully
        assert len(results) == 3
        for i, result in enumerate(results):
            assert not isinstance(result, Exception), f"Operation {i} failed: {result}"
        
        # Log performance metrics
        print(f"Concurrent Operations Performance: {performance_data}")
        
        # Save performance report
        perf_report = {
            "test_type": "concurrent_operations",
            "operations": ["test_execution", "doc_generation", "health_check"],
            "performance": performance_data,
            "requirements_met": {
                "duration_under_3min": performance_data["duration_seconds"] < 180,
                "memory_under_2gb": performance_data["peak_memory_mb"] < 2048
            }
        }
        
        report_file = large_test_project / "concurrent_operations_performance.json"
        with open(report_file, 'w') as f:
            json.dump(perf_report, f, indent=2)
        
        assert perf_report["requirements_met"]["duration_under_3min"]
        assert perf_report["requirements_met"]["memory_under_2gb"]

    def test_memory_usage_under_load(self, large_test_project):
        """Test memory usage under high load conditions."""
        
        initial_memory = psutil.Process().memory_info().rss
        
        # Create multiple instances of components
        components = []
        
        try:
            # Create multiple test orchestrators
            for i in range(5):
                orchestrator = TestSuiteOrchestrator(
                    project_root=large_test_project,
                    config_path=None
                )
                components.append(orchestrator)
            
            # Create multiple documentation generators
            for i in range(3):
                doc_generator = DocumentationGenerator(
                    source_dirs=[large_test_project / "docs"],
                    output_dir=large_test_project / "docs" / f"_build_{i}"
                )
                components.append(doc_generator)
            
            # Create multiple health checkers
            for i in range(3):
                health_checker = ProjectHealthChecker(project_root=large_test_project)
                components.append(health_checker)
            
            # Measure peak memory usage
            peak_memory = psutil.Process().memory_info().rss
            memory_increase = (peak_memory - initial_memory) / (1024 * 1024)  # MB
            
            # Memory usage should be reasonable even with multiple instances
            assert memory_increase < 1024  # Should not increase by more than 1GB
            
            print(f"Memory increase under load: {memory_increase:.2f} MB")
            
        finally:
            # Cleanup
            components.clear()

    @pytest.mark.asyncio
    async def test_scalability_with_project_size(self, large_test_project):
        """Test how performance scales with project size."""
        
        # Test with different project sizes
        project_sizes = [
            {"name": "small", "test_files": 10, "doc_files": 5},
            {"name": "medium", "test_files": 25, "doc_files": 15},
            {"name": "large", "test_files": 50, "doc_files": 30}
        ]
        
        scalability_results = []
        
        for size_config in project_sizes:
            # Create subset of project for testing
            test_dir = large_test_project / f"test_{size_config['name']}"
            test_dir.mkdir(exist_ok=True)
            (test_dir / "tests").mkdir(exist_ok=True)
            (test_dir / "docs").mkdir(exist_ok=True)
            
            # Copy subset of files
            test_files = list((large_test_project / "tests").glob("**/*.py"))[:size_config["test_files"]]
            doc_files = list((large_test_project / "docs").glob("**/*.md"))[:size_config["doc_files"]]
            
            for test_file in test_files:
                dest = test_dir / "tests" / test_file.name
                shutil.copy2(test_file, dest)
            
            for doc_file in doc_files:
                dest = test_dir / "docs" / doc_file.name
                shutil.copy2(doc_file, dest)
            
            # Measure performance for this size
            metrics = PerformanceMetrics()
            metrics.start_monitoring()
            
            # Run health check on subset
            health_checker = ProjectHealthChecker(project_root=test_dir)
            health_report = await health_checker.run_health_check()
            
            metrics.stop_monitoring()
            performance_data = metrics.get_metrics()
            
            scalability_results.append({
                "size": size_config["name"],
                "test_files": size_config["test_files"],
                "doc_files": size_config["doc_files"],
                "duration": performance_data["duration_seconds"],
                "memory": performance_data["peak_memory_mb"]
            })
        
        # Analyze scalability
        # Performance should scale reasonably with project size
        small_duration = next(r["duration"] for r in scalability_results if r["size"] == "small")
        large_duration = next(r["duration"] for r in scalability_results if r["size"] == "large")
        
        # Large project should not take more than 10x longer than small project
        scalability_ratio = large_duration / small_duration if small_duration > 0 else 1
        assert scalability_ratio < 10, f"Poor scalability: {scalability_ratio}x slower"
        
        print(f"Scalability results: {scalability_results}")
        
        # Save scalability report
        scalability_report = {
            "test_type": "scalability_analysis",
            "results": scalability_results,
            "scalability_ratio": scalability_ratio,
            "scalability_acceptable": scalability_ratio < 10
        }
        
        report_file = large_test_project / "scalability_performance.json"
        with open(report_file, 'w') as f:
            json.dump(scalability_report, f, indent=2)
        
        assert scalability_report["scalability_acceptable"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])