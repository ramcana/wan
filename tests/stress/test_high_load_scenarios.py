"""
Stress tests for high-load scenarios and resource constraints.

This module implements stress tests to validate system behavior under
extreme conditions and resource constraints.

Requirements: 1.7, 4.7
"""

import asyncio
import concurrent.futures
import threading
import time
import tempfile
import shutil
import json
import random
from pathlib import Path
from typing import Dict, List, Any
import pytest
import yaml
import psutil

from tools.test-runner.orchestrator import TestSuiteOrchestrator
from tools.doc_generator.documentation_generator import DocumentationGenerator
from tools.config_manager.config_unifier import ConfigurationUnifier
from tools.health_checker.health_checker import ProjectHealthChecker


class StressTestMetrics:
    """Collect and analyze stress test metrics."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.operations_completed = 0
        self.operations_failed = 0
        self.response_times = []
        self.memory_samples = []
        self.cpu_samples = []
        self.errors = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start monitoring system resources."""
        self.start_time = time.time()
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring system resources."""
        self.end_time = time.time()
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_resources(self):
        """Monitor CPU and memory usage during stress test."""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                # Sample memory usage
                memory_info = process.memory_info()
                self.memory_samples.append(memory_info.rss / (1024 * 1024))  # MB
                
                # Sample CPU usage
                cpu_percent = process.cpu_percent()
                self.cpu_samples.append(cpu_percent)
                
                time.sleep(0.1)  # Sample every 100ms
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
    
    def record_operation(self, success: bool, duration: float, error: str = None):
        """Record the result of an operation."""
        if success:
            self.operations_completed += 1
        else:
            self.operations_failed += 1
            if error:
                self.errors.append(error)
        
        self.response_times.append(duration)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get stress test summary metrics."""
        total_duration = self.end_time - self.start_time if self.end_time else 0
        total_operations = self.operations_completed + self.operations_failed
        
        return {
            "duration_seconds": total_duration,
            "total_operations": total_operations,
            "operations_completed": self.operations_completed,
            "operations_failed": self.operations_failed,
            "success_rate": self.operations_completed / total_operations if total_operations > 0 else 0,
            "average_response_time": sum(self.response_times) / len(self.response_times) if self.response_times else 0,
            "max_response_time": max(self.response_times) if self.response_times else 0,
            "min_response_time": min(self.response_times) if self.response_times else 0,
            "peak_memory_mb": max(self.memory_samples) if self.memory_samples else 0,
            "average_memory_mb": sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else 0,
            "peak_cpu_percent": max(self.cpu_samples) if self.cpu_samples else 0,
            "average_cpu_percent": sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0,
            "error_count": len(self.errors),
            "unique_errors": len(set(self.errors))
        }


class TestHighLoadScenarios:
    """Stress tests for high-load scenarios."""

    @pytest.fixture
    def stress_test_project(self):
        """Create a project for stress testing."""
        temp_dir = tempfile.mkdtemp()
        project_dir = Path(temp_dir) / "stress_project"
        project_dir.mkdir()
        
        # Create extensive project structure for stress testing
        (project_dir / "tests" / "unit").mkdir(parents=True)
        (project_dir / "tests" / "integration").mkdir(parents=True)
        (project_dir / "docs" / "sections").mkdir(parents=True)
        (project_dir / "config" / "environments").mkdir(parents=True)
        
        # Create many test files to stress the system
        for i in range(100):
            test_content = f"""
import pytest
import time

def test_operation_{i}_1():
    # Simulate some work
    time.sleep(0.001)
    assert True

def test_operation_{i}_2():
    result = sum(range({i % 100}))
    assert result >= 0

@pytest.mark.asyncio
async def test_async_operation_{i}():
    await asyncio.sleep(0.001)
    assert True
"""
            (project_dir / "tests" / "unit" / f"test_stress_{i}.py").write_text(test_content)
        
        # Create documentation files
        for i in range(50):
            doc_content = f"""
# Stress Test Documentation {i}

This is documentation file {i} for stress testing.

## Section A
Content for section A in document {i}.

## Section B
Content for section B in document {i}.

## Links
- [Document {i-1}](stress_doc_{i-1}.md)
- [Document {i+1}](stress_doc_{i+1}.md)
"""
            (project_dir / "docs" / "sections" / f"stress_doc_{i}.md").write_text(doc_content)
        
        # Create configuration files
        for env in ["dev", "test", "staging", "prod"]:
            for i in range(10):
                config_data = {
                    "system": {
                        "name": f"stress_project_{env}_{i}",
                        "version": f"1.{i}.0",
                        "environment": env
                    },
                    "services": {
                        f"service_{j}": {
                            "port": 8000 + i * 10 + j,
                            "timeout": 30 + j,
                            "workers": j + 1
                        } for j in range(5)
                    }
                }
                
                config_file = project_dir / "config" / "environments" / f"{env}_{i}.yaml"
                with open(config_file, 'w') as f:
                    yaml.dump(config_data, f)
        
        yield project_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_concurrent_health_checks_stress(self, stress_test_project):
        """Stress test with many concurrent health checks."""
        
        metrics = StressTestMetrics()
        metrics.start_monitoring()
        
        async def run_health_check(check_id: int):
            """Run a single health check."""
            start_time = time.time()
            try:
                health_checker = ProjectHealthChecker(project_root=stress_test_project)
                result = await health_checker.run_health_check()
                duration = time.time() - start_time
                metrics.record_operation(True, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                metrics.record_operation(False, duration, str(e))
                raise e
        
        # Run many concurrent health checks
        concurrent_checks = 20
        tasks = [run_health_check(i) for i in range(concurrent_checks)]
        
        # Execute with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=120  # 2 minute timeout
            )
        except asyncio.TimeoutError:
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            results = []
        
        metrics.stop_monitoring()
        summary = metrics.get_summary()
        
        # Stress test assertions
        assert summary["operations_completed"] > 0, "Some operations should complete"
        assert summary["success_rate"] > 0.5, "Success rate should be > 50%"
        assert summary["peak_memory_mb"] < 2048, "Memory usage should be reasonable"
        assert summary["average_response_time"] < 30, "Average response time should be < 30s"
        
        # Log stress test results
        print(f"Concurrent Health Checks Stress Test: {summary}")
        
        # Save stress test report
        report_file = stress_test_project / "concurrent_health_checks_stress.json"
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2)

    def test_rapid_configuration_changes_stress(self, stress_test_project):
        """Stress test with rapid configuration changes."""
        
        metrics = StressTestMetrics()
        metrics.start_monitoring()
        
        config_unifier = ConfigurationUnifier(
            config_sources=[stress_test_project / "config"]
        )
        
        # Perform rapid configuration operations
        for i in range(100):
            start_time = time.time()
            try:
                # Create temporary config file
                temp_config = {
                    "system": {"name": f"temp_config_{i}", "version": "1.0.0"},
                    "temp_setting": f"value_{i}"
                }
                
                temp_file = stress_test_project / "config" / f"temp_{i}.yaml"
                with open(temp_file, 'w') as f:
                    yaml.dump(temp_config, f)
                
                # Validate configuration
                validation_result = config_unifier.validate_configuration(temp_config)
                
                # Remove temporary file
                temp_file.unlink()
                
                duration = time.time() - start_time
                metrics.record_operation(True, duration)
                
            except Exception as e:
                duration = time.time() - start_time
                metrics.record_operation(False, duration, str(e))
        
        metrics.stop_monitoring()
        summary = metrics.get_summary()
        
        # Stress test assertions
        assert summary["operations_completed"] > 50, "Most operations should complete"
        assert summary["success_rate"] > 0.8, "Success rate should be > 80%"
        assert summary["average_response_time"] < 1.0, "Operations should be fast"
        
        print(f"Rapid Configuration Changes Stress Test: {summary}")

    @pytest.mark.asyncio
    async def test_massive_test_execution_stress(self, stress_test_project):
        """Stress test with massive test suite execution."""
        
        metrics = StressTestMetrics()
        metrics.start_monitoring()
        
        # Execute test suite multiple times concurrently
        async def run_test_suite(suite_id: int):
            start_time = time.time()
            try:
                test_orchestrator = TestSuiteOrchestrator(
                    project_root=stress_test_project,
                    config_path=None
                )
                
                # Run subset of tests to manage load
                result = await test_orchestrator.run_category("unit")
                duration = time.time() - start_time
                metrics.record_operation(True, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                metrics.record_operation(False, duration, str(e))
                raise e
        
        # Run multiple test suites concurrently
        concurrent_suites = 5  # Reduced to manage system load
        tasks = [run_test_suite(i) for i in range(concurrent_suites)]
        
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=300  # 5 minute timeout
            )
        except asyncio.TimeoutError:
            for task in tasks:
                if not task.done():
                    task.cancel()
            results = []
        
        metrics.stop_monitoring()
        summary = metrics.get_summary()
        
        # Stress test assertions
        assert summary["operations_completed"] > 0, "Some test suites should complete"
        assert summary["peak_memory_mb"] < 4096, "Memory usage should be manageable"
        
        print(f"Massive Test Execution Stress Test: {summary}")

    def test_documentation_generation_under_load(self, stress_test_project):
        """Stress test documentation generation with many files."""
        
        metrics = StressTestMetrics()
        metrics.start_monitoring()
        
        # Generate documentation multiple times with different configurations
        for i in range(10):
            start_time = time.time()
            try:
                doc_generator = DocumentationGenerator(
                    source_dirs=[stress_test_project / "docs"],
                    output_dir=stress_test_project / "docs" / f"_build_{i}"
                )
                
                doc_generator.consolidate_existing_docs()
                doc_generator.generate_search_index()
                validation_report = doc_generator.validate_links()
                
                duration = time.time() - start_time
                metrics.record_operation(True, duration)
                
            except Exception as e:
                duration = time.time() - start_time
                metrics.record_operation(False, duration, str(e))
        
        metrics.stop_monitoring()
        summary = metrics.get_summary()
        
        # Stress test assertions
        assert summary["operations_completed"] > 5, "Most operations should complete"
        assert summary["success_rate"] > 0.7, "Success rate should be > 70%"
        assert summary["peak_memory_mb"] < 1024, "Memory usage should be reasonable"
        
        print(f"Documentation Generation Under Load: {summary}")

    def test_memory_exhaustion_resilience(self, stress_test_project):
        """Test system resilience under memory pressure."""
        
        metrics = StressTestMetrics()
        metrics.start_monitoring()
        
        # Create memory pressure by allocating large objects
        memory_hogs = []
        
        try:
            # Gradually increase memory pressure
            for i in range(10):
                start_time = time.time()
                try:
                    # Allocate memory
                    memory_hog = bytearray(50 * 1024 * 1024)  # 50MB
                    memory_hogs.append(memory_hog)
                    
                    # Test system operation under memory pressure
                    health_checker = ProjectHealthChecker(project_root=stress_test_project)
                    result = asyncio.run(health_checker.run_health_check())
                    
                    duration = time.time() - start_time
                    metrics.record_operation(True, duration)
                    
                except (MemoryError, OSError) as e:
                    # Expected under extreme memory pressure
                    duration = time.time() - start_time
                    metrics.record_operation(False, duration, str(e))
                    break
                except Exception as e:
                    duration = time.time() - start_time
                    metrics.record_operation(False, duration, str(e))
        
        finally:
            # Release memory
            memory_hogs.clear()
        
        metrics.stop_monitoring()
        summary = metrics.get_summary()
        
        # System should handle some level of memory pressure
        assert summary["operations_completed"] > 0, "Some operations should complete"
        
        print(f"Memory Exhaustion Resilience Test: {summary}")

    def test_file_descriptor_exhaustion(self, stress_test_project):
        """Test system behavior when file descriptors are exhausted."""
        
        metrics = StressTestMetrics()
        metrics.start_monitoring()
        
        open_files = []
        
        try:
            # Open many files to exhaust file descriptors
            for i in range(100):
                start_time = time.time()
                try:
                    # Create and open temporary files
                    temp_file = stress_test_project / f"temp_fd_{i}.txt"
                    temp_file.write_text(f"Temporary file {i}")
                    
                    file_handle = open(temp_file, 'r')
                    open_files.append((temp_file, file_handle))
                    
                    # Test system operation with many open files
                    if i % 10 == 0:  # Test every 10 iterations
                        config_unifier = ConfigurationUnifier(
                            config_sources=[stress_test_project / "config"]
                        )
                        result = config_unifier.migrate_existing_configs()
                    
                    duration = time.time() - start_time
                    metrics.record_operation(True, duration)
                    
                except (OSError, IOError) as e:
                    # Expected when file descriptors are exhausted
                    duration = time.time() - start_time
                    metrics.record_operation(False, duration, str(e))
                    if "too many open files" in str(e).lower():
                        break
                except Exception as e:
                    duration = time.time() - start_time
                    metrics.record_operation(False, duration, str(e))
        
        finally:
            # Close all open files
            for temp_file, file_handle in open_files:
                try:
                    file_handle.close()
                    temp_file.unlink()
                except:
                    pass
        
        metrics.stop_monitoring()
        summary = metrics.get_summary()
        
        # System should handle reasonable number of file operations
        assert summary["operations_completed"] > 0, "Some operations should complete"
        
        print(f"File Descriptor Exhaustion Test: {summary}")

    @pytest.mark.asyncio
    async def test_long_running_operations_stability(self, stress_test_project):
        """Test system stability during long-running operations."""
        
        metrics = StressTestMetrics()
        metrics.start_monitoring()
        
        # Run operations for extended period
        end_time = time.time() + 60  # Run for 1 minute
        operation_count = 0
        
        while time.time() < end_time:
            start_time = time.time()
            try:
                # Alternate between different types of operations
                if operation_count % 3 == 0:
                    # Health check
                    health_checker = ProjectHealthChecker(project_root=stress_test_project)
                    result = await health_checker.run_health_check()
                elif operation_count % 3 == 1:
                    # Configuration validation
                    config_unifier = ConfigurationUnifier(
                        config_sources=[stress_test_project / "config"]
                    )
                    result = config_unifier.migrate_existing_configs()
                else:
                    # Documentation validation
                    doc_generator = DocumentationGenerator(
                        source_dirs=[stress_test_project / "docs"],
                        output_dir=stress_test_project / "docs" / "_build_long"
                    )
                    result = doc_generator.validate_links()
                
                duration = time.time() - start_time
                metrics.record_operation(True, duration)
                operation_count += 1
                
                # Brief pause between operations
                await asyncio.sleep(0.1)
                
            except Exception as e:
                duration = time.time() - start_time
                metrics.record_operation(False, duration, str(e))
                operation_count += 1
        
        metrics.stop_monitoring()
        summary = metrics.get_summary()
        
        # System should maintain stability over time
        assert summary["operations_completed"] > 10, "Many operations should complete"
        assert summary["success_rate"] > 0.8, "Success rate should remain high"
        
        print(f"Long Running Operations Stability Test: {summary}")

    def test_rapid_component_creation_destruction(self, stress_test_project):
        """Test rapid creation and destruction of components."""
        
        metrics = StressTestMetrics()
        metrics.start_monitoring()
        
        # Rapidly create and destroy components
        for i in range(200):
            start_time = time.time()
            try:
                # Create components
                health_checker = ProjectHealthChecker(project_root=stress_test_project)
                doc_generator = DocumentationGenerator(
                    source_dirs=[stress_test_project / "docs"],
                    output_dir=stress_test_project / "docs" / f"_temp_{i}"
                )
                config_unifier = ConfigurationUnifier(
                    config_sources=[stress_test_project / "config"]
                )
                
                # Use components briefly
                if i % 10 == 0:  # Occasional actual operation
                    result = asyncio.run(health_checker.run_health_check())
                
                # Components should be garbage collected automatically
                del health_checker, doc_generator, config_unifier
                
                duration = time.time() - start_time
                metrics.record_operation(True, duration)
                
            except Exception as e:
                duration = time.time() - start_time
                metrics.record_operation(False, duration, str(e))
        
        # Force garbage collection
        import gc
        gc.collect()
        
        metrics.stop_monitoring()
        summary = metrics.get_summary()
        
        # System should handle rapid component lifecycle
        assert summary["operations_completed"] > 150, "Most operations should complete"
        assert summary["success_rate"] > 0.9, "Success rate should be very high"
        assert summary["peak_memory_mb"] < 1024, "Memory should not grow excessively"
        
        print(f"Rapid Component Creation/Destruction Test: {summary}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])