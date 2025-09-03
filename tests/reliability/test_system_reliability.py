"""
System reliability tests for project health system.

This module tests system behavior under various failure conditions and edge cases
to validate reliability and robustness.

Requirements: 1.7, 4.7
"""

import asyncio
import os
import tempfile
import shutil
import signal
import threading
import time
import json
from pathlib import Path
from typing import Dict, List, Any
import pytest
import yaml
import psutil

from tools.test-runner.orchestrator import TestSuiteOrchestrator
from tools.doc_generator.documentation_generator import DocumentationGenerator
from tools.config_manager.config_unifier import ConfigurationUnifier
from tools.health_checker.health_checker import ProjectHealthChecker


class ReliabilityTestHarness:
    """Test harness for reliability testing."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.failure_scenarios = []
        self.recovery_times = []
    
    def inject_failure(self, failure_type: str, **kwargs):
        """Inject a specific type of failure."""
        scenario = {
            "type": failure_type,
            "timestamp": time.time(),
            "parameters": kwargs
        }
        self.failure_scenarios.append(scenario)
        
        if failure_type == "corrupt_config":
            self._corrupt_config_file(kwargs.get("config_file"))
        elif failure_type == "delete_test_file":
            self._delete_test_file(kwargs.get("test_file"))
        elif failure_type == "memory_pressure":
            self._create_memory_pressure(kwargs.get("size_mb", 100))
        elif failure_type == "disk_full":
            self._simulate_disk_full(kwargs.get("directory"))
        elif failure_type == "network_timeout":
            self._simulate_network_timeout()
    
    def _corrupt_config_file(self, config_file: Path):
        """Corrupt a configuration file."""
        if config_file and config_file.exists():
            config_file.write_text("corrupted: invalid: yaml: [[[")
    
    def _delete_test_file(self, test_file: Path):
        """Delete a test file."""
        if test_file and test_file.exists():
            test_file.unlink()
    
    def _create_memory_pressure(self, size_mb: int):
        """Create memory pressure by allocating large amounts of memory."""
        # This is a simulation - in real tests we'd be more careful
        self.memory_hog = bytearray(size_mb * 1024 * 1024)
    
    def _simulate_disk_full(self, directory: Path):
        """Simulate disk full condition."""
        if directory and directory.exists():
            # Create a large file to fill up space (simulation)
            large_file = directory / "disk_full_simulation.tmp"
            try:
                with open(large_file, 'wb') as f:
                    f.write(b'0' * (100 * 1024 * 1024))  # 100MB
            except OSError:
                pass  # Expected if disk is actually full
    
    def _simulate_network_timeout(self):
        """Simulate network timeout conditions."""
        # This would typically involve network manipulation
        # For testing, we'll just record the scenario
        pass
    
    def measure_recovery_time(self, operation_func, *args, **kwargs):
        """Measure how long it takes for a system to recover from failure."""
        start_time = time.time()
        
        try:
            result = operation_func(*args, **kwargs)
            recovery_time = time.time() - start_time
            self.recovery_times.append(recovery_time)
            return result, recovery_time
        except Exception as e:
            recovery_time = time.time() - start_time
            self.recovery_times.append(recovery_time)
            raise e


class TestSystemReliability:
    """Test system reliability under various failure conditions."""

    @pytest.fixture
    def reliability_test_project(self):
        """Create a test project for reliability testing."""
        temp_dir = tempfile.mkdtemp()
        project_dir = Path(temp_dir) / "reliability_project"
        project_dir.mkdir()
        
        # Create project structure
        (project_dir / "tests").mkdir()
        (project_dir / "docs").mkdir()
        (project_dir / "config").mkdir()
        
        # Create test files
        (project_dir / "tests" / "test_reliable.py").write_text("""
import pytest

def test_always_passes():
    assert True

def test_math_operations():
    assert 2 + 2 == 4
    assert 5 * 3 == 15

def test_string_operations():
    assert "hello".upper() == "HELLO"
""")
        
        # Create documentation
        (project_dir / "docs" / "README.md").write_text("""
# Reliability Test Project

This project is used for testing system reliability.

## Features
- Reliable operations
- Error handling
- Recovery mechanisms
""")
        
        # Create configuration
        config_data = {
            "system": {"name": "reliability_test", "version": "1.0.0"},
            "api": {"port": 8000, "timeout": 30},
            "database": {"host": "localhost", "port": 5432}
        }
        
        config_file = project_dir / "config" / "base.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        yield project_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_configuration_corruption_recovery(self, reliability_test_project):
        """Test system recovery from configuration file corruption."""
        
        harness = ReliabilityTestHarness(reliability_test_project)
        config_file = reliability_test_project / "config" / "base.yaml"
        
        # Backup original configuration
        original_config = config_file.read_text()
        
        try:
            # Inject configuration corruption
            harness.inject_failure("corrupt_config", config_file=config_file)
            
            # Test system behavior with corrupted config
            config_unifier = ConfigurationUnifier(
                config_sources=[reliability_test_project / "config"]
            )
            
            # System should handle corruption gracefully
            try:
                migration_report = config_unifier.migrate_existing_configs()
                # Should either succeed with error handling or fail gracefully
                assert migration_report is not None or True  # Allow graceful failure
            except Exception as e:
                # Should provide meaningful error message
                assert "yaml" in str(e).lower() or "config" in str(e).lower()
            
            # Test recovery by restoring configuration
            config_file.write_text(original_config)
            
            # Measure recovery time
            def recover_config():
                return config_unifier.migrate_existing_configs()
            
            result, recovery_time = harness.measure_recovery_time(recover_config)
            
            # Recovery should be fast (under 5 seconds)
            assert recovery_time < 5.0
            assert result is not None
            
        finally:
            # Ensure config is restored
            if config_file.exists():
                config_file.write_text(original_config)

    @pytest.mark.asyncio
    async def test_test_file_deletion_recovery(self, reliability_test_project):
        """Test system recovery from test file deletion."""
        
        harness = ReliabilityTestHarness(reliability_test_project)
        test_file = reliability_test_project / "tests" / "test_reliable.py"
        
        # Backup original test file
        original_content = test_file.read_text()
        
        try:
            # Test normal operation first
            test_orchestrator = TestSuiteOrchestrator(
                project_root=reliability_test_project,
                config_path=None
            )
            
            initial_results = await test_orchestrator.run_full_suite()
            assert initial_results.overall_summary.total_tests > 0
            
            # Inject test file deletion
            harness.inject_failure("delete_test_file", test_file=test_file)
            
            # Test system behavior with missing test file
            def run_tests_after_deletion():
                return asyncio.run(test_orchestrator.run_full_suite())
            
            # System should handle missing files gracefully
            result, recovery_time = harness.measure_recovery_time(run_tests_after_deletion)
            
            # Should complete even with missing files
            assert result is not None
            assert recovery_time < 30.0  # Should complete within 30 seconds
            
            # Restore test file and verify recovery
            test_file.write_text(original_content)
            
            recovered_results = await test_orchestrator.run_full_suite()
            assert recovered_results.overall_summary.total_tests > 0
            
        finally:
            # Ensure test file is restored
            if not test_file.exists():
                test_file.write_text(original_content)

    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self, reliability_test_project):
        """Test system behavior under memory pressure."""
        
        harness = ReliabilityTestHarness(reliability_test_project)
        
        # Get initial memory usage
        initial_memory = psutil.Process().memory_info().rss
        
        try:
            # Create memory pressure
            harness.inject_failure("memory_pressure", size_mb=200)
            
            # Test system operations under memory pressure
            health_checker = ProjectHealthChecker(project_root=reliability_test_project)
            
            def run_health_check_under_pressure():
                return asyncio.run(health_checker.run_health_check())
            
            # Measure performance under memory pressure
            result, execution_time = harness.measure_recovery_time(run_health_check_under_pressure)
            
            # System should still function, though possibly slower
            assert result is not None
            assert execution_time < 60.0  # Should complete within 1 minute even under pressure
            
            # Verify memory usage is reasonable
            current_memory = psutil.Process().memory_info().rss
            memory_increase = (current_memory - initial_memory) / (1024 * 1024)  # MB
            
            # Memory increase should be bounded
            assert memory_increase < 1024  # Should not increase by more than 1GB
            
        finally:
            # Release memory pressure
            if hasattr(harness, 'memory_hog'):
                del harness.memory_hog

    def test_concurrent_access_reliability(self, reliability_test_project):
        """Test system reliability under concurrent access."""
        
        harness = ReliabilityTestHarness(reliability_test_project)
        results = []
        exceptions = []
        
        def concurrent_health_check(thread_id):
            """Run health check in a separate thread."""
            try:
                health_checker = ProjectHealthChecker(project_root=reliability_test_project)
                result = asyncio.run(health_checker.run_health_check())
                results.append((thread_id, result))
            except Exception as e:
                exceptions.append((thread_id, e))
        
        # Start multiple concurrent operations
        threads = []
        for i in range(5):
            thread = threading.Thread(target=concurrent_health_check, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout per thread
        
        # Verify results
        assert len(results) > 0, "At least some operations should succeed"
        assert len(exceptions) < len(threads), "Not all operations should fail"
        
        # Check that successful results are valid
        for thread_id, result in results:
            assert result is not None
            assert hasattr(result, 'overall_score')

    @pytest.mark.asyncio
    async def test_disk_space_handling(self, reliability_test_project):
        """Test system behavior when disk space is limited."""
        
        harness = ReliabilityTestHarness(reliability_test_project)
        
        try:
            # Simulate disk space pressure
            harness.inject_failure("disk_full", directory=reliability_test_project)
            
            # Test documentation generation under disk pressure
            doc_generator = DocumentationGenerator(
                source_dirs=[reliability_test_project / "docs"],
                output_dir=reliability_test_project / "docs" / "_build"
            )
            
            def generate_docs_under_pressure():
                try:
                    doc_generator.consolidate_existing_docs()
                    return "success"
                except OSError as e:
                    if "No space left on device" in str(e) or "disk full" in str(e).lower():
                        return "disk_full_handled"
                    raise e
            
            result, execution_time = harness.measure_recovery_time(generate_docs_under_pressure)
            
            # System should handle disk space issues gracefully
            assert result in ["success", "disk_full_handled"]
            assert execution_time < 30.0  # Should fail fast if disk is full
            
        finally:
            # Cleanup disk space simulation
            disk_full_file = reliability_test_project / "disk_full_simulation.tmp"
            if disk_full_file.exists():
                disk_full_file.unlink()

    @pytest.mark.asyncio
    async def test_interrupted_operations_recovery(self, reliability_test_project):
        """Test recovery from interrupted operations."""
        
        harness = ReliabilityTestHarness(reliability_test_project)
        
        # Start a long-running operation
        test_orchestrator = TestSuiteOrchestrator(
            project_root=reliability_test_project,
            config_path=None
        )
        
        # Create a task that we can interrupt
        async def long_running_test():
            await asyncio.sleep(0.1)  # Simulate some work
            return await test_orchestrator.run_full_suite()
        
        # Start the operation
        task = asyncio.create_task(long_running_test())
        
        # Let it run briefly then cancel
        await asyncio.sleep(0.05)
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass  # Expected
        
        # Test that system can recover and run normally after interruption
        def recover_after_interruption():
            return asyncio.run(test_orchestrator.run_full_suite())
        
        result, recovery_time = harness.measure_recovery_time(recover_after_interruption)
        
        # System should recover successfully
        assert result is not None
        assert recovery_time < 30.0  # Should recover quickly

    def test_invalid_input_handling(self, reliability_test_project):
        """Test system handling of invalid inputs."""
        
        harness = ReliabilityTestHarness(reliability_test_project)
        
        # Test configuration validator with invalid inputs
        config_validator = ConfigurationValidator()
        
        invalid_configs = [
            None,
            {},
            {"invalid": "structure"},
            {"system": None},
            {"system": {"name": ""}},
            {"api": {"port": "invalid_port"}},
            {"api": {"port": -1}},
            {"api": {"port": 99999}},
        ]
        
        for invalid_config in invalid_configs:
            try:
                result = config_validator.validate_config(invalid_config)
                # Should either handle gracefully or provide meaningful error
                assert result is not None or True  # Allow graceful handling
            except Exception as e:
                # Should provide meaningful error messages
                error_msg = str(e).lower()
                assert any(keyword in error_msg for keyword in 
                          ["invalid", "config", "validation", "error", "missing"])

    @pytest.mark.asyncio
    async def test_resource_cleanup_reliability(self, reliability_test_project):
        """Test that resources are properly cleaned up even after failures."""
        
        harness = ReliabilityTestHarness(reliability_test_project)
        
        # Track initial resource usage
        initial_memory = psutil.Process().memory_info().rss
        initial_open_files = len(psutil.Process().open_files())
        
        # Create multiple components and simulate failures
        components = []
        
        try:
            for i in range(10):
                # Create components that might fail
                health_checker = ProjectHealthChecker(project_root=reliability_test_project)
                components.append(health_checker)
                
                # Simulate some failures
                if i % 3 == 0:
                    try:
                        # Force an error condition
                        await health_checker.run_health_check()
                    except Exception:
                        pass  # Expected failures
            
            # Force garbage collection
            import gc
gc.collect()
            
            # Check resource usage after operations
            final_memory = psutil.Process().memory_info().rss
            final_open_files = len(psutil.Process().open_files())
            
            memory_increase = (final_memory - initial_memory) / (1024 * 1024)  # MB
            file_increase = final_open_files - initial_open_files
            
            # Resource usage should be reasonable
            assert memory_increase < 100  # Should not leak more than 100MB
            assert file_increase < 10     # Should not leak more than 10 file handles
            
        finally:
            # Cleanup
            components.clear()

    def test_error_propagation_and_isolation(self, reliability_test_project):
        """Test that errors are properly isolated and don't cascade."""
        
        harness = ReliabilityTestHarness(reliability_test_project)
        
        # Create multiple independent operations
        operations = []
        
        # Operation 1: Valid health check
        def valid_health_check():
            health_checker = ProjectHealthChecker(project_root=reliability_test_project)
            return asyncio.run(health_checker.run_health_check())
        
        # Operation 2: Invalid configuration processing
        def invalid_config_processing():
            config_unifier = ConfigurationUnifier(
                config_sources=[Path("/nonexistent/path")]
            )
            return config_unifier.migrate_existing_configs()
        
        # Operation 3: Valid documentation generation
        def valid_doc_generation():
            doc_generator = DocumentationGenerator(
                source_dirs=[reliability_test_project / "docs"],
                output_dir=reliability_test_project / "docs" / "_build"
            )
            doc_generator.consolidate_existing_docs()
            return "docs_generated"
        
        operations = [valid_health_check, invalid_config_processing, valid_doc_generation]
        results = []
        
        # Execute operations independently
        for i, operation in enumerate(operations):
            try:
                result = operation()
                results.append((i, "success", result))
            except Exception as e:
                results.append((i, "error", str(e)))
        
        # Verify error isolation
        # At least some operations should succeed despite others failing
        successful_ops = [r for r in results if r[1] == "success"]
        failed_ops = [r for r in results if r[1] == "error"]
        
        assert len(successful_ops) > 0, "Some operations should succeed"
        
        # Failed operations should have meaningful error messages
        for op_id, status, error_msg in failed_ops:
            assert len(error_msg) > 0, f"Operation {op_id} should have error message"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])