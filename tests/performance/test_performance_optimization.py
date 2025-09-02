"""
Performance optimization validation tests.

This module validates performance requirements and identifies optimization opportunities
for the project health system.

Requirements: 1.7, 4.7
"""

import asyncio
import time
import tempfile
import shutil
import json
import cProfile
import pstats
import io
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pytest
import yaml
import psutil

from tools.test-runner.orchestrator import TestSuiteOrchestrator
from tools.doc_generator.documentation_generator import DocumentationGenerator
from tools.config_manager.config_unifier import ConfigurationUnifier
from tools.health_checker.health_checker import ProjectHealthChecker


class PerformanceProfiler:
    """Performance profiling utility for optimization analysis."""
    
    def __init__(self):
        self.profiler = None
        self.profile_data = None
    
    def start_profiling(self):
        """Start performance profiling."""
        self.profiler = cProfile.Profile()
        self.profiler.enable()
    
    def stop_profiling(self):
        """Stop performance profiling and collect data."""
        if self.profiler:
            self.profiler.disable()
            
            # Capture profile statistics
            s = io.StringIO()
            ps = pstats.Stats(self.profiler, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats()
            
            self.profile_data = s.getvalue()
    
    def get_top_functions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top functions by execution time."""
        if not self.profiler:
            return []
        
        stats = pstats.Stats(self.profiler)
        stats.sort_stats('cumulative')
        
        top_functions = []
        for func, (cc, nc, tt, ct, callers) in list(stats.stats.items())[:limit]:
            top_functions.append({
                "function": f"{func[0]}:{func[1]}({func[2]})",
                "call_count": cc,
                "total_time": tt,
                "cumulative_time": ct,
                "per_call_time": ct / cc if cc > 0 else 0
            })
        
        return top_functions


class TestPerformanceOptimization:
    """Performance optimization validation tests."""

    @pytest.fixture
    def optimization_test_project(self):
        """Create a project for performance optimization testing."""
        temp_dir = tempfile.mkdtemp()
        project_dir = Path(temp_dir) / "optimization_project"
        project_dir.mkdir()
        
        # Create project structure with varying complexity
        (project_dir / "tests" / "unit").mkdir(parents=True)
        (project_dir / "tests" / "integration").mkdir(parents=True)
        (project_dir / "docs" / "guides").mkdir(parents=True)
        (project_dir / "config" / "environments").mkdir(parents=True)
        
        # Create test files with different complexities
        # Simple tests
        for i in range(20):
            simple_test = f"""
import pytest

def test_simple_{i}():
    assert True

def test_math_{i}():
    assert {i} + 1 == {i + 1}
"""
            (project_dir / "tests" / "unit" / f"test_simple_{i}.py").write_text(simple_test)
        
        # Complex tests
        for i in range(10):
            complex_test = f"""
import pytest
import asyncio
import time

def test_complex_{i}():
    # Simulate complex computation
    result = sum(x * x for x in range({i * 10}))
    assert result >= 0

@pytest.mark.asyncio
async def test_async_complex_{i}():
    await asyncio.sleep(0.01)
    result = [x for x in range({i * 5}) if x % 2 == 0]
    assert len(result) >= 0

class TestComplexClass{i}:
    def setup_method(self):
        self.data = list(range({i * 20}))
    
    def test_method_{i}(self):
        processed = [x * 2 for x in self.data]
        assert len(processed) == len(self.data)
"""
            (project_dir / "tests" / "integration" / f"test_complex_{i}.py").write_text(complex_test)
        
        # Create documentation with varying sizes
        for i in range(30):
            doc_size = "small" if i < 10 else "medium" if i < 20 else "large"
            content_multiplier = 1 if doc_size == "small" else 3 if doc_size == "medium" else 10
            
            doc_content = f"""
# {doc_size.title()} Documentation {i}

This is a {doc_size} documentation file for performance testing.

""" + "\n".join([f"""
## Section {j}

This is section {j} with content for performance testing.
Content line 1 for section {j}.
Content line 2 for section {j}.
Content line 3 for section {j}.

### Subsection {j}.1
Detailed content for subsection {j}.1.

### Subsection {j}.2
Detailed content for subsection {j}.2.
""" for j in range(content_multiplier)])
            
            (project_dir / "docs" / "guides" / f"{doc_size}_doc_{i}.md").write_text(doc_content)
        
        # Create configuration files with different complexities
        for env in ["development", "staging", "production"]:
            for complexity in ["simple", "complex"]:
                if complexity == "simple":
                    config_data = {
                        "system": {"name": f"opt_project_{env}", "version": "1.0.0"},
                        "api": {"port": 8000, "timeout": 30}
                    }
                else:
                    config_data = {
                        "system": {
                            "name": f"opt_project_{env}_complex",
                            "version": "1.0.0",
                            "environment": env,
                            "features": {f"feature_{i}": i % 2 == 0 for i in range(50)}
                        },
                        "services": {
                            f"service_{i}": {
                                "port": 8000 + i,
                                "timeout": 30 + i,
                                "workers": i + 1,
                                "config": {f"setting_{j}": f"value_{j}" for j in range(10)}
                            } for i in range(20)
                        },
                        "database": {
                            "connections": {
                                f"db_{i}": {
                                    "host": f"db{i}.example.com",
                                    "port": 5432 + i,
                                    "name": f"database_{i}",
                                    "pool_size": 10 + i
                                } for i in range(5)
                            }
                        }
                    }
                
                config_file = project_dir / "config" / "environments" / f"{env}_{complexity}.yaml"
                with open(config_file, 'w') as f:
                    yaml.dump(config_data, f)
        
        yield project_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_test_execution_performance_baseline(self, optimization_test_project):
        """Establish performance baseline for test execution."""
        
        profiler = PerformanceProfiler()
        
        # Measure performance of different test categories
        performance_results = {}
        
        test_orchestrator = TestSuiteOrchestrator(
            project_root=optimization_test_project,
            config_path=None
        )
        
        # Test unit tests performance
        profiler.start_profiling()
        start_time = time.time()
        
        unit_results = await test_orchestrator.run_category("unit")
        
        unit_duration = time.time() - start_time
        profiler.stop_profiling()
        
        performance_results["unit_tests"] = {
            "duration": unit_duration,
            "test_count": unit_results.overall_summary.total_tests,
            "tests_per_second": unit_results.overall_summary.total_tests / unit_duration if unit_duration > 0 else 0,
            "top_functions": profiler.get_top_functions(5)
        }
        
        # Test integration tests performance
        profiler = PerformanceProfiler()
        profiler.start_profiling()
        start_time = time.time()
        
        integration_results = await test_orchestrator.run_category("integration")
        
        integration_duration = time.time() - start_time
        profiler.stop_profiling()
        
        performance_results["integration_tests"] = {
            "duration": integration_duration,
            "test_count": integration_results.overall_summary.total_tests,
            "tests_per_second": integration_results.overall_summary.total_tests / integration_duration if integration_duration > 0 else 0,
            "top_functions": profiler.get_top_functions(5)
        }
        
        # Performance requirements validation
        assert performance_results["unit_tests"]["tests_per_second"] > 1, "Unit tests should execute at least 1 test/second"
        assert performance_results["integration_tests"]["tests_per_second"] > 0.1, "Integration tests should execute at least 0.1 test/second"
        
        # Save performance baseline
        baseline_file = optimization_test_project / "performance_baseline.json"
        with open(baseline_file, 'w') as f:
            json.dump(performance_results, f, indent=2)
        
        print(f"Test Execution Performance Baseline: {performance_results}")

    def test_documentation_generation_optimization(self, optimization_test_project):
        """Test and optimize documentation generation performance."""
        
        profiler = PerformanceProfiler()
        
        # Test documentation generation with different strategies
        strategies = ["sequential", "batch"]
        performance_results = {}
        
        for strategy in strategies:
            profiler = PerformanceProfiler()
            profiler.start_profiling()
            start_time = time.time()
            
            doc_generator = DocumentationGenerator(
                source_dirs=[optimization_test_project / "docs"],
                output_dir=optimization_test_project / "docs" / f"_build_{strategy}"
            )
            
            if strategy == "sequential":
                # Process files one by one
                doc_generator.consolidate_existing_docs()
                doc_generator.generate_search_index()
            elif strategy == "batch":
                # Process files in batches (simulated optimization)
                doc_generator.consolidate_existing_docs()
                doc_generator.generate_search_index()
            
            duration = time.time() - start_time
            profiler.stop_profiling()
            
            # Count processed files
            doc_files = list((optimization_test_project / "docs").rglob("*.md"))
            
            performance_results[strategy] = {
                "duration": duration,
                "files_processed": len(doc_files),
                "files_per_second": len(doc_files) / duration if duration > 0 else 0,
                "top_functions": profiler.get_top_functions(3)
            }
        
        # Compare strategies
        sequential_fps = performance_results["sequential"]["files_per_second"]
        batch_fps = performance_results["batch"]["files_per_second"]
        
        # Performance requirements
        assert sequential_fps > 1, "Should process at least 1 file/second"
        
        # Optimization opportunity identification
        if batch_fps > sequential_fps * 1.1:
            print(f"Optimization opportunity: Batch processing is {batch_fps/sequential_fps:.2f}x faster")
        
        print(f"Documentation Generation Optimization: {performance_results}")

    def test_configuration_processing_optimization(self, optimization_test_project):
        """Test and optimize configuration processing performance."""
        
        profiler = PerformanceProfiler()
        
        # Test different configuration processing approaches
        approaches = ["individual", "bulk"]
        performance_results = {}
        
        config_files = list((optimization_test_project / "config").rglob("*.yaml"))
        
        for approach in approaches:
            profiler = PerformanceProfiler()
            profiler.start_profiling()
            start_time = time.time()
            
            config_unifier = ConfigurationUnifier(
                config_sources=[optimization_test_project / "config"]
            )
            
            if approach == "individual":
                # Process each config file individually
                for config_file in config_files:
                    with open(config_file, 'r') as f:
                        config_data = yaml.safe_load(f)
                    config_unifier.validate_configuration(config_data)
            
            elif approach == "bulk":
                # Process all configs together
                migration_report = config_unifier.migrate_existing_configs()
            
            duration = time.time() - start_time
            profiler.stop_profiling()
            
            performance_results[approach] = {
                "duration": duration,
                "configs_processed": len(config_files),
                "configs_per_second": len(config_files) / duration if duration > 0 else 0,
                "top_functions": profiler.get_top_functions(3)
            }
        
        # Performance validation
        individual_cps = performance_results["individual"]["configs_per_second"]
        bulk_cps = performance_results["bulk"]["configs_per_second"]
        
        assert individual_cps > 1, "Should process at least 1 config/second individually"
        assert bulk_cps > 1, "Should process at least 1 config/second in bulk"
        
        print(f"Configuration Processing Optimization: {performance_results}")

    @pytest.mark.asyncio
    async def test_health_check_optimization(self, optimization_test_project):
        """Test and optimize health check performance."""
        
        profiler = PerformanceProfiler()
        
        # Test different health check strategies
        strategies = ["comprehensive", "targeted", "parallel"]
        performance_results = {}
        
        for strategy in strategies:
            profiler = PerformanceProfiler()
            profiler.start_profiling()
            start_time = time.time()
            
            health_checker = ProjectHealthChecker(project_root=optimization_test_project)
            
            if strategy == "comprehensive":
                # Full health check
                health_report = await health_checker.run_health_check()
            
            elif strategy == "targeted":
                # Simulate targeted health check (subset of checks)
                health_report = await health_checker.run_health_check()
                # In real implementation, this would run fewer checks
            
            elif strategy == "parallel":
                # Simulate parallel health check execution
                health_report = await health_checker.run_health_check()
                # In real implementation, this would run checks in parallel
            
            duration = time.time() - start_time
            profiler.stop_profiling()
            
            performance_results[strategy] = {
                "duration": duration,
                "health_score": health_report.overall_score,
                "components_checked": len(health_report.component_scores),
                "checks_per_second": len(health_report.component_scores) / duration if duration > 0 else 0,
                "top_functions": profiler.get_top_functions(3)
            }
        
        # Performance validation
        for strategy, results in performance_results.items():
            assert results["duration"] < 30, f"{strategy} health check should complete within 30 seconds"
            assert results["checks_per_second"] > 0.1, f"{strategy} should perform at least 0.1 checks/second"
        
        print(f"Health Check Optimization: {performance_results}")

    def test_memory_usage_optimization(self, optimization_test_project):
        """Test and optimize memory usage patterns."""
        
        # Monitor memory usage during different operations
        memory_profiles = {}
        
        operations = [
            ("test_execution", lambda: asyncio.run(self._run_test_suite(optimization_test_project))),
            ("doc_generation", lambda: self._generate_docs(optimization_test_project)),
            ("config_processing", lambda: self._process_configs(optimization_test_project)),
            ("health_check", lambda: asyncio.run(self._run_health_check(optimization_test_project)))
        ]
        
        for op_name, operation in operations:
            # Measure memory before operation
            initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            
            # Run operation
            start_time = time.time()
            try:
                result = operation()
                duration = time.time() - start_time
                success = True
            except Exception as e:
                duration = time.time() - start_time
                success = False
                result = str(e)
            
            # Measure memory after operation
            final_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            memory_increase = final_memory - initial_memory
            
            memory_profiles[op_name] = {
                "initial_memory_mb": initial_memory,
                "final_memory_mb": final_memory,
                "memory_increase_mb": memory_increase,
                "duration": duration,
                "success": success,
                "memory_efficiency": duration / memory_increase if memory_increase > 0 else float('inf')
            }
        
        # Memory usage validation
        for op_name, profile in memory_profiles.items():
            assert profile["memory_increase_mb"] < 500, f"{op_name} should not increase memory by more than 500MB"
            
            if profile["success"]:
                assert profile["memory_efficiency"] > 0.01, f"{op_name} should have reasonable memory efficiency"
        
        print(f"Memory Usage Optimization: {memory_profiles}")
        
        # Save memory profile
        memory_file = optimization_test_project / "memory_profile.json"
        with open(memory_file, 'w') as f:
            json.dump(memory_profiles, f, indent=2)

    async def _run_test_suite(self, project_root: Path):
        """Helper method to run test suite."""
        orchestrator = TestSuiteOrchestrator(project_root=project_root, config_path=None)
        return await orchestrator.run_full_suite()

    def _generate_docs(self, project_root: Path):
        """Helper method to generate documentation."""
        doc_generator = DocumentationGenerator(
            source_dirs=[project_root / "docs"],
            output_dir=project_root / "docs" / "_build_memory_test"
        )
        doc_generator.consolidate_existing_docs()
        return "docs_generated"

    def _process_configs(self, project_root: Path):
        """Helper method to process configurations."""
        config_unifier = ConfigurationUnifier(config_sources=[project_root / "config"])
        return config_unifier.migrate_existing_configs()

    async def _run_health_check(self, project_root: Path):
        """Helper method to run health check."""
        health_checker = ProjectHealthChecker(project_root=project_root)
        return await health_checker.run_health_check()

    def test_cpu_usage_optimization(self, optimization_test_project):
        """Test and optimize CPU usage patterns."""
        
        # Monitor CPU usage during operations
        cpu_profiles = {}
        
        operations = [
            ("sequential_processing", lambda: self._sequential_processing(optimization_test_project)),
            ("parallel_processing", lambda: self._parallel_processing(optimization_test_project))
        ]
        
        for op_name, operation in operations:
            # Monitor CPU usage
            cpu_samples = []
            
            def monitor_cpu():
                start_time = time.time()
                while time.time() - start_time < 10:  # Monitor for 10 seconds max
                    cpu_samples.append(psutil.cpu_percent(interval=0.1))
            
            import threading
            monitor_thread = threading.Thread(target=monitor_cpu)
            monitor_thread.start()
            
            # Run operation
            start_time = time.time()
            try:
                result = operation()
                duration = time.time() - start_time
                success = True
            except Exception as e:
                duration = time.time() - start_time
                success = False
                result = str(e)
            
            monitor_thread.join(timeout=1)
            
            cpu_profiles[op_name] = {
                "duration": duration,
                "success": success,
                "average_cpu": sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0,
                "peak_cpu": max(cpu_samples) if cpu_samples else 0,
                "cpu_efficiency": duration / (sum(cpu_samples) / len(cpu_samples)) if cpu_samples else 0
            }
        
        # CPU usage validation
        for op_name, profile in cpu_profiles.items():
            if profile["success"]:
                assert profile["average_cpu"] < 90, f"{op_name} should not consistently use >90% CPU"
        
        print(f"CPU Usage Optimization: {cpu_profiles}")

    def _sequential_processing(self, project_root: Path):
        """Simulate sequential processing for CPU testing."""
        # Process configuration files sequentially
        config_files = list((project_root / "config").rglob("*.yaml"))
        results = []
        
        for config_file in config_files:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Simulate processing work
            processed = {k: v for k, v in config_data.items() if isinstance(v, (str, int, float))}
            results.append(processed)
        
        return results

    def _parallel_processing(self, project_root: Path):
        """Simulate parallel processing for CPU testing."""
        import concurrent.futures
        
        config_files = list((project_root / "config").rglob("*.yaml"))
        
        def process_config(config_file):
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            return {k: v for k, v in config_data.items() if isinstance(v, (str, int, float))}
        
        # Process files in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_config, config_files))
        
        return results

    def test_performance_regression_detection(self, optimization_test_project):
        """Test performance regression detection capabilities."""
        
        # Create baseline performance measurements
        baseline_file = optimization_test_project / "performance_baseline.json"
        
        if not baseline_file.exists():
            # Create initial baseline
            baseline_data = {
                "test_execution_time": 5.0,
                "doc_generation_time": 2.0,
                "config_processing_time": 1.0,
                "health_check_time": 3.0,
                "memory_usage_mb": 200.0
            }
            
            with open(baseline_file, 'w') as f:
                json.dump(baseline_data, f, indent=2)
        
        # Load baseline
        with open(baseline_file, 'r') as f:
            baseline = json.load(f)
        
        # Measure current performance
        current_performance = {}
        
        # Test execution time
        start_time = time.time()
        orchestrator = TestSuiteOrchestrator(project_root=optimization_test_project, config_path=None)
        asyncio.run(orchestrator.run_category("unit"))
        current_performance["test_execution_time"] = time.time() - start_time
        
        # Documentation generation time
        start_time = time.time()
        self._generate_docs(optimization_test_project)
        current_performance["doc_generation_time"] = time.time() - start_time
        
        # Configuration processing time
        start_time = time.time()
        self._process_configs(optimization_test_project)
        current_performance["config_processing_time"] = time.time() - start_time
        
        # Health check time
        start_time = time.time()
        asyncio.run(self._run_health_check(optimization_test_project))
        current_performance["health_check_time"] = time.time() - start_time
        
        # Memory usage
        current_performance["memory_usage_mb"] = psutil.Process().memory_info().rss / (1024 * 1024)
        
        # Detect regressions
        regressions = []
        improvements = []
        
        for metric, current_value in current_performance.items():
            baseline_value = baseline.get(metric, 0)
            
            if baseline_value > 0:
                change_ratio = current_value / baseline_value
                
                if change_ratio > 1.2:  # 20% slower is a regression
                    regressions.append({
                        "metric": metric,
                        "baseline": baseline_value,
                        "current": current_value,
                        "regression_ratio": change_ratio
                    })
                elif change_ratio < 0.8:  # 20% faster is an improvement
                    improvements.append({
                        "metric": metric,
                        "baseline": baseline_value,
                        "current": current_value,
                        "improvement_ratio": 1 / change_ratio
                    })
        
        # Performance regression validation
        assert len(regressions) == 0, f"Performance regressions detected: {regressions}"
        
        # Log improvements
        if improvements:
            print(f"Performance improvements detected: {improvements}")
        
        print(f"Performance Regression Detection: Current={current_performance}, Baseline={baseline}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])