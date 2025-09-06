import pytest
#!/usr/bin/env python3
"""
Test Runner Engine

Advanced test execution engine with isolation, timeout handling, and comprehensive reporting.
Supports parallel execution, retry logic, and detailed performance profiling.
"""

import asyncio
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import threading
import queue


@dataclass
class TestExecutionResult:
    """Result of executing a single test file"""
    test_file: str
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float
    timeout_occurred: bool
    retry_count: int
    test_counts: Dict[str, int]  # passed, failed, skipped, errors
    coverage_data: Optional[Dict[str, Any]] = None


@dataclass
class TestSuiteExecutionReport:
    """Complete execution report for test suite"""
    total_files: int
    successful_files: int
    failed_files: int
    total_execution_time: float
    file_results: List[TestExecutionResult]
    performance_summary: Dict[str, Any]
    retry_summary: Dict[str, int]
    timeout_summary: Dict[str, int]


class TestIsolationManager:
    """Manages test isolation and cleanup"""
    
    def __init__(self):
        self.temp_dirs = []
        self.processes = []
    
    def create_isolated_environment(self) -> Path:
        """Create isolated temporary directory for test execution"""
        temp_dir = Path(tempfile.mkdtemp(prefix='test_isolation_'))
        self.temp_dirs.append(temp_dir)
        return temp_dir
    
    def cleanup_environment(self, temp_dir: Path):
        """Clean up isolated environment"""
        try:
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            if temp_dir in self.temp_dirs:
                self.temp_dirs.remove(temp_dir)
        except Exception as e:
            print(f"Warning: Failed to cleanup {temp_dir}: {e}")
    
    def cleanup_all(self):
        """Clean up all created environments"""
        for temp_dir in self.temp_dirs[:]:
            self.cleanup_environment(temp_dir)
        
        # Terminate any remaining processes
        for process in self.processes:
            try:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass


class TestTimeoutManager:
    """Manages test timeouts with configurable limits"""
    
    def __init__(self):
        self.default_timeout = 30  # seconds
        self.timeout_overrides = {
            'integration': 60,
            'e2e': 120,
            'performance': 300,
            'stress': 600
        }
    
    def get_timeout_for_file(self, test_file: Path) -> int:
        """Get appropriate timeout for test file based on its type"""
        file_str = str(test_file).lower()
        
        for test_type, timeout in self.timeout_overrides.items():
            if test_type in file_str:
                return timeout
        
        return self.default_timeout
    
    def set_timeout_override(self, pattern: str, timeout: int):
        """Set custom timeout for files matching pattern"""
        self.timeout_overrides[pattern] = timeout


class TestRetryManager:
    """Manages test retry logic for flaky tests"""
    
    def __init__(self):
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
        self.exponential_backoff = True
        self.flaky_test_patterns = [
            'network', 'async', 'timing', 'race', 'concurrent'
        ]
    
    def should_retry(self, test_file: Path, attempt: int, result: TestExecutionResult) -> bool:
        """Determine if test should be retried"""
        if attempt >= self.max_retries:
            return False
        
        # Don't retry syntax errors or import errors
        if result.exit_code == 2:  # Pytest collection error
            return False
        
        # Retry timeouts and intermittent failures
        if result.timeout_occurred:
            return True
        
        # Retry if file matches flaky patterns
        file_str = str(test_file).lower()
        if any(pattern in file_str for pattern in self.flaky_test_patterns):
            return True
        
        # Retry if exit code suggests intermittent failure
        if result.exit_code in [1, 3, 4]:  # Test failures, interrupted, internal error
            return True
        
        return False
    
    def get_retry_delay(self, attempt: int) -> float:
        """Get delay before retry"""
        if self.exponential_backoff:
            return self.retry_delay * (2 ** attempt)
        return self.retry_delay


class TestCoverageCollector:
    """Collects test coverage information"""
    
    def __init__(self):
        self.coverage_enabled = True
        self.coverage_config = None
    
    def setup_coverage(self, test_file: Path, temp_dir: Path) -> List[str]:
        """Setup coverage collection for test execution"""
        if not self.coverage_enabled:
            return []
        
        coverage_file = temp_dir / '.coverage'
        
        return [
            '--cov=.',
            f'--cov-report=json:{temp_dir}/coverage.json',
            '--cov-report=term-missing',
            '--cov-fail-under=0'  # Don't fail on low coverage during audit
        ]
    
    def collect_coverage_data(self, temp_dir: Path) -> Optional[Dict[str, Any]]:
        """Collect coverage data from test execution"""
        coverage_file = temp_dir / 'coverage.json'
        
        if coverage_file.exists():
            try:
                with open(coverage_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to read coverage data: {e}")
        
        return None


class TestExecutor:
    """Executes individual test files with full isolation and monitoring"""
    
    def __init__(self):
        self.isolation_manager = TestIsolationManager()
        self.timeout_manager = TestTimeoutManager()
        self.retry_manager = TestRetryManager()
        self.coverage_collector = TestCoverageCollector()
    
    def execute_test_file(self, test_file: Path, project_root: Path) -> TestExecutionResult:
        """Execute a single test file with full monitoring"""
        attempt = 0
        last_result = None
        
        while attempt <= self.retry_manager.max_retries:
            try:
                result = self._execute_single_attempt(test_file, project_root, attempt)
                
                if result.success or not self.retry_manager.should_retry(test_file, attempt, result):
                    result.retry_count = attempt
                    return result
                
                last_result = result
                attempt += 1
                
                if attempt <= self.retry_manager.max_retries:
                    delay = self.retry_manager.get_retry_delay(attempt)
                    time.sleep(delay)
                    
            except Exception as e:
                # Create error result
                result = TestExecutionResult(
                    test_file=str(test_file),
                    success=False,
                    exit_code=-1,
                    stdout="",
                    stderr=f"Execution error: {str(e)}",
                    execution_time=0.0,
                    timeout_occurred=False,
                    retry_count=attempt,
                    test_counts={'passed': 0, 'failed': 0, 'skipped': 0, 'errors': 1}
                )
                
                if not self.retry_manager.should_retry(test_file, attempt, result):
                    return result
                
                attempt += 1
        
        # Return last result if all retries exhausted
        if last_result:
            last_result.retry_count = attempt - 1
            return last_result
        
        # Fallback error result
        return TestExecutionResult(
            test_file=str(test_file),
            success=False,
            exit_code=-1,
            stdout="",
            stderr="All retry attempts failed",
            execution_time=0.0,
            timeout_occurred=False,
            retry_count=attempt - 1,
            test_counts={'passed': 0, 'failed': 0, 'skipped': 0, 'errors': 1}
        )
    
    def _execute_single_attempt(self, test_file: Path, project_root: Path, attempt: int) -> TestExecutionResult:
        """Execute single test attempt"""
        # Create isolated environment
        temp_dir = self.isolation_manager.create_isolated_environment()
        
        try:
            # Setup test execution
            timeout = self.timeout_manager.get_timeout_for_file(test_file)
            coverage_args = self.coverage_collector.setup_coverage(test_file, temp_dir)
            
            # Build command
            cmd = [
                sys.executable, '-m', 'pytest',
                str(test_file),
                '-v',
                '--tb=short',
                '--no-header',
                '--json-report',
                f'--json-report-file={temp_dir}/report.json'
            ] + coverage_args
            
            # Set environment variables
            env = os.environ.copy()
            env['PYTHONPATH'] = str(project_root)
            env['PYTEST_CURRENT_TEST'] = str(test_file)
            
            # Execute test
            start_time = time.time()
            timeout_occurred = False
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=project_root,
                    env=env
                )
                execution_time = time.time() - start_time
                
            except subprocess.TimeoutExpired as e:
                execution_time = timeout
                timeout_occurred = True
                result = subprocess.CompletedProcess(
                    cmd, -1, 
                    stdout=e.stdout or "", 
                    stderr=f"Test timed out after {timeout}s"
                )
            
            # Parse test results
            test_counts = self._parse_test_counts(result.stdout, temp_dir)
            
            # Collect coverage data
            coverage_data = self.coverage_collector.collect_coverage_data(temp_dir)
            
            # Determine success
            success = result.returncode == 0 and not timeout_occurred
            
            return TestExecutionResult(
                test_file=str(test_file),
                success=success,
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                execution_time=execution_time,
                timeout_occurred=timeout_occurred,
                retry_count=0,  # Will be set by caller
                test_counts=test_counts,
                coverage_data=coverage_data
            )
            
        finally:
            # Cleanup isolated environment
            self.isolation_manager.cleanup_environment(temp_dir)
    
    def _parse_test_counts(self, stdout: str, temp_dir: Path) -> Dict[str, int]:
        """Parse test counts from pytest output"""
        counts = {'passed': 0, 'failed': 0, 'skipped': 0, 'errors': 0}
        
        # Try to read from JSON report first
        json_report = temp_dir / 'report.json'
        if json_report.exists():
            try:
                with open(json_report, 'r') as f:
                    data = json.load(f)
                
                summary = data.get('summary', {})
                counts['passed'] = summary.get('passed', 0)
                counts['failed'] = summary.get('failed', 0)
                counts['skipped'] = summary.get('skipped', 0)
                counts['errors'] = summary.get('error', 0)
                
                return counts
            except Exception:
                pass
        
        # Fallback to parsing stdout
        import re

        patterns = {
            'passed': r'(\d+) passed',
            'failed': r'(\d+) failed',
            'skipped': r'(\d+) skipped',
            'errors': r'(\d+) error'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, stdout)
            if match:
                counts[key] = int(match.group(1))
        
        return counts


class ParallelTestRunner:
    """Runs tests in parallel with resource management"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(multiprocessing.cpu_count(), 8)
        self.executor = TestExecutor()
    
    def run_tests_parallel(self, test_files: List[Path], project_root: Path) -> TestSuiteExecutionReport:
        """Run tests in parallel"""
        start_time = time.time()
        results = []
        
        print(f"Running {len(test_files)} test files with {self.max_workers} workers...")
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all test files
            future_to_file = {
                executor.submit(self._run_single_test, test_file, project_root): test_file
                for test_file in test_files
            }
            
            # Collect results as they complete
            for i, future in enumerate(as_completed(future_to_file), 1):
                test_file = future_to_file[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    status = "✓" if result.success else "✗"
                    print(f"[{i}/{len(test_files)}] {status} {test_file.name} ({result.execution_time:.2f}s)")
                    
                except Exception as e:
                    print(f"[{i}/{len(test_files)}] ✗ {test_file.name} (error: {e})")
                    
                    # Create error result
                    error_result = TestExecutionResult(
                        test_file=str(test_file),
                        success=False,
                        exit_code=-1,
                        stdout="",
                        stderr=f"Execution error: {str(e)}",
                        execution_time=0.0,
                        timeout_occurred=False,
                        retry_count=0,
                        test_counts={'passed': 0, 'failed': 0, 'skipped': 0, 'errors': 1}
                    )
                    results.append(error_result)
        
        total_time = time.time() - start_time
        
        # Generate summary
        successful_files = len([r for r in results if r.success])
        failed_files = len(results) - successful_files
        
        performance_summary = self._generate_performance_summary(results)
        retry_summary = self._generate_retry_summary(results)
        timeout_summary = self._generate_timeout_summary(results)
        
        return TestSuiteExecutionReport(
            total_files=len(test_files),
            successful_files=successful_files,
            failed_files=failed_files,
            total_execution_time=total_time,
            file_results=results,
            performance_summary=performance_summary,
            retry_summary=retry_summary,
            timeout_summary=timeout_summary
        )
    
    def _run_single_test(self, test_file: Path, project_root: Path) -> TestExecutionResult:
        """Run single test (for process pool)"""
        executor = TestExecutor()  # Create new executor for each process
        return executor.execute_test_file(test_file, project_root)
    
    def _generate_performance_summary(self, results: List[TestExecutionResult]) -> Dict[str, Any]:
        """Generate performance summary"""
        execution_times = [r.execution_time for r in results]
        
        if not execution_times:
            return {}
        
        return {
            'total_time': sum(execution_times),
            'average_time': sum(execution_times) / len(execution_times),
            'min_time': min(execution_times),
            'max_time': max(execution_times),
            'slowest_files': [
                {'file': r.test_file, 'time': r.execution_time}
                for r in sorted(results, key=lambda x: x.execution_time, reverse=True)[:10]
            ]
        }
    
    def _generate_retry_summary(self, results: List[TestExecutionResult]) -> Dict[str, int]:
        """Generate retry summary"""
        retry_counts = {}
        
        for result in results:
            retry_count = result.retry_count
            retry_counts[f'retry_{retry_count}'] = retry_counts.get(f'retry_{retry_count}', 0) + 1
        
        return retry_counts
    
    def _generate_timeout_summary(self, results: List[TestExecutionResult]) -> Dict[str, int]:
        """Generate timeout summary"""
        timeout_files = [r for r in results if r.timeout_occurred]
        
        return {
            'total_timeouts': len(timeout_files),
            'timeout_files': [r.test_file for r in timeout_files]
        }


def run_test_file_isolated(test_file_path: str, project_root_path: str) -> dict:
    """Standalone function for running a single test file (for multiprocessing)"""
    test_file = Path(test_file_path)
    project_root = Path(project_root_path)
    
    executor = TestExecutor()
    result = executor.execute_test_file(test_file, project_root)
    
    return asdict(result)


def main():
    """Main entry point for test runner"""
    import argparse

    parser = argparse.ArgumentParser(description="Advanced test runner with isolation and monitoring")
    parser.add_argument('test_files', nargs='*', help='Test files to run (default: discover all)')
    parser.add_argument('--project-root', type=Path, default=Path.cwd(), help='Project root directory')
    parser.add_argument('--parallel', type=int, help='Number of parallel workers')
    parser.add_argument('--timeout', type=int, help='Default timeout in seconds')
    parser.add_argument('--output', type=Path, help='Output file for results')
    
    args = parser.parse_args()
    
    # Setup runner
    runner = ParallelTestRunner(max_workers=args.parallel)
    
    if args.timeout:
        runner.executor.timeout_manager.default_timeout = args.timeout
    
    # Discover test files if not specified
    if args.test_files:
        test_files = [Path(f) for f in args.test_files]
    else:
        from test_auditor import TestDiscoveryEngine
        discovery = TestDiscoveryEngine(args.project_root)
        test_files = discovery.discover_test_files()
    
    # Run tests
    report = runner.run_tests_parallel(test_files, args.project_root)
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        print(f"Results saved to {args.output}")
    
    # Print summary
    print(f"\nExecution Summary:")
    print(f"Total files: {report.total_files}")
    print(f"Successful: {report.successful_files}")
    print(f"Failed: {report.failed_files}")
    print(f"Total time: {report.total_execution_time:.2f}s")
    
    return 0 if report.failed_files == 0 else 1


if __name__ == '__main__':
    sys.exit(main())