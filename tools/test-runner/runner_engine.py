"""
Test Runner Engine - Core test execution with timeout handling and discovery
"""

import asyncio
import logging
import subprocess
import time
import signal
import os
import glob
from pathlib import Path
from typing import List, Dict, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue
import sys

from orchestrator import TestCategory, TestStatus, TestDetail, TestConfig

logger = logging.getLogger(__name__)


class TestDiscoveryMethod(Enum):
    """Methods for discovering tests"""
    PYTEST = "pytest"
    UNITTEST = "unittest"
    CUSTOM = "custom"


@dataclass
class TestExecutionContext:
    """Context for test execution"""
    category: TestCategory
    test_files: List[Path]
    timeout: int
    parallel: bool
    environment: Dict[str, str] = field(default_factory=dict)
    working_directory: Optional[Path] = None


@dataclass
class ExecutionProgress:
    """Progress tracking for test execution"""
    total_tests: int
    completed_tests: int
    current_test: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    
    @property
    def progress_percentage(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return (self.completed_tests / self.total_tests) * 100
    
    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time


class ProgressMonitor:
    """Monitors and reports test execution progress"""
    
    def __init__(self):
        self.progress_callbacks: List[Callable[[ExecutionProgress], None]] = []
        self.current_progress: Optional[ExecutionProgress] = None
    
    def add_progress_callback(self, callback: Callable[[ExecutionProgress], None]):
        """Add a callback to be called on progress updates"""
        self.progress_callbacks.append(callback)
    
    def update_progress(self, progress: ExecutionProgress):
        """Update current progress and notify callbacks"""
        self.current_progress = progress
        for callback in self.progress_callbacks:
            try:
                callback(progress)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
    
    def start_execution(self, total_tests: int):
        """Start tracking execution progress"""
        self.current_progress = ExecutionProgress(
            total_tests=total_tests,
            completed_tests=0
        )
        self.update_progress(self.current_progress)
    
    def complete_test(self, test_name: str):
        """Mark a test as completed"""
        if self.current_progress:
            self.current_progress.completed_tests += 1
            self.current_progress.current_test = test_name
            self.update_progress(self.current_progress)


class TestDiscovery:
    """Discovers and categorizes tests based on patterns and file structure"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.test_root = Path("tests")
    
    def discover_tests(self, category: TestCategory) -> List[Path]:
        """
        Discover test files for a specific category
        
        Args:
            category: Test category to discover tests for
            
        Returns:
            List of test file paths
        """
        category_config = self.config.categories.get(category.value, {})
        patterns = category_config.get('patterns', [])
        
        if not patterns:
            # Default patterns based on category
            patterns = self._get_default_patterns(category)
        
        discovered_files = []
        
        for pattern in patterns:
            # Convert relative patterns to absolute paths
            if not pattern.startswith('/'):
                pattern = str(self.test_root / pattern)
            
            # Use glob to find matching files
            matching_files = glob.glob(pattern, recursive=True)
            discovered_files.extend([Path(f) for f in matching_files if Path(f).is_file()])
        
        # Remove duplicates and filter valid Python test files
        unique_files = list(set(discovered_files))
        valid_files = [f for f in unique_files if self._is_valid_test_file(f)]
        
        logger.info(f"Discovered {len(valid_files)} test files for category {category.value}")
        return valid_files
    
    def _get_default_patterns(self, category: TestCategory) -> List[str]:
        """Get default file patterns for a category"""
        patterns = {
            TestCategory.UNIT: ["tests/unit/test_*.py"],
            TestCategory.INTEGRATION: ["tests/integration/test_*_integration.py"],
            TestCategory.PERFORMANCE: ["tests/performance/test_*_performance.py", 
                                     "tests/performance/test_*_benchmark.py"],
            TestCategory.E2E: ["tests/e2e/test_*_e2e.py", "tests/e2e/test_*_workflow.py"]
        }
        return patterns.get(category, [])
    
    def _is_valid_test_file(self, file_path: Path) -> bool:
        """Check if a file is a valid Python test file"""
        if not file_path.suffix == '.py':
            return False
        
        if not file_path.exists():
            return False
        
        # Check if file contains test functions/classes
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Simple heuristic: file should contain test functions or classes
                return ('def test_' in content or 
                       'class Test' in content or 
                       'import pytest' in content or
                       'import unittest' in content)
        except Exception as e:
            logger.warning(f"Could not read test file {file_path}: {e}")
            return False
    
    def categorize_test_file(self, file_path: Path) -> Optional[TestCategory]:
        """Automatically categorize a test file based on its path and content"""
        path_str = str(file_path).lower()
        
        # Path-based categorization
        if '/unit/' in path_str or 'test_unit_' in path_str:
            return TestCategory.UNIT
        elif '/integration/' in path_str or '_integration' in path_str:
            return TestCategory.INTEGRATION
        elif '/performance/' in path_str or '_performance' in path_str or '_benchmark' in path_str:
            return TestCategory.PERFORMANCE
        elif '/e2e/' in path_str or '_e2e' in path_str or '_workflow' in path_str:
            return TestCategory.E2E
        
        # Content-based categorization (basic heuristics)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().lower()
                
                if 'pytest.mark.benchmark' in content or 'benchmark' in content:
                    return TestCategory.PERFORMANCE
                elif 'integration' in content or 'end_to_end' in content:
                    return TestCategory.INTEGRATION
                elif 'mock' in content or 'unittest.mock' in content:
                    return TestCategory.UNIT
                
        except Exception:
            pass
        
        # Default to unit tests
        return TestCategory.UNIT


class TimeoutManager:
    """Manages test execution timeouts with graceful handling"""
    
    def __init__(self):
        self.active_processes: Dict[int, subprocess.Popen] = {}
        self.timeout_handlers: Dict[int, threading.Timer] = {}
    
    def execute_with_timeout(self, command: List[str], timeout: int, 
                           cwd: Optional[Path] = None, 
                           env: Optional[Dict[str, str]] = None) -> tuple[int, str, str]:
        """
        Execute command with timeout handling
        
        Args:
            command: Command to execute
            timeout: Timeout in seconds
            cwd: Working directory
            env: Environment variables
            
        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        logger.debug(f"Executing command with {timeout}s timeout: {' '.join(command)}")
        
        # Prepare environment
        exec_env = os.environ.copy()
        if env:
            exec_env.update(env)
        
        try:
            # Start process
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=cwd,
                env=exec_env,
                preexec_fn=os.setsid if os.name != 'nt' else None
            )
            
            self.active_processes[process.pid] = process
            
            # Set up timeout handler
            timeout_occurred = threading.Event()
            
            def timeout_handler():
                timeout_occurred.set()
                self._terminate_process(process)
            
            timer = threading.Timer(timeout, timeout_handler)
            self.timeout_handlers[process.pid] = timer
            timer.start()
            
            try:
                # Wait for process completion
                stdout, stderr = process.communicate()
                timer.cancel()
                
                if timeout_occurred.is_set():
                    return -1, stdout, f"Process timed out after {timeout} seconds\n{stderr}"
                
                return process.returncode, stdout, stderr
                
            finally:
                # Clean up
                if process.pid in self.active_processes:
                    del self.active_processes[process.pid]
                if process.pid in self.timeout_handlers:
                    del self.timeout_handlers[process.pid]
                
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            return -1, "", str(e)
    
    def _terminate_process(self, process: subprocess.Popen):
        """Gracefully terminate a process"""
        try:
            if os.name == 'nt':  # Windows
                process.terminate()
            else:  # Unix-like
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                
            # Give process time to terminate gracefully
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if graceful termination failed
                if os.name == 'nt':
                    process.kill()
                else:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    
        except Exception as e:
            logger.warning(f"Error terminating process {process.pid}: {e}")


class TestRunnerEngine:
    """
    Core test execution engine with timeout handling and progress monitoring
    """
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.discovery = TestDiscovery(config)
        self.timeout_manager = TimeoutManager()
        self.progress_monitor = ProgressMonitor()
        
        # Default test runners
        self.test_runners = {
            TestDiscoveryMethod.PYTEST: self._run_pytest_tests,
            TestDiscoveryMethod.UNITTEST: self._run_unittest_tests
        }
    
    async def execute_category_tests(self, context: TestExecutionContext) -> List[TestDetail]:
        """
        Execute tests for a category with full monitoring and timeout handling
        
        Args:
            context: Test execution context
            
        Returns:
            List of test execution details
        """
        logger.info(f"Executing tests for category {context.category.value}")
        
        # Discover tests if not provided
        if not context.test_files:
            context.test_files = self.discovery.discover_tests(context.category)
        
        if not context.test_files:
            logger.warning(f"No test files found for category {context.category.value}")
            return []
        
        # Start progress monitoring
        total_tests = len(context.test_files)
        self.progress_monitor.start_execution(total_tests)
        
        test_results = []
        
        if context.parallel and len(context.test_files) > 1:
            # Run tests in parallel
            test_results = await self._execute_tests_parallel(context)
        else:
            # Run tests sequentially
            test_results = await self._execute_tests_sequential(context)
        
        logger.info(f"Completed execution of {len(test_results)} tests for {context.category.value}")
        return test_results
    
    async def _execute_tests_sequential(self, context: TestExecutionContext) -> List[TestDetail]:
        """Execute tests sequentially"""
        results = []
        
        for test_file in context.test_files:
            result = await self._execute_single_test_file(test_file, context)
            results.extend(result)
            
            # Update progress
            self.progress_monitor.complete_test(str(test_file))
        
        return results
    
    async def _execute_tests_parallel(self, context: TestExecutionContext) -> List[TestDetail]:
        """Execute tests in parallel using asyncio"""
        tasks = []
        
        for test_file in context.test_files:
            task = asyncio.create_task(self._execute_single_test_file(test_file, context))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results_lists = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results and handle exceptions
        all_results = []
        for i, result in enumerate(results_lists):
            if isinstance(result, Exception):
                logger.error(f"Test file {context.test_files[i]} failed with exception: {result}")
                all_results.append(TestDetail(
                    name=str(context.test_files[i]),
                    status=TestStatus.ERROR,
                    duration=0,
                    error_message=str(result),
                    category=context.category
                ))
            else:
                all_results.extend(result)
        
        return all_results
    
    async def _execute_single_test_file(self, test_file: Path, context: TestExecutionContext) -> List[TestDetail]:
        """Execute a single test file and return detailed results"""
        logger.debug(f"Executing test file: {test_file}")
        
        start_time = time.time()
        
        try:
            # Determine test runner method
            runner_method = self._determine_test_runner(test_file)
            
            # Execute tests
            if runner_method in self.test_runners:
                return await self.test_runners[runner_method](test_file, context)
            else:
                # Fallback to pytest
                return await self._run_pytest_tests(test_file, context)
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Error executing test file {test_file}: {e}")
            
            return [TestDetail(
                name=str(test_file),
                status=TestStatus.ERROR,
                duration=duration,
                error_message=str(e),
                category=context.category
            )]
    
    def _determine_test_runner(self, test_file: Path) -> TestDiscoveryMethod:
        """Determine the appropriate test runner for a file"""
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                if 'import pytest' in content or 'pytest.mark' in content:
                    return TestDiscoveryMethod.PYTEST
                elif 'import unittest' in content or 'unittest.TestCase' in content:
                    return TestDiscoveryMethod.UNITTEST
                    
        except Exception:
            pass
        
        # Default to pytest
        return TestDiscoveryMethod.PYTEST
    
    async def _run_pytest_tests(self, test_file: Path, context: TestExecutionContext) -> List[TestDetail]:
        """Run tests using pytest"""
        command = [
            sys.executable, '-m', 'pytest',
            str(test_file),
            '-v',  # Verbose output
            '--tb=short',  # Short traceback format
            '--json-report',  # JSON output for parsing
            '--json-report-file=/tmp/pytest_report.json'
        ]
        
        # Add category-specific options
        if context.category == TestCategory.PERFORMANCE:
            command.extend(['--benchmark-only', '--benchmark-json=/tmp/benchmark.json'])
        
        return_code, stdout, stderr = self.timeout_manager.execute_with_timeout(
            command=command,
            timeout=context.timeout,
            cwd=context.working_directory,
            env=context.environment
        )
        
        return self._parse_pytest_output(stdout, stderr, return_code, test_file, context)
    
    async def _run_unittest_tests(self, test_file: Path, context: TestExecutionContext) -> List[TestDetail]:
        """Run tests using unittest"""
        # Convert file path to module path
        module_path = str(test_file).replace('/', '.').replace('\\', '.').replace('.py', '')
        
        command = [
            sys.executable, '-m', 'unittest',
            module_path,
            '-v'
        ]
        
        return_code, stdout, stderr = self.timeout_manager.execute_with_timeout(
            command=command,
            timeout=context.timeout,
            cwd=context.working_directory,
            env=context.environment
        )
        
        return self._parse_unittest_output(stdout, stderr, return_code, test_file, context)
    
    def _parse_pytest_output(self, stdout: str, stderr: str, return_code: int, 
                           test_file: Path, context: TestExecutionContext) -> List[TestDetail]:
        """Parse pytest output to extract test results"""
        results = []
        
        # Try to parse JSON report if available
        try:
            import json
            with open('/tmp/pytest_report.json', 'r') as f:
                report = json.load(f)
                
            for test in report.get('tests', []):
                status = TestStatus.PASSED
                if test['outcome'] == 'failed':
                    status = TestStatus.FAILED
                elif test['outcome'] == 'skipped':
                    status = TestStatus.SKIPPED
                
                results.append(TestDetail(
                    name=test['nodeid'],
                    status=status,
                    duration=test.get('duration', 0),
                    error_message=test.get('call', {}).get('longrepr') if status == TestStatus.FAILED else None,
                    output=stdout,
                    category=context.category
                ))
                
        except Exception:
            # Fallback to parsing text output
            results = self._parse_text_output(stdout, stderr, return_code, test_file, context)
        
        return results
    
    def _parse_unittest_output(self, stdout: str, stderr: str, return_code: int,
                             test_file: Path, context: TestExecutionContext) -> List[TestDetail]:
        """Parse unittest output to extract test results"""
        return self._parse_text_output(stdout, stderr, return_code, test_file, context)
    
    def _parse_text_output(self, stdout: str, stderr: str, return_code: int,
                          test_file: Path, context: TestExecutionContext) -> List[TestDetail]:
        """Parse text output when structured output is not available"""
        # This is a simplified parser - in practice, you'd want more sophisticated parsing
        
        if return_code == -1:  # Timeout
            return [TestDetail(
                name=str(test_file),
                status=TestStatus.TIMEOUT,
                duration=context.timeout,
                error_message="Test execution timed out",
                output=stdout + stderr,
                category=context.category
            )]
        
        # Simple heuristic based on return code and output
        if return_code == 0:
            status = TestStatus.PASSED
            error_message = None
        else:
            status = TestStatus.FAILED
            error_message = stderr or "Test failed"
        
        return [TestDetail(
            name=str(test_file),
            status=status,
            duration=0,  # Would need to parse from output
            error_message=error_message,
            output=stdout,
            category=context.category
        )]
    
    def add_progress_callback(self, callback: Callable[[ExecutionProgress], None]):
        """Add a progress monitoring callback"""
        self.progress_monitor.add_progress_callback(callback)
    
    def get_current_progress(self) -> Optional[ExecutionProgress]:
        """Get current execution progress"""
        return self.progress_monitor.current_progress


# Example usage
if __name__ == "__main__":
    async def main():
        from pathlib import Path
        
        # Load config
        config = TestConfig.load_from_file(Path("tests/config/test-config.yaml"))
        
        # Create runner
        runner = TestRunnerEngine(config)
        
        # Add progress callback
        def progress_callback(progress: ExecutionProgress):
            print(f"Progress: {progress.progress_percentage:.1f}% "
                  f"({progress.completed_tests}/{progress.total_tests}) "
                  f"- Current: {progress.current_test}")
        
        runner.add_progress_callback(progress_callback)
        
        # Execute tests
        context = TestExecutionContext(
            category=TestCategory.UNIT,
            test_files=[],  # Will be discovered
            timeout=60,
            parallel=True
        )
        
        results = await runner.execute_category_tests(context)
        
        print(f"Executed {len(results)} tests")
        for result in results:
            print(f"  {result.name}: {result.status.value} ({result.duration:.2f}s)")
    
    asyncio.run(main())