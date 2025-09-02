"""
Test Execution Engine with Timeout Handling

This module provides a comprehensive test execution engine that supports:
- Configurable timeouts per test category
- Automatic retry logic for flaky tests with exponential backoff
- Test result aggregation and reporting
- Parallel test execution with resource management
"""

import asyncio
import concurrent.futures
import json
import logging
import multiprocessing
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import threading
import queue
import signal
import psutil


class TestCategory(Enum):
    """Test categories with different timeout requirements"""
    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"


class TestStatus(Enum):
    """Test execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RETRYING = "retrying"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestConfig:
    """Configuration for test execution"""
    # Timeout settings per category (in seconds)
    timeouts: Dict[TestCategory, int] = field(default_factory=lambda: {
        TestCategory.UNIT: 30,
        TestCategory.INTEGRATION: 120,
        TestCategory.E2E: 300,
        TestCategory.PERFORMANCE: 600,
        TestCategory.RELIABILITY: 900
    })
    
    # Retry settings
    max_retries: int = 3
    retry_delay_base: float = 1.0  # Base delay for exponential backoff
    retry_delay_max: float = 60.0  # Maximum delay between retries
    
    # Parallel execution settings
    max_workers: int = field(default_factory=lambda: min(4, multiprocessing.cpu_count()))
    memory_limit_mb: int = 2048  # Memory limit per worker
    
    # Resource management
    cpu_threshold: float = 0.8  # CPU usage threshold to throttle execution
    memory_threshold: float = 0.8  # Memory usage threshold
    
    # Flaky test detection
    flaky_threshold: int = 2  # Number of failures before marking as flaky
    flaky_success_rate: float = 0.7  # Success rate threshold for flaky tests


@dataclass
class TestResult:
    """Individual test result"""
    test_id: str
    category: TestCategory
    status: TestStatus
    duration: float
    start_time: datetime
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    retry_count: int = 0
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None


@dataclass
class TestSuiteResult:
    """Aggregated test suite results"""
    total_tests: int
    passed: int
    failed: int
    timeout: int
    error: int
    skipped: int
    total_duration: float
    start_time: datetime
    end_time: datetime
    results: List[TestResult]
    flaky_tests: Set[str] = field(default_factory=set)
    resource_usage: Dict[str, float] = field(default_factory=dict)


class ResourceMonitor:
    """Monitor system resources during test execution"""
    
    def __init__(self):
        self.monitoring = False
        self.cpu_usage = []
        self.memory_usage = []
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start resource monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            
    def _monitor_resources(self):
        """Monitor system resources"""
        while self.monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                
                self.cpu_usage.append(cpu_percent)
                self.memory_usage.append(memory_percent)
                
                # Keep only last 60 measurements (1 minute of data)
                if len(self.cpu_usage) > 60:
                    self.cpu_usage.pop(0)
                if len(self.memory_usage) > 60:
                    self.memory_usage.pop(0)
                    
            except Exception as e:
                logging.warning(f"Resource monitoring error: {e}")
                
            time.sleep(1)
            
    def get_average_usage(self) -> Tuple[float, float]:
        """Get average CPU and memory usage"""
        avg_cpu = sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0
        avg_memory = sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0
        return avg_cpu, avg_memory
        
    def should_throttle(self, config: TestConfig) -> bool:
        """Check if execution should be throttled due to resource usage"""
        if not self.cpu_usage or not self.memory_usage:
            return False
            
        current_cpu = self.cpu_usage[-1] / 100.0
        current_memory = self.memory_usage[-1] / 100.0
        
        return (current_cpu > config.cpu_threshold or 
                current_memory > config.memory_threshold)


class TestExecutionEngine:
    """Main test execution engine with timeout handling and retry logic"""
    
    def __init__(self, config: Optional[TestConfig] = None):
        self.config = config or TestConfig()
        self.logger = logging.getLogger(__name__)
        self.resource_monitor = ResourceMonitor()
        self.flaky_test_history: Dict[str, List[bool]] = {}  # Track test success/failure history
        self.active_processes: Set[subprocess.Popen] = set()
        self.shutdown_requested = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
        self._cleanup_processes()
        
    def _cleanup_processes(self):
        """Clean up active processes"""
        for process in list(self.active_processes):
            try:
                if process.poll() is None:  # Process is still running
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                self.active_processes.discard(process)
            except Exception as e:
                self.logger.warning(f"Error cleaning up process: {e}")
                
    def categorize_test(self, test_path: str) -> TestCategory:
        """Categorize test based on its path"""
        path_lower = test_path.lower()
        
        if "/unit/" in path_lower or "test_unit_" in path_lower:
            return TestCategory.UNIT
        elif "/integration/" in path_lower or "test_integration_" in path_lower:
            return TestCategory.INTEGRATION
        elif "/e2e/" in path_lower or "test_e2e_" in path_lower:
            return TestCategory.E2E
        elif "/performance/" in path_lower or "test_performance_" in path_lower:
            return TestCategory.PERFORMANCE
        elif "/reliability/" in path_lower or "test_reliability_" in path_lower:
            return TestCategory.RELIABILITY
        else:
            return TestCategory.UNIT  # Default to unit tests
            
    def discover_tests(self, test_dir: str = "tests") -> List[str]:
        """Discover all test files"""
        test_files = []
        test_path = Path(test_dir)
        
        if not test_path.exists():
            self.logger.warning(f"Test directory {test_dir} does not exist")
            return test_files
            
        for pattern in ["test_*.py", "*_test.py"]:
            test_files.extend(test_path.rglob(pattern))
            
        return [str(f) for f in test_files]
        
    def _calculate_retry_delay(self, retry_count: int) -> float:
        """Calculate delay for exponential backoff"""
        delay = self.config.retry_delay_base * (2 ** retry_count)
        return min(delay, self.config.retry_delay_max)
        
    def _is_flaky_test(self, test_id: str) -> bool:
        """Check if test is considered flaky based on history"""
        if test_id not in self.flaky_test_history:
            return False
            
        history = self.flaky_test_history[test_id]
        if len(history) < self.config.flaky_threshold:
            return False
            
        success_rate = sum(history) / len(history)
        return success_rate < self.config.flaky_success_rate
        
    def _update_flaky_history(self, test_id: str, success: bool):
        """Update flaky test history"""
        if test_id not in self.flaky_test_history:
            self.flaky_test_history[test_id] = []
            
        history = self.flaky_test_history[test_id]
        history.append(success)
        
        # Keep only last 10 results
        if len(history) > 10:
            history.pop(0)
            
    async def _execute_single_test(self, test_file: str, semaphore: asyncio.Semaphore) -> TestResult:
        """Execute a single test with timeout and retry logic"""
        async with semaphore:
            test_id = test_file
            category = self.categorize_test(test_file)
            timeout = self.config.timeouts[category]
            
            result = TestResult(
                test_id=test_id,
                category=category,
                status=TestStatus.PENDING,
                duration=0.0,
                start_time=datetime.now()
            )
            
            for retry_count in range(self.config.max_retries + 1):
                if self.shutdown_requested:
                    result.status = TestStatus.SKIPPED
                    result.error_message = "Execution cancelled"
                    break
                    
                # Wait for resources if needed
                while self.resource_monitor.should_throttle(self.config):
                    if self.shutdown_requested:
                        break
                    await asyncio.sleep(1)
                    
                if retry_count > 0:
                    delay = self._calculate_retry_delay(retry_count - 1)
                    self.logger.info(f"Retrying {test_id} in {delay:.1f}s (attempt {retry_count + 1})")
                    await asyncio.sleep(delay)
                    result.status = TestStatus.RETRYING
                    
                result.retry_count = retry_count
                result.status = TestStatus.RUNNING
                result.start_time = datetime.now()
                
                try:
                    # Execute test with timeout
                    process = await asyncio.create_subprocess_exec(
                        sys.executable, "-m", "pytest", test_file, "-v",
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    
                    try:
                        stdout, stderr = await asyncio.wait_for(
                            process.communicate(), timeout=timeout
                        )
                        
                        result.end_time = datetime.now()
                        result.duration = (result.end_time - result.start_time).total_seconds()
                        result.stdout = stdout.decode('utf-8', errors='ignore')
                        result.stderr = stderr.decode('utf-8', errors='ignore')
                        
                        if process.returncode == 0:
                            result.status = TestStatus.PASSED
                            self._update_flaky_history(test_id, True)
                            break
                        else:
                            result.status = TestStatus.FAILED
                            result.error_message = f"Test failed with return code {process.returncode}"
                            self._update_flaky_history(test_id, False)
                            
                    except asyncio.TimeoutError:
                        result.status = TestStatus.TIMEOUT
                        result.error_message = f"Test timed out after {timeout}s"
                        result.end_time = datetime.now()
                        result.duration = timeout
                        
                        # Kill the process
                        try:
                            process.kill()
                            await process.wait()
                        except:
                            pass
                            
                        self._update_flaky_history(test_id, False)
                        
                except Exception as e:
                    result.status = TestStatus.ERROR
                    result.error_message = f"Execution error: {str(e)}"
                    result.end_time = datetime.now()
                    result.duration = (result.end_time - result.start_time).total_seconds()
                    self._update_flaky_history(test_id, False)
                    
                # If test passed or we've exhausted retries, break
                if result.status == TestStatus.PASSED or retry_count >= self.config.max_retries:
                    break
                    
                # Only retry if test failed (not timeout or error, unless it's flaky)
                if result.status not in [TestStatus.FAILED, TestStatus.TIMEOUT]:
                    break
                    
                # Don't retry if not flaky and this isn't the first failure
                if not self._is_flaky_test(test_id) and retry_count > 0:
                    break
                    
            return result
            
    async def run_tests_async(self, test_files: List[str]) -> TestSuiteResult:
        """Run tests asynchronously with parallel execution"""
        start_time = datetime.now()
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        try:
            # Create semaphore to limit concurrent executions
            semaphore = asyncio.Semaphore(self.config.max_workers)
            
            # Execute all tests
            tasks = [
                self._execute_single_test(test_file, semaphore)
                for test_file in test_files
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            test_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Create error result for failed task
                    error_result = TestResult(
                        test_id=test_files[i],
                        category=self.categorize_test(test_files[i]),
                        status=TestStatus.ERROR,
                        duration=0.0,
                        start_time=start_time,
                        end_time=datetime.now(),
                        error_message=str(result)
                    )
                    test_results.append(error_result)
                else:
                    test_results.append(result)
                    
        finally:
            self.resource_monitor.stop_monitoring()
            
        end_time = datetime.now()
        
        # Aggregate results
        total_tests = len(test_results)
        passed = sum(1 for r in test_results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in test_results if r.status == TestStatus.FAILED)
        timeout = sum(1 for r in test_results if r.status == TestStatus.TIMEOUT)
        error = sum(1 for r in test_results if r.status == TestStatus.ERROR)
        skipped = sum(1 for r in test_results if r.status == TestStatus.SKIPPED)
        
        total_duration = (end_time - start_time).total_seconds()
        
        # Identify flaky tests
        flaky_tests = {test_id for test_id in self.flaky_test_history 
                      if self._is_flaky_test(test_id)}
        
        # Get resource usage
        avg_cpu, avg_memory = self.resource_monitor.get_average_usage()
        resource_usage = {
            'avg_cpu_percent': avg_cpu,
            'avg_memory_percent': avg_memory
        }
        
        return TestSuiteResult(
            total_tests=total_tests,
            passed=passed,
            failed=failed,
            timeout=timeout,
            error=error,
            skipped=skipped,
            total_duration=total_duration,
            start_time=start_time,
            end_time=end_time,
            results=test_results,
            flaky_tests=flaky_tests,
            resource_usage=resource_usage
        )
        
    def run_tests(self, test_files: Optional[List[str]] = None) -> TestSuiteResult:
        """Run tests synchronously (wrapper for async method)"""
        if test_files is None:
            test_files = self.discover_tests()
            
        if not test_files:
            self.logger.warning("No test files found")
            return TestSuiteResult(
                total_tests=0,
                passed=0,
                failed=0,
                timeout=0,
                error=0,
                skipped=0,
                total_duration=0.0,
                start_time=datetime.now(),
                end_time=datetime.now(),
                results=[]
            )
            
        self.logger.info(f"Running {len(test_files)} test files with {self.config.max_workers} workers")
        
        # Run async method
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(self.run_tests_async(test_files))
        finally:
            loop.close()
            
    def generate_report(self, result: TestSuiteResult, output_file: Optional[str] = None) -> str:
        """Generate detailed test execution report"""
        report = {
            'summary': {
                'total_tests': result.total_tests,
                'passed': result.passed,
                'failed': result.failed,
                'timeout': result.timeout,
                'error': result.error,
                'skipped': result.skipped,
                'success_rate': result.passed / result.total_tests if result.total_tests > 0 else 0,
                'total_duration': result.total_duration,
                'start_time': result.start_time.isoformat(),
                'end_time': result.end_time.isoformat()
            },
            'resource_usage': result.resource_usage,
            'flaky_tests': list(result.flaky_tests),
            'test_results': []
        }
        
        # Add individual test results
        for test_result in result.results:
            report['test_results'].append({
                'test_id': test_result.test_id,
                'category': test_result.category.value,
                'status': test_result.status.value,
                'duration': test_result.duration,
                'retry_count': test_result.retry_count,
                'error_message': test_result.error_message,
                'start_time': test_result.start_time.isoformat(),
                'end_time': test_result.end_time.isoformat() if test_result.end_time else None
            })
            
        # Generate text summary
        text_report = f"""
Test Execution Report
====================

Summary:
  Total Tests: {result.total_tests}
  Passed: {result.passed}
  Failed: {result.failed}
  Timeout: {result.timeout}
  Error: {result.error}
  Skipped: {result.skipped}
  Success Rate: {result.passed / result.total_tests * 100:.1f}%
  Total Duration: {result.total_duration:.2f}s

Resource Usage:
  Average CPU: {result.resource_usage.get('avg_cpu_percent', 0):.1f}%
  Average Memory: {result.resource_usage.get('avg_memory_percent', 0):.1f}%

Flaky Tests: {len(result.flaky_tests)}
{chr(10).join(f"  - {test}" for test in sorted(result.flaky_tests))}

Failed Tests:
{chr(10).join(f"  - {r.test_id}: {r.error_message}" for r in result.results if r.status == TestStatus.FAILED)}

Timeout Tests:
{chr(10).join(f"  - {r.test_id}: {r.error_message}" for r in result.results if r.status == TestStatus.TIMEOUT)}
"""
        
        if output_file:
            # Save JSON report
            json_file = output_file.replace('.txt', '.json')
            with open(json_file, 'w') as f:
                json.dump(report, f, indent=2)
                
            # Save text report
            with open(output_file, 'w') as f:
                f.write(text_report)
                
            self.logger.info(f"Reports saved to {output_file} and {json_file}")
            
        return text_report


def main():
    """CLI entry point for test execution engine"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Execution Engine with Timeout Handling")
    parser.add_argument("--test-dir", default="tests", help="Test directory to scan")
    parser.add_argument("--max-workers", type=int, help="Maximum parallel workers")
    parser.add_argument("--timeout-unit", type=int, default=30, help="Timeout for unit tests (seconds)")
    parser.add_argument("--timeout-integration", type=int, default=120, help="Timeout for integration tests (seconds)")
    parser.add_argument("--timeout-e2e", type=int, default=300, help="Timeout for e2e tests (seconds)")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retry attempts")
    parser.add_argument("--output", help="Output file for test report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create config
    config = TestConfig()
    if args.max_workers:
        config.max_workers = args.max_workers
    if args.max_retries:
        config.max_retries = args.max_retries
        
    config.timeouts[TestCategory.UNIT] = args.timeout_unit
    config.timeouts[TestCategory.INTEGRATION] = args.timeout_integration
    config.timeouts[TestCategory.E2E] = args.timeout_e2e
    
    # Create engine and run tests
    engine = TestExecutionEngine(config)
    
    try:
        result = engine.run_tests()
        report = engine.generate_report(result, args.output)
        
        print(report)
        
        # Exit with appropriate code
        if result.failed > 0 or result.timeout > 0 or result.error > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\nTest execution cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"Test execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()