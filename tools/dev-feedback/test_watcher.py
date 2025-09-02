#!/usr/bin/env python3
"""
Test Watcher with Selective Execution

This module provides watch mode for tests with selective execution and fast feedback.
"""

import os
import sys
import time
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional, Set, Callable
from dataclasses import dataclass
import logging
from datetime import datetime
import fnmatch
import hashlib

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    print("Installing watchdog for file watching...")
    subprocess.run([sys.executable, "-m", "pip", "install", "watchdog"], check=True)
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

@dataclass
class TestResult:
    """Test execution result"""
    test_file: str
    status: str  # 'pass', 'fail', 'skip'
    duration: float
    output: str
    error: Optional[str] = None

@dataclass
class WatchConfig:
    """Configuration for test watcher"""
    watch_dirs: List[Path]
    test_patterns: List[str]
    ignore_patterns: List[str]
    test_categories: List[str]
    fast_mode: bool
    debounce_delay: float
    max_parallel_tests: int

class TestFileHandler(FileSystemEventHandler):
    """File system event handler for test watching"""
    
    def __init__(self, watcher: 'TestWatcher'):
        self.watcher = watcher
        self.logger = logging.getLogger(__name__)
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Check if file should trigger test execution
        if self.watcher.should_trigger_tests(file_path):
            self.logger.info(f"File changed: {file_path}")
            self.watcher.schedule_test_run(file_path)
    
    def on_created(self, event):
        self.on_modified(event)

class TestWatcher:
    """Watch files and run tests with selective execution"""
    
    def __init__(self, config: WatchConfig, project_root: Optional[Path] = None):
        self.config = config
        self.project_root = project_root or Path.cwd()
        self.logger = logging.getLogger(__name__)
        
        # State management
        self.observer = Observer()
        self.running = False
        self.pending_runs = set()
        self.last_run_time = {}
        self.file_hashes = {}
        
        # Test execution
        self.test_runner_lock = threading.Lock()
        self.active_tests = set()
        
        # Initialize file hashes
        self._initialize_file_hashes()
    
    def _initialize_file_hashes(self):
        """Initialize file hashes for change detection"""
        for watch_dir in self.config.watch_dirs:
            if watch_dir.exists():
                for file_path in watch_dir.rglob("*"):
                    if file_path.is_file():
                        self.file_hashes[str(file_path)] = self._get_file_hash(file_path)
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get hash of file content"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""
    
    def should_trigger_tests(self, file_path: Path) -> bool:
        """Check if file change should trigger test execution"""
        file_str = str(file_path)
        
        # Check if file actually changed (content-based)
        current_hash = self._get_file_hash(file_path)
        if file_str in self.file_hashes and self.file_hashes[file_str] == current_hash:
            return False
        
        self.file_hashes[file_str] = current_hash
        
        # Check ignore patterns
        for pattern in self.config.ignore_patterns:
            if fnmatch.fnmatch(file_str, pattern):
                return False
        
        # Check if it's a Python file or test file
        if file_path.suffix == '.py':
            return True
        
        # Check if it's a configuration file that might affect tests
        config_extensions = ['.yaml', '.yml', '.json', '.toml', '.ini']
        if file_path.suffix in config_extensions:
            return True
        
        return False
    
    def schedule_test_run(self, changed_file: Path):
        """Schedule test run with debouncing"""
        current_time = time.time()
        file_str = str(changed_file)
        
        # Debouncing: avoid running tests too frequently for the same file
        if file_str in self.last_run_time:
            if current_time - self.last_run_time[file_str] < self.config.debounce_delay:
                return
        
        self.last_run_time[file_str] = current_time
        self.pending_runs.add(changed_file)
        
        # Schedule execution after debounce delay
        threading.Timer(self.config.debounce_delay, self._execute_pending_tests).start()
    
    def _execute_pending_tests(self):
        """Execute pending test runs"""
        if not self.pending_runs:
            return
        
        with self.test_runner_lock:
            changed_files = list(self.pending_runs)
            self.pending_runs.clear()
            
            self.logger.info(f"Running tests for {len(changed_files)} changed files...")
            
            # Determine which tests to run
            tests_to_run = self._determine_tests_to_run(changed_files)
            
            if tests_to_run:
                self._run_tests(tests_to_run)
            else:
                self.logger.info("No tests to run for changed files")
    
    def _determine_tests_to_run(self, changed_files: List[Path]) -> List[Path]:
        """Determine which tests to run based on changed files"""
        tests_to_run = set()
        
        for changed_file in changed_files:
            # If it's a test file, run it directly
            if self._is_test_file(changed_file):
                tests_to_run.add(changed_file)
            else:
                # Find related test files
                related_tests = self._find_related_tests(changed_file)
                tests_to_run.update(related_tests)
        
        # Filter by test patterns and categories
        filtered_tests = []
        for test_file in tests_to_run:
            if self._matches_test_criteria(test_file):
                filtered_tests.append(test_file)
        
        return filtered_tests
    
    def _is_test_file(self, file_path: Path) -> bool:
        """Check if file is a test file"""
        for pattern in self.config.test_patterns:
            if fnmatch.fnmatch(file_path.name, pattern):
                return True
        return False
    
    def _find_related_tests(self, source_file: Path) -> List[Path]:
        """Find test files related to a source file"""
        related_tests = []
        
        # Strategy 1: Look for test files with similar names
        source_name = source_file.stem
        test_name_patterns = [
            f"test_{source_name}.py",
            f"test_{source_name}_*.py",
            f"{source_name}_test.py",
            f"*_test_{source_name}.py"
        ]
        
        for watch_dir in self.config.watch_dirs:
            if watch_dir.exists():
                for pattern in test_name_patterns:
                    for test_file in watch_dir.rglob(pattern):
                        if test_file.is_file():
                            related_tests.append(test_file)
        
        # Strategy 2: Look in parallel test directory structure
        try:
            # Convert source path to test path
            relative_path = source_file.relative_to(self.project_root)
            
            # Common test directory mappings
            test_mappings = [
                ("backend", "backend/tests"),
                ("frontend/src", "frontend/src/tests"),
                ("core", "tests/unit"),
                ("infrastructure", "tests/integration")
            ]
            
            for src_prefix, test_prefix in test_mappings:
                if str(relative_path).startswith(src_prefix):
                    test_path = Path(test_prefix) / relative_path.relative_to(src_prefix)
                    test_path = test_path.with_name(f"test_{test_path.stem}.py")
                    full_test_path = self.project_root / test_path
                    
                    if full_test_path.exists():
                        related_tests.append(full_test_path)
        
        except ValueError:
            # File is not under project root
            pass
        
        return related_tests
    
    def _matches_test_criteria(self, test_file: Path) -> bool:
        """Check if test file matches configured criteria"""
        # Check test categories
        if self.config.test_categories:
            file_category = self._get_test_category(test_file)
            if file_category not in self.config.test_categories:
                return False
        
        # Check fast mode
        if self.config.fast_mode and self._is_slow_test(test_file):
            return False
        
        return True
    
    def _get_test_category(self, test_file: Path) -> str:
        """Determine test category from file path"""
        path_str = str(test_file)
        
        if "/unit/" in path_str or "test_unit_" in test_file.name:
            return "unit"
        elif "/integration/" in path_str or "test_integration_" in test_file.name:
            return "integration"
        elif "/e2e/" in path_str or "test_e2e_" in test_file.name:
            return "e2e"
        elif "/performance/" in path_str or "test_performance_" in test_file.name:
            return "performance"
        else:
            return "unit"  # Default to unit tests
    
    def _is_slow_test(self, test_file: Path) -> bool:
        """Check if test is considered slow"""
        # Heuristics for identifying slow tests
        slow_indicators = [
            "integration",
            "e2e",
            "performance",
            "slow",
            "benchmark"
        ]
        
        path_str = str(test_file).lower()
        return any(indicator in path_str for indicator in slow_indicators)
    
    def _run_tests(self, test_files: List[Path]):
        """Run the specified test files"""
        if not test_files:
            return
        
        start_time = time.time()
        self.logger.info(f"🧪 Running {len(test_files)} test files...")
        
        results = []
        
        # Run tests (limit parallel execution)
        for i in range(0, len(test_files), self.config.max_parallel_tests):
            batch = test_files[i:i + self.config.max_parallel_tests]
            batch_results = self._run_test_batch(batch)
            results.extend(batch_results)
        
        # Report results
        duration = time.time() - start_time
        self._report_test_results(results, duration)
    
    def _run_test_batch(self, test_files: List[Path]) -> List[TestResult]:
        """Run a batch of test files"""
        results = []
        
        for test_file in test_files:
            if str(test_file) in self.active_tests:
                continue  # Skip if already running
            
            self.active_tests.add(str(test_file))
            
            try:
                result = self._run_single_test(test_file)
                results.append(result)
            finally:
                self.active_tests.discard(str(test_file))
        
        return results
    
    def _run_single_test(self, test_file: Path) -> TestResult:
        """Run a single test file"""
        start_time = time.time()
        
        try:
            # Determine test runner based on file location
            if "frontend" in str(test_file):
                # Frontend test (Vitest)
                cmd = ["npm", "run", "test", "--", str(test_file), "--run"]
                cwd = self.project_root / "frontend"
            else:
                # Backend test (pytest)
                cmd = [sys.executable, "-m", "pytest", str(test_file), "-v"]
                cwd = self.project_root
            
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=60  # 1 minute timeout per test
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                status = "pass"
                error = None
            else:
                status = "fail"
                error = result.stderr
            
            return TestResult(
                test_file=str(test_file),
                status=status,
                duration=duration,
                output=result.stdout,
                error=error
            )
        
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return TestResult(
                test_file=str(test_file),
                status="fail",
                duration=duration,
                output="",
                error="Test timed out"
            )
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_file=str(test_file),
                status="fail",
                duration=duration,
                output="",
                error=str(e)
            )
    
    def _report_test_results(self, results: List[TestResult], total_duration: float):
        """Report test execution results"""
        passed = len([r for r in results if r.status == "pass"])
        failed = len([r for r in results if r.status == "fail"])
        
        # Summary
        self.logger.info(f"📊 Test Results: {passed} passed, {failed} failed in {total_duration:.2f}s")
        
        # Failed tests details
        if failed > 0:
            self.logger.error("❌ Failed tests:")
            for result in results:
                if result.status == "fail":
                    self.logger.error(f"  - {Path(result.test_file).name}: {result.error}")
        
        # Success message
        if failed == 0:
            self.logger.info("✅ All tests passed!")
    
    def start_watching(self):
        """Start watching for file changes"""
        self.logger.info("🔍 Starting test watcher...")
        self.logger.info(f"Watching directories: {[str(d) for d in self.config.watch_dirs]}")
        self.logger.info(f"Test patterns: {self.config.test_patterns}")
        self.logger.info(f"Categories: {self.config.test_categories or 'all'}")
        self.logger.info(f"Fast mode: {self.config.fast_mode}")
        
        # Setup file system observers
        handler = TestFileHandler(self)
        
        for watch_dir in self.config.watch_dirs:
            if watch_dir.exists():
                self.observer.schedule(handler, str(watch_dir), recursive=True)
                self.logger.info(f"Watching: {watch_dir}")
        
        self.observer.start()
        self.running = True
        
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop_watching()
    
    def stop_watching(self):
        """Stop watching for file changes"""
        self.logger.info("🛑 Stopping test watcher...")
        self.running = False
        self.observer.stop()
        self.observer.join()
    
    def run_all_tests(self):
        """Run all tests matching criteria"""
        self.logger.info("🧪 Running all tests...")
        
        all_tests = []
        for watch_dir in self.config.watch_dirs:
            if watch_dir.exists():
                for pattern in self.config.test_patterns:
                    for test_file in watch_dir.rglob(pattern):
                        if test_file.is_file() and self._matches_test_criteria(test_file):
                            all_tests.append(test_file)
        
        if all_tests:
            self._run_tests(all_tests)
        else:
            self.logger.info("No tests found matching criteria")

def create_default_config(project_root: Path) -> WatchConfig:
    """Create default watch configuration"""
    return WatchConfig(
        watch_dirs=[
            project_root / "backend",
            project_root / "frontend" / "src",
            project_root / "core",
            project_root / "infrastructure",
            project_root / "tests"
        ],
        test_patterns=[
            "test_*.py",
            "*_test.py",
            "*.test.ts",
            "*.test.js"
        ],
        ignore_patterns=[
            "*.pyc",
            "__pycache__/*",
            "node_modules/*",
            ".git/*",
            "*.log",
            "*.tmp"
        ],
        test_categories=[],  # Empty means all categories
        fast_mode=False,
        debounce_delay=1.0,  # 1 second
        max_parallel_tests=3
    )

def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test watcher with selective execution")
    parser.add_argument('--category', action='append', help='Test categories to run (unit, integration, e2e, performance)')
    parser.add_argument('--pattern', action='append', help='Test file patterns to watch')
    parser.add_argument('--files', action='append', help='Specific test files to watch')
    parser.add_argument('--fast', action='store_true', help='Fast mode (skip slow tests)')
    parser.add_argument('--run-all', action='store_true', help='Run all tests once and exit')
    parser.add_argument('--debounce', type=float, default=1.0, help='Debounce delay in seconds')
    parser.add_argument('--max-parallel', type=int, default=3, help='Maximum parallel test executions')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create configuration
    project_root = Path.cwd()
    config = create_default_config(project_root)
    
    # Apply command line options
    if args.category:
        config.test_categories = args.category
    
    if args.pattern:
        config.test_patterns = args.pattern
    
    if args.files:
        # Convert specific files to watch directories and patterns
        specific_files = [Path(f) for f in args.files]
        config.watch_dirs = list(set(f.parent for f in specific_files))
        config.test_patterns = [f.name for f in specific_files]
    
    config.fast_mode = args.fast
    config.debounce_delay = args.debounce
    config.max_parallel_tests = args.max_parallel
    
    # Create and start watcher
    watcher = TestWatcher(config, project_root)
    
    if args.run_all:
        watcher.run_all_tests()
    else:
        try:
            watcher.start_watching()
        except KeyboardInterrupt:
            print("\nTest watcher stopped.")

if __name__ == "__main__":
    main()