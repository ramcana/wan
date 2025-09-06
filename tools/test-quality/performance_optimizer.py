import pytest
#!/usr/bin/env python3
"""
Test Performance Optimization System

Implements test performance profiling to identify slow tests, creates optimization
recommendations based on performance analysis, implements test caching and
memoization for expensive operations, and adds performance regression detection.
"""

import json
import sqlite3
import time
import statistics
import subprocess
import sys
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
import functools
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class TestPerformanceMetric:
    """Performance metrics for a single test"""
    test_id: str
    test_file: str
    test_name: str
    duration: float
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    setup_time: Optional[float] = None
    teardown_time: Optional[float] = None
    timestamp: Optional[datetime] = None
    status: str = 'passed'  # passed, failed, skipped


@dataclass
class TestPerformanceProfile:
    """Complete performance profile for test execution"""
    total_duration: float
    test_count: int
    slow_tests: List[TestPerformanceMetric]
    fast_tests: List[TestPerformanceMetric]
    average_duration: float
    median_duration: float
    percentile_95: float
    memory_peak: Optional[float] = None
    cpu_peak: Optional[float] = None
    parallel_efficiency: Optional[float] = None


@dataclass
class PerformanceRegression:
    """Detected performance regression"""
    test_id: str
    current_duration: float
    baseline_duration: float
    regression_factor: float
    severity: str  # 'minor', 'moderate', 'severe'
    detected_at: datetime
    confidence: float


@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation"""
    test_id: str
    recommendation_type: str  # 'caching', 'mocking', 'parallelization', 'refactoring'
    description: str
    estimated_improvement: float  # Expected time savings in seconds
    implementation_effort: str  # 'low', 'medium', 'high'
    priority: str  # 'low', 'medium', 'high', 'critical'
    code_examples: List[str] = None


class TestPerformanceProfiler:
    """Profiles test performance and identifies bottlenecks"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.profile_data = {}
        self.slow_test_threshold = 5.0  # seconds
        self.very_slow_threshold = 30.0  # seconds
    
    def profile_test_suite(self, test_files: List[Path] = None) -> TestPerformanceProfile:
        """Profile performance of test suite"""
        print("Starting test performance profiling...")
        
        # Discover test files if not provided
        if not test_files:
            test_files = self._discover_test_files()
        
        # Run tests with performance monitoring
        test_metrics = self._run_performance_tests(test_files)
        
        # Analyze performance data
        profile = self._analyze_performance_data(test_metrics)
        
        return profile
    
    def _discover_test_files(self) -> List[Path]:
        """Discover test files in the project"""
        test_files = []
        
        # Look for test files in common locations
        test_patterns = ['test_*.py', '*_test.py']
        test_dirs = ['tests', 'test', 'backend/tests', 'frontend/tests']
        
        for test_dir in test_dirs:
            test_path = self.project_root / test_dir
            if test_path.exists():
                for pattern in test_patterns:
                    test_files.extend(test_path.rglob(pattern))
        
        return test_files
    
    def _run_performance_tests(self, test_files: List[Path]) -> List[TestPerformanceMetric]:
        """Run tests with performance monitoring"""
        metrics = []
        
        for test_file in test_files:
            print(f"Profiling {test_file}...")
            
            # Run individual test file with timing
            file_metrics = self._profile_test_file(test_file)
            metrics.extend(file_metrics)
        
        return metrics
    
    def _profile_test_file(self, test_file: Path) -> List[TestPerformanceMetric]:
        """Profile performance of a single test file"""
        metrics = []
        
        try:
            # Run pytest with detailed output and timing
            cmd = [
                sys.executable, '-m', 'pytest',
                str(test_file),
                '-v',
                '--tb=no',
                '--durations=0',
                '--quiet'
            ]
            
            start_time = time.time()
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per file
            )
            total_time = time.time() - start_time
            
            # Parse pytest output for individual test timings
            if result.returncode == 0 or 'failed' in result.stdout.lower():
                metrics.extend(self._parse_pytest_output(test_file, result.stdout, total_time))
            else:
                # If pytest failed, create a metric for the whole file
                metrics.append(TestPerformanceMetric(
                    test_id=f"{test_file}::FAILED",
                    test_file=str(test_file),
                    test_name="FILE_EXECUTION_FAILED",
                    duration=total_time,
                    timestamp=datetime.now(),
                    status='failed'
                ))
        
        except subprocess.TimeoutExpired:
            metrics.append(TestPerformanceMetric(
                test_id=f"{test_file}::TIMEOUT",
                test_file=str(test_file),
                test_name="FILE_EXECUTION_TIMEOUT",
                duration=300.0,
                timestamp=datetime.now(),
                status='timeout'
            ))
        
        except Exception as e:
            print(f"Error profiling {test_file}: {e}")
        
        return metrics
    
    def _parse_pytest_output(self, test_file: Path, output: str, total_time: float) -> List[TestPerformanceMetric]:
        """Parse pytest output to extract individual test timings"""
        metrics = []
        
        # Look for test duration information in pytest output
        lines = output.split('\n')
        
        for line in lines:
            # Parse lines like: "test_file.py::test_function PASSED [100%]"
            if '::' in line and ('PASSED' in line or 'FAILED' in line or 'SKIPPED' in line):
                parts = line.split()
                if len(parts) >= 2:
                    test_path = parts[0]
                    status = 'passed' if 'PASSED' in line else 'failed' if 'FAILED' in line else 'skipped'
                    
                    # Extract test name
                    if '::' in test_path:
                        file_part, test_name = test_path.split('::', 1)
                        
                        # Estimate duration (pytest doesn't always show individual timings)
                        # We'll use a simple heuristic based on total time and test count
                        estimated_duration = total_time / max(len([l for l in lines if '::' in l]), 1)
                        
                        metrics.append(TestPerformanceMetric(
                            test_id=test_path,
                            test_file=str(test_file),
                            test_name=test_name,
                            duration=estimated_duration,
                            timestamp=datetime.now(),
                            status=status
                        ))
        
        # If no individual tests found, create one metric for the whole file
        if not metrics:
            metrics.append(TestPerformanceMetric(
                test_id=f"{test_file}::ALL_TESTS",
                test_file=str(test_file),
                test_name="ALL_TESTS",
                duration=total_time,
                timestamp=datetime.now(),
                status='unknown'
            ))
        
        return metrics
    
    def _analyze_performance_data(self, metrics: List[TestPerformanceMetric]) -> TestPerformanceProfile:
        """Analyze performance metrics to create profile"""
        if not metrics:
            return TestPerformanceProfile(
                total_duration=0.0,
                test_count=0,
                slow_tests=[],
                fast_tests=[],
                average_duration=0.0,
                median_duration=0.0,
                percentile_95=0.0
            )
        
        # Calculate statistics
        durations = [m.duration for m in metrics if m.status != 'skipped']
        total_duration = sum(durations)
        test_count = len(metrics)
        
        if durations:
            average_duration = statistics.mean(durations)
            median_duration = statistics.median(durations)
            percentile_95 = self._percentile(durations, 95)
        else:
            average_duration = median_duration = percentile_95 = 0.0
        
        # Identify slow and fast tests
        slow_tests = [m for m in metrics if m.duration > self.slow_test_threshold]
        fast_tests = [m for m in metrics if m.duration < 1.0 and m.status == 'passed']
        
        # Sort by duration
        slow_tests.sort(key=lambda x: x.duration, reverse=True)
        fast_tests.sort(key=lambda x: x.duration)
        
        return TestPerformanceProfile(
            total_duration=total_duration,
            test_count=test_count,
            slow_tests=slow_tests,
            fast_tests=fast_tests,
            average_duration=average_duration,
            median_duration=median_duration,
            percentile_95=percentile_95
        )
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))


class TestCacheManager:
    """Manages test caching and memoization for expensive operations"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.cache_dir = project_root / '.test_cache'
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache = {}
        self.cache_stats = {'hits': 0, 'misses': 0}
    
    def cached_test_data(self, cache_key: str, generator_func: Callable, ttl: int = 3600):
        """Decorator for caching expensive test data generation"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key based on function and arguments
                full_cache_key = f"{cache_key}_{self._hash_args(args, kwargs)}"
                
                # Check memory cache first
                if full_cache_key in self.memory_cache:
                    cache_entry = self.memory_cache[full_cache_key]
                    if time.time() - cache_entry['timestamp'] < ttl:
                        self.cache_stats['hits'] += 1
                        return cache_entry['data']
                
                # Check disk cache
                cache_file = self.cache_dir / f"{full_cache_key}.pkl"
                if cache_file.exists():
                    try:
                        with open(cache_file, 'rb') as f:
                            cache_entry = pickle.load(f)
                        
                        if time.time() - cache_entry['timestamp'] < ttl:
                            # Load into memory cache
                            self.memory_cache[full_cache_key] = cache_entry
                            self.cache_stats['hits'] += 1
                            return cache_entry['data']
                    except Exception:
                        pass  # Cache file corrupted, regenerate
                
                # Generate new data
                self.cache_stats['misses'] += 1
                data = func(*args, **kwargs)
                
                # Cache the result
                cache_entry = {
                    'data': data,
                    'timestamp': time.time()
                }
                
                # Store in memory cache
                self.memory_cache[full_cache_key] = cache_entry
                
                # Store in disk cache
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(cache_entry, f)
                except Exception as e:
                    print(f"Warning: Could not cache to disk: {e}")
                
                return data
            
            return wrapper
        return decorator
    
    def _hash_args(self, args: tuple, kwargs: dict) -> str:
        """Create hash of function arguments for cache key"""
        content = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def clear_cache(self, pattern: str = None):
        """Clear cache entries matching pattern"""
        if pattern:
            # Clear specific pattern
            keys_to_remove = [k for k in self.memory_cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self.memory_cache[key]
            
            # Clear disk cache
            for cache_file in self.cache_dir.glob(f"*{pattern}*.pkl"):
                cache_file.unlink()
        else:
            # Clear all cache
            self.memory_cache.clear()
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (self.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'hit_rate': hit_rate,
            'memory_cache_size': len(self.memory_cache),
            'disk_cache_files': len(list(self.cache_dir.glob("*.pkl")))
        }


class PerformanceRegressionDetector:
    """Detects performance regressions in test execution"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.db_path = project_root / '.test_performance' / 'performance.db'
        self.regression_threshold = 1.5  # 50% slower is considered regression
        self.severe_threshold = 2.0  # 100% slower is severe
        self._init_database()
    
    def _init_database(self):
        """Initialize performance tracking database"""
        self.db_path.parent.mkdir(exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS test_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT NOT NULL,
                    test_file TEXT NOT NULL,
                    test_name TEXT NOT NULL,
                    duration REAL NOT NULL,
                    memory_usage REAL,
                    cpu_usage REAL,
                    timestamp TEXT NOT NULL,
                    commit_hash TEXT,
                    branch TEXT,
                    status TEXT DEFAULT 'passed'
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS performance_regressions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT NOT NULL,
                    current_duration REAL NOT NULL,
                    baseline_duration REAL NOT NULL,
                    regression_factor REAL NOT NULL,
                    severity TEXT NOT NULL,
                    detected_at TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE
                )
            ''')
            
            conn.execute('CREATE INDEX IF NOT EXISTS idx_test_id ON test_performance(test_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON test_performance(timestamp)')
            
            conn.commit()
    
    def record_performance(self, metrics: List[TestPerformanceMetric]):
        """Record performance metrics for regression detection"""
        commit_hash = self._get_current_commit()
        branch = self._get_current_branch()
        
        with sqlite3.connect(self.db_path) as conn:
            for metric in metrics:
                conn.execute('''
                    INSERT INTO test_performance
                    (test_id, test_file, test_name, duration, memory_usage, cpu_usage,
                     timestamp, commit_hash, branch, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metric.test_id,
                    metric.test_file,
                    metric.test_name,
                    metric.duration,
                    metric.memory_usage,
                    metric.cpu_usage,
                    metric.timestamp.isoformat() if metric.timestamp else datetime.now().isoformat(),
                    commit_hash,
                    branch,
                    metric.status
                ))
            
            conn.commit()
    
    def detect_regressions(self, current_metrics: List[TestPerformanceMetric]) -> List[PerformanceRegression]:
        """Detect performance regressions compared to baseline"""
        regressions = []
        
        with sqlite3.connect(self.db_path) as conn:
            for metric in current_metrics:
                if metric.status != 'passed':
                    continue  # Only check regressions for passing tests
                
                # Get baseline performance (average of last 10 successful runs)
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT AVG(duration) as baseline_duration, COUNT(*) as sample_count
                    FROM test_performance
                    WHERE test_id = ? AND status = 'passed' AND timestamp < ?
                    ORDER BY timestamp DESC
                    LIMIT 10
                ''', (metric.test_id, datetime.now().isoformat()))
                
                result = cursor.fetchone()
                if result and result[0] and result[1] >= 3:  # Need at least 3 samples
                    baseline_duration = result[0]
                    sample_count = result[1]
                    
                    # Calculate regression factor
                    regression_factor = metric.duration / baseline_duration
                    
                    if regression_factor >= self.regression_threshold:
                        # Determine severity
                        if regression_factor >= self.severe_threshold:
                            severity = 'severe'
                        elif regression_factor >= 1.8:
                            severity = 'moderate'
                        else:
                            severity = 'minor'
                        
                        # Calculate confidence based on sample size and regression magnitude
                        confidence = min(0.95, 0.5 + (sample_count / 20) + (regression_factor - 1) / 2)
                        
                        regression = PerformanceRegression(
                            test_id=metric.test_id,
                            current_duration=metric.duration,
                            baseline_duration=baseline_duration,
                            regression_factor=regression_factor,
                            severity=severity,
                            detected_at=datetime.now(),
                            confidence=confidence
                        )
                        
                        regressions.append(regression)
                        
                        # Record regression in database
                        conn.execute('''
                            INSERT INTO performance_regressions
                            (test_id, current_duration, baseline_duration, regression_factor,
                             severity, detected_at, confidence)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            regression.test_id,
                            regression.current_duration,
                            regression.baseline_duration,
                            regression.regression_factor,
                            regression.severity,
                            regression.detected_at.isoformat(),
                            regression.confidence
                        ))
            
            conn.commit()
        
        return regressions
    
    def get_performance_history(self, test_id: str, days: int = 30) -> List[TestPerformanceMetric]:
        """Get performance history for a specific test"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT test_id, test_file, test_name, duration, memory_usage, cpu_usage,
                       timestamp, status
                FROM test_performance
                WHERE test_id = ? AND timestamp >= ?
                ORDER BY timestamp ASC
            ''', (test_id, cutoff_date.isoformat()))
            
            history = []
            for row in cursor.fetchall():
                history.append(TestPerformanceMetric(
                    test_id=row[0],
                    test_file=row[1],
                    test_name=row[2],
                    duration=row[3],
                    memory_usage=row[4],
                    cpu_usage=row[5],
                    timestamp=datetime.fromisoformat(row[6]),
                    status=row[7]
                ))
            
            return history
    
    def _get_current_commit(self) -> Optional[str]:
        """Get current git commit hash"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    
    def _get_current_branch(self) -> Optional[str]:
        """Get current git branch"""
        try:
            result = subprocess.run(
                ['git', 'branch', '--show-current'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None


class TestOptimizationRecommendationEngine:
    """Generates optimization recommendations based on performance analysis"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.slow_test_threshold = 5.0
        self.very_slow_threshold = 30.0
    
    def generate_recommendations(self, profile: TestPerformanceProfile, 
                               regressions: List[PerformanceRegression] = None) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Recommendations for slow tests
        recommendations.extend(self._recommend_slow_test_optimizations(profile.slow_tests))
        
        # Recommendations for performance regressions
        if regressions:
            recommendations.extend(self._recommend_regression_fixes(regressions))
        
        # General optimization recommendations
        recommendations.extend(self._recommend_general_optimizations(profile))
        
        # Sort by priority and estimated improvement
        recommendations.sort(key=lambda x: (
            {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}[x.priority],
            x.estimated_improvement
        ), reverse=True)
        
        return recommendations
    
    def _recommend_slow_test_optimizations(self, slow_tests: List[TestPerformanceMetric]) -> List[OptimizationRecommendation]:
        """Generate recommendations for slow tests"""
        recommendations = []
        
        for test in slow_tests:
            if test.duration > self.very_slow_threshold:
                # Very slow tests need immediate attention
                recommendations.append(OptimizationRecommendation(
                    test_id=test.test_id,
                    recommendation_type='refactoring',
                    description=f"Refactor very slow test {test.test_name} ({test.duration:.1f}s)",
                    estimated_improvement=test.duration * 0.7,  # Assume 70% improvement possible
                    implementation_effort='high',
                    priority='critical',
                    code_examples=[
                        "# Consider breaking down into smaller, focused tests",
                        "# Use mocking for external dependencies",
                        "# Optimize database operations with fixtures"
                    ]
                ))
                
                # Also recommend caching
                recommendations.append(OptimizationRecommendation(
                    test_id=test.test_id,
                    recommendation_type='caching',
                    description=f"Implement caching for expensive operations in {test.test_name}",
                    estimated_improvement=test.duration * 0.5,
                    implementation_effort='medium',
                    priority='high',
                    code_examples=[
                        "@pytest.fixture(scope='session')",
                        "def expensive_setup():",
                        "    # Cache expensive setup operations"
                    ]
                ))
            
            elif test.duration > self.slow_test_threshold:
                # Moderately slow tests
                recommendations.append(OptimizationRecommendation(
                    test_id=test.test_id,
                    recommendation_type='mocking',
                    description=f"Add mocking to reduce dependencies in {test.test_name} ({test.duration:.1f}s)",
                    estimated_improvement=test.duration * 0.4,
                    implementation_effort='medium',
                    priority='medium',
                    code_examples=[
                        "@patch('module.expensive_function')",
                        "def test_function(mock_expensive):",
                        "    mock_expensive.return_value = 'mocked_result'"
                    ]
                ))
        
        return recommendations
    
    def _recommend_regression_fixes(self, regressions: List[PerformanceRegression]) -> List[OptimizationRecommendation]:
        """Generate recommendations for performance regressions"""
        recommendations = []
        
        for regression in regressions:
            priority = 'critical' if regression.severity == 'severe' else 'high'
            
            recommendations.append(OptimizationRecommendation(
                test_id=regression.test_id,
                recommendation_type='regression_fix',
                description=f"Fix performance regression in {regression.test_id} "
                           f"({regression.regression_factor:.1f}x slower)",
                estimated_improvement=regression.current_duration - regression.baseline_duration,
                implementation_effort='medium',
                priority=priority,
                code_examples=[
                    "# Review recent changes that may have introduced inefficiencies",
                    "# Profile the test to identify bottlenecks",
                    "# Consider reverting problematic changes"
                ]
            ))
        
        return recommendations
    
    def _recommend_general_optimizations(self, profile: TestPerformanceProfile) -> List[OptimizationRecommendation]:
        """Generate general optimization recommendations"""
        recommendations = []
        
        # Recommend parallelization if many tests
        if profile.test_count > 50 and profile.total_duration > 60:
            recommendations.append(OptimizationRecommendation(
                test_id='GENERAL',
                recommendation_type='parallelization',
                description=f"Enable parallel test execution for {profile.test_count} tests",
                estimated_improvement=profile.total_duration * 0.6,  # Assume 60% improvement
                implementation_effort='low',
                priority='high',
                code_examples=[
                    "# Add to pytest.ini:",
                    "[tool:pytest]",
                    "addopts = -n auto  # Requires pytest-xdist"
                ]
            ))
        
        # Recommend test organization if average duration is high
        if profile.average_duration > 2.0:
            recommendations.append(OptimizationRecommendation(
                test_id='GENERAL',
                recommendation_type='organization',
                description="Reorganize tests to separate fast and slow tests",
                estimated_improvement=profile.total_duration * 0.2,
                implementation_effort='medium',
                priority='medium',
                code_examples=[
                    "# Use pytest markers:",
                    "@pytest.mark.slow",
                    "def test_expensive_operation():",
                    "    pass",
                    "",
                    "# Run fast tests only:",
                    "pytest -m 'not slow'"
                ]
            ))
        
        return recommendations


class TestPerformanceOptimizer:
    """Main system that orchestrates all performance optimization components"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.profiler = TestPerformanceProfiler(project_root)
        self.cache_manager = TestCacheManager(project_root)
        self.regression_detector = PerformanceRegressionDetector(project_root)
        self.recommendation_engine = TestOptimizationRecommendationEngine(project_root)
    
    def optimize_test_performance(self, test_files: List[Path] = None) -> Dict[str, Any]:
        """Run comprehensive test performance optimization"""
        print("Starting test performance optimization...")
        
        # Profile test performance
        print("Profiling test performance...")
        profile = self.profiler.profile_test_suite(test_files)
        
        # Record performance metrics
        print("Recording performance metrics...")
        all_metrics = profile.slow_tests + profile.fast_tests
        self.regression_detector.record_performance(all_metrics)
        
        # Detect regressions
        print("Detecting performance regressions...")
        regressions = self.regression_detector.detect_regressions(all_metrics)
        
        # Generate recommendations
        print("Generating optimization recommendations...")
        recommendations = self.recommendation_engine.generate_recommendations(profile, regressions)
        
        # Get cache statistics
        cache_stats = self.cache_manager.get_cache_stats()
        
        # Compile results
        optimization_result = {
            'performance_profile': asdict(profile),
            'regressions': [asdict(r) for r in regressions],
            'recommendations': [asdict(r) for r in recommendations],
            'cache_statistics': cache_stats,
            'optimization_metadata': {
                'total_tests_analyzed': profile.test_count,
                'slow_tests_found': len(profile.slow_tests),
                'regressions_detected': len(regressions),
                'recommendations_generated': len(recommendations),
                'analysis_timestamp': datetime.now().isoformat()
            }
        }
        
        return optimization_result
    
    def save_optimization_report(self, optimization_result: Dict[str, Any], output_path: Path):
        """Save optimization report to file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(optimization_result, f, indent=2, default=str)
        
        print(f"Performance optimization report saved to {output_path}")


def main():
    """Main entry point for test performance optimizer"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test performance optimization system")
    parser.add_argument('--project-root', type=Path, default=Path.cwd(), help='Project root directory')
    parser.add_argument('--test-files', nargs='*', help='Specific test files to analyze')
    parser.add_argument('--output', type=Path, help='Output file for optimization report')
    parser.add_argument('--slow-threshold', type=float, default=5.0, help='Slow test threshold in seconds')
    parser.add_argument('--clear-cache', action='store_true', help='Clear test cache before analysis')
    
    args = parser.parse_args()
    
    # Setup optimizer
    optimizer = TestPerformanceOptimizer(args.project_root)
    optimizer.profiler.slow_test_threshold = args.slow_threshold
    
    # Clear cache if requested
    if args.clear_cache:
        optimizer.cache_manager.clear_cache()
        print("Test cache cleared.")
    
    # Prepare test files
    test_files = None
    if args.test_files:
        test_files = [Path(f) for f in args.test_files]
    
    # Run optimization
    result = optimizer.optimize_test_performance(test_files)
    
    # Save report
    if args.output:
        optimizer.save_optimization_report(result, args.output)
    else:
        # Save to default location
        default_output = args.project_root / '.test_performance' / f'optimization_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        optimizer.save_optimization_report(result, default_output)
    
    # Print summary
    profile = result['performance_profile']
    regressions = result['regressions']
    recommendations = result['recommendations']
    
    print(f"\n=== Performance Optimization Summary ===")
    print(f"Total tests analyzed: {profile['test_count']}")
    print(f"Total execution time: {profile['total_duration']:.1f}s")
    print(f"Average test duration: {profile['average_duration']:.2f}s")
    print(f"Slow tests (>{args.slow_threshold}s): {len(profile['slow_tests'])}")
    print(f"Performance regressions: {len(regressions)}")
    print(f"Optimization recommendations: {len(recommendations)}")
    
    if profile['slow_tests']:
        print(f"\nSlowest tests:")
        for test in profile['slow_tests'][:5]:  # Show top 5
            print(f"  {test['test_name']}: {test['duration']:.1f}s")
    
    if regressions:
        print(f"\nPerformance regressions:")
        for regression in regressions[:3]:  # Show top 3
            print(f"  {regression['test_id']}: {regression['regression_factor']:.1f}x slower ({regression['severity']})")
    
    if recommendations:
        print(f"\nTop recommendations:")
        for rec in recommendations[:3]:  # Show top 3
            print(f"  {rec['recommendation_type']}: {rec['description']}")
            print(f"    Estimated improvement: {rec['estimated_improvement']:.1f}s")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())