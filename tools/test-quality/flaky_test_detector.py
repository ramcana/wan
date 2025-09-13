import pytest
from unittest.mock import Mock, patch
#!/usr/bin/env python3
"""
Flaky Test Detection and Management System

Implements statistical analysis to identify intermittently failing tests,
creates flaky test tracking and reporting dashboard, implements automatic
test quarantine for consistently flaky tests, and adds fix recommendations
based on failure patterns.
"""

import json
import sqlite3
import subprocess
import sys
import statistics
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict, Counter
import hashlib


@dataclass
class TestExecution:
    """Single test execution record"""
    test_id: str
    test_file: str
    test_name: str
    status: str  # 'passed', 'failed', 'skipped', 'error'
    duration: float
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    timestamp: Optional[datetime] = None
    commit_hash: Optional[str] = None
    branch: Optional[str] = None
    environment: Optional[str] = None


@dataclass
class FlakyTestPattern:
    """Pattern analysis for a flaky test"""
    test_id: str
    total_runs: int
    passed_runs: int
    failed_runs: int
    skipped_runs: int
    error_runs: int
    flakiness_score: float  # 0.0 = stable, 1.0 = completely flaky
    failure_rate: float
    common_errors: List[Tuple[str, int]]  # (error_type, count)
    failure_patterns: Dict[str, Any]
    first_seen: datetime
    last_seen: datetime
    confidence: float


@dataclass
class FlakyTestRecommendation:
    """Recommendation for fixing a flaky test"""
    test_id: str
    recommendation_type: str  # 'timing', 'isolation', 'mocking', 'environment', 'refactoring'
    description: str
    confidence: float
    implementation_effort: str  # 'low', 'medium', 'high'
    priority: str  # 'low', 'medium', 'high', 'critical'
    code_examples: List[str] = None
    related_patterns: List[str] = None


@dataclass
class QuarantineDecision:
    """Decision about quarantining a flaky test"""
    test_id: str
    should_quarantine: bool
    reason: str
    quarantine_duration: Optional[int] = None  # days
    conditions_for_release: List[str] = None


class FlakyTestStatisticalAnalyzer:
    """Performs statistical analysis to identify flaky tests"""
    
    def __init__(self):
        self.min_runs_for_analysis = 5
        self.flakiness_threshold = 0.1  # 10% failure rate
        self.high_flakiness_threshold = 0.3  # 30% failure rate
        self.confidence_threshold = 0.8
    
    def analyze_test_flakiness(self, executions: List[TestExecution]) -> List[FlakyTestPattern]:
        """Analyze test executions to identify flaky patterns"""
        # Group executions by test
        test_groups = defaultdict(list)
        for execution in executions:
            test_groups[execution.test_id].append(execution)
        
        flaky_patterns = []
        
        for test_id, test_executions in test_groups.items():
            if len(test_executions) < self.min_runs_for_analysis:
                continue
            
            pattern = self._analyze_single_test(test_id, test_executions)
            
            # Only include tests that show flaky behavior
            if pattern.flakiness_score >= self.flakiness_threshold:
                flaky_patterns.append(pattern)
        
        # Sort by flakiness score (most flaky first)
        flaky_patterns.sort(key=lambda x: x.flakiness_score, reverse=True)
        
        return flaky_patterns
    
    def _analyze_single_test(self, test_id: str, executions: List[TestExecution]) -> FlakyTestPattern:
        """Analyze flakiness for a single test"""
        # Count outcomes
        status_counts = Counter(ex.status for ex in executions)
        total_runs = len(executions)
        passed_runs = status_counts.get('passed', 0)
        failed_runs = status_counts.get('failed', 0)
        skipped_runs = status_counts.get('skipped', 0)
        error_runs = status_counts.get('error', 0)
        
        # Calculate failure rate (excluding skipped tests)
        non_skipped_runs = total_runs - skipped_runs
        failure_rate = (failed_runs + error_runs) / non_skipped_runs if non_skipped_runs > 0 else 0
        
        # Calculate flakiness score
        flakiness_score = self._calculate_flakiness_score(executions)
        
        # Analyze error patterns
        common_errors = self._analyze_error_patterns(executions)
        
        # Analyze failure patterns
        failure_patterns = self._analyze_failure_patterns(executions)
        
        # Calculate confidence based on sample size and consistency
        confidence = self._calculate_confidence(executions, flakiness_score)
        
        # Get time range
        timestamps = [ex.timestamp for ex in executions if ex.timestamp]
        first_seen = min(timestamps) if timestamps else datetime.now()
        last_seen = max(timestamps) if timestamps else datetime.now()
        
        return FlakyTestPattern(
            test_id=test_id,
            total_runs=total_runs,
            passed_runs=passed_runs,
            failed_runs=failed_runs,
            skipped_runs=skipped_runs,
            error_runs=error_runs,
            flakiness_score=flakiness_score,
            failure_rate=failure_rate,
            common_errors=common_errors,
            failure_patterns=failure_patterns,
            first_seen=first_seen,
            last_seen=last_seen,
            confidence=confidence
        )
    
    def _calculate_flakiness_score(self, executions: List[TestExecution]) -> float:
        """Calculate flakiness score based on execution patterns"""
        if len(executions) < 2:
            return 0.0
        
        # Sort by timestamp
        sorted_executions = sorted(executions, key=lambda x: x.timestamp or datetime.min)
        
        # Count status transitions
        transitions = 0
        for i in range(1, len(sorted_executions)):
            prev_status = sorted_executions[i-1].status
            curr_status = sorted_executions[i].status
            
            # Count transitions between passed/failed (ignore skipped)
            if prev_status in ['passed', 'failed', 'error'] and curr_status in ['passed', 'failed', 'error']:
                if prev_status != curr_status:
                    transitions += 1
        
        # Calculate flakiness based on transitions and failure rate
        max_possible_transitions = len(sorted_executions) - 1
        transition_score = transitions / max_possible_transitions if max_possible_transitions > 0 else 0
        
        # Combine with failure rate
        status_counts = Counter(ex.status for ex in executions)
        non_skipped = len(executions) - status_counts.get('skipped', 0)
        failure_rate = (status_counts.get('failed', 0) + status_counts.get('error', 0)) / non_skipped if non_skipped > 0 else 0
        
        # Flakiness is high when there are many transitions and moderate failure rate
        # Pure failures (100% fail rate) are not flaky, they're just broken
        if failure_rate >= 0.95:  # Almost always fails
            return failure_rate * 0.3  # Lower flakiness score for consistently failing tests
        elif failure_rate <= 0.05:  # Almost always passes
            return failure_rate * 2  # Very low flakiness
        else:
            # Moderate failure rate with transitions indicates flakiness
            return min(1.0, transition_score + failure_rate * 0.7)
    
    def _analyze_error_patterns(self, executions: List[TestExecution]) -> List[Tuple[str, int]]:
        """Analyze common error patterns"""
        error_types = Counter()
        
        for execution in executions:
            if execution.status in ['failed', 'error'] and execution.error_type:
                error_types[execution.error_type] += 1
        
        # Return most common errors
        return error_types.most_common(5)
    
    def _analyze_failure_patterns(self, executions: List[TestExecution]) -> Dict[str, Any]:
        """Analyze patterns in test failures"""
        patterns = {
            'time_based': self._analyze_time_patterns(executions),
            'environment_based': self._analyze_environment_patterns(executions),
            'duration_based': self._analyze_duration_patterns(executions),
            'sequence_based': self._analyze_sequence_patterns(executions)
        }
        
        return patterns
    
    def _analyze_time_patterns(self, executions: List[TestExecution]) -> Dict[str, Any]:
        """Analyze time-based failure patterns"""
        failed_executions = [ex for ex in executions if ex.status in ['failed', 'error'] and ex.timestamp]
        
        if not failed_executions:
            return {}
        
        # Analyze by hour of day
        hour_failures = Counter(ex.timestamp.hour for ex in failed_executions)
        
        # Analyze by day of week
        weekday_failures = Counter(ex.timestamp.weekday() for ex in failed_executions)
        
        return {
            'hour_distribution': dict(hour_failures),
            'weekday_distribution': dict(weekday_failures),
            'total_failed': len(failed_executions)
        }
    
    def _analyze_environment_patterns(self, executions: List[TestExecution]) -> Dict[str, Any]:
        """Analyze environment-based failure patterns"""
        env_stats = defaultdict(lambda: {'total': 0, 'failed': 0})
        
        for execution in executions:
            env = execution.environment or 'unknown'
            env_stats[env]['total'] += 1
            if execution.status in ['failed', 'error']:
                env_stats[env]['failed'] += 1
        
        # Calculate failure rates by environment
        env_failure_rates = {}
        for env, stats in env_stats.items():
            failure_rate = stats['failed'] / stats['total'] if stats['total'] > 0 else 0
            env_failure_rates[env] = {
                'failure_rate': failure_rate,
                'total_runs': stats['total'],
                'failed_runs': stats['failed']
            }
        
        return env_failure_rates
    
    def _analyze_duration_patterns(self, executions: List[TestExecution]) -> Dict[str, Any]:
        """Analyze duration-based patterns"""
        passed_durations = [ex.duration for ex in executions if ex.status == 'passed']
        failed_durations = [ex.duration for ex in executions if ex.status in ['failed', 'error']]
        
        patterns = {}
        
        if passed_durations:
            patterns['passed_avg_duration'] = statistics.mean(passed_durations)
            patterns['passed_median_duration'] = statistics.median(passed_durations)
        
        if failed_durations:
            patterns['failed_avg_duration'] = statistics.mean(failed_durations)
            patterns['failed_median_duration'] = statistics.median(failed_durations)
        
        # Check if failed tests tend to be faster (timeout issues) or slower (performance issues)
        if passed_durations and failed_durations:
            patterns['duration_difference'] = statistics.mean(failed_durations) - statistics.mean(passed_durations)
        
        return patterns
    
    def _analyze_sequence_patterns(self, executions: List[TestExecution]) -> Dict[str, Any]:
        """Analyze sequential failure patterns"""
        sorted_executions = sorted(executions, key=lambda x: x.timestamp or datetime.min)
        
        # Look for streaks of failures or passes
        current_streak = 1
        current_status = sorted_executions[0].status if sorted_executions else None
        max_fail_streak = 0
        max_pass_streak = 0
        
        for i in range(1, len(sorted_executions)):
            if sorted_executions[i].status == current_status:
                current_streak += 1
            else:
                if current_status in ['failed', 'error']:
                    max_fail_streak = max(max_fail_streak, current_streak)
                elif current_status == 'passed':
                    max_pass_streak = max(max_pass_streak, current_streak)
                
                current_status = sorted_executions[i].status
                current_streak = 1
        
        # Handle the last streak
        if current_status in ['failed', 'error']:
            max_fail_streak = max(max_fail_streak, current_streak)
        elif current_status == 'passed':
            max_pass_streak = max(max_pass_streak, current_streak)
        
        return {
            'max_failure_streak': max_fail_streak,
            'max_success_streak': max_pass_streak,
            'total_sequences': len(sorted_executions)
        }
    
    def _calculate_confidence(self, executions: List[TestExecution], flakiness_score: float) -> float:
        """Calculate confidence in flakiness assessment"""
        sample_size = len(executions)
        
        # Base confidence on sample size
        size_confidence = min(1.0, sample_size / 20)  # Full confidence at 20+ samples
        
        # Adjust based on flakiness score consistency
        score_confidence = 1.0 - abs(flakiness_score - 0.5) * 0.5  # Higher confidence for moderate flakiness
        
        # Time span confidence (more confident with longer observation period)
        timestamps = [ex.timestamp for ex in executions if ex.timestamp]
        if len(timestamps) >= 2:
            time_span = (max(timestamps) - min(timestamps)).days
            time_confidence = min(1.0, time_span / 7)  # Full confidence after 7 days
        else:
            time_confidence = 0.1
        
        return (size_confidence + score_confidence + time_confidence) / 3


class FlakyTestTracker:
    """Tracks flaky test executions and maintains historical data"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.db_path = project_root / '.flaky_tests' / 'flaky_tests.db'
        self._init_database()
    
    def _init_database(self):
        """Initialize flaky test tracking database"""
        self.db_path.parent.mkdir(exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS test_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT NOT NULL,
                    test_file TEXT NOT NULL,
                    test_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    duration REAL NOT NULL,
                    error_message TEXT,
                    error_type TEXT,
                    timestamp TEXT NOT NULL,
                    commit_hash TEXT,
                    branch TEXT,
                    environment TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS flaky_test_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT NOT NULL UNIQUE,
                    total_runs INTEGER NOT NULL,
                    passed_runs INTEGER NOT NULL,
                    failed_runs INTEGER NOT NULL,
                    skipped_runs INTEGER NOT NULL,
                    error_runs INTEGER NOT NULL,
                    flakiness_score REAL NOT NULL,
                    failure_rate REAL NOT NULL,
                    confidence REAL NOT NULL,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    quarantined BOOLEAN DEFAULT FALSE,
                    quarantine_reason TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS flaky_test_recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT NOT NULL,
                    recommendation_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    implementation_effort TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    applied BOOLEAN DEFAULT FALSE
                )
            ''')
            
            conn.execute('CREATE INDEX IF NOT EXISTS idx_test_id ON test_executions(test_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON test_executions(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_status ON test_executions(status)')
            
            conn.commit()
    
    def record_test_execution(self, execution: TestExecution):
        """Record a single test execution"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO test_executions
                (test_id, test_file, test_name, status, duration, error_message, error_type,
                 timestamp, commit_hash, branch, environment)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                execution.test_id,
                execution.test_file,
                execution.test_name,
                execution.status,
                execution.duration,
                execution.error_message,
                execution.error_type,
                execution.timestamp.isoformat() if execution.timestamp else datetime.now().isoformat(),
                execution.commit_hash,
                execution.branch,
                execution.environment
            ))
            conn.commit()
    
    def record_test_executions(self, executions: List[TestExecution]):
        """Record multiple test executions"""
        with sqlite3.connect(self.db_path) as conn:
            for execution in executions:
                conn.execute('''
                    INSERT INTO test_executions
                    (test_id, test_file, test_name, status, duration, error_message, error_type,
                     timestamp, commit_hash, branch, environment)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    execution.test_id,
                    execution.test_file,
                    execution.test_name,
                    execution.status,
                    execution.duration,
                    execution.error_message,
                    execution.error_type,
                    execution.timestamp.isoformat() if execution.timestamp else datetime.now().isoformat(),
                    execution.commit_hash,
                    execution.branch,
                    execution.environment
                ))
            conn.commit()
    
    def get_test_executions(self, test_id: str = None, days: int = 30) -> List[TestExecution]:
        """Get test executions from the database"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            if test_id:
                cursor = conn.execute('''
                    SELECT test_id, test_file, test_name, status, duration, error_message, error_type,
                           timestamp, commit_hash, branch, environment
                    FROM test_executions
                    WHERE test_id = ? AND timestamp >= ?
                    ORDER BY timestamp ASC
                ''', (test_id, cutoff_date.isoformat()))
            else:
                cursor = conn.execute('''
                    SELECT test_id, test_file, test_name, status, duration, error_message, error_type,
                           timestamp, commit_hash, branch, environment
                    FROM test_executions
                    WHERE timestamp >= ?
                    ORDER BY timestamp ASC
                ''', (cutoff_date.isoformat(),))
            
            executions = []
            for row in cursor.fetchall():
                executions.append(TestExecution(
                    test_id=row[0],
                    test_file=row[1],
                    test_name=row[2],
                    status=row[3],
                    duration=row[4],
                    error_message=row[5],
                    error_type=row[6],
                    timestamp=datetime.fromisoformat(row[7]),
                    commit_hash=row[8],
                    branch=row[9],
                    environment=row[10]
                ))
            
            return executions
    
    def update_flaky_pattern(self, pattern: FlakyTestPattern):
        """Update or insert flaky test pattern"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO flaky_test_patterns
                (test_id, total_runs, passed_runs, failed_runs, skipped_runs, error_runs,
                 flakiness_score, failure_rate, confidence, first_seen, last_seen, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                pattern.test_id,
                pattern.total_runs,
                pattern.passed_runs,
                pattern.failed_runs,
                pattern.skipped_runs,
                pattern.error_runs,
                pattern.flakiness_score,
                pattern.failure_rate,
                pattern.confidence,
                pattern.first_seen.isoformat(),
                pattern.last_seen.isoformat(),
                datetime.now().isoformat()
            ))
            conn.commit()
    
    def get_flaky_patterns(self, min_flakiness: float = 0.1) -> List[FlakyTestPattern]:
        """Get flaky test patterns from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT test_id, total_runs, passed_runs, failed_runs, skipped_runs, error_runs,
                       flakiness_score, failure_rate, confidence, first_seen, last_seen,
                       quarantined, quarantine_reason
                FROM flaky_test_patterns
                WHERE flakiness_score >= ?
                ORDER BY flakiness_score DESC
            ''', (min_flakiness,))
            
            patterns = []
            for row in cursor.fetchall():
                # Note: We're not loading full failure patterns from DB for simplicity
                pattern = FlakyTestPattern(
                    test_id=row[0],
                    total_runs=row[1],
                    passed_runs=row[2],
                    failed_runs=row[3],
                    skipped_runs=row[4],
                    error_runs=row[5],
                    flakiness_score=row[6],
                    failure_rate=row[7],
                    confidence=row[8],
                    first_seen=datetime.fromisoformat(row[9]),
                    last_seen=datetime.fromisoformat(row[10]),
                    common_errors=[],  # Would need separate table for full implementation
                    failure_patterns={}  # Would need separate table for full implementation
                )
                patterns.append(pattern)
            
            return patterns


class FlakyTestRecommendationEngine:
    """Generates recommendations for fixing flaky tests"""
    
    def __init__(self):
        self.error_pattern_recommendations = {
            'TimeoutError': {
                'type': 'timing',
                'description': 'Add explicit waits or increase timeout values',
                'effort': 'low',
                'examples': [
                    'import time',
                    'time.sleep(0.1)  # Add small delay',
                    'WebDriverWait(driver, 10).until(condition)'
                ]
            },
            'ConnectionError': {
                'type': 'mocking',
                'description': 'Mock external service connections',
                'effort': 'medium',
                'examples': [
                    '@patch("requests.get")',
                    'def test_api_call(mock_get):',
                    '    mock_get.return_value.status_code = 200'
                ]
            },
            'FileNotFoundError': {
                'type': 'isolation',
                'description': 'Ensure proper test isolation and cleanup',
                'effort': 'medium',
                'examples': [
                    '@pytest.fixture(autouse=True)',
                    'def cleanup_files():',
                    '    yield',
                    '    # Cleanup code here'
                ]
            },
            'AssertionError': {
                'type': 'refactoring',
                'description': 'Review test logic for race conditions',
                'effort': 'high',
                'examples': [
                    '# Instead of exact equality, use ranges for timing-sensitive tests',
                    'assert abs(result - expected) < tolerance'
                ]
            }
        }
    
    def generate_recommendations(self, patterns: List[FlakyTestPattern]) -> List[FlakyTestRecommendation]:
        """Generate recommendations for fixing flaky tests"""
        recommendations = []
        
        for pattern in patterns:
            test_recommendations = self._analyze_pattern_for_recommendations(pattern)
            recommendations.extend(test_recommendations)
        
        # Sort by priority and confidence
        recommendations.sort(key=lambda x: (
            {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}[x.priority],
            x.confidence
        ), reverse=True)
        
        return recommendations
    
    def _analyze_pattern_for_recommendations(self, pattern: FlakyTestPattern) -> List[FlakyTestRecommendation]:
        """Analyze a single flaky pattern to generate recommendations"""
        recommendations = []
        
        # Recommendations based on common errors
        for error_type, count in pattern.common_errors:
            if error_type in self.error_pattern_recommendations:
                template = self.error_pattern_recommendations[error_type]
                
                recommendations.append(FlakyTestRecommendation(
                    test_id=pattern.test_id,
                    recommendation_type=template['type'],
                    description=f"{template['description']} (seen {count} times)",
                    confidence=min(0.9, count / pattern.total_runs + 0.3),
                    implementation_effort=template['effort'],
                    priority=self._determine_priority(pattern, count / pattern.total_runs),
                    code_examples=template['examples'],
                    related_patterns=[error_type]
                ))
        
        # Recommendations based on failure patterns
        if pattern.failure_patterns:
            recommendations.extend(self._analyze_failure_patterns_for_recommendations(pattern))
        
        # General recommendations based on flakiness score
        if pattern.flakiness_score > 0.5:
            recommendations.append(FlakyTestRecommendation(
                test_id=pattern.test_id,
                recommendation_type='refactoring',
                description=f"High flakiness score ({pattern.flakiness_score:.2f}) - consider complete test rewrite",
                confidence=0.8,
                implementation_effort='high',
                priority='high',
                code_examples=[
                    '# Break down into smaller, more focused tests',
                    '# Remove dependencies on external state',
                    '# Use deterministic test data'
                ]
            ))
        
        return recommendations
    
    def _analyze_failure_patterns_for_recommendations(self, pattern: FlakyTestPattern) -> List[FlakyTestRecommendation]:
        """Generate recommendations based on failure patterns"""
        recommendations = []
        
        # Time-based patterns
        time_patterns = pattern.failure_patterns.get('time_based', {})
        if time_patterns:
            hour_dist = time_patterns.get('hour_distribution', {})
            if hour_dist:
                # Check if failures cluster around certain hours
                max_hour_failures = max(hour_dist.values()) if hour_dist else 0
                total_failures = sum(hour_dist.values())
                
                if max_hour_failures > total_failures * 0.3:  # 30% of failures in one hour
                    recommendations.append(FlakyTestRecommendation(
                        test_id=pattern.test_id,
                        recommendation_type='timing',
                        description="Failures cluster at specific times - check for time-dependent logic",
                        confidence=0.7,
                        implementation_effort='medium',
                        priority='medium',
                        code_examples=[
                            '# Use fixed time in tests',
                            'with freeze_time("2023-01-01 12:00:00"):',
                            '    result = time_dependent_function()'
                        ]
                    ))
        
        # Environment-based patterns
        env_patterns = pattern.failure_patterns.get('environment_based', {})
        if env_patterns:
            # Check if certain environments have higher failure rates
            env_failure_rates = [(env, data['failure_rate']) for env, data in env_patterns.items()]
            env_failure_rates.sort(key=lambda x: x[1], reverse=True)
            
            if env_failure_rates and env_failure_rates[0][1] > 0.5:
                recommendations.append(FlakyTestRecommendation(
                    test_id=pattern.test_id,
                    recommendation_type='environment',
                    description=f"High failure rate in {env_failure_rates[0][0]} environment",
                    confidence=0.8,
                    implementation_effort='medium',
                    priority='high',
                    code_examples=[
                        '# Add environment-specific configuration',
                        'if os.environ.get("TEST_ENV") == "ci":',
                        '    timeout = 30  # Longer timeout for CI'
                    ]
                ))
        
        # Duration-based patterns
        duration_patterns = pattern.failure_patterns.get('duration_based', {})
        if duration_patterns:
            duration_diff = duration_patterns.get('duration_difference', 0)
            if abs(duration_diff) > 1.0:  # Significant duration difference
                if duration_diff > 0:
                    # Failed tests take longer - possible timeout issue
                    recommendations.append(FlakyTestRecommendation(
                        test_id=pattern.test_id,
                        recommendation_type='timing',
                        description="Failed tests take longer - possible timeout or performance issue",
                        confidence=0.6,
                        implementation_effort='medium',
                        priority='medium',
                        code_examples=[
                            '# Increase timeout or optimize slow operations',
                            '@pytest.mark.timeout(30)',
                            'def test_slow_operation():'
                        ]
                    ))
                else:
                    # Failed tests are faster - possible early termination
                    recommendations.append(FlakyTestRecommendation(
                        test_id=pattern.test_id,
                        recommendation_type='isolation',
                        description="Failed tests terminate early - check for missing setup or race conditions",
                        confidence=0.7,
                        implementation_effort='medium',
                        priority='medium',
                        code_examples=[
                            '# Ensure proper setup before assertions',
                            'setup_complete.wait()',
                            'assert precondition_met()'
                        ]
                    ))
        
        return recommendations
    
    def _determine_priority(self, pattern: FlakyTestPattern, error_frequency: float) -> str:
        """Determine priority based on pattern characteristics"""
        if pattern.flakiness_score > 0.7 or error_frequency > 0.5:
            return 'critical'
        elif pattern.flakiness_score > 0.4 or error_frequency > 0.3:
            return 'high'
        elif pattern.flakiness_score > 0.2 or error_frequency > 0.1:
            return 'medium'
        else:
            return 'low'


class FlakyTestQuarantineManager:
    """Manages quarantine decisions for flaky tests"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.tracker = FlakyTestTracker(project_root)
        self.quarantine_threshold = 0.4  # Quarantine tests with >40% flakiness
        self.severe_quarantine_threshold = 0.7  # Immediate quarantine for >70% flakiness
    
    def evaluate_quarantine_decisions(self, patterns: List[FlakyTestPattern]) -> List[QuarantineDecision]:
        """Evaluate which tests should be quarantined"""
        decisions = []
        
        for pattern in patterns:
            decision = self._evaluate_single_test_quarantine(pattern)
            decisions.append(decision)
        
        return decisions
    
    def _evaluate_single_test_quarantine(self, pattern: FlakyTestPattern) -> QuarantineDecision:
        """Evaluate quarantine decision for a single test"""
        should_quarantine = False
        reason = ""
        quarantine_duration = None
        conditions = []
        
        # Immediate quarantine for severely flaky tests
        if pattern.flakiness_score >= self.severe_quarantine_threshold:
            should_quarantine = True
            reason = f"Severe flakiness ({pattern.flakiness_score:.2f}) - immediate quarantine required"
            quarantine_duration = 14  # 2 weeks
            conditions = [
                "Flakiness score reduced below 0.3",
                "At least 10 consecutive successful runs",
                "Root cause identified and fixed"
            ]
        
        # Regular quarantine for moderately flaky tests
        elif pattern.flakiness_score >= self.quarantine_threshold:
            # Additional checks for quarantine decision
            if pattern.confidence >= 0.7 and pattern.total_runs >= 10:
                should_quarantine = True
                reason = f"Consistent flakiness ({pattern.flakiness_score:.2f}) with high confidence"
                quarantine_duration = 7  # 1 week
                conditions = [
                    "Flakiness score reduced below 0.2",
                    "At least 5 consecutive successful runs"
                ]
        
        # Check for recent regression
        elif pattern.failure_rate > 0.8 and pattern.total_runs >= 5:
            # High recent failure rate might indicate a regression
            should_quarantine = True
            reason = f"High recent failure rate ({pattern.failure_rate:.2f}) - possible regression"
            quarantine_duration = 3  # 3 days
            conditions = [
                "Failure rate reduced below 0.1",
                "Recent changes reviewed and fixed"
            ]
        
        # No quarantine needed
        else:
            reason = f"Flakiness score ({pattern.flakiness_score:.2f}) below quarantine threshold"
        
        return QuarantineDecision(
            test_id=pattern.test_id,
            should_quarantine=should_quarantine,
            reason=reason,
            quarantine_duration=quarantine_duration,
            conditions_for_release=conditions
        )
    
    def apply_quarantine(self, decision: QuarantineDecision):
        """Apply quarantine decision to a test"""
        if decision.should_quarantine:
            # Update database to mark test as quarantined
            with sqlite3.connect(self.tracker.db_path) as conn:
                conn.execute('''
                    UPDATE flaky_test_patterns
                    SET quarantined = TRUE, quarantine_reason = ?
                    WHERE test_id = ?
                ''', (decision.reason, decision.test_id))
                conn.commit()
            
            # Create pytest marker file for quarantined tests
            self._create_quarantine_marker(decision)
    
    def _create_quarantine_marker(self, decision: QuarantineDecision):
        """Create pytest marker for quarantined test"""
        quarantine_file = self.project_root / '.flaky_tests' / 'quarantined_tests.txt'
        quarantine_file.parent.mkdir(exist_ok=True)
        
        # Read existing quarantined tests
        quarantined_tests = set()
        if quarantine_file.exists():
            with open(quarantine_file, 'r') as f:
                quarantined_tests = set(line.strip() for line in f if line.strip())
        
        # Add new quarantined test
        quarantined_tests.add(decision.test_id)
        
        # Write back to file
        with open(quarantine_file, 'w') as f:
            for test_id in sorted(quarantined_tests):
                f.write(f"{test_id}\n")


class FlakyTestDetectionSystem:
    """Main system that orchestrates flaky test detection and management"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.analyzer = FlakyTestStatisticalAnalyzer()
        self.tracker = FlakyTestTracker(project_root)
        self.recommendation_engine = FlakyTestRecommendationEngine()
        self.quarantine_manager = FlakyTestQuarantineManager(project_root)
    
    def run_flaky_test_analysis(self, days: int = 30) -> Dict[str, Any]:
        """Run comprehensive flaky test analysis"""
        print("Starting flaky test analysis...")
        
        # Get test execution data
        print("Loading test execution history...")
        executions = self.tracker.get_test_executions(days=days)
        
        if not executions:
            print("No test execution data found.")
            return {
                'flaky_patterns': [],
                'recommendations': [],
                'quarantine_decisions': [],
                'analysis_metadata': {
                    'executions_analyzed': 0,
                    'analysis_timestamp': datetime.now().isoformat()
                }
            }
        
        # Analyze flakiness patterns
        print("Analyzing flakiness patterns...")
        patterns = self.analyzer.analyze_test_flakiness(executions)
        
        # Update patterns in database
        for pattern in patterns:
            self.tracker.update_flaky_pattern(pattern)
        
        # Generate recommendations
        print("Generating fix recommendations...")
        recommendations = self.recommendation_engine.generate_recommendations(patterns)
        
        # Evaluate quarantine decisions
        print("Evaluating quarantine decisions...")
        quarantine_decisions = self.quarantine_manager.evaluate_quarantine_decisions(patterns)
        
        # Apply quarantine decisions
        for decision in quarantine_decisions:
            if decision.should_quarantine:
                self.quarantine_manager.apply_quarantine(decision)
        
        # Compile results
        analysis_result = {
            'flaky_patterns': [asdict(p) for p in patterns],
            'recommendations': [asdict(r) for r in recommendations],
            'quarantine_decisions': [asdict(d) for d in quarantine_decisions],
            'analysis_metadata': {
                'executions_analyzed': len(executions),
                'flaky_tests_found': len(patterns),
                'recommendations_generated': len(recommendations),
                'tests_quarantined': len([d for d in quarantine_decisions if d.should_quarantine]),
                'analysis_timestamp': datetime.now().isoformat()
            }
        }
        
        return analysis_result
    
    def record_test_run_results(self, test_results_file: Path):
        """Record test run results from pytest output"""
        print(f"Recording test results from {test_results_file}...")
        
        # Parse test results and create execution records
        executions = self._parse_test_results(test_results_file)
        
        # Record in database
        self.tracker.record_test_executions(executions)
        
        print(f"Recorded {len(executions)} test executions.")
    
    def _parse_test_results(self, results_file: Path) -> List[TestExecution]:
        """Parse test results from file"""
        executions = []
        
        try:
            # This is a simplified parser - in practice, you'd want to parse
            # actual pytest output formats (JUnit XML, JSON, etc.)
            with open(results_file, 'r') as f:
                content = f.read()
            
            # Extract basic test information
            # This would need to be adapted based on your actual test output format
            lines = content.split('\n')
            for line in lines:
                if '::' in line and any(status in line for status in ['PASSED', 'FAILED', 'SKIPPED', 'ERROR']):
                    parts = line.split()
                    if len(parts) >= 2:
                        test_path = parts[0]
                        status = 'passed' if 'PASSED' in line else 'failed' if 'FAILED' in line else 'skipped'
                        
                        if '::' in test_path:
                            file_part, test_name = test_path.split('::', 1)
                            
                            execution = TestExecution(
                                test_id=test_path,
                                test_file=file_part,
                                test_name=test_name,
                                status=status,
                                duration=1.0,  # Would extract from actual output
                                timestamp=datetime.now(),
                                commit_hash=self._get_current_commit(),
                                branch=self._get_current_branch()
                            )
                            executions.append(execution)
        
        except Exception as e:
            print(f"Error parsing test results: {e}")
        
        return executions
    
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
    
    def save_analysis_report(self, analysis_result: Dict[str, Any], output_path: Path):
        """Save flaky test analysis report"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(analysis_result, f, indent=2, default=str)
        
        print(f"Flaky test analysis report saved to {output_path}")


def main():
    """Main entry point for flaky test detector"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Flaky test detection and management system")
    parser.add_argument('--project-root', type=Path, default=Path.cwd(), help='Project root directory')
    parser.add_argument('--days', type=int, default=30, help='Days of history to analyze')
    parser.add_argument('--output', type=Path, help='Output file for analysis report')
    parser.add_argument('--record-results', type=Path, help='Record test results from file')
    parser.add_argument('--flakiness-threshold', type=float, default=0.1, help='Flakiness threshold')
    
    args = parser.parse_args()
    
    # Setup system
    system = FlakyTestDetectionSystem(args.project_root)
    system.analyzer.flakiness_threshold = args.flakiness_threshold
    
    # Record test results if provided
    if args.record_results:
        system.record_test_run_results(args.record_results)
    
    # Run analysis
    result = system.run_flaky_test_analysis(args.days)
    
    # Save report
    if args.output:
        system.save_analysis_report(result, args.output)
    else:
        # Save to default location
        default_output = args.project_root / '.flaky_tests' / f'flaky_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        system.save_analysis_report(result, default_output)
    
    # Print summary
    metadata = result['analysis_metadata']
    patterns = result['flaky_patterns']
    recommendations = result['recommendations']
    quarantine_decisions = result['quarantine_decisions']
    
    print(f"\n=== Flaky Test Analysis Summary ===")
    print(f"Executions analyzed: {metadata['executions_analyzed']}")
    print(f"Flaky tests found: {metadata['flaky_tests_found']}")
    print(f"Recommendations generated: {metadata['recommendations_generated']}")
    print(f"Tests quarantined: {metadata['tests_quarantined']}")
    
    if patterns:
        print(f"\nTop flaky tests:")
        for pattern in patterns[:5]:  # Show top 5
            print(f"  {pattern['test_id']}: {pattern['flakiness_score']:.2f} flakiness score")
            print(f"    Failure rate: {pattern['failure_rate']:.1%}")
            print(f"    Total runs: {pattern['total_runs']}")
    
    if recommendations:
        print(f"\nTop recommendations:")
        for rec in recommendations[:3]:  # Show top 3
            print(f"  {rec['recommendation_type']}: {rec['description']}")
            print(f"    Priority: {rec['priority']}, Effort: {rec['implementation_effort']}")
    
    quarantined = [d for d in quarantine_decisions if d['should_quarantine']]
    if quarantined:
        print(f"\nQuarantined tests:")
        for decision in quarantined:
            print(f"  {decision['test_id']}: {decision['reason']}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
