#!/usr/bin/env python3
"""
Comprehensive Test Coverage Analysis System

Enhanced coverage reporting system that identifies untested code paths,
enforces coverage thresholds for new code, generates detailed reports
with actionable recommendations, and tracks coverage trends over time.
"""

import json
import sqlite3
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import hashlib
import os

# Import existing coverage analyzer
sys.path.append(str(Path(__file__).parent.parent / 'test-auditor'))
from coverage_analyzer import CoverageAnalyzer, CoverageReport, FileCoverage, CoverageGap


@dataclass
class CoverageTrend:
    """Coverage trend data point"""
    timestamp: datetime
    overall_percentage: float
    total_files: int
    covered_files: int
    total_lines: int
    covered_lines: int
    commit_hash: Optional[str] = None
    branch: Optional[str] = None


@dataclass
class NewCodeCoverage:
    """Coverage analysis for new/changed code"""
    file_path: str
    new_lines: List[int]
    covered_new_lines: List[int]
    uncovered_new_lines: List[int]
    new_code_coverage: float
    meets_threshold: bool
    threshold_required: float


@dataclass
class CoverageThresholdResult:
    """Result of coverage threshold enforcement"""
    passed: bool
    overall_coverage: float
    required_threshold: float
    new_code_results: List[NewCodeCoverage]
    violations: List[str]
    recommendations: List[str]


class CoverageTrendTracker:
    """Tracks coverage trends over time"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.db_path = project_root / '.coverage_analysis' / 'trends.db'
        self._init_database()
    
    def _init_database(self):
        """Initialize the trends database"""
        self.db_path.parent.mkdir(exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS coverage_trends (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    overall_percentage REAL NOT NULL,
                    total_files INTEGER NOT NULL,
                    covered_files INTEGER NOT NULL,
                    total_lines INTEGER NOT NULL,
                    covered_lines INTEGER NOT NULL,
                    commit_hash TEXT,
                    branch TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS file_coverage_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trend_id INTEGER NOT NULL,
                    file_path TEXT NOT NULL,
                    coverage_percentage REAL NOT NULL,
                    total_lines INTEGER NOT NULL,
                    covered_lines INTEGER NOT NULL,
                    FOREIGN KEY (trend_id) REFERENCES coverage_trends (id)
                )
            ''')
            
            conn.commit()
    
    def record_coverage(self, report: CoverageReport) -> int:
        """Record coverage data point"""
        commit_hash = self._get_current_commit()
        branch = self._get_current_branch()
        
        trend = CoverageTrend(
            timestamp=datetime.now(),
            overall_percentage=report.overall_percentage,
            total_files=report.total_files,
            covered_files=report.covered_files,
            total_lines=report.total_lines,
            covered_lines=report.covered_lines,
            commit_hash=commit_hash,
            branch=branch
        )
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Insert trend record
            cursor.execute('''
                INSERT INTO coverage_trends 
                (timestamp, overall_percentage, total_files, covered_files, 
                 total_lines, covered_lines, commit_hash, branch)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trend.timestamp.isoformat(),
                trend.overall_percentage,
                trend.total_files,
                trend.covered_files,
                trend.total_lines,
                trend.covered_lines,
                trend.commit_hash,
                trend.branch
            ))
            
            trend_id = cursor.lastrowid
            
            # Insert file coverage history
            for file_cov in report.file_coverages:
                cursor.execute('''
                    INSERT INTO file_coverage_history
                    (trend_id, file_path, coverage_percentage, total_lines, covered_lines)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    trend_id,
                    file_cov.file_path,
                    file_cov.coverage_percentage,
                    file_cov.total_lines,
                    file_cov.covered_lines
                ))
            
            conn.commit()
            return trend_id
    
    def get_trends(self, days: int = 30) -> List[CoverageTrend]:
        """Get coverage trends for the last N days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp, overall_percentage, total_files, covered_files,
                       total_lines, covered_lines, commit_hash, branch
                FROM coverage_trends
                WHERE timestamp >= ?
                ORDER BY timestamp ASC
            ''', (cutoff_date.isoformat(),))
            
            trends = []
            for row in cursor.fetchall():
                trends.append(CoverageTrend(
                    timestamp=datetime.fromisoformat(row[0]),
                    overall_percentage=row[1],
                    total_files=row[2],
                    covered_files=row[3],
                    total_lines=row[4],
                    covered_lines=row[5],
                    commit_hash=row[6],
                    branch=row[7]
                ))
            
            return trends
    
    def get_file_trends(self, file_path: str, days: int = 30) -> List[Tuple[datetime, float]]:
        """Get coverage trends for a specific file"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT ct.timestamp, fch.coverage_percentage
                FROM coverage_trends ct
                JOIN file_coverage_history fch ON ct.id = fch.trend_id
                WHERE fch.file_path = ? AND ct.timestamp >= ?
                ORDER BY ct.timestamp ASC
            ''', (file_path, cutoff_date.isoformat()))
            
            return [(datetime.fromisoformat(row[0]), row[1]) for row in cursor.fetchall()]
    
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


class NewCodeCoverageAnalyzer:
    """Analyzes coverage for new/changed code"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.new_code_threshold = 80.0  # Default threshold for new code
    
    def set_new_code_threshold(self, threshold: float):
        """Set coverage threshold for new code"""
        self.new_code_threshold = threshold
    
    def analyze_new_code_coverage(self, report: CoverageReport, base_branch: str = 'main') -> List[NewCodeCoverage]:
        """Analyze coverage for new/changed code compared to base branch"""
        changed_files = self._get_changed_files(base_branch)
        new_code_results = []
        
        for file_path in changed_files:
            # Get new lines in this file
            new_lines = self._get_new_lines(file_path, base_branch)
            if not new_lines:
                continue
            
            # Find coverage for this file
            file_coverage = None
            for fc in report.file_coverages:
                if fc.file_path == file_path:
                    file_coverage = fc
                    break
            
            if not file_coverage:
                # File not in coverage report (might be new file)
                new_code_results.append(NewCodeCoverage(
                    file_path=file_path,
                    new_lines=new_lines,
                    covered_new_lines=[],
                    uncovered_new_lines=new_lines,
                    new_code_coverage=0.0,
                    meets_threshold=False,
                    threshold_required=self.new_code_threshold
                ))
                continue
            
            # Determine which new lines are covered
            executed_lines = set(range(1, file_coverage.total_lines + 1)) - set(file_coverage.missing_lines)
            covered_new_lines = [line for line in new_lines if line in executed_lines]
            uncovered_new_lines = [line for line in new_lines if line not in executed_lines]
            
            new_code_coverage = (len(covered_new_lines) / len(new_lines) * 100) if new_lines else 100.0
            meets_threshold = new_code_coverage >= self.new_code_threshold
            
            new_code_results.append(NewCodeCoverage(
                file_path=file_path,
                new_lines=new_lines,
                covered_new_lines=covered_new_lines,
                uncovered_new_lines=uncovered_new_lines,
                new_code_coverage=new_code_coverage,
                meets_threshold=meets_threshold,
                threshold_required=self.new_code_threshold
            ))
        
        return new_code_results
    
    def _get_changed_files(self, base_branch: str) -> List[str]:
        """Get list of files changed compared to base branch"""
        try:
            result = subprocess.run(
                ['git', 'diff', '--name-only', f'{base_branch}...HEAD'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            
            files = result.stdout.strip().split('\n')
            # Filter for Python files
            return [f for f in files if f.endswith('.py') and not f.startswith('test')]
        
        except (subprocess.CalledProcessError, FileNotFoundError):
            return []
    
    def _get_new_lines(self, file_path: str, base_branch: str) -> List[int]:
        """Get line numbers of new/changed lines in a file"""
        try:
            result = subprocess.run(
                ['git', 'diff', f'{base_branch}...HEAD', '--unified=0', file_path],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            
            new_lines = []
            for line in result.stdout.split('\n'):
                if line.startswith('@@'):
                    # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
                    parts = line.split()
                    if len(parts) >= 3:
                        new_part = parts[2]  # +new_start,new_count
                        if new_part.startswith('+'):
                            new_info = new_part[1:].split(',')
                            start_line = int(new_info[0])
                            count = int(new_info[1]) if len(new_info) > 1 else 1
                            new_lines.extend(range(start_line, start_line + count))
            
            return sorted(list(set(new_lines)))
        
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            return []


class CoverageThresholdEnforcer:
    """Enforces coverage thresholds for new code"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.new_code_analyzer = NewCodeCoverageAnalyzer(project_root)
        self.overall_threshold = 80.0
        self.new_code_threshold = 85.0
        self.critical_files_threshold = 90.0
        self.critical_file_patterns = {'core/', 'api/', 'services/'}
    
    def set_thresholds(self, overall: float = None, new_code: float = None, critical: float = None):
        """Set coverage thresholds"""
        if overall is not None:
            self.overall_threshold = overall
        if new_code is not None:
            self.new_code_threshold = new_code
            self.new_code_analyzer.set_new_code_threshold(new_code)
        if critical is not None:
            self.critical_files_threshold = critical
    
    def enforce_thresholds(self, report: CoverageReport, base_branch: str = 'main') -> CoverageThresholdResult:
        """Enforce coverage thresholds"""
        violations = []
        recommendations = []
        
        # Check overall coverage
        overall_passed = report.overall_percentage >= self.overall_threshold
        if not overall_passed:
            violations.append(
                f"Overall coverage {report.overall_percentage:.1f}% below threshold {self.overall_threshold:.1f}%"
            )
            recommendations.append(
                f"Increase overall coverage by {self.overall_threshold - report.overall_percentage:.1f} percentage points"
            )
        
        # Check new code coverage
        new_code_results = self.new_code_analyzer.analyze_new_code_coverage(report, base_branch)
        new_code_passed = all(result.meets_threshold for result in new_code_results)
        
        for result in new_code_results:
            if not result.meets_threshold:
                violations.append(
                    f"New code in {result.file_path} has {result.new_code_coverage:.1f}% coverage, "
                    f"below threshold {result.threshold_required:.1f}%"
                )
                recommendations.append(
                    f"Add tests for {len(result.uncovered_new_lines)} uncovered lines in {result.file_path}"
                )
        
        # Check critical files
        for file_cov in report.file_coverages:
            if self._is_critical_file(file_cov.file_path):
                if file_cov.coverage_percentage < self.critical_files_threshold:
                    violations.append(
                        f"Critical file {file_cov.file_path} has {file_cov.coverage_percentage:.1f}% coverage, "
                        f"below threshold {self.critical_files_threshold:.1f}%"
                    )
                    recommendations.append(
                        f"Prioritize testing for critical file {file_cov.file_path}"
                    )
        
        passed = overall_passed and new_code_passed and len(violations) == 0
        
        return CoverageThresholdResult(
            passed=passed,
            overall_coverage=report.overall_percentage,
            required_threshold=self.overall_threshold,
            new_code_results=new_code_results,
            violations=violations,
            recommendations=recommendations
        )
    
    def _is_critical_file(self, file_path: str) -> bool:
        """Check if file is considered critical"""
        return any(pattern in file_path for pattern in self.critical_file_patterns)


class DetailedCoverageReporter:
    """Generates detailed coverage reports with actionable recommendations"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.trend_tracker = CoverageTrendTracker(project_root)
    
    def generate_detailed_report(self, report: CoverageReport, threshold_result: CoverageThresholdResult = None) -> Dict[str, Any]:
        """Generate comprehensive coverage report"""
        detailed_report = {
            'summary': self._generate_summary(report),
            'file_analysis': self._generate_file_analysis(report),
            'gap_analysis': self._generate_gap_analysis(report),
            'recommendations': self._generate_detailed_recommendations(report, threshold_result),
            'trends': self._generate_trend_analysis(),
            'actionable_items': self._generate_actionable_items(report, threshold_result),
            'generated_at': datetime.now().isoformat()
        }
        
        return detailed_report
    
    def _generate_summary(self, report: CoverageReport) -> Dict[str, Any]:
        """Generate coverage summary"""
        return {
            'overall_coverage': report.overall_percentage,
            'total_files': report.total_files,
            'covered_files': report.covered_files,
            'uncovered_files': report.total_files - report.covered_files,
            'total_lines': report.total_lines,
            'covered_lines': report.covered_lines,
            'uncovered_lines': report.total_lines - report.covered_lines,
            'coverage_gaps': len(report.coverage_gaps),
            'threshold_violations': len(report.threshold_violations)
        }
    
    def _generate_file_analysis(self, report: CoverageReport) -> List[Dict[str, Any]]:
        """Generate per-file analysis"""
        file_analysis = []
        
        for file_cov in sorted(report.file_coverages, key=lambda x: x.coverage_percentage):
            analysis = {
                'file_path': file_cov.file_path,
                'coverage_percentage': file_cov.coverage_percentage,
                'total_lines': file_cov.total_lines,
                'covered_lines': file_cov.covered_lines,
                'uncovered_lines': len(file_cov.missing_lines),
                'missing_line_ranges': self._group_missing_lines(file_cov.missing_lines),
                'function_coverage': self._analyze_function_coverage(file_cov),
                'priority': self._determine_file_priority(file_cov),
                'recommendations': self._generate_file_recommendations(file_cov)
            }
            file_analysis.append(analysis)
        
        return file_analysis
    
    def _generate_gap_analysis(self, report: CoverageReport) -> Dict[str, Any]:
        """Generate coverage gap analysis"""
        gaps_by_type = {}
        gaps_by_severity = {}
        
        for gap in report.coverage_gaps:
            # Group by type
            if gap.gap_type not in gaps_by_type:
                gaps_by_type[gap.gap_type] = []
            gaps_by_type[gap.gap_type].append(gap)
            
            # Group by severity
            if gap.severity not in gaps_by_severity:
                gaps_by_severity[gap.severity] = []
            gaps_by_severity[gap.severity].append(gap)
        
        return {
            'total_gaps': len(report.coverage_gaps),
            'gaps_by_type': {k: len(v) for k, v in gaps_by_type.items()},
            'gaps_by_severity': {k: len(v) for k, v in gaps_by_severity.items()},
            'critical_gaps': [asdict(gap) for gap in report.coverage_gaps if gap.severity == 'critical'],
            'high_priority_gaps': [asdict(gap) for gap in report.coverage_gaps if gap.severity == 'high']
        }
    
    def _generate_detailed_recommendations(self, report: CoverageReport, threshold_result: CoverageThresholdResult = None) -> List[Dict[str, Any]]:
        """Generate detailed, prioritized recommendations"""
        recommendations = []
        
        # High-priority recommendations
        if threshold_result and threshold_result.violations:
            for violation in threshold_result.violations:
                recommendations.append({
                    'priority': 'high',
                    'category': 'threshold_violation',
                    'description': violation,
                    'action_required': True
                })
        
        # Coverage gap recommendations
        critical_gaps = [g for g in report.coverage_gaps if g.severity == 'critical']
        if critical_gaps:
            recommendations.append({
                'priority': 'critical',
                'category': 'coverage_gaps',
                'description': f"Address {len(critical_gaps)} critical coverage gaps",
                'action_required': True,
                'details': [gap.suggestion for gap in critical_gaps[:5]]  # Top 5
            })
        
        # File-specific recommendations
        low_coverage_files = [fc for fc in report.file_coverages if fc.coverage_percentage < 50]
        if low_coverage_files:
            recommendations.append({
                'priority': 'medium',
                'category': 'low_coverage_files',
                'description': f"Improve coverage for {len(low_coverage_files)} files with <50% coverage",
                'action_required': False,
                'details': [f"{fc.file_path}: {fc.coverage_percentage:.1f}%" for fc in low_coverage_files[:10]]
            })
        
        return recommendations
    
    def _generate_trend_analysis(self) -> Dict[str, Any]:
        """Generate coverage trend analysis"""
        trends = self.trend_tracker.get_trends(30)  # Last 30 days
        
        if len(trends) < 2:
            return {'message': 'Insufficient data for trend analysis'}
        
        # Calculate trend direction
        recent_coverage = trends[-1].overall_percentage
        older_coverage = trends[0].overall_percentage
        trend_direction = 'improving' if recent_coverage > older_coverage else 'declining' if recent_coverage < older_coverage else 'stable'
        
        # Calculate average change per day
        days_span = (trends[-1].timestamp - trends[0].timestamp).days
        coverage_change = recent_coverage - older_coverage
        daily_change = coverage_change / days_span if days_span > 0 else 0
        
        return {
            'trend_direction': trend_direction,
            'coverage_change': coverage_change,
            'daily_change_rate': daily_change,
            'data_points': len(trends),
            'date_range': {
                'start': trends[0].timestamp.isoformat(),
                'end': trends[-1].timestamp.isoformat()
            }
        }
    
    def _generate_actionable_items(self, report: CoverageReport, threshold_result: CoverageThresholdResult = None) -> List[Dict[str, Any]]:
        """Generate specific actionable items"""
        items = []
        
        # Immediate actions for threshold violations
        if threshold_result and not threshold_result.passed:
            for result in threshold_result.new_code_results:
                if not result.meets_threshold:
                    items.append({
                        'type': 'immediate',
                        'title': f"Add tests for new code in {result.file_path}",
                        'description': f"Cover {len(result.uncovered_new_lines)} uncovered lines",
                        'effort': 'medium',
                        'impact': 'high',
                        'lines': result.uncovered_new_lines[:10]  # Show first 10 lines
                    })
        
        # Function-specific actions
        for file_cov in report.file_coverages:
            uncovered_functions = [name for name, data in file_cov.functions.items() 
                                 if data.get('coverage_percentage', 0) == 0]
            if uncovered_functions:
                items.append({
                    'type': 'function_testing',
                    'title': f"Add tests for uncovered functions in {file_cov.file_path}",
                    'description': f"Test {len(uncovered_functions)} uncovered functions",
                    'effort': 'high' if len(uncovered_functions) > 5 else 'medium',
                    'impact': 'medium',
                    'functions': uncovered_functions[:5]  # Show first 5 functions
                })
        
        return items
    
    def _group_missing_lines(self, missing_lines: List[int]) -> List[str]:
        """Group consecutive missing lines into ranges"""
        if not missing_lines:
            return []
        
        ranges = []
        missing_lines.sort()
        start = missing_lines[0]
        end = start
        
        for line in missing_lines[1:]:
            if line == end + 1:
                end = line
            else:
                if start == end:
                    ranges.append(str(start))
                else:
                    ranges.append(f"{start}-{end}")
                start = end = line
        
        # Add the last range
        if start == end:
            ranges.append(str(start))
        else:
            ranges.append(f"{start}-{end}")
        
        return ranges
    
    def _analyze_function_coverage(self, file_cov: FileCoverage) -> Dict[str, Any]:
        """Analyze function-level coverage"""
        if not file_cov.functions:
            return {'total': 0, 'covered': 0, 'percentage': 0}
        
        total_functions = len(file_cov.functions)
        covered_functions = sum(1 for func_data in file_cov.functions.values() 
                              if func_data.get('coverage_percentage', 0) > 0)
        
        return {
            'total': total_functions,
            'covered': covered_functions,
            'percentage': (covered_functions / total_functions * 100) if total_functions > 0 else 0,
            'uncovered': [name for name, data in file_cov.functions.items() 
                         if data.get('coverage_percentage', 0) == 0]
        }
    
    def _determine_file_priority(self, file_cov: FileCoverage) -> str:
        """Determine priority level for file coverage improvement"""
        if file_cov.coverage_percentage < 30:
            return 'critical'
        elif file_cov.coverage_percentage < 60:
            return 'high'
        elif file_cov.coverage_percentage < 80:
            return 'medium'
        else:
            return 'low'
    
    def _generate_file_recommendations(self, file_cov: FileCoverage) -> List[str]:
        """Generate specific recommendations for a file"""
        recommendations = []
        
        if file_cov.coverage_percentage < 50:
            recommendations.append("Priority: Add basic test coverage for core functionality")
        
        uncovered_functions = [name for name, data in file_cov.functions.items() 
                             if data.get('coverage_percentage', 0) == 0]
        if uncovered_functions:
            recommendations.append(f"Add tests for {len(uncovered_functions)} uncovered functions")
        
        if len(file_cov.missing_lines) > 20:
            recommendations.append("Consider breaking down large uncovered code blocks")
        
        return recommendations


class ComprehensiveCoverageSystem:
    """Main system that orchestrates all coverage analysis components"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.coverage_analyzer = CoverageAnalyzer(project_root)
        self.trend_tracker = CoverageTrendTracker(project_root)
        self.threshold_enforcer = CoverageThresholdEnforcer(project_root)
        self.reporter = DetailedCoverageReporter(project_root)
    
    def run_comprehensive_analysis(self, test_files: List[Path] = None, base_branch: str = 'main') -> Dict[str, Any]:
        """Run complete coverage analysis with all features"""
        print("Starting comprehensive coverage analysis...")
        
        # Discover test files if not provided
        if not test_files:
            from test_auditor import TestDiscoveryEngine
            discovery = TestDiscoveryEngine(self.project_root)
            test_files = discovery.discover_test_files()
        
        # Run basic coverage analysis
        print("Analyzing test coverage...")
        coverage_report = self.coverage_analyzer.analyze_coverage(test_files)
        
        # Record trend data
        print("Recording coverage trends...")
        trend_id = self.trend_tracker.record_coverage(coverage_report)
        
        # Enforce thresholds
        print("Enforcing coverage thresholds...")
        threshold_result = self.threshold_enforcer.enforce_thresholds(coverage_report, base_branch)
        
        # Generate detailed report
        print("Generating detailed report...")
        detailed_report = self.reporter.generate_detailed_report(coverage_report, threshold_result)
        
        # Combine all results
        comprehensive_result = {
            'basic_coverage': asdict(coverage_report),
            'threshold_enforcement': asdict(threshold_result),
            'detailed_analysis': detailed_report,
            'trend_id': trend_id,
            'analysis_metadata': {
                'test_files_analyzed': len(test_files),
                'base_branch': base_branch,
                'analysis_timestamp': datetime.now().isoformat()
            }
        }
        
        return comprehensive_result
    
    def save_report(self, analysis_result: Dict[str, Any], output_path: Path):
        """Save comprehensive analysis report"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(analysis_result, f, indent=2, default=str)
        
        print(f"Comprehensive coverage report saved to {output_path}")


def main():
    """Main entry point for comprehensive coverage system"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive test coverage analysis system")
    parser.add_argument('--project-root', type=Path, default=Path.cwd(), help='Project root directory')
    parser.add_argument('--test-files', nargs='*', help='Specific test files to analyze')
    parser.add_argument('--output', type=Path, help='Output file for comprehensive report')
    parser.add_argument('--base-branch', default='main', help='Base branch for new code analysis')
    parser.add_argument('--overall-threshold', type=float, default=80.0, help='Overall coverage threshold')
    parser.add_argument('--new-code-threshold', type=float, default=85.0, help='New code coverage threshold')
    
    args = parser.parse_args()
    
    # Setup system
    system = ComprehensiveCoverageSystem(args.project_root)
    system.threshold_enforcer.set_thresholds(
        overall=args.overall_threshold,
        new_code=args.new_code_threshold
    )
    
    # Prepare test files
    test_files = None
    if args.test_files:
        test_files = [Path(f) for f in args.test_files]
    
    # Run analysis
    result = system.run_comprehensive_analysis(test_files, args.base_branch)
    
    # Save report
    if args.output:
        system.save_report(result, args.output)
    else:
        # Save to default location
        default_output = args.project_root / '.coverage_analysis' / f'comprehensive_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        system.save_report(result, default_output)
    
    # Print summary
    basic_coverage = result['basic_coverage']
    threshold_result = result['threshold_enforcement']
    
    print(f"\n=== Coverage Analysis Summary ===")
    print(f"Overall coverage: {basic_coverage['overall_percentage']:.1f}%")
    print(f"Files covered: {basic_coverage['covered_files']}/{basic_coverage['total_files']}")
    print(f"Lines covered: {basic_coverage['covered_lines']}/{basic_coverage['total_lines']}")
    print(f"Coverage gaps: {len(basic_coverage['coverage_gaps'])}")
    
    print(f"\n=== Threshold Enforcement ===")
    print(f"Thresholds passed: {'✓' if threshold_result['passed'] else '✗'}")
    print(f"Violations: {len(threshold_result['violations'])}")
    
    if threshold_result['violations']:
        print("\nViolations:")
        for violation in threshold_result['violations']:
            print(f"  - {violation}")
    
    if threshold_result['recommendations']:
        print("\nRecommendations:")
        for rec in threshold_result['recommendations']:
            print(f"  - {rec}")
    
    return 0 if threshold_result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())