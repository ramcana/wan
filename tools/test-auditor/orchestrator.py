#!/usr/bin/env python3
"""
Test Suite Audit Orchestrator

Main orchestrator that coordinates all test auditing components to provide
a comprehensive analysis of the test suite health and quality.
"""

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any

from test_auditor import TestAuditor, TestSuiteAuditReport
from test_runner import ParallelTestRunner, TestSuiteExecutionReport
from coverage_analyzer import CoverageAnalyzer, CoverageReport


@dataclass
class ComprehensiveTestAnalysis:
    """Complete test suite analysis combining all audit components"""
    audit_report: TestSuiteAuditReport
    execution_report: TestSuiteExecutionReport
    coverage_report: CoverageReport
    analysis_summary: Dict[str, Any]
    recommendations: List[str]
    action_plan: List[Dict[str, Any]]
    health_score: float


class TestSuiteHealthScorer:
    """Calculates overall health score for test suite"""
    
    def __init__(self):
        self.weights = {
            'syntax_health': 0.20,      # No syntax/import errors
            'test_completeness': 0.15,   # Tests have assertions, not empty
            'execution_success': 0.25,   # Tests pass consistently
            'performance': 0.15,         # Tests run in reasonable time
            'coverage': 0.20,            # Good code coverage
            'reliability': 0.05          # Low flakiness, few retries
        }
    
    def calculate_health_score(self, analysis: ComprehensiveTestAnalysis) -> float:
        """Calculate overall health score (0-100)"""
        scores = {}
        
        # Syntax health score
        scores['syntax_health'] = self._calculate_syntax_health(analysis.audit_report)
        
        # Test completeness score
        scores['test_completeness'] = self._calculate_completeness_score(analysis.audit_report)
        
        # Execution success score
        scores['execution_success'] = self._calculate_execution_score(analysis.execution_report)
        
        # Performance score
        scores['performance'] = self._calculate_performance_score(analysis.execution_report)
        
        # Coverage score
        scores['coverage'] = self._calculate_coverage_score(analysis.coverage_report)
        
        # Reliability score
        scores['reliability'] = self._calculate_reliability_score(analysis.execution_report)
        
        # Calculate weighted average
        total_score = sum(
            scores[component] * self.weights[component]
            for component in scores
        )
        
        return min(100.0, max(0.0, total_score))
    
    def _calculate_syntax_health(self, audit_report: TestSuiteAuditReport) -> float:
        """Calculate syntax health score"""
        if audit_report.total_files == 0:
            return 0.0
        
        broken_files = len(audit_report.broken_files)
        syntax_errors = len([
            issue for fa in audit_report.file_analyses
            for issue in fa.issues
            if issue.issue_type in ['syntax_error', 'import_error', 'missing_import']
        ])
        
        # Penalize broken files and syntax errors
        penalty = (broken_files * 10 + syntax_errors * 5) / audit_report.total_files
        
        return max(0.0, 100.0 - penalty)
    
    def _calculate_completeness_score(self, audit_report: TestSuiteAuditReport) -> float:
        """Calculate test completeness score"""
        if audit_report.total_tests == 0:
            return 0.0
        
        empty_tests = len([
            issue for fa in audit_report.file_analyses
            for issue in fa.issues
            if issue.issue_type == 'empty_test'
        ])
        
        no_assertion_tests = len([
            issue for fa in audit_report.file_analyses
            for issue in fa.issues
            if issue.issue_type == 'no_assertions'
        ])
        
        # Penalize incomplete tests
        penalty = (empty_tests * 15 + no_assertion_tests * 10) / audit_report.total_tests * 100
        
        return max(0.0, 100.0 - penalty)
    
    def _calculate_execution_score(self, execution_report: TestSuiteExecutionReport) -> float:
        """Calculate execution success score"""
        if execution_report.total_files == 0:
            return 0.0
        
        success_rate = execution_report.successful_files / execution_report.total_files * 100
        return success_rate
    
    def _calculate_performance_score(self, execution_report: TestSuiteExecutionReport) -> float:
        """Calculate performance score"""
        if not execution_report.performance_summary:
            return 50.0  # Neutral score if no data
        
        avg_time = execution_report.performance_summary.get('average_time', 0)
        max_time = execution_report.performance_summary.get('max_time', 0)
        
        # Good performance: avg < 5s, max < 30s
        # Poor performance: avg > 20s, max > 120s
        
        avg_score = max(0, 100 - (avg_time - 5) * 10) if avg_time > 5 else 100
        max_score = max(0, 100 - (max_time - 30) * 2) if max_time > 30 else 100
        
        return (avg_score + max_score) / 2
    
    def _calculate_coverage_score(self, coverage_report: CoverageReport) -> float:
        """Calculate coverage score"""
        return coverage_report.overall_percentage
    
    def _calculate_reliability_score(self, execution_report: TestSuiteExecutionReport) -> float:
        """Calculate reliability score based on retries and timeouts"""
        if execution_report.total_files == 0:
            return 100.0
        
        # Count files that needed retries
        retry_files = sum(
            1 for result in execution_report.file_results
            if result.retry_count > 0
        )
        
        # Count timeout files
        timeout_files = len(execution_report.timeout_summary.get('timeout_files', []))
        
        # Penalize unreliable tests
        reliability_penalty = (retry_files * 5 + timeout_files * 10) / execution_report.total_files * 100
        
        return max(0.0, 100.0 - reliability_penalty)


class ActionPlanGenerator:
    """Generates actionable plans to improve test suite health"""
    
    def generate_action_plan(self, analysis: ComprehensiveTestAnalysis) -> List[Dict[str, Any]]:
        """Generate prioritized action plan"""
        actions = []
        
        # Critical issues first
        actions.extend(self._generate_critical_actions(analysis))
        
        # High priority improvements
        actions.extend(self._generate_high_priority_actions(analysis))
        
        # Medium priority improvements
        actions.extend(self._generate_medium_priority_actions(analysis))
        
        # Long-term improvements
        actions.extend(self._generate_long_term_actions(analysis))
        
        return actions
    
    def _generate_critical_actions(self, analysis: ComprehensiveTestAnalysis) -> List[Dict[str, Any]]:
        """Generate critical actions that must be addressed immediately"""
        actions = []
        
        # Fix broken files
        if analysis.audit_report.broken_files:
            actions.append({
                'priority': 'critical',
                'category': 'syntax_fixes',
                'title': 'Fix broken test files',
                'description': f'Fix {len(analysis.audit_report.broken_files)} broken test files with syntax or import errors',
                'files': analysis.audit_report.broken_files,
                'estimated_effort': 'high',
                'impact': 'high'
            })
        
        # Fix critical coverage gaps
        critical_gaps = [g for g in analysis.coverage_report.coverage_gaps if g.severity == 'critical']
        if critical_gaps:
            actions.append({
                'priority': 'critical',
                'category': 'coverage',
                'title': 'Address critical coverage gaps',
                'description': f'Add tests for {len(critical_gaps)} critical uncovered code paths',
                'gaps': [asdict(g) for g in critical_gaps],
                'estimated_effort': 'high',
                'impact': 'high'
            })
        
        return actions
    
    def _generate_high_priority_actions(self, analysis: ComprehensiveTestAnalysis) -> List[Dict[str, Any]]:
        """Generate high priority actions"""
        actions = []
        
        # Fix failing tests
        failing_files = [
            result for result in analysis.execution_report.file_results
            if not result.success and not result.timeout_occurred
        ]
        
        if failing_files:
            actions.append({
                'priority': 'high',
                'category': 'test_fixes',
                'title': 'Fix failing tests',
                'description': f'Fix {len(failing_files)} test files that are currently failing',
                'files': [result.test_file for result in failing_files],
                'estimated_effort': 'medium',
                'impact': 'high'
            })
        
        # Add missing assertions
        no_assertion_issues = [
            issue for fa in analysis.audit_report.file_analyses
            for issue in fa.issues
            if issue.issue_type == 'no_assertions'
        ]
        
        if no_assertion_issues:
            actions.append({
                'priority': 'high',
                'category': 'test_quality',
                'title': 'Add missing test assertions',
                'description': f'Add assertions to {len(no_assertion_issues)} test functions',
                'issues': [asdict(issue) for issue in no_assertion_issues],
                'estimated_effort': 'medium',
                'impact': 'medium'
            })
        
        return actions
    
    def _generate_medium_priority_actions(self, analysis: ComprehensiveTestAnalysis) -> List[Dict[str, Any]]:
        """Generate medium priority actions"""
        actions = []
        
        # Improve coverage
        if analysis.coverage_report.overall_percentage < 70:
            actions.append({
                'priority': 'medium',
                'category': 'coverage',
                'title': 'Improve test coverage',
                'description': f'Increase overall coverage from {analysis.coverage_report.overall_percentage:.1f}% to 70%+',
                'current_coverage': analysis.coverage_report.overall_percentage,
                'target_coverage': 70.0,
                'estimated_effort': 'high',
                'impact': 'medium'
            })
        
        # Optimize slow tests
        slow_tests = [
            result for result in analysis.execution_report.file_results
            if result.execution_time > 10.0
        ]
        
        if slow_tests:
            actions.append({
                'priority': 'medium',
                'category': 'performance',
                'title': 'Optimize slow tests',
                'description': f'Optimize {len(slow_tests)} test files that take over 10 seconds',
                'files': [result.test_file for result in slow_tests],
                'estimated_effort': 'medium',
                'impact': 'low'
            })
        
        return actions
    
    def _generate_long_term_actions(self, analysis: ComprehensiveTestAnalysis) -> List[Dict[str, Any]]:
        """Generate long-term improvement actions"""
        actions = []
        
        # Implement test automation
        actions.append({
            'priority': 'low',
            'category': 'automation',
            'title': 'Implement continuous testing',
            'description': 'Set up automated test execution in CI/CD pipeline',
            'estimated_effort': 'medium',
            'impact': 'high'
        })
        
        # Add performance monitoring
        actions.append({
            'priority': 'low',
            'category': 'monitoring',
            'title': 'Add test performance monitoring',
            'description': 'Implement monitoring to track test performance over time',
            'estimated_effort': 'low',
            'impact': 'medium'
        })
        
        return actions


class TestSuiteOrchestrator:
    """Main orchestrator for comprehensive test suite analysis"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.auditor = TestAuditor(project_root)
        self.runner = ParallelTestRunner()
        self.coverage_analyzer = CoverageAnalyzer(project_root)
        self.health_scorer = TestSuiteHealthScorer()
        self.action_planner = ActionPlanGenerator()
    
    def run_comprehensive_analysis(self) -> ComprehensiveTestAnalysis:
        """Run complete test suite analysis"""
        print("Starting comprehensive test suite analysis...")
        
        start_time = time.time()
        
        # Step 1: Audit test suite
        print("1. Auditing test suite structure and quality...")
        audit_report = self.auditor.audit_test_suite()
        
        # Discover test files for execution and coverage
        test_files = self.auditor.discovery_engine.discover_test_files()
        
        # Step 2: Execute tests
        print("2. Executing tests with monitoring...")
        execution_report = self.runner.run_tests_parallel(test_files, self.project_root)
        
        # Step 3: Analyze coverage
        print("3. Analyzing test coverage...")
        coverage_report = self.coverage_analyzer.analyze_coverage(test_files)
        
        # Step 4: Generate analysis summary
        print("4. Generating analysis summary...")
        analysis_summary = self._generate_analysis_summary(
            audit_report, execution_report, coverage_report, time.time() - start_time
        )
        
        # Step 5: Generate comprehensive recommendations
        print("5. Generating recommendations...")
        recommendations = self._generate_comprehensive_recommendations(
            audit_report, execution_report, coverage_report
        )
        
        # Create comprehensive analysis
        analysis = ComprehensiveTestAnalysis(
            audit_report=audit_report,
            execution_report=execution_report,
            coverage_report=coverage_report,
            analysis_summary=analysis_summary,
            recommendations=recommendations,
            action_plan=[],  # Will be filled next
            health_score=0.0  # Will be calculated next
        )
        
        # Step 6: Calculate health score
        analysis.health_score = self.health_scorer.calculate_health_score(analysis)
        
        # Step 7: Generate action plan
        analysis.action_plan = self.action_planner.generate_action_plan(analysis)
        
        print(f"Analysis complete! Health score: {analysis.health_score:.1f}/100")
        
        return analysis
    
    def _generate_analysis_summary(
        self, 
        audit_report: TestSuiteAuditReport,
        execution_report: TestSuiteExecutionReport,
        coverage_report: CoverageReport,
        total_time: float
    ) -> Dict[str, Any]:
        """Generate comprehensive analysis summary"""
        
        return {
            'analysis_duration': total_time,
            'test_discovery': {
                'total_files': audit_report.total_files,
                'total_tests': audit_report.total_tests,
                'test_distribution': self._calculate_test_distribution(audit_report)
            },
            'quality_metrics': {
                'syntax_health': len(audit_report.broken_files) == 0,
                'import_health': sum(len(fa.missing_imports) for fa in audit_report.file_analyses) == 0,
                'assertion_completeness': len([
                    issue for fa in audit_report.file_analyses
                    for issue in fa.issues
                    if issue.issue_type == 'no_assertions'
                ]) == 0
            },
            'execution_metrics': {
                'success_rate': execution_report.successful_files / execution_report.total_files if execution_report.total_files > 0 else 0,
                'average_execution_time': execution_report.performance_summary.get('average_time', 0),
                'timeout_rate': len(execution_report.timeout_summary.get('timeout_files', [])) / execution_report.total_files if execution_report.total_files > 0 else 0
            },
            'coverage_metrics': {
                'overall_percentage': coverage_report.overall_percentage,
                'files_with_coverage': coverage_report.covered_files,
                'critical_gaps': len([g for g in coverage_report.coverage_gaps if g.severity == 'critical'])
            }
        }
    
    def _calculate_test_distribution(self, audit_report: TestSuiteAuditReport) -> Dict[str, int]:
        """Calculate distribution of tests across categories"""
        distribution = {
            'unit': 0,
            'integration': 0,
            'e2e': 0,
            'performance': 0,
            'other': 0
        }
        
        for file_analysis in audit_report.file_analyses:
            file_path = file_analysis.file_path.lower()
            
            if 'unit' in file_path:
                distribution['unit'] += file_analysis.total_tests
            elif 'integration' in file_path:
                distribution['integration'] += file_analysis.total_tests
            elif 'e2e' in file_path or 'end_to_end' in file_path:
                distribution['e2e'] += file_analysis.total_tests
            elif 'performance' in file_path or 'perf' in file_path:
                distribution['performance'] += file_analysis.total_tests
            else:
                distribution['other'] += file_analysis.total_tests
        
        return distribution
    
    def _generate_comprehensive_recommendations(
        self,
        audit_report: TestSuiteAuditReport,
        execution_report: TestSuiteExecutionReport,
        coverage_report: CoverageReport
    ) -> List[str]:
        """Generate comprehensive recommendations combining all analyses"""
        
        recommendations = []
        
        # Add audit recommendations
        recommendations.extend(audit_report.recommendations)
        
        # Add coverage recommendations
        recommendations.extend(coverage_report.recommendations)
        
        # Add execution-specific recommendations
        if execution_report.failed_files > 0:
            recommendations.append(
                f"Fix {execution_report.failed_files} failing test files to improve reliability"
            )
        
        if execution_report.timeout_summary.get('total_timeouts', 0) > 0:
            recommendations.append(
                f"Investigate and fix {execution_report.timeout_summary['total_timeouts']} test timeouts"
            )
        
        # Add performance recommendations
        if execution_report.performance_summary:
            avg_time = execution_report.performance_summary.get('average_time', 0)
            if avg_time > 10:
                recommendations.append(
                    f"Optimize test performance - average execution time is {avg_time:.1f}s"
                )
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def save_analysis(self, analysis: ComprehensiveTestAnalysis, output_file: Path):
        """Save comprehensive analysis to file"""
        with open(output_file, 'w') as f:
            json.dump(asdict(analysis), f, indent=2, default=str)
    
    def print_summary(self, analysis: ComprehensiveTestAnalysis):
        """Print analysis summary"""
        print("\n" + "="*60)
        print("COMPREHENSIVE TEST SUITE ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"\nHealth Score: {analysis.health_score:.1f}/100")
        
        print(f"\nTest Discovery:")
        print(f"  Files: {analysis.audit_report.total_files}")
        print(f"  Tests: {analysis.audit_report.total_tests}")
        
        print(f"\nExecution Results:")
        print(f"  Success Rate: {analysis.execution_report.successful_files}/{analysis.execution_report.total_files}")
        print(f"  Average Time: {analysis.execution_report.performance_summary.get('average_time', 0):.2f}s")
        
        print(f"\nCoverage:")
        print(f"  Overall: {analysis.coverage_report.overall_percentage:.1f}%")
        print(f"  Files Covered: {analysis.coverage_report.covered_files}/{analysis.coverage_report.total_files}")
        
        print(f"\nIssues:")
        print(f"  Broken Files: {len(analysis.audit_report.broken_files)}")
        print(f"  Critical Issues: {len(analysis.audit_report.critical_issues)}")
        print(f"  Coverage Gaps: {len(analysis.coverage_report.coverage_gaps)}")
        
        if analysis.action_plan:
            print(f"\nTop Priority Actions:")
            for i, action in enumerate(analysis.action_plan[:5], 1):
                print(f"  {i}. [{action['priority'].upper()}] {action['title']}")
        
        if analysis.recommendations:
            print(f"\nKey Recommendations:")
            for i, rec in enumerate(analysis.recommendations[:5], 1):
                print(f"  {i}. {rec}")


def main():
    """Main entry point for orchestrator"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive test suite analysis orchestrator")
    parser.add_argument('--project-root', type=Path, default=Path.cwd(), help='Project root directory')
    parser.add_argument('--output', type=Path, default='comprehensive_test_analysis.json', help='Output file')
    parser.add_argument('--summary-only', action='store_true', help='Print summary only, no detailed output')
    
    args = parser.parse_args()
    
    # Run comprehensive analysis
    orchestrator = TestSuiteOrchestrator(args.project_root)
    analysis = orchestrator.run_comprehensive_analysis()
    
    # Save results
    orchestrator.save_analysis(analysis, args.output)
    print(f"\nDetailed analysis saved to {args.output}")
    
    # Print summary
    orchestrator.print_summary(analysis)
    
    # Return appropriate exit code
    return 0 if analysis.health_score >= 70 else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
