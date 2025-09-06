#!/usr/bin/env python3
"""
Test Quality Integration Example

Demonstrates how to integrate all test quality tools into a comprehensive
development workflow, including CI/CD integration and automated monitoring.
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from coverage_system import ComprehensiveCoverageSystem
from performance_optimizer import TestPerformanceOptimizer
from flaky_test_detector import FlakyTestDetectionSystem


class TestQualityIntegration:
    """Integrated test quality management system"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.coverage_system = ComprehensiveCoverageSystem(project_root)
        self.performance_optimizer = TestPerformanceOptimizer(project_root)
        self.flaky_detector = FlakyTestDetectionSystem(project_root)
        
        # Quality gates configuration
        self.quality_gates = {
            'coverage': {
                'overall_threshold': 80.0,
                'new_code_threshold': 85.0,
                'critical_files_threshold': 90.0
            },
            'performance': {
                'slow_test_threshold': 5.0,
                'regression_threshold': 1.5,
                'total_time_limit': 300.0  # 5 minutes
            },
            'flakiness': {
                'flakiness_threshold': 0.1,
                'quarantine_threshold': 0.4,
                'max_flaky_tests': 5
            }
        }
    
    def run_pre_commit_checks(self) -> Dict[str, Any]:
        """Run quality checks suitable for pre-commit hooks"""
        print("🔍 Running pre-commit test quality checks...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'passed': True,
            'blocking_issues': []
        }
        
        # Quick coverage check for new code
        try:
            coverage_result = self.coverage_system.run_comprehensive_analysis(base_branch='HEAD~1')
            threshold_result = coverage_result['threshold_enforcement']
            
            results['checks']['coverage'] = {
                'passed': threshold_result['passed'],
                'overall_coverage': threshold_result['overall_coverage'],
                'new_code_violations': len([r for r in threshold_result['new_code_results'] if not r['meets_threshold']])
            }
            
            if not threshold_result['passed']:
                results['passed'] = False
                results['blocking_issues'].extend(threshold_result['violations'])
        
        except Exception as e:
            results['checks']['coverage'] = {'error': str(e)}
            results['passed'] = False
            results['blocking_issues'].append(f"Coverage check failed: {e}")
        
        # Quick performance regression check
        try:
            # Run a subset of tests for performance check
            perf_result = self.performance_optimizer.optimize_test_performance()
            regressions = perf_result['regressions']
            
            severe_regressions = [r for r in regressions if r['severity'] == 'severe']
            
            results['checks']['performance'] = {
                'passed': len(severe_regressions) == 0,
                'total_regressions': len(regressions),
                'severe_regressions': len(severe_regressions)
            }
            
            if severe_regressions:
                results['passed'] = False
                results['blocking_issues'].extend([
                    f"Severe performance regression in {r['test_id']}" for r in severe_regressions
                ])
        
        except Exception as e:
            results['checks']['performance'] = {'error': str(e)}
        
        return results
    
    def run_ci_quality_gates(self) -> Dict[str, Any]:
        """Run comprehensive quality gates for CI/CD"""
        print("🚪 Running CI quality gates...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'gates': {},
            'passed': True,
            'warnings': [],
            'failures': []
        }
        
        # Coverage gate
        print("📊 Checking coverage gate...")
        try:
            self.coverage_system.threshold_enforcer.set_thresholds(
                overall=self.quality_gates['coverage']['overall_threshold'],
                new_code=self.quality_gates['coverage']['new_code_threshold'],
                critical=self.quality_gates['coverage']['critical_files_threshold']
            )
            
            coverage_result = self.coverage_system.run_comprehensive_analysis()
            threshold_result = coverage_result['threshold_enforcement']
            
            coverage_passed = threshold_result['passed']
            results['gates']['coverage'] = {
                'passed': coverage_passed,
                'overall_coverage': threshold_result['overall_coverage'],
                'violations': threshold_result['violations'],
                'recommendations': threshold_result['recommendations']
            }
            
            if not coverage_passed:
                results['passed'] = False
                results['failures'].extend(threshold_result['violations'])
        
        except Exception as e:
            results['gates']['coverage'] = {'error': str(e)}
            results['passed'] = False
            results['failures'].append(f"Coverage gate failed: {e}")
        
        # Performance gate
        print("⚡ Checking performance gate...")
        try:
            self.performance_optimizer.profiler.slow_test_threshold = self.quality_gates['performance']['slow_test_threshold']
            
            perf_result = self.performance_optimizer.optimize_test_performance()
            profile = perf_result['performance_profile']
            regressions = perf_result['regressions']
            
            # Check performance criteria
            total_time_ok = profile['total_duration'] <= self.quality_gates['performance']['total_time_limit']
            no_severe_regressions = not any(r['severity'] == 'severe' for r in regressions)
            slow_tests_acceptable = len(profile['slow_tests']) <= 10  # Max 10 slow tests
            
            performance_passed = total_time_ok and no_severe_regressions and slow_tests_acceptable
            
            results['gates']['performance'] = {
                'passed': performance_passed,
                'total_duration': profile['total_duration'],
                'slow_tests_count': len(profile['slow_tests']),
                'regressions_count': len(regressions),
                'severe_regressions': [r for r in regressions if r['severity'] == 'severe']
            }
            
            if not performance_passed:
                results['passed'] = False
                if not total_time_ok:
                    results['failures'].append(f"Test suite too slow: {profile['total_duration']:.1f}s > {self.quality_gates['performance']['total_time_limit']}s")
                if not no_severe_regressions:
                    results['failures'].append("Severe performance regressions detected")
                if not slow_tests_acceptable:
                    results['failures'].append(f"Too many slow tests: {len(profile['slow_tests'])} > 10")
        
        except Exception as e:
            results['gates']['performance'] = {'error': str(e)}
            results['warnings'].append(f"Performance gate check failed: {e}")
        
        # Flakiness gate
        print("🎲 Checking flakiness gate...")
        try:
            self.flaky_detector.analyzer.flakiness_threshold = self.quality_gates['flakiness']['flakiness_threshold']
            
            flaky_result = self.flaky_detector.run_flaky_test_analysis()
            patterns = flaky_result['flaky_patterns']
            
            # Check flakiness criteria
            flaky_count = len(patterns)
            high_flaky_count = len([p for p in patterns if p['flakiness_score'] > 0.5])
            max_flaky_ok = flaky_count <= self.quality_gates['flakiness']['max_flaky_tests']
            no_high_flaky = high_flaky_count == 0
            
            flakiness_passed = max_flaky_ok and no_high_flaky
            
            results['gates']['flakiness'] = {
                'passed': flakiness_passed,
                'flaky_tests_count': flaky_count,
                'high_flaky_count': high_flaky_count,
                'quarantined_count': flaky_result['analysis_metadata']['tests_quarantined']
            }
            
            if not flakiness_passed:
                if not max_flaky_ok:
                    results['failures'].append(f"Too many flaky tests: {flaky_count} > {self.quality_gates['flakiness']['max_flaky_tests']}")
                if not no_high_flaky:
                    results['failures'].append(f"Highly flaky tests detected: {high_flaky_count}")
            
            # Warnings for moderate issues
            if flaky_count > 0:
                results['warnings'].append(f"{flaky_count} flaky tests detected - consider fixing")
        
        except Exception as e:
            results['gates']['flakiness'] = {'error': str(e)}
            results['warnings'].append(f"Flakiness gate check failed: {e}")
        
        return results
    
    def run_nightly_analysis(self) -> Dict[str, Any]:
        """Run comprehensive nightly analysis with detailed reporting"""
        print("🌙 Running nightly test quality analysis...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reports_dir = self.project_root / '.test_quality_reports' / 'nightly'
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'nightly',
            'reports_generated': [],
            'summary': {},
            'recommendations': []
        }
        
        # Comprehensive coverage analysis
        print("📊 Running comprehensive coverage analysis...")
        try:
            coverage_result = self.coverage_system.run_comprehensive_analysis()
            coverage_report_path = reports_dir / f'coverage_nightly_{timestamp}.json'
            self.coverage_system.save_report(coverage_result, coverage_report_path)
            
            results['reports_generated'].append(str(coverage_report_path))
            results['summary']['coverage'] = {
                'overall_percentage': coverage_result['basic_coverage']['overall_percentage'],
                'files_covered': coverage_result['basic_coverage']['covered_files'],
                'total_files': coverage_result['basic_coverage']['total_files'],
                'gaps_found': len(coverage_result['basic_coverage']['coverage_gaps'])
            }
            
            # Add coverage recommendations
            detailed_analysis = coverage_result['detailed_analysis']
            for rec in detailed_analysis['recommendations'][:5]:  # Top 5
                results['recommendations'].append({
                    'type': 'coverage',
                    'priority': rec['priority'],
                    'description': rec['description']
                })
        
        except Exception as e:
            print(f"Coverage analysis failed: {e}")
            results['summary']['coverage'] = {'error': str(e)}
        
        # Comprehensive performance analysis
        print("⚡ Running comprehensive performance analysis...")
        try:
            perf_result = self.performance_optimizer.optimize_test_performance()
            perf_report_path = reports_dir / f'performance_nightly_{timestamp}.json'
            self.performance_optimizer.save_optimization_report(perf_result, perf_report_path)
            
            results['reports_generated'].append(str(perf_report_path))
            results['summary']['performance'] = {
                'total_duration': perf_result['performance_profile']['total_duration'],
                'slow_tests': len(perf_result['performance_profile']['slow_tests']),
                'regressions': len(perf_result['regressions']),
                'cache_hit_rate': perf_result['cache_statistics']['hit_rate']
            }
            
            # Add performance recommendations
            for rec in perf_result['recommendations'][:5]:  # Top 5
                results['recommendations'].append({
                    'type': 'performance',
                    'priority': rec['priority'],
                    'description': rec['description'],
                    'estimated_improvement': rec['estimated_improvement']
                })
        
        except Exception as e:
            print(f"Performance analysis failed: {e}")
            results['summary']['performance'] = {'error': str(e)}
        
        # Comprehensive flaky test analysis
        print("🎲 Running comprehensive flaky test analysis...")
        try:
            flaky_result = self.flaky_detector.run_flaky_test_analysis(days=7)  # Weekly analysis
            flaky_report_path = reports_dir / f'flaky_nightly_{timestamp}.json'
            self.flaky_detector.save_analysis_report(flaky_result, flaky_report_path)
            
            results['reports_generated'].append(str(flaky_report_path))
            results['summary']['flakiness'] = {
                'flaky_tests': flaky_result['analysis_metadata']['flaky_tests_found'],
                'quarantined': flaky_result['analysis_metadata']['tests_quarantined'],
                'executions_analyzed': flaky_result['analysis_metadata']['executions_analyzed']
            }
            
            # Add flakiness recommendations
            for rec in flaky_result['recommendations'][:5]:  # Top 5
                results['recommendations'].append({
                    'type': 'flakiness',
                    'priority': rec['priority'],
                    'description': rec['description'],
                    'confidence': rec['confidence']
                })
        
        except Exception as e:
            print(f"Flaky test analysis failed: {e}")
            results['summary']['flakiness'] = {'error': str(e)}
        
        # Save comprehensive nightly report
        nightly_report_path = reports_dir / f'nightly_comprehensive_{timestamp}.json'
        with open(nightly_report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        results['comprehensive_report'] = str(nightly_report_path)
        
        return results
    
    def generate_quality_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for quality dashboard"""
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {},
            'trends': {},
            'alerts': []
        }
        
        # Current metrics
        try:
            # Coverage metrics
            coverage_result = self.coverage_system.run_comprehensive_analysis()
            dashboard_data['metrics']['coverage'] = {
                'overall_percentage': coverage_result['basic_coverage']['overall_percentage'],
                'trend_direction': coverage_result['detailed_analysis']['trends'].get('trend_direction', 'unknown')
            }
            
            # Performance metrics
            perf_result = self.performance_optimizer.optimize_test_performance()
            dashboard_data['metrics']['performance'] = {
                'total_duration': perf_result['performance_profile']['total_duration'],
                'slow_tests_count': len(perf_result['performance_profile']['slow_tests']),
                'regressions_count': len(perf_result['regressions'])
            }
            
            # Flakiness metrics
            flaky_result = self.flaky_detector.run_flaky_test_analysis()
            dashboard_data['metrics']['flakiness'] = {
                'flaky_tests_count': flaky_result['analysis_metadata']['flaky_tests_found'],
                'quarantined_count': flaky_result['analysis_metadata']['tests_quarantined']
            }
            
            # Generate alerts
            if coverage_result['basic_coverage']['overall_percentage'] < 70:
                dashboard_data['alerts'].append({
                    'type': 'coverage',
                    'severity': 'high',
                    'message': f"Coverage below 70%: {coverage_result['basic_coverage']['overall_percentage']:.1f}%"
                })
            
            if len(perf_result['regressions']) > 0:
                dashboard_data['alerts'].append({
                    'type': 'performance',
                    'severity': 'medium',
                    'message': f"{len(perf_result['regressions'])} performance regressions detected"
                })
            
            if flaky_result['analysis_metadata']['flaky_tests_found'] > 5:
                dashboard_data['alerts'].append({
                    'type': 'flakiness',
                    'severity': 'medium',
                    'message': f"{flaky_result['analysis_metadata']['flaky_tests_found']} flaky tests detected"
                })
        
        except Exception as e:
            dashboard_data['alerts'].append({
                'type': 'system',
                'severity': 'high',
                'message': f"Quality analysis failed: {e}"
            })
        
        return dashboard_data


def example_pre_commit_integration():
    """Example of pre-commit hook integration"""
    print("🔗 Pre-commit Integration Example")
    
    project_root = Path.cwd()
    integration = TestQualityIntegration(project_root)
    
    # Run pre-commit checks
    results = integration.run_pre_commit_checks()
    
    print(f"Pre-commit checks: {'✅ PASSED' if results['passed'] else '❌ FAILED'}")
    
    if results['blocking_issues']:
        print("Blocking issues:")
        for issue in results['blocking_issues']:
            print(f"  - {issue}")
        return 1
    
    return 0


def example_ci_integration():
    """Example of CI/CD pipeline integration"""
    print("🚀 CI/CD Integration Example")
    
    project_root = Path.cwd()
    integration = TestQualityIntegration(project_root)
    
    # Run CI quality gates
    results = integration.run_ci_quality_gates()
    
    print(f"Quality gates: {'✅ PASSED' if results['passed'] else '❌ FAILED'}")
    
    if results['failures']:
        print("Gate failures:")
        for failure in results['failures']:
            print(f"  ❌ {failure}")
    
    if results['warnings']:
        print("Warnings:")
        for warning in results['warnings']:
            print(f"  ⚠️ {warning}")
    
    # Save results for CI artifacts
    ci_results_path = project_root / 'ci_quality_results.json'
    with open(ci_results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"CI results saved to: {ci_results_path}")
    
    return 0 if results['passed'] else 1


def example_nightly_analysis():
    """Example of nightly comprehensive analysis"""
    print("🌙 Nightly Analysis Example")
    
    project_root = Path.cwd()
    integration = TestQualityIntegration(project_root)
    
    # Run nightly analysis
    results = integration.run_nightly_analysis()
    
    print("Nightly analysis completed!")
    print(f"Reports generated: {len(results['reports_generated'])}")
    
    # Print summary
    for analysis_type, summary in results['summary'].items():
        if 'error' not in summary:
            print(f"\n{analysis_type.title()} Summary:")
            for key, value in summary.items():
                print(f"  {key}: {value}")
    
    # Print top recommendations
    if results['recommendations']:
        print(f"\nTop Recommendations:")
        for rec in results['recommendations'][:5]:
            print(f"  {rec['type']}: {rec['description']}")
    
    return 0


def example_dashboard_integration():
    """Example of dashboard data generation"""
    print("📊 Dashboard Integration Example")
    
    project_root = Path.cwd()
    integration = TestQualityIntegration(project_root)
    
    # Generate dashboard data
    dashboard_data = integration.generate_quality_dashboard_data()
    
    # Save dashboard data
    dashboard_path = project_root / 'quality_dashboard_data.json'
    with open(dashboard_path, 'w') as f:
        json.dump(dashboard_data, f, indent=2, default=str)
    
    print(f"Dashboard data saved to: {dashboard_path}")
    
    # Print current status
    metrics = dashboard_data['metrics']
    print(f"\nCurrent Quality Metrics:")
    if 'coverage' in metrics:
        print(f"  Coverage: {metrics['coverage']['overall_percentage']:.1f}%")
    if 'performance' in metrics:
        print(f"  Test Duration: {metrics['performance']['total_duration']:.1f}s")
        print(f"  Slow Tests: {metrics['performance']['slow_tests_count']}")
    if 'flakiness' in metrics:
        print(f"  Flaky Tests: {metrics['flakiness']['flaky_tests_count']}")
    
    # Print alerts
    if dashboard_data['alerts']:
        print(f"\nActive Alerts:")
        for alert in dashboard_data['alerts']:
            severity_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(alert['severity'], '⚪')
            print(f"  {severity_emoji} {alert['message']}")
    
    return 0


def main():
    """Run integration examples"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Quality Integration Examples")
    parser.add_argument('example', choices=['pre-commit', 'ci', 'nightly', 'dashboard'], 
                       help='Example to run')
    
    args = parser.parse_args()
    
    examples = {
        'pre-commit': example_pre_commit_integration,
        'ci': example_ci_integration,
        'nightly': example_nightly_analysis,
        'dashboard': example_dashboard_integration
    }
    
    return examples[args.example]()


if __name__ == '__main__':
    sys.exit(main())