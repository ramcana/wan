#!/usr/bin/env python3
"""
Test Quality Improvement CLI

Unified command-line interface for all test quality improvement tools:
- Coverage analysis and threshold enforcement
- Performance optimization and regression detection
- Flaky test detection and management
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from coverage_system import ComprehensiveCoverageSystem
from performance_optimizer import TestPerformanceOptimizer
from flaky_test_detector import FlakyTestDetectionSystem


def setup_coverage_commands(subparsers):
    """Setup coverage-related commands"""
    coverage_parser = subparsers.add_parser('coverage', help='Coverage analysis commands')
    coverage_subparsers = coverage_parser.add_subparsers(dest='coverage_command', help='Coverage operations')
    
    # Coverage analyze command
    analyze_parser = coverage_subparsers.add_parser('analyze', help='Run comprehensive coverage analysis')
    analyze_parser.add_argument('--test-files', nargs='*', help='Specific test files to analyze')
    analyze_parser.add_argument('--output', type=Path, help='Output file for report')
    analyze_parser.add_argument('--base-branch', default='main', help='Base branch for new code analysis')
    analyze_parser.add_argument('--overall-threshold', type=float, default=80.0, help='Overall coverage threshold')
    analyze_parser.add_argument('--new-code-threshold', type=float, default=85.0, help='New code coverage threshold')
    analyze_parser.add_argument('--fail-on-violation', action='store_true', help='Exit with error if thresholds violated')
    
    # Coverage trends command
    trends_parser = coverage_subparsers.add_parser('trends', help='Analyze coverage trends')
    trends_parser.add_argument('--days', type=int, default=30, help='Number of days to analyze')
    trends_parser.add_argument('--file', type=str, help='Analyze trends for specific file')
    trends_parser.add_argument('--output', type=Path, help='Output file for trends report')
    
    coverage_parser.set_defaults(func=handle_coverage_commands)


def setup_performance_commands(subparsers):
    """Setup performance-related commands"""
    perf_parser = subparsers.add_parser('performance', help='Performance optimization commands')
    perf_subparsers = perf_parser.add_subparsers(dest='performance_command', help='Performance operations')
    
    # Performance analyze command
    analyze_parser = perf_subparsers.add_parser('analyze', help='Run performance analysis')
    analyze_parser.add_argument('--test-files', nargs='*', help='Specific test files to analyze')
    analyze_parser.add_argument('--output', type=Path, help='Output file for report')
    analyze_parser.add_argument('--slow-threshold', type=float, default=5.0, help='Slow test threshold in seconds')
    analyze_parser.add_argument('--clear-cache', action='store_true', help='Clear test cache before analysis')
    
    # Performance trends command
    trends_parser = perf_subparsers.add_parser('trends', help='Analyze performance trends')
    trends_parser.add_argument('--test-id', type=str, help='Specific test to analyze')
    trends_parser.add_argument('--days', type=int, default=30, help='Number of days to analyze')
    trends_parser.add_argument('--output', type=Path, help='Output file for trends report')
    
    perf_parser.set_defaults(func=handle_performance_commands)


def setup_flaky_commands(subparsers):
    """Setup flaky test related commands"""
    flaky_parser = subparsers.add_parser('flaky', help='Flaky test detection commands')
    flaky_subparsers = flaky_parser.add_subparsers(dest='flaky_command', help='Flaky test operations')
    
    # Flaky analyze command
    analyze_parser = flaky_subparsers.add_parser('analyze', help='Run flaky test analysis')
    analyze_parser.add_argument('--days', type=int, default=30, help='Days of history to analyze')
    analyze_parser.add_argument('--output', type=Path, help='Output file for analysis report')
    analyze_parser.add_argument('--flakiness-threshold', type=float, default=0.1, help='Flakiness threshold')
    
    # Record results command
    record_parser = flaky_subparsers.add_parser('record', help='Record test results')
    record_parser.add_argument('results_file', type=Path, help='Test results file to record')
    
    # Quarantine management
    quarantine_parser = flaky_subparsers.add_parser('quarantine', help='Manage quarantined tests')
    quarantine_parser.add_argument('--list', action='store_true', help='List quarantined tests')
    quarantine_parser.add_argument('--release', type=str, help='Release test from quarantine')
    
    flaky_parser.set_defaults(func=handle_flaky_commands)


def setup_comprehensive_commands(subparsers):
    """Setup comprehensive analysis commands"""
    comp_parser = subparsers.add_parser('analyze-all', help='Run comprehensive test quality analysis')
    comp_parser.add_argument('--test-files', nargs='*', help='Specific test files to analyze')
    comp_parser.add_argument('--output-dir', type=Path, help='Output directory for all reports')
    comp_parser.add_argument('--days', type=int, default=30, help='Days of history to analyze')
    comp_parser.add_argument('--coverage-threshold', type=float, default=80.0, help='Coverage threshold')
    comp_parser.add_argument('--performance-threshold', type=float, default=5.0, help='Slow test threshold')
    comp_parser.add_argument('--flakiness-threshold', type=float, default=0.1, help='Flakiness threshold')
    comp_parser.add_argument('--fail-on-issues', action='store_true', help='Exit with error if issues found')
    comp_parser.set_defaults(func=handle_comprehensive_analysis)


def handle_coverage_commands(args):
    """Handle coverage-related commands"""
    project_root = Path.cwd()
    system = ComprehensiveCoverageSystem(project_root)
    
    if args.coverage_command == 'analyze':
        # Configure thresholds
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
        
        # Print summary
        print_coverage_summary(result)
        
        # Check if we should fail on violations
        threshold_result = result['threshold_enforcement']
        if args.fail_on_violation and not threshold_result['passed']:
            return 1
        
        return 0
    
    elif args.coverage_command == 'trends':
        from coverage_system import CoverageTrendTracker
        tracker = CoverageTrendTracker(project_root)
        
        if args.file:
            trends = tracker.get_file_trends(args.file, args.days)
            print(f"Coverage trends for {args.file} (last {args.days} days):")
            for timestamp, coverage in trends[-10:]:
                print(f"  {timestamp.strftime('%Y-%m-%d %H:%M')}: {coverage:.1f}%")
        else:
            trends = tracker.get_trends(args.days)
            if len(trends) >= 2:
                first_coverage = trends[0].overall_percentage
                last_coverage = trends[-1].overall_percentage
                change = last_coverage - first_coverage
                print(f"Overall coverage change: {change:+.1f}% ({first_coverage:.1f}% → {last_coverage:.1f}%)")
        
        return 0
    
    else:
        print("Unknown coverage command")
        return 1


def handle_performance_commands(args):
    """Handle performance-related commands"""
    project_root = Path.cwd()
    optimizer = TestPerformanceOptimizer(project_root)
    
    if args.performance_command == 'analyze':
        # Configure thresholds
        optimizer.profiler.slow_test_threshold = args.slow_threshold
        
        # Clear cache if requested
        if args.clear_cache:
            optimizer.cache_manager.clear_cache()
        
        # Prepare test files
        test_files = None
        if args.test_files:
            test_files = [Path(f) for f in args.test_files]
        
        # Run optimization
        result = optimizer.optimize_test_performance(test_files)
        
        # Save report
        if args.output:
            optimizer.save_optimization_report(result, args.output)
        
        # Print summary
        print_performance_summary(result)
        
        return 0
    
    elif args.performance_command == 'trends':
        from performance_optimizer import PerformanceRegressionDetector
        detector = PerformanceRegressionDetector(project_root)
        
        if args.test_id:
            history = detector.get_performance_history(args.test_id, args.days)
            print(f"Performance history for {args.test_id} (last {args.days} days):")
            for metric in history[-10:]:
                print(f"  {metric.timestamp.strftime('%Y-%m-%d %H:%M')}: {metric.duration:.2f}s ({metric.status})")
        else:
            print("Please specify --test-id for performance trends")
            return 1
        
        return 0
    
    else:
        print("Unknown performance command")
        return 1


def handle_flaky_commands(args):
    """Handle flaky test related commands"""
    project_root = Path.cwd()
    system = FlakyTestDetectionSystem(project_root)
    
    if args.flaky_command == 'analyze':
        # Configure threshold
        system.analyzer.flakiness_threshold = args.flakiness_threshold
        
        # Run analysis
        result = system.run_flaky_test_analysis(args.days)
        
        # Save report
        if args.output:
            system.save_analysis_report(result, args.output)
        
        # Print summary
        print_flaky_summary(result)
        
        return 0
    
    elif args.flaky_command == 'record':
        # Record test results
        system.record_test_run_results(args.results_file)
        return 0
    
    elif args.flaky_command == 'quarantine':
        if args.list:
            # List quarantined tests
            patterns = system.tracker.get_flaky_patterns()
            quarantined = [p for p in patterns if hasattr(p, 'quarantined') and p.quarantined]
            
            if quarantined:
                print("Quarantined tests:")
                for pattern in quarantined:
                    print(f"  {pattern.test_id}: {pattern.flakiness_score:.2f} flakiness score")
            else:
                print("No tests currently quarantined.")
        
        elif args.release:
            # Release test from quarantine
            with sqlite3.connect(system.tracker.db_path) as conn:
                conn.execute('''
                    UPDATE flaky_test_patterns
                    SET quarantined = FALSE, quarantine_reason = NULL
                    WHERE test_id = ?
                ''', (args.release,))
                conn.commit()
            print(f"Released {args.release} from quarantine.")
        
        return 0
    
    else:
        print("Unknown flaky command")
        return 1


def handle_comprehensive_analysis(args):
    """Handle comprehensive test quality analysis"""
    project_root = Path.cwd()
    
    print("🧪 Starting Comprehensive Test Quality Analysis")
    print("=" * 60)
    
    # Setup output directory
    output_dir = args.output_dir or project_root / '.test_quality_reports'
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare test files
    test_files = None
    if args.test_files:
        test_files = [Path(f) for f in args.test_files]
    
    issues_found = False
    
    # 1. Coverage Analysis
    print("\n📊 Running Coverage Analysis...")
    try:
        coverage_system = ComprehensiveCoverageSystem(project_root)
        coverage_system.threshold_enforcer.set_thresholds(overall=args.coverage_threshold)
        
        coverage_result = coverage_system.run_comprehensive_analysis(test_files)
        coverage_output = output_dir / f'coverage_report_{timestamp}.json'
        coverage_system.save_report(coverage_result, coverage_output)
        
        print_coverage_summary(coverage_result)
        
        # Check for coverage issues
        threshold_result = coverage_result['threshold_enforcement']
        if not threshold_result['passed']:
            issues_found = True
            print("❌ Coverage thresholds not met!")
        
    except Exception as e:
        print(f"❌ Coverage analysis failed: {e}")
        issues_found = True
    
    # 2. Performance Analysis
    print("\n⚡ Running Performance Analysis...")
    try:
        performance_optimizer = TestPerformanceOptimizer(project_root)
        performance_optimizer.profiler.slow_test_threshold = args.performance_threshold
        
        performance_result = performance_optimizer.optimize_test_performance(test_files)
        performance_output = output_dir / f'performance_report_{timestamp}.json'
        performance_optimizer.save_optimization_report(performance_result, performance_output)
        
        print_performance_summary(performance_result)
        
        # Check for performance issues
        profile = performance_result['performance_profile']
        if len(profile['slow_tests']) > 0 or len(performance_result['regressions']) > 0:
            issues_found = True
            print("⚠️ Performance issues detected!")
        
    except Exception as e:
        print(f"❌ Performance analysis failed: {e}")
        issues_found = True
    
    # 3. Flaky Test Analysis
    print("\n🎲 Running Flaky Test Analysis...")
    try:
        flaky_system = FlakyTestDetectionSystem(project_root)
        flaky_system.analyzer.flakiness_threshold = args.flakiness_threshold
        
        flaky_result = flaky_system.run_flaky_test_analysis(args.days)
        flaky_output = output_dir / f'flaky_report_{timestamp}.json'
        flaky_system.save_analysis_report(flaky_result, flaky_output)
        
        print_flaky_summary(flaky_result)
        
        # Check for flaky test issues
        if flaky_result['analysis_metadata']['flaky_tests_found'] > 0:
            issues_found = True
            print("⚠️ Flaky tests detected!")
        
    except Exception as e:
        print(f"❌ Flaky test analysis failed: {e}")
        issues_found = True
    
    # 4. Generate Comprehensive Report
    print("\n📋 Generating Comprehensive Report...")
    try:
        comprehensive_report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'project_root': str(project_root),
            'test_files_analyzed': len(test_files) if test_files else 'all',
            'coverage_analysis': coverage_result if 'coverage_result' in locals() else None,
            'performance_analysis': performance_result if 'performance_result' in locals() else None,
            'flaky_test_analysis': flaky_result if 'flaky_result' in locals() else None,
            'summary': {
                'issues_found': issues_found,
                'analysis_successful': True
            }
        }
        
        comprehensive_output = output_dir / f'comprehensive_report_{timestamp}.json'
        with open(comprehensive_output, 'w') as f:
            json.dump(comprehensive_report, f, indent=2, default=str)
        
        print(f"📄 Comprehensive report saved to: {comprehensive_output}")
        
    except Exception as e:
        print(f"❌ Failed to generate comprehensive report: {e}")
        issues_found = True
    
    # Final Summary
    print("\n" + "=" * 60)
    print("🏁 Test Quality Analysis Complete")
    
    if issues_found:
        print("❌ Issues found - review reports for details")
        if args.fail_on_issues:
            return 1
    else:
        print("✅ No significant issues detected")
    
    print(f"📁 Reports saved to: {output_dir}")
    
    return 0


def print_coverage_summary(result):
    """Print coverage analysis summary"""
    basic_coverage = result['basic_coverage']
    threshold_result = result['threshold_enforcement']
    
    print(f"  Overall coverage: {basic_coverage['overall_percentage']:.1f}%")
    print(f"  Files covered: {basic_coverage['covered_files']}/{basic_coverage['total_files']}")
    print(f"  Threshold status: {'✅ PASSED' if threshold_result['passed'] else '❌ FAILED'}")
    
    if threshold_result['violations']:
        print(f"  Violations: {len(threshold_result['violations'])}")


def print_performance_summary(result):
    """Print performance analysis summary"""
    profile = result['performance_profile']
    regressions = result['regressions']
    
    print(f"  Total execution time: {profile['total_duration']:.1f}s")
    print(f"  Average test duration: {profile['average_duration']:.2f}s")
    print(f"  Slow tests: {len(profile['slow_tests'])}")
    print(f"  Performance regressions: {len(regressions)}")
    
    if profile['slow_tests']:
        slowest = profile['slow_tests'][0]
        print(f"  Slowest test: {slowest['test_name']} ({slowest['duration']:.1f}s)")


def print_flaky_summary(result):
    """Print flaky test analysis summary"""
    metadata = result['analysis_metadata']
    patterns = result['flaky_patterns']
    
    print(f"  Executions analyzed: {metadata['executions_analyzed']}")
    print(f"  Flaky tests found: {metadata['flaky_tests_found']}")
    print(f"  Tests quarantined: {metadata['tests_quarantined']}")
    
    if patterns:
        flakiest = patterns[0]
        print(f"  Most flaky test: {flakiest['test_id']} ({flakiest['flakiness_score']:.2f} score)")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Test Quality Improvement System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run comprehensive analysis
  python test_quality_cli.py analyze-all
  
  # Coverage analysis only
  python test_quality_cli.py coverage analyze --overall-threshold 85
  
  # Performance analysis only
  python test_quality_cli.py performance analyze --slow-threshold 3.0
  
  # Flaky test analysis only
  python test_quality_cli.py flaky analyze --days 14
  
  # Coverage trends
  python test_quality_cli.py coverage trends --days 7
  
  # Record test results for flaky detection
  python test_quality_cli.py flaky record test_results.xml
        """
    )
    
    parser.add_argument('--project-root', type=Path, default=Path.cwd(), help='Project root directory')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    setup_coverage_commands(subparsers)
    setup_performance_commands(subparsers)
    setup_flaky_commands(subparsers)
    setup_comprehensive_commands(subparsers)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Change to project root if specified
    if args.project_root != Path.cwd():
        import os
os.chdir(args.project_root)
    
    # Execute command
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())