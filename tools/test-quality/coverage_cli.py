#!/usr/bin/env python3
"""
Coverage Analysis CLI

Command-line interface for the comprehensive test coverage analysis system.
Provides easy access to coverage analysis, threshold enforcement, and trend tracking.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from coverage_system import ComprehensiveCoverageSystem, CoverageTrendTracker


def setup_coverage_command(subparsers):
    """Setup coverage analysis command"""
    coverage_parser = subparsers.add_parser('analyze', help='Run comprehensive coverage analysis')
    coverage_parser.add_argument('--test-files', nargs='*', help='Specific test files to analyze')
    coverage_parser.add_argument('--output', type=Path, help='Output file for report')
    coverage_parser.add_argument('--base-branch', default='main', help='Base branch for new code analysis')
    coverage_parser.add_argument('--overall-threshold', type=float, default=80.0, help='Overall coverage threshold')
    coverage_parser.add_argument('--new-code-threshold', type=float, default=85.0, help='New code coverage threshold')
    coverage_parser.add_argument('--fail-on-violation', action='store_true', help='Exit with error code if thresholds violated')
    coverage_parser.set_defaults(func=run_coverage_analysis)


def setup_trends_command(subparsers):
    """Setup trends analysis command"""
    trends_parser = subparsers.add_parser('trends', help='Analyze coverage trends')
    trends_parser.add_argument('--days', type=int, default=30, help='Number of days to analyze')
    trends_parser.add_argument('--file', type=str, help='Analyze trends for specific file')
    trends_parser.add_argument('--output', type=Path, help='Output file for trends report')
    trends_parser.add_argument('--format', choices=['json', 'text'], default='text', help='Output format')
    trends_parser.set_defaults(func=run_trends_analysis)


def setup_thresholds_command(subparsers):
    """Setup threshold management command"""
    thresholds_parser = subparsers.add_parser('thresholds', help='Manage coverage thresholds')
    thresholds_parser.add_argument('--set-overall', type=float, help='Set overall coverage threshold')
    thresholds_parser.add_argument('--set-new-code', type=float, help='Set new code coverage threshold')
    thresholds_parser.add_argument('--set-critical', type=float, help='Set critical files coverage threshold')
    thresholds_parser.add_argument('--check-only', action='store_true', help='Only check thresholds, don\'t run full analysis')
    thresholds_parser.add_argument('--base-branch', default='main', help='Base branch for new code analysis')
    thresholds_parser.set_defaults(func=manage_thresholds)


def setup_report_command(subparsers):
    """Setup report generation command"""
    report_parser = subparsers.add_parser('report', help='Generate detailed coverage report')
    report_parser.add_argument('--input', type=Path, help='Input coverage data file')
    report_parser.add_argument('--output', type=Path, help='Output report file')
    report_parser.add_argument('--format', choices=['json', 'html', 'markdown'], default='json', help='Report format')
    report_parser.add_argument('--include-trends', action='store_true', help='Include trend analysis in report')
    report_parser.set_defaults(func=generate_report)


def run_coverage_analysis(args):
    """Run comprehensive coverage analysis"""
    project_root = Path.cwd()
    system = ComprehensiveCoverageSystem(project_root)
    
    # Configure thresholds
    system.threshold_enforcer.set_thresholds(
        overall=args.overall_threshold,
        new_code=args.new_code_threshold
    )
    
    # Prepare test files
    test_files = None
    if args.test_files:
        test_files = [Path(f) for f in args.test_files]
    
    try:
        # Run analysis
        result = system.run_comprehensive_analysis(test_files, args.base_branch)
        
        # Save report
        if args.output:
            system.save_report(result, args.output)
            print(f"Report saved to {args.output}")
        
        # Print summary
        print_coverage_summary(result)
        
        # Check if we should fail on violations
        threshold_result = result['threshold_enforcement']
        if args.fail_on_violation and not threshold_result['passed']:
            print("\n❌ Coverage thresholds not met!")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"Error running coverage analysis: {e}")
        return 1


def run_trends_analysis(args):
    """Run coverage trends analysis"""
    project_root = Path.cwd()
    tracker = CoverageTrendTracker(project_root)
    
    try:
        if args.file:
            # Analyze trends for specific file
            trends = tracker.get_file_trends(args.file, args.days)
            
            if args.format == 'json':
                trend_data = {
                    'file_path': args.file,
                    'days_analyzed': args.days,
                    'data_points': len(trends),
                    'trends': [{'timestamp': t[0].isoformat(), 'coverage': t[1]} for t in trends]
                }
                
                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(trend_data, f, indent=2)
                else:
                    print(json.dumps(trend_data, indent=2))
            else:
                print(f"\nCoverage trends for {args.file} (last {args.days} days):")
                print(f"Data points: {len(trends)}")
                
                if trends:
                    print("\nRecent trends:")
                    for timestamp, coverage in trends[-10:]:  # Show last 10 data points
                        print(f"  {timestamp.strftime('%Y-%m-%d %H:%M')}: {coverage:.1f}%")
                else:
                    print("No trend data available for this file.")
        
        else:
            # Analyze overall trends
            trends = tracker.get_trends(args.days)
            
            if args.format == 'json':
                trend_data = {
                    'days_analyzed': args.days,
                    'data_points': len(trends),
                    'trends': [
                        {
                            'timestamp': t.timestamp.isoformat(),
                            'overall_percentage': t.overall_percentage,
                            'total_files': t.total_files,
                            'covered_files': t.covered_files,
                            'commit_hash': t.commit_hash,
                            'branch': t.branch
                        } for t in trends
                    ]
                }
                
                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(trend_data, f, indent=2)
                else:
                    print(json.dumps(trend_data, indent=2))
            else:
                print(f"\nOverall coverage trends (last {args.days} days):")
                print(f"Data points: {len(trends)}")
                
                if len(trends) >= 2:
                    first_coverage = trends[0].overall_percentage
                    last_coverage = trends[-1].overall_percentage
                    change = last_coverage - first_coverage
                    
                    print(f"Coverage change: {change:+.1f}% ({first_coverage:.1f}% → {last_coverage:.1f}%)")
                    
                    if change > 0:
                        print("📈 Coverage is improving!")
                    elif change < 0:
                        print("📉 Coverage is declining.")
                    else:
                        print("📊 Coverage is stable.")
                    
                    print("\nRecent data points:")
                    for trend in trends[-10:]:  # Show last 10 data points
                        print(f"  {trend.timestamp.strftime('%Y-%m-%d %H:%M')}: {trend.overall_percentage:.1f}% "
                              f"({trend.covered_files}/{trend.total_files} files)")
                else:
                    print("Insufficient data for trend analysis.")
        
        return 0
        
    except Exception as e:
        print(f"Error analyzing trends: {e}")
        return 1


def manage_thresholds(args):
    """Manage coverage thresholds"""
    project_root = Path.cwd()
    system = ComprehensiveCoverageSystem(project_root)
    
    # Set thresholds if provided
    if args.set_overall or args.set_new_code or args.set_critical:
        system.threshold_enforcer.set_thresholds(
            overall=args.set_overall,
            new_code=args.set_new_code,
            critical=args.set_critical
        )
        
        print("Updated coverage thresholds:")
        if args.set_overall:
            print(f"  Overall coverage: {args.set_overall}%")
        if args.set_new_code:
            print(f"  New code coverage: {args.set_new_code}%")
        if args.set_critical:
            print(f"  Critical files coverage: {args.set_critical}%")
    
    if args.check_only:
        # Just check thresholds without full analysis
        try:
            # Run basic coverage analysis
            coverage_report = system.coverage_analyzer.analyze_coverage([])
            
            # Check thresholds
            threshold_result = system.threshold_enforcer.enforce_thresholds(coverage_report, args.base_branch)
            
            print(f"\nThreshold Check Results:")
            print(f"Overall threshold: {'✓' if threshold_result.passed else '✗'}")
            print(f"Current coverage: {threshold_result.overall_coverage:.1f}%")
            print(f"Required threshold: {threshold_result.required_threshold:.1f}%")
            
            if threshold_result.violations:
                print(f"\nViolations ({len(threshold_result.violations)}):")
                for violation in threshold_result.violations:
                    print(f"  - {violation}")
            
            if threshold_result.recommendations:
                print(f"\nRecommendations:")
                for rec in threshold_result.recommendations:
                    print(f"  - {rec}")
            
            return 0 if threshold_result.passed else 1
            
        except Exception as e:
            print(f"Error checking thresholds: {e}")
            return 1
    
    return 0


def generate_report(args):
    """Generate detailed coverage report"""
    project_root = Path.cwd()
    
    try:
        if args.input:
            # Load existing coverage data
            with open(args.input, 'r') as f:
                coverage_data = json.load(f)
        else:
            # Run fresh analysis
            system = ComprehensiveCoverageSystem(project_root)
            coverage_data = system.run_comprehensive_analysis()
        
        if args.format == 'json':
            output_data = coverage_data
        elif args.format == 'html':
            output_data = generate_html_report(coverage_data)
        elif args.format == 'markdown':
            output_data = generate_markdown_report(coverage_data)
        
        if args.output:
            if args.format == 'json':
                with open(args.output, 'w') as f:
                    json.dump(output_data, f, indent=2, default=str)
            else:
                with open(args.output, 'w') as f:
                    f.write(output_data)
            print(f"Report saved to {args.output}")
        else:
            if args.format == 'json':
                print(json.dumps(output_data, indent=2, default=str))
            else:
                print(output_data)
        
        return 0
        
    except Exception as e:
        print(f"Error generating report: {e}")
        return 1


def generate_html_report(coverage_data):
    """Generate HTML coverage report"""
    basic_coverage = coverage_data['basic_coverage']
    detailed_analysis = coverage_data['detailed_analysis']
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Coverage Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background: white; border-radius: 3px; }}
        .file-list {{ margin-top: 20px; }}
        .file-item {{ margin: 5px 0; padding: 10px; background: #f9f9f9; border-radius: 3px; }}
        .coverage-bar {{ width: 200px; height: 20px; background: #ddd; border-radius: 10px; overflow: hidden; }}
        .coverage-fill {{ height: 100%; background: #4CAF50; }}
        .low-coverage {{ background: #f44336; }}
        .medium-coverage {{ background: #ff9800; }}
    </style>
</head>
<body>
    <h1>Coverage Analysis Report</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <div class="metric">
            <strong>Overall Coverage:</strong> {basic_coverage['overall_percentage']:.1f}%
        </div>
        <div class="metric">
            <strong>Files Covered:</strong> {basic_coverage['covered_files']}/{basic_coverage['total_files']}
        </div>
        <div class="metric">
            <strong>Lines Covered:</strong> {basic_coverage['covered_lines']}/{basic_coverage['total_lines']}
        </div>
        <div class="metric">
            <strong>Coverage Gaps:</strong> {len(basic_coverage['coverage_gaps'])}
        </div>
    </div>
    
    <div class="file-list">
        <h2>File Coverage</h2>
"""
    
    for file_analysis in detailed_analysis['file_analysis']:
        coverage_pct = file_analysis['coverage_percentage']
        coverage_class = 'low-coverage' if coverage_pct < 50 else 'medium-coverage' if coverage_pct < 80 else ''
        
        html += f"""
        <div class="file-item">
            <strong>{file_analysis['file_path']}</strong>
            <div class="coverage-bar">
                <div class="coverage-fill {coverage_class}" style="width: {coverage_pct}%"></div>
            </div>
            <span>{coverage_pct:.1f}% ({file_analysis['covered_lines']}/{file_analysis['total_lines']} lines)</span>
        </div>
"""
    
    html += """
    </div>
</body>
</html>
"""
    
    return html


def generate_markdown_report(coverage_data):
    """Generate Markdown coverage report"""
    basic_coverage = coverage_data['basic_coverage']
    detailed_analysis = coverage_data['detailed_analysis']
    threshold_result = coverage_data['threshold_enforcement']
    
    md = f"""# Coverage Analysis Report

## Summary

- **Overall Coverage:** {basic_coverage['overall_percentage']:.1f}%
- **Files Covered:** {basic_coverage['covered_files']}/{basic_coverage['total_files']}
- **Lines Covered:** {basic_coverage['covered_lines']}/{basic_coverage['total_lines']}
- **Coverage Gaps:** {len(basic_coverage['coverage_gaps'])}
- **Threshold Status:** {'✅ Passed' if threshold_result['passed'] else '❌ Failed'}

## File Coverage

| File | Coverage | Lines | Priority |
|------|----------|-------|----------|
"""
    
    for file_analysis in detailed_analysis['file_analysis']:
        coverage_pct = file_analysis['coverage_percentage']
        priority_emoji = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}.get(file_analysis['priority'], '⚪')
        
        md += f"| {file_analysis['file_path']} | {coverage_pct:.1f}% | {file_analysis['covered_lines']}/{file_analysis['total_lines']} | {priority_emoji} {file_analysis['priority']} |\n"
    
    if threshold_result['violations']:
        md += f"\n## Threshold Violations\n\n"
        for violation in threshold_result['violations']:
            md += f"- ❌ {violation}\n"
    
    if detailed_analysis['recommendations']:
        md += f"\n## Recommendations\n\n"
        for rec in detailed_analysis['recommendations']:
            priority_emoji = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢'}.get(rec['priority'], '⚪')
            md += f"- {priority_emoji} **{rec['category']}:** {rec['description']}\n"
    
    return md


def print_coverage_summary(result):
    """Print coverage analysis summary"""
    basic_coverage = result['basic_coverage']
    threshold_result = result['threshold_enforcement']
    detailed_analysis = result['detailed_analysis']
    
    print(f"\n{'='*50}")
    print(f"📊 COVERAGE ANALYSIS SUMMARY")
    print(f"{'='*50}")
    
    print(f"\n📈 Overall Metrics:")
    print(f"  Coverage: {basic_coverage['overall_percentage']:.1f}%")
    print(f"  Files: {basic_coverage['covered_files']}/{basic_coverage['total_files']} covered")
    print(f"  Lines: {basic_coverage['covered_lines']}/{basic_coverage['total_lines']} covered")
    print(f"  Gaps: {len(basic_coverage['coverage_gaps'])} identified")
    
    print(f"\n🎯 Threshold Status:")
    status_emoji = "✅" if threshold_result['passed'] else "❌"
    print(f"  {status_emoji} Overall: {'PASSED' if threshold_result['passed'] else 'FAILED'}")
    
    if threshold_result['violations']:
        print(f"\n⚠️  Violations ({len(threshold_result['violations'])}):")
        for violation in threshold_result['violations'][:5]:  # Show first 5
            print(f"    • {violation}")
        if len(threshold_result['violations']) > 5:
            print(f"    ... and {len(threshold_result['violations']) - 5} more")
    
    if threshold_result['recommendations']:
        print(f"\n💡 Top Recommendations:")
        for rec in threshold_result['recommendations'][:3]:  # Show top 3
            print(f"    • {rec}")
    
    # Show trend if available
    trend_analysis = detailed_analysis.get('trends', {})
    if 'trend_direction' in trend_analysis:
        trend_emoji = {'improving': '📈', 'declining': '📉', 'stable': '📊'}.get(trend_analysis['trend_direction'], '📊')
        print(f"\n{trend_emoji} Trend: {trend_analysis['trend_direction'].title()}")
        if trend_analysis.get('coverage_change'):
            print(f"    Change: {trend_analysis['coverage_change']:+.1f}% over {trend_analysis.get('data_points', 0)} data points")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Comprehensive Test Coverage Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run basic coverage analysis
  python coverage_cli.py analyze
  
  # Run with custom thresholds
  python coverage_cli.py analyze --overall-threshold 85 --new-code-threshold 90
  
  # Analyze coverage trends
  python coverage_cli.py trends --days 14
  
  # Check thresholds only
  python coverage_cli.py thresholds --check-only
  
  # Generate HTML report
  python coverage_cli.py report --format html --output coverage_report.html
        """
    )
    
    parser.add_argument('--project-root', type=Path, default=Path.cwd(), help='Project root directory')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    setup_coverage_command(subparsers)
    setup_trends_command(subparsers)
    setup_thresholds_command(subparsers)
    setup_report_command(subparsers)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Change to project root if specified
    if args.project_root != Path.cwd():
        import os
        os.chdir(args.project_root)
    
    # Execute command
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
