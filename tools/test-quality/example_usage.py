#!/usr/bin/env python3
"""
Example Usage of Comprehensive Coverage System

This script demonstrates how to use the coverage analysis system
programmatically and shows various analysis scenarios.
"""

import json
from pathlib import Path
from datetime import datetime

from coverage_system import (
    ComprehensiveCoverageSystem,
    CoverageTrendTracker,
    NewCodeCoverageAnalyzer,
    CoverageThresholdEnforcer,
    DetailedCoverageReporter
)


def example_basic_analysis():
    """Example: Basic coverage analysis"""
    print("=== Basic Coverage Analysis ===")
    
    project_root = Path.cwd()
    system = ComprehensiveCoverageSystem(project_root)
    
    # Run comprehensive analysis
    result = system.run_comprehensive_analysis()
    
    # Extract key metrics
    basic_coverage = result['basic_coverage']
    print(f"Overall coverage: {basic_coverage['overall_percentage']:.1f}%")
    print(f"Files covered: {basic_coverage['covered_files']}/{basic_coverage['total_files']}")
    print(f"Coverage gaps: {len(basic_coverage['coverage_gaps'])}")
    
    # Show top recommendations
    detailed_analysis = result['detailed_analysis']
    print("\nTop recommendations:")
    for i, rec in enumerate(detailed_analysis['recommendations'][:3], 1):
        print(f"{i}. {rec['description']}")


def example_threshold_enforcement():
    """Example: Coverage threshold enforcement"""
    print("\n=== Threshold Enforcement ===")
    
    project_root = Path.cwd()
    system = ComprehensiveCoverageSystem(project_root)
    
    # Configure strict thresholds
    system.threshold_enforcer.set_thresholds(
        overall=85.0,
        new_code=90.0,
        critical=95.0
    )
    
    # Run analysis with threshold checking
    result = system.run_comprehensive_analysis(base_branch='main')
    
    threshold_result = result['threshold_enforcement']
    print(f"Thresholds passed: {'✅' if threshold_result['passed'] else '❌'}")
    print(f"Overall coverage: {threshold_result['overall_coverage']:.1f}%")
    print(f"Required threshold: {threshold_result['required_threshold']:.1f}%")
    
    if threshold_result['violations']:
        print(f"\nViolations ({len(threshold_result['violations'])}):")
        for violation in threshold_result['violations']:
            print(f"  - {violation}")
    
    # Show new code analysis
    print(f"\nNew code analysis:")
    for result in threshold_result['new_code_results']:
        status = "✅" if result['meets_threshold'] else "❌"
        print(f"  {status} {result['file_path']}: {result['new_code_coverage']:.1f}%")


def example_trend_analysis():
    """Example: Coverage trend analysis"""
    print("\n=== Trend Analysis ===")
    
    project_root = Path.cwd()
    tracker = CoverageTrendTracker(project_root)
    
    # Get recent trends
    trends = tracker.get_trends(days=30)
    
    if len(trends) >= 2:
        first_coverage = trends[0].overall_percentage
        last_coverage = trends[-1].overall_percentage
        change = last_coverage - first_coverage
        
        print(f"Coverage trend over {len(trends)} data points:")
        print(f"  Start: {first_coverage:.1f}%")
        print(f"  End: {last_coverage:.1f}%")
        print(f"  Change: {change:+.1f}%")
        
        if change > 0:
            print("  📈 Coverage is improving!")
        elif change < 0:
            print("  📉 Coverage is declining.")
        else:
            print("  📊 Coverage is stable.")
        
        # Show recent data points
        print("\nRecent trends:")
        for trend in trends[-5:]:
            print(f"  {trend.timestamp.strftime('%Y-%m-%d')}: {trend.overall_percentage:.1f}%")
    else:
        print("Insufficient trend data available.")


def example_file_specific_analysis():
    """Example: File-specific coverage analysis"""
    print("\n=== File-Specific Analysis ===")
    
    project_root = Path.cwd()
    system = ComprehensiveCoverageSystem(project_root)
    
    # Run analysis
    result = system.run_comprehensive_analysis()
    
    # Analyze specific files
    detailed_analysis = result['detailed_analysis']
    file_analysis = detailed_analysis['file_analysis']
    
    # Find files with low coverage
    low_coverage_files = [f for f in file_analysis if f['coverage_percentage'] < 60]
    
    print(f"Files with <60% coverage ({len(low_coverage_files)}):")
    for file_info in low_coverage_files[:5]:  # Show top 5
        print(f"  📄 {file_info['file_path']}: {file_info['coverage_percentage']:.1f}%")
        print(f"     Lines: {file_info['covered_lines']}/{file_info['total_lines']}")
        print(f"     Priority: {file_info['priority']}")
        
        # Show function coverage
        func_coverage = file_info['function_coverage']
        if func_coverage['uncovered']:
            print(f"     Uncovered functions: {', '.join(func_coverage['uncovered'][:3])}")
        print()


def example_gap_analysis():
    """Example: Coverage gap analysis"""
    print("\n=== Gap Analysis ===")
    
    project_root = Path.cwd()
    system = ComprehensiveCoverageSystem(project_root)
    
    # Run analysis
    result = system.run_comprehensive_analysis()
    
    # Analyze gaps
    detailed_analysis = result['detailed_analysis']
    gap_analysis = detailed_analysis['gap_analysis']
    
    print(f"Total coverage gaps: {gap_analysis['total_gaps']}")
    print(f"Gaps by type: {gap_analysis['gaps_by_type']}")
    print(f"Gaps by severity: {gap_analysis['gaps_by_severity']}")
    
    # Show critical gaps
    critical_gaps = gap_analysis.get('critical_gaps', [])
    if critical_gaps:
        print(f"\nCritical gaps ({len(critical_gaps)}):")
        for gap in critical_gaps[:3]:  # Show top 3
            print(f"  🔴 {gap['file_path']}:{gap['line_start']}-{gap['line_end']}")
            print(f"     Type: {gap['gap_type']}, Name: {gap['name']}")
            print(f"     Suggestion: {gap['suggestion']}")
    
    # Show high priority gaps
    high_gaps = gap_analysis.get('high_priority_gaps', [])
    if high_gaps:
        print(f"\nHigh priority gaps ({len(high_gaps)}):")
        for gap in high_gaps[:3]:  # Show top 3
            print(f"  🟠 {gap['file_path']}:{gap['line_start']}-{gap['line_end']}")
            print(f"     Type: {gap['gap_type']}, Name: {gap['name']}")


def example_actionable_items():
    """Example: Generate actionable items"""
    print("\n=== Actionable Items ===")
    
    project_root = Path.cwd()
    system = ComprehensiveCoverageSystem(project_root)
    
    # Run analysis
    result = system.run_comprehensive_analysis()
    
    # Get actionable items
    detailed_analysis = result['detailed_analysis']
    actionable_items = detailed_analysis['actionable_items']
    
    print(f"Actionable items found: {len(actionable_items)}")
    
    # Group by type
    items_by_type = {}
    for item in actionable_items:
        item_type = item['type']
        if item_type not in items_by_type:
            items_by_type[item_type] = []
        items_by_type[item_type].append(item)
    
    for item_type, items in items_by_type.items():
        print(f"\n{item_type.replace('_', ' ').title()} ({len(items)} items):")
        
        for item in items[:3]:  # Show top 3 per type
            effort_emoji = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}.get(item['effort'], '⚪')
            impact_emoji = {'low': '🔵', 'medium': '🟡', 'high': '🔴'}.get(item['impact'], '⚪')
            
            print(f"  {effort_emoji} {impact_emoji} {item['title']}")
            print(f"     {item['description']}")
            print(f"     Effort: {item['effort']}, Impact: {item['impact']}")


def example_custom_thresholds():
    """Example: Custom threshold configuration"""
    print("\n=== Custom Threshold Configuration ===")
    
    project_root = Path.cwd()
    enforcer = CoverageThresholdEnforcer(project_root)
    
    # Configure custom thresholds
    enforcer.set_thresholds(
        overall=75.0,      # Lower overall threshold
        new_code=95.0,     # Very high new code threshold
        critical=98.0      # Extremely high critical file threshold
    )
    
    # Add custom critical file patterns
    enforcer.critical_file_patterns.add('models/')
    enforcer.critical_file_patterns.add('security/')
    
    print("Custom thresholds configured:")
    print(f"  Overall: {enforcer.overall_threshold}%")
    print(f"  New code: {enforcer.new_code_threshold}%")
    print(f"  Critical files: {enforcer.critical_files_threshold}%")
    print(f"  Critical patterns: {enforcer.critical_file_patterns}")


def example_report_generation():
    """Example: Generate different report formats"""
    print("\n=== Report Generation ===")
    
    project_root = Path.cwd()
    system = ComprehensiveCoverageSystem(project_root)
    
    # Run analysis
    result = system.run_comprehensive_analysis()
    
    # Save comprehensive JSON report
    json_output = project_root / 'coverage_analysis.json'
    system.save_report(result, json_output)
    print(f"JSON report saved to: {json_output}")
    
    # Generate HTML report using CLI
    from coverage_cli import generate_html_report
    html_content = generate_html_report(result)
    
    html_output = project_root / 'coverage_report.html'
    with open(html_output, 'w') as f:
        f.write(html_content)
    print(f"HTML report saved to: {html_output}")
    
    # Generate Markdown report
    from coverage_cli import generate_markdown_report
    md_content = generate_markdown_report(result)
    
    md_output = project_root / 'COVERAGE_REPORT.md'
    with open(md_output, 'w') as f:
        f.write(md_content)
    print(f"Markdown report saved to: {md_output}")


def example_ci_integration():
    """Example: CI/CD integration pattern"""
    print("\n=== CI/CD Integration Example ===")
    
    project_root = Path.cwd()
    system = ComprehensiveCoverageSystem(project_root)
    
    # Configure for CI environment
    system.threshold_enforcer.set_thresholds(
        overall=80.0,
        new_code=85.0
    )
    
    try:
        # Run analysis
        result = system.run_comprehensive_analysis(base_branch='origin/main')
        
        # Check results
        threshold_result = result['threshold_enforcement']
        
        if threshold_result['passed']:
            print("✅ Coverage requirements met - CI can proceed")
            
            # Save artifacts for later use
            system.save_report(result, project_root / 'ci_coverage_report.json')
            
            return 0  # Success exit code
        else:
            print("❌ Coverage requirements not met - CI should fail")
            
            # Print specific violations for developer feedback
            for violation in threshold_result['violations']:
                print(f"  VIOLATION: {violation}")
            
            # Print actionable recommendations
            detailed_analysis = result['detailed_analysis']
            for rec in detailed_analysis['recommendations'][:3]:
                print(f"  RECOMMENDATION: {rec['description']}")
            
            return 1  # Failure exit code
            
    except Exception as e:
        print(f"❌ Coverage analysis failed: {e}")
        return 1


def main():
    """Run all examples"""
    print("🧪 Comprehensive Coverage System Examples")
    print("=" * 50)
    
    try:
        example_basic_analysis()
        example_threshold_enforcement()
        example_trend_analysis()
        example_file_specific_analysis()
        example_gap_analysis()
        example_actionable_items()
        example_custom_thresholds()
        example_report_generation()
        
        # CI example (commented out to avoid affecting actual CI)
        # exit_code = example_ci_integration()
        # print(f"\nCI integration would exit with code: {exit_code}")
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Example execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
