"""
Example usage of the code quality checking system.
"""

from pathlib import Path
from tools.code_quality import QualityChecker, QualityConfig


def main():
    """Demonstrate code quality checking system usage."""
    
    # Example 1: Basic quality check with default configuration
    print("=== Example 1: Basic Quality Check ===")
    checker = QualityChecker()
    report = checker.check_quality(Path("tools/code-quality"))
    
    print(f"Files analyzed: {report.files_analyzed}")
    print(f"Total issues: {report.total_issues}")
    print(f"Quality score: {report.quality_score:.1f}/100")
    print(f"Documentation coverage: {report.metrics.documentation_coverage:.1f}%")
    print()
    
    # Example 2: Quality check with custom configuration
    print("=== Example 2: Custom Configuration ===")
    config = QualityConfig(
        max_line_length=100,
        require_function_docstrings=False,
        max_cyclomatic_complexity=15
    )
    checker = QualityChecker(config)
    report = checker.check_quality(Path("tools/code-quality"))
    
    print(f"Quality score with relaxed rules: {report.quality_score:.1f}/100")
    print()
    
    # Example 3: Load configuration from file
    print("=== Example 3: Configuration from File ===")
    config_path = Path("tools/code-quality/config/quality-config.yaml")
    if config_path.exists():
        checker = QualityChecker.from_config_file(config_path)
        report = checker.check_quality(Path("tools/code-quality"))
        print(f"Quality score from config file: {report.quality_score:.1f}/100")
    else:
        print("Config file not found")
    print()
    
    # Example 4: Specific checks only
    print("=== Example 4: Specific Checks ===")
    checker = QualityChecker()
    report = checker.check_quality(
        Path("tools/code-quality"), 
        checks=['documentation', 'complexity']
    )
    
    print(f"Documentation and complexity issues: {report.total_issues}")
    print()
    
    # Example 5: Auto-fix issues
    print("=== Example 5: Auto-fix Issues ===")
    # Note: This would modify files, so we'll just show the concept
    # report = checker.fix_issues(Path("tools/code-quality"))
    # print(f"Fixed {report.auto_fixable_issues} issues")
    print("Auto-fix would repair formatting and import issues")
    print()
    
    # Example 6: Generate different report formats
    print("=== Example 6: Report Generation ===")
    checker = QualityChecker()
    report = checker.check_quality(Path("tools/code-quality"))
    
    # JSON report
    json_report = checker.generate_report(report, 'json')
    print(f"JSON report length: {len(json_report)} characters")
    
    # Text report
    text_report = checker.generate_report(report, 'text')
    print("Text report preview:")
    print(text_report[:300] + "..." if len(text_report) > 300 else text_report)
    print()
    
    # Example 7: Issue analysis
    print("=== Example 7: Issue Analysis ===")
    if report.issues:
        print("Top 5 issues:")
        for i, issue in enumerate(report.issues[:5], 1):
            print(f"{i}. {issue.file_path.name}:{issue.line_number} - {issue.message}")
        
        # Group issues by type
        from collections import defaultdict
        issues_by_type = defaultdict(int)
        for issue in report.issues:
            issues_by_type[issue.issue_type.value] += 1
        
        print("\nIssues by type:")
        for issue_type, count in sorted(issues_by_type.items()):
            print(f"  {issue_type}: {count}")
    
    print()
    
    # Example 8: Quality metrics analysis
    print("=== Example 8: Quality Metrics ===")
    metrics = report.metrics
    print(f"Code lines: {metrics.code_lines}")
    print(f"Functions: {metrics.functions_count}")
    print(f"Classes: {metrics.classes_count}")
    print(f"Average complexity: {metrics.complexity_score:.1f}")
    print(f"Maintainability index: {metrics.maintainability_index:.1f}")
    
    # Quality assessment
    if report.quality_score >= 90:
        print("🟢 Excellent code quality!")
    elif report.quality_score >= 75:
        print("🟡 Good code quality with room for improvement")
    elif report.quality_score >= 60:
        print("🟠 Moderate code quality, consider refactoring")
    else:
        print("🔴 Poor code quality, immediate attention needed")


if __name__ == "__main__":
    main()