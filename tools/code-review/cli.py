"""
Code Review and Refactoring CLI

Command-line interface for the code review and refactoring assistance tools.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from tools.code-review.code_reviewer import CodeReviewer, ReviewSeverity, IssueCategory
from tools.code-review.refactoring_engine import RefactoringEngine, RefactoringType
from tools.code-review.technical_debt_tracker import (
    TechnicalDebtTracker, TechnicalDebtItem, DebtCategory, 
    DebtSeverity, DebtStatus
)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Code Review and Refactoring Assistance Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Review entire project
  python -m tools.code-review.cli review --project-root .

  # Review specific file
  python -m tools.code-review.cli review --file src/main.py

  # Generate refactoring suggestions
  python -m tools.code-review.cli refactor --project-root .

  # Track technical debt
  python -m tools.code-review.cli debt list

  # Add technical debt item
  python -m tools.code-review.cli debt add --file src/main.py --title "Complex function" --severity high

  # Generate comprehensive report
  python -m tools.code-review.cli report --output-dir reports/
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Review command
    review_parser = subparsers.add_parser('review', help='Perform code review')
    review_parser.add_argument('--project-root', default='.', help='Project root directory')
    review_parser.add_argument('--file', help='Specific file to review')
    review_parser.add_argument('--output', help='Output file for results')
    review_parser.add_argument('--format', choices=['json', 'text'], default='text', help='Output format')
    review_parser.add_argument('--severity', choices=['critical', 'high', 'medium', 'low'], 
                              help='Filter by minimum severity')
    review_parser.add_argument('--category', choices=[c.value for c in IssueCategory], 
                              help='Filter by category')
    
    # Refactor command
    refactor_parser = subparsers.add_parser('refactor', help='Generate refactoring suggestions')
    refactor_parser.add_argument('--project-root', default='.', help='Project root directory')
    refactor_parser.add_argument('--file', help='Specific file to analyze')
    refactor_parser.add_argument('--output', help='Output file for suggestions')
    refactor_parser.add_argument('--type', choices=[t.value for t in RefactoringType], 
                                help='Filter by refactoring type')
    refactor_parser.add_argument('--priority', type=int, help='Filter by minimum priority')
    
    # Debt command
    debt_parser = subparsers.add_parser('debt', help='Manage technical debt')
    debt_subparsers = debt_parser.add_subparsers(dest='debt_action', help='Debt actions')
    
    # Debt list
    list_parser = debt_subparsers.add_parser('list', help='List technical debt items')
    list_parser.add_argument('--category', choices=[c.value for c in DebtCategory], 
                            help='Filter by category')
    list_parser.add_argument('--severity', choices=[s.value for s in DebtSeverity], 
                            help='Filter by severity')
    list_parser.add_argument('--status', choices=[s.value for s in DebtStatus], 
                            help='Filter by status')
    list_parser.add_argument('--limit', type=int, default=20, help='Limit number of results')
    
    # Debt add
    add_parser = debt_subparsers.add_parser('add', help='Add technical debt item')
    add_parser.add_argument('--file', required=True, help='File path')
    add_parser.add_argument('--title', required=True, help='Debt item title')
    add_parser.add_argument('--description', help='Detailed description')
    add_parser.add_argument('--category', choices=[c.value for c in DebtCategory], 
                           default='code_quality', help='Debt category')
    add_parser.add_argument('--severity', choices=[s.value for s in DebtSeverity], 
                           default='medium', help='Debt severity')
    add_parser.add_argument('--line-start', type=int, default=1, help='Start line number')
    add_parser.add_argument('--line-end', type=int, help='End line number')
    add_parser.add_argument('--effort', type=float, default=4.0, help='Estimated effort in hours')
    add_parser.add_argument('--business-impact', help='Business impact description')
    add_parser.add_argument('--technical-impact', help='Technical impact description')
    
    # Debt update
    update_parser = debt_subparsers.add_parser('update', help='Update technical debt item')
    update_parser.add_argument('--id', required=True, help='Debt item ID')
    update_parser.add_argument('--status', choices=[s.value for s in DebtStatus], 
                              help='New status')
    update_parser.add_argument('--assignee', help='Assignee')
    update_parser.add_argument('--notes', help='Resolution notes')
    
    # Debt metrics
    metrics_parser = debt_subparsers.add_parser('metrics', help='Show debt metrics')
    metrics_parser.add_argument('--output', help='Output file for metrics')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate comprehensive report')
    report_parser.add_argument('--project-root', default='.', help='Project root directory')
    report_parser.add_argument('--output-dir', default='reports', help='Output directory')
    report_parser.add_argument('--format', choices=['json', 'html'], default='json', 
                              help='Report format')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'review':
            handle_review_command(args)
        elif args.command == 'refactor':
            handle_refactor_command(args)
        elif args.command == 'debt':
            handle_debt_command(args)
        elif args.command == 'report':
            handle_report_command(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def handle_review_command(args):
    """Handle code review command"""
    reviewer = CodeReviewer(args.project_root)
    
    if args.file:
        # Review single file
        issues = reviewer.review_file(args.file)
    else:
        # Review entire project
        result = reviewer.review_project()
        issues = reviewer.issues
        print(f"Reviewed {result['files_reviewed']} files")
        print(f"Found {result['issues']} issues")
    
    # Filter issues
    if args.severity:
        min_severity = ReviewSeverity(args.severity)
        severity_order = [ReviewSeverity.CRITICAL, ReviewSeverity.HIGH, 
                         ReviewSeverity.MEDIUM, ReviewSeverity.LOW]
        min_index = severity_order.index(min_severity)
        issues = [i for i in issues if severity_order.index(i.severity) <= min_index]
    
    if args.category:
        category = IssueCategory(args.category)
        issues = [i for i in issues if i.category == category]
    
    # Output results
    if args.format == 'json':
        output_data = [
            {
                "file_path": issue.file_path,
                "line_number": issue.line_number,
                "column": issue.column,
                "severity": issue.severity.value,
                "category": issue.category.value,
                "message": issue.message,
                "suggestion": issue.suggestion,
                "rule_id": issue.rule_id
            }
            for issue in issues
        ]
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
        else:
            print(json.dumps(output_data, indent=2))
    else:
        # Text format
        if not issues:
            print("No issues found!")
            return
        
        for issue in issues:
            print(f"\n{issue.severity.value.upper()}: {issue.message}")
            print(f"  File: {issue.file_path}:{issue.line_number}:{issue.column}")
            print(f"  Category: {issue.category.value}")
            print(f"  Rule: {issue.rule_id}")
            print(f"  Suggestion: {issue.suggestion}")


def handle_refactor_command(args):
    """Handle refactoring command"""
    engine = RefactoringEngine(args.project_root)
    
    if args.file:
        # Analyze single file
        suggestions = engine.analyze_file(args.file)
    else:
        # Analyze entire project
        suggestions = []
        for py_file in Path(args.project_root).glob("**/*.py"):
            suggestions.extend(engine.analyze_file(str(py_file)))
    
    engine.suggestions = suggestions
    
    # Filter suggestions
    if args.type:
        refactor_type = RefactoringType(args.type)
        suggestions = [s for s in suggestions if s.refactoring_type == refactor_type]
    
    if args.priority:
        suggestions = [s for s in suggestions if s.priority <= args.priority]
    
    # Sort by priority
    suggestions.sort(key=lambda x: x.priority)
    
    # Output results
    output_data = [
        {
            "file_path": suggestion.file_path,
            "start_line": suggestion.start_line,
            "end_line": suggestion.end_line,
            "type": suggestion.refactoring_type.value,
            "title": suggestion.title,
            "description": suggestion.description,
            "benefits": suggestion.benefits,
            "effort_estimate": suggestion.effort_estimate,
            "confidence": suggestion.confidence,
            "priority": suggestion.priority
        }
        for suggestion in suggestions
    ]
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
    else:
        if not suggestions:
            print("No refactoring suggestions found!")
            return
        
        for suggestion in suggestions:
            print(f"\nPRIORITY {suggestion.priority}: {suggestion.title}")
            print(f"  File: {suggestion.file_path}:{suggestion.start_line}-{suggestion.end_line}")
            print(f"  Type: {suggestion.refactoring_type.value}")
            print(f"  Description: {suggestion.description}")
            print(f"  Benefits: {', '.join(suggestion.benefits)}")
            print(f"  Effort: {suggestion.effort_estimate}")
            print(f"  Confidence: {suggestion.confidence:.1%}")


def handle_debt_command(args):
    """Handle technical debt command"""
    tracker = TechnicalDebtTracker()
    
    if args.debt_action == 'list':
        items = tracker.debt_items
        
        # Apply filters
        if args.category:
            category = DebtCategory(args.category)
            items = [i for i in items if i.category == category]
        
        if args.severity:
            severity = DebtSeverity(args.severity)
            items = [i for i in items if i.severity == severity]
        
        if args.status:
            status = DebtStatus(args.status)
            items = [i for i in items if i.status == status]
        
        # Sort by priority and limit
        items = sorted(items, key=lambda x: x.priority_score, reverse=True)[:args.limit]
        
        if not items:
            print("No technical debt items found!")
            return
        
        print(f"Technical Debt Items (showing {len(items)}):")
        print("-" * 80)
        
        for item in items:
            print(f"\nID: {item.id}")
            print(f"Title: {item.title}")
            print(f"File: {item.file_path}:{item.line_start}")
            print(f"Category: {item.category.value}")
            print(f"Severity: {item.severity.value}")
            print(f"Status: {item.status.value}")
            print(f"Priority Score: {item.priority_score:.1f}")
            print(f"Estimated Effort: {item.estimated_effort_hours} hours")
            if item.assignee:
                print(f"Assignee: {item.assignee}")
            if item.description:
                print(f"Description: {item.description}")
    
    elif args.debt_action == 'add':
        item = TechnicalDebtItem(
            id="",  # Will be generated
            title=args.title,
            description=args.description or "",
            file_path=args.file,
            line_start=args.line_start,
            line_end=args.line_end or args.line_start,
            category=DebtCategory(args.category),
            severity=DebtSeverity(args.severity),
            status=DebtStatus.IDENTIFIED,
            created_date=datetime.now(),
            updated_date=datetime.now(),
            estimated_effort_hours=args.effort,
            business_impact=args.business_impact or "",
            technical_impact=args.technical_impact or "",
            priority_score=0  # Will be calculated
        )
        
        item_id = tracker.add_debt_item(item)
        print(f"Added technical debt item: {item_id}")
    
    elif args.debt_action == 'update':
        updates = {}
        if args.status:
            updates['status'] = args.status
        if args.assignee:
            updates['assignee'] = args.assignee
        if args.notes:
            updates['resolution_notes'] = args.notes
        
        if tracker.update_debt_item(args.id, updates):
            print(f"Updated debt item: {args.id}")
        else:
            print(f"Debt item not found: {args.id}")
    
    elif args.debt_action == 'metrics':
        metrics = tracker.calculate_debt_metrics()
        
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "total_items": metrics.total_items,
            "total_estimated_hours": metrics.total_estimated_hours,
            "items_by_category": metrics.items_by_category,
            "items_by_severity": metrics.items_by_severity,
            "items_by_status": metrics.items_by_status,
            "average_age_days": metrics.average_age_days,
            "oldest_item_days": metrics.oldest_item_days,
            "debt_trend": metrics.debt_trend,
            "resolution_rate": metrics.resolution_rate
        }
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
        else:
            print("Technical Debt Metrics:")
            print("-" * 40)
            print(f"Total Items: {metrics.total_items}")
            print(f"Total Estimated Hours: {metrics.total_estimated_hours:.1f}")
            print(f"Average Age: {metrics.average_age_days:.1f} days")
            print(f"Oldest Item: {metrics.oldest_item_days:.1f} days")
            print(f"Trend: {metrics.debt_trend}")
            print(f"Resolution Rate: {metrics.resolution_rate:.1f} items/week")
            
            print("\nBy Category:")
            for category, count in metrics.items_by_category.items():
                print(f"  {category}: {count}")
            
            print("\nBy Severity:")
            for severity, count in metrics.items_by_severity.items():
                print(f"  {severity}: {count}")
            
            print("\nBy Status:")
            for status, count in metrics.items_by_status.items():
                print(f"  {status}: {count}")


def handle_report_command(args):
    """Handle comprehensive report command"""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("Generating comprehensive code review report...")
    
    # Code review
    print("  Running code review...")
    reviewer = CodeReviewer(args.project_root)
    review_result = reviewer.review_project()
    
    # Refactoring suggestions
    print("  Generating refactoring suggestions...")
    engine = RefactoringEngine(args.project_root)
    suggestions = []
    for py_file in Path(args.project_root).glob("**/*.py"):
        suggestions.extend(engine.analyze_file(str(py_file)))
    engine.suggestions = suggestions
    
    # Technical debt
    print("  Analyzing technical debt...")
    tracker = TechnicalDebtTracker()
    debt_metrics = tracker.calculate_debt_metrics()
    debt_recommendations = tracker.generate_recommendations()
    
    # Generate reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.format == 'json':
        # JSON report
        report_file = output_dir / f"code_review_report_{timestamp}.json"
        reviewer.generate_report(str(report_file))
        
        refactor_file = output_dir / f"refactoring_suggestions_{timestamp}.json"
        engine.generate_suggestions_report(str(refactor_file))
        
        debt_file = output_dir / f"technical_debt_report_{timestamp}.json"
        tracker.export_debt_report(str(debt_file))
        
        print(f"\nReports generated:")
        print(f"  Code Review: {report_file}")
        print(f"  Refactoring: {refactor_file}")
        print(f"  Technical Debt: {debt_file}")
    
    else:
        # HTML report (simplified)
        report_file = output_dir / f"code_review_report_{timestamp}.html"
        generate_html_report(
            report_file, review_result, suggestions, debt_metrics, debt_recommendations
        )
        print(f"\nHTML report generated: {report_file}")


def generate_html_report(output_file: Path, review_result: Dict, suggestions: List, 
                        debt_metrics, debt_recommendations: List):
    """Generate HTML report"""
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Code Review Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e9e9e9; border-radius: 3px; }}
        .issue {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }}
        .critical {{ border-left-color: #d32f2f; }}
        .high {{ border-left-color: #f57c00; }}
        .medium {{ border-left-color: #fbc02d; }}
        .low {{ border-left-color: #388e3c; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Code Review Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>Summary</h2>
        <div class="metric">Total Issues: {review_result.get('issues', 0)}</div>
        <div class="metric">Refactoring Suggestions: {len(suggestions)}</div>
        <div class="metric">Technical Debt Items: {debt_metrics.total_items}</div>
        <div class="metric">Estimated Debt Hours: {debt_metrics.total_estimated_hours:.1f}</div>
    </div>
    
    <div class="section">
        <h2>Technical Debt Metrics</h2>
        <p>Average Age: {debt_metrics.average_age_days:.1f} days</p>
        <p>Trend: {debt_metrics.debt_trend}</p>
        <p>Resolution Rate: {debt_metrics.resolution_rate:.1f} items/week</p>
    </div>
    
    <div class="section">
        <h2>Recommendations</h2>
        <ul>
"""
    
    for rec in debt_recommendations[:5]:
        html_content += f"<li><strong>{rec.recommendation_type}:</strong> {rec.description}</li>\n"
    
    html_content += """
        </ul>
    </div>
</body>
</html>
"""
    
    with open(output_file, 'w') as f:
        f.write(html_content)


if __name__ == '__main__':
    main()