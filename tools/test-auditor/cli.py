#!/usr/bin/env python3
"""
Test Auditor CLI

Command-line interface for the comprehensive test suite auditor.
Provides various commands for analyzing and reporting on test suite health.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from test_auditor import TestAuditor, TestSuiteAuditReport


class TestAuditorCLI:
    """Command-line interface for test auditor"""
    
    def __init__(self):
        self.project_root = Path.cwd()
    
    def run(self, args: Optional[list] = None) -> int:
        """Run the CLI with given arguments"""
        parser = self._create_parser()
        parsed_args = parser.parse_args(args)
        
        try:
            return parsed_args.func(parsed_args)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser"""
        parser = argparse.ArgumentParser(
            description="Comprehensive test suite auditor",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s audit                    # Run full audit
  %(prog)s audit --output report.json  # Save to specific file
  %(prog)s summary                  # Show quick summary
  %(prog)s issues --severity critical  # Show only critical issues
  %(prog)s performance              # Show performance analysis
            """
        )
        
        parser.add_argument(
            '--project-root',
            type=Path,
            default=Path.cwd(),
            help='Project root directory (default: current directory)'
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Audit command
        audit_parser = subparsers.add_parser('audit', help='Run comprehensive test audit')
        audit_parser.add_argument(
            '--output', '-o',
            type=Path,
            default='test_audit_report.json',
            help='Output file for audit report'
        )
        audit_parser.add_argument(
            '--format',
            choices=['json', 'text', 'html'],
            default='json',
            help='Output format'
        )
        audit_parser.set_defaults(func=self._audit_command)
        
        # Summary command
        summary_parser = subparsers.add_parser('summary', help='Show audit summary')
        summary_parser.add_argument(
            '--report',
            type=Path,
            help='Use existing report file (default: run new audit)'
        )
        summary_parser.set_defaults(func=self._summary_command)
        
        # Issues command
        issues_parser = subparsers.add_parser('issues', help='Show test issues')
        issues_parser.add_argument(
            '--severity',
            choices=['critical', 'high', 'medium', 'low'],
            help='Filter by severity level'
        )
        issues_parser.add_argument(
            '--type',
            help='Filter by issue type'
        )
        issues_parser.add_argument(
            '--report',
            type=Path,
            help='Use existing report file'
        )
        issues_parser.set_defaults(func=self._issues_command)
        
        # Performance command
        perf_parser = subparsers.add_parser('performance', help='Show performance analysis')
        perf_parser.add_argument(
            '--threshold',
            type=float,
            default=5.0,
            help='Slow test threshold in seconds'
        )
        perf_parser.add_argument(
            '--report',
            type=Path,
            help='Use existing report file'
        )
        perf_parser.set_defaults(func=self._performance_command)
        
        # Files command
        files_parser = subparsers.add_parser('files', help='Show file analysis')
        files_parser.add_argument(
            '--broken-only',
            action='store_true',
            help='Show only broken files'
        )
        files_parser.add_argument(
            '--report',
            type=Path,
            help='Use existing report file'
        )
        files_parser.set_defaults(func=self._files_command)
        
        return parser
    
    def _audit_command(self, args) -> int:
        """Run comprehensive audit"""
        self.project_root = args.project_root
        
        print(f"Running test audit on {self.project_root}")
        auditor = TestAuditor(self.project_root)
        report = auditor.audit_test_suite()
        
        # Save report
        if args.format == 'json':
            self._save_json_report(report, args.output)
        elif args.format == 'text':
            self._save_text_report(report, args.output)
        elif args.format == 'html':
            self._save_html_report(report, args.output)
        
        print(f"Audit complete! Report saved to {args.output}")
        self._print_summary(report)
        
        return 0
    
    def _summary_command(self, args) -> int:
        """Show audit summary"""
        report = self._load_or_run_audit(args.report)
        self._print_summary(report)
        return 0
    
    def _issues_command(self, args) -> int:
        """Show test issues"""
        report = self._load_or_run_audit(args.report)
        
        # Collect all issues
        all_issues = []
        for file_analysis in report.file_analyses:
            all_issues.extend(file_analysis.issues)
        
        # Filter issues
        filtered_issues = all_issues
        if args.severity:
            filtered_issues = [i for i in filtered_issues if i.severity == args.severity]
        if args.type:
            filtered_issues = [i for i in filtered_issues if i.issue_type == args.type]
        
        # Display issues
        if not filtered_issues:
            print("No issues found matching criteria.")
            return 0
        
        print(f"Found {len(filtered_issues)} issues:")
        print()
        
        for issue in filtered_issues:
            print(f"[{issue.severity.upper()}] {issue.issue_type}")
            print(f"  File: {issue.test_file}")
            if issue.test_name:
                print(f"  Test: {issue.test_name}")
            if issue.line_number:
                print(f"  Line: {issue.line_number}")
            print(f"  Description: {issue.description}")
            if issue.suggestion:
                print(f"  Suggestion: {issue.suggestion}")
            print()
        
        return 0
    
    def _performance_command(self, args) -> int:
        """Show performance analysis"""
        report = self._load_or_run_audit(args.report)
        
        print("Test Performance Analysis")
        print("=" * 50)
        print()
        
        # Overall stats
        total_time = report.execution_summary.get('total_execution_time', 0)
        avg_time = report.execution_summary.get('average_execution_time', 0)
        
        print(f"Total execution time: {total_time:.2f}s")
        print(f"Average file time: {avg_time:.2f}s")
        print()
        
        # Slow files
        slow_files = [fa for fa in report.file_analyses if fa.execution_time > args.threshold]
        if slow_files:
            print(f"Files slower than {args.threshold}s:")
            for fa in sorted(slow_files, key=lambda x: x.execution_time, reverse=True):
                print(f"  {fa.execution_time:.2f}s - {fa.file_path}")
        else:
            print(f"No files slower than {args.threshold}s")
        
        print()
        
        # Timeout issues
        timeout_issues = [
            issue for fa in report.file_analyses 
            for issue in fa.issues 
            if issue.issue_type == 'timeout'
        ]
        
        if timeout_issues:
            print("Files with timeout issues:")
            for issue in timeout_issues:
                print(f"  {issue.test_file}")
        
        return 0
    
    def _files_command(self, args) -> int:
        """Show file analysis"""
        report = self._load_or_run_audit(args.report)
        
        files_to_show = report.file_analyses
        if args.broken_only:
            files_to_show = [
                fa for fa in files_to_show 
                if fa.has_syntax_errors or fa.has_import_errors or fa.failing_tests > 0
            ]
        
        print(f"Test File Analysis ({len(files_to_show)} files)")
        print("=" * 60)
        print()
        
        for fa in files_to_show:
            status_indicators = []
            if fa.has_syntax_errors:
                status_indicators.append("SYNTAX_ERROR")
            if fa.has_import_errors:
                status_indicators.append("IMPORT_ERROR")
            if fa.failing_tests > 0:
                status_indicators.append("FAILING_TESTS")
            
            status = " | ".join(status_indicators) if status_indicators else "OK"
            
            print(f"File: {fa.file_path}")
            print(f"  Status: {status}")
            print(f"  Tests: {fa.total_tests} total, {fa.passing_tests} passing, {fa.failing_tests} failing")
            print(f"  Time: {fa.execution_time:.2f}s")
            print(f"  Issues: {len(fa.issues)}")
            
            if fa.missing_imports:
                print(f"  Missing imports: {', '.join(fa.missing_imports)}")
            if fa.missing_fixtures:
                print(f"  Missing fixtures: {', '.join(fa.missing_fixtures)}")
            
            print()
        
        return 0
    
    def _load_or_run_audit(self, report_file: Optional[Path]) -> TestSuiteAuditReport:
        """Load existing report or run new audit"""
        if report_file and report_file.exists():
            with open(report_file, 'r') as f:
                data = json.load(f)
            return TestSuiteAuditReport(**data)
        else:
            print("Running new audit...")
            auditor = TestAuditor(self.project_root)
            return auditor.audit_test_suite()
    
    def _print_summary(self, report: TestSuiteAuditReport):
        """Print audit summary"""
        print()
        print("Test Suite Audit Summary")
        print("=" * 40)
        print(f"Total files: {report.total_files}")
        print(f"Total tests: {report.total_tests}")
        print(f"Passing tests: {report.passing_tests}")
        print(f"Failing tests: {report.failing_tests}")
        print(f"Skipped tests: {report.skipped_tests}")
        print(f"Broken files: {len(report.broken_files)}")
        print(f"Critical issues: {len(report.critical_issues)}")
        
        if report.execution_summary:
            total_time = report.execution_summary.get('total_execution_time', 0)
            print(f"Total execution time: {total_time:.2f}s")
        
        if report.recommendations:
            print()
            print("Top Recommendations:")
            for i, rec in enumerate(report.recommendations[:5], 1):
                print(f"{i}. {rec}")
    
    def _save_json_report(self, report: TestSuiteAuditReport, output_file: Path):
        """Save report as JSON"""
        from dataclasses import asdict
        
        with open(output_file, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
    
    def _save_text_report(self, report: TestSuiteAuditReport, output_file: Path):
        """Save report as text"""
        with open(output_file, 'w') as f:
            f.write("Test Suite Audit Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total files: {report.total_files}\n")
            f.write(f"Total tests: {report.total_tests}\n")
            f.write(f"Passing tests: {report.passing_tests}\n")
            f.write(f"Failing tests: {report.failing_tests}\n")
            f.write(f"Skipped tests: {report.skipped_tests}\n\n")
            
            if report.broken_files:
                f.write("Broken Files:\n")
                for file in report.broken_files:
                    f.write(f"  - {file}\n")
                f.write("\n")
            
            if report.critical_issues:
                f.write("Critical Issues:\n")
                for issue in report.critical_issues:
                    f.write(f"  - {issue.test_file}: {issue.description}\n")
                f.write("\n")
            
            if report.recommendations:
                f.write("Recommendations:\n")
                for i, rec in enumerate(report.recommendations, 1):
                    f.write(f"{i}. {rec}\n")
    
    def _save_html_report(self, report: TestSuiteAuditReport, output_file: Path):
        """Save report as HTML"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Test Suite Audit Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .summary {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
        .issue {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }}
        .critical {{ border-left-color: #d32f2f; }}
        .high {{ border-left-color: #f57c00; }}
        .medium {{ border-left-color: #fbc02d; }}
        .low {{ border-left-color: #388e3c; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Test Suite Audit Report</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Total files:</strong> {report.total_files}</p>
        <p><strong>Total tests:</strong> {report.total_tests}</p>
        <p><strong>Passing tests:</strong> {report.passing_tests}</p>
        <p><strong>Failing tests:</strong> {report.failing_tests}</p>
        <p><strong>Skipped tests:</strong> {report.skipped_tests}</p>
        <p><strong>Critical issues:</strong> {len(report.critical_issues)}</p>
    </div>
    
    <h2>Issues by Severity</h2>
"""
        
        # Group issues by severity
        issues_by_severity = {}
        for fa in report.file_analyses:
            for issue in fa.issues:
                if issue.severity not in issues_by_severity:
                    issues_by_severity[issue.severity] = []
                issues_by_severity[issue.severity].append(issue)
        
        for severity in ['critical', 'high', 'medium', 'low']:
            if severity in issues_by_severity:
                html_content += f"<h3>{severity.title()} Issues</h3>\n"
                for issue in issues_by_severity[severity]:
                    html_content += f'<div class="issue {severity}">\n'
                    html_content += f"<strong>{issue.test_file}</strong><br>\n"
                    html_content += f"Type: {issue.issue_type}<br>\n"
                    html_content += f"Description: {issue.description}<br>\n"
                    if issue.suggestion:
                        html_content += f"Suggestion: {issue.suggestion}<br>\n"
                    html_content += "</div>\n"
        
        html_content += """
</body>
</html>
"""
        
        with open(output_file, 'w') as f:
            f.write(html_content)


def main():
    """Main entry point"""
    cli = TestAuditorCLI()
    return cli.run()


if __name__ == '__main__':
    sys.exit(main())