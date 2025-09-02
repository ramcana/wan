"""
Command-line interface for code quality checking system.
"""

import argparse
import sys
from pathlib import Path
import logging
import json
import yaml

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tools.code_quality.quality_checker import QualityChecker
from tools.code_quality.models import QualityConfig


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Code Quality Checking System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s check src/                    # Check all Python files in src/
  %(prog)s check --config quality.yaml  # Use custom configuration
  %(prog)s fix src/ --auto-fix-only      # Auto-fix issues where possible
  %(prog)s report --output report.html  # Generate HTML report
        """
    )
    
    parser.add_argument('--version', action='version', version='1.0.0')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--config', '-c', type=Path, help='Configuration file path')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Check command
    check_parser = subparsers.add_parser('check', help='Check code quality')
    check_parser.add_argument('path', type=Path, help='Path to check (file or directory)')
    check_parser.add_argument('--checks', nargs='+', 
                             choices=['formatting', 'style', 'documentation', 'type_hints', 'complexity'],
                             help='Specific checks to run (default: all)')
    check_parser.add_argument('--output', '-o', type=Path, help='Output file for results')
    check_parser.add_argument('--format', choices=['json', 'yaml', 'text'], default='text',
                             help='Output format')
    check_parser.add_argument('--fail-on-error', action='store_true',
                             help='Exit with non-zero code if errors found')
    
    # Fix command
    fix_parser = subparsers.add_parser('fix', help='Fix code quality issues')
    fix_parser.add_argument('path', type=Path, help='Path to fix (file or directory)')
    fix_parser.add_argument('--auto-fix-only', action='store_true',
                           help='Only fix issues marked as auto-fixable')
    fix_parser.add_argument('--dry-run', action='store_true',
                           help='Show what would be fixed without making changes')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate quality report')
    report_parser.add_argument('path', type=Path, help='Path to analyze')
    report_parser.add_argument('--output', '-o', type=Path, required=True,
                              help='Output file for report')
    report_parser.add_argument('--format', choices=['json', 'yaml', 'html'], default='html',
                              help='Report format')
    report_parser.add_argument('--template', type=Path, help='Custom report template')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_subparsers = config_parser.add_subparsers(dest='config_command')
    
    # Generate default config
    gen_config_parser = config_subparsers.add_parser('generate', help='Generate default configuration')
    gen_config_parser.add_argument('--output', '-o', type=Path, default=Path('quality-config.yaml'),
                                  help='Output configuration file')
    gen_config_parser.add_argument('--format', choices=['yaml', 'json'], default='yaml',
                                  help='Configuration format')
    
    # Validate config
    validate_config_parser = config_subparsers.add_parser('validate', help='Validate configuration')
    validate_config_parser.add_argument('config_file', type=Path, help='Configuration file to validate')
    
    # Enforcement command
    enforce_parser = subparsers.add_parser('enforce', help='Quality enforcement system')
    enforce_subparsers = enforce_parser.add_subparsers(dest='enforce_command', help='Enforcement commands')
    
    # Setup hooks
    hooks_parser = enforce_subparsers.add_parser('setup-hooks', help='Set up pre-commit hooks')
    hooks_parser.add_argument('--config', type=Path, help='Configuration file')
    
    # Setup CI
    ci_parser = enforce_subparsers.add_parser('setup-ci', help='Set up CI/CD integration')
    ci_parser.add_argument('--platform', choices=['github', 'gitlab', 'jenkins'], 
                          default='github', help='CI platform')
    
    # Enforcement status
    enforce_subparsers.add_parser('status', help='Show enforcement system status')
    
    # Create dashboard
    enforce_subparsers.add_parser('dashboard', help='Create quality metrics dashboard')
    
    # Run enforcement checks
    enforce_check_parser = enforce_subparsers.add_parser('check', help='Run enforcement checks')
    enforce_check_parser.add_argument('files', nargs='*', help='Files to check')
    
    # Generate enforcement report
    enforce_report_parser = enforce_subparsers.add_parser('report', help='Generate enforcement report')
    enforce_report_parser.add_argument('--format', choices=['console', 'html', 'json'], 
                                      default='console', help='Report format')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    setup_logging(args.verbose)
    
    try:
        if args.command == 'check':
            return handle_check_command(args)
        elif args.command == 'fix':
            return handle_fix_command(args)
        elif args.command == 'report':
            return handle_report_command(args)
        elif args.command == 'config':
            return handle_config_command(args)
        elif args.command == 'enforce':
            return handle_enforce_command(args)
        else:
            parser.print_help()
            return 1
    
    except Exception as e:
        logging.error(f"Command failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def handle_check_command(args) -> int:
    """Handle check command."""
    if not args.path.exists():
        print(f"Error: Path {args.path} does not exist")
        return 1
    
    # Load configuration
    if args.config:
        checker = QualityChecker.from_config_file(args.config)
    else:
        checker = QualityChecker()
    
    # Run quality check
    print(f"Checking code quality in {args.path}...")
    report = checker.check_quality(args.path, args.checks)
    
    # Generate output
    if args.output:
        output_content = checker.generate_report(report, args.format)
        with open(args.output, 'w') as f:
            f.write(output_content)
        print(f"Results written to {args.output}")
    else:
        # Print to stdout
        output_content = checker.generate_report(report, args.format)
        print(output_content)
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Files analyzed: {report.files_analyzed}")
    print(f"  Total issues: {report.total_issues}")
    print(f"  Errors: {report.errors}")
    print(f"  Warnings: {report.warnings}")
    print(f"  Quality score: {report.quality_score:.1f}/100")
    
    # Exit with error code if requested and errors found
    if args.fail_on_error and report.errors > 0:
        return 1
    
    return 0


def handle_fix_command(args) -> int:
    """Handle fix command."""
    if not args.path.exists():
        print(f"Error: Path {args.path} does not exist")
        return 1
    
    # Load configuration
    if hasattr(args, 'config') and args.config:
        checker = QualityChecker.from_config_file(args.config)
    else:
        checker = QualityChecker()
    
    if args.dry_run:
        print(f"Dry run: Checking what would be fixed in {args.path}...")
        report = checker.check_quality(args.path)
        fixable_issues = [issue for issue in report.issues if issue.auto_fixable]
        
        print(f"Found {len(fixable_issues)} auto-fixable issues:")
        for issue in fixable_issues[:10]:  # Show first 10
            print(f"  {issue.file_path}:{issue.line_number} - {issue.message}")
        
        if len(fixable_issues) > 10:
            print(f"  ... and {len(fixable_issues) - 10} more")
        
        return 0
    
    # Fix issues
    print(f"Fixing code quality issues in {args.path}...")
    report = checker.fix_issues(args.path, args.auto_fix_only)
    
    print(f"Fixed {report.auto_fixable_issues} issues")
    
    return 0


def handle_report_command(args) -> int:
    """Handle report command."""
    if not args.path.exists():
        print(f"Error: Path {args.path} does not exist")
        return 1
    
    # Load configuration
    if hasattr(args, 'config') and args.config:
        checker = QualityChecker.from_config_file(args.config)
    else:
        checker = QualityChecker()
    
    # Generate report
    print(f"Generating quality report for {args.path}...")
    report = checker.check_quality(args.path)
    
    if args.format == 'html':
        html_content = generate_html_report(report, args.template)
        with open(args.output, 'w') as f:
            f.write(html_content)
    else:
        output_content = checker.generate_report(report, args.format)
        with open(args.output, 'w') as f:
            f.write(output_content)
    
    print(f"Report generated: {args.output}")
    return 0


def handle_config_command(args) -> int:
    """Handle config command."""
    if args.config_command == 'generate':
        return generate_default_config(args)
    elif args.config_command == 'validate':
        return validate_config_file(args)
    else:
        print("Error: No config subcommand specified")
        return 1


def handle_enforce_command(args) -> int:
    """Handle enforcement command."""
    try:
        from tools.code_quality.enforcement.enforcement_cli import EnforcementCLI
        
        cli = EnforcementCLI()
        
        if args.enforce_command == 'setup-hooks':
            success = cli.setup_hooks(getattr(args, 'config', None))
            return 0 if success else 1
        
        elif args.enforce_command == 'setup-ci':
            success = cli.setup_ci(args.platform)
            return 0 if success else 1
        
        elif args.enforce_command == 'status':
            cli.status()
            return 0
        
        elif args.enforce_command == 'dashboard':
            success = cli.create_dashboard()
            return 0 if success else 1
        
        elif args.enforce_command == 'check':
            success = cli.run_checks(getattr(args, 'files', None))
            return 0 if success else 1
        
        elif args.enforce_command == 'report':
            cli.generate_report(args.format)
            return 0
        
        else:
            print("Error: No enforcement subcommand specified")
            return 1
    
    except ImportError as e:
        print(f"Error: Enforcement system not available: {e}")
        return 1
    except Exception as e:
        print(f"Error: Enforcement command failed: {e}")
        return 1


def generate_default_config(args) -> int:
    """Generate default configuration file."""
    config = QualityConfig()
    config_dict = config.to_dict()
    
    if args.format == 'json':
        with open(args.output, 'w') as f:
            json.dump(config_dict, f, indent=2)
    else:
        with open(args.output, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    print(f"Default configuration generated: {args.output}")
    return 0


def validate_config_file(args) -> int:
    """Validate configuration file."""
    if not args.config_file.exists():
        print(f"Error: Configuration file {args.config_file} does not exist")
        return 1
    
    try:
        checker = QualityChecker.from_config_file(args.config_file)
        print(f"Configuration file {args.config_file} is valid")
        return 0
    except Exception as e:
        print(f"Error: Invalid configuration file: {e}")
        return 1


def generate_html_report(report, template_path=None) -> str:
    """Generate HTML report."""
    if template_path and template_path.exists():
        # Use custom template
        with open(template_path, 'r') as f:
            template = f.read()
        # Simple template substitution (could use Jinja2 for more complex templates)
        return template.format(report=report)
    
    # Default HTML template
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Code Quality Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .metric {{ text-align: center; padding: 10px; background-color: #e9e9e9; border-radius: 5px; }}
        .issues {{ margin: 20px 0; }}
        .issue {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }}
        .error {{ border-left-color: #d32f2f; }}
        .warning {{ border-left-color: #f57c00; }}
        .info {{ border-left-color: #1976d2; }}
        .quality-score {{ font-size: 2em; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Code Quality Report</h1>
        <p>Generated: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Project: {report.project_path}</p>
    </div>
    
    <div class="summary">
        <div class="metric">
            <div class="quality-score">{report.quality_score:.1f}</div>
            <div>Quality Score</div>
        </div>
        <div class="metric">
            <div>{report.files_analyzed}</div>
            <div>Files Analyzed</div>
        </div>
        <div class="metric">
            <div>{report.total_issues}</div>
            <div>Total Issues</div>
        </div>
        <div class="metric">
            <div>{report.errors}</div>
            <div>Errors</div>
        </div>
        <div class="metric">
            <div>{report.warnings}</div>
            <div>Warnings</div>
        </div>
    </div>
    
    <h2>Metrics</h2>
    <ul>
        <li>Documentation Coverage: {report.metrics.documentation_coverage:.1f}%</li>
        <li>Type Hint Coverage: {report.metrics.type_hint_coverage:.1f}%</li>
        <li>Average Complexity: {report.metrics.complexity_score:.1f}</li>
        <li>Maintainability Index: {report.metrics.maintainability_index:.1f}</li>
    </ul>
    
    <h2>Issues</h2>
    <div class="issues">
"""
    
    for issue in report.issues[:50]:  # Show first 50 issues
        css_class = issue.severity.value
        html += f"""
        <div class="issue {css_class}">
            <strong>{issue.file_path}:{issue.line_number}</strong> - {issue.message}
            <br><small>{issue.rule_code} | {issue.issue_type.value}</small>
            {f'<br><em>Suggestion: {issue.suggestion}</em>' if issue.suggestion else ''}
        </div>
        """
    
    if len(report.issues) > 50:
        html += f"<p>... and {len(report.issues) - 50} more issues</p>"
    
    html += """
    </div>
</body>
</html>
    """
    
    return html


if __name__ == '__main__':
    sys.exit(main())