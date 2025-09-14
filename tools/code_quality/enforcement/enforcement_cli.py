"""
CLI for automated quality enforcement system.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

from .pre_commit_hooks import PreCommitHookManager
from .ci_integration import CIIntegration

logger = logging.getLogger(__name__)


class EnforcementCLI:
    """Command-line interface for quality enforcement."""

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize enforcement CLI."""
        self.project_root = project_root or Path.cwd()
        self.hook_manager = PreCommitHookManager(self.project_root)
        self.ci_integration = CIIntegration(self.project_root)

    def setup_hooks(self, config_file: Optional[Path] = None) -> bool:
        """Set up pre-commit hooks."""
        config = None
        if config_file and config_file.exists():
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

        success = self.hook_manager.install_hooks(config)
        if success:
            print("✅ Pre-commit hooks installed successfully")
        else:
            print("❌ Failed to install pre-commit hooks")

        return success

    def setup_ci(self, platform: str = 'github') -> bool:
        """Set up CI/CD integration."""
        success = False

        if platform.lower() == 'github':
            success = self.ci_integration.setup_github_actions()
            if success:
                print("✅ GitHub Actions workflow created successfully")
        elif platform.lower() == 'gitlab':
            success = self.ci_integration.setup_gitlab_ci()
            if success:
                print("✅ GitLab CI pipeline created successfully")
        elif platform.lower() == 'jenkins':
            success = self.ci_integration.setup_jenkins()
            if success:
                print("✅ Jenkins pipeline created successfully")
        else:
            print(f"❌ Unsupported CI platform: {platform}")
            return False

        if not success:
            print(f"❌ Failed to setup {platform} integration")

        return success

    def run_checks(self, files: Optional[List[str]] = None) -> bool:
        """Run quality checks."""
        file_paths = None
        if files:
            file_paths = [Path(f) for f in files]

        # Run pre-commit hooks
        hook_results = self.hook_manager.run_hooks(file_paths)

        # Run CI-style checks
        ci_results = self.ci_integration.run_quality_checks(file_paths)

        # Display results
        print("\n" + "="*50)
        print("QUALITY CHECK RESULTS")
        print("="*50)

        if hook_results['success'] and ci_results['success']:
            print("✅ All quality checks passed!")
            return True
        else:
            print("❌ Quality checks failed!")

            if hook_results['failures']:
                print("\nPre-commit hook failures:")
                for failure in hook_results['failures']:
                    print(f"  - {failure}")

            if ci_results['errors']:
                print("\nCI check errors:")
                for error in ci_results['errors']:
                    print(f"  - {error}")

            return False

    def status(self) -> None:
        """Show enforcement system status."""
        print("QUALITY ENFORCEMENT STATUS")
        print("="*30)

        # Hook status
        hook_status = self.hook_manager.get_hook_status()
        print(f"Pre-commit hooks: {{'✅ Installed' if hook_status['installed'] else '❌ Not installed'}}")
        print(f"Pre-commit available: {{'✅ Yes' if hook_status['pre_commit_available'] else '❌ No'}}")
        print(f"Config exists: {{'✅ Yes' if hook_status['config_exists'] else '❌ No'}}")
        print(f"Config valid: {{'✅ Yes' if hook_status['config_valid'] else '❌ No'}}")

        # CI status
        ci_status = self.ci_integration.get_ci_status()
        print(f"GitHub Actions: {{'✅ Configured' if ci_status['github_actions'] else '❌ Not configured'}}")
        print(f"GitLab CI: {{'✅ Configured' if ci_status['gitlab_ci'] else '❌ Not configured'}}")
        print(f"Jenkins: {{'✅ Configured' if ci_status['jenkins'] else '❌ Not configured'}}")
        print(f"Quality config: {{'✅ Exists' if ci_status['quality_config'] else '❌ Missing'}}")
        print(f"Metrics tracking: {{'✅ Active' if ci_status['metrics_tracking'] else '❌ Inactive'}}")

    def create_dashboard(self) -> bool:
        """Create quality metrics dashboard."""
        config = self.ci_integration.create_quality_metrics_dashboard()
        if config:
            print("✅ Quality metrics dashboard configuration created")
            print(f"Configuration saved to: {self.ci_integration.quality_config}")
            return True
        else:
            print("❌ Failed to create dashboard configuration")
            return False

    def generate_report(self, format_type: str = 'console') -> None:
        """Generate quality report."""
        results = self.ci_integration.run_quality_checks()

        if format_type == 'console':
            print(self.ci_integration.generate_quality_report(results))
        elif format_type == 'html':
            report_file = self.project_root / "quality-report.html"
            html_report = self._generate_html_report(results)
            with open(report_file, 'w') as f:
                f.write(html_report)
            print(f"✅ HTML report generated: {report_file}")
        elif format_type == 'json':
            import json
            report_file = self.project_root / "quality-report.json"
            with open(report_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✅ JSON report generated: {report_file}")

    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML quality report."""
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Code Quality Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .success {{ color: green; }}
        .failure {{ color: red; }}
        .warning {{ color: orange; }}
        .metrics {{ display: flex; gap: 20px; margin: 20px 0; }}
        .metric-card {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; flex: 1; }}
        .details {{ margin-top: 20px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Code Quality Report</h1>
        <p class="{status_class}">Status: {status}</p>
        <p>Generated: {timestamp}</p>
    </div>

    <div class="metrics">
        <div class="metric-card">
            <h3>Overall Score</h3>
            <p style="font-size: 24px; font-weight: bold;">{overall_score}/10</p>
        </div>
        <div class="metric-card">
            <h3>Errors</h3>
            <p style="font-size: 24px; font-weight: bold; color: red;">{total_errors}</p>
        </div>
        <div class="metric-card">
            <h3>Warnings</h3>
            <p style="font-size: 24px; font-weight: bold; color: orange;">{total_warnings}</p>
        </div>
        <div class="metric-card">
            <h3>Files Checked</h3>
            <p style="font-size: 24px; font-weight: bold;">{files_checked}</p>
        </div>
    </div>

    <div class="details">
        <h2>Detailed Results</h2>
        <table>
            <tr>
                <th>File</th>
                <th>Score</th>
                <th>Errors</th>
                <th>Warnings</th>
            </tr>
            {file_rows}
        </table>
    </div>
</body>
</html>
        """

        from datetime import datetime

        status = "PASSED" if results['success'] else "FAILED"
        status_class = "success" if results['success'] else "failure"
        metrics = results.get('metrics', {})

        file_rows = ""
        for file_path, check_result in results.get('checks', {}).items():
            file_rows += f"""
            <tr>
                <td>{file_path}</td>
                <td>{check_result.get('score', 'N/A')}</td>
                <td>{check_result.get('errors', 0)}</td>
                <td>{check_result.get('warnings', 0)}</td>
            </tr>
            """

        return html_template.format(
            status=status,
            status_class=status_class,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            overall_score=metrics.get('overall_score', 'N/A'),
            total_errors=metrics.get('total_errors', 0),
            total_warnings=metrics.get('total_warnings', 0),
            files_checked=metrics.get('files_checked', 0),
            file_rows=file_rows
        )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Quality enforcement system")
    parser.add_argument('--project-root', type=Path, help="Project root directory")

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Setup hooks command
    hooks_parser = subparsers.add_parser('setup-hooks', help='Set up pre-commit hooks')
    hooks_parser.add_argument('--config', type=Path, help='Configuration file')

    # Setup CI command
    ci_parser = subparsers.add_parser('setup-ci', help='Set up CI/CD integration')
    ci_parser.add_argument('--platform', choices=['github', 'gitlab', 'jenkins'],
                          default='github', help='CI platform')

    # Run checks command
    check_parser = subparsers.add_parser('check', help='Run quality checks')
    check_parser.add_argument('files', nargs='*', help='Files to check')

    # Status command
    subparsers.add_parser('status', help='Show enforcement system status')

    # Dashboard command
    subparsers.add_parser('dashboard', help='Create quality metrics dashboard')

    # Report command
    report_parser = subparsers.add_parser('report', help='Generate quality report')
    report_parser.add_argument('--format', choices=['console', 'html', 'json'],
                              default='console', help='Report format')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Initialize CLI
    cli = EnforcementCLI(args.project_root)

    # Execute command
    try:
        if args.command == 'setup-hooks':
            success = cli.setup_hooks(args.config)
            return 0 if success else 1

        elif args.command == 'setup-ci':
            success = cli.setup_ci(args.platform)
            return 0 if success else 1

        elif args.command == 'check':
            success = cli.run_checks(args.files)
            return 0 if success else 1

        elif args.command == 'status':
            cli.status()
            return 0

        elif args.command == 'dashboard':
            success = cli.create_dashboard()
            return 0 if success else 1

        elif args.command == 'report':
            cli.generate_report(args.format)
            return 0

    except Exception as e:
        logger.error(f"Command failed: {e}")
        print(f"❌ Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
