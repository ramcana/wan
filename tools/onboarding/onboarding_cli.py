#!/usr/bin/env python3
"""
Onboarding CLI

Interactive command-line interface for comprehensive developer onboarding.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import logging
import click

from setup_wizard import SetupWizard
from developer_checklist import DeveloperChecklist

@click.group()
@click.option('--verbose', is_flag=True, help='Enable verbose output')
@click.option('--project-root', type=click.Path(exists=True), help='Project root directory')
@click.option('--developer', type=str, help='Developer name for progress tracking')
@click.pass_context
def cli(ctx, verbose, project_root, developer):
    """Comprehensive Developer Onboarding CLI"""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['project_root'] = Path(project_root) if project_root else Path.cwd()
    ctx.obj['developer'] = developer
    
    # Setup logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

@cli.command()
@click.pass_context
def start(ctx):
    """Start interactive onboarding process"""
    verbose = ctx.obj['verbose']
    project_root = ctx.obj['project_root']
    developer = ctx.obj['developer']
    
    click.echo("🚀 Starting WAN22 Developer Onboarding...")
    
    # Create and run setup wizard
    wizard = SetupWizard(project_root, developer)
    
    try:
        success = wizard.run_wizard()
        if success:
            click.echo("\n✅ Onboarding completed successfully!")
            click.echo("Welcome to the WAN22 team! 🎉")
        else:
            click.echo("\n❌ Onboarding incomplete.")
            click.echo("You can resume anytime by running this command again.")
    except KeyboardInterrupt:
        click.echo("\n\n⏸️  Onboarding paused.")
        click.echo("You can resume anytime by running 'python tools/onboarding/onboarding_cli.py start'")

@cli.command()
@click.pass_context
def progress(ctx):
    """Check onboarding progress"""
    verbose = ctx.obj['verbose']
    project_root = ctx.obj['project_root']
    developer = ctx.obj['developer']
    
    checklist = DeveloperChecklist(project_root, developer)
    status = checklist.get_status_summary()
    
    click.echo(f"\n📋 ONBOARDING PROGRESS")
    click.echo("=" * 50)
    
    if developer:
        click.echo(f"Developer: {developer}")
    
    # Overall progress
    click.echo(f"Overall Completion: {status['overall']['completion_percentage']:.1f}%")
    click.echo(f"Items Completed: {status['overall']['completed_items']}/{status['overall']['total_items']}")
    click.echo(f"Critical Items: {status['overall']['critical_completed']}/{status['overall']['critical_total']}")
    click.echo(f"Time Remaining: {status['overall']['estimated_time_remaining']}")
    
    # Progress by category
    click.echo(f"\n📊 PROGRESS BY CATEGORY:")
    for category, stats in status['categories'].items():
        completion = (stats['completed'] / stats['total']) * 100 if stats['total'] > 0 else 0
        click.echo(f"  {category}: {completion:.0f}% ({stats['completed']}/{stats['total']})")
    
    # Next steps
    if status['next_items']:
        click.echo(f"\n🎯 NEXT STEPS:")
        for item in status['next_items']:
            priority_icon = {'critical': '🔴', 'important': '🟡', 'optional': '🟢'}
            icon = priority_icon.get(item['priority'], '⚪')
            click.echo(f"  {icon} {item['title']} ({item['estimated_time']})")
    
    # Recommendations based on progress
    if status['overall']['completion_percentage'] < 50:
        click.echo(f"\n💡 RECOMMENDATION:")
        click.echo("  Run the setup wizard to complete critical items:")
        click.echo("  python tools/onboarding/onboarding_cli.py start")
    elif status['overall']['critical_completed'] < status['overall']['critical_total']:
        click.echo(f"\n💡 RECOMMENDATION:")
        click.echo("  Focus on completing critical items first.")
    else:
        click.echo(f"\n🎉 Great progress! You're ready to start developing!")

@cli.command()
@click.option('--item', type=str, help='Specific item ID to complete')
@click.option('--notes', type=str, help='Notes for the completed item')
@click.pass_context
def complete(ctx, item, notes):
    """Mark checklist item as completed"""
    verbose = ctx.obj['verbose']
    project_root = ctx.obj['project_root']
    developer = ctx.obj['developer']
    
    checklist = DeveloperChecklist(project_root, developer)
    
    if not item:
        # Show available items to complete
        next_items = checklist.get_next_items()
        
        if not next_items:
            click.echo("🎉 All items completed!")
            return
        
        click.echo("📋 Available items to complete:")
        for i, item_obj in enumerate(next_items[:10], 1):
            priority_icon = {'critical': '🔴', 'important': '🟡', 'optional': '🟢'}
            icon = priority_icon.get(item_obj.priority, '⚪')
            click.echo(f"  {i}. {icon} {item_obj.id}: {item_obj.title}")
        
        try:
            choice = click.prompt("Select item number (or 'q' to quit)", type=str)
            if choice.lower() == 'q':
                return
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(next_items):
                item = next_items[choice_num - 1].id
            else:
                click.echo("Invalid selection.")
                return
        except (ValueError, IndexError):
            click.echo("Invalid selection.")
            return
    
    # Complete the item
    if checklist.complete_item(item, notes):
        item_obj = checklist.progress.items[item]
        click.echo(f"✅ Completed: {item_obj.title}")
        
        # Show updated progress
        status = checklist.get_status_summary()
        click.echo(f"Progress: {status['overall']['completion_percentage']:.1f}% complete")
    else:
        click.echo(f"❌ Failed to complete item: {item}")

@cli.command()
@click.pass_context
def validate(ctx):
    """Validate development environment setup"""
    verbose = ctx.obj['verbose']
    project_root = ctx.obj['project_root']
    
    click.echo("🔍 Validating development environment...")
    
    # Import here to avoid circular imports
    sys.path.append(str(project_root / "tools"))
    from dev_environment.environment_validator import EnvironmentValidator
    
    validator = EnvironmentValidator(project_root)
    health = validator.run_full_validation()
    
    # Display results
    click.echo(f"\n🏥 ENVIRONMENT HEALTH REPORT")
    click.echo("=" * 50)
    
    # Color-coded status
    if health.overall_status == 'healthy':
        click.secho(f"Overall Status: {health.overall_status.upper()}", fg='green')
    elif health.overall_status == 'warning':
        click.secho(f"Overall Status: {health.overall_status.upper()}", fg='yellow')
    else:
        click.secho(f"Overall Status: {health.overall_status.upper()}", fg='red')
    
    click.echo(f"Health Score: {health.score:.1f}/100")
    click.echo(f"Checks: {health.passed_checks} passed, {health.warning_checks} warnings, {health.failed_checks} failed")
    
    # Show critical issues
    failed_results = [r for r in health.results if r.status == 'fail']
    if failed_results:
        click.echo(f"\n❌ CRITICAL ISSUES ({len(failed_results)}):")
        for result in failed_results:
            click.echo(f"  - {result.name}: {result.message}")
            if result.fix_suggestion:
                click.echo(f"    Fix: {result.fix_suggestion}")
    
    # Show warnings
    warning_results = [r for r in health.results if r.status == 'warning']
    if warning_results:
        click.echo(f"\n⚠️  WARNINGS ({len(warning_results)}):")
        for result in warning_results:
            click.echo(f"  - {result.name}: {result.message}")
            if result.fix_suggestion:
                click.echo(f"    Suggestion: {result.fix_suggestion}")
    
    # Recommendations
    if health.failed_checks > 0:
        click.echo(f"\n💡 RECOMMENDATION:")
        click.echo("  Run the setup wizard to fix critical issues:")
        click.echo("  python tools/onboarding/onboarding_cli.py start")
    elif health.warning_checks > 0:
        click.echo(f"\n💡 RECOMMENDATION:")
        click.echo("  Address warnings for optimal development experience.")
    else:
        click.echo(f"\n✅ Environment is healthy and ready for development!")

@cli.command()
@click.option('--format', type=click.Choice(['text', 'markdown', 'json']), default='text', help='Report format')
@click.option('--output', type=click.Path(), help='Output file path')
@click.pass_context
def report(ctx, format, output):
    """Generate comprehensive onboarding report"""
    verbose = ctx.obj['verbose']
    project_root = ctx.obj['project_root']
    developer = ctx.obj['developer']
    
    checklist = DeveloperChecklist(project_root, developer)
    
    if format == 'markdown':
        report_content = checklist.generate_report()
    elif format == 'json':
        import json
        status = checklist.get_status_summary()
        report_content = json.dumps(status, indent=2)
    else:  # text
        status = checklist.get_status_summary()
        
        lines = [
            f"WAN22 Developer Onboarding Report",
            f"=" * 40,
            f"",
            f"Developer: {checklist.progress.developer_name}",
            f"Started: {checklist.progress.started_at}",
            f"Last Updated: {checklist.progress.last_updated}",
            f"",
            f"Overall Progress: {status['overall']['completion_percentage']:.1f}%",
            f"Items Completed: {status['overall']['completed_items']}/{status['overall']['total_items']}",
            f"Critical Items: {status['overall']['critical_completed']}/{status['overall']['critical_total']}",
            f"Time Remaining: {status['overall']['estimated_time_remaining']}",
            f"",
            f"Progress by Category:",
        ]
        
        for category, stats in status['categories'].items():
            completion = (stats['completed'] / stats['total']) * 100 if stats['total'] > 0 else 0
            lines.append(f"  {category}: {completion:.1f}% ({stats['completed']}/{stats['total']})")
        
        if status['next_items']:
            lines.extend([
                f"",
                f"Next Steps:",
            ])
            for item in status['next_items']:
                lines.append(f"  - {item['title']} ({item['estimated_time']})")
        
        report_content = "\n".join(lines)
    
    if output:
        output_path = Path(output)
        with open(output_path, 'w') as f:
            f.write(report_content)
        click.echo(f"📄 Report saved to: {output_path}")
    else:
        click.echo(report_content)

@cli.command()
@click.pass_context
def status(ctx):
    """Quick status check"""
    verbose = ctx.obj['verbose']
    project_root = ctx.obj['project_root']
    developer = ctx.obj['developer']
    
    click.echo("🔍 Quick onboarding status check...")
    
    # Check if onboarding has been started
    checklist = DeveloperChecklist(project_root, developer)
    status = checklist.get_status_summary()
    
    click.echo(f"\n📊 QUICK STATUS")
    click.echo("-" * 30)
    
    # Overall status
    if status['overall']['completion_percentage'] == 0:
        click.secho("Status: Not started", fg='red')
        click.echo("Run 'python tools/onboarding/onboarding_cli.py start' to begin")
    elif status['overall']['completion_percentage'] < 50:
        click.secho("Status: In progress (early stage)", fg='yellow')
        click.echo("Continue with the setup wizard")
    elif status['overall']['completion_percentage'] < 80:
        click.secho("Status: In progress (advanced)", fg='yellow')
        click.echo("Most critical items completed")
    else:
        click.secho("Status: Nearly complete", fg='green')
        click.echo("Ready for development!")
    
    click.echo(f"Progress: {status['overall']['completion_percentage']:.0f}%")
    click.echo(f"Critical: {status['overall']['critical_completed']}/{status['overall']['critical_total']}")
    
    # Next action
    if status['next_items']:
        next_item = status['next_items'][0]
        priority_icon = {'critical': '🔴', 'important': '🟡', 'optional': '🟢'}
        icon = priority_icon.get(next_item['priority'], '⚪')
        click.echo(f"Next: {icon} {next_item['title']}")

@cli.command()
@click.pass_context
def reset(ctx):
    """Reset onboarding progress"""
    verbose = ctx.obj['verbose']
    project_root = ctx.obj['project_root']
    developer = ctx.obj['developer']
    
    if not developer:
        developer = click.prompt("Enter developer name to reset")
    
    progress_file = project_root / ".kiro" / "onboarding" / f"{developer}_progress.json"
    
    if progress_file.exists():
        if click.confirm(f"Reset onboarding progress for {developer}?"):
            progress_file.unlink()
            click.echo(f"✅ Onboarding progress reset for {developer}")
        else:
            click.echo("Reset cancelled")
    else:
        click.echo(f"No onboarding progress found for {developer}")

@cli.command()
@click.pass_context
def help_docs(ctx):
    """Show available documentation"""
    project_root = ctx.obj['project_root']
    
    docs_dir = project_root / "tools" / "onboarding" / "docs"
    
    click.echo("📚 Available Onboarding Documentation:")
    click.echo("=" * 50)
    
    if docs_dir.exists():
        docs = [
            ("getting-started.md", "Quick start guide for new developers"),
            ("project-overview.md", "Project architecture and structure overview"),
            ("development-setup.md", "Detailed development environment setup"),
            ("coding-standards.md", "Coding standards and best practices"),
            ("troubleshooting.md", "Common issues and solutions")
        ]
        
        for doc_file, description in docs:
            doc_path = docs_dir / doc_file
            if doc_path.exists():
                click.echo(f"✅ {doc_file}")
                click.echo(f"   {description}")
                click.echo(f"   Path: {doc_path}")
            else:
                click.echo(f"❌ {doc_file} (missing)")
            click.echo()
    else:
        click.echo("❌ Documentation directory not found")
        click.echo(f"Expected location: {docs_dir}")
    
    click.echo("💡 To read documentation:")
    click.echo("  • Use your preferred text editor or markdown viewer")
    click.echo("  • VS Code: code tools/onboarding/docs/getting-started.md")
    click.echo("  • Command line: cat tools/onboarding/docs/getting-started.md")

if __name__ == "__main__":
    cli()
