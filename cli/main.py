#!/usr/bin/env python3
"""
WAN Project Quality & Maintenance Toolkit
Main CLI entry point that unifies all project tools.
"""

import typer
from pathlib import Path
from typing import Optional
import sys
import os

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cli.commands import test, clean, config, docs, quality, health, deploy, wan

app = typer.Typer(
    name="wan-cli",
    help="WAN Project Quality & Maintenance Toolkit - Your unified development companion",
    add_completion=False,
    rich_markup_mode="rich"
)

# Add command groups
app.add_typer(wan.app, name="wan", help="WAN Model Management - T2V, I2V, TI2V operations")
app.add_typer(test.app, name="test", help="Testing and validation commands")
app.add_typer(clean.app, name="clean", help="Cleanup and maintenance commands") 
app.add_typer(config.app, name="config", help="Configuration management commands")
app.add_typer(docs.app, name="docs", help="Documentation generation and validation")
app.add_typer(quality.app, name="quality", help="Code quality and analysis commands")
app.add_typer(health.app, name="health", help="System health monitoring commands")
app.add_typer(deploy.app, name="deploy", help="Deployment and production commands")

@app.command()
def status():
    """Show overall project status and health"""
    typer.echo("Checking project status...")
    
    # Quick health checks
    from tools.health_checker.health_checker import HealthChecker
    checker = HealthChecker()
    results = checker.run_quick_check()
    
    if results.overall_health > 0.8:
        typer.echo("Project is in good health!")
    elif results.overall_health > 0.6:
        typer.echo("Project has some issues that need attention")
    else:
        typer.echo("Project needs immediate attention")
    
    typer.echo(f"Overall Health Score: {results.overall_health:.1%}")

@app.command()
def init():
    """Initialize project for first-time setup"""
    typer.echo("Initializing WAN project...")
    
    # Run essential setup tasks
    from tools.onboarding.setup_wizard import SetupWizard
    wizard = SetupWizard()
    wizard.run_interactive_setup()

@app.command()
def quick():
    """Run quick validation suite (fast feedback loop)"""
    typer.echo("Running quick validation...")
    
    # Fast checks only
    from cli.workflows.quick_validation import run_quick_validation
    success = run_quick_validation()
    
    if success:
        typer.echo("Quick validation passed!")
    else:
        typer.echo("Quick validation failed - run 'wan-cli test --help' for detailed testing")
        raise typer.Exit(1)

@app.callback()
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    config_file: Optional[Path] = typer.Option(None, "--config", help="Custom config file path")
):
    """
    WAN Project Quality & Maintenance Toolkit
    
    A unified CLI that brings together all your project tools:
    • Testing and validation
    • Code quality and cleanup  
    • Configuration management
    • Documentation generation
    • Health monitoring
    • Deployment automation
    
    Get started: wan-cli init
    Quick check: wan-cli quick
    Full status: wan-cli status
    """
    # Set global context
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['config_file'] = config_file
    
    if verbose:
        typer.echo("Verbose mode enabled")

if __name__ == "__main__":
    app()