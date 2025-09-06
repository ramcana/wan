#!/usr/bin/env python3
"""
Development Environment CLI

Interactive command-line interface for development environment management.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging
import click

from setup_dev_environment import DevEnvironmentSetup
from dependency_detector import DependencyDetector
from environment_validator import EnvironmentValidator

@click.group()
@click.option('--verbose', is_flag=True, help='Enable verbose output')
@click.option('--project-root', type=click.Path(exists=True), help='Project root directory')
@click.pass_context
def cli(ctx, verbose, project_root):
    """Development Environment Management CLI"""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['project_root'] = Path(project_root) if project_root else Path.cwd()
    
    # Setup logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

@cli.command()
@click.option('--python', is_flag=True, help='Setup Python environment only')
@click.option('--nodejs', is_flag=True, help='Setup Node.js environment only')
@click.option('--tools', is_flag=True, help='Setup development tools only')
@click.option('--structure', is_flag=True, help='Create project structure only')
@click.option('--config', is_flag=True, help='Setup configuration files only')
@click.option('--report', type=click.Path(), help='Generate setup report to file')
@click.pass_context
def setup(ctx, python, nodejs, tools, structure, config, report):
    """Setup development environment"""
    verbose = ctx.obj['verbose']
    project_root = ctx.obj['project_root']
    
    setup_manager = DevEnvironmentSetup(project_root, verbose)
    
    if not any([python, nodejs, tools, structure, config]):
        # Interactive full setup
        click.echo("🚀 Starting interactive development environment setup...")
        click.echo(f"Project root: {project_root}")
        
        if click.confirm("Run full automated setup?", default=True):
            success = setup_manager.run_full_setup()
            if success:
                click.echo("✅ Setup completed successfully!")
            else:
                click.echo("❌ Setup failed. Check the logs above.")
                sys.exit(1)
        else:
            click.echo("Setup cancelled.")
    else:
        # Run specific setup steps
        success = True
        
        if structure:
            click.echo("Creating project structure...")
            success &= setup_manager.create_project_structure()
        
        if python:
            click.echo("Setting up Python environment...")
            success &= setup_manager.setup_python_environment()
        
        if nodejs:
            click.echo("Setting up Node.js environment...")
            success &= setup_manager.setup_nodejs_environment()
        
        if tools:
            click.echo("Setting up development tools...")
            success &= setup_manager.setup_development_tools()
        
        if config:
            click.echo("Setting up configuration files...")
            success &= setup_manager.setup_configuration_files()
        
        if not success:
            click.echo("❌ Some setup steps failed.")
            sys.exit(1)
    
    if report:
        setup_manager.generate_setup_report(Path(report))
        click.echo(f"Setup report saved to {report}")

@cli.command()
@click.option('--python', is_flag=True, help='Validate Python environment only')
@click.option('--nodejs', is_flag=True, help='Validate Node.js environment only')
@click.option('--structure', is_flag=True, help='Validate project structure only')
@click.option('--tools', is_flag=True, help='Validate development tools only')
@click.option('--ports', is_flag=True, help='Validate ports and services only')
@click.option('--gpu', is_flag=True, help='Validate GPU environment only')
@click.option('--export', type=click.Path(), help='Export report to JSON file')
@click.pass_context
def validate(ctx, python, nodejs, structure, tools, ports, gpu, export):
    """Validate development environment"""
    verbose = ctx.obj['verbose']
    project_root = ctx.obj['project_root']
    
    validator = EnvironmentValidator(project_root)
    
    if not any([python, nodejs, structure, tools, ports, gpu]):
        # Run full validation
        health = validator.run_full_validation()
        
        click.echo(f"\n🏥 ENVIRONMENT HEALTH REPORT")
        click.echo("=" * 50)
        
        # Color-coded status
        status_colors = {
            'healthy': 'green',
            'warning': 'yellow',
            'critical': 'red'
        }
        
        click.echo(f"Overall Status: ", nl=False)
        click.secho(health.overall_status.upper(), 
                   fg=status_colors.get(health.overall_status, 'white'))
        
        click.echo(f"Health Score: {health.score:.1f}/100")
        click.echo(f"Checks: {health.passed_checks} passed, {health.warning_checks} warnings, {health.failed_checks} failed")
        
        # Show failed checks
        failed_results = [r for r in health.results if r.status == 'fail']
        if failed_results:
            click.echo(f"\n❌ FAILED CHECKS ({len(failed_results)}):")
            for result in failed_results:
                click.echo(f"  - {result.name}: {result.message}")
                if result.fix_suggestion:
                    click.echo(f"    Fix: {result.fix_suggestion}")
        
        # Show warnings
        warning_results = [r for r in health.results if r.status == 'warning']
        if warning_results:
            click.echo(f"\n⚠️  WARNING CHECKS ({len(warning_results)}):")
            for result in warning_results:
                click.echo(f"  - {result.name}: {result.message}")
                if result.fix_suggestion:
                    click.echo(f"    Suggestion: {result.fix_suggestion}")
        
        # Show passed checks in verbose mode
        if verbose:
            passed_results = [r for r in health.results if r.status == 'pass']
            if passed_results:
                click.echo(f"\n✅ PASSED CHECKS ({len(passed_results)}):")
                for result in passed_results:
                    click.echo(f"  - {result.name}: {result.message}")
        
        if export:
            validator.export_health_report(health, Path(export))
            click.echo(f"\nReport exported to {export}")
    
    else:
        # Run specific validations
        results = []
        
        if python:
            results.extend(validator.validate_python_environment())
        if nodejs:
            results.extend(validator.validate_nodejs_environment())
        if structure:
            results.extend(validator.validate_project_structure())
        if tools:
            results.extend(validator.validate_development_tools())
        if ports:
            results.extend(validator.validate_ports_and_services())
        if gpu:
            results.extend(validator.validate_gpu_environment())
        
        # Display results
        for result in results:
            status_icons = {"pass": "✅", "warning": "⚠️", "fail": "❌"}
            click.echo(f"{status_icons[result.status]} {result.name}: {result.message}")
            if result.fix_suggestion:
                click.echo(f"   Fix: {result.fix_suggestion}")

@cli.command()
@click.option('--missing', is_flag=True, help='Show only missing dependencies')
@click.option('--guide', is_flag=True, help='Generate installation guide')
@click.option('--export', type=click.Path(), help='Export report to JSON file')
@click.pass_context
def deps(ctx, missing, guide, export):
    """Check dependencies"""
    verbose = ctx.obj['verbose']
    project_root = ctx.obj['project_root']
    
    detector = DependencyDetector(project_root)
    
    if guide:
        # Generate installation guide
        guide_text = detector.generate_installation_guide()
        click.echo(guide_text)
    elif missing:
        # Show only missing dependencies
        missing_deps = detector.get_missing_dependencies()
        
        if not missing_deps:
            click.echo("✅ All dependencies are installed!")
        else:
            click.echo("❌ Missing dependencies:")
            for category, deps in missing_deps.items():
                click.echo(f"\n{category.upper()}:")
                for dep in deps:
                    click.echo(f"  - {dep.name}")
                    if dep.installation_command:
                        click.echo(f"    Install: {dep.installation_command}")
    else:
        # Show all dependencies
        all_deps = detector.get_all_dependencies()
        
        for category, deps in all_deps.items():
            click.echo(f"\n{category.upper()} DEPENDENCIES:")
            click.echo("-" * 40)
            
            for dep in deps:
                status = "✅" if dep.is_installed else "❌"
                version_info = f" (v{dep.installed_version})" if dep.installed_version else ""
                click.echo(f"{status} {dep.name}{version_info}")
                
                if not dep.is_installed and dep.installation_command:
                    click.echo(f"   Install: {dep.installation_command}")
    
    if export:
        detector.export_dependency_report(Path(export))
        click.echo(f"Report exported to {export}")

@cli.command()
@click.pass_context
def health(ctx):
    """Quick health check"""
    verbose = ctx.obj['verbose']
    project_root = ctx.obj['project_root']
    
    click.echo("🔍 Running quick health check...")
    
    # Check dependencies
    detector = DependencyDetector(project_root)
    missing_deps = detector.get_missing_dependencies()
    
    # Check environment
    validator = EnvironmentValidator(project_root)
    health = validator.run_full_validation()
    
    # Summary
    click.echo(f"\n📊 HEALTH SUMMARY")
    click.echo("-" * 30)
    
    # Dependencies
    total_missing = sum(len(deps) for deps in missing_deps.values())
    if total_missing == 0:
        click.secho("✅ Dependencies: All installed", fg='green')
    else:
        click.secho(f"❌ Dependencies: {total_missing} missing", fg='red')
    
    # Environment health
    if health.overall_status == 'healthy':
        click.secho(f"✅ Environment: Healthy ({health.score:.0f}/100)", fg='green')
    elif health.overall_status == 'warning':
        click.secho(f"⚠️  Environment: Warnings ({health.score:.0f}/100)", fg='yellow')
    else:
        click.secho(f"❌ Environment: Critical issues ({health.score:.0f}/100)", fg='red')
    
    # Quick recommendations
    if total_missing > 0 or health.failed_checks > 0:
        click.echo(f"\n💡 QUICK FIXES:")
        
        if total_missing > 0:
            click.echo("  - Run 'python tools/dev-environment/dev_environment_cli.py setup' to install dependencies")
        
        if health.failed_checks > 0:
            click.echo("  - Run 'python tools/dev-environment/dev_environment_cli.py validate' for detailed issues")

@cli.command()
@click.pass_context
def doctor(ctx):
    """Comprehensive environment diagnosis"""
    verbose = ctx.obj['verbose']
    project_root = ctx.obj['project_root']
    
    click.echo("🩺 Running comprehensive environment diagnosis...")
    
    # Initialize components
    detector = DependencyDetector(project_root)
    validator = EnvironmentValidator(project_root)
    
    # Get comprehensive data
    all_deps = detector.get_all_dependencies()
    missing_deps = detector.get_missing_dependencies()
    health = validator.run_full_validation()
    
    # Detailed report
    click.echo(f"\n📋 DIAGNOSIS REPORT")
    click.echo("=" * 50)
    
    # System info
    system_info = detector.system_info
    click.echo(f"Platform: {system_info.platform}")
    click.echo(f"Python: {system_info.python_version}")
    click.echo(f"Node.js: {system_info.node_version or 'Not installed'}")
    click.echo(f"Git: {system_info.git_version or 'Not installed'}")
    click.echo(f"CUDA: {'Available' if system_info.cuda_available else 'Not available'}")
    
    # Dependencies summary
    click.echo(f"\n📦 DEPENDENCIES SUMMARY")
    click.echo("-" * 30)
    
    for category, deps in all_deps.items():
        installed = len([d for d in deps if d.is_installed])
        total = len(deps)
        click.echo(f"{category.title()}: {installed}/{total} installed")
    
    # Environment health details
    click.echo(f"\n🏥 ENVIRONMENT HEALTH")
    click.echo("-" * 30)
    click.echo(f"Overall Score: {health.score:.1f}/100")
    click.echo(f"Status: {health.overall_status}")
    click.echo(f"Passed: {health.passed_checks}")
    click.echo(f"Warnings: {health.warning_checks}")
    click.echo(f"Failed: {health.failed_checks}")
    
    # Critical issues
    critical_issues = [r for r in health.results if r.status == 'fail']
    if critical_issues:
        click.echo(f"\n🚨 CRITICAL ISSUES")
        click.echo("-" * 30)
        for issue in critical_issues:
            click.echo(f"❌ {issue.name}: {issue.message}")
            if issue.fix_suggestion:
                click.echo(f"   Fix: {issue.fix_suggestion}")
    
    # Recommendations
    click.echo(f"\n💡 RECOMMENDATIONS")
    click.echo("-" * 30)
    
    if missing_deps:
        click.echo("1. Install missing dependencies:")
        click.echo("   python tools/dev-environment/dev_environment_cli.py setup")
    
    if health.failed_checks > 0:
        click.echo("2. Fix critical environment issues:")
        click.echo("   python tools/dev-environment/dev_environment_cli.py validate")
    
    if health.warning_checks > 0:
        click.echo("3. Address warnings for optimal setup:")
        click.echo("   python tools/dev-environment/dev_environment_cli.py validate --verbose")
    
    if health.overall_status == 'healthy':
        click.echo("✅ Environment is healthy! You're ready to develop.")

if __name__ == "__main__":
    cli()