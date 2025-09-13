#!/usr/bin/env python3
"""
Fast Feedback Development Tools CLI

Interactive command-line interface for fast feedback development tools.
"""

import os
import sys
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional
import logging
import click

from test_watcher import TestWatcher, WatchConfig, create_default_config
from config_watcher import ConfigWatcher, create_example_services
from debug_tools import DebugTools

@click.group()
@click.option('--verbose', is_flag=True, help='Enable verbose output')
@click.option('--project-root', type=click.Path(exists=True), help='Project root directory')
@click.pass_context
def cli(ctx, verbose, project_root):
    """Fast Feedback Development Tools CLI"""
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['project_root'] = Path(project_root) if project_root else Path.cwd()
    
    # Setup logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

@cli.command()
@click.option('--category', multiple=True, help='Test categories to watch (unit, integration, e2e, performance)')
@click.option('--pattern', multiple=True, help='Test file patterns to watch')
@click.option('--fast', is_flag=True, help='Fast mode (skip slow tests)')
@click.option('--debounce', type=float, default=1.0, help='Debounce delay in seconds')
@click.option('--max-parallel', type=int, default=3, help='Maximum parallel test executions')
@click.pass_context
def watch_tests(ctx, category, pattern, fast, debounce, max_parallel):
    """Watch files and run tests automatically"""
    verbose = ctx.obj['verbose']
    project_root = ctx.obj['project_root']
    
    click.echo("🧪 Starting test watcher...")
    
    # Create configuration
    config = create_default_config(project_root)
    
    if category:
        config.test_categories = list(category)
    
    if pattern:
        config.test_patterns = list(pattern)
    
    config.fast_mode = fast
    config.debounce_delay = debounce
    config.max_parallel_tests = max_parallel
    
    # Display configuration
    click.echo(f"Project root: {project_root}")
    click.echo(f"Watch directories: {[str(d) for d in config.watch_dirs]}")
    click.echo(f"Test patterns: {config.test_patterns}")
    click.echo(f"Categories: {config.test_categories or 'all'}")
    click.echo(f"Fast mode: {config.fast_mode}")
    click.echo(f"Debounce delay: {config.debounce_delay}s")
    
    # Create and start watcher
    watcher = TestWatcher(config, project_root)
    
    try:
        # Run initial test suite if requested
        if click.confirm("Run all tests once before starting watch mode?", default=False):
            watcher.run_all_tests()
        
        click.echo("\n👀 Watching for file changes... (Press Ctrl+C to stop)")
        watcher.start_watching()
        
    except KeyboardInterrupt:
        click.echo("\n🛑 Test watcher stopped.")

@cli.command()
@click.option('--files', multiple=True, help='Specific config files to watch')
@click.option('--reload-services', is_flag=True, help='Enable service reloading')
@click.option('--debounce', type=float, default=0.5, help='Debounce delay in seconds')
@click.pass_context
def watch_config(ctx, files, reload_services, debounce):
    """Watch configuration files for changes"""
    verbose = ctx.obj['verbose']
    project_root = ctx.obj['project_root']
    
    click.echo("⚙️ Starting configuration watcher...")
    
    # Create watcher
    watcher = ConfigWatcher(project_root)
    watcher.debounce_delay = debounce
    
    # Register change handler
    def change_handler(change):
        click.echo(f"📝 Config changed: {change.file_path.name} ({change.change_type})")
        
        if change.change_type == 'modified' and change.new_content:
            click.echo("  ✅ Configuration reloaded successfully")
        elif change.change_type == 'deleted':
            click.echo("  ⚠️ Configuration file deleted")
    
    watcher.register_change_handler(change_handler)
    
    # Register services if requested
    if reload_services:
        services = create_example_services(project_root)
        for service in services:
            watcher.register_service(service)
        click.echo(f"Registered {len(services)} services for auto-reload")
    
    # Override watch files if specified
    if files:
        specific_files = [Path(f) for f in files]
        watcher.watch_dirs = list(set(f.parent for f in specific_files))
        watcher.config_patterns = [f.name for f in specific_files]
        watcher._load_initial_configs()
        click.echo(f"Watching specific files: {list(files)}")
    
    # Display current configuration
    config_count = len(watcher.get_all_configs())
    click.echo(f"Watching {config_count} configuration files")
    
    try:
        click.echo("\n👀 Watching for configuration changes... (Press Ctrl+C to stop)")
        watcher.start_watching()
        
    except KeyboardInterrupt:
        click.echo("\n🛑 Configuration watcher stopped.")

@cli.command()
@click.option('--enable', is_flag=True, help='Enable debug logging')
@click.option('--level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']), 
              default='DEBUG', help='Debug log level')
@click.option('--session', type=str, help='Debug session ID')
@click.pass_context
def debug(ctx, enable, level, session):
    """Start interactive debug session"""
    verbose = ctx.obj['verbose']
    project_root = ctx.obj['project_root']
    
    click.echo("🐛 Starting debug session...")
    
    # Create debug tools
    debug_tools = DebugTools(project_root)
    
    if enable:
        debug_tools.enable_debug_logging(level)
        click.echo(f"✅ Debug logging enabled at level {level}")
    
    # Start debug session
    session_id = debug_tools.start_debug_session(session)
    click.echo(f"Debug session started: {session_id}")
    
    try:
        # Interactive debug loop
        while True:
            click.echo("\n🐛 DEBUG MENU")
            click.echo("1. Analyze logs")
            click.echo("2. Show recent errors")
            click.echo("3. Search logs")
            click.echo("4. Performance report")
            click.echo("5. Export debug report")
            click.echo("6. Clear logs")
            click.echo("0. Exit")
            
            choice = click.prompt("Select option", type=int, default=0)
            
            if choice == 0:
                break
            elif choice == 1:
                analysis = debug_tools.analyze_logs()
                click.echo(f"\n📊 LOG ANALYSIS")
                click.echo(f"Total entries: {analysis['total_entries']}")
                click.echo(f"Errors: {analysis['error_count']}")
                click.echo(f"Warnings: {analysis['warning_count']}")
                
                if analysis['patterns_found']:
                    click.echo(f"\nPatterns found: {len(analysis['patterns_found'])}")
                    for pattern in analysis['patterns_found'][:3]:
                        click.echo(f"  - {pattern['pattern']}")
                
                if analysis['recommendations']:
                    click.echo(f"\nRecommendations:")
                    for rec in analysis['recommendations']:
                        click.echo(f"  - {rec}")
            
            elif choice == 2:
                count = click.prompt("Number of recent errors", type=int, default=5)
                errors = debug_tools.get_recent_errors(count)
                
                if errors:
                    click.echo(f"\n❌ RECENT ERRORS ({len(errors)}):")
                    for error in errors:
                        click.echo(f"  [{error.timestamp}] {error.message}")
                else:
                    click.echo("✅ No recent errors found")
            
            elif choice == 3:
                query = click.prompt("Search query")
                results = debug_tools.search_logs(query)
                
                click.echo(f"\n🔍 SEARCH RESULTS ({len(results)}):")
                for result in results[-5:]:
                    click.echo(f"  [{result.timestamp}] {result.level}: {result.message}")
            
            elif choice == 4:
                report = debug_tools.get_performance_report()
                click.echo(f"\n⚡ PERFORMANCE REPORT")
                click.echo(f"Functions profiled: {report['total_functions_profiled']}")
                
                if report['slow_functions']:
                    click.echo(f"\nSlow functions:")
                    for func in report['slow_functions'][:3]:
                        click.echo(f"  - {func['function']}: {func['avg_time']:.2f}s avg")
            
            elif choice == 5:
                filename = click.prompt("Report filename", default="debug_report.json")
                output_file = project_root / filename
                debug_tools.export_debug_report(output_file, session_id)
                click.echo(f"✅ Report exported to {output_file}")
            
            elif choice == 6:
                debug_tools.clear_logs()
                click.echo("✅ Logs cleared")
    
    except KeyboardInterrupt:
        pass
    
    finally:
        # End debug session
        session = debug_tools.end_debug_session()
        if session:
            duration = session.end_time - session.start_time
            click.echo(f"\n🛑 Debug session ended: {session.session_id}")
            click.echo(f"Duration: {duration}")
            click.echo(f"Errors: {session.error_count}, Warnings: {session.warning_count}")

@cli.command()
@click.pass_context
def dashboard(ctx):
    """Start development dashboard"""
    verbose = ctx.obj['verbose']
    project_root = ctx.obj['project_root']
    
    click.echo("📊 Starting development dashboard...")
    
    # This would start a web-based dashboard
    # For now, we'll show a text-based status
    
    # Initialize tools
    debug_tools = DebugTools(project_root)
    config_watcher = ConfigWatcher(project_root)
    
    try:
        while True:
            # Clear screen (simple version)
            click.clear()
            
            click.echo("🚀 WAN22 DEVELOPMENT DASHBOARD")
            click.echo("=" * 50)
            click.echo(f"Project: {project_root.name}")
            click.echo(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Configuration status
            configs = config_watcher.get_all_configs()
            click.echo(f"\n⚙️ Configuration: {len(configs)} files loaded")
            
            # Recent logs
            recent_errors = debug_tools.get_recent_errors(3)
            if recent_errors:
                click.echo(f"\n❌ Recent Errors ({len(recent_errors)}):")
                for error in recent_errors:
                    click.echo(f"  - {error.message[:60]}...")
            else:
                click.echo(f"\n✅ No recent errors")
            
            # Performance
            perf_report = debug_tools.get_performance_report()
            if perf_report['slow_functions']:
                click.echo(f"\n⚡ Performance: {len(perf_report['slow_functions'])} slow functions")
            else:
                click.echo(f"\n⚡ Performance: All functions running efficiently")
            
            click.echo(f"\n📝 Commands:")
            click.echo("  - Press 'r' to refresh")
            click.echo("  - Press 'q' to quit")
            
            # Wait for input or auto-refresh
            import select
            import sys
            
            if sys.stdin in select.select([sys.stdin], [], [], 5)[0]:
                line = input()
                if line.lower() == 'q':
                    break
            
    except KeyboardInterrupt:
        pass
    
    click.echo("\n🛑 Dashboard stopped.")

@cli.command()
@click.pass_context
def status(ctx):
    """Show development environment status"""
    verbose = ctx.obj['verbose']
    project_root = ctx.obj['project_root']
    
    click.echo("📊 DEVELOPMENT ENVIRONMENT STATUS")
    click.echo("=" * 50)
    
    # Project info
    click.echo(f"Project: {project_root.name}")
    click.echo(f"Root: {project_root}")
    
    # Check if tools are available
    tools_status = {}
    
    # Test watcher
    try:
        config = create_default_config(project_root)
        test_dirs = [d for d in config.watch_dirs if d.exists()]
        tools_status['test_watcher'] = f"✅ Ready ({len(test_dirs)} directories)"
    except Exception as e:
        tools_status['test_watcher'] = f"❌ Error: {e}"
    
    # Config watcher
    try:
        watcher = ConfigWatcher(project_root)
        config_count = len(watcher.get_all_configs())
        tools_status['config_watcher'] = f"✅ Ready ({config_count} configs)"
    except Exception as e:
        tools_status['config_watcher'] = f"❌ Error: {e}"
    
    # Debug tools
    try:
        debug_tools = DebugTools(project_root)
        tools_status['debug_tools'] = f"✅ Ready (log: {debug_tools.log_file})"
    except Exception as e:
        tools_status['debug_tools'] = f"❌ Error: {e}"
    
    # Display status
    click.echo(f"\n🛠️ TOOLS STATUS:")
    for tool, status in tools_status.items():
        click.echo(f"  {tool}: {status}")
    
    # Quick health check
    click.echo(f"\n🏥 QUICK HEALTH CHECK:")
    
    # Check for common directories
    important_dirs = ['backend', 'frontend', 'tests', 'config', 'docs']
    for dir_name in important_dirs:
        dir_path = project_root / dir_name
        status = "✅" if dir_path.exists() else "❌"
        click.echo(f"  {dir_name}/: {status}")
    
    # Check for configuration files
    config_files = [
        'backend/requirements.txt',
        'frontend/package.json',
        'config/unified-config.yaml'
    ]
    for config_file in config_files:
        file_path = project_root / config_file
        status = "✅" if file_path.exists() else "❌"
        click.echo(f"  {config_file}: {status}")

if __name__ == "__main__":
    cli()
