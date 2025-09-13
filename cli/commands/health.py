"""System health monitoring commands"""

import typer
from pathlib import Path
from typing import Optional
import sys

app = typer.Typer()

@app.command()
def check(
    quick: bool = typer.Option(False, "--quick", help="Run quick health check only"),
    detailed: bool = typer.Option(False, "--detailed", help="Run detailed health analysis"),
    fix: bool = typer.Option(False, "--fix", help="Auto-fix health issues where possible")
):
    """🏥 Check overall system health"""
    
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from tools.health_checker.health_checker import HealthChecker
    
    checker = HealthChecker()
    
    if quick:
        typer.echo("⚡ Running quick health check...")
        results = checker.run_quick_check()
    elif detailed:
        typer.echo("🔍 Running detailed health analysis...")
        results = checker.run_detailed_check()
    else:
        typer.echo("🏥 Running standard health check...")
        results = checker.run_standard_check()
    
    typer.echo(f"📊 Overall Health Score: {results.overall_health:.1%}")
    
    if results.critical_issues:
        typer.echo(f"🚨 Critical Issues: {len(results.critical_issues)}")
    
    if results.warnings:
        typer.echo(f"⚠️ Warnings: {len(results.warnings)}")
    
    if fix and (results.critical_issues or results.warnings):
        typer.echo("🔧 Attempting to fix health issues...")
        checker.fix_health_issues(results)

@app.command()
def monitor(
    duration: int = typer.Option(60, "--duration", help="Monitoring duration in seconds"),
    interval: int = typer.Option(5, "--interval", help="Check interval in seconds"),
    alert_threshold: float = typer.Option(0.7, "--threshold", help="Alert threshold (0-1)")
):
    """📊 Monitor system health in real-time"""
    
    from tools.health_checker.health_analytics import HealthMonitor
    
    monitor = HealthMonitor()
    
    typer.echo(f"📊 Starting health monitoring for {duration}s...")
    monitor.start_monitoring(
        duration=duration,
        interval=interval,
        alert_threshold=alert_threshold
    )

@app.command()
def dashboard(
    port: int = typer.Option(8080, "--port", "-p", help="Dashboard port"),
    host: str = typer.Option("localhost", "--host", help="Dashboard host")
):
    """📈 Launch health monitoring dashboard"""
    
    from tools.health_checker.dashboard_server import HealthDashboard
    
    dashboard = HealthDashboard()
    
    typer.echo(f"📈 Starting health dashboard at http://{host}:{port}")
    dashboard.start_server(host=host, port=port)

@app.command()
def report(
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    format: str = typer.Option("html", "--format", help="Report format (html, pdf, json)")
):
    """📋 Generate health report"""
    
    from tools.health_checker.health_reporter import HealthReporter
    
    reporter = HealthReporter()
    
    typer.echo("📋 Generating health report...")
    report_path = reporter.generate_report(output_path=output, format=format)
    
    typer.echo(f"✅ Health report generated: {report_path}")

@app.command()
def baseline(
    save: bool = typer.Option(True, "--save/--no-save", help="Save baseline metrics"),
    compare: bool = typer.Option(False, "--compare", help="Compare with existing baseline")
):
    """📏 Establish or compare performance baseline"""
    
    from tools.health_checker.baseline_and_improvement import BaselineManager
    
    manager = BaselineManager()
    
    if compare:
        typer.echo("📊 Comparing current metrics with baseline...")
        comparison = manager.compare_with_baseline()
        
        typer.echo(f"Performance Change: {comparison.performance_delta:+.1%}")
        typer.echo(f"Health Change: {comparison.health_delta:+.1%}")
        
        if comparison.regressions:
            typer.echo(f"⚠️ Regressions detected: {len(comparison.regressions)}")
    else:
        typer.echo("📏 Establishing performance baseline...")
        baseline = manager.establish_baseline()
        
        if save:
            manager.save_baseline(baseline)
            typer.echo("✅ Baseline saved for future comparisons")

@app.command()
def alerts(
    setup: bool = typer.Option(False, "--setup", help="Setup alert system"),
    test: bool = typer.Option(False, "--test", help="Test alert system"),
    list_alerts: bool = typer.Option(False, "--list", help="List active alerts")
):
    """🚨 Manage health alerts"""
    
    from tools.health_checker.automated_alerting import AlertManager
    
    manager = AlertManager()
    
    if setup:
        typer.echo("🚨 Setting up alert system...")
        manager.setup_alerts()
        typer.echo("✅ Alert system configured")
    elif test:
        typer.echo("🧪 Testing alert system...")
        manager.test_alerts()
        typer.echo("✅ Alert test completed")
    elif list_alerts:
        alerts = manager.get_active_alerts()
        
        if alerts:
            typer.echo(f"🚨 Active Alerts ({len(alerts)}):")
            for alert in alerts:
                typer.echo(f"  {alert.severity}: {alert.message}")
        else:
            typer.echo("✅ No active alerts")
    else:
        typer.echo("Use --setup, --test, or --list to manage alerts")

@app.command()
def optimize(
    target: str = typer.Option("performance", "--target", help="Optimization target (performance, memory, disk)"),
    aggressive: bool = typer.Option(False, "--aggressive", help="Use aggressive optimization")
):
    """⚡ Optimize system performance"""
    
    from tools.health_checker.performance_optimizer import PerformanceOptimizer
    
    optimizer = PerformanceOptimizer()
    
    typer.echo(f"⚡ Optimizing {target}...")
    results = optimizer.optimize(target=target, aggressive=aggressive)
    
    typer.echo(f"✅ Optimization complete:")
    typer.echo(f"Performance Improvement: {results.performance_gain:.1%}")
    typer.echo(f"Memory Saved: {results.memory_saved_mb:.1f} MB")
    typer.echo(f"Disk Space Freed: {results.disk_freed_mb:.1f} MB")
