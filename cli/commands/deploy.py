"""Deployment and production commands"""

import typer
from pathlib import Path
from typing import Optional
import sys

app = typer.Typer()

@app.command()
def validate(
    environment: str = typer.Option("production", "--env", help="Target environment"),
    check_dependencies: bool = typer.Option(True, "--deps/--no-deps", help="Check dependencies"),
    check_config: bool = typer.Option(True, "--config/--no-config", help="Validate configuration")
):
    """✅ Validate deployment readiness"""
    
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from backend.scripts.deployment.deployment_validator import DeploymentValidator
    
    validator = DeploymentValidator()
    
    typer.echo(f"✅ Validating deployment for {environment}...")
    results = validator.validate_deployment(
        environment=environment,
        check_dependencies=check_dependencies,
        check_config=check_config
    )
    
    if results.is_ready:
        typer.echo("✅ Deployment validation passed!")
    else:
        typer.echo(f"❌ Deployment validation failed: {len(results.issues)} issues found")
        validator.show_issues(results.issues)
        raise typer.Exit(1)

@app.command()
def backup(
    target: str = typer.Option("config", "--target", help="Backup target (config, data, full)"),
    destination: Optional[Path] = typer.Option(None, "--dest", help="Backup destination")
):
    """💾 Create deployment backup"""
    
    from backend.scripts.deployment.config_backup_restore import BackupManager
    
    manager = BackupManager()
    
    typer.echo(f"💾 Creating {target} backup...")
    backup_path = manager.create_backup(target=target, destination=destination)
    
    typer.echo(f"✅ Backup created: {backup_path}")

@app.command()
def deploy(
    environment: str = typer.Argument(..., help="Target environment"),
    version: Optional[str] = typer.Option(None, "--version", help="Specific version to deploy"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Simulate deployment without executing"),
    skip_validation: bool = typer.Option(False, "--skip-validation", help="Skip pre-deployment validation")
):
    """🚀 Deploy to target environment"""
    
    from backend.scripts.deployment.deploy import DeploymentManager
    
    manager = DeploymentManager()
    
    if not skip_validation:
        typer.echo("✅ Running pre-deployment validation...")
        # Run validation first
        ctx = typer.Context(validate)
        ctx.invoke(validate, environment=environment)
    
    if dry_run:
        typer.echo(f"🧪 Simulating deployment to {environment}...")
        results = manager.simulate_deployment(environment, version)
        typer.echo("📋 Deployment simulation complete")
        manager.show_deployment_plan(results)
    else:
        typer.echo(f"🚀 Deploying to {environment}...")
        results = manager.deploy(environment, version)
        
        if results.success:
            typer.echo("✅ Deployment completed successfully!")
        else:
            typer.echo("❌ Deployment failed")
            raise typer.Exit(1)

@app.command()
def rollback(
    environment: str = typer.Argument(..., help="Target environment"),
    version: Optional[str] = typer.Option(None, "--version", help="Version to rollback to"),
    confirm: bool = typer.Option(False, "--confirm", help="Skip confirmation prompt")
):
    """⏪ Rollback deployment"""
    
    from backend.scripts.deployment.rollback_manager import RollbackManager
    
    manager = RollbackManager()
    
    if not confirm:
        confirmed = typer.confirm(f"Are you sure you want to rollback {environment}?")
        if not confirmed:
            typer.echo("Rollback cancelled")
            return
    
    typer.echo(f"⏪ Rolling back {environment}...")
    results = manager.rollback(environment, version)
    
    if results.success:
        typer.echo("✅ Rollback completed successfully!")
    else:
        typer.echo("❌ Rollback failed")
        raise typer.Exit(1)

@app.command()
def status(
    environment: str = typer.Argument(..., help="Target environment"),
    detailed: bool = typer.Option(False, "--detailed", help="Show detailed status")
):
    """📊 Check deployment status"""
    
    from backend.api.deployment_health import DeploymentHealthChecker
    
    checker = DeploymentHealthChecker()
    status = checker.get_deployment_status(environment, detailed=detailed)
    
    typer.echo(f"📊 Deployment Status for {environment}:")
    typer.echo(f"Status: {status.status}")
    typer.echo(f"Version: {status.version}")
    typer.echo(f"Health: {status.health_score:.1%}")
    
    if detailed:
        checker.show_detailed_status(status)

@app.command()
def migrate(
    from_env: str = typer.Argument(..., help="Source environment"),
    to_env: str = typer.Argument(..., help="Target environment"),
    migrate_data: bool = typer.Option(False, "--data", help="Include data migration"),
    backup_first: bool = typer.Option(True, "--backup", help="Create backup before migration")
):
    """🔄 Migrate between environments"""
    
    from backend.scripts.deployment.model_migration import EnvironmentMigrator
    
    migrator = EnvironmentMigrator()
    
    if backup_first:
        typer.echo("💾 Creating backup before migration...")
        # Create backup
        ctx = typer.Context(backup)
        ctx.invoke(backup, target="full")
    
    typer.echo(f"🔄 Migrating from {from_env} to {to_env}...")
    results = migrator.migrate_environment(
        from_env=from_env,
        to_env=to_env,
        include_data=migrate_data
    )
    
    if results.success:
        typer.echo("✅ Migration completed successfully!")
    else:
        typer.echo("❌ Migration failed")
        raise typer.Exit(1)

@app.command()
def monitor(
    environment: str = typer.Argument(..., help="Environment to monitor"),
    duration: int = typer.Option(300, "--duration", help="Monitoring duration in seconds"),
    alert_on_issues: bool = typer.Option(True, "--alerts/--no-alerts", help="Send alerts on issues")
):
    """📊 Monitor deployment health"""
    
    from backend.scripts.deployment.monitoring_setup import DeploymentMonitor
    
    monitor = DeploymentMonitor()
    
    typer.echo(f"📊 Starting deployment monitoring for {environment}...")
    monitor.start_monitoring(
        environment=environment,
        duration=duration,
        alert_on_issues=alert_on_issues
    )

@app.command()
def logs(
    environment: str = typer.Argument(..., help="Environment to check"),
    lines: int = typer.Option(100, "--lines", "-n", help="Number of lines to show"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
    level: str = typer.Option("INFO", "--level", help="Log level filter")
):
    """📋 View deployment logs"""
    
    from backend.scripts.deployment.deploy import DeploymentManager
    
    manager = DeploymentManager()
    
    if follow:
        typer.echo(f"📋 Following logs for {environment} (Ctrl+C to stop)...")
        manager.follow_logs(environment, level=level)
    else:
        typer.echo(f"📋 Showing last {lines} log lines for {environment}...")
        logs = manager.get_logs(environment, lines=lines, level=level)
        typer.echo(logs)