"""
Command-line interface for managing user preferences.
"""

import json
from pathlib import Path
from typing import Optional
import click
from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm, Prompt

from .preferences import PreferenceManager, UserPreferences


console = Console()


@click.group(name="preferences")
def preferences_cli():
    """Manage user preferences for the startup manager."""
    pass


@preferences_cli.command()
@click.option('--preferences-dir', type=click.Path(path_type=Path), help='Custom preferences directory')
def show(preferences_dir: Optional[Path]):
    """Show current user preferences."""
    manager = PreferenceManager(preferences_dir)
    preferences = manager.load_preferences()
    
    console.print("\n[bold blue]Current User Preferences[/bold blue]")
    
    # Create table for preferences
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Description", style="dim")
    
    # UI Preferences
    table.add_row("auto_open_browser", str(preferences.auto_open_browser), "Automatically open browser after startup")
    table.add_row("show_progress_bars", str(preferences.show_progress_bars), "Show progress bars during operations")
    table.add_row("verbose_output", str(preferences.verbose_output), "Show verbose output by default")
    table.add_row("confirm_destructive_actions", str(preferences.confirm_destructive_actions), "Ask for confirmation before destructive actions")
    
    # Recovery Preferences
    table.add_row("preferred_recovery_strategy", preferences.preferred_recovery_strategy, "Preferred recovery strategy")
    table.add_row("auto_retry_failed_operations", str(preferences.auto_retry_failed_operations), "Automatically retry failed operations")
    table.add_row("max_auto_retries", str(preferences.max_auto_retries), "Maximum automatic retry attempts")
    
    # Port Management
    table.add_row("preferred_backend_port", str(preferences.preferred_backend_port or "Auto"), "Preferred backend port")
    table.add_row("preferred_frontend_port", str(preferences.preferred_frontend_port or "Auto"), "Preferred frontend port")
    table.add_row("allow_port_auto_increment", str(preferences.allow_port_auto_increment), "Allow automatic port increment")
    
    # Security
    table.add_row("allow_admin_elevation", str(preferences.allow_admin_elevation), "Allow automatic admin elevation")
    table.add_row("trust_local_processes", str(preferences.trust_local_processes), "Trust local processes for port conflicts")
    
    # Logging
    table.add_row("keep_detailed_logs", str(preferences.keep_detailed_logs), "Keep detailed logs of operations")
    table.add_row("log_retention_days", str(preferences.log_retention_days), "Days to keep log files")
    
    # Advanced
    table.add_row("enable_experimental_features", str(preferences.enable_experimental_features), "Enable experimental features")
    table.add_row("startup_timeout_multiplier", str(preferences.startup_timeout_multiplier), "Multiplier for startup timeouts")
    
    console.print(table)


@preferences_cli.command()
@click.option('--preferences-dir', type=click.Path(path_type=Path), help='Custom preferences directory')
def edit(preferences_dir: Optional[Path]):
    """Interactively edit user preferences."""
    manager = PreferenceManager(preferences_dir)
    preferences = manager.load_preferences()
    
    console.print("\n[bold blue]Edit User Preferences[/bold blue]")
    console.print("Press Enter to keep current value, or type new value:")
    
    # UI Preferences
    console.print("\n[bold yellow]UI Preferences[/bold yellow]")
    preferences.auto_open_browser = Confirm.ask(
        f"Auto open browser after startup (current: {preferences.auto_open_browser})",
        default=preferences.auto_open_browser
    )
    
    preferences.show_progress_bars = Confirm.ask(
        f"Show progress bars (current: {preferences.show_progress_bars})",
        default=preferences.show_progress_bars
    )
    
    preferences.verbose_output = Confirm.ask(
        f"Verbose output by default (current: {preferences.verbose_output})",
        default=preferences.verbose_output
    )
    
    preferences.confirm_destructive_actions = Confirm.ask(
        f"Confirm destructive actions (current: {preferences.confirm_destructive_actions})",
        default=preferences.confirm_destructive_actions
    )
    
    # Recovery Preferences
    console.print("\n[bold yellow]Recovery Preferences[/bold yellow]")
    preferences.preferred_recovery_strategy = Prompt.ask(
        f"Preferred recovery strategy (current: {preferences.preferred_recovery_strategy})",
        choices=["auto", "manual", "aggressive"],
        default=preferences.preferred_recovery_strategy
    )
    
    preferences.auto_retry_failed_operations = Confirm.ask(
        f"Auto retry failed operations (current: {preferences.auto_retry_failed_operations})",
        default=preferences.auto_retry_failed_operations
    )
    
    max_retries = Prompt.ask(
        f"Maximum auto retries (current: {preferences.max_auto_retries})",
        default=str(preferences.max_auto_retries)
    )
    try:
        preferences.max_auto_retries = int(max_retries)
    except ValueError:
        console.print("[red]Invalid number, keeping current value[/red]")
    
    # Port Management
    console.print("\n[bold yellow]Port Management[/bold yellow]")
    backend_port = Prompt.ask(
        f"Preferred backend port (current: {preferences.preferred_backend_port or 'Auto'})",
        default=str(preferences.preferred_backend_port) if preferences.preferred_backend_port else ""
    )
    if backend_port and backend_port.isdigit():
        port_num = int(backend_port)
        if 1024 <= port_num <= 65535:
            preferences.preferred_backend_port = port_num
        else:
            console.print("[red]Invalid port range, keeping current value[/red]")
    elif not backend_port:
        preferences.preferred_backend_port = None
    
    frontend_port = Prompt.ask(
        f"Preferred frontend port (current: {preferences.preferred_frontend_port or 'Auto'})",
        default=str(preferences.preferred_frontend_port) if preferences.preferred_frontend_port else ""
    )
    if frontend_port and frontend_port.isdigit():
        port_num = int(frontend_port)
        if 1024 <= port_num <= 65535:
            preferences.preferred_frontend_port = port_num
        else:
            console.print("[red]Invalid port range, keeping current value[/red]")
    elif not frontend_port:
        preferences.preferred_frontend_port = None
    
    preferences.allow_port_auto_increment = Confirm.ask(
        f"Allow port auto increment (current: {preferences.allow_port_auto_increment})",
        default=preferences.allow_port_auto_increment
    )
    
    # Security
    console.print("\n[bold yellow]Security Preferences[/bold yellow]")
    preferences.allow_admin_elevation = Confirm.ask(
        f"Allow admin elevation (current: {preferences.allow_admin_elevation})",
        default=preferences.allow_admin_elevation
    )
    
    preferences.trust_local_processes = Confirm.ask(
        f"Trust local processes (current: {preferences.trust_local_processes})",
        default=preferences.trust_local_processes
    )
    
    # Logging
    console.print("\n[bold yellow]Logging Preferences[/bold yellow]")
    preferences.keep_detailed_logs = Confirm.ask(
        f"Keep detailed logs (current: {preferences.keep_detailed_logs})",
        default=preferences.keep_detailed_logs
    )
    
    log_retention = Prompt.ask(
        f"Log retention days (current: {preferences.log_retention_days})",
        default=str(preferences.log_retention_days)
    )
    try:
        preferences.log_retention_days = int(log_retention)
    except ValueError:
        console.print("[red]Invalid number, keeping current value[/red]")
    
    # Advanced
    console.print("\n[bold yellow]Advanced Preferences[/bold yellow]")
    preferences.enable_experimental_features = Confirm.ask(
        f"Enable experimental features (current: {preferences.enable_experimental_features})",
        default=preferences.enable_experimental_features
    )
    
    timeout_multiplier = Prompt.ask(
        f"Startup timeout multiplier (current: {preferences.startup_timeout_multiplier})",
        default=str(preferences.startup_timeout_multiplier)
    )
    try:
        multiplier = float(timeout_multiplier)
        if 0.5 <= multiplier <= 5.0:
            preferences.startup_timeout_multiplier = multiplier
        else:
            console.print("[red]Invalid range (0.5-5.0), keeping current value[/red]")
    except ValueError:
        console.print("[red]Invalid number, keeping current value[/red]")
    
    # Save preferences
    manager._preferences = preferences
    manager.save_preferences()
    
    console.print("\n[bold green]Preferences saved successfully![/bold green]")


@preferences_cli.command()
@click.option('--preferences-dir', type=click.Path(path_type=Path), help='Custom preferences directory')
def reset(preferences_dir: Optional[Path]):
    """Reset preferences to default values."""
    if not Confirm.ask("Are you sure you want to reset all preferences to defaults?"):
        console.print("Reset cancelled.")
        return
    
    manager = PreferenceManager(preferences_dir)
    
    # Create backup before reset
    backup_path = manager.create_backup("pre_reset_backup")
    console.print(f"Created backup at: {backup_path}")
    
    # Reset to defaults
    default_preferences = UserPreferences()
    manager._preferences = default_preferences
    manager.save_preferences()
    
    console.print("[bold green]Preferences reset to defaults![/bold green]")


@preferences_cli.command()
@click.option('--preferences-dir', type=click.Path(path_type=Path), help='Custom preferences directory')
def backup(preferences_dir: Optional[Path]):
    """Create a backup of current preferences and configuration."""
    manager = PreferenceManager(preferences_dir)
    
    backup_name = Prompt.ask("Backup name (optional)", default="")
    if not backup_name:
        backup_name = None
    
    backup_path = manager.create_backup(backup_name)
    console.print(f"[bold green]Backup created at: {backup_path}[/bold green]")


@preferences_cli.command()
@click.option('--preferences-dir', type=click.Path(path_type=Path), help='Custom preferences directory')
def restore(preferences_dir: Optional[Path]):
    """Restore preferences and configuration from a backup."""
    manager = PreferenceManager(preferences_dir)
    
    backups = manager.list_backups()
    if not backups:
        console.print("[yellow]No backups found.[/yellow]")
        return
    
    console.print("\n[bold blue]Available Backups[/bold blue]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Name", style="cyan")
    table.add_column("Created", style="green")
    table.add_column("Description", style="dim")
    
    for backup in backups:
        created_at = backup.get("created_at", "Unknown")
        if created_at != "Unknown":
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                created_at = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                pass
        
        table.add_row(
            backup["name"],
            created_at,
            backup.get("description", "No description")
        )
    
    console.print(table)
    
    backup_name = Prompt.ask("Enter backup name to restore")
    
    if backup_name not in [b["name"] for b in backups]:
        console.print("[red]Backup not found.[/red]")
        return
    
    if not Confirm.ask(f"Are you sure you want to restore backup '{backup_name}'?"):
        console.print("Restore cancelled.")
        return
    
    try:
        success = manager.restore_backup(backup_name)
        if success:
            console.print(f"[bold green]Successfully restored backup '{backup_name}'![/bold green]")
        else:
            console.print(f"[red]Failed to restore backup '{backup_name}'.[/red]")
    except Exception as e:
        console.print(f"[red]Error restoring backup: {e}[/red]")


@preferences_cli.command()
@click.option('--preferences-dir', type=click.Path(path_type=Path), help='Custom preferences directory')
@click.option('--keep', default=10, help='Number of backups to keep')
def cleanup(preferences_dir: Optional[Path], keep: int):
    """Clean up old backups."""
    manager = PreferenceManager(preferences_dir)
    
    removed_count = manager.cleanup_old_backups(keep_count=keep)
    
    if removed_count > 0:
        console.print(f"[bold green]Cleaned up {removed_count} old backups.[/bold green]")
    else:
        console.print("[yellow]No backups to clean up.[/yellow]")


@preferences_cli.command()
@click.option('--preferences-dir', type=click.Path(path_type=Path), help='Custom preferences directory')
@click.option('--target-version', default='2.0.0', help='Target version to migrate to')
def migrate(preferences_dir: Optional[Path], target_version: str):
    """Migrate configuration to a newer version."""
    manager = PreferenceManager(preferences_dir)
    
    current_version = manager.load_version_info().version
    console.print(f"Current version: {current_version}")
    console.print(f"Target version: {target_version}")
    
    if not Confirm.ask("Proceed with migration?"):
        console.print("Migration cancelled.")
        return
    
    try:
        migrated = manager.migrate_configuration(target_version)
        if migrated:
            console.print(f"[bold green]Successfully migrated to version {target_version}![/bold green]")
            
            # Show migration notes
            version_info = manager.load_version_info()
            if version_info.migration_notes:
                console.print("\n[bold yellow]Migration Notes:[/bold yellow]")
                for note in version_info.migration_notes:
                    console.print(f"  • {note}")
        else:
            console.print("[yellow]No migration needed.[/yellow]")
    except Exception as e:
        console.print(f"[red]Migration failed: {e}[/red]")


if __name__ == "__main__":
    preferences_cli()