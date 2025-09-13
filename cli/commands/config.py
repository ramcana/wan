"""Configuration management commands"""

import typer
from pathlib import Path
from typing import Optional
import sys

app = typer.Typer()

@app.command()
def validate(
    config_file: Optional[Path] = typer.Option(None, "--file", help="Specific config file to validate"),
    fix: bool = typer.Option(False, "--fix", help="Attempt to fix validation issues")
):
    """✅ Validate configuration files"""
    
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from tools.config_manager.config_validator import ConfigValidator
    
    validator = ConfigValidator()
    
    if config_file:
        results = validator.validate_file(config_file)
        files_checked = [config_file]
    else:
        results = validator.validate_all_configs()
        files_checked = validator.get_config_files()
    
    typer.echo(f"✅ Validated {len(files_checked)} configuration files")
    
    if results.has_errors:
        typer.echo(f"❌ Found {len(results.errors)} errors")
        
        if fix:
            typer.echo("🔧 Attempting to fix issues...")
            validator.fix_issues(results)
        else:
            validator.show_errors(results)
            typer.echo("Use --fix to attempt automatic fixes")
    else:
        typer.echo("✅ All configurations are valid")

@app.command()
def unify():
    """🔗 Unify scattered configuration files"""
    
    from tools.config_manager.config_unifier import ConfigUnifier
    
    unifier = ConfigUnifier()
    unified_config = unifier.create_unified_config()
    
    typer.echo("🔗 Configuration files unified")
    typer.echo(f"📄 Unified config saved to: {unified_config}")

@app.command()
def migrate(
    from_version: str = typer.Argument(..., help="Source version"),
    to_version: str = typer.Argument(..., help="Target version"),
    backup: bool = typer.Option(True, "--backup", help="Create backup before migration")
):
    """🚀 Migrate configuration between versions"""
    
    from tools.config_manager.migration_cli import ConfigMigrator
    
    migrator = ConfigMigrator()
    
    if backup:
        typer.echo("💾 Creating configuration backup...")
        migrator.create_backup()
    
    typer.echo(f"🚀 Migrating from {from_version} to {to_version}...")
    success = migrator.migrate(from_version, to_version)
    
    if success:
        typer.echo("✅ Migration completed successfully")
    else:
        typer.echo("❌ Migration failed - check logs for details")
        raise typer.Exit(1)

@app.command()
def show(
    format: str = typer.Option("yaml", "--format", help="Output format (yaml, json, table)"),
    section: Optional[str] = typer.Option(None, "--section", help="Show specific section only")
):
    """📋 Show current configuration"""
    
    from tools.config_manager.config_api import ConfigAPI
    
    api = ConfigAPI()
    config = api.get_current_config()
    
    if section:
        config = config.get(section, {})
    
    if format == "json":
        import json
        typer.echo(json.dumps(config, indent=2))
    elif format == "table":
        api.show_config_table(config)
    else:  # yaml
        import yaml
        typer.echo(yaml.dump(config, default_flow_style=False))

@app.command()
def set(
    key: str = typer.Argument(..., help="Configuration key (dot notation)"),
    value: str = typer.Argument(..., help="Configuration value"),
    config_file: Optional[Path] = typer.Option(None, "--file", help="Target config file")
):
    """⚙️ Set configuration value"""
    
    from tools.config_manager.config_api import ConfigAPI
    
    api = ConfigAPI()
    api.set_config_value(key, value, config_file)
    
    typer.echo(f"✅ Set {key} = {value}")

@app.command()
def backup(
    destination: Optional[Path] = typer.Option(None, "--dest", help="Backup destination")
):
    """💾 Backup all configuration files"""
    
    from tools.config_manager.config_api import ConfigAPI
    
    api = ConfigAPI()
    backup_path = api.create_backup(destination)
    
    typer.echo(f"💾 Configuration backup created: {backup_path}")
