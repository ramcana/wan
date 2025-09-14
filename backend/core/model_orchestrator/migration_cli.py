"""
Migration CLI - Command-line interface for migration and compatibility tools.

This module provides CLI commands for migrating configurations, validating
manifests, and managing rollbacks.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .migration_manager import (
    ConfigurationMigrator, 
    ManifestValidator, 
    RollbackManager,
    LegacyPathAdapter,
    FeatureFlags,
    MigrationResult
)
from .exceptions import MigrationError, ValidationError


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    import logging
    
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def cmd_migrate_config(args) -> int:
    """Migrate legacy configuration to new manifest format."""
    setup_logging(args.verbose)
    
    migrator = ConfigurationMigrator()
    
    try:
        result = migrator.migrate_configuration(
            legacy_config_path=args.legacy_config,
            output_manifest_path=args.output_manifest,
            legacy_models_dir=args.legacy_models_dir,
            backup=not args.no_backup,
            scan_files=args.scan_files
        )
        
        # Print results
        if args.json:
            print(json.dumps({
                "success": result.success,
                "manifest_path": result.manifest_path,
                "backup_path": result.backup_path,
                "warnings": result.warnings,
                "errors": result.errors
            }, indent=2))
        else:
            print(f"Migration {'succeeded' if result.success else 'failed'}")
            print(f"Manifest: {result.manifest_path}")
            
            if result.backup_path:
                print(f"Backup: {result.backup_path}")
                
            if result.warnings:
                print("\nWarnings:")
                for warning in result.warnings:
                    print(f"  - {warning}")
                    
            if result.errors:
                print("\nErrors:")
                for error in result.errors:
                    print(f"  - {error}")
        
        return 0 if result.success else 1
        
    except Exception as e:
        if args.json:
            print(json.dumps({"success": False, "error": str(e)}, indent=2))
        else:
            print(f"Migration failed: {e}")
        return 1


def cmd_validate_manifest(args) -> int:
    """Validate a manifest file."""
    setup_logging(args.verbose)
    
    validator = ManifestValidator()
    
    try:
        # Validate manifest structure
        errors = validator.validate_manifest_file(args.manifest_path)
        
        # Validate compatibility if legacy config provided
        if args.legacy_config:
            compat_errors = validator.validate_configuration_compatibility(
                args.manifest_path, 
                args.legacy_config
            )
            errors.extend(compat_errors)
        
        # Print results
        if args.json:
            print(json.dumps({
                "valid": len(errors) == 0,
                "errors": [str(e) for e in errors]
            }, indent=2))
        else:
            if errors:
                print(f"Validation failed with {len(errors)} error(s):")
                for error in errors:
                    print(f"  - {error}")
            else:
                print("Manifest validation passed")
        
        return 0 if len(errors) == 0 else 1
        
    except Exception as e:
        if args.json:
            print(json.dumps({"valid": False, "error": str(e)}, indent=2))
        else:
            print(f"Validation failed: {e}")
        return 1


def cmd_migrate_paths(args) -> int:
    """Migrate model files from legacy paths to orchestrator paths."""
    setup_logging(args.verbose)
    
    adapter = LegacyPathAdapter(
        legacy_models_dir=args.legacy_models_dir,
        orchestrator_models_root=args.orchestrator_models_root
    )
    
    success_count = 0
    total_count = 0
    
    for model_name in args.model_names:
        total_count += 1
        
        if not args.json:
            print(f"Migrating model: {model_name}")
        
        try:
            success = adapter.migrate_model_files(model_name, dry_run=args.dry_run)
            if success:
                success_count += 1
                if not args.json:
                    action = "Would migrate" if args.dry_run else "Migrated"
                    legacy_path = adapter.get_legacy_path(model_name)
                    new_path = adapter.map_legacy_path(model_name)
                    print(f"  {action}: {legacy_path} -> {new_path}")
            else:
                if not args.json:
                    print(f"  Failed to migrate {model_name}")
                    
        except Exception as e:
            if not args.json:
                print(f"  Error migrating {model_name}: {e}")
    
    # Print summary
    if args.json:
        print(json.dumps({
            "success": success_count == total_count,
            "migrated": success_count,
            "total": total_count,
            "dry_run": args.dry_run
        }, indent=2))
    else:
        print(f"\nMigration summary: {success_count}/{total_count} models migrated")
    
    return 0 if success_count == total_count else 1


def cmd_create_rollback(args) -> int:
    """Create a rollback point."""
    setup_logging(args.verbose)
    
    rollback_manager = RollbackManager()
    
    try:
        rollback_id = rollback_manager.create_rollback_point(
            config_paths=args.config_paths,
            rollback_dir=args.rollback_dir
        )
        
        if args.json:
            print(json.dumps({"rollback_id": rollback_id}, indent=2))
        else:
            print(f"Created rollback point: {rollback_id}")
        
        return 0
        
    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            print(f"Failed to create rollback point: {e}")
        return 1


def cmd_execute_rollback(args) -> int:
    """Execute a rollback."""
    setup_logging(args.verbose)
    
    rollback_manager = RollbackManager()
    
    try:
        success = rollback_manager.execute_rollback(
            rollback_id=args.rollback_id,
            rollback_dir=args.rollback_dir
        )
        
        if args.json:
            print(json.dumps({"success": success}, indent=2))
        else:
            if success:
                print(f"Rollback completed: {args.rollback_id}")
            else:
                print(f"Rollback failed: {args.rollback_id}")
        
        return 0 if success else 1
        
    except Exception as e:
        if args.json:
            print(json.dumps({"success": False, "error": str(e)}, indent=2))
        else:
            print(f"Rollback failed: {e}")
        return 1


def cmd_list_rollbacks(args) -> int:
    """List available rollback points."""
    setup_logging(args.verbose)
    
    rollback_manager = RollbackManager()
    
    try:
        rollback_points = rollback_manager.list_rollback_points(args.rollback_dir)
        
        if args.json:
            print(json.dumps(rollback_points, indent=2))
        else:
            if rollback_points:
                print(f"Available rollback points ({len(rollback_points)}):")
                for point in rollback_points:
                    import datetime
                    timestamp = datetime.datetime.fromtimestamp(point['timestamp'])
                    print(f"  {point['rollback_id']} - {timestamp} ({len(point['backed_up_files'])} files)")
            else:
                print("No rollback points found")
        
        return 0
        
    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}, indent=2))
        else:
            print(f"Failed to list rollback points: {e}")
        return 1


def cmd_show_feature_flags(args) -> int:
    """Show current feature flags."""
    flags = FeatureFlags.from_env()
    
    if args.json:
        import dataclasses
        print(json.dumps(dataclasses.asdict(flags), indent=2))
    else:
        print("Current feature flags:")
        print(f"  Enable Orchestrator: {flags.enable_orchestrator}")
        print(f"  Enable Manifest Validation: {flags.enable_manifest_validation}")
        print(f"  Enable Legacy Fallback: {flags.enable_legacy_fallback}")
        print(f"  Enable Path Migration: {flags.enable_path_migration}")
        print(f"  Enable Automatic Download: {flags.enable_automatic_download}")
        print(f"  Strict Validation: {flags.strict_validation}")
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="WAN Model Orchestrator Migration Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--json",
        action="store_true", 
        help="Output results in JSON format"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Migrate config command
    migrate_parser = subparsers.add_parser(
        "migrate-config",
        help="Migrate legacy config.json to models.toml manifest"
    )
    migrate_parser.add_argument(
        "legacy_config",
        help="Path to legacy config.json file"
    )
    migrate_parser.add_argument(
        "output_manifest", 
        help="Path for output models.toml manifest"
    )
    migrate_parser.add_argument(
        "--legacy-models-dir",
        help="Legacy models directory (for file scanning)"
    )
    migrate_parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup of existing manifest"
    )
    migrate_parser.add_argument(
        "--scan-files",
        action="store_true",
        help="Scan existing model files to generate file specifications"
    )
    migrate_parser.set_defaults(func=cmd_migrate_config)
    
    # Validate manifest command
    validate_parser = subparsers.add_parser(
        "validate-manifest",
        help="Validate a models.toml manifest file"
    )
    validate_parser.add_argument(
        "manifest_path",
        help="Path to models.toml manifest file"
    )
    validate_parser.add_argument(
        "--legacy-config",
        help="Path to legacy config.json for compatibility checking"
    )
    validate_parser.set_defaults(func=cmd_validate_manifest)
    
    # Migrate paths command
    migrate_paths_parser = subparsers.add_parser(
        "migrate-paths",
        help="Migrate model files from legacy paths to orchestrator paths"
    )
    migrate_paths_parser.add_argument(
        "legacy_models_dir",
        help="Legacy models directory"
    )
    migrate_paths_parser.add_argument(
        "orchestrator_models_root",
        help="Orchestrator models root directory"
    )
    migrate_paths_parser.add_argument(
        "model_names",
        nargs="+",
        help="Model names to migrate"
    )
    migrate_paths_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without actually doing it"
    )
    migrate_paths_parser.set_defaults(func=cmd_migrate_paths)
    
    # Create rollback command
    rollback_create_parser = subparsers.add_parser(
        "create-rollback",
        help="Create a rollback point"
    )
    rollback_create_parser.add_argument(
        "config_paths",
        nargs="+",
        help="Configuration file paths to backup"
    )
    rollback_create_parser.add_argument(
        "--rollback-dir",
        default=".rollbacks",
        help="Directory to store rollback data (default: .rollbacks)"
    )
    rollback_create_parser.set_defaults(func=cmd_create_rollback)
    
    # Execute rollback command
    rollback_exec_parser = subparsers.add_parser(
        "execute-rollback",
        help="Execute a rollback to a previous state"
    )
    rollback_exec_parser.add_argument(
        "rollback_id",
        help="Rollback point identifier"
    )
    rollback_exec_parser.add_argument(
        "--rollback-dir",
        default=".rollbacks",
        help="Directory containing rollback data (default: .rollbacks)"
    )
    rollback_exec_parser.set_defaults(func=cmd_execute_rollback)
    
    # List rollbacks command
    rollback_list_parser = subparsers.add_parser(
        "list-rollbacks",
        help="List available rollback points"
    )
    rollback_list_parser.add_argument(
        "--rollback-dir",
        default=".rollbacks",
        help="Directory containing rollback data (default: .rollbacks)"
    )
    rollback_list_parser.set_defaults(func=cmd_list_rollbacks)
    
    # Show feature flags command
    flags_parser = subparsers.add_parser(
        "show-flags",
        help="Show current feature flags"
    )
    flags_parser.set_defaults(func=cmd_show_feature_flags)
    
    # Parse arguments and execute command
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())