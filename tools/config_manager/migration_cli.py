#!/usr/bin/env python3
"""
Configuration Migration CLI Tool

Command-line interface for migrating scattered configuration files
to the unified configuration system.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from tools.config_manager.config_unifier import ConfigurationUnifier, MigrationReport
from tools.config_manager.unified_config import UnifiedConfig


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    import logging
    
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def discover_command(args):
    """Handle the discover command"""
    unifier = ConfigurationUnifier(Path(args.project_root))
    sources = unifier.discover_config_files()
    
    print(f"Found {len(sources)} configuration files:")
    print()
    
    # Group by category
    by_category = {}
    for source in sources:
        if source.category not in by_category:
            by_category[source.category] = []
        by_category[source.category].append(source)
    
    for category, category_sources in sorted(by_category.items()):
        print(f"{category.upper()}:")
        for source in category_sources:
            confidence_str = f"({source.confidence:.1f})" if source.confidence < 1.0 else ""
            print(f"  - {source.path} [{source.format}] {confidence_str}")
        print()
    
    if args.output:
        # Save discovery results to file
        discovery_data = {
            'total_sources': len(sources),
            'sources': [
                {
                    'path': str(source.path),
                    'format': source.format,
                    'category': source.category,
                    'confidence': source.confidence
                }
                for source in sources
            ]
        }
        
        Path(args.output).write_text(json.dumps(discovery_data, indent=2))
        print(f"Discovery results saved to {args.output}")


def preview_command(args):
    """Handle the preview command"""
    unifier = ConfigurationUnifier(Path(args.project_root))
    preview = unifier.generate_migration_preview()
    
    print("Migration Preview")
    print("=" * 50)
    print(f"Total sources: {preview['total_sources']}")
    print()
    
    print("Sources by category:")
    for category, sources in preview['sources_by_category'].items():
        print(f"  {category}: {len(sources)} files")
        if args.verbose:
            for source in sources:
                conf_str = f" (confidence: {source['confidence']:.1f})"
                print(f"    - {source['path']}{conf_str}")
    print()
    
    print("Sources by format:")
    for format_type, count in preview['sources_by_format'].items():
        print(f"  {format_type}: {count} files")
    print()
    
    if preview['potential_conflicts']:
        print("Potential conflicts:")
        for conflict in preview['potential_conflicts']:
            print(f"  - {conflict['type']} in {conflict['category']}")
            if args.verbose:
                for source in conflict['sources']:
                    print(f"    - {source}")
        print()
    
    if preview['recommendations']:
        print("Recommendations:")
        for rec in preview['recommendations']:
            print(f"  - {rec['message']}")
        print()
    
    if args.output:
        Path(args.output).write_text(json.dumps(preview, indent=2))
        print(f"Preview saved to {args.output}")


def migrate_command(args):
    """Handle the migrate command"""
    unifier = ConfigurationUnifier(Path(args.project_root))
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(args.project_root) / "config" / "unified-config.yaml"
    
    print(f"Migrating configuration to {output_path}")
    
    # Perform migration
    report = unifier.migrate_to_unified_config(
        output_path=output_path,
        create_backup=not args.no_backup
    )
    
    # Display results
    print("\nMigration Results")
    print("=" * 50)
    print(f"Success: {report.success}")
    print(f"Sources found: {len(report.sources_found)}")
    print(f"Sources migrated: {len(report.sources_migrated)}")
    print(f"Sources skipped: {len(report.sources_skipped)}")
    
    if report.backup_path:
        print(f"Backup created: {report.backup_path}")
    
    if report.errors:
        print("\nErrors:")
        for error in report.errors:
            print(f"  - {error}")
    
    if report.warnings:
        print("\nWarnings:")
        for warning in report.warnings:
            print(f"  - {warning}")
    
    if args.verbose and report.sources_migrated:
        print("\nMigrated sources:")
        for source in report.sources_migrated:
            print(f"  - {source.path} [{source.category}]")
    
    if args.verbose and report.sources_skipped:
        print("\nSkipped sources:")
        for source in report.sources_skipped:
            print(f"  - {source.path} [{source.category}]")
    
    # Save report if requested
    if args.report:
        report_data = {
            'timestamp': report.timestamp.isoformat(),
            'success': report.success,
            'sources_found': len(report.sources_found),
            'sources_migrated': [str(s.path) for s in report.sources_migrated],
            'sources_skipped': [str(s.path) for s in report.sources_skipped],
            'backup_path': str(report.backup_path) if report.backup_path else None,
            'unified_config_path': str(report.unified_config_path),
            'errors': report.errors,
            'warnings': report.warnings
        }
        
        Path(args.report).write_text(json.dumps(report_data, indent=2))
        print(f"\nMigration report saved to {args.report}")


def rollback_command(args):
    """Handle the rollback command"""
    unifier = ConfigurationUnifier(Path(args.project_root))
    backup_path = Path(args.backup_path)
    
    if not backup_path.exists():
        print(f"Error: Backup path does not exist: {backup_path}")
        sys.exit(1)
    
    print(f"Rolling back migration from {backup_path}")
    
    success = unifier.rollback_migration(backup_path)
    
    if success:
        print("Rollback completed successfully")
    else:
        print("Rollback failed - check logs for details")
        sys.exit(1)


def validate_command(args):
    """Handle the validate command"""
    config_path = Path(args.config_path)
    
    if not config_path.exists():
        print(f"Error: Configuration file does not exist: {config_path}")
        sys.exit(1)
    
    try:
        config = UnifiedConfig.from_file(config_path)
        print(f"Configuration file {config_path} is valid")
        
        if args.verbose:
            print(f"System name: {config.system.name}")
            print(f"System version: {config.system.version}")
            print(f"API host: {config.api.host}")
            print(f"API port: {config.api.port}")
            print(f"Models base path: {config.models.base_path}")
    
    except Exception as e:
        print(f"Error: Configuration file is invalid: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Configuration Migration Tool for WAN22 Project"
    )
    
    parser.add_argument(
        '--project-root',
        default='.',
        help='Project root directory (default: current directory)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Discover command
    discover_parser = subparsers.add_parser(
        'discover',
        help='Discover configuration files in the project'
    )
    discover_parser.add_argument(
        '--output', '-o',
        help='Save discovery results to JSON file'
    )
    
    # Preview command
    preview_parser = subparsers.add_parser(
        'preview',
        help='Preview migration without performing it'
    )
    preview_parser.add_argument(
        '--output', '-o',
        help='Save preview results to JSON file'
    )
    
    # Migrate command
    migrate_parser = subparsers.add_parser(
        'migrate',
        help='Perform configuration migration'
    )
    migrate_parser.add_argument(
        '--output', '-o',
        help='Output path for unified configuration file'
    )
    migrate_parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup of original files'
    )
    migrate_parser.add_argument(
        '--report', '-r',
        help='Save migration report to JSON file'
    )
    
    # Rollback command
    rollback_parser = subparsers.add_parser(
        'rollback',
        help='Rollback a previous migration'
    )
    rollback_parser.add_argument(
        'backup_path',
        help='Path to backup directory'
    )
    
    # Validate command
    validate_parser = subparsers.add_parser(
        'validate',
        help='Validate a unified configuration file'
    )
    validate_parser.add_argument(
        'config_path',
        help='Path to configuration file to validate'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Execute command
    if args.command == 'discover':
        discover_command(args)
    elif args.command == 'preview':
        preview_command(args)
    elif args.command == 'migrate':
        migrate_command(args)
    elif args.command == 'rollback':
        rollback_command(args)
    elif args.command == 'validate':
        validate_command(args)


if __name__ == '__main__':
    main()