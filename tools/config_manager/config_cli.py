#!/usr/bin/env python3
"""
Configuration Management CLI

Command-line interface for managing unified configuration,
including get/set operations, validation, and monitoring.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from tools.config_manager.config_api import ConfigurationAPI, ConfigChangeEvent
from tools.config_manager.config_validator import ConfigurationValidator
from tools.config_manager.unified_config import UnifiedConfig


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    import logging
    
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def get_command(args):
    """Handle the get command"""
    with ConfigurationAPI(args.config_file, auto_reload=False) as api:
        try:
            if args.path:
                value = api.get_config(args.path)
                if args.format == 'json':
                    print(json.dumps(value, indent=2, default=str))
                else:
                    print(value)
            else:
                # Get entire configuration
                config = api.get_config()
                if args.format == 'json':
                    print(config.to_json())
                elif args.format == 'yaml':
                    print(config.to_yaml())
                else:
                    print(config.to_yaml())  # Default to YAML
        
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)


def set_command(args):
    """Handle the set command"""
    with ConfigurationAPI(args.config_file, validate_changes=not args.no_validate) as api:
        try:
            # Parse value
            value = args.value
            
            # Try to parse as JSON for complex types
            if args.type == 'auto':
                try:
                    value = json.loads(args.value)
                except json.JSONDecodeError:
                    # Keep as string
                    pass
            elif args.type == 'int':
                value = int(args.value)
            elif args.type == 'float':
                value = float(args.value)
            elif args.type == 'bool':
                value = args.value.lower() in ('true', '1', 'yes', 'on')
            elif args.type == 'json':
                value = json.loads(args.value)
            
            # Set the value
            success = api.set_config(args.path, value)
            
            if success:
                print(f"Successfully set {args.path} = {value}")
                
                # Save if requested
                if args.save:
                    if api.save_config():
                        print("Configuration saved to file")
                    else:
                        print("Warning: Failed to save configuration")
            else:
                print(f"Failed to set {args.path}")
                sys.exit(1)
        
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)


def validate_command(args):
    """Handle the validate command"""
    validator = ConfigurationValidator()
    
    try:
        if args.config_file:
            result = validator.validate_config_file(args.config_file)
        else:
            # Validate default configuration
            config = UnifiedConfig()
            result = validator.validate_config(config)
        
        if args.format == 'json':
            # Output as JSON
            output = {
                'valid': result.is_valid,
                'issues_count': len(result.issues),
                'warnings': result.warnings_count,
                'errors': result.errors_count,
                'critical': result.critical_count,
                'issues': [
                    {
                        'severity': issue.severity.value,
                        'category': issue.category,
                        'field_path': issue.field_path,
                        'message': issue.message,
                        'current_value': issue.current_value,
                        'suggested_value': issue.suggested_value,
                        'fix_suggestion': issue.fix_suggestion
                    }
                    for issue in result.issues
                ]
            }
            print(json.dumps(output, indent=2, default=str))
        else:
            # Output as human-readable report
            report = validator.generate_validation_report(result)
            print(report)
        
        # Exit with error code if validation failed
        if not result.is_valid:
            sys.exit(1)
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def info_command(args):
    """Handle the info command"""
    with ConfigurationAPI(args.config_file, auto_reload=False) as api:
        info = api.get_config_info()
        
        if args.format == 'json':
            print(json.dumps(info, indent=2, default=str))
        else:
            print("Configuration Information")
            print("=" * 30)
            for key, value in info.items():
                print(f"{key.replace('_', ' ').title()}: {value}")


def watch_command(args):
    """Handle the watch command"""
    print(f"Watching configuration file: {args.config_file or 'config/unified-config.yaml'}")
    print("Press Ctrl+C to stop...")
    
    def change_callback(event: ConfigChangeEvent):
        timestamp = event.timestamp.strftime("%H:%M:%S")
        print(f"[{timestamp}] {event.field_path}: {event.old_value} -> {event.new_value} (source: {event.source})")
    
    try:
        with ConfigurationAPI(args.config_file, auto_reload=True) as api:
            api.register_change_callback(change_callback)
            
            # Keep running until interrupted
            while True:
                time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nStopping configuration watcher...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def export_command(args):
    """Handle the export command"""
    with ConfigurationAPI(args.config_file, auto_reload=False) as api:
        try:
            exported = api.export_config(args.format)
            
            if args.output:
                Path(args.output).write_text(exported)
                print(f"Configuration exported to {args.output}")
            else:
                print(exported)
        
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)


def import_command(args):
    """Handle the import command"""
    with ConfigurationAPI(args.config_file, validate_changes=not args.no_validate) as api:
        try:
            if args.input:
                config_str = Path(args.input).read_text()
            else:
                # Read from stdin
                config_str = sys.stdin.read()
            
            success = api.import_config(config_str, args.format)
            
            if success:
                print("Configuration imported successfully")
                
                # Save if requested
                if args.save:
                    if api.save_config():
                        print("Configuration saved to file")
                    else:
                        print("Warning: Failed to save configuration")
            else:
                print("Failed to import configuration")
                sys.exit(1)
        
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)


def environment_command(args):
    """Handle the environment command"""
    with ConfigurationAPI(args.config_file) as api:
        try:
            success = api.apply_environment_overrides(args.environment)
            
            if success:
                print(f"Applied environment overrides for: {args.environment}")
                
                # Save if requested
                if args.save:
                    if api.save_config():
                        print("Configuration saved to file")
                    else:
                        print("Warning: Failed to save configuration")
            else:
                print(f"Failed to apply environment overrides for: {args.environment}")
                sys.exit(1)
        
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Configuration Management CLI for WAN22 Project"
    )
    
    parser.add_argument(
        '--config-file', '-c',
        help='Configuration file path (default: config/unified-config.yaml)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Get command
    get_parser = subparsers.add_parser('get', help='Get configuration value')
    get_parser.add_argument('path', nargs='?', help='Configuration path (e.g., api.port)')
    get_parser.add_argument('--format', choices=['auto', 'json', 'yaml'], default='auto')
    
    # Set command
    set_parser = subparsers.add_parser('set', help='Set configuration value')
    set_parser.add_argument('path', help='Configuration path (e.g., api.port)')
    set_parser.add_argument('value', help='New value')
    set_parser.add_argument('--type', choices=['auto', 'str', 'int', 'float', 'bool', 'json'], default='auto')
    set_parser.add_argument('--save', action='store_true', help='Save configuration to file')
    set_parser.add_argument('--no-validate', action='store_true', help='Skip validation')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate configuration')
    validate_parser.add_argument('--format', choices=['text', 'json'], default='text')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show configuration information')
    info_parser.add_argument('--format', choices=['text', 'json'], default='text')
    
    # Watch command
    watch_parser = subparsers.add_parser('watch', help='Watch for configuration changes')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export configuration')
    export_parser.add_argument('--format', choices=['yaml', 'json'], default='yaml')
    export_parser.add_argument('--output', '-o', help='Output file (default: stdout)')
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import configuration')
    import_parser.add_argument('--format', choices=['auto', 'yaml', 'json'], default='auto')
    import_parser.add_argument('--input', '-i', help='Input file (default: stdin)')
    import_parser.add_argument('--save', action='store_true', help='Save configuration to file')
    import_parser.add_argument('--no-validate', action='store_true', help='Skip validation')
    
    # Environment command
    env_parser = subparsers.add_parser('environment', help='Apply environment overrides')
    env_parser.add_argument('environment', choices=['development', 'staging', 'production', 'testing'])
    env_parser.add_argument('--save', action='store_true', help='Save configuration to file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Execute command
    if args.command == 'get':
        get_command(args)
    elif args.command == 'set':
        set_command(args)
    elif args.command == 'validate':
        validate_command(args)
    elif args.command == 'info':
        info_command(args)
    elif args.command == 'watch':
        watch_command(args)
    elif args.command == 'export':
        export_command(args)
    elif args.command == 'import':
        import_command(args)
    elif args.command == 'environment':
        environment_command(args)


if __name__ == '__main__':
    main()