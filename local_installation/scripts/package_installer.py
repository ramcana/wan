#!/usr/bin/env python3
"""
Package Installer CLI

Command-line interface for creating and managing installation packages.
"""

import argparse
import sys
import os
from pathlib import Path
import json
import logging

# Add the scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from installer_packager import InstallerPackager, VersionManager
from logging_system import setup_logging


def create_package(args):
    """Create a new installation package."""
    try:
        # Setup logging
        setup_logging(level=args.log_level, log_file="packaging.log")
        logger = logging.getLogger(__name__)
        
        logger.info(f"Creating package {args.name} version {args.version}")
        
        # Initialize packager
        packager = InstallerPackager(
            source_dir=args.source_dir,
            output_dir=args.output_dir
        )
        
        # Create package
        package_path = packager.create_package(
            version=args.version,
            package_name=args.name
        )
        
        print(f"✓ Package created successfully: {package_path}")
        
        # Verify package if requested
        if args.verify:
            print("Verifying package integrity...")
            if packager.verify_package_integrity(package_path):
                print("✓ Package integrity verified")
            else:
                print("✗ Package integrity verification failed")
                return 1
        
        return 0
        
    except Exception as e:
        print(f"✗ Failed to create package: {e}")
        return 1


def verify_package(args):
    """Verify package integrity."""
    try:
        packager = InstallerPackager()
        
        print(f"Verifying package: {args.package}")
        if packager.verify_package_integrity(args.package):
            print("✓ Package integrity verified")
            return 0
        else:
            print("✗ Package integrity verification failed")
            return 1
            
    except Exception as e:
        print(f"✗ Failed to verify package: {e}")
        return 1


def extract_package(args):
    """Extract a package."""
    try:
        packager = InstallerPackager()
        
        print(f"Extracting package: {args.package}")
        print(f"Extract to: {args.output}")
        
        if packager.extract_package(args.package, args.output):
            print("✓ Package extracted successfully")
            return 0
        else:
            print("✗ Package extraction failed")
            return 1
            
    except Exception as e:
        print(f"✗ Failed to extract package: {e}")
        return 1


def check_version(args):
    """Check current version and updates."""
    try:
        version_manager = VersionManager(args.installation_dir)
        
        current_version = version_manager.get_current_version()
        if current_version:
            print(f"Current version: {current_version}")
        else:
            print("No version information found")
        
        if args.check_updates:
            print("Checking for updates...")
            update_info = version_manager.check_for_updates()
            
            if "error" in update_info:
                print(f"✗ Failed to check updates: {update_info['error']}")
                return 1
            
            if update_info.get("update_available"):
                print(f"✓ Update available: {update_info['latest_version']}")
                print(f"  Download: {update_info['download_url']}")
                print(f"  Size: {update_info['size_mb']} MB")
                print(f"  Notes: {update_info['release_notes']}")
            else:
                print("✓ You have the latest version")
        
        return 0
        
    except Exception as e:
        print(f"✗ Failed to check version: {e}")
        return 1


def create_backup(args):
    """Create a backup of the current installation."""
    try:
        version_manager = VersionManager(args.installation_dir)
        
        print(f"Creating backup: {args.name or 'auto-generated'}")
        backup_path = version_manager.create_backup(args.name)
        print(f"✓ Backup created: {backup_path}")
        
        return 0
        
    except Exception as e:
        print(f"✗ Failed to create backup: {e}")
        return 1


def restore_backup(args):
    """Restore from a backup."""
    try:
        version_manager = VersionManager(args.installation_dir)
        
        print(f"Restoring backup: {args.name}")
        if version_manager.restore_backup(args.name):
            print("✓ Backup restored successfully")
            return 0
        else:
            print("✗ Backup restoration failed")
            return 1
            
    except Exception as e:
        print(f"✗ Failed to restore backup: {e}")
        return 1


def list_info(args):
    """List package or installation information."""
    try:
        if args.type == "packages":
            # List available packages in output directory
            output_dir = Path(args.output_dir or "dist")
            if output_dir.exists():
                packages = list(output_dir.glob("*.zip"))
                if packages:
                    print("Available packages:")
                    for package in packages:
                        print(f"  - {package.name}")
                        # Try to get integrity info
                        integrity_file = package.with_suffix(".integrity.json")
                        if integrity_file.exists():
                            with open(integrity_file, "r") as f:
                                integrity_data = json.load(f)
                            size_mb = integrity_data["size_bytes"] / (1024 * 1024)
                            print(f"    Size: {size_mb:.1f} MB")
                            print(f"    Created: {integrity_data['created']}")
                else:
                    print("No packages found")
            else:
                print("Package directory not found")
        
        elif args.type == "backups":
            # List available backups
            version_manager = VersionManager(args.installation_dir)
            backup_dir = Path(args.installation_dir or ".") / "backups"
            if backup_dir.exists():
                backups = [d for d in backup_dir.iterdir() if d.is_dir()]
                if backups:
                    print("Available backups:")
                    for backup in backups:
                        print(f"  - {backup.name}")
                else:
                    print("No backups found")
            else:
                print("Backup directory not found")
        
        return 0
        
    except Exception as e:
        print(f"✗ Failed to list information: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="WAN2.2 Installation Package Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a new package
  python package_installer.py create --version 1.0.0 --name WAN22-Installer
  
  # Verify a package
  python package_installer.py verify --package dist/WAN22-Installer-v1.0.0.zip
  
  # Extract a package
  python package_installer.py extract --package dist/WAN22-Installer-v1.0.0.zip --output extracted/
  
  # Check version and updates
  python package_installer.py version --check-updates
  
  # Create a backup
  python package_installer.py backup --name pre-update-backup
  
  # List available packages
  python package_installer.py list packages
        """
    )
    
    # Global arguments
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Logging level")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Create package command
    create_parser = subparsers.add_parser("create", help="Create installation package")
    create_parser.add_argument("--version", required=True, help="Package version")
    create_parser.add_argument("--name", default="WAN22-Installer", help="Package name")
    create_parser.add_argument("--source-dir", help="Source directory (default: current)")
    create_parser.add_argument("--output-dir", help="Output directory (default: dist)")
    create_parser.add_argument("--verify", action="store_true", help="Verify package after creation")
    create_parser.set_defaults(func=create_package)
    
    # Verify package command
    verify_parser = subparsers.add_parser("verify", help="Verify package integrity")
    verify_parser.add_argument("--package", required=True, help="Package file to verify")
    verify_parser.set_defaults(func=verify_package)
    
    # Extract package command
    extract_parser = subparsers.add_parser("extract", help="Extract package")
    extract_parser.add_argument("--package", required=True, help="Package file to extract")
    extract_parser.add_argument("--output", required=True, help="Output directory")
    extract_parser.set_defaults(func=extract_package)
    
    # Version command
    version_parser = subparsers.add_parser("version", help="Check version information")
    version_parser.add_argument("--installation-dir", help="Installation directory")
    version_parser.add_argument("--check-updates", action="store_true", help="Check for updates")
    version_parser.set_defaults(func=check_version)
    
    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Create backup")
    backup_parser.add_argument("--installation-dir", help="Installation directory")
    backup_parser.add_argument("--name", help="Backup name (auto-generated if not provided)")
    backup_parser.set_defaults(func=create_backup)
    
    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore backup")
    restore_parser.add_argument("--installation-dir", help="Installation directory")
    restore_parser.add_argument("--name", required=True, help="Backup name to restore")
    restore_parser.set_defaults(func=restore_backup)
    
    # List command
    list_parser = subparsers.add_parser("list", help="List information")
    list_parser.add_argument("type", choices=["packages", "backups"], help="Type of information to list")
    list_parser.add_argument("--output-dir", help="Package output directory")
    list_parser.add_argument("--installation-dir", help="Installation directory")
    list_parser.set_defaults(func=list_info)
    
    # Parse arguments and execute
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())