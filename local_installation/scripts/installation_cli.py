"""
Installation CLI Utility
Command-line interface for managing installation state, snapshots, and logs.
"""

import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from installation_flow_controller import InstallationFlowController
from rollback_manager import RollbackManager
from logging_system import setup_installation_logging


def list_snapshots(installation_path: str) -> None:
    """List all available snapshots."""
    rollback_manager = RollbackManager(installation_path)
    snapshots = rollback_manager.list_snapshots()
    
    if not snapshots:
        print("No snapshots found.")
        return
    
    print(f"Found {len(snapshots)} snapshots:")
    print("-" * 80)
    
    for snapshot in snapshots:
        info = rollback_manager.get_snapshot_info(snapshot.id)
        print(f"ID: {snapshot.id}")
        print(f"Timestamp: {snapshot.timestamp}")
        print(f"Phase: {snapshot.phase}")
        print(f"Description: {snapshot.description}")
        if info:
            print(f"Size: {info['size_mb']} MB")
            print(f"Files: {info['files_count']}, Directories: {info['directories_count']}")
        print("-" * 80)


def restore_snapshot(installation_path: str, snapshot_id: str) -> None:
    """Restore from a specific snapshot."""
    rollback_manager = RollbackManager(installation_path)
    
    print(f"Restoring snapshot: {snapshot_id}")
    success = rollback_manager.restore_snapshot(snapshot_id)
    
    if success:
        print("✅ Snapshot restored successfully!")
    else:
        print("❌ Failed to restore snapshot.")
        sys.exit(1)


def delete_snapshot(installation_path: str, snapshot_id: str) -> None:
    """Delete a specific snapshot."""
    rollback_manager = RollbackManager(installation_path)
    
    print(f"Deleting snapshot: {snapshot_id}")
    success = rollback_manager.delete_snapshot(snapshot_id)
    
    if success:
        print("✅ Snapshot deleted successfully!")
    else:
        print("❌ Failed to delete snapshot.")
        sys.exit(1)


def cleanup_snapshots(installation_path: str, keep_count: int, keep_days: int) -> None:
    """Clean up old snapshots."""
    rollback_manager = RollbackManager(installation_path)
    
    print(f"Cleaning up snapshots (keep {keep_count} recent, {keep_days} days)...")
    deleted_count = rollback_manager.cleanup_old_snapshots(keep_count, keep_days)
    
    print(f"✅ Cleaned up {deleted_count} old snapshots.")


def show_installation_status(installation_path: str) -> None:
    """Show current installation status."""
    flow_controller = InstallationFlowController(installation_path, dry_run=True)
    
    # Load current state
    state = flow_controller.load_state()
    
    print("Installation Status:")
    print("=" * 50)
    
    if state:
        print(f"Phase: {state.phase.value}")
        print(f"Progress: {state.progress:.1%}")
        print(f"Current Task: {state.current_task}")
        print(f"Errors: {len(state.errors)}")
        print(f"Warnings: {len(state.warnings)}")
        
        if state.errors:
            print("\nRecent Errors:")
            for error in state.errors[-3:]:  # Show last 3 errors
                print(f"  • {error}")
        
        if state.warnings:
            print("\nRecent Warnings:")
            for warning in state.warnings[-3:]:  # Show last 3 warnings
                print(f"  • {warning}")
    else:
        print("No installation state found.")
    
    # Show summary
    summary = flow_controller.get_installation_summary()
    print(f"\nSnapshots: {summary['snapshots_count']}")
    
    if 'logging_summary' in summary:
        log_summary = summary['logging_summary']
        print(f"Log Files: {len(log_summary['files'])}")
        print(f"Total Log Size: {log_summary['statistics']['total_size'] / 1024 / 1024:.1f} MB")
        print(f"Total Errors: {log_summary['statistics']['error_count']}")
        print(f"Total Warnings: {log_summary['statistics']['warning_count']}")


def validate_installation(installation_path: str) -> None:
    """Validate installation integrity."""
    flow_controller = InstallationFlowController(installation_path, dry_run=True)
    
    print("Validating installation integrity...")
    validation_result = flow_controller.validate_installation_integrity()
    
    if validation_result['valid']:
        print("✅ Installation appears to be valid.")
    else:
        print("❌ Installation validation failed.")
        
        if validation_result['issues']:
            print("\nIssues found:")
            for issue in validation_result['issues']:
                print(f"  • {issue}")
        
        if validation_result['recommendations']:
            print("\nRecommendations:")
            for rec in validation_result['recommendations']:
                print(f"  • {rec}")


def export_logs(installation_path: str, output_path: str) -> None:
    """Export logs for support purposes."""
    logging_system = setup_installation_logging(installation_path, enable_console=False)
    
    print(f"Exporting logs to: {output_path}")
    success = logging_system.export_logs(output_path, include_structured=True)
    
    if success:
        print("✅ Logs exported successfully!")
    else:
        print("❌ Failed to export logs.")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="WAN2.2 Installation Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--installation-path",
        default=".",
        help="Path to installation directory (default: current directory)"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show installation status')
    
    # Snapshots commands
    snapshots_parser = subparsers.add_parser('snapshots', help='Manage snapshots')
    snapshots_subparsers = snapshots_parser.add_subparsers(dest='snapshot_action')
    
    list_parser = snapshots_subparsers.add_parser('list', help='List all snapshots')
    
    restore_parser = snapshots_subparsers.add_parser('restore', help='Restore from snapshot')
    restore_parser.add_argument('snapshot_id', help='Snapshot ID to restore')
    
    delete_parser = snapshots_subparsers.add_parser('delete', help='Delete a snapshot')
    delete_parser.add_argument('snapshot_id', help='Snapshot ID to delete')
    
    cleanup_parser = snapshots_subparsers.add_parser('cleanup', help='Clean up old snapshots')
    cleanup_parser.add_argument('--keep-count', type=int, default=5, help='Number of snapshots to keep')
    cleanup_parser.add_argument('--keep-days', type=int, default=30, help='Days to keep snapshots')
    
    # Validation command
    validate_parser = subparsers.add_parser('validate', help='Validate installation integrity')
    
    # Logs command
    logs_parser = subparsers.add_parser('logs', help='Manage logs')
    logs_subparsers = logs_parser.add_subparsers(dest='logs_action')
    
    export_parser = logs_subparsers.add_parser('export', help='Export logs')
    export_parser.add_argument('output_path', help='Output path for exported logs')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    installation_path = Path(args.installation_path).resolve()
    
    try:
        if args.command == 'status':
            show_installation_status(str(installation_path))
        
        elif args.command == 'snapshots':
            if args.snapshot_action == 'list':
                list_snapshots(str(installation_path))
            elif args.snapshot_action == 'restore':
                restore_snapshot(str(installation_path), args.snapshot_id)
            elif args.snapshot_action == 'delete':
                delete_snapshot(str(installation_path), args.snapshot_id)
            elif args.snapshot_action == 'cleanup':
                cleanup_snapshots(str(installation_path), args.keep_count, args.keep_days)
            else:
                snapshots_parser.print_help()
        
        elif args.command == 'validate':
            validate_installation(str(installation_path))
        
        elif args.command == 'logs':
            if args.logs_action == 'export':
                export_logs(str(installation_path), args.output_path)
            else:
                logs_parser.print_help()
        
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()