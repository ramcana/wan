"""
Demo script showing integration between VersionManager and RollbackManager
Demonstrates version management, updates, and recovery procedures.
"""

import os
import sys
import tempfile
import shutil
import json
import logging
from pathlib import Path

# Add the scripts directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from version_manager import VersionManager
from rollback_manager import RollbackManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_version_management_and_rollback():
    """Demonstrate complete version management and rollback workflow."""
    
    # Create temporary installation directory
    install_dir = tempfile.mkdtemp()
    logger.info(f"Demo installation directory: {install_dir}")
    
    try:
        # Initialize managers
        version_manager = VersionManager(install_dir, dry_run=False)
        rollback_manager = RollbackManager(install_dir, dry_run=False)
        
        logger.info("=== Phase 1: Initial Setup ===")
        
        # Create initial installation files
        config_file = Path(install_dir) / "config.json"
        config_file.write_text(json.dumps({
            "version": "1.0.0",
            "settings": {
                "debug": False,
                "max_threads": 4,
                "gpu_enabled": True
            }
        }, indent=2))
        
        scripts_dir = Path(install_dir) / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "main.py").write_text("# Main application v1.0.0")
        (scripts_dir / "utils.py").write_text("# Utilities v1.0.0")
        
        models_dir = Path(install_dir) / "models"
        models_dir.mkdir()
        (models_dir / "model_v1.bin").write_text("Model data v1.0.0")
        
        logger.info("Initial installation files created")
        
        # Create initial recovery point
        initial_snapshot = rollback_manager.create_recovery_point("Initial installation v1.0.0")
        logger.info(f"Created initial recovery point: {initial_snapshot}")
        
        logger.info("=== Phase 2: Version Check and Update Preparation ===")
        
        # Check current version
        current_version = version_manager.get_current_version()
        logger.info(f"Current version: {current_version}")
        
        # Simulate checking for updates (this would normally contact GitHub)
        logger.info("Checking for updates...")
        # In a real scenario, this would use check_for_updates()
        # For demo, we'll simulate an available update
        
        # Create pre-update backup
        pre_update_snapshot = version_manager.backup_current_installation()
        logger.info(f"Created pre-update backup: {pre_update_snapshot}")
        
        logger.info("=== Phase 3: Simulate Update Process ===")
        
        # Simulate updating files to v1.1.0
        logger.info("Updating to v1.1.0...")
        
        config_data = json.loads(config_file.read_text())
        config_data["version"] = "1.1.0"
        config_data["settings"]["max_threads"] = 8  # Improved performance
        config_file.write_text(json.dumps(config_data, indent=2))
        
        (scripts_dir / "main.py").write_text("# Main application v1.1.0 - Enhanced features")
        (scripts_dir / "new_feature.py").write_text("# New feature added in v1.1.0")
        
        # Update version info
        version_manager.update_version_info("1.1.0")
        logger.info("Updated to v1.1.0 successfully")
        
        logger.info("=== Phase 4: Simulate Installation Failure During Next Update ===")
        
        # Create pre-update backup for v1.2.0 update
        pre_v12_snapshot = rollback_manager.create_snapshot(
            "Pre-update backup (v1.1.0 -> v1.2.0)",
            "pre-update",
            files_to_backup=["config.json"],
            dirs_to_backup=["scripts", "models"]
        )
        logger.info(f"Created pre-v1.2.0 backup: {pre_v12_snapshot}")
        
        # Simulate failed update to v1.2.0
        logger.info("Attempting update to v1.2.0...")
        
        # Corrupt configuration during update
        config_file.write_text('{"corrupted": "config", "version": "1.2.0-broken"}')
        
        # Delete some files
        (scripts_dir / "utils.py").unlink()
        
        # Create temporary artifacts from failed installation
        temp_dir = Path(install_dir) / "temp_download"
        temp_dir.mkdir()
        (temp_dir / "model_v12.partial").write_text("Incomplete model download")
        (Path(install_dir) / "installation.tmp").write_text("Temporary installation data")
        
        logger.error("Update to v1.2.0 failed! Configuration corrupted and files missing.")
        
        logger.info("=== Phase 5: Recovery Process ===")
        
        # Create emergency backup of current (broken) state
        emergency_snapshot = rollback_manager.create_emergency_backup()
        logger.info(f"Created emergency backup: {emergency_snapshot}")
        
        # Get recovery recommendations
        recommendations = rollback_manager.get_recovery_recommendations()
        logger.info("Recovery recommendations:")
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"  {i}. [{rec['priority'].upper()}] {rec['message']}")
            logger.info(f"     Action: {rec['action']}")
        
        # Attempt recovery
        logger.info("Attempting recovery from failed installation...")
        recovery_success = rollback_manager.recover_from_failed_installation("models")
        
        if recovery_success:
            logger.info("Recovery successful!")
            
            # Verify recovery
            try:
                with open(config_file, 'r') as f:
                    recovered_config = json.load(f)
                logger.info(f"Recovered to version: {recovered_config['version']}")
                if 'settings' in recovered_config and 'max_threads' in recovered_config['settings']:
                    logger.info(f"Max threads setting: {recovered_config['settings']['max_threads']}")
                else:
                    logger.warning("Configuration structure may still be corrupted")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Configuration may still be corrupted: {e}")
            
            # Check if files were restored
            if (scripts_dir / "utils.py").exists():
                logger.info("utils.py was successfully restored")
            
            # Clean up failed installation artifacts
            cleanup_success = rollback_manager.cleanup_failed_installation_artifacts()
            if cleanup_success:
                logger.info("Cleaned up failed installation artifacts")
                
                # Verify cleanup
                if not temp_dir.exists():
                    logger.info("Temporary download directory was cleaned up")
                if not (Path(install_dir) / "installation.tmp").exists():
                    logger.info("Temporary installation file was cleaned up")
        else:
            logger.error("Recovery failed!")
        
        logger.info("=== Phase 6: System Status ===")
        
        # Show final version
        final_version = version_manager.get_current_version()
        logger.info(f"Final system version: {final_version}")
        
        # Show version history
        history = version_manager.get_version_history()
        logger.info("Version history:")
        for entry in history:
            if entry.get('current'):
                logger.info(f"  * {entry['version']} (current)")
            else:
                logger.info(f"    {entry.get('version', 'unknown')} - {entry.get('description', 'N/A')}")
        
        # Show available snapshots
        snapshots = rollback_manager.list_snapshots()
        logger.info(f"Available snapshots: {len(snapshots)}")
        for snapshot in snapshots[:5]:  # Show first 5
            logger.info(f"  - {snapshot.id}: {snapshot.description} ({snapshot.phase})")
        
        # Show final recommendations
        final_recommendations = rollback_manager.get_recovery_recommendations()
        if final_recommendations:
            logger.info("Current system recommendations:")
            for rec in final_recommendations:
                if rec['priority'] in ['high', 'medium']:
                    logger.info(f"  - [{rec['priority'].upper()}] {rec['message']}")
        
        logger.info("=== Demo Complete ===")
        logger.info("The system has successfully demonstrated:")
        logger.info("  ✓ Version management and tracking")
        logger.info("  ✓ Pre-update backup creation")
        logger.info("  ✓ Recovery from failed installation")
        logger.info("  ✓ Cleanup of installation artifacts")
        logger.info("  ✓ Recovery recommendations")
        logger.info("  ✓ Snapshot management and validation")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    finally:
        # Clean up
        logger.info(f"Cleaning up demo directory: {install_dir}")
        shutil.rmtree(install_dir, ignore_errors=True)


if __name__ == "__main__":
    demo_version_management_and_rollback()