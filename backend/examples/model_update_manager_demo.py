"""
Model Update Manager Demo
Demonstrates the functionality of the Model Update Management System
including version checking, update detection, safe updates, and rollback capabilities.
"""

import asyncio
import json
import logging
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the update manager
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.core.model_update_manager import (
    ModelUpdateManager, UpdateStatus, UpdatePriority, UpdateType,
    ModelVersion, UpdateInfo, UpdateProgress
)


class ModelUpdateManagerDemo:
    """Demo class for Model Update Manager functionality"""
    
    def __init__(self):
        self.temp_dir = None
        self.update_manager = None
    
    async def setup(self):
        """Setup demo environment"""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp(prefix="update_manager_demo_")
        logger.info(f"Demo setup in: {self.temp_dir}")
        
        # Initialize update manager
        self.update_manager = ModelUpdateManager(models_dir=self.temp_dir)
        
        # Add callbacks for demonstration
        self.update_manager.add_update_callback(self._update_progress_callback)
        self.update_manager.add_notification_callback(self._notification_callback)
        
        await self.update_manager.initialize()
        
        # Create demo model directories
        await self._create_demo_models()
    
    async def cleanup(self):
        """Cleanup demo environment"""
        if self.update_manager:
            await self.update_manager.shutdown()
        
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            logger.info("Demo cleanup completed")
    
    async def _create_demo_models(self):
        """Create demo model directories with version information"""
        models = {
            "t2v-A14B": {
                "version": "1.8.0",
                "files": ["config.json", "model_index.json", "unet/config.json"]
            },
            "i2v-A14B": {
                "version": "1.5.2",
                "files": ["config.json", "model_index.json", "vae/config.json"]
            },
            "ti2v-5B": {
                "version": "1.3.1",
                "files": ["config.json", "model_index.json", "text_encoder/config.json"]
            }
        }
        
        for model_id, info in models.items():
            model_dir = Path(self.temp_dir) / model_id
            model_dir.mkdir(parents=True)
            
            # Create version file
            version_file = model_dir / "version.json"
            with open(version_file, 'w') as f:
                json.dump({
                    "version": info["version"],
                    "created_at": datetime.now().isoformat()
                }, f, indent=2)
            
            # Create demo files
            for file_path in info["files"]:
                full_path = model_dir / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                
                if file_path.endswith(".json"):
                    with open(full_path, 'w') as f:
                        json.dump({
                            "model_type": model_id,
                            "version": info["version"],
                            "demo": True
                        }, f, indent=2)
                else:
                    full_path.write_text(f"Demo content for {file_path}")
        
        logger.info("Created demo models with versions")
    
    def _update_progress_callback(self, progress: UpdateProgress):
        """Callback for update progress"""
        logger.info(f"Update Progress - {progress.model_id}: "
                   f"{progress.current_step} ({progress.progress_percent:.1f}%)")
        
        if progress.status == UpdateStatus.COMPLETED:
            logger.info(f"✅ Update completed for {progress.model_id}")
        elif progress.status == UpdateStatus.FAILED:
            logger.error(f"❌ Update failed for {progress.model_id}: {progress.error_message}")
    
    def _notification_callback(self, update_info: UpdateInfo):
        """Callback for update notifications"""
        logger.info(f"🔔 Update available for {update_info.model_id}: "
                   f"{update_info.current_version} → {update_info.latest_version} "
                   f"({update_info.priority.value} priority)")
    
    async def demo_version_checking(self):
        """Demonstrate version checking functionality"""
        logger.info("\n" + "="*50)
        logger.info("DEMO: Version Checking")
        logger.info("="*50)
        
        # Check current versions
        models = ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
        
        for model_id in models:
            current_version = await self.update_manager._get_current_version(model_id)
            logger.info(f"Current version of {model_id}: {current_version}")
        
        # Test version comparison
        test_cases = [
            ("1.0.0", "1.0.1", "Patch update"),
            ("1.0.0", "1.1.0", "Minor update"),
            ("1.0.0", "2.0.0", "Major update"),
            ("local-123456", "1.0.0", "Local to release")
        ]
        
        logger.info("\nVersion comparison tests:")
        for current, latest, description in test_cases:
            is_update = self.update_manager._is_update_available(current, latest)
            update_type = self.update_manager._determine_update_type(current, latest)
            logger.info(f"  {current} → {latest}: {description} "
                       f"(Update available: {is_update}, Type: {update_type.value})")
    
    async def demo_update_detection(self):
        """Demonstrate update detection"""
        logger.info("\n" + "="*50)
        logger.info("DEMO: Update Detection")
        logger.info("="*50)
        
        # Check for updates
        available_updates = await self.update_manager.check_for_updates()
        
        if available_updates:
            logger.info(f"Found {len(available_updates)} available updates:")
            
            for model_id, update_info in available_updates.items():
                logger.info(f"\n📦 {model_id}:")
                logger.info(f"  Current: {update_info.current_version}")
                logger.info(f"  Latest: {update_info.latest_version}")
                logger.info(f"  Type: {update_info.update_type.value}")
                logger.info(f"  Priority: {update_info.priority.value}")
                logger.info(f"  Size: {update_info.size_mb:.1f} MB")
                logger.info(f"  Changelog:")
                for change in update_info.changelog[:3]:  # Show first 3 items
                    logger.info(f"    • {change}")
        else:
            logger.info("No updates available")
    
    async def demo_update_scheduling(self):
        """Demonstrate update scheduling"""
        logger.info("\n" + "="*50)
        logger.info("DEMO: Update Scheduling")
        logger.info("="*50)
        
        # Schedule updates for different times
        models_to_schedule = ["t2v-A14B", "i2v-A14B"]
        
        for i, model_id in enumerate(models_to_schedule):
            # Schedule for different times (demo purposes - very short intervals)
            scheduled_time = datetime.now() + timedelta(seconds=10 + i * 5)
            
            success = await self.update_manager.schedule_update(
                model_id, 
                scheduled_time, 
                auto_approve=True
            )
            
            if success:
                logger.info(f"📅 Scheduled update for {model_id} at {scheduled_time.strftime('%H:%M:%S')}")
            else:
                logger.error(f"❌ Failed to schedule update for {model_id}")
        
        # Show scheduled updates
        logger.info("\nCurrently scheduled updates:")
        for model_id, schedule in self.update_manager._scheduled_updates.items():
            logger.info(f"  {model_id}: {schedule.scheduled_time.strftime('%H:%M:%S')} "
                       f"(auto-approve: {schedule.auto_approve})")
    
    async def demo_backup_and_rollback(self):
        """Demonstrate backup and rollback functionality"""
        logger.info("\n" + "="*50)
        logger.info("DEMO: Backup and Rollback")
        logger.info("="*50)
        
        model_id = "t2v-A14B"
        model_dir = Path(self.temp_dir) / model_id
        
        # Show original content
        version_file = model_dir / "version.json"
        with open(version_file, 'r') as f:
            original_version = json.load(f)
        
        logger.info(f"Original version: {original_version['version']}")
        
        # Create backup
        logger.info("Creating backup...")
        backup_path = await self.update_manager._create_backup(model_id)
        
        if backup_path:
            logger.info(f"✅ Backup created at: {Path(backup_path).name}")
            
            # Simulate model update (modify version)
            logger.info("Simulating model update...")
            with open(version_file, 'w') as f:
                json.dump({
                    "version": "2.0.0",
                    "updated_at": datetime.now().isoformat(),
                    "demo": "updated_version"
                }, f, indent=2)
            
            # Show updated content
            with open(version_file, 'r') as f:
                updated_version = json.load(f)
            logger.info(f"Updated version: {updated_version['version']}")
            
            # Get rollback info
            rollback_options = await self.update_manager.get_rollback_info(model_id)
            logger.info(f"Available rollback options: {len(rollback_options)}")
            
            if rollback_options:
                rollback_info = rollback_options[0]
                logger.info(f"  Backup date: {rollback_info.backup_date}")
                logger.info(f"  Backup size: {rollback_info.backup_size_mb:.1f} MB")
                
                # Perform rollback
                logger.info("Performing rollback...")
                rollback_success = await self.update_manager.perform_rollback(
                    model_id, rollback_info.backup_path
                )
                
                if rollback_success:
                    logger.info("✅ Rollback completed successfully")
                    
                    # Verify rollback
                    with open(version_file, 'r') as f:
                        restored_version = json.load(f)
                    logger.info(f"Restored version: {restored_version['version']}")
                    
                    if restored_version['version'] == original_version['version']:
                        logger.info("✅ Rollback verification passed")
                    else:
                        logger.error("❌ Rollback verification failed")
                else:
                    logger.error("❌ Rollback failed")
        else:
            logger.error("❌ Failed to create backup")
    
    async def demo_progress_tracking(self):
        """Demonstrate update progress tracking"""
        logger.info("\n" + "="*50)
        logger.info("DEMO: Progress Tracking")
        logger.info("="*50)
        
        model_id = "demo_progress_model"
        
        # Simulate update progress
        steps = [
            ("Preparing update", 0),
            ("Creating backup", 16.7),
            ("Downloading update", 33.3),
            ("Validating download", 50.0),
            ("Installing update", 66.7),
            ("Validating installation", 83.3),
            ("Finalizing update", 100.0)
        ]
        
        logger.info("Simulating update progress:")
        
        for i, (step_name, progress_percent) in enumerate(steps):
            progress = UpdateProgress(
                model_id=model_id,
                status=UpdateStatus.DOWNLOADING if i < 6 else UpdateStatus.COMPLETED,
                progress_percent=progress_percent,
                current_step=step_name,
                total_steps=len(steps),
                current_step_number=i + 1,
                started_at=datetime.now()
            )
            
            # Add to active updates temporarily
            async with self.update_manager._update_lock:
                self.update_manager._active_updates[model_id] = progress
            
            # Notify callbacks (will trigger our demo callback)
            await self.update_manager._notify_update_callbacks(progress)
            
            # Simulate processing time
            await asyncio.sleep(0.5)
        
        # Clean up
        async with self.update_manager._update_lock:
            if model_id in self.update_manager._active_updates:
                del self.update_manager._active_updates[model_id]
    
    async def demo_error_handling(self):
        """Demonstrate error handling scenarios"""
        logger.info("\n" + "="*50)
        logger.info("DEMO: Error Handling")
        logger.info("="*50)
        
        # Test various error scenarios
        error_scenarios = [
            {
                "name": "Invalid model ID",
                "model_id": "nonexistent_model",
                "description": "Attempting to update non-existent model"
            },
            {
                "name": "Missing backup directory",
                "model_id": "t2v-A14B",
                "backup_path": "/nonexistent/backup/path",
                "description": "Rollback with invalid backup path"
            }
        ]
        
        for scenario in error_scenarios:
            logger.info(f"\n🧪 Testing: {scenario['name']}")
            logger.info(f"Description: {scenario['description']}")
            
            try:
                if "backup_path" in scenario:
                    # Test rollback error
                    result = await self.update_manager.perform_rollback(
                        scenario["model_id"], scenario["backup_path"]
                    )
                    logger.info(f"Rollback result: {result}")
                else:
                    # Test update error
                    current_version = await self.update_manager._get_current_version(
                        scenario["model_id"]
                    )
                    logger.info(f"Version check result: {current_version}")
                
            except Exception as e:
                logger.info(f"Expected error caught: {type(e).__name__}: {e}")
    
    async def demo_cleanup_operations(self):
        """Demonstrate cleanup operations"""
        logger.info("\n" + "="*50)
        logger.info("DEMO: Cleanup Operations")
        logger.info("="*50)
        
        # Show current backup status
        backup_dirs = list(self.update_manager.backups_dir.glob("*"))
        logger.info(f"Current backups: {len(backup_dirs)}")
        
        for backup_dir in backup_dirs:
            if backup_dir.is_dir():
                logger.info(f"  📁 {backup_dir.name}")
        
        # Demonstrate cleanup (with very short retention for demo)
        original_retention = self.update_manager.backup_retention_days
        self.update_manager.backup_retention_days = 0  # Clean up everything for demo
        
        logger.info(f"\nRunning cleanup (retention: {self.update_manager.backup_retention_days} days)...")
        await self.update_manager.cleanup_old_backups()
        
        # Check results
        remaining_backups = list(self.update_manager.backups_dir.glob("*"))
        logger.info(f"Remaining backups after cleanup: {len(remaining_backups)}")
        
        # Restore original retention
        self.update_manager.backup_retention_days = original_retention
    
    async def run_all_demos(self):
        """Run all demo scenarios"""
        logger.info("🚀 Starting Model Update Manager Demo")
        logger.info("="*60)
        
        try:
            await self.setup()
            
            # Run individual demos
            await self.demo_version_checking()
            await self.demo_update_detection()
            await self.demo_update_scheduling()
            await self.demo_backup_and_rollback()
            await self.demo_progress_tracking()
            await self.demo_error_handling()
            await self.demo_cleanup_operations()
            
            logger.info("\n" + "="*60)
            logger.info("✅ All demos completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            raise
        
        finally:
            await self.cleanup()


async def main():
    """Main demo function"""
    demo = ModelUpdateManagerDemo()
    await demo.run_all_demos()


if __name__ == "__main__":
    asyncio.run(main())