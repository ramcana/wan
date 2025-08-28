"""
Unit tests for Model Update Manager
Tests model version checking, update detection, safe update processes,
rollback capability, and update scheduling functionality.
"""

import asyncio
import json
import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Configure pytest for async tests
pytest_plugins = ('pytest_asyncio',)

# Import the module under test
import sys
sys.path.append(str(Path(__file__).parent.parent))

from backend.core.model_update_manager import (
    ModelUpdateManager, UpdateStatus, UpdatePriority, UpdateType,
    ModelVersion, UpdateInfo, UpdateProgress, UpdateResult,
    UpdateSchedule, RollbackInfo
)


@pytest.mark.asyncio
class TestModelUpdateManager:
    """Test suite for ModelUpdateManager"""
    
    @pytest.fixture
    def temp_models_dir(self):
        """Create temporary models directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    async def update_manager(self, temp_models_dir):
        """Create ModelUpdateManager instance for testing"""
        manager = ModelUpdateManager(models_dir=temp_models_dir)
        await manager.initialize()
        yield manager
        await manager.shutdown()
    
    @pytest.fixture
    def mock_downloader(self):
        """Create mock enhanced downloader"""
        downloader = Mock()
        downloader.download_with_retry = AsyncMock()
        downloader.add_progress_callback = Mock()
        downloader.remove_progress_callback = Mock()
        downloader.cancel_download = AsyncMock()
        return downloader
    
    @pytest.fixture
    def mock_health_monitor(self):
        """Create mock health monitor"""
        monitor = Mock()
        monitor.check_model_integrity = AsyncMock()
        return monitor
    
    async def test_initialization(self, temp_models_dir):
        """Test update manager initialization"""
        manager = ModelUpdateManager(models_dir=temp_models_dir)
        
        assert manager.models_dir == Path(temp_models_dir)
        assert manager.updates_dir.exists()
        assert manager.backups_dir.exists()
        assert manager.temp_dir.exists()
        
        # Test initialization
        result = await manager.initialize()
        assert result is True
        
        await manager.shutdown()
    
    async def test_version_parsing(self, update_manager):
        """Test semantic version parsing"""
        # Test normal versions
        assert update_manager._parse_version("1.2.3") == (1, 2, 3)
        assert update_manager._parse_version("2.0.0") == (2, 0, 0)
        assert update_manager._parse_version("1.5") == (1, 5, 0)
        
        # Test versions with suffixes
        assert update_manager._parse_version("1.2.3-beta") == (1, 2, 3)
        assert update_manager._parse_version("2.1.0+build.123") == (2, 1, 0)
        
        # Test local versions
        assert update_manager._parse_version("local-123456") == (0, 0, 0)
    
    async def test_update_type_determination(self, update_manager):
        """Test update type determination"""
        # Major update
        update_type = update_manager._determine_update_type("1.0.0", "2.0.0")
        assert update_type == UpdateType.MAJOR
        
        # Minor update
        update_type = update_manager._determine_update_type("1.0.0", "1.1.0")
        assert update_type == UpdateType.MINOR
        
        # Patch update
        update_type = update_manager._determine_update_type("1.0.0", "1.0.1")
        assert update_type == UpdateType.PATCH
    
    async def test_update_priority_determination(self, update_manager):
        """Test update priority determination"""
        # Critical priority
        version_info = ModelVersion(
            version="1.1.0",
            release_date=datetime.now(),
            size_mb=100.0,
            checksum="test",
            download_url="test",
            changelog=["Critical security fix", "Bug fixes"]
        )
        priority = update_manager._determine_update_priority(version_info)
        assert priority == UpdatePriority.CRITICAL
        
        # High priority
        version_info.changelog = ["Performance improvements", "Memory optimization"]
        priority = update_manager._determine_update_priority(version_info)
        assert priority == UpdatePriority.HIGH
        
        # Medium priority
        version_info.changelog = ["New features", "UI improvements"]
        priority = update_manager._determine_update_priority(version_info)
        assert priority == UpdatePriority.MEDIUM
        
        # Low priority
        version_info.changelog = ["Minor tweaks", "Documentation updates"]
        priority = update_manager._determine_update_priority(version_info)
        assert priority == UpdatePriority.LOW
    
    async def test_check_for_updates(self, update_manager, temp_models_dir):
        """Test checking for available updates"""
        # Create a mock model directory
        model_dir = Path(temp_models_dir) / "t2v-A14B"
        model_dir.mkdir()
        
        # Create version file with old version
        version_file = model_dir / "version.json"
        with open(version_file, 'w') as f:
            json.dump({"version": "1.0.0"}, f)
        
        # Mock the fetch latest version method
        with patch.object(update_manager, '_fetch_latest_version') as mock_fetch:
            mock_version = ModelVersion(
                version="2.0.0",
                release_date=datetime.now(),
                size_mb=8500.0,
                checksum="test_checksum",
                download_url="https://test.com/model.tar.gz",
                changelog=["Major improvements", "Bug fixes"]
            )
            mock_fetch.return_value = mock_version
            
            updates = await update_manager.check_for_updates("t2v-A14B")
            
            assert "t2v-A14B" in updates
            update_info = updates["t2v-A14B"]
            assert update_info.current_version == "1.0.0"
            assert update_info.latest_version == "2.0.0"
            assert update_info.update_type == UpdateType.MAJOR
    
    async def test_schedule_update(self, update_manager):
        """Test update scheduling"""
        model_id = "t2v-A14B"
        scheduled_time = datetime.now() + timedelta(hours=1)
        
        result = await update_manager.schedule_update(
            model_id, scheduled_time, auto_approve=True
        )
        
        assert result is True
        assert model_id in update_manager._scheduled_updates
        
        schedule = update_manager._scheduled_updates[model_id]
        assert schedule.model_id == model_id
        assert schedule.scheduled_time == scheduled_time
        assert schedule.auto_approve is True
    
    async def test_create_backup(self, update_manager, temp_models_dir):
        """Test model backup creation"""
        # Create a mock model directory with files
        model_dir = Path(temp_models_dir) / "test_model"
        model_dir.mkdir()
        
        # Create some test files
        (model_dir / "config.json").write_text('{"test": "data"}')
        (model_dir / "model.bin").write_bytes(b"fake model data")
        
        # Create backup
        backup_path = await update_manager._create_backup("test_model")
        
        assert backup_path is not None
        backup_dir = Path(backup_path)
        assert backup_dir.exists()
        assert (backup_dir / "config.json").exists()
        assert (backup_dir / "model.bin").exists()
        assert (backup_dir / "backup_info.json").exists()
        
        # Verify backup info
        with open(backup_dir / "backup_info.json", 'r') as f:
            backup_info = json.load(f)
        
        assert backup_info["model_id"] == "test_model"
        assert "backup_date" in backup_info
        assert "backup_size_mb" in backup_info
    
    async def test_download_validation(self, update_manager, temp_models_dir):
        """Test download validation with checksums"""
        # Create a test file
        test_file = Path(temp_models_dir) / "test_download.txt"
        test_content = b"test content for checksum validation"
        test_file.write_bytes(test_content)
        
        # Calculate expected checksum
        import hashlib
        expected_checksum = hashlib.sha256(test_content).hexdigest()
        
        # Test valid checksum
        result = await update_manager._validate_download(str(test_file), expected_checksum)
        assert result is True
        
        # Test invalid checksum
        result = await update_manager._validate_download(str(test_file), "invalid_checksum")
        assert result is False
    
    async def test_rollback_functionality(self, update_manager, temp_models_dir):
        """Test rollback to previous version"""
        # Create original model directory
        model_dir = Path(temp_models_dir) / "test_model"
        model_dir.mkdir()
        (model_dir / "original_file.txt").write_text("original content")
        
        # Create backup
        backup_path = await update_manager._create_backup("test_model")
        assert backup_path is not None
        
        # Simulate failed update by modifying model directory
        (model_dir / "original_file.txt").write_text("corrupted content")
        (model_dir / "new_file.txt").write_text("new content")
        
        # Perform rollback
        result = await update_manager._perform_rollback("test_model", backup_path)
        assert result is True
        
        # Verify rollback
        assert (model_dir / "original_file.txt").read_text() == "original content"
        assert not (model_dir / "new_file.txt").exists()
    
    async def test_get_rollback_info(self, update_manager, temp_models_dir):
        """Test getting rollback information"""
        # Create model and backup
        model_dir = Path(temp_models_dir) / "test_model"
        model_dir.mkdir()
        (model_dir / "test_file.txt").write_text("test content")
        
        backup_path = await update_manager._create_backup("test_model")
        assert backup_path is not None
        
        # Get rollback info
        rollback_options = await update_manager.get_rollback_info("test_model")
        
        assert len(rollback_options) == 1
        rollback_info = rollback_options[0]
        assert rollback_info.model_id == "test_model"
        assert rollback_info.backup_path == backup_path
        assert rollback_info.is_valid is True
        assert rollback_info.can_restore is True
    
    async def test_update_progress_tracking(self, update_manager):
        """Test update progress tracking"""
        model_id = "test_model"
        
        # Create progress object
        progress = UpdateProgress(
            model_id=model_id,
            status=UpdateStatus.DOWNLOADING,
            progress_percent=50.0,
            current_step="Downloading update",
            total_steps=5,
            current_step_number=2
        )
        
        # Add to active updates
        async with update_manager._update_lock:
            update_manager._active_updates[model_id] = progress
        
        # Test getting progress
        retrieved_progress = await update_manager.get_update_progress(model_id)
        assert retrieved_progress is not None
        assert retrieved_progress.model_id == model_id
        assert retrieved_progress.status == UpdateStatus.DOWNLOADING
        assert retrieved_progress.progress_percent == 50.0
    
    async def test_update_callbacks(self, update_manager):
        """Test update progress callbacks"""
        callback_called = False
        callback_progress = None
        
        def test_callback(progress):
            nonlocal callback_called, callback_progress
            callback_called = True
            callback_progress = progress
        
        # Add callback
        update_manager.add_update_callback(test_callback)
        
        # Create and notify progress
        progress = UpdateProgress(
            model_id="test_model",
            status=UpdateStatus.DOWNLOADING,
            progress_percent=25.0,
            current_step="Test step",
            total_steps=4,
            current_step_number=1
        )
        
        await update_manager._notify_update_callbacks(progress)
        
        assert callback_called is True
        assert callback_progress is not None
        assert callback_progress.model_id == "test_model"
        assert callback_progress.progress_percent == 25.0
    
    async def test_notification_callbacks(self, update_manager):
        """Test update notification callbacks"""
        callback_called = False
        callback_update_info = None
        
        def test_callback(update_info):
            nonlocal callback_called, callback_update_info
            callback_called = True
            callback_update_info = update_info
        
        # Add callback
        update_manager.add_notification_callback(test_callback)
        
        # Create and notify update info
        update_info = UpdateInfo(
            model_id="test_model",
            current_version="1.0.0",
            latest_version="2.0.0",
            update_type=UpdateType.MAJOR,
            priority=UpdatePriority.HIGH,
            size_mb=1000.0
        )
        
        await update_manager._notify_notification_callbacks(update_info)
        
        assert callback_called is True
        assert callback_update_info is not None
        assert callback_update_info.model_id == "test_model"
        assert callback_update_info.latest_version == "2.0.0"
    
    async def test_cancel_update(self, update_manager, mock_downloader):
        """Test cancelling an active update"""
        update_manager.downloader = mock_downloader
        model_id = "test_model"
        
        # Create active update
        progress = UpdateProgress(
            model_id=model_id,
            status=UpdateStatus.DOWNLOADING,
            progress_percent=30.0,
            current_step="Downloading",
            total_steps=5,
            current_step_number=2,
            can_cancel=True
        )
        
        async with update_manager._update_lock:
            update_manager._active_updates[model_id] = progress
        
        # Cancel update
        result = await update_manager.cancel_update(model_id)
        
        assert result is True
        mock_downloader.cancel_download.assert_called_once_with(model_id)
        
        # Check status updated
        updated_progress = await update_manager.get_update_progress(model_id)
        assert updated_progress.status == UpdateStatus.CANCELLED
    
    async def test_cleanup_old_backups(self, update_manager, temp_models_dir):
        """Test cleanup of old backups"""
        # Set short retention period for testing
        update_manager.backup_retention_days = 1
        
        # Create old backup
        old_backup_dir = update_manager.backups_dir / "test_model_backup_old"
        old_backup_dir.mkdir()
        
        old_date = datetime.now() - timedelta(days=2)
        backup_info = {
            "model_id": "test_model",
            "backup_date": old_date.isoformat(),
            "backup_size_mb": 100.0
        }
        
        with open(old_backup_dir / "backup_info.json", 'w') as f:
            json.dump(backup_info, f)
        
        # Create recent backup
        recent_backup_dir = update_manager.backups_dir / "test_model_backup_recent"
        recent_backup_dir.mkdir()
        
        recent_date = datetime.now() - timedelta(hours=12)
        recent_backup_info = {
            "model_id": "test_model",
            "backup_date": recent_date.isoformat(),
            "backup_size_mb": 100.0
        }
        
        with open(recent_backup_dir / "backup_info.json", 'w') as f:
            json.dump(recent_backup_info, f)
        
        # Run cleanup
        await update_manager.cleanup_old_backups()
        
        # Check results
        assert not old_backup_dir.exists()  # Should be removed
        assert recent_backup_dir.exists()   # Should be kept
    
    async def test_version_cache_persistence(self, update_manager):
        """Test version cache loading and saving"""
        # Add version to cache
        version = ModelVersion(
            version="1.5.0",
            release_date=datetime.now(),
            size_mb=5000.0,
            checksum="test_checksum",
            download_url="https://test.com/model.tar.gz",
            changelog=["Test changelog"]
        )
        
        update_manager.version_cache["test_model"] = version
        
        # Save cache
        await update_manager._save_version_cache()
        
        # Clear cache and reload
        update_manager.version_cache.clear()
        await update_manager._load_version_cache()
        
        # Verify loaded data
        assert "test_model" in update_manager.version_cache
        loaded_version = update_manager.version_cache["test_model"]
        assert loaded_version.version == "1.5.0"
        assert loaded_version.size_mb == 5000.0
        assert loaded_version.checksum == "test_checksum"
    
    async def test_scheduled_updates_persistence(self, update_manager):
        """Test scheduled updates loading and saving"""
        # Add scheduled update
        scheduled_time = datetime.now() + timedelta(hours=2)
        schedule = UpdateSchedule(
            model_id="test_model",
            scheduled_time=scheduled_time,
            auto_approve=True
        )
        
        update_manager._scheduled_updates["test_model"] = schedule
        
        # Save schedules
        await update_manager._save_scheduled_updates()
        
        # Clear and reload
        update_manager._scheduled_updates.clear()
        await update_manager._load_scheduled_updates()
        
        # Verify loaded data
        assert "test_model" in update_manager._scheduled_updates
        loaded_schedule = update_manager._scheduled_updates["test_model"]
        assert loaded_schedule.model_id == "test_model"
        assert loaded_schedule.auto_approve is True
        # Note: datetime comparison might have slight differences due to serialization
        assert abs((loaded_schedule.scheduled_time - scheduled_time).total_seconds()) < 1
    
    async def test_error_handling_in_update_process(self, update_manager, temp_models_dir):
        """Test error handling during update process"""
        model_id = "test_model"
        
        # Create update info
        update_info = UpdateInfo(
            model_id=model_id,
            current_version="1.0.0",
            latest_version="2.0.0",
            update_type=UpdateType.MAJOR,
            priority=UpdatePriority.HIGH,
            size_mb=1000.0,
            checksum="test_checksum",
            download_url="https://invalid-url.com/model.tar.gz"
        )
        
        update_manager.update_cache[model_id] = update_info
        
        # Mock backup creation to succeed
        with patch.object(update_manager, '_create_backup') as mock_backup:
            mock_backup.return_value = "/fake/backup/path"
            
            # Mock download to fail
            with patch.object(update_manager, '_download_update') as mock_download:
                mock_download.return_value = None  # Simulate download failure
                
                # Perform update
                result = await update_manager.perform_update(model_id, user_approved=True)
                
                # Verify failure handling
                assert result.success is False
                assert result.final_status == UpdateStatus.FAILED
                assert "Failed to download update" in result.error_message
    
    async def test_integration_with_health_monitor(self, update_manager, mock_health_monitor, temp_models_dir):
        """Test integration with health monitor for validation"""
        update_manager.health_monitor = mock_health_monitor
        
        # Mock health check to return healthy
        mock_integrity_result = Mock()
        mock_integrity_result.is_healthy = True
        mock_health_monitor.check_model_integrity.return_value = mock_integrity_result
        
        # Test validation
        result = await update_manager._validate_installation("test_model")
        
        assert result is True
        mock_health_monitor.check_model_integrity.assert_called_once_with("test_model")
        
        # Test with unhealthy result
        mock_integrity_result.is_healthy = False
        result = await update_manager._validate_installation("test_model")
        
        assert result is False


# Integration tests
class TestModelUpdateManagerIntegration:
    """Integration tests for ModelUpdateManager with other components"""
    
    @pytest.fixture
    async def integrated_manager(self, temp_models_dir):
        """Create manager with mock dependencies"""
        downloader = Mock()
        downloader.download_with_retry = AsyncMock()
        downloader.add_progress_callback = Mock()
        downloader.remove_progress_callback = Mock()
        downloader.cancel_download = AsyncMock()
        
        health_monitor = Mock()
        health_monitor.check_model_integrity = AsyncMock()
        
        manager = ModelUpdateManager(
            models_dir=temp_models_dir,
            downloader=downloader,
            health_monitor=health_monitor
        )
        
        await manager.initialize()
        yield manager
        await manager.shutdown()
    
    async def test_full_update_workflow(self, integrated_manager, temp_models_dir):
        """Test complete update workflow from check to completion"""
        model_id = "test_model"
        
        # Create existing model
        model_dir = Path(temp_models_dir) / model_id
        model_dir.mkdir()
        (model_dir / "version.json").write_text('{"version": "1.0.0"}')
        (model_dir / "config.json").write_text('{"test": "config"}')
        
        # Mock successful download
        download_result = Mock()
        download_result.success = True
        download_result.download_path = str(Path(temp_models_dir) / "downloaded_model.tar.gz")
        integrated_manager.downloader.download_with_retry.return_value = download_result
        
        # Create fake download file
        Path(download_result.download_path).write_bytes(b"fake model data")
        
        # Mock successful health check
        integrity_result = Mock()
        integrity_result.is_healthy = True
        integrated_manager.health_monitor.check_model_integrity.return_value = integrity_result
        
        # Create update info
        update_info = UpdateInfo(
            model_id=model_id,
            current_version="1.0.0",
            latest_version="2.0.0",
            update_type=UpdateType.MAJOR,
            priority=UpdatePriority.HIGH,
            size_mb=100.0,
            checksum=hashlib.sha256(b"fake model data").hexdigest(),
            download_url="https://test.com/model.tar.gz"
        )
        
        integrated_manager.update_cache[model_id] = update_info
        
        # Mock installation process
        with patch.object(integrated_manager, '_install_update') as mock_install:
            mock_install.return_value = True
            
            # Perform update
            result = await integrated_manager.perform_update(model_id, user_approved=True)
            
            # Verify success
            assert result.success is True
            assert result.final_status == UpdateStatus.COMPLETED
            assert result.old_version == "1.0.0"
            assert result.new_version == "2.0.0"
            assert result.rollback_available is True


if __name__ == "__main__":
    pytest.main([__file__])