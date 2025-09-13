#!/usr/bin/env python3
"""
Test Suite for WAN Model Weight Management System

Tests the WAN weight management functionality including downloading,
caching, verification, and migration of model weights.
"""

import asyncio
import json
import logging
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Setup test environment
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.models.wan_models.wan_weight_manager import (
    WANWeightManager, WeightInfo, WeightType, WeightStatus, CacheEntry,
    get_wan_weight_manager, download_wan_model, verify_wan_model
)
from backend.core.models.wan_models.wan_model_updater import (
    WANModelUpdater, UpdateInfo, ModelVersion, MigrationStrategy,
    check_wan_model_updates, update_wan_model
)
from backend.core.models.wan_models.wan_model_config import get_wan_model_config

logger = logging.getLogger(__name__)


class TestWANWeightManager(unittest.TestCase):
    """Test WAN Weight Manager functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.models_dir = self.test_dir / "models"
        self.cache_dir = self.test_dir / "cache"
        
        # Create test weight manager
        self.weight_manager = WANWeightManager(
            models_dir=str(self.models_dir),
            cache_dir=str(self.cache_dir)
        )
        
        # Mock the enhanced downloader
        self.mock_downloader = AsyncMock()
        self.weight_manager._downloader = self.mock_downloader
    
    def tearDown(self):
        """Clean up test environment"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    async def test_initialization(self):
        """Test weight manager initialization"""
        # Test directory creation
        self.assertTrue(self.models_dir.exists())
        self.assertTrue(self.cache_dir.exists())
        
        # Test initialization
        success = await self.weight_manager.initialize()
        self.assertTrue(success)
    
    async def test_weight_info_creation(self):
        """Test creation of weight info objects"""
        config = get_wan_model_config("t2v-A14B")
        self.assertIsNotNone(config)
        
        weight_infos = self.weight_manager._create_weight_infos(config)
        self.assertGreater(len(weight_infos), 0)
        
        # Check that all required weight types are present
        weight_types = {wi.weight_type for wi in weight_infos}
        self.assertIn(WeightType.MODEL, weight_types)
        self.assertIn(WeightType.CONFIG, weight_types)
        
        # Check weight info properties
        for weight_info in weight_infos:
            self.assertEqual(weight_info.model_id, "t2v-A14B")
            self.assertTrue(weight_info.url)
            self.assertTrue(weight_info.file_path.name)
    
    async def test_download_model_weights(self):
        """Test downloading model weights"""
        # Mock successful download with side effect to create files
        def create_mock_file(model_id, download_url):
            mock_result = MagicMock()
            mock_result.success = True
            mock_result.download_path = str(self.models_dir / f"{model_id}.bin")
            mock_result.error_message = None
            
            # Create the mock file
            file_path = Path(mock_result.download_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(b"mock model data" * 100)
            
            return mock_result
        
        self.mock_downloader.download_with_retry.side_effect = create_mock_file
        
        # Test download with force_redownload to ensure download is triggered
        # Disable verification since we're using mock files
        success = await self.weight_manager.download_model_weights("t2v-A14B", force_redownload=True, verify_integrity=False)
        self.assertTrue(success)
        
        # Verify downloader was called
        self.mock_downloader.download_with_retry.assert_called()
    
    async def test_verify_model_integrity(self):
        """Test model integrity verification"""
        # Create mock weight files
        config = get_wan_model_config("t2v-A14B")
        weight_infos = self.weight_manager._create_weight_infos(config)
        
        for weight_info in weight_infos:
            weight_info.file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if weight_info.weight_type.value in ["config", "tokenizer", "scheduler"]:
                # Create valid JSON file
                weight_info.file_path.write_text('{"test": "config"}')
            else:
                # Create mock binary file
                weight_info.file_path.write_bytes(b"mock weight data" * 100)
        
        # Test verification
        success = await self.weight_manager.verify_model_integrity("t2v-A14B")
        self.assertTrue(success)
    
    async def test_cache_management(self):
        """Test cache management functionality"""
        # Create mock cache entry
        weight_info = WeightInfo(
            weight_type=WeightType.MODEL,
            model_id="test-model",
            file_path=self.models_dir / "test_model.bin",
            url="https://example.com/model.bin",
            status=WeightStatus.DOWNLOADED
        )
        
        # Create mock file
        weight_info.file_path.parent.mkdir(parents=True, exist_ok=True)
        weight_info.file_path.write_bytes(b"test data" * 1000)
        
        cache_entry = CacheEntry(
            model_id="test-model",
            weights={WeightType.MODEL: weight_info},
            cache_time=datetime.now(),
            total_size_bytes=9000
        )
        
        self.weight_manager._cache["test-model"] = cache_entry
        
        # Test cache info retrieval
        cache_info = await self.weight_manager.get_model_cache_info("test-model")
        self.assertIsNotNone(cache_info)
        self.assertEqual(cache_info["model_id"], "test-model")
        self.assertGreater(cache_info["total_size_mb"], 0)
    
    async def test_cache_cleanup(self):
        """Test cache cleanup functionality"""
        # Create multiple mock cache entries
        for i in range(3):
            model_id = f"test-model-{i}"
            weight_info = WeightInfo(
                weight_type=WeightType.MODEL,
                model_id=model_id,
                file_path=self.models_dir / f"test_model_{i}.bin",
                url=f"https://example.com/model_{i}.bin",
                status=WeightStatus.DOWNLOADED
            )
            
            # Create mock file
            weight_info.file_path.parent.mkdir(parents=True, exist_ok=True)
            weight_info.file_path.write_bytes(b"test data" * 1000)
            
            cache_entry = CacheEntry(
                model_id=model_id,
                weights={WeightType.MODEL: weight_info},
                cache_time=datetime.now() - timedelta(days=i * 10),  # Different ages
                total_size_bytes=9000
            )
            
            self.weight_manager._cache[model_id] = cache_entry
        
        # Test cleanup with retention policy
        cleanup_stats = await self.weight_manager.cleanup_cache(
            max_size_gb=0.001,  # Very small to force cleanup
            retention_days=15   # Should clean up oldest entries
        )
        
        self.assertIn("cleaned_models", cleanup_stats)
        self.assertIn("freed_size_gb", cleanup_stats)
        self.assertGreater(len(cleanup_stats["cleaned_models"]), 0)
    
    async def test_file_integrity_verification(self):
        """Test file integrity verification"""
        # Test valid JSON file
        json_file = self.test_dir / "test.json"
        json_file.write_text('{"valid": "json"}')
        
        weight_info = WeightInfo(
            weight_type=WeightType.CONFIG,
            model_id="test",
            file_path=json_file,
            url="https://example.com/config.json"
        )
        
        is_valid = await self.weight_manager._verify_file_integrity(weight_info)
        self.assertTrue(is_valid)
        
        # Test invalid JSON file
        json_file.write_text('invalid json content')
        is_valid = await self.weight_manager._verify_file_integrity(weight_info)
        self.assertFalse(is_valid)
        
        # Test binary file
        bin_file = self.test_dir / "test.bin"
        bin_file.write_bytes(b"binary data" * 100)
        
        weight_info.weight_type = WeightType.MODEL
        weight_info.file_path = bin_file
        
        is_valid = await self.weight_manager._verify_file_integrity(weight_info)
        self.assertTrue(is_valid)
        
        # Test empty file
        bin_file.write_bytes(b"")
        is_valid = await self.weight_manager._verify_file_integrity(weight_info)
        self.assertFalse(is_valid)
    
    async def test_checksum_calculation(self):
        """Test checksum calculation"""
        test_file = self.test_dir / "test_checksum.bin"
        test_data = b"test data for checksum calculation"
        test_file.write_bytes(test_data)
        
        checksum = await self.weight_manager._calculate_file_checksum(test_file)
        self.assertIsInstance(checksum, str)
        self.assertEqual(len(checksum), 64)  # SHA256 hex length
        
        # Verify checksum consistency
        checksum2 = await self.weight_manager._calculate_file_checksum(test_file)
        self.assertEqual(checksum, checksum2)


class TestWANModelUpdater(unittest.TestCase):
    """Test WAN Model Updater functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.models_dir = self.test_dir / "models"
        
        # Create test weight manager
        self.weight_manager = WANWeightManager(models_dir=str(self.models_dir))
        self.weight_manager._downloader = AsyncMock()
        
        # Create test updater
        self.updater = WANModelUpdater(self.weight_manager)
    
    def tearDown(self):
        """Clean up test environment"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    async def test_version_history(self):
        """Test version history retrieval"""
        # Mock version data
        mock_versions = [
            ModelVersion(
                version="1.0.0",
                release_date=datetime.now() - timedelta(days=30),
                changelog=["Initial release"]
            ),
            ModelVersion(
                version="1.1.0",
                release_date=datetime.now() - timedelta(days=5),
                changelog=["Performance improvements"]
            )
        ]
        
        # Mock the fetch method
        self.updater._fetch_model_versions = AsyncMock(return_value=mock_versions)
        
        versions = await self.updater.get_version_history("t2v-A14B")
        self.assertEqual(len(versions), 2)
        self.assertEqual(versions[0].version, "1.0.0")
        self.assertEqual(versions[1].version, "1.1.0")
    
    async def test_check_for_updates(self):
        """Test update checking"""
        # Mock version data with newer version
        mock_versions = [
            ModelVersion(
                version="1.0.0",
                release_date=datetime.now() - timedelta(days=30),
                changelog=["Initial release"]
            ),
            ModelVersion(
                version="1.1.0",
                release_date=datetime.now() - timedelta(days=5),
                changelog=["Performance improvements"],
                download_urls={"model": "https://example.com/v1.1/model.bin"}
            )
        ]
        
        self.updater._fetch_model_versions = AsyncMock(return_value=mock_versions)
        
        update_infos = await self.updater.check_for_updates("t2v-A14B")
        self.assertIn("t2v-A14B", update_infos)
        
        update_info = update_infos["t2v-A14B"]
        self.assertTrue(update_info.update_available)
        self.assertEqual(update_info.latest_version, "1.1.0")
        self.assertEqual(update_info.current_version, "1.0.0")
    
    async def test_backup_creation(self):
        """Test model backup creation"""
        # Create mock weight files
        config = get_wan_model_config("t2v-A14B")
        weight_infos = self.weight_manager._create_weight_infos(config)
        
        for weight_info in weight_infos[:2]:  # Create a few files
            weight_info.file_path.parent.mkdir(parents=True, exist_ok=True)
            weight_info.file_path.write_bytes(b"mock weight data")
        
        # Update cache
        cache_entry = CacheEntry(
            model_id="t2v-A14B",
            weights={wi.weight_type: wi for wi in weight_infos[:2]},
            cache_time=datetime.now()
        )
        self.weight_manager._cache["t2v-A14B"] = cache_entry
        
        # Create backup
        backup_path = await self.updater._create_model_backup("t2v-A14B")
        self.assertIsNotNone(backup_path)
        self.assertTrue(backup_path.exists())
        
        # Verify backup contents
        backup_metadata = backup_path / "backup_metadata.json"
        self.assertTrue(backup_metadata.exists())
        
        with open(backup_metadata, 'r') as f:
            metadata = json.load(f)
        
        self.assertEqual(metadata["model_id"], "t2v-A14B")
        self.assertGreater(len(metadata["files"]), 0)
    
    async def test_backup_restoration(self):
        """Test model restoration from backup"""
        # Create original files
        config = get_wan_model_config("t2v-A14B")
        weight_infos = self.weight_manager._create_weight_infos(config)
        
        original_data = b"original weight data"
        for weight_info in weight_infos[:2]:
            weight_info.file_path.parent.mkdir(parents=True, exist_ok=True)
            weight_info.file_path.write_bytes(original_data)
        
        # Create backup
        backup_dir = self.test_dir / "backup"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        for weight_info in weight_infos[:2]:
            backup_file = backup_dir / weight_info.file_path.name
            backup_file.write_bytes(original_data)
        
        # Create backup metadata
        metadata = {
            "model_id": "t2v-A14B",
            "backup_time": datetime.now().isoformat(),
            "original_version": "1.0.0",
            "files": [wi.file_path.name for wi in weight_infos[:2]]
        }
        
        metadata_file = backup_dir / "backup_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        # Modify original files
        modified_data = b"modified weight data"
        for weight_info in weight_infos[:2]:
            weight_info.file_path.write_bytes(modified_data)
        
        # Restore from backup
        success = await self.updater._restore_from_backup("t2v-A14B", backup_dir)
        self.assertTrue(success)
        
        # Verify restoration
        for weight_info in weight_infos[:2]:
            if weight_info.file_path.exists():
                restored_data = weight_info.file_path.read_bytes()
                self.assertEqual(restored_data, original_data)
    
    async def test_migration_strategies(self):
        """Test different migration strategies"""
        # Test conservative strategy
        self.updater.set_migration_strategy(MigrationStrategy.CONSERVATIVE)
        self.assertEqual(self.updater._migration_strategy, MigrationStrategy.CONSERVATIVE)
        
        # Test aggressive strategy
        self.updater.set_migration_strategy(MigrationStrategy.AGGRESSIVE)
        self.assertEqual(self.updater._migration_strategy, MigrationStrategy.AGGRESSIVE)
        
        # Test parallel strategy
        self.updater.set_migration_strategy(MigrationStrategy.PARALLEL)
        self.assertEqual(self.updater._migration_strategy, MigrationStrategy.PARALLEL)
    
    async def test_auto_update_settings(self):
        """Test auto-update configuration"""
        # Test enabling auto-update
        self.updater.enable_auto_update(True)
        self.assertTrue(self.updater._auto_update_enabled)
        
        # Test disabling auto-update
        self.updater.enable_auto_update(False)
        self.assertFalse(self.updater._auto_update_enabled)
    
    async def test_backup_cleanup(self):
        """Test old backup cleanup"""
        # Create mock backup directories with different ages
        backup_base = self.models_dir / ".backups" / "test-model"
        
        for i in range(3):
            backup_dir = backup_base / f"backup_{i}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Create mock backup file
            backup_file = backup_dir / "model.bin"
            backup_file.write_bytes(b"backup data" * 100)
            
            # Set different modification times
            import os
            import time
            old_time = time.time() - (i + 1) * 10 * 24 * 3600  # i+1 * 10 days ago
            os.utime(backup_dir, (old_time, old_time))
        
        # Test cleanup
        cleanup_stats = await self.updater.cleanup_old_backups(retention_days=15)
        
        self.assertIn("cleaned_backups", cleanup_stats)
        self.assertIn("freed_size_gb", cleanup_stats)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test environment"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    async def test_get_wan_weight_manager(self):
        """Test weight manager factory function"""
        with patch('core.models.wan_models.wan_weight_manager.WANWeightManager') as mock_manager:
            mock_instance = AsyncMock()
            mock_instance.initialize.return_value = True
            mock_manager.return_value = mock_instance
            
            manager = await get_wan_weight_manager(str(self.test_dir))
            
            mock_manager.assert_called_once_with(models_dir=str(self.test_dir))
            mock_instance.initialize.assert_called_once()
    
    async def test_download_wan_model(self):
        """Test download convenience function"""
        with patch('core.models.wan_models.wan_weight_manager.get_wan_weight_manager') as mock_get_manager:
            mock_manager = AsyncMock()
            mock_manager.download_model_weights.return_value = True
            mock_get_manager.return_value = mock_manager
            
            success = await download_wan_model("t2v-A14B", str(self.test_dir))
            
            self.assertTrue(success)
            mock_get_manager.assert_called_once_with(str(self.test_dir))
            mock_manager.download_model_weights.assert_called_once_with("t2v-A14B", force_redownload=False)
    
    async def test_verify_wan_model(self):
        """Test verify convenience function"""
        with patch('core.models.wan_models.wan_weight_manager.get_wan_weight_manager') as mock_get_manager:
            mock_manager = AsyncMock()
            mock_manager.verify_model_integrity.return_value = True
            mock_get_manager.return_value = mock_manager
            
            success = await verify_wan_model("t2v-A14B", str(self.test_dir))
            
            self.assertTrue(success)
            mock_get_manager.assert_called_once_with(str(self.test_dir))
            mock_manager.verify_model_integrity.assert_called_once_with("t2v-A14B")
    
    async def test_check_wan_model_updates(self):
        """Test update check convenience function"""
        with patch('core.models.wan_models.wan_model_updater.WANWeightManager') as mock_manager_class:
            mock_manager = AsyncMock()
            mock_manager.initialize.return_value = True
            mock_manager_class.return_value = mock_manager
            
            with patch('core.models.wan_models.wan_model_updater.WANModelUpdater') as mock_updater_class:
                mock_updater = AsyncMock()
                mock_updater.check_for_updates.return_value = {"t2v-A14B": UpdateInfo(
                    model_id="t2v-A14B",
                    current_version="1.0.0",
                    latest_version="1.1.0",
                    update_available=True
                )}
                mock_updater_class.return_value = mock_updater
                
                updates = await check_wan_model_updates("t2v-A14B")
                
                self.assertIn("t2v-A14B", updates)
                self.assertTrue(updates["t2v-A14B"].update_available)


# Test runner
async def run_async_tests():
    """Run all async tests"""
    test_classes = [
        TestWANWeightManager,
        TestWANModelUpdater,
        TestUtilityFunctions
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Running {test_class.__name__}")
        print(f"{'='*60}")
        
        # Get all test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for test_method_name in test_methods:
            total_tests += 1
            
            try:
                # Create test instance
                test_instance = test_class()
                test_instance.setUp()
                
                # Get test method
                test_method = getattr(test_instance, test_method_name)
                
                # Run test
                if asyncio.iscoroutinefunction(test_method):
                    await test_method()
                else:
                    test_method()
                
                print(f"✅ {test_method_name}")
                passed_tests += 1
                
                # Clean up
                test_instance.tearDown()
                
            except Exception as e:
                print(f"❌ {test_method_name}: {e}")
                failed_tests.append(f"{test_class.__name__}.{test_method_name}: {e}")
                
                # Try to clean up even on failure
                try:
                    test_instance.tearDown()
                except:
                    pass
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Test Summary")
    print(f"{'='*60}")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    
    if failed_tests:
        print(f"\nFailed tests:")
        for failure in failed_tests:
            print(f"  - {failure}")
    
    return len(failed_tests) == 0


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("WAN Model Weight Management System Test Suite")
    print("=" * 60)
    
    # Run tests
    success = asyncio.run(run_async_tests())
    
    if success:
        print("\n🎉 All tests passed!")
        exit(0)
    else:
        print("\n💥 Some tests failed!")
        exit(1)
