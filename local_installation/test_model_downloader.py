"""
Test script for the WAN2.2 Model Downloader
Tests the model downloading functionality with mock data.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.download_models import ModelDownloader, DownloadProgress
from scripts.interfaces import InstallationError, ErrorCategory


def test_model_downloader_initialization():
    """Test ModelDownloader initialization."""
    print("Testing ModelDownloader initialization...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        downloader = ModelDownloader(temp_dir)
        
        assert downloader.installation_path == Path(temp_dir)
        assert downloader.models_dir == Path(temp_dir) / "models"
        assert downloader.models_dir.exists()
        assert downloader.max_workers == 3
        
        print("✅ ModelDownloader initialization test passed")


def test_check_existing_models():
    """Test checking for existing models."""
    print("Testing check_existing_models...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        downloader = ModelDownloader(temp_dir)
        
        # Initially no models should exist
        existing = downloader.check_existing_models()
        assert existing == []
        
        # Create a mock model directory with files
        model_dir = downloader.models_dir / "WAN2.2-T2V-A14B"
        model_dir.mkdir(parents=True)
        
        # Create mock model files
        for file_name in downloader.MODEL_CONFIG["WAN2.2-T2V-A14B"].files:
            (model_dir / file_name).write_text("mock model data")
        
        # Mock the integrity verification to return True
        with patch.object(downloader, 'verify_model_integrity', return_value=True):
            existing = downloader.check_existing_models()
            assert "WAN2.2-T2V-A14B" in existing
        
        print("✅ check_existing_models test passed")


def test_verify_model_integrity():
    """Test model integrity verification."""
    print("Testing verify_model_integrity...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        downloader = ModelDownloader(temp_dir)
        
        # Create a mock model file
        model_dir = downloader.models_dir / "WAN2.2-T2V-A14B"
        model_dir.mkdir(parents=True)
        model_file = model_dir / "pytorch_model.bin"
        model_file.write_text("mock model data")
        
        # Test integrity verification (should pass with current implementation)
        result = downloader.verify_model_integrity(str(model_file))
        assert result == True
        
        # Test with non-existent file
        result = downloader.verify_model_integrity(str(model_dir / "nonexistent.bin"))
        assert result == False
        
        print("✅ verify_model_integrity test passed")


def test_configure_model_paths():
    """Test model path configuration."""
    print("Testing configure_model_paths...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        downloader = ModelDownloader(temp_dir)
        config_path = Path(temp_dir) / "config.json"
        
        # Create mock model directories
        for model_name in downloader.MODEL_CONFIG.keys():
            model_dir = downloader.models_dir / model_name
            model_dir.mkdir(parents=True)
        
        # Configure model paths
        result = downloader.configure_model_paths(str(config_path))
        assert result == True
        assert config_path.exists()
        
        # Verify configuration content
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        assert "models" in config
        assert "model_paths" in config["models"]
        assert len(config["models"]["model_paths"]) == len(downloader.MODEL_CONFIG)
        
        print("✅ configure_model_paths test passed")


def test_download_progress_tracking():
    """Test download progress tracking functionality."""
    print("Testing download progress tracking...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        downloader = ModelDownloader(temp_dir)
        
        # Create mock progress
        progress = DownloadProgress(
            model_name="WAN2.2-T2V-A14B",
            file_name="pytorch_model.bin",
            downloaded_bytes=1024*1024*100,  # 100 MB
            total_bytes=1024*1024*1000,     # 1 GB
            speed_mbps=50.0,
            eta_seconds=180.0,
            status="downloading"
        )
        
        downloader.download_progress["test_download"] = progress
        
        # Test getting progress
        all_progress = downloader.get_download_progress()
        assert "test_download" in all_progress
        assert all_progress["test_download"].downloaded_bytes == 1024*1024*100
        
        # Test total progress calculation
        total_progress, status = downloader.get_total_download_progress()
        assert total_progress == 10.0  # 100MB / 1GB = 10%
        assert "Downloading" in status
        
        print("✅ download progress tracking test passed")


def test_model_metadata_update():
    """Test model metadata update functionality."""
    print("Testing model metadata update...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        downloader = ModelDownloader(temp_dir)
        
        # Create mock model directories
        for model_name in list(downloader.MODEL_CONFIG.keys())[:2]:  # Just test with 2 models
            model_dir = downloader.models_dir / model_name
            model_dir.mkdir(parents=True)
        
        # Update metadata
        downloader._update_model_metadata()
        
        # Verify metadata file was created
        assert downloader.metadata_file.exists()
        
        # Verify metadata content
        metadata = downloader.load_json_file(downloader.metadata_file)
        assert "last_updated" in metadata
        assert "models" in metadata
        assert len(metadata["models"]) == 2
        
        print("✅ model metadata update test passed")


def test_cleanup_failed_downloads():
    """Test cleanup of failed downloads."""
    print("Testing cleanup_failed_downloads...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        downloader = ModelDownloader(temp_dir)
        
        # Create incomplete model directory (missing files)
        model_dir = downloader.models_dir / "WAN2.2-T2V-A14B"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("incomplete")
        
        # Mock check_existing_models to return empty (indicating incomplete model)
        with patch.object(downloader, 'check_existing_models', return_value=[]):
            downloader.cleanup_failed_downloads()
        
        # Verify incomplete model was removed
        assert not model_dir.exists()
        
        print("✅ cleanup_failed_downloads test passed")


def test_error_handling():
    """Test error handling in model downloader."""
    print("Testing error handling...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        downloader = ModelDownloader(temp_dir)
        
        # Test with invalid installation path
        try:
            invalid_downloader = ModelDownloader("/invalid/path/that/does/not/exist")
            # This should not raise an error during initialization
            # but should fail when trying to create directories
        except Exception as e:
            print(f"Expected error for invalid path: {e}")
        
        # Test configure_model_paths with invalid config path (Windows invalid path)
        try:
            result = downloader.configure_model_paths("C:\\invalid<>path\\config.json")
            # This should raise an InstallationError
            assert False, "Should have raised an error"
        except InstallationError as e:
            assert e.category == ErrorCategory.CONFIGURATION
            print(f"Caught expected configuration error: {e.message}")
        
        print("✅ error handling test passed")


def run_all_tests():
    """Run all model downloader tests."""
    print("=" * 60)
    print("Running WAN2.2 Model Downloader Tests")
    print("=" * 60)
    
    try:
        test_model_downloader_initialization()
        test_check_existing_models()
        test_verify_model_integrity()
        test_configure_model_paths()
        test_download_progress_tracking()
        test_model_metadata_update()
        test_cleanup_failed_downloads()
        test_error_handling()
        
        print("\n" + "=" * 60)
        print("✅ All Model Downloader Tests Passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
