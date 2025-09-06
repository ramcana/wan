"""
Simple tests for Model Update Manager
Basic functionality tests without complex async fixtures.
"""

import tempfile
import shutil
from pathlib import Path
import json
from datetime import datetime

# Import the module under test
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.core.model_update_manager import (
    ModelUpdateManager, UpdateType, UpdatePriority, ModelVersion
)


def test_version_parsing():
    """Test semantic version parsing"""
    temp_dir = tempfile.mkdtemp()
    try:
        manager = ModelUpdateManager(models_dir=temp_dir)
        
        # Test normal versions
        assert manager._parse_version("1.2.3") == (1, 2, 3)
        assert manager._parse_version("2.0.0") == (2, 0, 0)
        assert manager._parse_version("1.5") == (1, 5, 0)
        
        # Test versions with suffixes
        assert manager._parse_version("1.2.3-beta") == (1, 2, 3)
        assert manager._parse_version("2.1.0+build.123") == (2, 1, 0)
        
        # Test local versions
        assert manager._parse_version("local-123456") == (0, 0, 0)
        
    finally:
        shutil.rmtree(temp_dir)


def test_update_type_determination():
    """Test update type determination"""
    temp_dir = tempfile.mkdtemp()
    try:
        manager = ModelUpdateManager(models_dir=temp_dir)
        
        # Major update
        update_type = manager._determine_update_type("1.0.0", "2.0.0")
        assert update_type == UpdateType.MAJOR
        
        # Minor update
        update_type = manager._determine_update_type("1.0.0", "1.1.0")
        assert update_type == UpdateType.MINOR
        
        # Patch update
        update_type = manager._determine_update_type("1.0.0", "1.0.1")
        assert update_type == UpdateType.PATCH
        
    finally:
        shutil.rmtree(temp_dir)


def test_update_priority_determination():
    """Test update priority determination"""
    temp_dir = tempfile.mkdtemp()
    try:
        manager = ModelUpdateManager(models_dir=temp_dir)
        
        # Critical priority
        version_info = ModelVersion(
            version="1.1.0",
            release_date=datetime.now(),
            size_mb=100.0,
            checksum="test",
            download_url="test",
            changelog=["Critical security fix", "Bug fixes"]
        )
        priority = manager._determine_update_priority(version_info)
        assert priority == UpdatePriority.CRITICAL
        
        # High priority
        version_info.changelog = ["Performance improvements", "Memory optimization"]
        priority = manager._determine_update_priority(version_info)
        assert priority == UpdatePriority.HIGH
        
        # Medium priority
        version_info.changelog = ["New features", "UI improvements"]
        priority = manager._determine_update_priority(version_info)
        assert priority == UpdatePriority.MEDIUM
        
        # Low priority
        version_info.changelog = ["Minor tweaks", "Documentation updates"]
        priority = manager._determine_update_priority(version_info)
        assert priority == UpdatePriority.LOW
        
    finally:
        shutil.rmtree(temp_dir)


def test_is_update_available():
    """Test update availability checking"""
    temp_dir = tempfile.mkdtemp()
    try:
        manager = ModelUpdateManager(models_dir=temp_dir)
        
        # Test version comparisons
        assert manager._is_update_available("1.0.0", "1.0.1") is True
        assert manager._is_update_available("1.0.0", "1.1.0") is True
        assert manager._is_update_available("1.0.0", "2.0.0") is True
        assert manager._is_update_available("2.0.0", "1.0.0") is False
        assert manager._is_update_available("1.0.0", "1.0.0") is False
        
        # Local versions should always be updatable
        assert manager._is_update_available("local-123456", "1.0.0") is True
        
    finally:
        shutil.rmtree(temp_dir)


def test_directory_size_calculation():
    """Test directory size calculation"""
    temp_dir = tempfile.mkdtemp()
    try:
        manager = ModelUpdateManager(models_dir=temp_dir)
        
        # Create test directory with files
        test_dir = Path(temp_dir) / "test_model"
        test_dir.mkdir()
        
        # Create test files
        (test_dir / "file1.txt").write_text("Hello World")
        (test_dir / "file2.txt").write_text("Test Content")
        
        # Calculate size
        size = manager._calculate_directory_size(test_dir)
        
        # Should be greater than 0
        assert size > 0
        
        # Should be approximately the sum of file sizes
        expected_size = len("Hello World") + len("Test Content")
        assert abs(size - expected_size) < 10  # Allow small variance
        
    finally:
        shutil.rmtree(temp_dir)


def test_get_installed_models():
    """Test getting list of installed models"""
    temp_dir = tempfile.mkdtemp()
    try:
        manager = ModelUpdateManager(models_dir=temp_dir)
        
        # Initially no models
        models = manager._get_installed_models()
        assert len(models) == 0
        
        # Create mock model directories
        model_dirs = ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
        for model_id in model_dirs:
            model_dir = Path(temp_dir) / model_id
            model_dir.mkdir()
            # Create config file to make it look like a model
            (model_dir / "config.json").write_text('{"test": "config"}')
        
        # Should find the models
        models = manager._get_installed_models()
        assert len(models) == 3
        assert set(models) == set(model_dirs)
        
        # Hidden directories should be ignored
        hidden_dir = Path(temp_dir) / ".hidden"
        hidden_dir.mkdir()
        (hidden_dir / "config.json").write_text('{"test": "config"}')
        
        models = manager._get_installed_models()
        assert len(models) == 3  # Should still be 3
        assert ".hidden" not in models
        
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_version_parsing()
    test_update_type_determination()
    test_update_priority_determination()
    test_is_update_available()
    test_directory_size_calculation()
    test_get_installed_models()
    print("All tests passed!")