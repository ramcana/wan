#!/usr/bin/env python3
"""
Test script for the Model Configuration System.
Tests model path configuration, directory structure setup, file organization,
and metadata management.
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

# Add the current directory to the path to enable proper imports
sys.path.insert(0, os.path.dirname(__file__))

# Import the modules using the package structure
try:
    from scripts.model_configuration import (
        ModelConfigurationManager, ModelStatus, ModelType, ModelMetadata,
        create_model_configuration_manager
    )
    from scripts.interfaces import HardwareProfile, CPUInfo, MemoryInfo, GPUInfo, StorageInfo, OSInfo
except ImportError:
    # Fallback for direct execution - modify the imports to be absolute
    import importlib.util

    # Load interfaces module
    interfaces_spec = importlib.util.spec_from_file_location(
        "interfaces", os.path.join(os.path.dirname(__file__), "scripts", "interfaces.py")
    )
    interfaces = importlib.util.module_from_spec(interfaces_spec)
    interfaces_spec.loader.exec_module(interfaces)
    
    # Load base_classes module with interfaces available
    base_classes_spec = importlib.util.spec_from_file_location(
        "base_classes", os.path.join(os.path.dirname(__file__), "scripts", "base_classes.py")
    )
    base_classes = importlib.util.module_from_spec(base_classes_spec)
    # Make interfaces available to base_classes
    sys.modules['interfaces'] = interfaces
    base_classes_spec.loader.exec_module(base_classes)
    
    # Load model_configuration module with dependencies available
    model_config_spec = importlib.util.spec_from_file_location(
        "model_configuration", os.path.join(os.path.dirname(__file__), "scripts", "model_configuration.py")
    )
    model_config = importlib.util.module_from_spec(model_config_spec)
    # Make dependencies available
    sys.modules['base_classes'] = base_classes
    model_config_spec.loader.exec_module(model_config)
    
    # Import the classes
    ModelConfigurationManager = model_config.ModelConfigurationManager
    ModelStatus = model_config.ModelStatus
    ModelType = model_config.ModelType
    ModelMetadata = model_config.ModelMetadata
    create_model_configuration_manager = model_config.create_model_configuration_manager
    
    HardwareProfile = interfaces.HardwareProfile
    CPUInfo = interfaces.CPUInfo
    MemoryInfo = interfaces.MemoryInfo
    GPUInfo = interfaces.GPUInfo
    StorageInfo = interfaces.StorageInfo
    OSInfo = interfaces.OSInfo


def create_test_hardware_profile() -> HardwareProfile:
    """Create a test hardware profile for testing."""
    return HardwareProfile(
        cpu=CPUInfo(
            model="AMD Ryzen Threadripper PRO 5995WX",
            cores=64,
            threads=128,
            base_clock=2.7,
            boost_clock=4.5,
            architecture="x64"
        ),
        memory=MemoryInfo(
            total_gb=128,
            available_gb=120,
            type="DDR4",
            speed=3200
        ),
        gpu=GPUInfo(
            model="NVIDIA GeForce RTX 4080",
            vram_gb=16,
            cuda_version="12.1",
            driver_version="537.13",
            compute_capability="8.9"
        ),
        storage=StorageInfo(
            available_gb=500,
            type="NVMe SSD"
        ),
        os=OSInfo(
            name="Windows",
            version="11",
            architecture="x64"
        )
    )


def create_mock_model_files(model_dir: Path, model_name: str) -> list:
    """Create mock model files for testing."""
    files = [
        "pytorch_model.bin",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json"
    ]
    
    created_files = []
    for filename in files:
        file_path = model_dir / filename
        
        # Create different content based on file type
        if filename == "pytorch_model.bin":
            # Create a larger mock binary file
            content = b"MOCK_PYTORCH_MODEL_DATA" * 1000000  # ~24MB
        elif filename.endswith(".json"):
            content = json.dumps({
                "model_name": model_name,
                "version": "1.0.3",
                "architecture": "transformer",
                "mock_file": True
            }, indent=2).encode()
        else:
            content = f"Mock content for {filename} in {model_name}".encode()
        
        with open(file_path, 'wb') as f:
            f.write(content)
        
        created_files.append(str(file_path))
    
    return created_files


def test_directory_structure_setup():
    """Test directory structure setup."""
    print("Testing directory structure setup...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        installation_path = Path(temp_dir)
        models_dir = installation_path / "models"
        
        # Create model configuration manager
        manager = ModelConfigurationManager(str(installation_path), str(models_dir))
        
        # Verify directory structure
        assert manager.directory_structure.models_root.exists()
        assert manager.directory_structure.cache_dir.exists()
        assert manager.directory_structure.temp_dir.exists()
        assert manager.directory_structure.backup_dir.exists()
        
        # Verify model directories
        for model_name in manager.MODEL_DEFINITIONS.keys():
            assert manager.directory_structure.model_dirs[model_name].exists()
        
        print("✅ Directory structure setup test passed")


def test_model_file_organization():
    """Test model file organization."""
    print("Testing model file organization...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        installation_path = Path(temp_dir)
        models_dir = installation_path / "models"
        
        manager = ModelConfigurationManager(str(installation_path), str(models_dir))
        
        # Create mock source files in a temporary location
        source_dir = Path(temp_dir) / "source"
        source_dir.mkdir()
        
        model_name = "WAN2.2-T2V-A14B"
        source_files = create_mock_model_files(source_dir, model_name)
        
        # Test file organization
        success = manager.organize_model_files(model_name, source_files)
        assert success
        
        # Verify files were organized correctly
        model_dir = manager.directory_structure.model_dirs[model_name]
        for expected_file in manager.MODEL_DEFINITIONS[model_name]["files"]:
            assert (model_dir / expected_file).exists()
        
        print("✅ Model file organization test passed")


def test_model_validation():
    """Test model file validation."""
    print("Testing model validation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        installation_path = Path(temp_dir)
        models_dir = installation_path / "models"
        
        manager = ModelConfigurationManager(str(installation_path), str(models_dir))
        
        model_name = "WAN2.2-T2V-A14B"
        model_dir = manager.directory_structure.model_dirs[model_name]
        
        # Test validation with missing files
        is_valid, issues = manager.validate_model_files(model_name)
        assert not is_valid
        assert len(issues) > 0
        
        # Create mock model files
        mock_files = create_mock_model_files(model_dir, model_name)
        
        # Test validation with all files present
        is_valid, issues = manager.validate_model_files(model_name)
        assert is_valid
        assert len(issues) == 0
        
        print("✅ Model validation test passed")


def test_model_status():
    """Test model status tracking."""
    print("Testing model status tracking...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        installation_path = Path(temp_dir)
        models_dir = installation_path / "models"
        
        manager = ModelConfigurationManager(str(installation_path), str(models_dir))
        
        # Test status for missing model
        model_name = "WAN2.2-T2V-A14B"
        status = manager.get_model_status(model_name)
        assert status.status == ModelStatus.MISSING
        
        # Create mock model files
        model_dir = manager.directory_structure.model_dirs[model_name]
        mock_files = create_mock_model_files(model_dir, model_name)
        
        # Test status for available model
        status = manager.get_model_status(model_name)
        assert status.status == ModelStatus.AVAILABLE
        assert status.name == model_name
        assert status.model_type == ModelType.TEXT_TO_VIDEO
        
        # Test all models status
        all_status = manager.get_all_model_status()
        assert model_name in all_status
        assert all_status[model_name].status == ModelStatus.AVAILABLE
        
        print("✅ Model status test passed")


def test_configuration_generation():
    """Test configuration generation with hardware optimization."""
    print("Testing configuration generation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        installation_path = Path(temp_dir)
        models_dir = installation_path / "models"
        config_path = installation_path / "config.json"
        
        manager = ModelConfigurationManager(str(installation_path), str(models_dir))
        hardware_profile = create_test_hardware_profile()
        
        # Create a mock available model
        model_name = "WAN2.2-T2V-A14B"
        model_dir = manager.directory_structure.model_dirs[model_name]
        create_mock_model_files(model_dir, model_name)
        
        # Test configuration generation
        success = manager.configure_model_paths(str(config_path), hardware_profile)
        assert success
        assert config_path.exists()
        
        # Verify configuration content
        config = json.loads(config_path.read_text())
        assert "models" in config
        assert "model_paths" in config["models"]
        assert "models_directory" in config["models"]
        assert model_name in config["models"]["model_paths"]
        
        # Verify hardware optimization (high-performance system)
        assert config["models"]["cache_models"] == True
        assert config["models"]["preload_models"] == True
        assert config["models"]["model_precision"] == "bf16"
        
        print("✅ Configuration generation test passed")


def test_metadata_management():
    """Test metadata management and persistence."""
    print("Testing metadata management...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        installation_path = Path(temp_dir)
        models_dir = installation_path / "models"
        
        manager = ModelConfigurationManager(str(installation_path), str(models_dir))
        
        # Create mock model files and organize them
        model_name = "WAN2.2-T2V-A14B"
        source_dir = Path(temp_dir) / "source"
        source_dir.mkdir()
        source_files = create_mock_model_files(source_dir, model_name)
        
        manager.organize_model_files(model_name, source_files)
        
        # Verify metadata was created
        assert model_name in manager.metadata_cache
        metadata = manager.metadata_cache[model_name]
        assert metadata.name == model_name
        assert metadata.status == ModelStatus.AVAILABLE
        
        # Test metadata persistence
        metadata_file = manager.directory_structure.metadata_file
        assert metadata_file.exists()
        
        # Create new manager instance to test loading
        manager2 = ModelConfigurationManager(str(installation_path), str(models_dir))
        assert model_name in manager2.metadata_cache
        
        print("✅ Metadata management test passed")


def test_model_summary():
    """Test model summary generation."""
    print("Testing model summary generation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        installation_path = Path(temp_dir)
        models_dir = installation_path / "models"
        
        manager = ModelConfigurationManager(str(installation_path), str(models_dir))
        
        # Create one available model
        model_name = "WAN2.2-T2V-A14B"
        model_dir = manager.directory_structure.model_dirs[model_name]
        create_mock_model_files(model_dir, model_name)
        
        # Get summary
        summary = manager.get_model_summary()
        
        assert summary["total_models"] == 3  # All defined models
        assert summary["available_models"] == 1  # Only one created
        assert summary["missing_models"] == 2  # Two missing
        assert summary["corrupted_models"] == 0
        assert summary["total_size_gb"] > 0
        assert "models" in summary
        assert model_name in summary["models"]
        
        print("✅ Model summary test passed")


def test_backup_functionality():
    """Test backup functionality."""
    print("Testing backup functionality...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        installation_path = Path(temp_dir)
        models_dir = installation_path / "models"
        
        manager = ModelConfigurationManager(str(installation_path), str(models_dir))
        
        # Create some metadata
        model_name = "WAN2.2-T2V-A14B"
        source_dir = Path(temp_dir) / "source"
        source_dir.mkdir()
        source_files = create_mock_model_files(source_dir, model_name)
        manager.organize_model_files(model_name, source_files)
        
        # Create backup
        backup_path = manager.backup_model_configuration()
        assert Path(backup_path).exists()
        
        # Verify backup content
        backup_data = json.loads(Path(backup_path).read_text())
        assert "timestamp" in backup_data
        assert "models_directory" in backup_data
        assert "metadata" in backup_data
        assert model_name in backup_data["metadata"]
        
        print("✅ Backup functionality test passed")


def test_factory_function():
    """Test factory function."""
    print("Testing factory function...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = create_model_configuration_manager(temp_dir)
        assert isinstance(manager, ModelConfigurationManager)
        assert manager.installation_path == Path(temp_dir)
        
        print("✅ Factory function test passed")


def main():
    """Run all tests."""
    print("🧪 Running Model Configuration System Tests\n")
    
    try:
        test_directory_structure_setup()
        test_model_file_organization()
        test_model_validation()
        test_model_status()
        test_configuration_generation()
        test_metadata_management()
        test_model_summary()
        test_backup_functionality()
        test_factory_function()
        
        print("\n🎉 All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
