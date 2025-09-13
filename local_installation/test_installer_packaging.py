"""
Test suite for installer packaging system.
"""

import pytest
import tempfile
import shutil
import json
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from scripts.installer_packager import InstallerPackager, VersionManager
from scripts.interfaces import PackagingInterface


class TestInstallerPackager:
    """Test cases for InstallerPackager class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.source_dir = self.temp_dir / "source"
        self.output_dir = self.temp_dir / "output"
        
        # Create mock source structure
        self.source_dir.mkdir(parents=True)
        self._create_mock_source_structure()
        
        self.packager = InstallerPackager(
            source_dir=str(self.source_dir),
            output_dir=str(self.output_dir)
        )
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_mock_source_structure(self):
        """Create mock source directory structure."""
        # Create directories
        dirs = ["scripts", "application", "resources", "examples", "logs"]
        for dir_name in dirs:
            (self.source_dir / dir_name).mkdir(parents=True)
        
        # Create mock files
        files = {
            "install.bat": "@echo off\necho Installing...",
            "README.md": "# WAN2.2 Installer",
            "scripts/main_installer.py": "# Main installer",
            "scripts/detect_system.py": "# System detection",
            "application/main.py": "# Main application",
            "resources/requirements.txt": "torch>=2.0.0",
            "resources/default_config.json": '{"version": "1.0.0"}',
            "examples/example.py": "# Example script"
        }
        
        for file_path, content in files.items():
            file_full_path = self.source_dir / file_path
            file_full_path.parent.mkdir(parents=True, exist_ok=True)
            file_full_path.write_text(content)
    
    def test_create_package_basic(self):
        """Test basic package creation."""
        version = "1.0.0"
        package_name = "TestInstaller"
        
        package_path = self.packager.create_package(version, package_name)
        
        # Verify package was created
        assert Path(package_path).exists()
        assert package_path.endswith(f"{package_name}-v{version}.zip")
        
        # Verify integrity files were created
        integrity_file = Path(package_path).with_suffix(".integrity.json")
        checksum_file = Path(package_path).with_suffix(".zip.sha256")
        assert integrity_file.exists()
        assert checksum_file.exists()
    
    def test_package_structure(self):
        """Test that package contains correct structure."""
        version = "1.0.0"
        package_path = self.packager.create_package(version)
        
        # Extract and verify structure
        extract_dir = self.temp_dir / "extracted"
        with zipfile.ZipFile(package_path, 'r') as zipf:
            zipf.extractall(extract_dir)
        
        # Check required directories exist
        required_dirs = ["scripts", "application", "resources", "examples", "logs", "models", "loras"]
        for dir_name in required_dirs:
            assert (extract_dir / dir_name).exists()
        
        # Check required files exist
        required_files = ["install.bat", "README.md", "version_manifest.json"]
        for file_name in required_files:
            assert (extract_dir / file_name).exists()
        
        # Check offline resources
        offline_dir = extract_dir / "offline_resources"
        assert offline_dir.exists()
        assert (offline_dir / "offline_manifest.json").exists()
    
    def test_version_manifest_creation(self):
        """Test version manifest creation."""
        version = "1.2.3"
        package_path = self.packager.create_package(version)
        
        # Extract and check version manifest
        extract_dir = self.temp_dir / "extracted"
        with zipfile.ZipFile(package_path, 'r') as zipf:
            zipf.extractall(extract_dir)
        
        manifest_file = extract_dir / "version_manifest.json"
        assert manifest_file.exists()
        
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
        
        assert manifest["version"] == version
        assert "build_date" in manifest
        assert "components" in manifest
        assert "update_info" in manifest
        assert "compatibility" in manifest
    
    def test_offline_resources_embedding(self):
        """Test offline resources embedding."""
        package_path = self.packager.create_package("1.0.0")
        
        # Extract and check offline resources
        extract_dir = self.temp_dir / "extracted"
        with zipfile.ZipFile(package_path, 'r') as zipf:
            zipf.extractall(extract_dir)
        
        offline_dir = extract_dir / "offline_resources"
        assert offline_dir.exists()
        
        # Check offline manifest
        manifest_file = offline_dir / "offline_manifest.json"
        assert manifest_file.exists()
        
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
        
        assert "resources" in manifest
        assert "installation_modes" in manifest
        
        # Check Python config
        python_config = offline_dir / "python" / "python_config.json"
        assert python_config.exists()
        
        # Check packages config
        packages_config = offline_dir / "packages" / "critical_packages.json"
        assert packages_config.exists()
    
    def test_package_integrity_verification(self):
        """Test package integrity verification."""
        package_path = self.packager.create_package("1.0.0")
        
        # Verify integrity
        assert self.packager.verify_package_integrity(package_path)
        
        # Test with corrupted package
        with open(package_path, 'ab') as f:
            f.write(b"corrupted_data")
        
        assert not self.packager.verify_package_integrity(package_path)
    
    def test_package_extraction(self):
        """Test package extraction."""
        package_path = self.packager.create_package("1.0.0")
        extract_dir = self.temp_dir / "test_extract"
        
        # Extract package
        result = self.packager.extract_package(package_path, str(extract_dir))
        assert result
        
        # Verify extracted structure
        assert (extract_dir / "install.bat").exists()
        assert (extract_dir / "scripts").exists()
        assert (extract_dir / "version_manifest.json").exists()
    
    def test_interface_compliance(self):
        """Test that InstallerPackager implements PackagingInterface."""
        assert isinstance(self.packager, PackagingInterface)
        
        # Test all interface methods are implemented
        assert hasattr(self.packager, 'create_package')
        assert hasattr(self.packager, 'verify_package_integrity')
        assert hasattr(self.packager, 'extract_package')


class TestVersionManager:
    """Test cases for VersionManager class."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.version_manager = VersionManager(str(self.temp_dir))
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_get_current_version_no_file(self):
        """Test getting current version when no version file exists."""
        version = self.version_manager.get_current_version()
        assert version is None
    
    def test_get_current_version_with_file(self):
        """Test getting current version with version file."""
        # Create version manifest
        version_data = {"version": "1.2.3"}
        version_file = self.temp_dir / "version_manifest.json"
        with open(version_file, 'w') as f:
            json.dump(version_data, f)
        
        version = self.version_manager.get_current_version()
        assert version == "1.2.3"
    
    def test_check_for_updates(self):
        """Test checking for updates."""
        # Create version manifest
        version_data = {"version": "1.0.0"}
        version_file = self.temp_dir / "version_manifest.json"
        with open(version_file, 'w') as f:
            json.dump(version_data, f)
        
        update_info = self.version_manager.check_for_updates()
        
        assert "current_version" in update_info
        assert "latest_version" in update_info
        assert "update_available" in update_info
        assert update_info["current_version"] == "1.0.0"
    
    def test_create_backup(self):
        """Test creating backup."""
        # Create some files to backup
        (self.temp_dir / "config.json").write_text('{"test": true}')
        (self.temp_dir / "version_manifest.json").write_text('{"version": "1.0.0"}')
        
        backup_path = self.version_manager.create_backup("test_backup")
        
        assert Path(backup_path).exists()
        assert "test_backup" in backup_path
        
        # Verify backup contains files
        backup_dir = Path(backup_path)
        assert (backup_dir / "version_manifest.json").exists()
    
    def test_restore_backup(self):
        """Test restoring backup."""
        # Create backup first
        (self.temp_dir / "version_manifest.json").write_text('{"version": "1.0.0"}')
        backup_path = self.version_manager.create_backup("test_backup")
        
        # Modify original file
        (self.temp_dir / "version_manifest.json").write_text('{"version": "2.0.0"}')
        
        # Restore backup
        result = self.version_manager.restore_backup("test_backup")
        assert result
        
        # Verify restoration
        with open(self.temp_dir / "version_manifest.json", 'r') as f:
            data = json.load(f)
        assert data["version"] == "1.0.0"
    
    def test_restore_nonexistent_backup(self):
        """Test restoring non-existent backup."""
        result = self.version_manager.restore_backup("nonexistent_backup")
        assert not result


class TestPackagingIntegration:
    """Integration tests for packaging system."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.source_dir = self.temp_dir / "source"
        self.output_dir = self.temp_dir / "output"
        
        # Create realistic source structure
        self._create_realistic_source_structure()
        
        self.packager = InstallerPackager(
            source_dir=str(self.source_dir),
            output_dir=str(self.output_dir)
        )
    
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_realistic_source_structure(self):
        """Create realistic source directory structure."""
        # Create the actual local_installation structure
        structure = {
            "install.bat": "@echo off\necho WAN2.2 Installer",
            "README.md": "# WAN2.2 Local Installation",
            "scripts/__init__.py": "",
            "scripts/main_installer.py": "# Main installer implementation",
            "scripts/detect_system.py": "# System detection",
            "scripts/setup_dependencies.py": "# Dependency setup",
            "scripts/download_models.py": "# Model downloader",
            "scripts/generate_config.py": "# Config generator",
            "scripts/validate_installation.py": "# Installation validator",
            "application/__init__.py": "",
            "application/main.py": "# Main application",
            "application/ui.py": "# User interface",
            "resources/requirements.txt": "torch>=2.0.0\ntransformers>=4.30.0",
            "resources/default_config.json": '{"system": {"threads": 8}}',
            "examples/dependency_management_example.py": "# Example",
            "logs/.gitkeep": ""
        }
        
        for file_path, content in structure.items():
            full_path = self.source_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
    
    def test_full_packaging_workflow(self):
        """Test complete packaging workflow."""
        version = "1.0.0"
        package_name = "WAN22-Installer"
        
        # Create package
        package_path = self.packager.create_package(version, package_name)
        
        # Verify package exists and has correct name
        assert Path(package_path).exists()
        assert f"{package_name}-v{version}.zip" in package_path
        
        # Verify integrity
        assert self.packager.verify_package_integrity(package_path)
        
        # Extract package
        extract_dir = self.temp_dir / "extracted"
        assert self.packager.extract_package(package_path, str(extract_dir))
        
        # Verify extracted structure matches requirements
        required_structure = [
            "install.bat",
            "README.md",
            "scripts/main_installer.py",
            "scripts/detect_system.py",
            "application/main.py",
            "resources/requirements.txt",
            "version_manifest.json",
            "offline_resources/offline_manifest.json",
            "models/.gitkeep",
            "loras/.gitkeep"
        ]
        
        for item in required_structure:
            assert (extract_dir / item).exists(), f"Missing: {item}"
    
    def test_version_management_integration(self):
        """Test version management integration."""
        # Create package with version manager
        package_path = self.packager.create_package("1.0.0")
        
        # Extract to simulate installation
        install_dir = self.temp_dir / "installation"
        self.packager.extract_package(package_path, str(install_dir))
        
        # Test version manager with extracted installation
        version_manager = VersionManager(str(install_dir))
        
        # Verify version detection
        current_version = version_manager.get_current_version()
        assert current_version == "1.0.0"
        
        # Test backup creation
        backup_path = version_manager.create_backup("pre_update")
        assert Path(backup_path).exists()
        
        # Test update checking
        update_info = version_manager.check_for_updates()
        assert "current_version" in update_info
        assert update_info["current_version"] == "1.0.0"


def run_packaging_tests():
    """Run all packaging tests."""
    print("Running installer packaging tests...")
    
    # Run tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short"
    ])


if __name__ == "__main__":
    run_packaging_tests()
