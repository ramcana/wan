"""
Test suite for distribution manager system.
"""

import pytest
import tempfile
import shutil
import json
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch
import os

import sys
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from distribution_manager import DistributionManager
from interfaces import InstallationError, ErrorCategory


class TestDistributionManager:
    """Test cases for DistributionManager."""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        source_dir = tempfile.mkdtemp(prefix="test_source_")
        output_dir = tempfile.mkdtemp(prefix="test_output_")
        
        # Create basic source structure
        source_path = Path(source_dir)
        (source_path / "scripts").mkdir()
        (source_path / "application").mkdir()
        (source_path / "resources").mkdir()
        
        # Create some test files
        (source_path / "install.bat").write_text("@echo off\necho Test installer")
        (source_path / "README.md").write_text("# Test Project")
        (source_path / "requirements.txt").write_text("pytest\nnumpy")
        (source_path / "scripts" / "test_script.py").write_text("print('test')")
        
        yield source_dir, output_dir
        
        # Cleanup
        shutil.rmtree(source_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)
    
    @pytest.fixture
    def dist_manager(self, temp_dirs):
        """Create DistributionManager instance."""
        source_dir, output_dir = temp_dirs
        return DistributionManager(source_dir, output_dir)
    
    def test_init(self, temp_dirs):
        """Test DistributionManager initialization."""
        source_dir, output_dir = temp_dirs
        manager = DistributionManager(source_dir, output_dir)
        
        assert manager.source_dir == Path(source_dir)
        assert manager.output_dir == Path(output_dir)
        assert hasattr(manager, 'packager')
    
    def test_prepare_release_basic(self, dist_manager):
        """Test basic release preparation."""
        with patch.object(dist_manager.packager, 'create_package') as mock_create:
            # Mock the packager to return a test package
            test_package = dist_manager.output_dir / "test_package.zip"
            test_package.write_text("test package content")
            mock_create.return_value = str(test_package)
            
            result = dist_manager.prepare_release("1.0.0", "Test release", test_compatibility=False)
            
            assert result["version"] == "1.0.0"
            assert "release_dir" in result
            assert "artifacts" in result
            assert "manifest" in result
            
            # Check that release directory was created
            release_dir = Path(result["release_dir"])
            assert release_dir.exists()
            assert (release_dir / "release_manifest.json").exists()
    
    def test_generate_release_artifacts(self, dist_manager):
        """Test release artifact generation."""
        release_dir = dist_manager.output_dir / "test_release"
        release_dir.mkdir(parents=True)
        
        with patch.object(dist_manager.packager, 'create_package') as mock_create:
            # Create a test package
            test_package = dist_manager.output_dir / "test_package.zip"
            with zipfile.ZipFile(test_package, 'w') as zf:
                zf.writestr("install.bat", "@echo off")
                zf.writestr("version_manifest.json", '{"version": "1.0.0"}')
            mock_create.return_value = str(test_package)
            
            artifacts = dist_manager._generate_release_artifacts("1.0.0", release_dir)
            
            assert "installer" in artifacts
            assert "source" in artifacts
            assert "quickstart" in artifacts
            
            # Check that artifacts exist
            for artifact_path in artifacts.values():
                assert Path(artifact_path).exists()
    
    def test_create_source_archive(self, dist_manager):
        """Test source archive creation."""
        release_dir = dist_manager.output_dir / "test_release"
        release_dir.mkdir(parents=True)
        
        source_archive = dist_manager._create_source_archive(release_dir, "1.0.0")
        
        assert source_archive.exists()
        assert source_archive.suffix == ".zip"
        
        # Check archive contents
        with zipfile.ZipFile(source_archive, 'r') as zf:
            files = zf.namelist()
            assert any("install.bat" in f for f in files)
            assert any("README.md" in f for f in files)
    
    def test_create_quickstart_guide(self, dist_manager):
        """Test quick-start guide creation."""
        release_dir = dist_manager.output_dir / "test_release"
        release_dir.mkdir(parents=True)
        
        quickstart = dist_manager._create_quickstart_guide(release_dir, "1.0.0")
        
        assert quickstart.exists()
        assert quickstart.name == "QUICKSTART.md"
        
        content = quickstart.read_text()
        assert "Version: 1.0.0" in content
        assert "System Requirements" in content
        assert "Installation Steps" in content
    
    def test_create_release_manifest(self, dist_manager):
        """Test release manifest creation."""
        artifacts = {
            "installer": str(dist_manager.output_dir / "installer.zip"),
            "source": str(dist_manager.output_dir / "source.zip")
        }
        
        # Create test artifact files
        for artifact_path in artifacts.values():
            Path(artifact_path).write_text("test content")
        
        manifest = dist_manager._create_release_manifest("1.0.0", "Test release", artifacts)
        
        assert manifest["version"] == "1.0.0"
        assert manifest["release_notes"] == "Test release"
        assert "build_info" in manifest
        assert "artifacts" in manifest
        assert "system_requirements" in manifest
        
        # Check artifact information
        for artifact_type in artifacts.keys():
            assert artifact_type in manifest["artifacts"]
            assert "filename" in manifest["artifacts"][artifact_type]
            assert "size_bytes" in manifest["artifacts"][artifact_type]
            assert "sha256" in manifest["artifacts"][artifact_type]
    
    def test_run_compatibility_tests(self, dist_manager):
        """Test compatibility testing."""
        artifacts = {
            "installer": str(dist_manager.output_dir / "installer.zip")
        }
        
        # Create test installer package
        installer_path = Path(artifacts["installer"])
        with zipfile.ZipFile(installer_path, 'w') as zf:
            zf.writestr("install.bat", "@echo off")
            zf.writestr("version_manifest.json", '{"version": "1.0.0"}')
        
        with patch.object(dist_manager.packager, 'verify_package_integrity', return_value=True):
            with patch.object(dist_manager.packager, 'extract_package', return_value=True):
                results = dist_manager._run_compatibility_tests(artifacts)
                
                assert "test_timestamp" in results
                assert "test_system" in results
                assert "tests" in results
                assert "compatibility_score" in results
                assert "overall_status" in results
                
                # Check individual tests
                assert "package_integrity" in results["tests"]
                assert "archive_extraction" in results["tests"]
                assert "file_permissions" in results["tests"]
    
    def test_test_package_integrity(self, dist_manager):
        """Test package integrity testing."""
        artifacts = {"installer": str(dist_manager.output_dir / "installer.zip")}
        
        # Create test package
        Path(artifacts["installer"]).write_text("test content")
        
        with patch.object(dist_manager.packager, 'verify_package_integrity', return_value=True):
            result = dist_manager._test_package_integrity(artifacts)
            
            assert result["passed"] is True
            assert "message" in result
    
    def test_test_archive_extraction(self, dist_manager):
        """Test archive extraction testing."""
        artifacts = {"installer": str(dist_manager.output_dir / "installer.zip")}
        
        # Create test package with required files
        with zipfile.ZipFile(artifacts["installer"], 'w') as zf:
            zf.writestr("install.bat", "@echo off")
            zf.writestr("version_manifest.json", '{"version": "1.0.0"}')
        
        with patch.object(dist_manager.packager, 'extract_package', return_value=True):
            result = dist_manager._test_archive_extraction(artifacts)
            
            assert result["passed"] is True
            assert "message" in result
    
    def test_test_file_permissions(self, dist_manager):
        """Test file permissions testing."""
        artifacts = {
            "installer": str(dist_manager.output_dir / "installer.zip"),
            "script": str(dist_manager.output_dir / "test.bat")
        }
        
        # Create test files
        for artifact_path in artifacts.values():
            Path(artifact_path).write_text("test content")
        
        result = dist_manager._test_file_permissions(artifacts)
        
        assert "passed" in result
        assert "message" in result
    
    def test_test_path_compatibility(self, dist_manager):
        """Test path compatibility testing."""
        artifacts = {
            "good_file": str(dist_manager.output_dir / "good_file.zip"),
            "bad_file": str(dist_manager.output_dir / "bad<file>.zip")  # Contains problematic character
        }
        
        result = dist_manager._test_path_compatibility(artifacts)
        
        assert "passed" in result
        assert "issues" in result
        # Should fail due to problematic character
        assert not result["passed"]
        assert len(result["issues"]) > 0
    
    def test_test_size_validation(self, dist_manager):
        """Test size validation testing."""
        artifacts = {"installer": str(dist_manager.output_dir / "installer.zip")}
        
        # Create reasonably sized test file
        Path(artifacts["installer"]).write_text("test content" * 100)
        
        result = dist_manager._test_size_validation(artifacts)
        
        assert "passed" in result
        assert "message" in result
        assert result["passed"] is True  # Should pass for small test file
    
    def test_generate_distribution_checksums(self, dist_manager):
        """Test distribution checksum generation."""
        release_dir = dist_manager.output_dir / "test_release"
        release_dir.mkdir(parents=True)
        
        # Create test files
        (release_dir / "file1.txt").write_text("content1")
        (release_dir / "file2.txt").write_text("content2")
        
        dist_manager._generate_distribution_checksums(release_dir)
        
        checksums_file = release_dir / "CHECKSUMS.txt"
        assert checksums_file.exists()
        
        content = checksums_file.read_text()
        assert "file1.txt" in content
        assert "file2.txt" in content
        assert "SHA256" in content
    
    def test_calculate_file_checksum(self, dist_manager):
        """Test file checksum calculation."""
        test_file = dist_manager.output_dir / "test.txt"
        test_file.write_text("test content")
        
        checksum = dist_manager._calculate_file_checksum(str(test_file))
        
        assert len(checksum) == 64  # SHA256 hex length
        assert isinstance(checksum, str)
        
        # Same content should produce same checksum
        checksum2 = dist_manager._calculate_file_checksum(str(test_file))
        assert checksum == checksum2
    
    def test_create_release_package(self, dist_manager):
        """Test release package creation."""
        release_dir = dist_manager.output_dir / "test_release"
        release_dir.mkdir(parents=True)
        
        # Create test files in release directory
        (release_dir / "file1.txt").write_text("content1")
        (release_dir / "subdir").mkdir()
        (release_dir / "subdir" / "file2.txt").write_text("content2")
        
        release_package = dist_manager._create_release_package(release_dir, "1.0.0")
        
        assert release_package.exists()
        assert release_package.suffix == ".zip"
        assert "WAN22-Release-v1.0.0" in release_package.name
        
        # Check package contents
        with zipfile.ZipFile(release_package, 'r') as zf:
            files = zf.namelist()
            assert "file1.txt" in files
            assert "subdir/file2.txt" in files
    
    def test_validate_release(self, dist_manager):
        """Test release validation."""
        release_dir = dist_manager.output_dir / "test_release"
        release_dir.mkdir(parents=True)
        
        # Create test manifest
        manifest = {
            "version": "1.0.0",
            "artifacts": {
                "installer": {"filename": "installer.zip"},
                "quickstart": {"filename": "QUICKSTART.md"}
            }
        }
        
        manifest_file = release_dir / "release_manifest.json"
        with open(manifest_file, "w") as f:
            json.dump(manifest, f)
        
        # Create test artifacts
        (release_dir / "installer.zip").write_text("test installer")
        (release_dir / "QUICKSTART.md").write_text("test quickstart")
        (release_dir / "CHECKSUMS.txt").write_text("test checksums")
        
        result = dist_manager.validate_release(str(release_dir))
        
        assert "valid" in result
        assert "version" in result
        assert "checks" in result
        assert result["version"] == "1.0.0"
    
    def test_validate_release_missing_manifest(self, dist_manager):
        """Test release validation with missing manifest."""
        release_dir = dist_manager.output_dir / "test_release"
        release_dir.mkdir(parents=True)
        
        result = dist_manager.validate_release(str(release_dir))
        
        assert result["valid"] is False
        assert "error" in result
        assert "manifest not found" in result["error"]
    
    def test_validate_release_missing_artifacts(self, dist_manager):
        """Test release validation with missing artifacts."""
        release_dir = dist_manager.output_dir / "test_release"
        release_dir.mkdir(parents=True)
        
        # Create manifest with missing artifacts
        manifest = {
            "version": "1.0.0",
            "artifacts": {
                "installer": {"filename": "missing_installer.zip"},
                "quickstart": {"filename": "missing_quickstart.md"}
            }
        }
        
        manifest_file = release_dir / "release_manifest.json"
        with open(manifest_file, "w") as f:
            json.dump(manifest, f)
        
        result = dist_manager.validate_release(str(release_dir))
        
        assert result["valid"] is False
        assert "checks" in result
        assert result["failed_checks"] > 0
    
    def test_error_handling(self, dist_manager):
        """Test error handling in release preparation."""
        with patch.object(dist_manager.packager, 'create_package', side_effect=Exception("Test error")):
            with pytest.raises(InstallationError) as exc_info:
                dist_manager.prepare_release("1.0.0", "Test release")
            
            assert "Release preparation failed" in str(exc_info.value)
            assert exc_info.value.category == ErrorCategory.SYSTEM
