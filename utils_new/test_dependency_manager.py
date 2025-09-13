"""
Tests for DependencyManager class

Tests cover:
- Remote code availability checking
- Pipeline code fetching from various sources
- Version compatibility validation
- Dependency installation
- Security validation
- Fallback strategies
"""

import pytest
import tempfile
import shutil
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import requests

from dependency_manager import (
    DependencyManager,
    RemoteCodeStatus,
    FetchResult,
    VersionCompatibility,
    InstallationResult,
    SecurityValidation
)


class TestDependencyManager:
    """Test suite for DependencyManager class"""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def dependency_manager(self, temp_cache_dir):
        """Create DependencyManager instance for testing"""
        return DependencyManager(cache_dir=temp_cache_dir, trust_mode="safe")
    
    @pytest.fixture
    def trust_dependency_manager(self, temp_cache_dir):
        """Create DependencyManager instance with trust mode"""
        return DependencyManager(cache_dir=temp_cache_dir, trust_mode="trust")

    def test_initialization(self, temp_cache_dir):
        """Test DependencyManager initialization"""
        dm = DependencyManager(cache_dir=temp_cache_dir, trust_mode="safe")
        
        assert dm.cache_dir == Path(temp_cache_dir)
        assert dm.trust_mode == "safe"
        assert "huggingface.co" in dm.trusted_sources
        assert "WanPipeline" in dm.known_pipelines
        assert dm.cache_dir.exists()

    def test_extract_model_id(self, dependency_manager):
        """Test model ID extraction from various path formats"""
        # Test local path
        assert dependency_manager._extract_model_id("/path/to/model") == "model"
        
        # Test Hugging Face model ID
        assert dependency_manager._extract_model_id("Wan-AI/Wan2.2-T2V-A14B-Diffusers") == "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        
        # Test simple name
        assert dependency_manager._extract_model_id("my_model") == "my_model"

    def test_is_huggingface_model(self, dependency_manager):
        """Test Hugging Face model detection"""
        assert dependency_manager._is_huggingface_model("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
        assert dependency_manager._is_huggingface_model("user/model")
        assert not dependency_manager._is_huggingface_model("local_model")
        assert not dependency_manager._is_huggingface_model("/path/to/model")

    @patch('requests.get')
    def test_check_hf_remote_code_success(self, mock_get, dependency_manager):
        """Test successful remote code checking on Hugging Face"""
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "siblings": [
                {"rfilename": "pipeline_wan.py"},
                {"rfilename": "config.json"}
            ],
            "sha": "abc123"
        }
        mock_get.return_value = mock_response
        
        result = dependency_manager._check_hf_remote_code("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
        
        assert result.is_available
        assert result.source_url == "https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        assert result.version == "abc123"

    @patch('requests.get')
    def test_check_hf_remote_code_no_pipeline(self, mock_get, dependency_manager):
        """Test remote code checking when no pipeline files exist"""
        # Mock API response without pipeline files
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "siblings": [
                {"rfilename": "config.json"},
                {"rfilename": "model.safetensors"}
            ]
        }
        mock_get.return_value = mock_response
        
        result = dependency_manager._check_hf_remote_code("user/model")
        
        assert not result.is_available
        assert "No pipeline code found" in result.error_message

    @patch('requests.get')
    def test_check_hf_remote_code_api_error(self, mock_get, dependency_manager):
        """Test remote code checking with API error"""
        # Mock API error
        mock_get.side_effect = requests.RequestException("Connection error")
        
        result = dependency_manager._check_hf_remote_code("user/model")
        
        assert not result.is_available
        assert "Failed to check Hugging Face" in result.error_message

    def test_check_remote_code_availability_local_cache(self, dependency_manager):
        """Test remote code availability checking with local cache"""
        model_id = "test/model"
        cache_file = dependency_manager.cache_dir / f"{model_id.replace('/', '_')}_info.json"
        
        # Create cache file
        cache_info = {
            "source_url": "https://example.com/model",
            "version": "1.0.0",
            "security_hash": "hash123"
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_info, f)
        
        result = dependency_manager.check_remote_code_availability(f"test/model")
        
        assert result.is_available
        assert result.source_url == "https://example.com/model"
        assert result.version == "1.0.0"
        assert result.security_hash == "hash123"

    def test_validate_source_security_trusted(self, dependency_manager):
        """Test security validation for trusted sources"""
        result = dependency_manager._validate_source_security("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
        
        assert result.is_safe
        assert result.risk_level == "low"
        assert len(result.detected_risks) == 0

    def test_validate_source_security_untrusted(self, dependency_manager):
        """Test security validation for untrusted sources"""
        result = dependency_manager._validate_source_security("untrusted.com/model")
        
        assert not result.is_safe
        assert result.risk_level == "high"
        assert "Untrusted source domain" in result.detected_risks

    def test_fetch_pipeline_code_trust_disabled(self, dependency_manager):
        """Test pipeline code fetching with trust_remote_code=False"""
        result = dependency_manager.fetch_pipeline_code("test/model", trust_remote_code=False)
        
        assert not result.success
        assert "Remote code fetching disabled" in result.error_message
        assert result.fallback_options is not None

    @patch('dependency_manager.DependencyManager._validate_source_security')
    def test_fetch_pipeline_code_security_fail(self, mock_security, dependency_manager):
        """Test pipeline code fetching with security validation failure"""
        # Mock security validation failure
        mock_security.return_value = SecurityValidation(
            is_safe=False,
            risk_level="high",
            detected_risks=["Malicious code detected"]
        )
        
        result = dependency_manager.fetch_pipeline_code("untrusted/model", trust_remote_code=True)
        
        assert not result.success
        assert "Security validation failed" in result.error_message

    @patch('huggingface_hub.list_repo_files')
    @patch('huggingface_hub.hf_hub_download')
    def test_fetch_from_huggingface_success(self, mock_download, mock_list_files, trust_dependency_manager):
        """Test successful pipeline fetching from Hugging Face"""
        # Mock huggingface_hub functions
        mock_list_files.return_value = ["pipeline_wan.py", "config.json"]
        mock_download.return_value = "/path/to/downloaded/pipeline_wan.py"
        
        result = trust_dependency_manager._fetch_from_huggingface("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
        
        assert result.success
        assert result.code_path == "/path/to/downloaded/pipeline_wan.py"
        assert result.version == "latest"

    @patch('huggingface_hub.list_repo_files')
    def test_fetch_from_huggingface_no_pipeline(self, mock_list_files, trust_dependency_manager):
        """Test pipeline fetching when no pipeline files exist"""
        # Mock no pipeline files
        mock_list_files.return_value = ["config.json", "model.safetensors"]
        
        result = trust_dependency_manager._fetch_from_huggingface("user/model")
        
        assert not result.success
        assert "No pipeline files found" in result.error_message

    def test_fetch_from_huggingface_no_hub(self, trust_dependency_manager):
        """Test pipeline fetching without huggingface_hub installed"""
        with patch('builtins.__import__', side_effect=ImportError("No module named 'huggingface_hub'")):
            result = trust_dependency_manager._fetch_from_huggingface("user/model")
            
            assert not result.success
            assert "huggingface_hub not installed" in result.error_message

    def test_extract_code_version(self, dependency_manager, temp_cache_dir):
        """Test version extraction from pipeline code"""
        # Create test file with version
        test_file = Path(temp_cache_dir) / "test_pipeline.py"
        test_file.write_text('__version__ = "1.2.3"\nclass TestPipeline: pass')
        
        version = dependency_manager._extract_code_version(str(test_file))
        assert version == "1.2.3"
        
        # Test file without version
        test_file.write_text('class TestPipeline: pass')
        version = dependency_manager._extract_code_version(str(test_file))
        assert version == "unknown"

    def test_calculate_compatibility_score(self, dependency_manager):
        """Test version compatibility score calculation"""
        # Exact match
        score = dependency_manager._calculate_compatibility_score("1.0.0", "1.0.0")
        assert score == 1.0
        
        # Minor version difference
        score = dependency_manager._calculate_compatibility_score("1.0.0", "1.1.0")
        assert 0.5 < score < 1.0
        
        # Major version difference
        score = dependency_manager._calculate_compatibility_score("1.0.0", "2.0.0")
        assert score < 0.8
        
        # Unknown version
        score = dependency_manager._calculate_compatibility_score("unknown", "1.0.0")
        assert score == 0.5

    def test_validate_code_version(self, dependency_manager, temp_cache_dir):
        """Test code version validation"""
        # Create test file with version
        test_file = Path(temp_cache_dir) / "test_pipeline.py"
        test_file.write_text('__version__ = "1.0.0"\nclass TestPipeline: pass')
        
        result = dependency_manager.validate_code_version(str(test_file), "1.0.0")
        
        assert result.is_compatible
        assert result.local_version == "1.0.0"
        assert result.required_version == "1.0.0"
        assert result.compatibility_score == 1.0

    def test_validate_code_version_mismatch(self, dependency_manager, temp_cache_dir):
        """Test code version validation with version mismatch"""
        # Create test file with different version
        test_file = Path(temp_cache_dir) / "test_pipeline.py"
        test_file.write_text('__version__ = "1.0.0"\nclass TestPipeline: pass')
        
        result = dependency_manager.validate_code_version(str(test_file), "2.0.0")
        
        assert result.compatibility_score < 1.0
        assert len(result.warnings) > 0
        assert "Version mismatch detected" in result.warnings[0]

    @patch('subprocess.run')
    def test_is_package_installed_true(self, mock_run, dependency_manager):
        """Test package installation checking - package installed"""
        mock_run.return_value.returncode = 0
        
        result = dependency_manager._is_package_installed("torch")
        assert result is True

    @patch('subprocess.run')
    def test_is_package_installed_false(self, mock_run, dependency_manager):
        """Test package installation checking - package not installed"""
        mock_run.return_value.returncode = 1
        
        result = dependency_manager._is_package_installed("nonexistent_package")
        assert result is False

    @patch('subprocess.run')
    def test_install_package_success(self, mock_run, dependency_manager):
        """Test successful package installation"""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "Successfully installed torch"
        mock_run.return_value.stderr = ""
        
        result = dependency_manager._install_package("torch")
        
        assert result["success"] is True
        assert "Successfully installed torch" in result["log"]

    @patch('subprocess.run')
    def test_install_package_failure(self, mock_run, dependency_manager):
        """Test failed package installation"""
        mock_run.return_value.returncode = 1
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = "ERROR: Could not find a version"
        
        result = dependency_manager._install_package("nonexistent_package")
        
        assert result["success"] is False
        assert result["error"] == "ERROR: Could not find a version"

    @patch('subprocess.run')
    def test_install_package_timeout(self, mock_run, dependency_manager):
        """Test package installation timeout"""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired("pip", 300)
        
        result = dependency_manager._install_package("slow_package")
        
        assert result["success"] is False
        assert "timed out" in result["log"]

    @patch('dependency_manager.DependencyManager._is_package_installed')
    @patch('dependency_manager.DependencyManager._install_package')
    def test_install_dependencies_success(self, mock_install, mock_is_installed, dependency_manager):
        """Test successful dependency installation"""
        # Mock package not installed initially
        mock_is_installed.return_value = False
        mock_install.return_value = {
            "success": True,
            "log": "Successfully installed package",
            "error": None
        }
        
        result = dependency_manager.install_dependencies(["torch>=2.0.0", "transformers"])
        
        assert result.success
        assert len(result.installed_packages) == 2
        assert len(result.failed_packages) == 0

    @patch('dependency_manager.DependencyManager._is_package_installed')
    @patch('dependency_manager.DependencyManager._install_package')
    def test_install_dependencies_partial_failure(self, mock_install, mock_is_installed, dependency_manager):
        """Test dependency installation with partial failures"""
        mock_is_installed.return_value = False
        
        # Mock one success, one failure
        mock_install.side_effect = [
            {"success": True, "log": "Success", "error": None},
            {"success": False, "log": "Failed", "error": "Package not found"}
        ]
        
        result = dependency_manager.install_dependencies(["torch", "nonexistent"])
        
        assert not result.success
        assert len(result.installed_packages) == 1
        assert len(result.failed_packages) == 1
        assert "torch" in result.installed_packages
        assert "nonexistent" in result.failed_packages

    @patch('dependency_manager.DependencyManager._is_package_installed')
    def test_install_dependencies_already_installed(self, mock_is_installed, dependency_manager):
        """Test dependency installation when packages already installed"""
        mock_is_installed.return_value = True
        
        result = dependency_manager.install_dependencies(["torch", "transformers"])
        
        assert result.success
        assert len(result.installed_packages) == 2
        assert len(result.failed_packages) == 0

    def test_get_fallback_options(self, dependency_manager):
        """Test fallback options generation"""
        options = dependency_manager.get_fallback_options("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
        
        assert len(options) > 0
        assert any("Manual installation" in option for option in options)
        assert any("wan-pipeline" in option for option in options)  # Wan-specific option

    def test_get_fallback_options_generic(self, dependency_manager):
        """Test fallback options for generic models"""
        options = dependency_manager.get_fallback_options("user/generic-model")
        
        assert len(options) > 0
        assert any("Manual installation" in option for option in options)
        assert any("Alternative models" in option for option in options)

    def test_cache_download_info(self, dependency_manager, temp_cache_dir):
        """Test caching of download information"""
        # Create a test file to cache info about
        test_file = Path(temp_cache_dir) / "test_pipeline.py"
        test_file.write_text("# Test pipeline code")
        
        dependency_manager._cache_download_info("test/model", str(test_file), "huggingface")
        
        # Check cache file was created
        cache_file = dependency_manager.cache_dir / "test_model_info.json"
        assert cache_file.exists()
        
        # Check cache content
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
        
        assert cache_data["model_id"] == "test/model"
        assert cache_data["code_path"] == str(test_file)
        assert cache_data["source"] == "huggingface"

    def test_calculate_file_hash(self, dependency_manager, temp_cache_dir):
        """Test file hash calculation"""
        test_file = Path(temp_cache_dir) / "test_file.py"
        test_file.write_text("test content")
        
        hash1 = dependency_manager._calculate_file_hash(str(test_file))
        hash2 = dependency_manager._calculate_file_hash(str(test_file))
        
        # Same file should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hash length
        
        # Different content should produce different hash
        test_file.write_text("different content")
        hash3 = dependency_manager._calculate_file_hash(str(test_file))
        assert hash1 != hash3


class TestIntegrationScenarios:
    """Integration tests for common usage scenarios"""
    
    @pytest.fixture
    def dependency_manager(self):
        """Create DependencyManager for integration tests"""
        temp_dir = tempfile.mkdtemp()
        dm = DependencyManager(cache_dir=temp_dir, trust_mode="trust")
        yield dm
        shutil.rmtree(temp_dir)

    @patch('requests.get')
    @patch('huggingface_hub.list_repo_files')
    @patch('huggingface_hub.hf_hub_download')
    def test_full_wan_model_workflow(self, mock_download, mock_list_files, mock_get, dependency_manager):
        """Test complete workflow for Wan model dependency management"""
        model_id = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        
        # Mock API responses
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "siblings": [{"rfilename": "pipeline_wan.py"}],
            "sha": "abc123"
        }
        mock_get.return_value = mock_response
        
        mock_list_files.return_value = ["pipeline_wan.py"]
        mock_download.return_value = "/cached/pipeline_wan.py"
        
        # Check availability
        status = dependency_manager.check_remote_code_availability(model_id)
        assert status.is_available
        
        # Fetch code
        result = dependency_manager.fetch_pipeline_code(model_id, trust_remote_code=True)
        assert result.success
        assert result.code_path == "/cached/pipeline_wan.py"

    def test_security_restricted_workflow(self, dependency_manager):
        """Test workflow with security restrictions"""
        dependency_manager.trust_mode = "safe"
        
        # Try to fetch from untrusted source
        result = dependency_manager.fetch_pipeline_code("untrusted.com/model", trust_remote_code=True)
        
        assert not result.success
        assert "Security validation failed" in result.error_message
        assert result.fallback_options is not None

    @patch('dependency_manager.DependencyManager._is_package_installed')
    @patch('dependency_manager.DependencyManager._install_package')
    def test_dependency_installation_workflow(self, mock_install, mock_is_installed, dependency_manager):
        """Test complete dependency installation workflow"""
        # Simulate mixed installation scenario
        def mock_is_installed_side_effect(package):
            return "torch" in package  # torch already installed
        
        def mock_install_side_effect(package):
            if "transformers" in package:
                return {"success": True, "log": "Success", "error": None}
            else:
                return {"success": False, "log": "Failed", "error": "Not found"}
        
        mock_is_installed.side_effect = mock_is_installed_side_effect
        mock_install.side_effect = mock_install_side_effect
        
        requirements = ["torch>=2.0.0", "transformers>=4.25.0", "nonexistent_package"]
        result = dependency_manager.install_dependencies(requirements)
        
        # Should have partial success
        assert not result.success  # Overall failure due to one failed package
        assert "torch>=2.0.0" in result.installed_packages  # Already installed
        assert "transformers>=4.25.0" in result.installed_packages  # Successfully installed
        assert "nonexistent_package" in result.failed_packages  # Failed to install


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
