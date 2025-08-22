"""
Verification tests to ensure all requirements 6.1-6.4 are implemented correctly
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from dependency_manager import DependencyManager


class TestRequirementsVerification:
    """Verify that all acceptance criteria for Requirement 6 are implemented"""
    
    @pytest.fixture
    def dependency_manager(self):
        """Create DependencyManager for testing"""
        temp_dir = tempfile.mkdtemp()
        dm = DependencyManager(cache_dir=temp_dir, trust_mode="trust")
        yield dm
        shutil.rmtree(temp_dir)

    @patch('huggingface_hub.list_repo_files')
    @patch('huggingface_hub.hf_hub_download')
    def test_requirement_6_1_automatic_fetching(self, mock_download, mock_list_files, dependency_manager):
        """
        Requirement 6.1: WHEN trust_remote_code=True is enabled and pipeline code is missing 
        THEN the system SHALL attempt automatic fetching from Hugging Face
        """
        # Mock successful fetching
        mock_list_files.return_value = ["pipeline_wan.py"]
        mock_download.return_value = "/cached/pipeline_wan.py"
        
        # Test automatic fetching
        result = dependency_manager.fetch_pipeline_code(
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers", 
            trust_remote_code=True
        )
        
        assert result.success
        assert result.code_path == "/cached/pipeline_wan.py"
        mock_list_files.assert_called_once()
        mock_download.assert_called_once()

    def test_requirement_6_2_version_validation(self, dependency_manager):
        """
        Requirement 6.2: WHEN pipeline code versions don't match model requirements 
        THEN the system SHALL validate compatibility and suggest updates
        """
        # Create test file with version mismatch
        temp_dir = Path(dependency_manager.cache_dir)
        test_file = temp_dir / "test_pipeline.py"
        test_file.write_text('__version__ = "1.0.0"\nclass TestPipeline: pass')
        
        # Test version validation with mismatch
        result = dependency_manager.validate_code_version(str(test_file), "2.0.0")
        
        assert not result.is_compatible or result.compatibility_score < 1.0
        assert len(result.warnings) > 0
        assert "Version mismatch detected" in result.warnings[0]
        assert len(result.recommendations) > 0
        assert "updating pipeline code" in result.recommendations[0]

    @patch('huggingface_hub.list_repo_files')
    def test_requirement_6_3_fallback_options(self, mock_list_files, dependency_manager):
        """
        Requirement 6.3: WHEN remote code download fails 
        THEN the system SHALL provide fallback options and manual installation instructions
        """
        # Mock download failure
        mock_list_files.side_effect = Exception("Download failed")
        
        # Test fallback options are provided
        result = dependency_manager.fetch_pipeline_code(
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers", 
            trust_remote_code=True
        )
        
        assert not result.success
        assert result.fallback_options is not None
        assert len(result.fallback_options) > 0
        
        # Verify fallback options contain manual installation instructions
        fallback_text = " ".join(result.fallback_options)
        assert "Manual installation" in fallback_text
        assert "wan-pipeline" in fallback_text  # Wan-specific instruction

    def test_requirement_6_4_security_restrictions(self, dependency_manager):
        """
        Requirement 6.4: WHEN environment security policies restrict remote code 
        THEN the system SHALL provide local installation alternatives
        """
        # Test with trust_remote_code=False (security restriction)
        result = dependency_manager.fetch_pipeline_code(
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers", 
            trust_remote_code=False
        )
        
        assert not result.success
        assert "Remote code fetching disabled" in result.error_message
        assert result.fallback_options is not None
        
        # Verify local installation alternatives are provided
        fallback_text = " ".join(result.fallback_options)
        assert "local" in fallback_text.lower()
        assert "manual" in fallback_text.lower()

    def test_requirement_6_4_untrusted_source(self, dependency_manager):
        """
        Requirement 6.4: Test security validation for untrusted sources
        """
        dependency_manager.trust_mode = "safe"
        
        # Test with untrusted source
        result = dependency_manager.fetch_pipeline_code(
            "untrusted.com/malicious-model", 
            trust_remote_code=True
        )
        
        assert not result.success
        assert "Security validation failed" in result.error_message
        assert result.fallback_options is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])