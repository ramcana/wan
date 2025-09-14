"""Tests for HuggingFace storage backend."""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil

from .hf_store import HFStore
from .base_store import DownloadResult


class TestHFStore:
    """Test cases for HFStore."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @patch('huggingface_hub.snapshot_download')
    @patch('huggingface_hub.repo_info')
    def test_can_handle_hf_urls(self, mock_repo_info, mock_snapshot_download):
        """Test that HFStore can handle HuggingFace URLs."""
        store = HFStore()
        
        assert store.can_handle("hf://microsoft/DialoGPT-medium")
        assert store.can_handle("hf://microsoft/DialoGPT-medium@main")
        assert not store.can_handle("https://example.com/model")
        assert not store.can_handle("s3://bucket/model")
        assert not store.can_handle("local://path/to/model")

    @patch('huggingface_hub.snapshot_download')
    @patch('huggingface_hub.repo_info')
    def test_parse_hf_url(self, mock_repo_info, mock_snapshot_download):
        """Test parsing of HuggingFace URLs."""
        store = HFStore()
        
        # Test URL without revision
        repo_id, revision = store._parse_hf_url("hf://microsoft/DialoGPT-medium")
        assert repo_id == "microsoft/DialoGPT-medium"
        assert revision is None
        
        # Test URL with revision
        repo_id, revision = store._parse_hf_url("hf://microsoft/DialoGPT-medium@main")
        assert repo_id == "microsoft/DialoGPT-medium"
        assert revision == "main"
        
        # Test URL with commit hash
        repo_id, revision = store._parse_hf_url("hf://microsoft/DialoGPT-medium@abc123")
        assert repo_id == "microsoft/DialoGPT-medium"
        assert revision == "abc123"

    @patch('huggingface_hub.snapshot_download')
    @patch('huggingface_hub.repo_info')
    def test_parse_hf_url_invalid(self, mock_repo_info, mock_snapshot_download):
        """Test parsing of invalid HuggingFace URLs."""
        store = HFStore()
        
        with pytest.raises(ValueError, match="Invalid HuggingFace URL"):
            store._parse_hf_url("https://example.com/model")

    @patch('huggingface_hub.snapshot_download')
    @patch('huggingface_hub.repo_info')
    def test_download_success(self, mock_repo_info, mock_snapshot_download):
        """Test successful download from HuggingFace."""
        # Create mock downloaded directory with some files
        mock_download_dir = self.temp_dir / "downloaded"
        mock_download_dir.mkdir()
        (mock_download_dir / "model.bin").write_bytes(b"fake model data" * 100)
        (mock_download_dir / "config.json").write_text('{"model_type": "test"}')
        
        mock_snapshot_download.return_value = str(mock_download_dir)
        
        store = HFStore(token="fake_token")
        
        result = store.download(
            source_url="hf://microsoft/DialoGPT-medium",
            local_dir=self.temp_dir / "target",
            allow_patterns=["*.bin", "*.json"]
        )
        
        assert result.success is True
        assert result.bytes_downloaded > 0
        assert result.files_downloaded == 2
        assert result.duration_seconds is not None
        assert result.error_message is None
        
        # Verify snapshot_download was called with correct arguments
        mock_snapshot_download.assert_called_once()
        call_args = mock_snapshot_download.call_args[1]
        assert call_args["repo_id"] == "microsoft/DialoGPT-medium"
        assert call_args["local_dir"] == self.temp_dir / "target"
        assert call_args["allow_patterns"] == ["*.bin", "*.json"]
        assert call_args["token"] == "fake_token"
        assert call_args["local_dir_use_symlinks"] is False

    @patch('huggingface_hub.snapshot_download')
    @patch('huggingface_hub.repo_info')
    def test_download_with_revision(self, mock_repo_info, mock_snapshot_download):
        """Test download with specific revision."""
        mock_download_dir = self.temp_dir / "downloaded"
        mock_download_dir.mkdir()
        (mock_download_dir / "model.bin").write_bytes(b"fake model data")
        
        mock_snapshot_download.return_value = str(mock_download_dir)
        
        store = HFStore()
        
        result = store.download(
            source_url="hf://microsoft/DialoGPT-medium@v1.0",
            local_dir=self.temp_dir / "target"
        )
        
        assert result.success is True
        
        # Verify revision was passed
        call_args = mock_snapshot_download.call_args[1]
        assert call_args["revision"] == "v1.0"

    @patch('huggingface_hub.snapshot_download')
    @patch('huggingface_hub.repo_info')
    def test_download_hf_hub_error(self, mock_repo_info, mock_snapshot_download):
        """Test handling of HuggingFace Hub errors."""
        from huggingface_hub.utils import HfHubHTTPError
        
        # Mock HTTP error response
        mock_response = Mock()
        mock_response.status_code = 404
        error = HfHubHTTPError("Repository not found", response=mock_response)
        mock_snapshot_download.side_effect = error
        
        store = HFStore()
        
        result = store.download(
            source_url="hf://nonexistent/model",
            local_dir=self.temp_dir / "target"
        )
        
        assert result.success is False
        assert result.bytes_downloaded == 0
        assert result.files_downloaded == 0
        assert "HuggingFace Hub error" in result.error_message
        assert result.duration_seconds is not None

    @patch('huggingface_hub.snapshot_download')
    @patch('huggingface_hub.repo_info')
    def test_download_generic_error(self, mock_repo_info, mock_snapshot_download):
        """Test handling of generic errors during download."""
        mock_snapshot_download.side_effect = Exception("Network error")
        
        store = HFStore()
        
        result = store.download(
            source_url="hf://microsoft/DialoGPT-medium",
            local_dir=self.temp_dir / "target"
        )
        
        assert result.success is False
        assert result.bytes_downloaded == 0
        assert result.files_downloaded == 0
        assert "Download failed: Network error" in result.error_message

    @patch('huggingface_hub.snapshot_download')
    @patch('huggingface_hub.repo_info')
    def test_verify_availability_success(self, mock_repo_info, mock_snapshot_download):
        """Test successful repository availability check."""
        mock_repo_info.return_value = {"id": "microsoft/DialoGPT-medium"}
        
        store = HFStore(token="fake_token")
        
        result = store.verify_availability("hf://microsoft/DialoGPT-medium")
        
        assert result is True
        mock_repo_info.assert_called_once_with(
            "microsoft/DialoGPT-medium",
            revision=None,
            token="fake_token"
        )

    @patch('huggingface_hub.snapshot_download')
    @patch('huggingface_hub.repo_info')
    def test_verify_availability_with_revision(self, mock_repo_info, mock_snapshot_download):
        """Test repository availability check with revision."""
        mock_repo_info.return_value = {"id": "microsoft/DialoGPT-medium"}
        
        store = HFStore()
        
        result = store.verify_availability("hf://microsoft/DialoGPT-medium@main")
        
        assert result is True
        mock_repo_info.assert_called_once_with(
            "microsoft/DialoGPT-medium",
            revision="main",
            token=None
        )

    @patch('huggingface_hub.snapshot_download')
    @patch('huggingface_hub.repo_info')
    def test_verify_availability_not_found(self, mock_repo_info, mock_snapshot_download):
        """Test repository availability check for non-existent repo."""
        from huggingface_hub.utils import HfHubHTTPError
        
        mock_response = Mock()
        mock_response.status_code = 404
        error = HfHubHTTPError("Repository not found", response=mock_response)
        mock_repo_info.side_effect = error
        
        store = HFStore()
        
        result = store.verify_availability("hf://nonexistent/model")
        
        assert result is False

    @patch('huggingface_hub.snapshot_download')
    @patch('huggingface_hub.repo_info')
    def test_verify_availability_auth_required(self, mock_repo_info, mock_snapshot_download):
        """Test repository availability check for private repo."""
        from huggingface_hub.utils import HfHubHTTPError
        
        mock_response = Mock()
        mock_response.status_code = 401
        error = HfHubHTTPError("Authentication required", response=mock_response)
        mock_repo_info.side_effect = error
        
        store = HFStore()
        
        result = store.verify_availability("hf://private/model")
        
        assert result is False

    @patch('huggingface_hub.snapshot_download')
    @patch('huggingface_hub.repo_info')
    def test_estimate_download_size(self, mock_repo_info, mock_snapshot_download):
        """Test download size estimation."""
        store = HFStore()
        
        # Currently returns 0 as size estimation is not implemented
        size = store.estimate_download_size("hf://microsoft/DialoGPT-medium")
        assert size == 0

    @patch('huggingface_hub.snapshot_download')
    @patch('huggingface_hub.repo_info')
    def test_calculate_directory_size(self, mock_repo_info, mock_snapshot_download):
        """Test directory size calculation."""
        store = HFStore()
        
        # Create test directory with files
        test_dir = self.temp_dir / "test"
        test_dir.mkdir()
        (test_dir / "file1.txt").write_bytes(b"hello" * 100)  # 500 bytes
        (test_dir / "file2.txt").write_bytes(b"world" * 200)  # 1000 bytes
        
        subdir = test_dir / "subdir"
        subdir.mkdir()
        (subdir / "file3.txt").write_bytes(b"test" * 50)  # 200 bytes
        
        total_size = store._calculate_directory_size(test_dir)
        assert total_size == 1700  # 500 + 1000 + 200

    @patch('huggingface_hub.snapshot_download')
    @patch('huggingface_hub.repo_info')
    def test_count_files(self, mock_repo_info, mock_snapshot_download):
        """Test file counting."""
        store = HFStore()
        
        # Create test directory with files
        test_dir = self.temp_dir / "test"
        test_dir.mkdir()
        (test_dir / "file1.txt").write_text("content1")
        (test_dir / "file2.txt").write_text("content2")
        
        subdir = test_dir / "subdir"
        subdir.mkdir()
        (subdir / "file3.txt").write_text("content3")
        
        file_count = store._count_files(test_dir)
        assert file_count == 3

    @patch.dict(os.environ, {"HF_TOKEN": "env_token"})
    @patch('huggingface_hub.snapshot_download')
    @patch('huggingface_hub.repo_info')
    def test_token_from_environment(self, mock_repo_info, mock_snapshot_download):
        """Test token loading from environment variable."""
        store = HFStore()
        assert store.token == "env_token"

    @patch('huggingface_hub.snapshot_download')
    @patch('huggingface_hub.repo_info')
    def test_token_from_parameter(self, mock_repo_info, mock_snapshot_download):
        """Test token from constructor parameter."""
        store = HFStore(token="param_token")
        assert store.token == "param_token"

    @patch.dict(os.environ, {}, clear=True)
    @patch('huggingface_hub.snapshot_download')
    @patch('huggingface_hub.repo_info')
    def test_no_token(self, mock_repo_info, mock_snapshot_download):
        """Test behavior when no token is provided."""
        store = HFStore()
        assert store.token is None

    def test_missing_huggingface_hub(self):
        """Test error when huggingface_hub is not installed."""
        with patch.dict('sys.modules', {'huggingface_hub': None}):
            with pytest.raises(ImportError, match="huggingface_hub is required"):
                HFStore()

    @patch('huggingface_hub.snapshot_download')
    @patch('huggingface_hub.repo_info')
    def test_hf_transfer_disabled(self, mock_repo_info, mock_snapshot_download):
        """Test behavior when hf_transfer is explicitly disabled."""
        store = HFStore(enable_hf_transfer=False)
        assert store._hf_transfer_available is False
    
    @patch('huggingface_hub.snapshot_download')
    @patch('huggingface_hub.repo_info')
    def test_hf_transfer_fallback_warning(self, mock_repo_info, mock_snapshot_download):
        """Test that warning is logged once when hf_transfer is unavailable."""
        # Capture log messages
        with patch('backend.core.model_orchestrator.storage_backends.hf_store.logger') as mock_logger:
            # Mock the import statement inside the try block to fail
            with patch('sys.modules', {'hf_transfer': None}):
                # Remove hf_transfer from sys.modules if it exists
                import sys
                if 'hf_transfer' in sys.modules:
                    del sys.modules['hf_transfer']
                
                store = HFStore(enable_hf_transfer=True)
                
                # Verify warning was logged
                mock_logger.warning.assert_called_once()
                warning_call = mock_logger.warning.call_args[0][0]
                assert "hf_transfer not available" in warning_call
                assert "falling back to standard downloads" in warning_call
                assert "pip install hf_transfer" in warning_call
                
                # Verify hf_transfer is marked as unavailable
                assert store._hf_transfer_available is False
                assert store._hf_transfer_warning_logged is True
    
    @patch('huggingface_hub.snapshot_download')
    @patch('huggingface_hub.repo_info')
    def test_hf_transfer_available(self, mock_repo_info, mock_snapshot_download):
        """Test behavior when hf_transfer is available."""
        # Mock successful hf_transfer import
        mock_hf_transfer = Mock()
        with patch.dict('sys.modules', {'hf_transfer': mock_hf_transfer}):
            with patch('backend.core.model_orchestrator.storage_backends.hf_store.logger') as mock_logger:
                store = HFStore(enable_hf_transfer=True)
                
                # Verify success message was logged
                mock_logger.info.assert_called_with("hf_transfer enabled for faster downloads")
                
                # Verify hf_transfer is marked as available
                assert store._hf_transfer_available is True
                assert store._hf_transfer_warning_logged is False
    
    @patch('huggingface_hub.snapshot_download')
    @patch('huggingface_hub.repo_info')
    def test_hf_transfer_environment_variable_set(self, mock_repo_info, mock_snapshot_download):
        """Test that HF_HUB_ENABLE_HF_TRANSFER environment variable is set when hf_transfer is available."""
        # Mock successful hf_transfer import
        mock_hf_transfer = Mock()
        with patch.dict('sys.modules', {'hf_transfer': mock_hf_transfer}):
            with patch.dict(os.environ, {}, clear=True):
                store = HFStore(enable_hf_transfer=True)
                
                # Verify environment variable was set
                assert os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1"
    
    @patch('huggingface_hub.snapshot_download')
    @patch('huggingface_hub.repo_info')
    def test_download_continues_without_hf_transfer(self, mock_repo_info, mock_snapshot_download):
        """Test that download continues normally even when hf_transfer is unavailable."""
        # Create mock downloaded directory
        mock_download_dir = self.temp_dir / "downloaded"
        mock_download_dir.mkdir()
        (mock_download_dir / "model.bin").write_bytes(b"fake model data")
        
        mock_snapshot_download.return_value = str(mock_download_dir)
        
        # Mock hf_transfer import failure
        with patch('sys.modules', {'hf_transfer': None}):
            import sys
            if 'hf_transfer' in sys.modules:
                del sys.modules['hf_transfer']
            
            store = HFStore(enable_hf_transfer=True)
            
            # Download should still work
            result = store.download(
                source_url="hf://microsoft/DialoGPT-medium",
                local_dir=self.temp_dir / "target"
            )
            
            assert result.success is True
            assert result.bytes_downloaded > 0
            assert store._hf_transfer_available is False