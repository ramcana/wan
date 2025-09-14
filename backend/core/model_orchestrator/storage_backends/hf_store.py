"""HuggingFace Hub storage backend."""

import os
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Callable, Dict
from urllib.parse import urlparse

from .base_store import StorageBackend, DownloadResult
from ..credential_manager import SecureCredentialManager, CredentialConfig

logger = logging.getLogger(__name__)


@dataclass
class HFFileMetadata:
    """HuggingFace file metadata for integrity verification."""
    etag: str
    size: int
    last_modified: Optional[str] = None


class HFStore(StorageBackend):
    """HuggingFace Hub storage backend using huggingface_hub."""

    def __init__(self, token: Optional[str] = None, enable_hf_transfer: bool = True, credential_manager: Optional[SecureCredentialManager] = None):
        """Initialize HuggingFace store.
        
        Args:
            token: HuggingFace API token (if None, will try to get from credential manager or environment)
            enable_hf_transfer: Whether to enable hf_transfer for faster downloads
            credential_manager: Secure credential manager for handling credentials
        """
        self.credential_manager = credential_manager or SecureCredentialManager()
        
        # Get token from secure credential manager first, then fallback to parameter/env
        credentials = self.credential_manager.get_credentials_for_source("hf://")
        self.token = credentials.get("token") or token or os.getenv("HF_TOKEN")
        
        self.enable_hf_transfer = enable_hf_transfer
        
        # Try to import required dependencies
        try:
            from huggingface_hub import snapshot_download, repo_info, hf_hub_download
            from huggingface_hub.utils import HfHubHTTPError
            from huggingface_hub.file_download import http_get
            self._snapshot_download = snapshot_download
            self._repo_info = repo_info
            self._hf_hub_download = hf_hub_download
            self._http_get = http_get
            self._HfHubHTTPError = HfHubHTTPError
        except ImportError as e:
            raise ImportError(
                "huggingface_hub is required for HuggingFace storage backend. "
                "Install with: pip install huggingface_hub"
            ) from e
        
        # Check if hf_transfer is available and enabled
        self._hf_transfer_available = False
        self._hf_transfer_warning_logged = False
        if enable_hf_transfer:
            try:
                import hf_transfer  # noqa: F401
                self._hf_transfer_available = True
                # Set environment variable to enable hf_transfer
                os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
                logger.info("hf_transfer enabled for faster downloads")
            except ImportError:
                # Log once that hf_transfer is unavailable, then continue without it
                if not self._hf_transfer_warning_logged:
                    logger.warning(
                        "hf_transfer not available, falling back to standard downloads. "
                        "Install hf_transfer for faster multi-connection downloads: pip install hf_transfer"
                    )
                    self._hf_transfer_warning_logged = True

    def can_handle(self, source_url: str) -> bool:
        """Check if this backend can handle HuggingFace URLs."""
        return source_url.startswith("hf://")

    def _parse_hf_url(self, source_url: str) -> tuple[str, Optional[str]]:
        """Parse HuggingFace URL to extract repo_id and revision.
        
        Args:
            source_url: URL in format hf://repo_id or hf://repo_id@revision
            
        Returns:
            Tuple of (repo_id, revision)
        """
        if not source_url.startswith("hf://"):
            raise ValueError(f"Invalid HuggingFace URL: {source_url}")
        
        # Remove hf:// prefix
        repo_path = source_url[5:]
        
        # Check for revision (after @)
        if "@" in repo_path:
            repo_id, revision = repo_path.rsplit("@", 1)
        else:
            repo_id, revision = repo_path, None
            
        return repo_id, revision

    def download(
        self,
        source_url: str,
        local_dir: Path,
        file_specs: Optional[List] = None,
        allow_patterns: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> DownloadResult:
        """Download model from HuggingFace Hub.
        
        Args:
            source_url: HuggingFace URL (hf://repo_id or hf://repo_id@revision)
            local_dir: Local directory to download to
            file_specs: List of FileSpec objects (not used for HF, patterns used instead)
            allow_patterns: List of file patterns to download
            progress_callback: Optional progress callback
            
        Returns:
            DownloadResult with download statistics
        """
        start_time = time.time()
        
        try:
            repo_id, revision = self._parse_hf_url(source_url)
            
            logger.info(
                f"Starting HuggingFace download: {repo_id}"
                f"{f'@{revision}' if revision else ''} -> {local_dir}"
            )
            
            # Prepare download arguments
            download_kwargs = {
                "repo_id": repo_id,
                "local_dir": local_dir,
                "local_dir_use_symlinks": False,  # Use actual files, not symlinks
                "token": self.token,
                "resume_download": True,  # Enable resume for large files
            }
            
            if revision:
                download_kwargs["revision"] = revision
                
            if allow_patterns:
                download_kwargs["allow_patterns"] = allow_patterns
                logger.info(f"Using allow_patterns: {allow_patterns}")
            
            # Create local directory if it doesn't exist
            local_dir.mkdir(parents=True, exist_ok=True)
            
            # Perform the download
            downloaded_path = self._snapshot_download(**download_kwargs)
            
            # Calculate statistics
            duration = time.time() - start_time
            bytes_downloaded = self._calculate_directory_size(Path(downloaded_path))
            files_downloaded = self._count_files(Path(downloaded_path))
            
            logger.info(
                f"HuggingFace download completed: {files_downloaded} files, "
                f"{bytes_downloaded} bytes in {duration:.2f}s"
            )
            
            # Extract metadata for integrity verification
            metadata = self._extract_file_metadata(Path(downloaded_path))
            
            return DownloadResult(
                success=True,
                bytes_downloaded=bytes_downloaded,
                files_downloaded=files_downloaded,
                duration_seconds=duration,
                metadata=metadata
            )
            
        except self._HfHubHTTPError as e:
            error_msg = f"HuggingFace Hub error: {e}"
            logger.error(error_msg)
            return DownloadResult(
                success=False,
                bytes_downloaded=0,
                files_downloaded=0,
                error_message=error_msg,
                duration_seconds=time.time() - start_time
            )
        except Exception as e:
            error_msg = f"Download failed: {e}"
            logger.error(error_msg)
            return DownloadResult(
                success=False,
                bytes_downloaded=0,
                files_downloaded=0,
                error_message=error_msg,
                duration_seconds=time.time() - start_time
            )

    def verify_availability(self, source_url: str) -> bool:
        """Verify if the HuggingFace repository is available."""
        try:
            repo_id, revision = self._parse_hf_url(source_url)
            
            # Use repo_info to check if repository exists and is accessible
            self._repo_info(repo_id, revision=revision, token=self.token)
            return True
            
        except self._HfHubHTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Repository not found: {repo_id}")
            elif e.response.status_code == 401:
                logger.warning(f"Authentication required for: {repo_id}")
            else:
                logger.warning(f"Repository access error: {e}")
            return False
        except Exception as e:
            logger.warning(f"Error checking repository availability: {e}")
            return False

    def estimate_download_size(
        self,
        source_url: str,
        file_specs: Optional[List] = None,
        allow_patterns: Optional[List[str]] = None
    ) -> int:
        """Estimate download size from HuggingFace repository.
        
        Note: This is a rough estimate as HuggingFace doesn't provide
        easy access to file sizes without downloading. Returns 0 for now.
        """
        try:
            repo_id, revision = self._parse_hf_url(source_url)
            
            # For now, we can't easily get file sizes from HF without downloading
            # This would require using the HF API to list files and their sizes
            # which is more complex and not always available
            logger.info(f"Size estimation not available for HuggingFace repo: {repo_id}")
            return 0
            
        except Exception as e:
            logger.warning(f"Error estimating download size: {e}")
            return 0

    def _calculate_directory_size(self, directory: Path) -> int:
        """Calculate total size of all files in directory."""
        total_size = 0
        try:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception as e:
            logger.warning(f"Error calculating directory size: {e}")
        return total_size

    def _count_files(self, directory: Path) -> int:
        """Count total number of files in directory."""
        try:
            return len([f for f in directory.rglob("*") if f.is_file()])
        except Exception as e:
            logger.warning(f"Error counting files: {e}")
            return 0
    
    def _extract_file_metadata(self, directory: Path) -> Dict[str, HFFileMetadata]:
        """
        Extract file metadata from downloaded HuggingFace files.
        
        This attempts to extract ETags and other metadata that can be used
        for integrity verification when SHA256 checksums are not available.
        """
        metadata = {}
        
        try:
            # Look for HuggingFace cache metadata files
            # HF typically stores metadata in .cache files or similar
            for file_path in directory.rglob("*"):
                if file_path.is_file() and not file_path.name.startswith('.'):
                    try:
                        stat = file_path.stat()
                        
                        # Generate ETag equivalent (MD5 for simple files)
                        import hashlib
                        md5_hash = hashlib.md5()
                        with open(file_path, "rb") as f:
                            for chunk in iter(lambda: f.read(8192), b""):
                                md5_hash.update(chunk)
                        etag = md5_hash.hexdigest()
                        
                        # Create relative path from directory
                        relative_path = file_path.relative_to(directory)
                        
                        metadata[str(relative_path)] = HFFileMetadata(
                            etag=etag,
                            size=stat.st_size,
                            last_modified=time.ctime(stat.st_mtime)
                        )
                        
                    except Exception as e:
                        logger.debug(f"Failed to extract metadata for {file_path}: {e}")
                        
        except Exception as e:
            logger.warning(f"Error extracting file metadata: {e}")
        
        return metadata
    
    def get_file_metadata(
        self,
        source_url: str,
        file_path: str
    ) -> Optional[HFFileMetadata]:
        """
        Get metadata for a specific file from HuggingFace Hub.
        
        This can be used to get ETag information before downloading
        for integrity verification purposes.
        """
        try:
            repo_id, revision = self._parse_hf_url(source_url)
            
            # Use HF API to get file info
            # This is a simplified implementation - in practice you'd use
            # the HF API to get actual ETag/metadata
            from huggingface_hub import HfApi
            api = HfApi(token=self.token)
            
            # Get repository info which includes file metadata
            repo_files = api.list_repo_files(repo_id, revision=revision)
            
            if file_path in repo_files:
                # In a real implementation, you'd get the actual ETag from the API
                # For now, we'll return None to indicate metadata not available
                return None
                
        except Exception as e:
            logger.debug(f"Failed to get file metadata: {e}")
            
        return None