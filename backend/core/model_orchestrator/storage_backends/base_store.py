"""Base storage backend interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Callable
from pathlib import Path


@dataclass
class DownloadResult:
    """Result of a download operation."""
    success: bool
    bytes_downloaded: int
    files_downloaded: int
    error_message: Optional[str] = None
    duration_seconds: Optional[float] = None
    metadata: Optional[dict] = None  # Additional metadata (e.g., ETags, checksums)


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def can_handle(self, source_url: str) -> bool:
        """Check if this backend can handle the given source URL."""
        pass

    @abstractmethod
    def download(
        self,
        source_url: str,
        local_dir: Path,
        file_specs: Optional[List] = None,
        allow_patterns: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> DownloadResult:
        """Download files from source to local directory."""
        pass

    @abstractmethod
    def verify_availability(self, source_url: str) -> bool:
        """Verify if the source is available."""
        pass

    @abstractmethod
    def estimate_download_size(
        self,
        source_url: str,
        file_specs: Optional[List] = None,
        allow_patterns: Optional[List[str]] = None
    ) -> int:
        """Estimate the total download size in bytes."""
        pass