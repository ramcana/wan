"""S3/MinIO storage backend with parallel downloads and resume capability."""

import os
import time
import logging
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any
from urllib.parse import urlparse
from dataclasses import dataclass

from .base_store import StorageBackend, DownloadResult
from ..credential_manager import SecureCredentialManager, CredentialConfig

logger = logging.getLogger(__name__)


@dataclass
class S3Config:
    """Configuration for S3/MinIO backend."""
    
    endpoint_url: Optional[str] = None  # Custom endpoint for MinIO
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    region_name: str = "us-east-1"
    max_concurrent_downloads: int = 4
    chunk_size: int = 8 * 1024 * 1024  # 8MB chunks
    max_retries: int = 3
    retry_backoff: float = 1.0


class S3Store(StorageBackend):
    """S3/MinIO storage backend with parallel downloads and resume capability."""

    def __init__(self, config: Optional[S3Config] = None, credential_manager: Optional[SecureCredentialManager] = None):
        """Initialize S3 store.
        
        Args:
            config: S3 configuration. If None, will use environment variables.
            credential_manager: Secure credential manager for handling credentials
        """
        self.config = config or S3Config()
        self.credential_manager = credential_manager or SecureCredentialManager()
        
        # Get credentials from secure credential manager first, then fallback to config/env
        credentials = self.credential_manager.get_credentials_for_source("s3://")
        
        self.config.endpoint_url = (
            credentials.get("endpoint_url") or 
            self.config.endpoint_url or 
            os.getenv("AWS_ENDPOINT_URL")
        )
        self.config.access_key_id = (
            credentials.get("access_key_id") or 
            self.config.access_key_id or 
            os.getenv("AWS_ACCESS_KEY_ID")
        )
        self.config.secret_access_key = (
            credentials.get("secret_access_key") or 
            self.config.secret_access_key or 
            os.getenv("AWS_SECRET_ACCESS_KEY")
        )
        self.config.region_name = os.getenv("AWS_DEFAULT_REGION", self.config.region_name)
        
        # Try to import boto3
        try:
            import boto3
            from botocore.exceptions import ClientError, NoCredentialsError
            self._boto3 = boto3
            self._ClientError = ClientError
            self._NoCredentialsError = NoCredentialsError
        except ImportError as e:
            raise ImportError(
                "boto3 is required for S3/MinIO storage backend. "
                "Install with: pip install boto3"
            ) from e
        
        # Initialize S3 client
        self._client = None
        self._client_lock = threading.Lock()

    def _get_client(self):
        """Get or create S3 client (thread-safe)."""
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    client_kwargs = {
                        "service_name": "s3",
                        "region_name": self.config.region_name,
                    }
                    
                    if self.config.endpoint_url:
                        client_kwargs["endpoint_url"] = self.config.endpoint_url
                        
                    if self.config.access_key_id and self.config.secret_access_key:
                        # Use secure credential context to temporarily set credentials
                        with self.credential_manager.get_secure_context("s3://"):
                            client_kwargs["aws_access_key_id"] = self.config.access_key_id
                            client_kwargs["aws_secret_access_key"] = self.config.secret_access_key
                    
                    self._client = self._boto3.client(**client_kwargs)
                    
        return self._client

    def can_handle(self, source_url: str) -> bool:
        """Check if this backend can handle S3 URLs."""
        return source_url.startswith("s3://")

    def _parse_s3_url(self, source_url: str) -> tuple[str, str]:
        """Parse S3 URL to extract bucket and key prefix.
        
        Args:
            source_url: URL in format s3://bucket/key/prefix
            
        Returns:
            Tuple of (bucket, key_prefix)
        """
        if not source_url.startswith("s3://"):
            raise ValueError(f"Invalid S3 URL: {source_url}")
        
        parsed = urlparse(source_url)
        bucket = parsed.netloc
        key_prefix = parsed.path.lstrip("/")
        
        return bucket, key_prefix

    def _list_objects(self, bucket: str, prefix: str, allow_patterns: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """List objects in S3 bucket with optional pattern filtering."""
        client = self._get_client()
        objects = []
        
        try:
            paginator = client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
            
            for page in page_iterator:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        
                        # Apply pattern filtering if specified
                        if allow_patterns:
                            import fnmatch
                            if not any(fnmatch.fnmatch(key, pattern) for pattern in allow_patterns):
                                continue
                        
                        objects.append({
                            'Key': key,
                            'Size': obj['Size'],
                            'ETag': obj['ETag'].strip('"'),
                            'LastModified': obj['LastModified']
                        })
                        
        except self._ClientError as e:
            logger.error(f"Error listing S3 objects: {e}")
            raise
            
        return objects

    def _download_file_with_resume(
        self,
        bucket: str,
        key: str,
        local_path: Path,
        expected_size: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> int:
        """Download a single file with resume capability using HTTP Range requests.
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            local_path: Local file path to download to
            expected_size: Expected file size for validation
            progress_callback: Optional progress callback
            
        Returns:
            Number of bytes downloaded
        """
        client = self._get_client()
        
        # Check if partial file exists
        start_byte = 0
        if local_path.exists():
            start_byte = local_path.stat().st_size
            logger.info(f"Resuming download from byte {start_byte} for {key}")
        
        # Get object metadata to check total size
        try:
            head_response = client.head_object(Bucket=bucket, Key=key)
            total_size = head_response['ContentLength']
            etag = head_response['ETag'].strip('"')
            
            # If file is already complete, verify and return
            if start_byte == total_size:
                if expected_size and total_size != expected_size:
                    raise ValueError(f"Size mismatch for {key}: expected {expected_size}, got {total_size}")
                logger.info(f"File {key} already complete, skipping download")
                return 0
                
        except self._ClientError as e:
            logger.error(f"Error getting object metadata for {key}: {e}")
            raise
        
        # Ensure parent directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download remaining bytes
        bytes_downloaded = 0
        retry_count = 0
        
        while retry_count < self.config.max_retries:
            try:
                # Prepare range request
                range_header = None
                if start_byte < total_size:
                    range_header = f"bytes={start_byte}-"
                
                # Download with range request
                download_kwargs = {"Bucket": bucket, "Key": key}
                if range_header:
                    download_kwargs["Range"] = range_header
                
                response = client.get_object(**download_kwargs)
                
                # Open file in append mode if resuming, write mode if starting fresh
                mode = "ab" if start_byte > 0 else "wb"
                
                with open(local_path, mode) as f:
                    while True:
                        chunk = response['Body'].read(self.config.chunk_size)
                        if not chunk:
                            break
                        
                        f.write(chunk)
                        bytes_downloaded += len(chunk)
                        
                        # Call progress callback if provided
                        if progress_callback:
                            progress_callback(start_byte + bytes_downloaded, total_size)
                
                # Verify final file size
                final_size = local_path.stat().st_size
                if final_size != total_size:
                    raise ValueError(f"Download incomplete for {key}: expected {total_size}, got {final_size}")
                
                logger.info(f"Successfully downloaded {key}: {bytes_downloaded} bytes")
                return bytes_downloaded
                
            except Exception as e:
                retry_count += 1
                if retry_count >= self.config.max_retries:
                    logger.error(f"Failed to download {key} after {self.config.max_retries} retries: {e}")
                    raise
                
                # Exponential backoff
                wait_time = self.config.retry_backoff * (2 ** (retry_count - 1))
                logger.warning(f"Download failed for {key}, retrying in {wait_time}s (attempt {retry_count})")
                time.sleep(wait_time)
                
                # Update start_byte for resume
                if local_path.exists():
                    start_byte = local_path.stat().st_size
        
        return bytes_downloaded

    def download(
        self,
        source_url: str,
        local_dir: Path,
        file_specs: Optional[List] = None,
        allow_patterns: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> DownloadResult:
        """Download files from S3/MinIO with parallel downloads and resume capability.
        
        Args:
            source_url: S3 URL (s3://bucket/prefix)
            local_dir: Local directory to download to
            file_specs: List of FileSpec objects for validation
            allow_patterns: List of file patterns to download
            progress_callback: Optional progress callback
            
        Returns:
            DownloadResult with download statistics
        """
        start_time = time.time()
        
        try:
            bucket, key_prefix = self._parse_s3_url(source_url)
            
            logger.info(f"Starting S3 download: {source_url} -> {local_dir}")
            
            # List objects to download
            objects = self._list_objects(bucket, key_prefix, allow_patterns)
            
            if not objects:
                logger.warning(f"No objects found at {source_url}")
                return DownloadResult(
                    success=True,
                    bytes_downloaded=0,
                    files_downloaded=0,
                    duration_seconds=time.time() - start_time
                )
            
            # Create file specs lookup for validation
            file_specs_dict = {}
            if file_specs:
                for spec in file_specs:
                    file_specs_dict[spec.path] = spec
            
            # Create local directory
            local_dir.mkdir(parents=True, exist_ok=True)
            
            # Download files in parallel
            total_bytes_downloaded = 0
            files_downloaded = 0
            download_errors = []
            
            with ThreadPoolExecutor(max_workers=self.config.max_concurrent_downloads) as executor:
                # Submit download tasks
                future_to_object = {}
                
                for obj in objects:
                    key = obj['Key']
                    size = obj['Size']
                    
                    # Calculate relative path (remove prefix)
                    if key.startswith(key_prefix):
                        relative_path = key[len(key_prefix):].lstrip("/")
                    else:
                        relative_path = key
                    
                    local_path = local_dir / relative_path
                    
                    # Get expected size from file specs if available
                    expected_size = None
                    if relative_path in file_specs_dict:
                        expected_size = file_specs_dict[relative_path].size
                    
                    future = executor.submit(
                        self._download_file_with_resume,
                        bucket,
                        key,
                        local_path,
                        expected_size,
                        progress_callback
                    )
                    future_to_object[future] = (key, relative_path, size)
                
                # Collect results
                for future in as_completed(future_to_object):
                    key, relative_path, expected_size = future_to_object[future]
                    
                    try:
                        bytes_downloaded = future.result()
                        total_bytes_downloaded += bytes_downloaded
                        files_downloaded += 1
                        
                        logger.debug(f"Downloaded {key}: {bytes_downloaded} bytes")
                        
                    except Exception as e:
                        error_msg = f"Failed to download {key}: {e}"
                        logger.error(error_msg)
                        download_errors.append(error_msg)
            
            # Check if any downloads failed
            if download_errors:
                error_message = f"Some downloads failed: {'; '.join(download_errors)}"
                return DownloadResult(
                    success=False,
                    bytes_downloaded=total_bytes_downloaded,
                    files_downloaded=files_downloaded,
                    error_message=error_message,
                    duration_seconds=time.time() - start_time
                )
            
            duration = time.time() - start_time
            logger.info(
                f"S3 download completed: {files_downloaded} files, "
                f"{total_bytes_downloaded} bytes in {duration:.2f}s"
            )
            
            return DownloadResult(
                success=True,
                bytes_downloaded=total_bytes_downloaded,
                files_downloaded=files_downloaded,
                duration_seconds=duration
            )
            
        except Exception as e:
            error_msg = f"S3 download failed: {e}"
            logger.error(error_msg)
            return DownloadResult(
                success=False,
                bytes_downloaded=0,
                files_downloaded=0,
                error_message=error_msg,
                duration_seconds=time.time() - start_time
            )

    def verify_availability(self, source_url: str) -> bool:
        """Verify if the S3 source is available."""
        try:
            bucket, key_prefix = self._parse_s3_url(source_url)
            client = self._get_client()
            
            # Try to list objects with the prefix to verify access
            response = client.list_objects_v2(
                Bucket=bucket,
                Prefix=key_prefix,
                MaxKeys=1
            )
            
            # If we can list objects, the source is available
            return True
            
        except self._NoCredentialsError:
            logger.warning("No AWS credentials configured for S3 access")
            return False
        except self._ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == 'NoSuchBucket':
                logger.warning(f"S3 bucket not found: {bucket}")
            elif error_code == 'AccessDenied':
                logger.warning(f"Access denied to S3 bucket: {bucket}")
            else:
                logger.warning(f"S3 access error: {e}")
            return False
        except Exception as e:
            logger.warning(f"Error checking S3 availability: {e}")
            return False

    def estimate_download_size(
        self,
        source_url: str,
        file_specs: Optional[List] = None,
        allow_patterns: Optional[List[str]] = None
    ) -> int:
        """Estimate download size from S3 source."""
        try:
            bucket, key_prefix = self._parse_s3_url(source_url)
            
            # List objects to get their sizes
            objects = self._list_objects(bucket, key_prefix, allow_patterns)
            
            total_size = sum(obj['Size'] for obj in objects)
            
            logger.info(f"Estimated download size for {source_url}: {total_size} bytes")
            return total_size
            
        except Exception as e:
            logger.warning(f"Error estimating S3 download size: {e}")
            return 0