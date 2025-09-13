"""
Network Failure Recovery System for WAN2.2 Installation.

This module provides comprehensive network failure recovery capabilities including:
- Authentication and rate-limiting detection
- Alternative download sources and mirror selection
- Resume capability for partial downloads
- Proxy and authentication configuration handling
- Network connectivity testing and timeout management
- Intelligent retry strategies for network operations

Requirements addressed: 1.3, 6.3, 6.4
"""

import logging
import time
import os
import socket
import urllib.request
import urllib.parse
import urllib.error
import ssl
import json
import hashlib
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
import threading
import queue
import tempfile
import shutil

from interfaces import ErrorCategory, ValidationResult
from base_classes import BaseInstallationComponent


class NetworkErrorType(Enum):
    """Types of network errors for specific handling."""
    CONNECTION_TIMEOUT = "connection_timeout"
    READ_TIMEOUT = "read_timeout"
    CONNECTION_REFUSED = "connection_refused"
    DNS_RESOLUTION = "dns_resolution"
    SSL_CERTIFICATE = "ssl_certificate"
    AUTHENTICATION = "authentication"
    RATE_LIMITING = "rate_limiting"
    PROXY_ERROR = "proxy_error"
    BANDWIDTH_LIMIT = "bandwidth_limit"
    SERVER_ERROR = "server_error"
    PARTIAL_DOWNLOAD = "partial_download"
    UNKNOWN = "unknown"


class DownloadSource(Enum):
    """Available download sources."""
    PRIMARY = "primary"
    MIRROR_1 = "mirror_1"
    MIRROR_2 = "mirror_2"
    CDN = "cdn"
    BACKUP = "backup"
    LOCAL_CACHE = "local_cache"


@dataclass
class NetworkConfiguration:
    """Network configuration settings."""
    connect_timeout: int = 30
    read_timeout: int = 300
    max_retries: int = 3
    retry_delay: float = 2.0
    max_retry_delay: float = 60.0
    backoff_multiplier: float = 2.0
    chunk_size: int = 8192
    resume_threshold: int = 1024 * 1024  # 1MB
    bandwidth_limit_mbps: Optional[float] = None
    user_agent: str = "WAN2.2-Installer/1.0"
    verify_ssl: bool = True
    follow_redirects: bool = True
    max_redirects: int = 5


@dataclass
class ProxyConfiguration:
    """Proxy configuration settings."""
    http_proxy: Optional[str] = None
    https_proxy: Optional[str] = None
    no_proxy: Optional[str] = None
    proxy_auth: Optional[Tuple[str, str]] = None
    auto_detect: bool = True


@dataclass
class AuthenticationConfiguration:
    """Authentication configuration for downloads."""
    username: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None
    api_key: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class DownloadMirror:
    """Configuration for a download mirror."""
    name: str
    base_url: str
    priority: int
    auth_config: Optional[AuthenticationConfiguration] = None
    proxy_config: Optional[ProxyConfiguration] = None
    timeout_multiplier: float = 1.0
    available: bool = True
    last_check: Optional[datetime] = None
    failure_count: int = 0
    success_count: int = 0


@dataclass
class DownloadProgress:
    """Progress information for a download."""
    url: str
    filename: str
    total_size: int
    downloaded_size: int
    start_time: datetime
    current_speed_mbps: float
    average_speed_mbps: float
    eta_seconds: Optional[int]
    resume_count: int = 0
    source: DownloadSource = DownloadSource.PRIMARY


@dataclass
class NetworkTestResult:
    """Result of network connectivity test."""
    success: bool
    latency_ms: float
    bandwidth_mbps: float
    error_message: Optional[str] = None
    test_duration: float = 0.0
    proxy_working: bool = False
    dns_working: bool = False


class NetworkErrorDetector:
    """Detects and classifies network errors."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Error patterns for classification
        self.error_patterns = {
            NetworkErrorType.CONNECTION_TIMEOUT: [
                "connection timeout", "timed out", "timeout error",
                "connection timed out", "read timeout"
            ],
            NetworkErrorType.CONNECTION_REFUSED: [
                "connection refused", "connection reset", "connection aborted",
                "no route to host", "network unreachable"
            ],
            NetworkErrorType.DNS_RESOLUTION: [
                "name resolution failed", "dns", "hostname", "getaddrinfo",
                "name or service not known", "nodename nor servname provided"
            ],
            NetworkErrorType.SSL_CERTIFICATE: [
                "ssl", "certificate", "cert", "tls", "handshake",
                "certificate verify failed", "ssl certificate problem"
            ],
            NetworkErrorType.AUTHENTICATION: [
                "401", "unauthorized", "authentication", "login",
                "invalid credentials", "access denied", "forbidden"
            ],
            NetworkErrorType.RATE_LIMITING: [
                "429", "rate limit", "too many requests", "quota exceeded",
                "throttled", "rate exceeded", "limit exceeded"
            ],
            NetworkErrorType.PROXY_ERROR: [
                "proxy", "407", "proxy authentication", "tunnel",
                "proxy connection failed", "bad gateway"
            ],
            NetworkErrorType.SERVER_ERROR: [
                "500", "502", "503", "504", "internal server error",
                "bad gateway", "service unavailable", "gateway timeout"
            ]
        }
    
    def detect_error_type(self, error: Exception, response_code: Optional[int] = None) -> NetworkErrorType:
        """Detect the type of network error."""
        error_message = str(error).lower()
        
        # Check response code first
        if response_code:
            if response_code == 401:
                return NetworkErrorType.AUTHENTICATION
            elif response_code == 429:
                return NetworkErrorType.RATE_LIMITING
            elif response_code == 407:
                return NetworkErrorType.PROXY_ERROR
            elif 500 <= response_code < 600:
                return NetworkErrorType.SERVER_ERROR
        
        # Check error message patterns
        for error_type, patterns in self.error_patterns.items():
            if any(pattern in error_message for pattern in patterns):
                return error_type
        
        # Check specific exception types
        if isinstance(error, socket.timeout):
            return NetworkErrorType.CONNECTION_TIMEOUT
        elif isinstance(error, socket.gaierror):
            return NetworkErrorType.DNS_RESOLUTION
        elif isinstance(error, ssl.SSLError):
            return NetworkErrorType.SSL_CERTIFICATE
        elif isinstance(error, ConnectionRefusedError):
            return NetworkErrorType.CONNECTION_REFUSED
        
        return NetworkErrorType.UNKNOWN
    
    def is_retryable_error(self, error_type: NetworkErrorType) -> bool:
        """Determine if an error type is retryable."""
        retryable_errors = {
            NetworkErrorType.CONNECTION_TIMEOUT,
            NetworkErrorType.READ_TIMEOUT,
            NetworkErrorType.CONNECTION_REFUSED,
            NetworkErrorType.SERVER_ERROR,
            NetworkErrorType.PARTIAL_DOWNLOAD,
            NetworkErrorType.BANDWIDTH_LIMIT
        }
        return error_type in retryable_errors
    
    def requires_alternative_source(self, error_type: NetworkErrorType) -> bool:
        """Determine if error requires switching to alternative source."""
        alternative_source_errors = {
            NetworkErrorType.AUTHENTICATION,
            NetworkErrorType.RATE_LIMITING,
            NetworkErrorType.SERVER_ERROR,
            NetworkErrorType.DNS_RESOLUTION
        }
        return error_type in alternative_source_errors


class NetworkConnectivityTester:
    """Tests network connectivity and performance."""
    
    def __init__(self, config: NetworkConfiguration, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
    
    def test_basic_connectivity(self, timeout: int = 10) -> NetworkTestResult:
        """Test basic internet connectivity."""
        start_time = time.time()
        
        try:
            # Test multiple endpoints for reliability
            test_endpoints = [
                ("8.8.8.8", 53),  # Google DNS
                ("1.1.1.1", 53),  # Cloudflare DNS
                ("208.67.222.222", 53)  # OpenDNS
            ]
            
            successful_tests = 0
            total_latency = 0.0
            
            for host, port in test_endpoints:
                try:
                    sock_start = time.time()
                    sock = socket.create_connection((host, port), timeout=timeout)
                    sock_end = time.time()
                    sock.close()
                    
                    latency = (sock_end - sock_start) * 1000
                    total_latency += latency
                    successful_tests += 1
                    
                except Exception as e:
                    self.logger.debug(f"Connectivity test failed for {host}:{port}: {e}")
            
            if successful_tests == 0:
                return NetworkTestResult(
                    success=False,
                    latency_ms=0.0,
                    bandwidth_mbps=0.0,
                    error_message="No connectivity to test endpoints",
                    test_duration=time.time() - start_time
                )
            
            average_latency = total_latency / successful_tests
            
            # Test DNS resolution
            dns_working = self._test_dns_resolution()
            
            # Test proxy if configured
            proxy_working = self._test_proxy_connectivity()
            
            return NetworkTestResult(
                success=True,
                latency_ms=average_latency,
                bandwidth_mbps=0.0,  # Basic test doesn't measure bandwidth
                test_duration=time.time() - start_time,
                dns_working=dns_working,
                proxy_working=proxy_working
            )
            
        except Exception as e:
            return NetworkTestResult(
                success=False,
                latency_ms=0.0,
                bandwidth_mbps=0.0,
                error_message=str(e),
                test_duration=time.time() - start_time
            )
    
    def test_bandwidth(self, test_url: str = "http://speedtest.ftp.otenet.gr/files/test1Mb.db",
                      timeout: int = 30) -> NetworkTestResult:
        """Test network bandwidth."""
        start_time = time.time()
        
        try:
            # Download a test file to measure bandwidth
            request = urllib.request.Request(test_url)
            request.add_header('User-Agent', self.config.user_agent)
            
            download_start = time.time()
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read()
            download_end = time.time()
            
            download_time = download_end - download_start
            data_size_mb = len(data) / (1024 * 1024)
            bandwidth_mbps = data_size_mb / download_time if download_time > 0 else 0.0
            
            return NetworkTestResult(
                success=True,
                latency_ms=0.0,
                bandwidth_mbps=bandwidth_mbps,
                test_duration=time.time() - start_time
            )
            
        except Exception as e:
            return NetworkTestResult(
                success=False,
                latency_ms=0.0,
                bandwidth_mbps=0.0,
                error_message=str(e),
                test_duration=time.time() - start_time
            )
    
    def _test_dns_resolution(self) -> bool:
        """Test DNS resolution."""
        try:
            socket.gethostbyname("google.com")
            socket.gethostbyname("github.com")
            return True
        except socket.gaierror:
            return False
    
    def _test_proxy_connectivity(self) -> bool:
        """Test proxy connectivity if configured."""
        proxy_config = self._get_proxy_config()
        if not proxy_config.http_proxy and not proxy_config.https_proxy:
            return True  # No proxy configured, so it's "working"
        
        try:
            # Test with proxy
            proxy_handler = urllib.request.ProxyHandler({
                'http': proxy_config.http_proxy,
                'https': proxy_config.https_proxy
            })
            opener = urllib.request.build_opener(proxy_handler)
            
            request = urllib.request.Request("http://httpbin.org/ip")
            response = opener.open(request, timeout=10)
            response.read()
            return True
            
        except Exception:
            return False
    
    def _get_proxy_config(self) -> ProxyConfiguration:
        """Get proxy configuration from environment."""
        return ProxyConfiguration(
            http_proxy=os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy'),
            https_proxy=os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy'),
            no_proxy=os.environ.get('NO_PROXY') or os.environ.get('no_proxy')
        )


class DownloadMirrorManager:
    """Manages alternative download sources and mirrors."""
    
    def __init__(self, config: NetworkConfiguration, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.mirrors: Dict[str, List[DownloadMirror]] = {}
        self.mirror_health_cache: Dict[str, Tuple[bool, datetime]] = {}
        self._initialize_default_mirrors()
    
    def _initialize_default_mirrors(self):
        """Initialize default mirror configurations."""
        # Hugging Face mirrors
        self.mirrors["huggingface"] = [
            DownloadMirror(
                name="huggingface_primary",
                base_url="https://huggingface.co",
                priority=1
            ),
            DownloadMirror(
                name="huggingface_cdn",
                base_url="https://cdn-lfs.huggingface.co",
                priority=2
            ),
            DownloadMirror(
                name="huggingface_mirror",
                base_url="https://hf-mirror.com",
                priority=3
            )
        ]
        
        # GitHub mirrors
        self.mirrors["github"] = [
            DownloadMirror(
                name="github_primary",
                base_url="https://github.com",
                priority=1
            ),
            DownloadMirror(
                name="github_releases",
                base_url="https://api.github.com",
                priority=2
            )
        ]
        
        # PyPI mirrors
        self.mirrors["pypi"] = [
            DownloadMirror(
                name="pypi_primary",
                base_url="https://pypi.org",
                priority=1
            ),
            DownloadMirror(
                name="pypi_simple",
                base_url="https://pypi.python.org",
                priority=2
            )
        ]
    
    def add_mirror(self, service: str, mirror: DownloadMirror):
        """Add a new mirror for a service."""
        if service not in self.mirrors:
            self.mirrors[service] = []
        
        self.mirrors[service].append(mirror)
        self.mirrors[service].sort(key=lambda m: m.priority)
        self.logger.info(f"Added mirror {mirror.name} for service {service}")
    
    def get_available_mirrors(self, service: str) -> List[DownloadMirror]:
        """Get available mirrors for a service, sorted by priority and health."""
        if service not in self.mirrors:
            return []
        
        available_mirrors = []
        for mirror in self.mirrors[service]:
            if mirror.available and self._is_mirror_healthy(mirror):
                available_mirrors.append(mirror)
        
        # Sort by priority and success rate
        available_mirrors.sort(key=lambda m: (
            m.priority,
            -m.success_count / max(m.success_count + m.failure_count, 1)
        ))
        
        return available_mirrors
    
    def mark_mirror_failure(self, mirror: DownloadMirror, error: Exception):
        """Mark a mirror as having failed."""
        mirror.failure_count += 1
        mirror.last_check = datetime.now()
        
        # Disable mirror if it has too many failures
        if mirror.failure_count >= 3:
            mirror.available = False
            self.logger.warning(f"Disabled mirror {mirror.name} due to repeated failures")
        
        self.logger.debug(f"Mirror {mirror.name} failure: {error}")
    
    def mark_mirror_success(self, mirror: DownloadMirror):
        """Mark a mirror as having succeeded."""
        mirror.success_count += 1
        mirror.last_check = datetime.now()
        mirror.available = True
        self.logger.debug(f"Mirror {mirror.name} success")
    
    def _is_mirror_healthy(self, mirror: DownloadMirror) -> bool:
        """Check if a mirror is healthy based on recent performance."""
        cache_key = f"{mirror.name}_{mirror.base_url}"
        
        # Check cache first
        if cache_key in self.mirror_health_cache:
            is_healthy, check_time = self.mirror_health_cache[cache_key]
            if datetime.now() - check_time < timedelta(minutes=5):
                return is_healthy
        
        # Perform health check
        try:
            test_url = f"{mirror.base_url.rstrip('/')}/robots.txt"
            request = urllib.request.Request(test_url)
            request.add_header('User-Agent', self.config.user_agent)
            
            with urllib.request.urlopen(request, timeout=10) as response:
                is_healthy = response.status == 200
        except Exception:
            is_healthy = False
        
        # Cache result
        self.mirror_health_cache[cache_key] = (is_healthy, datetime.now())
        return is_healthy
    
    def get_mirror_statistics(self) -> Dict[str, Any]:
        """Get statistics about mirror performance."""
        stats = {}
        
        for service, mirrors in self.mirrors.items():
            service_stats = {
                'total_mirrors': len(mirrors),
                'available_mirrors': len([m for m in mirrors if m.available]),
                'mirrors': []
            }
            
            for mirror in mirrors:
                total_attempts = mirror.success_count + mirror.failure_count
                success_rate = mirror.success_count / total_attempts if total_attempts > 0 else 0.0
                
                mirror_stats = {
                    'name': mirror.name,
                    'base_url': mirror.base_url,
                    'priority': mirror.priority,
                    'available': mirror.available,
                    'success_count': mirror.success_count,
                    'failure_count': mirror.failure_count,
                    'success_rate': success_rate,
                    'last_check': mirror.last_check.isoformat() if mirror.last_check else None
                }
                service_stats['mirrors'].append(mirror_stats)
            
            stats[service] = service_stats
        
        return stats


class ResumableDownloader:
    """Handles resumable downloads with progress tracking."""
    
    def __init__(self, config: NetworkConfiguration, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.active_downloads: Dict[str, DownloadProgress] = {}
        self._download_lock = threading.Lock()
    
    def download_with_resume(self, url: str, destination: Path, 
                           expected_size: Optional[int] = None,
                           checksum: Optional[str] = None,
                           progress_callback: Optional[Callable[[DownloadProgress], None]] = None) -> bool:
        """Download a file with resume capability."""
        download_id = f"{url}_{destination.name}"
        
        try:
            # Check if file already exists and is complete
            if destination.exists() and expected_size:
                if destination.stat().st_size == expected_size:
                    if checksum and self._verify_checksum(destination, checksum):
                        self.logger.info(f"File {destination.name} already exists and is valid")
                        return True
            
            # Initialize download progress
            progress = DownloadProgress(
                url=url,
                filename=destination.name,
                total_size=expected_size or 0,
                downloaded_size=0,
                start_time=datetime.now(),
                current_speed_mbps=0.0,
                average_speed_mbps=0.0,
                eta_seconds=None
            )
            
            with self._download_lock:
                self.active_downloads[download_id] = progress
            
            # Determine resume position
            resume_pos = 0
            if destination.exists():
                resume_pos = destination.stat().st_size
                progress.downloaded_size = resume_pos
                progress.resume_count += 1
                self.logger.info(f"Resuming download from position {resume_pos}")
            
            # Create request with resume headers
            request = urllib.request.Request(url)
            request.add_header('User-Agent', self.config.user_agent)
            
            if resume_pos > 0:
                request.add_header('Range', f'bytes={resume_pos}-')
            
            # Perform download
            success = self._perform_download(request, destination, progress, 
                                           resume_pos, progress_callback)
            
            # Verify checksum if provided
            if success and checksum:
                if not self._verify_checksum(destination, checksum):
                    self.logger.error(f"Checksum verification failed for {destination.name}")
                    success = False
            
            return success
            
        except Exception as e:
            self.logger.error(f"Download failed for {url}: {e}")
            return False
        finally:
            with self._download_lock:
                self.active_downloads.pop(download_id, None)
    
    def _perform_download(self, request: urllib.request.Request, destination: Path,
                         progress: DownloadProgress, resume_pos: int,
                         progress_callback: Optional[Callable[[DownloadProgress], None]]) -> bool:
        """Perform the actual download with progress tracking."""
        try:
            with urllib.request.urlopen(request, timeout=self.config.read_timeout) as response:
                # Get total size from headers
                if progress.total_size == 0:
                    content_length = response.headers.get('Content-Length')
                    if content_length:
                        if resume_pos > 0:
                            progress.total_size = resume_pos + int(content_length)
                        else:
                            progress.total_size = int(content_length)
                
                # Open file for writing
                mode = 'ab' if resume_pos > 0 else 'wb'
                with open(destination, mode) as f:
                    bytes_downloaded = 0
                    last_update = time.time()
                    speed_samples = []
                    
                    while True:
                        chunk = response.read(self.config.chunk_size)
                        if not chunk:
                            break
                        
                        f.write(chunk)
                        bytes_downloaded += len(chunk)
                        progress.downloaded_size = resume_pos + bytes_downloaded
                        
                        # Update speed calculation
                        current_time = time.time()
                        if current_time - last_update >= 1.0:  # Update every second
                            elapsed = current_time - last_update
                            current_speed = (len(chunk) / elapsed) / (1024 * 1024)  # MB/s
                            speed_samples.append(current_speed)
                            
                            # Keep only last 10 samples for average
                            if len(speed_samples) > 10:
                                speed_samples.pop(0)
                            
                            progress.current_speed_mbps = current_speed
                            progress.average_speed_mbps = sum(speed_samples) / len(speed_samples)
                            
                            # Calculate ETA
                            if progress.total_size > 0 and progress.average_speed_mbps > 0:
                                remaining_mb = (progress.total_size - progress.downloaded_size) / (1024 * 1024)
                                progress.eta_seconds = int(remaining_mb / progress.average_speed_mbps)
                            
                            last_update = current_time
                            
                            # Call progress callback
                            if progress_callback:
                                progress_callback(progress)
                            
                            # Apply bandwidth limit if configured
                            if self.config.bandwidth_limit_mbps:
                                if progress.current_speed_mbps > self.config.bandwidth_limit_mbps:
                                    sleep_time = 0.1
                                    time.sleep(sleep_time)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Download error: {e}")
            return False
    
    def _verify_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        """Verify file checksum."""
        try:
            hash_algo = 'sha256'  # Default to SHA256
            if ':' in expected_checksum:
                hash_algo, expected_checksum = expected_checksum.split(':', 1)
            
            hasher = hashlib.new(hash_algo)
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            
            actual_checksum = hasher.hexdigest()
            return actual_checksum.lower() == expected_checksum.lower()
            
        except Exception as e:
            self.logger.error(f"Checksum verification error: {e}")
            return False
    
    def get_download_progress(self, download_id: str) -> Optional[DownloadProgress]:
        """Get progress for an active download."""
        with self._download_lock:
            return self.active_downloads.get(download_id)
    
    def get_active_downloads(self) -> Dict[str, DownloadProgress]:
        """Get all active downloads."""
        with self._download_lock:
            return self.active_downloads.copy()
    
    def cancel_download(self, download_id: str) -> bool:
        """Cancel an active download."""
        with self._download_lock:
            if download_id in self.active_downloads:
                del self.active_downloads[download_id]
                return True
        return False


class NetworkFailureRecovery(BaseInstallationComponent):
    """
    Comprehensive network failure recovery system.
    
    Provides intelligent handling of network failures including:
    - Error detection and classification
    - Alternative source selection
    - Resumable downloads
    - Proxy and authentication handling
    - Network connectivity testing
    """
    
    def __init__(self, installation_path: str, logger: Optional[logging.Logger] = None):
        super().__init__(installation_path, logger)
        
        # Initialize configuration
        self.network_config = NetworkConfiguration()
        self.proxy_config = ProxyConfiguration()
        self.auth_config = AuthenticationConfiguration()
        
        # Initialize components
        self.error_detector = NetworkErrorDetector(logger)
        self.connectivity_tester = NetworkConnectivityTester(self.network_config, logger)
        self.mirror_manager = DownloadMirrorManager(self.network_config, logger)
        self.downloader = ResumableDownloader(self.network_config, logger)
        
        # Recovery statistics
        self.recovery_stats = {
            'total_attempts': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'mirror_switches': 0,
            'resumed_downloads': 0,
            'authentication_fixes': 0,
            'proxy_fixes': 0
        }
        
        self._load_configuration()
    
    def _load_configuration(self):
        """Load network configuration from files or environment."""
        try:
            # Load from environment variables
            self.proxy_config = ProxyConfiguration(
                http_proxy=os.environ.get('HTTP_PROXY'),
                https_proxy=os.environ.get('HTTPS_PROXY'),
                no_proxy=os.environ.get('NO_PROXY')
            )
            
            # Load authentication from config file if exists
            config_file = Path(self.installation_path) / "config" / "network.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    
                    # Update network configuration
                    for key, value in config_data.get('network', {}).items():
                        if hasattr(self.network_config, key):
                            setattr(self.network_config, key, value)
                    
                    # Update authentication configuration
                    auth_data = config_data.get('authentication', {})
                    self.auth_config = AuthenticationConfiguration(**auth_data)
                    
                    self.logger.info("Loaded network configuration from file")
            
        except Exception as e:
            self.logger.warning(f"Failed to load network configuration: {e}")
    
    def test_network_connectivity(self) -> NetworkTestResult:
        """Test network connectivity and return detailed results."""
        self.logger.info("Testing network connectivity...")
        
        # Test basic connectivity
        basic_result = self.connectivity_tester.test_basic_connectivity()
        
        if not basic_result.success:
            self.logger.warning(f"Basic connectivity test failed: {basic_result.error_message}")
            return basic_result
        
        # Test bandwidth if basic connectivity works
        try:
            bandwidth_result = self.connectivity_tester.test_bandwidth()
            if bandwidth_result.success:
                basic_result.bandwidth_mbps = bandwidth_result.bandwidth_mbps
        except Exception as e:
            self.logger.debug(f"Bandwidth test failed: {e}")
        
        self.logger.info(f"Network connectivity test completed: "
                        f"latency={basic_result.latency_ms:.1f}ms, "
                        f"bandwidth={basic_result.bandwidth_mbps:.1f}Mbps")
        
        return basic_result
    
    def recover_from_network_failure(self, operation: Callable, operation_name: str,
                                   url: str, error: Exception,
                                   max_attempts: int = 3) -> Any:
        """
        Recover from network failure using intelligent strategies.
        
        Args:
            operation: The operation that failed
            operation_name: Human-readable name for the operation
            url: The URL that failed
            error: The original error
            max_attempts: Maximum recovery attempts
            
        Returns:
            Result of successful operation
            
        Raises:
            Exception: If all recovery attempts fail
        """
        self.recovery_stats['total_attempts'] += 1
        
        # Detect error type
        error_type = self.error_detector.detect_error_type(error)
        self.logger.info(f"Detected network error type: {error_type.value} for {operation_name}")
        
        # Try recovery strategies based on error type
        recovery_strategies = self._get_recovery_strategies(error_type, url)
        
        last_error = error
        for attempt in range(max_attempts):
            for strategy_name, strategy_func in recovery_strategies:
                try:
                    self.logger.info(f"Attempting recovery strategy: {strategy_name} "
                                   f"(attempt {attempt + 1}/{max_attempts})")
                    
                    result = strategy_func(operation, operation_name, url, error)
                    
                    if result is not None:
                        self.recovery_stats['successful_recoveries'] += 1
                        self.logger.info(f"Recovery successful using strategy: {strategy_name}")
                        return result
                        
                except Exception as e:
                    last_error = e
                    self.logger.warning(f"Recovery strategy {strategy_name} failed: {e}")
            
            # Wait before next attempt
            if attempt < max_attempts - 1:
                delay = self.network_config.retry_delay * (2 ** attempt)
                delay = min(delay, self.network_config.max_retry_delay)
                self.logger.info(f"Waiting {delay:.1f}s before next recovery attempt...")
                time.sleep(delay)
        
        # All recovery attempts failed
        self.recovery_stats['failed_recoveries'] += 1
        self.logger.error(f"All recovery attempts failed for {operation_name}")
        raise last_error
    
    def _get_recovery_strategies(self, error_type: NetworkErrorType, 
                               url: str) -> List[Tuple[str, Callable]]:
        """Get recovery strategies for a specific error type."""
        strategies = []
        
        # Common strategies for all error types
        strategies.append(("retry_with_backoff", self._retry_with_backoff))
        
        # Error-specific strategies
        if error_type == NetworkErrorType.AUTHENTICATION:
            strategies.extend([
                ("fix_authentication", self._fix_authentication),
                ("try_alternative_source", self._try_alternative_source)
            ])
        elif error_type == NetworkErrorType.RATE_LIMITING:
            strategies.extend([
                ("wait_for_rate_limit", self._wait_for_rate_limit),
                ("try_alternative_source", self._try_alternative_source)
            ])
        elif error_type == NetworkErrorType.PROXY_ERROR:
            strategies.extend([
                ("fix_proxy_configuration", self._fix_proxy_configuration),
                ("bypass_proxy", self._bypass_proxy)
            ])
        elif error_type == NetworkErrorType.DNS_RESOLUTION:
            strategies.extend([
                ("try_alternative_dns", self._try_alternative_dns),
                ("try_alternative_source", self._try_alternative_source)
            ])
        elif error_type == NetworkErrorType.SSL_CERTIFICATE:
            strategies.extend([
                ("fix_ssl_configuration", self._fix_ssl_configuration),
                ("try_alternative_source", self._try_alternative_source)
            ])
        elif error_type in [NetworkErrorType.CONNECTION_TIMEOUT, NetworkErrorType.CONNECTION_REFUSED]:
            strategies.extend([
                ("increase_timeout", self._increase_timeout),
                ("try_alternative_source", self._try_alternative_source)
            ])
        elif error_type == NetworkErrorType.SERVER_ERROR:
            strategies.extend([
                ("try_alternative_source", self._try_alternative_source),
                ("wait_for_server_recovery", self._wait_for_server_recovery)
            ])
        
        # Always try alternative source as last resort
        if ("try_alternative_source", self._try_alternative_source) not in strategies:
            strategies.append(("try_alternative_source", self._try_alternative_source))
        
        return strategies
    
    def _retry_with_backoff(self, operation: Callable, operation_name: str,
                           url: str, error: Exception) -> Any:
        """Retry operation with exponential backoff."""
        delay = self.network_config.retry_delay
        
        for attempt in range(self.network_config.max_retries):
            try:
                time.sleep(delay)
                return operation()
            except Exception as e:
                if attempt == self.network_config.max_retries - 1:
                    raise e
                delay = min(delay * self.network_config.backoff_multiplier,
                           self.network_config.max_retry_delay)
        
        return None
    
    def _fix_authentication(self, operation: Callable, operation_name: str,
                           url: str, error: Exception) -> Any:
        """Fix authentication issues."""
        self.recovery_stats['authentication_fixes'] += 1
        
        # Try different authentication methods
        auth_methods = [
            self._try_token_auth,
            self._try_api_key_auth,
            self._try_basic_auth,
            self._prompt_for_credentials
        ]
        
        for auth_method in auth_methods:
            try:
                if auth_method(url):
                    return operation()
            except Exception:
                continue
        
        return None
    
    def _try_alternative_source(self, operation: Callable, operation_name: str,
                               url: str, error: Exception) -> Any:
        """Try alternative download source."""
        self.recovery_stats['mirror_switches'] += 1
        
        # Determine service type from URL
        service = self._detect_service_from_url(url)
        if not service:
            return None
        
        # Get available mirrors
        mirrors = self.mirror_manager.get_available_mirrors(service)
        
        for mirror in mirrors:
            try:
                # Replace base URL with mirror URL
                mirror_url = self._convert_url_to_mirror(url, mirror)
                self.logger.info(f"Trying alternative source: {mirror.name}")
                
                # Update operation to use mirror URL
                # This is a simplified approach - in practice, you'd need to modify
                # the operation to use the new URL
                result = operation()  # Would need to pass mirror_url somehow
                
                self.mirror_manager.mark_mirror_success(mirror)
                return result
                
            except Exception as e:
                self.mirror_manager.mark_mirror_failure(mirror, e)
                continue
        
        return None
    
    def _wait_for_rate_limit(self, operation: Callable, operation_name: str,
                            url: str, error: Exception) -> Any:
        """Wait for rate limit to reset."""
        # Extract retry-after header if available
        retry_after = 60  # Default to 1 minute
        
        if hasattr(error, 'headers') and 'Retry-After' in error.headers:
            try:
                retry_after = int(error.headers['Retry-After'])
            except ValueError:
                pass
        
        self.logger.info(f"Rate limited, waiting {retry_after} seconds...")
        time.sleep(retry_after)
        
        return operation()
    
    def _fix_proxy_configuration(self, operation: Callable, operation_name: str,
                                url: str, error: Exception) -> Any:
        """Fix proxy configuration issues."""
        self.recovery_stats['proxy_fixes'] += 1
        
        # Try different proxy configurations
        proxy_configs = [
            self._get_system_proxy_config(),
            self._get_auto_detected_proxy_config(),
            ProxyConfiguration()  # No proxy
        ]
        
        for proxy_config in proxy_configs:
            try:
                # Apply proxy configuration
                self._apply_proxy_config(proxy_config)
                return operation()
            except Exception:
                continue
        
        return None
    
    def _bypass_proxy(self, operation: Callable, operation_name: str,
                     url: str, error: Exception) -> Any:
        """Bypass proxy for the operation."""
        # Temporarily disable proxy
        original_proxy = self.proxy_config
        self.proxy_config = ProxyConfiguration()
        
        try:
            return operation()
        finally:
            self.proxy_config = original_proxy
    
    def _try_alternative_dns(self, operation: Callable, operation_name: str,
                            url: str, error: Exception) -> Any:
        """Try alternative DNS servers."""
        # This would require system-level DNS configuration changes
        # For now, just retry the operation
        return operation()
    
    def _fix_ssl_configuration(self, operation: Callable, operation_name: str,
                              url: str, error: Exception) -> Any:
        """Fix SSL configuration issues."""
        # Try with different SSL settings
        original_verify = self.network_config.verify_ssl
        
        try:
            # Temporarily disable SSL verification (not recommended for production)
            self.network_config.verify_ssl = False
            self.logger.warning("Temporarily disabling SSL verification")
            return operation()
        finally:
            self.network_config.verify_ssl = original_verify
    
    def _increase_timeout(self, operation: Callable, operation_name: str,
                         url: str, error: Exception) -> Any:
        """Increase timeout for the operation."""
        original_timeout = self.network_config.read_timeout
        
        try:
            # Double the timeout
            self.network_config.read_timeout *= 2
            self.logger.info(f"Increased timeout to {self.network_config.read_timeout}s")
            return operation()
        finally:
            self.network_config.read_timeout = original_timeout
    
    def _wait_for_server_recovery(self, operation: Callable, operation_name: str,
                                 url: str, error: Exception) -> Any:
        """Wait for server to recover from error."""
        wait_time = 30  # Wait 30 seconds for server recovery
        self.logger.info(f"Server error detected, waiting {wait_time}s for recovery...")
        time.sleep(wait_time)
        return operation()
    
    def download_with_recovery(self, url: str, destination: Path,
                              expected_size: Optional[int] = None,
                              checksum: Optional[str] = None,
                              progress_callback: Optional[Callable[[DownloadProgress], None]] = None) -> bool:
        """Download a file with comprehensive recovery capabilities."""
        def download_operation():
            return self.downloader.download_with_resume(
                url, destination, expected_size, checksum, progress_callback
            )
        
        try:
            return self.recover_from_network_failure(
                download_operation, f"download_{destination.name}", url, Exception("Initial download")
            )
        except Exception as e:
            self.logger.error(f"Download failed after all recovery attempts: {e}")
            return False
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get network failure recovery statistics."""
        stats = self.recovery_stats.copy()
        
        # Add mirror statistics
        stats['mirror_statistics'] = self.mirror_manager.get_mirror_statistics()
        
        # Add active downloads
        stats['active_downloads'] = len(self.downloader.get_active_downloads())
        
        # Calculate success rate
        total_attempts = stats['total_attempts']
        if total_attempts > 0:
            stats['success_rate'] = stats['successful_recoveries'] / total_attempts
        else:
            stats['success_rate'] = 0.0
        
        return stats
    
    def configure_authentication(self, username: Optional[str] = None,
                               password: Optional[str] = None,
                               token: Optional[str] = None,
                               api_key: Optional[str] = None,
                               headers: Optional[Dict[str, str]] = None):
        """Configure authentication settings."""
        self.auth_config = AuthenticationConfiguration(
            username=username,
            password=password,
            token=token,
            api_key=api_key,
            headers=headers or {}
        )
        self.logger.info("Updated authentication configuration")
    
    def configure_proxy(self, http_proxy: Optional[str] = None,
                       https_proxy: Optional[str] = None,
                       no_proxy: Optional[str] = None,
                       proxy_auth: Optional[Tuple[str, str]] = None):
        """Configure proxy settings."""
        self.proxy_config = ProxyConfiguration(
            http_proxy=http_proxy,
            https_proxy=https_proxy,
            no_proxy=no_proxy,
            proxy_auth=proxy_auth
        )
        self.logger.info("Updated proxy configuration")
    
    def validate_network_configuration(self) -> ValidationResult:
        """Validate current network configuration."""
        issues = []
        warnings = []
        
        # Test basic connectivity
        connectivity_result = self.test_network_connectivity()
        if not connectivity_result.success:
            issues.append(f"Network connectivity test failed: {connectivity_result.error_message}")
        else:
            if connectivity_result.latency_ms > 1000:
                warnings.append(f"High network latency: {connectivity_result.latency_ms:.1f}ms")
            if connectivity_result.bandwidth_mbps < 1.0:
                warnings.append(f"Low bandwidth: {connectivity_result.bandwidth_mbps:.1f}Mbps")
        
        # Check proxy configuration
        if self.proxy_config.http_proxy or self.proxy_config.https_proxy:
            if not connectivity_result.proxy_working:
                issues.append("Proxy is configured but not working properly")
        
        # Check DNS
        if not connectivity_result.dns_working:
            issues.append("DNS resolution is not working")
        
        success = len(issues) == 0
        message = "Network configuration is valid" if success else f"Found {len(issues)} issues"
        
        return ValidationResult(
            success=success,
            message=message,
            details={'issues': issues, 'warnings': warnings},
            warnings=warnings
        )
    
    # Helper methods
    def _detect_service_from_url(self, url: str) -> Optional[str]:
        """Detect service type from URL."""
        if 'huggingface.co' in url or 'hf.co' in url:
            return 'huggingface'
        elif 'github.com' in url:
            return 'github'
        elif 'pypi.org' in url or 'pypi.python.org' in url:
            return 'pypi'
        return None
    
    def _convert_url_to_mirror(self, original_url: str, mirror: DownloadMirror) -> str:
        """Convert original URL to use mirror."""
        # This is a simplified implementation
        # In practice, you'd need more sophisticated URL conversion logic
        parsed = urllib.parse.urlparse(original_url)
        mirror_parsed = urllib.parse.urlparse(mirror.base_url)
        
        return original_url.replace(f"{parsed.scheme}://{parsed.netloc}", 
                                  f"{mirror_parsed.scheme}://{mirror_parsed.netloc}")
    
    def _try_token_auth(self, url: str) -> bool:
        """Try token-based authentication."""
        return self.auth_config.token is not None
    
    def _try_api_key_auth(self, url: str) -> bool:
        """Try API key authentication."""
        return self.auth_config.api_key is not None
    
    def _try_basic_auth(self, url: str) -> bool:
        """Try basic authentication."""
        return self.auth_config.username is not None and self.auth_config.password is not None
    
    def _prompt_for_credentials(self, url: str) -> bool:
        """Prompt user for credentials."""
        # This would show a dialog or prompt in a real implementation
        return False
    
    def _get_system_proxy_config(self) -> ProxyConfiguration:
        """Get system proxy configuration."""
        return ProxyConfiguration(
            http_proxy=os.environ.get('HTTP_PROXY'),
            https_proxy=os.environ.get('HTTPS_PROXY'),
            no_proxy=os.environ.get('NO_PROXY')
        )
    
    def _get_auto_detected_proxy_config(self) -> ProxyConfiguration:
        """Auto-detect proxy configuration."""
        # This would implement proxy auto-detection logic
        return ProxyConfiguration()
    
    def _apply_proxy_config(self, proxy_config: ProxyConfiguration):
        """Apply proxy configuration."""
        self.proxy_config = proxy_config
