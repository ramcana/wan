"""
Comprehensive error handling and recovery system for WAN2.2 installation.

This module provides advanced error handling capabilities including:
- Error categorization and classification
- Automatic retry mechanisms for transient failures
- Fallback options for common failure scenarios
- Recovery action suggestions
- Error context tracking and logging
"""

import logging
import time
import traceback
import platform
import psutil
import os
import sys
import socket
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
from datetime import datetime

from interfaces import (
    InstallationError, ErrorCategory, IErrorHandler,
    HardwareProfile, ValidationResult
)
from base_classes import BaseInstallationComponent
from retry_system import IntelligentRetrySystem, RetryConfiguration, RetryStrategy


class RecoveryAction(Enum):
    """Available recovery actions for errors."""
    RETRY = "retry"
    RETRY_WITH_FALLBACK = "retry_with_fallback"
    ELEVATE_PERMISSIONS = "elevate"
    ABORT = "abort"
    CONTINUE = "continue"
    ROLLBACK = "rollback"
    MANUAL_INTERVENTION = "manual"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SystemInfo:
    """Comprehensive system information."""
    os_version: str
    python_version: str
    available_memory_gb: float
    available_disk_gb: float
    cpu_usage_percent: float
    gpu_info: Optional[Dict[str, Any]]
    installed_packages: Dict[str, str]
    environment_vars: Dict[str, str]


@dataclass
class ResourceSnapshot:
    """Current resource usage snapshot."""
    memory_usage_mb: int
    disk_usage_gb: float
    cpu_usage_percent: float
    gpu_memory_usage_mb: int
    network_bandwidth_mbps: float
    open_file_handles: int
    process_count: int


@dataclass
class NetworkStatus:
    """Network connectivity status."""
    connectivity: bool
    latency_ms: float
    bandwidth_mbps: float
    proxy_configured: bool
    dns_resolution: bool
    external_ip: Optional[str]


@dataclass
class EnhancedErrorContext:
    """Enhanced context information for an error with comprehensive system state."""
    timestamp: datetime = field(default_factory=datetime.now)
    phase: str = ""
    task: str = ""
    component: str = ""
    method: str = ""
    hardware_profile: Optional[HardwareProfile] = None
    system_info: Optional[SystemInfo] = None
    environment_vars: Dict[str, str] = field(default_factory=dict)
    stack_trace: str = ""
    retry_count: int = 0
    previous_errors: List[str] = field(default_factory=list)
    recovery_attempts: List[str] = field(default_factory=list)
    system_resources: Optional[ResourceSnapshot] = None
    network_status: Optional[NetworkStatus] = None


# Keep backward compatibility
@dataclass
class ErrorContext:
    """Legacy context information for an error."""
    timestamp: datetime = field(default_factory=datetime.now)
    phase: str = ""
    task: str = ""
    hardware_profile: Optional[HardwareProfile] = None
    system_info: Dict[str, Any] = field(default_factory=dict)
    stack_trace: str = ""
    retry_count: int = 0
    previous_errors: List[str] = field(default_factory=list)


@dataclass
class RetryConfig:
    """Configuration for retry mechanisms."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True


class SystemStateCollector:
    """Collects comprehensive system state information for error context."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def collect_system_info(self) -> SystemInfo:
        """Collect comprehensive system information."""
        try:
            # Get OS information
            os_version = f"{platform.system()} {platform.release()} ({platform.architecture()[0]})"
            
            # Get Python version
            python_version = f"Python {sys.version.split()[0]}"
            
            # Get memory information
            memory = psutil.virtual_memory()
            available_memory_gb = memory.available / (1024**3)
            
            # Get disk information
            disk = psutil.disk_usage('.')
            available_disk_gb = disk.free / (1024**3)
            
            # Get CPU usage
            cpu_usage_percent = psutil.cpu_percent(interval=1)
            
            # Get GPU information (if available)
            gpu_info = self._get_gpu_info()
            
            # Get installed packages
            installed_packages = self._get_installed_packages()
            
            # Get relevant environment variables
            environment_vars = self._get_relevant_env_vars()
            
            return SystemInfo(
                os_version=os_version,
                python_version=python_version,
                available_memory_gb=available_memory_gb,
                available_disk_gb=available_disk_gb,
                cpu_usage_percent=cpu_usage_percent,
                gpu_info=gpu_info,
                installed_packages=installed_packages,
                environment_vars=environment_vars
            )
        except Exception as e:
            self.logger.warning(f"Failed to collect system info: {e}")
            return SystemInfo(
                os_version="Unknown",
                python_version="Unknown",
                available_memory_gb=0.0,
                available_disk_gb=0.0,
                cpu_usage_percent=0.0,
                gpu_info=None,
                installed_packages={},
                environment_vars={}
            )
    
    def collect_resource_snapshot(self) -> ResourceSnapshot:
        """Collect current resource usage snapshot."""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage_mb = (memory.total - memory.available) / (1024**2)
            
            # Disk usage
            disk = psutil.disk_usage('.')
            disk_usage_gb = (disk.total - disk.free) / (1024**3)
            
            # CPU usage
            cpu_usage_percent = psutil.cpu_percent()
            
            # GPU memory (placeholder - would need GPU-specific libraries)
            gpu_memory_usage_mb = 0
            
            # Network bandwidth (placeholder - would need network monitoring)
            network_bandwidth_mbps = 0.0
            
            # Open file handles
            try:
                process = psutil.Process()
                open_file_handles = len(process.open_files())
            except:
                open_file_handles = 0
            
            # Process count
            process_count = len(psutil.pids())
            
            return ResourceSnapshot(
                memory_usage_mb=int(memory_usage_mb),
                disk_usage_gb=disk_usage_gb,
                cpu_usage_percent=cpu_usage_percent,
                gpu_memory_usage_mb=gpu_memory_usage_mb,
                network_bandwidth_mbps=network_bandwidth_mbps,
                open_file_handles=open_file_handles,
                process_count=process_count
            )
        except Exception as e:
            self.logger.warning(f"Failed to collect resource snapshot: {e}")
            return ResourceSnapshot(
                memory_usage_mb=0,
                disk_usage_gb=0.0,
                cpu_usage_percent=0.0,
                gpu_memory_usage_mb=0,
                network_bandwidth_mbps=0.0,
                open_file_handles=0,
                process_count=0
            )
    
    def collect_network_status(self) -> NetworkStatus:
        """Collect network connectivity status."""
        try:
            # Test basic connectivity
            connectivity = self._test_connectivity()
            
            # Test latency
            latency_ms = self._test_latency()
            
            # Bandwidth (placeholder)
            bandwidth_mbps = 0.0
            
            # Check proxy configuration
            proxy_configured = bool(os.environ.get('HTTP_PROXY') or os.environ.get('HTTPS_PROXY'))
            
            # Test DNS resolution
            dns_resolution = self._test_dns_resolution()
            
            # Get external IP (optional)
            external_ip = self._get_external_ip()
            
            return NetworkStatus(
                connectivity=connectivity,
                latency_ms=latency_ms,
                bandwidth_mbps=bandwidth_mbps,
                proxy_configured=proxy_configured,
                dns_resolution=dns_resolution,
                external_ip=external_ip
            )
        except Exception as e:
            self.logger.warning(f"Failed to collect network status: {e}")
            return NetworkStatus(
                connectivity=False,
                latency_ms=0.0,
                bandwidth_mbps=0.0,
                proxy_configured=False,
                dns_resolution=False,
                external_ip=None
            )
    
    def _get_gpu_info(self) -> Optional[Dict[str, Any]]:
        """Get GPU information if available."""
        try:
            # Try to get GPU info using various methods
            gpu_info = {}
            
            # Try nvidia-ml-py if available
            try:
                import pynvml
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                if device_count > 0:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_info = {
                        'name': name,
                        'memory_total_mb': memory_info.total / (1024**2),
                        'memory_used_mb': memory_info.used / (1024**2),
                        'memory_free_mb': memory_info.free / (1024**2)
                    }
                pynvml.nvmlShutdown()
            except ImportError:
                pass
            
            return gpu_info if gpu_info else None
        except Exception:
            return None
    
    def _get_installed_packages(self) -> Dict[str, str]:
        """Get installed Python packages."""
        try:
            import pkg_resources
            packages = {}
            for dist in pkg_resources.working_set:
                packages[dist.project_name] = dist.version
            return packages
        except Exception:
            return {}
    
    def _get_relevant_env_vars(self) -> Dict[str, str]:
        """Get relevant environment variables."""
        relevant_vars = [
            'PATH', 'PYTHONPATH', 'PYTHON_HOME', 'VIRTUAL_ENV',
            'HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY',
            'CUDA_VISIBLE_DEVICES', 'CUDA_PATH',
            'TEMP', 'TMP', 'TMPDIR'
        ]
        
        env_vars = {}
        for var in relevant_vars:
            value = os.environ.get(var)
            if value:
                env_vars[var] = value
        
        return env_vars
    
    def _test_connectivity(self) -> bool:
        """Test basic internet connectivity."""
        try:
            sock = socket.create_connection(("8.8.8.8", 53), timeout=3)
            sock.close()
            return True
        except OSError:
            return False
    
    def _test_latency(self) -> float:
        """Test network latency."""
        try:
            import time
            start_time = time.time()
            sock = socket.create_connection(("8.8.8.8", 53), timeout=3)
            end_time = time.time()
            sock.close()
            return (end_time - start_time) * 1000  # Convert to milliseconds
        except OSError:
            return 0.0
    
    def _test_dns_resolution(self) -> bool:
        """Test DNS resolution."""
        try:
            socket.gethostbyname("google.com")
            return True
        except socket.gaierror:
            return False
    
    def _get_external_ip(self) -> Optional[str]:
        """Get external IP address (optional)."""
        try:
            import urllib.request
            with urllib.request.urlopen('https://api.ipify.org', timeout=5) as response:
                return response.read().decode('utf-8')
        except Exception:
            return None


class ErrorClassifier:
    """Classifies errors and determines appropriate handling strategies."""
    
    # Error patterns for automatic classification
    NETWORK_PATTERNS = [
        "connection", "timeout", "network", "dns", "ssl", "certificate",
        "download", "upload", "http", "https", "url", "socket", "proxy"
    ]
    
    PERMISSION_PATTERNS = [
        "permission", "access", "denied", "forbidden", "unauthorized",
        "privilege", "administrator", "elevation", "readonly"
    ]
    
    SYSTEM_PATTERNS = [
        "memory", "disk", "space", "driver", "hardware", "cpu", "gpu",
        "ram", "storage", "system", "os", "platform", "architecture"
    ]
    
    CONFIGURATION_PATTERNS = [
        "config", "setting", "parameter", "option", "value", "format",
        "syntax", "parse", "json", "yaml", "toml", "invalid"
    ]
    
    def classify_error(self, error: Exception, context: ErrorContext) -> ErrorCategory:
        """Classify an error based on its message and context."""
        error_message = str(error).lower()
        
        # Check for specific error types first
        if isinstance(error, PermissionError):
            return ErrorCategory.PERMISSION
        elif isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorCategory.NETWORK
        elif isinstance(error, (OSError, SystemError)):
            return ErrorCategory.SYSTEM
        elif isinstance(error, (ValueError, SyntaxError, KeyError)):
            return ErrorCategory.CONFIGURATION
        
        # Pattern-based classification
        if any(pattern in error_message for pattern in self.NETWORK_PATTERNS):
            return ErrorCategory.NETWORK
        elif any(pattern in error_message for pattern in self.PERMISSION_PATTERNS):
            return ErrorCategory.PERMISSION
        elif any(pattern in error_message for pattern in self.SYSTEM_PATTERNS):
            return ErrorCategory.SYSTEM
        elif any(pattern in error_message for pattern in self.CONFIGURATION_PATTERNS):
            return ErrorCategory.CONFIGURATION
        
        # Default to system error
        return ErrorCategory.SYSTEM
    
    def determine_severity(self, error: InstallationError, context: ErrorContext) -> ErrorSeverity:
        """Determine the severity of an error."""
        # Critical errors that should abort installation
        critical_patterns = [
            "insufficient memory", "disk full", "insufficient disk space", "hardware not supported",
            "python not found", "critical system error"
        ]
        
        # High severity errors that need immediate attention
        high_patterns = [
            "permission denied", "access denied", "network unreachable",
            "model download failed", "dependency installation failed"
        ]
        
        # Medium severity errors that can be retried
        medium_patterns = [
            "timeout", "connection reset", "temporary failure",
            "file locked", "resource busy"
        ]
        
        error_message = error.message.lower()
        
        if any(pattern in error_message for pattern in critical_patterns):
            return ErrorSeverity.CRITICAL
        elif any(pattern in error_message for pattern in high_patterns):
            return ErrorSeverity.HIGH
        elif any(pattern in error_message for pattern in medium_patterns):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW


class RetryManager:
    """Manages retry logic for transient failures."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def should_retry(self, error: InstallationError, context: ErrorContext, 
                    config: RetryConfig) -> bool:
        """Determine if an error should be retried."""
        # Don't retry if we've exceeded max attempts
        if context.retry_count >= config.max_attempts:
            return False
        
        # Don't retry critical errors or permission errors
        if error.category in [ErrorCategory.PERMISSION]:
            return False
        
        # Retry network and temporary system errors
        if error.category in [ErrorCategory.NETWORK, ErrorCategory.SYSTEM]:
            # Check for specific non-retryable patterns
            non_retryable = [
                "disk full", "insufficient memory", "hardware not supported",
                "permission denied", "access denied"
            ]
            if any(pattern in error.message.lower() for pattern in non_retryable):
                return False
            return True
        
        # Don't retry configuration errors by default
        return False
    
    def calculate_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate delay before next retry attempt."""
        import random
        
        delay = min(
            config.base_delay * (config.backoff_multiplier ** attempt),
            config.max_delay
        )
        
        if config.jitter:
            # Add random jitter to prevent thundering herd
            delay *= (0.5 + random.random() * 0.5)
        
        return delay
    
    def execute_with_retry(self, func: Callable, config: RetryConfig,
                          context: ErrorContext) -> Any:
        """Execute a function with retry logic."""
        last_error = None
        
        for attempt in range(config.max_attempts):
            try:
                context.retry_count = attempt
                return func()
            except Exception as e:
                last_error = e
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt < config.max_attempts - 1:
                    delay = self.calculate_delay(attempt, config)
                    self.logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
        
        # All retries failed
        if last_error:
            raise last_error


class FallbackManager:
    """Manages fallback options for common failure scenarios."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.fallback_strategies = self._initialize_fallback_strategies()
    
    def _initialize_fallback_strategies(self) -> Dict[str, List[Callable]]:
        """Initialize fallback strategies for different scenarios."""
        return {
            "python_download": [
                self._fallback_python_embedded,
                self._fallback_python_system,
                self._fallback_python_manual
            ],
            "model_download": [
                self._fallback_model_mirror,
                self._fallback_model_local,
                self._fallback_model_skip
            ],
            "package_install": [
                self._fallback_package_simple,
                self._fallback_package_offline,
                self._fallback_package_minimal
            ],
            "config_generation": [
                self._fallback_config_template,
                self._fallback_config_minimal,
                self._fallback_config_default
            ]
        }
    
    def get_fallback_options(self, scenario: str, context: ErrorContext) -> List[Callable]:
        """Get fallback options for a specific scenario."""
        return self.fallback_strategies.get(scenario, [])
    
    def _fallback_python_embedded(self, context: ErrorContext) -> bool:
        """Fallback to embedded Python installation."""
        self.logger.info("Attempting fallback: embedded Python installation")
        # Implementation would go here
        return False
    
    def _fallback_python_system(self, context: ErrorContext) -> bool:
        """Fallback to system Python installation."""
        self.logger.info("Attempting fallback: system Python installation")
        # Implementation would go here
        return False
    
    def _fallback_python_manual(self, context: ErrorContext) -> bool:
        """Fallback to manual Python installation guidance."""
        self.logger.info("Attempting fallback: manual Python installation")
        # Implementation would go here
        return False
    
    def _fallback_model_mirror(self, context: ErrorContext) -> bool:
        """Fallback to alternative model download mirror."""
        self.logger.info("Attempting fallback: alternative model mirror")
        # Implementation would go here
        return False
    
    def _fallback_model_local(self, context: ErrorContext) -> bool:
        """Fallback to local model files if available."""
        self.logger.info("Attempting fallback: local model files")
        # Implementation would go here
        return False
    
    def _fallback_model_skip(self, context: ErrorContext) -> bool:
        """Fallback to skip model download for now."""
        self.logger.info("Attempting fallback: skip model download")
        # Implementation would go here
        return False
    
    def _fallback_package_simple(self, context: ErrorContext) -> bool:
        """Fallback to simple package installation."""
        self.logger.info("Attempting fallback: simple package installation")
        # Implementation would go here
        return False
    
    def _fallback_package_offline(self, context: ErrorContext) -> bool:
        """Fallback to offline package installation."""
        self.logger.info("Attempting fallback: offline package installation")
        # Implementation would go here
        return False
    
    def _fallback_package_minimal(self, context: ErrorContext) -> bool:
        """Fallback to minimal package set."""
        self.logger.info("Attempting fallback: minimal package set")
        # Implementation would go here
        return False
    
    def _fallback_config_template(self, context: ErrorContext) -> bool:
        """Fallback to configuration template."""
        self.logger.info("Attempting fallback: configuration template")
        # Implementation would go here
        return False
    
    def _fallback_config_minimal(self, context: ErrorContext) -> bool:
        """Fallback to minimal configuration."""
        self.logger.info("Attempting fallback: minimal configuration")
        # Implementation would go here
        return False
    
    def _fallback_config_default(self, context: ErrorContext) -> bool:
        """Fallback to default configuration."""
        self.logger.info("Attempting fallback: default configuration")
        # Implementation would go here
        return False


class ComprehensiveErrorHandler(IErrorHandler, BaseInstallationComponent):
    """Comprehensive error handler with advanced recovery capabilities."""
    
    def __init__(self, installation_path: str, logger: Optional[logging.Logger] = None):
        super().__init__(installation_path, logger)
        self.classifier = ErrorClassifier()
        self.retry_manager = RetryManager(logger)
        self.fallback_manager = FallbackManager(logger)
        self.system_state_collector = SystemStateCollector()
        self.error_log_file = Path(installation_path) / "logs" / "errors.json"
        self.error_history: List[Dict[str, Any]] = []
        
        # Initialize intelligent retry system
        self.intelligent_retry_system = IntelligentRetrySystem(installation_path, logger)
        
        # Default retry configurations for different error categories
        self.retry_configs = {
            ErrorCategory.NETWORK: RetryConfig(max_attempts=5, base_delay=2.0),
            ErrorCategory.SYSTEM: RetryConfig(max_attempts=3, base_delay=1.0),
            ErrorCategory.CONFIGURATION: RetryConfig(max_attempts=2, base_delay=0.5),
            ErrorCategory.PERMISSION: RetryConfig(max_attempts=1, base_delay=0.0)
        }
    
    def create_enhanced_error_context(self, phase: str = "", task: str = "", 
                                    component: str = "", method: str = "",
                                    hardware_profile: Optional[HardwareProfile] = None,
                                    retry_count: int = 0,
                                    previous_errors: List[str] = None,
                                    recovery_attempts: List[str] = None) -> EnhancedErrorContext:
        """Create enhanced error context with comprehensive system state."""
        try:
            # Collect comprehensive system state
            system_info = self.system_state_collector.collect_system_info()
            system_resources = self.system_state_collector.collect_resource_snapshot()
            network_status = self.system_state_collector.collect_network_status()
            
            # Get current stack trace
            stack_trace = traceback.format_exc()
            
            return EnhancedErrorContext(
                timestamp=datetime.now(),
                phase=phase,
                task=task,
                component=component,
                method=method,
                hardware_profile=hardware_profile,
                system_info=system_info,
                environment_vars=system_info.environment_vars if system_info else {},
                stack_trace=stack_trace,
                retry_count=retry_count,
                previous_errors=previous_errors or [],
                recovery_attempts=recovery_attempts or [],
                system_resources=system_resources,
                network_status=network_status
            )
        except Exception as e:
            self.logger.warning(f"Failed to create enhanced error context: {e}")
            # Fallback to basic context
            return EnhancedErrorContext(
                timestamp=datetime.now(),
                phase=phase,
                task=task,
                component=component,
                method=method,
                hardware_profile=hardware_profile,
                stack_trace=traceback.format_exc(),
                retry_count=retry_count,
                previous_errors=previous_errors or [],
                recovery_attempts=recovery_attempts or []
            )
    
    def handle_error(self, error: InstallationError, context: Optional[ErrorContext] = None) -> RecoveryAction:
        """Handle installation error and return recovery action."""
        if context is None:
            context = ErrorContext()
        
        # Classify error if not already classified
        if not hasattr(error, 'category') or error.category is None:
            if isinstance(error, InstallationError):
                # Already classified
                pass
            else:
                # Classify the error
                category = self.classifier.classify_error(error, context)
                error = InstallationError(str(error), category)
        
        # Determine severity
        severity = self.classifier.determine_severity(error, context)
        
        # Log the error
        self.log_error(error, context.__dict__)
        
        # Add to error history
        self._add_to_error_history(error, context, severity)
        
        # Determine recovery action
        recovery_action = self._determine_recovery_action(error, context, severity)
        
        self.logger.info(f"Error recovery action: {recovery_action.value}")
        return recovery_action
    
    def log_error(self, error: InstallationError, context: Union[Dict[str, Any], EnhancedErrorContext, ErrorContext]) -> None:
        """Log error with comprehensive context information."""
        # Convert context to JSON-serializable format
        if isinstance(context, (EnhancedErrorContext, ErrorContext)):
            context_dict = self._serialize_error_context(context)
        else:
            context_dict = context
            
        serializable_context = {}
        for key, value in context_dict.items():
            try:
                if isinstance(value, datetime):
                    serializable_context[key] = value.isoformat()
                elif hasattr(value, '__dict__'):
                    # Convert objects to dict representation
                    serializable_context[key] = self._serialize_object(value)
                else:
                    json.dumps(value)  # Test if it's serializable
                    serializable_context[key] = value
            except (TypeError, ValueError):
                serializable_context[key] = str(value)
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_message": error.message,
            "error_category": error.category.value if error.category else "unknown",
            "context": serializable_context,
            "stack_trace": serializable_context.get("stack_trace", "") if isinstance(serializable_context, dict) else ""
        }
        
        # Log to file
        try:
            self.error_log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Load existing errors
            existing_errors = []
            if self.error_log_file.exists():
                try:
                    with open(self.error_log_file, 'r', encoding='utf-8') as f:
                        existing_errors = json.load(f)
                except (json.JSONDecodeError, IOError):
                    existing_errors = []
            
            # Add new error
            existing_errors.append(log_entry)
            
            # Keep only last 100 errors to prevent file from growing too large
            if len(existing_errors) > 100:
                existing_errors = existing_errors[-100:]
            
            # Save back to file
            with open(self.error_log_file, 'w', encoding='utf-8') as f:
                json.dump(existing_errors, f, indent=2, ensure_ascii=False)
        
        except Exception as e:
            self.logger.warning(f"Failed to log error to file: {e}")
        
        # Log to console/logger
        self.logger.error(f"Installation error: {error.message}")
        self.logger.error(f"Category: {error.category.value if error.category else 'unknown'}")
        if error.recovery_suggestions:
            self.logger.error(f"Recovery suggestions: {error.recovery_suggestions}")
    
    def suggest_recovery(self, error: InstallationError, context: Optional[ErrorContext] = None) -> List[str]:
        """Suggest comprehensive recovery actions for the error."""
        suggestions = error.recovery_suggestions.copy() if error.recovery_suggestions else []
        
        # Add category-specific suggestions
        if error.category == ErrorCategory.NETWORK:
            suggestions.extend([
                "Check your internet connection",
                "Verify firewall and proxy settings",
                "Try downloading from an alternative source",
                "Wait a few minutes and retry",
                "Use a VPN if in a restricted network"
            ])
        elif error.category == ErrorCategory.PERMISSION:
            suggestions.extend([
                "Run the installer as Administrator",
                "Check file and folder permissions",
                "Close any applications that might be using the files",
                "Temporarily disable antivirus software",
                "Ensure you have write access to the installation directory"
            ])
        elif error.category == ErrorCategory.SYSTEM:
            suggestions.extend([
                "Check available disk space (at least 50GB recommended)",
                "Verify system meets minimum requirements",
                "Close other applications to free up memory",
                "Update system drivers, especially GPU drivers",
                "Restart your computer and try again"
            ])
        elif error.category == ErrorCategory.CONFIGURATION:
            suggestions.extend([
                "Check configuration file syntax",
                "Verify all required settings are present",
                "Reset to default configuration",
                "Check for conflicting settings",
                "Validate hardware-specific parameters"
            ])
        
        # Add context-specific suggestions
        if context and context.hardware_profile:
            if context.hardware_profile.memory.available_gb < 8:
                suggestions.append("Consider upgrading system memory (8GB+ recommended)")
            if context.hardware_profile.gpu and "nvidia" in context.hardware_profile.gpu.model.lower():
                suggestions.append("Ensure NVIDIA drivers are up to date")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion not in seen:
                seen.add(suggestion)
                unique_suggestions.append(suggestion)
        
        return unique_suggestions
    
    def execute_with_retry_and_fallback(self, func: Callable, scenario: str,
                                      context: Optional[ErrorContext] = None) -> Any:
        """Execute a function with retry logic and fallback options."""
        if context is None:
            context = ErrorContext()
        
        # First, try with retry logic
        try:
            retry_config = self.retry_configs.get(ErrorCategory.NETWORK, RetryConfig())
            return self.retry_manager.execute_with_retry(func, retry_config, context)
        except Exception as e:
            self.logger.warning(f"All retry attempts failed for {scenario}: {e}")
            
            # Try fallback options
            fallback_options = self.fallback_manager.get_fallback_options(scenario, context)
            
            for i, fallback_func in enumerate(fallback_options):
                try:
                    self.logger.info(f"Trying fallback option {i + 1}/{len(fallback_options)}")
                    result = fallback_func(context)
                    if result:
                        self.logger.info(f"Fallback option {i + 1} succeeded")
                        return result
                except Exception as fallback_error:
                    self.logger.warning(f"Fallback option {i + 1} failed: {fallback_error}")
            
            # All fallbacks failed
            self.logger.error(f"All retry attempts and fallbacks failed for {scenario}")
            raise e
    
    def _determine_recovery_action(self, error: InstallationError, 
                                 context: ErrorContext, severity: ErrorSeverity) -> RecoveryAction:
        """Determine the appropriate recovery action for an error."""
        # Critical errors should abort
        if severity == ErrorSeverity.CRITICAL:
            return RecoveryAction.ABORT
        
        # Permission errors need elevation
        if error.category == ErrorCategory.PERMISSION:
            return RecoveryAction.ELEVATE_PERMISSIONS
        
        # Network errors can be retried
        if error.category == ErrorCategory.NETWORK:
            if context.retry_count < 3:
                return RecoveryAction.RETRY
            else:
                return RecoveryAction.RETRY_WITH_FALLBACK
        
        # System errors depend on severity
        if error.category == ErrorCategory.SYSTEM:
            if severity == ErrorSeverity.CRITICAL:
                return RecoveryAction.ABORT
            elif severity == ErrorSeverity.HIGH:
                return RecoveryAction.MANUAL_INTERVENTION
            elif context.retry_count < 2:
                return RecoveryAction.RETRY
            else:
                return RecoveryAction.ROLLBACK
        
        # Configuration errors can usually be fixed
        if error.category == ErrorCategory.CONFIGURATION:
            return RecoveryAction.RETRY_WITH_FALLBACK
        
        # Default action
        return RecoveryAction.CONTINUE
    
    def _add_to_error_history(self, error: InstallationError, 
                            context: ErrorContext, severity: ErrorSeverity) -> None:
        """Add error to internal history for pattern analysis."""
        error_entry = {
            "timestamp": context.timestamp.isoformat(),
            "message": error.message,
            "category": error.category.value if error.category else "unknown",
            "severity": severity.value,
            "phase": context.phase,
            "task": context.task,
            "retry_count": context.retry_count
        }
        
        self.error_history.append(error_entry)
        
        # Keep only last 50 errors in memory
        if len(self.error_history) > 50:
            self.error_history = self.error_history[-50:]
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get statistics about errors encountered during installation."""
        if not self.error_history:
            return {"total_errors": 0}
        
        stats = {
            "total_errors": len(self.error_history),
            "by_category": {},
            "by_severity": {},
            "by_phase": {},
            "most_common_errors": []
        }
        
        # Count by category
        for error in self.error_history:
            category = error.get("category", "unknown")
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1
            
            severity = error.get("severity", "unknown")
            stats["by_severity"][severity] = stats["by_severity"].get(severity, 0) + 1
            
            phase = error.get("phase", "unknown")
            stats["by_phase"][phase] = stats["by_phase"].get(phase, 0) + 1
        
        # Find most common error messages
        error_counts = {}
        for error in self.error_history:
            message = error.get("message", "")
            error_counts[message] = error_counts.get(message, 0) + 1
        
        stats["most_common_errors"] = sorted(
            error_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]
        
        return stats
    
    def _serialize_error_context(self, context: Union[EnhancedErrorContext, ErrorContext]) -> Dict[str, Any]:
        """Serialize error context to dictionary."""
        result = {}
        for field in context.__dataclass_fields__:
            value = getattr(context, field)
            if value is not None:
                if isinstance(value, datetime):
                    result[field] = value.isoformat()
                elif hasattr(value, '__dict__'):
                    result[field] = self._serialize_object(value)
                else:
                    result[field] = value
        return result
    
    def _serialize_object(self, obj: Any) -> Dict[str, Any]:
        """Serialize an object to dictionary."""
        if hasattr(obj, '__dict__'):
            result = {}
            for key, value in obj.__dict__.items():
                if not key.startswith('_'):
                    try:
                        if isinstance(value, datetime):
                            result[key] = value.isoformat()
                        elif isinstance(value, (str, int, float, bool, type(None))):
                            result[key] = value
                        elif isinstance(value, (list, dict)):
                            json.dumps(value)  # Test if serializable
                            result[key] = value
                        else:
                            result[key] = str(value)
                    except (TypeError, ValueError):
                        result[key] = str(value)
            return result
        else:
            return str(obj)
    
    def execute_with_intelligent_retry(self, operation: Callable, operation_name: str,
                                     error_category: Optional[ErrorCategory] = None,
                                     context: Optional[Dict[str, Any]] = None,
                                     custom_config: Optional[RetryConfiguration] = None) -> Any:
        """
        Execute an operation with intelligent retry logic.
        
        This method integrates the intelligent retry system with the existing error handler,
        providing enhanced retry capabilities with user control and exponential backoff.
        
        Args:
            operation: The operation to execute
            operation_name: Human-readable name for the operation
            error_category: Optional error category for strategy selection
            context: Additional context for retry strategy selection
            custom_config: Optional custom retry configuration
            
        Returns:
            Result of the successful operation
            
        Raises:
            Exception: The final exception if all retries fail
        """
        try:
            return self.intelligent_retry_system.execute_with_retry(
                operation=operation,
                operation_name=operation_name,
                error_category=error_category,
                context=context,
                custom_config=custom_config
            )
        except Exception as e:
            # Log the final failure with enhanced context
            enhanced_context = self.create_enhanced_error_context(
                task=operation_name,
                component="IntelligentRetrySystem"
            )
            
            if isinstance(e, InstallationError):
                self.log_error(e, enhanced_context)
            else:
                # Convert to InstallationError for consistent handling
                category = self.classifier.classify_error(e, ErrorContext())
                installation_error = InstallationError(str(e), category)
                self.log_error(installation_error, enhanced_context)
            
            raise
    
    def configure_retry_behavior(self, error_category: ErrorCategory, 
                               config: RetryConfiguration) -> None:
        """
        Configure retry behavior for a specific error category.
        
        Args:
            error_category: The error category to configure
            config: The retry configuration to apply
        """
        # Update both legacy and intelligent retry systems
        legacy_config = RetryConfig(
            max_attempts=config.max_attempts,
            base_delay=config.base_delay,
            max_delay=config.max_delay,
            backoff_multiplier=config.backoff_multiplier,
            jitter=config.jitter
        )
        self.retry_configs[error_category] = legacy_config
        
        # Update intelligent retry system's global configuration for this category
        # Note: The simplified retry system uses global config, so we update it
        if error_category == ErrorCategory.NETWORK:
            self.intelligent_retry_system.set_global_configuration(config)
        
        self.logger.info(f"Updated retry configuration for {error_category.value}")
    
    def get_retry_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive retry statistics from both legacy and intelligent retry systems.
        
        Returns:
            Dictionary containing retry statistics
        """
        # Get statistics from intelligent retry system
        intelligent_stats = self.intelligent_retry_system.get_session_statistics()
        
        # Get error statistics from legacy system
        error_stats = self.get_error_statistics()
        
        # Combine statistics
        combined_stats = {
            'intelligent_retry_system': intelligent_stats,
            'error_handler_stats': error_stats,
            'active_retry_sessions': len(self.intelligent_retry_system.get_active_sessions()),
            'retry_configurations': {
                category.value: {
                    'max_attempts': config.max_attempts,
                    'base_delay': config.base_delay,
                    'max_delay': config.max_delay,
                    'backoff_multiplier': config.backoff_multiplier,
                    'jitter': config.jitter
                }
                for category, config in self.retry_configs.items()
            }
        }
        
        return combined_stats
    
    def set_user_prompt_enabled(self, enabled: bool) -> None:
        """
        Enable or disable user prompts for retry decisions.
        
        Args:
            enabled: Whether to enable user prompts
        """
        self.intelligent_retry_system.global_config.user_prompt = enabled
        self.logger.info(f"User prompts for retry decisions: {'enabled' if enabled else 'disabled'}")
    
    def cancel_active_retry_session(self, operation_name: str) -> bool:
        """
        Cancel an active retry session.
        
        Args:
            operation_name: Name of the operation to cancel
            
        Returns:
            True if session was cancelled, False if not found
        """
        return self.intelligent_retry_system.cancel_session(operation_name)
    
    def get_active_retry_sessions(self) -> Dict[str, Any]:
        """
        Get information about currently active retry sessions.
        
        Returns:
            Dictionary of active retry sessions
        """
        active_sessions = self.intelligent_retry_system.get_active_sessions()
        
        # Convert to serializable format
        serializable_sessions = {}
        for name, session in active_sessions.items():
            serializable_sessions[name] = {
                'operation_name': session.operation_name,
                'start_time': session.start_time.isoformat(),
                'total_attempts': session.total_attempts,
                'successful': session.successful,
                'user_decisions': session.user_decisions,
                'attempts_count': len(session.attempts)
            }
        
        return serializable_sessions