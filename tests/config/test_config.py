"""
Test Configuration System

This module provides centralized configuration for the test suite, including
test categories, timeouts, resource limits, and environment-specific settings.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent


class TestCategory(Enum):
    """Test categories with different requirements."""
    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    PERFORMANCE = "performance"
    STRESS = "stress"
    RELIABILITY = "reliability"
    SECURITY = "security"


class TestPriority(Enum):
    """Test priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class TestTimeouts:
    """Test timeout configuration."""
    unit: int = 30  # seconds
    integration: int = 120  # 2 minutes
    e2e: int = 300  # 5 minutes
    performance: int = 600  # 10 minutes
    stress: int = 1800  # 30 minutes
    reliability: int = 3600  # 1 hour
    security: int = 180  # 3 minutes


@dataclass
class ResourceLimits:
    """Resource limits for tests."""
    max_memory_mb: int = 2048
    max_cpu_percent: float = 80.0
    max_disk_space_mb: int = 5120
    max_network_requests: int = 1000
    max_file_handles: int = 1000
    max_processes: int = 10


@dataclass
class RetryConfiguration:
    """Retry configuration for flaky tests."""
    enabled: bool = True
    max_attempts: int = 3
    delay_seconds: float = 1.0
    exponential_backoff: bool = True
    backoff_multiplier: float = 2.0
    max_delay_seconds: float = 30.0
    retry_on_errors: List[str] = field(default_factory=lambda: [
        "ConnectionError", "TimeoutError", "TemporaryFailure"
    ])


@dataclass
class ParallelConfiguration:
    """Parallel execution configuration."""
    enabled: bool = True
    max_workers: int = 4
    worker_timeout: int = 300
    shared_resources: List[str] = field(default_factory=lambda: [
        "database", "network", "filesystem"
    ])
    isolation_level: str = "process"  # process, thread, or none


@dataclass
class CoverageConfiguration:
    """Code coverage configuration."""
    enabled: bool = True
    min_coverage_percent: float = 80.0
    fail_under: float = 70.0
    include_patterns: List[str] = field(default_factory=lambda: [
        "backend/*", "frontend/src/*", "scripts/*", "tools/*"
    ])
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "*/tests/*", "*/test_*", "*/__pycache__/*", "*/node_modules/*"
    ])
    report_formats: List[str] = field(default_factory=lambda: ["html", "xml", "json"])


@dataclass
class LoggingConfiguration:
    """Test logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    capture_stdout: bool = True
    capture_stderr: bool = True
    log_file_enabled: bool = True
    log_file_path: str = "tests/logs/test.log"
    max_log_size_mb: int = 100
    backup_count: int = 5


@dataclass
class DatabaseConfiguration:
    """Test database configuration."""
    engine: str = "sqlite"
    url: str = "sqlite:///tests/fixtures/data/test.db"
    pool_size: int = 5
    max_overflow: int = 10
    echo: bool = False
    isolation_level: str = "READ_UNCOMMITTED"
    auto_create_tables: bool = True
    auto_cleanup: bool = True


@dataclass
class NetworkConfiguration:
    """Test network configuration."""
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    allowed_hosts: List[str] = field(default_factory=lambda: [
        "localhost", "127.0.0.1", "::1"
    ])
    mock_external_requests: bool = True
    record_requests: bool = True
    request_log_path: str = "tests/logs/requests.log"


@dataclass
class SecurityConfiguration:
    """Test security configuration."""
    allow_admin_elevation: bool = False
    sandbox_enabled: bool = True
    network_isolation: bool = True
    file_system_isolation: bool = True
    process_isolation: bool = True
    sensitive_data_masking: bool = True
    audit_logging: bool = True


class TestConfiguration:
    """Main test configuration class."""
    
    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file or Path(__file__).parent / "test-config.yaml"
        self.timeouts = TestTimeouts()
        self.resource_limits = ResourceLimits()
        self.retry_config = RetryConfiguration()
        self.parallel_config = ParallelConfiguration()
        self.coverage_config = CoverageConfiguration()
        self.logging_config = LoggingConfiguration()
        self.database_config = DatabaseConfiguration()
        self.network_config = NetworkConfiguration()
        self.security_config = SecurityConfiguration()
        
        # Load configuration from file if it exists
        if self.config_file.exists():
            self.load_from_file()
        
        # Override with environment variables
        self.load_from_environment()
    
    def load_from_file(self):
        """Load configuration from file."""
        try:
            if self.config_file.suffix.lower() == '.json':
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
            elif self.config_file.suffix.lower() in ['.yaml', '.yml']:
                try:
                    import yaml
                    with open(self.config_file, 'r') as f:
                        config_data = yaml.safe_load(f)
                except ImportError:
                    print("Warning: PyYAML not installed, cannot load YAML config")
                    return
            else:
                print(f"Warning: Unsupported config file format: {self.config_file.suffix}")
                return
            
            self._update_from_dict(config_data)
            
        except Exception as e:
            print(f"Warning: Failed to load config from {self.config_file}: {e}")
    
    def load_from_environment(self):
        """Load configuration from environment variables."""
        env_mappings = {
            # Timeouts
            "TEST_TIMEOUT_UNIT": ("timeouts", "unit", int),
            "TEST_TIMEOUT_INTEGRATION": ("timeouts", "integration", int),
            "TEST_TIMEOUT_E2E": ("timeouts", "e2e", int),
            "TEST_TIMEOUT_PERFORMANCE": ("timeouts", "performance", int),
            
            # Resource limits
            "TEST_MAX_MEMORY_MB": ("resource_limits", "max_memory_mb", int),
            "TEST_MAX_CPU_PERCENT": ("resource_limits", "max_cpu_percent", float),
            "TEST_MAX_DISK_SPACE_MB": ("resource_limits", "max_disk_space_mb", int),
            
            # Retry configuration
            "TEST_RETRY_ENABLED": ("retry_config", "enabled", lambda x: x.lower() == 'true'),
            "TEST_RETRY_MAX_ATTEMPTS": ("retry_config", "max_attempts", int),
            "TEST_RETRY_DELAY": ("retry_config", "delay_seconds", float),
            
            # Parallel configuration
            "TEST_PARALLEL_ENABLED": ("parallel_config", "enabled", lambda x: x.lower() == 'true'),
            "TEST_PARALLEL_MAX_WORKERS": ("parallel_config", "max_workers", int),
            
            # Coverage configuration
            "TEST_COVERAGE_ENABLED": ("coverage_config", "enabled", lambda x: x.lower() == 'true'),
            "TEST_COVERAGE_MIN_PERCENT": ("coverage_config", "min_coverage_percent", float),
            
            # Logging configuration
            "TEST_LOG_LEVEL": ("logging_config", "level", str),
            "TEST_LOG_FILE_ENABLED": ("logging_config", "log_file_enabled", lambda x: x.lower() == 'true'),
            
            # Database configuration
            "TEST_DATABASE_URL": ("database_config", "url", str),
            "TEST_DATABASE_ECHO": ("database_config", "echo", lambda x: x.lower() == 'true'),
            
            # Network configuration
            "TEST_NETWORK_TIMEOUT": ("network_config", "timeout_seconds", int),
            "TEST_MOCK_EXTERNAL_REQUESTS": ("network_config", "mock_external_requests", lambda x: x.lower() == 'true'),
            
            # Security configuration
            "TEST_ALLOW_ADMIN": ("security_config", "allow_admin_elevation", lambda x: x.lower() == 'true'),
            "TEST_SANDBOX_ENABLED": ("security_config", "sandbox_enabled", lambda x: x.lower() == 'true'),
        }
        
        for env_var, (config_section, config_key, converter) in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                try:
                    converted_value = converter(env_value)
                    config_obj = getattr(self, config_section)
                    setattr(config_obj, config_key, converted_value)
                except (ValueError, TypeError) as e:
                    print(f"Warning: Invalid value for {env_var}: {env_value} ({e})")
    
    def _update_from_dict(self, config_data: Dict[str, Any]):
        """Update configuration from dictionary."""
        for section_name, section_data in config_data.items():
            if hasattr(self, section_name) and isinstance(section_data, dict):
                config_obj = getattr(self, section_name)
                for key, value in section_data.items():
                    if hasattr(config_obj, key):
                        setattr(config_obj, key, value)
    
    def get_timeout_for_category(self, category: TestCategory) -> int:
        """Get timeout for test category."""
        timeout_map = {
            TestCategory.UNIT: self.timeouts.unit,
            TestCategory.INTEGRATION: self.timeouts.integration,
            TestCategory.E2E: self.timeouts.e2e,
            TestCategory.PERFORMANCE: self.timeouts.performance,
            TestCategory.STRESS: self.timeouts.stress,
            TestCategory.RELIABILITY: self.timeouts.reliability,
            TestCategory.SECURITY: self.timeouts.security,
        }
        return timeout_map.get(category, self.timeouts.unit)
    
    def should_retry_test(self, test_name: str, error_type: str) -> bool:
        """Check if a test should be retried."""
        if not self.retry_config.enabled:
            return False
        
        # Check if error type is in retry list
        return any(retry_error in error_type for retry_error in self.retry_config.retry_on_errors)
    
    def get_retry_delay(self, attempt: int) -> float:
        """Get retry delay for attempt number."""
        if not self.retry_config.exponential_backoff:
            return self.retry_config.delay_seconds
        
        delay = self.retry_config.delay_seconds * (self.retry_config.backoff_multiplier ** (attempt - 1))
        return min(delay, self.retry_config.max_delay_seconds)
    
    def is_parallel_execution_enabled(self, test_category: TestCategory) -> bool:
        """Check if parallel execution is enabled for test category."""
        if not self.parallel_config.enabled:
            return False
        
        # Some test categories should not run in parallel
        sequential_categories = [TestCategory.E2E, TestCategory.STRESS]
        return test_category not in sequential_categories
    
    def get_coverage_threshold(self, test_category: TestCategory) -> float:
        """Get coverage threshold for test category."""
        # Different categories may have different coverage requirements
        category_thresholds = {
            TestCategory.UNIT: self.coverage_config.min_coverage_percent,
            TestCategory.INTEGRATION: self.coverage_config.min_coverage_percent * 0.8,
            TestCategory.E2E: self.coverage_config.min_coverage_percent * 0.6,
            TestCategory.PERFORMANCE: self.coverage_config.min_coverage_percent * 0.4,
        }
        return category_thresholds.get(test_category, self.coverage_config.min_coverage_percent)
    
    def get_database_url(self, test_id: str) -> str:
        """Get database URL for test."""
        if self.database_config.url.startswith("sqlite://"):
            # Create test-specific database
            base_path = Path(self.database_config.url.replace("sqlite:///", ""))
            test_db_path = base_path.parent / f"test_{test_id.replace('::', '_')}.db"
            return f"sqlite:///{test_db_path}"
        
        return self.database_config.url
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "timeouts": {
                "unit": self.timeouts.unit,
                "integration": self.timeouts.integration,
                "e2e": self.timeouts.e2e,
                "performance": self.timeouts.performance,
                "stress": self.timeouts.stress,
                "reliability": self.timeouts.reliability,
                "security": self.timeouts.security,
            },
            "resource_limits": {
                "max_memory_mb": self.resource_limits.max_memory_mb,
                "max_cpu_percent": self.resource_limits.max_cpu_percent,
                "max_disk_space_mb": self.resource_limits.max_disk_space_mb,
                "max_network_requests": self.resource_limits.max_network_requests,
                "max_file_handles": self.resource_limits.max_file_handles,
                "max_processes": self.resource_limits.max_processes,
            },
            "retry_config": {
                "enabled": self.retry_config.enabled,
                "max_attempts": self.retry_config.max_attempts,
                "delay_seconds": self.retry_config.delay_seconds,
                "exponential_backoff": self.retry_config.exponential_backoff,
                "backoff_multiplier": self.retry_config.backoff_multiplier,
                "max_delay_seconds": self.retry_config.max_delay_seconds,
                "retry_on_errors": self.retry_config.retry_on_errors,
            },
            "parallel_config": {
                "enabled": self.parallel_config.enabled,
                "max_workers": self.parallel_config.max_workers,
                "worker_timeout": self.parallel_config.worker_timeout,
                "shared_resources": self.parallel_config.shared_resources,
                "isolation_level": self.parallel_config.isolation_level,
            },
            "coverage_config": {
                "enabled": self.coverage_config.enabled,
                "min_coverage_percent": self.coverage_config.min_coverage_percent,
                "fail_under": self.coverage_config.fail_under,
                "include_patterns": self.coverage_config.include_patterns,
                "exclude_patterns": self.coverage_config.exclude_patterns,
                "report_formats": self.coverage_config.report_formats,
            },
            "logging_config": {
                "level": self.logging_config.level,
                "format": self.logging_config.format,
                "capture_stdout": self.logging_config.capture_stdout,
                "capture_stderr": self.logging_config.capture_stderr,
                "log_file_enabled": self.logging_config.log_file_enabled,
                "log_file_path": self.logging_config.log_file_path,
                "max_log_size_mb": self.logging_config.max_log_size_mb,
                "backup_count": self.logging_config.backup_count,
            },
            "database_config": {
                "engine": self.database_config.engine,
                "url": self.database_config.url,
                "pool_size": self.database_config.pool_size,
                "max_overflow": self.database_config.max_overflow,
                "echo": self.database_config.echo,
                "isolation_level": self.database_config.isolation_level,
                "auto_create_tables": self.database_config.auto_create_tables,
                "auto_cleanup": self.database_config.auto_cleanup,
            },
            "network_config": {
                "timeout_seconds": self.network_config.timeout_seconds,
                "max_retries": self.network_config.max_retries,
                "retry_delay": self.network_config.retry_delay,
                "allowed_hosts": self.network_config.allowed_hosts,
                "mock_external_requests": self.network_config.mock_external_requests,
                "record_requests": self.network_config.record_requests,
                "request_log_path": self.network_config.request_log_path,
            },
            "security_config": {
                "allow_admin_elevation": self.security_config.allow_admin_elevation,
                "sandbox_enabled": self.security_config.sandbox_enabled,
                "network_isolation": self.security_config.network_isolation,
                "file_system_isolation": self.security_config.file_system_isolation,
                "process_isolation": self.security_config.process_isolation,
                "sensitive_data_masking": self.security_config.sensitive_data_masking,
                "audit_logging": self.security_config.audit_logging,
            }
        }
    
    def save_to_file(self, file_path: Optional[Path] = None):
        """Save configuration to file."""
        if file_path is None:
            file_path = self.config_file
        
        config_dict = self.to_dict()
        
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        elif file_path.suffix.lower() in ['.yaml', '.yml']:
            try:
                import yaml
                with open(file_path, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            except ImportError:
                print("Warning: PyYAML not installed, saving as JSON")
                json_path = file_path.with_suffix('.json')
                with open(json_path, 'w') as f:
                    json.dump(config_dict, f, indent=2)


# Global configuration instance
_test_config = TestConfiguration()


def get_test_config() -> TestConfiguration:
    """Get the global test configuration."""
    return _test_config


def reload_test_config(config_file: Optional[Path] = None):
    """Reload the global test configuration."""
    global _test_config
    _test_config = TestConfiguration(config_file)


# Export main classes and functions
__all__ = [
    'TestConfiguration', 'TestCategory', 'TestPriority', 'TestTimeouts',
    'ResourceLimits', 'RetryConfiguration', 'ParallelConfiguration',
    'CoverageConfiguration', 'LoggingConfiguration', 'DatabaseConfiguration',
    'NetworkConfiguration', 'SecurityConfiguration', 'get_test_config',
    'reload_test_config'
]
