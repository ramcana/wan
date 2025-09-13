"""
Environment Validator for Test Isolation

This module provides validation and setup for test environments, ensuring
that tests run in properly isolated and configured environments.
"""

import os
import sys
import json
import tempfile
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pytest

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class EnvironmentType(Enum):
    """Test environment types."""
    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    PERFORMANCE = "performance"
    STRESS = "stress"
    RELIABILITY = "reliability"


class ValidationLevel(Enum):
    """Validation levels."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    STRICT = "strict"
    COMPREHENSIVE = "comprehensive"


@dataclass
class EnvironmentRequirements:
    """Environment requirements specification."""
    python_version: Optional[str] = None
    node_version: Optional[str] = None
    npm_version: Optional[str] = None
    required_packages: List[str] = field(default_factory=list)
    required_ports: List[int] = field(default_factory=list)
    required_env_vars: List[str] = field(default_factory=list)
    forbidden_env_vars: List[str] = field(default_factory=list)
    min_memory_mb: Optional[int] = None
    min_disk_space_mb: Optional[int] = None
    required_commands: List[str] = field(default_factory=list)
    network_access: bool = False
    admin_privileges: bool = False
    docker_required: bool = False


@dataclass
class ValidationResult:
    """Environment validation result."""
    valid: bool
    environment_type: EnvironmentType
    validation_level: ValidationLevel
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    environment_info: Dict[str, Any] = field(default_factory=dict)


class EnvironmentValidator:
    """Validates and sets up test environments."""
    
    def __init__(self):
        self.validation_cache: Dict[str, ValidationResult] = {}
    
    def validate_environment(self, env_type: EnvironmentType, 
                           validation_level: ValidationLevel = ValidationLevel.STANDARD,
                           requirements: Optional[EnvironmentRequirements] = None) -> ValidationResult:
        """Validate the test environment."""
        cache_key = f"{env_type.value}_{validation_level.value}"
        
        if cache_key in self.validation_cache:
            return self.validation_cache[cache_key]
        
        if requirements is None:
            requirements = self._get_default_requirements(env_type)
        
        result = ValidationResult(
            valid=True,
            environment_type=env_type,
            validation_level=validation_level
        )
        
        # Collect environment information
        result.environment_info = self._collect_environment_info()
        
        # Perform validations based on level
        if validation_level in [ValidationLevel.MINIMAL, ValidationLevel.STANDARD, 
                              ValidationLevel.STRICT, ValidationLevel.COMPREHENSIVE]:
            self._validate_basic_requirements(result, requirements)
        
        if validation_level in [ValidationLevel.STANDARD, ValidationLevel.STRICT, 
                              ValidationLevel.COMPREHENSIVE]:
            self._validate_system_resources(result, requirements)
            self._validate_network_access(result, requirements)
        
        if validation_level in [ValidationLevel.STRICT, ValidationLevel.COMPREHENSIVE]:
            self._validate_security_requirements(result, requirements)
            self._validate_dependencies(result, requirements)
        
        if validation_level == ValidationLevel.COMPREHENSIVE:
            self._validate_performance_requirements(result, requirements)
            self._validate_isolation_requirements(result, requirements)
        
        # Cache result
        self.validation_cache[cache_key] = result
        
        return result
    
    def setup_environment(self, env_type: EnvironmentType, 
                         test_id: str,
                         requirements: Optional[EnvironmentRequirements] = None) -> Dict[str, Any]:
        """Set up a test environment."""
        if requirements is None:
            requirements = self._get_default_requirements(env_type)
        
        setup_info = {
            "test_id": test_id,
            "environment_type": env_type.value,
            "temp_dirs": [],
            "temp_files": [],
            "env_vars": {},
            "processes": [],
            "ports": [],
            "cleanup_callbacks": []
        }
        
        # Create temporary directories
        temp_base = Path(tempfile.mkdtemp(prefix=f"test_{env_type.value}_{test_id}_"))
        setup_info["temp_dirs"].append(temp_base)
        
        # Set up environment variables
        test_env_vars = self._get_test_environment_variables(env_type, test_id, temp_base)
        for var_name, var_value in test_env_vars.items():
            original_value = os.environ.get(var_name)
            os.environ[var_name] = var_value
            setup_info["env_vars"][var_name] = original_value
        
        # Reserve required ports
        for port in requirements.required_ports:
            if self._is_port_available(port):
                setup_info["ports"].append(port)
            else:
                # Find alternative port
                alt_port = self._find_available_port(port + 1000)
                setup_info["ports"].append(alt_port)
        
        # Create test-specific directories
        test_dirs = {
            "data": temp_base / "data",
            "logs": temp_base / "logs",
            "config": temp_base / "config",
            "temp": temp_base / "temp",
            "cache": temp_base / "cache"
        }
        
        for dir_name, dir_path in test_dirs.items():
            dir_path.mkdir(parents=True, exist_ok=True)
            setup_info[f"{dir_name}_dir"] = dir_path
        
        return setup_info
    
    def cleanup_environment(self, setup_info: Dict[str, Any]):
        """Clean up a test environment."""
        # Run cleanup callbacks
        for callback in setup_info.get("cleanup_callbacks", []):
            try:
                callback()
            except Exception as e:
                print(f"Warning: Cleanup callback failed: {e}")
        
        # Restore environment variables
        for var_name, original_value in setup_info.get("env_vars", {}).items():
            if original_value is None:
                os.environ.pop(var_name, None)
            else:
                os.environ[var_name] = original_value
        
        # Clean up temporary directories
        for temp_dir in setup_info.get("temp_dirs", []):
            try:
                if Path(temp_dir).exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                print(f"Warning: Failed to clean up temp dir {temp_dir}: {e}")
        
        # Clean up temporary files
        for temp_file in setup_info.get("temp_files", []):
            try:
                if Path(temp_file).exists():
                    Path(temp_file).unlink()
            except Exception as e:
                print(f"Warning: Failed to clean up temp file {temp_file}: {e}")
        
        # Clean up processes
        for pid in setup_info.get("processes", []):
            try:
                import psutil
                process = psutil.Process(pid)
                if process.is_running():
                    process.terminate()
                    process.wait(timeout=5)
            except Exception:
                pass
    
    def _get_default_requirements(self, env_type: EnvironmentType) -> EnvironmentRequirements:
        """Get default requirements for environment type."""
        base_requirements = EnvironmentRequirements(
            python_version="3.8.0",
            required_packages=["pytest", "pytest-cov"],
            required_env_vars=["TESTING"],
            forbidden_env_vars=["PRODUCTION", "LIVE"],
            min_memory_mb=512,
            min_disk_space_mb=1024
        )
        
        if env_type == EnvironmentType.UNIT:
            return base_requirements
        
        elif env_type == EnvironmentType.INTEGRATION:
            base_requirements.required_ports = [8000, 3000]
            base_requirements.required_packages.extend(["requests", "psutil"])
            base_requirements.min_memory_mb = 1024
            
        elif env_type == EnvironmentType.E2E:
            base_requirements.required_ports = [8000, 3000, 4444]  # Selenium port
            base_requirements.required_packages.extend(["selenium", "requests", "psutil"])
            base_requirements.network_access = True
            base_requirements.min_memory_mb = 2048
            
        elif env_type == EnvironmentType.PERFORMANCE:
            base_requirements.required_packages.extend(["psutil", "memory_profiler"])
            base_requirements.min_memory_mb = 2048
            base_requirements.min_disk_space_mb = 5120
            
        elif env_type == EnvironmentType.STRESS:
            base_requirements.required_packages.extend(["psutil", "memory_profiler", "locust"])
            base_requirements.network_access = True
            base_requirements.min_memory_mb = 4096
            base_requirements.min_disk_space_mb = 10240
            
        elif env_type == EnvironmentType.RELIABILITY:
            base_requirements.required_packages.extend(["psutil", "requests"])
            base_requirements.network_access = True
            base_requirements.min_memory_mb = 1024
            base_requirements.docker_required = True
        
        return base_requirements
    
    def _collect_environment_info(self) -> Dict[str, Any]:
        """Collect information about the current environment."""
        info = {
            "python_version": sys.version,
            "platform": sys.platform,
            "executable": sys.executable,
            "path": sys.path[:5],  # First 5 entries
            "cwd": os.getcwd(),
            "user": os.environ.get("USER", os.environ.get("USERNAME", "unknown")),
            "env_vars": {
                key: value for key, value in os.environ.items()
                if any(keyword in key.upper() for keyword in ["TEST", "PYTEST", "WAN22", "DEBUG"])
            }
        }
        
        # Add system information if available
        try:
            import psutil
            info.update({
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available,
                "disk_free": psutil.disk_usage('.').free,
                "cpu_count": psutil.cpu_count()
            })
        except ImportError:
            pass
        
        return info
    
    def _validate_basic_requirements(self, result: ValidationResult, requirements: EnvironmentRequirements):
        """Validate basic requirements."""
        # Check Python version
        if requirements.python_version:
            current_version = sys.version_info
            required_parts = requirements.python_version.split('.')
            required_version = tuple(int(part) for part in required_parts)
            
            if current_version < required_version:
                result.errors.append(
                    f"Python version {requirements.python_version} required, "
                    f"but {sys.version} is installed"
                )
                result.valid = False
        
        # Check required environment variables
        for env_var in requirements.required_env_vars:
            if env_var not in os.environ:
                result.warnings.append(f"Required environment variable {env_var} is not set")
        
        # Check forbidden environment variables
        for env_var in requirements.forbidden_env_vars:
            if env_var in os.environ:
                result.errors.append(f"Forbidden environment variable {env_var} is set")
                result.valid = False
        
        # Check required packages
        for package in requirements.required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                result.errors.append(f"Required package {package} is not installed")
                result.valid = False
    
    def _validate_system_resources(self, result: ValidationResult, requirements: EnvironmentRequirements):
        """Validate system resources."""
        try:
            import psutil
            
            # Check memory
            if requirements.min_memory_mb:
                available_mb = psutil.virtual_memory().available // (1024 * 1024)
                if available_mb < requirements.min_memory_mb:
                    result.warnings.append(
                        f"Low memory: {available_mb}MB available, "
                        f"{requirements.min_memory_mb}MB recommended"
                    )
            
            # Check disk space
            if requirements.min_disk_space_mb:
                free_mb = psutil.disk_usage('.').free // (1024 * 1024)
                if free_mb < requirements.min_disk_space_mb:
                    result.warnings.append(
                        f"Low disk space: {free_mb}MB available, "
                        f"{requirements.min_disk_space_mb}MB recommended"
                    )
        
        except ImportError:
            result.warnings.append("psutil not available, cannot check system resources")
    
    def _validate_network_access(self, result: ValidationResult, requirements: EnvironmentRequirements):
        """Validate network access."""
        if requirements.network_access:
            try:
                socket.create_connection(("8.8.8.8", 53), timeout=3)
            except OSError:
                result.warnings.append("Network access required but not available")
        
        # Check required ports
        for port in requirements.required_ports:
            if not self._is_port_available(port):
                result.warnings.append(f"Port {port} is not available")
    
    def _validate_security_requirements(self, result: ValidationResult, requirements: EnvironmentRequirements):
        """Validate security requirements."""
        if requirements.admin_privileges:
            if os.name == 'nt':  # Windows
                try:
                    import ctypes
                    if not ctypes.windll.shell32.IsUserAnAdmin():
                        result.warnings.append("Admin privileges required but not available")
                except:
                    result.warnings.append("Cannot check admin privileges on Windows")
            else:  # Unix-like
                if os.geteuid() != 0:
                    result.warnings.append("Root privileges required but not available")
    
    def _validate_dependencies(self, result: ValidationResult, requirements: EnvironmentRequirements):
        """Validate external dependencies."""
        # Check required commands
        for command in requirements.required_commands:
            try:
                subprocess.run([command, "--version"], 
                             capture_output=True, 
                             timeout=5, 
                             check=True)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
                result.warnings.append(f"Required command '{command}' is not available")
        
        # Check Docker if required
        if requirements.docker_required:
            try:
                subprocess.run(["docker", "--version"], 
                             capture_output=True, 
                             timeout=5, 
                             check=True)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
                result.errors.append("Docker is required but not available")
                result.valid = False
    
    def _validate_performance_requirements(self, result: ValidationResult, requirements: EnvironmentRequirements):
        """Validate performance requirements."""
        try:
            import psutil
            
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 80:
                result.warnings.append(f"High CPU usage: {cpu_percent}%")
            
            # Check memory usage
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 80:
                result.warnings.append(f"High memory usage: {memory_percent}%")
            
            # Check disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io and disk_io.read_bytes > 1024**3:  # 1GB
                result.warnings.append("High disk I/O detected")
        
        except ImportError:
            result.warnings.append("Cannot validate performance requirements without psutil")
    
    def _validate_isolation_requirements(self, result: ValidationResult, requirements: EnvironmentRequirements):
        """Validate isolation requirements."""
        # Check for conflicting processes
        try:
            import psutil
            
            conflicting_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or []).lower()
                    if any(keyword in cmdline for keyword in ['pytest', 'test', 'wan22']):
                        if proc.pid != os.getpid():
                            conflicting_processes.append(proc.info['name'])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            if conflicting_processes:
                result.warnings.append(
                    f"Potentially conflicting processes detected: {', '.join(set(conflicting_processes))}"
                )
        
        except ImportError:
            pass
        
        # Check for test environment variables from other tests
        test_env_vars = [key for key in os.environ.keys() 
                        if key.startswith(('TEST_', 'PYTEST_', 'WAN22_TEST_'))]
        
        if len(test_env_vars) > 3:  # Allow a few standard ones
            result.warnings.append(
                f"Many test environment variables detected: {len(test_env_vars)}"
            )
    
    def _get_test_environment_variables(self, env_type: EnvironmentType, 
                                      test_id: str, temp_base: Path) -> Dict[str, str]:
        """Get environment variables for test environment."""
        base_vars = {
            "TESTING": "true",
            "PYTEST_RUNNING": "true",
            "WAN22_TEST_MODE": "true",
            "TEST_ID": test_id,
            "TEST_TYPE": env_type.value,
            "TEST_TEMP_DIR": str(temp_base),
            "LOG_LEVEL": "debug",
            "DISABLE_BROWSER_OPEN": "true",
            "NO_COLOR": "1"  # Disable colored output in tests
        }
        
        # Environment-specific variables
        if env_type == EnvironmentType.UNIT:
            base_vars.update({
                "UNIT_TEST": "true",
                "SKIP_INTEGRATION": "true"
            })
        
        elif env_type == EnvironmentType.INTEGRATION:
            base_vars.update({
                "INTEGRATION_TEST": "true",
                "TEST_DATABASE_URL": f"sqlite:///{temp_base}/test.db",
                "TEST_BACKEND_PORT": "8000",
                "TEST_FRONTEND_PORT": "3000"
            })
        
        elif env_type == EnvironmentType.E2E:
            base_vars.update({
                "E2E_TEST": "true",
                "HEADLESS_BROWSER": "true",
                "TEST_TIMEOUT": "60"
            })
        
        elif env_type == EnvironmentType.PERFORMANCE:
            base_vars.update({
                "PERFORMANCE_TEST": "true",
                "PROFILE_MEMORY": "true",
                "PROFILE_CPU": "true"
            })
        
        return base_vars
    
    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(('localhost', port))
                return True
        except OSError:
            return False
    
    def _find_available_port(self, start_port: int = 8000, max_attempts: int = 100) -> int:
        """Find an available port."""
        for i in range(max_attempts):
            port = start_port + i
            if self._is_port_available(port):
                return port
        
        raise RuntimeError(f"Could not find available port starting from {start_port}")


# Global validator instance
_environment_validator = EnvironmentValidator()


# Pytest fixtures for environment validation
@pytest.fixture(scope="session")
def environment_validator():
    """Provide environment validator."""
    return _environment_validator


@pytest.fixture
def validate_unit_environment():
    """Validate unit test environment."""
    result = _environment_validator.validate_environment(EnvironmentType.UNIT)
    if not result.valid:
        pytest.skip(f"Unit test environment validation failed: {'; '.join(result.errors)}")
    return result


@pytest.fixture
def validate_integration_environment():
    """Validate integration test environment."""
    result = _environment_validator.validate_environment(EnvironmentType.INTEGRATION)
    if not result.valid:
        pytest.skip(f"Integration test environment validation failed: {'; '.join(result.errors)}")
    return result


@pytest.fixture
def validate_e2e_environment():
    """Validate E2E test environment."""
    result = _environment_validator.validate_environment(EnvironmentType.E2E)
    if not result.valid:
        pytest.skip(f"E2E test environment validation failed: {'; '.join(result.errors)}")
    return result


@pytest.fixture
def validate_performance_environment():
    """Validate performance test environment."""
    result = _environment_validator.validate_environment(EnvironmentType.PERFORMANCE)
    if not result.valid:
        pytest.skip(f"Performance test environment validation failed: {'; '.join(result.errors)}")
    return result


@pytest.fixture
def setup_unit_environment(request):
    """Set up unit test environment."""
    test_id = f"{request.module.__name__}::{request.function.__name__}"
    setup_info = _environment_validator.setup_environment(EnvironmentType.UNIT, test_id)
    
    yield setup_info
    
    _environment_validator.cleanup_environment(setup_info)


@pytest.fixture
def setup_integration_environment(request):
    """Set up integration test environment."""
    test_id = f"{request.module.__name__}::{request.function.__name__}"
    setup_info = _environment_validator.setup_environment(EnvironmentType.INTEGRATION, test_id)
    
    yield setup_info
    
    _environment_validator.cleanup_environment(setup_info)


# Export main classes and functions
__all__ = [
    'EnvironmentValidator', 'EnvironmentType', 'ValidationLevel', 
    'EnvironmentRequirements', 'ValidationResult', 'environment_validator',
    'validate_unit_environment', 'validate_integration_environment',
    'validate_e2e_environment', 'validate_performance_environment',
    'setup_unit_environment', 'setup_integration_environment'
]
