"""
Pre-installation validation system for WAN2.2 local installation.
Provides comprehensive validation before installation begins to prevent predictable failures.
"""

import os
import sys
import json
import time
import shutil
import psutil
import socket
import subprocess
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from urllib.request import urlopen
from urllib.error import URLError

from interfaces import ValidationResult, HardwareProfile, InstallationError, ErrorCategory
from base_classes import BaseInstallationComponent


@dataclass
class SystemRequirement:
    """System requirement specification."""
    name: str
    minimum_value: Any
    current_value: Any
    unit: str
    met: bool
    details: Optional[str] = None


@dataclass
class NetworkTest:
    """Network connectivity test result."""
    test_name: str
    target: str
    success: bool
    latency_ms: Optional[float] = None
    bandwidth_mbps: Optional[float] = None
    error: Optional[str] = None


@dataclass
class PermissionTest:
    """Permission validation test result."""
    path: str
    readable: bool
    writable: bool
    executable: bool
    error: Optional[str] = None


@dataclass
class ConflictDetection:
    """Existing installation conflict detection result."""
    path: str
    conflict_type: str
    severity: str  # "warning", "error"
    description: str
    resolution: Optional[str] = None


@dataclass
class PreValidationReport:
    """Complete pre-installation validation report."""
    timestamp: str
    system_requirements: List[SystemRequirement]
    network_tests: List[NetworkTest]
    permission_tests: List[PermissionTest]
    conflicts: List[ConflictDetection]
    overall_success: bool
    errors: List[str]
    warnings: List[str]
    estimated_install_time_minutes: Optional[int] = None


class IPreInstallationValidator(ABC):
    """Interface for pre-installation validation."""
    
    @abstractmethod
    def validate_system_requirements(self) -> ValidationResult:
        """Validate system requirements (disk space, memory, permissions)."""
        pass
    
    @abstractmethod
    def validate_network_connectivity(self) -> ValidationResult:
        """Validate network connectivity and bandwidth."""
        pass
    
    @abstractmethod
    def validate_permissions(self) -> ValidationResult:
        """Validate file system permissions."""
        pass
    
    @abstractmethod
    def validate_existing_installation(self) -> ValidationResult:
        """Validate for existing installation conflicts."""
        pass
    
    @abstractmethod
    def generate_validation_report(self) -> PreValidationReport:
        """Generate comprehensive pre-validation report."""
        pass


class TimeoutManager:
    """Manages operation timeouts with automatic cleanup."""
    
    def __init__(self, timeout_seconds: int, cleanup_func: Optional[callable] = None):
        self.timeout_seconds = timeout_seconds
        self.cleanup_func = cleanup_func
        self.start_time = None
        self.timed_out = False
        self._timer = None
    
    def __enter__(self):
        self.start_time = time.time()
        self._timer = threading.Timer(self.timeout_seconds, self._timeout_handler)
        self._timer.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._timer:
            self._timer.cancel()
        
        if self.timed_out and self.cleanup_func:
            try:
                self.cleanup_func()
            except Exception as e:
                print(f"Cleanup failed: {e}")
    
    def _timeout_handler(self):
        self.timed_out = True
        if self.cleanup_func:
            try:
                self.cleanup_func()
            except Exception:
                pass
    
    def elapsed_time(self) -> float:
        if self.start_time:
            return time.time() - self.start_time
        return 0.0
    
    def is_timed_out(self) -> bool:
        return self.timed_out


class PreInstallationValidator(BaseInstallationComponent, IPreInstallationValidator):
    """Comprehensive pre-installation validator."""
    
    def __init__(self, installation_path: str, hardware_profile: Optional[HardwareProfile] = None):
        super().__init__(installation_path)
        self.hardware_profile = hardware_profile
        
        # System requirements
        self.min_requirements = {
            "disk_space_gb": 80,  # 80GB minimum for models and dependencies
            "memory_gb": 16,      # 16GB RAM minimum
            "python_version": "3.8.0",
            "free_memory_gb": 8   # 8GB free memory during installation
        }
        
        # Network test targets
        self.network_targets = {
            "huggingface_hub": "https://huggingface.co",
            "pypi": "https://pypi.org",
            "github": "https://github.com",
            "google_dns": "8.8.8.8"
        }
        
        # Timeout configurations
        self.timeouts = {
            "network_test": 30,
            "bandwidth_test": 60,
            "disk_test": 120,
            "permission_test": 30,
            "conflict_detection": 60
        }
        
        # Temporary files for cleanup
        self.temp_files = []
    
    def validate_system_requirements(self) -> ValidationResult:
        """Validate system requirements (disk space, memory, permissions)."""
        self.logger.info("Starting system requirements validation...")
        
        try:
            requirements = []
            errors = []
            warnings = []
            
            with TimeoutManager(self.timeouts["disk_test"], self._cleanup_temp_files):
                # Check disk space
                disk_req = self._check_disk_space()
                requirements.append(disk_req)
                if not disk_req.met:
                    errors.append(f"Insufficient disk space: {disk_req.current_value}{disk_req.unit} available, {disk_req.minimum_value}{disk_req.unit} required")
                
                # Check memory
                memory_req = self._check_memory()
                requirements.append(memory_req)
                if not memory_req.met:
                    errors.append(f"Insufficient memory: {memory_req.current_value}{memory_req.unit} available, {memory_req.minimum_value}{memory_req.unit} required")
                
                # Check free memory during installation
                free_memory_req = self._check_free_memory()
                requirements.append(free_memory_req)
                if not free_memory_req.met:
                    warnings.append(f"Low free memory: {free_memory_req.current_value}{free_memory_req.unit} available, {free_memory_req.minimum_value}{free_memory_req.unit} recommended")
                
                # Check Python version
                python_req = self._check_python_version()
                requirements.append(python_req)
                if not python_req.met:
                    errors.append(f"Python version too old: {python_req.current_value} found, {python_req.minimum_value}+ required")
                
                # Check CPU capabilities
                cpu_req = self._check_cpu_capabilities()
                requirements.append(cpu_req)
                if not cpu_req.met:
                    warnings.append(f"CPU may be insufficient: {cpu_req.details}")
                
                # Check GPU if available
                if self.hardware_profile and self.hardware_profile.gpu:
                    gpu_req = self._check_gpu_capabilities()
                    requirements.append(gpu_req)
                    if not gpu_req.met:
                        warnings.append(f"GPU issues detected: {gpu_req.details}")
            
            success = len(errors) == 0
            message = "System requirements validated successfully" if success else f"Found {len(errors)} requirement issues"
            
            return ValidationResult(
                success=success,
                message=message,
                details={
                    "requirements": [asdict(req) for req in requirements],
                    "installation_path": str(self.installation_path)
                },
                warnings=warnings
            )
            
        except Exception as e:
            self.logger.error(f"System requirements validation failed: {e}")
            return ValidationResult(
                success=False,
                message=f"System requirements validation failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def validate_network_connectivity(self) -> ValidationResult:
        """Validate network connectivity and bandwidth."""
        self.logger.info("Starting network connectivity validation...")
        
        try:
            network_tests = []
            errors = []
            warnings = []
            
            with TimeoutManager(self.timeouts["network_test"]):
                # Test basic connectivity to each target
                for name, target in self.network_targets.items():
                    test_result = self._test_network_connectivity(name, target)
                    network_tests.append(test_result)
                    
                    if not test_result.success:
                        if name in ["huggingface_hub", "pypi"]:
                            errors.append(f"Cannot reach {name}: {test_result.error}")
                        else:
                            warnings.append(f"Cannot reach {name}: {test_result.error}")
                
                # Test bandwidth if basic connectivity works
                if any(test.success for test in network_tests):
                    bandwidth_test = self._test_bandwidth()
                    network_tests.append(bandwidth_test)
                    
                    if bandwidth_test.success and bandwidth_test.bandwidth_mbps:
                        if bandwidth_test.bandwidth_mbps < 10:  # Less than 10 Mbps
                            warnings.append(f"Slow network connection: {bandwidth_test.bandwidth_mbps:.1f} Mbps")
                    elif not bandwidth_test.success:
                        warnings.append(f"Bandwidth test failed: {bandwidth_test.error}")
            
            success = len(errors) == 0
            message = "Network connectivity validated successfully" if success else f"Found {len(errors)} network issues"
            
            return ValidationResult(
                success=success,
                message=message,
                details={
                    "network_tests": [asdict(test) for test in network_tests]
                },
                warnings=warnings
            )
            
        except Exception as e:
            self.logger.error(f"Network connectivity validation failed: {e}")
            return ValidationResult(
                success=False,
                message=f"Network connectivity validation failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def validate_permissions(self) -> ValidationResult:
        """Validate file system permissions."""
        self.logger.info("Starting permissions validation...")
        
        try:
            permission_tests = []
            errors = []
            warnings = []
            
            with TimeoutManager(self.timeouts["permission_test"], self._cleanup_temp_files):
                # Test installation directory permissions
                install_perm = self._test_directory_permissions(self.installation_path, "Installation directory")
                permission_tests.append(install_perm)
                
                if not install_perm.writable:
                    errors.append(f"Cannot write to installation directory: {self.installation_path}")
                
                # Test system directories that might be needed
                system_dirs = [
                    (Path.home(), "User home directory"),
                    (Path.cwd(), "Current working directory"),
                ]
                
                if os.name == 'nt':  # Windows
                    system_dirs.extend([
                        (Path(os.environ.get('TEMP', 'C:\\temp')), "Temporary directory"),
                        (Path(os.environ.get('APPDATA', 'C:\\Users\\Default\\AppData\\Roaming')), "AppData directory")
                    ])
                else:  # Unix-like
                    system_dirs.extend([
                        (Path('/tmp'), "Temporary directory"),
                        (Path('/usr/local'), "Local installation directory")
                    ])
                
                for dir_path, description in system_dirs:
                    if dir_path.exists():
                        perm_test = self._test_directory_permissions(dir_path, description)
                        permission_tests.append(perm_test)
                        
                        if not perm_test.readable:
                            warnings.append(f"Cannot read {description}: {dir_path}")
                
                # Test Python executable permissions
                python_perm = self._test_python_permissions()
                if python_perm:
                    permission_tests.append(python_perm)
                    if not python_perm.executable:
                        errors.append("Python executable permissions issue")
            
            success = len(errors) == 0
            message = "Permissions validated successfully" if success else f"Found {len(errors)} permission issues"
            
            return ValidationResult(
                success=success,
                message=message,
                details={
                    "permission_tests": [asdict(test) for test in permission_tests]
                },
                warnings=warnings
            )
            
        except Exception as e:
            self.logger.error(f"Permissions validation failed: {e}")
            return ValidationResult(
                success=False,
                message=f"Permissions validation failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def validate_existing_installation(self) -> ValidationResult:
        """Validate for existing installation conflicts."""
        self.logger.info("Starting existing installation conflict detection...")
        
        try:
            conflicts = []
            errors = []
            warnings = []
            
            with TimeoutManager(self.timeouts["conflict_detection"]):
                # Check for existing WAN2.2 installations
                wan22_conflicts = self._detect_wan22_conflicts()
                conflicts.extend(wan22_conflicts)
                
                # Check for Python environment conflicts
                python_conflicts = self._detect_python_conflicts()
                conflicts.extend(python_conflicts)
                
                # Check for model directory conflicts
                model_conflicts = self._detect_model_conflicts()
                conflicts.extend(model_conflicts)
                
                # Check for port conflicts (if web UI is planned)
                port_conflicts = self._detect_port_conflicts()
                conflicts.extend(port_conflicts)
                
                # Categorize conflicts
                for conflict in conflicts:
                    if conflict.severity == "error":
                        errors.append(f"Installation conflict: {conflict.description}")
                    else:
                        warnings.append(f"Potential conflict: {conflict.description}")
            
            success = len(errors) == 0
            message = "No installation conflicts detected" if success else f"Found {len(errors)} installation conflicts"
            
            return ValidationResult(
                success=success,
                message=message,
                details={
                    "conflicts": [asdict(conflict) for conflict in conflicts]
                },
                warnings=warnings
            )
            
        except Exception as e:
            self.logger.error(f"Existing installation validation failed: {e}")
            return ValidationResult(
                success=False,
                message=f"Existing installation validation failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def generate_validation_report(self) -> PreValidationReport:
        """Generate comprehensive pre-validation report."""
        self.logger.info("Generating pre-installation validation report...")
        
        # Run all validation tests
        system_result = self.validate_system_requirements()
        network_result = self.validate_network_connectivity()
        permission_result = self.validate_permissions()
        conflict_result = self.validate_existing_installation()
        
        # Collect all errors and warnings
        all_errors = []
        all_warnings = []
        
        for result in [system_result, network_result, permission_result, conflict_result]:
            if not result.success:
                all_errors.append(result.message)
            if result.warnings:
                all_warnings.extend(result.warnings)
        
        # Estimate installation time based on system capabilities and network speed
        estimated_time = self._estimate_installation_time(
            system_result.details or {},
            network_result.details or {}
        )
        
        # Create report
        report = PreValidationReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            system_requirements=system_result.details.get("requirements", []) if system_result.details else [],
            network_tests=network_result.details.get("network_tests", []) if network_result.details else [],
            permission_tests=permission_result.details.get("permission_tests", []) if permission_result.details else [],
            conflicts=conflict_result.details.get("conflicts", []) if conflict_result.details else [],
            overall_success=all([system_result.success, network_result.success, 
                               permission_result.success, conflict_result.success]),
            errors=all_errors,
            warnings=all_warnings,
            estimated_install_time_minutes=estimated_time
        )
        
        # Save report to file
        report_path = self.installation_path / "logs" / "pre_validation_report.json"
        self.ensure_directory(report_path.parent)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(report), f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Pre-validation report saved to: {report_path}")
        return report
    
    # Helper methods for system requirements
    
    def _check_disk_space(self) -> SystemRequirement:
        """Check available disk space."""
        try:
            disk_usage = shutil.disk_usage(self.installation_path.parent)
            available_gb = disk_usage.free / (1024**3)
            required_gb = self.min_requirements["disk_space_gb"]
            
            return SystemRequirement(
                name="Disk Space",
                minimum_value=required_gb,
                current_value=round(available_gb, 1),
                unit="GB",
                met=available_gb >= required_gb,
                details=f"Free space on {self.installation_path.parent}"
            )
        except Exception as e:
            return SystemRequirement(
                name="Disk Space",
                minimum_value=self.min_requirements["disk_space_gb"],
                current_value=0,
                unit="GB",
                met=False,
                details=f"Error checking disk space: {e}"
            )
    
    def _check_memory(self) -> SystemRequirement:
        """Check total system memory."""
        try:
            memory = psutil.virtual_memory()
            total_gb = memory.total / (1024**3)
            required_gb = self.min_requirements["memory_gb"]
            
            return SystemRequirement(
                name="Total Memory",
                minimum_value=required_gb,
                current_value=round(total_gb, 1),
                unit="GB",
                met=total_gb >= required_gb,
                details=f"Total system RAM"
            )
        except Exception as e:
            return SystemRequirement(
                name="Total Memory",
                minimum_value=self.min_requirements["memory_gb"],
                current_value=0,
                unit="GB",
                met=False,
                details=f"Error checking memory: {e}"
            )
    
    def _check_free_memory(self) -> SystemRequirement:
        """Check available free memory."""
        try:
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            required_gb = self.min_requirements["free_memory_gb"]
            
            return SystemRequirement(
                name="Available Memory",
                minimum_value=required_gb,
                current_value=round(available_gb, 1),
                unit="GB",
                met=available_gb >= required_gb,
                details=f"Currently available RAM"
            )
        except Exception as e:
            return SystemRequirement(
                name="Available Memory",
                minimum_value=self.min_requirements["free_memory_gb"],
                current_value=0,
                unit="GB",
                met=False,
                details=f"Error checking available memory: {e}"
            )
    
    def _check_python_version(self) -> SystemRequirement:
        """Check Python version."""
        try:
            current_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            required_version = self.min_requirements["python_version"]
            
            # Simple version comparison
            current_parts = [int(x) for x in current_version.split('.')]
            required_parts = [int(x) for x in required_version.split('.')]
            
            met = current_parts >= required_parts
            
            return SystemRequirement(
                name="Python Version",
                minimum_value=required_version,
                current_value=current_version,
                unit="",
                met=met,
                details=f"Python {current_version} at {sys.executable}"
            )
        except Exception as e:
            return SystemRequirement(
                name="Python Version",
                minimum_value=self.min_requirements["python_version"],
                current_value="unknown",
                unit="",
                met=False,
                details=f"Error checking Python version: {e}"
            )
    
    def _check_cpu_capabilities(self) -> SystemRequirement:
        """Check CPU capabilities."""
        try:
            cpu_count = psutil.cpu_count(logical=True)
            cpu_freq = psutil.cpu_freq()
            
            # Basic CPU adequacy check
            adequate = cpu_count >= 4  # At least 4 cores recommended
            
            details = f"{cpu_count} cores"
            if cpu_freq:
                details += f", {cpu_freq.current:.0f}MHz"
            
            return SystemRequirement(
                name="CPU Capabilities",
                minimum_value="4 cores",
                current_value=f"{cpu_count} cores",
                unit="",
                met=adequate,
                details=details
            )
        except Exception as e:
            return SystemRequirement(
                name="CPU Capabilities",
                minimum_value="4 cores",
                current_value="unknown",
                unit="",
                met=False,
                details=f"Error checking CPU: {e}"
            )
    
    def _check_gpu_capabilities(self) -> SystemRequirement:
        """Check GPU capabilities if available."""
        try:
            if not self.hardware_profile or not self.hardware_profile.gpu:
                return SystemRequirement(
                    name="GPU Capabilities",
                    minimum_value="Optional",
                    current_value="Not detected",
                    unit="",
                    met=True,
                    details="GPU acceleration not required but recommended"
                )
            
            gpu = self.hardware_profile.gpu
            adequate = gpu.vram_gb >= 8  # At least 8GB VRAM recommended
            
            return SystemRequirement(
                name="GPU Capabilities",
                minimum_value="8GB VRAM",
                current_value=f"{gpu.vram_gb}GB VRAM",
                unit="",
                met=adequate,
                details=f"{gpu.model} with {gpu.vram_gb}GB VRAM"
            )
        except Exception as e:
            return SystemRequirement(
                name="GPU Capabilities",
                minimum_value="8GB VRAM",
                current_value="unknown",
                unit="",
                met=False,
                details=f"Error checking GPU: {e}"
            )
    
    # Helper methods for network testing
    
    def _test_network_connectivity(self, name: str, target: str) -> NetworkTest:
        """Test connectivity to a specific target."""
        try:
            start_time = time.time()
            
            if target.startswith('http'):
                # HTTP connectivity test
                with urlopen(target, timeout=10) as response:
                    latency = (time.time() - start_time) * 1000
                    return NetworkTest(
                        test_name=name,
                        target=target,
                        success=response.status == 200,
                        latency_ms=round(latency, 2)
                    )
            else:
                # IP connectivity test (ping-like)
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10)
                result = sock.connect_ex((target, 53))  # DNS port
                sock.close()
                
                latency = (time.time() - start_time) * 1000
                return NetworkTest(
                    test_name=name,
                    target=target,
                    success=result == 0,
                    latency_ms=round(latency, 2)
                )
                
        except Exception as e:
            return NetworkTest(
                test_name=name,
                target=target,
                success=False,
                error=str(e)
            )
    
    def _test_bandwidth(self) -> NetworkTest:
        """Test network bandwidth with a small download."""
        try:
            # Use a small file from a reliable source for bandwidth testing
            test_url = "https://httpbin.org/bytes/1048576"  # 1MB test file
            
            start_time = time.time()
            with urlopen(test_url, timeout=30) as response:
                data = response.read()
                download_time = time.time() - start_time
            
            if download_time > 0:
                bandwidth_mbps = (len(data) * 8) / (download_time * 1000000)  # Convert to Mbps
                return NetworkTest(
                    test_name="Bandwidth Test",
                    target=test_url,
                    success=True,
                    bandwidth_mbps=round(bandwidth_mbps, 2)
                )
            else:
                return NetworkTest(
                    test_name="Bandwidth Test",
                    target=test_url,
                    success=False,
                    error="Download completed too quickly to measure"
                )
                
        except Exception as e:
            return NetworkTest(
                test_name="Bandwidth Test",
                target="bandwidth test",
                success=False,
                error=str(e)
            )
    
    # Helper methods for permission testing
    
    def _test_directory_permissions(self, path: Path, description: str) -> PermissionTest:
        """Test read/write/execute permissions for a directory."""
        try:
            readable = os.access(path, os.R_OK) if path.exists() else False
            writable = os.access(path, os.W_OK) if path.exists() else False
            executable = os.access(path, os.X_OK) if path.exists() else False
            
            # Test actual write capability
            if writable and path.exists():
                test_file = path / f".wan22_permission_test_{int(time.time())}"
                try:
                    test_file.write_text("test")
                    test_file.unlink()
                    self.temp_files.append(test_file)
                except Exception:
                    writable = False
            
            return PermissionTest(
                path=str(path),
                readable=readable,
                writable=writable,
                executable=executable
            )
            
        except Exception as e:
            return PermissionTest(
                path=str(path),
                readable=False,
                writable=False,
                executable=False,
                error=str(e)
            )
    
    def _test_python_permissions(self) -> Optional[PermissionTest]:
        """Test Python executable permissions."""
        try:
            python_path = Path(sys.executable)
            
            return PermissionTest(
                path=str(python_path),
                readable=os.access(python_path, os.R_OK),
                writable=False,  # Don't need to write to Python executable
                executable=os.access(python_path, os.X_OK)
            )
            
        except Exception as e:
            return PermissionTest(
                path=sys.executable,
                readable=False,
                writable=False,
                executable=False,
                error=str(e)
            )
    
    # Helper methods for conflict detection
    
    def _detect_wan22_conflicts(self) -> List[ConflictDetection]:
        """Detect existing WAN2.2 installations."""
        conflicts = []
        
        try:
            # Check if installation directory already exists
            if self.installation_path.exists():
                contents = list(self.installation_path.iterdir())
                if contents:
                    conflicts.append(ConflictDetection(
                        path=str(self.installation_path),
                        conflict_type="existing_installation",
                        severity="warning",
                        description=f"Installation directory already exists with {len(contents)} items",
                        resolution="Contents will be backed up before installation"
                    ))
            
            # Check for WAN2.2 in common locations
            common_locations = [
                Path.home() / "WAN22",
                Path.home() / "wan22",
                Path("C:/WAN22") if os.name == 'nt' else Path("/opt/wan22"),
                Path.cwd() / "WAN22"
            ]
            
            for location in common_locations:
                if location.exists() and location != self.installation_path:
                    conflicts.append(ConflictDetection(
                        path=str(location),
                        conflict_type="existing_installation",
                        severity="warning",
                        description=f"Existing WAN2.2 installation found at {location}",
                        resolution="Multiple installations are supported"
                    ))
                    
        except Exception as e:
            conflicts.append(ConflictDetection(
                path="unknown",
                conflict_type="detection_error",
                severity="warning",
                description=f"Error detecting WAN2.2 conflicts: {e}",
                resolution="Manual verification recommended"
            ))
        
        return conflicts
    
    def _detect_python_conflicts(self) -> List[ConflictDetection]:
        """Detect Python environment conflicts."""
        conflicts = []
        
        try:
            # Check for existing virtual environments
            venv_path = self.installation_path / "venv"
            if venv_path.exists():
                conflicts.append(ConflictDetection(
                    path=str(venv_path),
                    conflict_type="existing_venv",
                    severity="warning",
                    description="Virtual environment already exists",
                    resolution="Will be recreated if needed"
                ))
            
            # Check for conflicting Python packages in system Python
            try:
                import torch
                conflicts.append(ConflictDetection(
                    path=sys.executable,
                    conflict_type="system_packages",
                    severity="warning",
                    description="PyTorch already installed in system Python",
                    resolution="Virtual environment will isolate dependencies"
                ))
            except ImportError:
                pass
                
        except Exception as e:
            conflicts.append(ConflictDetection(
                path="unknown",
                conflict_type="detection_error",
                severity="warning",
                description=f"Error detecting Python conflicts: {e}",
                resolution="Manual verification recommended"
            ))
        
        return conflicts
    
    def _detect_model_conflicts(self) -> List[ConflictDetection]:
        """Detect model directory conflicts."""
        conflicts = []
        
        try:
            models_path = self.installation_path / "models"
            if models_path.exists():
                model_dirs = [d for d in models_path.iterdir() if d.is_dir()]
                if model_dirs:
                    conflicts.append(ConflictDetection(
                        path=str(models_path),
                        conflict_type="existing_models",
                        severity="warning",
                        description=f"Found {len(model_dirs)} existing model directories",
                        resolution="Existing models will be validated and reused if compatible"
                    ))
                    
        except Exception as e:
            conflicts.append(ConflictDetection(
                path="unknown",
                conflict_type="detection_error",
                severity="warning",
                description=f"Error detecting model conflicts: {e}",
                resolution="Manual verification recommended"
            ))
        
        return conflicts
    
    def _detect_port_conflicts(self) -> List[ConflictDetection]:
        """Detect port conflicts for web UI."""
        conflicts = []
        
        try:
            # Check common ports that WAN2.2 might use
            ports_to_check = [7860, 8080, 8000, 5000]  # Common Gradio/web UI ports
            
            for port in ports_to_check:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    result = sock.bind(('localhost', port))
                    sock.close()
                except OSError:
                    conflicts.append(ConflictDetection(
                        path=f"localhost:{port}",
                        conflict_type="port_conflict",
                        severity="warning",
                        description=f"Port {port} is already in use",
                        resolution="Alternative port will be selected automatically"
                    ))
                    
        except Exception as e:
            conflicts.append(ConflictDetection(
                path="unknown",
                conflict_type="detection_error",
                severity="warning",
                description=f"Error detecting port conflicts: {e}",
                resolution="Manual verification recommended"
            ))
        
        return conflicts
    
    def _estimate_installation_time(self, system_details: Dict, network_details: Dict) -> Optional[int]:
        """Estimate installation time in minutes based on system capabilities."""
        try:
            base_time = 45  # Base installation time in minutes
            
            # Adjust for system performance
            requirements = system_details.get("requirements", [])
            for req in requirements:
                if req["name"] == "Available Memory" and req["current_value"] < 12:
                    base_time += 15  # Slower with less memory
                elif req["name"] == "CPU Capabilities" and "cores" in str(req["current_value"]):
                    cores = int(req["current_value"].split()[0])
                    if cores < 6:
                        base_time += 10  # Slower with fewer cores
            
            # Adjust for network speed
            network_tests = network_details.get("network_tests", [])
            for test in network_tests:
                if test["test_name"] == "Bandwidth Test" and test.get("bandwidth_mbps"):
                    bandwidth = test["bandwidth_mbps"]
                    if bandwidth < 10:
                        base_time += 30  # Much slower with slow network
                    elif bandwidth < 50:
                        base_time += 15  # Somewhat slower
            
            return min(base_time, 120)  # Cap at 2 hours
            
        except Exception:
            return None
    
    def _cleanup_temp_files(self):
        """Clean up temporary files created during validation."""
        for temp_file in self.temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception:
                pass
        self.temp_files.clear()


def main():
    """Main function for standalone testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="WAN2.2 Pre-Installation Validator")
    parser.add_argument("--installation-path", default="./wan22_installation", 
                       help="Path for WAN2.2 installation")
    parser.add_argument("--report-only", action="store_true",
                       help="Generate report only, don't validate individual components")
    
    args = parser.parse_args()
    
    validator = PreInstallationValidator(args.installation_path)
    
    if args.report_only:
        report = validator.generate_validation_report()
        print(f"Pre-validation report generated: {report.overall_success}")
        if not report.overall_success:
            print(f"Errors: {len(report.errors)}")
            print(f"Warnings: {len(report.warnings)}")
    else:
        print("Running individual validation tests...")
        
        # Run each validation test
        tests = [
            ("System Requirements", validator.validate_system_requirements),
            ("Network Connectivity", validator.validate_network_connectivity),
            ("Permissions", validator.validate_permissions),
            ("Existing Installation", validator.validate_existing_installation)
        ]
        
        for test_name, test_func in tests:
            print(f"\n{test_name}:")
            result = test_func()
            print(f"  Success: {result.success}")
            print(f"  Message: {result.message}")
            if result.warnings:
                print(f"  Warnings: {len(result.warnings)}")


if __name__ == "__main__":
    main()