"""
Test Environment Validator

This module provides comprehensive validation of test environment dependencies,
service availability, and setup requirements with detailed error reporting
and guidance.
"""

import os
import sys
import subprocess
import socket
import importlib
import platform
import psutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import requests
import time


class ValidationLevel(Enum):
    """Validation level enumeration"""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class ValidationStatus(Enum):
    """Validation status enumeration"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ValidationResult:
    """Result of a validation check"""
    name: str
    status: ValidationStatus
    level: ValidationLevel
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class EnvironmentRequirement:
    """Definition of an environment requirement"""
    name: str
    type: str  # 'python_package', 'system_command', 'service', 'file', 'directory', 'env_var'
    required: bool = True
    version_check: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)


class EnvironmentValidator:
    """
    Comprehensive test environment validator for checking dependencies,
    service availability, and environment setup requirements.
    """
    
    def __init__(self, requirements_file: Optional[Union[str, Path]] = None):
        """
        Initialize environment validator
        
        Args:
            requirements_file: Path to requirements configuration file
        """
        self.requirements_file = Path(requirements_file) if requirements_file else None
        self.requirements: List[EnvironmentRequirement] = []
        self.validation_results: List[ValidationResult] = []
        
        # Load requirements
        self._load_requirements()
        
        # Add built-in requirements
        self._add_builtin_requirements()
    
    def _load_requirements(self) -> None:
        """Load requirements from configuration file"""
        if self.requirements_file and self.requirements_file.exists():
            try:
                import yaml
                with open(self.requirements_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                requirements_data = config.get('requirements', [])
                for req_data in requirements_data:
                    requirement = EnvironmentRequirement(**req_data)
                    self.requirements.append(requirement)
                    
            except Exception as e:
                print(f"Warning: Failed to load requirements file: {e}")
    
    def _add_builtin_requirements(self) -> None:
        """Add built-in environment requirements"""
        builtin_requirements = [
            # Python version
            EnvironmentRequirement(
                name="python_version",
                type="python_version",
                required=True,
                version_check=">=3.8",
                config={"min_version": "3.8"}
            ),
            
            # Essential Python packages
            EnvironmentRequirement(
                name="pytest",
                type="python_package",
                required=True,
                version_check=">=6.0.0"
            ),
            
            EnvironmentRequirement(
                name="pytest-asyncio",
                type="python_package",
                required=True
            ),
            
            EnvironmentRequirement(
                name="pytest-cov",
                type="python_package",
                required=False
            ),
            
            # System resources
            EnvironmentRequirement(
                name="memory_check",
                type="system_resource",
                required=True,
                config={"min_memory_gb": 2}
            ),
            
            EnvironmentRequirement(
                name="disk_space_check",
                type="system_resource",
                required=True,
                config={"min_disk_gb": 1}
            ),
            
            # Test directories
            EnvironmentRequirement(
                name="tests_directory",
                type="directory",
                required=True,
                config={"path": "tests"}
            ),
            
            EnvironmentRequirement(
                name="fixtures_directory",
                type="directory",
                required=False,
                config={"path": "tests/fixtures"}
            ),
        ]
        
        self.requirements.extend(builtin_requirements)
    
    def validate_environment(self) -> List[ValidationResult]:
        """
        Validate complete test environment
        
        Returns:
            List of validation results
        """
        self.validation_results.clear()
        
        for requirement in self.requirements:
            try:
                result = self._validate_requirement(requirement)
                self.validation_results.append(result)
            except Exception as e:
                error_result = ValidationResult(
                    name=requirement.name,
                    status=ValidationStatus.FAILED,
                    level=ValidationLevel.CRITICAL if requirement.required else ValidationLevel.WARNING,
                    message=f"Validation error: {e}",
                    suggestions=["Check requirement configuration"]
                )
                self.validation_results.append(error_result)
        
        return self.validation_results
    
    def _validate_requirement(self, requirement: EnvironmentRequirement) -> ValidationResult:
        """Validate a specific requirement"""
        if requirement.type == "python_package":
            return self._validate_python_package(requirement)
        elif requirement.type == "python_version":
            return self._validate_python_version(requirement)
        elif requirement.type == "system_command":
            return self._validate_system_command(requirement)
        elif requirement.type == "service":
            return self._validate_service(requirement)
        elif requirement.type == "file":
            return self._validate_file(requirement)
        elif requirement.type == "directory":
            return self._validate_directory(requirement)
        elif requirement.type == "env_var":
            return self._validate_environment_variable(requirement)
        elif requirement.type == "system_resource":
            return self._validate_system_resource(requirement)
        else:
            return ValidationResult(
                name=requirement.name,
                status=ValidationStatus.SKIPPED,
                level=ValidationLevel.INFO,
                message=f"Unknown requirement type: {requirement.type}"
            )
    
    def _validate_python_package(self, requirement: EnvironmentRequirement) -> ValidationResult:
        """Validate Python package availability and version"""
        try:
            module = importlib.import_module(requirement.name)
            
            # Check version if specified
            if requirement.version_check:
                version = getattr(module, '__version__', None)
                if version:
                    if self._check_version(version, requirement.version_check):
                        return ValidationResult(
                            name=requirement.name,
                            status=ValidationStatus.PASSED,
                            level=ValidationLevel.INFO,
                            message=f"Package {requirement.name} version {version} is available",
                            details={"version": version}
                        )
                    else:
                        return ValidationResult(
                            name=requirement.name,
                            status=ValidationStatus.FAILED,
                            level=ValidationLevel.CRITICAL if requirement.required else ValidationLevel.WARNING,
                            message=f"Package {requirement.name} version {version} does not meet requirement {requirement.version_check}",
                            details={"current_version": version, "required_version": requirement.version_check},
                            suggestions=[f"pip install '{requirement.name}{requirement.version_check}'"]
                        )
                else:
                    return ValidationResult(
                        name=requirement.name,
                        status=ValidationStatus.PASSED,
                        level=ValidationLevel.WARNING,
                        message=f"Package {requirement.name} is available but version cannot be determined",
                        suggestions=["Consider upgrading to a version that supports __version__"]
                    )
            else:
                return ValidationResult(
                    name=requirement.name,
                    status=ValidationStatus.PASSED,
                    level=ValidationLevel.INFO,
                    message=f"Package {requirement.name} is available"
                )
                
        except ImportError:
            return ValidationResult(
                name=requirement.name,
                status=ValidationStatus.FAILED,
                level=ValidationLevel.CRITICAL if requirement.required else ValidationLevel.WARNING,
                message=f"Package {requirement.name} is not installed",
                suggestions=[
                    f"pip install {requirement.name}",
                    "Check if package name is correct",
                    "Ensure virtual environment is activated"
                ]
            )
    
    def _validate_python_version(self, requirement: EnvironmentRequirement) -> ValidationResult:
        """Validate Python version"""
        current_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        if requirement.version_check:
            if self._check_version(current_version, requirement.version_check):
                return ValidationResult(
                    name=requirement.name,
                    status=ValidationStatus.PASSED,
                    level=ValidationLevel.INFO,
                    message=f"Python version {current_version} meets requirement {requirement.version_check}",
                    details={"version": current_version}
                )
            else:
                return ValidationResult(
                    name=requirement.name,
                    status=ValidationStatus.FAILED,
                    level=ValidationLevel.CRITICAL,
                    message=f"Python version {current_version} does not meet requirement {requirement.version_check}",
                    details={"current_version": current_version, "required_version": requirement.version_check},
                    suggestions=[
                        f"Upgrade Python to version {requirement.version_check}",
                        "Use pyenv or conda to manage Python versions"
                    ]
                )
        else:
            return ValidationResult(
                name=requirement.name,
                status=ValidationStatus.PASSED,
                level=ValidationLevel.INFO,
                message=f"Python version {current_version} is available",
                details={"version": current_version}
            )
    
    def _validate_system_command(self, requirement: EnvironmentRequirement) -> ValidationResult:
        """Validate system command availability"""
        try:
            result = subprocess.run(
                [requirement.name, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                version_output = result.stdout.strip() or result.stderr.strip()
                return ValidationResult(
                    name=requirement.name,
                    status=ValidationStatus.PASSED,
                    level=ValidationLevel.INFO,
                    message=f"Command {requirement.name} is available",
                    details={"version_output": version_output}
                )
            else:
                return ValidationResult(
                    name=requirement.name,
                    status=ValidationStatus.FAILED,
                    level=ValidationLevel.CRITICAL if requirement.required else ValidationLevel.WARNING,
                    message=f"Command {requirement.name} returned error code {result.returncode}",
                    suggestions=[f"Install {requirement.name}", "Check PATH environment variable"]
                )
                
        except FileNotFoundError:
            return ValidationResult(
                name=requirement.name,
                status=ValidationStatus.FAILED,
                level=ValidationLevel.CRITICAL if requirement.required else ValidationLevel.WARNING,
                message=f"Command {requirement.name} not found",
                suggestions=[
                    f"Install {requirement.name}",
                    "Add to PATH environment variable",
                    "Check spelling of command name"
                ]
            )
        except subprocess.TimeoutExpired:
            return ValidationResult(
                name=requirement.name,
                status=ValidationStatus.FAILED,
                level=ValidationLevel.WARNING,
                message=f"Command {requirement.name} timed out",
                suggestions=["Check if command is responsive", "Try manual execution"]
            )
    
    def _validate_service(self, requirement: EnvironmentRequirement) -> ValidationResult:
        """Validate service availability"""
        host = requirement.config.get("host", "localhost")
        port = requirement.config.get("port")
        url = requirement.config.get("url")
        
        if url:
            return self._validate_http_service(requirement, url)
        elif port:
            return self._validate_tcp_service(requirement, host, port)
        else:
            return ValidationResult(
                name=requirement.name,
                status=ValidationStatus.FAILED,
                level=ValidationLevel.WARNING,
                message="Service validation requires either 'url' or 'port' in config"
            )
    
    def _validate_http_service(self, requirement: EnvironmentRequirement, url: str) -> ValidationResult:
        """Validate HTTP service availability"""
        try:
            timeout = requirement.config.get("timeout", 5)
            response = requests.get(url, timeout=timeout)
            
            if response.status_code < 400:
                return ValidationResult(
                    name=requirement.name,
                    status=ValidationStatus.PASSED,
                    level=ValidationLevel.INFO,
                    message=f"HTTP service at {url} is available",
                    details={"status_code": response.status_code, "url": url}
                )
            else:
                return ValidationResult(
                    name=requirement.name,
                    status=ValidationStatus.FAILED,
                    level=ValidationLevel.CRITICAL if requirement.required else ValidationLevel.WARNING,
                    message=f"HTTP service at {url} returned status {response.status_code}",
                    details={"status_code": response.status_code, "url": url},
                    suggestions=["Check service configuration", "Verify service is running"]
                )
                
        except requests.exceptions.ConnectionError:
            return ValidationResult(
                name=requirement.name,
                status=ValidationStatus.FAILED,
                level=ValidationLevel.CRITICAL if requirement.required else ValidationLevel.WARNING,
                message=f"Cannot connect to HTTP service at {url}",
                suggestions=[
                    "Start the service",
                    "Check URL is correct",
                    "Verify network connectivity"
                ]
            )
        except requests.exceptions.Timeout:
            return ValidationResult(
                name=requirement.name,
                status=ValidationStatus.FAILED,
                level=ValidationLevel.WARNING,
                message=f"HTTP service at {url} timed out",
                suggestions=["Check service responsiveness", "Increase timeout"]
            )
    
    def _validate_tcp_service(self, requirement: EnvironmentRequirement, host: str, port: int) -> ValidationResult:
        """Validate TCP service availability"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                return ValidationResult(
                    name=requirement.name,
                    status=ValidationStatus.PASSED,
                    level=ValidationLevel.INFO,
                    message=f"TCP service at {host}:{port} is available",
                    details={"host": host, "port": port}
                )
            else:
                return ValidationResult(
                    name=requirement.name,
                    status=ValidationStatus.FAILED,
                    level=ValidationLevel.CRITICAL if requirement.required else ValidationLevel.WARNING,
                    message=f"Cannot connect to TCP service at {host}:{port}",
                    suggestions=[
                        "Start the service",
                        "Check host and port are correct",
                        "Verify firewall settings"
                    ]
                )
                
        except Exception as e:
            return ValidationResult(
                name=requirement.name,
                status=ValidationStatus.FAILED,
                level=ValidationLevel.WARNING,
                message=f"Error checking TCP service: {e}",
                suggestions=["Check network configuration"]
            )
    
    def _validate_file(self, requirement: EnvironmentRequirement) -> ValidationResult:
        """Validate file existence"""
        file_path = Path(requirement.config.get("path", requirement.name))
        
        if file_path.exists():
            if file_path.is_file():
                return ValidationResult(
                    name=requirement.name,
                    status=ValidationStatus.PASSED,
                    level=ValidationLevel.INFO,
                    message=f"File {file_path} exists",
                    details={"path": str(file_path), "size": file_path.stat().st_size}
                )
            else:
                return ValidationResult(
                    name=requirement.name,
                    status=ValidationStatus.FAILED,
                    level=ValidationLevel.WARNING,
                    message=f"Path {file_path} exists but is not a file",
                    suggestions=["Check path specification"]
                )
        else:
            return ValidationResult(
                name=requirement.name,
                status=ValidationStatus.FAILED,
                level=ValidationLevel.CRITICAL if requirement.required else ValidationLevel.WARNING,
                message=f"File {file_path} does not exist",
                suggestions=[
                    f"Create file {file_path}",
                    "Check path is correct",
                    "Verify file permissions"
                ]
            )
    
    def _validate_directory(self, requirement: EnvironmentRequirement) -> ValidationResult:
        """Validate directory existence"""
        dir_path = Path(requirement.config.get("path", requirement.name))
        
        if dir_path.exists():
            if dir_path.is_dir():
                return ValidationResult(
                    name=requirement.name,
                    status=ValidationStatus.PASSED,
                    level=ValidationLevel.INFO,
                    message=f"Directory {dir_path} exists",
                    details={"path": str(dir_path)}
                )
            else:
                return ValidationResult(
                    name=requirement.name,
                    status=ValidationStatus.FAILED,
                    level=ValidationLevel.WARNING,
                    message=f"Path {dir_path} exists but is not a directory",
                    suggestions=["Check path specification"]
                )
        else:
            level = ValidationLevel.CRITICAL if requirement.required else ValidationLevel.WARNING
            suggestions = [f"Create directory: mkdir -p {dir_path}", "Check path is correct"]
            
            # Auto-create directory if not required
            if not requirement.required:
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    return ValidationResult(
                        name=requirement.name,
                        status=ValidationStatus.PASSED,
                        level=ValidationLevel.INFO,
                        message=f"Directory {dir_path} created automatically",
                        details={"path": str(dir_path)}
                    )
                except Exception as e:
                    suggestions.append(f"Auto-creation failed: {e}")
            
            return ValidationResult(
                name=requirement.name,
                status=ValidationStatus.FAILED,
                level=level,
                message=f"Directory {dir_path} does not exist",
                suggestions=suggestions
            )
    
    def _validate_environment_variable(self, requirement: EnvironmentRequirement) -> ValidationResult:
        """Validate environment variable"""
        var_name = requirement.config.get("name", requirement.name)
        value = os.getenv(var_name)
        
        if value is not None:
            expected_value = requirement.config.get("value")
            if expected_value and value != expected_value:
                return ValidationResult(
                    name=requirement.name,
                    status=ValidationStatus.FAILED,
                    level=ValidationLevel.WARNING,
                    message=f"Environment variable {var_name} has unexpected value",
                    details={"current_value": value, "expected_value": expected_value},
                    suggestions=[f"Set {var_name}={expected_value}"]
                )
            else:
                return ValidationResult(
                    name=requirement.name,
                    status=ValidationStatus.PASSED,
                    level=ValidationLevel.INFO,
                    message=f"Environment variable {var_name} is set",
                    details={"value": value}
                )
        else:
            return ValidationResult(
                name=requirement.name,
                status=ValidationStatus.FAILED,
                level=ValidationLevel.CRITICAL if requirement.required else ValidationLevel.WARNING,
                message=f"Environment variable {var_name} is not set",
                suggestions=[
                    f"Set environment variable: export {var_name}=<value>",
                    "Add to shell profile (.bashrc, .zshrc, etc.)"
                ]
            )
    
    def _validate_system_resource(self, requirement: EnvironmentRequirement) -> ValidationResult:
        """Validate system resources"""
        if requirement.name == "memory_check":
            return self._validate_memory(requirement)
        elif requirement.name == "disk_space_check":
            return self._validate_disk_space(requirement)
        else:
            return ValidationResult(
                name=requirement.name,
                status=ValidationStatus.SKIPPED,
                level=ValidationLevel.INFO,
                message=f"Unknown system resource check: {requirement.name}"
            )
    
    def _validate_memory(self, requirement: EnvironmentRequirement) -> ValidationResult:
        """Validate available memory"""
        min_memory_gb = requirement.config.get("min_memory_gb", 2)
        
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024 ** 3)
        
        if available_gb >= min_memory_gb:
            return ValidationResult(
                name=requirement.name,
                status=ValidationStatus.PASSED,
                level=ValidationLevel.INFO,
                message=f"Available memory: {available_gb:.1f}GB (required: {min_memory_gb}GB)",
                details={"available_gb": available_gb, "required_gb": min_memory_gb}
            )
        else:
            return ValidationResult(
                name=requirement.name,
                status=ValidationStatus.FAILED,
                level=ValidationLevel.WARNING,
                message=f"Insufficient memory: {available_gb:.1f}GB (required: {min_memory_gb}GB)",
                details={"available_gb": available_gb, "required_gb": min_memory_gb},
                suggestions=[
                    "Close unnecessary applications",
                    "Add more RAM",
                    "Reduce test parallelization"
                ]
            )
    
    def _validate_disk_space(self, requirement: EnvironmentRequirement) -> ValidationResult:
        """Validate available disk space"""
        min_disk_gb = requirement.config.get("min_disk_gb", 1)
        
        disk_usage = psutil.disk_usage('.')
        available_gb = disk_usage.free / (1024 ** 3)
        
        if available_gb >= min_disk_gb:
            return ValidationResult(
                name=requirement.name,
                status=ValidationStatus.PASSED,
                level=ValidationLevel.INFO,
                message=f"Available disk space: {available_gb:.1f}GB (required: {min_disk_gb}GB)",
                details={"available_gb": available_gb, "required_gb": min_disk_gb}
            )
        else:
            return ValidationResult(
                name=requirement.name,
                status=ValidationStatus.FAILED,
                level=ValidationLevel.WARNING,
                message=f"Insufficient disk space: {available_gb:.1f}GB (required: {min_disk_gb}GB)",
                details={"available_gb": available_gb, "required_gb": min_disk_gb},
                suggestions=[
                    "Clean up temporary files",
                    "Remove unused files",
                    "Move files to external storage"
                ]
            )
    
    def _check_version(self, current: str, requirement: str) -> bool:
        """Check if current version meets requirement"""
        try:
            from packaging import version
            
            if requirement.startswith(">="):
                return version.parse(current) >= version.parse(requirement[2:])
            elif requirement.startswith("<="):
                return version.parse(current) <= version.parse(requirement[2:])
            elif requirement.startswith(">"):
                return version.parse(current) > version.parse(requirement[1:])
            elif requirement.startswith("<"):
                return version.parse(current) < version.parse(requirement[1:])
            elif requirement.startswith("=="):
                return version.parse(current) == version.parse(requirement[2:])
            else:
                return version.parse(current) >= version.parse(requirement)
                
        except Exception:
            # Fallback to string comparison if packaging not available
            return current >= requirement.lstrip(">=<!")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary"""
        if not self.validation_results:
            return {"status": "not_run", "message": "Validation not run"}
        
        passed = sum(1 for r in self.validation_results if r.status == ValidationStatus.PASSED)
        failed = sum(1 for r in self.validation_results if r.status == ValidationStatus.FAILED)
        skipped = sum(1 for r in self.validation_results if r.status == ValidationStatus.SKIPPED)
        
        critical_failures = sum(1 for r in self.validation_results 
                              if r.status == ValidationStatus.FAILED and r.level == ValidationLevel.CRITICAL)
        
        overall_status = "passed" if critical_failures == 0 else "failed"
        
        return {
            "status": overall_status,
            "total_checks": len(self.validation_results),
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "critical_failures": critical_failures,
            "ready_for_testing": critical_failures == 0
        }
    
    def generate_report(self, format: str = "text") -> str:
        """Generate validation report"""
        if format == "text":
            return self._generate_text_report()
        elif format == "json":
            return self._generate_json_report()
        else:
            raise ValueError(f"Unsupported report format: {format}")
    
    def _generate_text_report(self) -> str:
        """Generate text validation report"""
        lines = []
        lines.append("Test Environment Validation Report")
        lines.append("=" * 40)
        lines.append("")
        
        summary = self.get_validation_summary()
        lines.append(f"Overall Status: {summary['status'].upper()}")
        lines.append(f"Total Checks: {summary['total_checks']}")
        lines.append(f"Passed: {summary['passed']}")
        lines.append(f"Failed: {summary['failed']}")
        lines.append(f"Skipped: {summary['skipped']}")
        lines.append(f"Critical Failures: {summary['critical_failures']}")
        lines.append(f"Ready for Testing: {'Yes' if summary['ready_for_testing'] else 'No'}")
        lines.append("")
        
        # Group results by status
        for status in [ValidationStatus.FAILED, ValidationStatus.PASSED, ValidationStatus.SKIPPED]:
            status_results = [r for r in self.validation_results if r.status == status]
            if status_results:
                lines.append(f"{status.value.upper()} ({len(status_results)}):")
                lines.append("-" * 20)
                
                for result in status_results:
                    lines.append(f"  {result.name}: {result.message}")
                    if result.suggestions:
                        lines.append("    Suggestions:")
                        for suggestion in result.suggestions:
                            lines.append(f"      - {suggestion}")
                    lines.append("")
        
        return "\n".join(lines)
    
    def _generate_json_report(self) -> str:
        """Generate JSON validation report"""
        import json
        
        report_data = {
            "summary": self.get_validation_summary(),
            "results": [
                {
                    "name": r.name,
                    "status": r.status.value,
                    "level": r.level.value,
                    "message": r.message,
                    "details": r.details,
                    "suggestions": r.suggestions
                }
                for r in self.validation_results
            ]
        }
        
        return json.dumps(report_data, indent=2)


def validate_test_environment(requirements_file: Optional[str] = None) -> EnvironmentValidator:
    """
    Convenience function to validate test environment
    
    Args:
        requirements_file: Path to requirements configuration file
        
    Returns:
        EnvironmentValidator with validation results
    """
    validator = EnvironmentValidator(requirements_file)
    validator.validate_environment()
    return validator