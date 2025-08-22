"""
Environment Validator for Local Testing Framework

Validates system prerequisites, dependencies, and configuration files
to ensure the testing environment is properly set up.
"""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import importlib.util

from .models.test_results import ValidationResult, ValidationStatus, EnvironmentValidationResults
from .models.configuration import LocalTestConfiguration


class EnvironmentValidator:
    """
    Main environment validation orchestrator that checks system prerequisites,
    dependencies, and configuration files.
    """
    
    def __init__(self, config: Optional[LocalTestConfiguration] = None):
        """Initialize the environment validator with configuration"""
        self.config = config or LocalTestConfiguration()
        self.platform_info = self._detect_platform()
        
    def _detect_platform(self) -> Dict[str, str]:
        """Detect platform information"""
        return {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version()
        }
    
    def validate_python_version(self) -> ValidationResult:
        """
        Validate Python version meets minimum requirements
        Uses `python --version` command to verify installation
        """
        try:
            # Get current Python version
            current_version = platform.python_version()
            required_version = self.config.environment_requirements.min_python_version
            
            # Parse version numbers for comparison
            current_parts = [int(x) for x in current_version.split('.')]
            required_parts = [int(x) for x in required_version.split('.')]
            
            # Compare versions
            meets_requirement = current_parts >= required_parts
            
            if meets_requirement:
                return ValidationResult(
                    component="python_version",
                    status=ValidationStatus.PASSED,
                    message=f"Python {current_version} meets requirement (>= {required_version})",
                    details={
                        "current_version": current_version,
                        "required_version": required_version,
                        "executable_path": sys.executable,
                        "platform_info": self.platform_info
                    }
                )
            else:
                return ValidationResult(
                    component="python_version",
                    status=ValidationStatus.FAILED,
                    message=f"Python {current_version} does not meet requirement (>= {required_version})",
                    details={
                        "current_version": current_version,
                        "required_version": required_version,
                        "executable_path": sys.executable
                    },
                    remediation_steps=[
                        f"Upgrade Python to version {required_version} or higher",
                        "Download from https://python.org/downloads/",
                        "Verify installation with: python --version"
                    ]
                )
                
        except Exception as e:
            return ValidationResult(
                component="python_version",
                status=ValidationStatus.FAILED,
                message=f"Failed to validate Python version: {str(e)}",
                details={"error": str(e)},
                remediation_steps=[
                    "Ensure Python is properly installed",
                    "Verify Python is in system PATH",
                    "Try running: python --version"
                ]
            )
    
    def validate_dependencies(self) -> ValidationResult:
        """
        Validate that all required packages from requirements.txt are installed
        """
        try:
            requirements_path = Path(self.config.requirements_path)
            
            if not requirements_path.exists():
                return ValidationResult(
                    component="dependencies",
                    status=ValidationStatus.FAILED,
                    message=f"Requirements file not found: {requirements_path}",
                    details={"requirements_path": str(requirements_path)},
                    remediation_steps=[
                        f"Create {requirements_path} file with required packages",
                        "Add necessary dependencies for the application"
                    ]
                )
            
            # Read requirements file
            with open(requirements_path, 'r') as f:
                requirements = f.read().strip().split('\n')
            
            # Filter out empty lines and comments
            requirements = [req.strip() for req in requirements 
                          if req.strip() and not req.strip().startswith('#')]
            
            missing_packages = []
            installed_packages = []
            
            for requirement in requirements:
                # Parse package name (handle version specifiers)
                package_name = requirement.split('==')[0].split('>=')[0].split('<=')[0].split('>')[0].split('<')[0].strip()
                
                # Map package names to their import names
                import_name_map = {
                    'Pillow': 'PIL',
                    'opencv-python': 'cv2',
                    'pyyaml': 'yaml',
                    'huggingface-hub': 'huggingface_hub',
                    'imageio-ffmpeg': 'imageio_ffmpeg',
                    'python-multipart': 'multipart',
                    'nvidia-ml-py': 'pynvml'
                }
                
                import_name = import_name_map.get(package_name, package_name.replace('-', '_'))
                
                try:
                    # Try to import the package
                    spec = importlib.util.find_spec(import_name)
                    if spec is not None:
                        installed_packages.append(requirement)
                    else:
                        missing_packages.append(requirement)
                except (ImportError, ModuleNotFoundError):
                    missing_packages.append(requirement)
            
            if not missing_packages:
                return ValidationResult(
                    component="dependencies",
                    status=ValidationStatus.PASSED,
                    message=f"All {len(installed_packages)} required packages are installed",
                    details={
                        "total_packages": len(requirements),
                        "installed_packages": installed_packages,
                        "requirements_file": str(requirements_path)
                    }
                )
            else:
                return ValidationResult(
                    component="dependencies",
                    status=ValidationStatus.FAILED,
                    message=f"{len(missing_packages)} required packages are missing",
                    details={
                        "missing_packages": missing_packages,
                        "installed_packages": installed_packages,
                        "requirements_file": str(requirements_path)
                    },
                    remediation_steps=[
                        f"Install missing packages: pip install {' '.join(missing_packages)}",
                        f"Or install all requirements: pip install -r {requirements_path}",
                        "Verify installation with: pip list"
                    ]
                )
                
        except Exception as e:
            return ValidationResult(
                component="dependencies",
                status=ValidationStatus.FAILED,
                message=f"Failed to validate dependencies: {str(e)}",
                details={"error": str(e)},
                remediation_steps=[
                    "Check if requirements.txt file is readable",
                    "Verify pip is installed and working",
                    "Try: pip install -r requirements.txt"
                ]
            )
    
    def validate_cuda_availability(self) -> ValidationResult:
        """
        Validate CUDA availability using torch.cuda.is_available()
        Also detect multi-GPU setup and non-NVIDIA hardware fallbacks
        """
        try:
            # Try to import torch
            try:
                import torch
            except ImportError:
                return ValidationResult(
                    component="cuda_availability",
                    status=ValidationStatus.WARNING,
                    message="PyTorch not installed - CUDA validation skipped",
                    details={"torch_available": False},
                    remediation_steps=[
                        "Install PyTorch: pip install torch",
                        "For CUDA support: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
                    ]
                )
            
            # Check CUDA availability
            cuda_available = torch.cuda.is_available()
            
            if cuda_available:
                # Get CUDA device information
                device_count = torch.cuda.device_count()
                devices_info = []
                
                for i in range(device_count):
                    device_props = torch.cuda.get_device_properties(i)
                    devices_info.append({
                        "device_id": i,
                        "name": device_props.name,
                        "total_memory": device_props.total_memory,
                        "major": device_props.major,
                        "minor": device_props.minor,
                        "multi_processor_count": device_props.multi_processor_count
                    })
                
                # Check for multi-GPU setup
                multi_gpu = device_count > 1
                
                return ValidationResult(
                    component="cuda_availability",
                    status=ValidationStatus.PASSED,
                    message=f"CUDA available with {device_count} device(s)",
                    details={
                        "cuda_available": True,
                        "device_count": device_count,
                        "devices": devices_info,
                        "multi_gpu": multi_gpu,
                        "cuda_version": torch.version.cuda,
                        "pytorch_version": torch.__version__
                    }
                )
            else:
                # CUDA not available - check for alternative hardware
                fallback_info = self._detect_hardware_fallbacks()
                
                return ValidationResult(
                    component="cuda_availability",
                    status=ValidationStatus.WARNING,
                    message="CUDA not available - using CPU fallback",
                    details={
                        "cuda_available": False,
                        "fallback_options": fallback_info,
                        "pytorch_version": torch.__version__
                    },
                    remediation_steps=[
                        "Install CUDA-compatible PyTorch if NVIDIA GPU is available",
                        "Check NVIDIA drivers: nvidia-smi",
                        "For CPU-only usage, performance will be significantly slower",
                        "Consider using cloud GPU instances for better performance"
                    ]
                )
                
        except Exception as e:
            return ValidationResult(
                component="cuda_availability",
                status=ValidationStatus.FAILED,
                message=f"Failed to validate CUDA availability: {str(e)}",
                details={"error": str(e)},
                remediation_steps=[
                    "Install PyTorch: pip install torch",
                    "Check NVIDIA drivers if GPU is available",
                    "Verify CUDA installation"
                ]
            )
    
    def _detect_hardware_fallbacks(self) -> Dict[str, any]:
        """Detect non-NVIDIA hardware and fallback options"""
        fallback_info = {
            "cpu_cores": os.cpu_count(),
            "platform": self.platform_info["system"],
            "architecture": self.platform_info["machine"]
        }
        
        # Try to detect other GPU types
        try:
            # Check for AMD GPU (basic detection)
            if self.platform_info["system"] == "Linux":
                try:
                    result = subprocess.run(["lspci"], capture_output=True, text=True, timeout=5)
                    if "AMD" in result.stdout or "Radeon" in result.stdout:
                        fallback_info["amd_gpu_detected"] = True
                except:
                    pass
            
            # Check for Intel GPU
            if "Intel" in self.platform_info["processor"]:
                fallback_info["intel_integrated"] = True
                
        except Exception:
            pass
        
        return fallback_info

    def validate_configuration_files(self) -> ValidationResult:
        """
        Validate config.json file with required fields check
        """
        try:
            config_path = Path(self.config.config_path)
            
            if not config_path.exists():
                return ValidationResult(
                    component="configuration",
                    status=ValidationStatus.FAILED,
                    message=f"Configuration file not found: {config_path}",
                    details={"config_path": str(config_path)},
                    remediation_steps=[
                        f"Create {config_path} file",
                        "Include required sections: system, directories, optimization, performance",
                        "Use sample configuration as template"
                    ]
                )
            
            # Read and parse config file
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
            except json.JSONDecodeError as e:
                return ValidationResult(
                    component="configuration",
                    status=ValidationStatus.FAILED,
                    message=f"Invalid JSON in configuration file: {str(e)}",
                    details={"config_path": str(config_path), "json_error": str(e)},
                    remediation_steps=[
                        "Fix JSON syntax errors in config.json",
                        "Validate JSON format using online JSON validator",
                        "Check for missing commas, brackets, or quotes"
                    ]
                )
            
            # Check required fields
            required_fields = self.config.environment_requirements.required_config_fields
            missing_fields = []
            present_fields = []
            
            for field in required_fields:
                if field in config_data:
                    present_fields.append(field)
                else:
                    missing_fields.append(field)
            
            if missing_fields:
                return ValidationResult(
                    component="configuration",
                    status=ValidationStatus.FAILED,
                    message=f"Missing required configuration fields: {', '.join(missing_fields)}",
                    details={
                        "config_path": str(config_path),
                        "missing_fields": missing_fields,
                        "present_fields": present_fields,
                        "required_fields": required_fields
                    },
                    remediation_steps=[
                        f"Add missing fields to {config_path}: {', '.join(missing_fields)}",
                        "Refer to documentation for field specifications",
                        "Use sample configuration as reference"
                    ]
                )
            
            # Additional validation for specific fields
            validation_issues = []
            
            # Check directories section
            if "directories" in config_data:
                dirs = config_data["directories"]
                if isinstance(dirs, dict):
                    for dir_key, dir_path in dirs.items():
                        if dir_path and not Path(dir_path).parent.exists():
                            validation_issues.append(f"Directory path parent does not exist: {dir_path}")
            
            if validation_issues:
                return ValidationResult(
                    component="configuration",
                    status=ValidationStatus.WARNING,
                    message=f"Configuration validation issues found: {len(validation_issues)}",
                    details={
                        "config_path": str(config_path),
                        "issues": validation_issues,
                        "present_fields": present_fields
                    },
                    remediation_steps=[
                        "Review and fix configuration issues",
                        "Ensure directory paths are valid",
                        "Check file permissions"
                    ]
                )
            
            return ValidationResult(
                component="configuration",
                status=ValidationStatus.PASSED,
                message=f"Configuration file valid with all {len(required_fields)} required fields",
                details={
                    "config_path": str(config_path),
                    "present_fields": present_fields,
                    "config_size": config_path.stat().st_size
                }
            )
            
        except Exception as e:
            return ValidationResult(
                component="configuration",
                status=ValidationStatus.FAILED,
                message=f"Failed to validate configuration: {str(e)}",
                details={"error": str(e)},
                remediation_steps=[
                    "Check if config.json file is readable",
                    "Verify file permissions",
                    "Ensure valid JSON format"
                ]
            )

    def validate_environment_variables(self) -> ValidationResult:
        """
        Validate .env file and required environment variables
        Includes cross-platform environment variable setup
        """
        try:
            env_path = Path(self.config.env_path)
            required_vars = self.config.environment_requirements.required_env_vars
            
            # Check if .env file exists
            env_file_exists = env_path.exists()
            env_vars_from_file = {}
            
            if env_file_exists:
                try:
                    with open(env_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#') and '=' in line:
                                key, value = line.split('=', 1)
                                env_vars_from_file[key.strip()] = value.strip()
                except Exception as e:
                    return ValidationResult(
                        component="environment_variables",
                        status=ValidationStatus.FAILED,
                        message=f"Failed to read .env file: {str(e)}",
                        details={"env_path": str(env_path), "error": str(e)},
                        remediation_steps=[
                            "Check .env file format",
                            "Ensure file is readable",
                            "Use format: VARIABLE_NAME=value"
                        ]
                    )
            
            # Check required environment variables
            missing_vars = []
            present_vars = []
            
            for var in required_vars:
                # Check both system environment and .env file
                if var in os.environ or var in env_vars_from_file:
                    present_vars.append(var)
                else:
                    missing_vars.append(var)
            
            # Generate cross-platform setup instructions
            platform_commands = self._generate_env_setup_commands(missing_vars)
            
            if missing_vars:
                remediation_steps = [
                    f"Set missing environment variables: {', '.join(missing_vars)}"
                ]
                remediation_steps.extend(platform_commands)
                
                return ValidationResult(
                    component="environment_variables",
                    status=ValidationStatus.FAILED,
                    message=f"Missing required environment variables: {', '.join(missing_vars)}",
                    details={
                        "env_path": str(env_path),
                        "env_file_exists": env_file_exists,
                        "missing_vars": missing_vars,
                        "present_vars": present_vars,
                        "platform_commands": platform_commands
                    },
                    remediation_steps=remediation_steps
                )
            
            # Validate specific variables
            validation_issues = []
            
            # Check HF_TOKEN if present
            hf_token = os.environ.get('HF_TOKEN') or env_vars_from_file.get('HF_TOKEN')
            if hf_token:
                if len(hf_token) < 10:  # Basic validation
                    validation_issues.append("HF_TOKEN appears to be too short")
                elif not hf_token.startswith('hf_'):
                    validation_issues.append("HF_TOKEN should start with 'hf_'")
            
            if validation_issues:
                return ValidationResult(
                    component="environment_variables",
                    status=ValidationStatus.WARNING,
                    message=f"Environment variable validation issues: {len(validation_issues)}",
                    details={
                        "issues": validation_issues,
                        "present_vars": present_vars
                    },
                    remediation_steps=[
                        "Review environment variable values",
                        "Ensure HF_TOKEN is valid Hugging Face token",
                        "Check token permissions on Hugging Face"
                    ]
                )
            
            return ValidationResult(
                component="environment_variables",
                status=ValidationStatus.PASSED,
                message=f"All {len(required_vars)} required environment variables are set",
                details={
                    "env_path": str(env_path),
                    "env_file_exists": env_file_exists,
                    "present_vars": present_vars,
                    "vars_from_file": len(env_vars_from_file),
                    "vars_from_system": len([v for v in required_vars if v in os.environ])
                }
            )
            
        except Exception as e:
            return ValidationResult(
                component="environment_variables",
                status=ValidationStatus.FAILED,
                message=f"Failed to validate environment variables: {str(e)}",
                details={"error": str(e)},
                remediation_steps=[
                    "Check .env file format and permissions",
                    "Verify environment variable syntax",
                    "Ensure required variables are set"
                ]
            )

    def _generate_env_setup_commands(self, missing_vars: List[str]) -> List[str]:
        """Generate cross-platform environment variable setup commands"""
        if not missing_vars:
            return []
        
        commands = []
        system = self.platform_info["system"]
        
        if system == "Windows":
            commands.append("Windows (Command Prompt):")
            for var in missing_vars:
                commands.append(f"  setx {var} \"your_value_here\"")
            commands.append("Windows (PowerShell):")
            for var in missing_vars:
                commands.append(f"  $env:{var} = \"your_value_here\"")
        else:  # Linux/macOS
            commands.append("Linux/macOS (bash):")
            for var in missing_vars:
                commands.append(f"  export {var}=\"your_value_here\"")
            commands.append("Add to ~/.bashrc or ~/.zshrc for persistence:")
            for var in missing_vars:
                commands.append(f"  echo 'export {var}=\"your_value_here\"' >> ~/.bashrc")
        
        commands.append("Or add to .env file:")
        for var in missing_vars:
            commands.append(f"  {var}=your_value_here")
        
        return commands

    def validate_security_basics(self) -> ValidationResult:
        """
        Basic security validation for HTTPS and file permissions
        """
        try:
            security_issues = []
            security_checks = []
            
            # Check file permissions on sensitive files
            sensitive_files = [self.config.config_path, self.config.env_path]
            
            for file_path in sensitive_files:
                path = Path(file_path)
                if path.exists():
                    try:
                        stat_info = path.stat()
                        # Check if file is world-readable (basic check)
                        if stat_info.st_mode & 0o004:  # World readable
                            security_issues.append(f"{file_path} is world-readable")
                        security_checks.append(f"{file_path}: permissions OK")
                    except Exception:
                        security_checks.append(f"{file_path}: permission check failed")
            
            # Check for HTTPS configuration in config.json
            try:
                config_path = Path(self.config.config_path)
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                    
                    # Look for SSL/HTTPS settings
                    if 'ssl' in config_data or 'https' in config_data:
                        security_checks.append("HTTPS configuration found")
                    else:
                        security_issues.append("No HTTPS configuration found")
            except Exception:
                security_issues.append("Could not check HTTPS configuration")
            
            if security_issues:
                return ValidationResult(
                    component="security",
                    status=ValidationStatus.WARNING,
                    message=f"Security issues found: {len(security_issues)}",
                    details={
                        "issues": security_issues,
                        "checks_passed": security_checks
                    },
                    remediation_steps=[
                        "Review file permissions on sensitive files",
                        "Consider enabling HTTPS for production",
                        "Restrict access to configuration files",
                        "Use environment variables for secrets"
                    ]
                )
            
            return ValidationResult(
                component="security",
                status=ValidationStatus.PASSED,
                message="Basic security checks passed",
                details={"checks_passed": security_checks}
            )
            
        except Exception as e:
            return ValidationResult(
                component="security",
                status=ValidationStatus.WARNING,
                message=f"Security validation failed: {str(e)}",
                details={"error": str(e)},
                remediation_steps=[
                    "Review security configuration manually",
                    "Check file permissions",
                    "Ensure sensitive data is protected"
                ]
            )

    def validate_full_environment(self) -> EnvironmentValidationResults:
        """
        Run complete environment validation and return comprehensive results
        """
        # Run individual validations
        python_result = self.validate_python_version()
        dependencies_result = self.validate_dependencies()
        cuda_result = self.validate_cuda_availability()
        
        config_result = self.validate_configuration_files()
        env_vars_result = self.validate_environment_variables()
        
        # Determine overall status
        results = [python_result, dependencies_result, cuda_result, config_result, env_vars_result]
        failed_results = [r for r in results if r.status == ValidationStatus.FAILED]
        
        if failed_results:
            overall_status = ValidationStatus.FAILED
        elif any(r.status == ValidationStatus.WARNING for r in results):
            overall_status = ValidationStatus.WARNING
        else:
            overall_status = ValidationStatus.PASSED
        
        # Collect all remediation steps
        all_remediation_steps = []
        for result in failed_results:
            all_remediation_steps.extend(result.remediation_steps)
        
        return EnvironmentValidationResults(
            python_version=python_result,
            cuda_availability=cuda_result,
            dependencies=dependencies_result,
            configuration=config_result,
            environment_variables=env_vars_result,
            overall_status=overall_status,
            remediation_steps=all_remediation_steps
        )
    
    def generate_environment_report(self, results: EnvironmentValidationResults) -> str:
        """Generate a human-readable environment validation report"""
        report_lines = [
            "Environment Validation Report",
            "=" * 30,
            f"Timestamp: {results.timestamp}",
            f"Platform: {self.platform_info['system']} {self.platform_info['release']}",
            f"Python: {self.platform_info['python_version']}",
            "",
            "Validation Results:",
            "-" * 20
        ]
        
        # Add individual validation results
        for component, result in [
            ("Python Version", results.python_version),
            ("CUDA Availability", results.cuda_availability),
            ("Dependencies", results.dependencies),
            ("Configuration", results.configuration),
            ("Environment Variables", results.environment_variables)
        ]:
            status_symbol = {
                ValidationStatus.PASSED: "✓",
                ValidationStatus.FAILED: "✗",
                ValidationStatus.WARNING: "⚠",
                ValidationStatus.SKIPPED: "○"
            }.get(result.status, "?")
            
            report_lines.append(f"{status_symbol} {component}: {result.message}")
        
        # Add overall status
        report_lines.extend([
            "",
            f"Overall Status: {results.overall_status.value.upper()}",
            ""
        ])
        
        # Add remediation steps if any
        if results.remediation_steps:
            report_lines.extend([
                "Remediation Steps:",
                "-" * 18
            ])
            for i, step in enumerate(results.remediation_steps, 1):
                report_lines.append(f"{i}. {step}")
        
        return "\n".join(report_lines)
    
    def generate_remediation_instructions(self, results: EnvironmentValidationResults) -> str:
        """
        Generate comprehensive remediation instructions with specific commands
        """
        failed_validations = results.get_failed_validations()
        
        if not failed_validations:
            return "No remediation needed - all validations passed!"
        
        instructions = [
            "Remediation Instructions",
            "=" * 25,
            f"Found {len(failed_validations)} issue(s) that need attention:",
            ""
        ]
        
        for i, validation in enumerate(failed_validations, 1):
            instructions.extend([
                f"{i}. {validation.component.upper()}: {validation.message}",
                "   Steps to fix:"
            ])
            
            for j, step in enumerate(validation.remediation_steps, 1):
                instructions.append(f"   {i}.{j} {step}")
            
            instructions.append("")
        
        # Add platform-specific notes
        instructions.extend([
            "Platform-Specific Notes:",
            "-" * 23,
            f"Operating System: {self.platform_info['system']} {self.platform_info['release']}",
            f"Python Version: {self.platform_info['python_version']}",
            ""
        ])
        
        # Add general recommendations
        instructions.extend([
            "General Recommendations:",
            "-" * 24,
            "1. Fix issues in the order listed above",
            "2. Restart your terminal/IDE after environment changes",
            "3. Re-run validation after each fix to verify success",
            "4. Keep backup copies of configuration files",
            ""
        ])
        
        # Add quick verification commands
        instructions.extend([
            "Quick Verification Commands:",
            "-" * 29,
            "• Python version: python --version",
            "• Installed packages: pip list",
            "• Environment variables: echo $HF_TOKEN (Linux/macOS) or echo %HF_TOKEN% (Windows)",
            "• CUDA availability: python -c \"import torch; print(torch.cuda.is_available())\"",
            ""
        ])
        
        return "\n".join(instructions)
    
    def get_automated_fix_commands(self, results: EnvironmentValidationResults) -> List[str]:
        """
        Generate automated fix commands that can be executed programmatically
        """
        failed_validations = results.get_failed_validations()
        commands = []
        
        for validation in failed_validations:
            if validation.component == "dependencies":
                # Extract package installation commands
                for step in validation.remediation_steps:
                    if step.startswith("Install missing packages:"):
                        cmd = step.replace("Install missing packages: ", "")
                        commands.append(cmd)
                    elif step.startswith("Or install all requirements:"):
                        cmd = step.replace("Or install all requirements: ", "")
                        commands.append(cmd)
            
            elif validation.component == "environment_variables":
                # Extract environment variable setup commands
                missing_vars = validation.details.get("missing_vars", [])
                platform_commands = validation.details.get("platform_commands", [])
                
                # Add platform-specific commands
                for cmd in platform_commands:
                    if cmd.strip() and not cmd.endswith(":"):
                        commands.append(cmd.strip())
        
        return commands