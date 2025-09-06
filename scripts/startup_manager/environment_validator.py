"""
Environment Validator for WAN22 Startup Manager.
Validates system requirements, dependencies, and configurations before server startup.
"""

import sys
import os
import subprocess
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import importlib.util
import venv


class ValidationStatus(Enum):
    """Status of validation checks."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    FIXED = "fixed"


@dataclass
class ValidationIssue:
    """Represents a validation issue found during environment checking."""
    component: str
    issue_type: str
    message: str
    status: ValidationStatus
    auto_fixable: bool = False
    fix_command: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class ValidationResult:
    """Result of environment validation."""
    is_valid: bool
    issues: List[ValidationIssue]
    fixes_applied: List[str]
    warnings: List[str]
    system_info: Dict[str, Any]


class DependencyValidator:
    """Validates Python and Node.js dependencies and environments."""
    
    def __init__(self):
        self.python_min_version = (3, 8, 0)
        self.node_min_version = "16.0.0"
        self.npm_min_version = "8.0.0"
        
    def validate_python_environment(self) -> List[ValidationIssue]:
        """Validate Python version, virtual environment, and dependencies."""
        issues = []
        
        # Check Python version
        python_version_issue = self._check_python_version()
        if python_version_issue:
            issues.append(python_version_issue)
        
        # Check virtual environment
        venv_issue = self._check_virtual_environment()
        if venv_issue:
            issues.append(venv_issue)
        
        # Check backend dependencies
        backend_deps_issues = self._check_backend_dependencies()
        issues.extend(backend_deps_issues)
        
        return issues
    
    def validate_node_environment(self) -> List[ValidationIssue]:
        """Validate Node.js and npm versions and frontend dependencies."""
        issues = []
        
        # Check Node.js installation and version
        node_issue = self._check_node_version()
        if node_issue:
            issues.append(node_issue)
        
        # Check npm version
        npm_issue = self._check_npm_version()
        if npm_issue:
            issues.append(npm_issue)
        
        # Check frontend dependencies
        frontend_deps_issues = self._check_frontend_dependencies()
        issues.extend(frontend_deps_issues)
        
        return issues
    
    def _check_python_version(self) -> Optional[ValidationIssue]:
        """Check if Python version meets minimum requirements."""
        current_version = sys.version_info[:3]
        
        if current_version < self.python_min_version:
            return ValidationIssue(
                component="python",
                issue_type="version_too_old",
                message=f"Python version {'.'.join(map(str, current_version))} is below minimum required {'.'.join(map(str, self.python_min_version))}",
                status=ValidationStatus.FAILED,
                auto_fixable=False,
                details={"current_version": current_version, "required_version": self.python_min_version}
            )
        
        return None
    
    def _check_virtual_environment(self) -> Optional[ValidationIssue]:
        """Check if running in a virtual environment."""
        # Check for virtual environment indicators
        in_venv = (
            hasattr(sys, 'real_prefix') or  # virtualenv
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or  # venv
            'VIRTUAL_ENV' in os.environ  # environment variable
        )
        
        if not in_venv:
            return ValidationIssue(
                component="python",
                issue_type="no_virtual_env",
                message="Not running in a virtual environment. This may cause dependency conflicts.",
                status=ValidationStatus.WARNING,
                auto_fixable=True,
                fix_command="python -m venv venv && venv\\Scripts\\activate",
                details={"recommendation": "Use virtual environment for isolated dependencies"}
            )
        
        return None
    
    def _check_backend_dependencies(self) -> List[ValidationIssue]:
        """Check if backend dependencies are installed."""
        issues = []
        requirements_file = Path("backend/requirements.txt")
        
        if not requirements_file.exists():
            issues.append(ValidationIssue(
                component="backend",
                issue_type="missing_requirements_file",
                message="Backend requirements.txt file not found",
                status=ValidationStatus.FAILED,
                auto_fixable=False,
                details={"expected_path": str(requirements_file)}
            ))
            return issues
        
        # Read requirements
        try:
            with open(requirements_file, 'r', encoding='utf-8') as f:
                requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        except Exception as e:
            issues.append(ValidationIssue(
                component="backend",
                issue_type="requirements_read_error",
                message=f"Failed to read requirements.txt: {e}",
                status=ValidationStatus.FAILED,
                auto_fixable=False
            ))
            return issues
        
        # Check critical dependencies
        critical_deps = ['fastapi', 'uvicorn', 'pydantic']
        missing_deps = []
        
        for dep in critical_deps:
            if not self._is_package_installed(dep):
                missing_deps.append(dep)
        
        if missing_deps:
            issues.append(ValidationIssue(
                component="backend",
                issue_type="missing_dependencies",
                message=f"Missing critical backend dependencies: {', '.join(missing_deps)}",
                status=ValidationStatus.FAILED,
                auto_fixable=True,
                fix_command=f"pip install {' '.join(missing_deps)}",
                details={"missing_packages": missing_deps}
            ))
        
        # Check if all requirements are installed
        uninstalled_reqs = []
        for req in requirements:
            # Simple package name extraction (handles basic cases)
            package_name = req.split('==')[0].split('>=')[0].split('<=')[0].split('>')[0].split('<')[0].strip()
            if not self._is_package_installed(package_name):
                uninstalled_reqs.append(req)
        
        if uninstalled_reqs:
            issues.append(ValidationIssue(
                component="backend",
                issue_type="uninstalled_requirements",
                message=f"Some backend requirements are not installed: {len(uninstalled_reqs)} packages",
                status=ValidationStatus.WARNING,
                auto_fixable=True,
                fix_command="pip install -r backend/requirements.txt",
                details={"uninstalled_count": len(uninstalled_reqs)}
            ))
        
        return issues
    
    def _check_node_version(self) -> Optional[ValidationIssue]:
        """Check Node.js installation and version."""
        try:
            result = subprocess.run(['node', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, 'node --version')
            
            version_str = result.stdout.strip().lstrip('v')
            current_version = tuple(map(int, version_str.split('.')))
            required_version = tuple(map(int, self.node_min_version.split('.')))
            
            if current_version < required_version:
                return ValidationIssue(
                    component="nodejs",
                    issue_type="version_too_old",
                    message=f"Node.js version {version_str} is below minimum required {self.node_min_version}",
                    status=ValidationStatus.FAILED,
                    auto_fixable=False,
                    details={"current_version": version_str, "required_version": self.node_min_version}
                )
                
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return ValidationIssue(
                component="nodejs",
                issue_type="not_installed",
                message="Node.js is not installed or not accessible",
                status=ValidationStatus.FAILED,
                auto_fixable=False,
                fix_command="Download and install Node.js from https://nodejs.org/",
                details={"required_version": self.node_min_version}
            )
        
        return None
    
    def _check_npm_version(self) -> Optional[ValidationIssue]:
        """Check npm installation and version."""
        try:
            result = subprocess.run(['npm', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, 'npm --version')
            
            version_str = result.stdout.strip()
            current_version = tuple(map(int, version_str.split('.')))
            required_version = tuple(map(int, self.npm_min_version.split('.')))
            
            if current_version < required_version:
                return ValidationIssue(
                    component="npm",
                    issue_type="version_too_old",
                    message=f"npm version {version_str} is below minimum required {self.npm_min_version}",
                    status=ValidationStatus.FAILED,
                    auto_fixable=True,
                    fix_command="npm install -g npm@latest",
                    details={"current_version": version_str, "required_version": self.npm_min_version}
                )
                
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return ValidationIssue(
                component="npm",
                issue_type="not_installed",
                message="npm is not installed or not accessible",
                status=ValidationStatus.FAILED,
                auto_fixable=False,
                details={"required_version": self.npm_min_version}
            )
        
        return None
    
    def _check_frontend_dependencies(self) -> List[ValidationIssue]:
        """Check if frontend dependencies are installed."""
        issues = []
        package_json_path = Path("frontend/package.json")
        node_modules_path = Path("frontend/node_modules")
        
        if not package_json_path.exists():
            issues.append(ValidationIssue(
                component="frontend",
                issue_type="missing_package_json",
                message="Frontend package.json file not found",
                status=ValidationStatus.FAILED,
                auto_fixable=False,
                details={"expected_path": str(package_json_path)}
            ))
            return issues
        
        # Check if node_modules exists
        if not node_modules_path.exists():
            issues.append(ValidationIssue(
                component="frontend",
                issue_type="dependencies_not_installed",
                message="Frontend dependencies not installed (node_modules missing)",
                status=ValidationStatus.FAILED,
                auto_fixable=True,
                fix_command="cd frontend && npm install",
                details={"node_modules_path": str(node_modules_path)}
            ))
            return issues
        
        # Check package-lock.json for consistency
        package_lock_path = Path("frontend/package-lock.json")
        if package_lock_path.exists():
            try:
                # Check if package-lock is newer than node_modules
                lock_mtime = package_lock_path.stat().st_mtime
                modules_mtime = node_modules_path.stat().st_mtime
                
                if lock_mtime > modules_mtime:
                    issues.append(ValidationIssue(
                        component="frontend",
                        issue_type="dependencies_outdated",
                        message="Frontend dependencies may be outdated (package-lock.json is newer than node_modules)",
                        status=ValidationStatus.WARNING,
                        auto_fixable=True,
                        fix_command="cd frontend && npm install",
                        details={"recommendation": "Run npm install to update dependencies"}
                    ))
            except OSError:
                pass  # Ignore file stat errors
        
        return issues
    
    def _is_package_installed(self, package_name: str) -> bool:
        """Check if a Python package is installed."""
        try:
            importlib.import_module(package_name)
            return True
        except ImportError:
            # Try with underscores (some packages use different import names)
            try:
                importlib.import_module(package_name.replace('-', '_'))
                return True
            except ImportError:
                return False


class ConfigurationValidator:
    """Validates and repairs configuration files."""
    
    def __init__(self):
        self.required_backend_config_fields = {
            "model_path": str,
            "device": str,
            "max_memory": (int, float),
            "batch_size": int,
            "num_inference_steps": int,
            "guidance_scale": (int, float),
            "height": int,
            "width": int,
            "num_frames": int,
            "fps": int
        }
        
        self.required_frontend_config_fields = {
            "name": str,
            "version": str,
            "scripts": dict,
            "dependencies": dict,
            "devDependencies": dict
        }
    
    def validate_backend_config(self) -> List[ValidationIssue]:
        """Validate backend config.json file."""
        issues = []
        config_path = Path("backend/config.json")
        
        if not config_path.exists():
            issues.append(ValidationIssue(
                component="backend_config",
                issue_type="missing_config_file",
                message="Backend config.json file not found",
                status=ValidationStatus.FAILED,
                auto_fixable=True,
                fix_command="create_default_backend_config",
                details={"expected_path": str(config_path)}
            ))
            return issues
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        except json.JSONDecodeError as e:
            issues.append(ValidationIssue(
                component="backend_config",
                issue_type="invalid_json",
                message=f"Backend config.json contains invalid JSON: {e}",
                status=ValidationStatus.FAILED,
                auto_fixable=True,
                fix_command="repair_backend_config_json",
                details={"json_error": str(e)}
            ))
            return issues
        except Exception as e:
            issues.append(ValidationIssue(
                component="backend_config",
                issue_type="config_read_error",
                message=f"Failed to read backend config.json: {e}",
                status=ValidationStatus.FAILED,
                auto_fixable=False,
                details={"error": str(e)}
            ))
            return issues
        
        # Check required fields
        missing_fields = []
        invalid_types = []
        
        for field, expected_type in self.required_backend_config_fields.items():
            if field not in config_data:
                missing_fields.append(field)
            else:
                value = config_data[field]
                if isinstance(expected_type, tuple):
                    # Multiple allowed types
                    if not any(isinstance(value, t) for t in expected_type):
                        invalid_types.append((field, expected_type, type(value).__name__))
                else:
                    # Single expected type
                    if not isinstance(value, expected_type):
                        invalid_types.append((field, expected_type.__name__, type(value).__name__))
        
        if missing_fields:
            issues.append(ValidationIssue(
                component="backend_config",
                issue_type="missing_required_fields",
                message=f"Backend config.json missing required fields: {', '.join(missing_fields)}",
                status=ValidationStatus.FAILED,
                auto_fixable=True,
                fix_command="add_missing_backend_config_fields",
                details={"missing_fields": missing_fields}
            ))
        
        if invalid_types:
            type_errors = [f"{field} (expected {expected}, got {actual})" for field, expected, actual in invalid_types]
            issues.append(ValidationIssue(
                component="backend_config",
                issue_type="invalid_field_types",
                message=f"Backend config.json has invalid field types: {', '.join(type_errors)}",
                status=ValidationStatus.WARNING,
                auto_fixable=True,
                fix_command="fix_backend_config_types",
                details={"type_errors": invalid_types}
            ))
        
        # Validate specific field values
        validation_issues = self._validate_backend_config_values(config_data)
        issues.extend(validation_issues)
        
        return issues
    
    def validate_frontend_config(self) -> List[ValidationIssue]:
        """Validate frontend configuration files."""
        issues = []
        
        # Validate package.json
        package_json_issues = self._validate_package_json()
        issues.extend(package_json_issues)
        
        # Validate vite.config.ts
        vite_config_issues = self._validate_vite_config()
        issues.extend(vite_config_issues)
        
        return issues
    
    def _validate_package_json(self) -> List[ValidationIssue]:
        """Validate frontend package.json file."""
        issues = []
        package_json_path = Path("frontend/package.json")
        
        if not package_json_path.exists():
            issues.append(ValidationIssue(
                component="frontend_config",
                issue_type="missing_package_json",
                message="Frontend package.json file not found",
                status=ValidationStatus.FAILED,
                auto_fixable=True,
                fix_command="create_default_package_json",
                details={"expected_path": str(package_json_path)}
            ))
            return issues
        
        try:
            with open(package_json_path, 'r', encoding='utf-8') as f:
                package_data = json.load(f)
        except json.JSONDecodeError as e:
            issues.append(ValidationIssue(
                component="frontend_config",
                issue_type="invalid_package_json",
                message=f"Frontend package.json contains invalid JSON: {e}",
                status=ValidationStatus.FAILED,
                auto_fixable=True,
                fix_command="repair_package_json",
                details={"json_error": str(e)}
            ))
            return issues
        
        # Check required fields
        missing_fields = []
        for field, expected_type in self.required_frontend_config_fields.items():
            if field not in package_data:
                missing_fields.append(field)
            elif not isinstance(package_data[field], expected_type):
                issues.append(ValidationIssue(
                    component="frontend_config",
                    issue_type="invalid_package_json_field",
                    message=f"Frontend package.json field '{field}' has invalid type",
                    status=ValidationStatus.WARNING,
                    auto_fixable=True,
                    fix_command="fix_package_json_field_types",
                    details={"field": field, "expected_type": expected_type.__name__, "actual_type": type(package_data[field]).__name__}
                ))
        
        if missing_fields:
            issues.append(ValidationIssue(
                component="frontend_config",
                issue_type="missing_package_json_fields",
                message=f"Frontend package.json missing required fields: {', '.join(missing_fields)}",
                status=ValidationStatus.FAILED,
                auto_fixable=True,
                fix_command="add_missing_package_json_fields",
                details={"missing_fields": missing_fields}
            ))
        
        # Check for required scripts
        if "scripts" in package_data:
            required_scripts = ["dev", "build", "preview"]
            missing_scripts = [script for script in required_scripts if script not in package_data["scripts"]]
            if missing_scripts:
                issues.append(ValidationIssue(
                    component="frontend_config",
                    issue_type="missing_npm_scripts",
                    message=f"Frontend package.json missing required scripts: {', '.join(missing_scripts)}",
                    status=ValidationStatus.WARNING,
                    auto_fixable=True,
                    fix_command="add_missing_npm_scripts",
                    details={"missing_scripts": missing_scripts}
                ))
        
        return issues
    
    def _validate_vite_config(self) -> List[ValidationIssue]:
        """Validate frontend vite.config.ts file."""
        issues = []
        vite_config_path = Path("frontend/vite.config.ts")
        
        if not vite_config_path.exists():
            issues.append(ValidationIssue(
                component="frontend_config",
                issue_type="missing_vite_config",
                message="Frontend vite.config.ts file not found",
                status=ValidationStatus.WARNING,
                auto_fixable=True,
                fix_command="create_default_vite_config",
                details={"expected_path": str(vite_config_path)}
            ))
            return issues
        
        try:
            with open(vite_config_path, 'r', encoding='utf-8') as f:
                vite_config_content = f.read()
        except Exception as e:
            issues.append(ValidationIssue(
                component="frontend_config",
                issue_type="vite_config_read_error",
                message=f"Failed to read vite.config.ts: {e}",
                status=ValidationStatus.FAILED,
                auto_fixable=False,
                details={"error": str(e)}
            ))
            return issues
        
        # Basic syntax checks
        required_patterns = [
            ("import", "Missing import statements"),
            ("export default", "Missing default export"),
            ("defineConfig", "Missing defineConfig call")
        ]
        
        for pattern, error_msg in required_patterns:
            if pattern not in vite_config_content:
                issues.append(ValidationIssue(
                    component="frontend_config",
                    issue_type="invalid_vite_config",
                    message=f"Vite config issue: {error_msg}",
                    status=ValidationStatus.WARNING,
                    auto_fixable=True,
                    fix_command="repair_vite_config",
                    details={"missing_pattern": pattern, "error": error_msg}
                ))
        
        return issues
    
    def _validate_backend_config_values(self, config_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate specific backend configuration values."""
        issues = []
        
        # Validate device setting
        if "device" in config_data:
            device = config_data["device"]
            valid_devices = ["cpu", "cuda", "auto"]
            if device not in valid_devices:
                issues.append(ValidationIssue(
                    component="backend_config",
                    issue_type="invalid_device_setting",
                    message=f"Invalid device setting '{device}'. Valid options: {', '.join(valid_devices)}",
                    status=ValidationStatus.WARNING,
                    auto_fixable=True,
                    fix_command="fix_device_setting",
                    details={"current_device": device, "valid_devices": valid_devices}
                ))
        
        # Validate resolution settings
        if "height" in config_data and "width" in config_data:
            height, width = config_data["height"], config_data["width"]
            if height <= 0 or width <= 0:
                issues.append(ValidationIssue(
                    component="backend_config",
                    issue_type="invalid_resolution",
                    message=f"Invalid resolution {width}x{height}. Must be positive integers.",
                    status=ValidationStatus.FAILED,
                    auto_fixable=True,
                    fix_command="fix_resolution_settings",
                    details={"current_width": width, "current_height": height}
                ))
            elif height % 8 != 0 or width % 8 != 0:
                issues.append(ValidationIssue(
                    component="backend_config",
                    issue_type="non_optimal_resolution",
                    message=f"Resolution {width}x{height} is not optimal. Dimensions should be multiples of 8.",
                    status=ValidationStatus.WARNING,
                    auto_fixable=True,
                    fix_command="optimize_resolution_settings",
                    details={"current_width": width, "current_height": height}
                ))
        
        # Validate memory settings
        if "max_memory" in config_data:
            max_memory = config_data["max_memory"]
            if max_memory <= 0:
                issues.append(ValidationIssue(
                    component="backend_config",
                    issue_type="invalid_memory_setting",
                    message=f"Invalid max_memory setting {max_memory}. Must be positive.",
                    status=ValidationStatus.FAILED,
                    auto_fixable=True,
                    fix_command="fix_memory_settings",
                    details={"current_memory": max_memory}
                ))
        
        return issues
    
    def auto_repair_config(self, issues: List[ValidationIssue]) -> List[str]:
        """Attempt to automatically repair configuration issues."""
        repairs_applied = []
        
        for issue in issues:
            if issue.auto_fixable and issue.fix_command:
                try:
                    if issue.fix_command == "create_default_backend_config":
                        self._create_default_backend_config()
                        repairs_applied.append("Created default backend config.json")
                        issue.status = ValidationStatus.FIXED
                    
                    elif issue.fix_command == "create_default_package_json":
                        self._create_default_package_json()
                        repairs_applied.append("Created default frontend package.json")
                        issue.status = ValidationStatus.FIXED
                    
                    elif issue.fix_command == "create_default_vite_config":
                        self._create_default_vite_config()
                        repairs_applied.append("Created default vite.config.ts")
                        issue.status = ValidationStatus.FIXED
                    
                    elif issue.fix_command == "add_missing_backend_config_fields":
                        self._add_missing_backend_config_fields(issue.details["missing_fields"])
                        repairs_applied.append(f"Added missing backend config fields: {', '.join(issue.details['missing_fields'])}")
                        issue.status = ValidationStatus.FIXED
                    
                    elif issue.fix_command == "fix_device_setting":
                        self._fix_device_setting()
                        repairs_applied.append("Fixed device setting to 'auto'")
                        issue.status = ValidationStatus.FIXED
                    
                    elif issue.fix_command == "fix_resolution_settings":
                        self._fix_resolution_settings()
                        repairs_applied.append("Fixed resolution settings to 512x512")
                        issue.status = ValidationStatus.FIXED
                
                except Exception as e:
                    # Repair attempt failed, leave issue as is
                    pass
        
        return repairs_applied
    
    def _create_default_backend_config(self):
        """Create a default backend configuration file."""
        default_config = {
            "model_path": "models/",
            "device": "auto",
            "max_memory": 8.0,
            "batch_size": 1,
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "height": 512,
            "width": 512,
            "num_frames": 16,
            "fps": 8
        }
        
        config_path = Path("backend/config.json")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
    
    def _create_default_package_json(self):
        """Create a default frontend package.json file."""
        default_package = {
            "name": "wan22-frontend",
            "version": "1.0.0",
            "type": "module",
            "scripts": {
                "dev": "vite",
                "build": "tsc && vite build",
                "preview": "vite preview"
            },
            "dependencies": {
                "react": "^18.2.0",
                "react-dom": "^18.2.0"
            },
            "devDependencies": {
                "@types/react": "^18.2.0",
                "@types/react-dom": "^18.2.0",
                "@vitejs/plugin-react": "^4.0.0",
                "typescript": "^5.0.0",
                "vite": "^4.4.0"
            }
        }
        
        package_path = Path("frontend/package.json")
        package_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(package_path, 'w', encoding='utf-8') as f:
            json.dump(default_package, f, indent=2, ensure_ascii=False)
    
    def _create_default_vite_config(self):
        """Create a default vite.config.ts file."""
        default_vite_config = '''import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: true
  },
  build: {
    outDir: 'dist',
    sourcemap: true
  }
})
'''
        
        vite_config_path = Path("frontend/vite.config.ts")
        vite_config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(vite_config_path, 'w', encoding='utf-8') as f:
            f.write(default_vite_config)
    
    def _add_missing_backend_config_fields(self, missing_fields: List[str]):
        """Add missing fields to backend config.json."""
        config_path = Path("backend/config.json")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        default_values = {
            "model_path": "models/",
            "device": "auto",
            "max_memory": 8.0,
            "batch_size": 1,
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "height": 512,
            "width": 512,
            "num_frames": 16,
            "fps": 8
        }
        
        for field in missing_fields:
            if field in default_values:
                config_data[field] = default_values[field]
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    def _fix_device_setting(self):
        """Fix invalid device setting in backend config."""
        config_path = Path("backend/config.json")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        config_data["device"] = "auto"
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    def _fix_resolution_settings(self):
        """Fix invalid resolution settings in backend config."""
        config_path = Path("backend/config.json")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        config_data["height"] = 512
        config_data["width"] = 512
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)


class EnvironmentValidator:
    """Main environment validator that orchestrates all validation checks."""
    
    def __init__(self):
        self.dependency_validator = DependencyValidator()
        self.configuration_validator = ConfigurationValidator()
        self.system_info = self._collect_system_info()
    
    def validate_all(self) -> ValidationResult:
        """Run all environment validation checks."""
        all_issues = []
        fixes_applied = []
        warnings = []
        
        # Validate Python environment
        python_issues = self.dependency_validator.validate_python_environment()
        all_issues.extend(python_issues)
        
        # Validate Node.js environment
        node_issues = self.dependency_validator.validate_node_environment()
        all_issues.extend(node_issues)
        
        # Validate backend configuration
        backend_config_issues = self.configuration_validator.validate_backend_config()
        all_issues.extend(backend_config_issues)
        
        # Validate frontend configuration
        frontend_config_issues = self.configuration_validator.validate_frontend_config()
        all_issues.extend(frontend_config_issues)
        
        # Separate warnings from critical issues
        critical_issues = [issue for issue in all_issues if issue.status == ValidationStatus.FAILED]
        warning_issues = [issue for issue in all_issues if issue.status == ValidationStatus.WARNING]
        
        warnings.extend([issue.message for issue in warning_issues])
        
        # Determine overall validity
        is_valid = len(critical_issues) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            issues=all_issues,
            fixes_applied=fixes_applied,
            warnings=warnings,
            system_info=self.system_info
        )
    
    def auto_fix_issues(self, issues: List[ValidationIssue]) -> List[str]:
        """Attempt to automatically fix issues that are marked as auto-fixable."""
        fixes_applied = []
        
        # Separate dependency issues from configuration issues
        dependency_issues = [issue for issue in issues if issue.component in ["backend", "frontend"] and "install" in issue.fix_command]
        config_issues = [issue for issue in issues if issue.component in ["backend_config", "frontend_config"]]
        
        # Handle dependency fixes
        for issue in dependency_issues:
            if issue.auto_fixable and issue.fix_command:
                try:
                    if issue.component == "backend" and "pip install" in issue.fix_command:
                        # Handle pip install commands
                        result = subprocess.run(
                            issue.fix_command.split(),
                            capture_output=True,
                            text=True,
                            timeout=300  # 5 minutes timeout for pip installs
                        )
                        if result.returncode == 0:
                            fixes_applied.append(f"Fixed {issue.component}: {issue.issue_type}")
                            issue.status = ValidationStatus.FIXED
                    
                    elif issue.component == "frontend" and "npm install" in issue.fix_command:
                        # Handle npm install commands
                        cmd_parts = issue.fix_command.split(" && ")
                        for cmd in cmd_parts:
                            if cmd.startswith("cd "):
                                continue  # Skip cd commands, we'll handle directory
                            
                            result = subprocess.run(
                                cmd.split(),
                                cwd="frontend",
                                capture_output=True,
                                text=True,
                                timeout=300
                            )
                            if result.returncode == 0:
                                fixes_applied.append(f"Fixed {issue.component}: {issue.issue_type}")
                                issue.status = ValidationStatus.FIXED
                
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                    # Fix attempt failed, leave issue as is
                    pass
        
        # Handle configuration fixes
        config_fixes = self.configuration_validator.auto_repair_config(config_issues)
        fixes_applied.extend(config_fixes)
        
        return fixes_applied
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information for debugging and logging."""
        import platform
        import os
        
        info = {
            "platform": platform.platform(),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "working_directory": str(Path.cwd()),
            "environment_variables": {
                "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV"),
                "PATH": os.environ.get("PATH", "")[:200] + "..." if len(os.environ.get("PATH", "")) > 200 else os.environ.get("PATH", ""),
            }
        }
        
        # Add Node.js info if available
        try:
            node_result = subprocess.run(['node', '--version'], capture_output=True, text=True, timeout=5)
            if node_result.returncode == 0:
                info["node_version"] = node_result.stdout.strip()
        except:
            info["node_version"] = "Not available"
        
        try:
            npm_result = subprocess.run(['npm', '--version'], capture_output=True, text=True, timeout=5)
            if npm_result.returncode == 0:
                info["npm_version"] = npm_result.stdout.strip()
        except:
            info["npm_version"] = "Not available"
        
        return info