"""
Core utility functions for system detection and path management.
"""

import os
import sys
import platform
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import shutil


class SystemDetector:
    """Utility class for detecting system information and capabilities."""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get comprehensive system information."""
        return {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.architecture()[0],
            "processor": platform.processor(),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "working_directory": os.getcwd(),
            "user": os.getenv("USERNAME") or os.getenv("USER", "unknown"),
            "is_admin": SystemDetector.is_admin(),
            "virtual_env": SystemDetector.get_virtual_env_info()
        }
    
    @staticmethod
    def is_windows() -> bool:
        """Check if running on Windows."""
        return platform.system().lower() == "windows"
    
    @staticmethod
    def is_admin() -> bool:
        """Check if running with administrator privileges."""
        if SystemDetector.is_windows():
            try:
                import ctypes
                return ctypes.windll.shell32.IsUserAnAdmin() != 0
            except Exception:
                return False
        else:
            return os.geteuid() == 0
    
    @staticmethod
    def get_virtual_env_info() -> Dict[str, Any]:
        """Get information about the current virtual environment."""
        venv_info = {
            "active": False,
            "path": None,
            "type": None
        }
        
        # Check for various virtual environment indicators
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            venv_info["active"] = True
            venv_info["path"] = sys.prefix
            
            # Determine virtual environment type
            if os.path.exists(os.path.join(sys.prefix, 'pyvenv.cfg')):
                venv_info["type"] = "venv"
            elif 'conda' in sys.prefix.lower():
                venv_info["type"] = "conda"
            elif 'virtualenv' in sys.prefix.lower():
                venv_info["type"] = "virtualenv"
            else:
                venv_info["type"] = "unknown"
        
        return venv_info
    
    @staticmethod
    def check_python_version(min_version: Tuple[int, int] = (3, 8)) -> Dict[str, Any]:
        """Check if Python version meets minimum requirements."""
        current_version = sys.version_info[:2]
        meets_requirement = current_version >= min_version
        
        return {
            "current_version": f"{current_version[0]}.{current_version[1]}",
            "required_version": f"{min_version[0]}.{min_version[1]}",
            "meets_requirement": meets_requirement,
            "version_info": sys.version_info
        }
    
    @staticmethod
    def check_command_available(command: str) -> Dict[str, Any]:
        """Check if a command is available in the system PATH."""
        result = {
            "available": False,
            "path": None,
            "version": None
        }
        
        # Check if command exists
        command_path = shutil.which(command)
        if command_path:
            result["available"] = True
            result["path"] = command_path
            
            # Try to get version
            try:
                version_result = subprocess.run(
                    [command, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if version_result.returncode == 0:
                    result["version"] = version_result.stdout.strip()
            except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError, OSError):
                pass
        
        return result
    
    @staticmethod
    def check_node_environment() -> Dict[str, Any]:
        """Check Node.js and npm availability and versions."""
        node_info = SystemDetector.check_command_available("node")
        npm_info = SystemDetector.check_command_available("npm")
        
        return {
            "node": node_info,
            "npm": npm_info,
            "environment_ready": node_info["available"] and npm_info["available"]
        }


class PathManager:
    """Utility class for managing project paths and file operations."""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self._ensure_project_root()
    
    def _ensure_project_root(self) -> None:
        """Ensure we're working from the correct project root."""
        # If project_root was explicitly set, don't search for it
        if self.project_root != Path.cwd():
            return
            
        # Look for key project files to confirm we're in the right directory
        key_files = ["backend", "frontend", "README.md"]
        
        current_path = self.project_root
        for _ in range(3):  # Search up to 3 levels up
            if all((current_path / file).exists() for file in key_files):
                self.project_root = current_path
                return
            current_path = current_path.parent
        
        # If we can't find the project root, use current directory
        self.project_root = Path.cwd()
    
    def get_backend_path(self) -> Path:
        """Get path to backend directory."""
        return self.project_root / "backend"
    
    def get_frontend_path(self) -> Path:
        """Get path to frontend directory."""
        return self.project_root / "frontend"
    
    def get_scripts_path(self) -> Path:
        """Get path to scripts directory."""
        return self.project_root / "scripts"
    
    def get_logs_path(self) -> Path:
        """Get path to logs directory, creating if necessary."""
        logs_path = self.project_root / "logs"
        logs_path.mkdir(exist_ok=True)
        return logs_path
    
    def get_config_path(self, config_name: str = "startup_config.json") -> Path:
        """Get path to configuration file."""
        return self.project_root / config_name
    
    def validate_project_structure(self) -> Dict[str, Any]:
        """Validate that required project directories and files exist."""
        validation_result = {
            "valid": True,
            "missing_directories": [],
            "missing_files": [],
            "warnings": []
        }
        
        # Required directories
        required_dirs = [
            self.get_backend_path(),
            self.get_frontend_path()
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                validation_result["valid"] = False
                validation_result["missing_directories"].append(str(dir_path))
        
        # Required files
        required_files = [
            self.get_backend_path() / "main.py",
            self.get_backend_path() / "requirements.txt",
            self.get_frontend_path() / "package.json"
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                validation_result["valid"] = False
                validation_result["missing_files"].append(str(file_path))
        
        # Optional but recommended files
        recommended_files = [
            self.project_root / "README.md",
            self.get_backend_path() / "config.json"
        ]
        
        for file_path in recommended_files:
            if not file_path.exists():
                validation_result["warnings"].append(f"Recommended file missing: {file_path}")
        
        return validation_result
    
    def create_directory_structure(self) -> List[str]:
        """Create missing directories for the project."""
        created_dirs = []
        
        directories_to_create = [
            self.get_logs_path(),
            self.get_scripts_path(),
            self.get_scripts_path() / "startup_manager"
        ]
        
        for dir_path in directories_to_create:
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                created_dirs.append(str(dir_path))
        
        return created_dirs
    
    def get_relative_path(self, path: Path) -> str:
        """Get relative path from project root."""
        try:
            return str(path.relative_to(self.project_root))
        except ValueError:
            return str(path)
    
    def resolve_path(self, path_str: str) -> Path:
        """Resolve a path string relative to project root."""
        path = Path(path_str)
        if path.is_absolute():
            return path
        return self.project_root / path
