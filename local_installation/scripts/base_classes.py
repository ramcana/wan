"""
Base classes providing common functionality for the installation system.
These classes implement shared behavior and utilities used across components.
"""

import os
import json
import logging
import subprocess
from typing import Dict, Any, Optional, List
from pathlib import Path

from interfaces import (
    InstallationError, ErrorCategory, InstallationState, 
    InstallationPhase, IProgressReporter, IErrorHandler
)


class BaseInstallationComponent:
    """Base class for all installation components."""
    
    def __init__(self, installation_path: str, logger: Optional[logging.Logger] = None):
        self.installation_path = Path(installation_path)
        self.logger = logger or self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up component logger."""
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def ensure_directory(self, path: Path) -> None:
        """Ensure directory exists, create if necessary."""
        try:
            path.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Ensured directory exists: {path}")
        except Exception as e:
            raise InstallationError(
                f"Failed to create directory {path}: {str(e)}",
                ErrorCategory.SYSTEM,
                ["Check file permissions", "Ensure sufficient disk space"]
            )
    
    def run_command(self, command: List[str], cwd: Optional[Path] = None, 
                   capture_output: bool = True) -> subprocess.CompletedProcess:
        """Run system command with error handling."""
        try:
            self.logger.debug(f"Running command: {' '.join(command)}")
            result = subprocess.run(
                command,
                cwd=cwd or self.installation_path,
                capture_output=capture_output,
                text=True,
                check=True
            )
            self.logger.debug(f"Command completed successfully")
            return result
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {e}")
            raise InstallationError(
                f"Command failed: {' '.join(command)}\nError: {e.stderr}",
                ErrorCategory.SYSTEM,
                ["Check system requirements", "Verify file permissions"]
            )
        except FileNotFoundError as e:
            raise InstallationError(
                f"Command not found: {command[0]}",
                ErrorCategory.SYSTEM,
                ["Install required software", "Check PATH environment variable"]
            )
    
    def load_json_file(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON file with error handling."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise InstallationError(
                f"Configuration file not found: {file_path}",
                ErrorCategory.CONFIGURATION,
                ["Check file path", "Restore missing configuration files"]
            )
        except json.JSONDecodeError as e:
            raise InstallationError(
                f"Invalid JSON in file {file_path}: {str(e)}",
                ErrorCategory.CONFIGURATION,
                ["Check file syntax", "Restore from backup"]
            )
    
    def save_json_file(self, data: Dict[str, Any], file_path: Path) -> None:
        """Save data to JSON file with error handling."""
        try:
            self.ensure_directory(file_path.parent)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.logger.debug(f"Saved JSON file: {file_path}")
        except Exception as e:
            raise InstallationError(
                f"Failed to save file {file_path}: {str(e)}",
                ErrorCategory.SYSTEM,
                ["Check file permissions", "Ensure sufficient disk space"]
            )


class ConsoleProgressReporter(IProgressReporter):
    """Console-based progress reporter for batch file interface."""
    
    def __init__(self):
        self.current_phase = None
        self.last_progress = 0.0
    
    def update_progress(self, phase: InstallationPhase, progress: float, 
                       task: str) -> None:
        """Update and display installation progress."""
        if phase != self.current_phase:
            self.current_phase = phase
            print(f"\n=== {phase.value.upper()} PHASE ===")
        
        # Show progress bar
        bar_length = 40
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        percentage = int(progress * 100)
        
        print(f"\r[{bar}] {percentage}% - {task}", end='', flush=True)
        
        if progress >= 1.0:
            print()  # New line when phase completes
    
    def report_error(self, error: InstallationError) -> None:
        """Report error to console."""
        print(f"\n❌ ERROR: {error.message}")
        if error.recovery_suggestions:
            print("💡 Suggestions:")
            for suggestion in error.recovery_suggestions:
                print(f"   • {suggestion}")
    
    def report_warning(self, message: str) -> None:
        """Report warning to console."""
        print(f"\n⚠️  WARNING: {message}")
    
    def report_success(self, message: str) -> None:
        """Report success to console."""
        print(f"\n✅ {message}")


class DefaultErrorHandler(IErrorHandler):
    """Default error handler with logging and recovery suggestions."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def handle_error(self, error: InstallationError) -> str:
        """Handle installation error and return recovery action."""
        self.log_error(error, {})
        
        # Determine recovery action based on error category
        if error.category == ErrorCategory.NETWORK:
            return "retry"
        elif error.category == ErrorCategory.PERMISSION:
            return "elevate"
        elif error.category == ErrorCategory.SYSTEM:
            return "abort"
        else:
            return "continue"
    
    def log_error(self, error: InstallationError, context: Dict[str, Any]) -> None:
        """Log error with context information."""
        self.logger.error(f"Installation error: {error.message}")
        self.logger.error(f"Category: {error.category.value}")
        if context:
            self.logger.error(f"Context: {context}")
        if error.recovery_suggestions:
            self.logger.error(f"Recovery suggestions: {error.recovery_suggestions}")
    
    def suggest_recovery(self, error: InstallationError) -> List[str]:
        """Suggest recovery actions for the error."""
        base_suggestions = error.recovery_suggestions.copy() if error.recovery_suggestions else []
        
        # Add category-specific suggestions
        if error.category == ErrorCategory.NETWORK:
            base_suggestions.extend([
                "Check internet connection",
                "Try again later",
                "Use alternative download source"
            ])
        elif error.category == ErrorCategory.PERMISSION:
            base_suggestions.extend([
                "Run as administrator",
                "Check file permissions",
                "Close other applications using the files"
            ])
        elif error.category == ErrorCategory.SYSTEM:
            base_suggestions.extend([
                "Check system requirements",
                "Free up disk space",
                "Update system drivers"
            ])
        
        return base_suggestions


class InstallationStateManager:
    """Manages installation state and persistence."""
    
    def __init__(self, installation_path: str):
        self.installation_path = Path(installation_path)
        self.state_file = self.installation_path / "logs" / "installation_state.json"
        self.current_state = InstallationState(
            phase=InstallationPhase.DETECTION,
            progress=0.0,
            current_task="Initializing",
            errors=[],
            warnings=[],
            hardware_profile=None,
            installation_path=str(installation_path)
        )
    
    def save_state(self) -> None:
        """Save current installation state to file."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            state_data = {
                "phase": self.current_state.phase.value,
                "progress": self.current_state.progress,
                "current_task": self.current_state.current_task,
                "errors": self.current_state.errors,
                "warnings": self.current_state.warnings,
                "installation_path": self.current_state.installation_path
            }
            
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2)
        except Exception as e:
            # Don't fail installation if state saving fails
            logging.warning(f"Failed to save installation state: {e}")
    
    def load_state(self) -> Optional[InstallationState]:
        """Load installation state from file."""
        try:
            if not self.state_file.exists():
                return None
            
            with open(self.state_file, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            
            self.current_state.phase = InstallationPhase(state_data["phase"])
            self.current_state.progress = state_data["progress"]
            self.current_state.current_task = state_data["current_task"]
            self.current_state.errors = state_data["errors"]
            self.current_state.warnings = state_data["warnings"]
            
            return self.current_state
        except Exception as e:
            logging.warning(f"Failed to load installation state: {e}")
            return None
    
    def update_state(self, phase: Optional[InstallationPhase] = None,
                    progress: Optional[float] = None,
                    task: Optional[str] = None,
                    error: Optional[str] = None,
                    warning: Optional[str] = None) -> None:
        """Update installation state."""
        if phase is not None:
            self.current_state.phase = phase
        if progress is not None:
            self.current_state.progress = progress
        if task is not None:
            self.current_state.current_task = task
        if error is not None:
            self.current_state.errors.append(error)
        if warning is not None:
            self.current_state.warnings.append(warning)
        
        self.save_state()
    
    def clear_state(self) -> None:
        """Clear installation state file."""
        try:
            if self.state_file.exists():
                self.state_file.unlink()
        except Exception as e:
            logging.warning(f"Failed to clear installation state: {e}")


class ConfigurationTemplate:
    """Template system for generating configurations."""
    
    @staticmethod
    def get_high_performance_template() -> Dict[str, Any]:
        """Configuration template for high-performance systems."""
        return {
            "system": {
                "default_quantization": "bf16",
                "enable_offload": False,
                "vae_tile_size": 512,
                "max_queue_size": 20,
                "worker_threads": 32
            },
            "optimization": {
                "max_vram_usage_gb": 14,
                "cpu_threads": 64,
                "memory_pool_gb": 32
            },
            "models": {
                "cache_models": True,
                "preload_models": True,
                "model_precision": "fp16"
            }
        }
    
    @staticmethod
    def get_mid_range_template() -> Dict[str, Any]:
        """Configuration template for mid-range systems."""
        return {
            "system": {
                "default_quantization": "fp16",
                "enable_offload": True,
                "vae_tile_size": 256,
                "max_queue_size": 10,
                "worker_threads": 8
            },
            "optimization": {
                "max_vram_usage_gb": 8,
                "cpu_threads": 16,
                "memory_pool_gb": 8
            },
            "models": {
                "cache_models": False,
                "preload_models": False,
                "model_precision": "fp16"
            }
        }
    
    @staticmethod
    def get_budget_template() -> Dict[str, Any]:
        """Configuration template for budget systems."""
        return {
            "system": {
                "default_quantization": "int8",
                "enable_offload": True,
                "vae_tile_size": 128,
                "max_queue_size": 5,
                "worker_threads": 4
            },
            "optimization": {
                "max_vram_usage_gb": 4,
                "cpu_threads": 8,
                "memory_pool_gb": 4
            },
            "models": {
                "cache_models": False,
                "preload_models": False,
                "model_precision": "int8"
            }
        }