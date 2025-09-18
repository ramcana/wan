"""
Recovery Engine for Server Startup Management System

This module provides error classification, recovery strategies, and intelligent
failure handling for the WAN22 server startup process.
"""

import time
import logging
import traceback
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Union
from pathlib import Path
import json
import re
import subprocess
import psutil
import socket


class ErrorType(Enum):
    """Classification of different error types that can occur during startup"""
    PORT_CONFLICT = "port_conflict"
    PERMISSION_DENIED = "permission_denied"
    DEPENDENCY_MISSING = "dependency_missing"
    CONFIG_INVALID = "config_invalid"
    PROCESS_FAILED = "process_failed"
    NETWORK_ERROR = "network_error"
    FIREWALL_BLOCKED = "firewall_blocked"
    VIRTUAL_ENV_ERROR = "virtual_env_error"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN = "unknown"


@dataclass
class RecoveryAction:
    """Represents a specific recovery action that can be taken"""
    name: str
    description: str
    action_func: Callable
    priority: int = 1  # Lower number = higher priority
    success_rate: float = 0.0  # Track success rate for learning
    auto_executable: bool = True
    requires_user_input: bool = False
    estimated_time: float = 5.0  # Estimated time in seconds


@dataclass
class StartupError:
    """Represents an error that occurred during startup"""
    type: ErrorType
    message: str
    details: Optional[Dict[str, Any]] = None
    suggested_actions: List[str] = field(default_factory=list)
    auto_fixable: bool = False
    original_exception: Optional[Exception] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class RecoveryResult:
    """Result of a recovery attempt"""
    success: bool
    action_taken: str
    message: str
    details: Optional[Dict[str, Any]] = None
    retry_recommended: bool = False
    fallback_needed: bool = False


class ErrorPatternMatcher:
    """Matches error messages to specific error types using patterns"""
    
    def __init__(self):
        self.patterns = {
            ErrorType.PERMISSION_DENIED: [
                r"WinError 10013",
                r"Permission denied",
                r"Access is denied",
                r"PermissionError",
                r"OSError.*\[Errno 13\]",
                r"socket.*permission denied"
            ],
            ErrorType.PORT_CONFLICT: [
                r"Address already in use",
                r"WinError 10048",
                r"OSError.*\[Errno 98\]",
                r"port.*already.*use",
                r"bind.*failed.*address.*use"
            ],
            ErrorType.DEPENDENCY_MISSING: [
                r"ModuleNotFoundError",
                r"ImportError",
                r"No module named",
                r"command not found",
                r"is not recognized as an internal or external command"
            ],
            ErrorType.CONFIG_INVALID: [
                r"JSONDecodeError",
                r"Invalid configuration",
                r"Config.*not found",
                r"KeyError.*config",
                r"Configuration.*missing"
            ],
            ErrorType.FIREWALL_BLOCKED: [
                r"WinError 10060",
                r"Connection timed out",
                r"No connection could be made",
                r"firewall.*block",
                r"Windows Defender.*block"
            ],
            ErrorType.VIRTUAL_ENV_ERROR: [
                r"virtual environment",
                r"venv.*not.*activated",
                r"pip.*not found",
                r"python.*not found.*venv"
            ],
            ErrorType.TIMEOUT_ERROR: [
                r"TimeoutError",
                r"timeout.*exceeded",
                r"Connection timeout",
                r"Read timeout"
            ],
            ErrorType.PROCESS_FAILED: [
                r"Process.*failed",
                r"subprocess.*error",
                r"CalledProcessError",
                r"Exit code.*non-zero"
            ]
        }
    
    def classify_error(self, error_message: str, exception: Optional[Exception] = None) -> ErrorType:
        """Classify an error based on its message and exception type"""
        error_text = str(error_message).lower()
        
        # Check exception type first for more accurate classification
        if exception:
            if isinstance(exception, PermissionError):
                return ErrorType.PERMISSION_DENIED
            elif type(exception).__name__ == 'TimeoutError':
                return ErrorType.TIMEOUT_ERROR
            elif isinstance(exception, OSError) and hasattr(exception, 'errno'):
                if exception.errno == 13:  # Permission denied
                    return ErrorType.PERMISSION_DENIED
                elif exception.errno == 98 or exception.errno == 10048:  # Address in use
                    return ErrorType.PORT_CONFLICT
                elif exception.errno == 10060:  # Connection timeout
                    return ErrorType.FIREWALL_BLOCKED
            elif isinstance(exception, (ImportError, ModuleNotFoundError)):
                return ErrorType.DEPENDENCY_MISSING
            elif isinstance(exception, (json.JSONDecodeError, KeyError)):
                return ErrorType.CONFIG_INVALID
            elif isinstance(exception, subprocess.CalledProcessError):
                return ErrorType.PROCESS_FAILED
        
        # Pattern matching fallback
        for error_type, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, error_text, re.IGNORECASE):
                    return error_type
        
        return ErrorType.UNKNOWN


class RetryStrategy:
    """Implements exponential backoff retry logic"""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.logger = logging.getLogger(__name__)
    
    def execute_with_retry(self, operation: Callable, operation_name: str, 
                          *args, **kwargs) -> Any:
        """Execute an operation with exponential backoff retry"""
        last_exception = None
        
        for attempt in range(1, self.max_attempts + 1):
            try:
                self.logger.info(f"Attempting {operation_name} (attempt {attempt}/{self.max_attempts})")
                result = operation(*args, **kwargs)
                self.logger.info(f"{operation_name} succeeded on attempt {attempt}")
                return result
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"{operation_name} failed on attempt {attempt}: {str(e)}")
                
                if attempt < self.max_attempts:
                    delay = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
                    self.logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                else:
                    self.logger.error(f"{operation_name} failed after {self.max_attempts} attempts")
        
        # If we get here, all attempts failed
        raise last_exception


class RecoveryEngine:
    """Main recovery engine that handles error classification and recovery strategies"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.pattern_matcher = ErrorPatternMatcher()
        self.retry_strategy = RetryStrategy()
        
        # Recovery action registry
        self.recovery_actions: Dict[ErrorType, List[RecoveryAction]] = {}
        self.success_history: Dict[str, List[bool]] = {}
        
        # Configuration
        self.config_path = config_path or Path("recovery_config.json")
        self.load_configuration()
        
        # Initialize recovery actions
        self._register_recovery_actions()
    
    def load_configuration(self):
        """Load recovery engine configuration"""
        default_config = {
            "max_retry_attempts": 3,
            "base_retry_delay": 1.0,
            "max_retry_delay": 60.0,
            "auto_fix_enabled": True,
            "learning_enabled": True,
            "success_rate_threshold": 0.7
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.config = {**default_config, **config}
            except Exception as e:
                self.logger.warning(f"Failed to load recovery config: {e}, using defaults")
                self.config = default_config
        else:
            self.config = default_config
            self.save_configuration()
    
    def save_configuration(self):
        """Save current configuration"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save recovery config: {e}")
    
    def classify_error(self, error_message: str, exception: Optional[Exception] = None,
                      context: Optional[Dict[str, Any]] = None) -> StartupError:
        """Classify an error and create a StartupError object"""
        error_type = self.pattern_matcher.classify_error(error_message, exception)
        
        # Create detailed error information
        startup_error = StartupError(
            type=error_type,
            message=error_message,
            original_exception=exception,
            context=context or {}
        )
        
        # Add suggested actions based on error type
        startup_error.suggested_actions = self._get_suggested_actions(error_type)
        startup_error.auto_fixable = self._is_auto_fixable(error_type)
        
        # Add specific details based on error type
        startup_error.details = self._get_error_details(error_type, exception, context)
        
        return startup_error
    
    def _get_suggested_actions(self, error_type: ErrorType) -> List[str]:
        """Get human-readable suggested actions for an error type"""
        suggestions = {
            ErrorType.PERMISSION_DENIED: [
                "Run as administrator",
                "Add firewall exception for Python/Node.js",
                "Try different port range (8080-8090)",
                "Check Windows Defender settings"
            ],
            ErrorType.PORT_CONFLICT: [
                "Kill existing process using the port",
                "Use different port automatically",
                "Check for zombie processes",
                "Restart network services"
            ],
            ErrorType.DEPENDENCY_MISSING: [
                "Install missing dependencies",
                "Activate virtual environment",
                "Check Python/Node.js installation",
                "Update package managers"
            ],
            ErrorType.CONFIG_INVALID: [
                "Repair configuration file",
                "Reset to default configuration",
                "Validate JSON syntax",
                "Check file permissions"
            ],
            ErrorType.FIREWALL_BLOCKED: [
                "Add firewall exception",
                "Temporarily disable firewall",
                "Check Windows Defender settings",
                "Try different network interface"
            ],
            ErrorType.VIRTUAL_ENV_ERROR: [
                "Activate virtual environment",
                "Create new virtual environment",
                "Check Python installation",
                "Update pip and setuptools"
            ],
            ErrorType.TIMEOUT_ERROR: [
                "Increase timeout values",
                "Check network connectivity",
                "Restart network services",
                "Try different ports"
            ],
            ErrorType.PROCESS_FAILED: [
                "Check process logs",
                "Verify executable permissions",
                "Check system resources",
                "Restart with different parameters"
            ]
        }
        
        return suggestions.get(error_type, ["Check logs for more details", "Try manual restart"])
    
    def _is_auto_fixable(self, error_type: ErrorType) -> bool:
        """Determine if an error type can be automatically fixed"""
        auto_fixable_types = {
            ErrorType.PORT_CONFLICT,
            ErrorType.CONFIG_INVALID,
            ErrorType.VIRTUAL_ENV_ERROR,
            ErrorType.DEPENDENCY_MISSING
        }
        return error_type in auto_fixable_types
    
    def _get_error_details(self, error_type: ErrorType, exception: Optional[Exception],
                          context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Get detailed information about the error"""
        details = {
            "error_type": error_type.value,
            "timestamp": time.time(),
            "context": context or {}
        }
        
        if exception:
            details["exception_type"] = type(exception).__name__
            details["traceback"] = traceback.format_exc()
        
        # Add type-specific details
        if error_type == ErrorType.PORT_CONFLICT and context:
            details["conflicting_port"] = context.get("port")
            details["process_using_port"] = self._get_process_using_port(context.get("port"))
        
        return details
    
    def _get_process_using_port(self, port: int) -> Optional[Dict[str, Any]]:
        """Get information about the process using a specific port"""
        try:
            for conn in psutil.net_connections():
                if conn.laddr.port == port:
                    try:
                        process = psutil.Process(conn.pid)
                        return {
                            "pid": conn.pid,
                            "name": process.name(),
                            "cmdline": process.cmdline()
                        }
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        return {"pid": conn.pid, "name": "Unknown", "cmdline": []}
        except Exception as e:
            self.logger.warning(f"Failed to get process info for port {port}: {e}")
        
        return None
    
    def _register_recovery_actions(self):
        """Register all available recovery actions"""
        
        # Port conflict recovery actions
        self.register_recovery_action(
            ErrorType.PORT_CONFLICT,
            RecoveryAction(
                name="kill_process_on_port",
                description="Kill process using the conflicted port",
                action_func=self._kill_process_on_port,
                priority=1,
                auto_executable=True,
                requires_user_input=False,
                estimated_time=3.0
            )
        )
        
        self.register_recovery_action(
            ErrorType.PORT_CONFLICT,
            RecoveryAction(
                name="find_alternative_port",
                description="Find and use alternative port",
                action_func=self._find_alternative_port,
                priority=2,
                auto_executable=True,
                requires_user_input=False,
                estimated_time=2.0
            )
        )
        
        # Permission denied recovery actions
        self.register_recovery_action(
            ErrorType.PERMISSION_DENIED,
            RecoveryAction(
                name="try_alternative_ports",
                description="Try ports in safe range (8080-8090)",
                action_func=self._try_alternative_ports,
                priority=1,
                auto_executable=True,
                requires_user_input=False,
                estimated_time=5.0
            )
        )
        
        self.register_recovery_action(
            ErrorType.PERMISSION_DENIED,
            RecoveryAction(
                name="check_firewall_exceptions",
                description="Check and suggest firewall exceptions",
                action_func=self._check_firewall_exceptions,
                priority=2,
                auto_executable=False,
                requires_user_input=True,
                estimated_time=10.0
            )
        )
        
        # Dependency missing recovery actions
        self.register_recovery_action(
            ErrorType.DEPENDENCY_MISSING,
            RecoveryAction(
                name="install_missing_dependencies",
                description="Install missing Python/Node.js dependencies",
                action_func=self._install_missing_dependencies,
                priority=1,
                auto_executable=True,
                requires_user_input=False,
                estimated_time=30.0
            )
        )
        
        self.register_recovery_action(
            ErrorType.DEPENDENCY_MISSING,
            RecoveryAction(
                name="activate_virtual_environment",
                description="Activate Python virtual environment",
                action_func=self._activate_virtual_environment,
                priority=2,
                auto_executable=True,
                requires_user_input=False,
                estimated_time=5.0
            )
        )
        
        # Configuration invalid recovery actions
        self.register_recovery_action(
            ErrorType.CONFIG_INVALID,
            RecoveryAction(
                name="repair_config_file",
                description="Repair or recreate configuration file",
                action_func=self._repair_config_file,
                priority=1,
                auto_executable=True,
                requires_user_input=False,
                estimated_time=3.0
            )
        )
        
        # Virtual environment error recovery actions
        self.register_recovery_action(
            ErrorType.VIRTUAL_ENV_ERROR,
            RecoveryAction(
                name="create_virtual_environment",
                description="Create new virtual environment",
                action_func=self._create_virtual_environment,
                priority=1,
                auto_executable=True,
                requires_user_input=False,
                estimated_time=20.0
            )
        )
        
        # Firewall blocked recovery actions
        self.register_recovery_action(
            ErrorType.FIREWALL_BLOCKED,
            RecoveryAction(
                name="suggest_firewall_exception",
                description="Provide firewall exception instructions",
                action_func=self._suggest_firewall_exception,
                priority=1,
                auto_executable=False,
                requires_user_input=True,
                estimated_time=15.0
            )
        )
        
        # Process failed recovery actions
        self.register_recovery_action(
            ErrorType.PROCESS_FAILED,
            RecoveryAction(
                name="restart_with_different_params",
                description="Restart process with different parameters",
                action_func=self._restart_with_different_params,
                priority=1,
                auto_executable=True,
                requires_user_input=False,
                estimated_time=10.0
            )
        )
        
        # Timeout error recovery actions
        self.register_recovery_action(
            ErrorType.TIMEOUT_ERROR,
            RecoveryAction(
                name="increase_timeout",
                description="Increase timeout values and retry",
                action_func=self._increase_timeout,
                priority=1,
                auto_executable=True,
                requires_user_input=False,
                estimated_time=5.0
            )
        )
    
    def register_recovery_action(self, error_type: ErrorType, action: RecoveryAction):
        """Register a recovery action for a specific error type"""
        if error_type not in self.recovery_actions:
            self.recovery_actions[error_type] = []
        
        self.recovery_actions[error_type].append(action)
        
        # Sort by priority (lower number = higher priority)
        self.recovery_actions[error_type].sort(key=lambda x: x.priority)
    
    def get_recovery_actions(self, error_type: ErrorType) -> List[RecoveryAction]:
        """Get available recovery actions for an error type"""
        actions = self.recovery_actions.get(error_type, [])
        
        # Sort by success rate if learning is enabled
        if self.config.get("learning_enabled", True):
            actions = sorted(actions, key=lambda x: x.success_rate, reverse=True)
        
        return actions
    
    def attempt_recovery(self, error: StartupError, context: Optional[Dict[str, Any]] = None) -> RecoveryResult:
        """Attempt to recover from an error"""
        self.logger.info(f"Attempting recovery for error type: {error.type.value}")
        
        actions = self.get_recovery_actions(error.type)
        
        if not actions:
            return RecoveryResult(
                success=False,
                action_taken="none",
                message=f"No recovery actions available for error type: {error.type.value}",
                fallback_needed=True
            )
        
        # Try each recovery action in order of priority/success rate
        for action in actions:
            if not self.config.get("auto_fix_enabled", True) and action.auto_executable:
                continue
            
            try:
                self.logger.info(f"Trying recovery action: {action.name}")
                
                # Execute the recovery action
                result = action.action_func(error, context or {})
                
                # Update success rate
                self._update_success_rate(action.name, result.success)
                
                if result.success:
                    self.logger.info(f"Recovery action '{action.name}' succeeded")
                    return result
                else:
                    self.logger.warning(f"Recovery action '{action.name}' failed: {result.message}")
                    
            except Exception as e:
                self.logger.error(f"Recovery action '{action.name}' raised exception: {e}")
                self._update_success_rate(action.name, False)
        
        # If we get here, all recovery actions failed
        return RecoveryResult(
            success=False,
            action_taken="all_failed",
            message="All recovery actions failed",
            fallback_needed=True
        )
    
    def _update_success_rate(self, action_name: str, success: bool):
        """Update success rate for a recovery action"""
        if action_name not in self.success_history:
            self.success_history[action_name] = []
        
        self.success_history[action_name].append(success)
        
        # Keep only last 20 attempts for rolling average
        if len(self.success_history[action_name]) > 20:
            self.success_history[action_name] = self.success_history[action_name][-20:]
        
        # Update success rate in the action
        success_rate = sum(self.success_history[action_name]) / len(self.success_history[action_name])
        
        # Find and update the action
        for error_type, actions in self.recovery_actions.items():
            for action in actions:
                if action.name == action_name:
                    action.success_rate = success_rate
                    break
    
    # Recovery action implementations
    def _kill_process_on_port(self, error: StartupError, context: Dict[str, Any]) -> RecoveryResult:
        """Kill process using a specific port"""
        port = context.get("port") or error.details.get("conflicting_port")
        
        if not port:
            return RecoveryResult(
                success=False,
                action_taken="kill_process_on_port",
                message="No port specified for process killing"
            )
        
        try:
            killed_processes = []
            
            for conn in psutil.net_connections():
                if conn.laddr.port == port:
                    try:
                        process = psutil.Process(conn.pid)
                        process_name = process.name()
                        process.terminate()
                        
                        # Wait for process to terminate
                        try:
                            process.wait(timeout=5)
                            killed_processes.append(f"{process_name} (PID: {conn.pid})")
                        except psutil.TimeoutExpired:
                            # Force kill if terminate didn't work
                            process.kill()
                            killed_processes.append(f"{process_name} (PID: {conn.pid}) - force killed")
                            
                    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                        self.logger.warning(f"Could not kill process {conn.pid}: {e}")
            
            if killed_processes:
                return RecoveryResult(
                    success=True,
                    action_taken="kill_process_on_port",
                    message=f"Killed processes on port {port}: {', '.join(killed_processes)}",
                    details={"killed_processes": killed_processes, "port": port},
                    retry_recommended=True
                )
            else:
                return RecoveryResult(
                    success=False,
                    action_taken="kill_process_on_port",
                    message=f"No processes found using port {port}"
                )
                
        except Exception as e:
            return RecoveryResult(
                success=False,
                action_taken="kill_process_on_port",
                message=f"Failed to kill process on port {port}: {str(e)}"
            )
    
    def _find_alternative_port(self, error: StartupError, context: Dict[str, Any]) -> RecoveryResult:
        """Find an alternative available port"""
        original_port = context.get("port") or error.details.get("conflicting_port", 8000)
        
        # Try ports in a reasonable range
        port_ranges = [
            range(original_port + 1, original_port + 10),  # Try next 9 ports
            range(8080, 8090),  # Common alternative range
            range(3001, 3010),  # Frontend alternative range
            range(8000, 9010)   # Another common range
        ]
        
        for port_range in port_ranges:
            for port in port_range:
                if self._is_port_available(port):
                    return RecoveryResult(
                        success=True,
                        action_taken="find_alternative_port",
                        message=f"Found alternative port: {port}",
                        details={"alternative_port": port, "original_port": original_port},
                        retry_recommended=True
                    )
        
        return RecoveryResult(
            success=False,
            action_taken="find_alternative_port",
            message="No alternative ports available in tested ranges"
        )
    
    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(('localhost', port))
                return True
        except OSError:
            return False
    
    def _try_alternative_ports(self, error: StartupError, context: Dict[str, Any]) -> RecoveryResult:
        """Try ports in safe range for permission issues"""
        safe_ports = [8080, 8081, 8082, 8083, 8084, 8085, 8086, 8087, 8088, 8089]
        
        for port in safe_ports:
            if self._is_port_available(port):
                return RecoveryResult(
                    success=True,
                    action_taken="try_alternative_ports",
                    message=f"Found safe alternative port: {port}",
                    details={"safe_port": port},
                    retry_recommended=True
                )
        
        return RecoveryResult(
            success=False,
            action_taken="try_alternative_ports",
            message="No safe alternative ports available"
        )
    
    def _check_firewall_exceptions(self, error: StartupError, context: Dict[str, Any]) -> RecoveryResult:
        """Check firewall exceptions and provide guidance"""
        python_exe = context.get("python_exe", "python.exe")
        node_exe = context.get("node_exe", "node.exe")
        
        instructions = [
            "To add firewall exceptions:",
            "1. Open Windows Defender Firewall with Advanced Security",
            "2. Click 'Inbound Rules' -> 'New Rule'",
            "3. Select 'Program' -> Browse to your Python/Node.js executable",
            f"   - Python: {python_exe}",
            f"   - Node.js: {node_exe}",
            "4. Allow the connection for all profiles",
            "5. Restart the startup process"
        ]
        
        return RecoveryResult(
            success=False,  # Requires manual action
            action_taken="check_firewall_exceptions",
            message="Firewall exception instructions provided",
            details={"instructions": instructions},
            retry_recommended=True
        )
    
    def _install_missing_dependencies(self, error: StartupError, context: Dict[str, Any]) -> RecoveryResult:
        """Install missing dependencies"""
        try:
            results = []
            
            # Try to install Python dependencies
            if context.get("missing_python_deps"):
                try:
                    result = subprocess.run(
                        ["pip", "install", "-r", "requirements.txt"],
                        capture_output=True,
                        text=True,
                        timeout=120
                    )
                    if result.returncode == 0:
                        results.append("Python dependencies installed successfully")
                    else:
                        results.append(f"Python dependency installation failed: {result.stderr}")
                except subprocess.TimeoutExpired:
                    results.append("Python dependency installation timed out")
                except Exception as e:
                    results.append(f"Python dependency installation error: {str(e)}")
            
            # Try to install Node.js dependencies
            if context.get("missing_node_deps"):
                try:
                    result = subprocess.run(
                        ["npm", "install"],
                        capture_output=True,
                        text=True,
                        timeout=180,
                        cwd=context.get("frontend_path", "frontend")
                    )
                    if result.returncode == 0:
                        results.append("Node.js dependencies installed successfully")
                    else:
                        results.append(f"Node.js dependency installation failed: {result.stderr}")
                except subprocess.TimeoutExpired:
                    results.append("Node.js dependency installation timed out")
                except Exception as e:
                    results.append(f"Node.js dependency installation error: {str(e)}")
            
            success = any("successfully" in result for result in results)
            
            return RecoveryResult(
                success=success,
                action_taken="install_missing_dependencies",
                message="; ".join(results),
                details={"installation_results": results},
                retry_recommended=success
            )
            
        except Exception as e:
            return RecoveryResult(
                success=False,
                action_taken="install_missing_dependencies",
                message=f"Dependency installation failed: {str(e)}"
            )
    
    def _activate_virtual_environment(self, error: StartupError, context: Dict[str, Any]) -> RecoveryResult:
        """Activate virtual environment"""
        venv_paths = [
            Path("venv"),
            Path(".venv"),
            Path("env"),
            Path(".env")
        ]
        
        for venv_path in venv_paths:
            if venv_path.exists():
                activate_script = venv_path / "Scripts" / "activate.bat"
                if activate_script.exists():
                    return RecoveryResult(
                        success=True,
                        action_taken="activate_virtual_environment",
                        message=f"Virtual environment found at {venv_path}",
                        details={
                            "venv_path": str(venv_path),
                            "activate_script": str(activate_script),
                            "instruction": f"Run: {activate_script}"
                        },
                        retry_recommended=True
                    )
        
        return RecoveryResult(
            success=False,
            action_taken="activate_virtual_environment",
            message="No virtual environment found in common locations"
        )
    
    def _repair_config_file(self, error: StartupError, context: Dict[str, Any]) -> RecoveryResult:
        """Repair or recreate configuration file"""
        config_file = context.get("config_file", "config.json")
        
        try:
            # Try to create a basic valid configuration
            default_config = {
                "backend": {
                    "host": "localhost",
                    "port": 8000,
                    "reload": True
                },
                "frontend": {
                    "host": "localhost", 
                    "port": 3000,
                    "open_browser": True
                }
            }
            
            # Backup existing file if it exists
            config_path = Path(config_file)
            if config_path.exists():
                backup_path = config_path.with_suffix(f".backup.{int(time.time())}")
                config_path.rename(backup_path)
            
            # Write new configuration
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            return RecoveryResult(
                success=True,
                action_taken="repair_config_file",
                message=f"Configuration file repaired: {config_file}",
                details={"config_file": config_file, "default_config": default_config},
                retry_recommended=True
            )
            
        except Exception as e:
            return RecoveryResult(
                success=False,
                action_taken="repair_config_file",
                message=f"Failed to repair config file: {str(e)}"
            )
    
    def _create_virtual_environment(self, error: StartupError, context: Dict[str, Any]) -> RecoveryResult:
        """Create new virtual environment"""
        try:
            venv_path = Path("venv")
            
            # Create virtual environment
            result = subprocess.run(
                ["python", "-m", "venv", str(venv_path)],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                return RecoveryResult(
                    success=True,
                    action_taken="create_virtual_environment",
                    message=f"Virtual environment created at {venv_path}",
                    details={
                        "venv_path": str(venv_path),
                        "activate_script": str(venv_path / "Scripts" / "activate.bat")
                    },
                    retry_recommended=True
                )
            else:
                return RecoveryResult(
                    success=False,
                    action_taken="create_virtual_environment",
                    message=f"Failed to create virtual environment: {result.stderr}"
                )
                
        except Exception as e:
            return RecoveryResult(
                success=False,
                action_taken="create_virtual_environment",
                message=f"Virtual environment creation failed: {str(e)}"
            )
    
    def _suggest_firewall_exception(self, error: StartupError, context: Dict[str, Any]) -> RecoveryResult:
        """Provide firewall exception suggestions"""
        port = context.get("port", "8000/3000")
        
        instructions = [
            f"Firewall may be blocking port {port}. To resolve:",
            "1. Open Windows Defender Firewall",
            "2. Click 'Allow an app or feature through Windows Defender Firewall'",
            "3. Click 'Change Settings' then 'Allow another app'",
            "4. Browse and add Python.exe and Node.exe",
            "5. Ensure both Private and Public are checked",
            "6. Alternatively, try running as administrator"
        ]
        
        return RecoveryResult(
            success=False,  # Requires manual action
            action_taken="suggest_firewall_exception",
            message="Firewall exception instructions provided",
            details={"instructions": instructions, "port": port}
        )
    
    def _restart_with_different_params(self, error: StartupError, context: Dict[str, Any]) -> RecoveryResult:
        """Restart process with different parameters"""
        alternative_params = context.get("alternative_params", {})
        
        suggestions = {
            "backend": ["--host", "127.0.0.1", "--port", "8001"],
            "frontend": ["--host", "127.0.0.1", "--port", "3001"]
        }
        
        return RecoveryResult(
            success=True,
            action_taken="restart_with_different_params",
            message="Alternative parameters suggested",
            details={"alternative_params": suggestions},
            retry_recommended=True
        )
    
    def _increase_timeout(self, error: StartupError, context: Dict[str, Any]) -> RecoveryResult:
        """Increase timeout values and retry"""
        current_timeout = context.get("timeout", 30)
        new_timeout = min(current_timeout * 2, 120)  # Cap at 2 minutes
        
        return RecoveryResult(
            success=True,
            action_taken="increase_timeout",
            message=f"Timeout increased from {current_timeout}s to {new_timeout}s",
            details={"old_timeout": current_timeout, "new_timeout": new_timeout},
            retry_recommended=True
        )


class FailurePattern:
    """Represents a detected failure pattern"""
    
    def __init__(self, pattern_id: str, description: str, frequency: int = 1):
        self.pattern_id = pattern_id
        self.description = description
        self.frequency = frequency
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.error_types = set()
        self.contexts = []
        self.successful_recoveries = []
        self.failed_recoveries = []
    
    def add_occurrence(self, error_type: ErrorType, context: Dict[str, Any]):
        """Add a new occurrence of this failure pattern"""
        self.frequency += 1
        self.last_seen = time.time()
        self.error_types.add(error_type)
        self.contexts.append(context)
    
    def add_recovery_result(self, action_name: str, success: bool):
        """Record the result of a recovery attempt"""
        if success:
            self.successful_recoveries.append(action_name)
        else:
            self.failed_recoveries.append(action_name)
    
    def get_success_rate_for_action(self, action_name: str) -> float:
        """Get success rate for a specific recovery action"""
        successes = self.successful_recoveries.count(action_name)
        failures = self.failed_recoveries.count(action_name)
        total = successes + failures
        
        if total == 0:
            return 0.0
        
        return successes / total
    
    def get_most_successful_actions(self) -> List[str]:
        """Get recovery actions sorted by success rate for this pattern"""
        action_stats = {}
        
        for action in set(self.successful_recoveries + self.failed_recoveries):
            action_stats[action] = self.get_success_rate_for_action(action)
        
        return sorted(action_stats.keys(), key=lambda x: action_stats[x], reverse=True)


class FallbackConfiguration:
    """Manages fallback configurations when primary recovery methods fail"""
    
    def __init__(self):
        self.fallback_configs = {
            ErrorType.PORT_CONFLICT: {
                "safe_port_ranges": [(8080, 8090), (8000, 9010), (3001, 3010)],
                "alternative_hosts": ["127.0.0.1", "0.0.0.0"],
                "reduced_functionality": True
            },
            ErrorType.PERMISSION_DENIED: {
                "safe_port_ranges": [(8080, 8090)],
                "alternative_hosts": ["127.0.0.1"],
                "run_as_user": True,
                "disable_features": ["auto_open_browser", "system_integration"]
            },
            ErrorType.DEPENDENCY_MISSING: {
                "minimal_mode": True,
                "skip_optional_deps": True,
                "use_fallback_implementations": True
            },
            ErrorType.FIREWALL_BLOCKED: {
                "safe_port_ranges": [(8080, 8090)],
                "local_only": True,
                "disable_network_features": True
            }
        }
    
    def get_fallback_config(self, error_type: ErrorType) -> Dict[str, Any]:
        """Get fallback configuration for an error type"""
        return self.fallback_configs.get(error_type, {})
    
    def apply_fallback_config(self, error_type: ErrorType, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply fallback configuration to base configuration"""
        fallback = self.get_fallback_config(error_type)
        
        if not fallback:
            return base_config
        
        # Create a copy of the base config
        config = base_config.copy()
        
        # Apply fallback modifications
        if "safe_port_ranges" in fallback:
            config["safe_ports"] = fallback["safe_port_ranges"]
        
        if "alternative_hosts" in fallback:
            config["allowed_hosts"] = fallback["alternative_hosts"]
        
        if "minimal_mode" in fallback:
            config["minimal_mode"] = fallback["minimal_mode"]
        
        if "disable_features" in fallback:
            config["disabled_features"] = fallback["disable_features"]
        
        if "local_only" in fallback:
            config["host"] = "127.0.0.1"
            config["local_only"] = True
        
        return config


class IntelligentFailureHandler:
    """Handles intelligent failure detection and learning"""
    
    def __init__(self, recovery_engine):
        self.recovery_engine = recovery_engine
        self.failure_patterns: Dict[str, FailurePattern] = {}
        self.fallback_config = FallbackConfiguration()
        self.learning_enabled = True
        self.pattern_threshold = 3  # Minimum occurrences to consider a pattern
        
        # Load existing patterns from storage
        self.load_failure_patterns()
    
    def detect_failure_pattern(self, error: StartupError, context: Dict[str, Any]) -> Optional[FailurePattern]:
        """Detect if this error matches an existing failure pattern"""
        if not self.learning_enabled:
            return None
            
        # Create a pattern signature based on error characteristics
        pattern_signature = self._create_pattern_signature(error, context)
        
        if pattern_signature in self.failure_patterns:
            pattern = self.failure_patterns[pattern_signature]
            pattern.add_occurrence(error.type, context)
            return pattern
        
        # Check for similar patterns
        similar_pattern = self._find_similar_pattern(error, context)
        if similar_pattern:
            similar_pattern.add_occurrence(error.type, context)
            return similar_pattern
        
        # Create new pattern if this is a recurring issue
        if self._should_create_pattern(error, context):
            pattern = FailurePattern(
                pattern_id=pattern_signature,
                description=self._generate_pattern_description(error, context)
            )
            pattern.add_occurrence(error.type, context)
            self.failure_patterns[pattern_signature] = pattern
            return pattern
        
        return None
    
    def _create_pattern_signature(self, error: StartupError, context: Dict[str, Any]) -> str:
        """Create a unique signature for a failure pattern"""
        # Combine error type, key context elements, and error message patterns
        signature_parts = [
            error.type.value,
            str(context.get("port", "")),
            str(context.get("process_name", "")),
            self._extract_error_keywords(error.message)
        ]
        
        return "|".join(filter(None, signature_parts))
    
    def _extract_error_keywords(self, error_message: str) -> str:
        """Extract key words from error message for pattern matching"""
        # Common error keywords that help identify patterns
        keywords = [
            "permission", "denied", "access", "forbidden",
            "address", "use", "bind", "port", "socket",
            "timeout", "connection", "refused", "blocked",
            "module", "import", "found", "missing",
            "config", "json", "invalid", "corrupt"
        ]
        
        message_lower = error_message.lower()
        found_keywords = [kw for kw in keywords if kw in message_lower]
        
        return ",".join(sorted(found_keywords))
    
    def _find_similar_pattern(self, error: StartupError, context: Dict[str, Any]) -> Optional[FailurePattern]:
        """Find similar existing patterns"""
        current_keywords = self._extract_error_keywords(error.message)
        current_port = context.get("port")
        
        for pattern in self.failure_patterns.values():
            # Check if error types match
            if error.type in pattern.error_types:
                # Check for similar contexts
                for pattern_context in pattern.contexts:
                    if (current_port and pattern_context.get("port") == current_port) or \
                       (current_keywords and current_keywords in pattern.pattern_id):
                        return pattern
        
        return None
    
    def _should_create_pattern(self, error: StartupError, context: Dict[str, Any]) -> bool:
        """Determine if we should create a new failure pattern"""
        # For now, create patterns for specific error types that benefit from learning
        learning_error_types = {
            ErrorType.PORT_CONFLICT,
            ErrorType.PERMISSION_DENIED,
            ErrorType.FIREWALL_BLOCKED,
            ErrorType.DEPENDENCY_MISSING
        }
        
        return error.type in learning_error_types
    
    def _generate_pattern_description(self, error: StartupError, context: Dict[str, Any]) -> str:
        """Generate a human-readable description for the pattern"""
        descriptions = {
            ErrorType.PORT_CONFLICT: f"Port {context.get('port', 'unknown')} conflict pattern",
            ErrorType.PERMISSION_DENIED: f"Permission denied on port {context.get('port', 'unknown')}",
            ErrorType.FIREWALL_BLOCKED: f"Firewall blocking port {context.get('port', 'unknown')}",
            ErrorType.DEPENDENCY_MISSING: f"Missing dependency: {context.get('missing_dep', 'unknown')}"
        }
        
        return descriptions.get(error.type, f"Unknown pattern for {error.type.value}")
    
    def prioritize_recovery_actions(self, error: StartupError, context: Dict[str, Any]) -> List[RecoveryAction]:
        """Prioritize recovery actions based on learned patterns and success rates"""
        # Get base recovery actions
        base_actions = self.recovery_engine.get_recovery_actions(error.type)
        
        if not self.learning_enabled or not base_actions:
            return base_actions
        
        # Check if this matches a known failure pattern
        pattern = self.detect_failure_pattern(error, context)
        
        if pattern and pattern.frequency >= self.pattern_threshold:
            # Reorder actions based on pattern-specific success rates
            pattern_actions = []
            remaining_actions = []
            
            successful_actions = pattern.get_most_successful_actions()
            
            # First, add actions that have been successful for this pattern
            for action_name in successful_actions:
                for action in base_actions:
                    if action.name == action_name:
                        # Update action priority based on pattern success rate
                        action.success_rate = pattern.get_success_rate_for_action(action_name)
                        pattern_actions.append(action)
                        break
            
            # Add remaining actions
            used_names = {action.name for action in pattern_actions}
            remaining_actions = [action for action in base_actions if action.name not in used_names]
            
            # Combine pattern-optimized actions with remaining actions
            return pattern_actions + remaining_actions
        
        return base_actions
    
    def handle_recovery_failure(self, error: StartupError, context: Dict[str, Any], 
                               failed_actions: List[str]) -> RecoveryResult:
        """Handle the case when all primary recovery actions fail"""
        # Record the failure pattern
        pattern = self.detect_failure_pattern(error, context)
        if pattern:
            for action_name in failed_actions:
                pattern.add_recovery_result(action_name, False)
        
        # Try fallback configuration
        fallback_config = self.fallback_config.apply_fallback_config(error.type, context)
        
        if fallback_config != context:
            return RecoveryResult(
                success=True,
                action_taken="apply_fallback_configuration",
                message=f"Applied fallback configuration for {error.type.value}",
                details={"fallback_config": fallback_config},
                retry_recommended=True
            )
        
        # If no fallback is available, suggest manual intervention
        return RecoveryResult(
            success=False,
            action_taken="manual_intervention_required",
            message=f"All automatic recovery attempts failed for {error.type.value}. Manual intervention required.",
            details={
                "failed_actions": failed_actions,
                "suggested_manual_actions": self._get_manual_intervention_suggestions(error.type)
            },
            fallback_needed=False  # We've exhausted all options
        )
    
    def _get_manual_intervention_suggestions(self, error_type: ErrorType) -> List[str]:
        """Get manual intervention suggestions for an error type"""
        suggestions = {
            ErrorType.PORT_CONFLICT: [
                "Manually kill processes using the required ports",
                "Restart the computer to clear all port locks",
                "Use netstat -ano to identify processes using ports",
                "Configure application to use different default ports"
            ],
            ErrorType.PERMISSION_DENIED: [
                "Run the application as administrator",
                "Add firewall exceptions manually",
                "Check Windows Defender settings",
                "Verify user account has necessary permissions"
            ],
            ErrorType.DEPENDENCY_MISSING: [
                "Manually install missing dependencies",
                "Check Python/Node.js installation",
                "Verify virtual environment is properly configured",
                "Update package managers (pip, npm)"
            ],
            ErrorType.FIREWALL_BLOCKED: [
                "Manually configure firewall exceptions",
                "Temporarily disable firewall for testing",
                "Check antivirus software settings",
                "Contact system administrator"
            ]
        }
        
        return suggestions.get(error_type, ["Contact technical support"])
    
    def save_failure_patterns(self):
        """Save learned failure patterns to persistent storage"""
        try:
            patterns_data = {}
            
            for pattern_id, pattern in self.failure_patterns.items():
                patterns_data[pattern_id] = {
                    "description": pattern.description,
                    "frequency": pattern.frequency,
                    "first_seen": pattern.first_seen,
                    "last_seen": pattern.last_seen,
                    "error_types": list(pattern.error_types),
                    "successful_recoveries": pattern.successful_recoveries,
                    "failed_recoveries": pattern.failed_recoveries
                }
            
            patterns_file = Path("failure_patterns.json")
            with open(patterns_file, 'w') as f:
                json.dump(patterns_data, f, indent=2, default=str)
                
        except Exception as e:
            self.recovery_engine.logger.error(f"Failed to save failure patterns: {e}")
    
    def load_failure_patterns(self):
        """Load failure patterns from persistent storage"""
        try:
            patterns_file = Path("failure_patterns.json")
            
            if not patterns_file.exists():
                return
            
            with open(patterns_file, 'r') as f:
                patterns_data = json.load(f)
            
            for pattern_id, data in patterns_data.items():
                pattern = FailurePattern(pattern_id, data["description"])
                pattern.frequency = data["frequency"]
                pattern.first_seen = data["first_seen"]
                pattern.last_seen = data["last_seen"]
                pattern.error_types = set(ErrorType(et) for et in data["error_types"])
                pattern.successful_recoveries = data["successful_recoveries"]
                pattern.failed_recoveries = data["failed_recoveries"]
                
                self.failure_patterns[pattern_id] = pattern
                
        except Exception as e:
            self.recovery_engine.logger.warning(f"Failed to load failure patterns: {e}")


# Add intelligent failure handling to the main RecoveryEngine class
def _add_intelligent_failure_handling(self):
    """Add intelligent failure handling capabilities to RecoveryEngine"""
    self.intelligent_handler = IntelligentFailureHandler(self)
    
    # Override the attempt_recovery method to use intelligent handling
    original_attempt_recovery = self.attempt_recovery
    
    def intelligent_attempt_recovery(error: StartupError, context: Optional[Dict[str, Any]] = None) -> RecoveryResult:
        """Enhanced recovery attempt with intelligent failure handling"""
        context = context or {}
        
        # Get prioritized recovery actions
        actions = self.intelligent_handler.prioritize_recovery_actions(error, context)
        
        if not actions:
            return RecoveryResult(
                success=False,
                action_taken="none",
                message=f"No recovery actions available for error type: {error.type.value}",
                fallback_needed=True
            )
        
        failed_actions = []
        
        # Try each recovery action in prioritized order
        for action in actions:
            if not self.config.get("auto_fix_enabled", True) and action.auto_executable:
                continue
            
            try:
                self.logger.info(f"Trying prioritized recovery action: {action.name}")
                
                # Execute the recovery action
                result = action.action_func(error, context)
                
                # Update success rate and pattern learning
                self._update_success_rate(action.name, result.success)
                
                # Record result in failure pattern if applicable
                pattern = self.intelligent_handler.detect_failure_pattern(error, context)
                if pattern:
                    pattern.add_recovery_result(action.name, result.success)
                
                if result.success:
                    self.logger.info(f"Prioritized recovery action '{action.name}' succeeded")
                    return result
                else:
                    self.logger.warning(f"Prioritized recovery action '{action.name}' failed: {result.message}")
                    failed_actions.append(action.name)
                    
            except Exception as e:
                self.logger.error(f"Recovery action '{action.name}' raised exception: {e}")
                self._update_success_rate(action.name, False)
                failed_actions.append(action.name)
        
        # All primary actions failed, try intelligent fallback handling
        return self.intelligent_handler.handle_recovery_failure(error, context, failed_actions)
    
    # Replace the method
    self.attempt_recovery = intelligent_attempt_recovery


# Monkey patch the RecoveryEngine class to add intelligent failure handling
RecoveryEngine._add_intelligent_failure_handling = _add_intelligent_failure_handling

# Modify the RecoveryEngine __init__ to include intelligent handling
original_init = RecoveryEngine.__init__

def enhanced_init(self, config_path: Optional[Path] = None):
    """Enhanced initialization with intelligent failure handling"""
    original_init(self, config_path)
    self._add_intelligent_failure_handling()

RecoveryEngine.__init__ = enhanced_init
