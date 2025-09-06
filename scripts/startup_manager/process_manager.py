"""
Process Manager for WAN22 server startup management.
Handles server process lifecycle, health monitoring, and cleanup.
"""

import os
import sys
import time
import signal
import subprocess
import threading
import requests
import psutil
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

from .config import StartupConfig, BackendConfig, FrontendConfig


class ProcessStatus(Enum):
    """Process status enumeration."""
    STARTING = "starting"
    RUNNING = "running"
    FAILED = "failed"
    STOPPED = "stopped"
    UNKNOWN = "unknown"


@dataclass
class ProcessInfo:
    """Information about a managed process."""
    name: str
    pid: Optional[int] = None
    port: int = 0
    status: ProcessStatus = ProcessStatus.STARTING
    start_time: Optional[datetime] = None
    health_check_url: str = ""
    log_file: str = ""
    process: Optional[subprocess.Popen] = None
    working_directory: str = ""
    command: List[str] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    restart_count: int = 0
    last_restart: Optional[datetime] = None
    auto_restart: bool = True


@dataclass
class ProcessResult:
    """Result of a process operation."""
    success: bool
    process_info: Optional[ProcessInfo] = None
    error_message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success_result(cls, process_info: ProcessInfo) -> 'ProcessResult':
        """Create a successful process result."""
        return cls(success=True, process_info=process_info)

    @classmethod
    def failure_result(cls, error_message: str, details: Dict[str, Any] = None) -> 'ProcessResult':
        """Create a failed process result."""
        return cls(
            success=False, 
            error_message=error_message, 
            details=details or {}
        )


class HealthMonitor:
    """Health monitoring for server processes."""
    
    def __init__(self, check_interval: float = 5.0):
        self.check_interval = check_interval
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.processes: Dict[str, ProcessInfo] = {}
        self._stop_event = threading.Event()
    
    def add_process(self, process_info: ProcessInfo):
        """Add a process to monitor."""
        self.processes[process_info.name] = process_info
    
    def remove_process(self, process_name: str):
        """Remove a process from monitoring."""
        self.processes.pop(process_name, None)
    
    def start_monitoring(self):
        """Start health monitoring in background thread."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self._stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring = False
        self._stop_event.set()
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring and not self._stop_event.is_set():
            for process_name, process_info in self.processes.items():
                try:
                    self._check_process_health(process_info)
                except Exception as e:
                    print(f"Error monitoring {process_name}: {e}")
            
            self._stop_event.wait(self.check_interval)
    
    def _check_process_health(self, process_info: ProcessInfo):
        """Check health of a single process."""
        if not process_info.process:
            process_info.status = ProcessStatus.UNKNOWN
            return
        
        # Check if process is still running
        poll_result = process_info.process.poll()
        if poll_result is not None:
            process_info.status = ProcessStatus.FAILED
            return
        
        # Check HTTP endpoint if available
        if process_info.health_check_url:
            try:
                response = requests.get(
                    process_info.health_check_url, 
                    timeout=5.0
                )
                if response.status_code == 200:
                    process_info.status = ProcessStatus.RUNNING
                else:
                    process_info.status = ProcessStatus.FAILED
            except requests.RequestException:
                # HTTP check failed, but process might still be starting
                if process_info.status == ProcessStatus.STARTING:
                    # Allow some time for startup
                    if (process_info.start_time and 
                        (datetime.now() - process_info.start_time).seconds > 30):
                        process_info.status = ProcessStatus.FAILED
                else:
                    process_info.status = ProcessStatus.FAILED


class ProcessManager:
    """Manages server process lifecycle and health monitoring."""
    
    def __init__(self, config: StartupConfig):
        self.config = config
        self.processes: Dict[str, ProcessInfo] = {}
        self.health_monitor = HealthMonitor()
        self.project_root = Path.cwd()
        
        # Ensure log directory exists
        self.log_dir = self.project_root / "logs"
        self.log_dir.mkdir(exist_ok=True)
    
    def start_backend(self, port: int, backend_config: Optional[BackendConfig] = None) -> ProcessResult:
        """Start FastAPI backend server."""
        if backend_config is None:
            backend_config = self.config.backend
        
        # Set up backend process info
        process_info = ProcessInfo(
            name="backend",
            port=port,
            working_directory=str(self.project_root / "backend"),
            health_check_url=f"http://{backend_config.host}:{port}/health",
            log_file=str(self.log_dir / "backend_startup.log")
        )
        
        # Prepare command
        python_executable = sys.executable
        main_script = self.project_root / "backend" / "main.py"
        
        if not main_script.exists():
            # Try alternative locations
            alt_script = self.project_root / "backend" / "app.py"
            if alt_script.exists():
                main_script = alt_script
            else:
                return ProcessResult.failure_result(
                    f"Backend main script not found. Looked for: {main_script}, {alt_script}"
                )
        
        process_info.command = [
            python_executable, str(main_script),
            "--host", backend_config.host,
            "--port", str(port),
            "--log-level", backend_config.log_level
        ]
        
        if backend_config.reload:
            process_info.command.append("--reload")
        
        # Set up environment
        process_info.environment = os.environ.copy()
        process_info.environment.update({
            "PYTHONPATH": str(self.project_root),
            "WAN22_PORT": str(port),
            "WAN22_HOST": backend_config.host,
            "WAN22_LOG_LEVEL": backend_config.log_level
        })
        
        return self._start_process(process_info)
    
    def start_frontend(self, port: int, frontend_config: Optional[FrontendConfig] = None) -> ProcessResult:
        """Start React frontend development server."""
        if frontend_config is None:
            frontend_config = self.config.frontend
        
        # Set up frontend process info
        process_info = ProcessInfo(
            name="frontend",
            port=port,
            working_directory=str(self.project_root / "frontend"),
            health_check_url=f"http://{frontend_config.host}:{port}",
            log_file=str(self.log_dir / "frontend_startup.log")
        )
        
        # Detect package manager (npm or yarn)
        frontend_dir = Path(process_info.working_directory)
        package_manager = self._detect_package_manager(frontend_dir)
        
        if not package_manager:
            return ProcessResult.failure_result(
                "No package manager found. Please install npm or yarn."
            )
        
        # Prepare command
        if package_manager == "npm":
            process_info.command = ["npm", "run", "dev"]
        else:  # yarn
            process_info.command = ["yarn", "dev"]
        
        # Set up environment
        process_info.environment = os.environ.copy()
        process_info.environment.update({
            "PORT": str(port),
            "HOST": frontend_config.host,
            "BROWSER": "none" if not frontend_config.open_browser else "default"
        })
        
        # Add Vite-specific environment variables
        process_info.environment.update({
            "VITE_API_URL": f"http://{self.config.backend.host}:{self.config.backend.port or 8000}",
            "VITE_PORT": str(port)
        })
        
        return self._start_process(process_info)
    
    def _detect_package_manager(self, frontend_dir: Path) -> Optional[str]:
        """Detect which package manager to use (npm or yarn)."""
        if (frontend_dir / "yarn.lock").exists():
            # Check if yarn is available
            try:
                subprocess.run(["yarn", "--version"], 
                             capture_output=True, check=True, timeout=5)
                return "yarn"
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                pass
        
        # Check if npm is available
        try:
            subprocess.run(["npm", "--version"], 
                         capture_output=True, check=True, timeout=5)
            return "npm"
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return None
    
    def _start_process(self, process_info: ProcessInfo) -> ProcessResult:
        """Start a process with the given configuration."""
        try:
            # Ensure working directory exists
            working_dir = Path(process_info.working_directory)
            if not working_dir.exists():
                return ProcessResult.failure_result(
                    f"Working directory does not exist: {working_dir}"
                )
            
            # Open log file
            log_file = open(process_info.log_file, 'w', encoding='utf-8')
            
            # Start the process
            process_info.start_time = datetime.now()
            process_info.process = subprocess.Popen(
                process_info.command,
                cwd=process_info.working_directory,
                env=process_info.environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )
            
            process_info.pid = process_info.process.pid
            process_info.status = ProcessStatus.STARTING
            
            # Add to managed processes
            self.processes[process_info.name] = process_info
            self.health_monitor.add_process(process_info)
            
            # Start health monitoring if not already running
            if not self.health_monitor.monitoring:
                self.health_monitor.start_monitoring()
            
            return ProcessResult.success_result(process_info)
            
        except Exception as e:
            return ProcessResult.failure_result(
                f"Failed to start {process_info.name}: {str(e)}",
                {"command": process_info.command, "working_dir": process_info.working_directory}
            )
    
    def get_process_status(self, process_name: str) -> Optional[ProcessInfo]:
        """Get current status of a process."""
        return self.processes.get(process_name)
    
    def is_process_healthy(self, process_name: str) -> bool:
        """Check if a process is healthy."""
        process_info = self.processes.get(process_name)
        if not process_info:
            return False
        return process_info.status == ProcessStatus.RUNNING
    
    def wait_for_health(self, process_name: str, timeout: float = 30.0) -> bool:
        """Wait for a process to become healthy."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_process_healthy(process_name):
                return True
            time.sleep(1.0)
        return False
    
    def get_all_processes(self) -> Dict[str, ProcessInfo]:
        """Get information about all managed processes."""
        return self.processes.copy()
    
    def cleanup(self):
        """Clean up all processes and monitoring."""
        self.health_monitor.stop_monitoring()
        for process_name in list(self.processes.keys()):
            self.stop_process(process_name)
    
    def stop_process(self, process_name: str, force: bool = False) -> bool:
        """Stop a specific process."""
        process_info = self.processes.get(process_name)
        if not process_info or not process_info.process:
            return True
        
        try:
            if force or os.name == 'nt':
                # On Windows, use terminate
                process_info.process.terminate()
            else:
                # On Unix-like systems, try graceful shutdown first
                process_info.process.send_signal(signal.SIGTERM)
            
            # Wait for process to exit
            try:
                process_info.process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown failed
                process_info.process.kill()
                try:
                    process_info.process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    # Process is really stuck, but we'll consider it stopped
                    pass
            
            process_info.status = ProcessStatus.STOPPED
            self.health_monitor.remove_process(process_name)
            
            return True
            
        except Exception as e:
            print(f"Error stopping {process_name}: {e}")
            return False
        finally:
            # Clean up from managed processes
            self.processes.pop(process_name, None)
    
    def graceful_shutdown(self, process_name: str, timeout: float = 30.0) -> bool:
        """Gracefully shutdown a process with SIGTERM/SIGKILL escalation."""
        process_info = self.processes.get(process_name)
        if not process_info or not process_info.process:
            return True
        
        try:
            # Step 1: Try graceful shutdown with SIGTERM (or terminate on Windows)
            if os.name == 'nt':
                process_info.process.terminate()
            else:
                process_info.process.send_signal(signal.SIGTERM)
            
            # Step 2: Wait for graceful shutdown
            try:
                process_info.process.wait(timeout=timeout * 0.8)  # Use 80% of timeout for graceful
                process_info.status = ProcessStatus.STOPPED
                return True
            except subprocess.TimeoutExpired:
                pass
            
            # Step 3: Force kill if graceful shutdown failed
            process_info.process.kill()
            try:
                process_info.process.wait(timeout=timeout * 0.2)  # Use remaining 20% for force kill
                process_info.status = ProcessStatus.STOPPED
                return True
            except subprocess.TimeoutExpired:
                # Process is really stuck
                process_info.status = ProcessStatus.FAILED
                return False
                
        except Exception as e:
            logging.error(f"Error during graceful shutdown of {process_name}: {e}")
            return False
    
    def cleanup_zombie_processes(self) -> List[str]:
        """Clean up zombie processes and file locks."""
        cleaned_processes = []
        
        for process_name, process_info in list(self.processes.items()):
            if not process_info.process:
                continue
                
            # Check if process is actually dead but not cleaned up
            poll_result = process_info.process.poll()
            if poll_result is not None:
                # Process is dead, clean it up
                try:
                    # Close any open file handles
                    if hasattr(process_info.process, 'stdout') and process_info.process.stdout:
                        process_info.process.stdout.close()
                    if hasattr(process_info.process, 'stderr') and process_info.process.stderr:
                        process_info.process.stderr.close()
                    if hasattr(process_info.process, 'stdin') and process_info.process.stdin:
                        process_info.process.stdin.close()
                    
                    process_info.status = ProcessStatus.STOPPED
                    self.health_monitor.remove_process(process_name)
                    cleaned_processes.append(process_name)
                    
                except Exception as e:
                    logging.warning(f"Error cleaning up zombie process {process_name}: {e}")
        
        # Also check for orphaned processes using psutil
        try:
            for process_name, process_info in list(self.processes.items()):
                if process_info.pid:
                    try:
                        proc = psutil.Process(process_info.pid)
                        if not proc.is_running():
                            process_info.status = ProcessStatus.STOPPED
                            self.health_monitor.remove_process(process_name)
                            if process_name not in cleaned_processes:
                                cleaned_processes.append(process_name)
                    except psutil.NoSuchProcess:
                        # Process doesn't exist anymore
                        process_info.status = ProcessStatus.STOPPED
                        self.health_monitor.remove_process(process_name)
                        if process_name not in cleaned_processes:
                            cleaned_processes.append(process_name)
        except Exception as e:
            logging.warning(f"Error during psutil cleanup: {e}")
        
        return cleaned_processes
    
    def restart_process(self, process_name: str, max_attempts: int = 3) -> ProcessResult:
        """Restart a process with exponential backoff."""
        process_info = self.processes.get(process_name)
        if not process_info:
            return ProcessResult.failure_result(f"Process {process_name} not found")
        
        # Check if we should restart (respect restart limits)
        if process_info.restart_count >= max_attempts:
            return ProcessResult.failure_result(
                f"Process {process_name} has exceeded maximum restart attempts ({max_attempts})"
            )
        
        # Calculate backoff delay
        backoff_delay = min(2 ** process_info.restart_count, 60)  # Cap at 60 seconds
        
        # Wait for backoff if this isn't the first restart
        if process_info.restart_count > 0:
            logging.info(f"Waiting {backoff_delay} seconds before restarting {process_name}")
            time.sleep(backoff_delay)
        
        # Stop the current process
        if process_info.process and process_info.process.poll() is None:
            if not self.graceful_shutdown(process_name):
                return ProcessResult.failure_result(f"Failed to stop {process_name} for restart")
        
        # Update restart tracking
        process_info.restart_count += 1
        process_info.last_restart = datetime.now()
        
        # Restart based on process type
        try:
            if process_name == "backend":
                result = self.start_backend(process_info.port)
            elif process_name == "frontend":
                result = self.start_frontend(process_info.port)
            else:
                # Generic restart using stored command and environment
                result = self._restart_generic_process(process_info)
            
            if result.success:
                # Copy restart tracking to new process info
                if result.process_info:
                    result.process_info.restart_count = process_info.restart_count
                    result.process_info.last_restart = process_info.last_restart
                    result.process_info.auto_restart = process_info.auto_restart
            
            return result
            
        except Exception as e:
            return ProcessResult.failure_result(
                f"Failed to restart {process_name}: {str(e)}",
                {"restart_count": process_info.restart_count, "backoff_delay": backoff_delay}
            )
    
    def _restart_generic_process(self, process_info: ProcessInfo) -> ProcessResult:
        """Restart a generic process using stored configuration."""
        if not process_info.command:
            return ProcessResult.failure_result("No command stored for process restart")
        
        # Create new process info with same configuration
        new_process_info = ProcessInfo(
            name=process_info.name,
            port=process_info.port,
            working_directory=process_info.working_directory,
            health_check_url=process_info.health_check_url,
            log_file=process_info.log_file,
            command=process_info.command.copy(),
            environment=process_info.environment.copy()
        )
        
        return self._start_process(new_process_info)
    
    def auto_restart_failed_processes(self) -> Dict[str, ProcessResult]:
        """Automatically restart failed processes that have auto_restart enabled."""
        restart_results = {}
        
        for process_name, process_info in list(self.processes.items()):
            if (process_info.status == ProcessStatus.FAILED and 
                process_info.auto_restart and
                process_info.restart_count < 3):  # Default max attempts
                
                # Check if enough time has passed since last restart
                if (process_info.last_restart and 
                    datetime.now() - process_info.last_restart < timedelta(minutes=1)):
                    continue  # Too soon to restart
                
                logging.info(f"Auto-restarting failed process: {process_name}")
                result = self.restart_process(process_name)
                restart_results[process_name] = result
        
        return restart_results
    
    def get_process_metrics(self, process_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed metrics for a process."""
        process_info = self.processes.get(process_name)
        if not process_info or not process_info.pid:
            return None
        
        try:
            proc = psutil.Process(process_info.pid)
            
            # Get memory info
            memory_info = proc.memory_info()
            memory_percent = proc.memory_percent()
            
            # Get CPU info
            cpu_percent = proc.cpu_percent()
            
            # Get process status
            status = proc.status()
            
            # Calculate uptime
            uptime = None
            if process_info.start_time:
                uptime = (datetime.now() - process_info.start_time).total_seconds()
            
            return {
                "pid": process_info.pid,
                "status": status,
                "cpu_percent": cpu_percent,
                "memory_rss": memory_info.rss,
                "memory_vms": memory_info.vms,
                "memory_percent": memory_percent,
                "uptime_seconds": uptime,
                "restart_count": process_info.restart_count,
                "last_restart": process_info.last_restart.isoformat() if process_info.last_restart else None,
                "port": process_info.port,
                "health_check_url": process_info.health_check_url
            }
            
        except psutil.NoSuchProcess:
            return None
        except Exception as e:
            logging.error(f"Error getting metrics for {process_name}: {e}")
            return None
    
    def set_auto_restart(self, process_name: str, enabled: bool) -> bool:
        """Enable or disable auto-restart for a process."""
        process_info = self.processes.get(process_name)
        if not process_info:
            return False
        
        process_info.auto_restart = enabled
        return True
    
    def reset_restart_count(self, process_name: str) -> bool:
        """Reset the restart count for a process."""
        process_info = self.processes.get(process_name)
        if not process_info:
            return False
        
        process_info.restart_count = 0
        process_info.last_restart = None
        return True