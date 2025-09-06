"""
Port Manager Component for WAN22 Server Startup Management System

This module handles port availability checking, conflict detection, and resolution
with Windows-specific handling for firewall and permission issues.
"""

import socket
import psutil
import subprocess
import logging
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum
import time
import json
from pathlib import Path


class PortStatus(Enum):
    """Port availability status"""
    AVAILABLE = "available"
    OCCUPIED = "occupied"
    BLOCKED = "blocked"  # Firewall or permission issue
    UNKNOWN = "unknown"


class ConflictResolution(Enum):
    """Port conflict resolution strategies"""
    KILL_PROCESS = "kill_process"
    USE_ALTERNATIVE = "use_alternative"
    SKIP = "skip"


@dataclass
class ProcessInfo:
    """Information about a process using a port"""
    pid: int
    name: str
    cmdline: List[str]
    port: int
    can_terminate: bool = False


@dataclass
class PortConflict:
    """Information about a port conflict"""
    port: int
    process: ProcessInfo
    resolution_options: List[ConflictResolution]


class PortAllocation(NamedTuple):
    """Result of port allocation"""
    backend: int
    frontend: int
    conflicts_resolved: List[str]
    alternative_ports_used: bool


class PortCheckResult(NamedTuple):
    """Result of port availability check"""
    port: int
    status: PortStatus
    process: Optional[ProcessInfo] = None
    error_message: Optional[str] = None


class PortManager:
    """
    Manages port allocation, conflict detection, and resolution for server startup.
    
    Handles Windows-specific issues like firewall blocking and provides intelligent
    port allocation with conflict resolution.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        self.default_ports = {"backend": 8000, "frontend": 3000}
        self.safe_port_ranges = [
            (8000, 8099),  # Common development ports
            (3000, 3099),  # Frontend development ports
            (9000, 9099),  # Alternative range
        ]
        self.allocated_ports: Dict[str, int] = {}
        self.config_path = config_path
        
    def check_port_availability(self, port: int, host: str = "localhost") -> PortCheckResult:
        """
        Check if a specific port is available for binding.
        
        Args:
            port: Port number to check
            host: Host address to check (default: localhost)
            
        Returns:
            PortCheckResult with status and details
        """
        try:
            # First check if port is in use by examining network connections
            process_info = self._get_process_using_port(port)
            if process_info:
                return PortCheckResult(
                    port=port,
                    status=PortStatus.OCCUPIED,
                    process=process_info
                )
            
            # Try to bind to the port to check availability
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.settimeout(1.0)  # Quick timeout for availability check
                
                try:
                    sock.bind((host, port))
                    return PortCheckResult(port=port, status=PortStatus.AVAILABLE)
                    
                except socket.error as e:
                    error_code = e.errno if hasattr(e, 'errno') else 0
                    
                    # Windows-specific error handling
                    if error_code == 10013:  # WSAEACCES - Permission denied
                        return PortCheckResult(
                            port=port,
                            status=PortStatus.BLOCKED,
                            error_message="Permission denied - likely firewall or admin rights issue"
                        )
                    elif error_code == 10048:  # WSAEADDRINUSE - Address already in use
                        return PortCheckResult(
                            port=port,
                            status=PortStatus.OCCUPIED,
                            error_message="Address already in use"
                        )
                    else:
                        return PortCheckResult(
                            port=port,
                            status=PortStatus.UNKNOWN,
                            error_message=f"Socket error: {str(e)}"
                        )
                        
        except Exception as e:
            self.logger.error(f"Error checking port {port}: {str(e)}")
            return PortCheckResult(
                port=port,
                status=PortStatus.UNKNOWN,
                error_message=f"Unexpected error: {str(e)}"
            )
    
    def _get_process_using_port(self, port: int) -> Optional[ProcessInfo]:
        """
        Get information about the process using a specific port.
        
        Args:
            port: Port number to check
            
        Returns:
            ProcessInfo if port is in use, None otherwise
        """
        try:
            for conn in psutil.net_connections(kind='inet'):
                if conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
                    if conn.pid:
                        try:
                            process = psutil.Process(conn.pid)
                            return ProcessInfo(
                                pid=conn.pid,
                                name=process.name(),
                                cmdline=process.cmdline(),
                                port=port,
                                can_terminate=self._can_safely_terminate_process(process)
                            )
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            # Process might have ended or we don't have permission
                            continue
            return None
            
        except Exception as e:
            self.logger.warning(f"Error getting process info for port {port}: {str(e)}")
            return None
    
    def _can_safely_terminate_process(self, process: psutil.Process) -> bool:
        """
        Determine if a process can be safely terminated.
        
        Args:
            process: Process to check
            
        Returns:
            True if process can be safely terminated
        """
        try:
            # Don't terminate system processes
            if process.pid <= 4:  # System processes on Windows
                return False
                
            # Don't terminate processes running as SYSTEM or other system users
            try:
                username = process.username()
                if username.lower() in ['nt authority\\system', 'system']:
                    return False
            except (psutil.AccessDenied, AttributeError):
                # If we can't get username, be conservative
                return False
            
            # Check if it's a development server (safe to terminate)
            cmdline = process.cmdline()
            safe_patterns = [
                'python', 'node', 'npm', 'yarn', 'uvicorn', 'fastapi',
                'react-scripts', 'vite', 'webpack-dev-server'
            ]
            
            cmdline_str = ' '.join(cmdline).lower()
            for pattern in safe_patterns:
                if pattern in cmdline_str:
                    return True
                    
            return False
            
        except Exception:
            return False
    
    def find_available_port(self, start_port: int, max_attempts: int = 100) -> Optional[int]:
        """
        Find the next available port starting from the given port.
        
        Args:
            start_port: Starting port number
            max_attempts: Maximum number of ports to try
            
        Returns:
            Available port number or None if none found
        """
        for port in range(start_port, start_port + max_attempts):
            # Skip well-known system ports
            if port < 1024:
                continue
                
            result = self.check_port_availability(port)
            if result.status == PortStatus.AVAILABLE:
                return port
                
        return None
    
    def find_available_ports_in_range(self, port_range: Tuple[int, int]) -> List[int]:
        """
        Find all available ports in a given range.
        
        Args:
            port_range: Tuple of (start_port, end_port)
            
        Returns:
            List of available port numbers
        """
        start_port, end_port = port_range
        available_ports = []
        
        for port in range(start_port, end_port + 1):
            result = self.check_port_availability(port)
            if result.status == PortStatus.AVAILABLE:
                available_ports.append(port)
                
        return available_ports
    
    def detect_port_conflicts(self, required_ports: Dict[str, int] = None) -> List[PortConflict]:
        """
        Detect conflicts for required ports.
        
        Args:
            required_ports: Dictionary of service names to port numbers
            
        Returns:
            List of detected port conflicts
        """
        if required_ports is None:
            required_ports = self.default_ports
            
        conflicts = []
        
        for service_name, port in required_ports.items():
            result = self.check_port_availability(port)
            
            if result.status == PortStatus.OCCUPIED and result.process:
                resolution_options = [ConflictResolution.USE_ALTERNATIVE]
                
                if result.process.can_terminate:
                    resolution_options.insert(0, ConflictResolution.KILL_PROCESS)
                    
                conflicts.append(PortConflict(
                    port=port,
                    process=result.process,
                    resolution_options=resolution_options
                ))
                
        return conflicts
    
    def check_firewall_exceptions(self, ports: List[int]) -> Dict[int, bool]:
        """
        Check if ports have firewall exceptions (Windows-specific).
        
        Args:
            ports: List of port numbers to check
            
        Returns:
            Dictionary mapping port numbers to exception status
        """
        exceptions = {}
        
        try:
            # Use netsh to check Windows Firewall rules
            for port in ports:
                try:
                    # Check if there's an inbound rule for this port
                    cmd = [
                        'netsh', 'advfirewall', 'firewall', 'show', 'rule',
                        f'name=all', 'dir=in', f'localport={port}'
                    ]
                    
                    result = subprocess.run(
                        cmd, 
                        capture_output=True, 
                        text=True, 
                        timeout=5,
                        creationflags=subprocess.CREATE_NO_WINDOW
                    )
                    
                    # If command succeeds and returns rules, port has exceptions
                    exceptions[port] = result.returncode == 0 and 'Rule Name:' in result.stdout
                    
                except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                    exceptions[port] = False
                    
        except Exception as e:
            self.logger.warning(f"Error checking firewall exceptions: {str(e)}")
            # Default to False for all ports if we can't check
            exceptions = {port: False for port in ports}
            
        return exceptions
    
    def suggest_firewall_fix(self, port: int, service_name: str = "WAN22") -> str:
        """
        Generate Windows firewall exception command for a port.
        
        Args:
            port: Port number
            service_name: Name for the firewall rule
            
        Returns:
            Command string to add firewall exception
        """
        return (
            f'netsh advfirewall firewall add rule name="{service_name}-{port}" '
            f'dir=in action=allow protocol=TCP localport={port}'
        )
    
    def allocate_ports(self, required_ports: Dict[str, int] = None) -> PortAllocation:
        """
        Allocate ports for services, resolving conflicts automatically.
        
        Args:
            required_ports: Dictionary of service names to preferred port numbers
            
        Returns:
            PortAllocation with assigned ports and resolution details
        """
        if required_ports is None:
            required_ports = self.default_ports.copy()
            
        allocated = {}
        conflicts_resolved = []
        alternative_ports_used = False
        
        for service_name, preferred_port in required_ports.items():
            result = self.check_port_availability(preferred_port)
            
            if result.status == PortStatus.AVAILABLE:
                allocated[service_name] = preferred_port
                self.logger.info(f"{service_name} allocated to preferred port {preferred_port}")
                
            else:
                # Port is not available, find alternative
                alternative_port = self._find_alternative_port(service_name, preferred_port)
                
                if alternative_port:
                    allocated[service_name] = alternative_port
                    alternative_ports_used = True
                    conflicts_resolved.append(
                        f"{service_name}: {preferred_port} -> {alternative_port}"
                    )
                    self.logger.info(
                        f"{service_name} moved from {preferred_port} to {alternative_port}"
                    )
                else:
                    raise RuntimeError(
                        f"Could not find available port for {service_name} "
                        f"(preferred: {preferred_port})"
                    )
        
        self.allocated_ports = allocated
        
        return PortAllocation(
            backend=allocated.get('backend', self.default_ports['backend']),
            frontend=allocated.get('frontend', self.default_ports['frontend']),
            conflicts_resolved=conflicts_resolved,
            alternative_ports_used=alternative_ports_used
        )
    
    def _find_alternative_port(self, service_name: str, preferred_port: int) -> Optional[int]:
        """
        Find an alternative port for a service.
        
        Args:
            service_name: Name of the service
            preferred_port: Originally preferred port
            
        Returns:
            Alternative port number or None if none found
        """
        # Try ports near the preferred port first
        nearby_range = 20
        alternative = self.find_available_port(
            preferred_port + 1, 
            max_attempts=nearby_range
        )
        
        if alternative:
            return alternative
            
        # Try safe port ranges
        for start_port, end_port in self.safe_port_ranges:
            if start_port <= preferred_port <= end_port:
                continue  # Skip the range containing the preferred port
                
            alternative = self.find_available_port(start_port, max_attempts=end_port - start_port)
            if alternative:
                return alternative
                
        return None
    
    def kill_process_on_port(self, port: int, force: bool = False) -> bool:
        """
        Terminate the process using a specific port.
        
        Args:
            port: Port number
            force: If True, use SIGKILL instead of SIGTERM
            
        Returns:
            True if process was terminated successfully
        """
        process_info = self._get_process_using_port(port)
        
        if not process_info:
            self.logger.info(f"No process found using port {port}")
            return True
            
        if not process_info.can_terminate:
            self.logger.warning(
                f"Cannot safely terminate process {process_info.name} "
                f"(PID: {process_info.pid}) on port {port}"
            )
            return False
            
        try:
            process = psutil.Process(process_info.pid)
            
            if force:
                process.kill()
                self.logger.info(f"Force killed process {process_info.name} (PID: {process_info.pid})")
            else:
                process.terminate()
                self.logger.info(f"Terminated process {process_info.name} (PID: {process_info.pid})")
                
                # Wait for graceful termination
                try:
                    process.wait(timeout=5)
                except psutil.TimeoutExpired:
                    self.logger.warning(f"Process {process_info.pid} didn't terminate gracefully, killing...")
                    process.kill()
                    
            return True
            
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            self.logger.error(f"Error terminating process on port {port}: {str(e)}")
            return False
    
    def resolve_port_conflicts(self, conflicts: List[PortConflict], 
                             strategy: ConflictResolution = ConflictResolution.USE_ALTERNATIVE) -> Dict[int, int]:
        """
        Resolve port conflicts using the specified strategy.
        
        Args:
            conflicts: List of port conflicts to resolve
            strategy: Resolution strategy to use
            
        Returns:
            Dictionary mapping original ports to resolved ports
        """
        resolution_map = {}
        
        for conflict in conflicts:
            original_port = conflict.port
            
            if strategy == ConflictResolution.KILL_PROCESS:
                if ConflictResolution.KILL_PROCESS in conflict.resolution_options:
                    if self.kill_process_on_port(original_port):
                        resolution_map[original_port] = original_port
                        self.logger.info(f"Resolved conflict on port {original_port} by killing process")
                    else:
                        # Fallback to alternative port
                        alt_port = self.find_available_port(original_port + 1)
                        if alt_port:
                            resolution_map[original_port] = alt_port
                            self.logger.info(f"Fallback: moved from port {original_port} to {alt_port}")
                        
            elif strategy == ConflictResolution.USE_ALTERNATIVE:
                alt_port = self.find_available_port(original_port + 1)
                if alt_port:
                    resolution_map[original_port] = alt_port
                    self.logger.info(f"Resolved conflict: moved from port {original_port} to {alt_port}")
                    
        return resolution_map
    
    def update_configuration_files(self, port_allocation: PortAllocation) -> bool:
        """
        Update configuration files with new port assignments.
        
        Args:
            port_allocation: Port allocation result
            
        Returns:
            True if configurations were updated successfully
        """
        try:
            # Update backend configuration
            backend_config_path = Path("backend/config.json")
            if backend_config_path.exists():
                self._update_backend_config(backend_config_path, port_allocation.backend)
                
            # Update frontend configuration
            frontend_config_paths = [
                Path("frontend/vite.config.ts"),
                Path("frontend/package.json")
            ]
            
            for config_path in frontend_config_paths:
                if config_path.exists():
                    self._update_frontend_config(config_path, port_allocation.frontend)
                    
            # Update startup configuration
            if self.config_path and self.config_path.exists():
                self._update_startup_config(self.config_path, port_allocation)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating configuration files: {str(e)}")
            return False
    
    def _update_backend_config(self, config_path: Path, port: int):
        """Update backend configuration with new port."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            config['server'] = config.get('server', {})
            config['server']['port'] = port
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
            self.logger.info(f"Updated backend config: port = {port}")
            
        except Exception as e:
            self.logger.error(f"Error updating backend config: {str(e)}")
    
    def _update_frontend_config(self, config_path: Path, port: int):
        """Update frontend configuration with new port."""
        try:
            if config_path.name == "vite.config.ts":
                self._update_vite_config(config_path, port)
            elif config_path.name == "package.json":
                self._update_package_json(config_path, port)
                
        except Exception as e:
            self.logger.error(f"Error updating frontend config {config_path}: {str(e)}")
    
    def _update_vite_config(self, config_path: Path, port: int):
        """Update Vite configuration file."""
        content = config_path.read_text()
        
        # Simple regex replacement for port in vite config
        import re
        pattern = r'port:\s*\d+'
        replacement = f'port: {port}'
        
        updated_content = re.sub(pattern, replacement, content)
        
        if updated_content != content:
            config_path.write_text(updated_content)
            self.logger.info(f"Updated Vite config: port = {port}")
    
    def _update_package_json(self, config_path: Path, port: int):
        """Update package.json with new port in scripts."""
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Update dev script if it exists
        if 'scripts' in config and 'dev' in config['scripts']:
            script = config['scripts']['dev']
            # Replace port in vite command
            import re
            pattern = r'--port\s+\d+'
            replacement = f'--port {port}'
            
            updated_script = re.sub(pattern, replacement, script)
            if updated_script == script:
                # Add port if not present
                updated_script = f"{script} --port {port}"
                
            config['scripts']['dev'] = updated_script
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
            self.logger.info(f"Updated package.json: port = {port}")
    
    def _update_startup_config(self, config_path: Path, port_allocation: PortAllocation):
        """Update startup configuration with new ports."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            config['backend'] = config.get('backend', {})
            config['backend']['port'] = port_allocation.backend
            
            config['frontend'] = config.get('frontend', {})
            config['frontend']['port'] = port_allocation.frontend
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
            self.logger.info(f"Updated startup config: backend={port_allocation.backend}, frontend={port_allocation.frontend}")
            
        except Exception as e:
            self.logger.error(f"Error updating startup config: {str(e)}")
    
    def get_port_status_report(self) -> Dict:
        """
        Generate a comprehensive port status report.
        
        Returns:
            Dictionary with port status information
        """
        report = {
            'default_ports': self.default_ports,
            'allocated_ports': self.allocated_ports,
            'port_checks': {},
            'conflicts': [],
            'firewall_status': {}
        }
        
        # Check status of default and allocated ports
        all_ports = set(self.default_ports.values())
        all_ports.update(self.allocated_ports.values())
        
        for port in all_ports:
            result = self.check_port_availability(port)
            report['port_checks'][port] = {
                'status': result.status.value,
                'process': result.process.__dict__ if result.process else None,
                'error_message': result.error_message
            }
            
        # Detect conflicts
        conflicts = self.detect_port_conflicts()
        report['conflicts'] = [
            {
                'port': conflict.port,
                'process': conflict.process.__dict__,
                'resolution_options': [opt.value for opt in conflict.resolution_options]
            }
            for conflict in conflicts
        ]
        
        # Check firewall status
        ports_to_check = list(all_ports)
        report['firewall_status'] = self.check_firewall_exceptions(ports_to_check)
        
        return report