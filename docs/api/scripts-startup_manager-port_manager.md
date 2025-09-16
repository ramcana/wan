---
title: scripts.startup_manager.port_manager
category: api
tags: [api, scripts]
---

# scripts.startup_manager.port_manager

Port Manager Component for WAN22 Server Startup Management System

This module handles port availability checking, conflict detection, and resolution
with Windows-specific handling for firewall and permission issues.

## Classes

### PortStatus

Port availability status

### ConflictResolution

Port conflict resolution strategies

### ProcessInfo

Information about a process using a port

### PortConflict

Information about a port conflict

### PortAllocation

Result of port allocation

### PortCheckResult

Result of port availability check

### PortManager

Manages port allocation, conflict detection, and resolution for server startup.

Handles Windows-specific issues like firewall blocking and provides intelligent
port allocation with conflict resolution.

#### Methods

##### __init__(self: Any, config_path: <ast.Subscript object at 0x0000019434077A90>)



##### check_port_availability(self: Any, port: int, host: str) -> PortCheckResult

Check if a specific port is available for binding.

Args:
    port: Port number to check
    host: Host address to check (default: localhost)
    
Returns:
    PortCheckResult with status and details

##### _get_process_using_port(self: Any, port: int) -> <ast.Subscript object at 0x0000019434319E40>

Get information about the process using a specific port.

Args:
    port: Port number to check
    
Returns:
    ProcessInfo if port is in use, None otherwise

##### _can_safely_terminate_process(self: Any, process: psutil.Process) -> bool

Determine if a process can be safely terminated.

Args:
    process: Process to check
    
Returns:
    True if process can be safely terminated

##### find_available_port(self: Any, start_port: int, max_attempts: int) -> <ast.Subscript object at 0x00000194345E2FE0>

Find the next available port starting from the given port.

Args:
    start_port: Starting port number
    max_attempts: Maximum number of ports to try
    
Returns:
    Available port number or None if none found

##### find_available_ports_in_range(self: Any, port_range: <ast.Subscript object at 0x00000194345E0F10>) -> <ast.Subscript object at 0x00000194345E1780>

Find all available ports in a given range.

Args:
    port_range: Tuple of (start_port, end_port)
    
Returns:
    List of available port numbers

##### detect_port_conflicts(self: Any, required_ports: <ast.Subscript object at 0x00000194345E2E30>) -> <ast.Subscript object at 0x00000194345E18D0>

Detect conflicts for required ports.

Args:
    required_ports: Dictionary of service names to port numbers
    
Returns:
    List of detected port conflicts

##### check_firewall_exceptions(self: Any, ports: <ast.Subscript object at 0x00000194345E2C80>) -> <ast.Subscript object at 0x00000194345E12A0>

Check if ports have firewall exceptions (Windows-specific).

Args:
    ports: List of port numbers to check
    
Returns:
    Dictionary mapping port numbers to exception status

##### suggest_firewall_fix(self: Any, port: int, service_name: str) -> str

Generate Windows firewall exception command for a port.

Args:
    port: Port number
    service_name: Name for the firewall rule
    
Returns:
    Command string to add firewall exception

##### allocate_ports(self: Any, required_ports: <ast.Subscript object at 0x00000194345E1090>) -> PortAllocation

Allocate ports for services, resolving conflicts automatically.

Args:
    required_ports: Dictionary of service names to preferred port numbers
    
Returns:
    PortAllocation with assigned ports and resolution details

##### _find_alternative_port(self: Any, service_name: str, preferred_port: int) -> <ast.Subscript object at 0x0000019431B5F100>

Find an alternative port for a service.

Args:
    service_name: Name of the service
    preferred_port: Originally preferred port
    
Returns:
    Alternative port number or None if none found

##### kill_process_on_port(self: Any, port: int, force: bool) -> bool

Terminate the process using a specific port.

Args:
    port: Port number
    force: If True, use SIGKILL instead of SIGTERM
    
Returns:
    True if process was terminated successfully

##### resolve_port_conflicts(self: Any, conflicts: <ast.Subscript object at 0x00000194341FA4A0>, strategy: ConflictResolution) -> <ast.Subscript object at 0x00000194341F8490>

Resolve port conflicts using the specified strategy.

Args:
    conflicts: List of port conflicts to resolve
    strategy: Resolution strategy to use
    
Returns:
    Dictionary mapping original ports to resolved ports

##### update_configuration_files(self: Any, port_allocation: PortAllocation) -> bool

Update configuration files with new port assignments.

Args:
    port_allocation: Port allocation result
    
Returns:
    True if configurations were updated successfully

##### _update_backend_config(self: Any, config_path: Path, port: int)

Update backend configuration with new port.

##### _update_frontend_config(self: Any, config_path: Path, port: int)

Update frontend configuration with new port.

##### _update_vite_config(self: Any, config_path: Path, port: int)

Update Vite configuration file.

##### _update_package_json(self: Any, config_path: Path, port: int)

Update package.json with new port in scripts.

##### _update_startup_config(self: Any, config_path: Path, port_allocation: PortAllocation)

Update startup configuration with new ports.

##### get_port_status_report(self: Any) -> Dict

Generate a comprehensive port status report.

Returns:
    Dictionary with port status information

## Constants

### AVAILABLE

Type: `str`

Value: `available`

### OCCUPIED

Type: `str`

Value: `occupied`

### BLOCKED

Type: `str`

Value: `blocked`

### UNKNOWN

Type: `str`

Value: `unknown`

### KILL_PROCESS

Type: `str`

Value: `kill_process`

### USE_ALTERNATIVE

Type: `str`

Value: `use_alternative`

### SKIP

Type: `str`

Value: `skip`

