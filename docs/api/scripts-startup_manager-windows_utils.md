---
title: scripts.startup_manager.windows_utils
category: api
tags: [api, scripts]
---

# scripts.startup_manager.windows_utils

Windows-specific utilities for the startup manager.
Handles UAC, Windows services, firewall exceptions, and other Windows-specific features.

## Classes

### WindowsPermissionManager

Manages Windows permissions and UAC elevation

#### Methods

##### is_admin() -> bool

Check if the current process is running with administrator privileges

##### request_elevation(script_path: str, args: <ast.Subscript object at 0x000001942A2AB430>) -> bool

Request UAC elevation for the current script

##### can_bind_privileged_ports() -> bool

Check if we can bind to privileged ports (< 1024)

### WindowsFirewallManager

Manages Windows Firewall exceptions

#### Methods

##### check_firewall_exception(program_path: str) -> bool

Check if a program has a firewall exception

##### add_firewall_exception(program_path: str, rule_name: str, description: str) -> bool

Add a firewall exception for a program

##### add_port_exception(port: int, protocol: str, rule_name: str) -> bool

Add a firewall exception for a specific port

### WindowsServiceManager

Manages Windows services for background server management

#### Methods

##### is_service_installed(service_name: str) -> bool

Check if a Windows service is installed

##### install_service(service_name: str, display_name: str, executable_path: str, description: str) -> bool

Install a Windows service

##### start_service(service_name: str) -> bool

Start a Windows service

##### stop_service(service_name: str) -> bool

Stop a Windows service

##### uninstall_service(service_name: str) -> bool

Uninstall a Windows service

### WindowsRegistryManager

Manages Windows Registry operations for startup configuration

#### Methods

##### get_startup_entry(app_name: str) -> <ast.Subscript object at 0x000001942A260070>

Get a startup entry from the Windows Registry

##### set_startup_entry(app_name: str, command: str) -> bool

Set a startup entry in the Windows Registry

##### remove_startup_entry(app_name: str) -> bool

Remove a startup entry from the Windows Registry

### WindowsSystemInfo

Provides Windows system information

#### Methods

##### get_windows_version() -> <ast.Subscript object at 0x000001942CE3E620>

Get Windows version information

##### get_defender_status() -> <ast.Subscript object at 0x00000194275CF760>

Get Windows Defender status

##### is_wsl_available() -> bool

Check if Windows Subsystem for Linux is available

### WindowsOptimizer

Windows-specific optimizations for the startup manager

#### Methods

##### __init__(self: Any)



##### optimize_for_development(self: Any, ports: <ast.Subscript object at 0x00000194275CDD20>) -> <ast.Subscript object at 0x00000194275C5B70>

Apply Windows optimizations for development environment

##### setup_service_mode(self: Any, service_name: str) -> bool

Set up Windows service for background server management

##### _create_service_wrapper(self: Any, script_path: Path)

Create a Windows service wrapper script

