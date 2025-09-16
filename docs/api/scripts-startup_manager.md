---
title: scripts.startup_manager
category: api
tags: [api, scripts]
---

# scripts.startup_manager

WAN22 Server Startup Manager

Main entry point for the intelligent server startup system.
Provides CLI interface and orchestrates all startup components.

## Classes

### StartupManager

Main startup manager orchestrator

#### Methods

##### __init__(self: Any, cli_interface: InteractiveCLI, config: StartupConfig)



##### run_startup_sequence(self: Any, backend_port: <ast.Subscript object at 0x000001943451C5E0>, frontend_port: <ast.Subscript object at 0x000001943451E620>) -> bool

Run the complete startup sequence

##### _validate_environment(self: Any) -> bool

Validate the development environment

##### _manage_ports(self: Any, backend_port: <ast.Subscript object at 0x0000019433F919F0>, frontend_port: <ast.Subscript object at 0x0000019433F91930>) -> <ast.Subscript object at 0x00000194340F2B30>

Manage port allocation and conflicts

##### _start_processes(self: Any, ports: <ast.Subscript object at 0x00000194340F2A70>) -> bool

Start the server processes

##### _verify_health(self: Any, ports: <ast.Subscript object at 0x0000019433FD6FE0>) -> bool

Verify server health

##### _display_success_summary(self: Any, ports: <ast.Subscript object at 0x00000194346456F0>, stats: <ast.Subscript object at 0x0000019434647DC0>, optimization_suggestions: <ast.Subscript object at 0x0000019434647EE0>)

Display startup success summary with performance metrics and optimization suggestions

##### _attempt_environment_fixes(self: Any, issues: list) -> bool

Attempt to automatically fix environment issues

