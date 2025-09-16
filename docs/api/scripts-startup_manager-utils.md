---
title: scripts.startup_manager.utils
category: api
tags: [api, scripts]
---

# scripts.startup_manager.utils

Core utility functions for system detection and path management.

## Classes

### SystemDetector

Utility class for detecting system information and capabilities.

#### Methods

##### get_system_info() -> <ast.Subscript object at 0x0000019434647070>

Get comprehensive system information.

##### is_windows() -> bool

Check if running on Windows.

##### is_admin() -> bool

Check if running with administrator privileges.

##### get_virtual_env_info() -> <ast.Subscript object at 0x0000019434646530>

Get information about the current virtual environment.

##### check_python_version(min_version: <ast.Subscript object at 0x0000019434647B50>) -> <ast.Subscript object at 0x0000019434459390>

Check if Python version meets minimum requirements.

##### check_command_available(command: str) -> <ast.Subscript object at 0x000001943445B910>

Check if a command is available in the system PATH.

##### check_node_environment() -> <ast.Subscript object at 0x0000019434458820>

Check Node.js and npm availability and versions.

### PathManager

Utility class for managing project paths and file operations.

#### Methods

##### __init__(self: Any, project_root: <ast.Subscript object at 0x0000019434458880>)



##### _ensure_project_root(self: Any) -> None

Ensure we're working from the correct project root.

##### get_backend_path(self: Any) -> Path

Get path to backend directory.

##### get_frontend_path(self: Any) -> Path

Get path to frontend directory.

##### get_scripts_path(self: Any) -> Path

Get path to scripts directory.

##### get_logs_path(self: Any) -> Path

Get path to logs directory, creating if necessary.

##### get_config_path(self: Any, config_name: str) -> Path

Get path to configuration file.

##### validate_project_structure(self: Any) -> <ast.Subscript object at 0x000001942FA6F8E0>

Validate that required project directories and files exist.

##### create_directory_structure(self: Any) -> <ast.Subscript object at 0x0000019431B400D0>

Create missing directories for the project.

##### get_relative_path(self: Any, path: Path) -> str

Get relative path from project root.

##### resolve_path(self: Any, path_str: str) -> Path

Resolve a path string relative to project root.

