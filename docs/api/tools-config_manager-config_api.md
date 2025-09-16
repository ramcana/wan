---
title: tools.config_manager.config_api
category: api
tags: [api, tools]
---

# tools.config_manager.config_api

Configuration API and Management System

This module provides a comprehensive API for configuration management,
including get/set operations, hot-reloading, and change notifications.

## Classes

### ConfigChangeEvent

Represents a configuration change event

### ConfigurationChangeHandler

Handles configuration change notifications

#### Methods

##### __init__(self: Any)



##### register_callback(self: Any, callback: <ast.Subscript object at 0x000001942F6D7160>)

Register a callback for configuration changes

##### unregister_callback(self: Any, callback: <ast.Subscript object at 0x000001942F6D6E00>)

Unregister a configuration change callback

##### notify_change(self: Any, event: ConfigChangeEvent)

Notify all registered callbacks of a configuration change

### ConfigFileWatcher

Watches configuration files for changes

#### Methods

##### __init__(self: Any, config_api: ConfigurationAPI)



##### on_modified(self: Any, event: Any)

Handle file modification events

### ConfigurationAPI

Comprehensive configuration API with hot-reloading and change notifications

#### Methods

##### __init__(self: Any, config_file_path: <ast.Subscript object at 0x000001942F6D5930>, auto_reload: bool, validate_changes: bool)



##### _load_config(self: Any) -> UnifiedConfig

Load configuration from file or create default

##### _setup_file_watcher(self: Any)

Setup file system watcher for configuration changes

##### _reload_from_file(self: Any)

Reload configuration from file

##### _detect_and_notify_changes(self: Any, old_config: UnifiedConfig, new_config: UnifiedConfig, source: str)

Detect changes between configurations and notify callbacks

##### _compare_dicts(self: Any, old_dict: Dict, new_dict: Dict, path_prefix: str, source: str)

Recursively compare dictionaries and generate change events

##### get_config(self: Any, path: <ast.Subscript object at 0x000001942F72C820>) -> Any

Get configuration value by path

Args:
    path: Dot-separated path to configuration value (e.g., 'api.port')
         If None, returns entire configuration

Returns:
    Configuration value or entire configuration

##### set_config(self: Any, path: str, value: Any, validate: bool) -> bool

Set configuration value by path

Args:
    path: Dot-separated path to configuration value
    value: New value to set
    validate: Whether to validate the change (uses instance default if None)

Returns:
    True if successful, False otherwise

##### update_config(self: Any, updates: <ast.Subscript object at 0x000001942F6B0250>, validate: bool) -> <ast.Subscript object at 0x000001942F6CBC10>

Update multiple configuration values

Args:
    updates: Dictionary of path -> value updates
    validate: Whether to validate changes

Returns:
    Dictionary of path -> success status

##### reload_config(self: Any) -> bool

Manually reload configuration from file

Returns:
    True if successful, False otherwise

##### save_config(self: Any, file_path: <ast.Subscript object at 0x000001942F6CB580>) -> bool

Save current configuration to file

Args:
    file_path: Path to save to (uses default if None)

Returns:
    True if successful, False otherwise

##### validate_current_config(self: Any) -> ValidationResult

Validate the current configuration

Returns:
    Validation result

##### register_change_callback(self: Any, callback: <ast.Subscript object at 0x000001942F6CA5C0>)

Register a callback for configuration changes

##### unregister_change_callback(self: Any, callback: <ast.Subscript object at 0x0000019431B1D030>)

Unregister a configuration change callback

##### get_config_info(self: Any) -> <ast.Subscript object at 0x0000019431B4BFA0>

Get information about the current configuration

Returns:
    Dictionary with configuration metadata

##### apply_environment_overrides(self: Any, environment: str) -> bool

Apply environment-specific configuration overrides

Args:
    environment: Environment name (development, staging, production, testing)

Returns:
    True if successful, False otherwise

##### export_config(self: Any, format: str) -> str

Export current configuration as string

Args:
    format: Export format ('yaml', 'json')

Returns:
    Configuration as formatted string

##### import_config(self: Any, config_str: str, format: str) -> bool

Import configuration from string

Args:
    config_str: Configuration as string
    format: Format of the string ('yaml', 'json', 'auto')

Returns:
    True if successful, False otherwise

##### shutdown(self: Any)

Shutdown the configuration API and cleanup resources

##### __enter__(self: Any)

Context manager entry

##### __exit__(self: Any, exc_type: Any, exc_val: Any, exc_tb: Any)

Context manager exit

