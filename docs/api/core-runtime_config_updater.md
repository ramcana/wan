---
title: core.runtime_config_updater
category: api
tags: [api, core]
---

# core.runtime_config_updater

Runtime Configuration Update System

Provides hot-reload capabilities for configuration changes without requiring
application restart, with proper validation and rollback mechanisms.

## Classes

### RuntimeConfigurationUpdater

Manages runtime configuration updates without application restart

#### Methods

##### __init__(self: Any, config_manager: <ast.Subscript object at 0x000001942A18FBE0>)



##### start_monitoring(self: Any, watch_paths: <ast.Subscript object at 0x000001942A18EF50>) -> None

Start monitoring configuration files for changes

Args:
    watch_paths: List of paths to monitor, defaults to config file directory

##### stop_monitoring(self: Any) -> None

Stop monitoring configuration files

##### add_update_callback(self: Any, section: str, callback: Callable) -> None

Add callback for configuration updates

Args:
    section: Configuration section ('user_preferences', 'admin_policies', 'feature_flags', 'any')
    callback: Async function to call on updates

##### remove_update_callback(self: Any, section: str, callback: Callable) -> None

Remove configuration update callback

##### get_rollback_history(self: Any) -> <ast.Subscript object at 0x000001942880C940>

Get list of available rollback points

##### _create_rollback_point(self: Any) -> None

Create a rollback point with current configuration

##### __del__(self: Any)

Cleanup when updater is destroyed

### ConfigurationChangeHandler

File system event handler for configuration file changes

#### Methods

##### __init__(self: Any, updater: RuntimeConfigurationUpdater)



##### on_modified(self: Any, event: Any)

Handle file modification events

## Constants

### WATCHDOG_AVAILABLE

Type: `bool`

Value: `True`

### WATCHDOG_AVAILABLE

Type: `bool`

Value: `False`

