---
title: tools.dev-feedback.config_watcher
category: api
tags: [api, tools]
---

# tools.dev-feedback.config_watcher

Configuration Watcher with Hot-Reloading

This module provides hot-reloading for configuration changes during development.

## Classes

### ConfigChange

Configuration change event

### ServiceConfig

Service configuration for reloading

### ConfigFileHandler

File system event handler for configuration watching

#### Methods

##### __init__(self: Any, watcher: ConfigWatcher)



##### on_modified(self: Any, event: Any)



##### on_created(self: Any, event: Any)



##### on_deleted(self: Any, event: Any)



### ConfigWatcher

Watch configuration files and provide hot-reloading

#### Methods

##### __init__(self: Any, project_root: <ast.Subscript object at 0x000001942A28B9A0>)



##### _load_initial_configs(self: Any)

Load initial configuration files into cache

##### _read_config_file(self: Any, file_path: Path) -> Any

Read and parse configuration file

##### is_config_file(self: Any, file_path: Path) -> bool

Check if file is a configuration file

##### handle_config_change(self: Any, file_path: Path, change_type: str)

Handle configuration file change with debouncing

##### _process_config_change(self: Any, file_path: Path, change_type: str)

Process configuration change after debouncing

##### _validate_config(self: Any, file_path: Path, content: Any) -> <ast.Subscript object at 0x0000019428120F70>

Validate configuration content

##### _validate_unified_config(self: Any, config: <ast.Subscript object at 0x00000194281238E0>) -> <ast.Subscript object at 0x0000019428122F80>

Validate unified configuration

##### _validate_package_json(self: Any, config: <ast.Subscript object at 0x0000019428121630>) -> <ast.Subscript object at 0x0000019428120370>

Validate package.json

##### _notify_change_handlers(self: Any, change: ConfigChange)

Notify registered change handlers

##### _handle_service_reloading(self: Any, change: ConfigChange)

Handle service reloading based on configuration changes

##### _should_reload_service(self: Any, service_config: ServiceConfig, changed_file: Path) -> bool

Check if service should be reloaded for the changed file

##### _reload_service(self: Any, service_name: str)

Reload a service

##### register_change_handler(self: Any, handler: <ast.Subscript object at 0x0000019428927E20>)

Register a configuration change handler

##### register_service(self: Any, service_config: ServiceConfig)

Register a service for automatic reloading

##### get_config(self: Any, file_path: str) -> <ast.Subscript object at 0x0000019428925E10>

Get cached configuration content

##### get_all_configs(self: Any) -> <ast.Subscript object at 0x0000019428925B70>

Get all cached configurations

##### start_watching(self: Any)

Start watching for configuration changes

##### stop_watching(self: Any)

Stop watching for configuration changes

