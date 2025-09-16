---
title: tools.health-checker.checkers.configuration_health_checker
category: api
tags: [api, tools]
---

# tools.health-checker.checkers.configuration_health_checker



## Classes

### ConfigurationHealthChecker

Checks the health of project configuration

#### Methods

##### __init__(self: Any, config: HealthConfig)



##### check_health(self: Any) -> ComponentHealth

Check configuration health

##### _discover_configuration_files(self: Any) -> <ast.Subscript object at 0x000001942CE2F730>

Discover configuration files throughout the project

##### _check_scattered_configuration(self: Any, config_files: <ast.Subscript object at 0x000001942CE2CEE0>) -> <ast.Subscript object at 0x000001942CE2E560>

Check for configuration files outside the config directory

##### _check_duplicate_configurations(self: Any, config_files: <ast.Subscript object at 0x000001942CE2D960>) -> <ast.Subscript object at 0x000001942CE22080>

Check for duplicate configuration keys across files

##### _extract_config_keys(self: Any, config_file: Path) -> <ast.Subscript object at 0x000001942CE219F0>

Extract configuration keys from a file

##### _flatten_dict_keys(self: Any, data: <ast.Subscript object at 0x000001942CE236D0>, prefix: str) -> <ast.Subscript object at 0x000001942CE22710>

Flatten nested dictionary keys

##### _validate_configurations(self: Any, config_files: <ast.Subscript object at 0x000001942CE21900>) -> <ast.Subscript object at 0x0000019427B89780>

Validate configuration files for syntax errors

##### _check_missing_configurations(self: Any) -> <ast.Subscript object at 0x0000019427B883D0>

Check for missing essential configuration files

##### _check_configuration_security(self: Any, config_files: <ast.Subscript object at 0x0000019427B88C40>) -> <ast.Subscript object at 0x0000019427B8A440>

Check for security issues in configuration files

##### _check_unified_configuration(self: Any) -> <ast.Subscript object at 0x0000019427B8A7A0>

Check if unified configuration system exists

##### _calculate_configuration_score(self: Any, metrics: <ast.Subscript object at 0x0000019427B8ADD0>, issues: <ast.Subscript object at 0x0000019427B890F0>) -> float

Calculate configuration health score

##### _determine_status(self: Any, score: float) -> str

Determine health status from score

