---
title: tools.config-analyzer.config_dependency_mapper
category: api
tags: [api, tools]
---

# tools.config-analyzer.config_dependency_mapper



## Classes

### ConfigMapping

Maps configuration files to their purpose and relationships.

### ConfigDependencyMapper

Maps configuration dependencies for consolidation planning.

#### Methods

##### __init__(self: Any, project_root: Path)



##### analyze_core_configs(self: Any) -> <ast.Subscript object at 0x00000194302C1270>

Analyze core configuration files and their relationships.

##### _analyze_config_file(self: Any, file_path: str, purpose: str, component: str, priority: str) -> ConfigMapping

Analyze a single configuration file.

##### _load_file_content(self: Any, file_path: Path) -> <ast.Subscript object at 0x00000194302EE2F0>

Load configuration file content.

##### _parse_env_file(self: Any, content: str) -> <ast.Subscript object at 0x00000194302ED7B0>

Parse environment file content.

##### _count_settings(self: Any, content: <ast.Subscript object at 0x00000194302ED600>) -> int

Count total settings in configuration.

##### _extract_key_settings(self: Any, content: <ast.Subscript object at 0x00000194302ECD30>) -> <ast.Subscript object at 0x00000194302EC130>

Extract key configuration settings.

##### _find_dependencies(self: Any, file_path: str, content: <ast.Subscript object at 0x00000194302A3F40>) -> <ast.Subscript object at 0x00000194302A0850>

Find files that this configuration depends on.

##### _find_usage(self: Any, file_path: str) -> <ast.Subscript object at 0x0000019434137940>

Find components that use this configuration file.

##### generate_consolidation_plan(self: Any) -> <ast.Subscript object at 0x0000019434076B60>

Generate a consolidation plan based on the analysis.

##### generate_report(self: Any) -> <ast.Subscript object at 0x00000194340754E0>

Generate comprehensive dependency mapping report.

