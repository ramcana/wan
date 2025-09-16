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



##### analyze_core_configs(self: Any) -> <ast.Subscript object at 0x00000194281DD270>

Analyze core configuration files and their relationships.

##### _analyze_config_file(self: Any, file_path: str, purpose: str, component: str, priority: str) -> ConfigMapping

Analyze a single configuration file.

##### _load_file_content(self: Any, file_path: Path) -> <ast.Subscript object at 0x00000194281962F0>

Load configuration file content.

##### _parse_env_file(self: Any, content: str) -> <ast.Subscript object at 0x00000194281957B0>

Parse environment file content.

##### _count_settings(self: Any, content: <ast.Subscript object at 0x0000019428195600>) -> int

Count total settings in configuration.

##### _extract_key_settings(self: Any, content: <ast.Subscript object at 0x0000019428194D30>) -> <ast.Subscript object at 0x000001942818D180>

Extract key configuration settings.

##### _find_dependencies(self: Any, file_path: str, content: <ast.Subscript object at 0x000001942818E050>) -> <ast.Subscript object at 0x000001942818C9D0>

Find files that this configuration depends on.

##### _find_usage(self: Any, file_path: str) -> <ast.Subscript object at 0x00000194279B95A0>

Find components that use this configuration file.

##### generate_consolidation_plan(self: Any) -> <ast.Subscript object at 0x00000194279D6AA0>

Generate a consolidation plan based on the analysis.

##### generate_report(self: Any) -> <ast.Subscript object at 0x00000194279D53C0>

Generate comprehensive dependency mapping report.

