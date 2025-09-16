---
title: tools.config-analyzer.config_landscape_analyzer
category: api
tags: [api, tools]
---

# tools.config-analyzer.config_landscape_analyzer



## Classes

### ConfigFile

Represents a configuration file in the project.

#### Methods

##### __post_init__(self: Any)



### ConfigConflict

Represents a conflict between configuration settings.

### ConfigDependency

Represents a dependency relationship between configs.

### ConfigAnalysisReport

Complete analysis report of the configuration landscape.

### ConfigLandscapeAnalyzer

Analyzes the configuration landscape of a project.

#### Methods

##### __init__(self: Any, project_root: Path)



##### scan_project(self: Any) -> <ast.Subscript object at 0x0000019431B37040>

Scan the project for all configuration files.

##### _is_config_file(self: Any, file_path: Path) -> bool

Determine if a file is a configuration file.

##### _analyze_config_file(self: Any, file_path: Path, relative_path: Path) -> <ast.Subscript object at 0x0000019431B35780>

Analyze a single configuration file.

##### _determine_file_type(self: Any, file_path: Path) -> str

Determine the configuration file type.

##### _load_config_content(self: Any, file_path: Path, file_type: str) -> <ast.Subscript object at 0x0000019431A9F580>

Load configuration file content.

##### _parse_env_file(self: Any, content: str) -> <ast.Subscript object at 0x0000019431A9C160>

Parse environment file content.

##### _extract_settings(self: Any, content: <ast.Subscript object at 0x000001943028BF70>, prefix: str) -> <ast.Subscript object at 0x000001943028B3D0>

Extract all setting keys from configuration content.

##### analyze_dependencies(self: Any) -> <ast.Subscript object at 0x00000194302893F0>

Analyze dependencies between configuration files.

##### detect_conflicts(self: Any) -> <ast.Subscript object at 0x000001942F393B20>

Detect conflicts between configuration settings.

##### _determine_conflict_severity(self: Any, setting_name: str, values: <ast.Subscript object at 0x000001942F393C40>) -> str

Determine the severity of a configuration conflict.

##### find_duplicate_settings(self: Any) -> <ast.Subscript object at 0x00000194340B30D0>

Find duplicate settings across configuration files.

##### generate_recommendations(self: Any) -> <ast.Subscript object at 0x00000194302A2E00>

Generate recommendations for configuration consolidation.

##### create_migration_plan(self: Any) -> <ast.Subscript object at 0x0000019434136C50>

Create a migration plan for configuration consolidation.

##### _suggest_target_section(self: Any, config_file: ConfigFile) -> str

Suggest which section of unified config this file should map to.

##### _determine_migration_priority(self: Any, config_file: ConfigFile) -> str

Determine migration priority for a configuration file.

##### _requires_manual_review(self: Any, config_file: ConfigFile) -> bool

Determine if a config file requires manual review during migration.

##### _has_complex_structure(self: Any, content: Any, depth: int) -> bool

Check if configuration has complex nested structure.

##### generate_report(self: Any) -> ConfigAnalysisReport

Generate comprehensive configuration analysis report.

## Constants

### CONFIG_EXTENSIONS

Type: `unknown`

### CONFIG_PATTERNS

Type: `unknown`

