---
title: tools.project-structure-analyzer.component_analyzer
category: api
tags: [api, tools]
---

# tools.project-structure-analyzer.component_analyzer



## Classes

### ImportInfo

Information about an import statement.

### ComponentDependency

Represents a dependency between components.

### ComponentInfo

Information about a project component.

### ComponentRelationshipMap

Complete map of component relationships.

### ComponentRelationshipAnalyzer

Analyzes relationships and dependencies between project components.

#### Methods

##### __init__(self: Any, root_path: str)

Initialize analyzer with project root path.

##### analyze(self: Any) -> ComponentRelationshipMap

Perform complete component relationship analysis.

##### _identify_components(self: Any)

Identify major components in the project.

##### _should_ignore_dir(self: Any, dirname: str) -> bool

Check if directory should be ignored.

##### _is_significant_component(self: Any, path: Path, files: <ast.Subscript object at 0x0000019427C75BD0>) -> bool

Determine if a directory represents a significant component.

##### _determine_component_type(self: Any, path: Path, files: <ast.Subscript object at 0x0000019427C75060>) -> str

Determine the type of component.

##### _determine_component_purpose(self: Any, name: str, files: <ast.Subscript object at 0x0000019427C74700>) -> <ast.Subscript object at 0x0000019429CBBDC0>

Determine the purpose of a component.

##### _analyze_python_imports(self: Any)

Analyze Python import statements to find dependencies.

##### _extract_imports_from_file(self: Any, file_path: Path) -> <ast.Subscript object at 0x0000019429CD74F0>

Extract import information from a Python file.

##### _process_imports(self: Any, source_component: str, file_path: str, imports: <ast.Subscript object at 0x0000019429CD72E0>)

Process imports and create dependency relationships.

##### _resolve_import_to_component(self: Any, import_info: ImportInfo) -> <ast.Subscript object at 0x0000019427FEB1C0>

Resolve an import to a component name.

##### _analyze_config_references(self: Any)

Analyze configuration file references.

##### _is_config_file(self: Any, filename: str) -> bool

Check if a file is a configuration file.

##### _analyze_file_references(self: Any)

Analyze file path references in code.

##### _find_component_for_file(self: Any, file_path: str) -> <ast.Subscript object at 0x000001942807E590>

Find which component a file belongs to.

##### _analyze_api_interactions(self: Any)

Analyze API calls and service interactions.

##### _find_api_handler_component(self: Any, endpoint: str) -> <ast.Subscript object at 0x00000194280C8160>

Find which component handles an API endpoint.

##### _calculate_component_metrics(self: Any)

Calculate complexity and importance metrics for components.

##### _detect_circular_dependencies(self: Any) -> <ast.Subscript object at 0x000001942CBC0100>

Detect circular dependencies between components.

##### _identify_critical_components(self: Any) -> <ast.Subscript object at 0x0000019427C41480>

Identify components that many others depend on.

##### _identify_isolated_components(self: Any) -> <ast.Subscript object at 0x0000019427C425C0>

Identify components with no or very few dependencies.

##### _identify_entry_points(self: Any) -> <ast.Subscript object at 0x0000019427C43760>

Identify components that serve as entry points.

##### save_analysis(self: Any, analysis: ComponentRelationshipMap, output_path: str)

Save analysis results to JSON file.

##### generate_summary_report(self: Any, analysis: ComponentRelationshipMap) -> str

Generate a human-readable summary report.

