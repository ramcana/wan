---
title: tools.project-structure-analyzer.structure_analyzer
category: api
tags: [api, tools]
---

# tools.project-structure-analyzer.structure_analyzer



## Classes

### FileInfo

Information about a file in the project.

### DirectoryInfo

Information about a directory in the project.

### ProjectStructure

Complete project structure analysis.

### ProjectStructureAnalyzer

Analyzes project directory structure and identifies components.

#### Methods

##### __init__(self: Any, root_path: str, ignore_patterns: <ast.Subscript object at 0x0000019427F12E60>)

Initialize the analyzer with project root path.

##### analyze(self: Any) -> ProjectStructure

Perform complete project structure analysis.

##### _should_ignore(self: Any, name: str, parent_path: str) -> bool

Check if a file or directory should be ignored.

##### _analyze_file(self: Any, file_path: Path) -> FileInfo

Analyze a single file and determine its characteristics.

##### _analyze_directory(self: Any, dir_path: Path, files: <ast.Subscript object at 0x000001942CDDBF10>) -> DirectoryInfo

Analyze a directory and determine its characteristics.

##### _is_config_file(self: Any, name: str, extension: str) -> bool

Determine if a file is a configuration file.

##### _is_documentation_file(self: Any, name: str, extension: str) -> bool

Determine if a file is documentation.

##### _is_test_file(self: Any, name: str, path: str) -> bool

Determine if a file is a test file.

##### _is_script_file(self: Any, name: str, extension: str, path: str) -> bool

Determine if a file is a script.

##### _determine_file_purpose(self: Any, name: str, extension: str, path: str) -> <ast.Subscript object at 0x000001942791A380>

Determine the purpose of a file based on its characteristics.

##### _determine_directory_purpose(self: Any, name: str, path: str, files: <ast.Subscript object at 0x000001942791B970>, is_package: bool) -> <ast.Subscript object at 0x0000019428929BA0>

Determine the purpose of a directory.

##### _identify_entry_points(self: Any, files: <ast.Subscript object at 0x000001942892B7C0>) -> <ast.Subscript object at 0x0000019428928D60>

Identify potential application entry points.

##### _identify_main_components(self: Any, directories: <ast.Subscript object at 0x000001942892A290>) -> <ast.Subscript object at 0x000001942892B100>

Identify the main components of the project.

##### save_analysis(self: Any, analysis: ProjectStructure, output_path: str) -> None

Save analysis results to JSON file.

##### generate_summary_report(self: Any, analysis: ProjectStructure) -> str

Generate a human-readable summary report.

## Constants

### CONFIG_EXTENSIONS

Type: `unknown`

### CONFIG_NAMES

Type: `unknown`

### DOC_EXTENSIONS

Type: `unknown`

### DOC_NAMES

Type: `unknown`

### TEST_PATTERNS

Type: `unknown`

### SCRIPT_PATTERNS

Type: `unknown`

### DEFAULT_IGNORE_PATTERNS

Type: `unknown`

