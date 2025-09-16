---
title: tools.codebase-cleanup.dead_code_analyzer
category: api
tags: [api, tools]
---

# tools.codebase-cleanup.dead_code_analyzer



## Classes

### DeadFunction

Represents a dead/unused function

### DeadClass

Represents a dead/unused class

### UnusedImport

Represents an unused import

### DeadFile

Represents a dead/unused file

### DeadCodeReport

Report of dead code analysis

### DeadCodeAnalyzer

Comprehensive dead code analysis system that identifies:
- Unused functions and methods
- Dead classes that are never instantiated
- Unused imports
- Dead files that are never imported or referenced

#### Methods

##### __init__(self: Any, root_path: str, backup_dir: str)



##### analyze_dead_code(self: Any, include_tests: bool) -> DeadCodeReport

Perform comprehensive dead code analysis

Args:
    include_tests: Whether to include test files in analysis
    
Returns:
    DeadCodeReport with all findings and recommendations

##### _get_files_to_analyze(self: Any, include_tests: bool) -> <ast.Subscript object at 0x0000019427AD8A00>

Get list of code files to analyze

##### _build_project_map(self: Any, files: <ast.Subscript object at 0x0000019427ADBCD0>) -> <ast.Subscript object at 0x000001942A26FF70>

Build a map of the project structure including:
- All defined functions, classes, and variables
- All usage references
- Import relationships

##### _analyze_python_file(self: Any, file_path: Path, project_map: Dict)

Analyze Python file for definitions and usages

##### _analyze_javascript_file(self: Any, file_path: Path, project_map: Dict)

Analyze JavaScript/TypeScript file for definitions and usages

##### _find_dead_functions(self: Any, files: <ast.Subscript object at 0x000001942A2597B0>, project_map: Dict) -> <ast.Subscript object at 0x000001942A259240>

Find functions that are never called

##### _find_dead_python_functions(self: Any, file_path: Path, project_map: Dict) -> <ast.Subscript object at 0x000001942A260220>

Find dead functions in a Python file

##### _check_same_file_usage(self: Any, tree: ast.AST, func_name: str, func_node: ast.FunctionDef) -> bool

Check if function is used elsewhere in the same file

##### _find_dead_classes(self: Any, files: <ast.Subscript object at 0x0000019427B6F310>, project_map: Dict) -> <ast.Subscript object at 0x0000019427B6C4F0>

Find classes that are never instantiated

##### _find_dead_python_classes(self: Any, file_path: Path, project_map: Dict) -> <ast.Subscript object at 0x0000019428927070>

Find dead classes in a Python file

##### _find_unused_imports(self: Any, files: <ast.Subscript object at 0x00000194289272E0>) -> <ast.Subscript object at 0x00000194289274C0>

Find imports that are never used

##### _find_unused_python_imports(self: Any, file_path: Path) -> <ast.Subscript object at 0x000001942C653C40>

Find unused imports in a Python file

##### _find_dead_files(self: Any, files: <ast.Subscript object at 0x00000194285215D0>, project_map: Dict) -> <ast.Subscript object at 0x0000019428901A80>

Find files that are never imported or referenced

##### _calculate_potential_lines_removed(self: Any, dead_functions: <ast.Subscript object at 0x0000019428901C00>, dead_classes: <ast.Subscript object at 0x0000019428901CC0>, unused_imports: <ast.Subscript object at 0x0000019428901D80>, dead_files: <ast.Subscript object at 0x0000019428901E40>) -> int

Calculate potential lines of code that could be removed

##### _generate_recommendations(self: Any, dead_functions: <ast.Subscript object at 0x0000019428902AA0>, dead_classes: <ast.Subscript object at 0x0000019428902B60>, unused_imports: <ast.Subscript object at 0x0000019428902C20>, dead_files: <ast.Subscript object at 0x0000019428902CE0>) -> <ast.Subscript object at 0x00000194283B85E0>

Generate recommendations for handling dead code

##### safe_remove_dead_code(self: Any, report: DeadCodeReport) -> <ast.Subscript object at 0x00000194283BA2C0>

Safely remove dead code with comprehensive testing

Args:
    report: DeadCodeReport from analysis
    
Returns:
    Dict mapping operation to result message

##### _remove_unused_imports(self: Any, unused_imports: <ast.Subscript object at 0x00000194283BA470>) -> int

Remove unused import statements

##### _remove_dead_functions(self: Any, dead_functions: <ast.Subscript object at 0x00000194283BBB20>) -> int

Remove dead function definitions

##### _remove_dead_classes(self: Any, dead_classes: <ast.Subscript object at 0x00000194282F9EA0>) -> int

Remove dead class definitions

##### _remove_dead_files(self: Any, dead_files: <ast.Subscript object at 0x00000194282FA140>) -> int

Remove dead files

##### create_backup(self: Any, files_to_backup: <ast.Subscript object at 0x00000194282FA920>) -> str

Create backup of files before removal

##### save_report(self: Any, report: DeadCodeReport, output_path: str) -> None

Save dead code analysis report to file

