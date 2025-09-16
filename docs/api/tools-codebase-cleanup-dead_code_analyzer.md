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

##### _get_files_to_analyze(self: Any, include_tests: bool) -> <ast.Subscript object at 0x0000019431ABAC50>

Get list of code files to analyze

##### _build_project_map(self: Any, files: <ast.Subscript object at 0x0000019431ABA6B0>) -> <ast.Subscript object at 0x000001942FC36140>

Build a map of the project structure including:
- All defined functions, classes, and variables
- All usage references
- Import relationships

##### _analyze_python_file(self: Any, file_path: Path, project_map: Dict)

Analyze Python file for definitions and usages

##### _analyze_javascript_file(self: Any, file_path: Path, project_map: Dict)

Analyze JavaScript/TypeScript file for definitions and usages

##### _find_dead_functions(self: Any, files: <ast.Subscript object at 0x0000019434131F00>, project_map: Dict) -> <ast.Subscript object at 0x000001943448F070>

Find functions that are never called

##### _find_dead_python_functions(self: Any, file_path: Path, project_map: Dict) -> <ast.Subscript object at 0x000001942F220A90>

Find dead functions in a Python file

##### _check_same_file_usage(self: Any, tree: ast.AST, func_name: str, func_node: ast.FunctionDef) -> bool

Check if function is used elsewhere in the same file

##### _find_dead_classes(self: Any, files: <ast.Subscript object at 0x000001942F02C6D0>, project_map: Dict) -> <ast.Subscript object at 0x000001942F02D600>

Find classes that are never instantiated

##### _find_dead_python_classes(self: Any, file_path: Path, project_map: Dict) -> <ast.Subscript object at 0x000001942EFC3C10>

Find dead classes in a Python file

##### _find_unused_imports(self: Any, files: <ast.Subscript object at 0x000001942EFC3AC0>) -> <ast.Subscript object at 0x000001942EFC35E0>

Find imports that are never used

##### _find_unused_python_imports(self: Any, file_path: Path) -> <ast.Subscript object at 0x000001942F805030>

Find unused imports in a Python file

##### _find_dead_files(self: Any, files: <ast.Subscript object at 0x000001942F805180>, project_map: Dict) -> <ast.Subscript object at 0x000001942F8387F0>

Find files that are never imported or referenced

##### _calculate_potential_lines_removed(self: Any, dead_functions: <ast.Subscript object at 0x000001942F838970>, dead_classes: <ast.Subscript object at 0x000001942F838A30>, unused_imports: <ast.Subscript object at 0x000001942F838AF0>, dead_files: <ast.Subscript object at 0x000001942F838BB0>) -> int

Calculate potential lines of code that could be removed

##### _generate_recommendations(self: Any, dead_functions: <ast.Subscript object at 0x000001942F839810>, dead_classes: <ast.Subscript object at 0x000001942F8398D0>, unused_imports: <ast.Subscript object at 0x000001942F839990>, dead_files: <ast.Subscript object at 0x000001942F839A50>) -> <ast.Subscript object at 0x000001942F83B310>

Generate recommendations for handling dead code

##### safe_remove_dead_code(self: Any, report: DeadCodeReport) -> <ast.Subscript object at 0x0000019431C0D030>

Safely remove dead code with comprehensive testing

Args:
    report: DeadCodeReport from analysis
    
Returns:
    Dict mapping operation to result message

##### _remove_unused_imports(self: Any, unused_imports: <ast.Subscript object at 0x0000019431C0D1E0>) -> int

Remove unused import statements

##### _remove_dead_functions(self: Any, dead_functions: <ast.Subscript object at 0x0000019431C0E890>) -> int

Remove dead function definitions

##### _remove_dead_classes(self: Any, dead_classes: <ast.Subscript object at 0x0000019431B7D000>) -> int

Remove dead class definitions

##### _remove_dead_files(self: Any, dead_files: <ast.Subscript object at 0x0000019431B7E4D0>) -> int

Remove dead files

##### create_backup(self: Any, files_to_backup: <ast.Subscript object at 0x0000019431B7E770>) -> str

Create backup of files before removal

##### save_report(self: Any, report: DeadCodeReport, output_path: str) -> None

Save dead code analysis report to file

