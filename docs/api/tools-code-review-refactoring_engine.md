---
title: tools.code-review.refactoring_engine
category: api
tags: [api, tools]
---

# tools.code-review.refactoring_engine

Refactoring Recommendation Engine

This module provides intelligent refactoring recommendations based on code analysis
and quality metrics to help improve code maintainability and performance.

## Classes

### RefactoringType

Types of refactoring recommendations

### RefactoringPattern

Represents a refactoring pattern

### RefactoringSuggestion

Represents a specific refactoring suggestion

### RefactoringEngine

Main refactoring recommendation engine

#### Methods

##### __init__(self: Any, project_root: str)



##### _load_refactoring_patterns(self: Any) -> <ast.Subscript object at 0x0000019427B5F370>

Load refactoring patterns from configuration

##### analyze_file(self: Any, file_path: str) -> <ast.Subscript object at 0x0000019427B5DE10>

Analyze a file and generate refactoring suggestions

##### _analyze_methods(self: Any, file_path: str, tree: ast.AST, lines: <ast.Subscript object at 0x0000019427B5DBD0>) -> <ast.Subscript object at 0x0000019428D9B010>

Analyze methods for refactoring opportunities

##### _analyze_classes(self: Any, file_path: str, tree: ast.AST, lines: <ast.Subscript object at 0x0000019427A941C0>) -> <ast.Subscript object at 0x000001942C7FFEB0>

Analyze classes for refactoring opportunities

##### _analyze_conditionals(self: Any, file_path: str, tree: ast.AST, lines: <ast.Subscript object at 0x000001942C7FEDA0>) -> <ast.Subscript object at 0x000001942C7FE320>

Analyze conditional statements for refactoring opportunities

##### _analyze_naming(self: Any, file_path: str, tree: ast.AST, lines: <ast.Subscript object at 0x00000194283DB010>) -> <ast.Subscript object at 0x00000194283D96C0>

Analyze naming for improvement opportunities

##### _analyze_imports(self: Any, file_path: str, tree: ast.AST, lines: <ast.Subscript object at 0x00000194283D87F0>) -> <ast.Subscript object at 0x000001942C52CFA0>

Analyze imports for optimization opportunities

##### _get_method_length(self: Any, node: ast.FunctionDef) -> int

Calculate method length in lines

##### _get_class_length(self: Any, node: ast.ClassDef) -> int

Calculate class length in lines

##### _calculate_complexity(self: Any, node: ast.FunctionDef) -> int

Calculate cyclomatic complexity

##### _get_nesting_level(self: Any, node: ast.If) -> int

Get nesting level of conditional

##### _get_boolean_complexity(self: Any, node: ast.AST) -> int

Calculate boolean expression complexity

##### _is_poor_name(self: Any, name: str) -> bool

Check if a name is poorly chosen

##### _suggest_better_name(self: Any, name: str) -> str

Suggest a better name

##### _get_node_code(self: Any, node: ast.AST, lines: <ast.Subscript object at 0x000001942CBC74F0>) -> str

Extract code for a given AST node

##### _generate_extract_method_suggestion(self: Any, node: ast.FunctionDef, lines: <ast.Subscript object at 0x000001942CBC6F80>) -> str

Generate extract method refactoring suggestion

##### _generate_complexity_reduction_suggestion(self: Any, node: ast.FunctionDef, lines: <ast.Subscript object at 0x000001942CBC7AC0>) -> str

Generate complexity reduction suggestion

##### _generate_extract_class_suggestion(self: Any, node: ast.ClassDef, lines: <ast.Subscript object at 0x000001942CBC51E0>) -> str

Generate extract class refactoring suggestion

##### _generate_conditional_simplification(self: Any, node: ast.If, lines: <ast.Subscript object at 0x000001942CD5FA00>) -> str

Generate conditional simplification suggestion

##### generate_suggestions_report(self: Any, output_path: str)

Generate refactoring suggestions report

## Constants

### EXTRACT_METHOD

Type: `str`

Value: `extract_method`

### EXTRACT_CLASS

Type: `str`

Value: `extract_class`

### RENAME_VARIABLE

Type: `str`

Value: `rename_variable`

### SIMPLIFY_CONDITIONAL

Type: `str`

Value: `simplify_conditional`

### REMOVE_DUPLICATION

Type: `str`

Value: `remove_duplication`

### OPTIMIZE_IMPORTS

Type: `str`

Value: `optimize_imports`

### ADD_TYPE_HINTS

Type: `str`

Value: `add_type_hints`

### IMPROVE_NAMING

Type: `str`

Value: `improve_naming`

### REDUCE_COMPLEXITY

Type: `str`

Value: `reduce_complexity`

### OPTIMIZE_PERFORMANCE

Type: `str`

Value: `optimize_performance`

