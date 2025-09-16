---
title: tools.code_quality.validators.type_hint_validator
category: api
tags: [api, tools]
---

# tools.code_quality.validators.type_hint_validator

Type hint validation and enforcement.

## Classes

### TypeHintValidator

Validates type hints for functions and methods.

#### Methods

##### __init__(self: Any, config: QualityConfig)

Initialize validator with configuration.

##### validate_type_hints(self: Any, file_path: Path, tree: ast.AST) -> <ast.Subscript object at 0x00000194340B02B0>

Validate type hints in the given AST.

Returns:
    Tuple of (issues, metrics)

##### _validate_function_type_hints(self: Any, file_path: Path, node: ast.FunctionDef) -> <ast.Subscript object at 0x000001942F3A4D60>

Validate type hints for a specific function.

##### _run_mypy_check(self: Any, file_path: Path) -> <ast.Subscript object at 0x000001942F331480>

Run mypy type checker on the file.

##### _parse_mypy_output(self: Any, file_path: Path, line: str) -> QualityIssue

Parse mypy output line into QualityIssue.

##### _is_void_function(self: Any, node: ast.FunctionDef) -> bool

Check if function is void (doesn't return a value).

##### _check_inconsistent_annotations(self: Any, file_path: Path, node: ast.FunctionDef) -> <ast.Subscript object at 0x000001942F3044C0>

Check for inconsistent type annotations.

##### _check_complex_annotations(self: Any, file_path: Path, node: ast.FunctionDef) -> <ast.Subscript object at 0x000001942F3DA980>

Check for overly complex type annotations.

##### _is_complex_annotation(self: Any, annotation: ast.AST) -> bool

Check if type annotation is overly complex.

##### _calculate_annotation_complexity(self: Any, node: ast.AST) -> int

Calculate complexity score for type annotation.

