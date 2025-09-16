---
title: tools.code_quality.validators.documentation_validator
category: api
tags: [api, tools]
---

# tools.code_quality.validators.documentation_validator

Documentation completeness validator.

## Classes

### DocumentationValidator

Validates documentation completeness for functions, classes, and modules.

#### Methods

##### __init__(self: Any, config: QualityConfig)

Initialize validator with configuration.

##### validate_documentation(self: Any, file_path: Path, tree: ast.AST) -> <ast.Subscript object at 0x000001942C6E2C50>

Validate documentation completeness in the given AST.

Returns:
    Tuple of (issues, metrics)

##### _validate_function_documentation(self: Any, file_path: Path, node: ast.FunctionDef) -> <ast.Subscript object at 0x000001942C69DDB0>

Validate function documentation.

##### _validate_class_documentation(self: Any, file_path: Path, node: ast.ClassDef) -> <ast.Subscript object at 0x0000019427B530D0>

Validate class documentation.

##### _has_parameter_docs(self: Any, docstring: str, param_names: <ast.Subscript object at 0x0000019427B53340>) -> bool

Check if docstring contains parameter documentation.

##### _has_return_docs(self: Any, docstring: str) -> bool

Check if docstring contains return documentation.

##### _has_attribute_docs(self: Any, docstring: str) -> bool

Check if docstring contains attribute documentation.

##### _has_return_statement(self: Any, node: ast.FunctionDef) -> bool

Check if function has return statements.

##### _has_attributes(self: Any, node: ast.ClassDef) -> bool

Check if class has attributes.

