---
title: tools.health-checker.checkers.code_quality_checker
category: api
tags: [api, tools]
---

# tools.health-checker.checkers.code_quality_checker



## Classes

### CodeQualityChecker

Checks code quality metrics and issues

#### Methods

##### __init__(self: Any, config: HealthConfig)



##### check_health(self: Any) -> ComponentHealth

Check code quality health

##### _discover_python_files(self: Any) -> <ast.Subscript object at 0x0000019427A224D0>

Discover Python files in the project

##### _check_syntax_errors(self: Any, python_files: <ast.Subscript object at 0x0000019427A22260>) -> <ast.Subscript object at 0x0000019427A200D0>

Check for syntax errors in Python files

##### _check_code_complexity(self: Any, python_files: <ast.Subscript object at 0x0000019427A21900>) -> <ast.Subscript object at 0x0000019427A21B40>

Check for high complexity functions

##### _calculate_cyclomatic_complexity(self: Any, node: ast.FunctionDef) -> int

Calculate cyclomatic complexity of a function

##### _check_code_smells(self: Any, python_files: <ast.Subscript object at 0x0000019428122500>) -> <ast.Subscript object at 0x0000019428123610>

Check for common code smells

##### _check_import_organization(self: Any, python_files: <ast.Subscript object at 0x0000019428123E80>) -> <ast.Subscript object at 0x0000019427AD8D00>

Check import organization issues

##### _check_todo_comments(self: Any, python_files: <ast.Subscript object at 0x0000019427AD9F90>) -> <ast.Subscript object at 0x0000019427AD9780>

Check for TODO/FIXME comments

##### _run_external_quality_tools(self: Any) -> <ast.Subscript object at 0x0000019427ADA8F0>

Run external code quality tools if available

##### _calculate_code_quality_score(self: Any, metrics: <ast.Subscript object at 0x0000019427AD9300>, issues: <ast.Subscript object at 0x0000019427ADB4F0>) -> float

Calculate code quality score

##### _determine_status(self: Any, score: float) -> str

Determine health status from score

