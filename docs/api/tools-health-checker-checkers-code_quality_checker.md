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

##### _discover_python_files(self: Any) -> <ast.Subscript object at 0x0000019430089C60>

Discover Python files in the project

##### _check_syntax_errors(self: Any, python_files: <ast.Subscript object at 0x0000019430089B10>) -> <ast.Subscript object at 0x0000019430088D30>

Check for syntax errors in Python files

##### _check_code_complexity(self: Any, python_files: <ast.Subscript object at 0x0000019430088B20>) -> <ast.Subscript object at 0x000001943010F850>

Check for high complexity functions

##### _calculate_cyclomatic_complexity(self: Any, node: ast.FunctionDef) -> int

Calculate cyclomatic complexity of a function

##### _check_code_smells(self: Any, python_files: <ast.Subscript object at 0x000001943010C880>) -> <ast.Subscript object at 0x00000194300DEF20>

Check for common code smells

##### _check_import_organization(self: Any, python_files: <ast.Subscript object at 0x00000194300DED10>) -> <ast.Subscript object at 0x0000019431B72F80>

Check import organization issues

##### _check_todo_comments(self: Any, python_files: <ast.Subscript object at 0x0000019431BEAA70>) -> <ast.Subscript object at 0x0000019431BE99C0>

Check for TODO/FIXME comments

##### _run_external_quality_tools(self: Any) -> <ast.Subscript object at 0x0000019431B961A0>

Run external code quality tools if available

##### _calculate_code_quality_score(self: Any, metrics: <ast.Subscript object at 0x0000019431B96140>, issues: <ast.Subscript object at 0x0000019431B97A90>) -> float

Calculate code quality score

##### _determine_status(self: Any, score: float) -> str

Determine health status from score

