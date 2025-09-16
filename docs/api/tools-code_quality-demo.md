---
title: tools.code_quality.demo
category: api
tags: [api, tools]
---

# tools.code_quality.demo

Demonstration of the code quality checking system.

## Classes

### QualityIssueType



### QualitySeverity



### QualityIssue



### QualityMetrics



#### Methods

##### documentation_coverage(self: Any) -> float



### QualityReport



#### Methods

##### total_issues(self: Any) -> int



##### errors(self: Any) -> int



##### warnings(self: Any) -> int



##### quality_score(self: Any) -> float



### SimpleQualityChecker

Simplified quality checker for demonstration.

#### Methods

##### check_quality(self: Any, file_path: Path) -> QualityReport

Check quality of a Python file.

##### _calculate_complexity(self: Any, node: ast.FunctionDef) -> int

Calculate cyclomatic complexity.

## Constants

### FORMATTING

Type: `str`

Value: `formatting`

### STYLE

Type: `str`

Value: `style`

### DOCUMENTATION

Type: `str`

Value: `documentation`

### TYPE_HINTS

Type: `str`

Value: `type_hints`

### COMPLEXITY

Type: `str`

Value: `complexity`

### ERROR

Type: `str`

Value: `error`

### WARNING

Type: `str`

Value: `warning`

### INFO

Type: `str`

Value: `info`

