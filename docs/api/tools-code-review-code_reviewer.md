---
title: tools.code-review.code_reviewer
category: api
tags: [api, tools]
---

# tools.code-review.code_reviewer

Code Review and Refactoring Assistance System

This module provides automated code review suggestions, refactoring recommendations,
and technical debt tracking to improve code quality and maintainability.

## Classes

### ReviewSeverity

Severity levels for code review issues

### IssueCategory

Categories of code review issues

### CodeIssue

Represents a code review issue

### RefactoringRecommendation

Represents a refactoring recommendation

### TechnicalDebtItem

Represents a technical debt item

### CodeReviewer

Main code review and refactoring assistance system

#### Methods

##### __init__(self: Any, project_root: str)



##### _load_config(self: Any) -> <ast.Subscript object at 0x000001942C83E8C0>

Load code review configuration

##### review_file(self: Any, file_path: str) -> <ast.Subscript object at 0x000001942C83D270>

Review a single file and return issues

##### review_project(self: Any, include_patterns: <ast.Subscript object at 0x000001942C83D120>) -> <ast.Subscript object at 0x0000019427F97160>

Review entire project

##### _generate_refactoring_recommendations(self: Any, file_path: str, issues: <ast.Subscript object at 0x0000019427F97430>) -> <ast.Subscript object at 0x0000019428D3BD30>

Generate refactoring recommendations based on issues

##### _update_technical_debt(self: Any)

Update technical debt tracking based on current issues

##### _calculate_priority_score(self: Any, issues: <ast.Subscript object at 0x000001942CC8A3E0>) -> float

Calculate priority score for technical debt

##### _generate_summary(self: Any) -> <ast.Subscript object at 0x000001942CC8AFB0>

Generate review summary

##### generate_report(self: Any, output_path: str)

Generate comprehensive code review report

### ComplexityAnalyzer

Analyzes code complexity

#### Methods

##### analyze(self: Any, file_path: str, tree: ast.AST, content: str) -> <ast.Subscript object at 0x000001942B33EB30>



##### _calculate_complexity(self: Any, node: ast.FunctionDef) -> int

Calculate cyclomatic complexity

### MaintainabilityAnalyzer

Analyzes code maintainability

#### Methods

##### analyze(self: Any, file_path: str, tree: ast.AST, content: str) -> <ast.Subscript object at 0x0000019427AD9BD0>



### PerformanceAnalyzer

Analyzes potential performance issues

#### Methods

##### analyze(self: Any, file_path: str, tree: ast.AST, content: str) -> <ast.Subscript object at 0x0000019427AD83D0>



##### _count_nested_loops(self: Any, node: ast.AST) -> int

Count nested loop levels

### SecurityAnalyzer

Analyzes potential security issues

#### Methods

##### analyze(self: Any, file_path: str, tree: ast.AST, content: str) -> <ast.Subscript object at 0x0000019427AD97E0>



##### _get_function_name(self: Any, node: ast.AST) -> str

Get function name from AST node

## Constants

### CRITICAL

Type: `str`

Value: `critical`

### HIGH

Type: `str`

Value: `high`

### MEDIUM

Type: `str`

Value: `medium`

### LOW

Type: `str`

Value: `low`

### INFO

Type: `str`

Value: `info`

### COMPLEXITY

Type: `str`

Value: `complexity`

### MAINTAINABILITY

Type: `str`

Value: `maintainability`

### PERFORMANCE

Type: `str`

Value: `performance`

### SECURITY

Type: `str`

Value: `security`

### STYLE

Type: `str`

Value: `style`

### DOCUMENTATION

Type: `str`

Value: `documentation`

### TESTING

Type: `str`

Value: `testing`

### ARCHITECTURE

Type: `str`

Value: `architecture`

