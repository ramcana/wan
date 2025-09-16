---
title: tools.health-checker.health_models
category: api
tags: [api, tools]
---

# tools.health-checker.health_models

Health monitoring data models and enums

## Classes

### Severity

Issue severity levels

### HealthCategory

Health check categories

### HealthIssue

Represents a project health issue

### Recommendation

Actionable recommendation for improvement

### ComponentHealth

Health status for a specific component

### HealthTrends

Historical health trend data

### HealthReport

Comprehensive project health report

#### Methods

##### get_issues_by_severity(self: Any, severity: Severity) -> <ast.Subscript object at 0x0000019428D0C820>

Get all issues of a specific severity

##### get_issues_by_category(self: Any, category: HealthCategory) -> <ast.Subscript object at 0x0000019428D0CA00>

Get all issues in a specific category

##### get_critical_issues(self: Any) -> <ast.Subscript object at 0x000001942C800880>

Get all critical issues

##### get_component_score(self: Any, component: str) -> float

Get score for a specific component

### HealthConfig

Configuration for health monitoring

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

### TESTS

Type: `str`

Value: `tests`

### DOCUMENTATION

Type: `str`

Value: `documentation`

### CONFIGURATION

Type: `str`

Value: `configuration`

### CODE_QUALITY

Type: `str`

Value: `code_quality`

### PERFORMANCE

Type: `str`

Value: `performance`

### SECURITY

Type: `str`

Value: `security`

### DEPENDENCIES

Type: `str`

Value: `dependencies`

