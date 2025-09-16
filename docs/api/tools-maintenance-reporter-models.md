---
title: tools.maintenance-reporter.models
category: api
tags: [api, tools]
---

# tools.maintenance-reporter.models

Maintenance reporting data models and types.

## Classes

### MaintenanceOperationType

Types of maintenance operations.

### MaintenanceStatus

Status of maintenance operations.

### ImpactLevel

Impact level of maintenance operations.

### MaintenanceOperation

Individual maintenance operation record.

#### Methods

##### to_dict(self: Any) -> <ast.Subscript object at 0x000001942CC0BC70>

Convert to dictionary for serialization.

### MaintenanceImpactAnalysis

Analysis of maintenance operation impact.

#### Methods

##### to_dict(self: Any) -> <ast.Subscript object at 0x000001942CC08CD0>

Convert to dictionary for serialization.

### MaintenanceRecommendation

Maintenance recommendation based on project analysis.

#### Methods

##### to_dict(self: Any) -> <ast.Subscript object at 0x0000019427F38760>

Convert to dictionary for serialization.

### MaintenanceScheduleOptimization

Maintenance scheduling optimization analysis.

#### Methods

##### to_dict(self: Any) -> <ast.Subscript object at 0x000001942C859090>

Convert to dictionary for serialization.

### MaintenanceReport

Comprehensive maintenance report.

#### Methods

##### to_dict(self: Any) -> <ast.Subscript object at 0x000001942C85A110>

Convert to dictionary for serialization.

##### to_json(self: Any) -> str

Convert to JSON string.

### MaintenanceAuditTrail

Audit trail for maintenance operations.

#### Methods

##### to_dict(self: Any) -> <ast.Subscript object at 0x000001942C85A8C0>

Convert to dictionary for serialization.

## Constants

### TEST_REPAIR

Type: `str`

Value: `test_repair`

### CODE_CLEANUP

Type: `str`

Value: `code_cleanup`

### DOCUMENTATION_UPDATE

Type: `str`

Value: `documentation_update`

### CONFIGURATION_CONSOLIDATION

Type: `str`

Value: `configuration_consolidation`

### QUALITY_IMPROVEMENT

Type: `str`

Value: `quality_improvement`

### DEPENDENCY_UPDATE

Type: `str`

Value: `dependency_update`

### PERFORMANCE_OPTIMIZATION

Type: `str`

Value: `performance_optimization`

### SECURITY_UPDATE

Type: `str`

Value: `security_update`

### SCHEDULED

Type: `str`

Value: `scheduled`

### IN_PROGRESS

Type: `str`

Value: `in_progress`

### COMPLETED

Type: `str`

Value: `completed`

### FAILED

Type: `str`

Value: `failed`

### CANCELLED

Type: `str`

Value: `cancelled`

### ROLLBACK

Type: `str`

Value: `rollback`

### LOW

Type: `str`

Value: `low`

### MEDIUM

Type: `str`

Value: `medium`

### HIGH

Type: `str`

Value: `high`

### CRITICAL

Type: `str`

Value: `critical`

