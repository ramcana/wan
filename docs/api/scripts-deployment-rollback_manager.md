---
title: scripts.deployment.rollback_manager
category: api
tags: [api, scripts]
---

# scripts.deployment.rollback_manager

Rollback Manager for Enhanced Model Availability System

This script provides rollback capabilities for failed deployments of the
enhanced model availability system, ensuring system stability and data integrity.

## Classes

### RollbackType

Types of rollback operations

### RollbackStatus

Status of rollback operations

### RollbackPoint

Represents a rollback point in the system

### RollbackResult

Result of a rollback operation

### RollbackManager

Manages rollback operations for enhanced model availability system

#### Methods

##### __init__(self: Any, backup_dir: str)



##### _load_rollback_points(self: Any) -> <ast.Subscript object at 0x000001942C557430>

Load rollback points from storage

## Constants

### FULL_SYSTEM

Type: `str`

Value: `full_system`

### CONFIGURATION

Type: `str`

Value: `configuration`

### MODELS_ONLY

Type: `str`

Value: `models_only`

### DATABASE

Type: `str`

Value: `database`

### CODE_ONLY

Type: `str`

Value: `code_only`

### PENDING

Type: `str`

Value: `pending`

### IN_PROGRESS

Type: `str`

Value: `in_progress`

### COMPLETED

Type: `str`

Value: `completed`

### FAILED

Type: `str`

Value: `failed`

### PARTIAL

Type: `str`

Value: `partial`

