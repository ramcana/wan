---
title: core.model_update_manager
category: api
tags: [api, core]
---

# core.model_update_manager



## Classes

### UpdateStatus

Update status enumeration

### UpdatePriority

Update priority levels

### UpdateType

Types of updates

### ModelVersion

Model version information

### UpdateInfo

Information about an available update

### UpdateProgress

Update progress tracking

### UpdateResult

Result of an update operation

### UpdateSchedule

Update scheduling configuration

### RollbackInfo

Information about a rollback operation

### ModelUpdateManager

Comprehensive model update management system with version checking,
update detection, safe update processes, rollback capability, and scheduling.

#### Methods

##### __init__(self: Any, models_dir: <ast.Subscript object at 0x000001942754A980>, downloader: Any, health_monitor: Any)

Initialize the model update manager.

Args:
    models_dir: Directory containing models
    downloader: Enhanced model downloader instance
    health_monitor: Model health monitor instance

##### add_update_callback(self: Any, callback: <ast.Subscript object at 0x000001942CE64E20>)

Add a callback for update progress

##### add_notification_callback(self: Any, callback: <ast.Subscript object at 0x000001942CE64AC0>)

Add a callback for update notifications

##### _get_installed_models(self: Any) -> <ast.Subscript object at 0x000001942CEC84F0>

Get list of installed models

##### _is_update_available(self: Any, current_version: str, latest_version: str) -> bool

Check if an update is available

##### _parse_version(self: Any, version: str) -> <ast.Subscript object at 0x000001942CB73820>

Parse semantic version string

##### _determine_update_type(self: Any, current_version: str, latest_version: str) -> UpdateType

Determine the type of update

##### _determine_update_priority(self: Any, version_info: ModelVersion) -> UpdatePriority

Determine update priority based on changelog

##### _calculate_directory_size(self: Any, directory: Path) -> int

Calculate total size of a directory

## Constants

### AVAILABLE

Type: `str`

Value: `available`

### DOWNLOADING

Type: `str`

Value: `downloading`

### VALIDATING

Type: `str`

Value: `validating`

### INSTALLING

Type: `str`

Value: `installing`

### COMPLETED

Type: `str`

Value: `completed`

### FAILED

Type: `str`

Value: `failed`

### CANCELLED

Type: `str`

Value: `cancelled`

### ROLLBACK_REQUIRED

Type: `str`

Value: `rollback_required`

### ROLLBACK_IN_PROGRESS

Type: `str`

Value: `rollback_in_progress`

### ROLLBACK_COMPLETED

Type: `str`

Value: `rollback_completed`

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

### MAJOR

Type: `str`

Value: `major`

### MINOR

Type: `str`

Value: `minor`

### PATCH

Type: `str`

Value: `patch`

### HOTFIX

Type: `str`

Value: `hotfix`

