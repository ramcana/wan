---
title: tools.maintenance-scheduler.rollback_manager
category: api
tags: [api, tools]
---

# tools.maintenance-scheduler.rollback_manager



## Classes

### RollbackPoint

Represents a rollback point for maintenance operations.

### RollbackManager

Manages rollback points and operations for safe automated maintenance.

Provides capabilities to:
- Create rollback points before maintenance operations
- Store file backups, git states, and configuration snapshots
- Execute rollbacks when operations fail or need to be undone
- Clean up old rollback points

#### Methods

##### __init__(self: Any, config: <ast.Subscript object at 0x00000194302690F0>)



##### get_rollback_points(self: Any, task_id: <ast.Subscript object at 0x000001942F221360>) -> <ast.Subscript object at 0x000001942F222650>

Get available rollback points.

##### cleanup_old_rollback_points(self: Any) -> int

Clean up old and invalid rollback points.

##### get_rollback_statistics(self: Any) -> <ast.Subscript object at 0x000001943193A950>

Get statistics about rollback points.

##### _get_files_to_backup(self: Any, task: MaintenanceTask) -> <ast.Subscript object at 0x00000194344A24D0>

Determine which files to backup based on task type.

##### _get_config_files_to_backup(self: Any, task: MaintenanceTask) -> <ast.Subscript object at 0x00000194344A2D40>

Get configuration files to backup.

##### _calculate_backup_checksum(self: Any, backup_dir: Path) -> str

Calculate checksum for backup directory.

##### _calculate_backup_size(self: Any, backup_dir: Path) -> int

Calculate total size of backup directory.

##### _verify_backup_integrity(self: Any, rollback_point: RollbackPoint, backup_dir: Path) -> bool

Verify backup integrity using checksum.

##### _get_current_branch(self: Any) -> <ast.Subscript object at 0x00000194344A38E0>

Get current git branch.

##### _get_git_status(self: Any) -> <ast.Subscript object at 0x00000194344A1210>

Get git status.

##### _load_rollback_points(self: Any) -> None

Load rollback points from storage.

##### _save_rollback_points(self: Any) -> None

Save rollback points to storage.

##### _serialize_rollback_point(self: Any, point: RollbackPoint) -> Dict

Serialize rollback point to dictionary.

##### _deserialize_rollback_point(self: Any, data: Dict) -> <ast.Subscript object at 0x00000194319C7250>

Deserialize rollback point from dictionary.

