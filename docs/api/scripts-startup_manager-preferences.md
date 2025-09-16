---
title: scripts.startup_manager.preferences
category: api
tags: [api, scripts]
---

# scripts.startup_manager.preferences

User preference management system for the startup manager.
Handles persistent user preferences, configuration migration, and backup/restore.

## Classes

### UserPreferences

User preferences for startup manager behavior.

### ConfigurationVersion

Configuration version information for migration tracking.

### PreferenceManager

Manages user preferences, configuration migration, and backup/restore.

#### Methods

##### __init__(self: Any, preferences_dir: <ast.Subscript object at 0x000001942CBE78E0>)



##### load_preferences(self: Any) -> UserPreferences

Load user preferences from file or create defaults.

##### save_preferences(self: Any) -> None

Save current preferences to file.

##### load_version_info(self: Any) -> ConfigurationVersion

Load configuration version information.

##### save_version_info(self: Any) -> None

Save version information to file.

##### create_backup(self: Any, backup_name: <ast.Subscript object at 0x000001942C5DD7B0>) -> Path

Create a backup of current configuration and preferences.

##### restore_backup(self: Any, backup_name: str) -> bool

Restore configuration from a backup.

##### list_backups(self: Any) -> <ast.Subscript object at 0x0000019428D802B0>

List available backups with their information.

##### cleanup_old_backups(self: Any, keep_count: int) -> int

Clean up old backups, keeping only the most recent ones.

##### migrate_configuration(self: Any, target_version: str) -> bool

Migrate configuration to target version.

##### _migrate_to_2_0_0(self: Any) -> <ast.Subscript object at 0x000001942C5E6CB0>

Migrate configuration from 1.x to 2.0.0.

##### _backup_corrupted_file(self: Any, file_path: Path, backup_name: str) -> None

Backup a corrupted file for debugging.

##### apply_preferences_to_config(self: Any, config: StartupConfig) -> StartupConfig

Apply user preferences to a startup configuration.

##### preferences(self: Any) -> <ast.Subscript object at 0x00000194285DE7A0>

Get current preferences.

##### version_info(self: Any) -> <ast.Subscript object at 0x0000019428D25810>

Get current version information.

