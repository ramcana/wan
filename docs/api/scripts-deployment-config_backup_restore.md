---
title: scripts.deployment.config_backup_restore
category: api
tags: [api, scripts]
---

# scripts.deployment.config_backup_restore

Configuration Backup and Restore Tools for Enhanced Model Availability System

This script provides comprehensive backup and restore capabilities for all
configuration files and settings related to the enhanced model availability system.

## Classes

### BackupType

Types of configuration backups

### BackupStatus

Status of backup operations

### ConfigFile

Represents a configuration file

### BackupManifest

Manifest of a configuration backup

### RestoreResult

Result of a restore operation

### ConfigurationBackupManager

Manages configuration backups and restores

#### Methods

##### __init__(self: Any, backup_dir: str)



##### _get_config_patterns(self: Any) -> <ast.Subscript object at 0x000001942C5F1A20>

Get configuration file patterns for different backup types

##### _load_manifests(self: Any) -> <ast.Subscript object at 0x00000194289D2710>

Load backup manifests from storage

## Constants

### FULL

Type: `str`

Value: `full`

### CONFIGURATION_ONLY

Type: `str`

Value: `configuration_only`

### USER_PREFERENCES

Type: `str`

Value: `user_preferences`

### SYSTEM_SETTINGS

Type: `str`

Value: `system_settings`

### CUSTOM

Type: `str`

Value: `custom`

### CREATED

Type: `str`

Value: `created`

### VERIFIED

Type: `str`

Value: `verified`

### CORRUPTED

Type: `str`

Value: `corrupted`

### RESTORED

Type: `str`

Value: `restored`

### FAILED

Type: `str`

Value: `failed`

