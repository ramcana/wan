---
title: core.model_orchestrator.test_model_resolver
category: api
tags: [api, core]
---

# core.model_orchestrator.test_model_resolver

Unit tests for ModelResolver class.

Tests cross-platform path handling, atomic operations, and Windows long path scenarios.

## Classes

### TestModelResolver

Test cases for ModelResolver class.

#### Methods

##### setUp(self: Any)

Set up test fixtures.

##### tearDown(self: Any)

Clean up test fixtures.

##### test_init_valid_models_root(self: Any)

Test ModelResolver initialization with valid models_root.

##### test_init_empty_models_root(self: Any)

Test ModelResolver initialization with empty models_root.

##### test_init_none_models_root(self: Any)

Test ModelResolver initialization with None models_root.

##### test_local_dir_basic(self: Any)

Test basic local directory path generation.

##### test_local_dir_with_variant(self: Any)

Test local directory path generation with variant.

##### test_local_dir_empty_model_id(self: Any)

Test local directory with empty model_id.

##### test_local_dir_none_model_id(self: Any)

Test local directory with None model_id.

##### test_temp_dir_basic(self: Any)

Test temporary directory path generation.

##### test_temp_dir_with_variant(self: Any)

Test temporary directory path generation with variant.

##### test_temp_dir_uniqueness(self: Any)

Test that temp directories are unique.

##### test_temp_dir_empty_model_id(self: Any)

Test temp directory with empty model_id.

##### test_normalize_model_id(self: Any)

Test model ID normalization.

##### test_validate_path_constraints_valid_path(self: Any)

Test path validation with valid path.

##### test_validate_path_constraints_windows_reserved_names(self: Any)

Test path validation with Windows reserved names.

##### test_validate_path_constraints_long_path(self: Any)

Test path validation with very long path.

##### test_validate_path_constraints_invalid_characters(self: Any)

Test path validation with invalid characters.

##### test_ensure_directory_exists_new_directory(self: Any)

Test creating a new directory.

##### test_ensure_directory_exists_existing_directory(self: Any)

Test with existing directory.

##### test_ensure_directory_exists_nested_directories(self: Any)

Test creating nested directories.

##### test_ensure_directory_exists_permission_error(self: Any, mock_mkdir: Any)

Test handling permission errors when creating directories.

##### test_ensure_directory_exists_os_error(self: Any, mock_mkdir: Any)

Test handling OS errors when creating directories.

##### test_is_same_filesystem_same_root(self: Any)

Test filesystem detection for paths under same root.

##### test_get_models_root(self: Any)

Test getting models root directory.

##### test_windows_detection(self: Any, mock_system: Any)

Test Windows platform detection.

##### test_wsl_detection(self: Any, mock_system: Any)

Test WSL detection.

##### test_wsl_detection_no_proc_version(self: Any, mock_open: Any, mock_system: Any)

Test WSL detection when /proc/version doesn't exist.

##### test_path_on_windows_drive_wsl(self: Any)

Test Windows drive detection in WSL.

##### test_path_on_windows_drive_non_wsl(self: Any)

Test Windows drive detection on non-WSL systems.

##### test_get_invalid_path_chars_windows(self: Any)

Test invalid path characters on Windows.

##### test_get_invalid_path_chars_unix(self: Any)

Test invalid path characters on Unix.

##### test_atomic_operations_same_filesystem(self: Any)

Test that temp and final paths are on same filesystem.

### TestPathIssue

Test cases for PathIssue class.

#### Methods

##### test_path_issue_creation(self: Any)

Test PathIssue creation.

##### test_path_issue_repr(self: Any)

Test PathIssue string representation.

