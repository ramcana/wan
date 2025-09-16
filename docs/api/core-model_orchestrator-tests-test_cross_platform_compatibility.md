---
title: core.model_orchestrator.tests.test_cross_platform_compatibility
category: api
tags: [api, core]
---

# core.model_orchestrator.tests.test_cross_platform_compatibility

Cross-platform compatibility tests for Model Orchestrator.

Tests Windows, WSL, and Unix-specific behaviors including:
- Path handling and long path support
- File system operations and atomic moves
- Lock mechanisms and process synchronization
- Case sensitivity and reserved names

## Classes

### TestWindowsCompatibility

Test Windows-specific behaviors and limitations.

#### Methods

##### test_long_path_handling(self: Any)

Test handling of Windows long paths (>260 characters).

##### test_reserved_filename_handling(self: Any)

Test handling of Windows reserved filenames.

##### test_case_insensitive_filesystem(self: Any)

Test behavior on case-insensitive Windows filesystem.

##### test_unc_path_support(self: Any)

Test UNC path support for network drives.

##### test_windows_file_locking(self: Any)

Test Windows-specific file locking behavior.

##### test_junction_vs_symlink_preference(self: Any)

Test preference for junctions over symlinks on Windows.

### TestWSLCompatibility

Test WSL (Windows Subsystem for Linux) specific behaviors.

#### Methods

##### test_wsl_path_translation(self: Any)

Test path translation between WSL and Windows paths.

##### test_wsl_case_sensitivity(self: Any)

Test case sensitivity behavior in WSL.

##### _is_wsl()

Detect if running in WSL environment.

### TestUnixCompatibility

Test Unix/Linux specific behaviors.

#### Methods

##### test_unix_file_permissions(self: Any)

Test Unix file permission handling.

##### test_unix_symlink_support(self: Any)

Test symlink creation and handling on Unix.

##### test_unix_case_sensitivity(self: Any)

Test case-sensitive filesystem behavior on Unix.

##### test_unix_file_locking(self: Any)

Test Unix fcntl-based file locking.

### TestCrossPlatformPathHandling

Test path handling that works across all platforms.

#### Methods

##### test_path_normalization(self: Any)

Test path normalization across platforms.

##### test_special_character_handling(self: Any)

Test handling of special characters in model names.

##### test_unicode_path_support(self: Any)

Test Unicode character support in paths.

##### test_atomic_operations_cross_platform(self: Any)

Test atomic file operations work on all platforms.

##### test_temp_directory_same_volume(self: Any)

Test that temp directories are on same volume for atomic operations.

### TestPlatformSpecificErrorHandling

Test platform-specific error handling and messages.

#### Methods

##### test_windows_error_messages(self: Any)

Test Windows-specific error messages and guidance.

##### test_unix_error_messages(self: Any)

Test Unix-specific error messages and guidance.

##### test_cross_platform_error_consistency(self: Any)

Test that error messages are consistent across platforms.

