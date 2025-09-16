---
title: core.model_orchestrator.test_lock_manager
category: api
tags: [api, core]
---

# core.model_orchestrator.test_lock_manager

Unit tests for the LockManager class.

## Classes

### TestLockManager

Test cases for LockManager functionality.

#### Methods

##### temp_lock_dir(self: Any)

Create a temporary directory for lock files.

##### lock_manager(self: Any, temp_lock_dir: Any)

Create a LockManager instance with temporary directory.

##### test_init_creates_lock_directory(self: Any, temp_lock_dir: Any)

Test that LockManager creates the lock directory if it doesn't exist.

##### test_acquire_and_release_lock(self: Any, lock_manager: Any)

Test basic lock acquisition and release.

##### test_concurrent_lock_acquisition_same_process(self: Any, lock_manager: Any)

Test that concurrent lock acquisition in same process blocks correctly.

##### test_lock_timeout(self: Any, lock_manager: Any)

Test that lock acquisition times out appropriately.

##### test_multiple_different_models(self: Any, lock_manager: Any)

Test that locks for different models don't interfere.

##### test_lock_file_creation_and_cleanup(self: Any, lock_manager: Any, temp_lock_dir: Any)

Test that lock files are created and cleaned up properly.

##### test_stale_lock_detection(self: Any, lock_manager: Any, temp_lock_dir: Any)

Test detection and cleanup of stale locks.

##### test_cleanup_stale_locks(self: Any, lock_manager: Any, temp_lock_dir: Any)

Test the cleanup_stale_locks method.

##### test_is_locked_method(self: Any, lock_manager: Any)

Test the is_locked method accuracy.

##### test_windows_locking(self: Any, mock_msvcrt: Any, lock_manager: Any)

Test Windows-specific locking behavior.

##### test_unix_locking(self: Any, lock_manager: Any)

Test Unix-specific locking behavior.

##### test_lock_exception_handling(self: Any, lock_manager: Any, temp_lock_dir: Any)

Test proper exception handling during lock operations.

##### test_process_running_detection(self: Any, lock_manager: Any)

Test the _is_process_running method.

##### test_lock_manager_destructor(self: Any, lock_manager: Any)

Test that destructor cleans up active locks.

### TestLockManagerIntegration

Integration tests for LockManager with real file operations.

#### Methods

##### test_real_file_locking(self: Any)

Test actual file locking with real file operations.

##### test_concurrent_processes_simulation(self: Any)

Test concurrent access simulation using threading.

