---
title: core.model_orchestrator.lock_manager
category: api
tags: [api, core]
---

# core.model_orchestrator.lock_manager

Cross-process locking system for model orchestrator.

This module provides OS-appropriate file locking to prevent concurrent download conflicts
and ensure atomic model operations across multiple processes.

## Classes

### LockManager

Cross-process file locking manager with timeout and cleanup capabilities.

#### Methods

##### __init__(self: Any, lock_dir: str)

Initialize the lock manager.

Args:
    lock_dir: Directory to store lock files

##### acquire_model_lock(self: Any, model_id: str, timeout: float) -> ContextManager

Acquire an exclusive lock for a model.

Args:
    model_id: Unique identifier for the model
    timeout: Maximum time to wait for lock acquisition in seconds
    
Returns:
    Context manager for the lock
    
Raises:
    LockTimeoutError: If lock cannot be acquired within timeout
    LockError: If lock operation fails

##### _lock_context(self: Any, model_id: str, timeout: float)

Context manager for safe lock acquisition and release.

##### _acquire_lock(self: Any, model_id: str, timeout: float) -> <ast.Subscript object at 0x000001942CDD9F30>

Acquire a file lock with timeout and retry logic.

##### _try_lock(self: Any, fd: int) -> bool

Try to acquire an exclusive lock on the file descriptor.

##### _release_lock(self: Any, model_id: str, lock_file: Path, lock_fd: int)

Release the file lock and clean up.

##### is_locked(self: Any, model_id: str) -> bool

Check if a model is currently locked.

Args:
    model_id: Model identifier to check
    
Returns:
    True if model is locked, False otherwise

##### _is_stale_lock(self: Any, lock_file: Path, max_age: timedelta) -> bool

Check if a lock file is stale (from a crashed process).

##### _is_process_running(self: Any, pid: int) -> bool

Check if a process with the given PID is still running.

##### _remove_stale_lock(self: Any, lock_file: Path)

Remove a stale lock file.

##### cleanup_stale_locks(self: Any, max_age: timedelta) -> int

Clean up stale lock files from crashed processes.

Args:
    max_age: Maximum age for lock files before considering them stale
    
Returns:
    Number of stale locks removed

##### __del__(self: Any)

Cleanup any remaining active locks on destruction.

## Constants

### HAS_FCNTL

Type: `bool`

Value: `True`

### HAS_MSVCRT

Type: `bool`

Value: `True`

### HAS_FCNTL

Type: `bool`

Value: `False`

### HAS_MSVCRT

Type: `bool`

Value: `False`

