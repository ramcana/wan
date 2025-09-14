"""Cross-process locking system for model orchestrator.

This module provides OS-appropriate file locking to prevent concurrent download conflicts
and ensure atomic model operations across multiple processes.
"""

import os
import time
import uuid
import logging
from pathlib import Path
from typing import Optional, ContextManager
from contextlib import contextmanager
from datetime import datetime, timedelta

# OS-specific imports
try:
    import fcntl  # Unix/Linux
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False

try:
    import msvcrt  # Windows
    HAS_MSVCRT = True
except ImportError:
    HAS_MSVCRT = False

from .exceptions import LockTimeoutError, LockError

logger = logging.getLogger(__name__)


class LockManager:
    """Cross-process file locking manager with timeout and cleanup capabilities."""
    
    def __init__(self, lock_dir: str):
        """Initialize the lock manager.
        
        Args:
            lock_dir: Directory to store lock files
        """
        self.lock_dir = Path(lock_dir)
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        self._active_locks = {}  # Track active locks for cleanup
        
    def acquire_model_lock(self, model_id: str, timeout: float = 300.0) -> ContextManager:
        """Acquire an exclusive lock for a model.
        
        Args:
            model_id: Unique identifier for the model
            timeout: Maximum time to wait for lock acquisition in seconds
            
        Returns:
            Context manager for the lock
            
        Raises:
            LockTimeoutError: If lock cannot be acquired within timeout
            LockError: If lock operation fails
        """
        return self._lock_context(model_id, timeout)
    
    @contextmanager
    def _lock_context(self, model_id: str, timeout: float):
        """Context manager for safe lock acquisition and release."""
        lock_file = None
        lock_fd = None
        
        try:
            lock_file, lock_fd = self._acquire_lock(model_id, timeout)
            logger.info(f"Acquired lock for model {model_id}")
            yield
        finally:
            if lock_fd is not None:
                self._release_lock(model_id, lock_file, lock_fd)
                logger.info(f"Released lock for model {model_id}")
    
    def _acquire_lock(self, model_id: str, timeout: float) -> tuple[Path, int]:
        """Acquire a file lock with timeout and retry logic."""
        lock_file = self.lock_dir / f"{model_id}.lock"
        start_time = time.time()
        retry_delay = 0.1  # Start with 100ms
        max_retry_delay = 2.0  # Cap at 2 seconds
        
        while time.time() - start_time < timeout:
            lock_fd = None
            try:
                # Create lock file with process info
                lock_fd = os.open(str(lock_file), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
                
                # Try to acquire exclusive lock
                if self._try_lock(lock_fd):
                    # Write lock metadata
                    lock_info = f"{os.getpid()}:{uuid.uuid4().hex}:{datetime.now().isoformat()}\n"
                    os.write(lock_fd, lock_info.encode())
                    os.fsync(lock_fd)  # Ensure data is written
                    
                    # Track active lock
                    self._active_locks[model_id] = (lock_file, lock_fd)
                    return lock_file, lock_fd
                else:
                    # Lock is held by another process
                    os.close(lock_fd)
                    lock_fd = None
                    
                    # Check if lock is stale
                    if self._is_stale_lock(lock_file):
                        logger.warning(f"Removing stale lock for model {model_id}")
                        self._remove_stale_lock(lock_file)
                        continue
                    
            except (OSError, IOError) as e:
                if lock_fd is not None:
                    try:
                        os.close(lock_fd)
                    except:
                        pass
                    lock_fd = None
                
                # If file doesn't exist, another process might have cleaned it up
                if not lock_file.exists():
                    continue
                    
                logger.debug(f"Lock acquisition attempt failed for {model_id}: {e}")
            
            # Exponential backoff with jitter
            time.sleep(retry_delay + (retry_delay * 0.1 * (time.time() % 1)))
            retry_delay = min(retry_delay * 1.5, max_retry_delay)
        
        raise LockTimeoutError(f"Could not acquire lock for model {model_id} within {timeout} seconds")
    
    def _try_lock(self, fd: int) -> bool:
        """Try to acquire an exclusive lock on the file descriptor."""
        try:
            if HAS_FCNTL:
                # Unix/Linux: Use fcntl for file locking
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            elif HAS_MSVCRT:
                # Windows: Use msvcrt for file locking
                # Note: msvcrt.locking locks from current position to EOF
                # We need to seek to beginning and lock at least 1 byte
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return True
            else:
                # Fallback: No locking available, assume success
                logger.warning("No file locking mechanism available, proceeding without locks")
                return True
        except (OSError, IOError):
            return False
    
    def _release_lock(self, model_id: str, lock_file: Path, lock_fd: int):
        """Release the file lock and clean up."""
        try:
            if HAS_FCNTL:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            elif HAS_MSVCRT:
                # Windows: Unlock the same region we locked
                try:
                    os.lseek(lock_fd, 0, os.SEEK_SET)
                    msvcrt.locking(lock_fd, msvcrt.LK_UNLCK, 1)
                except (OSError, IOError):
                    # If unlock fails, we'll still close the file
                    pass
            
            os.close(lock_fd)
            
            # Remove lock file
            try:
                lock_file.unlink()
            except FileNotFoundError:
                pass  # Already removed
            except (OSError, IOError):
                # On Windows, file might still be locked by system, retry once
                import time
                time.sleep(0.01)
                try:
                    lock_file.unlink()
                except:
                    pass  # Best effort
            
            # Remove from active locks
            self._active_locks.pop(model_id, None)
            
        except (OSError, IOError) as e:
            logger.error(f"Error releasing lock for model {model_id}: {e}")
            # Don't raise exception on release failure, just log it
            # This prevents cascading failures during cleanup
    
    def is_locked(self, model_id: str) -> bool:
        """Check if a model is currently locked.
        
        Args:
            model_id: Model identifier to check
            
        Returns:
            True if model is locked, False otherwise
        """
        lock_file = self.lock_dir / f"{model_id}.lock"
        
        if not lock_file.exists():
            return False
        
        # Check if we have this lock active
        if model_id in self._active_locks:
            return True
        
        # Try to acquire lock to test if it's available
        # On Windows, this is more reliable than trying to read the lock
        try:
            test_fd = os.open(str(lock_file), os.O_WRONLY | os.O_APPEND)
            try:
                if self._try_lock(test_fd):
                    # Lock was available, release it immediately
                    if HAS_FCNTL:
                        fcntl.flock(test_fd, fcntl.LOCK_UN)
                    elif HAS_MSVCRT:
                        try:
                            os.lseek(test_fd, 0, os.SEEK_SET)
                            msvcrt.locking(test_fd, msvcrt.LK_UNLCK, 1)
                        except:
                            pass
                    return False
                else:
                    return True
            finally:
                os.close(test_fd)
        except (OSError, IOError):
            # If we can't test the lock, assume it's locked
            return True
    
    def _is_stale_lock(self, lock_file: Path, max_age: timedelta = timedelta(hours=1)) -> bool:
        """Check if a lock file is stale (from a crashed process)."""
        try:
            if not lock_file.exists():
                return False
            
            # Check file age
            file_age = datetime.now() - datetime.fromtimestamp(lock_file.stat().st_mtime)
            if file_age < max_age:
                return False
            
            # Try to read lock info
            with open(lock_file, 'r') as f:
                lock_info = f.read().strip()
            
            if ':' not in lock_info:
                return True  # Invalid format, consider stale
            
            pid_str = lock_info.split(':')[0]
            try:
                pid = int(pid_str)
            except ValueError:
                return True  # Invalid PID, consider stale
            
            # Check if process is still running
            return not self._is_process_running(pid)
            
        except (OSError, IOError, ValueError):
            # If we can't read the lock file, consider it stale
            return True
    
    def _is_process_running(self, pid: int) -> bool:
        """Check if a process with the given PID is still running."""
        try:
            if os.name == 'nt':  # Windows
                import subprocess
                result = subprocess.run(['tasklist', '/FI', f'PID eq {pid}'], 
                                      capture_output=True, text=True)
                return str(pid) in result.stdout
            else:  # Unix/Linux
                os.kill(pid, 0)  # Send signal 0 to check if process exists
                return True
        except (OSError, subprocess.SubprocessError):
            return False
    
    def _remove_stale_lock(self, lock_file: Path):
        """Remove a stale lock file."""
        try:
            lock_file.unlink()
        except FileNotFoundError:
            pass  # Already removed
        except (OSError, IOError) as e:
            logger.warning(f"Could not remove stale lock {lock_file}: {e}")
    
    def cleanup_stale_locks(self, max_age: timedelta = timedelta(hours=1)) -> int:
        """Clean up stale lock files from crashed processes.
        
        Args:
            max_age: Maximum age for lock files before considering them stale
            
        Returns:
            Number of stale locks removed
        """
        removed_count = 0
        
        try:
            for lock_file in self.lock_dir.glob("*.lock"):
                if self._is_stale_lock(lock_file, max_age):
                    logger.info(f"Removing stale lock: {lock_file}")
                    self._remove_stale_lock(lock_file)
                    removed_count += 1
        except (OSError, IOError) as e:
            logger.error(f"Error during stale lock cleanup: {e}")
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} stale lock files")
        
        return removed_count
    
    def __del__(self):
        """Cleanup any remaining active locks on destruction."""
        for model_id, (lock_file, lock_fd) in list(self._active_locks.items()):
            try:
                self._release_lock(model_id, lock_file, lock_fd)
            except:
                pass  # Best effort cleanup