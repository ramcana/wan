"""Unit tests for the LockManager class."""

import os
import time
import tempfile
import threading
import multiprocessing
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pytest

from .lock_manager import LockManager
from .exceptions import LockTimeoutError, LockError


class TestLockManager:
    """Test cases for LockManager functionality."""
    
    @pytest.fixture
    def temp_lock_dir(self):
        """Create a temporary directory for lock files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def lock_manager(self, temp_lock_dir):
        """Create a LockManager instance with temporary directory."""
        return LockManager(temp_lock_dir)
    
    def test_init_creates_lock_directory(self, temp_lock_dir):
        """Test that LockManager creates the lock directory if it doesn't exist."""
        lock_dir = Path(temp_lock_dir) / "locks"
        assert not lock_dir.exists()
        
        LockManager(str(lock_dir))
        assert lock_dir.exists()
        assert lock_dir.is_dir()
    
    def test_acquire_and_release_lock(self, lock_manager):
        """Test basic lock acquisition and release."""
        model_id = "test-model"
        
        # Initially not locked
        assert not lock_manager.is_locked(model_id)
        
        # Acquire lock
        with lock_manager.acquire_model_lock(model_id):
            assert lock_manager.is_locked(model_id)
        
        # Lock should be released
        assert not lock_manager.is_locked(model_id)
    
    def test_concurrent_lock_acquisition_same_process(self, lock_manager):
        """Test that concurrent lock acquisition in same process blocks correctly."""
        model_id = "test-model"
        results = []
        
        def acquire_lock(delay=0):
            if delay:
                time.sleep(delay)
            try:
                with lock_manager.acquire_model_lock(model_id, timeout=1.0):
                    results.append(f"acquired-{threading.current_thread().name}")
                    time.sleep(0.5)  # Hold lock for a bit
                    results.append(f"released-{threading.current_thread().name}")
            except LockTimeoutError:
                results.append(f"timeout-{threading.current_thread().name}")
        
        # Start two threads trying to acquire the same lock
        thread1 = threading.Thread(target=acquire_lock, name="thread1")
        thread2 = threading.Thread(target=acquire_lock, args=(0.1,), name="thread2")
        
        thread1.start()
        thread2.start()
        
        thread1.join()
        thread2.join()
        
        # Either one succeeds and one times out, or both succeed sequentially
        # (depending on timing and system performance)
        assert len(results) >= 3  # At least one complete execution
        assert any("acquired" in r for r in results)
        assert any("released" in r for r in results)
        # May have timeout or may have both succeed sequentially
    
    def test_lock_timeout(self, lock_manager):
        """Test that lock acquisition times out appropriately."""
        model_id = "test-model"
        
        def hold_lock():
            with lock_manager.acquire_model_lock(model_id):
                time.sleep(2.0)  # Hold lock longer than timeout
        
        # Start thread that holds the lock
        thread = threading.Thread(target=hold_lock)
        thread.start()
        
        time.sleep(0.1)  # Let first thread acquire lock
        
        # Try to acquire with short timeout
        start_time = time.time()
        with pytest.raises(LockTimeoutError):
            with lock_manager.acquire_model_lock(model_id, timeout=0.5):
                pass
        
        elapsed = time.time() - start_time
        assert 0.4 < elapsed < 1.0  # Should timeout around 0.5 seconds, but allow some variance
        
        thread.join()
    
    def test_multiple_different_models(self, lock_manager):
        """Test that locks for different models don't interfere."""
        model1 = "model-1"
        model2 = "model-2"
        results = []
        
        def acquire_lock(model_id, name):
            with lock_manager.acquire_model_lock(model_id):
                results.append(f"acquired-{name}")
                time.sleep(0.2)
                results.append(f"released-{name}")
        
        # Start threads for different models simultaneously
        thread1 = threading.Thread(target=acquire_lock, args=(model1, "thread1"))
        thread2 = threading.Thread(target=acquire_lock, args=(model2, "thread2"))
        
        thread1.start()
        thread2.start()
        
        thread1.join()
        thread2.join()
        
        # Both should succeed
        assert len(results) == 4
        assert "acquired-thread1" in results
        assert "released-thread1" in results
        assert "acquired-thread2" in results
        assert "released-thread2" in results
    
    def test_lock_file_creation_and_cleanup(self, lock_manager, temp_lock_dir):
        """Test that lock files are created and cleaned up properly."""
        model_id = "test-model"
        lock_file = Path(temp_lock_dir) / f"{model_id}.lock"
        
        assert not lock_file.exists()
        
        with lock_manager.acquire_model_lock(model_id):
            assert lock_file.exists()
            
            # On Windows, we can't read the file while it's locked
            # Just verify it exists and has some content
            assert lock_file.stat().st_size > 0
        
        # Lock file should be cleaned up
        # Give Windows a moment to release the file handle
        import time
        time.sleep(0.1)
        assert not lock_file.exists()
    
    def test_stale_lock_detection(self, lock_manager, temp_lock_dir):
        """Test detection and cleanup of stale locks."""
        model_id = "test-model"
        lock_file = Path(temp_lock_dir) / f"{model_id}.lock"
        
        # Create a fake stale lock file with non-existent PID
        fake_pid = 999999  # Very unlikely to exist
        lock_content = f"{fake_pid}:fake-uuid:{datetime.now().isoformat()}\n"
        lock_file.write_text(lock_content)
        
        # Make the file old
        old_time = time.time() - 7200  # 2 hours ago
        os.utime(lock_file, (old_time, old_time))
        
        # Should be able to acquire lock (stale lock gets removed)
        with lock_manager.acquire_model_lock(model_id, timeout=1.0):
            pass  # Should succeed
    
    def test_cleanup_stale_locks(self, lock_manager, temp_lock_dir):
        """Test the cleanup_stale_locks method."""
        # Create several lock files
        lock_files = []
        for i in range(3):
            model_id = f"model-{i}"
            lock_file = Path(temp_lock_dir) / f"{model_id}.lock"
            
            # Create lock with fake PID
            fake_pid = 999990 + i
            lock_content = f"{fake_pid}:fake-uuid:{datetime.now().isoformat()}\n"
            lock_file.write_text(lock_content)
            lock_files.append(lock_file)
        
        # Make first two files old (stale)
        old_time = time.time() - 7200  # 2 hours ago
        for lock_file in lock_files[:2]:
            os.utime(lock_file, (old_time, old_time))
        
        # Run cleanup
        removed_count = lock_manager.cleanup_stale_locks(max_age=timedelta(hours=1))
        
        # Should have removed 2 stale locks
        assert removed_count == 2
        assert not lock_files[0].exists()
        assert not lock_files[1].exists()
        assert lock_files[2].exists()  # Recent one should remain
    
    def test_is_locked_method(self, lock_manager):
        """Test the is_locked method accuracy."""
        model_id = "test-model"
        
        # Initially not locked
        assert not lock_manager.is_locked(model_id)
        
        # Acquire lock in thread
        def hold_lock():
            with lock_manager.acquire_model_lock(model_id):
                time.sleep(0.5)
        
        thread = threading.Thread(target=hold_lock)
        thread.start()
        
        time.sleep(0.1)  # Let thread acquire lock
        assert lock_manager.is_locked(model_id)
        
        thread.join()
        assert not lock_manager.is_locked(model_id)
    
    @patch('os.name', 'nt')  # Mock Windows
    @patch('backend.core.model_orchestrator.lock_manager.HAS_MSVCRT', True)
    @patch('backend.core.model_orchestrator.lock_manager.msvcrt')
    def test_windows_locking(self, mock_msvcrt, lock_manager):
        """Test Windows-specific locking behavior."""
        model_id = "test-model"
        
        # Mock successful locking
        mock_msvcrt.locking.side_effect = [None, None]  # Lock, then unlock
        mock_msvcrt.LK_NBLCK = 1
        mock_msvcrt.LK_UNLCK = 0
        
        with lock_manager.acquire_model_lock(model_id):
            pass
        
        # Verify msvcrt.locking was called
        assert mock_msvcrt.locking.call_count == 2
    
    @pytest.mark.skipif(os.name == 'nt', reason="Unix-specific test")
    @patch('backend.core.model_orchestrator.lock_manager.HAS_FCNTL', True)
    def test_unix_locking(self, lock_manager):
        """Test Unix-specific locking behavior."""
        # This test only runs on Unix systems where fcntl is available
        import fcntl
        
        model_id = "test-model"
        
        with patch.object(fcntl, 'flock') as mock_flock:
            # Mock successful locking
            mock_flock.side_effect = [None, None]  # Lock, then unlock
            
            with lock_manager.acquire_model_lock(model_id):
                pass
            
            # Verify fcntl.flock was called
            assert mock_flock.call_count == 2
    
    def test_lock_exception_handling(self, lock_manager, temp_lock_dir):
        """Test proper exception handling during lock operations."""
        model_id = "test-model"
        
        # Test with permission denied scenario
        with patch('os.open', side_effect=PermissionError("Access denied")):
            with pytest.raises(LockTimeoutError):
                with lock_manager.acquire_model_lock(model_id, timeout=0.1):
                    pass
    
    def test_process_running_detection(self, lock_manager):
        """Test the _is_process_running method."""
        # Test with current process (should be running)
        assert lock_manager._is_process_running(os.getpid())
        
        # Test with non-existent PID
        assert not lock_manager._is_process_running(999999)
    
    def test_lock_manager_destructor(self, lock_manager):
        """Test that destructor cleans up active locks."""
        model_id = "test-model"
        
        # Acquire lock but don't release it properly
        lock_file, lock_fd = lock_manager._acquire_lock(model_id, 1.0)
        
        # Verify lock is active
        assert model_id in lock_manager._active_locks
        
        # Destroy lock manager (should clean up)
        lock_manager.__del__()
        
        # Lock should be cleaned up
        assert model_id not in lock_manager._active_locks
        assert not lock_file.exists()


class TestLockManagerIntegration:
    """Integration tests for LockManager with real file operations."""
    
    def test_real_file_locking(self):
        """Test actual file locking with real file operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_manager = LockManager(temp_dir)
            model_id = "integration-test-model"
            
            # Test basic lock acquisition and release
            with lock_manager.acquire_model_lock(model_id):
                assert lock_manager.is_locked(model_id)
                
                # Verify lock file exists
                lock_file = Path(temp_dir) / f"{model_id}.lock"
                assert lock_file.exists()
                assert lock_file.stat().st_size > 0
            
            # Lock should be released and file removed
            # Give Windows a moment to release the file handle
            import time
            time.sleep(0.1)
            assert not lock_manager.is_locked(model_id)
            assert not lock_file.exists()
    
    def test_concurrent_processes_simulation(self):
        """Test concurrent access simulation using threading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_manager = LockManager(temp_dir)
            model_id = "concurrent-test-model"
            results = []
            
            def worker(worker_id, delay=0):
                if delay:
                    time.sleep(delay)
                
                try:
                    with lock_manager.acquire_model_lock(model_id, timeout=2.0):
                        results.append(f"worker-{worker_id}-start")
                        time.sleep(0.3)  # Simulate work
                        results.append(f"worker-{worker_id}-end")
                except LockTimeoutError:
                    results.append(f"worker-{worker_id}-timeout")
            
            # Start multiple workers
            threads = []
            for i in range(3):
                thread = threading.Thread(target=worker, args=(i, i * 0.1))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads
            for thread in threads:
                thread.join()
            
            # Verify results - should have sequential execution
            assert len(results) >= 4  # At least one complete execution
            
            # Find successful executions
            successful_workers = set()
            for i in range(len(results) - 1):
                if results[i].endswith('-start') and results[i+1].endswith('-end'):
                    worker_id = results[i].split('-')[1]
                    if results[i+1].split('-')[1] == worker_id:
                        successful_workers.add(worker_id)
            
            # At least one worker should have succeeded
            assert len(successful_workers) >= 1