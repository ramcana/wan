"""
Test Isolation and Cleanup Utilities

This module provides comprehensive test isolation mechanisms including:
- Database isolation and cleanup
- File system isolation
- Process isolation
- Network isolation
- Environment variable isolation
- Mock management
"""

import os
import sys
import tempfile
import shutil
import sqlite3
import json
import threading
import time
import socket
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Callable
from contextlib import contextmanager
from unittest.mock import Mock, patch, MagicMock
import pytest
from dataclasses import dataclass, field


@dataclass
class IsolationContext:
    """Context for test isolation state."""
    temp_dirs: List[Path] = field(default_factory=list)
    temp_files: List[Path] = field(default_factory=list)
    databases: List[Path] = field(default_factory=list)
    processes: List[int] = field(default_factory=list)
    ports: List[int] = field(default_factory=list)
    env_vars: Dict[str, Optional[str]] = field(default_factory=dict)
    mocks: List[Any] = field(default_factory=list)
    cleanup_callbacks: List[Callable] = field(default_factory=list)


class TestIsolationManager:
    """Manages test isolation and cleanup."""
    
    def __init__(self):
        self.contexts: Dict[str, IsolationContext] = {}
        self.global_context = IsolationContext()
        self._lock = threading.Lock()
    
    def create_context(self, test_id: str) -> IsolationContext:
        """Create a new isolation context for a test."""
        with self._lock:
            context = IsolationContext()
            self.contexts[test_id] = context
            return context
    
    def get_context(self, test_id: str) -> Optional[IsolationContext]:
        """Get the isolation context for a test."""
        return self.contexts.get(test_id)
    
    def cleanup_context(self, test_id: str):
        """Clean up a specific test context."""
        context = self.contexts.get(test_id)
        if context:
            self._cleanup_context(context)
            del self.contexts[test_id]
    
    def cleanup_all(self):
        """Clean up all contexts."""
        with self._lock:
            for context in self.contexts.values():
                self._cleanup_context(context)
            self.contexts.clear()
            self._cleanup_context(self.global_context)
    
    def _cleanup_context(self, context: IsolationContext):
        """Clean up a single context."""
        # Run custom cleanup callbacks first
        for callback in context.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                print(f"Warning: Cleanup callback failed: {e}")
        
        # Clean up temporary files
        for temp_file in context.temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception as e:
                print(f"Warning: Failed to remove temp file {temp_file}: {e}")
        
        # Clean up temporary directories
        for temp_dir in context.temp_dirs:
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                print(f"Warning: Failed to remove temp dir {temp_dir}: {e}")
        
        # Clean up databases
        for db_path in context.databases:
            try:
                if db_path.exists():
                    db_path.unlink()
            except Exception as e:
                print(f"Warning: Failed to remove database {db_path}: {e}")
        
        # Clean up processes
        for pid in context.processes:
            try:
                import psutil
                process = psutil.Process(pid)
                if process.is_running():
                    process.terminate()
                    process.wait(timeout=5)
            except Exception:
                pass
        
        # Restore environment variables
        for var_name, original_value in context.env_vars.items():
            try:
                if original_value is None:
                    os.environ.pop(var_name, None)
                else:
                    os.environ[var_name] = original_value
            except Exception as e:
                print(f"Warning: Failed to restore env var {var_name}: {e}")
        
        # Clean up mocks
        for mock_obj in context.mocks:
            try:
                if hasattr(mock_obj, 'stop'):
                    mock_obj.stop()
                elif hasattr(mock_obj, 'reset_mock'):
                    mock_obj.reset_mock()
            except Exception as e:
                print(f"Warning: Failed to clean up mock: {e}")


# Global isolation manager instance
_isolation_manager = TestIsolationManager()


class DatabaseIsolation:
    """Provides database isolation for tests."""
    
    def __init__(self, test_id: str):
        self.test_id = test_id
        self.context = _isolation_manager.get_context(test_id) or _isolation_manager.create_context(test_id)
        self.temp_db_dir = None
    
    def create_temp_database(self, db_name: str = "test.db") -> Path:
        """Create a temporary SQLite database for testing."""
        if not self.temp_db_dir:
            self.temp_db_dir = Path(tempfile.mkdtemp(prefix=f"test_db_{self.test_id}_"))
            self.context.temp_dirs.append(self.temp_db_dir)
        
        db_path = self.temp_db_dir / db_name
        self.context.databases.append(db_path)
        
        # Create empty database
        conn = sqlite3.connect(str(db_path))
        conn.close()
        
        return db_path
    
    def create_test_tables(self, db_path: Path, schema_sql: str):
        """Create test tables in the database."""
        conn = sqlite3.connect(str(db_path))
        try:
            conn.executescript(schema_sql)
            conn.commit()
        finally:
            conn.close()
    
    def populate_test_data(self, db_path: Path, data: Dict[str, List[Dict[str, Any]]]):
        """Populate database with test data."""
        conn = sqlite3.connect(str(db_path))
        try:
            for table_name, rows in data.items():
                if rows:
                    # Get column names from first row
                    columns = list(rows[0].keys())
                    placeholders = ', '.join(['?' for _ in columns])
                    
                    insert_sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
                    
                    for row in rows:
                        values = [row[col] for col in columns]
                        conn.execute(insert_sql, values)
            
            conn.commit()
        finally:
            conn.close()
    
    @contextmanager
    def isolated_database(self, schema_sql: str = "", test_data: Optional[Dict[str, List[Dict[str, Any]]]] = None):
        """Context manager for isolated database testing."""
        db_path = self.create_temp_database()
        
        if schema_sql:
            self.create_test_tables(db_path, schema_sql)
        
        if test_data:
            self.populate_test_data(db_path, test_data)
        
        yield db_path
    
    @contextmanager
    def database_transaction(self, db_path: Path):
        """Context manager for database transaction with automatic rollback."""
        conn = sqlite3.connect(str(db_path))
        conn.execute("BEGIN TRANSACTION")
        
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def create_database_snapshot(self, db_path: Path) -> Path:
        """Create a snapshot of the database for rollback purposes."""
        snapshot_path = db_path.parent / f"{db_path.stem}_snapshot{db_path.suffix}"
        shutil.copy2(db_path, snapshot_path)
        self.context.databases.append(snapshot_path)
        return snapshot_path
    
    def restore_database_snapshot(self, db_path: Path, snapshot_path: Path):
        """Restore database from snapshot."""
        if snapshot_path.exists():
            shutil.copy2(snapshot_path, db_path)


class FileSystemIsolation:
    """Provides file system isolation for tests."""
    
    def __init__(self, test_id: str):
        self.test_id = test_id
        self.context = _isolation_manager.get_context(test_id) or _isolation_manager.create_context(test_id)
    
    def create_temp_dir(self, prefix: str = "test_") -> Path:
        """Create a temporary directory."""
        temp_dir = Path(tempfile.mkdtemp(prefix=f"{prefix}{self.test_id}_"))
        self.context.temp_dirs.append(temp_dir)
        return temp_dir
    
    def create_temp_file(self, content: str = "", suffix: str = ".txt", prefix: str = "test_") -> Path:
        """Create a temporary file with content."""
        fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=f"{prefix}{self.test_id}_")
        temp_file = Path(temp_path)
        
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(content)
        except:
            os.close(fd)
            raise
        
        self.context.temp_files.append(temp_file)
        return temp_file
    
    def create_test_file_structure(self, base_dir: Path, structure: Dict[str, Any]):
        """Create a test file structure from a dictionary."""
        for name, content in structure.items():
            path = base_dir / name
            
            if isinstance(content, dict):
                # It's a directory
                path.mkdir(parents=True, exist_ok=True)
                self.create_test_file_structure(path, content)
            else:
                # It's a file
                path.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(content, str):
                    path.write_text(content)
                elif isinstance(content, bytes):
                    path.write_bytes(content)
                elif content is None:
                    path.touch()
    
    @contextmanager
    def isolated_filesystem(self, structure: Optional[Dict[str, Any]] = None):
        """Context manager for isolated filesystem testing."""
        temp_dir = self.create_temp_dir()
        
        if structure:
            self.create_test_file_structure(temp_dir, structure)
        
        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            yield temp_dir
        finally:
            os.chdir(original_cwd)


class ProcessIsolation:
    """Provides process isolation for tests."""
    
    def __init__(self, test_id: str):
        self.test_id = test_id
        self.context = _isolation_manager.get_context(test_id) or _isolation_manager.create_context(test_id)
    
    def track_process(self, pid: int):
        """Track a process for cleanup."""
        self.context.processes.append(pid)
    
    def find_available_port(self, start_port: int = 8000, max_attempts: int = 100) -> int:
        """Find an available port for testing."""
        for i in range(max_attempts):
            port = start_port + i
            if self._is_port_available(port):
                self.context.ports.append(port)
                return port
        
        raise RuntimeError(f"Could not find available port starting from {start_port}")
    
    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(('localhost', port))
                return True
        except OSError:
            return False
    
    @contextmanager
    def isolated_process_environment(self):
        """Context manager for isolated process environment."""
        # Store original process list
        try:
            import psutil
            original_processes = set(p.pid for p in psutil.process_iter())
        except ImportError:
            original_processes = set()
        
        yield
        
        # Clean up any new processes
        try:
            import psutil
            current_processes = set(p.pid for p in psutil.process_iter())
            new_processes = current_processes - original_processes
            
            for pid in new_processes:
                try:
                    process = psutil.Process(pid)
                    # Only terminate processes that look like test processes
                    cmdline = ' '.join(process.cmdline()).lower()
                    if any(keyword in cmdline for keyword in ['test', 'pytest', 'temp', self.test_id]):
                        process.terminate()
                        self.context.processes.append(pid)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except ImportError:
            pass


class EnvironmentIsolation:
    """Provides environment variable isolation for tests."""
    
    def __init__(self, test_id: str):
        self.test_id = test_id
        self.context = _isolation_manager.get_context(test_id) or _isolation_manager.create_context(test_id)
    
    def set_env_var(self, name: str, value: str):
        """Set an environment variable with cleanup tracking."""
        if name not in self.context.env_vars:
            self.context.env_vars[name] = os.environ.get(name)
        
        os.environ[name] = value
    
    def unset_env_var(self, name: str):
        """Unset an environment variable with cleanup tracking."""
        if name not in self.context.env_vars:
            self.context.env_vars[name] = os.environ.get(name)
        
        os.environ.pop(name, None)
    
    @contextmanager
    def isolated_environment(self, env_vars: Optional[Dict[str, str]] = None, 
                           unset_vars: Optional[List[str]] = None):
        """Context manager for isolated environment variables."""
        if env_vars:
            for name, value in env_vars.items():
                self.set_env_var(name, value)
        
        if unset_vars:
            for name in unset_vars:
                self.unset_env_var(name)
        
        yield


class MockManager:
    """Manages mocks for test isolation."""
    
    def __init__(self, test_id: str):
        self.test_id = test_id
        self.context = _isolation_manager.get_context(test_id) or _isolation_manager.create_context(test_id)
        self.active_patches = []
    
    def create_mock(self, spec=None, **kwargs) -> Mock:
        """Create a mock with cleanup tracking."""
        mock_obj = Mock(spec=spec, **kwargs)
        self.context.mocks.append(mock_obj)
        return mock_obj
    
    def patch_object(self, target, attribute, new=None, **kwargs):
        """Patch an object attribute with cleanup tracking."""
        if new is None:
            new = self.create_mock()
        
        patcher = patch.object(target, attribute, new, **kwargs)
        mock_obj = patcher.start()
        self.context.mocks.append(patcher)
        self.active_patches.append(patcher)
        return mock_obj
    
    def patch_module(self, target, new=None, **kwargs):
        """Patch a module with cleanup tracking."""
        if new is None:
            new = self.create_mock()
        
        patcher = patch(target, new, **kwargs)
        mock_obj = patcher.start()
        self.context.mocks.append(patcher)
        self.active_patches.append(patcher)
        return mock_obj
    
    def mock_external_service(self, service_name: str, responses: Dict[str, Any] = None):
        """Mock external service dependencies."""
        if responses is None:
            responses = {"status": "ok", "data": {}}
        
        mock_service = self.create_mock()
        mock_service.get.return_value = responses
        mock_service.post.return_value = responses
        mock_service.put.return_value = responses
        mock_service.delete.return_value = responses
        
        # Mock common HTTP libraries
        self.patch_module('requests.get', return_value=Mock(json=lambda: responses, status_code=200))
        self.patch_module('requests.post', return_value=Mock(json=lambda: responses, status_code=201))
        self.patch_module('requests.put', return_value=Mock(json=lambda: responses, status_code=200))
        self.patch_module('requests.delete', return_value=Mock(json=lambda: responses, status_code=204))
        
        return mock_service
    
    def mock_database_connection(self, db_type: str = "sqlite"):
        """Mock database connections."""
        mock_connection = self.create_mock()
        mock_cursor = self.create_mock()
        
        mock_connection.cursor.return_value = mock_cursor
        mock_connection.commit.return_value = None
        mock_connection.rollback.return_value = None
        mock_connection.close.return_value = None
        
        mock_cursor.execute.return_value = None
        mock_cursor.fetchone.return_value = None
        mock_cursor.fetchall.return_value = []
        mock_cursor.fetchmany.return_value = []
        
        if db_type == "sqlite":
            self.patch_module('sqlite3.connect', return_value=mock_connection)
        elif db_type == "postgresql":
            self.patch_module('psycopg2.connect', return_value=mock_connection)
        elif db_type == "mysql":
            self.patch_module('mysql.connector.connect', return_value=mock_connection)
        
        return mock_connection
    
    def mock_file_system_operations(self):
        """Mock file system operations for testing."""
        mock_path = self.create_mock()
        mock_path.exists.return_value = True
        mock_path.is_file.return_value = True
        mock_path.is_dir.return_value = False
        mock_path.read_text.return_value = "test content"
        mock_path.write_text.return_value = None
        
        self.patch_module('pathlib.Path', return_value=mock_path)
        self.patch_module('os.path.exists', return_value=True)
        self.patch_module('os.makedirs', return_value=None)
        
        return mock_path
    
    def add_cleanup_callback(self, callback: Callable):
        """Add a custom cleanup callback."""
        self.context.cleanup_callbacks.append(callback)
    
    def cleanup_mocks(self):
        """Clean up all mocks for this manager."""
        for patcher in self.active_patches:
            try:
                patcher.stop()
            except Exception:
                pass
        self.active_patches.clear()


# Pytest fixtures for test isolation
@pytest.fixture
def test_id(request):
    """Generate a unique test ID."""
    return f"{request.module.__name__}::{request.function.__name__}"


@pytest.fixture
def isolation_manager(test_id):
    """Provide isolation manager for a test."""
    context = _isolation_manager.create_context(test_id)
    yield _isolation_manager
    _isolation_manager.cleanup_context(test_id)


@pytest.fixture
def database_isolation(test_id):
    """Provide database isolation for a test."""
    return DatabaseIsolation(test_id)


@pytest.fixture
def filesystem_isolation(test_id):
    """Provide filesystem isolation for a test."""
    return FileSystemIsolation(test_id)


@pytest.fixture
def process_isolation(test_id):
    """Provide process isolation for a test."""
    return ProcessIsolation(test_id)


@pytest.fixture
def environment_isolation(test_id):
    """Provide environment isolation for a test."""
    return EnvironmentIsolation(test_id)


@pytest.fixture
def mock_manager(test_id):
    """Provide mock manager for a test."""
    manager = MockManager(test_id)
    yield manager
    manager.cleanup_mocks()


@pytest.fixture(autouse=True)
def auto_cleanup():
    """Automatically clean up after each test."""
    yield
    # Cleanup is handled by individual isolation fixtures


# Cleanup hook for pytest session end
def pytest_sessionfinish(session, exitstatus):
    """Clean up all isolation contexts at session end."""
    _isolation_manager.cleanup_all()


# Export main classes and functions
__all__ = [
    'TestIsolationManager', 'DatabaseIsolation', 'FileSystemIsolation',
    'ProcessIsolation', 'EnvironmentIsolation', 'MockManager',
    'IsolationContext', 'database_isolation', 'filesystem_isolation',
    'process_isolation', 'environment_isolation', 'mock_manager',
    'isolation_manager', 'test_id'
]