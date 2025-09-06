"""
Fixture Manager for Test Isolation and Cleanup

This module provides a centralized fixture management system that handles
setup and teardown of test environments, ensuring proper isolation and cleanup.
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
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Callable, Union
from contextlib import contextmanager
from unittest.mock import Mock, patch, MagicMock
import pytest
from dataclasses import dataclass, field

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from tests.utils.test_data_factories import TestDataFactory
except ImportError:
    # Fallback if TestDataFactory is not available
    class TestDataFactory:
        class database:
            @staticmethod
            def create_user_table_schema():
                return "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);"
            
            @staticmethod
            def create_user_test_data():
                return [{"id": 1, "name": "test_user"}]
            
            @staticmethod
            def create_full_schema():
                return TestDataFactory.database.create_user_table_schema()
            
            @staticmethod
            def create_full_test_data():
                return {"users": TestDataFactory.database.create_user_test_data()}
        
        class file:
            @staticmethod
            def create_config_file_content(config_type="basic"):
                return '{"test": true}'
            
            @staticmethod
            def create_test_file_structure():
                return {"config": {"app.json": '{"test": true}'}}

try:
    from tests.utils.test_isolation import TestIsolationManager, IsolationContext
except ImportError:
    # Fallback if test_isolation is not available
    class TestIsolationManager:
        def __init__(self):
            pass
    
    class IsolationContext:
        def __init__(self):
            pass


@dataclass
class FixtureState:
    """State tracking for fixtures."""
    active_fixtures: Set[str] = field(default_factory=set)
    cleanup_callbacks: List[Callable] = field(default_factory=list)
    temp_resources: List[Path] = field(default_factory=list)
    mock_objects: List[Any] = field(default_factory=list)
    processes: List[int] = field(default_factory=list)
    databases: List[Path] = field(default_factory=list)


class FixtureManager:
    """Manages test fixtures with proper setup and teardown."""
    
    def __init__(self):
        self.states: Dict[str, FixtureState] = {}
        self.isolation_manager = TestIsolationManager()
        self._lock = threading.Lock()
    
    def get_state(self, test_id: str) -> FixtureState:
        """Get or create fixture state for a test."""
        with self._lock:
            if test_id not in self.states:
                self.states[test_id] = FixtureState()
            return self.states[test_id]
    
    def register_fixture(self, test_id: str, fixture_name: str):
        """Register an active fixture."""
        state = self.get_state(test_id)
        state.active_fixtures.add(fixture_name)
    
    def add_cleanup_callback(self, test_id: str, callback: Callable):
        """Add a cleanup callback for a test."""
        state = self.get_state(test_id)
        state.cleanup_callbacks.append(callback)
    
    def cleanup_test(self, test_id: str):
        """Clean up all fixtures for a test."""
        state = self.states.get(test_id)
        if not state:
            return
        
        # Run cleanup callbacks
        for callback in state.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                print(f"Warning: Cleanup callback failed: {e}")
        
        # Clean up temporary resources
        for resource in state.temp_resources:
            try:
                if resource.is_file():
                    resource.unlink()
                elif resource.is_dir():
                    shutil.rmtree(resource, ignore_errors=True)
            except Exception as e:
                print(f"Warning: Failed to clean up resource {resource}: {e}")
        
        # Clean up databases
        for db_path in state.databases:
            try:
                if db_path.exists():
                    db_path.unlink()
            except Exception as e:
                print(f"Warning: Failed to clean up database {db_path}: {e}")
        
        # Clean up processes
        for pid in state.processes:
            try:
                import psutil
                process = psutil.Process(pid)
                if process.is_running():
                    process.terminate()
                    process.wait(timeout=5)
            except Exception:
                pass
        
        # Clean up mocks
        for mock_obj in state.mock_objects:
            try:
                if hasattr(mock_obj, 'stop'):
                    mock_obj.stop()
                elif hasattr(mock_obj, 'reset_mock'):
                    mock_obj.reset_mock()
            except Exception:
                pass
        
        # Remove state
        del self.states[test_id]
    
    def cleanup_all(self):
        """Clean up all test states."""
        test_ids = list(self.states.keys())
        for test_id in test_ids:
            self.cleanup_test(test_id)


# Global fixture manager instance
_fixture_manager = FixtureManager()


class DatabaseFixture:
    """Database fixture with isolation and cleanup."""
    
    def __init__(self, test_id: str):
        self.test_id = test_id
        self.fixture_manager = _fixture_manager
        self.state = self.fixture_manager.get_state(test_id)
        self.temp_db_dir = None
    
    def create_test_database(self, schema: str = "", test_data: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> Path:
        """Create a test database with schema and data."""
        if not self.temp_db_dir:
            self.temp_db_dir = Path(tempfile.mkdtemp(prefix=f"test_db_{self.test_id}_"))
            self.state.temp_resources.append(self.temp_db_dir)
        
        db_path = self.temp_db_dir / "test.db"
        self.state.databases.append(db_path)
        
        # Create database
        conn = sqlite3.connect(str(db_path))
        try:
            if schema:
                conn.executescript(schema)
            
            if test_data:
                for table_name, rows in test_data.items():
                    if rows:
                        columns = list(rows[0].keys())
                        placeholders = ', '.join(['?' for _ in columns])
                        insert_sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
                        
                        for row in rows:
                            values = [row[col] for col in columns]
                            conn.execute(insert_sql, values)
            
            conn.commit()
        finally:
            conn.close()
        
        return db_path
    
    def create_user_database(self) -> Path:
        """Create a database with user test data."""
        schema = TestDataFactory.database.create_user_table_schema()
        test_data = {"users": TestDataFactory.database.create_user_test_data()}
        return self.create_test_database(schema, test_data)
    
    def create_full_database(self) -> Path:
        """Create a database with all test tables and data."""
        schema = TestDataFactory.database.create_full_schema()
        test_data = TestDataFactory.database.create_full_test_data()
        return self.create_test_database(schema, test_data)


class FileSystemFixture:
    """File system fixture with isolation and cleanup."""
    
    def __init__(self, test_id: str):
        self.test_id = test_id
        self.fixture_manager = _fixture_manager
        self.state = self.fixture_manager.get_state(test_id)
    
    def create_temp_directory(self, prefix: str = "test_") -> Path:
        """Create a temporary directory."""
        temp_dir = Path(tempfile.mkdtemp(prefix=f"{prefix}{self.test_id}_"))
        self.state.temp_resources.append(temp_dir)
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
        
        self.state.temp_resources.append(temp_file)
        return temp_file
    
    def create_config_file(self, config_type: str = "basic") -> Path:
        """Create a configuration file."""
        content = TestDataFactory.file.create_config_file_content(config_type)
        return self.create_temp_file(content, suffix=".json", prefix="config_")
    
    def create_log_file(self, entries: int = 10) -> Path:
        """Create a log file with test entries."""
        content = TestDataFactory.file.create_log_file_content(entries=entries)
        return self.create_temp_file(content, suffix=".log", prefix="test_")
    
    def create_project_structure(self, structure: Optional[Dict[str, Any]] = None) -> Path:
        """Create a test project structure."""
        if structure is None:
            structure = TestDataFactory.file.create_test_file_structure()
        
        base_dir = self.create_temp_directory("project_")
        self._create_structure_recursive(base_dir, structure)
        return base_dir
    
    def _create_structure_recursive(self, base_path: Path, structure: Dict[str, Any]):
        """Recursively create file structure."""
        for name, content in structure.items():
            path = base_path / name
            
            if isinstance(content, dict):
                path.mkdir(parents=True, exist_ok=True)
                self._create_structure_recursive(path, content)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(content, str):
                    path.write_text(content)
                elif isinstance(content, bytes):
                    path.write_bytes(content)
                elif content is None:
                    path.touch()


class ProcessFixture:
    """Process fixture with isolation and cleanup."""
    
    def __init__(self, test_id: str):
        self.test_id = test_id
        self.fixture_manager = _fixture_manager
        self.state = self.fixture_manager.get_state(test_id)
    
    def find_available_port(self, start_port: int = 8000, max_attempts: int = 100) -> int:
        """Find an available port."""
        for i in range(max_attempts):
            port = start_port + i
            if self._is_port_available(port):
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
    
    def start_mock_server(self, port: Optional[int] = None) -> Dict[str, Any]:
        """Start a mock server for testing."""
        if port is None:
            port = self.find_available_port()
        
        # Create a simple mock server using Python's http.server
        import http.server
        import socketserver
        import threading
        
        class MockHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = json.dumps({"status": "ok", "test": True})
                self.wfile.write(response.encode())
            
            def do_POST(self):
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                
                self.send_response(201)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = json.dumps({"status": "created", "received": post_data.decode()})
                self.wfile.write(response.encode())
        
        httpd = socketserver.TCPServer(("localhost", port), MockHandler)
        server_thread = threading.Thread(target=httpd.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        
        # Add cleanup callback
        def cleanup_server():
            httpd.shutdown()
            httpd.server_close()
        
        self.fixture_manager.add_cleanup_callback(self.test_id, cleanup_server)
        
        return {
            "port": port,
            "url": f"http://localhost:{port}",
            "server": httpd,
            "thread": server_thread
        }


class MockFixture:
    """Mock fixture with automatic cleanup."""
    
    def __init__(self, test_id: str):
        self.test_id = test_id
        self.fixture_manager = _fixture_manager
        self.state = self.fixture_manager.get_state(test_id)
        self.active_patches = []
    
    def create_mock(self, spec=None, **kwargs) -> Mock:
        """Create a mock with cleanup tracking."""
        mock_obj = Mock(spec=spec, **kwargs)
        self.state.mock_objects.append(mock_obj)
        return mock_obj
    
    def patch_object(self, target, attribute, new=None, **kwargs):
        """Patch an object attribute."""
        if new is None:
            new = self.create_mock()
        
        patcher = patch.object(target, attribute, new, **kwargs)
        mock_obj = patcher.start()
        self.active_patches.append(patcher)
        self.state.mock_objects.append(patcher)
        return mock_obj
    
    def patch_module(self, target, new=None, **kwargs):
        """Patch a module."""
        if new is None:
            new = self.create_mock()
        
        patcher = patch(target, new, **kwargs)
        mock_obj = patcher.start()
        self.active_patches.append(patcher)
        self.state.mock_objects.append(patcher)
        return mock_obj
    
    def mock_subprocess_run(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        """Mock subprocess.run."""
        mock_result = Mock()
        mock_result.returncode = returncode
        mock_result.stdout = stdout
        mock_result.stderr = stderr
        
        return self.patch_module('subprocess.run', return_value=mock_result)
    
    def mock_psutil_process(self, pid: int = 1234, name: str = "test_process", status: str = "running"):
        """Mock psutil.Process."""
        mock_process = self.create_mock()
        mock_process.pid = pid
        mock_process.name.return_value = name
        mock_process.status.return_value = status
        mock_process.cpu_percent.return_value = 10.0
        mock_process.memory_info.return_value = Mock(rss=1024*1024*100)
        mock_process.terminate.return_value = None
        mock_process.kill.return_value = None
        mock_process.is_running.return_value = (status == "running")
        
        return self.patch_module('psutil.Process', return_value=mock_process)


class EnvironmentFixture:
    """Environment fixture with isolation and cleanup."""
    
    def __init__(self, test_id: str):
        self.test_id = test_id
        self.fixture_manager = _fixture_manager
        self.state = self.fixture_manager.get_state(test_id)
        self.original_env = {}
    
    def set_env_var(self, name: str, value: str):
        """Set an environment variable with cleanup tracking."""
        if name not in self.original_env:
            self.original_env[name] = os.environ.get(name)
        
        os.environ[name] = value
        
        # Add cleanup callback
        def restore_env():
            if self.original_env[name] is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = self.original_env[name]
        
        self.fixture_manager.add_cleanup_callback(self.test_id, restore_env)
    
    def unset_env_var(self, name: str):
        """Unset an environment variable with cleanup tracking."""
        if name not in self.original_env:
            self.original_env[name] = os.environ.get(name)
        
        os.environ.pop(name, None)
        
        # Add cleanup callback
        def restore_env():
            if self.original_env[name] is not None:
                os.environ[name] = self.original_env[name]
        
        self.fixture_manager.add_cleanup_callback(self.test_id, restore_env)
    
    def set_test_environment(self):
        """Set up a standard test environment."""
        test_vars = {
            "TESTING": "true",
            "WAN22_TEST_MODE": "true",
            "LOG_LEVEL": "debug",
            "DISABLE_BROWSER_OPEN": "true",
            "PYTEST_RUNNING": "true"
        }
        
        for name, value in test_vars.items():
            self.set_env_var(name, value)


# Pytest fixtures using the fixture manager
@pytest.fixture
def test_id(request):
    """Generate a unique test ID."""
    return f"{request.module.__name__}::{request.function.__name__}"


@pytest.fixture
def database_fixture(test_id):
    """Provide database fixture."""
    fixture = DatabaseFixture(test_id)
    _fixture_manager.register_fixture(test_id, "database")
    yield fixture
    # Cleanup is handled by the fixture manager


@pytest.fixture
def filesystem_fixture(test_id):
    """Provide filesystem fixture."""
    fixture = FileSystemFixture(test_id)
    _fixture_manager.register_fixture(test_id, "filesystem")
    yield fixture
    # Cleanup is handled by the fixture manager


@pytest.fixture
def process_fixture(test_id):
    """Provide process fixture."""
    fixture = ProcessFixture(test_id)
    _fixture_manager.register_fixture(test_id, "process")
    yield fixture
    # Cleanup is handled by the fixture manager


@pytest.fixture
def mock_fixture(test_id):
    """Provide mock fixture."""
    fixture = MockFixture(test_id)
    _fixture_manager.register_fixture(test_id, "mock")
    yield fixture
    # Cleanup is handled by the fixture manager


@pytest.fixture
def environment_fixture(test_id):
    """Provide environment fixture."""
    fixture = EnvironmentFixture(test_id)
    _fixture_manager.register_fixture(test_id, "environment")
    yield fixture
    # Cleanup is handled by the fixture manager


@pytest.fixture(autouse=True)
def auto_fixture_cleanup(test_id):
    """Automatically clean up fixtures after each test."""
    yield
    _fixture_manager.cleanup_test(test_id)


# Composite fixtures for common scenarios
@pytest.fixture
def isolated_test_environment(test_id, database_fixture, filesystem_fixture, 
                            process_fixture, mock_fixture, environment_fixture):
    """Provide a complete isolated test environment."""
    environment_fixture.set_test_environment()
    
    return {
        "database": database_fixture,
        "filesystem": filesystem_fixture,
        "process": process_fixture,
        "mock": mock_fixture,
        "environment": environment_fixture,
        "test_id": test_id
    }


@pytest.fixture
def test_isolation_suite(test_id):
    """Provide comprehensive test isolation suite."""
    from tests.utils.test_isolation import (
        DatabaseIsolation, FileSystemIsolation, ProcessIsolation,
        EnvironmentIsolation, MockManager
    )
    
    return {
        "database": DatabaseIsolation(test_id),
        "filesystem": FileSystemIsolation(test_id),
        "process": ProcessIsolation(test_id),
        "environment": EnvironmentIsolation(test_id),
        "mock": MockManager(test_id),
        "test_id": test_id
    }


@pytest.fixture
def web_test_environment(isolated_test_environment):
    """Provide an isolated environment for web testing."""
    env = isolated_test_environment
    
    # Start mock servers
    backend_server = env["process"].start_mock_server(8000)
    frontend_server = env["process"].start_mock_server(3000)
    
    # Create test configuration
    config_file = env["filesystem"].create_config_file("development")
    
    # Set up environment variables
    env["environment"].set_env_var("BACKEND_URL", backend_server["url"])
    env["environment"].set_env_var("FRONTEND_URL", frontend_server["url"])
    env["environment"].set_env_var("CONFIG_FILE", str(config_file))
    
    env.update({
        "backend_server": backend_server,
        "frontend_server": frontend_server,
        "config_file": config_file
    })
    
    return env


@pytest.fixture
def database_test_environment(isolated_test_environment):
    """Provide an isolated environment for database testing."""
    env = isolated_test_environment
    
    # Create test database
    db_path = env["database"].create_full_database()
    
    # Set up environment variables
    env["environment"].set_env_var("DATABASE_URL", f"sqlite:///{db_path}")
    env["environment"].set_env_var("TEST_DATABASE", str(db_path))
    
    env.update({
        "database_path": db_path
    })
    
    return env


# Session-level cleanup
def pytest_sessionfinish(session, exitstatus):
    """Clean up all fixtures at session end."""
    _fixture_manager.cleanup_all()


# Export main classes and fixtures
__all__ = [
    'FixtureManager', 'DatabaseFixture', 'FileSystemFixture', 'ProcessFixture',
    'MockFixture', 'EnvironmentFixture', 'database_fixture', 'filesystem_fixture',
    'process_fixture', 'mock_fixture', 'environment_fixture', 'isolated_test_environment',
    'web_test_environment', 'database_test_environment', 'test_id'
]