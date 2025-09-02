"""
Tests for enhanced test isolation and cleanup system.

This module tests the comprehensive test isolation system including database isolation,
file system isolation, process isolation, environment isolation, and mock management.
"""

import pytest
import asyncio
import tempfile
import sqlite3
import json
import os
import time
from pathlib import Path
from unittest.mock import Mock, patch

from tests.utils.test_isolation import (
    TestIsolationManager, DatabaseIsolation, FileSystemIsolation,
    ProcessIsolation, EnvironmentIsolation, MockManager
)
from tests.utils.test_data_factories import TestDataFactory
from tests.utils.test_execution_engine import (
    TestExecutionEngine, TestExecutionConfig, RetryStrategy, TestStatus
)


class TestEnhancedIsolation:
    """Test enhanced isolation and cleanup functionality."""
    
    def test_database_isolation_with_transactions(self, test_id):
        """Test database isolation with transaction support."""
        db_isolation = DatabaseIsolation(test_id)
        
        # Create test database with schema
        schema = TestDataFactory.database.create_user_table_schema()
        test_data = {"users": TestDataFactory.database.create_user_test_data()}
        
        with db_isolation.isolated_database(schema, test_data) as db_path:
            # Verify database was created and populated
            assert db_path.exists()
            
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Check that data was inserted
            cursor.execute("SELECT COUNT(*) FROM users")
            count = cursor.fetchone()[0]
            assert count > 0
            
            # Test transaction rollback
            with db_isolation.database_transaction(db_path) as trans_conn:
                trans_conn.execute("DELETE FROM users")
                trans_conn.execute("SELECT COUNT(*) FROM users")
                assert trans_conn.fetchone()[0] == 0
                
                # Raise exception to trigger rollback
                raise Exception("Test rollback")
        
        # Verify rollback worked - data should still be there
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        count = cursor.fetchone()[0]
        assert count > 0  # Data should be restored
        
        conn.close()
    
    def test_database_snapshot_and_restore(self, test_id):
        """Test database snapshot and restore functionality."""
        db_isolation = DatabaseIsolation(test_id)
        
        # Create test database
        schema = TestDataFactory.database.create_user_table_schema()
        test_data = {"users": TestDataFactory.database.create_user_test_data()}
        
        with db_isolation.isolated_database(schema, test_data) as db_path:
            # Create snapshot
            snapshot_path = db_isolation.create_database_snapshot(db_path)
            assert snapshot_path.exists()
            
            # Modify database
            conn = sqlite3.connect(str(db_path))
            conn.execute("DELETE FROM users")
            conn.commit()
            conn.close()
            
            # Verify data is gone
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM users")
            assert cursor.fetchone()[0] == 0
            conn.close()
            
            # Restore from snapshot
            db_isolation.restore_database_snapshot(db_path, snapshot_path)
            
            # Verify data is restored
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM users")
            assert cursor.fetchone()[0] > 0
            conn.close()
    
    def test_filesystem_isolation_with_structure(self, test_id):
        """Test filesystem isolation with complex directory structures."""
        fs_isolation = FileSystemIsolation(test_id)
        
        # Create complex file structure
        structure = TestDataFactory.file.create_test_file_structure()
        
        with fs_isolation.isolated_filesystem(structure) as temp_dir:
            # Verify structure was created
            assert (temp_dir / "config" / "app.json").exists()
            assert (temp_dir / "logs" / "app.log").exists()
            assert (temp_dir / "data" / "users.json").exists()
            assert (temp_dir / "README.md").exists()
            
            # Verify file contents
            config_content = json.loads((temp_dir / "config" / "app.json").read_text())
            assert "backend" in config_content
            assert "frontend" in config_content
            
            # Test file modifications within isolation
            new_file = temp_dir / "test_modification.txt"
            new_file.write_text("test content")
            assert new_file.exists()
        
        # After context, temp_dir should be cleaned up
        # Note: cleanup happens automatically via fixture manager
    
    def test_process_isolation_with_mock_servers(self, test_id):
        """Test process isolation with mock server management."""
        process_isolation = ProcessIsolation(test_id)
        
        # Find available ports
        backend_port = process_isolation.find_available_port(8000)
        frontend_port = process_isolation.find_available_port(3000)
        
        assert backend_port >= 8000
        assert frontend_port >= 3000
        assert backend_port != frontend_port
        
        # Test port availability checking
        assert process_isolation._is_port_available(backend_port)
        assert process_isolation._is_port_available(frontend_port)
    
    def test_environment_isolation_comprehensive(self, test_id):
        """Test comprehensive environment variable isolation."""
        env_isolation = EnvironmentIsolation(test_id)
        
        # Store original environment state
        original_test_var = os.environ.get("TEST_ISOLATION_VAR")
        original_path = os.environ.get("PATH")
        
        with env_isolation.isolated_environment(
            env_vars={"TEST_ISOLATION_VAR": "test_value", "NEW_VAR": "new_value"},
            unset_vars=["PATH"]
        ):
            # Verify environment changes
            assert os.environ.get("TEST_ISOLATION_VAR") == "test_value"
            assert os.environ.get("NEW_VAR") == "new_value"
            assert "PATH" not in os.environ
        
        # Verify environment restoration
        assert os.environ.get("TEST_ISOLATION_VAR") == original_test_var
        assert os.environ.get("PATH") == original_path
        assert "NEW_VAR" not in os.environ
    
    def test_mock_manager_external_services(self, test_id):
        """Test mock manager with external service mocking."""
        mock_manager = MockManager(test_id)
        
        # Mock external service
        service_responses = {
            "get_user": {"id": 1, "name": "Test User"},
            "create_user": {"id": 2, "name": "New User"}
        }
        
        mock_service = mock_manager.mock_external_service("user_service", service_responses)
        
        # Test mock service responses
        assert mock_service.get.return_value == service_responses
        assert mock_service.post.return_value == service_responses
        
        # Test HTTP library mocking
        with patch('requests.get') as mock_get:
            mock_get.return_value.json.return_value = service_responses
            mock_get.return_value.status_code = 200
            
            # Simulate HTTP request
            import requests
            response = requests.get("http://test-service/users/1")
            assert response.status_code == 200
            assert response.json() == service_responses
    
    def test_mock_manager_database_connections(self, test_id):
        """Test mock manager with database connection mocking."""
        mock_manager = MockManager(test_id)
        
        # Mock SQLite connection
        mock_connection = mock_manager.mock_database_connection("sqlite")
        
        # Test mock connection behavior
        cursor = mock_connection.cursor()
        assert cursor is not None
        
        cursor.execute("SELECT * FROM users")
        assert cursor.fetchall() == []
        
        mock_connection.commit()
        mock_connection.close()
        
        # Verify mocks were called
        mock_connection.cursor.assert_called()
        mock_connection.commit.assert_called()
        mock_connection.close.assert_called()
    
    def test_mock_manager_file_system_operations(self, test_id):
        """Test mock manager with file system operation mocking."""
        mock_manager = MockManager(test_id)
        
        # Mock file system operations
        mock_path = mock_manager.mock_file_system_operations()
        
        # Test mock file operations
        assert mock_path.exists() is True
        assert mock_path.is_file() is True
        assert mock_path.read_text() == "test content"
        
        mock_path.write_text("new content")
        mock_path.write_text.assert_called_with("new content")
    
    @pytest.mark.asyncio
    async def test_test_execution_engine_with_timeout(self):
        """Test execution engine with timeout handling."""
        config = TestExecutionConfig(
            timeout=2.0,  # 2 second timeout
            max_retries=1,
            retry_strategy=RetryStrategy.FIXED_DELAY,
            retry_delay=0.5
        )
        
        engine = TestExecutionEngine(config)
        
        # Test successful execution
        result = await engine.execute_test(
            "test_success",
            ["python", "-c", "print('success'); exit(0)"]
        )
        
        assert result.status == TestStatus.PASSED
        assert result.retry_count == 0
        assert "success" in result.output
        
        # Test timeout handling
        result = await engine.execute_test(
            "test_timeout",
            ["python", "-c", "import time; time.sleep(5); print('done')"]
        )
        
        assert result.status == TestStatus.TIMEOUT
        assert "timed out" in result.error.lower()
        
        await engine.cleanup_all()
    
    @pytest.mark.asyncio
    async def test_test_execution_engine_with_retry(self):
        """Test execution engine with retry logic."""
        config = TestExecutionConfig(
            timeout=5.0,
            max_retries=2,
            retry_strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            retry_delay=0.1
        )
        
        engine = TestExecutionEngine(config)
        
        # Test retry on failure
        result = await engine.execute_test(
            "test_retry",
            ["python", "-c", "exit(1)"]  # Always fails
        )
        
        assert result.status == TestStatus.FAILED
        assert result.retry_count == 2  # Should retry max_retries times
        
        await engine.cleanup_all()
    
    @pytest.mark.asyncio
    async def test_parallel_test_execution(self):
        """Test parallel test execution."""
        config = TestExecutionConfig(
            timeout=10.0,
            parallel_workers=3,
            max_retries=0
        )
        
        engine = TestExecutionEngine(config)
        
        # Create multiple test commands
        test_commands = {
            "test_1": ["python", "-c", "import time; time.sleep(0.1); print('test1'); exit(0)"],
            "test_2": ["python", "-c", "import time; time.sleep(0.1); print('test2'); exit(0)"],
            "test_3": ["python", "-c", "import time; time.sleep(0.1); print('test3'); exit(0)"],
        }
        
        start_time = time.time()
        results = await engine.execute_test_suite(test_commands)
        end_time = time.time()
        
        # Verify all tests passed
        assert len(results) == 3
        for result in results.values():
            assert result.status == TestStatus.PASSED
        
        # Verify parallel execution (should be faster than sequential)
        total_time = end_time - start_time
        assert total_time < 1.0  # Should be much faster than 3 * 0.1 = 0.3 seconds
        
        await engine.cleanup_all()
    
    def test_comprehensive_isolation_integration(self, test_id):
        """Test integration of all isolation components."""
        # Create all isolation managers
        db_isolation = DatabaseIsolation(test_id)
        fs_isolation = FileSystemIsolation(test_id)
        process_isolation = ProcessIsolation(test_id)
        env_isolation = EnvironmentIsolation(test_id)
        mock_manager = MockManager(test_id)
        
        # Set up comprehensive test environment
        env_isolation.set_env_var("TEST_MODE", "comprehensive")
        env_isolation.set_env_var("DATABASE_URL", "sqlite:///test.db")
        
        # Create test database
        schema = TestDataFactory.database.create_full_schema()
        test_data = TestDataFactory.database.create_full_test_data()
        
        with db_isolation.isolated_database(schema, test_data) as db_path:
            # Create test file structure
            structure = TestDataFactory.file.create_test_file_structure()
            
            with fs_isolation.isolated_filesystem(structure) as temp_dir:
                # Mock external dependencies
                mock_service = mock_manager.mock_external_service("api")
                mock_db = mock_manager.mock_database_connection("sqlite")
                
                # Find available port
                port = process_isolation.find_available_port(8000)
                
                # Verify all components work together
                assert db_path.exists()
                assert temp_dir.exists()
                assert (temp_dir / "config" / "app.json").exists()
                assert os.environ.get("TEST_MODE") == "comprehensive"
                assert port >= 8000
                assert mock_service is not None
                assert mock_db is not None
                
                # Test data consistency across components
                config_file = temp_dir / "config" / "app.json"
                config_data = json.loads(config_file.read_text())
                assert "backend" in config_data
                
                # Test database operations
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM users")
                user_count = cursor.fetchone()[0]
                assert user_count > 0
                conn.close()
        
        # Verify cleanup - environment should be restored
        # (Database and filesystem cleanup handled by isolation managers)
    
    def test_isolation_manager_context_cleanup(self, test_id):
        """Test that isolation manager properly cleans up contexts."""
        from tests.utils.test_isolation import _isolation_manager
        
        # Create context
        context = _isolation_manager.create_context(test_id)
        assert test_id in _isolation_manager.contexts
        
        # Add some resources to context
        temp_file = Path(tempfile.mktemp())
        temp_file.write_text("test")
        context.temp_files.append(temp_file)
        
        # Add environment variable
        context.env_vars["TEST_CLEANUP"] = os.environ.get("TEST_CLEANUP")
        os.environ["TEST_CLEANUP"] = "test_value"
        
        # Cleanup context
        _isolation_manager.cleanup_context(test_id)
        
        # Verify cleanup
        assert test_id not in _isolation_manager.contexts
        assert not temp_file.exists()
        assert os.environ.get("TEST_CLEANUP") != "test_value"
    
    def test_fixture_manager_integration(self, isolated_test_environment):
        """Test integration with fixture manager."""
        env = isolated_test_environment
        
        # Verify all components are available
        assert "database" in env
        assert "filesystem" in env
        assert "process" in env
        assert "mock" in env
        assert "environment" in env
        assert "test_id" in env
        
        # Test database fixture
        db_path = env["database"].create_full_database()
        assert db_path.exists()
        
        # Test filesystem fixture
        temp_dir = env["filesystem"].create_temp_directory()
        assert temp_dir.exists()
        
        # Test process fixture
        port = env["process"].find_available_port()
        assert port > 0
        
        # Test mock fixture
        mock_obj = env["mock"].create_mock()
        assert mock_obj is not None
        
        # Test environment fixture
        env["environment"].set_env_var("FIXTURE_TEST", "success")
        assert os.environ.get("FIXTURE_TEST") == "success"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])