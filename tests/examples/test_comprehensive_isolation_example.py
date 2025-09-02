"""
Comprehensive Test Isolation Example

This example demonstrates how to use the enhanced test isolation and cleanup system
for complex testing scenarios involving databases, file systems, processes, and mocks.
"""

import pytest
import asyncio
import json
import sqlite3
import os
import time
from pathlib import Path
from unittest.mock import patch

from tests.utils.test_data_factories import TestDataFactory
from tests.utils.test_execution_engine import TestExecutionEngine, TestExecutionConfig, RetryStrategy


class TestComprehensiveIsolationExample:
    """Example tests showing comprehensive isolation usage."""
    
    def test_web_application_simulation(self, isolated_test_environment):
        """Simulate testing a web application with database, config, and external services."""
        env = isolated_test_environment
        
        # Step 1: Set up database with test data
        schema = """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            created_at TEXT NOT NULL
        );
        
        CREATE TABLE posts (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );
        """
        
        test_data = {
            "users": [
                {"id": 1, "username": "testuser", "email": "test@example.com", "created_at": "2023-01-01"},
                {"id": 2, "username": "admin", "email": "admin@example.com", "created_at": "2023-01-01"}
            ],
            "posts": [
                {"id": 1, "user_id": 1, "title": "Test Post", "content": "This is a test post", "created_at": "2023-01-02"},
                {"id": 2, "user_id": 2, "title": "Admin Post", "content": "Admin announcement", "created_at": "2023-01-03"}
            ]
        }
        
        with env["database"].isolated_database(schema, test_data) as db_path:
            # Step 2: Create application configuration
            app_config = {
                "database": {"url": f"sqlite:///{db_path}"},
                "api": {"host": "localhost", "port": 8000},
                "external_services": {
                    "email_service": {"url": "http://email-service:8080"},
                    "auth_service": {"url": "http://auth-service:8081"}
                },
                "logging": {"level": "debug", "file": "app.log"}
            }
            
            config_file = env["filesystem"].create_temp_file(
                json.dumps(app_config, indent=2),
                suffix=".json",
                prefix="app_config_"
            )
            
            # Step 3: Set up environment variables
            env["environment"].set_env_var("APP_CONFIG", str(config_file))
            env["environment"].set_env_var("TESTING", "true")
            env["environment"].set_env_var("LOG_LEVEL", "debug")
            
            # Step 4: Mock external services
            email_responses = {"send_email": {"status": "sent", "id": "email_123"}}
            auth_responses = {"validate_token": {"valid": True, "user_id": 1}}
            
            email_service = env["mock"].mock_external_service("email_service", email_responses)
            auth_service = env["mock"].mock_external_service("auth_service", auth_responses)
            
            # Step 5: Test application functionality
            
            # Test database operations
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Verify test data
            cursor.execute("SELECT COUNT(*) FROM users")
            assert cursor.fetchone()[0] == 2
            
            cursor.execute("SELECT COUNT(*) FROM posts")
            assert cursor.fetchone()[0] == 2
            
            # Test user creation
            cursor.execute(
                "INSERT INTO users (username, email, created_at) VALUES (?, ?, ?)",
                ("newuser", "new@example.com", "2023-01-04")
            )
            conn.commit()
            
            cursor.execute("SELECT COUNT(*) FROM users")
            assert cursor.fetchone()[0] == 3
            
            conn.close()
            
            # Test configuration loading
            assert config_file.exists()
            loaded_config = json.loads(config_file.read_text())
            assert loaded_config["database"]["url"] == f"sqlite:///{db_path}"
            
            # Test environment variables
            assert os.environ.get("APP_CONFIG") == str(config_file)
            assert os.environ.get("TESTING") == "true"
            
            # Test external service mocking
            assert email_service.get.return_value == email_responses
            assert auth_service.get.return_value == auth_responses
    
    def test_microservice_integration(self, test_isolation_suite):
        """Test microservice integration with multiple databases and services."""
        isolation = test_isolation_suite
        
        # Create multiple databases for different services
        user_db_schema = """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL
        );
        """
        
        order_db_schema = """
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            user_id INTEGER NOT NULL,
            product_name TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            total_price REAL NOT NULL
        );
        """
        
        # Set up user service database
        user_db = isolation["database"].create_temp_database("user_service.db")
        isolation["database"].create_test_tables(user_db, user_db_schema)
        isolation["database"].populate_test_data(user_db, {
            "users": [
                {"id": 1, "username": "customer1", "email": "customer1@example.com"},
                {"id": 2, "username": "customer2", "email": "customer2@example.com"}
            ]
        })
        
        # Set up order service database
        order_db = isolation["database"].create_temp_database("order_service.db")
        isolation["database"].create_test_tables(order_db, order_db_schema)
        isolation["database"].populate_test_data(order_db, {
            "orders": [
                {"id": 1, "user_id": 1, "product_name": "Widget", "quantity": 2, "total_price": 19.99},
                {"id": 2, "user_id": 2, "product_name": "Gadget", "quantity": 1, "total_price": 29.99}
            ]
        })
        
        # Create service configuration files
        service_configs = isolation["filesystem"].create_temp_directory("service_configs")
        
        user_service_config = service_configs / "user_service.json"
        user_service_config.write_text(json.dumps({
            "database_url": f"sqlite:///{user_db}",
            "port": 8001,
            "external_services": {"auth": "http://auth:8080"}
        }))
        
        order_service_config = service_configs / "order_service.json"
        order_service_config.write_text(json.dumps({
            "database_url": f"sqlite:///{order_db}",
            "port": 8002,
            "external_services": {"user": "http://user-service:8001", "payment": "http://payment:8080"}
        }))
        
        # Set up environment for microservices
        isolation["environment"].set_env_var("USER_SERVICE_CONFIG", str(user_service_config))
        isolation["environment"].set_env_var("ORDER_SERVICE_CONFIG", str(order_service_config))
        isolation["environment"].set_env_var("SERVICE_MESH_ENABLED", "true")
        
        # Mock external dependencies
        auth_service = isolation["mock"].mock_external_service("auth", {
            "validate": {"valid": True, "user_id": 1}
        })
        
        payment_service = isolation["mock"].mock_external_service("payment", {
            "process": {"status": "success", "transaction_id": "txn_123"}
        })
        
        # Test cross-service operations
        
        # Verify user service data
        user_conn = sqlite3.connect(str(user_db))
        user_cursor = user_conn.cursor()
        user_cursor.execute("SELECT * FROM users WHERE id = 1")
        user = user_cursor.fetchone()
        assert user[1] == "customer1"  # username
        user_conn.close()
        
        # Verify order service data
        order_conn = sqlite3.connect(str(order_db))
        order_cursor = order_conn.cursor()
        order_cursor.execute("SELECT * FROM orders WHERE user_id = 1")
        order = order_cursor.fetchone()
        assert order[2] == "Widget"  # product_name
        order_conn.close()
        
        # Test service configuration
        user_config = json.loads(user_service_config.read_text())
        assert user_config["port"] == 8001
        
        order_config = json.loads(order_service_config.read_text())
        assert order_config["port"] == 8002
        
        # Test environment setup
        assert os.environ.get("SERVICE_MESH_ENABLED") == "true"
        
        # Test external service mocks
        assert auth_service.get.return_value["validate"]["valid"] is True
        assert payment_service.get.return_value["process"]["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_performance_testing_with_isolation(self, test_execution_engine):
        """Test performance testing with proper isolation and cleanup."""
        
        # Create test scripts for performance testing
        test_scripts = {
            "cpu_intensive": [
                "python", "-c", 
                "import time; start = time.time(); "
                "sum(i*i for i in range(100000)); "
                "print(f'CPU test completed in {time.time() - start:.2f}s')"
            ],
            "memory_intensive": [
                "python", "-c",
                "import time; start = time.time(); "
                "data = [i for i in range(100000)]; "
                "print(f'Memory test completed in {time.time() - start:.2f}s')"
            ],
            "io_intensive": [
                "python", "-c",
                "import time, tempfile, os; start = time.time(); "
                "with tempfile.NamedTemporaryFile(delete=False) as f: "
                "    f.write(b'test' * 10000); temp_file = f.name; "
                "os.unlink(temp_file); "
                "print(f'IO test completed in {time.time() - start:.2f}s')"
            ]
        }
        
        # Execute performance tests
        results = await test_execution_engine.execute_test_suite(test_scripts)
        
        # Verify all tests completed
        assert len(results) == 3
        
        for test_name, result in results.items():
            assert result.status.value in ["passed", "failed"]  # Should complete, not timeout
            assert result.duration > 0
            assert "completed" in result.output
            
            print(f"{test_name}: {result.duration:.2f}s - {result.status.value}")
        
        # Get performance summary
        summary = test_execution_engine.get_summary()
        assert summary["total_tests"] == 3
        assert summary["total_duration"] > 0
        
        print(f"Performance test summary: {summary}")
    
    def test_error_handling_and_recovery(self, isolated_test_environment):
        """Test error handling and recovery mechanisms."""
        env = isolated_test_environment
        
        # Create a database with intentional issues for testing error handling
        problematic_schema = """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL
        );
        
        -- This will cause issues if we try to insert duplicate usernames
        """
        
        with env["database"].isolated_database(problematic_schema) as db_path:
            conn = sqlite3.connect(str(db_path))
            
            try:
                # Insert first user - should succeed
                conn.execute("INSERT INTO users (username) VALUES (?)", ("user1",))
                conn.commit()
                
                # Try to insert duplicate username - should fail
                with pytest.raises(sqlite3.IntegrityError):
                    conn.execute("INSERT INTO users (username) VALUES (?)", ("user1",))
                    conn.commit()
                
                # Test recovery - rollback and insert different user
                conn.rollback()
                conn.execute("INSERT INTO users (username) VALUES (?)", ("user2",))
                conn.commit()
                
                # Verify recovery worked
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM users")
                assert cursor.fetchone()[0] == 2
                
            finally:
                conn.close()
        
        # Test file system error handling
        temp_dir = env["filesystem"].create_temp_directory()
        
        # Create a file and then try to create a directory with the same name
        test_file = temp_dir / "test_item"
        test_file.write_text("test content")
        
        # This should fail
        with pytest.raises(FileExistsError):
            test_file.mkdir()
        
        # Test mock error handling
        mock_service = env["mock"].create_mock()
        mock_service.get.side_effect = Exception("Service unavailable")
        
        with pytest.raises(Exception, match="Service unavailable"):
            mock_service.get()
        
        # Test recovery - reset mock
        mock_service.reset_mock()
        mock_service.get.return_value = {"status": "ok"}
        
        result = mock_service.get()
        assert result["status"] == "ok"
    
    def test_concurrent_test_isolation(self, test_isolation_suite):
        """Test that isolation works correctly with concurrent operations."""
        isolation = test_isolation_suite
        
        # Create multiple temporary resources concurrently
        import threading
        import queue
        
        results = queue.Queue()
        
        def create_isolated_resources(thread_id):
            try:
                # Each thread creates its own isolated resources
                db = isolation["database"].create_temp_database(f"thread_{thread_id}.db")
                temp_dir = isolation["filesystem"].create_temp_directory(f"thread_{thread_id}_")
                port = isolation["process"].find_available_port(8000 + thread_id * 10)
                
                # Set thread-specific environment
                isolation["environment"].set_env_var(f"THREAD_{thread_id}_VAR", f"value_{thread_id}")
                
                results.put({
                    "thread_id": thread_id,
                    "db": db,
                    "temp_dir": temp_dir,
                    "port": port,
                    "env_var": os.environ.get(f"THREAD_{thread_id}_VAR")
                })
                
            except Exception as e:
                results.put({"thread_id": thread_id, "error": str(e)})
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=create_isolated_resources, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Collect results
        thread_results = []
        while not results.empty():
            thread_results.append(results.get())
        
        # Verify isolation worked correctly
        assert len(thread_results) == 3
        
        for result in thread_results:
            assert "error" not in result, f"Thread {result['thread_id']} failed: {result.get('error')}"
            assert result["db"].exists()
            assert result["temp_dir"].exists()
            assert result["port"] > 0
            assert result["env_var"] == f"value_{result['thread_id']}"
        
        # Verify resources are unique across threads
        dbs = [r["db"] for r in thread_results]
        temp_dirs = [r["temp_dir"] for r in thread_results]
        ports = [r["port"] for r in thread_results]
        
        assert len(set(str(db) for db in dbs)) == 3  # All databases unique
        assert len(set(str(td) for td in temp_dirs)) == 3  # All temp dirs unique
        assert len(set(ports)) == 3  # All ports unique


if __name__ == "__main__":
    pytest.main([__file__, "-v"])