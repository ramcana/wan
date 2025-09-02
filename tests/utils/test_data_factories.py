"""
Test Data Factories

This module provides factories for creating consistent test data across the test suite.
It includes factories for configuration data, process data, user data, and other common
test scenarios to ensure consistent and reliable test data generation.
"""

import json
import time
import uuid
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


class ProcessStatus(Enum):
    """Process status enumeration."""
    RUNNING = "running"
    STOPPED = "stopped"
    STARTING = "starting"
    STOPPING = "stopping"
    ERROR = "error"


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class TestUser:
    """Test user data structure."""
    id: int
    name: str
    email: str
    active: bool = True
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    permissions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestProcess:
    """Test process data structure."""
    pid: int
    name: str
    status: ProcessStatus
    cpu_percent: float = 0.0
    memory_mb: int = 0
    start_time: float = field(default_factory=time.time)
    command_line: List[str] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)


@dataclass
class TestConfiguration:
    """Test configuration data structure."""
    backend: Dict[str, Any] = field(default_factory=dict)
    frontend: Dict[str, Any] = field(default_factory=dict)
    logging: Dict[str, Any] = field(default_factory=dict)
    recovery: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    security: Dict[str, Any] = field(default_factory=dict)


class ConfigurationFactory:
    """Factory for creating test configuration data."""
    
    @staticmethod
    def create_basic_config() -> TestConfiguration:
        """Create a basic test configuration."""
        return TestConfiguration(
            backend={
                "port": 8000,
                "host": "localhost",
                "log_level": "info",
                "timeout": 30,
                "reload": False
            },
            frontend={
                "port": 3000,
                "host": "localhost",
                "open_browser": False,
                "hot_reload": True
            },
            logging={
                "level": "info",
                "file_enabled": True,
                "console_enabled": True,
                "max_file_size": 10485760,
                "backup_count": 5
            },
            recovery={
                "enabled": True,
                "max_retry_attempts": 3,
                "retry_delay": 2.0,
                "exponential_backoff": True,
                "auto_kill_processes": False,
                "fallback_ports": [8080, 8081, 8082, 3001, 3002, 3003]
            },
            environment={
                "python_min_version": "3.8.0",
                "node_min_version": "16.0.0",
                "npm_min_version": "8.0.0",
                "check_virtual_env": True,
                "validate_dependencies": True,
                "auto_install_missing": False
            },
            security={
                "allow_admin_elevation": True,
                "firewall_auto_exception": False,
                "trusted_port_range": [8000, 9000]
            }
        )


class ProcessFactory:
    """Factory for creating test process data."""
    
    @staticmethod
    def create_running_process(name: str = "test_process", pid: Optional[int] = None) -> TestProcess:
        """Create a running test process."""
        if pid is None:
            pid = random.randint(1000, 9999)
        
        return TestProcess(
            pid=pid,
            name=name,
            status=ProcessStatus.RUNNING,
            cpu_percent=random.uniform(0.1, 15.0),
            memory_mb=random.randint(50, 500),
            start_time=time.time() - random.randint(10, 3600),
            command_line=[name, "--test-mode"],
            environment={"TEST_MODE": "true", "PROCESS_NAME": name}
        )
    
    @staticmethod
    def create_process_list(count: int = 5) -> List[TestProcess]:
        """Create a list of test processes."""
        processes = []
        for i in range(count):
            processes.append(ProcessFactory.create_running_process(f"test_process_{i}"))
        return processes


class UserFactory:
    """Factory for creating test user data."""
    
    @staticmethod
    def create_basic_user(name: str = "test_user", email: Optional[str] = None) -> TestUser:
        """Create a basic test user."""
        if email is None:
            email = f"{name.lower().replace(' ', '.')}@example.com"
        
        return TestUser(
            id=random.randint(1, 10000),
            name=name,
            email=email,
            active=True,
            permissions=["read", "write"],
            metadata={"created_by": "test_factory"}
        )
    
    @staticmethod
    def create_user_list(count: int = 10) -> List[TestUser]:
        """Create a list of test users."""
        users = []
        for i in range(count):
            users.append(UserFactory.create_basic_user(f"user_{i}"))
        return users


class FileFactory:
    """Factory for creating test file structures and content."""
    
    @staticmethod
    def create_config_file_content(config_type: str = "basic") -> str:
        """Create configuration file content."""
        if config_type == "basic":
            config = ConfigurationFactory.create_basic_config()
        else:
            config = ConfigurationFactory.create_basic_config()
        
        config_dict = {
            "backend": config.backend,
            "frontend": config.frontend,
            "logging": config.logging,
            "recovery": config.recovery,
            "environment": config.environment,
            "security": config.security
        }
        
        return json.dumps(config_dict, indent=2)
    
    @staticmethod
    def create_test_file_structure() -> Dict[str, Any]:
        """Create a test file structure."""
        return {
            "config": {
                "app.json": FileFactory.create_config_file_content("basic"),
                "test.json": FileFactory.create_config_file_content("basic")
            },
            "logs": {
                "app.log": "Test log content",
                "error.log": "Test error log"
            },
            "data": {
                "users.json": json.dumps([user.__dict__ for user in UserFactory.create_user_list(5)], indent=2, default=str)
            },
            "temp": {},
            "README.md": "# Test Project\n\nThis is a test project structure."
        }


class DatabaseFactory:
    """Factory for creating test database schemas and data."""
    
    @staticmethod
    def create_user_table_schema() -> str:
        """Create user table schema."""
        return """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            active BOOLEAN DEFAULT 1,
            created_at TEXT NOT NULL,
            permissions TEXT,
            metadata TEXT
        );
        """
    
    @staticmethod
    def create_full_schema() -> str:
        """Create full database schema."""
        return DatabaseFactory.create_user_table_schema()
    
    @staticmethod
    def create_user_test_data() -> List[Dict[str, Any]]:
        """Create user test data for database."""
        users = UserFactory.create_user_list(5)
        return [
            {
                "id": user.id,
                "name": user.name,
                "email": user.email,
                "active": user.active,
                "created_at": user.created_at,
                "permissions": json.dumps(user.permissions),
                "metadata": json.dumps(user.metadata)
            }
            for user in users
        ]
    
    @staticmethod
    def create_full_test_data() -> Dict[str, List[Dict[str, Any]]]:
        """Create full test data for all tables."""
        return {
            "users": DatabaseFactory.create_user_test_data()
        }


class NetworkFactory:
    """Factory for creating test network data."""
    
    @staticmethod
    def create_http_request_data(method: str = "GET", url: str = "/api/test") -> Dict[str, Any]:
        """Create HTTP request test data."""
        return {
            "method": method,
            "url": url,
            "headers": {
                "Content-Type": "application/json",
                "User-Agent": "test-client/1.0",
                "Accept": "application/json"
            },
            "body": json.dumps({"test": True, "timestamp": time.time()}) if method in ["POST", "PUT"] else None,
            "timestamp": time.time()
        }
    
    @staticmethod
    def create_http_response_data(status_code: int = 200, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create HTTP response test data."""
        if data is None:
            data = {"success": True, "message": "Test response"}
        
        return {
            "status_code": status_code,
            "headers": {
                "Content-Type": "application/json",
                "Server": "test-server/1.0"
            },
            "body": json.dumps(data),
            "timestamp": time.time()
        }


# Main factory class that combines all factories
class TestDataFactory:
    """Main factory class providing access to all test data factories."""
    
    config = ConfigurationFactory
    process = ProcessFactory
    user = UserFactory
    file = FileFactory
    database = DatabaseFactory
    network = NetworkFactory
    
    @staticmethod
    def create_complete_test_environment() -> Dict[str, Any]:
        """Create a complete test environment with all data types."""
        return {
            "config": TestDataFactory.config.create_basic_config(),
            "processes": TestDataFactory.process.create_process_list(5),
            "users": TestDataFactory.user.create_user_list(10),
            "file_structure": TestDataFactory.file.create_test_file_structure(),
            "database_data": TestDataFactory.database.create_full_test_data(),
            "network_requests": [
                TestDataFactory.network.create_http_request_data("GET", "/api/users"),
                TestDataFactory.network.create_http_request_data("POST", "/api/users"),
                TestDataFactory.network.create_http_request_data("PUT", "/api/users/1")
            ],
            "network_responses": [
                TestDataFactory.network.create_http_response_data(200),
                TestDataFactory.network.create_http_response_data(201),
                TestDataFactory.network.create_http_response_data(404, {"error": "Not found"})
            ]
        }


# Export main classes and functions
__all__ = [
    'TestDataFactory', 'ConfigurationFactory', 'ProcessFactory', 'UserFactory',
    'FileFactory', 'DatabaseFactory', 'NetworkFactory', 'TestUser', 'TestProcess',
    'TestConfiguration', 'ProcessStatus', 'LogLevel'
]