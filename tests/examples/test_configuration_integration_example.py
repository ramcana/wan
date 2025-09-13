"""
Example Test: Configuration Integration with Test Isolation

This test demonstrates the comprehensive test isolation and cleanup system
including database isolation, filesystem isolation, process management,
and environment variable handling.
"""

import pytest
import json
import time
import sqlite3
from pathlib import Path
from unittest.mock import Mock, patch

# Import test isolation and fixture systems
from tests.fixtures.fixture_manager import (
    database_fixture, filesystem_fixture, process_fixture, 
    mock_fixture, environment_fixture, isolated_test_environment,
    web_test_environment, database_test_environment
)
from tests.config.environment_validator import (
    validate_unit_environment, validate_integration_environment,
    setup_unit_environment, setup_integration_environment
)
from tests.config.test_config import get_test_config, TestCategory
from tests.utils.test_data_factories import TestDataFactory


class TestConfigurationIntegration:
    """Test configuration integration with isolation system."""
    
    def test_basic_isolation_setup(self, isolated_test_environment):
        """Test basic isolation environment setup."""
        env = isolated_test_environment
        
        # Verify all components are available
        assert "database" in env
        assert "filesystem" in env
        assert "process" in env
        assert "mock" in env
        assert "environment" in env
        assert "test_id" in env
        
        # Test environment variables are set
        import os
        assert os.environ.get("TESTING") == "true"
        assert os.environ.get("WAN22_TEST_MODE") == "true"
        assert os.environ.get("PYTEST_RUNNING") == "true"
    
    async def test_database_isolation(self, database_fixture):
        """Test database isolation and cleanup."""
        # Create a test database with schema and data
        db_path = database_fixture.create_full_database()
        
        # Verify database exists and has data
        assert db_path.exists()
        
        conn = sqlite3.connect(str(db_path))
        try:
            # Check tables exist
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            assert "users" in tables
            assert "processes" in tables
            assert "configurations" in tables
            
            # Check data exists
            cursor = conn.execute("SELECT COUNT(*) FROM users")
            user_count = cursor.fetchone()[0]
            assert user_count > 0
        finally:
            conn.close()
        
        # Register a mock fixture
        self.fixture_manager.register_fixture(
            "mock_service",
            FixtureType.MOCK,
            FixtureScope.FUNCTION,
            config={"mock_type": "Mock"}
        )
        
        mock_service = await self.fixture_manager.get_fixture("mock_service")
        assert mock_service is not None
        
        print(f"Test data fixture: {test_data}")
        print(f"Mock service fixture: {type(mock_service)}")
    
    def test_environment_validation_usage(self):
        """Test using the environment validator"""
        # Get validation summary
        summary = self.env_validator.get_validation_summary()
        
        assert "status" in summary
        assert "total_checks" in summary
        assert "ready_for_testing" in summary
        
        # Generate a report
        text_report = self.env_validator.generate_report("text")
        assert "Test Environment Validation Report" in text_report
        
        json_report = self.env_validator.generate_report("json")
        assert json_report.startswith("{")
        
        print(f"Environment status: {summary['status']}")
        print(f"Total checks: {summary['total_checks']}")
        print(f"Ready for testing: {summary['ready_for_testing']}")
    
    @pytest.mark.asyncio
    async def test_integrated_workflow(self):
        """Test integrated workflow using all systems"""
        # 1. Check environment is ready
        summary = self.env_validator.get_validation_summary()
        if not summary["ready_for_testing"]:
            pytest.skip("Environment not ready")
        
        # 2. Get test configuration
        timeout = self.config.get_timeout("integration")
        fixture_dir = self.config.get_fixture_directory("shared")
        
        # 3. Setup test fixtures
        self.fixture_manager.register_fixture(
            "integration_config",
            FixtureType.CONFIG,
            FixtureScope.FUNCTION,
            config={
                "timeout": timeout,
                "fixture_dir": str(fixture_dir),
                "test_mode": True
            }
        )
        
        # 4. Use fixtures in test
        config = await self.fixture_manager.get_fixture("integration_config")
        
        assert config["timeout"] == timeout
        assert config["test_mode"] is True
        
        # 5. Simulate test execution with timeout
        start_time = asyncio.get_event_loop().time()
        
        # Simulate some work
        await asyncio.sleep(0.1)
        
        elapsed_time = asyncio.get_event_loop().time() - start_time
        assert elapsed_time < timeout  # Should complete within timeout
        
        print(f"Integrated test completed in {elapsed_time:.3f}s (timeout: {timeout}s)")
    
    def teardown_method(self):
        """Cleanup after each test method"""
        # Cleanup function-scoped fixtures
        self.fixture_manager.cleanup_scope(FixtureScope.FUNCTION)
    
    @classmethod
    def teardown_class(cls):
        """Cleanup after the test class"""
        # Cleanup all fixtures
        cls.fixture_manager.cleanup_all()


def test_standalone_usage():
    """Example of standalone usage without class setup"""
    # Quick environment check
    validator = validate_test_environment()
    summary = validator.get_validation_summary()
    
    if summary["ready_for_testing"]:
        print("✓ Environment is ready for testing")
    else:
        print(f"✗ Environment has {summary['critical_failures']} critical issues")
    
    # Quick config check
    config = get_test_config()
    unit_timeout = config.get_timeout("unit")
    print(f"Unit test timeout: {unit_timeout}s")
    
    # Quick fixture usage
    async def use_fixture():
        fixture_manager = get_fixture_manager()
        fixture_manager.register_fixture(
            "quick_test",
            FixtureType.CONFIG,
            FixtureScope.FUNCTION,
            config={"value": "quick"}
        )
        
        result = await fixture_manager.get_fixture("quick_test")
        fixture_manager.cleanup_all()
        return result
    
    result = asyncio.run(use_fixture())
    assert result["value"] == "quick"
    print("✓ Quick fixture test passed")


if __name__ == "__main__":
    # Run the standalone test
    test_standalone_usage()
    
    # Run pytest on this file
    pytest.main([__file__, "-v"])
