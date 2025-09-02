"""
Example demonstrating integration of test configuration, fixture management,
and environment validation systems.
"""

import pytest
import asyncio
from pathlib import Path

from tests.config.test_config import get_test_config, TestCategory
from tests.fixtures.fixture_manager import get_fixture_manager, FixtureType, FixtureScope
from tests.config.environment_validator import validate_test_environment


class TestConfigurationIntegrationExample:
    """Example test class showing integrated usage"""
    
    @classmethod
    def setup_class(cls):
        """Setup for the test class"""
        # Validate environment before running tests
        cls.env_validator = validate_test_environment()
        summary = cls.env_validator.get_validation_summary()
        
        if not summary["ready_for_testing"]:
            pytest.skip(f"Environment not ready for testing: {summary['critical_failures']} critical failures")
        
        # Get test configuration
        cls.config = get_test_config()
        
        # Get fixture manager
        cls.fixture_manager = get_fixture_manager()
    
    def test_configuration_system_usage(self):
        """Test using the configuration system"""
        # Get configuration for unit tests
        unit_config = self.config.get_category_config(TestCategory.UNIT)
        assert unit_config is not None
        
        # Check timeout settings
        timeout = self.config.get_timeout("unit")
        assert timeout > 0
        
        # Check if parallel execution is enabled
        parallel_enabled = self.config.is_parallel_enabled("unit")
        assert isinstance(parallel_enabled, bool)
        
        # Get test patterns
        patterns = self.config.get_test_patterns("unit")
        assert isinstance(patterns, list)
        
        print(f"Unit test timeout: {timeout}s")
        print(f"Parallel execution: {parallel_enabled}")
        print(f"Test patterns: {patterns}")
    
    @pytest.mark.asyncio
    async def test_fixture_manager_usage(self):
        """Test using the fixture manager"""
        # Register a test fixture
        self.fixture_manager.register_fixture(
            "test_data",
            FixtureType.CONFIG,
            FixtureScope.FUNCTION,
            config={"test_value": "example", "numbers": [1, 2, 3]}
        )
        
        # Get the fixture
        test_data = await self.fixture_manager.get_fixture("test_data")
        assert test_data["test_value"] == "example"
        assert test_data["numbers"] == [1, 2, 3]
        
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