"""
Tests for the test fixture manager
"""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from tests.fixtures.fixture_manager import (
    TestFixtureManager, FixtureType, FixtureScope, FixtureDefinition,
    FixtureError, get_fixture_manager, reset_fixture_manager
)


class TestTestFixtureManager:
    """Test cases for TestFixtureManager class"""
    
    def setup_method(self):
        """Setup for each test"""
        reset_fixture_manager()
        self.temp_dir = Path(tempfile.mkdtemp())
        self.manager = TestFixtureManager(base_path=self.temp_dir)
    
    def teardown_method(self):
        """Cleanup after each test"""
        self.manager.cleanup_all()
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_init_creates_directories(self):
        """Test that initialization creates required directories"""
        expected_dirs = [
            self.temp_dir / "data",
            self.temp_dir / "mocks",
            self.temp_dir / "configs",
            self.temp_dir / "temp"
        ]
        
        for directory in expected_dirs:
            assert directory.exists()
            assert directory.is_dir()
    
    def test_register_fixture(self):
        """Test fixture registration"""
        self.manager.register_fixture(
            "test_fixture",
            FixtureType.DATA,
            FixtureScope.FUNCTION,
            dependencies=["dependency1"],
            config={"key": "value"}
        )
        
        assert "test_fixture" in self.manager._fixtures
        definition = self.manager._fixtures["test_fixture"]
        assert definition.name == "test_fixture"
        assert definition.fixture_type == FixtureType.DATA
        assert definition.scope == FixtureScope.FUNCTION
        assert definition.dependencies == ["dependency1"]
        assert definition.config == {"key": "value"}
    
    @pytest.mark.asyncio
    async def test_get_data_fixture(self):
        """Test getting data fixture from JSON file"""
        # Create test data file
        test_data = {"test": "data", "number": 42}
        data_file = self.temp_dir / "data" / "test_data.json"
        data_file.parent.mkdir(exist_ok=True)
        
        with open(data_file, 'w') as f:
            json.dump(test_data, f)
        
        # Register fixture
        self.manager.register_fixture(
            "test_data",
            FixtureType.DATA,
            FixtureScope.FUNCTION,
            data_path=data_file
        )
        
        # Get fixture
        result = await self.manager.get_fixture("test_data")
        assert result == test_data
    
    @pytest.mark.asyncio
    async def test_get_config_fixture(self):
        """Test getting config fixture"""
        config_data = {"api_url": "http://test.com", "timeout": 30}
        
        self.manager.register_fixture(
            "test_config",
            FixtureType.CONFIG,
            FixtureScope.SESSION,
            config=config_data
        )
        
        result = await self.manager.get_fixture("test_config")
        assert result == config_data
    
    @pytest.mark.asyncio
    async def test_get_mock_fixture(self):
        """Test getting mock fixture"""
        self.manager.register_fixture(
            "test_mock",
            FixtureType.MOCK,
            FixtureScope.FUNCTION,
            config={"mock_type": "Mock"}
        )
        
        result = await self.manager.get_fixture("test_mock")
        assert isinstance(result, Mock)
    
    @pytest.mark.asyncio
    async def test_get_temporary_fixture(self):
        """Test getting temporary fixture"""
        self.manager.register_fixture(
            "temp_dir",
            FixtureType.TEMPORARY,
            FixtureScope.FUNCTION,
            config={"type": "directory"}
        )
        
        result = await self.manager.get_fixture("temp_dir")
        assert isinstance(result, Path)
        assert result.exists()
        assert result.is_dir()
    
    @pytest.mark.asyncio
    async def test_fixture_dependencies(self):
        """Test fixture dependency resolution"""
        # Register dependency fixture
        self.manager.register_fixture(
            "dependency",
            FixtureType.CONFIG,
            FixtureScope.SESSION,
            config={"dep_value": "test"}
        )
        
        # Register fixture with dependency
        def setup_with_dependency(value, **config):
            # This would normally access the dependency fixture
            return {"main_value": "test", "has_dependency": True}
        
        self.manager.register_fixture(
            "main_fixture",
            FixtureType.CONFIG,
            FixtureScope.FUNCTION,
            dependencies=["dependency"],
            setup_func=setup_with_dependency
        )
        
        # Get main fixture (should resolve dependency first)
        result = await self.manager.get_fixture("main_fixture")
        assert result["has_dependency"] is True
        
        # Verify dependency was also created
        dependency_result = await self.manager.get_fixture("dependency")
        assert dependency_result["dep_value"] == "test"
    
    @pytest.mark.asyncio
    async def test_fixture_scope_reuse(self):
        """Test that fixtures are reused within their scope"""
        self.manager.register_fixture(
            "session_fixture",
            FixtureType.CONFIG,
            FixtureScope.SESSION,
            config={"value": "session_test"}
        )
        
        # Get fixture twice
        result1 = await self.manager.get_fixture("session_fixture")
        result2 = await self.manager.get_fixture("session_fixture")
        
        # Should be the same instance
        assert result1 is result2
    
    def test_cleanup_scope(self):
        """Test cleanup of fixtures by scope"""
        # Create some fixtures in different scopes
        self.manager.register_fixture(
            "function_fixture",
            FixtureType.CONFIG,
            FixtureScope.FUNCTION,
            config={"value": "function"}
        )
        
        self.manager.register_fixture(
            "session_fixture",
            FixtureType.CONFIG,
            FixtureScope.SESSION,
            config={"value": "session"}
        )
        
        # Create instances
        asyncio.run(self.manager.get_fixture("function_fixture"))
        asyncio.run(self.manager.get_fixture("session_fixture"))
        
        # Verify instances exist
        assert len(self.manager._scope_instances[FixtureScope.FUNCTION]) == 1
        assert len(self.manager._scope_instances[FixtureScope.SESSION]) == 1
        
        # Cleanup function scope
        self.manager.cleanup_scope(FixtureScope.FUNCTION)
        
        # Verify function scope cleaned up but session scope remains
        assert len(self.manager._scope_instances[FixtureScope.FUNCTION]) == 0
        assert len(self.manager._scope_instances[FixtureScope.SESSION]) == 1
    
    def test_cleanup_fixture(self):
        """Test cleanup of specific fixture"""
        self.manager.register_fixture(
            "test_fixture",
            FixtureType.CONFIG,
            FixtureScope.FUNCTION,
            config={"value": "test"}
        )
        
        # Create instance
        asyncio.run(self.manager.get_fixture("test_fixture"))
        
        # Verify instance exists
        assert "test_fixture" in self.manager._instances
        
        # Cleanup specific fixture
        self.manager.cleanup_fixture("test_fixture")
        
        # Verify fixture cleaned up
        assert "test_fixture" not in self.manager._instances
    
    @pytest.mark.asyncio
    async def test_fixture_not_found_error(self):
        """Test error when requesting non-existent fixture"""
        with pytest.raises(FixtureError, match="Fixture 'nonexistent' not registered"):
            await self.manager.get_fixture("nonexistent")
    
    @pytest.mark.asyncio
    async def test_dependency_not_found_error(self):
        """Test error when fixture dependency not found"""
        self.manager.register_fixture(
            "bad_fixture",
            FixtureType.CONFIG,
            FixtureScope.FUNCTION,
            dependencies=["nonexistent_dependency"]
        )
        
        with pytest.raises(FixtureError, match="Dependency 'nonexistent_dependency' not found"):
            await self.manager.get_fixture("bad_fixture")
    
    def test_list_fixtures(self):
        """Test listing fixtures"""
        self.manager.register_fixture(
            "data_fixture",
            FixtureType.DATA,
            FixtureScope.FUNCTION
        )
        
        self.manager.register_fixture(
            "mock_fixture",
            FixtureType.MOCK,
            FixtureScope.FUNCTION
        )
        
        # List all fixtures
        all_fixtures = self.manager.list_fixtures()
        assert "data_fixture" in all_fixtures
        assert "mock_fixture" in all_fixtures
        
        # List by type
        data_fixtures = self.manager.list_fixtures(FixtureType.DATA)
        assert "data_fixture" in data_fixtures
        assert "mock_fixture" not in data_fixtures
    
    def test_get_fixture_info(self):
        """Test getting fixture information"""
        self.manager.register_fixture(
            "info_fixture",
            FixtureType.CONFIG,
            FixtureScope.SESSION,
            config={"info": "test"}
        )
        
        info = self.manager.get_fixture_info("info_fixture")
        assert info is not None
        assert info.name == "info_fixture"
        assert info.fixture_type == FixtureType.CONFIG
        assert info.scope == FixtureScope.SESSION
        
        # Test non-existent fixture
        no_info = self.manager.get_fixture_info("nonexistent")
        assert no_info is None
    
    def test_create_fixture_data_file(self):
        """Test creating fixture data files"""
        test_data = {"test": "data", "numbers": [1, 2, 3]}
        
        # Create JSON file
        json_path = self.manager.create_fixture_data_file("test_json", test_data, "json")
        assert json_path.exists()
        assert json_path.suffix == ".json"
        
        # Verify content
        with open(json_path, 'r') as f:
            loaded_data = json.load(f)
        assert loaded_data == test_data
    
    def test_fixture_scope_context_manager(self):
        """Test fixture scope context manager"""
        self.manager.register_fixture(
            "context_fixture",
            FixtureType.CONFIG,
            FixtureScope.FUNCTION,
            config={"value": "context_test"}
        )
        
        with self.manager.fixture_scope(FixtureScope.FUNCTION):
            # Create fixture within context
            asyncio.run(self.manager.get_fixture("context_fixture"))
            
            # Verify fixture exists
            assert len(self.manager._scope_instances[FixtureScope.FUNCTION]) == 1
        
        # After context, fixture should be cleaned up
        assert len(self.manager._scope_instances[FixtureScope.FUNCTION]) == 0
    
    def test_global_fixture_manager(self):
        """Test global fixture manager instance"""
        # First call should create instance
        manager1 = get_fixture_manager()
        assert manager1 is not None
        
        # Second call should return same instance
        manager2 = get_fixture_manager()
        assert manager1 is manager2
        
        # Reset should clear global instance
        reset_fixture_manager()
        manager3 = get_fixture_manager()
        assert manager3 is not manager1


if __name__ == "__main__":
    pytest.main([__file__])
