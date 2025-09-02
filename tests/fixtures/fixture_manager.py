"""
Test Fixture Management System

This module provides a comprehensive test fixture manager for shared test data,
mocks, setup/teardown operations, and fixture dependency resolution.
"""

import json
import pickle
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, Type
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import tempfile
import shutil
import asyncio
from unittest.mock import Mock, MagicMock, patch


class FixtureType(Enum):
    """Fixture type enumeration"""
    DATA = "data"
    MOCK = "mock"
    CONFIG = "config"
    TEMPORARY = "temporary"
    SERVICE = "service"


class FixtureScope(Enum):
    """Fixture scope enumeration"""
    FUNCTION = "function"
    CLASS = "class"
    MODULE = "module"
    SESSION = "session"


@dataclass
class FixtureDefinition:
    """Definition of a test fixture"""
    name: str
    fixture_type: FixtureType
    scope: FixtureScope
    setup_func: Optional[Callable] = None
    teardown_func: Optional[Callable] = None
    dependencies: List[str] = field(default_factory=list)
    data_path: Optional[Path] = None
    config: Dict[str, Any] = field(default_factory=dict)
    is_async: bool = False


@dataclass
class FixtureInstance:
    """Instance of a loaded fixture"""
    definition: FixtureDefinition
    value: Any
    is_active: bool = True
    cleanup_funcs: List[Callable] = field(default_factory=list)


class TestFixtureManager:
    """
    Comprehensive test fixture manager for shared test data, mocks,
    setup/teardown operations, and fixture dependency resolution.
    """
    
    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        """
        Initialize fixture manager
        
        Args:
            base_path: Base path for fixture files
        """
        self.base_path = Path(base_path) if base_path else Path(__file__).parent
        
        # Fixture storage
        self._fixtures: Dict[str, FixtureDefinition] = {}
        self._instances: Dict[str, FixtureInstance] = {}
        self._scope_instances: Dict[FixtureScope, Dict[str, FixtureInstance]] = {
            scope: {} for scope in FixtureScope
        }
        
        # Dependency graph
        self._dependency_graph: Dict[str, List[str]] = {}
        
        # Temporary resources
        self._temp_dirs: List[Path] = []
        self._temp_files: List[Path] = []
        
        # Initialize fixture directories
        self._init_directories()
        
        # Load built-in fixtures
        self._load_builtin_fixtures()
    
    def _init_directories(self) -> None:
        """Initialize fixture directories"""
        directories = [
            self.base_path / "data",
            self.base_path / "mocks",
            self.base_path / "configs",
            self.base_path / "temp"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _load_builtin_fixtures(self) -> None:
        """Load built-in fixture definitions"""
        # Register common data fixtures
        self.register_fixture(
            "sample_config",
            FixtureType.CONFIG,
            FixtureScope.SESSION,
            data_path=self.base_path / "configs" / "sample_config.json"
        )
        
        self.register_fixture(
            "test_database",
            FixtureType.SERVICE,
            FixtureScope.MODULE,
            setup_func=self._setup_test_database,
            teardown_func=self._teardown_test_database
        )
        
        self.register_fixture(
            "mock_api_client",
            FixtureType.MOCK,
            FixtureScope.FUNCTION,
            setup_func=self._setup_mock_api_client
        )
        
        self.register_fixture(
            "temp_directory",
            FixtureType.TEMPORARY,
            FixtureScope.FUNCTION,
            setup_func=self._setup_temp_directory,
            teardown_func=self._cleanup_temp_directory
        )
    
    def register_fixture(self,
                        name: str,
                        fixture_type: FixtureType,
                        scope: FixtureScope,
                        setup_func: Optional[Callable] = None,
                        teardown_func: Optional[Callable] = None,
                        dependencies: Optional[List[str]] = None,
                        data_path: Optional[Union[str, Path]] = None,
                        config: Optional[Dict[str, Any]] = None,
                        is_async: bool = False) -> None:
        """
        Register a new fixture definition
        
        Args:
            name: Fixture name
            fixture_type: Type of fixture
            scope: Fixture scope
            setup_func: Setup function
            teardown_func: Teardown function
            dependencies: List of dependency fixture names
            data_path: Path to fixture data file
            config: Fixture configuration
            is_async: Whether fixture functions are async
        """
        definition = FixtureDefinition(
            name=name,
            fixture_type=fixture_type,
            scope=scope,
            setup_func=setup_func,
            teardown_func=teardown_func,
            dependencies=dependencies or [],
            data_path=Path(data_path) if data_path else None,
            config=config or {},
            is_async=is_async
        )
        
        self._fixtures[name] = definition
        self._dependency_graph[name] = definition.dependencies
    
    async def get_fixture(self, name: str) -> Any:
        """
        Get fixture instance, creating it if necessary
        
        Args:
            name: Fixture name
            
        Returns:
            Fixture value
        """
        if name not in self._fixtures:
            raise FixtureError(f"Fixture '{name}' not registered")
        
        definition = self._fixtures[name]
        
        # Check if fixture already exists in appropriate scope
        scope_instances = self._scope_instances[definition.scope]
        if name in scope_instances and scope_instances[name].is_active:
            return scope_instances[name].value
        
        # Resolve dependencies first
        await self._resolve_dependencies(name)
        
        # Create fixture instance
        instance = await self._create_fixture_instance(definition)
        
        # Store in appropriate scope
        scope_instances[name] = instance
        self._instances[name] = instance
        
        return instance.value
    
    async def _resolve_dependencies(self, name: str) -> None:
        """Resolve fixture dependencies"""
        definition = self._fixtures[name]
        
        for dependency in definition.dependencies:
            if dependency not in self._fixtures:
                raise FixtureError(f"Dependency '{dependency}' not found for fixture '{name}'")
            
            # Recursively resolve dependencies
            await self.get_fixture(dependency)
    
    async def _create_fixture_instance(self, definition: FixtureDefinition) -> FixtureInstance:
        """Create a new fixture instance"""
        value = None
        cleanup_funcs = []
        
        try:
            if definition.fixture_type == FixtureType.DATA:
                value = await self._load_data_fixture(definition)
            elif definition.fixture_type == FixtureType.CONFIG:
                value = await self._load_config_fixture(definition)
            elif definition.fixture_type == FixtureType.MOCK:
                value = await self._create_mock_fixture(definition)
            elif definition.fixture_type == FixtureType.TEMPORARY:
                value = await self._create_temporary_fixture(definition)
            elif definition.fixture_type == FixtureType.SERVICE:
                value = await self._create_service_fixture(definition)
            
            # Run setup function if provided
            if definition.setup_func:
                if definition.is_async:
                    setup_result = await definition.setup_func(value, **definition.config)
                else:
                    setup_result = definition.setup_func(value, **definition.config)
                
                if setup_result is not None:
                    value = setup_result
            
            # Add teardown function to cleanup
            if definition.teardown_func:
                cleanup_funcs.append(definition.teardown_func)
            
            return FixtureInstance(
                definition=definition,
                value=value,
                cleanup_funcs=cleanup_funcs
            )
            
        except Exception as e:
            raise FixtureError(f"Failed to create fixture '{definition.name}': {e}")
    
    async def _load_data_fixture(self, definition: FixtureDefinition) -> Any:
        """Load data fixture from file"""
        if not definition.data_path or not definition.data_path.exists():
            raise FixtureError(f"Data file not found for fixture '{definition.name}'")
        
        file_ext = definition.data_path.suffix.lower()
        
        try:
            if file_ext == '.json':
                with open(definition.data_path, 'r') as f:
                    return json.load(f)
            elif file_ext in ['.yaml', '.yml']:
                with open(definition.data_path, 'r') as f:
                    return yaml.safe_load(f)
            elif file_ext == '.pkl':
                with open(definition.data_path, 'rb') as f:
                    return pickle.load(f)
            else:
                with open(definition.data_path, 'r') as f:
                    return f.read()
        except Exception as e:
            raise FixtureError(f"Failed to load data fixture '{definition.name}': {e}")
    
    async def _load_config_fixture(self, definition: FixtureDefinition) -> Dict[str, Any]:
        """Load configuration fixture"""
        if definition.data_path and definition.data_path.exists():
            return await self._load_data_fixture(definition)
        else:
            return definition.config.copy()
    
    async def _create_mock_fixture(self, definition: FixtureDefinition) -> Mock:
        """Create mock fixture"""
        mock_type = definition.config.get('mock_type', 'Mock')
        
        if mock_type == 'MagicMock':
            return MagicMock(**definition.config.get('mock_kwargs', {}))
        else:
            return Mock(**definition.config.get('mock_kwargs', {}))
    
    async def _create_temporary_fixture(self, definition: FixtureDefinition) -> Path:
        """Create temporary fixture (file or directory)"""
        temp_type = definition.config.get('type', 'directory')
        
        if temp_type == 'directory':
            temp_path = Path(tempfile.mkdtemp())
            self._temp_dirs.append(temp_path)
        else:
            fd, temp_file = tempfile.mkstemp()
            temp_path = Path(temp_file)
            self._temp_files.append(temp_path)
            os.close(fd)
        
        return temp_path
    
    async def _create_service_fixture(self, definition: FixtureDefinition) -> Any:
        """Create service fixture (database, API, etc.)"""
        # This is a placeholder - actual service setup would be defined in setup_func
        return definition.config.copy()
    
    # Built-in fixture setup functions
    def _setup_test_database(self, value: Any, **config) -> Dict[str, Any]:
        """Setup test database fixture"""
        return {
            'connection_string': 'sqlite:///:memory:',
            'tables': [],
            'data': {}
        }
    
    def _teardown_test_database(self, value: Any) -> None:
        """Teardown test database fixture"""
        # Cleanup database resources
        pass
    
    def _setup_mock_api_client(self, value: Any, **config) -> Mock:
        """Setup mock API client fixture"""
        mock_client = Mock()
        mock_client.get.return_value = Mock(status_code=200, json=lambda: {'status': 'ok'})
        mock_client.post.return_value = Mock(status_code=201, json=lambda: {'id': 1})
        return mock_client
    
    def _setup_temp_directory(self, value: Path, **config) -> Path:
        """Setup temporary directory fixture"""
        # Create subdirectories if specified
        subdirs = config.get('subdirs', [])
        for subdir in subdirs:
            (value / subdir).mkdir(parents=True, exist_ok=True)
        
        # Create files if specified
        files = config.get('files', {})
        for filename, content in files.items():
            (value / filename).write_text(content)
        
        return value
    
    def _cleanup_temp_directory(self, value: Path) -> None:
        """Cleanup temporary directory fixture"""
        if value.exists():
            shutil.rmtree(value)
    
    def cleanup_scope(self, scope: FixtureScope) -> None:
        """Cleanup all fixtures in a specific scope"""
        scope_instances = self._scope_instances[scope]
        
        for name, instance in list(scope_instances.items()):
            self._cleanup_fixture_instance(instance)
            del scope_instances[name]
            if name in self._instances:
                del self._instances[name]
    
    def cleanup_fixture(self, name: str) -> None:
        """Cleanup a specific fixture"""
        if name in self._instances:
            instance = self._instances[name]
            self._cleanup_fixture_instance(instance)
            
            # Remove from scope instances
            scope_instances = self._scope_instances[instance.definition.scope]
            if name in scope_instances:
                del scope_instances[name]
            
            del self._instances[name]
    
    def _cleanup_fixture_instance(self, instance: FixtureInstance) -> None:
        """Cleanup a fixture instance"""
        try:
            # Run cleanup functions
            for cleanup_func in instance.cleanup_funcs:
                if instance.definition.is_async:
                    asyncio.create_task(cleanup_func(instance.value))
                else:
                    cleanup_func(instance.value)
            
            instance.is_active = False
            
        except Exception as e:
            print(f"Warning: Failed to cleanup fixture '{instance.definition.name}': {e}")
    
    def cleanup_all(self) -> None:
        """Cleanup all fixtures and temporary resources"""
        # Cleanup all fixture instances
        for scope in FixtureScope:
            self.cleanup_scope(scope)
        
        # Cleanup temporary directories
        for temp_dir in self._temp_dirs:
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"Warning: Failed to cleanup temp directory {temp_dir}: {e}")
        
        # Cleanup temporary files
        for temp_file in self._temp_files:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception as e:
                    print(f"Warning: Failed to cleanup temp file {temp_file}: {e}")
        
        self._temp_dirs.clear()
        self._temp_files.clear()
    
    def list_fixtures(self, fixture_type: Optional[FixtureType] = None) -> List[str]:
        """List registered fixtures, optionally filtered by type"""
        if fixture_type:
            return [name for name, definition in self._fixtures.items() 
                   if definition.fixture_type == fixture_type]
        else:
            return list(self._fixtures.keys())
    
    def get_fixture_info(self, name: str) -> Optional[FixtureDefinition]:
        """Get fixture definition information"""
        return self._fixtures.get(name)
    
    @contextmanager
    def fixture_scope(self, scope: FixtureScope):
        """Context manager for fixture scope lifecycle"""
        try:
            yield
        finally:
            self.cleanup_scope(scope)
    
    def create_fixture_data_file(self, name: str, data: Any, file_format: str = 'json') -> Path:
        """Create a fixture data file"""
        data_dir = self.base_path / "data"
        data_dir.mkdir(exist_ok=True)
        
        if file_format == 'json':
            file_path = data_dir / f"{name}.json"
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        elif file_format in ['yaml', 'yml']:
            file_path = data_dir / f"{name}.yaml"
            with open(file_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        elif file_format == 'pickle':
            file_path = data_dir / f"{name}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        return file_path


class FixtureError(Exception):
    """Fixture-related error"""
    pass


# Global fixture manager instance
_global_fixture_manager: Optional[TestFixtureManager] = None


def get_fixture_manager(base_path: Optional[Union[str, Path]] = None) -> TestFixtureManager:
    """
    Get global fixture manager instance
    
    Args:
        base_path: Base path for fixture files
        
    Returns:
        TestFixtureManager instance
    """
    global _global_fixture_manager
    
    if _global_fixture_manager is None:
        _global_fixture_manager = TestFixtureManager(base_path)
    
    return _global_fixture_manager


def reset_fixture_manager() -> None:
    """Reset global fixture manager (useful for testing)"""
    global _global_fixture_manager
    if _global_fixture_manager:
        _global_fixture_manager.cleanup_all()
    _global_fixture_manager = None