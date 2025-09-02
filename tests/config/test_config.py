"""
Test Configuration Management System

This module provides a unified test configuration system with YAML-based configuration
management, environment-specific overrides, and comprehensive validation.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


class TestCategory(Enum):
    """Test category enumeration"""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    E2E = "e2e"


class Environment(Enum):
    """Environment enumeration"""
    DEVELOPMENT = "development"
    CI = "ci"
    LOCAL = "local"
    PRODUCTION = "production"


@dataclass
class CategoryConfig:
    """Configuration for a specific test category"""
    description: str
    timeout: int
    parallel: bool
    patterns: List[str]
    coverage_threshold: Optional[int] = None
    requires_services: Optional[List[str]] = None
    requires_full_stack: Optional[bool] = None
    baseline_file: Optional[str] = None


@dataclass
class TestEnvironmentConfig:
    """Test environment configuration"""
    python_version: str
    required_packages: List[str]
    optional_packages: List[str] = field(default_factory=list)


@dataclass
class CoverageConfig:
    """Test coverage configuration"""
    minimum_threshold: int
    exclude_patterns: List[str]
    report_formats: List[str]


@dataclass
class FixtureConfig:
    """Test fixture configuration"""
    shared_data_dir: str
    mock_data_dir: str
    test_configs_dir: str


@dataclass
class ReportingConfig:
    """Test reporting configuration"""
    output_dir: str
    formats: List[str]
    include_coverage: bool
    include_performance: bool


@dataclass
class ParallelExecutionConfig:
    """Parallel execution configuration"""
    max_workers: int
    categories_parallel: List[str]
    categories_sequential: List[str]


@dataclass
class ResourceLimits:
    """Resource limits for test execution"""
    max_memory_mb: Optional[int] = None
    max_cpu_percent: Optional[int] = None
    max_execution_time: Optional[int] = None
    max_concurrent_tests: Optional[int] = None


class TestConfig:
    """
    Unified test configuration management system with YAML-based configuration,
    environment-specific overrides, and comprehensive validation.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None, environment: Optional[str] = None):
        """
        Initialize test configuration
        
        Args:
            config_path: Path to the main configuration file
            environment: Environment name for overrides
        """
        self.config_path = Path(config_path) if config_path else self._get_default_config_path()
        self.environment = Environment(environment) if environment else self._detect_environment()
        
        # Configuration data
        self._config_data: Dict[str, Any] = {}
        self._environment_overrides: Dict[str, Any] = {}
        
        # Parsed configuration objects
        self.categories: Dict[TestCategory, CategoryConfig] = {}
        self.test_environment: Optional[TestEnvironmentConfig] = None
        self.coverage: Optional[CoverageConfig] = None
        self.fixtures: Optional[FixtureConfig] = None
        self.reporting: Optional[ReportingConfig] = None
        self.parallel_execution: Optional[ParallelExecutionConfig] = None
        self.resource_limits: Optional[ResourceLimits] = None
        
        # Load configuration
        self.load_configuration()
    
    def _get_default_config_path(self) -> Path:
        """Get default configuration file path"""
        return Path(__file__).parent / "test-config.yaml"
    
    def _detect_environment(self) -> Environment:
        """Detect current environment"""
        if os.getenv("CI"):
            return Environment.CI
        elif os.getenv("PYTEST_CURRENT_TEST"):
            return Environment.LOCAL
        else:
            return Environment.DEVELOPMENT
    
    def load_configuration(self) -> None:
        """Load configuration from YAML files"""
        try:
            # Load main configuration
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self._config_data = yaml.safe_load(f) or {}
            else:
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            # Load environment-specific overrides
            self._load_environment_overrides()
            
            # Apply overrides
            self._apply_environment_overrides()
            
            # Parse configuration into objects
            self._parse_configuration()
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load test configuration: {e}")
    
    def _load_environment_overrides(self) -> None:
        """Load environment-specific configuration overrides"""
        env_config_path = self.config_path.parent / f"environments" / f"{self.environment.value}.yaml"
        
        if env_config_path.exists():
            try:
                with open(env_config_path, 'r') as f:
                    self._environment_overrides = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Warning: Failed to load environment overrides from {env_config_path}: {e}")
    
    def _apply_environment_overrides(self) -> None:
        """Apply environment-specific overrides to configuration"""
        if not self._environment_overrides:
            return
        
        def deep_merge(base: Dict, override: Dict) -> Dict:
            """Deep merge two dictionaries"""
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        self._config_data = deep_merge(self._config_data, self._environment_overrides)
    
    def _parse_configuration(self) -> None:
        """Parse configuration data into structured objects"""
        # Parse test categories
        categories_data = self._config_data.get("test_categories", {})
        for category_name, category_data in categories_data.items():
            try:
                category = TestCategory(category_name)
                self.categories[category] = CategoryConfig(**category_data)
            except ValueError:
                print(f"Warning: Unknown test category '{category_name}', skipping")
        
        # Parse test environment
        env_data = self._config_data.get("test_environment", {})
        if env_data:
            self.test_environment = TestEnvironmentConfig(**env_data)
        
        # Parse coverage configuration
        coverage_data = self._config_data.get("coverage", {})
        if coverage_data:
            self.coverage = CoverageConfig(**coverage_data)
        
        # Parse fixture configuration
        fixture_data = self._config_data.get("fixtures", {})
        if fixture_data:
            self.fixtures = FixtureConfig(**fixture_data)
        
        # Parse reporting configuration
        reporting_data = self._config_data.get("reporting", {})
        if reporting_data:
            self.reporting = ReportingConfig(**reporting_data)
        
        # Parse parallel execution configuration
        parallel_data = self._config_data.get("parallel_execution", {})
        if parallel_data:
            self.parallel_execution = ParallelExecutionConfig(**parallel_data)
        
        # Parse resource limits
        limits_data = self._config_data.get("resource_limits", {})
        if limits_data:
            self.resource_limits = ResourceLimits(**limits_data)
    
    def get_category_config(self, category: Union[TestCategory, str]) -> Optional[CategoryConfig]:
        """Get configuration for a specific test category"""
        if isinstance(category, str):
            try:
                category = TestCategory(category)
            except ValueError:
                return None
        
        return self.categories.get(category)
    
    def get_timeout(self, category: Union[TestCategory, str]) -> int:
        """Get timeout for a specific test category"""
        config = self.get_category_config(category)
        return config.timeout if config else 300  # Default 5 minutes
    
    def is_parallel_enabled(self, category: Union[TestCategory, str]) -> bool:
        """Check if parallel execution is enabled for a category"""
        config = self.get_category_config(category)
        return config.parallel if config else False
    
    def get_test_patterns(self, category: Union[TestCategory, str]) -> List[str]:
        """Get test file patterns for a category"""
        config = self.get_category_config(category)
        return config.patterns if config else []
    
    def get_coverage_threshold(self, category: Union[TestCategory, str] = None) -> int:
        """Get coverage threshold for a category or global"""
        if category:
            config = self.get_category_config(category)
            if config and config.coverage_threshold:
                return config.coverage_threshold
        
        return self.coverage.minimum_threshold if self.coverage else 70
    
    def get_max_workers(self) -> int:
        """Get maximum number of parallel workers"""
        return self.parallel_execution.max_workers if self.parallel_execution else 4
    
    def get_output_directory(self) -> Path:
        """Get test output directory"""
        output_dir = self.reporting.output_dir if self.reporting else "test_results"
        return Path(output_dir)
    
    def get_fixture_directory(self, fixture_type: str = "shared") -> Path:
        """Get fixture directory path"""
        if not self.fixtures:
            return Path("tests/fixtures")
        
        if fixture_type == "shared":
            return Path(self.fixtures.shared_data_dir)
        elif fixture_type == "mocks":
            return Path(self.fixtures.mock_data_dir)
        elif fixture_type == "configs":
            return Path(self.fixtures.test_configs_dir)
        else:
            return Path(self.fixtures.shared_data_dir)
    
    def validate_configuration(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate categories
        if not self.categories:
            issues.append("No test categories defined")
        
        for category, config in self.categories.items():
            if config.timeout <= 0:
                issues.append(f"Invalid timeout for category {category.value}: {config.timeout}")
            
            if not config.patterns:
                issues.append(f"No test patterns defined for category {category.value}")
        
        # Validate coverage
        if self.coverage:
            if self.coverage.minimum_threshold < 0 or self.coverage.minimum_threshold > 100:
                issues.append(f"Invalid coverage threshold: {self.coverage.minimum_threshold}")
        
        # Validate parallel execution
        if self.parallel_execution:
            if self.parallel_execution.max_workers <= 0:
                issues.append(f"Invalid max_workers: {self.parallel_execution.max_workers}")
        
        # Validate resource limits
        if self.resource_limits:
            if self.resource_limits.max_memory_mb and self.resource_limits.max_memory_mb <= 0:
                issues.append(f"Invalid max_memory_mb: {self.resource_limits.max_memory_mb}")
            
            if self.resource_limits.max_cpu_percent and (
                self.resource_limits.max_cpu_percent <= 0 or self.resource_limits.max_cpu_percent > 100
            ):
                issues.append(f"Invalid max_cpu_percent: {self.resource_limits.max_cpu_percent}")
        
        return issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return self._config_data.copy()
    
    def save_configuration(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save current configuration to file"""
        save_path = Path(path) if path else self.config_path
        
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                yaml.dump(self._config_data, f, default_flow_style=False, indent=2)
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {e}")
    
    def create_environment_override(self, environment: str, overrides: Dict[str, Any]) -> None:
        """Create environment-specific configuration override"""
        env_dir = self.config_path.parent / "environments"
        env_dir.mkdir(exist_ok=True)
        
        env_file = env_dir / f"{environment}.yaml"
        
        try:
            with open(env_file, 'w') as f:
                yaml.dump(overrides, f, default_flow_style=False, indent=2)
        except Exception as e:
            raise ConfigurationError(f"Failed to create environment override: {e}")


class ConfigurationError(Exception):
    """Configuration-related error"""
    pass


# Global configuration instance
_global_config: Optional[TestConfig] = None


def get_test_config(config_path: Optional[Union[str, Path]] = None, 
                   environment: Optional[str] = None,
                   force_reload: bool = False) -> TestConfig:
    """
    Get global test configuration instance
    
    Args:
        config_path: Path to configuration file
        environment: Environment name
        force_reload: Force reload of configuration
    
    Returns:
        TestConfig instance
    """
    global _global_config
    
    if _global_config is None or force_reload:
        _global_config = TestConfig(config_path, environment)
    
    return _global_config


def reset_test_config() -> None:
    """Reset global test configuration (useful for testing)"""
    global _global_config
    _global_config = None