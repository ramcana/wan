"""
Configuration models for the testing framework
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from enum import Enum


class TestMode(Enum):
    """Available test modes"""
    ENVIRONMENT = "environment"
    PERFORMANCE = "performance"
    INTEGRATION = "integration"
    DIAGNOSTIC = "diagnostic"
    FULL = "full"


@dataclass
class PerformanceTargets:
    """Performance target definitions"""
    target_720p_time_minutes: float = 9.0
    target_1080p_time_minutes: float = 17.0
    max_vram_usage_gb: float = 12.0
    vram_warning_threshold: float = 0.9
    cpu_warning_threshold: float = 80.0
    expected_vram_reduction_percent: float = 80.0


@dataclass
class EnvironmentRequirements:
    """Environment requirement specifications"""
    min_python_version: str = "3.8.0"
    required_packages: List[str] = field(default_factory=list)
    required_env_vars: List[str] = field(default_factory=lambda: ["HF_TOKEN"])
    required_config_fields: List[str] = field(default_factory=lambda: [
        "system", "directories", "optimization", "performance"
    ])


@dataclass
class LocalTestConfiguration:
    """Test framework configuration"""
    test_modes: List[TestMode] = field(default_factory=lambda: [TestMode.FULL])
    performance_targets: PerformanceTargets = field(default_factory=PerformanceTargets)
    environment_requirements: EnvironmentRequirements = field(default_factory=EnvironmentRequirements)
    config_path: str = "config.json"
    requirements_path: str = "requirements.txt"
    env_path: str = ".env"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "test_modes": [mode.value for mode in self.test_modes],
            "performance_targets": {
                "target_720p_time_minutes": self.performance_targets.target_720p_time_minutes,
                "target_1080p_time_minutes": self.performance_targets.target_1080p_time_minutes,
                "max_vram_usage_gb": self.performance_targets.max_vram_usage_gb,
                "vram_warning_threshold": self.performance_targets.vram_warning_threshold,
                "cpu_warning_threshold": self.performance_targets.cpu_warning_threshold,
                "expected_vram_reduction_percent": self.performance_targets.expected_vram_reduction_percent
            },
            "environment_requirements": {
                "min_python_version": self.environment_requirements.min_python_version,
                "required_packages": self.environment_requirements.required_packages,
                "required_env_vars": self.environment_requirements.required_env_vars,
                "required_config_fields": self.environment_requirements.required_config_fields
            },
            "config_path": self.config_path,
            "requirements_path": self.requirements_path,
            "env_path": self.env_path
        }