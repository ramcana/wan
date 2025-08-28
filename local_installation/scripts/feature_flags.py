"""
Feature Flags System for Reliability Enhancements

This module provides a comprehensive feature flag system for gradual rollout
of reliability enhancements. It supports A/B testing, canary deployments,
and progressive feature enablement.

Requirements addressed:
- 1.4: User configurable retry limits and feature control
- 8.1: Health report generation with feature usage metrics
- 8.5: Cross-instance monitoring of feature adoption
"""

import json
import logging
import hashlib
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum


class RolloutStrategy(Enum):
    """Feature rollout strategies."""
    DISABLED = "disabled"
    CANARY = "canary"  # Small percentage of users
    GRADUAL = "gradual"  # Increasing percentage over time
    FULL = "full"  # All users
    A_B_TEST = "a_b_test"  # Split testing


class FeatureState(Enum):
    """Feature states."""
    DISABLED = "disabled"
    ENABLED = "enabled"
    TESTING = "testing"
    DEPRECATED = "deprecated"


@dataclass
class FeatureFlag:
    """Individual feature flag configuration."""
    name: str
    description: str
    state: FeatureState = FeatureState.DISABLED
    rollout_strategy: RolloutStrategy = RolloutStrategy.DISABLED
    rollout_percentage: float = 0.0
    target_groups: List[str] = None
    dependencies: List[str] = None
    created_date: str = None
    last_modified: str = None
    owner: str = "system"
    tags: List[str] = None
    
    def __post_init__(self):
        if self.target_groups is None:
            self.target_groups = []
        if self.dependencies is None:
            self.dependencies = []
        if self.tags is None:
            self.tags = []
        if self.created_date is None:
            self.created_date = datetime.now().isoformat()
        if self.last_modified is None:
            self.last_modified = self.created_date


@dataclass
class RolloutConfig:
    """Rollout configuration for gradual deployment."""
    start_date: str
    end_date: str
    initial_percentage: float = 5.0
    target_percentage: float = 100.0
    increment_percentage: float = 10.0
    increment_interval_hours: int = 24
    success_threshold: float = 0.95  # Success rate required to continue rollout
    rollback_threshold: float = 0.80  # Success rate below which to rollback


class FeatureFlagManager:
    """Manages feature flags for reliability system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize feature flag manager."""
        self.config_path = config_path or self._get_default_config_path()
        self.flags: Dict[str, FeatureFlag] = {}
        self.rollout_configs: Dict[str, RolloutConfig] = {}
        self.usage_metrics: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self._load_flags()
        
        # Initialize default reliability flags
        self._initialize_default_flags()
    
    def _get_default_config_path(self) -> str:
        """Get default feature flags configuration path."""
        script_dir = Path(__file__).parent
        return str(script_dir / "feature_flags.json")
    
    def _load_flags(self) -> bool:
        """Load feature flags from configuration file."""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                
                # Load feature flags
                for flag_name, flag_data in data.get("flags", {}).items():
                    flag_data["state"] = FeatureState(flag_data["state"])
                    flag_data["rollout_strategy"] = RolloutStrategy(flag_data["rollout_strategy"])
                    self.flags[flag_name] = FeatureFlag(**flag_data)
                
                # Load rollout configurations
                for config_name, config_data in data.get("rollout_configs", {}).items():
                    self.rollout_configs[config_name] = RolloutConfig(**config_data)
                
                # Load usage metrics
                self.usage_metrics = data.get("usage_metrics", {})
                
                self.logger.info(f"Loaded {len(self.flags)} feature flags")
                return True
            else:
                self.logger.info("No existing feature flags configuration found")
                return True
        except Exception as e:
            self.logger.error(f"Failed to load feature flags: {e}")
            return False
    
    def _save_flags(self) -> bool:
        """Save feature flags to configuration file."""
        try:
            data = {
                "flags": {},
                "rollout_configs": {},
                "usage_metrics": self.usage_metrics,
                "last_updated": datetime.now().isoformat()
            }
            
            # Serialize feature flags
            for flag_name, flag in self.flags.items():
                flag_dict = asdict(flag)
                flag_dict["state"] = flag.state.value
                flag_dict["rollout_strategy"] = flag.rollout_strategy.value
                data["flags"][flag_name] = flag_dict
            
            # Serialize rollout configurations
            for config_name, config in self.rollout_configs.items():
                data["rollout_configs"][config_name] = asdict(config)
            
            # Ensure directory exists
            Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info("Saved feature flags configuration")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save feature flags: {e}")
            return False
    
    def _initialize_default_flags(self):
        """Initialize default reliability feature flags."""
        default_flags = [
            {
                "name": "enhanced_error_context",
                "description": "Enhanced error context capture with full system state",
                "state": FeatureState.ENABLED,
                "rollout_strategy": RolloutStrategy.FULL,
                "rollout_percentage": 100.0,
                "tags": ["reliability", "error_handling"]
            },
            {
                "name": "missing_method_recovery",
                "description": "Automatic recovery from missing method errors",
                "state": FeatureState.TESTING,
                "rollout_strategy": RolloutStrategy.CANARY,
                "rollout_percentage": 10.0,
                "tags": ["reliability", "recovery", "experimental"]
            },
            {
                "name": "model_validation_recovery",
                "description": "Automatic model validation and recovery system",
                "state": FeatureState.ENABLED,
                "rollout_strategy": RolloutStrategy.GRADUAL,
                "rollout_percentage": 50.0,
                "tags": ["reliability", "models", "recovery"]
            },
            {
                "name": "network_failure_recovery",
                "description": "Network failure recovery with alternative sources",
                "state": FeatureState.ENABLED,
                "rollout_strategy": RolloutStrategy.FULL,
                "rollout_percentage": 100.0,
                "tags": ["reliability", "network", "recovery"]
            },
            {
                "name": "dependency_recovery",
                "description": "Dependency installation failure recovery",
                "state": FeatureState.ENABLED,
                "rollout_strategy": RolloutStrategy.GRADUAL,
                "rollout_percentage": 75.0,
                "tags": ["reliability", "dependencies", "recovery"]
            },
            {
                "name": "pre_installation_validation",
                "description": "Comprehensive pre-installation validation",
                "state": FeatureState.ENABLED,
                "rollout_strategy": RolloutStrategy.FULL,
                "rollout_percentage": 100.0,
                "tags": ["reliability", "validation"]
            },
            {
                "name": "diagnostic_monitoring",
                "description": "Continuous diagnostic monitoring and health checks",
                "state": FeatureState.TESTING,
                "rollout_strategy": RolloutStrategy.CANARY,
                "rollout_percentage": 20.0,
                "tags": ["reliability", "monitoring", "experimental"]
            },
            {
                "name": "health_reporting",
                "description": "Comprehensive health reporting and analytics",
                "state": FeatureState.ENABLED,
                "rollout_strategy": RolloutStrategy.GRADUAL,
                "rollout_percentage": 60.0,
                "tags": ["reliability", "reporting", "analytics"]
            },
            {
                "name": "timeout_management",
                "description": "Advanced timeout management with context awareness",
                "state": FeatureState.ENABLED,
                "rollout_strategy": RolloutStrategy.FULL,
                "rollout_percentage": 100.0,
                "tags": ["reliability", "timeouts"]
            },
            {
                "name": "user_guidance_enhancements",
                "description": "Enhanced user guidance and support system",
                "state": FeatureState.ENABLED,
                "rollout_strategy": RolloutStrategy.GRADUAL,
                "rollout_percentage": 80.0,
                "tags": ["reliability", "user_experience"]
            },
            {
                "name": "intelligent_retry_system",
                "description": "Intelligent retry system with exponential backoff",
                "state": FeatureState.ENABLED,
                "rollout_strategy": RolloutStrategy.FULL,
                "rollout_percentage": 100.0,
                "tags": ["reliability", "retry", "core"]
            },
            {
                "name": "predictive_failure_analysis",
                "description": "Predictive failure analysis and prevention",
                "state": FeatureState.TESTING,
                "rollout_strategy": RolloutStrategy.CANARY,
                "rollout_percentage": 5.0,
                "tags": ["reliability", "prediction", "experimental", "ai"]
            }
        ]
        
        # Add flags that don't already exist
        for flag_data in default_flags:
            flag_name = flag_data["name"]
            if flag_name not in self.flags:
                self.flags[flag_name] = FeatureFlag(**flag_data)
        
        # Save updated configuration
        self._save_flags()
    
    def is_enabled(self, flag_name: str, user_id: Optional[str] = None, 
                   context: Optional[Dict[str, Any]] = None) -> bool:
        """Check if a feature flag is enabled for a user/context."""
        try:
            flag = self.flags.get(flag_name)
            if not flag:
                self.logger.warning(f"Feature flag '{flag_name}' not found")
                return False
            
            # Check if feature is disabled
            if flag.state == FeatureState.DISABLED:
                return False
            
            # Check dependencies
            if not self._check_dependencies(flag):
                return False
            
            # Check rollout strategy
            enabled = self._check_rollout_strategy(flag, user_id, context)
            
            # Track usage
            self._track_usage(flag_name, enabled, user_id, context)
            
            return enabled
            
        except Exception as e:
            self.logger.error(f"Error checking feature flag '{flag_name}': {e}")
            return False
    
    def _check_dependencies(self, flag: FeatureFlag) -> bool:
        """Check if feature dependencies are satisfied."""
        for dependency in flag.dependencies:
            if not self.is_enabled(dependency):
                return False
        return True
    
    def _check_rollout_strategy(self, flag: FeatureFlag, user_id: Optional[str], 
                               context: Optional[Dict[str, Any]]) -> bool:
        """Check rollout strategy to determine if feature should be enabled."""
        if flag.rollout_strategy == RolloutStrategy.DISABLED:
            return False
        elif flag.rollout_strategy == RolloutStrategy.FULL:
            return True
        elif flag.rollout_strategy == RolloutStrategy.CANARY:
            return self._canary_check(flag, user_id)
        elif flag.rollout_strategy == RolloutStrategy.GRADUAL:
            return self._gradual_rollout_check(flag, user_id)
        elif flag.rollout_strategy == RolloutStrategy.A_B_TEST:
            return self._ab_test_check(flag, user_id, context)
        
        return False
    
    def _canary_check(self, flag: FeatureFlag, user_id: Optional[str]) -> bool:
        """Check if user is in canary group."""
        if not user_id:
            user_id = "anonymous"
        
        # Use consistent hashing to determine canary group membership
        hash_input = f"{flag.name}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        percentage = (hash_value % 100) / 100.0
        
        return percentage < (flag.rollout_percentage / 100.0)
    
    def _gradual_rollout_check(self, flag: FeatureFlag, user_id: Optional[str]) -> bool:
        """Check gradual rollout based on time and percentage."""
        rollout_config = self.rollout_configs.get(flag.name)
        if not rollout_config:
            # Use simple percentage-based rollout
            return self._canary_check(flag, user_id)
        
        # Calculate current rollout percentage based on time
        start_date = datetime.fromisoformat(rollout_config.start_date)
        current_date = datetime.now()
        
        if current_date < start_date:
            return False
        
        # Calculate time-based percentage
        hours_elapsed = (current_date - start_date).total_seconds() / 3600
        increments = int(hours_elapsed / rollout_config.increment_interval_hours)
        current_percentage = min(
            rollout_config.initial_percentage + (increments * rollout_config.increment_percentage),
            rollout_config.target_percentage
        )
        
        # Update flag percentage
        flag.rollout_percentage = current_percentage
        
        # Check if user is in rollout group
        return self._canary_check(flag, user_id)
    
    def _ab_test_check(self, flag: FeatureFlag, user_id: Optional[str], 
                      context: Optional[Dict[str, Any]]) -> bool:
        """Check A/B test assignment."""
        if not user_id:
            user_id = "anonymous"
        
        # Use consistent hashing for A/B test assignment
        hash_input = f"ab_test:{flag.name}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        
        # Assign to group A or B (50/50 split)
        group = "A" if (hash_value % 2) == 0 else "B"
        
        # Feature is enabled for group A
        return group == "A"
    
    def _track_usage(self, flag_name: str, enabled: bool, user_id: Optional[str], 
                    context: Optional[Dict[str, Any]]):
        """Track feature flag usage for analytics."""
        try:
            if flag_name not in self.usage_metrics:
                self.usage_metrics[flag_name] = {
                    "total_checks": 0,
                    "enabled_count": 0,
                    "disabled_count": 0,
                    "unique_users": set(),
                    "last_updated": datetime.now().isoformat()
                }
            
            metrics = self.usage_metrics[flag_name]
            metrics["total_checks"] += 1
            
            if enabled:
                metrics["enabled_count"] += 1
            else:
                metrics["disabled_count"] += 1
            
            if user_id:
                metrics["unique_users"].add(user_id)
            
            metrics["last_updated"] = datetime.now().isoformat()
            
            # Convert set to list for JSON serialization
            if isinstance(metrics["unique_users"], set):
                metrics["unique_users"] = list(metrics["unique_users"])
            
        except Exception as e:
            self.logger.error(f"Failed to track usage for '{flag_name}': {e}")
    
    def create_flag(self, name: str, description: str, **kwargs) -> bool:
        """Create a new feature flag."""
        try:
            if name in self.flags:
                self.logger.warning(f"Feature flag '{name}' already exists")
                return False
            
            flag_data = {
                "name": name,
                "description": description,
                **kwargs
            }
            
            self.flags[name] = FeatureFlag(**flag_data)
            self._save_flags()
            
            self.logger.info(f"Created feature flag '{name}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create feature flag '{name}': {e}")
            return False
    
    def update_flag(self, name: str, **updates) -> bool:
        """Update an existing feature flag."""
        try:
            if name not in self.flags:
                self.logger.error(f"Feature flag '{name}' not found")
                return False
            
            flag = self.flags[name]
            
            # Update flag attributes
            for key, value in updates.items():
                if hasattr(flag, key):
                    if key == "state" and isinstance(value, str):
                        value = FeatureState(value)
                    elif key == "rollout_strategy" and isinstance(value, str):
                        value = RolloutStrategy(value)
                    
                    setattr(flag, key, value)
            
            # Update last modified timestamp
            flag.last_modified = datetime.now().isoformat()
            
            self._save_flags()
            
            self.logger.info(f"Updated feature flag '{name}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update feature flag '{name}': {e}")
            return False
    
    def delete_flag(self, name: str) -> bool:
        """Delete a feature flag."""
        try:
            if name not in self.flags:
                self.logger.error(f"Feature flag '{name}' not found")
                return False
            
            del self.flags[name]
            
            # Remove from usage metrics
            if name in self.usage_metrics:
                del self.usage_metrics[name]
            
            # Remove rollout config
            if name in self.rollout_configs:
                del self.rollout_configs[name]
            
            self._save_flags()
            
            self.logger.info(f"Deleted feature flag '{name}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete feature flag '{name}': {e}")
            return False
    
    def setup_gradual_rollout(self, flag_name: str, rollout_config: RolloutConfig) -> bool:
        """Setup gradual rollout for a feature flag."""
        try:
            if flag_name not in self.flags:
                self.logger.error(f"Feature flag '{flag_name}' not found")
                return False
            
            # Update flag rollout strategy
            flag = self.flags[flag_name]
            flag.rollout_strategy = RolloutStrategy.GRADUAL
            flag.rollout_percentage = rollout_config.initial_percentage
            
            # Store rollout configuration
            self.rollout_configs[flag_name] = rollout_config
            
            self._save_flags()
            
            self.logger.info(f"Setup gradual rollout for '{flag_name}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup gradual rollout for '{flag_name}': {e}")
            return False
    
    def get_flag_status(self, flag_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a feature flag."""
        try:
            if flag_name not in self.flags:
                return None
            
            flag = self.flags[flag_name]
            metrics = self.usage_metrics.get(flag_name, {})
            
            status = {
                "flag": asdict(flag),
                "metrics": metrics,
                "rollout_config": asdict(self.rollout_configs[flag_name]) if flag_name in self.rollout_configs else None
            }
            
            # Convert enums to strings for JSON serialization
            status["flag"]["state"] = flag.state.value
            status["flag"]["rollout_strategy"] = flag.rollout_strategy.value
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get status for '{flag_name}': {e}")
            return None
    
    def get_all_flags_status(self) -> Dict[str, Any]:
        """Get status of all feature flags."""
        try:
            status = {}
            for flag_name in self.flags:
                status[flag_name] = self.get_flag_status(flag_name)
            
            return {
                "flags": status,
                "summary": {
                    "total_flags": len(self.flags),
                    "enabled_flags": len([f for f in self.flags.values() if f.state == FeatureState.ENABLED]),
                    "testing_flags": len([f for f in self.flags.values() if f.state == FeatureState.TESTING]),
                    "disabled_flags": len([f for f in self.flags.values() if f.state == FeatureState.DISABLED])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get all flags status: {e}")
            return {}
    
    def export_configuration(self, output_path: str) -> bool:
        """Export feature flags configuration."""
        try:
            config_data = {
                "export_timestamp": datetime.now().isoformat(),
                "flags": {},
                "rollout_configs": {},
                "usage_metrics": self.usage_metrics
            }
            
            # Export flags
            for flag_name, flag in self.flags.items():
                flag_dict = asdict(flag)
                flag_dict["state"] = flag.state.value
                flag_dict["rollout_strategy"] = flag.rollout_strategy.value
                config_data["flags"][flag_name] = flag_dict
            
            # Export rollout configs
            for config_name, config in self.rollout_configs.items():
                config_data["rollout_configs"][config_name] = asdict(config)
            
            with open(output_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            self.logger.info(f"Exported feature flags configuration to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export configuration: {e}")
            return False


# Global feature flag manager instance
_flag_manager = None

def get_feature_flag_manager() -> FeatureFlagManager:
    """Get global feature flag manager instance."""
    global _flag_manager
    if _flag_manager is None:
        _flag_manager = FeatureFlagManager()
    return _flag_manager

def is_feature_enabled(flag_name: str, user_id: Optional[str] = None, 
                      context: Optional[Dict[str, Any]] = None) -> bool:
    """Check if a feature is enabled."""
    return get_feature_flag_manager().is_enabled(flag_name, user_id, context)

def get_reliability_features() -> Dict[str, bool]:
    """Get status of all reliability features."""
    manager = get_feature_flag_manager()
    features = {}
    
    reliability_flags = [
        "enhanced_error_context",
        "missing_method_recovery", 
        "model_validation_recovery",
        "network_failure_recovery",
        "dependency_recovery",
        "pre_installation_validation",
        "diagnostic_monitoring",
        "health_reporting",
        "timeout_management",
        "user_guidance_enhancements",
        "intelligent_retry_system",
        "predictive_failure_analysis"
    ]
    
    for flag_name in reliability_flags:
        features[flag_name] = manager.is_enabled(flag_name)
    
    return features