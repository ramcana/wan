"""
Quantization Fallback and Preference System

This module provides automatic fallback mechanisms, user preference persistence,
and quantization compatibility validation for the WAN22 system optimization.
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import threading

from quantization_controller import (
    QuantizationMethod, QuantizationStatus, QuantizationResult,
    HardwareProfile, ModelInfo, UserPreferences
)

logger = logging.getLogger(__name__)


@dataclass
class FallbackRule:
    """Rule for automatic fallback behavior"""
    trigger_condition: str  # "timeout", "error", "memory_exceeded", "quality_low"
    from_method: QuantizationMethod
    to_method: QuantizationMethod
    max_attempts: int
    cooldown_seconds: int
    enabled: bool = True


@dataclass
class CompatibilityCheck:
    """Result of compatibility validation"""
    is_compatible: bool
    confidence_score: float  # 0.0 to 1.0
    warnings: List[str]
    recommendations: List[str]
    estimated_performance: Dict[str, float]
    risk_factors: List[str]


@dataclass
class FallbackHistory:
    """History of fallback operations"""
    timestamp: datetime
    original_method: QuantizationMethod
    fallback_method: QuantizationMethod
    reason: str
    model_name: str
    success: bool
    performance_impact: Optional[float]


@dataclass
class PreferenceProfile:
    """User preference profile with context"""
    name: str
    description: str
    preferences: UserPreferences
    hardware_requirements: Dict[str, Any]
    use_cases: List[str]
    created_at: datetime
    last_used: datetime


class QuantizationFallbackSystem:
    """
    Manages automatic fallback mechanisms and user preference persistence
    for quantization operations.
    """
    
    def __init__(self, preferences_path: str = "quantization_preferences.json",
                 fallback_rules_path: str = "quantization_fallback_rules.json",
                 history_path: str = "quantization_history.json"):
        """
        Initialize the fallback system.
        
        Args:
            preferences_path: Path to user preferences file
            fallback_rules_path: Path to fallback rules configuration
            history_path: Path to fallback history file
        """
        self.preferences_path = Path(preferences_path)
        self.fallback_rules_path = Path(fallback_rules_path)
        self.history_path = Path(history_path)
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.fallback_rules = self._load_fallback_rules()
        self.fallback_history = self._load_fallback_history()
        self.preference_profiles = self._load_preference_profiles()
        
        # Runtime state
        self._attempt_counts: Dict[str, int] = {}
        self._cooldown_timers: Dict[str, datetime] = {}
        self._compatibility_cache: Dict[str, CompatibilityCheck] = {}
        
        self.logger.info("QuantizationFallbackSystem initialized")
    
    def should_attempt_fallback(self, result: QuantizationResult, model_info: ModelInfo,
                              hardware_profile: HardwareProfile) -> Tuple[bool, Optional[QuantizationMethod]]:
        """
        Determine if fallback should be attempted based on result and rules.
        
        Args:
            result: Result of failed quantization attempt
            model_info: Information about the model
            hardware_profile: Current hardware profile
            
        Returns:
            Tuple of (should_fallback, fallback_method)
        """
        if result.success:
            return False, None
        
        # Generate cache key for attempt tracking
        cache_key = f"{model_info.name}_{result.method_used.value}"
        
        # Check attempt limits
        current_attempts = self._attempt_counts.get(cache_key, 0)
        
        # Check cooldown
        if cache_key in self._cooldown_timers:
            if datetime.now() < self._cooldown_timers[cache_key]:
                self.logger.info(f"Fallback for {cache_key} is in cooldown period")
                return False, None
        
        # Find applicable fallback rule
        trigger_condition = self._determine_trigger_condition(result)
        fallback_rule = self._find_fallback_rule(result.method_used, trigger_condition)
        
        if not fallback_rule or not fallback_rule.enabled:
            self.logger.info(f"No applicable fallback rule for {result.method_used.value} with {trigger_condition}")
            return False, None
        
        if current_attempts >= fallback_rule.max_attempts:
            self.logger.info(f"Max fallback attempts ({fallback_rule.max_attempts}) reached for {cache_key}")
            return False, None
        
        # Validate fallback method compatibility
        compatibility = self.validate_quantization_compatibility(
            model_info, fallback_rule.to_method, hardware_profile
        )
        
        if not compatibility.is_compatible:
            self.logger.warning(f"Fallback method {fallback_rule.to_method.value} is not compatible")
            return False, None
        
        # Update attempt count
        self._attempt_counts[cache_key] = current_attempts + 1
        
        self.logger.info(f"Fallback approved: {result.method_used.value} -> {fallback_rule.to_method.value}")
        return True, fallback_rule.to_method
    
    def record_fallback_attempt(self, original_method: QuantizationMethod, 
                              fallback_method: QuantizationMethod,
                              reason: str, model_name: str, success: bool,
                              performance_impact: Optional[float] = None) -> None:
        """
        Record a fallback attempt in history.
        
        Args:
            original_method: Original quantization method that failed
            fallback_method: Fallback method that was used
            reason: Reason for fallback
            model_name: Name of the model
            success: Whether the fallback was successful
            performance_impact: Performance impact of fallback (if measurable)
        """
        history_entry = FallbackHistory(
            timestamp=datetime.now(),
            original_method=original_method,
            fallback_method=fallback_method,
            reason=reason,
            model_name=model_name,
            success=success,
            performance_impact=performance_impact
        )
        
        self.fallback_history.append(history_entry)
        
        # Set cooldown timer if fallback failed
        if not success:
            cache_key = f"{model_name}_{original_method.value}"
            fallback_rule = self._find_fallback_rule(original_method, reason)
            if fallback_rule:
                cooldown_end = datetime.now() + timedelta(seconds=fallback_rule.cooldown_seconds)
                self._cooldown_timers[cache_key] = cooldown_end
        
        # Save history
        self._save_fallback_history()
        
        self.logger.info(f"Recorded fallback: {original_method.value} -> {fallback_method.value} ({'success' if success else 'failed'})")
    
    def validate_quantization_compatibility(self, model_info: ModelInfo, 
                                          method: QuantizationMethod,
                                          hardware_profile: HardwareProfile) -> CompatibilityCheck:
        """
        Validate quantization compatibility with comprehensive checks.
        
        Args:
            model_info: Information about the model
            method: Quantization method to validate
            hardware_profile: Current hardware profile
            
        Returns:
            Detailed compatibility check result
        """
        # Check cache first
        cache_key = f"{model_info.name}_{method.value}_{hardware_profile.gpu_model}"
        if cache_key in self._compatibility_cache:
            cached_result = self._compatibility_cache[cache_key]
            # Use cached result if it's less than 1 hour old
            if hasattr(cached_result, 'timestamp'):
                if datetime.now() - cached_result.timestamp < timedelta(hours=1):
                    return cached_result
        
        warnings = []
        recommendations = []
        risk_factors = []
        confidence_score = 1.0
        
        # Hardware compatibility checks
        if method == QuantizationMethod.NONE:
            estimated_performance = {
                "memory_usage_gb": model_info.estimated_vram_usage / 1024,
                "speed_multiplier": 1.0,
                "quality_retention": 1.0
            }
        
        elif method == QuantizationMethod.FP16:
            if not hardware_profile.supports_bf16:
                confidence_score *= 0.9
                warnings.append("Hardware may have limited FP16 performance")
            
            estimated_performance = {
                "memory_usage_gb": model_info.estimated_vram_usage / 1024 * 0.8,
                "speed_multiplier": 1.1,
                "quality_retention": 0.98
            }
        
        elif method == QuantizationMethod.BF16:
            if not hardware_profile.supports_bf16:
                confidence_score = 0.0
                warnings.append("Hardware does not support BF16")
                recommendations.append("Use FP16 instead")
                return CompatibilityCheck(
                    is_compatible=False,
                    confidence_score=confidence_score,
                    warnings=warnings,
                    recommendations=recommendations,
                    estimated_performance={},
                    risk_factors=["Unsupported hardware"]
                )
            
            estimated_performance = {
                "memory_usage_gb": model_info.estimated_vram_usage / 1024 * 0.7,
                "speed_multiplier": 1.2,
                "quality_retention": 0.97
            }
        
        elif method == QuantizationMethod.INT8:
            if not hardware_profile.supports_int8:
                confidence_score = 0.0
                warnings.append("INT8 quantization libraries not available")
                recommendations.append("Install bitsandbytes for INT8 support")
                return CompatibilityCheck(
                    is_compatible=False,
                    confidence_score=confidence_score,
                    warnings=warnings,
                    recommendations=recommendations,
                    estimated_performance={},
                    risk_factors=["Missing dependencies"]
                )
            
            # INT8 has higher quality risk for certain components
            if "transformer" in model_info.components:
                confidence_score *= 0.8
                risk_factors.append("Transformer quantization may impact quality")
                recommendations.append("Monitor output quality carefully")
            
            estimated_performance = {
                "memory_usage_gb": model_info.estimated_vram_usage / 1024 * 0.5,
                "speed_multiplier": 0.9,  # May be slower due to dequantization
                "quality_retention": 0.85
            }
        
        elif method == QuantizationMethod.FP8:
            if not hardware_profile.supports_fp8:
                confidence_score = 0.0
                warnings.append("Hardware does not support FP8")
                recommendations.append("Use BF16 or INT8 instead")
                return CompatibilityCheck(
                    is_compatible=False,
                    confidence_score=confidence_score,
                    warnings=warnings,
                    recommendations=recommendations,
                    estimated_performance={},
                    risk_factors=["Unsupported hardware"]
                )
            
            confidence_score *= 0.7  # Experimental
            risk_factors.append("FP8 is experimental and may be unstable")
            warnings.append("FP8 quantization is experimental")
            
            estimated_performance = {
                "memory_usage_gb": model_info.estimated_vram_usage / 1024 * 0.4,
                "speed_multiplier": 1.3,
                "quality_retention": 0.80
            }
        
        # VRAM capacity checks
        estimated_vram_gb = estimated_performance.get("memory_usage_gb", 0)
        available_vram_gb = hardware_profile.vram_gb * 0.9  # 90% threshold
        
        if estimated_vram_gb > available_vram_gb:
            confidence_score *= 0.5
            risk_factors.append("Estimated VRAM usage exceeds available memory")
            warnings.append(f"May require {estimated_vram_gb:.1f}GB but only {available_vram_gb:.1f}GB available")
            recommendations.append("Consider more aggressive quantization")
        
        # Model size considerations
        if model_info.size_gb > 10 and method in [QuantizationMethod.NONE, QuantizationMethod.FP16]:
            confidence_score *= 0.8
            warnings.append("Large model with minimal quantization may cause memory issues")
            recommendations.append("Consider BF16 or INT8 quantization")
        
        # Architecture-specific considerations
        if "wan" in model_info.architecture.lower():
            if method == QuantizationMethod.INT8:
                confidence_score *= 0.9
                warnings.append("WAN models may be sensitive to aggressive quantization")
        
        # Historical performance analysis
        historical_success_rate = self._get_historical_success_rate(model_info.name, method)
        if historical_success_rate < 0.8:
            confidence_score *= historical_success_rate
            warnings.append(f"Historical success rate for this combination: {historical_success_rate:.1%}")
        
        # Final compatibility determination
        is_compatible = confidence_score > 0.5 and len([r for r in risk_factors if "Unsupported" in r or "Missing" in r]) == 0
        
        result = CompatibilityCheck(
            is_compatible=is_compatible,
            confidence_score=confidence_score,
            warnings=warnings,
            recommendations=recommendations,
            estimated_performance=estimated_performance,
            risk_factors=risk_factors
        )
        
        # Cache result
        result.timestamp = datetime.now()  # Add timestamp for cache validation
        self._compatibility_cache[cache_key] = result
        
        return result
    
    def get_recommended_method(self, model_info: ModelInfo, 
                             hardware_profile: HardwareProfile,
                             user_preferences: UserPreferences) -> Tuple[QuantizationMethod, float]:
        """
        Get recommended quantization method based on model, hardware, and preferences.
        
        Args:
            model_info: Information about the model
            hardware_profile: Current hardware profile
            user_preferences: User preferences
            
        Returns:
            Tuple of (recommended_method, confidence_score)
        """
        # Check model-specific preferences first
        if model_info.name in user_preferences.model_specific_preferences:
            preferred_method = user_preferences.model_specific_preferences[model_info.name]
            compatibility = self.validate_quantization_compatibility(model_info, preferred_method, hardware_profile)
            if compatibility.is_compatible:
                return preferred_method, compatibility.confidence_score
        
        # Evaluate all supported methods
        method_scores = {}
        
        for method in QuantizationMethod:
            compatibility = self.validate_quantization_compatibility(model_info, method, hardware_profile)
            
            if not compatibility.is_compatible:
                continue
            
            # Calculate score based on multiple factors
            score = compatibility.confidence_score
            
            # Preference bonus
            if method == user_preferences.preferred_method:
                score *= 1.2
            
            # Performance considerations
            perf = compatibility.estimated_performance
            if perf:
                # Balance memory savings vs quality retention
                memory_factor = 1.0 - (perf.get("memory_usage_gb", 10) / 16)  # Normalize to 16GB
                quality_factor = perf.get("quality_retention", 1.0)
                speed_factor = perf.get("speed_multiplier", 1.0)
                
                performance_score = (memory_factor * 0.4 + quality_factor * 0.4 + speed_factor * 0.2)
                score *= performance_score
            
            # Historical success rate
            historical_rate = self._get_historical_success_rate(model_info.name, method)
            score *= historical_rate
            
            method_scores[method] = score
        
        if not method_scores:
            # Fallback to NONE if nothing else works
            return QuantizationMethod.NONE, 0.5
        
        # Return method with highest score
        best_method = max(method_scores.items(), key=lambda x: x[1])
        return best_method[0], best_method[1]
    
    def create_preference_profile(self, name: str, description: str,
                                preferences: UserPreferences,
                                hardware_requirements: Dict[str, Any],
                                use_cases: List[str]) -> PreferenceProfile:
        """
        Create a new preference profile.
        
        Args:
            name: Profile name
            description: Profile description
            preferences: User preferences
            hardware_requirements: Hardware requirements
            use_cases: List of use cases
            
        Returns:
            Created preference profile
        """
        profile = PreferenceProfile(
            name=name,
            description=description,
            preferences=preferences,
            hardware_requirements=hardware_requirements,
            use_cases=use_cases,
            created_at=datetime.now(),
            last_used=datetime.now()
        )
        
        self.preference_profiles[name] = profile
        self._save_preference_profiles()
        
        self.logger.info(f"Created preference profile: {name}")
        return profile
    
    def get_matching_profiles(self, hardware_profile: HardwareProfile,
                            use_case: str) -> List[PreferenceProfile]:
        """
        Get preference profiles matching current hardware and use case.
        
        Args:
            hardware_profile: Current hardware profile
            use_case: Current use case
            
        Returns:
            List of matching preference profiles
        """
        matching_profiles = []
        
        for profile in self.preference_profiles.values():
            # Check hardware compatibility
            hw_reqs = profile.hardware_requirements
            
            if "min_vram_gb" in hw_reqs:
                if hardware_profile.vram_gb < hw_reqs["min_vram_gb"]:
                    continue
            
            if "required_gpu_features" in hw_reqs:
                required_features = hw_reqs["required_gpu_features"]
                if "bf16" in required_features and not hardware_profile.supports_bf16:
                    continue
                if "int8" in required_features and not hardware_profile.supports_int8:
                    continue
            
            # Check use case match
            if use_case in profile.use_cases or "general" in profile.use_cases:
                matching_profiles.append(profile)
        
        # Sort by last used (most recent first)
        matching_profiles.sort(key=lambda p: p.last_used, reverse=True)
        
        return matching_profiles
    
    def update_fallback_rules(self, rules: List[FallbackRule]) -> None:
        """Update fallback rules configuration"""
        self.fallback_rules = {
            f"{rule.from_method.value}_{rule.trigger_condition}": rule
            for rule in rules
        }
        self._save_fallback_rules()
        self.logger.info(f"Updated {len(rules)} fallback rules")
    
    def get_fallback_statistics(self) -> Dict[str, Any]:
        """Get statistics about fallback operations"""
        if not self.fallback_history:
            return {"total_fallbacks": 0}
        
        total_fallbacks = len(self.fallback_history)
        successful_fallbacks = sum(1 for h in self.fallback_history if h.success)
        
        # Group by method transitions
        method_transitions = {}
        for history in self.fallback_history:
            transition = f"{history.original_method.value} -> {history.fallback_method.value}"
            if transition not in method_transitions:
                method_transitions[transition] = {"total": 0, "successful": 0}
            method_transitions[transition]["total"] += 1
            if history.success:
                method_transitions[transition]["successful"] += 1
        
        # Calculate success rates
        for transition_data in method_transitions.values():
            transition_data["success_rate"] = transition_data["successful"] / transition_data["total"]
        
        # Recent trends (last 30 days)
        recent_cutoff = datetime.now() - timedelta(days=30)
        recent_fallbacks = [h for h in self.fallback_history if h.timestamp > recent_cutoff]
        
        return {
            "total_fallbacks": total_fallbacks,
            "successful_fallbacks": successful_fallbacks,
            "overall_success_rate": successful_fallbacks / total_fallbacks if total_fallbacks > 0 else 0,
            "method_transitions": method_transitions,
            "recent_fallbacks_30d": len(recent_fallbacks),
            "most_common_reasons": self._get_most_common_reasons(),
            "average_performance_impact": self._calculate_average_performance_impact()
        }
    
    def _determine_trigger_condition(self, result: QuantizationResult) -> str:
        """Determine trigger condition from quantization result"""
        if result.status == QuantizationStatus.TIMEOUT:
            return "timeout"
        elif result.status == QuantizationStatus.FAILED:
            if "memory" in " ".join(result.errors).lower():
                return "memory_exceeded"
            return "error"
        elif result.quality_score and result.quality_score < 0.8:
            return "quality_low"
        else:
            return "error"
    
    def _find_fallback_rule(self, method: QuantizationMethod, trigger: str) -> Optional[FallbackRule]:
        """Find applicable fallback rule"""
        rule_key = f"{method.value}_{trigger}"
        return self.fallback_rules.get(rule_key)
    
    def _get_historical_success_rate(self, model_name: str, method: QuantizationMethod) -> float:
        """Get historical success rate for model/method combination"""
        relevant_history = [
            h for h in self.fallback_history
            if h.model_name == model_name and h.fallback_method == method
        ]
        
        if not relevant_history:
            return 0.9  # Default optimistic rate
        
        successful = sum(1 for h in relevant_history if h.success)
        return successful / len(relevant_history)
    
    def _get_most_common_reasons(self) -> Dict[str, int]:
        """Get most common fallback reasons"""
        reasons = {}
        for history in self.fallback_history:
            reason = history.reason
            reasons[reason] = reasons.get(reason, 0) + 1
        
        return dict(sorted(reasons.items(), key=lambda x: x[1], reverse=True))
    
    def _calculate_average_performance_impact(self) -> Optional[float]:
        """Calculate average performance impact of fallbacks"""
        impacts = [h.performance_impact for h in self.fallback_history if h.performance_impact is not None]
        return sum(impacts) / len(impacts) if impacts else None
    
    def _load_fallback_rules(self) -> Dict[str, FallbackRule]:
        """Load fallback rules from configuration"""
        try:
            if self.fallback_rules_path.exists():
                with open(self.fallback_rules_path, 'r') as f:
                    data = json.load(f)
                    rules = {}
                    for rule_data in data.get('rules', []):
                        rule = FallbackRule(
                            trigger_condition=rule_data['trigger_condition'],
                            from_method=QuantizationMethod(rule_data['from_method']),
                            to_method=QuantizationMethod(rule_data['to_method']),
                            max_attempts=rule_data['max_attempts'],
                            cooldown_seconds=rule_data['cooldown_seconds'],
                            enabled=rule_data.get('enabled', True)
                        )
                        key = f"{rule.from_method.value}_{rule.trigger_condition}"
                        rules[key] = rule
                    return rules
        except Exception as e:
            self.logger.error(f"Failed to load fallback rules: {e}")
        
        # Default fallback rules
        return self._create_default_fallback_rules()
    
    def _create_default_fallback_rules(self) -> Dict[str, FallbackRule]:
        """Create default fallback rules"""
        default_rules = [
            FallbackRule("timeout", QuantizationMethod.INT8, QuantizationMethod.BF16, 2, 300),
            FallbackRule("timeout", QuantizationMethod.BF16, QuantizationMethod.FP16, 2, 300),
            FallbackRule("timeout", QuantizationMethod.FP16, QuantizationMethod.NONE, 1, 600),
            FallbackRule("memory_exceeded", QuantizationMethod.NONE, QuantizationMethod.BF16, 1, 60),
            FallbackRule("memory_exceeded", QuantizationMethod.FP16, QuantizationMethod.BF16, 1, 60),
            FallbackRule("memory_exceeded", QuantizationMethod.BF16, QuantizationMethod.INT8, 1, 60),
            FallbackRule("error", QuantizationMethod.INT8, QuantizationMethod.BF16, 2, 180),
            FallbackRule("error", QuantizationMethod.BF16, QuantizationMethod.FP16, 2, 180),
            FallbackRule("quality_low", QuantizationMethod.INT8, QuantizationMethod.BF16, 1, 300),
        ]
        
        rules_dict = {}
        for rule in default_rules:
            key = f"{rule.from_method.value}_{rule.trigger_condition}"
            rules_dict[key] = rule
        
        # Save default rules
        self._save_fallback_rules()
        
        return rules_dict
    
    def _save_fallback_rules(self) -> None:
        """Save fallback rules to configuration"""
        try:
            rules_data = {
                'rules': [
                    {
                        'trigger_condition': rule.trigger_condition,
                        'from_method': rule.from_method.value,
                        'to_method': rule.to_method.value,
                        'max_attempts': rule.max_attempts,
                        'cooldown_seconds': rule.cooldown_seconds,
                        'enabled': rule.enabled
                    }
                    for rule in self.fallback_rules.values()
                ]
            }
            
            with open(self.fallback_rules_path, 'w') as f:
                json.dump(rules_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save fallback rules: {e}")
    
    def _load_fallback_history(self) -> List[FallbackHistory]:
        """Load fallback history"""
        try:
            if self.history_path.exists():
                with open(self.history_path, 'r') as f:
                    data = json.load(f)
                    history = []
                    for item in data.get('history', []):
                        history.append(FallbackHistory(
                            timestamp=datetime.fromisoformat(item['timestamp']),
                            original_method=QuantizationMethod(item['original_method']),
                            fallback_method=QuantizationMethod(item['fallback_method']),
                            reason=item['reason'],
                            model_name=item['model_name'],
                            success=item['success'],
                            performance_impact=item.get('performance_impact')
                        ))
                    return history
        except Exception as e:
            self.logger.error(f"Failed to load fallback history: {e}")
        
        return []
    
    def _save_fallback_history(self) -> None:
        """Save fallback history"""
        try:
            # Keep only last 1000 entries
            recent_history = self.fallback_history[-1000:]
            
            history_data = {
                'history': [
                    {
                        'timestamp': h.timestamp.isoformat(),
                        'original_method': h.original_method.value,
                        'fallback_method': h.fallback_method.value,
                        'reason': h.reason,
                        'model_name': h.model_name,
                        'success': h.success,
                        'performance_impact': h.performance_impact
                    }
                    for h in recent_history
                ]
            }
            
            with open(self.history_path, 'w') as f:
                json.dump(history_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save fallback history: {e}")
    
    def _load_preference_profiles(self) -> Dict[str, PreferenceProfile]:
        """Load preference profiles"""
        profiles_path = Path("quantization_profiles.json")
        try:
            if profiles_path.exists():
                with open(profiles_path, 'r') as f:
                    data = json.load(f)
                    profiles = {}
                    for name, profile_data in data.get('profiles', {}).items():
                        prefs_data = profile_data['preferences']
                        preferences = UserPreferences(
                            preferred_method=QuantizationMethod(prefs_data['preferred_method']),
                            auto_fallback_enabled=prefs_data['auto_fallback_enabled'],
                            timeout_seconds=prefs_data['timeout_seconds'],
                            skip_quality_check=prefs_data['skip_quality_check'],
                            remember_model_settings=prefs_data['remember_model_settings'],
                            model_specific_preferences={
                                k: QuantizationMethod(v) for k, v in prefs_data['model_specific_preferences'].items()
                            }
                        )
                        
                        profile = PreferenceProfile(
                            name=name,
                            description=profile_data['description'],
                            preferences=preferences,
                            hardware_requirements=profile_data['hardware_requirements'],
                            use_cases=profile_data['use_cases'],
                            created_at=datetime.fromisoformat(profile_data['created_at']),
                            last_used=datetime.fromisoformat(profile_data['last_used'])
                        )
                        profiles[name] = profile
                    return profiles
        except Exception as e:
            self.logger.error(f"Failed to load preference profiles: {e}")
        
        return {}
    
    def _save_preference_profiles(self) -> None:
        """Save preference profiles"""
        profiles_path = Path("quantization_profiles.json")
        try:
            profiles_data = {
                'profiles': {}
            }
            
            for name, profile in self.preference_profiles.items():
                profiles_data['profiles'][name] = {
                    'description': profile.description,
                    'preferences': {
                        'preferred_method': profile.preferences.preferred_method.value,
                        'auto_fallback_enabled': profile.preferences.auto_fallback_enabled,
                        'timeout_seconds': profile.preferences.timeout_seconds,
                        'skip_quality_check': profile.preferences.skip_quality_check,
                        'remember_model_settings': profile.preferences.remember_model_settings,
                        'model_specific_preferences': {
                            k: v.value for k, v in profile.preferences.model_specific_preferences.items()
                        }
                    },
                    'hardware_requirements': profile.hardware_requirements,
                    'use_cases': profile.use_cases,
                    'created_at': profile.created_at.isoformat(),
                    'last_used': profile.last_used.isoformat()
                }
            
            with open(profiles_path, 'w') as f:
                json.dump(profiles_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save preference profiles: {e}")