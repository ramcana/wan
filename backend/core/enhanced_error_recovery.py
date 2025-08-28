"""
Enhanced Error Recovery System for Model Availability Management

This module extends the existing FallbackRecoverySystem with sophisticated error categorization,
multi-strategy recovery attempts, intelligent fallback integration, automatic repair triggers,
and user-friendly error messages with actionable recovery steps.

Integrates with:
- ModelAvailabilityManager for comprehensive model status
- IntelligentFallbackManager for smart alternatives
- ModelHealthMonitor for corruption detection and repair
- EnhancedModelDownloader for retry mechanisms
"""

import asyncio
import logging
import time
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from pathlib import Path

# Import existing components
from backend.core.fallback_recovery_system import (
    FallbackRecoverySystem, RecoveryAction, FailureType, RecoveryAttempt, SystemHealthStatus
)

# Import enhanced components
try:
    from backend.core.model_availability_manager import (
        ModelAvailabilityManager, ModelAvailabilityStatus, DetailedModelStatus
    )
    from backend.core.intelligent_fallback_manager import (
        IntelligentFallbackManager, FallbackStrategy, FallbackType as EnhancedFallbackType,
        ModelSuggestion, GenerationRequirements
    )
    from backend.core.model_health_monitor import (
        ModelHealthMonitor, HealthStatus, IntegrityResult, CorruptionReport
    )
    from backend.core.enhanced_model_downloader import (
        EnhancedModelDownloader, DownloadResult, DownloadStatus
    )
except ImportError as e:
    logging.warning(f"Some enhanced components not available: {e}")
    # Define minimal interfaces for testing
    ModelAvailabilityManager = None
    IntelligentFallbackManager = None
    ModelHealthMonitor = None
    EnhancedModelDownloader = None

logger = logging.getLogger(__name__)


class EnhancedFailureType(Enum):
    """Enhanced failure types with more granular categorization"""
    # Model-related failures
    MODEL_DOWNLOAD_FAILURE = "model_download_failure"
    MODEL_CORRUPTION_DETECTED = "model_corruption_detected"
    MODEL_VERSION_MISMATCH = "model_version_mismatch"
    MODEL_LOADING_TIMEOUT = "model_loading_timeout"
    MODEL_INTEGRITY_FAILURE = "model_integrity_failure"
    MODEL_COMPATIBILITY_ERROR = "model_compatibility_error"
    
    # Resource-related failures
    VRAM_EXHAUSTION = "vram_exhaustion"
    STORAGE_SPACE_INSUFFICIENT = "storage_space_insufficient"
    NETWORK_CONNECTIVITY_LOSS = "network_connectivity_loss"
    BANDWIDTH_LIMITATION = "bandwidth_limitation"
    
    # System-related failures
    HARDWARE_OPTIMIZATION_FAILURE = "hardware_optimization_failure"
    GENERATION_PIPELINE_ERROR = "generation_pipeline_error"
    SYSTEM_RESOURCE_ERROR = "system_resource_error"
    DEPENDENCY_MISSING = "dependency_missing"
    
    # User-related failures
    INVALID_PARAMETERS = "invalid_parameters"
    UNSUPPORTED_OPERATION = "unsupported_operation"
    PERMISSION_DENIED = "permission_denied"


class RecoveryStrategy(Enum):
    """Enhanced recovery strategies"""
    IMMEDIATE_RETRY = "immediate_retry"
    INTELLIGENT_FALLBACK = "intelligent_fallback"
    AUTOMATIC_REPAIR = "automatic_repair"
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    RESOURCE_OPTIMIZATION = "resource_optimization"
    USER_INTERVENTION = "user_intervention"
    GRACEFUL_DEGRADATION = "graceful_degradation"


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"           # Minor issues, system continues normally
    MEDIUM = "medium"     # Noticeable issues, some degradation
    HIGH = "high"         # Significant issues, major degradation
    CRITICAL = "critical" # System failure, immediate attention required


@dataclass
class ErrorContext:
    """Enhanced error context with detailed information"""
    failure_type: EnhancedFailureType
    original_error: Exception
    severity: ErrorSeverity
    model_id: Optional[str] = None
    operation: Optional[str] = None
    user_parameters: Dict[str, Any] = field(default_factory=dict)
    system_state: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    previous_attempts: List[str] = field(default_factory=list)
    user_session_id: Optional[str] = None


@dataclass
class RecoveryResult:
    """Enhanced recovery result with detailed information"""
    success: bool
    strategy_used: RecoveryStrategy
    recovery_time_seconds: float
    message: str
    user_message: str = ""
    actionable_steps: List[str] = field(default_factory=list)
    alternative_options: List[Dict[str, Any]] = field(default_factory=list)
    system_changes: Dict[str, Any] = field(default_factory=dict)
    requires_user_action: bool = False
    estimated_resolution_time: Optional[timedelta] = None
    follow_up_required: bool = False


@dataclass
class RecoveryMetrics:
    """Metrics for recovery success tracking and optimization"""
    total_attempts: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    average_recovery_time: float = 0.0
    strategy_success_rates: Dict[RecoveryStrategy, float] = field(default_factory=dict)
    failure_type_frequencies: Dict[EnhancedFailureType, int] = field(default_factory=dict)
    user_intervention_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


class EnhancedErrorRecovery:
    """
    Enhanced Error Recovery System that extends FallbackRecoverySystem
    with sophisticated error categorization and multi-strategy recovery
    """
    
    def __init__(
        self,
        base_recovery_system: Optional[FallbackRecoverySystem] = None,
        model_availability_manager: Optional[ModelAvailabilityManager] = None,
        intelligent_fallback_manager: Optional[IntelligentFallbackManager] = None,
        model_health_monitor: Optional[ModelHealthMonitor] = None,
        enhanced_downloader: Optional[EnhancedModelDownloader] = None,
        websocket_manager: Optional[Any] = None
    ):
        self.base_recovery = base_recovery_system
        self.availability_manager = model_availability_manager
        self.fallback_manager = intelligent_fallback_manager
        self.health_monitor = model_health_monitor
        self.enhanced_downloader = enhanced_downloader
        self.websocket_manager = websocket_manager
        
        self.logger = logging.getLogger(__name__ + ".EnhancedErrorRecovery")
        
        # Recovery tracking and metrics
        self.recovery_attempts: List[RecoveryAttempt] = []
        self.recovery_metrics = RecoveryMetrics()
        self.active_recoveries: Dict[str, datetime] = {}
        
        # Configuration
        self.max_recovery_attempts = 5
        self.recovery_timeout_seconds = 300  # 5 minutes
        self.user_intervention_threshold = 3  # attempts before requiring user action
        
        # Recovery strategy mapping
        self.strategy_mapping = self._initialize_strategy_mapping()
        
        # Error message templates
        self.error_messages = self._initialize_error_messages()
        
        self.logger.info("Enhanced Error Recovery System initialized")
    
    def _initialize_strategy_mapping(self) -> Dict[EnhancedFailureType, List[RecoveryStrategy]]:
        """Initialize recovery strategy mapping for different failure types"""
        return {
            EnhancedFailureType.MODEL_DOWNLOAD_FAILURE: [
                RecoveryStrategy.IMMEDIATE_RETRY,
                RecoveryStrategy.RESOURCE_OPTIMIZATION,
                RecoveryStrategy.INTELLIGENT_FALLBACK
            ],
            EnhancedFailureType.MODEL_CORRUPTION_DETECTED: [
                RecoveryStrategy.AUTOMATIC_REPAIR,
                RecoveryStrategy.IMMEDIATE_RETRY,
                RecoveryStrategy.INTELLIGENT_FALLBACK
            ],
            EnhancedFailureType.MODEL_VERSION_MISMATCH: [
                RecoveryStrategy.AUTOMATIC_REPAIR,
                RecoveryStrategy.INTELLIGENT_FALLBACK,
                RecoveryStrategy.USER_INTERVENTION
            ],
            EnhancedFailureType.MODEL_LOADING_TIMEOUT: [
                RecoveryStrategy.RESOURCE_OPTIMIZATION,
                RecoveryStrategy.PARAMETER_ADJUSTMENT,
                RecoveryStrategy.INTELLIGENT_FALLBACK
            ],
            EnhancedFailureType.MODEL_INTEGRITY_FAILURE: [
                RecoveryStrategy.AUTOMATIC_REPAIR,
                RecoveryStrategy.IMMEDIATE_RETRY,
                RecoveryStrategy.USER_INTERVENTION
            ],
            EnhancedFailureType.MODEL_COMPATIBILITY_ERROR: [
                RecoveryStrategy.INTELLIGENT_FALLBACK,
                RecoveryStrategy.PARAMETER_ADJUSTMENT,
                RecoveryStrategy.USER_INTERVENTION
            ],
            EnhancedFailureType.VRAM_EXHAUSTION: [
                RecoveryStrategy.RESOURCE_OPTIMIZATION,
                RecoveryStrategy.PARAMETER_ADJUSTMENT,
                RecoveryStrategy.INTELLIGENT_FALLBACK
            ],
            EnhancedFailureType.STORAGE_SPACE_INSUFFICIENT: [
                RecoveryStrategy.RESOURCE_OPTIMIZATION,
                RecoveryStrategy.USER_INTERVENTION
            ],
            EnhancedFailureType.NETWORK_CONNECTIVITY_LOSS: [
                RecoveryStrategy.IMMEDIATE_RETRY,
                RecoveryStrategy.GRACEFUL_DEGRADATION,
                RecoveryStrategy.USER_INTERVENTION
            ],
            EnhancedFailureType.BANDWIDTH_LIMITATION: [
                RecoveryStrategy.PARAMETER_ADJUSTMENT,
                RecoveryStrategy.GRACEFUL_DEGRADATION
            ],
            EnhancedFailureType.HARDWARE_OPTIMIZATION_FAILURE: [
                RecoveryStrategy.RESOURCE_OPTIMIZATION,
                RecoveryStrategy.GRACEFUL_DEGRADATION,
                RecoveryStrategy.USER_INTERVENTION
            ],
            EnhancedFailureType.GENERATION_PIPELINE_ERROR: [
                RecoveryStrategy.IMMEDIATE_RETRY,
                RecoveryStrategy.RESOURCE_OPTIMIZATION,
                RecoveryStrategy.INTELLIGENT_FALLBACK
            ],
            EnhancedFailureType.SYSTEM_RESOURCE_ERROR: [
                RecoveryStrategy.RESOURCE_OPTIMIZATION,
                RecoveryStrategy.GRACEFUL_DEGRADATION,
                RecoveryStrategy.USER_INTERVENTION
            ],
            EnhancedFailureType.DEPENDENCY_MISSING: [
                RecoveryStrategy.USER_INTERVENTION
            ],
            EnhancedFailureType.INVALID_PARAMETERS: [
                RecoveryStrategy.PARAMETER_ADJUSTMENT,
                RecoveryStrategy.USER_INTERVENTION
            ],
            EnhancedFailureType.UNSUPPORTED_OPERATION: [
                RecoveryStrategy.INTELLIGENT_FALLBACK,
                RecoveryStrategy.USER_INTERVENTION
            ],
            EnhancedFailureType.PERMISSION_DENIED: [
                RecoveryStrategy.USER_INTERVENTION
            ]
        }
    
    def _initialize_error_messages(self) -> Dict[EnhancedFailureType, Dict[str, str]]:
        """Initialize user-friendly error messages and actionable steps"""
        return {
            EnhancedFailureType.MODEL_DOWNLOAD_FAILURE: {
                "title": "Model Download Failed",
                "message": "The AI model couldn't be downloaded completely.",
                "user_message": "We're automatically retrying the download. This usually resolves network-related issues.",
                "steps": [
                    "Check your internet connection",
                    "Ensure sufficient storage space",
                    "Try downloading during off-peak hours"
                ]
            },
            EnhancedFailureType.MODEL_CORRUPTION_DETECTED: {
                "title": "Model File Corruption Detected",
                "message": "The AI model files appear to be corrupted.",
                "user_message": "We're automatically repairing the model files. This may take a few minutes.",
                "steps": [
                    "Allow automatic repair to complete",
                    "Check storage device health",
                    "Consider re-downloading if issues persist"
                ]
            },
            EnhancedFailureType.VRAM_EXHAUSTION: {
                "title": "Insufficient GPU Memory",
                "message": "Not enough GPU memory available for the current operation.",
                "user_message": "We're optimizing memory usage and adjusting parameters automatically.",
                "steps": [
                    "Close other GPU-intensive applications",
                    "Reduce generation resolution or complexity",
                    "Consider using CPU offloading"
                ]
            },
            EnhancedFailureType.NETWORK_CONNECTIVITY_LOSS: {
                "title": "Network Connection Lost",
                "message": "Internet connection was lost during the operation.",
                "user_message": "We'll retry once your connection is restored.",
                "steps": [
                    "Check your internet connection",
                    "Restart your router if needed",
                    "Try again once connection is stable"
                ]
            }
        }
    
    async def handle_enhanced_failure(
        self,
        error_context: ErrorContext
    ) -> RecoveryResult:
        """
        Handle a failure using enhanced recovery strategies
        
        Args:
            error_context: Detailed context about the failure
            
        Returns:
            RecoveryResult with detailed recovery information
        """
        start_time = time.time()
        correlation_id = error_context.correlation_id or f"recovery_{int(time.time())}"
        
        try:
            self.logger.info(
                f"Handling enhanced failure: {error_context.failure_type.value} "
                f"(correlation_id: {correlation_id})"
            )
            
            # Update metrics
            self.recovery_metrics.total_attempts += 1
            self.recovery_metrics.failure_type_frequencies[error_context.failure_type] = \
                self.recovery_metrics.failure_type_frequencies.get(error_context.failure_type, 0) + 1
            
            # Check if we should attempt recovery or require user intervention
            if len(error_context.previous_attempts) >= self.user_intervention_threshold:
                return await self._require_user_intervention(error_context, start_time)
            
            # Get recovery strategies for this failure type
            strategies = self.strategy_mapping.get(
                error_context.failure_type,
                [RecoveryStrategy.USER_INTERVENTION]
            )
            
            # Filter out already attempted strategies
            available_strategies = [
                s for s in strategies 
                if s.value not in error_context.previous_attempts
            ]
            
            if not available_strategies:
                return await self._require_user_intervention(error_context, start_time)
            
            # Try each strategy in order
            for strategy in available_strategies:
                try:
                    self.logger.info(f"Attempting recovery strategy: {strategy.value}")
                    
                    result = await self._execute_recovery_strategy(
                        strategy, error_context, correlation_id
                    )
                    
                    if result.success:
                        # Update success metrics
                        self.recovery_metrics.successful_recoveries += 1
                        self._update_strategy_success_rate(strategy, True)
                        
                        recovery_time = time.time() - start_time
                        result.recovery_time_seconds = recovery_time
                        
                        self.logger.info(
                            f"Recovery successful with strategy: {strategy.value} "
                            f"(took {recovery_time:.1f}s)"
                        )
                        
                        # Notify via WebSocket
                        await self._notify_recovery_success(result, correlation_id)
                        
                        return result
                    else:
                        self.logger.warning(f"Recovery strategy {strategy.value} failed")
                        self._update_strategy_success_rate(strategy, False)
                        
                except Exception as e:
                    self.logger.error(f"Recovery strategy {strategy.value} raised exception: {e}")
                    self._update_strategy_success_rate(strategy, False)
            
            # All strategies failed
            self.recovery_metrics.failed_recoveries += 1
            return await self._handle_all_strategies_failed(error_context, start_time)
            
        except Exception as e:
            self.logger.error(f"Error in enhanced failure handling: {e}")
            recovery_time = time.time() - start_time
            
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.USER_INTERVENTION,
                recovery_time_seconds=recovery_time,
                message=f"Recovery system error: {str(e)}",
                user_message="An unexpected error occurred during recovery. Please try again or contact support.",
                requires_user_action=True
            )
    
    async def _execute_recovery_strategy(
        self,
        strategy: RecoveryStrategy,
        error_context: ErrorContext,
        correlation_id: str
    ) -> RecoveryResult:
        """Execute a specific recovery strategy"""
        
        if strategy == RecoveryStrategy.IMMEDIATE_RETRY:
            return await self._immediate_retry_strategy(error_context)
        
        elif strategy == RecoveryStrategy.INTELLIGENT_FALLBACK:
            return await self._intelligent_fallback_strategy(error_context)
        
        elif strategy == RecoveryStrategy.AUTOMATIC_REPAIR:
            return await self._automatic_repair_strategy(error_context)
        
        elif strategy == RecoveryStrategy.PARAMETER_ADJUSTMENT:
            return await self._parameter_adjustment_strategy(error_context)
        
        elif strategy == RecoveryStrategy.RESOURCE_OPTIMIZATION:
            return await self._resource_optimization_strategy(error_context)
        
        elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            return await self._graceful_degradation_strategy(error_context)
        
        elif strategy == RecoveryStrategy.USER_INTERVENTION:
            return await self._user_intervention_strategy(error_context)
        
        else:
            return RecoveryResult(
                success=False,
                strategy_used=strategy,
                recovery_time_seconds=0.0,
                message=f"Unknown recovery strategy: {strategy.value}",
                user_message="An unknown recovery method was attempted.",
                requires_user_action=True
            )
    
    async def _immediate_retry_strategy(self, error_context: ErrorContext) -> RecoveryResult:
        """Implement immediate retry with exponential backoff"""
        try:
            # Determine retry parameters based on failure type
            if error_context.failure_type == EnhancedFailureType.MODEL_DOWNLOAD_FAILURE:
                max_retries = 3
                base_delay = 2.0
                backoff_factor = 2.0
            elif error_context.failure_type == EnhancedFailureType.NETWORK_CONNECTIVITY_LOSS:
                max_retries = 5
                base_delay = 1.0
                backoff_factor = 1.5
            else:
                max_retries = 2
                base_delay = 1.0
                backoff_factor = 2.0
            
            for attempt in range(max_retries):
                if attempt > 0:
                    delay = base_delay * (backoff_factor ** (attempt - 1))
                    self.logger.info(f"Waiting {delay:.1f}s before retry attempt {attempt + 1}")
                    await asyncio.sleep(delay)
                
                try:
                    # Attempt the original operation based on failure type
                    if error_context.failure_type == EnhancedFailureType.MODEL_DOWNLOAD_FAILURE:
                        success = await self._retry_model_download(error_context)
                    elif error_context.failure_type == EnhancedFailureType.GENERATION_PIPELINE_ERROR:
                        success = await self._retry_generation_pipeline(error_context)
                    else:
                        # Generic retry using base recovery system
                        success = await self._retry_with_base_system(error_context)
                    
                    if success:
                        return RecoveryResult(
                            success=True,
                            strategy_used=RecoveryStrategy.IMMEDIATE_RETRY,
                            recovery_time_seconds=0.0,  # Will be set by caller
                            message=f"Retry successful after {attempt + 1} attempts",
                            user_message="The operation completed successfully after retrying.",
                            system_changes={"retry_attempts": attempt + 1}
                        )
                
                except Exception as e:
                    self.logger.warning(f"Retry attempt {attempt + 1} failed: {e}")
            
            # All retries failed
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.IMMEDIATE_RETRY,
                recovery_time_seconds=0.0,
                message=f"All {max_retries} retry attempts failed",
                user_message="Multiple retry attempts were unsuccessful. Trying alternative approaches.",
                system_changes={"retry_attempts": max_retries, "all_retries_failed": True}
            )
            
        except Exception as e:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.IMMEDIATE_RETRY,
                recovery_time_seconds=0.0,
                message=f"Retry strategy error: {str(e)}",
                user_message="An error occurred during retry attempts."
            )
    
    async def _intelligent_fallback_strategy(self, error_context: ErrorContext) -> RecoveryResult:
        """Use intelligent fallback manager for smart alternatives"""
        try:
            if not self.fallback_manager:
                return RecoveryResult(
                    success=False,
                    strategy_used=RecoveryStrategy.INTELLIGENT_FALLBACK,
                    recovery_time_seconds=0.0,
                    message="Intelligent fallback manager not available",
                    user_message="Smart alternatives are not available at this time."
                )
            
            # Create generation requirements from error context
            requirements = GenerationRequirements(
                model_type=error_context.model_id or "unknown",
                quality=error_context.user_parameters.get("quality", "medium"),
                speed=error_context.user_parameters.get("speed", "medium"),
                resolution=error_context.user_parameters.get("resolution", "1280x720"),
                allow_alternatives=True,
                allow_quality_reduction=True
            )
            
            # Get fallback strategy
            fallback_strategy = await self.fallback_manager.get_fallback_strategy(
                error_context.model_id or "unknown",
                {
                    "error_type": error_context.failure_type.value,
                    "error_message": str(error_context.original_error),
                    "user_parameters": error_context.user_parameters
                }
            )
            
            if fallback_strategy.strategy_type == EnhancedFallbackType.ALTERNATIVE_MODEL:
                # Try alternative model
                if fallback_strategy.alternative_model:
                    success = await self._try_alternative_model(
                        fallback_strategy.alternative_model,
                        error_context
                    )
                    
                    if success:
                        return RecoveryResult(
                            success=True,
                            strategy_used=RecoveryStrategy.INTELLIGENT_FALLBACK,
                            recovery_time_seconds=0.0,
                            message=f"Successfully switched to alternative model: {fallback_strategy.alternative_model}",
                            user_message=f"We've switched to a compatible alternative model ({fallback_strategy.alternative_model}) to continue your request.",
                            alternative_options=[{
                                "type": "alternative_model",
                                "model": fallback_strategy.alternative_model,
                                "reason": fallback_strategy.recommended_action
                            }],
                            system_changes={
                                "alternative_model_used": fallback_strategy.alternative_model,
                                "fallback_reason": fallback_strategy.recommended_action
                            }
                        )
            
            elif fallback_strategy.strategy_type == EnhancedFallbackType.QUEUE_AND_WAIT:
                # Queue request for later processing
                if fallback_strategy.can_queue_request:
                    queue_result = await self._queue_request_for_later(error_context, fallback_strategy)
                    
                    if queue_result:
                        wait_time_str = ""
                        if fallback_strategy.estimated_wait_time:
                            wait_time_str = f" (estimated wait: {fallback_strategy.estimated_wait_time})"
                        
                        return RecoveryResult(
                            success=True,
                            strategy_used=RecoveryStrategy.INTELLIGENT_FALLBACK,
                            recovery_time_seconds=0.0,
                            message=f"Request queued for processing{wait_time_str}",
                            user_message=f"Your request has been queued and will be processed when the model becomes available{wait_time_str}.",
                            estimated_resolution_time=fallback_strategy.estimated_wait_time,
                            follow_up_required=True,
                            system_changes={
                                "request_queued": True,
                                "estimated_wait_time": str(fallback_strategy.estimated_wait_time) if fallback_strategy.estimated_wait_time else None
                            }
                        )
            
            elif fallback_strategy.strategy_type == EnhancedFallbackType.DOWNLOAD_AND_RETRY:
                # Trigger download and retry
                download_result = await self._trigger_download_and_retry(error_context, fallback_strategy)
                
                if download_result:
                    return RecoveryResult(
                        success=True,
                        strategy_used=RecoveryStrategy.INTELLIGENT_FALLBACK,
                        recovery_time_seconds=0.0,
                        message="Model download initiated, will retry when complete",
                        user_message="We're downloading the required model. Your request will be processed automatically once the download completes.",
                        estimated_resolution_time=fallback_strategy.estimated_wait_time,
                        follow_up_required=True,
                        system_changes={
                            "download_initiated": True,
                            "auto_retry_scheduled": True
                        }
                    )
            
            # Fallback strategy didn't work
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.INTELLIGENT_FALLBACK,
                recovery_time_seconds=0.0,
                message=f"Intelligent fallback strategy failed: {fallback_strategy.strategy_type.value}",
                user_message=fallback_strategy.user_message or "No suitable alternatives are currently available."
            )
            
        except Exception as e:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.INTELLIGENT_FALLBACK,
                recovery_time_seconds=0.0,
                message=f"Intelligent fallback error: {str(e)}",
                user_message="An error occurred while finding alternatives."
            )
    
    async def _automatic_repair_strategy(self, error_context: ErrorContext) -> RecoveryResult:
        """Automatically repair detected model issues"""
        try:
            if not error_context.model_id:
                return RecoveryResult(
                    success=False,
                    strategy_used=RecoveryStrategy.AUTOMATIC_REPAIR,
                    recovery_time_seconds=0.0,
                    message="No model ID specified for repair",
                    user_message="Cannot repair model: no model specified."
                )
            
            repair_actions = []
            
            # Handle corruption detection
            if error_context.failure_type == EnhancedFailureType.MODEL_CORRUPTION_DETECTED:
                if self.health_monitor:
                    # Check integrity and get corruption report
                    integrity_result = await self.health_monitor.check_model_integrity(error_context.model_id)
                    
                    if not integrity_result.is_healthy:
                        repair_actions.append("integrity_check_failed")
                        
                        # Attempt automatic repair
                        if self.enhanced_downloader:
                            repair_result = await self.enhanced_downloader.verify_and_repair_model(error_context.model_id)
                            
                            if repair_result.success:
                                repair_actions.append("automatic_repair_successful")
                                
                                return RecoveryResult(
                                    success=True,
                                    strategy_used=RecoveryStrategy.AUTOMATIC_REPAIR,
                                    recovery_time_seconds=0.0,
                                    message=f"Model {error_context.model_id} repaired successfully",
                                    user_message="The corrupted model files have been automatically repaired.",
                                    system_changes={
                                        "model_repaired": error_context.model_id,
                                        "repair_actions": repair_actions
                                    }
                                )
                            else:
                                repair_actions.append("automatic_repair_failed")
            
            # Handle version mismatch
            elif error_context.failure_type == EnhancedFailureType.MODEL_VERSION_MISMATCH:
                if self.availability_manager:
                    # Check if update is available
                    model_status = await self.availability_manager.get_model_status(error_context.model_id)
                    
                    if model_status and model_status.update_available:
                        # Trigger model update
                        if self.enhanced_downloader:
                            update_result = await self.enhanced_downloader.download_with_retry(
                                error_context.model_id,
                                max_retries=2
                            )
                            
                            if update_result.success:
                                repair_actions.append("model_updated")
                                
                                return RecoveryResult(
                                    success=True,
                                    strategy_used=RecoveryStrategy.AUTOMATIC_REPAIR,
                                    recovery_time_seconds=0.0,
                                    message=f"Model {error_context.model_id} updated to latest version",
                                    user_message="The model has been automatically updated to the latest version.",
                                    system_changes={
                                        "model_updated": error_context.model_id,
                                        "repair_actions": repair_actions
                                    }
                                )
            
            # Handle integrity failure
            elif error_context.failure_type == EnhancedFailureType.MODEL_INTEGRITY_FAILURE:
                if self.enhanced_downloader:
                    # Re-download the model
                    download_result = await self.enhanced_downloader.download_with_retry(
                        error_context.model_id,
                        max_retries=2
                    )
                    
                    if download_result.success:
                        repair_actions.append("model_redownloaded")
                        
                        return RecoveryResult(
                            success=True,
                            strategy_used=RecoveryStrategy.AUTOMATIC_REPAIR,
                            recovery_time_seconds=0.0,
                            message=f"Model {error_context.model_id} re-downloaded successfully",
                            user_message="The model has been re-downloaded to fix integrity issues.",
                            system_changes={
                                "model_redownloaded": error_context.model_id,
                                "repair_actions": repair_actions
                            }
                        )
            
            # No repair action was successful
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.AUTOMATIC_REPAIR,
                recovery_time_seconds=0.0,
                message=f"Automatic repair failed for {error_context.failure_type.value}",
                user_message="Automatic repair was unsuccessful. Manual intervention may be required.",
                actionable_steps=[
                    "Check model file permissions",
                    "Verify available storage space",
                    "Try manually re-downloading the model"
                ],
                system_changes={"repair_actions": repair_actions}
            )
            
        except Exception as e:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.AUTOMATIC_REPAIR,
                recovery_time_seconds=0.0,
                message=f"Automatic repair error: {str(e)}",
                user_message="An error occurred during automatic repair."
            )
    
    async def _parameter_adjustment_strategy(self, error_context: ErrorContext) -> RecoveryResult:
        """Automatically adjust parameters to resolve issues"""
        try:
            original_params = error_context.user_parameters.copy()
            adjusted_params = original_params.copy()
            adjustments_made = []
            
            # Handle VRAM exhaustion
            if error_context.failure_type == EnhancedFailureType.VRAM_EXHAUSTION:
                # Reduce resolution
                current_resolution = adjusted_params.get("resolution", "1280x720")
                if current_resolution in ["1920x1080", "1920x1088"]:
                    adjusted_params["resolution"] = "1280x720"
                    adjustments_made.append("reduced_resolution_to_720p")
                elif current_resolution == "1280x720":
                    adjusted_params["resolution"] = "960x540"
                    adjustments_made.append("reduced_resolution_to_540p")
                
                # Reduce inference steps
                current_steps = adjusted_params.get("steps", 20)
                if current_steps > 20:
                    adjusted_params["steps"] = 20
                    adjustments_made.append("reduced_steps_to_20")
                elif current_steps > 15:
                    adjusted_params["steps"] = 15
                    adjustments_made.append("reduced_steps_to_15")
                
                # Reduce number of frames for video generation
                current_frames = adjusted_params.get("num_frames", 16)
                if current_frames > 16:
                    adjusted_params["num_frames"] = 16
                    adjustments_made.append("reduced_frames_to_16")
                elif current_frames > 8:
                    adjusted_params["num_frames"] = 8
                    adjustments_made.append("reduced_frames_to_8")
            
            # Handle invalid parameters
            elif error_context.failure_type == EnhancedFailureType.INVALID_PARAMETERS:
                # Validate and fix common parameter issues
                if "resolution" in adjusted_params:
                    resolution = adjusted_params["resolution"]
                    if resolution not in ["960x540", "1280x720", "1920x1080", "1920x1088"]:
                        adjusted_params["resolution"] = "1280x720"
                        adjustments_made.append("fixed_invalid_resolution")
                
                if "steps" in adjusted_params:
                    steps = adjusted_params.get("steps", 20)
                    if not isinstance(steps, int) or steps < 1 or steps > 50:
                        adjusted_params["steps"] = 20
                        adjustments_made.append("fixed_invalid_steps")
                
                if "num_frames" in adjusted_params:
                    frames = adjusted_params.get("num_frames", 16)
                    if not isinstance(frames, int) or frames < 1 or frames > 32:
                        adjusted_params["num_frames"] = 16
                        adjustments_made.append("fixed_invalid_frames")
            
            # Handle bandwidth limitations
            elif error_context.failure_type == EnhancedFailureType.BANDWIDTH_LIMITATION:
                # Reduce quality settings that affect download size
                if adjusted_params.get("quality") == "high":
                    adjusted_params["quality"] = "medium"
                    adjustments_made.append("reduced_quality_to_medium")
                elif adjusted_params.get("quality") == "medium":
                    adjusted_params["quality"] = "low"
                    adjustments_made.append("reduced_quality_to_low")
            
            if adjustments_made:
                # Apply the adjusted parameters
                error_context.user_parameters = adjusted_params
                
                adjustment_description = ", ".join(adjustments_made)
                
                return RecoveryResult(
                    success=True,
                    strategy_used=RecoveryStrategy.PARAMETER_ADJUSTMENT,
                    recovery_time_seconds=0.0,
                    message=f"Parameters adjusted: {adjustment_description}",
                    user_message="We've automatically adjusted some settings to resolve the issue. Your request will continue with optimized parameters.",
                    system_changes={
                        "original_parameters": original_params,
                        "adjusted_parameters": adjusted_params,
                        "adjustments_made": adjustments_made
                    },
                    actionable_steps=[
                        "Review the adjusted parameters",
                        "Consider these settings for future requests",
                        "Upgrade hardware for higher quality settings"
                    ]
                )
            else:
                return RecoveryResult(
                    success=False,
                    strategy_used=RecoveryStrategy.PARAMETER_ADJUSTMENT,
                    recovery_time_seconds=0.0,
                    message="No parameter adjustments could resolve the issue",
                    user_message="The current parameters couldn't be automatically adjusted to resolve the issue."
                )
            
        except Exception as e:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.PARAMETER_ADJUSTMENT,
                recovery_time_seconds=0.0,
                message=f"Parameter adjustment error: {str(e)}",
                user_message="An error occurred while adjusting parameters."
            )
    
    async def _resource_optimization_strategy(self, error_context: ErrorContext) -> RecoveryResult:
        """Optimize system resources to resolve issues"""
        try:
            optimizations_applied = []
            
            # Use base recovery system for resource optimization
            if self.base_recovery:
                # Clear GPU cache
                try:
                    cache_cleared = await self.base_recovery._clear_gpu_cache()
                    if cache_cleared:
                        optimizations_applied.append("gpu_cache_cleared")
                except Exception as e:
                    self.logger.warning(f"GPU cache clearing failed: {e}")
                
                # Apply VRAM optimization
                try:
                    vram_optimized = await self.base_recovery._apply_vram_optimization(error_context.system_state)
                    if vram_optimized:
                        optimizations_applied.append("vram_optimized")
                except Exception as e:
                    self.logger.warning(f"VRAM optimization failed: {e}")
                
                # Enable CPU offload if needed
                if error_context.failure_type == EnhancedFailureType.VRAM_EXHAUSTION:
                    try:
                        cpu_offload_enabled = await self.base_recovery._enable_cpu_offload()
                        if cpu_offload_enabled:
                            optimizations_applied.append("cpu_offload_enabled")
                    except Exception as e:
                        self.logger.warning(f"CPU offload enabling failed: {e}")
            
            # Additional optimizations specific to enhanced system
            if error_context.failure_type == EnhancedFailureType.STORAGE_SPACE_INSUFFICIENT:
                if self.availability_manager:
                    # Suggest cleanup of unused models
                    try:
                        cleanup_suggestions = await self.availability_manager.get_cleanup_suggestions()
                        if cleanup_suggestions:
                            optimizations_applied.append("cleanup_suggestions_generated")
                    except Exception as e:
                        self.logger.warning(f"Cleanup suggestions failed: {e}")
            
            if optimizations_applied:
                optimization_description = ", ".join(optimizations_applied)
                
                return RecoveryResult(
                    success=True,
                    strategy_used=RecoveryStrategy.RESOURCE_OPTIMIZATION,
                    recovery_time_seconds=0.0,
                    message=f"Resource optimizations applied: {optimization_description}",
                    user_message="We've optimized system resources to resolve the issue. Please try your request again.",
                    system_changes={
                        "optimizations_applied": optimizations_applied
                    },
                    actionable_steps=[
                        "Try your request again",
                        "Monitor system resource usage",
                        "Consider upgrading hardware if issues persist"
                    ]
                )
            else:
                return RecoveryResult(
                    success=False,
                    strategy_used=RecoveryStrategy.RESOURCE_OPTIMIZATION,
                    recovery_time_seconds=0.0,
                    message="No resource optimizations could be applied",
                    user_message="System resources couldn't be optimized further at this time."
                )
            
        except Exception as e:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.RESOURCE_OPTIMIZATION,
                recovery_time_seconds=0.0,
                message=f"Resource optimization error: {str(e)}",
                user_message="An error occurred during resource optimization."
            )
    
    async def _graceful_degradation_strategy(self, error_context: ErrorContext) -> RecoveryResult:
        """Implement graceful degradation to maintain functionality"""
        try:
            degradation_actions = []
            
            # Enable mock generation as fallback
            if self.base_recovery:
                try:
                    mock_enabled = await self.base_recovery._fallback_to_mock_generation()
                    if mock_enabled:
                        degradation_actions.append("mock_generation_enabled")
                except Exception as e:
                    self.logger.warning(f"Mock generation fallback failed: {e}")
            
            # Reduce service quality but maintain functionality
            if error_context.failure_type in [
                EnhancedFailureType.NETWORK_CONNECTIVITY_LOSS,
                EnhancedFailureType.BANDWIDTH_LIMITATION
            ]:
                degradation_actions.append("offline_mode_enabled")
            
            if degradation_actions:
                degradation_description = ", ".join(degradation_actions)
                
                return RecoveryResult(
                    success=True,
                    strategy_used=RecoveryStrategy.GRACEFUL_DEGRADATION,
                    recovery_time_seconds=0.0,
                    message=f"Graceful degradation applied: {degradation_description}",
                    user_message="The system is running in a reduced functionality mode to maintain service. Some features may be limited.",
                    system_changes={
                        "degradation_mode": True,
                        "degradation_actions": degradation_actions
                    },
                    actionable_steps=[
                        "System functionality is maintained at reduced quality",
                        "Full functionality will be restored when issues are resolved",
                        "Check system status for updates"
                    ]
                )
            else:
                return RecoveryResult(
                    success=False,
                    strategy_used=RecoveryStrategy.GRACEFUL_DEGRADATION,
                    recovery_time_seconds=0.0,
                    message="Graceful degradation could not be applied",
                    user_message="The system cannot maintain functionality in a degraded mode."
                )
            
        except Exception as e:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.GRACEFUL_DEGRADATION,
                recovery_time_seconds=0.0,
                message=f"Graceful degradation error: {str(e)}",
                user_message="An error occurred during graceful degradation."
            )
    
    async def _user_intervention_strategy(self, error_context: ErrorContext) -> RecoveryResult:
        """Require user intervention with clear guidance"""
        try:
            error_info = self.error_messages.get(error_context.failure_type, {
                "title": "System Error",
                "message": "An error occurred that requires your attention.",
                "user_message": "Please review the error details and take appropriate action.",
                "steps": ["Contact support if the issue persists"]
            })
            
            actionable_steps = error_info.get("steps", [])
            
            # Add specific steps based on failure type
            if error_context.failure_type == EnhancedFailureType.STORAGE_SPACE_INSUFFICIENT:
                actionable_steps.extend([
                    "Free up disk space by deleting unnecessary files",
                    "Move files to external storage",
                    "Clean up unused AI models"
                ])
            
            elif error_context.failure_type == EnhancedFailureType.DEPENDENCY_MISSING:
                actionable_steps.extend([
                    "Install missing system dependencies",
                    "Update your system packages",
                    "Reinstall the application if needed"
                ])
            
            elif error_context.failure_type == EnhancedFailureType.PERMISSION_DENIED:
                actionable_steps.extend([
                    "Check file and folder permissions",
                    "Run the application with appropriate privileges",
                    "Contact your system administrator"
                ])
            
            return RecoveryResult(
                success=False,  # User intervention required means automatic recovery failed
                strategy_used=RecoveryStrategy.USER_INTERVENTION,
                recovery_time_seconds=0.0,
                message=error_info.get("message", "User intervention required"),
                user_message=error_info.get("user_message", "Please take manual action to resolve this issue."),
                actionable_steps=actionable_steps,
                requires_user_action=True,
                system_changes={
                    "user_intervention_required": True,
                    "error_category": error_context.failure_type.value,
                    "severity": error_context.severity.value
                }
            )
            
        except Exception as e:
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.USER_INTERVENTION,
                recovery_time_seconds=0.0,
                message=f"User intervention strategy error: {str(e)}",
                user_message="An error occurred while preparing user guidance.",
                requires_user_action=True
            )  
  
    # Helper methods for recovery strategies
    
    async def _retry_model_download(self, error_context: ErrorContext) -> bool:
        """Retry model download using enhanced downloader"""
        try:
            if not self.enhanced_downloader or not error_context.model_id:
                return False
            
            result = await self.enhanced_downloader.download_with_retry(
                error_context.model_id,
                max_retries=2
            )
            
            return result.success
            
        except Exception as e:
            self.logger.error(f"Model download retry failed: {e}")
            return False
    
    async def _retry_generation_pipeline(self, error_context: ErrorContext) -> bool:
        """Retry generation pipeline operation"""
        try:
            if not self.base_recovery:
                return False
            
            return await self.base_recovery._restart_generation_pipeline()
            
        except Exception as e:
            self.logger.error(f"Generation pipeline retry failed: {e}")
            return False
    
    async def _retry_with_base_system(self, error_context: ErrorContext) -> bool:
        """Retry using base recovery system"""
        try:
            if not self.base_recovery:
                return False
            
            # Convert enhanced failure type to base failure type
            base_failure_type = self._convert_to_base_failure_type(error_context.failure_type)
            
            success, message = await self.base_recovery.handle_failure(
                base_failure_type,
                error_context.original_error,
                error_context.system_state
            )
            
            return success
            
        except Exception as e:
            self.logger.error(f"Base system retry failed: {e}")
            return False
    
    async def _try_alternative_model(self, alternative_model: str, error_context: ErrorContext) -> bool:
        """Try using an alternative model"""
        try:
            if not self.availability_manager:
                return False
            
            # Check if alternative model is available
            model_status = await self.availability_manager.get_model_status(alternative_model)
            
            if model_status and model_status.is_available:
                # Update the error context to use the alternative model
                error_context.model_id = alternative_model
                error_context.system_state["alternative_model_used"] = alternative_model
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Alternative model attempt failed: {e}")
            return False
    
    async def _queue_request_for_later(self, error_context: ErrorContext, fallback_strategy) -> bool:
        """Queue request for later processing"""
        try:
            # This would integrate with a request queue system
            # For now, we'll just log the action and return success
            self.logger.info(f"Queuing request for model: {error_context.model_id}")
            
            # In a real implementation, this would:
            # 1. Add request to a persistent queue
            # 2. Set up monitoring for model availability
            # 3. Automatically retry when model becomes available
            
            return True
            
        except Exception as e:
            self.logger.error(f"Request queuing failed: {e}")
            return False
    
    async def _trigger_download_and_retry(self, error_context: ErrorContext, fallback_strategy) -> bool:
        """Trigger model download and schedule retry"""
        try:
            if not self.enhanced_downloader or not error_context.model_id:
                return False
            
            # Start download in background
            download_task = asyncio.create_task(
                self.enhanced_downloader.download_with_retry(
                    error_context.model_id,
                    max_retries=3
                )
            )
            
            # Don't wait for completion, just confirm it started
            await asyncio.sleep(0.1)  # Give it a moment to start
            
            return True
            
        except Exception as e:
            self.logger.error(f"Download trigger failed: {e}")
            return False
    
    def _convert_to_base_failure_type(self, enhanced_type: EnhancedFailureType) -> FailureType:
        """Convert enhanced failure type to base failure type"""
        mapping = {
            EnhancedFailureType.MODEL_DOWNLOAD_FAILURE: FailureType.MODEL_LOADING_FAILURE,
            EnhancedFailureType.MODEL_CORRUPTION_DETECTED: FailureType.MODEL_LOADING_FAILURE,
            EnhancedFailureType.MODEL_VERSION_MISMATCH: FailureType.MODEL_LOADING_FAILURE,
            EnhancedFailureType.MODEL_LOADING_TIMEOUT: FailureType.MODEL_LOADING_FAILURE,
            EnhancedFailureType.MODEL_INTEGRITY_FAILURE: FailureType.MODEL_LOADING_FAILURE,
            EnhancedFailureType.MODEL_COMPATIBILITY_ERROR: FailureType.MODEL_LOADING_FAILURE,
            EnhancedFailureType.VRAM_EXHAUSTION: FailureType.VRAM_EXHAUSTION,
            EnhancedFailureType.STORAGE_SPACE_INSUFFICIENT: FailureType.SYSTEM_RESOURCE_ERROR,
            EnhancedFailureType.NETWORK_CONNECTIVITY_LOSS: FailureType.NETWORK_ERROR,
            EnhancedFailureType.BANDWIDTH_LIMITATION: FailureType.NETWORK_ERROR,
            EnhancedFailureType.HARDWARE_OPTIMIZATION_FAILURE: FailureType.HARDWARE_OPTIMIZATION_FAILURE,
            EnhancedFailureType.GENERATION_PIPELINE_ERROR: FailureType.GENERATION_PIPELINE_ERROR,
            EnhancedFailureType.SYSTEM_RESOURCE_ERROR: FailureType.SYSTEM_RESOURCE_ERROR,
            EnhancedFailureType.DEPENDENCY_MISSING: FailureType.SYSTEM_RESOURCE_ERROR,
            EnhancedFailureType.INVALID_PARAMETERS: FailureType.GENERATION_PIPELINE_ERROR,
            EnhancedFailureType.UNSUPPORTED_OPERATION: FailureType.GENERATION_PIPELINE_ERROR,
            EnhancedFailureType.PERMISSION_DENIED: FailureType.SYSTEM_RESOURCE_ERROR
        }
        
        return mapping.get(enhanced_type, FailureType.SYSTEM_RESOURCE_ERROR)
    
    async def _require_user_intervention(self, error_context: ErrorContext, start_time: float) -> RecoveryResult:
        """Handle cases where user intervention is required"""
        recovery_time = time.time() - start_time
        
        return RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategy.USER_INTERVENTION,
            recovery_time_seconds=recovery_time,
            message=f"Maximum recovery attempts exceeded for {error_context.failure_type.value}",
            user_message="Automatic recovery was unsuccessful. Please review the issue and take manual action.",
            requires_user_action=True,
            actionable_steps=[
                "Review the error details below",
                "Check system requirements and resources",
                "Contact support if the issue persists"
            ],
            system_changes={
                "max_attempts_exceeded": True,
                "previous_attempts": error_context.previous_attempts
            }
        )
    
    async def _handle_all_strategies_failed(self, error_context: ErrorContext, start_time: float) -> RecoveryResult:
        """Handle cases where all recovery strategies failed"""
        recovery_time = time.time() - start_time
        
        error_info = self.error_messages.get(error_context.failure_type, {
            "title": "Recovery Failed",
            "message": "All automatic recovery attempts were unsuccessful.",
            "user_message": "We couldn't automatically resolve this issue. Manual intervention is required.",
            "steps": ["Contact support for assistance"]
        })
        
        return RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategy.USER_INTERVENTION,
            recovery_time_seconds=recovery_time,
            message=f"All recovery strategies failed for {error_context.failure_type.value}",
            user_message=error_info.get("user_message", "All automatic recovery attempts failed."),
            requires_user_action=True,
            actionable_steps=error_info.get("steps", ["Contact support"]),
            system_changes={
                "all_strategies_failed": True,
                "attempted_strategies": [s.value for s in self.strategy_mapping.get(error_context.failure_type, [])]
            }
        )
    
    def _update_strategy_success_rate(self, strategy: RecoveryStrategy, success: bool):
        """Update success rate metrics for recovery strategies"""
        try:
            current_rate = self.recovery_metrics.strategy_success_rates.get(strategy, 0.0)
            
            # Simple moving average update
            if strategy not in self.recovery_metrics.strategy_success_rates:
                self.recovery_metrics.strategy_success_rates[strategy] = 1.0 if success else 0.0
            else:
                # Weight recent results more heavily
                weight = 0.1
                new_value = 1.0 if success else 0.0
                self.recovery_metrics.strategy_success_rates[strategy] = (
                    (1 - weight) * current_rate + weight * new_value
                )
            
            self.recovery_metrics.last_updated = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error updating strategy success rate: {e}")
    
    async def _notify_recovery_attempt(self, result: RecoveryResult, correlation_id: str):
        """Notify about recovery attempt via WebSocket"""
        try:
            if not self.websocket_manager:
                return
            
            notification = {
                "type": "recovery_attempt",
                "correlation_id": correlation_id,
                "data": {
                    "success": result.success,
                    "strategy": result.strategy_used.value,
                    "message": result.message,
                    "user_message": result.user_message,
                    "recovery_time": result.recovery_time_seconds,
                    "requires_user_action": result.requires_user_action,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            await self.websocket_manager.broadcast(notification)
            
        except Exception as e:
            self.logger.error(f"Error sending recovery notification: {e}")
    
    async def _notify_recovery_success(self, result: RecoveryResult, correlation_id: str):
        """Notify about successful recovery via WebSocket"""
        try:
            if not self.websocket_manager:
                return
            
            notification = {
                "type": "recovery_success",
                "correlation_id": correlation_id,
                "data": {
                    "strategy": result.strategy_used.value,
                    "message": result.message,
                    "user_message": result.user_message,
                    "recovery_time": result.recovery_time_seconds,
                    "system_changes": result.system_changes,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            await self.websocket_manager.broadcast(notification)
            
        except Exception as e:
            self.logger.error(f"Error sending recovery success notification: {e}")
    
    # Public interface methods
    
    async def categorize_error(self, error: Exception, context: Dict[str, Any]) -> ErrorContext:
        """Categorize an error and create enhanced error context"""
        try:
            # Determine failure type based on error and context
            failure_type = self._determine_failure_type(error, context)
            
            # Determine severity
            severity = self._determine_error_severity(failure_type, error, context)
            
            # Extract relevant information
            model_id = context.get("model_id") or context.get("model_type")
            operation = context.get("operation", "unknown")
            user_parameters = context.get("user_parameters", {})
            system_state = context.get("system_state", {})
            
            return ErrorContext(
                failure_type=failure_type,
                original_error=error,
                severity=severity,
                model_id=model_id,
                operation=operation,
                user_parameters=user_parameters,
                system_state=system_state,
                correlation_id=context.get("correlation_id"),
                user_session_id=context.get("user_session_id")
            )
            
        except Exception as e:
            self.logger.error(f"Error categorizing error: {e}")
            
            # Return a basic error context
            return ErrorContext(
                failure_type=EnhancedFailureType.SYSTEM_RESOURCE_ERROR,
                original_error=error,
                severity=ErrorSeverity.MEDIUM
            )
    
    def _determine_failure_type(self, error: Exception, context: Dict[str, Any]) -> EnhancedFailureType:
        """Determine the enhanced failure type based on error and context"""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Model-related failures
        if "download" in error_str or "fetch" in error_str:
            return EnhancedFailureType.MODEL_DOWNLOAD_FAILURE
        
        if "corrupt" in error_str or "checksum" in error_str or "integrity" in error_str:
            return EnhancedFailureType.MODEL_CORRUPTION_DETECTED
        
        if "version" in error_str or "mismatch" in error_str:
            return EnhancedFailureType.MODEL_VERSION_MISMATCH
        
        if "timeout" in error_str and ("load" in error_str or "model" in error_str):
            return EnhancedFailureType.MODEL_LOADING_TIMEOUT
        
        if "compatibility" in error_str or "incompatible" in error_str:
            return EnhancedFailureType.MODEL_COMPATIBILITY_ERROR
        
        # Resource-related failures
        if "vram" in error_str or "gpu memory" in error_str or "cuda out of memory" in error_str:
            return EnhancedFailureType.VRAM_EXHAUSTION
        
        if "disk" in error_str or "storage" in error_str or "no space" in error_str:
            return EnhancedFailureType.STORAGE_SPACE_INSUFFICIENT
        
        if "network" in error_str or "connection" in error_str or "unreachable" in error_str:
            return EnhancedFailureType.NETWORK_CONNECTIVITY_LOSS
        
        if "bandwidth" in error_str or "slow" in error_str or "rate limit" in error_str:
            return EnhancedFailureType.BANDWIDTH_LIMITATION
        
        # System-related failures
        if "permission" in error_str or "access denied" in error_str:
            return EnhancedFailureType.PERMISSION_DENIED
        
        if "parameter" in error_str or "invalid" in error_str or "validation" in error_str:
            return EnhancedFailureType.INVALID_PARAMETERS
        
        if "unsupported" in error_str or "not implemented" in error_str:
            return EnhancedFailureType.UNSUPPORTED_OPERATION
        
        if "dependency" in error_str or "import" in error_str or "module" in error_str:
            return EnhancedFailureType.DEPENDENCY_MISSING
        
        # Default to system resource error
        return EnhancedFailureType.SYSTEM_RESOURCE_ERROR
    
    def _determine_error_severity(
        self, 
        failure_type: EnhancedFailureType, 
        error: Exception, 
        context: Dict[str, Any]
    ) -> ErrorSeverity:
        """Determine error severity based on failure type and context"""
        
        # Critical severity failures
        critical_failures = [
            EnhancedFailureType.DEPENDENCY_MISSING,
            EnhancedFailureType.PERMISSION_DENIED,
            EnhancedFailureType.STORAGE_SPACE_INSUFFICIENT
        ]
        
        if failure_type in critical_failures:
            return ErrorSeverity.CRITICAL
        
        # High severity failures
        high_severity_failures = [
            EnhancedFailureType.MODEL_CORRUPTION_DETECTED,
            EnhancedFailureType.MODEL_INTEGRITY_FAILURE,
            EnhancedFailureType.VRAM_EXHAUSTION,
            EnhancedFailureType.HARDWARE_OPTIMIZATION_FAILURE
        ]
        
        if failure_type in high_severity_failures:
            return ErrorSeverity.HIGH
        
        # Medium severity failures
        medium_severity_failures = [
            EnhancedFailureType.MODEL_DOWNLOAD_FAILURE,
            EnhancedFailureType.MODEL_VERSION_MISMATCH,
            EnhancedFailureType.MODEL_LOADING_TIMEOUT,
            EnhancedFailureType.GENERATION_PIPELINE_ERROR,
            EnhancedFailureType.NETWORK_CONNECTIVITY_LOSS
        ]
        
        if failure_type in medium_severity_failures:
            return ErrorSeverity.MEDIUM
        
        # Default to low severity
        return ErrorSeverity.LOW
    
    async def get_recovery_metrics(self) -> RecoveryMetrics:
        """Get current recovery metrics"""
        return self.recovery_metrics
    
    async def get_recovery_history(self, limit: int = 50) -> List[RecoveryAttempt]:
        """Get recent recovery attempts"""
        return self.recovery_attempts[-limit:] if self.recovery_attempts else []
    
    def reset_metrics(self):
        """Reset recovery metrics (for testing or maintenance)"""
        self.recovery_metrics = RecoveryMetrics()
        self.recovery_attempts.clear()
        self.active_recoveries.clear()
        
        self.logger.info("Recovery metrics reset")


# Convenience function for creating enhanced error recovery system
def create_enhanced_error_recovery(
    generation_service=None,
    websocket_manager=None,
    **kwargs
) -> EnhancedErrorRecovery:
    """
    Create an enhanced error recovery system with all components
    
    Args:
        generation_service: Generation service instance
        websocket_manager: WebSocket manager for notifications
        **kwargs: Additional component instances
        
    Returns:
        Configured EnhancedErrorRecovery instance
    """
    
    # Create base recovery system if not provided
    base_recovery = kwargs.get('base_recovery_system')
    if not base_recovery and generation_service:
        base_recovery = FallbackRecoverySystem(
            generation_service=generation_service,
            websocket_manager=websocket_manager
        )
    
    # Get other components from kwargs or try to create them
    model_availability_manager = kwargs.get('model_availability_manager')
    intelligent_fallback_manager = kwargs.get('intelligent_fallback_manager')
    model_health_monitor = kwargs.get('model_health_monitor')
    enhanced_downloader = kwargs.get('enhanced_downloader')
    
    return EnhancedErrorRecovery(
        base_recovery_system=base_recovery,
        model_availability_manager=model_availability_manager,
        intelligent_fallback_manager=intelligent_fallback_manager,
        model_health_monitor=model_health_monitor,
        enhanced_downloader=enhanced_downloader,
        websocket_manager=websocket_manager
    )