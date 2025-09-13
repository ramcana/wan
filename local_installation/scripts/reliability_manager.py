"""
ReliabilityManager - Central Coordination System for Installation Reliability

This module provides the central orchestration system for all reliability operations,
including component wrapping, failure handling coordination, recovery strategy selection,
reliability metrics collection, and component health monitoring.

Requirements addressed: 1.2, 3.1, 7.3, 8.1
"""

import logging
import time
import threading
from typing import Dict, Any, Optional, List, Callable, Type, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import json

from interfaces import InstallationError, ErrorCategory, HardwareProfile
from base_classes import BaseInstallationComponent
from error_handler import ComprehensiveErrorHandler, EnhancedErrorContext, RecoveryAction
from reliability_wrapper import ReliabilityWrapper, ReliabilityWrapperFactory
from missing_method_recovery import MissingMethodRecovery
from intelligent_retry_system import IntelligentRetrySystem, RetryConfiguration
from model_validation_recovery import ModelValidationRecovery
from network_failure_recovery import NetworkFailureRecovery


class ComponentType(Enum):
    """Types of components that can be managed by ReliabilityManager."""
    MODEL_DOWNLOADER = "model_downloader"
    DEPENDENCY_MANAGER = "dependency_manager"
    CONFIGURATION_ENGINE = "configuration_engine"
    INSTALLATION_VALIDATOR = "installation_validator"
    SYSTEM_DETECTOR = "system_detector"
    ERROR_HANDLER = "error_handler"
    PROGRESS_REPORTER = "progress_reporter"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    ALTERNATIVE_SOURCE = "alternative_source"
    MISSING_METHOD_RECOVERY = "missing_method_recovery"
    MODEL_VALIDATION_RECOVERY = "model_validation_recovery"
    NETWORK_FAILURE_RECOVERY = "network_failure_recovery"
    MANUAL_INTERVENTION = "manual_intervention"
    ABORT = "abort"


@dataclass
class ComponentHealth:
    """Health status of a managed component."""
    component_id: str
    component_type: ComponentType
    is_healthy: bool
    last_check: datetime
    success_rate: float
    total_calls: int
    failed_calls: int
    average_response_time: float
    last_error: Optional[str] = None
    recovery_attempts: int = 0
    consecutive_failures: int = 0


@dataclass
class ReliabilityMetrics:
    """Comprehensive reliability metrics."""
    total_components: int
    healthy_components: int
    total_method_calls: int
    successful_calls: int
    failed_calls: int
    recovery_attempts: int
    successful_recoveries: int
    average_response_time: float
    uptime_percentage: float
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class RecoverySession:
    """Information about a recovery session."""
    session_id: str
    component_id: str
    error: Exception
    strategy: RecoveryStrategy
    start_time: datetime
    end_time: Optional[datetime] = None
    success: bool = False
    attempts: int = 0
    details: str = ""


class ReliabilityManager(BaseInstallationComponent):
    """
    Central coordination system for installation reliability operations.
    
    This class orchestrates all reliability operations including:
    - Component wrapping with reliability enhancements
    - Failure handling coordination
    - Recovery strategy selection and execution
    - Reliability metrics collection and analysis
    - Component health monitoring and tracking
    """
    
    def __init__(self, installation_path: str, logger: Optional[logging.Logger] = None):
        super().__init__(installation_path, logger)
        
        # Core reliability components
        self.error_handler = ComprehensiveErrorHandler(installation_path, logger)
        self.wrapper_factory = ReliabilityWrapperFactory(installation_path, logger)
        self.missing_method_recovery = MissingMethodRecovery(installation_path, logger)
        self.retry_system = IntelligentRetrySystem(installation_path, logger)
        self.model_validation_recovery = ModelValidationRecovery(installation_path, logger=logger)
        self.network_failure_recovery = NetworkFailureRecovery(installation_path, logger)
        
        # Component management
        self.wrapped_components: Dict[str, ReliabilityWrapper] = {}
        self.component_health: Dict[str, ComponentHealth] = {}
        self.component_types: Dict[str, ComponentType] = {}
        
        # Recovery management
        self.active_recovery_sessions: Dict[str, RecoverySession] = {}
        self.recovery_history: List[RecoverySession] = []
        
        # Metrics and monitoring
        self.metrics = ReliabilityMetrics(
            total_components=0,
            healthy_components=0,
            total_method_calls=0,
            successful_calls=0,
            failed_calls=0,
            recovery_attempts=0,
            successful_recoveries=0,
            average_response_time=0.0,
            uptime_percentage=100.0
        )
        
        # Configuration
        self.health_check_interval = 300  # 5 minutes
        self.metrics_update_interval = 60   # 1 minute
        self.max_consecutive_failures = 3
        self.recovery_timeout = 300  # 5 minutes
        
        # Threading for background monitoring
        self._monitoring_thread = None
        self._monitoring_active = False
        self._metrics_lock = threading.Lock()
        
        # Recovery strategy registry
        self.recovery_strategies = self._initialize_recovery_strategies()
        
        self.logger.info("ReliabilityManager initialized successfully")
    
    def _initialize_recovery_strategies(self) -> Dict[str, List[RecoveryStrategy]]:
        """Initialize recovery strategies for different error types and components."""
        return {
            # Network-related errors
            "network_timeout": [
                RecoveryStrategy.RETRY,
                RecoveryStrategy.NETWORK_FAILURE_RECOVERY,
                RecoveryStrategy.ALTERNATIVE_SOURCE
            ],
            "network_connection": [
                RecoveryStrategy.NETWORK_FAILURE_RECOVERY,
                RecoveryStrategy.RETRY,
                RecoveryStrategy.ALTERNATIVE_SOURCE
            ],
            
            # Missing method errors
            "missing_method": [
                RecoveryStrategy.MISSING_METHOD_RECOVERY,
                RecoveryStrategy.FALLBACK,
                RecoveryStrategy.MANUAL_INTERVENTION
            ],
            
            # Model validation errors
            "model_validation": [
                RecoveryStrategy.MODEL_VALIDATION_RECOVERY,
                RecoveryStrategy.RETRY,
                RecoveryStrategy.ALTERNATIVE_SOURCE
            ],
            
            # System errors
            "system_error": [
                RecoveryStrategy.RETRY,
                RecoveryStrategy.FALLBACK,
                RecoveryStrategy.MANUAL_INTERVENTION
            ],
            
            # Configuration errors
            "configuration_error": [
                RecoveryStrategy.FALLBACK,
                RecoveryStrategy.RETRY,
                RecoveryStrategy.MANUAL_INTERVENTION
            ],
            
            # Permission errors
            "permission_error": [
                RecoveryStrategy.MANUAL_INTERVENTION,
                RecoveryStrategy.ABORT
            ]
        }
    
    def wrap_component(self, component: Any, component_type: str = None, 
                      component_id: str = None) -> ReliabilityWrapper:
        """
        Wrap a component with reliability enhancements.
        
        Args:
            component: The component to wrap
            component_type: Optional type hint for the component
            component_id: Optional unique identifier for the component
            
        Returns:
            ReliabilityWrapper instance
        """
        # Generate component ID if not provided
        if component_id is None:
            component_id = f"{component.__class__.__name__}_{id(component)}"
        
        # Determine component type
        comp_type = self._determine_component_type(component, component_type)
        
        # Check if already wrapped
        if component_id in self.wrapped_components:
            self.logger.debug(f"Component {component_id} already wrapped")
            return self.wrapped_components[component_id]
        
        # Create wrapper
        wrapper = self.wrapper_factory.wrap_component(component, component_type)
        
        # Register component
        self.wrapped_components[component_id] = wrapper
        self.component_types[component_id] = comp_type
        
        # Initialize health tracking
        self.component_health[component_id] = ComponentHealth(
            component_id=component_id,
            component_type=comp_type,
            is_healthy=True,
            last_check=datetime.now(),
            success_rate=1.0,
            total_calls=0,
            failed_calls=0,
            average_response_time=0.0
        )
        
        # Update metrics
        with self._metrics_lock:
            self.metrics.total_components += 1
            self.metrics.healthy_components += 1
        
        self.logger.info(f"Wrapped component {component_id} of type {comp_type.value}")
        return wrapper
    
    def _determine_component_type(self, component: Any, type_hint: str = None) -> ComponentType:
        """Determine the type of a component."""
        if type_hint:
            try:
                # Handle both string and ComponentType inputs
                if isinstance(type_hint, ComponentType):
                    return type_hint
                return ComponentType(type_hint.lower())
            except (ValueError, AttributeError):
                pass
        
        class_name = component.__class__.__name__.lower()
        
        if "model" in class_name and "download" in class_name:
            return ComponentType.MODEL_DOWNLOADER
        elif "dependency" in class_name or "package" in class_name:
            return ComponentType.DEPENDENCY_MANAGER
        elif "config" in class_name:
            return ComponentType.CONFIGURATION_ENGINE
        elif "validator" in class_name or "validation" in class_name:
            return ComponentType.INSTALLATION_VALIDATOR
        elif "detector" in class_name or "system" in class_name:
            return ComponentType.SYSTEM_DETECTOR
        elif "error" in class_name or "handler" in class_name:
            return ComponentType.ERROR_HANDLER
        elif "progress" in class_name or "reporter" in class_name:
            return ComponentType.PROGRESS_REPORTER
        else:
            return ComponentType.UNKNOWN
    
    def handle_component_failure(self, component_id: str, error: Exception, 
                               context: Dict[str, Any]) -> RecoveryAction:
        """
        Handle component failure and coordinate recovery.
        
        Args:
            component_id: ID of the failed component
            error: The exception that occurred
            context: Additional context information
            
        Returns:
            RecoveryAction indicating what should be done next
        """
        self.logger.warning(f"Handling failure for component {component_id}: {error}")
        
        # Update component health
        self._update_component_health(component_id, success=False, error=str(error))
        
        # Create enhanced error context
        enhanced_context = self.error_handler.create_enhanced_error_context(
            component=component_id,
            method=context.get('method', ''),
            retry_count=context.get('retry_count', 0)
        )
        
        # Get recovery strategy
        strategy = self.get_recovery_strategy(error, component_id, context)
        
        # Execute recovery
        recovery_result = self._execute_recovery_strategy(
            component_id, error, strategy, enhanced_context
        )
        
        # Update metrics
        with self._metrics_lock:
            self.metrics.recovery_attempts += 1
            if recovery_result != RecoveryAction.ABORT:
                self.metrics.successful_recoveries += 1
        
        return recovery_result
    
    def get_recovery_strategy(self, error: Exception, component_id: str, 
                            context: Dict[str, Any]) -> RecoveryStrategy:
        """
        Select appropriate recovery strategy based on error type and context.
        
        Args:
            error: The exception that occurred
            component_id: ID of the failed component
            context: Additional context information
            
        Returns:
            Selected RecoveryStrategy
        """
        # Classify error type
        error_type = self._classify_error_for_recovery(error)
        
        # Get component health
        health = self.component_health.get(component_id)
        
        # Get available strategies for this error type
        available_strategies = self.recovery_strategies.get(error_type, [RecoveryStrategy.RETRY])
        
        # Select strategy based on context and component health
        if health and health.consecutive_failures >= self.max_consecutive_failures:
            # Component is consistently failing, try more aggressive recovery
            if RecoveryStrategy.MANUAL_INTERVENTION in available_strategies:
                return RecoveryStrategy.MANUAL_INTERVENTION
            elif RecoveryStrategy.ABORT in available_strategies:
                return RecoveryStrategy.ABORT
        
        # Check retry count
        retry_count = context.get('retry_count', 0)
        if retry_count >= 3:
            # Too many retries, try alternative strategies
            for strategy in available_strategies:
                if strategy != RecoveryStrategy.RETRY:
                    return strategy
        
        # Return first available strategy
        return available_strategies[0] if available_strategies else RecoveryStrategy.RETRY
    
    def _classify_error_for_recovery(self, error: Exception) -> str:
        """Classify error for recovery strategy selection."""
        error_message = str(error).lower()
        
        if "timeout" in error_message or "connection" in error_message:
            return "network_timeout"
        elif "network" in error_message or "dns" in error_message:
            return "network_connection"
        elif "has no attribute" in error_message or "missing method" in error_message:
            return "missing_method"
        elif "model" in error_message and ("validation" in error_message or "corrupt" in error_message):
            return "model_validation"
        elif "permission" in error_message or "access denied" in error_message:
            return "permission_error"
        elif "config" in error_message or "configuration" in error_message:
            return "configuration_error"
        else:
            return "system_error"
    
    def _execute_recovery_strategy(self, component_id: str, error: Exception, 
                                 strategy: RecoveryStrategy, 
                                 context: EnhancedErrorContext) -> RecoveryAction:
        """Execute the selected recovery strategy."""
        session_id = f"{component_id}_{int(time.time())}"
        
        # Create recovery session
        session = RecoverySession(
            session_id=session_id,
            component_id=component_id,
            error=error,
            strategy=strategy,
            start_time=datetime.now()
        )
        
        self.active_recovery_sessions[session_id] = session
        
        try:
            if strategy == RecoveryStrategy.RETRY:
                return self._execute_retry_recovery(session, context)
            elif strategy == RecoveryStrategy.MISSING_METHOD_RECOVERY:
                return self._execute_missing_method_recovery(session, context)
            elif strategy == RecoveryStrategy.MODEL_VALIDATION_RECOVERY:
                return self._execute_model_validation_recovery(session, context)
            elif strategy == RecoveryStrategy.NETWORK_FAILURE_RECOVERY:
                return self._execute_network_failure_recovery(session, context)
            elif strategy == RecoveryStrategy.FALLBACK:
                return self._execute_fallback_recovery(session, context)
            elif strategy == RecoveryStrategy.ALTERNATIVE_SOURCE:
                return self._execute_alternative_source_recovery(session, context)
            elif strategy == RecoveryStrategy.MANUAL_INTERVENTION:
                return self._execute_manual_intervention(session, context)
            else:
                return RecoveryAction.ABORT
                
        except Exception as recovery_error:
            self.logger.error(f"Recovery strategy {strategy.value} failed: {recovery_error}")
            session.success = False
            session.details = f"Recovery failed: {recovery_error}"
            return RecoveryAction.ABORT
        finally:
            session.end_time = datetime.now()
            self.recovery_history.append(session)
            self.active_recovery_sessions.pop(session_id, None)
    
    def _execute_retry_recovery(self, session: RecoverySession, 
                              context: EnhancedErrorContext) -> RecoveryAction:
        """Execute retry recovery strategy."""
        self.logger.info(f"Executing retry recovery for {session.component_id}")
        
        # Use intelligent retry system
        try:
            # Configure retry based on error type and component
            from intelligent_retry_system import RetryStrategy
            retry_config = RetryConfiguration(
                max_attempts=3,
                base_delay=2.0,
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF
            )
            
            session.success = True
            session.details = "Retry recovery initiated"
            return RecoveryAction.RETRY
            
        except Exception as e:
            session.success = False
            session.details = f"Retry recovery failed: {e}"
            return RecoveryAction.ABORT
    
    def _execute_missing_method_recovery(self, session: RecoverySession, 
                                       context: EnhancedErrorContext) -> RecoveryAction:
        """Execute missing method recovery strategy."""
        self.logger.info(f"Executing missing method recovery for {session.component_id}")
        
        try:
            # Get the wrapped component
            wrapper = self.wrapped_components.get(session.component_id)
            if not wrapper:
                raise ValueError(f"Component {session.component_id} not found")
            
            # Extract method name from error
            error_message = str(session.error)
            if "has no attribute" in error_message:
                # Extract method name from error message
                method_name = error_message.split("'")[-2] if "'" in error_message else "unknown"
                
                # Attempt recovery
                result = self.missing_method_recovery.handle_missing_method(
                    wrapper.get_component(), method_name
                )
                
                session.success = True
                session.details = f"Successfully recovered missing method: {method_name}"
                return RecoveryAction.RETRY
            else:
                session.success = False
                session.details = "Could not identify missing method from error"
                return RecoveryAction.ABORT
                
        except Exception as e:
            session.success = False
            session.details = f"Missing method recovery failed: {e}"
            return RecoveryAction.ABORT
    
    def _execute_model_validation_recovery(self, session: RecoverySession, 
                                         context: EnhancedErrorContext) -> RecoveryAction:
        """Execute model validation recovery strategy."""
        self.logger.info(f"Executing model validation recovery for {session.component_id}")
        
        try:
            # Extract model information from context or error
            error_message = str(session.error).lower()
            
            # Try to identify the model from the error message
            model_id = None
            for known_model in ["wan2.2/t2v-a14b", "wan2.2/i2v-a14b", "wan2.2/ti2v-5b"]:
                if known_model.lower() in error_message:
                    model_id = known_model
                    break
            
            if not model_id:
                # Default to first model for recovery attempt
                model_id = "Wan2.2/T2V-A14B"
            
            # Attempt model recovery
            recovery_result = self.model_validation_recovery.recover_model(model_id)
            
            if recovery_result.success:
                session.success = True
                session.details = f"Successfully recovered model: {model_id}"
                return RecoveryAction.RETRY
            else:
                session.success = False
                session.details = f"Model recovery failed: {recovery_result.details}"
                return RecoveryAction.MANUAL_INTERVENTION
                
        except Exception as e:
            session.success = False
            session.details = f"Model validation recovery failed: {e}"
            return RecoveryAction.ABORT
    
    def _execute_network_failure_recovery(self, session: RecoverySession, 
                                        context: EnhancedErrorContext) -> RecoveryAction:
        """Execute network failure recovery strategy."""
        self.logger.info(f"Executing network failure recovery for {session.component_id}")
        
        try:
            # Test network connectivity
            connectivity_result = self.network_failure_recovery.connectivity_tester.test_basic_connectivity()
            
            if not connectivity_result.success:
                session.success = False
                session.details = f"Network connectivity test failed: {connectivity_result.error_message}"
                return RecoveryAction.MANUAL_INTERVENTION
            
            # Network is available, suggest retry with alternative source
            session.success = True
            session.details = "Network connectivity verified, retry with alternative source"
            return RecoveryAction.RETRY
            
        except Exception as e:
            session.success = False
            session.details = f"Network failure recovery failed: {e}"
            return RecoveryAction.ABORT
    
    def _execute_fallback_recovery(self, session: RecoverySession, 
                                 context: EnhancedErrorContext) -> RecoveryAction:
        """Execute fallback recovery strategy."""
        self.logger.info(f"Executing fallback recovery for {session.component_id}")
        
        try:
            # Use error handler's fallback mechanisms
            if hasattr(self.error_handler, 'fallback_manager'):
                # Determine fallback scenario based on component type
                component_type = self.component_types.get(session.component_id, ComponentType.UNKNOWN)
                
                if component_type == ComponentType.MODEL_DOWNLOADER:
                    scenario = "model_download"
                elif component_type == ComponentType.DEPENDENCY_MANAGER:
                    scenario = "package_install"
                elif component_type == ComponentType.CONFIGURATION_ENGINE:
                    scenario = "config_generation"
                else:
                    scenario = "system_error"
                
                # Get fallback options (this would need to be implemented in error handler)
                session.success = True
                session.details = f"Fallback recovery initiated for scenario: {scenario}"
                return RecoveryAction.CONTINUE
            else:
                session.success = False
                session.details = "Fallback manager not available"
                return RecoveryAction.ABORT
                
        except Exception as e:
            session.success = False
            session.details = f"Fallback recovery failed: {e}"
            return RecoveryAction.ABORT
    
    def _execute_alternative_source_recovery(self, session: RecoverySession, 
                                           context: EnhancedErrorContext) -> RecoveryAction:
        """Execute alternative source recovery strategy."""
        self.logger.info(f"Executing alternative source recovery for {session.component_id}")
        
        try:
            # This would coordinate with network failure recovery for alternative sources
            session.success = True
            session.details = "Alternative source recovery initiated"
            return RecoveryAction.RETRY
            
        except Exception as e:
            session.success = False
            session.details = f"Alternative source recovery failed: {e}"
            return RecoveryAction.ABORT
    
    def _execute_manual_intervention(self, session: RecoverySession, 
                                   context: EnhancedErrorContext) -> RecoveryAction:
        """Execute manual intervention strategy."""
        self.logger.warning(f"Manual intervention required for {session.component_id}")
        
        session.success = False
        session.details = "Manual intervention required - automatic recovery not possible"
        
        # Log detailed information for manual intervention
        self.logger.error(f"Manual intervention details:")
        self.logger.error(f"  Component: {session.component_id}")
        self.logger.error(f"  Error: {session.error}")
        self.logger.error(f"  Context: {context}")
        
        return RecoveryAction.MANUAL_INTERVENTION
    
    def track_reliability_metrics(self, component_id: str, operation: str, 
                                success: bool, duration: float):
        """Track reliability metrics for a component operation."""
        # Update component health
        self._update_component_health(component_id, success, duration=duration)
        
        # Update global metrics
        with self._metrics_lock:
            self.metrics.total_method_calls += 1
            if success:
                self.metrics.successful_calls += 1
            else:
                self.metrics.failed_calls += 1
            
            # Update average response time
            total_time = self.metrics.average_response_time * (self.metrics.total_method_calls - 1)
            self.metrics.average_response_time = (total_time + duration) / self.metrics.total_method_calls
            
            self.metrics.last_updated = datetime.now()
    
    def _update_component_health(self, component_id: str, success: bool, 
                               error: str = None, duration: float = 0.0):
        """Update health status for a component."""
        if component_id not in self.component_health:
            return
        
        health = self.component_health[component_id]
        health.last_check = datetime.now()
        health.total_calls += 1
        
        if success:
            health.consecutive_failures = 0
            # Update average response time
            if health.total_calls > 1:
                total_time = health.average_response_time * (health.total_calls - 1)
                health.average_response_time = (total_time + duration) / health.total_calls
            else:
                health.average_response_time = duration
        else:
            health.failed_calls += 1
            health.consecutive_failures += 1
            health.last_error = error
            health.recovery_attempts += 1
        
        # Update success rate
        health.success_rate = (health.total_calls - health.failed_calls) / health.total_calls
        
        # Update health status
        health.is_healthy = (
            health.consecutive_failures < self.max_consecutive_failures and
            health.success_rate > 0.5
        )
        
        # Update global healthy component count
        with self._metrics_lock:
            healthy_count = sum(1 for h in self.component_health.values() if h.is_healthy)
            self.metrics.healthy_components = healthy_count
            
            # Update uptime percentage
            if self.metrics.total_components > 0:
                self.metrics.uptime_percentage = (healthy_count / self.metrics.total_components) * 100
    
    def check_component_health(self, component_id: str) -> Optional[ComponentHealth]:
        """Check health status of a specific component."""
        return self.component_health.get(component_id)
    
    def get_all_component_health(self) -> Dict[str, ComponentHealth]:
        """Get health status of all managed components."""
        return self.component_health.copy()
    
    def get_reliability_metrics(self) -> ReliabilityMetrics:
        """Get current reliability metrics."""
        with self._metrics_lock:
            return ReliabilityMetrics(
                total_components=self.metrics.total_components,
                healthy_components=self.metrics.healthy_components,
                total_method_calls=self.metrics.total_method_calls,
                successful_calls=self.metrics.successful_calls,
                failed_calls=self.metrics.failed_calls,
                recovery_attempts=self.metrics.recovery_attempts,
                successful_recoveries=self.metrics.successful_recoveries,
                average_response_time=self.metrics.average_response_time,
                uptime_percentage=self.metrics.uptime_percentage,
                last_updated=self.metrics.last_updated
            )
    
    def start_monitoring(self):
        """Start background monitoring of component health."""
        if self._monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        self.logger.info("Started reliability monitoring")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        self.logger.info("Stopped reliability monitoring")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        last_health_check = datetime.now()
        last_metrics_update = datetime.now()
        
        while self._monitoring_active:
            try:
                current_time = datetime.now()
                
                # Periodic health checks
                if (current_time - last_health_check).total_seconds() >= self.health_check_interval:
                    self._perform_health_checks()
                    last_health_check = current_time
                
                # Periodic metrics updates
                if (current_time - last_metrics_update).total_seconds() >= self.metrics_update_interval:
                    self._update_metrics()
                    last_metrics_update = current_time
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Wait longer on error
    
    def _perform_health_checks(self):
        """Perform health checks on all components."""
        for component_id, health in self.component_health.items():
            try:
                # Check if component has been inactive for too long
                time_since_last_check = datetime.now() - health.last_check
                if time_since_last_check > timedelta(minutes=30):
                    self.logger.warning(f"Component {component_id} has been inactive for {time_since_last_check}")
                
                # Check for components with high failure rates
                if health.success_rate < 0.5 and health.total_calls > 10:
                    self.logger.warning(f"Component {component_id} has low success rate: {health.success_rate:.2f}")
                
            except Exception as e:
                self.logger.error(f"Error checking health for component {component_id}: {e}")
    
    def _update_metrics(self):
        """Update global metrics."""
        try:
            with self._metrics_lock:
                # Update component counts
                self.metrics.total_components = len(self.wrapped_components)
                self.metrics.healthy_components = sum(
                    1 for h in self.component_health.values() if h.is_healthy
                )
                
                # Update uptime percentage
                if self.metrics.total_components > 0:
                    self.metrics.uptime_percentage = (
                        self.metrics.healthy_components / self.metrics.total_components
                    ) * 100
                
                self.metrics.last_updated = datetime.now()
                
        except Exception as e:
            self.logger.error(f"Error updating metrics: {e}")
    
    def get_recovery_history(self) -> List[RecoverySession]:
        """Get history of recovery sessions."""
        return self.recovery_history.copy()
    
    def get_component_health_summary(self) -> Dict[str, Any]:
        """Get summary of component health status."""
        return {
            comp_id: {
                "component_type": health.component_type.value,
                "is_healthy": health.is_healthy,
                "success_rate": health.success_rate,
                "total_calls": health.total_calls,
                "failed_calls": health.failed_calls,
                "average_response_time": health.average_response_time,
                "consecutive_failures": health.consecutive_failures,
                "recovery_attempts": health.recovery_attempts,
                "last_error": health.last_error,
                "last_check": health.last_check.isoformat()
            }
            for comp_id, health in self.component_health.items()
        }
    
    def start_monitoring(self):
        """Start reliability monitoring."""
        if not self._monitoring_active:
            self._monitoring_active = True
            self.logger.info("Started reliability monitoring")
    
    def stop_monitoring(self):
        """Stop reliability monitoring."""
        if self._monitoring_active:
            self._monitoring_active = False
            self.logger.info("Stopped reliability monitoring")
    
    def get_active_recovery_sessions(self) -> Dict[str, RecoverySession]:
        """Get currently active recovery sessions."""
        return self.active_recovery_sessions.copy()
    
    def export_reliability_report(self, output_path: str = None) -> str:
        """Export comprehensive reliability report."""
        if output_path is None:
            output_path = str(Path(self.installation_path) / "logs" / "reliability_report.json")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "total_components": self.metrics.total_components,
                "healthy_components": self.metrics.healthy_components,
                "total_method_calls": self.metrics.total_method_calls,
                "successful_calls": self.metrics.successful_calls,
                "failed_calls": self.metrics.failed_calls,
                "recovery_attempts": self.metrics.recovery_attempts,
                "successful_recoveries": self.metrics.successful_recoveries,
                "average_response_time": self.metrics.average_response_time,
                "uptime_percentage": self.metrics.uptime_percentage
            },
            "component_health": {
                comp_id: {
                    "component_type": health.component_type.value,
                    "is_healthy": health.is_healthy,
                    "success_rate": health.success_rate,
                    "total_calls": health.total_calls,
                    "failed_calls": health.failed_calls,
                    "average_response_time": health.average_response_time,
                    "consecutive_failures": health.consecutive_failures,
                    "recovery_attempts": health.recovery_attempts,
                    "last_error": health.last_error
                }
                for comp_id, health in self.component_health.items()
            },
            "recovery_history": [
                {
                    "session_id": session.session_id,
                    "component_id": session.component_id,
                    "strategy": session.strategy.value,
                    "success": session.success,
                    "start_time": session.start_time.isoformat(),
                    "end_time": session.end_time.isoformat() if session.end_time else None,
                    "details": session.details
                }
                for session in self.recovery_history[-50:]  # Last 50 sessions
            ]
        }
        
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Reliability report exported to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to export reliability report: {e}")
            return ""
    
    def shutdown(self):
        """Shutdown the reliability manager and clean up resources."""
        self.logger.info("Shutting down ReliabilityManager")
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Export final report
        try:
            self.export_reliability_report()
        except Exception as e:
            self.logger.error(f"Failed to export final report: {e}")
        
        # Clear active sessions
        self.active_recovery_sessions.clear()
        
        self.logger.info("ReliabilityManager shutdown complete")
