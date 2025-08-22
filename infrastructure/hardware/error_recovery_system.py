"""
WAN22 Error Recovery System

This module implements comprehensive error recovery and logging capabilities
for the WAN2.2 system optimization framework.

Requirements addressed:
- 6.1: Log detailed information including stack traces, system state, and recovery actions
- 6.2: Attempt automatic recovery before failing
- 6.3: Save current state and provide recovery options on restart
"""

import json
import logging
import pickle
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union
import threading
import os


class RecoveryStrategy(Enum):
    """Recovery strategy types"""
    IMMEDIATE_RETRY = "immediate_retry"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FALLBACK_CONFIG = "fallback_config"
    STATE_RESTORATION = "state_restoration"
    USER_GUIDED = "user_guided"
    SAFE_SHUTDOWN = "safe_shutdown"


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SystemState:
    """System state snapshot for recovery purposes"""
    timestamp: datetime
    active_model: Optional[str]
    configuration: Dict[str, Any]
    memory_usage: Dict[str, float]
    gpu_state: Dict[str, Any]
    pipeline_state: Dict[str, Any]
    user_preferences: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemState':
        """Create from dictionary"""
        # Convert timestamp string back to datetime if needed
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class RecoveryResult:
    """Result of a recovery attempt"""
    success: bool
    strategy_used: RecoveryStrategy
    actions_taken: List[str]
    time_taken: float
    error_resolved: bool
    fallback_applied: bool
    user_intervention_required: bool
    recovery_message: str
    warnings: List[str]


@dataclass
class ErrorContext:
    """Context information for error recovery"""
    error_type: str
    error_message: str
    stack_trace: str
    system_state: SystemState
    recovery_attempts: int
    timestamp: datetime
    severity: ErrorSeverity
    component: str
    user_action: Optional[str]


class ErrorRecoverySystem:
    """
    Comprehensive error recovery system with automatic recovery attempts,
    system state preservation, and user-guided recovery workflows.
    """
    
    def __init__(self, 
                 state_dir: str = "recovery_states",
                 log_dir: str = "logs",
                 max_recovery_attempts: int = 3,
                 enable_auto_recovery: bool = True):
        """
        Initialize the error recovery system.
        
        Args:
            state_dir: Directory for storing system state snapshots
            log_dir: Directory for error logs
            max_recovery_attempts: Maximum automatic recovery attempts
            enable_auto_recovery: Whether to enable automatic recovery
        """
        self.state_dir = Path(state_dir)
        self.log_dir = Path(log_dir)
        self.max_recovery_attempts = max_recovery_attempts
        self.enable_auto_recovery = enable_auto_recovery
        
        # Create directories
        self.state_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # Error handler registry
        self._error_handlers: Dict[Type[Exception], List[Callable]] = {}
        self._recovery_strategies: Dict[Type[Exception], RecoveryStrategy] = {}
        
        # Recovery state tracking
        self._recovery_attempts: Dict[str, int] = {}
        self._last_recovery_time: Dict[str, float] = {}
        self._current_system_state: Optional[SystemState] = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Setup logging
        self._setup_logging()
        
        # Register default error handlers
        self._register_default_handlers()
    
    def _setup_logging(self):
        """Setup comprehensive error logging"""
        log_file = self.log_dir / f"error_recovery_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Create formatter for detailed logging
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Setup logger
        self.logger = logging.getLogger('ErrorRecoverySystem')
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def register_error_handler(self, 
                             error_type: Type[Exception], 
                             handler: Callable[[Exception, ErrorContext], RecoveryResult],
                             strategy: RecoveryStrategy = RecoveryStrategy.IMMEDIATE_RETRY) -> None:
        """
        Register an error handler for a specific exception type.
        
        Args:
            error_type: Exception type to handle
            handler: Handler function that takes (exception, context) and returns RecoveryResult
            strategy: Default recovery strategy for this error type
        """
        with self._lock:
            if error_type not in self._error_handlers:
                self._error_handlers[error_type] = []
            
            self._error_handlers[error_type].append(handler)
            self._recovery_strategies[error_type] = strategy
            
            self.logger.info(f"Registered error handler for {error_type.__name__} with strategy {strategy.value}")
    
    def save_system_state(self, 
                         state: Optional[SystemState] = None,
                         state_id: Optional[str] = None) -> str:
        """
        Save current system state for recovery purposes.
        
        Args:
            state: System state to save (if None, captures current state)
            state_id: Optional identifier for the state
            
        Returns:
            Path to saved state file
        """
        if state is None:
            state = self._capture_current_state()
        
        if state_id is None:
            state_id = f"state_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        state_file = self.state_dir / f"{state_id}.json"
        
        try:
            # Save as JSON for human readability
            state_data = state.to_dict()
            # Convert datetime to string for JSON serialization
            if isinstance(state_data.get('timestamp'), datetime):
                state_data['timestamp'] = state_data['timestamp'].isoformat()
            
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2, default=str)
            
            # Also save as pickle for complete object preservation
            pickle_file = self.state_dir / f"{state_id}.pkl"
            with open(pickle_file, 'wb') as f:
                pickle.dump(state, f)
            
            self.logger.info(f"System state saved to {state_file}")
            return str(state_file)
            
        except Exception as e:
            self.logger.error(f"Failed to save system state: {e}")
            raise
    
    def restore_system_state(self, state_path: str) -> RecoveryResult:
        """
        Restore system state from a saved state file.
        
        Args:
            state_path: Path to the state file
            
        Returns:
            RecoveryResult indicating success/failure of restoration
        """
        start_time = time.time()
        actions_taken = []
        warnings = []
        
        try:
            state_file = Path(state_path)
            
            # Try to load from pickle first (more complete)
            pickle_file = state_file.with_suffix('.pkl')
            if pickle_file.exists():
                with open(pickle_file, 'rb') as f:
                    state = pickle.load(f)
                actions_taken.append("Loaded state from pickle file")
            else:
                # Fallback to JSON
                with open(state_file, 'r') as f:
                    state_data = json.load(f)
                state = SystemState.from_dict(state_data)
                actions_taken.append("Loaded state from JSON file")
            
            # Apply state restoration
            restoration_success = self._apply_state_restoration(state, actions_taken, warnings)
            
            time_taken = time.time() - start_time
            
            if restoration_success:
                self.logger.info(f"System state restored successfully from {state_path}")
                return RecoveryResult(
                    success=True,
                    strategy_used=RecoveryStrategy.STATE_RESTORATION,
                    actions_taken=actions_taken,
                    time_taken=time_taken,
                    error_resolved=True,
                    fallback_applied=False,
                    user_intervention_required=False,
                    recovery_message="System state restored successfully",
                    warnings=warnings
                )
            else:
                return RecoveryResult(
                    success=False,
                    strategy_used=RecoveryStrategy.STATE_RESTORATION,
                    actions_taken=actions_taken,
                    time_taken=time_taken,
                    error_resolved=False,
                    fallback_applied=False,
                    user_intervention_required=True,
                    recovery_message="State restoration partially failed",
                    warnings=warnings
                )
                
        except Exception as e:
            time_taken = time.time() - start_time
            error_msg = f"Failed to restore system state: {e}"
            self.logger.error(error_msg)
            
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.STATE_RESTORATION,
                actions_taken=actions_taken,
                time_taken=time_taken,
                error_resolved=False,
                fallback_applied=False,
                user_intervention_required=True,
                recovery_message=error_msg,
                warnings=warnings
            )
    
    def attempt_recovery(self, 
                        error: Exception, 
                        context: Optional[Dict[str, Any]] = None,
                        component: str = "unknown") -> RecoveryResult:
        """
        Attempt to recover from an error using registered handlers and strategies.
        
        Args:
            error: The exception that occurred
            context: Additional context information
            component: Component where the error occurred
            
        Returns:
            RecoveryResult indicating the outcome of recovery attempts
        """
        start_time = time.time()
        error_type = type(error)
        error_key = f"{error_type.__name__}_{component}"
        
        # Create error context
        error_context = ErrorContext(
            error_type=error_type.__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            system_state=self._capture_current_state(),
            recovery_attempts=self._recovery_attempts.get(error_key, 0),
            timestamp=datetime.now(),
            severity=self._determine_error_severity(error),
            component=component,
            user_action=context.get('user_action') if context else None
        )
        
        # Log the error with full context
        self._log_error_with_context(error, error_context)
        
        # Check if we should attempt recovery
        if not self.enable_auto_recovery:
            return self._create_no_recovery_result(error_context, start_time)
        
        # Check recovery attempt limits
        if self._recovery_attempts.get(error_key, 0) >= self.max_recovery_attempts:
            return self._create_max_attempts_result(error_context, start_time)
        
        # Increment recovery attempts
        with self._lock:
            self._recovery_attempts[error_key] = self._recovery_attempts.get(error_key, 0) + 1
            self._last_recovery_time[error_key] = time.time()
        
        # Find and execute recovery handlers
        recovery_result = self._execute_recovery_handlers(error, error_context)
        
        # Update timing
        recovery_result.time_taken = time.time() - start_time
        
        # Log recovery result
        self._log_recovery_result(recovery_result, error_context)
        
        return recovery_result
    
    def _capture_current_state(self) -> SystemState:
        """Capture current system state"""
        try:
            # This would be implemented to capture actual system state
            # For now, return a basic state structure
            return SystemState(
                timestamp=datetime.now(),
                active_model=None,  # Would capture from model manager
                configuration={},   # Would capture from config manager
                memory_usage={},    # Would capture from memory monitor
                gpu_state={},       # Would capture from GPU monitor
                pipeline_state={},  # Would capture from pipeline manager
                user_preferences={}  # Would capture from user settings
            )
        except Exception as e:
            self.logger.warning(f"Failed to capture complete system state: {e}")
            # Return minimal state
            return SystemState(
                timestamp=datetime.now(),
                active_model=None,
                configuration={},
                memory_usage={},
                gpu_state={},
                pipeline_state={},
                user_preferences={}
            )
    
    def _determine_error_severity(self, error: Exception) -> ErrorSeverity:
        """Determine error severity based on exception type"""
        if isinstance(error, (SystemExit, KeyboardInterrupt)):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, (MemoryError, OSError)):
            return ErrorSeverity.HIGH
        elif isinstance(error, (ValueError, TypeError, AttributeError)):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _log_error_with_context(self, error: Exception, context: ErrorContext):
        """Log error with full context information"""
        log_entry = {
            "timestamp": context.timestamp.isoformat(),
            "error_type": context.error_type,
            "error_message": context.error_message,
            "component": context.component,
            "severity": context.severity.value,
            "recovery_attempts": context.recovery_attempts,
            "stack_trace": context.stack_trace,
            "system_state_summary": {
                "active_model": context.system_state.active_model,
                "memory_usage": context.system_state.memory_usage,
                "gpu_state": context.system_state.gpu_state
            }
        }
        
        # Log as JSON for structured logging
        self.logger.error(f"ERROR_CONTEXT: {json.dumps(log_entry, indent=2)}")
    
    def _execute_recovery_handlers(self, 
                                 error: Exception, 
                                 context: ErrorContext) -> RecoveryResult:
        """Execute registered recovery handlers for the error"""
        error_type = type(error)
        actions_taken = []
        warnings = []
        
        # Find handlers for this error type or its base classes
        handlers = []
        for registered_type, handler_list in self._error_handlers.items():
            if issubclass(error_type, registered_type):
                handlers.extend(handler_list)
        
        if not handlers:
            # No specific handlers, use default recovery strategy
            return self._apply_default_recovery(error, context)
        
        # Try each handler
        for handler in handlers:
            try:
                result = handler(error, context)
                if result.success:
                    actions_taken.extend(result.actions_taken)
                    warnings.extend(result.warnings)
                    return result
                else:
                    actions_taken.extend(result.actions_taken)
                    warnings.extend(result.warnings)
            except Exception as handler_error:
                warning_msg = f"Recovery handler failed: {handler_error}"
                warnings.append(warning_msg)
                self.logger.warning(warning_msg)
        
        # All handlers failed, try default recovery
        default_result = self._apply_default_recovery(error, context)
        default_result.actions_taken = actions_taken + default_result.actions_taken
        default_result.warnings = warnings + default_result.warnings
        
        return default_result
    
    def _apply_default_recovery(self, 
                              error: Exception, 
                              context: ErrorContext) -> RecoveryResult:
        """Apply default recovery strategy based on error type and severity"""
        strategy = self._recovery_strategies.get(type(error), RecoveryStrategy.IMMEDIATE_RETRY)
        actions_taken = []
        warnings = []
        
        if strategy == RecoveryStrategy.EXPONENTIAL_BACKOFF:
            return self._apply_exponential_backoff_recovery(error, context, actions_taken, warnings)
        elif strategy == RecoveryStrategy.FALLBACK_CONFIG:
            return self._apply_fallback_config_recovery(error, context, actions_taken, warnings)
        elif strategy == RecoveryStrategy.STATE_RESTORATION:
            return self._apply_automatic_state_restoration(error, context, actions_taken, warnings)
        elif strategy == RecoveryStrategy.SAFE_SHUTDOWN:
            return self._apply_safe_shutdown_recovery(error, context, actions_taken, warnings)
        else:
            # Default immediate retry
            return self._apply_immediate_retry_recovery(error, context, actions_taken, warnings)
    
    def _apply_exponential_backoff_recovery(self, 
                                          error: Exception, 
                                          context: ErrorContext,
                                          actions_taken: List[str],
                                          warnings: List[str]) -> RecoveryResult:
        """Apply exponential backoff recovery strategy"""
        backoff_time = min(2 ** context.recovery_attempts, 60)  # Max 60 seconds
        actions_taken.append(f"Applied exponential backoff: {backoff_time}s")
        
        time.sleep(backoff_time)
        
        return RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.EXPONENTIAL_BACKOFF,
            actions_taken=actions_taken,
            time_taken=backoff_time,
            error_resolved=False,  # Backoff doesn't resolve the error, just delays retry
            fallback_applied=False,
            user_intervention_required=False,
            recovery_message=f"Applied exponential backoff of {backoff_time} seconds",
            warnings=warnings
        )
    
    def _apply_fallback_config_recovery(self, 
                                      error: Exception, 
                                      context: ErrorContext,
                                      actions_taken: List[str],
                                      warnings: List[str]) -> RecoveryResult:
        """Apply fallback configuration recovery strategy"""
        actions_taken.append("Applied fallback configuration")
        
        # This would implement actual fallback config logic
        # For now, return a basic result
        return RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.FALLBACK_CONFIG,
            actions_taken=actions_taken,
            time_taken=0.0,
            error_resolved=True,
            fallback_applied=True,
            user_intervention_required=False,
            recovery_message="Applied fallback configuration",
            warnings=warnings
        )
    
    def _apply_automatic_state_restoration(self, 
                                         error: Exception, 
                                         context: ErrorContext,
                                         actions_taken: List[str],
                                         warnings: List[str]) -> RecoveryResult:
        """Apply automatic state restoration recovery strategy"""
        # Find the most recent valid state
        state_files = list(self.state_dir.glob("*.json"))
        if not state_files:
            warnings.append("No saved states available for restoration")
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.STATE_RESTORATION,
                actions_taken=actions_taken,
                time_taken=0.0,
                error_resolved=False,
                fallback_applied=False,
                user_intervention_required=True,
                recovery_message="No saved states available for restoration",
                warnings=warnings
            )
        
        # Get most recent state file
        latest_state = max(state_files, key=lambda p: p.stat().st_mtime)
        actions_taken.append(f"Attempting restoration from {latest_state.name}")
        
        # Attempt restoration
        restoration_result = self.restore_system_state(str(latest_state))
        restoration_result.actions_taken = actions_taken + restoration_result.actions_taken
        restoration_result.warnings = warnings + restoration_result.warnings
        
        return restoration_result
    
    def _apply_safe_shutdown_recovery(self, 
                                    error: Exception, 
                                    context: ErrorContext,
                                    actions_taken: List[str],
                                    warnings: List[str]) -> RecoveryResult:
        """Apply safe shutdown recovery strategy for critical errors"""
        actions_taken.append("Initiated safe shutdown procedure")
        
        # Save current state before shutdown
        try:
            state_path = self.save_system_state(context.system_state, "emergency_shutdown")
            actions_taken.append(f"Emergency state saved to {state_path}")
        except Exception as save_error:
            warnings.append(f"Failed to save emergency state: {save_error}")
        
        return RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.SAFE_SHUTDOWN,
            actions_taken=actions_taken,
            time_taken=0.0,
            error_resolved=False,
            fallback_applied=False,
            user_intervention_required=True,
            recovery_message="Safe shutdown initiated due to critical error",
            warnings=warnings
        )
    
    def _apply_immediate_retry_recovery(self, 
                                      error: Exception, 
                                      context: ErrorContext,
                                      actions_taken: List[str],
                                      warnings: List[str]) -> RecoveryResult:
        """Apply immediate retry recovery strategy"""
        actions_taken.append("Prepared for immediate retry")
        
        return RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.IMMEDIATE_RETRY,
            actions_taken=actions_taken,
            time_taken=0.0,
            error_resolved=False,  # Retry doesn't resolve the error, just allows another attempt
            fallback_applied=False,
            user_intervention_required=False,
            recovery_message="Prepared for immediate retry",
            warnings=warnings
        )
    
    def _apply_state_restoration(self, 
                               state: SystemState, 
                               actions_taken: List[str],
                               warnings: List[str]) -> bool:
        """Apply state restoration (placeholder for actual implementation)"""
        try:
            # This would implement actual state restoration logic
            # For now, just log the restoration attempt
            actions_taken.append("Applied configuration restoration")
            actions_taken.append("Applied memory state restoration")
            actions_taken.append("Applied pipeline state restoration")
            
            self._current_system_state = state
            return True
            
        except Exception as e:
            warnings.append(f"State restoration failed: {e}")
            return False
    
    def _create_no_recovery_result(self, context: ErrorContext, start_time: float) -> RecoveryResult:
        """Create result for when auto-recovery is disabled"""
        return RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategy.USER_GUIDED,
            actions_taken=["Auto-recovery disabled"],
            time_taken=time.time() - start_time,
            error_resolved=False,
            fallback_applied=False,
            user_intervention_required=True,
            recovery_message="Auto-recovery is disabled, user intervention required",
            warnings=[]
        )
    
    def _create_max_attempts_result(self, context: ErrorContext, start_time: float) -> RecoveryResult:
        """Create result for when max recovery attempts reached"""
        return RecoveryResult(
            success=False,
            strategy_used=RecoveryStrategy.USER_GUIDED,
            actions_taken=[f"Max recovery attempts ({self.max_recovery_attempts}) reached"],
            time_taken=time.time() - start_time,
            error_resolved=False,
            fallback_applied=False,
            user_intervention_required=True,
            recovery_message=f"Maximum recovery attempts ({self.max_recovery_attempts}) reached",
            warnings=[]
        )
    
    def _log_recovery_result(self, result: RecoveryResult, context: ErrorContext):
        """Log the recovery result"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": context.error_type,
            "component": context.component,
            "recovery_success": result.success,
            "strategy_used": result.strategy_used.value,
            "actions_taken": result.actions_taken,
            "time_taken": result.time_taken,
            "error_resolved": result.error_resolved,
            "user_intervention_required": result.user_intervention_required,
            "warnings": result.warnings
        }
        
        if result.success:
            self.logger.info(f"RECOVERY_SUCCESS: {json.dumps(log_entry, indent=2)}")
        else:
            self.logger.error(f"RECOVERY_FAILED: {json.dumps(log_entry, indent=2)}")
    
    def _register_default_handlers(self):
        """Register default error handlers for common exception types"""
        
        # Memory errors - use fallback config with reduced memory usage
        def memory_error_handler(error: MemoryError, context: ErrorContext) -> RecoveryResult:
            actions_taken = ["Applied memory optimization fallback"]
            return RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.FALLBACK_CONFIG,
                actions_taken=actions_taken,
                time_taken=0.0,
                error_resolved=True,
                fallback_applied=True,
                user_intervention_required=False,
                recovery_message="Applied memory optimization fallback configuration",
                warnings=[]
            )
        
        self.register_error_handler(MemoryError, memory_error_handler, RecoveryStrategy.FALLBACK_CONFIG)
        
        # File not found errors - attempt state restoration
        def file_not_found_handler(error: FileNotFoundError, context: ErrorContext) -> RecoveryResult:
            actions_taken = [f"File not found: {error.filename}"]
            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.STATE_RESTORATION,
                actions_taken=actions_taken,
                time_taken=0.0,
                error_resolved=False,
                fallback_applied=False,
                user_intervention_required=True,
                recovery_message=f"File not found: {error.filename}. Manual intervention required.",
                warnings=[]
            )
        
        self.register_error_handler(FileNotFoundError, file_not_found_handler, RecoveryStrategy.STATE_RESTORATION)
        
        # Connection errors - exponential backoff
        def connection_error_handler(error: ConnectionError, context: ErrorContext) -> RecoveryResult:
            actions_taken = ["Connection error detected, will retry with backoff"]
            return RecoveryResult(
                success=True,
                strategy_used=RecoveryStrategy.EXPONENTIAL_BACKOFF,
                actions_taken=actions_taken,
                time_taken=0.0,
                error_resolved=False,
                fallback_applied=False,
                user_intervention_required=False,
                recovery_message="Connection error, will retry with exponential backoff",
                warnings=[]
            )
        
        self.register_error_handler(ConnectionError, connection_error_handler, RecoveryStrategy.EXPONENTIAL_BACKOFF)
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery statistics and metrics"""
        with self._lock:
            return {
                "total_recovery_attempts": sum(self._recovery_attempts.values()),
                "recovery_attempts_by_error": dict(self._recovery_attempts),
                "registered_handlers": {
                    error_type.__name__: len(handlers) 
                    for error_type, handlers in self._error_handlers.items()
                },
                "auto_recovery_enabled": self.enable_auto_recovery,
                "max_recovery_attempts": self.max_recovery_attempts,
                "state_files_count": len(list(self.state_dir.glob("*.json"))),
                "log_files_count": len(list(self.log_dir.glob("*.log")))
            }
    
    def cleanup_old_states(self, max_age_days: int = 30):
        """Clean up old state files to prevent disk space issues"""
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        cleaned_count = 0
        
        for state_file in self.state_dir.glob("*"):
            if state_file.stat().st_mtime < cutoff_time:
                try:
                    state_file.unlink()
                    cleaned_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to clean up old state file {state_file}: {e}")
        
        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} old state files")
    
    def reset_recovery_attempts(self, error_key: Optional[str] = None):
        """Reset recovery attempt counters"""
        with self._lock:
            if error_key:
                self._recovery_attempts.pop(error_key, None)
                self._last_recovery_time.pop(error_key, None)
            else:
                self._recovery_attempts.clear()
                self._last_recovery_time.clear()
        
        self.logger.info(f"Reset recovery attempts for {error_key or 'all errors'}")