"""
Intelligent Retry System with Exponential Backoff
"""

import logging
import time
import random
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# Import from local scripts
from interfaces import InstallationError, ErrorCategory
from base_classes import BaseInstallationComponent


class RetryStrategy(Enum):
    """Available retry strategies."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    IMMEDIATE = "immediate"
    FIBONACCI = "fibonacci"


@dataclass
class RetryConfiguration:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 2.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    user_prompt: bool = True


@dataclass
class RetryAttempt:
    """Information about a retry attempt."""
    attempt_number: int
    timestamp: datetime
    delay_before: float
    error: Optional[Exception]
    success: bool
    duration: float


@dataclass
class RetrySession:
    """Information about a complete retry session."""
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_attempts: int = 0
    successful: bool = False
    final_error: Optional[Exception] = None
    attempts: List[RetryAttempt] = field(default_factory=list)


class IntelligentRetrySystem(BaseInstallationComponent):
    """Intelligent retry system with exponential backoff and user control."""
    
    def __init__(self, installation_path: str, logger: Optional[logging.Logger] = None):
        super().__init__(installation_path, logger)
        self.active_sessions: Dict[str, RetrySession] = {}
        self.session_history: List[RetrySession] = []
        self.global_config = RetryConfiguration()
    
    def execute_with_retry(self, operation: Callable, operation_name: str,
                          error_category: Optional[str] = None,
                          context: Optional[Dict[str, Any]] = None,
                          custom_config: Optional[RetryConfiguration] = None) -> Any:
        """Execute an operation with intelligent retry logic."""
        session = RetrySession(operation_name=operation_name, start_time=datetime.now())
        self.active_sessions[operation_name] = session
        
        try:
            config = custom_config or self.global_config
            last_error = None
            
            for attempt_number in range(1, config.max_attempts + 1):
                session.total_attempts = attempt_number
                attempt_start = datetime.now()
                
                try:
                    result = operation()
                    
                    # Success!
                    attempt_duration = (datetime.now() - attempt_start).total_seconds()
                    attempt = RetryAttempt(
                        attempt_number=attempt_number,
                        timestamp=attempt_start,
                        delay_before=0.0,
                        error=None,
                        success=True,
                        duration=attempt_duration
                    )
                    session.attempts.append(attempt)
                    session.successful = True
                    
                    self.logger.info(f"{session.operation_name} succeeded on attempt {attempt_number}")
                    return result
                    
                except Exception as error:
                    last_error = error
                    attempt_duration = (datetime.now() - attempt_start).total_seconds()
                    
                    self.logger.warning(f"{session.operation_name} failed on attempt {attempt_number}: {error}")
                    
                    if attempt_number >= config.max_attempts:
                        attempt = RetryAttempt(
                            attempt_number=attempt_number,
                            timestamp=attempt_start,
                            delay_before=0.0,
                            error=error,
                            success=False,
                            duration=attempt_duration
                        )
                        session.attempts.append(attempt)
                        break
                    
                    # Calculate delay before next attempt
                    delay = self._calculate_delay(attempt_number - 1, config)
                    
                    attempt = RetryAttempt(
                        attempt_number=attempt_number,
                        timestamp=attempt_start,
                        delay_before=delay,
                        error=error,
                        success=False,
                        duration=attempt_duration
                    )
                    session.attempts.append(attempt)
                    
                    # Wait before next attempt
                    if delay > 0:
                        self.logger.info(f"Waiting {delay:.1f} seconds before retry...")
                        time.sleep(delay)
            
            # All retries failed
            session.final_error = last_error
            self.logger.error(f"{session.operation_name} failed after {session.total_attempts} attempts")
            
            if last_error:
                raise last_error
            else:
                raise RuntimeError(f"Operation {session.operation_name} failed with unknown error")
                
        finally:
            session.end_time = datetime.now()
            self.session_history.append(session)
            if operation_name in self.active_sessions:
                del self.active_sessions[operation_name]
    
    def _calculate_delay(self, attempt_number: int, config: RetryConfiguration) -> float:
        """Calculate delay before next retry attempt."""
        if config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = config.base_delay * (config.backoff_multiplier ** attempt_number)
        elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = config.base_delay * (attempt_number + 1)
        elif config.strategy == RetryStrategy.FIXED_DELAY:
            delay = config.base_delay
        elif config.strategy == RetryStrategy.IMMEDIATE:
            delay = 0.0
        else:
            delay = config.base_delay
        
        # Apply max delay limit
        delay = min(delay, config.max_delay)
        
        # Apply jitter if enabled
        if config.jitter and delay > 0:
            jitter_factor = 0.5 + random.random() * 0.5
            delay *= jitter_factor
        
        return delay
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get statistics about retry sessions."""
        if not self.session_history:
            return {
                'total_sessions': 0,
                'successful_sessions': 0,
                'failed_sessions': 0,
                'success_rate': 0.0,
                'average_attempts': 0.0
            }
        
        total_sessions = len(self.session_history)
        successful_sessions = sum(1 for session in self.session_history if session.successful)
        failed_sessions = total_sessions - successful_sessions
        success_rate = successful_sessions / total_sessions if total_sessions > 0 else 0.0
        
        total_attempts = sum(session.total_attempts for session in self.session_history)
        average_attempts = total_attempts / total_sessions if total_sessions > 0 else 0.0
        
        return {
            'total_sessions': total_sessions,
            'successful_sessions': successful_sessions,
            'failed_sessions': failed_sessions,
            'success_rate': success_rate,
            'average_attempts': average_attempts
        }
    
    def set_global_configuration(self, config: RetryConfiguration) -> None:
        """Set global retry configuration."""
        self.global_config = config
        self.logger.info("Global retry configuration updated")
    
    def get_active_sessions(self) -> Dict[str, RetrySession]:
        """Get currently active retry sessions."""
        return self.active_sessions.copy()
    
    def cancel_session(self, operation_name: str) -> bool:
        """Cancel an active retry session."""
        if operation_name in self.active_sessions:
            session = self.active_sessions[operation_name]
            session.end_time = datetime.now()
            session.final_error = RuntimeError("Session cancelled by user")
            self.session_history.append(session)
            del self.active_sessions[operation_name]
            self.logger.info(f"Cancelled retry session: {operation_name}")
            return True
        return False