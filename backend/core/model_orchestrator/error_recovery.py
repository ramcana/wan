"""
Error recovery and retry logic for the Model Orchestrator system.
"""

import asyncio
import logging
import random
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

from .exceptions import ErrorCode, ModelOrchestratorError

T = TypeVar('T')


class RetryStrategy(Enum):
    """Retry strategies for different types of failures."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    NO_RETRY = "no_retry"


class FailureCategory(Enum):
    """Categories of failures for recovery strategy selection."""
    TRANSIENT = "transient"  # Network timeouts, temporary unavailability
    AUTHENTICATION = "authentication"  # Auth failures, expired tokens
    RATE_LIMIT = "rate_limit"  # Rate limiting from upstream
    INTEGRITY = "integrity"  # Checksum failures, corrupted data
    RESOURCE = "resource"  # Disk space, memory issues
    PERMANENT = "permanent"  # Invalid model IDs, missing files
    CONCURRENCY = "concurrency"  # Lock timeouts, concurrent modifications


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    
    # Error-specific overrides
    error_overrides: Dict[ErrorCode, 'RetryConfig'] = field(default_factory=dict)


@dataclass
class RecoveryContext:
    """Context information for error recovery."""
    correlation_id: str
    operation: str
    model_id: Optional[str] = None
    variant: Optional[str] = None
    source_url: Optional[str] = None
    attempt: int = 1
    start_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ErrorRecoveryManager:
    """Manages error recovery strategies and retry logic."""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self.logger = logging.getLogger(__name__)
        
        # Map error codes to failure categories
        self._error_category_map = {
            # Transient errors
            ErrorCode.NETWORK_TIMEOUT: FailureCategory.TRANSIENT,
            ErrorCode.SOURCE_UNAVAILABLE: FailureCategory.TRANSIENT,
            ErrorCode.FILESYSTEM_ERROR: FailureCategory.TRANSIENT,
            
            # Authentication errors
            ErrorCode.AUTH_FAIL: FailureCategory.AUTHENTICATION,
            
            # Rate limiting
            ErrorCode.RATE_LIMIT: FailureCategory.RATE_LIMIT,
            
            # Integrity errors
            ErrorCode.CHECKSUM_FAIL: FailureCategory.INTEGRITY,
            ErrorCode.SIZE_MISMATCH: FailureCategory.INTEGRITY,
            ErrorCode.INCOMPLETE_DOWNLOAD: FailureCategory.INTEGRITY,
            
            # Resource errors
            ErrorCode.NO_SPACE: FailureCategory.RESOURCE,
            ErrorCode.PERMISSION_DENIED: FailureCategory.RESOURCE,
            
            # Permanent errors
            ErrorCode.MODEL_NOT_FOUND: FailureCategory.PERMANENT,
            ErrorCode.VARIANT_NOT_FOUND: FailureCategory.PERMANENT,
            ErrorCode.INVALID_MODEL_ID: FailureCategory.PERMANENT,
            ErrorCode.INVALID_CONFIG: FailureCategory.PERMANENT,
            ErrorCode.MISSING_MANIFEST: FailureCategory.PERMANENT,
            ErrorCode.SCHEMA_VERSION_MISMATCH: FailureCategory.PERMANENT,
            ErrorCode.PATH_TOO_LONG: FailureCategory.PERMANENT,
            
            # Concurrency errors
            ErrorCode.LOCK_TIMEOUT: FailureCategory.CONCURRENCY,
            ErrorCode.LOCK_ERROR: FailureCategory.CONCURRENCY,
            ErrorCode.CONCURRENT_MODIFICATION: FailureCategory.CONCURRENCY,
        }
        
        # Default retry strategies by failure category
        self._category_strategies = {
            FailureCategory.TRANSIENT: RetryConfig(
                max_attempts=5,
                base_delay=1.0,
                max_delay=30.0,
                backoff_factor=2.0,
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF
            ),
            FailureCategory.AUTHENTICATION: RetryConfig(
                max_attempts=2,
                base_delay=5.0,
                strategy=RetryStrategy.FIXED_DELAY
            ),
            FailureCategory.RATE_LIMIT: RetryConfig(
                max_attempts=10,
                base_delay=5.0,
                max_delay=300.0,
                backoff_factor=1.5,
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF
            ),
            FailureCategory.INTEGRITY: RetryConfig(
                max_attempts=3,
                base_delay=2.0,
                strategy=RetryStrategy.LINEAR_BACKOFF
            ),
            FailureCategory.RESOURCE: RetryConfig(
                max_attempts=2,
                base_delay=10.0,
                strategy=RetryStrategy.FIXED_DELAY
            ),
            FailureCategory.PERMANENT: RetryConfig(
                max_attempts=1,
                strategy=RetryStrategy.NO_RETRY
            ),
            FailureCategory.CONCURRENCY: RetryConfig(
                max_attempts=5,
                base_delay=0.5,
                max_delay=10.0,
                backoff_factor=1.5,
                jitter=True,
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF
            ),
        }
    
    def get_failure_category(self, error: Exception) -> FailureCategory:
        """Determine the failure category for an error."""
        if isinstance(error, ModelOrchestratorError):
            return self._error_category_map.get(error.error_code, FailureCategory.PERMANENT)
        
        # Handle standard Python exceptions
        if isinstance(error, (ConnectionError, TimeoutError)):
            return FailureCategory.TRANSIENT
        elif isinstance(error, PermissionError):
            return FailureCategory.RESOURCE
        elif isinstance(error, OSError):
            return FailureCategory.TRANSIENT
        else:
            return FailureCategory.PERMANENT
    
    def get_retry_config(self, error: Exception) -> RetryConfig:
        """Get the retry configuration for a specific error."""
        if isinstance(error, ModelOrchestratorError):
            # Check for error-specific overrides
            if error.error_code in self.config.error_overrides:
                return self.config.error_overrides[error.error_code]
        
        # Use category-based strategy
        category = self.get_failure_category(error)
        return self._category_strategies.get(category, self.config)
    
    def calculate_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate the delay before the next retry attempt."""
        if config.strategy == RetryStrategy.NO_RETRY:
            return 0.0
        elif config.strategy == RetryStrategy.FIXED_DELAY:
            delay = config.base_delay
        elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = config.base_delay * attempt
        elif config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = config.base_delay * (config.backoff_factor ** (attempt - 1))
        else:
            delay = config.base_delay
        
        # Apply maximum delay limit
        delay = min(delay, config.max_delay)
        
        # Add jitter to prevent thundering herd
        if config.jitter:
            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)
        
        return max(0.0, delay)
    
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if an operation should be retried."""
        config = self.get_retry_config(error)
        return attempt < config.max_attempts and config.strategy != RetryStrategy.NO_RETRY
    
    @contextmanager
    def recovery_context(self, operation: str, **kwargs):
        """Create a recovery context for an operation."""
        # Filter kwargs to only include RecoveryContext fields
        context_fields = {'model_id', 'variant', 'source_url', 'metadata'}
        context_kwargs = {k: v for k, v in kwargs.items() if k in context_fields}
        
        # Put any extra kwargs in metadata
        extra_kwargs = {k: v for k, v in kwargs.items() if k not in context_fields}
        if extra_kwargs:
            metadata = context_kwargs.get('metadata', {})
            metadata.update(extra_kwargs)
            context_kwargs['metadata'] = metadata
        
        context = RecoveryContext(
            correlation_id=str(uuid.uuid4()),
            operation=operation,
            **context_kwargs
        )
        
        self.logger.info(
            "Starting operation",
            extra={
                "correlation_id": context.correlation_id,
                "operation": context.operation,
                "model_id": context.model_id,
                "variant": context.variant,
                "source_url": context.source_url,
                **context.metadata
            }
        )
        
        try:
            yield context
        finally:
            duration = time.time() - context.start_time
            self.logger.info(
                "Operation completed",
                extra={
                    "correlation_id": context.correlation_id,
                    "operation": context.operation,
                    "duration_seconds": duration,
                    "attempts": context.attempt
                }
            )
    
    def retry_with_recovery(
        self,
        func: Callable[..., T],
        context: RecoveryContext,
        *args,
        **kwargs
    ) -> T:
        """Execute a function with retry logic and error recovery."""
        last_error = None
        
        while True:
            try:
                self.logger.debug(
                    "Attempting operation",
                    extra={
                        "correlation_id": context.correlation_id,
                        "operation": context.operation,
                        "attempt": context.attempt
                    }
                )
                
                result = func(*args, **kwargs)
                
                if context.attempt > 1:
                    self.logger.info(
                        "Operation succeeded after retry",
                        extra={
                            "correlation_id": context.correlation_id,
                            "operation": context.operation,
                            "attempt": context.attempt,
                            "total_attempts": context.attempt
                        }
                    )
                
                return result
                
            except Exception as error:
                last_error = error
                
                self.logger.warning(
                    "Operation failed",
                    extra={
                        "correlation_id": context.correlation_id,
                        "operation": context.operation,
                        "attempt": context.attempt,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        "error_code": getattr(error, 'error_code', None)
                    },
                    exc_info=True
                )
                
                if not self.should_retry(error, context.attempt):
                    self.logger.error(
                        "Operation failed permanently",
                        extra={
                            "correlation_id": context.correlation_id,
                            "operation": context.operation,
                            "total_attempts": context.attempt,
                            "final_error": str(error)
                        }
                    )
                    raise error
                
                # Apply recovery strategy before retry
                self._apply_recovery_strategy(error, context)
                
                # Calculate and apply delay
                config = self.get_retry_config(error)
                delay = self.calculate_delay(context.attempt, config)
                
                if delay > 0:
                    self.logger.info(
                        "Retrying after delay",
                        extra={
                            "correlation_id": context.correlation_id,
                            "operation": context.operation,
                            "attempt": context.attempt,
                            "delay_seconds": delay,
                            "next_attempt": context.attempt + 1
                        }
                    )
                    time.sleep(delay)
                
                context.attempt += 1
    
    def _apply_recovery_strategy(self, error: Exception, context: RecoveryContext):
        """Apply recovery strategies based on the error type."""
        category = self.get_failure_category(error)
        
        if category == FailureCategory.AUTHENTICATION:
            self._handle_auth_failure(error, context)
        elif category == FailureCategory.RATE_LIMIT:
            self._handle_rate_limit(error, context)
        elif category == FailureCategory.RESOURCE:
            self._handle_resource_failure(error, context)
        elif category == FailureCategory.INTEGRITY:
            self._handle_integrity_failure(error, context)
        elif category == FailureCategory.CONCURRENCY:
            self._handle_concurrency_failure(error, context)
    
    def _handle_auth_failure(self, error: Exception, context: RecoveryContext):
        """Handle authentication failures."""
        self.logger.warning(
            "Authentication failure detected, may need credential refresh",
            extra={
                "correlation_id": context.correlation_id,
                "operation": context.operation,
                "source_url": context.source_url
            }
        )
        # In a real implementation, this might trigger credential refresh
    
    def _handle_rate_limit(self, error: Exception, context: RecoveryContext):
        """Handle rate limiting."""
        self.logger.warning(
            "Rate limit encountered, backing off",
            extra={
                "correlation_id": context.correlation_id,
                "operation": context.operation,
                "source_url": context.source_url
            }
        )
        # Rate limit handling is primarily done through delay calculation
    
    def _handle_resource_failure(self, error: Exception, context: RecoveryContext):
        """Handle resource failures like disk space."""
        if isinstance(error, ModelOrchestratorError) and error.error_code == ErrorCode.NO_SPACE:
            self.logger.warning(
                "Disk space issue detected, consider garbage collection",
                extra={
                    "correlation_id": context.correlation_id,
                    "operation": context.operation,
                    "error_details": error.details
                }
            )
            # In a real implementation, this might trigger garbage collection
    
    def _handle_integrity_failure(self, error: Exception, context: RecoveryContext):
        """Handle integrity failures like checksum mismatches."""
        self.logger.warning(
            "Integrity failure detected, will re-download",
            extra={
                "correlation_id": context.correlation_id,
                "operation": context.operation,
                "error_details": getattr(error, 'details', {})
            }
        )
        # Integrity failures typically require re-downloading the affected files
    
    def _handle_concurrency_failure(self, error: Exception, context: RecoveryContext):
        """Handle concurrency failures like lock timeouts."""
        self.logger.warning(
            "Concurrency issue detected, will retry with jitter",
            extra={
                "correlation_id": context.correlation_id,
                "operation": context.operation,
                "model_id": context.model_id
            }
        )
        # Concurrency handling is primarily done through jittered delays


# Convenience decorators and functions

def with_retry(
    operation: str,
    config: Optional[RetryConfig] = None,
    **context_kwargs
):
    """Decorator to add retry logic to a function."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        def wrapper(*args, **kwargs) -> T:
            manager = ErrorRecoveryManager(config)
            with manager.recovery_context(operation, **context_kwargs) as context:
                return manager.retry_with_recovery(func, context, *args, **kwargs)
        return wrapper
    return decorator


def retry_operation(
    func: Callable[..., T],
    operation: str,
    config: Optional[RetryConfig] = None,
    *args,
    **kwargs
) -> T:
    """Execute a function with retry logic."""
    manager = ErrorRecoveryManager(config)
    
    # Separate context kwargs from function kwargs
    context_fields = {'model_id', 'variant', 'source_url', 'metadata'}
    context_kwargs = {k: v for k, v in kwargs.items() if k in context_fields}
    func_kwargs = {k: v for k, v in kwargs.items() if k not in context_fields}
    
    with manager.recovery_context(operation, **context_kwargs) as context:
        return manager.retry_with_recovery(func, context, *args, **func_kwargs)