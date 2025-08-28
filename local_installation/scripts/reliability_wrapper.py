"""
ReliabilityWrapper Component

This module provides transparent reliability enhancement for existing installation components.
It wraps components to automatically detect and recover from errors, track performance,
and provide comprehensive monitoring without changing the original component interfaces.
"""

import logging
import time
import functools
from typing import Any, Dict, Optional, Callable, List, Type
from datetime import datetime

from interfaces import InstallationError, ErrorCategory
from base_classes import BaseInstallationComponent
from error_handler import ComprehensiveErrorHandler, EnhancedErrorContext
from missing_method_recovery import MissingMethodRecovery


class ReliabilityMetrics:
    """Tracks reliability metrics for wrapped components."""
    
    def __init__(self):
        self.method_calls = {}
        self.success_count = 0
        self.failure_count = 0
        self.total_execution_time = 0.0
        self.error_history = []
        self.recovery_attempts = []
    
    def record_method_call(self, method_name: str, success: bool, duration: float, error: Optional[Exception] = None):
        """Record a method call with its outcome."""
        if method_name not in self.method_calls:
            self.method_calls[method_name] = {
                'total_calls': 0,
                'successful_calls': 0,
                'failed_calls': 0,
                'total_duration': 0.0,
                'average_duration': 0.0,
                'last_called': None,
                'errors': []
            }
        
        stats = self.method_calls[method_name]
        stats['total_calls'] += 1
        stats['total_duration'] += duration
        stats['average_duration'] = stats['total_duration'] / stats['total_calls']
        stats['last_called'] = datetime.now()
        
        if success:
            stats['successful_calls'] += 1
            self.success_count += 1
        else:
            stats['failed_calls'] += 1
            self.failure_count += 1
            if error:
                stats['errors'].append({
                    'timestamp': datetime.now(),
                    'error': str(error),
                    'type': type(error).__name__
                })
                self.error_history.append({
                    'method': method_name,
                    'timestamp': datetime.now(),
                    'error': str(error)
                })
        
        self.total_execution_time += duration
    
    def record_recovery_attempt(self, method_name: str, recovery_type: str, success: bool):
        """Record a recovery attempt."""
        self.recovery_attempts.append({
            'method': method_name,
            'recovery_type': recovery_type,
            'success': success,
            'timestamp': datetime.now()
        })
    
    def get_success_rate(self) -> float:
        """Get overall success rate."""
        total_calls = self.success_count + self.failure_count
        return (self.success_count / total_calls) if total_calls > 0 else 0.0
    
    def get_method_success_rate(self, method_name: str) -> float:
        """Get success rate for a specific method."""
        if method_name not in self.method_calls:
            return 0.0
        
        stats = self.method_calls[method_name]
        total = stats['total_calls']
        return (stats['successful_calls'] / total) if total > 0 else 0.0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            'total_calls': self.success_count + self.failure_count,
            'success_rate': self.get_success_rate(),
            'total_execution_time': self.total_execution_time,
            'average_execution_time': self.total_execution_time / (self.success_count + self.failure_count) if (self.success_count + self.failure_count) > 0 else 0.0,
            'method_stats': self.method_calls,
            'recent_errors': self.error_history[-5:],  # Last 5 errors
            'recovery_attempts': len(self.recovery_attempts),
            'successful_recoveries': len([r for r in self.recovery_attempts if r['success']])
        }


class ReliabilityWrapper:
    """
    Wraps components with reliability enhancements including automatic error recovery,
    performance monitoring, and comprehensive logging.
    """
    
    def __init__(self, component: Any, installation_path: str, 
                 error_handler: Optional[ComprehensiveErrorHandler] = None,
                 missing_method_recovery: Optional[MissingMethodRecovery] = None,
                 logger: Optional[logging.Logger] = None):
        self.component = component
        self.component_name = component.__class__.__name__
        self.installation_path = installation_path
        self.logger = logger or logging.getLogger(f"ReliabilityWrapper.{self.component_name}")
        
        # Initialize reliability systems
        self.error_handler = error_handler or ComprehensiveErrorHandler(installation_path, self.logger)
        self.missing_method_recovery = missing_method_recovery or MissingMethodRecovery(installation_path, self.logger)
        
        # Initialize metrics tracking
        self.metrics = ReliabilityMetrics()
        
        # Track wrapped methods to avoid double-wrapping
        self._wrapped_methods = set()
        
        self.logger.info(f"ReliabilityWrapper initialized for {self.component_name}")
    
    def __getattr__(self, name: str) -> Any:
        """
        Intercept attribute access to wrap methods with reliability enhancements.
        """
        # Get the original attribute
        try:
            original_attr = getattr(self.component, name)
        except AttributeError:
            # Handle missing attribute with recovery system
            self.logger.warning(f"Attribute '{name}' not found in {self.component_name}, attempting recovery")
            try:
                return self.missing_method_recovery.handle_missing_method(self.component, name)
            except Exception as e:
                self.logger.error(f"Failed to recover missing attribute '{name}': {e}")
                raise AttributeError(f"'{self.component_name}' object has no attribute '{name}' and recovery failed")
        
        # If it's not a callable, return as-is
        if not callable(original_attr):
            return original_attr
        
        # If already wrapped, return the wrapped version
        if name in self._wrapped_methods:
            return original_attr
        
        # Wrap the method with reliability enhancements
        wrapped_method = self._wrap_method(original_attr, name)
        
        # Store the wrapped method back on the component to avoid re-wrapping
        setattr(self.component, f"_wrapped_{name}", wrapped_method)
        self._wrapped_methods.add(name)
        
        return wrapped_method
    
    def _wrap_method(self, method: Callable, method_name: str) -> Callable:
        """
        Wrap a method with reliability enhancements.
        """
        @functools.wraps(method)
        def wrapped_method(*args, **kwargs):
            start_time = time.time()
            success = False
            error = None
            result = None
            
            try:
                self.logger.debug(f"Calling {self.component_name}.{method_name}")
                
                # Create enhanced error context for potential errors
                context = self.error_handler.create_enhanced_error_context(
                    component=self.component_name,
                    method=method_name,
                    retry_count=0
                )
                
                # Execute the original method
                result = method(*args, **kwargs)
                success = True
                
                self.logger.debug(f"Successfully completed {self.component_name}.{method_name}")
                return result
                
            except AttributeError as e:
                # Handle missing method errors
                if "has no attribute" in str(e):
                    self.logger.warning(f"Missing method detected: {self.component_name}.{method_name}")
                    try:
                        # Attempt recovery
                        result = self.missing_method_recovery.handle_missing_method(
                            self.component, method_name, *args, **kwargs
                        )
                        success = True
                        self.metrics.record_recovery_attempt(method_name, "missing_method", True)
                        self.logger.info(f"Successfully recovered missing method {self.component_name}.{method_name}")
                        return result
                    except Exception as recovery_error:
                        error = recovery_error
                        self.metrics.record_recovery_attempt(method_name, "missing_method", False)
                        self.logger.error(f"Failed to recover missing method {self.component_name}.{method_name}: {recovery_error}")
                else:
                    error = e
                
            except Exception as e:
                error = e
                self.logger.warning(f"Error in {self.component_name}.{method_name}: {e}")
                
                # Try to handle the error with the comprehensive error handler
                try:
                    if isinstance(e, InstallationError):
                        installation_error = e
                    else:
                        installation_error = InstallationError(
                            str(e), 
                            ErrorCategory.SYSTEM,
                            [f"Check {self.component_name}.{method_name} implementation"]
                        )
                    
                    # Create enhanced context with current call information
                    enhanced_context = self.error_handler.create_enhanced_error_context(
                        component=self.component_name,
                        method=method_name,
                        retry_count=0
                    )
                    
                    recovery_action = self.error_handler.handle_error(installation_error, enhanced_context)
                    
                    # Handle recovery action
                    if recovery_action.value == "retry":
                        self.logger.info(f"Retrying {self.component_name}.{method_name}")
                        try:
                            result = method(*args, **kwargs)
                            success = True
                            self.metrics.record_recovery_attempt(method_name, "retry", True)
                            return result
                        except Exception as retry_error:
                            self.metrics.record_recovery_attempt(method_name, "retry", False)
                            error = retry_error
                    elif recovery_action.value == "continue":
                        self.logger.info(f"Continuing after error in {self.component_name}.{method_name}")
                        success = True  # Mark as success for metrics since we're continuing
                        return None
                    
                except Exception as handler_error:
                    self.logger.error(f"Error handler failed for {self.component_name}.{method_name}: {handler_error}")
                    error = e  # Use original error
            
            finally:
                # Record metrics
                duration = time.time() - start_time
                self.metrics.record_method_call(method_name, success, duration, error)
                
                # Log performance info only for failed operations to avoid overhead
                if not success:
                    self.logger.warning(f"{self.component_name}.{method_name} failed after {duration:.2f}s")
            
            # If we get here, the method failed and couldn't be recovered
            if error:
                raise error
            else:
                raise RuntimeError(f"Unknown error in {self.component_name}.{method_name}")
        
        return wrapped_method
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get reliability metrics for this wrapped component."""
        return {
            'component_name': self.component_name,
            'metrics': self.metrics.get_performance_summary(),
            'wrapped_methods': list(self._wrapped_methods)
        }
    
    def reset_metrics(self):
        """Reset reliability metrics."""
        self.metrics = ReliabilityMetrics()
        self.logger.info(f"Reset metrics for {self.component_name}")
    
    def get_component(self) -> Any:
        """Get the original wrapped component."""
        return self.component


class ReliabilityWrapperFactory:
    """Factory for creating reliability wrappers for different component types."""
    
    def __init__(self, installation_path: str, logger: Optional[logging.Logger] = None):
        self.installation_path = installation_path
        self.logger = logger or logging.getLogger(__name__)
        self.error_handler = ComprehensiveErrorHandler(installation_path, self.logger)
        self.missing_method_recovery = MissingMethodRecovery(installation_path, self.logger)
        self.wrapped_components = {}
    
    def wrap_component(self, component: Any, component_type: Optional[str] = None) -> ReliabilityWrapper:
        """
        Wrap a component with reliability enhancements.
        
        Args:
            component: The component to wrap
            component_type: Optional type hint for the component
            
        Returns:
            ReliabilityWrapper instance
        """
        component_name = component_type or component.__class__.__name__
        
        if id(component) in self.wrapped_components:
            self.logger.debug(f"Component {component_name} already wrapped, returning existing wrapper")
            return self.wrapped_components[id(component)]
        
        wrapper = ReliabilityWrapper(
            component=component,
            installation_path=self.installation_path,
            error_handler=self.error_handler,
            missing_method_recovery=self.missing_method_recovery,
            logger=self.logger
        )
        
        self.wrapped_components[id(component)] = wrapper
        self.logger.info(f"Created reliability wrapper for {component_name}")
        
        return wrapper
    
    def wrap_model_downloader(self, model_downloader: Any) -> ReliabilityWrapper:
        """Wrap a ModelDownloader with specific reliability enhancements."""
        return self.wrap_component(model_downloader, "ModelDownloader")
    
    def wrap_dependency_manager(self, dependency_manager: Any) -> ReliabilityWrapper:
        """Wrap a DependencyManager with specific reliability enhancements."""
        return self.wrap_component(dependency_manager, "DependencyManager")
    
    def wrap_configuration_engine(self, config_engine: Any) -> ReliabilityWrapper:
        """Wrap a ConfigurationEngine with specific reliability enhancements."""
        return self.wrap_component(config_engine, "ConfigurationEngine")
    
    def wrap_installation_validator(self, validator: Any) -> ReliabilityWrapper:
        """Wrap an InstallationValidator with specific reliability enhancements."""
        return self.wrap_component(validator, "InstallationValidator")
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get metrics for all wrapped components."""
        return {
            wrapper_id: wrapper.get_metrics() 
            for wrapper_id, wrapper in self.wrapped_components.items()
        }
    
    def get_component_metrics(self, component: Any) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific component."""
        wrapper = self.wrapped_components.get(id(component))
        return wrapper.get_metrics() if wrapper else None
    
    def reset_all_metrics(self):
        """Reset metrics for all wrapped components."""
        for wrapper in self.wrapped_components.values():
            wrapper.reset_metrics()
        self.logger.info("Reset metrics for all wrapped components")
    
    def get_reliability_summary(self) -> Dict[str, Any]:
        """Get overall reliability summary."""
        all_metrics = self.get_all_metrics()
        
        total_calls = 0
        total_successes = 0
        total_failures = 0
        total_recoveries = 0
        
        for wrapper_metrics in all_metrics.values():
            metrics = wrapper_metrics['metrics']
            total_calls += metrics['total_calls']
            total_successes += metrics['total_calls'] * metrics['success_rate']
            total_failures += metrics['total_calls'] * (1 - metrics['success_rate'])
            total_recoveries += metrics['successful_recoveries']
        
        return {
            'total_components_wrapped': len(self.wrapped_components),
            'total_method_calls': total_calls,
            'overall_success_rate': (total_successes / total_calls) if total_calls > 0 else 0.0,
            'total_failures': int(total_failures),
            'total_recoveries': total_recoveries,
            'recovery_rate': (total_recoveries / total_failures) if total_failures > 0 else 0.0,
            'components': list(all_metrics.keys())
        }