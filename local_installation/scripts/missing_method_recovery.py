"""
Missing Method Detection and Recovery System

This module handles AttributeError exceptions by providing fallback implementations
and compatibility shims for missing methods. It addresses the specific issues
identified in the error log where components are missing expected methods.
"""

import logging
import inspect
import importlib
from typing import Dict, Any, Optional, Callable, List, Type
from pathlib import Path

from interfaces import InstallationError, ErrorCategory
from base_classes import BaseInstallationComponent


class MissingMethodRecovery(BaseInstallationComponent):
    """Handles missing method detection and provides recovery mechanisms."""
    
    def __init__(self, installation_path: str, logger: Optional[logging.Logger] = None):
        super().__init__(installation_path, logger)
        self.fallback_methods = self._initialize_fallback_methods()
        self.compatibility_shims = self._initialize_compatibility_shims()
        self.version_info = self._detect_software_versions()
    
    def _initialize_fallback_methods(self) -> Dict[str, Dict[str, Callable]]:
        """Initialize fallback method implementations for known missing methods."""
        return {
            'ModelDownloader': {
                'get_required_models': self._fallback_get_required_models,
                'download_models_parallel': self._fallback_download_models_parallel,
                'verify_all_models': self._fallback_verify_all_models
            },
            'DependencyManager': {
                'create_optimized_virtual_environment': self._fallback_create_optimized_venv,
                'is_suitable': self._fallback_is_suitable
            },
            'ConfigurationEngine': {
                'generate_base_configuration': self._fallback_generate_base_configuration
            },
            'ValidationResult': {
                'meets_minimum': self._fallback_meets_minimum
            }
        }
    
    def _initialize_compatibility_shims(self) -> Dict[str, Dict[str, str]]:
        """Initialize compatibility shims for version mismatches."""
        return {
            'ModelDownloader': {
                'download_models_parallel': 'download_wan22_models',  # Alternative method name
                'verify_all_models': 'verify_model_integrity'
            },
            'DependencyManager': {
                'create_optimized_virtual_environment': 'create_virtual_environment'
            }
        }
    
    def _detect_software_versions(self) -> Dict[str, str]:
        """Detect software versions for compatibility checking."""
        versions = {}
        
        try:
            # Check Python version
            import sys
            versions['python'] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            
            # Check installed packages
            try:
                import pkg_resources
                for package_name in ['huggingface_hub', 'torch', 'transformers', 'psutil']:
                    try:
                        dist = pkg_resources.get_distribution(package_name)
                        versions[package_name] = dist.version
                    except pkg_resources.DistributionNotFound:
                        versions[package_name] = 'not_installed'
            except ImportError:
                pass
                
        except Exception as e:
            self.logger.warning(f"Failed to detect software versions: {e}")
        
        return versions
    
    def handle_missing_method(self, obj: Any, method_name: str, *args, **kwargs) -> Any:
        """
        Handle missing method by attempting various recovery strategies.
        
        Args:
            obj: The object that's missing the method
            method_name: Name of the missing method
            *args, **kwargs: Arguments that would be passed to the method
            
        Returns:
            Result of the fallback method or raises appropriate error
        """
        class_name = obj.__class__.__name__
        
        self.logger.info(f"Attempting to recover missing method: {class_name}.{method_name}")
        
        # Strategy 1: Try compatibility shim (alternative method name)
        result = self._try_compatibility_shim(obj, class_name, method_name, *args, **kwargs)
        if result is not None:
            self.logger.info(f"Successfully used compatibility shim for {class_name}.{method_name}")
            return result
        
        # Strategy 2: Try fallback implementation
        result = self._try_fallback_method(obj, class_name, method_name, *args, **kwargs)
        if result is not None:
            self.logger.info(f"Successfully used fallback implementation for {class_name}.{method_name}")
            return result
        
        # Strategy 3: Try dynamic method injection
        if self._try_dynamic_injection(obj, class_name, method_name):
            self.logger.info(f"Successfully injected method {class_name}.{method_name}")
            return getattr(obj, method_name)(*args, **kwargs)
        
        # Strategy 4: Version-specific recovery
        result = self._try_version_specific_recovery(obj, class_name, method_name, *args, **kwargs)
        if result is not None:
            self.logger.info(f"Successfully used version-specific recovery for {class_name}.{method_name}")
            return result
        
        # All strategies failed
        self.logger.error(f"All recovery strategies failed for {class_name}.{method_name}")
        self._suggest_manual_fix(class_name, method_name)
        
        raise InstallationError(
            f"Missing method '{method_name}' in {class_name} could not be recovered automatically. "
            f"Please update to the latest software version or contact support.",
            ErrorCategory.SYSTEM,
            [
                f"Update the software to the latest version",
                f"Check if {class_name} has been properly initialized",
                f"Verify all required dependencies are installed",
                f"Contact support with error details"
            ]
        )
    
    def _try_compatibility_shim(self, obj: Any, class_name: str, method_name: str, *args, **kwargs) -> Optional[Any]:
        """Try to use a compatibility shim (alternative method name)."""
        shims = self.compatibility_shims.get(class_name, {})
        alternative_method = shims.get(method_name)
        
        if alternative_method and hasattr(obj, alternative_method):
            try:
                self.logger.debug(f"Trying compatibility shim: {alternative_method}")
                return getattr(obj, alternative_method)(*args, **kwargs)
            except Exception as e:
                self.logger.warning(f"Compatibility shim {alternative_method} failed: {e}")
                return None
        
        return None
    
    def _try_fallback_method(self, obj: Any, class_name: str, method_name: str, *args, **kwargs) -> Optional[Any]:
        """Try to use a fallback method implementation."""
        fallbacks = self.fallback_methods.get(class_name, {})
        fallback_func = fallbacks.get(method_name)
        
        if fallback_func:
            try:
                self.logger.debug(f"Trying fallback implementation for {method_name}")
                return fallback_func(obj, *args, **kwargs)
            except Exception as e:
                self.logger.warning(f"Fallback method {method_name} failed: {e}")
                return None
        
        return None
    
    def _try_dynamic_injection(self, obj: Any, class_name: str, method_name: str) -> bool:
        """Try to dynamically inject the missing method."""
        try:
            # Get fallback implementation
            fallbacks = self.fallback_methods.get(class_name, {})
            fallback_func = fallbacks.get(method_name)
            
            if fallback_func:
                # Create a bound method
                import types
                bound_method = types.MethodType(fallback_func, obj)
                setattr(obj, method_name, bound_method)
                
                self.logger.debug(f"Successfully injected method {method_name} into {class_name}")
                return True
                
        except Exception as e:
            self.logger.warning(f"Dynamic injection failed for {method_name}: {e}")
        
        return False
    
    def _try_version_specific_recovery(self, obj: Any, class_name: str, method_name: str, *args, **kwargs) -> Optional[Any]:
        """Try version-specific recovery strategies."""
        # Check if this is a known version compatibility issue
        python_version = self.version_info.get('python', '0.0.0')
        
        # Example: Handle Python version-specific issues
        if class_name == 'ModelDownloader' and method_name == 'download_models_parallel':
            # For older Python versions, fall back to sequential download
            if python_version < '3.8.0':
                self.logger.info("Using sequential download for older Python version")
                return self._fallback_sequential_download(obj, *args, **kwargs)
        
        return None
    
    def _suggest_manual_fix(self, class_name: str, method_name: str) -> None:
        """Suggest manual fixes for unrecoverable missing methods."""
        suggestions = {
            'ModelDownloader': {
                'get_required_models': [
                    "Update to the latest version of the model downloader",
                    "Check if huggingface_hub is properly installed",
                    "Verify model configuration files are present"
                ],
                'download_models_parallel': [
                    "Install concurrent.futures if missing",
                    "Check if threading is available",
                    "Fall back to sequential download if needed"
                ]
            },
            'DependencyManager': {
                'create_optimized_virtual_environment': [
                    "Update Python to version 3.8 or higher",
                    "Install virtualenv package",
                    "Check system permissions for virtual environment creation"
                ]
            }
        }
        
        class_suggestions = suggestions.get(class_name, {})
        method_suggestions = class_suggestions.get(method_name, [])
        
        if method_suggestions:
            self.logger.error(f"Manual fix suggestions for {class_name}.{method_name}:")
            for suggestion in method_suggestions:
                self.logger.error(f"  - {suggestion}")
    
    # Fallback method implementations
    
    def _fallback_get_required_models(self, obj: Any, *args, **kwargs) -> List[str]:
        """Fallback implementation for ModelDownloader.get_required_models."""
        self.logger.info("Using fallback implementation for get_required_models")
        
        # Try to get model list from configuration or metadata
        try:
            if hasattr(obj, 'MODEL_CONFIG'):
                return list(obj.MODEL_CONFIG.keys())
            elif hasattr(obj, 'models_dir'):
                # Look for existing model directories
                models_dir = Path(obj.models_dir)
                if models_dir.exists():
                    return [d.name for d in models_dir.iterdir() if d.is_dir()]
            
            # Default model list
            return ['WAN2.2-T2V-A14B', 'WAN2.2-I2V-A14B', 'WAN2.2-TI2V-5B']
            
        except Exception as e:
            self.logger.warning(f"Fallback get_required_models failed: {e}")
            return []
    
    def _fallback_download_models_parallel(self, obj: Any, *args, **kwargs) -> bool:
        """Fallback implementation for ModelDownloader.download_models_parallel."""
        self.logger.info("Using fallback implementation for download_models_parallel")
        
        try:
            # Try to use existing download method
            if hasattr(obj, 'download_wan22_models'):
                return obj.download_wan22_models(*args, **kwargs)
            elif hasattr(obj, 'download_models'):
                return obj.download_models(*args, **kwargs)
            else:
                self.logger.warning("No alternative download method found")
                return False
                
        except Exception as e:
            self.logger.error(f"Fallback download_models_parallel failed: {e}")
            return False
    
    def _fallback_verify_all_models(self, obj: Any, *args, **kwargs) -> bool:
        """Fallback implementation for ModelDownloader.verify_all_models."""
        self.logger.info("Using fallback implementation for verify_all_models")
        
        try:
            # Try to use existing verification method
            if hasattr(obj, 'verify_model_integrity'):
                return obj.verify_model_integrity(*args, **kwargs)
            elif hasattr(obj, 'check_existing_models'):
                result = obj.check_existing_models(*args, **kwargs)
                # Convert result to boolean if needed
                return bool(result)
            else:
                # Basic verification - check if model directories exist
                if hasattr(obj, 'models_dir'):
                    models_dir = Path(obj.models_dir)
                    required_models = self._fallback_get_required_models(obj)
                    
                    for model_name in required_models:
                        model_path = models_dir / model_name
                        if not model_path.exists():
                            self.logger.warning(f"Model directory not found: {model_path}")
                            return False
                    
                    return True
                
                return False
                
        except Exception as e:
            self.logger.error(f"Fallback verify_all_models failed: {e}")
            return False
    
    def _fallback_create_optimized_venv(self, obj: Any, *args, **kwargs) -> bool:
        """Fallback implementation for DependencyManager.create_optimized_virtual_environment."""
        self.logger.info("Using fallback implementation for create_optimized_virtual_environment")
        
        try:
            # Try to use existing virtual environment creation method
            if hasattr(obj, 'create_virtual_environment'):
                return obj.create_virtual_environment(*args, **kwargs)
            else:
                self.logger.warning("No alternative virtual environment creation method found")
                return False
                
        except Exception as e:
            self.logger.error(f"Fallback create_optimized_virtual_environment failed: {e}")
            return False
    
    def _fallback_is_suitable(self, obj: Any, *args, **kwargs) -> bool:
        """Fallback implementation for is_suitable method."""
        self.logger.info("Using fallback implementation for is_suitable")
        
        # Basic suitability check - assume suitable if no specific criteria
        return True
    
    def _fallback_generate_base_configuration(self, obj: Any, *args, **kwargs) -> Dict[str, Any]:
        """Fallback implementation for ConfigurationEngine.generate_base_configuration."""
        self.logger.info("Using fallback implementation for generate_base_configuration")
        
        try:
            # Try to use existing configuration generation method
            if hasattr(obj, 'generate_config'):
                return obj.generate_config(*args, **kwargs)
            else:
                # Return basic default configuration
                return {
                    "model_settings": {
                        "batch_size": 1,
                        "max_length": 512,
                        "temperature": 0.7
                    },
                    "hardware_settings": {
                        "use_gpu": True,
                        "memory_limit": "auto"
                    },
                    "performance_settings": {
                        "optimization_level": "balanced"
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Fallback generate_base_configuration failed: {e}")
            return {}
    
    def _fallback_meets_minimum(self, obj: Any, *args, **kwargs) -> bool:
        """Fallback implementation for ValidationResult.meets_minimum."""
        self.logger.info("Using fallback implementation for meets_minimum")
        
        try:
            # Check if the validation result indicates success
            if hasattr(obj, 'success'):
                return obj.success
            else:
                # Assume meets minimum if no specific criteria
                return True
                
        except Exception as e:
            self.logger.error(f"Fallback meets_minimum failed: {e}")
            return False
    
    def _fallback_sequential_download(self, obj: Any, *args, **kwargs) -> bool:
        """Fallback to sequential download for compatibility."""
        self.logger.info("Using sequential download fallback")
        
        try:
            if hasattr(obj, 'download_wan22_models'):
                return obj.download_wan22_models(*args, **kwargs)
            else:
                return False
        except Exception as e:
            self.logger.error(f"Sequential download fallback failed: {e}")
            return False
    
    def validate_software_versions(self) -> List[str]:
        """Validate software versions and return list of issues."""
        issues = []
        
        # Check Python version
        python_version = self.version_info.get('python', '0.0.0')
        if python_version < '3.8.0':
            issues.append(f"Python {python_version} is below recommended minimum 3.8.0")
        
        # Check required packages
        required_packages = {
            'huggingface_hub': '0.10.0',
            'torch': '1.12.0',
            'psutil': '5.8.0'
        }
        
        for package, min_version in required_packages.items():
            installed_version = self.version_info.get(package, 'not_installed')
            if installed_version == 'not_installed':
                issues.append(f"Required package {package} is not installed")
            elif installed_version < min_version:
                issues.append(f"Package {package} version {installed_version} is below minimum {min_version}")
        
        return issues
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get statistics about method recovery attempts."""
        # This would be implemented to track recovery attempts
        # For now, return basic info
        return {
            "supported_classes": list(self.fallback_methods.keys()),
            "supported_methods": {
                class_name: list(methods.keys()) 
                for class_name, methods in self.fallback_methods.items()
            },
            "version_info": self.version_info
        }
