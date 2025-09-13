"""
VRAM Configuration Manager for WAN22 System

Provides comprehensive VRAM configuration management including fallback systems,
persistent storage, and GPU selection interfaces.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import shutil

from vram_manager import VRAMManager, GPUInfo, VRAMConfig, VRAMDetectionError


@dataclass
class VRAMConfigProfile:
    """VRAM configuration profile"""
    name: str
    description: str
    manual_vram_gb: Dict[int, int]
    preferred_gpu: Optional[int] = None
    enable_multi_gpu: bool = False
    memory_fraction: float = 0.9
    enable_memory_growth: bool = True
    created_at: Optional[str] = None
    last_used: Optional[str] = None


@dataclass
class GPUSelectionCriteria:
    """Criteria for GPU selection"""
    min_vram_gb: Optional[int] = None
    max_vram_gb: Optional[int] = None
    preferred_models: Optional[List[str]] = None
    exclude_models: Optional[List[str]] = None
    require_cuda: bool = True
    min_compute_capability: Optional[Tuple[int, int]] = None


class VRAMConfigManager:
    """
    Advanced VRAM configuration management system
    
    Features:
    - Manual VRAM specification for detection failures
    - Validation system for manual VRAM settings
    - Persistent storage for VRAM configuration preferences
    - GPU selection interface for multi-GPU systems
    - Configuration profiles for different use cases
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.config_dir = Path(config_dir or "vram_configs")
        self.config_dir.mkdir(exist_ok=True)
        
        self.main_config_path = self.config_dir / "vram_config.json"
        self.profiles_path = self.config_dir / "profiles.json"
        self.backup_dir = self.config_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        self.vram_manager = VRAMManager(str(self.main_config_path))
        self.profiles: Dict[str, VRAMConfigProfile] = {}
        self.current_profile: Optional[str] = None
        
        self._load_profiles()
    
    def _load_profiles(self) -> None:
        """Load configuration profiles from file"""
        try:
            if self.profiles_path.exists():
                with open(self.profiles_path, 'r') as f:
                    profiles_data = json.load(f)
                
                self.profiles = {}
                for name, data in profiles_data.get('profiles', {}).items():
                    self.profiles[name] = VRAMConfigProfile(**data)
                
                self.current_profile = profiles_data.get('current_profile')
                self.logger.info(f"Loaded {len(self.profiles)} VRAM configuration profiles")
        except Exception as e:
            self.logger.warning(f"Failed to load VRAM profiles: {e}")
            self.profiles = {}
    
    def _save_profiles(self) -> None:
        """Save configuration profiles to file"""
        try:
            # Create backup first
            if self.profiles_path.exists():
                backup_path = self.backup_dir / f"profiles_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                shutil.copy2(self.profiles_path, backup_path)
            
            profiles_data = {
                'profiles': {name: asdict(profile) for name, profile in self.profiles.items()},
                'current_profile': self.current_profile,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.profiles_path, 'w') as f:
                json.dump(profiles_data, f, indent=2)
            
            self.logger.info("VRAM profiles saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save VRAM profiles: {e}")
    
    def create_manual_vram_config(self, gpu_vram_mapping: Dict[int, int], 
                                 profile_name: str = "manual_config",
                                 description: str = "Manual VRAM configuration") -> Tuple[bool, List[str]]:
        """
        Create manual VRAM configuration for detection failures
        
        Args:
            gpu_vram_mapping: Dictionary mapping GPU index to VRAM in GB
            profile_name: Name for the configuration profile
            description: Description of the configuration
            
        Returns:
            Tuple of (success, list_of_errors)
        """
        # Validate the manual configuration
        is_valid, errors = self.validate_manual_vram_config(gpu_vram_mapping)
        if not is_valid:
            return False, errors
        
        try:
            # Create configuration profile
            profile = VRAMConfigProfile(
                name=profile_name,
                description=description,
                manual_vram_gb=gpu_vram_mapping.copy(),
                created_at=datetime.now().isoformat()
            )
            
            # Save profile
            self.profiles[profile_name] = profile
            self._save_profiles()
            
            # Apply to VRAM manager
            self.vram_manager.set_manual_vram_config(gpu_vram_mapping)
            
            self.logger.info(f"Created manual VRAM config '{profile_name}': {gpu_vram_mapping}")
            return True, []
            
        except Exception as e:
            error_msg = f"Failed to create manual VRAM config: {e}"
            self.logger.error(error_msg)
            return False, [error_msg]
    
    def validate_manual_vram_config(self, gpu_vram_mapping: Dict[int, int]) -> Tuple[bool, List[str]]:
        """
        Validate manual VRAM configuration
        
        Args:
            gpu_vram_mapping: Dictionary mapping GPU index to VRAM in GB
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if not gpu_vram_mapping:
            errors.append("GPU VRAM mapping cannot be empty")
            return False, errors
        
        # Validate GPU indices and VRAM amounts
        for gpu_index, vram_gb in gpu_vram_mapping.items():
            # Validate GPU index
            if not isinstance(gpu_index, int):
                errors.append(f"GPU index must be integer, got {type(gpu_index).__name__}: {gpu_index}")
                continue
            
            if gpu_index < 0:
                errors.append(f"GPU index cannot be negative: {gpu_index}")
            
            if gpu_index > 15:  # Reasonable upper limit for GPU count
                errors.append(f"GPU index too high (max 15): {gpu_index}")
            
            # Validate VRAM amount
            if not isinstance(vram_gb, (int, float)):
                errors.append(f"VRAM amount must be numeric for GPU {gpu_index}, got {type(vram_gb).__name__}: {vram_gb}")
                continue
            
            if vram_gb <= 0:
                errors.append(f"VRAM amount must be positive for GPU {gpu_index}: {vram_gb}GB")
            
            if vram_gb < 1:
                errors.append(f"VRAM amount too low for GPU {gpu_index}: {vram_gb}GB (minimum 1GB)")
            
            if vram_gb > 128:  # Reasonable upper limit for current GPUs
                errors.append(f"VRAM amount too high for GPU {gpu_index}: {vram_gb}GB (maximum 128GB)")
        
        # Check for duplicate GPU indices
        if len(set(gpu_vram_mapping.keys())) != len(gpu_vram_mapping):
            errors.append("Duplicate GPU indices found in mapping")
        
        return len(errors) == 0, errors
    
    def get_gpu_selection_interface(self, criteria: Optional[GPUSelectionCriteria] = None) -> Dict[str, Any]:
        """
        Get GPU selection interface data for multi-GPU systems
        
        Args:
            criteria: Optional selection criteria to filter GPUs
            
        Returns:
            Dictionary containing GPU selection interface data
        """
        try:
            # Detect available GPUs
            detected_gpus = self.vram_manager.detect_vram_capacity()
            
            # Apply selection criteria if provided
            if criteria:
                detected_gpus = self._filter_gpus_by_criteria(detected_gpus, criteria)
            
            # Get current VRAM usage for each GPU
            gpu_usage = {}
            try:
                usage_list = self.vram_manager.get_current_vram_usage()
                for usage in usage_list:
                    gpu_usage[usage.gpu_index] = {
                        'used_mb': usage.used_mb,
                        'free_mb': usage.free_mb,
                        'usage_percent': usage.usage_percent
                    }
            except Exception as e:
                self.logger.warning(f"Failed to get VRAM usage: {e}")
            
            # Build interface data
            interface_data = {
                'available_gpus': [],
                'current_selection': self.vram_manager.config.preferred_gpu,
                'multi_gpu_enabled': self.vram_manager.config.enable_multi_gpu,
                'selection_criteria': asdict(criteria) if criteria else None,
                'recommendations': []
            }
            
            for gpu in detected_gpus:
                gpu_data = {
                    'index': gpu.index,
                    'name': gpu.name,
                    'total_memory_mb': gpu.total_memory_mb,
                    'total_memory_gb': gpu.total_memory_mb // 1024,
                    'driver_version': gpu.driver_version,
                    'cuda_version': gpu.cuda_version,
                    'temperature': gpu.temperature,
                    'utilization': gpu.utilization,
                    'power_usage': gpu.power_usage,
                    'is_available': gpu.is_available,
                    'current_usage': gpu_usage.get(gpu.index, {}),
                    'suitability_score': self._calculate_gpu_suitability(gpu, criteria)
                }
                interface_data['available_gpus'].append(gpu_data)
            
            # Sort by suitability score
            interface_data['available_gpus'].sort(key=lambda x: x['suitability_score'], reverse=True)
            
            # Generate recommendations
            interface_data['recommendations'] = self._generate_gpu_recommendations(detected_gpus, criteria)
            
            return interface_data
            
        except Exception as e:
            self.logger.error(f"Failed to get GPU selection interface: {e}")
            return {
                'available_gpus': [],
                'current_selection': None,
                'multi_gpu_enabled': False,
                'error': str(e)
            }
    
    def _filter_gpus_by_criteria(self, gpus: List[GPUInfo], criteria: GPUSelectionCriteria) -> List[GPUInfo]:
        """Filter GPUs based on selection criteria"""
        filtered_gpus = []
        
        for gpu in gpus:
            # Check VRAM requirements
            gpu_vram_gb = gpu.total_memory_mb // 1024
            
            if criteria.min_vram_gb and gpu_vram_gb < criteria.min_vram_gb:
                continue
            
            if criteria.max_vram_gb and gpu_vram_gb > criteria.max_vram_gb:
                continue
            
            # Check model preferences
            if criteria.preferred_models:
                if not any(model.lower() in gpu.name.lower() for model in criteria.preferred_models):
                    continue
            
            if criteria.exclude_models:
                if any(model.lower() in gpu.name.lower() for model in criteria.exclude_models):
                    continue
            
            # Check CUDA requirement
            if criteria.require_cuda and not gpu.cuda_version:
                continue
            
            filtered_gpus.append(gpu)
        
        return filtered_gpus
    
    def _calculate_gpu_suitability(self, gpu: GPUInfo, criteria: Optional[GPUSelectionCriteria]) -> float:
        """Calculate suitability score for a GPU (0-100)"""
        score = 50.0  # Base score
        
        # VRAM score (higher is better)
        vram_gb = gpu.total_memory_mb // 1024
        score += min(vram_gb * 2, 30)  # Up to 30 points for VRAM
        
        # Availability score
        if gpu.is_available:
            score += 10
        else:
            score -= 20
        
        # Temperature score (lower is better)
        if gpu.temperature:
            if gpu.temperature < 60:
                score += 5
            elif gpu.temperature > 80:
                score -= 10
        
        # Utilization score (lower current utilization is better for new tasks)
        if gpu.utilization:
            score += max(0, (100 - gpu.utilization) * 0.1)
        
        # Criteria-based scoring
        if criteria:
            if criteria.min_vram_gb and vram_gb >= criteria.min_vram_gb:
                score += 5
            
            if criteria.preferred_models:
                if any(model.lower() in gpu.name.lower() for model in criteria.preferred_models):
                    score += 15
        
        return min(100.0, max(0.0, score))
    
    def _generate_gpu_recommendations(self, gpus: List[GPUInfo], 
                                    criteria: Optional[GPUSelectionCriteria]) -> List[Dict[str, Any]]:
        """Generate GPU selection recommendations"""
        recommendations = []
        
        if not gpus:
            recommendations.append({
                'type': 'error',
                'message': 'No GPUs detected. Consider using manual VRAM configuration.',
                'action': 'create_manual_config'
            })
            return recommendations
        
        # Find best GPU
        best_gpu = max(gpus, key=lambda g: self._calculate_gpu_suitability(g, criteria))
        recommendations.append({
            'type': 'primary',
            'message': f'Recommended primary GPU: {best_gpu.name} (Index {best_gpu.index})',
            'gpu_index': best_gpu.index,
            'action': 'select_gpu'
        })
        
        # Multi-GPU recommendations
        if len(gpus) > 1:
            total_vram = sum(gpu.total_memory_mb for gpu in gpus) // 1024
            recommendations.append({
                'type': 'multi_gpu',
                'message': f'Multi-GPU setup available: {len(gpus)} GPUs with {total_vram}GB total VRAM',
                'gpu_count': len(gpus),
                'total_vram_gb': total_vram,
                'action': 'enable_multi_gpu'
            })
        
        # VRAM warnings
        for gpu in gpus:
            vram_gb = gpu.total_memory_mb // 1024
            if vram_gb < 8:
                recommendations.append({
                    'type': 'warning',
                    'message': f'GPU {gpu.index} ({gpu.name}) has low VRAM: {vram_gb}GB',
                    'gpu_index': gpu.index,
                    'action': 'consider_upgrade'
                })
        
        return recommendations
    
    def select_gpu(self, gpu_index: int) -> Tuple[bool, str]:
        """
        Select preferred GPU for processing
        
        Args:
            gpu_index: Index of GPU to select
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Validate GPU exists
            available_gpus = self.vram_manager.get_available_gpus()
            gpu_indices = [gpu.index for gpu in available_gpus]
            
            if gpu_index not in gpu_indices:
                return False, f"GPU {gpu_index} not found in available GPUs: {gpu_indices}"
            
            # Set preferred GPU
            self.vram_manager.set_preferred_gpu(gpu_index)
            
            # Update current profile if exists
            if self.current_profile and self.current_profile in self.profiles:
                self.profiles[self.current_profile].preferred_gpu = gpu_index
                self.profiles[self.current_profile].last_used = datetime.now().isoformat()
                self._save_profiles()
            
            selected_gpu = next(gpu for gpu in available_gpus if gpu.index == gpu_index)
            message = f"Selected GPU {gpu_index}: {selected_gpu.name}"
            self.logger.info(message)
            
            return True, message
            
        except Exception as e:
            error_msg = f"Failed to select GPU {gpu_index}: {e}"
            self.logger.error(error_msg)
            return False, error_msg
    
    def enable_multi_gpu_support(self, enabled: bool = True) -> Tuple[bool, str]:
        """
        Enable or disable multi-GPU support
        
        Args:
            enabled: Whether to enable multi-GPU support
            
        Returns:
            Tuple of (success, message)
        """
        try:
            self.vram_manager.enable_multi_gpu(enabled)
            
            # Update current profile if exists
            if self.current_profile and self.current_profile in self.profiles:
                self.profiles[self.current_profile].enable_multi_gpu = enabled
                self.profiles[self.current_profile].last_used = datetime.now().isoformat()
                self._save_profiles()
            
            status = "enabled" if enabled else "disabled"
            message = f"Multi-GPU support {status}"
            self.logger.info(message)
            
            return True, message
            
        except Exception as e:
            error_msg = f"Failed to {'enable' if enabled else 'disable'} multi-GPU support: {e}"
            self.logger.error(error_msg)
            return False, error_msg
    
    def create_profile(self, name: str, description: str, 
                      gpu_vram_mapping: Optional[Dict[int, int]] = None,
                      preferred_gpu: Optional[int] = None,
                      enable_multi_gpu: bool = False) -> Tuple[bool, str]:
        """
        Create a new VRAM configuration profile
        
        Args:
            name: Profile name
            description: Profile description
            gpu_vram_mapping: Manual VRAM mapping (optional)
            preferred_gpu: Preferred GPU index (optional)
            enable_multi_gpu: Enable multi-GPU support
            
        Returns:
            Tuple of (success, message)
        """
        try:
            if name in self.profiles:
                return False, f"Profile '{name}' already exists"
            
            # Validate manual VRAM config if provided
            if gpu_vram_mapping:
                is_valid, errors = self.validate_manual_vram_config(gpu_vram_mapping)
                if not is_valid:
                    return False, f"Invalid VRAM configuration: {'; '.join(errors)}"
            
            profile = VRAMConfigProfile(
                name=name,
                description=description,
                manual_vram_gb=gpu_vram_mapping or {},
                preferred_gpu=preferred_gpu,
                enable_multi_gpu=enable_multi_gpu,
                created_at=datetime.now().isoformat()
            )
            
            self.profiles[name] = profile
            self._save_profiles()
            
            message = f"Created VRAM profile '{name}'"
            self.logger.info(message)
            return True, message
            
        except Exception as e:
            error_msg = f"Failed to create profile '{name}': {e}"
            self.logger.error(error_msg)
            return False, error_msg
    
    def load_profile(self, name: str) -> Tuple[bool, str]:
        """
        Load a VRAM configuration profile
        
        Args:
            name: Profile name to load
            
        Returns:
            Tuple of (success, message)
        """
        try:
            if name not in self.profiles:
                return False, f"Profile '{name}' not found"
            
            profile = self.profiles[name]
            
            # Apply profile settings to VRAM manager
            if profile.manual_vram_gb:
                self.vram_manager.set_manual_vram_config(profile.manual_vram_gb)
            
            if profile.preferred_gpu is not None:
                self.vram_manager.set_preferred_gpu(profile.preferred_gpu)
            
            self.vram_manager.enable_multi_gpu(profile.enable_multi_gpu)
            
            # Update profile usage
            profile.last_used = datetime.now().isoformat()
            self.current_profile = name
            self._save_profiles()
            
            message = f"Loaded VRAM profile '{name}'"
            self.logger.info(message)
            return True, message
            
        except Exception as e:
            error_msg = f"Failed to load profile '{name}': {e}"
            self.logger.error(error_msg)
            return False, error_msg
    
    def delete_profile(self, name: str) -> Tuple[bool, str]:
        """
        Delete a VRAM configuration profile
        
        Args:
            name: Profile name to delete
            
        Returns:
            Tuple of (success, message)
        """
        try:
            if name not in self.profiles:
                return False, f"Profile '{name}' not found"
            
            del self.profiles[name]
            
            if self.current_profile == name:
                self.current_profile = None
            
            self._save_profiles()
            
            message = f"Deleted VRAM profile '{name}'"
            self.logger.info(message)
            return True, message
            
        except Exception as e:
            error_msg = f"Failed to delete profile '{name}': {e}"
            self.logger.error(error_msg)
            return False, error_msg
    
    def list_profiles(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available VRAM configuration profiles
        
        Returns:
            Dictionary of profile names to profile information
        """
        profiles_info = {}
        
        for name, profile in self.profiles.items():
            profiles_info[name] = {
                'name': profile.name,
                'description': profile.description,
                'gpu_count': len(profile.manual_vram_gb),
                'total_vram_gb': sum(profile.manual_vram_gb.values()) if profile.manual_vram_gb else 0,
                'preferred_gpu': profile.preferred_gpu,
                'multi_gpu_enabled': profile.enable_multi_gpu,
                'created_at': profile.created_at,
                'last_used': profile.last_used,
                'is_current': name == self.current_profile
            }
        
        return profiles_info
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status including VRAM configuration
        
        Returns:
            Dictionary containing system status information
        """
        try:
            status = {
                'detection_summary': self.vram_manager.get_detection_summary(),
                'current_profile': self.current_profile,
                'profiles_count': len(self.profiles),
                'config_valid': True,
                'last_detection': datetime.now().isoformat()
            }
            
            # Add current VRAM usage if available
            try:
                usage = self.vram_manager.get_current_vram_usage()
                status['current_usage'] = [asdict(u) for u in usage]
            except Exception as e:
                status['usage_error'] = str(e)
            
            # Add GPU selection interface data
            try:
                interface_data = self.get_gpu_selection_interface()
                status['gpu_selection'] = interface_data
            except Exception as e:
                status['selection_error'] = str(e)
            
            return status
            
        except Exception as e:
            return {
                'error': str(e),
                'config_valid': False,
                'last_detection': datetime.now().isoformat()
            }

    def get_fallback_config_options(self) -> Dict[str, Any]:
        """
        Get fallback configuration options for when detection fails
        
        Returns:
            Dictionary containing fallback options and guidance
        """
        return {
            'common_gpu_configs': {
                'RTX 4090': {'0': 24},
                'RTX 4080': {'0': 16},
                'RTX 4070 Ti': {'0': 12},
                'RTX 4070': {'0': 12},
                'RTX 4060 Ti': {'0': 16},
                'RTX 4060': {'0': 8},
                'RTX 3090': {'0': 24},
                'RTX 3080': {'0': 10},
                'RTX 3070': {'0': 8},
                'RTX 3060': {'0': 12},
                'GTX 1660 Ti': {'0': 6},
                'GTX 1660': {'0': 6}
            },
            'multi_gpu_examples': {
                '2x RTX 4080': {'0': 16, '1': 16},
                '2x RTX 3090': {'0': 24, '1': 24},
                '4x RTX 4090': {'0': 24, '1': 24, '2': 24, '3': 24}
            },
            'validation_rules': {
                'min_vram_gb': 1,
                'max_vram_gb': 128,
                'max_gpu_count': 16,
                'recommended_min_vram': 8
            },
            'troubleshooting': {
                'detection_failed': 'Try running as administrator or check GPU drivers',
                'low_vram': 'Consider enabling memory optimization or using smaller models',
                'multi_gpu_issues': 'Ensure all GPUs have sufficient power and cooling'
            }
        }
    
    def export_config(self, export_path: str) -> Tuple[bool, str]:
        """
        Export VRAM configuration to file
        
        Args:
            export_path: Path to export configuration file
            
        Returns:
            Tuple of (success, message)
        """
        try:
            export_data = {
                'profiles': {name: asdict(profile) for name, profile in self.profiles.items()},
                'current_profile': self.current_profile,
                'exported_at': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            # Ensure export directory exists
            export_file = Path(export_path)
            export_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            message = f"Configuration exported to {export_path}"
            self.logger.info(message)
            return True, message
            
        except Exception as e:
            error_msg = f"Failed to export configuration: {e}"
            self.logger.error(error_msg)
            return False, error_msg
    
    def import_config(self, import_path: str, merge: bool = True) -> Tuple[bool, str]:
        """
        Import VRAM configuration from file
        
        Args:
            import_path: Path to import configuration file
            merge: Whether to merge with existing profiles or replace them
            
        Returns:
            Tuple of (success, message)
        """
        try:
            if not Path(import_path).exists():
                return False, f"Import file not found: {import_path}"
            
            with open(import_path, 'r') as f:
                import_data = json.load(f)
            
            if 'profiles' not in import_data:
                return False, "Invalid configuration file: missing 'profiles' section"
            
            # Create backup before import
            if self.profiles:
                backup_path = self.backup_dir / f"pre_import_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                self.export_config(str(backup_path))
            
            imported_count = 0
            
            if not merge:
                # Replace all profiles
                self.profiles = {}
                self.current_profile = None
            
            # Import profiles
            for name, profile_data in import_data['profiles'].items():
                try:
                    profile = VRAMConfigProfile(**profile_data)
                    self.profiles[name] = profile
                    imported_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to import profile '{name}': {e}")
                    continue
            
            # Set current profile if specified and valid
            imported_current = import_data.get('current_profile')
            if imported_current and imported_current in self.profiles:
                self.current_profile = imported_current
            
            # Save imported configuration
            self._save_profiles()
            
            message = f"Successfully imported {imported_count} profile(s) from {import_path}"
            self.logger.info(message)
            return True, message
            
        except Exception as e:
            error_msg = f"Failed to import configuration: {e}"
            self.logger.error(error_msg)
            return False, error_msg
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        if self.vram_manager:
            self.vram_manager.cleanup()


# Utility functions for easy access
def create_fallback_config(gpu_vram_mapping: Dict[int, int]) -> Tuple[bool, List[str]]:
    """Convenience function to create fallback VRAM configuration"""
    manager = VRAMConfigManager()
    return manager.create_manual_vram_config(gpu_vram_mapping)


def get_gpu_selection_ui() -> Dict[str, Any]:
    """Convenience function to get GPU selection interface"""
    manager = VRAMConfigManager()
    return manager.get_gpu_selection_interface()


def validate_vram_config(gpu_vram_mapping: Dict[int, int]) -> Tuple[bool, List[str]]:
    """Convenience function to validate VRAM configuration"""
    manager = VRAMConfigManager()
    return manager.validate_manual_vram_config(gpu_vram_mapping)


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    manager = VRAMConfigManager()
    
    try:
        # Get system status
        status = manager.get_system_status()
        print("System Status:")
        print(json.dumps(status, indent=2, default=str))
        
        # Get GPU selection interface
        interface = manager.get_gpu_selection_interface()
        print("\nGPU Selection Interface:")
        for gpu in interface['available_gpus']:
            print(f"  GPU {gpu['index']}: {gpu['name']} - {gpu['total_memory_gb']}GB (Score: {gpu['suitability_score']:.1f})")
        
        # Show fallback options
        fallback = manager.get_fallback_config_options()
        print("\nCommon GPU Configurations:")
        for gpu_name, config in fallback['common_gpu_configs'].items():
            print(f"  {gpu_name}: {config}")
        
    except Exception as e:
        print(f"Demo failed: {e}")
    
    finally:
        manager.cleanup()
