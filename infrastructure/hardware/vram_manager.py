"""
VRAM Manager for WAN22 System Optimization

This module provides comprehensive VRAM detection and management capabilities
for the WAN22 system, supporting multiple detection methods and multi-GPU setups.
"""

import json
import logging
import subprocess
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import threading
from datetime import datetime

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    pynvml = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


@dataclass
class GPUInfo:
    """Information about a detected GPU"""
    index: int
    name: str
    total_memory_mb: int
    driver_version: str
    cuda_version: Optional[str] = None
    temperature: Optional[float] = None
    utilization: Optional[float] = None
    power_usage: Optional[float] = None
    is_available: bool = True


@dataclass
class VRAMUsage:
    """Current VRAM usage information"""
    gpu_index: int
    used_mb: int
    free_mb: int
    total_mb: int
    usage_percent: float
    timestamp: datetime


@dataclass
class VRAMConfig:
    """VRAM configuration settings"""
    manual_vram_gb: Optional[Dict[int, int]] = None
    preferred_gpu: Optional[int] = None
    enable_multi_gpu: bool = False
    memory_fraction: float = 0.9
    enable_memory_growth: bool = True


class VRAMDetectionError(Exception):
    """Exception raised when VRAM detection fails"""
    pass


class VRAMManager:
    """
    Comprehensive VRAM detection and management system
    
    Supports multiple detection methods:
    1. NVIDIA ML (nvml) - Primary method
    2. PyTorch CUDA memory info - Secondary method  
    3. nvidia-smi command parsing - Fallback method
    4. Manual configuration - Last resort
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or "vram_config.json"
        self.config = self._load_config()
        self.detected_gpus: List[GPUInfo] = []
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self._usage_history: Dict[int, List[VRAMUsage]] = {}
        
        # Initialize NVML if available
        self.nvml_initialized = False
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
                self.logger.info("NVML initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize NVML: {e}")
    
    def _load_config(self) -> VRAMConfig:
        """Load VRAM configuration from file"""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                return VRAMConfig(**config_data)
        except Exception as e:
            self.logger.warning(f"Failed to load VRAM config: {e}")
        
        return VRAMConfig()
    
    def _save_config(self) -> None:
        """Save current VRAM configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(asdict(self.config), f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save VRAM config: {e}")
    
    def detect_vram_capacity(self) -> List[GPUInfo]:
        """
        Detect VRAM capacity using multiple methods
        
        Returns:
            List of GPUInfo objects for all detected GPUs
            
        Raises:
            VRAMDetectionError: If all detection methods fail
        """
        self.logger.info("Starting VRAM detection...")
        
        # Try detection methods in order of preference
        detection_methods = [
            ("NVML", self._detect_via_nvml),
            ("PyTorch", self._detect_via_pytorch),
            ("nvidia-smi", self._detect_via_nvidia_smi),
            ("Manual Config", self._detect_via_manual_config)
        ]
        
        for method_name, method in detection_methods:
            try:
                self.logger.info(f"Attempting detection via {method_name}")
                gpus = method()
                if gpus:
                    self.detected_gpus = gpus
                    self.logger.info(f"Successfully detected {len(gpus)} GPU(s) via {method_name}")
                    for gpu in gpus:
                        self.logger.info(f"  GPU {gpu.index}: {gpu.name} - {gpu.total_memory_mb}MB")
                    return gpus
            except Exception as e:
                self.logger.warning(f"{method_name} detection failed: {e}")
                continue
        
        raise VRAMDetectionError("All VRAM detection methods failed")
    
    def _detect_via_nvml(self) -> List[GPUInfo]:
        """Detect GPUs using NVIDIA ML library"""
        if not self.nvml_initialized:
            raise VRAMDetectionError("NVML not available or not initialized")
        
        gpus = []
        device_count = pynvml.nvmlDeviceGetCount()
        
        for i in range(device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Get basic info
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                driver_version = pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
                
                # Get additional info if available
                try:
                    cuda_version = pynvml.nvmlSystemGetCudaDriverVersion()
                    cuda_version = f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}"
                except:
                    cuda_version = None
                
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temperature = None
                
                try:
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = utilization.gpu
                except:
                    gpu_util = None
                
                try:
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                except:
                    power_usage = None
                
                gpu_info = GPUInfo(
                    index=i,
                    name=name,
                    total_memory_mb=memory_info.total // (1024 * 1024),
                    driver_version=driver_version,
                    cuda_version=cuda_version,
                    temperature=temperature,
                    utilization=gpu_util,
                    power_usage=power_usage
                )
                
                gpus.append(gpu_info)
                
            except Exception as e:
                self.logger.warning(f"Failed to get info for GPU {i}: {e}")
                continue
        
        if not gpus:
            raise VRAMDetectionError("No GPUs detected via NVML")
        
        return gpus
    
    def _detect_via_pytorch(self) -> List[GPUInfo]:
        """Detect GPUs using PyTorch CUDA info"""
        if not TORCH_AVAILABLE:
            raise VRAMDetectionError("PyTorch not available")
        
        if not torch.cuda.is_available():
            raise VRAMDetectionError("CUDA not available in PyTorch")
        
        gpus = []
        device_count = torch.cuda.device_count()
        
        for i in range(device_count):
            try:
                # Get device properties
                props = torch.cuda.get_device_properties(i)
                
                # Get memory info
                torch.cuda.set_device(i)
                total_memory = torch.cuda.get_device_properties(i).total_memory
                
                gpu_info = GPUInfo(
                    index=i,
                    name=props.name,
                    total_memory_mb=total_memory // (1024 * 1024),
                    driver_version="Unknown",
                    cuda_version=torch.version.cuda
                )
                
                gpus.append(gpu_info)
                
            except Exception as e:
                self.logger.warning(f"Failed to get PyTorch info for GPU {i}: {e}")
                continue
        
        if not gpus:
            raise VRAMDetectionError("No GPUs detected via PyTorch")
        
        return gpus
    
    def _detect_via_nvidia_smi(self) -> List[GPUInfo]:
        """Detect GPUs using nvidia-smi command"""
        try:
            # Run nvidia-smi with XML output for easier parsing
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=index,name,memory.total,driver_version',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                raise VRAMDetectionError(f"nvidia-smi failed: {result.stderr}")
            
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                
                parts = [part.strip() for part in line.split(',')]
                if len(parts) >= 4:
                    try:
                        gpu_info = GPUInfo(
                            index=int(parts[0]),
                            name=parts[1],
                            total_memory_mb=int(parts[2]),
                            driver_version=parts[3]
                        )
                        gpus.append(gpu_info)
                    except (ValueError, IndexError) as e:
                        self.logger.warning(f"Failed to parse nvidia-smi line '{line}': {e}")
                        continue
            
            if not gpus:
                raise VRAMDetectionError("No GPUs parsed from nvidia-smi output")
            
            return gpus
            
        except subprocess.TimeoutExpired:
            raise VRAMDetectionError("nvidia-smi command timed out")
        except FileNotFoundError:
            raise VRAMDetectionError("nvidia-smi command not found")
    
    def _detect_via_manual_config(self) -> List[GPUInfo]:
        """Use manual VRAM configuration as fallback"""
        if not self.config.manual_vram_gb:
            raise VRAMDetectionError("No manual VRAM configuration available")
        
        gpus = []
        for gpu_index, vram_gb in self.config.manual_vram_gb.items():
            gpu_info = GPUInfo(
                index=gpu_index,
                name=f"Manual GPU {gpu_index}",
                total_memory_mb=vram_gb * 1024,
                driver_version="Manual",
                is_available=True
            )
            gpus.append(gpu_info)
        
        return gpus
    
    def get_available_gpus(self) -> List[GPUInfo]:
        """Get list of available GPUs"""
        if not self.detected_gpus:
            self.detect_vram_capacity()
        
        return [gpu for gpu in self.detected_gpus if gpu.is_available]
    
    def select_optimal_gpu(self) -> Optional[GPUInfo]:
        """Select the optimal GPU for processing"""
        available_gpus = self.get_available_gpus()
        
        if not available_gpus:
            return None
        
        # Use preferred GPU if specified and available
        if self.config.preferred_gpu is not None:
            for gpu in available_gpus:
                if gpu.index == self.config.preferred_gpu:
                    return gpu
        
        # Otherwise select GPU with most VRAM
        return max(available_gpus, key=lambda gpu: gpu.total_memory_mb)
    
    def set_manual_vram_config(self, gpu_vram_mapping: Dict[int, int]) -> None:
        """
        Set manual VRAM configuration
        
        Args:
            gpu_vram_mapping: Dictionary mapping GPU index to VRAM in GB
        """
        self.config.manual_vram_gb = gpu_vram_mapping
        self._save_config()
        self.logger.info(f"Manual VRAM config set: {gpu_vram_mapping}")
    
    def set_preferred_gpu(self, gpu_index: int) -> None:
        """Set preferred GPU for processing"""
        self.config.preferred_gpu = gpu_index
        self._save_config()
        self.logger.info(f"Preferred GPU set to: {gpu_index}")
    
    def enable_multi_gpu(self, enabled: bool = True) -> None:
        """Enable or disable multi-GPU support"""
        self.config.enable_multi_gpu = enabled
        self._save_config()
        self.logger.info(f"Multi-GPU support {'enabled' if enabled else 'disabled'}")
    
    def validate_manual_config(self, gpu_vram_mapping: Dict[int, int]) -> Tuple[bool, List[str]]:
        """
        Validate manual VRAM configuration
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if not gpu_vram_mapping:
            errors.append("GPU VRAM mapping cannot be empty")
            return False, errors
        
        for gpu_index, vram_gb in gpu_vram_mapping.items():
            if not isinstance(gpu_index, int) or gpu_index < 0:
                errors.append(f"Invalid GPU index: {gpu_index}")
            
            if not isinstance(vram_gb, int) or vram_gb <= 0:
                errors.append(f"Invalid VRAM amount for GPU {gpu_index}: {vram_gb}GB")
            
            if vram_gb > 128:  # Reasonable upper limit
                errors.append(f"VRAM amount too high for GPU {gpu_index}: {vram_gb}GB")
        
        return len(errors) == 0, errors
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """Get summary of VRAM detection results"""
        return {
            "total_gpus": len(self.detected_gpus),
            "available_gpus": len(self.get_available_gpus()),
            "gpus": [asdict(gpu) for gpu in self.detected_gpus],
            "config": asdict(self.config),
            "nvml_available": NVML_AVAILABLE and self.nvml_initialized,
            "torch_available": TORCH_AVAILABLE and (torch.cuda.is_available() if TORCH_AVAILABLE else False)
        }

    def get_current_vram_usage(self, gpu_index: Optional[int] = None) -> List[VRAMUsage]:
        """
        Get current VRAM usage for specified GPU or all GPUs
        
        Args:
            gpu_index: Specific GPU index, or None for all GPUs
            
        Returns:
            List of VRAMUsage objects
        """
        usage_list = []
        
        if gpu_index is not None:
            gpus_to_check = [gpu for gpu in self.detected_gpus if gpu.index == gpu_index]
        else:
            gpus_to_check = self.detected_gpus
        
        for gpu in gpus_to_check:
            try:
                usage = self._get_gpu_memory_usage(gpu.index)
                if usage:
                    usage_list.append(usage)
            except Exception as e:
                self.logger.warning(f"Failed to get VRAM usage for GPU {gpu.index}: {e}")
        
        return usage_list
    
    def _get_gpu_memory_usage(self, gpu_index: int) -> Optional[VRAMUsage]:
        """Get memory usage for a specific GPU"""
        # Try NVML first
        if self.nvml_initialized:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                used_mb = memory_info.used // (1024 * 1024)
                free_mb = memory_info.free // (1024 * 1024)
                total_mb = memory_info.total // (1024 * 1024)
                usage_percent = (used_mb / total_mb) * 100 if total_mb > 0 else 0
                
                return VRAMUsage(
                    gpu_index=gpu_index,
                    used_mb=used_mb,
                    free_mb=free_mb,
                    total_mb=total_mb,
                    usage_percent=usage_percent,
                    timestamp=datetime.now()
                )
            except Exception as e:
                self.logger.debug(f"NVML memory query failed for GPU {gpu_index}: {e}")
        
        # Try PyTorch as fallback
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                if gpu_index < torch.cuda.device_count():
                    torch.cuda.set_device(gpu_index)
                    allocated = torch.cuda.memory_allocated(gpu_index) // (1024 * 1024)
                    reserved = torch.cuda.memory_reserved(gpu_index) // (1024 * 1024)
                    total_mb = torch.cuda.get_device_properties(gpu_index).total_memory // (1024 * 1024)
                    
                    used_mb = max(allocated, reserved)
                    free_mb = total_mb - used_mb
                    usage_percent = (used_mb / total_mb) * 100 if total_mb > 0 else 0
                    
                    return VRAMUsage(
                        gpu_index=gpu_index,
                        used_mb=used_mb,
                        free_mb=free_mb,
                        total_mb=total_mb,
                        usage_percent=usage_percent,
                        timestamp=datetime.now()
                    )
            except Exception as e:
                self.logger.debug(f"PyTorch memory query failed for GPU {gpu_index}: {e}")
        
        return None
    
    def start_monitoring(self, interval_seconds: float = 1.0) -> None:
        """Start continuous VRAM monitoring"""
        if self.monitoring_active:
            self.logger.warning("VRAM monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info(f"Started VRAM monitoring with {interval_seconds}s interval")
    
    def stop_monitoring(self) -> None:
        """Stop VRAM monitoring"""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        self.logger.info("Stopped VRAM monitoring")
    
    def _monitoring_loop(self, interval_seconds: float) -> None:
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                current_usage = self.get_current_vram_usage()
                
                for usage in current_usage:
                    # Store usage history
                    if usage.gpu_index not in self._usage_history:
                        self._usage_history[usage.gpu_index] = []
                    
                    self._usage_history[usage.gpu_index].append(usage)
                    
                    # Keep only last 1000 entries per GPU
                    if len(self._usage_history[usage.gpu_index]) > 1000:
                        self._usage_history[usage.gpu_index] = self._usage_history[usage.gpu_index][-1000:]
                    
                    # Check for high usage and trigger optimization if needed
                    if usage.usage_percent > 90.0:
                        self.logger.warning(f"High VRAM usage on GPU {usage.gpu_index}: {usage.usage_percent:.1f}%")
                        self._trigger_memory_optimization(usage.gpu_index)
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in VRAM monitoring loop: {e}")
                time.sleep(interval_seconds)
    
    def _trigger_memory_optimization(self, gpu_index: int) -> None:
        """Trigger memory optimization for high VRAM usage"""
        try:
            self.logger.info(f"Triggering memory optimization for GPU {gpu_index}")
            
            # Clear PyTorch cache if available
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("Cleared PyTorch CUDA cache")
            
            # Additional optimization strategies could be added here
            # For example: reducing batch size, enabling gradient checkpointing, etc.
            
        except Exception as e:
            self.logger.error(f"Failed to optimize memory for GPU {gpu_index}: {e}")
    
    def get_usage_history(self, gpu_index: int, max_entries: int = 100) -> List[VRAMUsage]:
        """Get VRAM usage history for a specific GPU"""
        if gpu_index not in self._usage_history:
            return []
        
        history = self._usage_history[gpu_index]
        return history[-max_entries:] if len(history) > max_entries else history
    
    def get_usage_statistics(self, gpu_index: int) -> Dict[str, float]:
        """Get usage statistics for a specific GPU"""
        history = self.get_usage_history(gpu_index, max_entries=1000)
        
        if not history:
            return {}
        
        usage_percentages = [usage.usage_percent for usage in history]
        
        return {
            "current_usage_percent": usage_percentages[-1] if usage_percentages else 0.0,
            "average_usage_percent": sum(usage_percentages) / len(usage_percentages),
            "max_usage_percent": max(usage_percentages),
            "min_usage_percent": min(usage_percentages),
            "samples_count": len(usage_percentages)
        }
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        self.stop_monitoring()
        
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
                self.logger.info("NVML shutdown completed")
            except Exception as e:
                self.logger.warning(f"Error during NVML shutdown: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except:
            pass


# Utility functions for easy access
def detect_vram() -> List[GPUInfo]:
    """Convenience function to detect VRAM"""
    manager = VRAMManager()
    return manager.detect_vram_capacity()


def get_optimal_gpu() -> Optional[GPUInfo]:
    """Convenience function to get optimal GPU"""
    manager = VRAMManager()
    manager.detect_vram_capacity()
    return manager.select_optimal_gpu()


def get_vram_usage() -> List[VRAMUsage]:
    """Convenience function to get current VRAM usage"""
    manager = VRAMManager()
    manager.detect_vram_capacity()
    return manager.get_current_vram_usage()


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    manager = VRAMManager()
    
    try:
        # Detect GPUs
        gpus = manager.detect_vram_capacity()
        print(f"Detected {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f"  {gpu.name}: {gpu.total_memory_mb}MB")
        
        # Get current usage
        usage = manager.get_current_vram_usage()
        for u in usage:
            print(f"GPU {u.gpu_index}: {u.used_mb}MB used ({u.usage_percent:.1f}%)")
        
        # Get optimal GPU
        optimal = manager.select_optimal_gpu()
        if optimal:
            print(f"Optimal GPU: {optimal.name} (Index {optimal.index})")
        
    except VRAMDetectionError as e:
        print(f"VRAM detection failed: {e}")
    
    finally:
        manager.cleanup()