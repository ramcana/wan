"""
VRAM Optimization and Resource Management for Wan2.2 Video Generation
Implements proactive VRAM checking, automatic parameter optimization, and memory cleanup strategies
"""

import logging
import torch
import psutil
import gc
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Union
from enum import Enum
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)

class ResourceStatus(Enum):
    """Resource availability status"""
    OPTIMAL = "optimal"
    AVAILABLE = "available"
    LIMITED = "limited"
    INSUFFICIENT = "insufficient"
    CRITICAL = "critical"

class OptimizationLevel(Enum):
    """Optimization levels for resource management"""
    NONE = "none"
    BASIC = "basic"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"

@dataclass
class VRAMInfo:
    """VRAM usage information"""
    total_mb: float
    allocated_mb: float
    cached_mb: float
    free_mb: float
    utilization_percent: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_mb": self.total_mb,
            "allocated_mb": self.allocated_mb,
            "cached_mb": self.cached_mb,
            "free_mb": self.free_mb,
            "utilization_percent": self.utilization_percent
        }

@dataclass
class SystemResourceInfo:
    """System resource information"""
    vram: VRAMInfo
    ram_total_gb: float
    ram_available_gb: float
    ram_usage_percent: float
    cpu_usage_percent: float
    disk_free_gb: float
    gpu_temperature: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "vram": self.vram.to_dict(),
            "ram_total_gb": self.ram_total_gb,
            "ram_available_gb": self.ram_available_gb,
            "ram_usage_percent": self.ram_usage_percent,
            "cpu_usage_percent": self.cpu_usage_percent,
            "disk_free_gb": self.disk_free_gb,
            "gpu_temperature": self.gpu_temperature
        }

@dataclass
class ResourceRequirement:
    """Resource requirements for a specific generation task"""
    model_type: str
    resolution: str
    steps: int
    duration: int
    vram_mb: float
    ram_mb: float
    estimated_time_seconds: float
    optimization_level: OptimizationLevel = OptimizationLevel.BASIC
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "resolution": self.resolution,
            "steps": self.steps,
            "duration": self.duration,
            "vram_mb": self.vram_mb,
            "ram_mb": self.ram_mb,
            "estimated_time_seconds": self.estimated_time_seconds,
            "optimization_level": self.optimization_level.value
        }

@dataclass
class OptimizationSuggestion:
    """Optimization suggestion for resource management"""
    parameter: str
    current_value: Any
    suggested_value: Any
    reason: str
    impact: str
    vram_savings_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter": self.parameter,
            "current_value": self.current_value,
            "suggested_value": self.suggested_value,
            "reason": self.reason,
            "impact": self.impact,
            "vram_savings_mb": self.vram_savings_mb
        }

class VRAMOptimizer:
    """Handles VRAM optimization and memory management"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.optimization_config = config.get("optimization", {})
        self.max_vram_usage_percent = self.optimization_config.get("max_vram_usage_percent", 90)
        self.memory_safety_margin_mb = self.optimization_config.get("memory_safety_margin_mb", 1024)
        
        # Resource monitoring
        self.resource_history: List[SystemResourceInfo] = []
        self.monitoring_enabled = True
        self.monitoring_thread = None
        
        self._initialize_gpu_info()
        self._start_resource_monitoring()
    
    def _initialize_gpu_info(self):
        """Initialize GPU information"""
        try:
            self.gpu_available = torch.cuda.is_available()
            if self.gpu_available:
                self.gpu_count = torch.cuda.device_count()
                self.total_vram = torch.cuda.get_device_properties(0).total_memory
                self.gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"GPU initialized: {self.gpu_name}, VRAM: {self.total_vram/1024**3:.1f}GB")
            else:
                self.gpu_count = 0
                self.total_vram = 0
                self.gpu_name = "No GPU"
                logger.warning("No GPU available for VRAM optimization")
        except Exception as e:
            logger.error(f"GPU initialization failed: {e}")
            self.gpu_available = False
            self.gpu_count = 0
            self.total_vram = 0
            self.gpu_name = "Error"
    
    def get_vram_info(self) -> VRAMInfo:
        """Get current VRAM usage information"""
        if not self.gpu_available:
            return VRAMInfo(0, 0, 0, 0, 0)
        
        try:
            allocated = torch.cuda.memory_allocated(0)
            cached = torch.cuda.memory_reserved(0)
            
            total_mb = self.total_vram / (1024 * 1024)
            allocated_mb = allocated / (1024 * 1024)
            cached_mb = cached / (1024 * 1024)
            free_mb = total_mb - max(allocated_mb, cached_mb)
            utilization = (max(allocated_mb, cached_mb) / total_mb) * 100
            
            return VRAMInfo(
                total_mb=total_mb,
                allocated_mb=allocated_mb,
                cached_mb=cached_mb,
                free_mb=free_mb,
                utilization_percent=utilization
            )
        except Exception as e:
            logger.error(f"Failed to get VRAM info: {e}")
            return VRAMInfo(0, 0, 0, 0, 0)
    
    def get_system_resource_info(self) -> SystemResourceInfo:
        """Get comprehensive system resource information"""
        vram_info = self.get_vram_info()
        
        try:
            # RAM information
            memory = psutil.virtual_memory()
            ram_total_gb = memory.total / (1024**3)
            ram_available_gb = memory.available / (1024**3)
            ram_usage_percent = memory.percent
            
            # CPU information
            cpu_usage_percent = psutil.cpu_percent(interval=0.1)
            
            # Disk information
            disk_usage = psutil.disk_usage('.')
            disk_free_gb = disk_usage.free / (1024**3)
            
            # GPU temperature (if available)
            gpu_temperature = None
            try:
                if self.gpu_available:
                    # Try to get GPU temperature using nvidia-ml-py if available
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except ImportError:
                pass  # pynvml not available
            except Exception:
                pass  # GPU temperature not available
            
            return SystemResourceInfo(
                vram=vram_info,
                ram_total_gb=ram_total_gb,
                ram_available_gb=ram_available_gb,
                ram_usage_percent=ram_usage_percent,
                cpu_usage_percent=cpu_usage_percent,
                disk_free_gb=disk_free_gb,
                gpu_temperature=gpu_temperature
            )
        except Exception as e:
            logger.error(f"Failed to get system resource info: {e}")
            return SystemResourceInfo(
                vram=vram_info,
                ram_total_gb=0,
                ram_available_gb=0,
                ram_usage_percent=0,
                cpu_usage_percent=0,
                disk_free_gb=0
            )
    
    def check_vram_availability(self, required_mb: float, safety_margin: bool = True) -> Tuple[bool, str]:
        """Check if sufficient VRAM is available for generation"""
        if not self.gpu_available:
            return False, "No GPU available"
        
        try:
            vram_info = self.get_vram_info()
            
            # Apply safety margin if requested
            margin = self.memory_safety_margin_mb if safety_margin else 0
            required_with_margin = required_mb + margin
            
            if vram_info.free_mb >= required_with_margin:
                return True, f"Sufficient VRAM available: {vram_info.free_mb:.0f}MB free, {required_mb:.0f}MB required"
            else:
                shortage = required_with_margin - vram_info.free_mb
                return False, f"Insufficient VRAM: {shortage:.0f}MB short (need {required_mb:.0f}MB, have {vram_info.free_mb:.0f}MB free)"
        
        except Exception as e:
            logger.error(f"VRAM availability check failed: {e}")
            return False, f"VRAM check error: {str(e)}"
    
    def estimate_resource_requirements(self, model_type: str, resolution: str, 
                                     steps: int, duration: int = 4, 
                                     lora_count: int = 0) -> ResourceRequirement:
        """Estimate resource requirements for different generation modes"""
        try:
            # Base requirements calibrated from actual usage
            base_requirements = {
                "t2v-A14B": {"vram": 6500, "ram": 3000, "time_per_step": 0.8},
                "i2v-A14B": {"vram": 6000, "ram": 2800, "time_per_step": 0.7},
                "ti2v-5B": {"vram": 4500, "ram": 2200, "time_per_step": 0.5}
            }
            
            # Resolution multipliers based on actual measurements
            resolution_multipliers = {
                "480p": 0.6,
                "720p": 1.0,
                "1080p": 2.4,
                "1440p": 4.2,
                "2160p": 8.5  # 4K
            }
            
            # Get base requirements
            base = base_requirements.get(model_type, base_requirements["t2v-A14B"])
            res_multiplier = resolution_multipliers.get(resolution, 1.0)
            
            # Calculate VRAM requirements
            base_vram = base["vram"] * res_multiplier
            
            # Step scaling (non-linear due to memory accumulation)
            step_factor = 1.0 + ((steps - 50) / 50) * 0.3
            vram_mb = base_vram * step_factor
            
            # Duration scaling for RAM (longer videos need more buffer)
            duration_factor = 1.0 + ((duration - 4) / 4) * 0.2
            ram_mb = base["ram"] * res_multiplier * duration_factor
            
            # LoRA overhead
            if lora_count > 0:
                lora_overhead = 1.0 + (lora_count * 0.15)
                vram_mb *= lora_overhead
                ram_mb *= 1.1  # Slight RAM overhead for LoRA
            
            # Time estimation
            estimated_time = base["time_per_step"] * steps * res_multiplier * duration_factor
            
            return ResourceRequirement(
                model_type=model_type,
                resolution=resolution,
                steps=steps,
                duration=duration,
                vram_mb=vram_mb,
                ram_mb=ram_mb,
                estimated_time_seconds=estimated_time
            )
        
        except Exception as e:
            logger.error(f"Resource estimation failed: {e}")
            # Return conservative estimates
            return ResourceRequirement(
                model_type=model_type,
                resolution=resolution,
                steps=steps,
                duration=duration,
                vram_mb=8000,
                ram_mb=4000,
                estimated_time_seconds=120
            )
    
    def optimize_parameters_for_resources(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], List[OptimizationSuggestion]]:
        """Automatically optimize parameters based on available resources"""
        try:
            optimized_params = params.copy()
            suggestions = []
            
            # Get current resource status
            resource_info = self.get_system_resource_info()
            
            # Estimate requirements for current parameters
            requirement = self.estimate_resource_requirements(
                model_type=params.get("model_type", "t2v-A14B"),
                resolution=params.get("resolution", "720p"),
                steps=params.get("steps", 50),
                duration=params.get("duration", 4),
                lora_count=len(params.get("lora_config", {}))
            )
            
            # Check if optimization is needed
            vram_available, vram_message = self.check_vram_availability(requirement.vram_mb)
            
            if not vram_available:
                # Apply VRAM optimizations
                suggestions.extend(self._optimize_for_vram(optimized_params, requirement, resource_info))
            
            # Check RAM availability
            if resource_info.ram_available_gb * 1024 < requirement.ram_mb:
                suggestions.extend(self._optimize_for_ram(optimized_params, requirement, resource_info))
            
            # Apply performance optimizations if resources are limited
            if resource_info.vram.utilization_percent > 80 or resource_info.ram_usage_percent > 80:
                suggestions.extend(self._optimize_for_performance(optimized_params, requirement, resource_info))
            
            return optimized_params, suggestions
        
        except Exception as e:
            logger.error(f"Parameter optimization failed: {e}")
            return params, []
    
    def _optimize_for_vram(self, params: Dict[str, Any], requirement: ResourceRequirement, 
                          resource_info: SystemResourceInfo) -> List[OptimizationSuggestion]:
        """Optimize parameters to reduce VRAM usage"""
        suggestions = []
        
        try:
            # Resolution optimization
            current_res = params.get("resolution", "720p")
            if current_res == "1080p":
                params["resolution"] = "720p"
                vram_savings = requirement.vram_mb * 0.58  # ~58% reduction from 1080p to 720p
                suggestions.append(OptimizationSuggestion(
                    parameter="resolution",
                    current_value="1080p",
                    suggested_value="720p",
                    reason="Reduce VRAM usage for insufficient GPU memory",
                    impact="Significant VRAM reduction, moderate quality impact",
                    vram_savings_mb=vram_savings
                ))
            elif current_res == "720p" and resource_info.vram.free_mb < 4000:
                params["resolution"] = "480p"
                vram_savings = requirement.vram_mb * 0.4
                suggestions.append(OptimizationSuggestion(
                    parameter="resolution",
                    current_value="720p",
                    suggested_value="480p",
                    reason="Critical VRAM shortage",
                    impact="Major VRAM reduction, noticeable quality impact",
                    vram_savings_mb=vram_savings
                ))
            
            # Steps optimization
            current_steps = params.get("steps", 50)
            if current_steps > 40 and resource_info.vram.free_mb < 6000:
                new_steps = max(25, current_steps - 15)
                params["steps"] = new_steps
                vram_savings = requirement.vram_mb * 0.2
                suggestions.append(OptimizationSuggestion(
                    parameter="steps",
                    current_value=current_steps,
                    suggested_value=new_steps,
                    reason="Reduce inference steps to save VRAM",
                    impact="Moderate VRAM reduction, slight quality impact",
                    vram_savings_mb=vram_savings
                ))
            
            # Enable memory optimizations
            if "optimization_settings" not in params:
                params["optimization_settings"] = {}
            
            params["optimization_settings"].update({
                "enable_memory_efficient_attention": True,
                "enable_sequential_cpu_offload": True,
                "enable_vae_tiling": True
            })
            
            suggestions.append(OptimizationSuggestion(
                parameter="memory_optimizations",
                current_value="disabled",
                suggested_value="enabled",
                reason="Enable memory-efficient techniques",
                impact="Significant VRAM reduction, minimal quality impact",
                vram_savings_mb=requirement.vram_mb * 0.25
            ))
            
        except Exception as e:
            logger.error(f"VRAM optimization failed: {e}")
        
        return suggestions
    
    def _optimize_for_ram(self, params: Dict[str, Any], requirement: ResourceRequirement,
                         resource_info: SystemResourceInfo) -> List[OptimizationSuggestion]:
        """Optimize parameters to reduce RAM usage"""
        suggestions = []
        
        try:
            # Reduce duration if RAM is limited
            current_duration = params.get("duration", 4)
            if current_duration > 4 and resource_info.ram_available_gb < 3:
                new_duration = max(2, current_duration - 2)
                params["duration"] = new_duration
                suggestions.append(OptimizationSuggestion(
                    parameter="duration",
                    current_value=current_duration,
                    suggested_value=new_duration,
                    reason="Reduce video duration to save system RAM",
                    impact="RAM usage reduction, shorter output video"
                ))
            
            # Enable CPU offloading to reduce RAM pressure
            if "optimization_settings" not in params:
                params["optimization_settings"] = {}
            
            params["optimization_settings"]["enable_cpu_offload"] = True
            suggestions.append(OptimizationSuggestion(
                parameter="cpu_offload",
                current_value="disabled",
                suggested_value="enabled",
                reason="Offload model components to reduce RAM usage",
                impact="RAM usage reduction, slight performance impact"
            ))
            
        except Exception as e:
            logger.error(f"RAM optimization failed: {e}")
        
        return suggestions
    
    def _optimize_for_performance(self, params: Dict[str, Any], requirement: ResourceRequirement,
                                 resource_info: SystemResourceInfo) -> List[OptimizationSuggestion]:
        """Optimize parameters for better performance under resource constraints"""
        suggestions = []
        
        try:
            # Reduce guidance scale for faster inference
            current_guidance = params.get("guidance_scale", 7.5)
            if current_guidance > 10:
                new_guidance = 8.0
                params["guidance_scale"] = new_guidance
                suggestions.append(OptimizationSuggestion(
                    parameter="guidance_scale",
                    current_value=current_guidance,
                    suggested_value=new_guidance,
                    reason="Reduce guidance scale for faster inference",
                    impact="Faster generation, minimal quality impact"
                ))
            
            # Enable performance optimizations
            if "optimization_settings" not in params:
                params["optimization_settings"] = {}
            
            params["optimization_settings"].update({
                "enable_attention_slicing": True,
                "enable_xformers": True
            })
            
            suggestions.append(OptimizationSuggestion(
                parameter="performance_optimizations",
                current_value="disabled",
                suggested_value="enabled",
                reason="Enable performance optimizations for resource-constrained system",
                impact="Faster generation, lower memory usage"
            ))
            
        except Exception as e:
            logger.error(f"Performance optimization failed: {e}")
        
        return suggestions
    
    def cleanup_memory(self, aggressive: bool = False) -> Dict[str, Any]:
        """Implement memory cleanup and optimization strategies"""
        cleanup_results = {
            "vram_before_mb": 0,
            "vram_after_mb": 0,
            "vram_freed_mb": 0,
            "ram_before_mb": 0,
            "ram_after_mb": 0,
            "ram_freed_mb": 0,
            "actions_taken": []
        }
        
        try:
            # Get initial memory state
            initial_vram = self.get_vram_info()
            initial_ram = psutil.virtual_memory()
            
            cleanup_results["vram_before_mb"] = initial_vram.allocated_mb
            cleanup_results["ram_before_mb"] = (initial_ram.total - initial_ram.available) / (1024 * 1024)
            
            # GPU memory cleanup
            if self.gpu_available:
                # Clear GPU cache
                torch.cuda.empty_cache()
                cleanup_results["actions_taken"].append("cleared_gpu_cache")
                
                if aggressive:
                    # Force garbage collection
                    gc.collect()
                    torch.cuda.empty_cache()
                    cleanup_results["actions_taken"].append("aggressive_gpu_cleanup")
                    
                    # Reset peak memory stats
                    torch.cuda.reset_peak_memory_stats()
                    cleanup_results["actions_taken"].append("reset_memory_stats")
            
            # System memory cleanup
            gc.collect()
            cleanup_results["actions_taken"].append("garbage_collection")
            
            if aggressive:
                # Force multiple GC cycles
                for _ in range(3):
                    gc.collect()
                cleanup_results["actions_taken"].append("aggressive_garbage_collection")
            
            # Get final memory state
            final_vram = self.get_vram_info()
            final_ram = psutil.virtual_memory()
            
            cleanup_results["vram_after_mb"] = final_vram.allocated_mb
            cleanup_results["ram_after_mb"] = (final_ram.total - final_ram.available) / (1024 * 1024)
            
            cleanup_results["vram_freed_mb"] = cleanup_results["vram_before_mb"] - cleanup_results["vram_after_mb"]
            cleanup_results["ram_freed_mb"] = cleanup_results["ram_before_mb"] - cleanup_results["ram_after_mb"]
            
            logger.info(f"Memory cleanup completed: VRAM freed {cleanup_results['vram_freed_mb']:.1f}MB, "
                       f"RAM freed {cleanup_results['ram_freed_mb']:.1f}MB")
            
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
            cleanup_results["error"] = str(e)
        
        return cleanup_results
    
    def _start_resource_monitoring(self):
        """Start background resource monitoring"""
        if not self.monitoring_enabled:
            return
        
        def monitor_resources():
            while self.monitoring_enabled:
                try:
                    resource_info = self.get_system_resource_info()
                    self.resource_history.append(resource_info)
                    
                    # Keep only last 100 entries
                    if len(self.resource_history) > 100:
                        self.resource_history = self.resource_history[-100:]
                    
                    # Check for critical resource conditions
                    if resource_info.vram.utilization_percent > 95:
                        logger.warning(f"Critical VRAM usage: {resource_info.vram.utilization_percent:.1f}%")
                    
                    if resource_info.ram_usage_percent > 90:
                        logger.warning(f"Critical RAM usage: {resource_info.ram_usage_percent:.1f}%")
                    
                    time.sleep(5)  # Monitor every 5 seconds
                    
                except Exception as e:
                    logger.error(f"Resource monitoring error: {e}")
                    time.sleep(10)  # Wait longer on error
        
        self.monitoring_thread = threading.Thread(target=monitor_resources, daemon=True)
        self.monitoring_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring_enabled = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1)
        logger.info("Resource monitoring stopped")
    
    def get_resource_status(self) -> ResourceStatus:
        """Get overall resource status"""
        try:
            resource_info = self.get_system_resource_info()
            
            # Determine status based on resource availability
            if resource_info.vram.utilization_percent > 95 or resource_info.ram_usage_percent > 95:
                return ResourceStatus.CRITICAL
            elif resource_info.vram.utilization_percent > 85 or resource_info.ram_usage_percent > 85:
                return ResourceStatus.INSUFFICIENT
            elif resource_info.vram.utilization_percent > 70 or resource_info.ram_usage_percent > 70:
                return ResourceStatus.LIMITED
            elif resource_info.vram.utilization_percent > 50 or resource_info.ram_usage_percent > 50:
                return ResourceStatus.AVAILABLE
            else:
                return ResourceStatus.OPTIMAL
        
        except Exception as e:
            logger.error(f"Resource status check failed: {e}")
            return ResourceStatus.INSUFFICIENT
    
    def get_resource_history(self, last_n: int = 10) -> List[Dict[str, Any]]:
        """Get recent resource usage history"""
        try:
            recent_history = self.resource_history[-last_n:] if self.resource_history else []
            return [info.to_dict() for info in recent_history]
        except Exception as e:
            logger.error(f"Failed to get resource history: {e}")
            return []


# Global resource manager instance
_resource_manager = None

def get_resource_manager(config: Optional[Dict[str, Any]] = None) -> VRAMOptimizer:
    """Get the global resource manager instance"""
    global _resource_manager
    if _resource_manager is None:
        if config is None:
            # Load default config
            try:
                with open("config.json", 'r') as f:
                    config = json.load(f)
            except:
                config = {"optimization": {}}
        _resource_manager = VRAMOptimizer(config)
    return _resource_manager

# Convenience functions
def check_vram_availability(required_mb: float) -> Tuple[bool, str]:
    """Check if sufficient VRAM is available"""
    manager = get_resource_manager()
    return manager.check_vram_availability(required_mb)

def estimate_resource_requirements(model_type: str, resolution: str, steps: int, 
                                 duration: int = 4, lora_count: int = 0) -> ResourceRequirement:
    """Estimate resource requirements for generation"""
    manager = get_resource_manager()
    return manager.estimate_resource_requirements(model_type, resolution, steps, duration, lora_count)

def optimize_parameters_for_resources(params: Dict[str, Any]) -> Tuple[Dict[str, Any], List[OptimizationSuggestion]]:
    """Optimize parameters based on available resources"""
    manager = get_resource_manager()
    return manager.optimize_parameters_for_resources(params)

def cleanup_memory(aggressive: bool = False) -> Dict[str, Any]:
    """Clean up memory"""
    manager = get_resource_manager()
    return manager.cleanup_memory(aggressive)

def get_system_resource_info() -> SystemResourceInfo:
    """Get current system resource information"""
    manager = get_resource_manager()
    return manager.get_system_resource_info()