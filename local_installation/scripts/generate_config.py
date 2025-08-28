"""
Hardware-aware configuration generator for WAN2.2 installation.
Generates optimized configuration files based on detected hardware specifications.
"""

import json
import math
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import asdict

from interfaces import (
    IConfigurationEngine, HardwareProfile, CPUInfo, MemoryInfo, 
    GPUInfo, StorageInfo, InstallationError, ErrorCategory
)
from base_classes import BaseInstallationComponent, ConfigurationTemplate


class HardwareTier:
    """Hardware tier classification for configuration templates."""
    HIGH_PERFORMANCE = "high_performance"
    MID_RANGE = "mid_range"
    BUDGET = "budget"
    MINIMUM = "minimum"


class ConfigurationEngine(BaseInstallationComponent, IConfigurationEngine):
    """Hardware-aware configuration generator."""
    
    def __init__(self, installation_path: str):
        super().__init__(installation_path)
        self.config_templates = {
            HardwareTier.HIGH_PERFORMANCE: ConfigurationTemplate.get_high_performance_template(),
            HardwareTier.MID_RANGE: ConfigurationTemplate.get_mid_range_template(),
            HardwareTier.BUDGET: ConfigurationTemplate.get_budget_template(),
            HardwareTier.MINIMUM: self._get_minimum_template()
        }
    
    def _get_minimum_template(self) -> Dict[str, Any]:
        """Configuration template for minimum spec systems."""
        return {
            "system": {
                "default_quantization": "int8",
                "enable_offload": True,
                "vae_tile_size": 64,
                "max_queue_size": 2,
                "worker_threads": 2
            },
            "optimization": {
                "max_vram_usage_gb": 2,
                "cpu_threads": 4,
                "memory_pool_gb": 2
            },
            "models": {
                "cache_models": False,
                "preload_models": False,
                "model_precision": "int8"
            }
        }
    
    def classify_hardware_tier(self, hardware_profile: HardwareProfile) -> str:
        """Classify hardware into performance tiers."""
        cpu_score = self._calculate_cpu_score(hardware_profile.cpu)
        memory_score = self._calculate_memory_score(hardware_profile.memory)
        gpu_score = self._calculate_gpu_score(hardware_profile.gpu)
        
        total_score = cpu_score + memory_score + gpu_score
        
        self.logger.info(f"Hardware scores - CPU: {cpu_score}, Memory: {memory_score}, GPU: {gpu_score}, Total: {total_score}")
        
        if total_score >= 80:
            return HardwareTier.HIGH_PERFORMANCE
        elif total_score >= 60:
            return HardwareTier.MID_RANGE
        elif total_score >= 40:
            return HardwareTier.BUDGET
        else:
            return HardwareTier.MINIMUM
    
    def _calculate_cpu_score(self, cpu: CPUInfo) -> float:
        """Calculate CPU performance score (0-30)."""
        # Base score from core count
        core_score = min(cpu.cores * 0.8, 20)
        
        # Thread bonus
        thread_bonus = min((cpu.threads - cpu.cores) * 0.2, 5)
        
        # Clock speed bonus
        clock_bonus = min((cpu.boost_clock - 2.0) * 2, 5)
        
        return max(0, core_score + thread_bonus + clock_bonus)
    
    def _calculate_memory_score(self, memory: MemoryInfo) -> float:
        """Calculate memory performance score (0-25)."""
        # Base score from total memory
        memory_score = min(memory.total_gb * 0.3, 20)
        
        # Speed bonus for DDR4/DDR5
        speed_bonus = 0
        if "DDR5" in memory.type:
            speed_bonus = 5
        elif "DDR4" in memory.type:
            speed_bonus = 3
        elif "DDR3" in memory.type:
            speed_bonus = 1
        
        return memory_score + speed_bonus
    
    def _calculate_gpu_score(self, gpu: Optional[GPUInfo]) -> float:
        """Calculate GPU performance score (0-45)."""
        if not gpu:
            return 0
        
        # VRAM score
        vram_score = min(gpu.vram_gb * 2.5, 30)
        
        # Model-specific bonuses
        model_bonus = 0
        model_lower = gpu.model.lower()
        
        if "rtx 4090" in model_lower:
            model_bonus = 15
        elif "rtx 4080" in model_lower or "rtx 4070 ti" in model_lower:
            model_bonus = 12
        elif "rtx 4070" in model_lower or "rtx 3080" in model_lower:
            model_bonus = 10
        elif "rtx 3070" in model_lower or "rtx 4060 ti" in model_lower:
            model_bonus = 8
        elif "rtx 3060" in model_lower or "rtx 4060" in model_lower:
            model_bonus = 6
        elif "gtx 1660" in model_lower or "rtx 2060" in model_lower:
            model_bonus = 4
        elif "gtx 1060" in model_lower:
            model_bonus = 2
        
        return min(vram_score + model_bonus, 45)
    
    def generate_config(self, hardware_profile: HardwareProfile) -> Dict[str, Any]:
        """Generate configuration based on hardware profile."""
        try:
            # Classify hardware tier
            tier = self.classify_hardware_tier(hardware_profile)
            self.logger.info(f"Hardware classified as: {tier}")
            
            # Get base template
            base_config = self.config_templates[tier].copy()
            
            # Apply hardware-specific optimizations
            optimized_config = self.optimize_for_hardware(base_config, hardware_profile)
            
            # Add metadata
            optimized_config["metadata"] = {
                "hardware_tier": tier,
                "generated_for": {
                    "cpu": hardware_profile.cpu.model,
                    "memory_gb": hardware_profile.memory.total_gb,
                    "gpu": hardware_profile.gpu.model if hardware_profile.gpu else "None",
                    "os": f"{hardware_profile.os.name} {hardware_profile.os.version}"
                },
                "generation_timestamp": self._get_timestamp()
            }
            
            return optimized_config
            
        except Exception as e:
            raise InstallationError(
                f"Failed to generate configuration: {str(e)}",
                ErrorCategory.CONFIGURATION,
                ["Check hardware detection results", "Use default configuration"]
            )
    
    def optimize_for_hardware(self, base_config: Dict[str, Any], 
                            hardware: HardwareProfile) -> Dict[str, Any]:
        """Optimize configuration for specific hardware."""
        config = base_config.copy()
        
        # CPU optimizations
        config = self._optimize_cpu_settings(config, hardware.cpu)
        
        # Memory optimizations
        config = self._optimize_memory_settings(config, hardware.memory)
        
        # GPU optimizations
        if hardware.gpu:
            config = self._optimize_gpu_settings(config, hardware.gpu)
        else:
            config = self._optimize_cpu_only_settings(config)
        
        # Storage optimizations
        config = self._optimize_storage_settings(config, hardware.storage)
        
        return config
    
    def _optimize_cpu_settings(self, config: Dict[str, Any], cpu: CPUInfo) -> Dict[str, Any]:
        """Optimize CPU-related settings."""
        # Calculate optimal thread count (leave some threads for system)
        optimal_threads = max(1, min(cpu.threads - 2, cpu.threads * 0.8))
        config["optimization"]["cpu_threads"] = int(optimal_threads)
        
        # Adjust worker threads based on CPU capability
        if cpu.cores >= 32:
            config["system"]["worker_threads"] = min(32, cpu.cores // 2)
        elif cpu.cores >= 16:
            config["system"]["worker_threads"] = min(16, cpu.cores // 2)
        elif cpu.cores >= 8:
            config["system"]["worker_threads"] = min(8, cpu.cores)
        else:
            config["system"]["worker_threads"] = max(2, cpu.cores // 2)
        
        # High-performance CPU optimizations
        if cpu.cores >= 24 and cpu.boost_clock >= 4.0:
            config["system"]["enable_cpu_acceleration"] = True
            config["system"]["cpu_priority"] = "high"
        
        return config
    
    def _optimize_memory_settings(self, config: Dict[str, Any], memory: MemoryInfo) -> Dict[str, Any]:
        """Optimize memory-related settings."""
        # Calculate safe memory pool size (leave 25% for system)
        safe_memory = memory.available_gb * 0.75
        
        if safe_memory >= 64:
            config["optimization"]["memory_pool_gb"] = min(32, safe_memory * 0.5)
        elif safe_memory >= 32:
            config["optimization"]["memory_pool_gb"] = min(16, safe_memory * 0.4)
        elif safe_memory >= 16:
            config["optimization"]["memory_pool_gb"] = min(8, safe_memory * 0.3)
        else:
            config["optimization"]["memory_pool_gb"] = max(2, safe_memory * 0.2)
        
        # Adjust queue size based on available memory
        if memory.total_gb >= 64:
            config["system"]["max_queue_size"] = 20
        elif memory.total_gb >= 32:
            config["system"]["max_queue_size"] = 10
        elif memory.total_gb >= 16:
            config["system"]["max_queue_size"] = 5
        else:
            config["system"]["max_queue_size"] = 2
        
        return config
    
    def _optimize_gpu_settings(self, config: Dict[str, Any], gpu: GPUInfo) -> Dict[str, Any]:
        """Optimize GPU-related settings."""
        # VRAM usage optimization
        safe_vram = gpu.vram_gb * 0.85  # Leave 15% buffer
        config["optimization"]["max_vram_usage_gb"] = int(safe_vram)
        
        # Quantization based on VRAM
        if gpu.vram_gb >= 16:
            config["system"]["default_quantization"] = "bf16"
            config["models"]["model_precision"] = "fp16"
        elif gpu.vram_gb >= 12:
            config["system"]["default_quantization"] = "fp16"
            config["models"]["model_precision"] = "fp16"
        elif gpu.vram_gb >= 8:
            config["system"]["default_quantization"] = "fp16"
            config["models"]["model_precision"] = "fp16"
        else:
            config["system"]["default_quantization"] = "int8"
            config["models"]["model_precision"] = "int8"
        
        # VAE tile size based on VRAM
        if gpu.vram_gb >= 16:
            config["system"]["vae_tile_size"] = 512
        elif gpu.vram_gb >= 12:
            config["system"]["vae_tile_size"] = 384
        elif gpu.vram_gb >= 8:
            config["system"]["vae_tile_size"] = 256
        else:
            config["system"]["vae_tile_size"] = 128
        
        # Enable GPU-specific features
        config["system"]["enable_gpu_acceleration"] = True
        config["system"]["cuda_enabled"] = "cuda" in gpu.cuda_version.lower()
        
        # Model caching for high VRAM systems
        if gpu.vram_gb >= 16:
            config["models"]["cache_models"] = True
            config["models"]["preload_models"] = True
        elif gpu.vram_gb >= 12:
            config["models"]["cache_models"] = True
            config["models"]["preload_models"] = False
        else:
            config["models"]["cache_models"] = False
            config["models"]["preload_models"] = False
        
        # Offloading strategy
        if gpu.vram_gb >= 12:
            config["system"]["enable_offload"] = False
        else:
            config["system"]["enable_offload"] = True
        
        return config
    
    def _optimize_cpu_only_settings(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize settings for CPU-only systems."""
        config["system"]["enable_gpu_acceleration"] = False
        config["system"]["cuda_enabled"] = False
        config["system"]["enable_offload"] = True
        config["system"]["default_quantization"] = "int8"
        config["models"]["model_precision"] = "int8"
        config["models"]["cache_models"] = False
        config["models"]["preload_models"] = False
        config["optimization"]["max_vram_usage_gb"] = 0
        
        return config
    
    def _optimize_storage_settings(self, config: Dict[str, Any], storage: StorageInfo) -> Dict[str, Any]:
        """Optimize storage-related settings."""
        # Enable caching for fast storage
        if "nvme" in storage.type.lower() or "ssd" in storage.type.lower():
            config["system"]["enable_disk_cache"] = True
            config["system"]["cache_size_gb"] = min(10, storage.available_gb * 0.1)
        else:
            config["system"]["enable_disk_cache"] = False
            config["system"]["cache_size_gb"] = 1
        
        # Temporary file settings
        if storage.available_gb >= 100:
            config["system"]["temp_space_gb"] = 20
        elif storage.available_gb >= 50:
            config["system"]["temp_space_gb"] = 10
        else:
            config["system"]["temp_space_gb"] = 5
        
        return config
    
    def save_config(self, config: Dict[str, Any], config_path: str) -> bool:
        """Save configuration to file."""
        try:
            config_file = Path(config_path)
            self.save_json_file(config, config_file)
            self.logger.info(f"Configuration saved to: {config_file}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False
    
    def create_config_variants(self, hardware_profile: HardwareProfile) -> Dict[str, Dict[str, Any]]:
        """Create multiple configuration variants for different use cases."""
        base_config = self.generate_config(hardware_profile)
        
        variants = {
            "balanced": base_config,
            "performance": self._create_performance_variant(base_config),
            "memory_conservative": self._create_memory_conservative_variant(base_config),
            "quality_focused": self._create_quality_focused_variant(base_config)
        }
        
        return variants
    
    def _create_performance_variant(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create performance-focused configuration variant."""
        config = base_config.copy()
        
        # Increase thread usage
        config["optimization"]["cpu_threads"] = int(config["optimization"]["cpu_threads"] * 1.2)
        config["system"]["worker_threads"] = int(config["system"]["worker_threads"] * 1.5)
        
        # More aggressive memory usage
        config["optimization"]["memory_pool_gb"] = int(config["optimization"]["memory_pool_gb"] * 1.3)
        
        # Larger queue for throughput
        config["system"]["max_queue_size"] = int(config["system"]["max_queue_size"] * 1.5)
        
        config["metadata"]["variant"] = "performance"
        return config
    
    def _create_memory_conservative_variant(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create memory-conservative configuration variant."""
        config = base_config.copy()
        
        # Reduce memory usage
        config["optimization"]["memory_pool_gb"] = int(config["optimization"]["memory_pool_gb"] * 0.7)
        config["optimization"]["max_vram_usage_gb"] = int(config["optimization"]["max_vram_usage_gb"] * 0.8)
        
        # Smaller queue
        config["system"]["max_queue_size"] = max(2, int(config["system"]["max_queue_size"] * 0.6))
        
        # Enable more aggressive offloading
        config["system"]["enable_offload"] = True
        
        config["metadata"]["variant"] = "memory_conservative"
        return config
    
    def _create_quality_focused_variant(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create quality-focused configuration variant."""
        config = base_config.copy()
        
        # Higher precision if possible
        if config["system"]["default_quantization"] == "int8":
            config["system"]["default_quantization"] = "fp16"
        elif config["system"]["default_quantization"] == "fp16":
            config["system"]["default_quantization"] = "bf16"
        
        # Larger VAE tile size for better quality
        config["system"]["vae_tile_size"] = min(512, int(config["system"]["vae_tile_size"] * 1.5))
        
        # Enable model caching for consistency
        config["models"]["cache_models"] = True
        
        config["metadata"]["variant"] = "quality_focused"
        return config
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for metadata."""
        from datetime import datetime
        return datetime.now().isoformat()


def calculate_optimal_settings(hardware_profile: HardwareProfile) -> Dict[str, Any]:
    """Standalone function to calculate optimal settings for given hardware."""
    engine = ConfigurationEngine(".")
    return engine.generate_config(hardware_profile)


def get_hardware_recommendations(hardware_profile: HardwareProfile) -> List[str]:
    """Get hardware upgrade recommendations based on current specs."""
    recommendations = []
    
    # CPU recommendations
    if hardware_profile.cpu.cores < 8:
        recommendations.append("Consider upgrading to a CPU with 8+ cores for better performance")
    
    # Memory recommendations
    if hardware_profile.memory.total_gb < 16:
        recommendations.append("Upgrade to at least 16GB RAM for optimal performance")
    elif hardware_profile.memory.total_gb < 32:
        recommendations.append("Consider upgrading to 32GB RAM for high-resolution video generation")
    
    # GPU recommendations
    if not hardware_profile.gpu:
        recommendations.append("Add a dedicated GPU with 8GB+ VRAM for GPU acceleration")
    elif hardware_profile.gpu.vram_gb < 8:
        recommendations.append("Upgrade to a GPU with 8GB+ VRAM for better performance")
    elif hardware_profile.gpu.vram_gb < 12:
        recommendations.append("Consider upgrading to a GPU with 12GB+ VRAM for high-resolution generation")
    
    # Storage recommendations
    if "hdd" in hardware_profile.storage.type.lower():
        recommendations.append("Upgrade to an SSD for faster model loading and caching")
    
    return recommendations