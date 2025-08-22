"""
Hardware-Based Parameter Recommendation System for Wan2.2 Video Generation

This module provides intelligent parameter recommendations based on hardware
capabilities, performance analysis, and user preferences.
"""

import logging
import psutil
import platform
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import threading
import time

logger = logging.getLogger(__name__)

class HardwareClass(Enum):
    """Hardware performance classes"""
    LOW_END = "low_end"
    MID_RANGE = "mid_range"
    HIGH_END = "high_end"
    ENTHUSIAST = "enthusiast"

class RecommendationConfidence(Enum):
    """Confidence levels for recommendations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class HardwareProfile:
    """Hardware profile information"""
    gpu_name: str = ""
    gpu_memory_gb: float = 0.0
    cpu_cores: int = 0
    cpu_name: str = ""
    system_ram_gb: float = 0.0
    disk_type: str = "unknown"  # ssd, hdd, nvme
    hardware_class: HardwareClass = HardwareClass.LOW_END
    performance_score: float = 0.0
    detected_at: str = ""

@dataclass
class ParameterRecommendation:
    """Parameter recommendation with reasoning"""
    parameter: str
    recommended_value: Any
    current_value: Any
    confidence: RecommendationConfidence
    reasoning: str
    performance_impact: str  # "speed", "quality", "memory"
    estimated_improvement: float  # percentage
    alternative_values: List[Any] = field(default_factory=list)

@dataclass
class GenerationProfile:
    """Recommended generation profile"""
    name: str
    description: str
    parameters: Dict[str, Any]
    estimated_time_minutes: float
    quality_score: float  # 1-10
    memory_usage_gb: float
    success_probability: float
    use_cases: List[str] = field(default_factory=list)

class HardwareParameterRecommender:
    """Provides hardware-based parameter recommendations"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.hardware_profile: Optional[HardwareProfile] = None
        self.performance_history: List[Dict[str, Any]] = []
        self.user_preferences: Dict[str, Any] = {}
        
        # Load performance database
        self.performance_db = self._load_performance_database()
        
        # Initialize hardware detection
        self._detect_hardware()
        
        # Load user preferences
        self._load_user_preferences()
    
    def _detect_hardware(self):
        """Detect and profile system hardware"""
        try:
            # Detect GPU
            gpu_info = self._detect_gpu()
            
            # Detect CPU
            cpu_info = self._detect_cpu()
            
            # Detect memory
            memory_info = self._detect_memory()
            
            # Detect storage
            storage_info = self._detect_storage()
            
            # Create hardware profile
            self.hardware_profile = HardwareProfile(
                gpu_name=gpu_info.get("name", "Unknown"),
                gpu_memory_gb=gpu_info.get("memory_gb", 0.0),
                cpu_cores=cpu_info.get("cores", 0),
                cpu_name=cpu_info.get("name", "Unknown"),
                system_ram_gb=memory_info.get("total_gb", 0.0),
                disk_type=storage_info.get("type", "unknown"),
                detected_at=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
            # Classify hardware and calculate performance score
            self._classify_hardware()
            
            logger.info(f"Hardware detected: {self.hardware_profile.hardware_class.value} "
                       f"(Score: {self.hardware_profile.performance_score:.1f})")
            
        except Exception as e:
            logger.error(f"Hardware detection failed: {e}")
            # Create minimal profile
            self.hardware_profile = HardwareProfile()
    
    def _detect_gpu(self) -> Dict[str, Any]:
        """Detect GPU information"""
        gpu_info = {"name": "Unknown", "memory_gb": 0.0}
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_gb = gpu_memory / (1024**3)
                
                gpu_info = {
                    "name": gpu_name,
                    "memory_gb": gpu_memory_gb
                }
                
                logger.info(f"GPU detected: {gpu_name} ({gpu_memory_gb:.1f}GB)")
            else:
                logger.warning("CUDA not available")
        except Exception as e:
            logger.error(f"GPU detection failed: {e}")
        
        return gpu_info
    
    def _detect_cpu(self) -> Dict[str, Any]:
        """Detect CPU information"""
        try:
            cpu_count = psutil.cpu_count(logical=False)
            cpu_name = platform.processor() or "Unknown CPU"
            
            return {
                "cores": cpu_count,
                "name": cpu_name
            }
        except Exception as e:
            logger.error(f"CPU detection failed: {e}")
            return {"cores": 1, "name": "Unknown"}
    
    def _detect_memory(self) -> Dict[str, Any]:
        """Detect system memory information"""
        try:
            memory = psutil.virtual_memory()
            total_gb = memory.total / (1024**3)
            
            return {"total_gb": total_gb}
        except Exception as e:
            logger.error(f"Memory detection failed: {e}")
            return {"total_gb": 8.0}
    
    def _detect_storage(self) -> Dict[str, Any]:
        """Detect storage type"""
        try:
            # Simple heuristic - check if main drive is likely SSD
            # This is a simplified implementation
            return {"type": "ssd"}  # Default assumption
        except Exception as e:
            logger.error(f"Storage detection failed: {e}")
            return {"type": "unknown"}
    
    def _classify_hardware(self):
        """Classify hardware performance class"""
        if not self.hardware_profile:
            return
        
        score = 0
        
        # GPU scoring (most important for video generation)
        gpu_memory = self.hardware_profile.gpu_memory_gb
        if gpu_memory >= 24:
            score += 40  # Enthusiast
        elif gpu_memory >= 16:
            score += 30  # High-end
        elif gpu_memory >= 8:
            score += 20  # Mid-range
        elif gpu_memory >= 4:
            score += 10  # Low-end
        
        # CPU scoring
        cpu_cores = self.hardware_profile.cpu_cores
        if cpu_cores >= 16:
            score += 20
        elif cpu_cores >= 8:
            score += 15
        elif cpu_cores >= 4:
            score += 10
        else:
            score += 5
        
        # RAM scoring
        ram_gb = self.hardware_profile.system_ram_gb
        if ram_gb >= 32:
            score += 15
        elif ram_gb >= 16:
            score += 10
        elif ram_gb >= 8:
            score += 5
        
        # Storage scoring
        if self.hardware_profile.disk_type == "nvme":
            score += 10
        elif self.hardware_profile.disk_type == "ssd":
            score += 5
        
        self.hardware_profile.performance_score = score
        
        # Classify based on score
        if score >= 70:
            self.hardware_profile.hardware_class = HardwareClass.ENTHUSIAST
        elif score >= 50:
            self.hardware_profile.hardware_class = HardwareClass.HIGH_END
        elif score >= 30:
            self.hardware_profile.hardware_class = HardwareClass.MID_RANGE
        else:
            self.hardware_profile.hardware_class = HardwareClass.LOW_END
    
    def _load_performance_database(self) -> Dict[str, Any]:
        """Load performance database for recommendations"""
        try:
            db_path = Path("performance_database.json")
            if db_path.exists():
                with open(db_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load performance database: {e}")
        
        # Return default database
        return {
            "gpu_profiles": {
                "RTX 4090": {"class": "enthusiast", "optimal_resolution": "1280x720", "max_steps": 100},
                "RTX 4080": {"class": "high_end", "optimal_resolution": "1280x720", "max_steps": 80},
                "RTX 4070": {"class": "high_end", "optimal_resolution": "1280x720", "max_steps": 60},
                "RTX 3090": {"class": "high_end", "optimal_resolution": "1280x720", "max_steps": 80},
                "RTX 3080": {"class": "mid_range", "optimal_resolution": "1280x720", "max_steps": 50},
                "RTX 3070": {"class": "mid_range", "optimal_resolution": "1280x720", "max_steps": 40},
                "RTX 3060": {"class": "low_end", "optimal_resolution": "1280x720", "max_steps": 30}
            },
            "resolution_requirements": {
                "1280x720": {"min_vram_gb": 6, "recommended_vram_gb": 8},
                "1920x1080": {"min_vram_gb": 10, "recommended_vram_gb": 12}
            }
        }
    
    def _load_user_preferences(self):
        """Load user preferences"""
        try:
            prefs_path = Path("user_preferences.json")
            if prefs_path.exists():
                with open(prefs_path, 'r') as f:
                    self.user_preferences = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load user preferences: {e}")
        
        # Set defaults
        self.user_preferences.setdefault("priority", "balanced")  # speed, quality, balanced
        self.user_preferences.setdefault("max_generation_time", 300)  # seconds
        self.user_preferences.setdefault("preferred_resolution", "auto")
    
    def get_hardware_profile(self) -> Optional[HardwareProfile]:
        """Get current hardware profile"""
        return self.hardware_profile
    
    def get_parameter_recommendations(self, generation_type: str = "t2v", 
                                    current_params: Optional[Dict[str, Any]] = None) -> List[ParameterRecommendation]:
        """Get parameter recommendations based on hardware"""
        if not self.hardware_profile:
            return []
        
        recommendations = []
        current_params = current_params or {}
        
        # Resolution recommendation
        resolution_rec = self._recommend_resolution(generation_type, current_params)
        if resolution_rec:
            recommendations.append(resolution_rec)
        
        # Steps recommendation
        steps_rec = self._recommend_steps(generation_type, current_params)
        if steps_rec:
            recommendations.append(steps_rec)
        
        # Batch size recommendation
        batch_rec = self._recommend_batch_size(generation_type, current_params)
        if batch_rec:
            recommendations.append(batch_rec)
        
        # Memory optimization recommendations
        memory_recs = self._recommend_memory_optimizations(generation_type, current_params)
        recommendations.extend(memory_recs)
        
        return recommendations
    
    def _recommend_resolution(self, generation_type: str, current_params: Dict[str, Any]) -> Optional[ParameterRecommendation]:
        """Recommend optimal resolution"""
        if not self.hardware_profile:
            return None
        
        current_resolution = current_params.get("resolution", "1280x720")
        gpu_memory = self.hardware_profile.gpu_memory_gb
        
        # Determine optimal resolution based on VRAM
        if gpu_memory >= 12:
            recommended = "1280x720"
            confidence = RecommendationConfidence.HIGH
            reasoning = "Your GPU has sufficient VRAM for high-quality generation"
        elif gpu_memory >= 8:
            recommended = "1280x720"
            confidence = RecommendationConfidence.MEDIUM
            reasoning = "Your GPU can handle 720p with some optimizations"
        else:
            recommended = "1280x720"
            confidence = RecommendationConfidence.LOW
            reasoning = "Limited VRAM may require additional optimizations"
        
        if current_resolution != recommended:
            return ParameterRecommendation(
                parameter="resolution",
                recommended_value=recommended,
                current_value=current_resolution,
                confidence=confidence,
                reasoning=reasoning,
                performance_impact="memory",
                estimated_improvement=15.0,
                alternative_values=["1280x720"]
            )
        
        return None
    
    def _recommend_steps(self, generation_type: str, current_params: Dict[str, Any]) -> Optional[ParameterRecommendation]:
        """Recommend optimal number of steps"""
        if not self.hardware_profile:
            return None
        
        current_steps = current_params.get("steps", 50)
        hardware_class = self.hardware_profile.hardware_class
        
        # Recommend steps based on hardware class
        if hardware_class == HardwareClass.ENTHUSIAST:
            recommended = min(current_steps, 100)
            reasoning = "Your hardware can handle high step counts for maximum quality"
        elif hardware_class == HardwareClass.HIGH_END:
            recommended = min(current_steps, 80)
            reasoning = "Optimal balance of quality and speed for your hardware"
        elif hardware_class == HardwareClass.MID_RANGE:
            recommended = min(current_steps, 50)
            reasoning = "Balanced settings for good quality and reasonable speed"
        else:
            recommended = min(current_steps, 30)
            reasoning = "Lower steps recommended for faster generation on your hardware"
        
        if current_steps != recommended:
            return ParameterRecommendation(
                parameter="steps",
                recommended_value=recommended,
                current_value=current_steps,
                confidence=RecommendationConfidence.HIGH,
                reasoning=reasoning,
                performance_impact="speed",
                estimated_improvement=20.0,
                alternative_values=[30, 50, 80]
            )
        
        return None
    
    def _recommend_batch_size(self, generation_type: str, current_params: Dict[str, Any]) -> Optional[ParameterRecommendation]:
        """Recommend optimal batch size"""
        # For video generation, batch size is typically 1
        current_batch = current_params.get("batch_size", 1)
        
        if current_batch > 1:
            return ParameterRecommendation(
                parameter="batch_size",
                recommended_value=1,
                current_value=current_batch,
                confidence=RecommendationConfidence.VERY_HIGH,
                reasoning="Video generation requires batch size of 1 for optimal memory usage",
                performance_impact="memory",
                estimated_improvement=30.0
            )
        
        return None
    
    def _recommend_memory_optimizations(self, generation_type: str, current_params: Dict[str, Any]) -> List[ParameterRecommendation]:
        """Recommend memory optimization settings"""
        if not self.hardware_profile:
            return []
        
        recommendations = []
        gpu_memory = self.hardware_profile.gpu_memory_gb
        
        # CPU offloading recommendation
        if gpu_memory < 12:
            current_offload = current_params.get("enable_cpu_offload", False)
            if not current_offload:
                recommendations.append(ParameterRecommendation(
                    parameter="enable_cpu_offload",
                    recommended_value=True,
                    current_value=current_offload,
                    confidence=RecommendationConfidence.HIGH,
                    reasoning="CPU offloading will reduce VRAM usage significantly",
                    performance_impact="memory",
                    estimated_improvement=25.0
                ))
        
        # Mixed precision recommendation
        current_fp16 = current_params.get("enable_fp16", True)
        if not current_fp16:
            recommendations.append(ParameterRecommendation(
                parameter="enable_fp16",
                recommended_value=True,
                current_value=current_fp16,
                confidence=RecommendationConfidence.VERY_HIGH,
                reasoning="Half precision reduces memory usage with minimal quality impact",
                performance_impact="memory",
                estimated_improvement=40.0
            ))
        
        return recommendations
    
    def get_generation_profiles(self, generation_type: str = "t2v") -> List[GenerationProfile]:
        """Get recommended generation profiles"""
        if not self.hardware_profile:
            return []
        
        profiles = []
        hardware_class = self.hardware_profile.hardware_class
        gpu_memory = self.hardware_profile.gpu_memory_gb
        
        # Speed-optimized profile
        speed_params = {
            "resolution": "1280x720",
            "steps": 20 if hardware_class == HardwareClass.LOW_END else 30,
            "enable_cpu_offload": gpu_memory < 12,
            "enable_fp16": True,
            "batch_size": 1
        }
        
        profiles.append(GenerationProfile(
            name="Speed Optimized",
            description="Fast generation with acceptable quality",
            parameters=speed_params,
            estimated_time_minutes=2.0,
            quality_score=6.0,
            memory_usage_gb=min(gpu_memory * 0.8, 8.0),
            success_probability=0.95,
            use_cases=["Quick previews", "Rapid iteration", "Testing prompts"]
        ))
        
        # Balanced profile
        balanced_params = {
            "resolution": "1280x720",
            "steps": 40 if hardware_class in [HardwareClass.HIGH_END, HardwareClass.ENTHUSIAST] else 30,
            "enable_cpu_offload": gpu_memory < 10,
            "enable_fp16": True,
            "batch_size": 1
        }
        
        profiles.append(GenerationProfile(
            name="Balanced",
            description="Good balance of speed and quality",
            parameters=balanced_params,
            estimated_time_minutes=4.0,
            quality_score=8.0,
            memory_usage_gb=min(gpu_memory * 0.9, 10.0),
            success_probability=0.90,
            use_cases=["General use", "Content creation", "Social media"]
        ))
        
        # Quality-optimized profile (only for higher-end hardware)
        if hardware_class in [HardwareClass.HIGH_END, HardwareClass.ENTHUSIAST]:
            quality_params = {
                "resolution": "1280x720",
                "steps": 80 if hardware_class == HardwareClass.ENTHUSIAST else 60,
                "enable_cpu_offload": False,
                "enable_fp16": True,
                "batch_size": 1
            }
            
            profiles.append(GenerationProfile(
                name="Quality Optimized",
                description="Maximum quality for professional use",
                parameters=quality_params,
                estimated_time_minutes=8.0,
                quality_score=9.5,
                memory_usage_gb=min(gpu_memory * 0.95, 12.0),
                success_probability=0.85,
                use_cases=["Professional content", "Final renders", "High-quality output"]
            ))
        
        return profiles
    
    def get_optimization_suggestions(self, current_params: Dict[str, Any], 
                                   target_metric: str = "balanced") -> List[str]:
        """Get optimization suggestions based on target metric"""
        suggestions = []
        
        if not self.hardware_profile:
            return ["Hardware profile not available"]
        
        gpu_memory = self.hardware_profile.gpu_memory_gb
        hardware_class = self.hardware_profile.hardware_class
        
        if target_metric == "speed":
            suggestions.extend([
                "Reduce number of steps to 20-30 for faster generation",
                "Enable CPU offloading to reduce GPU load",
                "Use lower resolution if quality permits"
            ])
        elif target_metric == "quality":
            if hardware_class in [HardwareClass.HIGH_END, HardwareClass.ENTHUSIAST]:
                suggestions.extend([
                    "Increase steps to 60-80 for better quality",
                    "Disable CPU offloading if VRAM permits",
                    "Consider using higher resolution"
                ])
            else:
                suggestions.append("Hardware limitations may prevent quality improvements")
        elif target_metric == "memory":
            suggestions.extend([
                "Enable CPU offloading to reduce VRAM usage",
                "Use FP16 precision to halve memory requirements",
                "Clear GPU cache between generations"
            ])
        
        # Hardware-specific suggestions
        if gpu_memory < 8:
            suggestions.append("Consider upgrading GPU for better performance")
        elif gpu_memory > 16:
            suggestions.append("Your hardware can handle high-quality settings")
        
        return suggestions

# Global recommender instance
_recommender = None

def get_parameter_recommender(config: Optional[Dict[str, Any]] = None) -> HardwareParameterRecommender:
    """Get or create global parameter recommender instance"""
    global _recommender
    if _recommender is None:
        _recommender = HardwareParameterRecommender(config)
    return _recommender