#!/usr/bin/env python3
"""
Model Fallback and Recommendation System for WAN22 System Optimization

This module provides intelligent model fallback options and hardware-based recommendations
for failed model loads, including alternative models and reduced quality settings.

Requirements addressed:
- 7.4: Fallback options for failed model loads (alternative models, reduced quality)
- 7.5: Model recommendation based on hardware configuration
- 7.6: Input validation for image-to-video generation (resolution, format compatibility)
"""

import json
import logging
import os
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import re

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ModelType(Enum):
    """Types of models supported"""
    TEXT_TO_VIDEO = "text_to_video"
    IMAGE_TO_VIDEO = "image_to_video"
    VIDEO_TO_VIDEO = "video_to_video"
    TEXT_TO_IMAGE = "text_to_image"
    IMAGE_TO_IMAGE = "image_to_image"


class ModelSize(Enum):
    """Model size categories"""
    SMALL = "small"      # < 2GB
    MEDIUM = "medium"    # 2-8GB
    LARGE = "large"      # 8-16GB
    XLARGE = "xlarge"    # > 16GB


class QualityLevel(Enum):
    """Quality levels for fallback options"""
    HIGHEST = "highest"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    LOWEST = "lowest"


@dataclass
class HardwareProfile:
    """Hardware configuration profile"""
    gpu_model: str
    vram_gb: int
    cpu_cores: int
    ram_gb: int
    cuda_version: Optional[str] = None
    supports_bf16: bool = False
    supports_int8: bool = False
    supports_fp8: bool = False
    
    def get_vram_category(self) -> str:
        """Get VRAM category for recommendations"""
        if self.vram_gb >= 24:
            return "high"
        elif self.vram_gb >= 12:
            return "medium"
        elif self.vram_gb >= 8:
            return "low"
        else:
            return "minimal"


@dataclass
class ModelInfo:
    """Information about a model"""
    name: str
    model_path: str
    model_type: ModelType
    size_category: ModelSize
    estimated_vram_gb: float
    quality_level: QualityLevel
    supports_quantization: bool = True
    requires_trust_remote_code: bool = False
    min_resolution: Tuple[int, int] = (256, 256)
    max_resolution: Tuple[int, int] = (1024, 1024)
    supported_formats: List[str] = None
    description: str = ""
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ["mp4", "avi", "mov"]


@dataclass
class FallbackOption:
    """A fallback option for failed model loading"""
    model_info: ModelInfo
    reason: str
    quality_impact: str
    performance_impact: str
    compatibility_notes: str = ""


@dataclass
class ModelRecommendation:
    """A model recommendation based on hardware"""
    model_info: ModelInfo
    confidence_score: float  # 0.0 to 1.0
    reasoning: List[str]
    optimization_suggestions: List[str]
    expected_performance: str


@dataclass
class InputValidationResult:
    """Result of input validation for generation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    corrected_parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.corrected_parameters is None:
            self.corrected_parameters = {}


class ModelFallbackSystem:
    """
    Comprehensive model fallback and recommendation system.
    
    Features:
    - Intelligent fallback options for failed model loads
    - Hardware-based model recommendations
    - Input validation for image-to-video generation
    - Quality vs performance trade-off analysis
    """
    
    def __init__(self, models_config_path: str = "models_config.json"):
        """
        Initialize the ModelFallbackSystem.
        
        Args:
            models_config_path: Path to models configuration file
        """
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Load model database
        self.models_db = self._load_models_database(models_config_path)
        
        # Initialize fallback chains
        self.fallback_chains = self._initialize_fallback_chains()
        
        # Hardware compatibility matrix
        self.hardware_compatibility = self._initialize_hardware_compatibility()
        
        # Input validation rules
        self.validation_rules = self._initialize_validation_rules()
        
        self.logger.info("ModelFallbackSystem initialized")
    
    def _setup_logging(self):
        """Setup logging for the fallback system"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _load_models_database(self, config_path: str) -> Dict[str, ModelInfo]:
        """Load models database from configuration file"""
        models_db = {}
        
        # Try to load from file
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    for model_data in config_data.get("models", []):
                        model_info = ModelInfo(**model_data)
                        models_db[model_info.name] = model_info
                self.logger.info(f"Loaded {len(models_db)} models from {config_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load models config: {e}")
        
        # Add default models if database is empty
        if not models_db:
            models_db = self._get_default_models_database()
            self.logger.info(f"Using default models database with {len(models_db)} models")
        
        return models_db
    
    def _get_default_models_database(self) -> Dict[str, ModelInfo]:
        """Get default models database"""
        return {
            "wan22-ti2v-5b": ModelInfo(
                name="wan22-ti2v-5b",
                model_path="Wan-AI/Wan2.2-TI2V-5B",
                model_type=ModelType.TEXT_TO_VIDEO,
                size_category=ModelSize.XLARGE,
                estimated_vram_gb=12.0,
                quality_level=QualityLevel.HIGHEST,
                requires_trust_remote_code=True,
                max_resolution=(1024, 1024),
                description="High-quality text-to-video model with 5B parameters"
            ),
            "wan22-ti2v-1b": ModelInfo(
                name="wan22-ti2v-1b",
                model_path="Wan-AI/Wan2.2-TI2V-1B",
                model_type=ModelType.TEXT_TO_VIDEO,
                size_category=ModelSize.LARGE,
                estimated_vram_gb=8.0,
                quality_level=QualityLevel.HIGH,
                requires_trust_remote_code=True,
                max_resolution=(768, 768),
                description="Balanced text-to-video model with 1B parameters"
            ),
            "wan22-i2v-5b": ModelInfo(
                name="wan22-i2v-5b",
                model_path="Wan-AI/Wan2.2-I2V-5B",
                model_type=ModelType.IMAGE_TO_VIDEO,
                size_category=ModelSize.XLARGE,
                estimated_vram_gb=12.0,
                quality_level=QualityLevel.HIGHEST,
                requires_trust_remote_code=True,
                max_resolution=(1024, 1024),
                description="High-quality image-to-video model with 5B parameters"
            ),
            "wan22-i2v-1b": ModelInfo(
                name="wan22-i2v-1b",
                model_path="Wan-AI/Wan2.2-I2V-1B",
                model_type=ModelType.IMAGE_TO_VIDEO,
                size_category=ModelSize.LARGE,
                estimated_vram_gb=8.0,
                quality_level=QualityLevel.HIGH,
                requires_trust_remote_code=True,
                max_resolution=(768, 768),
                description="Balanced image-to-video model with 1B parameters"
            ),
            "stable-video-diffusion": ModelInfo(
                name="stable-video-diffusion",
                model_path="stabilityai/stable-video-diffusion-img2vid",
                model_type=ModelType.IMAGE_TO_VIDEO,
                size_category=ModelSize.LARGE,
                estimated_vram_gb=10.0,
                quality_level=QualityLevel.HIGH,
                requires_trust_remote_code=False,
                max_resolution=(576, 1024),
                description="Stable Video Diffusion for image-to-video generation"
            ),
            "animatediff": ModelInfo(
                name="animatediff",
                model_path="guoyww/animatediff-motion-adapter-v1-5-2",
                model_type=ModelType.TEXT_TO_VIDEO,
                size_category=ModelSize.MEDIUM,
                estimated_vram_gb=6.0,
                quality_level=QualityLevel.MEDIUM,
                requires_trust_remote_code=False,
                max_resolution=(512, 512),
                description="AnimateDiff motion adapter for text-to-video"
            )
        }
    
    def _initialize_fallback_chains(self) -> Dict[str, List[str]]:
        """Initialize fallback chains for different model types"""
        return {
            "wan22-ti2v-5b": [
                "wan22-ti2v-1b",
                "animatediff",
                "stable-video-diffusion"  # Can be used with text-to-image first
            ],
            "wan22-i2v-5b": [
                "wan22-i2v-1b",
                "stable-video-diffusion",
                "animatediff"  # Can be used with image input
            ],
            "wan22-ti2v-1b": [
                "animatediff",
                "stable-video-diffusion"
            ],
            "wan22-i2v-1b": [
                "stable-video-diffusion",
                "animatediff"
            ]
        }
    
    def _initialize_hardware_compatibility(self) -> Dict[str, Dict[str, float]]:
        """Initialize hardware compatibility matrix"""
        return {
            "rtx4090": {
                "wan22-ti2v-5b": 0.95,
                "wan22-i2v-5b": 0.95,
                "wan22-ti2v-1b": 1.0,
                "wan22-i2v-1b": 1.0,
                "stable-video-diffusion": 0.9,
                "animatediff": 1.0
            },
            "rtx4080": {
                "wan22-ti2v-5b": 0.8,  # Requires optimization
                "wan22-i2v-5b": 0.8,
                "wan22-ti2v-1b": 0.95,
                "wan22-i2v-1b": 0.95,
                "stable-video-diffusion": 0.85,
                "animatediff": 1.0
            },
            "rtx3090": {
                "wan22-ti2v-5b": 0.7,
                "wan22-i2v-5b": 0.7,
                "wan22-ti2v-1b": 0.9,
                "wan22-i2v-1b": 0.9,
                "stable-video-diffusion": 0.8,
                "animatediff": 0.95
            },
            "rtx3080": {
                "wan22-ti2v-5b": 0.5,  # Challenging
                "wan22-i2v-5b": 0.5,
                "wan22-ti2v-1b": 0.8,
                "wan22-i2v-1b": 0.8,
                "stable-video-diffusion": 0.7,
                "animatediff": 0.9
            }
        }
    
    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """Initialize input validation rules"""
        return {
            "resolution": {
                "min_width": 256,
                "min_height": 256,
                "max_width": 1024,
                "max_height": 1024,
                "aspect_ratios": ["16:9", "9:16", "1:1", "4:3", "3:4"],
                "recommended_sizes": [(512, 512), (768, 768), (1024, 1024)]
            },
            "formats": {
                "supported_image": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
                "supported_video": [".mp4", ".avi", ".mov", ".mkv"],
                "recommended_image": [".jpg", ".png"],
                "recommended_video": [".mp4"]
            },
            "generation": {
                "min_frames": 8,
                "max_frames": 64,
                "recommended_frames": [16, 24, 32],
                "min_fps": 8,
                "max_fps": 30,
                "recommended_fps": [12, 16, 24]
            }
        }
    
    def get_fallback_options(self, failed_model: str, error_type: str, 
                           hardware_profile: HardwareProfile) -> List[FallbackOption]:
        """
        Get fallback options for a failed model load.
        
        Args:
            failed_model: Name of the model that failed to load
            error_type: Type of error that occurred
            hardware_profile: Current hardware configuration
            
        Returns:
            List of fallback options sorted by suitability
        """
        fallback_options = []
        
        # Get fallback chain for the failed model
        fallback_chain = self.fallback_chains.get(failed_model, [])
        
        # Add fallback models
        for fallback_model_name in fallback_chain:
            if fallback_model_name in self.models_db:
                fallback_model = self.models_db[fallback_model_name]
                
                # Check if this fallback is suitable for the hardware
                if self._is_model_compatible(fallback_model, hardware_profile, error_type):
                    option = self._create_fallback_option(
                        fallback_model, failed_model, error_type, hardware_profile
                    )
                    fallback_options.append(option)
        
        # Add quantized versions of the original model
        if failed_model in self.models_db:
            original_model = self.models_db[failed_model]
            if original_model.supports_quantization and error_type in ["CUDA_OUT_OF_MEMORY", "INSUFFICIENT_VRAM"]:
                quantized_options = self._get_quantized_fallbacks(original_model, hardware_profile)
                fallback_options.extend(quantized_options)
        
        # Sort by quality and compatibility
        fallback_options.sort(key=lambda x: (
            x.model_info.quality_level.value,
            -x.model_info.estimated_vram_gb
        ))
        
        self.logger.info(f"Generated {len(fallback_options)} fallback options for {failed_model}")
        return fallback_options
    
    def _is_model_compatible(self, model: ModelInfo, hardware: HardwareProfile, 
                           error_type: str) -> bool:
        """Check if a model is compatible with the hardware given the error type"""
        # Check VRAM requirements
        if error_type in ["CUDA_OUT_OF_MEMORY", "INSUFFICIENT_VRAM"]:
            if model.estimated_vram_gb > hardware.vram_gb * 0.9:  # Leave 10% buffer
                return False
        
        # Check if model requires features not supported by hardware
        if model.requires_trust_remote_code and error_type == "TRUST_REMOTE_CODE_ERROR":
            return False
        
        return True
    
    def _create_fallback_option(self, fallback_model: ModelInfo, failed_model: str,
                              error_type: str, hardware: HardwareProfile) -> FallbackOption:
        """Create a fallback option with detailed information"""
        # Determine quality impact
        if failed_model in self.models_db:
            original_quality = self.models_db[failed_model].quality_level
            quality_impact = self._compare_quality_levels(original_quality, fallback_model.quality_level)
        else:
            quality_impact = "Unknown quality impact"
        
        # Determine performance impact
        performance_impact = self._estimate_performance_impact(fallback_model, hardware)
        
        # Create compatibility notes
        compatibility_notes = self._generate_compatibility_notes(fallback_model, hardware, error_type)
        
        return FallbackOption(
            model_info=fallback_model,
            reason=f"Fallback for {failed_model} due to {error_type}",
            quality_impact=quality_impact,
            performance_impact=performance_impact,
            compatibility_notes=compatibility_notes
        )
    
    def _get_quantized_fallbacks(self, model: ModelInfo, hardware: HardwareProfile) -> List[FallbackOption]:
        """Get quantized versions as fallback options"""
        quantized_options = []
        
        # 8-bit quantization
        if hardware.supports_int8:
            int8_model = ModelInfo(
                name=f"{model.name}-int8",
                model_path=model.model_path,
                model_type=model.model_type,
                size_category=ModelSize.MEDIUM if model.size_category == ModelSize.XLARGE else ModelSize.SMALL,
                estimated_vram_gb=model.estimated_vram_gb * 0.6,  # ~40% reduction
                quality_level=QualityLevel.HIGH if model.quality_level == QualityLevel.HIGHEST else QualityLevel.MEDIUM,
                supports_quantization=True,
                requires_trust_remote_code=model.requires_trust_remote_code,
                min_resolution=model.min_resolution,
                max_resolution=model.max_resolution,
                supported_formats=model.supported_formats,
                description=f"{model.description} (8-bit quantized)"
            )
            
            option = FallbackOption(
                model_info=int8_model,
                reason="8-bit quantization to reduce VRAM usage",
                quality_impact="Slight quality reduction (~5-10%)",
                performance_impact="Faster inference, ~40% less VRAM",
                compatibility_notes="Requires load_in_8bit=True parameter"
            )
            quantized_options.append(option)
        
        # bfloat16 precision
        if hardware.supports_bf16:
            bf16_model = ModelInfo(
                name=f"{model.name}-bf16",
                model_path=model.model_path,
                model_type=model.model_type,
                size_category=model.size_category,
                estimated_vram_gb=model.estimated_vram_gb * 0.8,  # ~20% reduction
                quality_level=model.quality_level,
                supports_quantization=True,
                requires_trust_remote_code=model.requires_trust_remote_code,
                min_resolution=model.min_resolution,
                max_resolution=model.max_resolution,
                supported_formats=model.supported_formats,
                description=f"{model.description} (bfloat16 precision)"
            )
            
            option = FallbackOption(
                model_info=bf16_model,
                reason="bfloat16 precision to reduce VRAM usage",
                quality_impact="Minimal quality impact",
                performance_impact="Similar speed, ~20% less VRAM",
                compatibility_notes="Requires torch_dtype=torch.bfloat16"
            )
            quantized_options.append(option)
        
        return quantized_options
    
    def _compare_quality_levels(self, original: QualityLevel, fallback: QualityLevel) -> str:
        """Compare quality levels and return impact description"""
        quality_order = [QualityLevel.LOWEST, QualityLevel.LOW, QualityLevel.MEDIUM, 
                        QualityLevel.HIGH, QualityLevel.HIGHEST]
        
        original_idx = quality_order.index(original)
        fallback_idx = quality_order.index(fallback)
        
        if fallback_idx == original_idx:
            return "No quality impact"
        elif fallback_idx < original_idx:
            diff = original_idx - fallback_idx
            if diff == 1:
                return "Slight quality reduction"
            elif diff == 2:
                return "Moderate quality reduction"
            else:
                return "Significant quality reduction"
        else:
            return "Quality improvement"
    
    def _estimate_performance_impact(self, model: ModelInfo, hardware: HardwareProfile) -> str:
        """Estimate performance impact of using this model"""
        vram_usage_ratio = model.estimated_vram_gb / hardware.vram_gb
        
        if vram_usage_ratio < 0.5:
            return "Fast generation, low VRAM usage"
        elif vram_usage_ratio < 0.7:
            return "Good performance, moderate VRAM usage"
        elif vram_usage_ratio < 0.9:
            return "Slower generation, high VRAM usage"
        else:
            return "May require CPU offloading, very high VRAM usage"
    
    def _generate_compatibility_notes(self, model: ModelInfo, hardware: HardwareProfile,
                                    error_type: str) -> str:
        """Generate compatibility notes for the model"""
        notes = []
        
        if model.requires_trust_remote_code:
            notes.append("Requires trust_remote_code=True")
        
        if model.estimated_vram_gb > hardware.vram_gb * 0.8:
            notes.append("May benefit from CPU offloading")
        
        if model.size_category == ModelSize.XLARGE and hardware.vram_gb < 16:
            notes.append("Consider using quantization for better performance")
        
        return "; ".join(notes) if notes else "No special requirements"
    
    def recommend_models(self, model_type: ModelType, hardware_profile: HardwareProfile,
                        quality_preference: QualityLevel = QualityLevel.HIGH) -> List[ModelRecommendation]:
        """
        Recommend models based on hardware configuration and preferences.
        
        Args:
            model_type: Type of model needed
            hardware_profile: Hardware configuration
            quality_preference: Preferred quality level
            
        Returns:
            List of model recommendations sorted by suitability
        """
        recommendations = []
        
        # Filter models by type
        compatible_models = [
            model for model in self.models_db.values()
            if model.model_type == model_type
        ]
        
        for model in compatible_models:
            # Calculate compatibility score
            compatibility_score = self._calculate_compatibility_score(model, hardware_profile)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(model, quality_preference)
            
            # Combined confidence score
            confidence_score = (compatibility_score * 0.6 + quality_score * 0.4)
            
            if confidence_score > 0.3:  # Only recommend if reasonably suitable
                recommendation = ModelRecommendation(
                    model_info=model,
                    confidence_score=confidence_score,
                    reasoning=self._generate_recommendation_reasoning(model, hardware_profile),
                    optimization_suggestions=self._generate_optimization_suggestions(model, hardware_profile),
                    expected_performance=self._estimate_performance_impact(model, hardware_profile)
                )
                recommendations.append(recommendation)
        
        # Sort by confidence score
        recommendations.sort(key=lambda x: x.confidence_score, reverse=True)
        
        self.logger.info(f"Generated {len(recommendations)} recommendations for {model_type.value}")
        return recommendations
    
    def _calculate_compatibility_score(self, model: ModelInfo, hardware: HardwareProfile) -> float:
        """Calculate compatibility score between model and hardware"""
        score = 1.0
        
        # VRAM compatibility
        vram_ratio = model.estimated_vram_gb / hardware.vram_gb
        if vram_ratio > 1.0:
            score *= 0.3  # Very challenging
        elif vram_ratio > 0.9:
            score *= 0.6  # Challenging
        elif vram_ratio > 0.7:
            score *= 0.8  # Moderate
        # else: Good compatibility (no penalty)
        
        # Hardware-specific compatibility
        gpu_key = hardware.gpu_model.lower().replace(" ", "").replace("-", "")
        if gpu_key in self.hardware_compatibility:
            model_compat = self.hardware_compatibility[gpu_key].get(model.name, 0.5)
            score *= model_compat
        
        # Feature compatibility
        if model.supports_quantization and hardware.supports_int8:
            score *= 1.1  # Bonus for quantization support
        
        return min(score, 1.0)
    
    def _calculate_quality_score(self, model: ModelInfo, preference: QualityLevel) -> float:
        """Calculate quality score based on preference"""
        quality_order = [QualityLevel.LOWEST, QualityLevel.LOW, QualityLevel.MEDIUM, 
                        QualityLevel.HIGH, QualityLevel.HIGHEST]
        
        model_idx = quality_order.index(model.quality_level)
        pref_idx = quality_order.index(preference)
        
        # Score based on how close the model quality is to preference
        diff = abs(model_idx - pref_idx)
        return max(0.2, 1.0 - (diff * 0.2))
    
    def _generate_recommendation_reasoning(self, model: ModelInfo, hardware: HardwareProfile) -> List[str]:
        """Generate reasoning for model recommendation"""
        reasoning = []
        
        vram_ratio = model.estimated_vram_gb / hardware.vram_gb
        if vram_ratio < 0.7:
            reasoning.append(f"Good VRAM fit ({model.estimated_vram_gb}GB required, {hardware.vram_gb}GB available)")
        elif vram_ratio < 0.9:
            reasoning.append(f"Moderate VRAM usage ({model.estimated_vram_gb}GB required, {hardware.vram_gb}GB available)")
        else:
            reasoning.append(f"High VRAM usage ({model.estimated_vram_gb}GB required, {hardware.vram_gb}GB available)")
        
        reasoning.append(f"Quality level: {model.quality_level.value}")
        reasoning.append(f"Model size: {model.size_category.value}")
        
        if model.supports_quantization:
            reasoning.append("Supports quantization for memory optimization")
        
        return reasoning
    
    def _generate_optimization_suggestions(self, model: ModelInfo, hardware: HardwareProfile) -> List[str]:
        """Generate optimization suggestions for the model"""
        suggestions = []
        
        vram_ratio = model.estimated_vram_gb / hardware.vram_gb
        
        if vram_ratio > 0.8:
            suggestions.append("Enable CPU offloading with device_map='auto'")
            suggestions.append("Use torch.bfloat16 or torch.float16 for reduced memory usage")
        
        if model.supports_quantization and vram_ratio > 0.7:
            suggestions.append("Consider 8-bit quantization with load_in_8bit=True")
        
        if hardware.vram_gb >= 16:
            suggestions.append("Enable attention slicing for memory efficiency")
        
        if model.size_category in [ModelSize.LARGE, ModelSize.XLARGE]:
            suggestions.append("Use low_cpu_mem_usage=True for faster loading")
        
        return suggestions
    
    def validate_generation_input(self, model_name: str, **generation_params) -> InputValidationResult:
        """
        Validate input parameters for image-to-video generation.
        
        Args:
            model_name: Name of the model to use
            **generation_params: Generation parameters to validate
            
        Returns:
            InputValidationResult with validation status and suggestions
        """
        errors = []
        warnings = []
        suggestions = []
        corrected_params = {}
        
        # Get model info
        model = self.models_db.get(model_name)
        if not model:
            errors.append(f"Unknown model: {model_name}")
            return InputValidationResult(False, errors, warnings, suggestions)
        
        # Validate resolution
        width = generation_params.get('width', 512)
        height = generation_params.get('height', 512)
        
        resolution_result = self._validate_resolution(width, height, model)
        errors.extend(resolution_result.get('errors', []))
        warnings.extend(resolution_result.get('warnings', []))
        suggestions.extend(resolution_result.get('suggestions', []))
        corrected_params.update(resolution_result.get('corrections', {}))
        
        # Validate input image (for I2V models)
        if model.model_type == ModelType.IMAGE_TO_VIDEO:
            image_path = generation_params.get('image_path')
            if image_path:
                image_result = self._validate_input_image(image_path)
                errors.extend(image_result.get('errors', []))
                warnings.extend(image_result.get('warnings', []))
                suggestions.extend(image_result.get('suggestions', []))
        
        # Validate generation parameters
        gen_result = self._validate_generation_params(generation_params, model)
        errors.extend(gen_result.get('errors', []))
        warnings.extend(gen_result.get('warnings', []))
        suggestions.extend(gen_result.get('suggestions', []))
        corrected_params.update(gen_result.get('corrections', {}))
        
        is_valid = len(errors) == 0
        
        self.logger.info(f"Input validation for {model_name}: {'PASSED' if is_valid else 'FAILED'}")
        
        return InputValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            corrected_parameters=corrected_params
        )
    
    def _validate_resolution(self, width: int, height: int, model: ModelInfo) -> Dict[str, Any]:
        """Validate resolution parameters"""
        result = {'errors': [], 'warnings': [], 'suggestions': [], 'corrections': {}}
        
        # Check minimum resolution
        if width < model.min_resolution[0] or height < model.min_resolution[1]:
            result['errors'].append(
                f"Resolution {width}x{height} is below minimum {model.min_resolution[0]}x{model.min_resolution[1]}"
            )
            result['corrections']['width'] = max(width, model.min_resolution[0])
            result['corrections']['height'] = max(height, model.min_resolution[1])
        
        # Check maximum resolution
        if width > model.max_resolution[0] or height > model.max_resolution[1]:
            result['errors'].append(
                f"Resolution {width}x{height} exceeds maximum {model.max_resolution[0]}x{model.max_resolution[1]}"
            )
            result['corrections']['width'] = min(width, model.max_resolution[0])
            result['corrections']['height'] = min(height, model.max_resolution[1])
        
        # Check if resolution is multiple of 8 (common requirement)
        if width % 8 != 0 or height % 8 != 0:
            result['warnings'].append("Resolution should be multiple of 8 for optimal performance")
            result['corrections']['width'] = (width // 8) * 8
            result['corrections']['height'] = (height // 8) * 8
        
        # Suggest optimal resolutions
        aspect_ratio = width / height
        if not (0.5 <= aspect_ratio <= 2.0):
            result['warnings'].append("Unusual aspect ratio may affect quality")
            result['suggestions'].append("Consider using standard aspect ratios like 16:9, 1:1, or 4:3")
        
        return result
    
    def _validate_input_image(self, image_path: str) -> Dict[str, Any]:
        """Validate input image for I2V generation"""
        result = {'errors': [], 'warnings': [], 'suggestions': []}
        
        if not os.path.exists(image_path):
            result['errors'].append(f"Input image not found: {image_path}")
            return result
        
        # Check file extension
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in self.validation_rules['formats']['supported_image']:
            result['errors'].append(f"Unsupported image format: {ext}")
            result['suggestions'].append(f"Use supported formats: {', '.join(self.validation_rules['formats']['recommended_image'])}")
        
        # Check file size (basic check)
        try:
            file_size = os.path.getsize(image_path)
            if file_size > 50 * 1024 * 1024:  # 50MB
                result['warnings'].append("Large image file may slow down processing")
                result['suggestions'].append("Consider resizing image to reduce file size")
        except OSError:
            result['warnings'].append("Could not check image file size")
        
        return result
    
    def _validate_generation_params(self, params: Dict[str, Any], model: ModelInfo) -> Dict[str, Any]:
        """Validate generation parameters"""
        result = {'errors': [], 'warnings': [], 'suggestions': [], 'corrections': {}}
        
        # Validate number of frames
        num_frames = params.get('num_frames', 16)
        if num_frames < self.validation_rules['generation']['min_frames']:
            result['errors'].append(f"Number of frames ({num_frames}) below minimum ({self.validation_rules['generation']['min_frames']})")
            result['corrections']['num_frames'] = self.validation_rules['generation']['min_frames']
        elif num_frames > self.validation_rules['generation']['max_frames']:
            result['errors'].append(f"Number of frames ({num_frames}) exceeds maximum ({self.validation_rules['generation']['max_frames']})")
            result['corrections']['num_frames'] = self.validation_rules['generation']['max_frames']
        
        # Validate FPS
        fps = params.get('fps', 16)
        if fps < self.validation_rules['generation']['min_fps']:
            result['warnings'].append(f"Low FPS ({fps}) may result in choppy video")
            result['suggestions'].append(f"Consider using FPS >= {self.validation_rules['generation']['min_fps']}")
        elif fps > self.validation_rules['generation']['max_fps']:
            result['warnings'].append(f"High FPS ({fps}) may increase generation time")
        
        # Validate guidance scale
        guidance_scale = params.get('guidance_scale', 7.5)
        if guidance_scale < 1.0:
            result['warnings'].append("Very low guidance scale may produce poor quality")
        elif guidance_scale > 20.0:
            result['warnings'].append("Very high guidance scale may cause artifacts")
        
        return result


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    fallback_system = ModelFallbackSystem()
    
    # Example hardware profile (RTX 4080)
    hardware = HardwareProfile(
        gpu_model="RTX 4080",
        vram_gb=16,
        cpu_cores=16,
        ram_gb=32,
        supports_bf16=True,
        supports_int8=True
    )
    
    # Test fallback options
    fallbacks = fallback_system.get_fallback_options(
        "wan22-ti2v-5b", 
        "CUDA_OUT_OF_MEMORY", 
        hardware
    )
    
    print(f"Found {len(fallbacks)} fallback options:")
    for i, option in enumerate(fallbacks):
        print(f"{i+1}. {option.model_info.name}")
        print(f"   Reason: {option.reason}")
        print(f"   Quality Impact: {option.quality_impact}")
        print(f"   Performance: {option.performance_impact}")
        print()
    
    # Test model recommendations
    recommendations = fallback_system.recommend_models(
        ModelType.TEXT_TO_VIDEO,
        hardware,
        QualityLevel.HIGH
    )
    
    print(f"Found {len(recommendations)} model recommendations:")
    for i, rec in enumerate(recommendations):
        print(f"{i+1}. {rec.model_info.name} (confidence: {rec.confidence_score:.2f})")
        print(f"   Reasoning: {'; '.join(rec.reasoning)}")
        print(f"   Optimizations: {'; '.join(rec.optimization_suggestions)}")
        print()
    
    # Test input validation
    validation = fallback_system.validate_generation_input(
        "wan22-ti2v-5b",
        width=1280,
        height=720,
        num_frames=24,
        fps=16
    )
    
    print(f"Input validation: {'PASSED' if validation.is_valid else 'FAILED'}")
    if validation.errors:
        print(f"Errors: {validation.errors}")
    if validation.warnings:
        print(f"Warnings: {validation.warnings}")
    if validation.suggestions:
        print(f"Suggestions: {validation.suggestions}")
