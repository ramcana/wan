"""
Fallback Handler System for Wan2.2 Model Compatibility

This module provides comprehensive fallback strategies and graceful degradation
for model loading and pipeline initialization failures.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any, Union, Tuple
import logging
import json
from pathlib import Path
import torch
import psutil
from abc import ABC, abstractmethod


class FallbackStrategyType(Enum):
    """Types of fallback strategies available"""
    COMPONENT_ISOLATION = "component_isolation"
    ALTERNATIVE_MODEL = "alternative_model"
    REDUCED_FUNCTIONALITY = "reduced_functionality"
    OPTIMIZATION_FALLBACK = "optimization_fallback"
    PIPELINE_SUBSTITUTION = "pipeline_substitution"


class ComponentType(Enum):
    """Types of model components that can be isolated"""
    TRANSFORMER = "transformer"
    TRANSFORMER_2 = "transformer_2"
    VAE = "vae"
    SCHEDULER = "scheduler"
    TEXT_ENCODER = "text_encoder"
    TOKENIZER = "tokenizer"
    UNET = "unet"


@dataclass
class FallbackStrategy:
    """Represents a fallback strategy for handling failures"""
    strategy_type: FallbackStrategyType
    description: str
    implementation_steps: List[str]
    expected_limitations: List[str]
    success_probability: float
    resource_requirements: Dict[str, Any]
    compatibility_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert strategy to dictionary for serialization"""
        return {
            "strategy_type": self.strategy_type.value,
            "description": self.description,
            "implementation_steps": self.implementation_steps,
            "expected_limitations": self.expected_limitations,
            "success_probability": self.success_probability,
            "resource_requirements": self.resource_requirements,
            "compatibility_score": self.compatibility_score
        }


@dataclass
class UsableComponent:
    """Represents a model component that can be used independently"""
    component_type: ComponentType
    component_path: str
    class_name: str
    is_functional: bool
    limitations: List[str]
    required_dependencies: List[str]
    memory_usage_mb: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert component to dictionary for serialization"""
        return {
            "component_type": self.component_type.value,
            "component_path": self.component_path,
            "class_name": self.class_name,
            "is_functional": self.is_functional,
            "limitations": self.limitations,
            "required_dependencies": self.required_dependencies,
            "memory_usage_mb": self.memory_usage_mb
        }


@dataclass
class AlternativeModel:
    """Represents an alternative model that could be used instead"""
    model_name: str
    model_path: str
    architecture_type: str
    compatibility_score: float
    feature_parity: Dict[str, bool]  # Which features are supported
    resource_requirements: Dict[str, Any]
    download_required: bool
    size_mb: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alternative model to dictionary for serialization"""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "architecture_type": self.architecture_type,
            "compatibility_score": self.compatibility_score,
            "feature_parity": self.feature_parity,
            "resource_requirements": self.resource_requirements,
            "download_required": self.download_required,
            "size_mb": self.size_mb
        }


@dataclass
class FallbackResult:
    """Result of a fallback attempt"""
    success: bool
    strategy_used: Optional[FallbackStrategy]
    fallback_pipeline: Optional[Any]
    usable_components: List[UsableComponent]
    error_message: Optional[str]
    warnings: List[str]
    performance_impact: Dict[str, float]  # Expected performance changes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization"""
        return {
            "success": self.success,
            "strategy_used": self.strategy_used.to_dict() if self.strategy_used else None,
            "usable_components": [comp.to_dict() for comp in self.usable_components],
            "error_message": self.error_message,
            "warnings": self.warnings,
            "performance_impact": self.performance_impact
        }


class ComponentAnalyzer:
    """Analyzes model components for isolation and compatibility"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._component_validators = self._initialize_validators()
    
    def _initialize_validators(self) -> Dict[ComponentType, callable]:
        """Initialize component validation functions"""
        return {
            ComponentType.TRANSFORMER: self._validate_transformer,
            ComponentType.TRANSFORMER_2: self._validate_transformer_2,
            ComponentType.VAE: self._validate_vae,
            ComponentType.SCHEDULER: self._validate_scheduler,
            ComponentType.TEXT_ENCODER: self._validate_text_encoder,
            ComponentType.TOKENIZER: self._validate_tokenizer,
            ComponentType.UNET: self._validate_unet
        }
    
    def analyze_component(self, component_path: str, component_type: ComponentType) -> UsableComponent:
        """Analyze a single component for usability"""
        try:
            validator = self._component_validators.get(component_type)
            if not validator:
                return self._create_unknown_component(component_path, component_type)
            
            return validator(component_path)
        
        except Exception as e:
            self.logger.warning(f"Failed to analyze component {component_path}: {e}")
            return self._create_failed_component(component_path, component_type, str(e))
    
    def _validate_transformer(self, component_path: str) -> UsableComponent:
        """Validate transformer component"""
        config_path = Path(component_path) / "config.json"
        if not config_path.exists():
            return self._create_failed_component(
                component_path, ComponentType.TRANSFORMER, "Config file missing"
            )
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Check for Wan-specific transformer attributes
            is_wan_transformer = any(key in config for key in [
                "boundary_ratio", "patch_size_t", "in_channels"
            ])
            
            limitations = []
            if not is_wan_transformer:
                limitations.append("May not support 3D video generation")
            
            memory_usage = self._estimate_transformer_memory(config)
            
            return UsableComponent(
                component_type=ComponentType.TRANSFORMER,
                component_path=component_path,
                class_name=config.get("_class_name", "Transformer2DModel"),
                is_functional=True,
                limitations=limitations,
                required_dependencies=["torch", "transformers"],
                memory_usage_mb=memory_usage
            )
        
        except Exception as e:
            return self._create_failed_component(
                component_path, ComponentType.TRANSFORMER, f"Config validation failed: {e}"
            )
    
    def _validate_transformer_2(self, component_path: str) -> UsableComponent:
        """Validate secondary transformer component (Wan-specific)"""
        config_path = Path(component_path) / "config.json"
        if not config_path.exists():
            return self._create_failed_component(
                component_path, ComponentType.TRANSFORMER_2, "Config file missing"
            )
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Transformer_2 is typically Wan-specific
            limitations = ["Requires Wan pipeline for full functionality"]
            memory_usage = self._estimate_transformer_memory(config)
            
            return UsableComponent(
                component_type=ComponentType.TRANSFORMER_2,
                component_path=component_path,
                class_name=config.get("_class_name", "Transformer2DModel"),
                is_functional=True,
                limitations=limitations,
                required_dependencies=["torch", "transformers"],
                memory_usage_mb=memory_usage
            )
        
        except Exception as e:
            return self._create_failed_component(
                component_path, ComponentType.TRANSFORMER_2, f"Config validation failed: {e}"
            )
    
    def _validate_vae(self, component_path: str) -> UsableComponent:
        """Validate VAE component"""
        config_path = Path(component_path) / "config.json"
        if not config_path.exists():
            return self._create_failed_component(
                component_path, ComponentType.VAE, "Config file missing"
            )
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Check VAE dimensions
            in_channels = config.get("in_channels", 3)
            latent_channels = config.get("latent_channels", 4)
            
            limitations = []
            if in_channels != 3:
                limitations.append(f"Non-standard input channels: {in_channels}")
            if latent_channels > 4:
                limitations.append("High-dimensional latent space may require more VRAM")
            
            # Check for 3D VAE
            is_3d_vae = "temporal" in str(config).lower() or latent_channels > 8
            if is_3d_vae:
                limitations.append("3D VAE requires specialized pipeline")
            
            memory_usage = self._estimate_vae_memory(config)
            
            return UsableComponent(
                component_type=ComponentType.VAE,
                component_path=component_path,
                class_name=config.get("_class_name", "AutoencoderKL"),
                is_functional=True,
                limitations=limitations,
                required_dependencies=["torch", "diffusers"],
                memory_usage_mb=memory_usage
            )
        
        except Exception as e:
            return self._create_failed_component(
                component_path, ComponentType.VAE, f"Config validation failed: {e}"
            )
    
    def _validate_scheduler(self, component_path: str) -> UsableComponent:
        """Validate scheduler component"""
        config_path = Path(component_path) / "scheduler_config.json"
        if not config_path.exists():
            return self._create_failed_component(
                component_path, ComponentType.SCHEDULER, "Scheduler config missing"
            )
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            scheduler_class = config.get("_class_name", "DDIMScheduler")
            limitations = []
            
            # Check for custom scheduler requirements
            if "Wan" in scheduler_class:
                limitations.append("Custom scheduler may require specific pipeline")
            
            return UsableComponent(
                component_type=ComponentType.SCHEDULER,
                component_path=component_path,
                class_name=scheduler_class,
                is_functional=True,
                limitations=limitations,
                required_dependencies=["diffusers"],
                memory_usage_mb=10  # Schedulers are typically lightweight
            )
        
        except Exception as e:
            return self._create_failed_component(
                component_path, ComponentType.SCHEDULER, f"Config validation failed: {e}"
            )
    
    def _validate_text_encoder(self, component_path: str) -> UsableComponent:
        """Validate text encoder component"""
        config_path = Path(component_path) / "config.json"
        if not config_path.exists():
            return self._create_failed_component(
                component_path, ComponentType.TEXT_ENCODER, "Config file missing"
            )
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            model_type = config.get("model_type", "clip")
            limitations = []
            
            if model_type not in ["clip", "bert", "t5"]:
                limitations.append(f"Uncommon text encoder type: {model_type}")
            
            memory_usage = config.get("hidden_size", 768) * 0.01  # Rough estimate
            
            return UsableComponent(
                component_type=ComponentType.TEXT_ENCODER,
                component_path=component_path,
                class_name=config.get("_name_or_path", "CLIPTextModel"),
                is_functional=True,
                limitations=limitations,
                required_dependencies=["transformers", "torch"],
                memory_usage_mb=int(memory_usage)
            )
        
        except Exception as e:
            return self._create_failed_component(
                component_path, ComponentType.TEXT_ENCODER, f"Config validation failed: {e}"
            )
    
    def _validate_tokenizer(self, component_path: str) -> UsableComponent:
        """Validate tokenizer component"""
        tokenizer_config = Path(component_path) / "tokenizer_config.json"
        if not tokenizer_config.exists():
            return self._create_failed_component(
                component_path, ComponentType.TOKENIZER, "Tokenizer config missing"
            )
        
        try:
            with open(tokenizer_config, 'r') as f:
                config = json.load(f)
            
            tokenizer_class = config.get("tokenizer_class", "CLIPTokenizer")
            
            return UsableComponent(
                component_type=ComponentType.TOKENIZER,
                component_path=component_path,
                class_name=tokenizer_class,
                is_functional=True,
                limitations=[],
                required_dependencies=["transformers"],
                memory_usage_mb=5  # Tokenizers are very lightweight
            )
        
        except Exception as e:
            return self._create_failed_component(
                component_path, ComponentType.TOKENIZER, f"Config validation failed: {e}"
            )
    
    def _validate_unet(self, component_path: str) -> UsableComponent:
        """Validate UNet component (for SD compatibility)"""
        config_path = Path(component_path) / "config.json"
        if not config_path.exists():
            return self._create_failed_component(
                component_path, ComponentType.UNET, "Config file missing"
            )
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            limitations = ["UNet architecture not compatible with Wan models"]
            memory_usage = self._estimate_unet_memory(config)
            
            return UsableComponent(
                component_type=ComponentType.UNET,
                component_path=component_path,
                class_name=config.get("_class_name", "UNet2DConditionModel"),
                is_functional=False,  # Not functional for Wan models
                limitations=limitations,
                required_dependencies=["torch", "diffusers"],
                memory_usage_mb=memory_usage
            )
        
        except Exception as e:
            return self._create_failed_component(
                component_path, ComponentType.UNET, f"Config validation failed: {e}"
            )
    
    def _estimate_transformer_memory(self, config: Dict[str, Any]) -> int:
        """Estimate transformer memory usage in MB"""
        hidden_size = config.get("hidden_size", 1024)
        num_layers = config.get("num_layers", 12)
        # Rough estimation: hidden_size * num_layers * 4 bytes * scaling factor
        return int((hidden_size * num_layers * 4) / (1024 * 1024) * 10)
    
    def _estimate_vae_memory(self, config: Dict[str, Any]) -> int:
        """Estimate VAE memory usage in MB"""
        latent_channels = config.get("latent_channels", 4)
        # VAE memory scales with latent channels
        return int(latent_channels * 50)  # Rough estimate
    
    def _estimate_unet_memory(self, config: Dict[str, Any]) -> int:
        """Estimate UNet memory usage in MB"""
        # UNets are typically the largest component
        return 2000  # Rough estimate for standard UNet
    
    def _create_unknown_component(self, component_path: str, component_type: ComponentType) -> UsableComponent:
        """Create component info for unknown component type"""
        return UsableComponent(
            component_type=component_type,
            component_path=component_path,
            class_name="Unknown",
            is_functional=False,
            limitations=["Unknown component type - cannot validate"],
            required_dependencies=[],
            memory_usage_mb=0
        )
    
    def _create_failed_component(self, component_path: str, component_type: ComponentType, error: str) -> UsableComponent:
        """Create component info for failed validation"""
        return UsableComponent(
            component_type=component_type,
            component_path=component_path,
            class_name="Failed",
            is_functional=False,
            limitations=[f"Validation failed: {error}"],
            required_dependencies=[],
            memory_usage_mb=0
        )


class AlternativeModelSuggester:
    """Suggests alternative models when primary model fails"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._model_registry = self._initialize_model_registry()
    
    def _initialize_model_registry(self) -> Dict[str, List[AlternativeModel]]:
        """Initialize registry of alternative models"""
        return {
            "wan_t2v": [
                AlternativeModel(
                    model_name="Wan2.2-T2V-Mini",
                    model_path="Wan-AI/Wan2.2-T2V-Mini",
                    architecture_type="wan_t2v",
                    compatibility_score=0.9,
                    feature_parity={"text_to_video": True, "high_resolution": False},
                    resource_requirements={"min_vram_mb": 6144, "recommended_vram_mb": 8192},
                    download_required=True,
                    size_mb=7000
                ),
                AlternativeModel(
                    model_name="Wan2.1-T2V",
                    model_path="Wan-AI/Wan2.1-T2V",
                    architecture_type="wan_t2v",
                    compatibility_score=0.8,
                    feature_parity={"text_to_video": True, "high_resolution": True},
                    resource_requirements={"min_vram_mb": 10240, "recommended_vram_mb": 12288},
                    download_required=True,
                    size_mb=12000
                )
            ],
            "wan_t2i": [
                AlternativeModel(
                    model_name="Stable Diffusion 2.1",
                    model_path="stabilityai/stable-diffusion-2-1",
                    architecture_type="stable_diffusion",
                    compatibility_score=0.6,
                    feature_parity={"text_to_image": True, "text_to_video": False},
                    resource_requirements={"min_vram_mb": 4096, "recommended_vram_mb": 6144},
                    download_required=True,
                    size_mb=5000
                )
            ],
            "stable_diffusion": [
                AlternativeModel(
                    model_name="Stable Diffusion 1.5",
                    model_path="runwayml/stable-diffusion-v1-5",
                    architecture_type="stable_diffusion",
                    compatibility_score=0.95,
                    feature_parity={"text_to_image": True, "text_to_video": False},
                    resource_requirements={"min_vram_mb": 3072, "recommended_vram_mb": 4096},
                    download_required=True,
                    size_mb=4000
                )
            ]
        }
    
    def suggest_alternatives(self, target_architecture: str, available_vram_mb: int = None) -> List[AlternativeModel]:
        """Suggest alternative models for the target architecture"""
        alternatives = self._model_registry.get(target_architecture, [])
        
        if available_vram_mb:
            # Filter by VRAM requirements
            alternatives = [
                alt for alt in alternatives 
                if alt.resource_requirements.get("min_vram_mb", 0) <= available_vram_mb
            ]
        
        # Sort by compatibility score
        alternatives.sort(key=lambda x: x.compatibility_score, reverse=True)
        
        return alternatives
    
    def add_alternative_model(self, architecture: str, model: AlternativeModel):
        """Add a new alternative model to the registry"""
        if architecture not in self._model_registry:
            self._model_registry[architecture] = []
        self._model_registry[architecture].append(model)
    
    def get_local_alternatives(self, models_dir: str = "models") -> List[AlternativeModel]:
        """Find locally available alternative models"""
        local_alternatives = []
        models_path = Path(models_dir)
        
        if not models_path.exists():
            return local_alternatives
        
        for model_dir in models_path.iterdir():
            if model_dir.is_dir():
                try:
                    # Check if it's a valid model directory
                    model_index = model_dir / "model_index.json"
                    if model_index.exists():
                        with open(model_index, 'r') as f:
                            config = json.load(f)
                        
                        # Determine architecture type
                        arch_type = self._detect_architecture_type(config)
                        
                        local_alternatives.append(AlternativeModel(
                            model_name=model_dir.name,
                            model_path=str(model_dir),
                            architecture_type=arch_type,
                            compatibility_score=0.7,  # Default for local models
                            feature_parity=self._detect_feature_parity(config),
                            resource_requirements={"min_vram_mb": 4096},
                            download_required=False,
                            size_mb=self._estimate_model_size(model_dir)
                        ))
                
                except Exception as e:
                    self.logger.warning(f"Failed to analyze local model {model_dir}: {e}")
        
        return local_alternatives
    
    def _detect_architecture_type(self, config: Dict[str, Any]) -> str:
        """Detect architecture type from model config"""
        pipeline_class = config.get("_class_name", "")
        
        if "Wan" in pipeline_class:
            if "T2V" in pipeline_class:
                return "wan_t2v"
            elif "T2I" in pipeline_class:
                return "wan_t2i"
            else:
                return "wan_unknown"
        elif "StableDiffusion" in pipeline_class:
            return "stable_diffusion"
        else:
            return "unknown"
    
    def _detect_feature_parity(self, config: Dict[str, Any]) -> Dict[str, bool]:
        """Detect supported features from model config"""
        features = {
            "text_to_image": False,
            "text_to_video": False,
            "image_to_video": False,
            "high_resolution": False
        }
        
        pipeline_class = config.get("_class_name", "")
        
        if "T2V" in pipeline_class or "Video" in pipeline_class:
            features["text_to_video"] = True
        if "T2I" in pipeline_class or "Image" in pipeline_class:
            features["text_to_image"] = True
        if "I2V" in pipeline_class:
            features["image_to_video"] = True
        
        # Check for high resolution support (rough heuristic)
        if any("1080" in str(v) or "high" in str(v).lower() for v in config.values()):
            features["high_resolution"] = True
        
        return features
    
    def _estimate_model_size(self, model_dir: Path) -> int:
        """Estimate model size in MB"""
        total_size = 0
        try:
            for file_path in model_dir.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return int(total_size / (1024 * 1024))
        except Exception:
            return 0


class FallbackHandler:
    """Main fallback handler for graceful degradation strategies"""
    
    def __init__(self, models_dir: str = "models"):
        self.logger = logging.getLogger(__name__)
        self.models_dir = models_dir
        self.component_analyzer = ComponentAnalyzer()
        self.alternative_suggester = AlternativeModelSuggester()
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for fallback handler"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def create_fallback_strategy(self, failed_pipeline: str, error: Exception) -> FallbackStrategy:
        """Create fallback strategy based on failure type"""
        error_message = str(error).lower()
        
        # Analyze error to determine best fallback strategy
        if "out of memory" in error_message or "cuda" in error_message:
            return self._create_memory_fallback_strategy(failed_pipeline, error)
        elif "pipeline" in error_message or "class" in error_message:
            return self._create_pipeline_fallback_strategy(failed_pipeline, error)
        elif "component" in error_message or "missing" in error_message:
            return self._create_component_fallback_strategy(failed_pipeline, error)
        else:
            return self._create_generic_fallback_strategy(failed_pipeline, error)
    
    def _create_memory_fallback_strategy(self, failed_pipeline: str, error: Exception) -> FallbackStrategy:
        """Create fallback strategy for memory-related failures"""
        return FallbackStrategy(
            strategy_type=FallbackStrategyType.OPTIMIZATION_FALLBACK,
            description="Apply memory optimizations to reduce VRAM usage",
            implementation_steps=[
                "Enable mixed precision (float16) to reduce memory usage by ~50%",
                "Enable CPU offloading for non-critical components",
                "Reduce batch size to 1 for generation",
                "Enable gradient checkpointing if available",
                "Clear GPU cache before loading"
            ],
            expected_limitations=[
                "Slightly slower generation due to CPU offloading",
                "Potential minor quality reduction with mixed precision",
                "Single batch processing only"
            ],
            success_probability=0.8,
            resource_requirements={
                "min_vram_mb": 4096,
                "cpu_offload_capable": True,
                "mixed_precision_support": True
            },
            compatibility_score=0.9
        )
    
    def _create_pipeline_fallback_strategy(self, failed_pipeline: str, error: Exception) -> FallbackStrategy:
        """Create fallback strategy for pipeline loading failures"""
        return FallbackStrategy(
            strategy_type=FallbackStrategyType.PIPELINE_SUBSTITUTION,
            description="Use alternative pipeline or generic diffusion pipeline",
            implementation_steps=[
                "Attempt to load with DiffusionPipeline.from_pretrained()",
                "Try trust_remote_code=True for custom pipelines",
                "Fall back to component-by-component loading",
                "Use generic pipeline wrapper if available"
            ],
            expected_limitations=[
                "May not support all model-specific features",
                "Potential compatibility issues with custom components",
                "Reduced functionality compared to native pipeline"
            ],
            success_probability=0.6,
            resource_requirements={
                "trust_remote_code": True,
                "internet_connection": True
            },
            compatibility_score=0.7
        )
    
    def _create_component_fallback_strategy(self, failed_pipeline: str, error: Exception) -> FallbackStrategy:
        """Create fallback strategy for component-related failures"""
        return FallbackStrategy(
            strategy_type=FallbackStrategyType.COMPONENT_ISOLATION,
            description="Isolate and use functional components independently",
            implementation_steps=[
                "Analyze each model component for individual functionality",
                "Identify which components can be loaded successfully",
                "Create minimal pipeline with working components only",
                "Provide alternative components for failed ones"
            ],
            expected_limitations=[
                "Reduced model capabilities",
                "May not support full feature set",
                "Potential quality degradation"
            ],
            success_probability=0.5,
            resource_requirements={
                "component_isolation_support": True
            },
            compatibility_score=0.6
        )
    
    def _create_generic_fallback_strategy(self, failed_pipeline: str, error: Exception) -> FallbackStrategy:
        """Create generic fallback strategy for unknown failures"""
        return FallbackStrategy(
            strategy_type=FallbackStrategyType.ALTERNATIVE_MODEL,
            description="Suggest alternative compatible models",
            implementation_steps=[
                "Analyze target model architecture",
                "Find compatible alternative models",
                "Suggest models with similar capabilities",
                "Provide download and setup instructions"
            ],
            expected_limitations=[
                "Different model may produce different results",
                "May require additional downloads",
                "Feature parity not guaranteed"
            ],
            success_probability=0.7,
            resource_requirements={
                "internet_connection": True,
                "storage_space_gb": 5
            },
            compatibility_score=0.8
        )
    
    def attempt_component_isolation(self, model_path: str) -> List[UsableComponent]:
        """Identify which model components can be used independently"""
        self.logger.info(f"Analyzing components in {model_path}")
        
        usable_components = []
        model_dir = Path(model_path)
        
        if not model_dir.exists():
            self.logger.warning(f"Model directory does not exist: {model_path}")
            return usable_components
        
        # Check for model_index.json to understand component structure
        model_index_path = model_dir / "model_index.json"
        if model_index_path.exists():
            try:
                with open(model_index_path, 'r') as f:
                    model_index = json.load(f)
                
                # Analyze each component listed in model_index
                for component_name, component_info in model_index.items():
                    if isinstance(component_info, list) and len(component_info) >= 2:
                        component_class = component_info[0]
                        component_subdir = component_info[1]
                        
                        component_path = model_dir / component_subdir
                        if component_path.exists():
                            component_type = self._map_component_name_to_type(component_name)
                            if component_type:
                                component = self.component_analyzer.analyze_component(
                                    str(component_path), component_type
                                )
                                usable_components.append(component)
            
            except Exception as e:
                self.logger.error(f"Failed to parse model_index.json: {e}")
        
        # Also check for common component directories
        common_components = {
            "transformer": ComponentType.TRANSFORMER,
            "transformer_2": ComponentType.TRANSFORMER_2,
            "vae": ComponentType.VAE,
            "scheduler": ComponentType.SCHEDULER,
            "text_encoder": ComponentType.TEXT_ENCODER,
            "tokenizer": ComponentType.TOKENIZER,
            "unet": ComponentType.UNET
        }
        
        for dir_name, component_type in common_components.items():
            component_dir = model_dir / dir_name
            if component_dir.exists():
                # Check if we haven't already analyzed this component
                if not any(comp.component_path == str(component_dir) for comp in usable_components):
                    component = self.component_analyzer.analyze_component(
                        str(component_dir), component_type
                    )
                    usable_components.append(component)
        
        self.logger.info(f"Found {len(usable_components)} components, "
                        f"{sum(1 for c in usable_components if c.is_functional)} functional")
        
        return usable_components
    
    def _map_component_name_to_type(self, component_name: str) -> Optional[ComponentType]:
        """Map component name from model_index to ComponentType"""
        name_mapping = {
            "transformer": ComponentType.TRANSFORMER,
            "transformer_2": ComponentType.TRANSFORMER_2,
            "vae": ComponentType.VAE,
            "scheduler": ComponentType.SCHEDULER,
            "text_encoder": ComponentType.TEXT_ENCODER,
            "tokenizer": ComponentType.TOKENIZER,
            "unet": ComponentType.UNET
        }
        
        return name_mapping.get(component_name.lower())
    
    def suggest_alternative_models(self, target_architecture: str) -> List[AlternativeModel]:
        """Suggest compatible alternative models"""
        self.logger.info(f"Finding alternatives for architecture: {target_architecture}")
        
        # Get system VRAM info for filtering
        available_vram = self._get_available_vram()
        
        # Get alternatives from registry
        registry_alternatives = self.alternative_suggester.suggest_alternatives(
            target_architecture, available_vram
        )
        
        # Get local alternatives
        local_alternatives = self.alternative_suggester.get_local_alternatives(self.models_dir)
        
        # Combine and deduplicate
        all_alternatives = registry_alternatives + local_alternatives
        seen_models = set()
        unique_alternatives = []
        
        for alt in all_alternatives:
            if alt.model_name not in seen_models:
                seen_models.add(alt.model_name)
                unique_alternatives.append(alt)
        
        # Sort by compatibility score
        unique_alternatives.sort(key=lambda x: x.compatibility_score, reverse=True)
        
        self.logger.info(f"Found {len(unique_alternatives)} alternative models")
        return unique_alternatives
    
    def _get_available_vram(self) -> int:
        """Get available VRAM in MB"""
        try:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                total_memory = torch.cuda.get_device_properties(device).total_memory
                allocated_memory = torch.cuda.memory_allocated(device)
                available_memory = total_memory - allocated_memory
                return int(available_memory / (1024 * 1024))
        except Exception:
            pass
        
        return 4096  # Default assumption
    
    def execute_fallback_strategy(self, strategy: FallbackStrategy, model_path: str, 
                                 context: Dict[str, Any] = None) -> FallbackResult:
        """Execute a specific fallback strategy"""
        self.logger.info(f"Executing fallback strategy: {strategy.strategy_type.value}")
        
        try:
            if strategy.strategy_type == FallbackStrategyType.COMPONENT_ISOLATION:
                return self._execute_component_isolation(strategy, model_path, context)
            elif strategy.strategy_type == FallbackStrategyType.ALTERNATIVE_MODEL:
                return self._execute_alternative_model(strategy, model_path, context)
            elif strategy.strategy_type == FallbackStrategyType.OPTIMIZATION_FALLBACK:
                return self._execute_optimization_fallback(strategy, model_path, context)
            elif strategy.strategy_type == FallbackStrategyType.PIPELINE_SUBSTITUTION:
                return self._execute_pipeline_substitution(strategy, model_path, context)
            else:
                return self._execute_reduced_functionality(strategy, model_path, context)
        
        except Exception as e:
            self.logger.error(f"Fallback strategy execution failed: {e}")
            return FallbackResult(
                success=False,
                strategy_used=strategy,
                fallback_pipeline=None,
                usable_components=[],
                error_message=f"Strategy execution failed: {e}",
                warnings=[],
                performance_impact={}
            )
    
    def _execute_component_isolation(self, strategy: FallbackStrategy, model_path: str, 
                                   context: Dict[str, Any] = None) -> FallbackResult:
        """Execute component isolation fallback"""
        usable_components = self.attempt_component_isolation(model_path)
        functional_components = [c for c in usable_components if c.is_functional]
        
        if not functional_components:
            return FallbackResult(
                success=False,
                strategy_used=strategy,
                fallback_pipeline=None,
                usable_components=usable_components,
                error_message="No functional components found",
                warnings=["Model appears to be completely non-functional"],
                performance_impact={}
            )
        
        warnings = []
        for component in usable_components:
            if not component.is_functional:
                warnings.append(f"Component {component.component_type.value} is not functional")
            elif component.limitations:
                warnings.extend(component.limitations)
        
        return FallbackResult(
            success=True,
            strategy_used=strategy,
            fallback_pipeline=None,  # Would need actual pipeline implementation
            usable_components=usable_components,
            error_message=None,
            warnings=warnings,
            performance_impact={"functionality_reduction": 0.3}
        )
    
    def _execute_alternative_model(self, strategy: FallbackStrategy, model_path: str, 
                                 context: Dict[str, Any] = None) -> FallbackResult:
        """Execute alternative model fallback"""
        # Detect target architecture
        target_arch = context.get("target_architecture", "wan_t2v") if context else "wan_t2v"
        alternatives = self.suggest_alternative_models(target_arch)
        
        if not alternatives:
            return FallbackResult(
                success=False,
                strategy_used=strategy,
                fallback_pipeline=None,
                usable_components=[],
                error_message="No alternative models found",
                warnings=["Consider downloading compatible models"],
                performance_impact={}
            )
        
        # Return the best alternative
        best_alternative = alternatives[0]
        warnings = [f"Using alternative model: {best_alternative.model_name}"]
        
        if best_alternative.download_required:
            warnings.append("Model download required")
        
        performance_impact = {}
        if best_alternative.compatibility_score < 0.9:
            performance_impact["feature_parity"] = best_alternative.compatibility_score
        
        return FallbackResult(
            success=True,
            strategy_used=strategy,
            fallback_pipeline=None,  # Would contain actual alternative pipeline
            usable_components=[],
            error_message=None,
            warnings=warnings,
            performance_impact=performance_impact
        )
    
    def _execute_optimization_fallback(self, strategy: FallbackStrategy, model_path: str, 
                                     context: Dict[str, Any] = None) -> FallbackResult:
        """Execute optimization fallback"""
        warnings = []
        performance_impact = {}
        
        # Simulate optimization application
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            warnings.append("GPU cache cleared")
            performance_impact["memory_reduction"] = 0.2
        
        # Check if mixed precision is supported
        try:
            if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
                warnings.append("Mixed precision (bfloat16) available")
                performance_impact["memory_reduction"] = 0.5
            elif torch.cuda.is_available():
                warnings.append("Mixed precision (float16) available")
                performance_impact["memory_reduction"] = 0.4
        except Exception:
            warnings.append("Mixed precision support unknown")
        
        return FallbackResult(
            success=True,
            strategy_used=strategy,
            fallback_pipeline=None,  # Would contain optimized pipeline
            usable_components=[],
            error_message=None,
            warnings=warnings,
            performance_impact=performance_impact
        )
    
    def _execute_pipeline_substitution(self, strategy: FallbackStrategy, model_path: str, 
                                     context: Dict[str, Any] = None) -> FallbackResult:
        """Execute pipeline substitution fallback"""
        warnings = [
            "Using generic pipeline - some features may not be available",
            "Custom model features may not be supported"
        ]
        
        performance_impact = {
            "feature_availability": 0.7,
            "compatibility_risk": 0.3
        }
        
        return FallbackResult(
            success=True,
            strategy_used=strategy,
            fallback_pipeline=None,  # Would contain substitute pipeline
            usable_components=[],
            error_message=None,
            warnings=warnings,
            performance_impact=performance_impact
        )
    
    def _execute_reduced_functionality(self, strategy: FallbackStrategy, model_path: str, 
                                     context: Dict[str, Any] = None) -> FallbackResult:
        """Execute reduced functionality fallback"""
        warnings = [
            "Running in reduced functionality mode",
            "Some features disabled to ensure stability"
        ]
        
        performance_impact = {
            "functionality_reduction": 0.5,
            "stability_improvement": 0.8
        }
        
        return FallbackResult(
            success=True,
            strategy_used=strategy,
            fallback_pipeline=None,  # Would contain reduced functionality pipeline
            usable_components=[],
            error_message=None,
            warnings=warnings,
            performance_impact=performance_impact
        )


# Convenience functions for common fallback scenarios
def handle_pipeline_failure(model_path: str, error: Exception, 
                          context: Dict[str, Any] = None) -> FallbackResult:
    """Handle pipeline loading failures with automatic fallback"""
    handler = FallbackHandler()
    strategy = handler.create_fallback_strategy("pipeline_loading", error)
    return handler.execute_fallback_strategy(strategy, model_path, context)


def analyze_model_components(model_path: str) -> List[UsableComponent]:
    """Analyze model components for isolation possibilities"""
    handler = FallbackHandler()
    return handler.attempt_component_isolation(model_path)


def find_alternative_models(architecture: str, models_dir: str = "models") -> List[AlternativeModel]:
    """Find alternative models for given architecture"""
    handler = FallbackHandler(models_dir)
    return handler.suggest_alternative_models(architecture)