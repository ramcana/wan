#!/usr/bin/env python3
"""
Wan2.2 Compatibility System - Final Integration
Integrates all compatibility components into a cohesive system
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
import json

# Import all compatibility components
from infrastructure.hardware.architecture_detector import ArchitectureDetector, ModelArchitecture
from model_index_schema import SchemaValidator, ModelIndexSchema
from pipeline_manager import PipelineManager, PipelineRequirements
from dependency_manager import DependencyManager, RemoteCodeStatus
from vae_compatibility_handler import VAECompatibilityHandler
from core.services.optimization_manager import OptimizationManager, OptimizationPlan
from core.services.wan_pipeline_loader import WanPipelineLoader, WanPipelineWrapper
from fallback_handler import FallbackHandler, FallbackStrategy
from frame_tensor_handler import FrameTensorHandler, ProcessedFrames
from video_encoder import VideoEncoder, EncodingResult
from smoke_test_runner import SmokeTestRunner, SmokeTestResult
from wan_diagnostic_collector import DiagnosticCollector, ModelDiagnostics
from safe_load_manager import SafeLoadManager, SafeLoadingOptions
from compatibility_registry import CompatibilityRegistry
from error_messaging_system import ErrorMessagingSystem, UserFriendlyError
from infrastructure.hardware.performance_profiler import PerformanceProfiler


@dataclass
class CompatibilitySystemConfig:
    """Configuration for the compatibility system"""
    enable_diagnostics: bool = True
    enable_performance_monitoring: bool = True
    enable_safe_loading: bool = True
    enable_optimization: bool = True
    enable_fallback: bool = True
    diagnostics_dir: str = "diagnostics"
    registry_path: str = "compatibility_registry.json"
    max_memory_usage_gb: float = 12.0
    default_precision: str = "bf16"
    enable_cpu_offload: bool = True
    enable_chunked_processing: bool = True
    log_level: str = "INFO"


@dataclass
class ModelLoadResult:
    """Result of model loading attempt"""
    success: bool
    pipeline: Optional[Any] = None
    architecture: Optional[ModelArchitecture] = None
    optimizations: Optional[OptimizationPlan] = None
    diagnostics: Optional[ModelDiagnostics] = None
    errors: List[str] = None
    warnings: List[str] = None
    load_time: float = 0.0
    memory_usage: float = 0.0

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


@dataclass
class GenerationResult:
    """Result of video generation"""
    success: bool
    output_path: Optional[str] = None
    frames: Optional[ProcessedFrames] = None
    encoding_result: Optional[EncodingResult] = None
    generation_time: float = 0.0
    memory_peak: float = 0.0
    errors: List[str] = None
    warnings: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


class Wan22CompatibilitySystem:
    """
    Integrated compatibility system for Wan2.2 models
    Coordinates all compatibility components for seamless model loading and generation
    """

    def __init__(self, config: Optional[CompatibilitySystemConfig] = None):
        self.config = config or CompatibilitySystemConfig()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Initialize all components
        self._initialize_components()
        
        # Performance tracking
        self.performance_profiler = PerformanceProfiler() if self.config.enable_performance_monitoring else None
        
        # Statistics
        self.stats = {
            "models_loaded": 0,
            "successful_loads": 0,
            "failed_loads": 0,
            "generations_completed": 0,
            "total_generation_time": 0.0,
            "average_generation_time": 0.0,
            "optimizations_applied": 0,
            "fallbacks_used": 0
        }
        
        self.logger.info("Wan2.2 Compatibility System initialized")

    def _initialize_components(self):
        """Initialize all compatibility system components"""
        try:
            # Core detection and validation
            self.architecture_detector = ArchitectureDetector()
            self.schema_validator = SchemaValidator()
            
            # Pipeline management
            self.pipeline_manager = PipelineManager()
            self.dependency_manager = DependencyManager()
            
            # Specialized handlers
            self.vae_handler = VAECompatibilityHandler()
            self.optimization_manager = OptimizationManager() if self.config.enable_optimization else None
            self.fallback_handler = FallbackHandler() if self.config.enable_fallback else None
            
            # Pipeline loading
            self.wan_pipeline_loader = WanPipelineLoader()
            
            # Video processing
            self.frame_handler = FrameTensorHandler()
            self.video_encoder = VideoEncoder()
            
            # Testing and validation
            self.smoke_test_runner = SmokeTestRunner()
            
            # Diagnostics and reporting
            if self.config.enable_diagnostics:
                self.diagnostic_collector = DiagnosticCollector(self.config.diagnostics_dir)
            else:
                self.diagnostic_collector = None
            
            # Security
            if self.config.enable_safe_loading:
                self.safe_load_manager = SafeLoadManager()
            else:
                self.safe_load_manager = None
            
            # Registry and error handling
            self.compatibility_registry = CompatibilityRegistry(self.config.registry_path)
            self.error_messaging = ErrorMessagingSystem()
            
            self.logger.info("All compatibility components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize compatibility components: {e}")
            raise

    def load_model(self, model_path: str, **kwargs) -> ModelLoadResult:
        """
        Load a model with full compatibility handling
        
        Args:
            model_path: Path to the model
            **kwargs: Additional loading arguments
            
        Returns:
            ModelLoadResult with all loading information
        """
        start_time = time.time()
        result = ModelLoadResult(success=False)
        
        try:
            self.logger.info(f"Loading model: {model_path}")
            self.stats["models_loaded"] += 1
            
            # Start performance monitoring
            if self.performance_profiler:
                self.performance_profiler.start_monitoring("model_load")
            
            # Step 1: Detect model architecture
            self.logger.debug("Detecting model architecture...")
            architecture = self.architecture_detector.detect_model_architecture(model_path)
            result.architecture = architecture
            
            if not architecture:
                result.errors.append("Could not detect model architecture")
                return result
            
            # Step 2: Validate model schema
            self.logger.debug("Validating model schema...")
            schema_result = self.schema_validator.validate_model_index(
                str(Path(model_path) / "model_index.json")
            )
            
            if not schema_result.is_valid:
                result.warnings.extend(schema_result.warnings)
                if schema_result.errors:
                    result.errors.extend(schema_result.errors)
            
            # Step 3: Check security requirements
            if self.safe_load_manager:
                self.logger.debug("Checking security requirements...")
                safe_options = self.safe_load_manager.get_safe_loading_options(model_path)
                kwargs.update({
                    "trust_remote_code": safe_options.allow_remote_code,
                    "use_safetensors": True
                })
            
            # Step 4: Handle dependencies
            self.logger.debug("Checking dependencies...")
            remote_status = self.dependency_manager.check_remote_code_availability(model_path)
            
            if not remote_status.is_available and architecture.signature.is_wan_architecture():
                # Try to fetch remote code
                fetch_result = self.dependency_manager.fetch_pipeline_code(
                    model_path, kwargs.get("trust_remote_code", False)
                )
                if not fetch_result.success:
                    result.warnings.append(f"Could not fetch remote pipeline code: {fetch_result.error}")
            
            # Step 5: Create optimization plan
            optimization_plan = None
            if self.optimization_manager:
                self.logger.debug("Creating optimization plan...")
                system_resources = self.optimization_manager.analyze_system_resources()
                model_requirements = self._estimate_model_requirements(architecture)
                
                optimization_plan = self.optimization_manager.recommend_optimizations(
                    model_requirements, system_resources
                )
                result.optimizations = optimization_plan
                
                if optimization_plan.use_mixed_precision or optimization_plan.enable_cpu_offload:
                    self.stats["optimizations_applied"] += 1
            
            # Step 6: Load the pipeline
            self.logger.debug("Loading pipeline...")
            try:
                if architecture.signature.is_wan_architecture():
                    # Use WanPipelineLoader for Wan models
                    pipeline = self.wan_pipeline_loader.load_wan_pipeline(
                        model_path, 
                        optimization_plan=optimization_plan,
                        **kwargs
                    )
                else:
                    # Use standard pipeline loading
                    pipeline_class = self.pipeline_manager.select_pipeline_class(architecture.signature)
                    pipeline = self.pipeline_manager.load_custom_pipeline(
                        model_path, pipeline_class, **kwargs
                    )
                
                result.pipeline = pipeline
                result.success = True
                self.stats["successful_loads"] += 1
                
            except Exception as load_error:
                self.logger.warning(f"Primary pipeline loading failed: {load_error}")
                result.errors.append(str(load_error))
                
                # Try fallback strategies
                if self.fallback_handler:
                    self.logger.info("Attempting fallback strategies...")
                    fallback_strategy = self.fallback_handler.create_fallback_strategy(
                        "pipeline_loading", load_error
                    )
                    
                    if fallback_strategy.strategy_type != "no_fallback":
                        try:
                            pipeline = self._apply_fallback_strategy(
                                model_path, fallback_strategy, **kwargs
                            )
                            if pipeline:
                                result.pipeline = pipeline
                                result.success = True
                                result.warnings.append(f"Using fallback strategy: {fallback_strategy.description}")
                                self.stats["fallbacks_used"] += 1
                                self.stats["successful_loads"] += 1
                        except Exception as fallback_error:
                            result.errors.append(f"Fallback failed: {fallback_error}")
                
                if not result.success:
                    self.stats["failed_loads"] += 1
            
            # Step 7: Run smoke test if successful
            if result.success and result.pipeline:
                self.logger.debug("Running smoke test...")
                try:
                    smoke_result = self.smoke_test_runner.run_pipeline_smoke_test(result.pipeline)
                    if not smoke_result.success:
                        result.warnings.append("Smoke test failed - pipeline may not work correctly")
                        result.warnings.extend(smoke_result.errors)
                except Exception as smoke_error:
                    result.warnings.append(f"Could not run smoke test: {smoke_error}")
            
            # Step 8: Collect diagnostics
            if self.diagnostic_collector:
                self.logger.debug("Collecting diagnostics...")
                try:
                    diagnostics = self.diagnostic_collector.collect_model_diagnostics(
                        model_path, result
                    )
                    result.diagnostics = diagnostics
                    
                    # Write diagnostic report
                    model_name = Path(model_path).name
                    self.diagnostic_collector.write_compatibility_report(model_name, diagnostics)
                    
                except Exception as diag_error:
                    self.logger.warning(f"Could not collect diagnostics: {diag_error}")
            
        except Exception as e:
            self.logger.error(f"Unexpected error during model loading: {e}")
            result.errors.append(f"System error: {e}")
            self.stats["failed_loads"] += 1
        
        finally:
            # Stop performance monitoring
            if self.performance_profiler:
                perf_data = self.performance_profiler.stop_monitoring("model_load")
                result.memory_usage = perf_data.get("peak_memory_mb", 0.0)
            
            result.load_time = time.time() - start_time
            
            # Log result
            if result.success:
                self.logger.info(f"Model loaded successfully in {result.load_time:.2f}s")
            else:
                self.logger.error(f"Model loading failed after {result.load_time:.2f}s")
                if result.errors:
                    self.logger.error(f"Errors: {'; '.join(result.errors)}")
        
        return result

    def generate_video(self, pipeline: Any, prompt: str, output_path: str, **kwargs) -> GenerationResult:
        """
        Generate video with full processing pipeline
        
        Args:
            pipeline: Loaded pipeline
            prompt: Text prompt for generation
            output_path: Path for output video
            **kwargs: Generation parameters
            
        Returns:
            GenerationResult with all generation information
        """
        start_time = time.time()
        result = GenerationResult(success=False)
        
        try:
            self.logger.info(f"Starting video generation: '{prompt[:50]}...'")
            
            # Start performance monitoring
            if self.performance_profiler:
                self.performance_profiler.start_monitoring("video_generation")
            
            # Step 1: Generate video frames
            self.logger.debug("Generating video frames...")
            try:
                if isinstance(pipeline, WanPipelineWrapper):
                    # Use wrapper's generate method
                    generation_output = pipeline.generate(prompt, **kwargs)
                else:
                    # Use standard pipeline
                    generation_output = pipeline(prompt, **kwargs)
                
            except Exception as gen_error:
                self.logger.error(f"Video generation failed: {gen_error}")
                result.errors.append(str(gen_error))
                return result
            
            # Step 2: Process output tensors
            self.logger.debug("Processing output tensors...")
            try:
                processed_frames = self.frame_handler.process_output_tensors(generation_output)
                result.frames = processed_frames
                
                # Validate frame data
                validation_result = self.frame_handler.validate_frame_dimensions(processed_frames.frames)
                if not validation_result.is_valid:
                    result.warnings.extend(validation_result.errors)
                
            except Exception as frame_error:
                self.logger.error(f"Frame processing failed: {frame_error}")
                result.errors.append(f"Frame processing error: {frame_error}")
                return result
            
            # Step 3: Encode video
            self.logger.debug("Encoding video...")
            try:
                encoding_result = self.video_encoder.encode_frames_to_video(
                    processed_frames, output_path, format="mp4"
                )
                result.encoding_result = encoding_result
                
                if encoding_result.success:
                    result.output_path = output_path
                    result.success = True
                else:
                    # Try fallback output
                    self.logger.warning("Video encoding failed, trying fallback...")
                    fallback_result = self.video_encoder.provide_fallback_output(
                        processed_frames, output_path
                    )
                    if fallback_result.success:
                        result.output_path = fallback_result.output_path
                        result.success = True
                        result.warnings.append("Used frame-by-frame fallback output")
                    else:
                        result.errors.append("Both video encoding and fallback failed")
                
            except Exception as encode_error:
                self.logger.error(f"Video encoding failed: {encode_error}")
                result.errors.append(f"Encoding error: {encode_error}")
                return result
            
            # Update statistics
            if result.success:
                self.stats["generations_completed"] += 1
                generation_time = time.time() - start_time
                self.stats["total_generation_time"] += generation_time
                self.stats["average_generation_time"] = (
                    self.stats["total_generation_time"] / self.stats["generations_completed"]
                )
            
        except Exception as e:
            self.logger.error(f"Unexpected error during video generation: {e}")
            result.errors.append(f"System error: {e}")
        
        finally:
            # Stop performance monitoring
            if self.performance_profiler:
                perf_data = self.performance_profiler.stop_monitoring("video_generation")
                result.memory_peak = perf_data.get("peak_memory_mb", 0.0)
            
            result.generation_time = time.time() - start_time
            
            # Add metadata
            result.metadata = {
                "prompt": prompt,
                "generation_params": kwargs,
                "system_info": self._get_system_info() if self.performance_profiler else {}
            }
            
            # Log result
            if result.success:
                self.logger.info(f"Video generated successfully in {result.generation_time:.2f}s")
            else:
                self.logger.error(f"Video generation failed after {result.generation_time:.2f}s")
        
        return result

    def get_user_friendly_error(self, error: Exception, context: Dict[str, Any]) -> UserFriendlyError:
        """Get user-friendly error message for any error"""
        return self.error_messaging.create_user_friendly_error(error, context)

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "compatibility_system": {
                "initialized": True,
                "components": {
                    "architecture_detector": bool(self.architecture_detector),
                    "pipeline_manager": bool(self.pipeline_manager),
                    "optimization_manager": bool(self.optimization_manager),
                    "fallback_handler": bool(self.fallback_handler),
                    "diagnostics": bool(self.diagnostic_collector),
                    "safe_loading": bool(self.safe_load_manager),
                    "performance_monitoring": bool(self.performance_profiler)
                }
            },
            "statistics": self.stats.copy(),
            "configuration": asdict(self.config)
        }
        
        # Add system resources if available
        if self.optimization_manager:
            try:
                system_resources = self.optimization_manager.analyze_system_resources()
                status["system_resources"] = asdict(system_resources)
            except Exception:
                pass
        
        return status

    def _estimate_model_requirements(self, architecture: ModelArchitecture) -> Any:
        """Estimate model resource requirements"""
        # This would normally analyze the model to estimate requirements
        # For now, return a basic estimate based on architecture type
        from core.services.optimization_manager import ModelRequirements
        
        if architecture.signature.is_wan_architecture():
            return ModelRequirements(
                min_vram_mb=8192,
                recommended_vram_mb=12288,
                supports_cpu_offload=True,
                supports_mixed_precision=True,
                estimated_load_time=30.0
            )
        else:
            return ModelRequirements(
                min_vram_mb=4096,
                recommended_vram_mb=6144,
                supports_cpu_offload=True,
                supports_mixed_precision=True,
                estimated_load_time=15.0
            )

    def _apply_fallback_strategy(self, model_path: str, strategy: FallbackStrategy, **kwargs) -> Any:
        """Apply a fallback strategy to load the model"""
        if strategy.strategy_type == "component_isolation":
            # Try to load individual components
            return self.fallback_handler.attempt_component_isolation(model_path)
        elif strategy.strategy_type == "alternative_model":
            # This would suggest alternative models
            alternatives = self.fallback_handler.suggest_alternative_models(model_path)
            if alternatives:
                self.logger.info(f"Suggested alternatives: {[alt.name for alt in alternatives]}")
        elif strategy.strategy_type == "reduced_functionality":
            # Try loading with reduced functionality
            reduced_kwargs = kwargs.copy()
            reduced_kwargs.update({
                "torch_dtype": "float16",
                "low_cpu_mem_usage": True,
                "device_map": "auto"
            })
            return self.pipeline_manager.load_custom_pipeline(
                model_path, "StableDiffusionPipeline", **reduced_kwargs
            )
        
        return None

    def _get_system_info(self) -> Dict[str, Any]:
        """Get basic system information"""
        import torch
        import platform
        
        info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
        }
        
        if torch.cuda.is_available():
            info.update({
                "cuda_available": True,
                "cuda_version": torch.version.cuda,
                "gpu_count": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
            })
        else:
            info["cuda_available"] = False
        
        return info

    def cleanup(self):
        """Cleanup system resources"""
        self.logger.info("Cleaning up compatibility system...")
        
        try:
            # Stop performance monitoring
            if self.performance_profiler:
                self.performance_profiler.cleanup()
            
            # Clear any cached models
            if hasattr(self.wan_pipeline_loader, 'cleanup'):
                self.wan_pipeline_loader.cleanup()
            
            # Save final statistics
            if self.diagnostic_collector:
                stats_path = Path(self.config.diagnostics_dir) / "system_stats.json"
                try:
                    with open(stats_path, 'w') as f:
                        json.dump(self.stats, f, indent=2)
                    self.logger.info(f"Saved system statistics to {stats_path}")
                except Exception as e:
                    self.logger.warning(f"Could not save statistics: {e}")
            
            self.logger.info("Compatibility system cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


# Global compatibility system instance
_compatibility_system: Optional[Wan22CompatibilitySystem] = None


def get_compatibility_system(config: Optional[CompatibilitySystemConfig] = None) -> Wan22CompatibilitySystem:
    """Get or create the global compatibility system instance"""
    global _compatibility_system
    
    if _compatibility_system is None:
        _compatibility_system = Wan22CompatibilitySystem(config)
    
    return _compatibility_system


def initialize_compatibility_system(config: Optional[CompatibilitySystemConfig] = None) -> Wan22CompatibilitySystem:
    """Initialize the compatibility system with configuration"""
    global _compatibility_system
    
    if _compatibility_system is not None:
        _compatibility_system.cleanup()
    
    _compatibility_system = Wan22CompatibilitySystem(config)
    return _compatibility_system


def cleanup_compatibility_system():
    """Cleanup the global compatibility system"""
    global _compatibility_system
    
    if _compatibility_system is not None:
        _compatibility_system.cleanup()
        _compatibility_system = None