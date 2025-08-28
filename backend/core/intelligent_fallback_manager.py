"""
Intelligent Fallback Manager for Enhanced Model Availability

This module provides smart alternatives when preferred models are unavailable,
implements model compatibility scoring algorithms, and manages fallback strategies
with request queuing and wait time estimation.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import json
import math

logger = logging.getLogger(__name__)

# Import performance monitoring (with fallback if not available)
try:
    from .performance_monitoring_system import get_performance_monitor
    PERFORMANCE_MONITORING_AVAILABLE = True
except ImportError:
    PERFORMANCE_MONITORING_AVAILABLE = False
    logger.warning("Performance monitoring not available")


class FallbackType(Enum):
    """Types of fallback strategies available"""
    ALTERNATIVE_MODEL = "alternative_model"
    QUEUE_AND_WAIT = "queue_and_wait"
    MOCK_GENERATION = "mock_generation"
    DOWNLOAD_AND_RETRY = "download_and_retry"
    REDUCE_REQUIREMENTS = "reduce_requirements"
    HYBRID_APPROACH = "hybrid_approach"


class ModelCapability(Enum):
    """Model capabilities for compatibility scoring"""
    TEXT_TO_VIDEO = "text_to_video"
    IMAGE_TO_VIDEO = "image_to_video"
    TEXT_IMAGE_TO_VIDEO = "text_image_to_video"
    HIGH_RESOLUTION = "high_resolution"
    FAST_GENERATION = "fast_generation"
    HIGH_QUALITY = "high_quality"


@dataclass
class GenerationRequirements:
    """Requirements for a generation request"""
    model_type: str
    quality: str = "medium"  # low, medium, high
    speed: str = "medium"    # fast, medium, slow
    resolution: str = "1280x720"
    max_wait_time: Optional[timedelta] = None
    allow_alternatives: bool = True
    allow_quality_reduction: bool = True
    priority: str = "normal"  # low, normal, high, critical


@dataclass
class ModelSuggestion:
    """Suggestion for an alternative model"""
    suggested_model: str
    compatibility_score: float  # 0.0 to 1.0
    performance_difference: float  # -1.0 to 1.0 (negative means worse performance)
    availability_status: str
    reason: str
    estimated_quality_difference: str  # "similar", "slightly_lower", "significantly_lower"
    capabilities_match: List[str] = field(default_factory=list)
    capabilities_missing: List[str] = field(default_factory=list)
    vram_requirement_gb: float = 0.0
    estimated_generation_time: Optional[timedelta] = None


@dataclass
class FallbackStrategy:
    """Strategy for handling unavailable models"""
    strategy_type: FallbackType
    recommended_action: str
    alternative_model: Optional[str] = None
    estimated_wait_time: Optional[timedelta] = None
    user_message: str = ""
    can_queue_request: bool = False
    confidence_score: float = 0.0  # 0.0 to 1.0
    fallback_options: List[Dict[str, Any]] = field(default_factory=list)
    requirements_adjustments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EstimatedWaitTime:
    """Estimated wait time for model availability"""
    model_id: str
    download_time: Optional[timedelta] = None
    queue_position: int = 0
    queue_wait_time: Optional[timedelta] = None
    total_wait_time: Optional[timedelta] = None
    confidence: str = "medium"  # low, medium, high
    factors: List[str] = field(default_factory=list)


@dataclass
class QueuedRequest:
    """Queued generation request waiting for model availability"""
    request_id: str
    model_id: str
    requirements: GenerationRequirements
    queued_at: datetime
    priority: str = "normal"
    estimated_start_time: Optional[datetime] = None
    callback: Optional[Any] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueueResult:
    """Result of queuing a request"""
    success: bool
    request_id: str
    queue_position: int
    estimated_wait_time: Optional[timedelta] = None
    message: str = ""
    error: Optional[str] = None


class IntelligentFallbackManager:
    """
    Intelligent fallback manager that provides smart alternatives when preferred models
    are unavailable, implements compatibility scoring, and manages request queuing.
    """
    
    def __init__(self, availability_manager=None, models_dir: Optional[str] = None):
        """
        Initialize the Intelligent Fallback Manager.
        
        Args:
            availability_manager: ModelAvailabilityManager instance
            models_dir: Directory for storing model metadata
        """
        self.availability_manager = availability_manager
        self.models_dir = Path(models_dir) if models_dir else Path("models")
        
        # Model compatibility database
        self._model_capabilities = self._initialize_model_capabilities()
        self._compatibility_matrix = self._initialize_compatibility_matrix()
        
        # Request queue management
        self._request_queue: List[QueuedRequest] = []
        self._queue_lock = asyncio.Lock()
        self._next_request_id = 1
        
        # Performance tracking for better estimates
        self._performance_history: Dict[str, List[Dict[str, Any]]] = {}
        self._download_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Configuration
        self.max_queue_size = 100
        self.default_download_speed_mbps = 50.0  # Conservative estimate
        self.queue_processing_interval = 5.0  # seconds
        
        # Cache for compatibility scores
        self._compatibility_cache: Dict[str, ModelSuggestion] = {}
        
        logger.info("Intelligent Fallback Manager initialized")
    
    async def initialize(self) -> bool:
        """Initialize the intelligent fallback manager"""
        try:
            # Initialize model capabilities and compatibility matrix
            self._model_capabilities = self._initialize_model_capabilities()
            self._compatibility_matrix = self._initialize_compatibility_matrix()
            
            logger.info("Intelligent Fallback Manager initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Intelligent Fallback Manager: {e}")
            return False
    
    def _initialize_model_capabilities(self) -> Dict[str, List[ModelCapability]]:
        """Initialize model capabilities database"""
        return {
            "t2v-A14B": [
                ModelCapability.TEXT_TO_VIDEO,
                ModelCapability.HIGH_QUALITY,
                ModelCapability.HIGH_RESOLUTION
            ],
            "i2v-A14B": [
                ModelCapability.IMAGE_TO_VIDEO,
                ModelCapability.HIGH_QUALITY,
                ModelCapability.HIGH_RESOLUTION
            ],
            "ti2v-5B": [
                ModelCapability.TEXT_IMAGE_TO_VIDEO,
                ModelCapability.FAST_GENERATION,
                ModelCapability.HIGH_QUALITY
            ]
        }
    
    def _initialize_compatibility_matrix(self) -> Dict[str, Dict[str, float]]:
        """Initialize model compatibility scoring matrix"""
        return {
            "t2v-A14B": {
                "i2v-A14B": 0.7,    # Can handle text prompts but needs image
                "ti2v-5B": 0.8      # Good alternative for text-to-video
            },
            "i2v-A14B": {
                "t2v-A14B": 0.6,    # Can generate from text but no image input
                "ti2v-5B": 0.9      # Excellent alternative for image-to-video
            },
            "ti2v-5B": {
                "t2v-A14B": 0.8,    # Good for text-only generation
                "i2v-A14B": 0.9     # Excellent for image-based generation
            }
        }
    
    async def suggest_alternative_model(
        self, 
        requested_model: str, 
        requirements: GenerationRequirements
    ) -> ModelSuggestion:
        """
        Suggest the best alternative model based on requirements and compatibility.
        
        Args:
            requested_model: The originally requested model
            requirements: Generation requirements and preferences
            
        Returns:
            ModelSuggestion with the best alternative and compatibility info
        """
        # Start performance monitoring
        performance_id = None
        if PERFORMANCE_MONITORING_AVAILABLE:
            try:
                monitor = get_performance_monitor()
                performance_id = monitor.track_fallback_strategy(
                    f"suggest_alternative_{requested_model}",
                    {
                        "requested_model": requested_model,
                        "quality_requirement": requirements.quality,
                        "speed_requirement": requirements.speed,
                        "resolution": requirements.resolution
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to start performance monitoring: {e}")
        
        try:
            logger.info(f"Finding alternative for {requested_model} with requirements: {requirements.quality}/{requirements.speed}")
            
            # Check cache first
            cache_key = f"{requested_model}_{requirements.quality}_{requirements.speed}_{requirements.resolution}"
            if cache_key in self._compatibility_cache:
                cached_suggestion = self._compatibility_cache[cache_key]
                # Update availability status
                if self.availability_manager:
                    status = await self.availability_manager._check_single_model_availability(cached_suggestion.suggested_model)
                    cached_suggestion.availability_status = status.availability_status.value
                
                # End performance monitoring for cached result
                if PERFORMANCE_MONITORING_AVAILABLE and performance_id:
                    try:
                        monitor = get_performance_monitor()
                        monitor.end_tracking(
                            performance_id,
                            success=True,
                            additional_metadata={
                                "cache_hit": True,
                                "suggested_model": cached_suggestion.suggested_model,
                                "compatibility_score": cached_suggestion.compatibility_score
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Failed to end performance monitoring: {e}")
                
                return cached_suggestion
            
            # Get available models
            available_models = await self._get_available_models()
            
            # Score all alternative models
            best_suggestion = None
            best_score = 0.0
            
            for model_id in available_models:
                if model_id == requested_model:
                    continue  # Skip the requested model
                
                suggestion = await self._score_model_compatibility(
                    requested_model, model_id, requirements
                )
                
                if suggestion.compatibility_score > best_score:
                    best_score = suggestion.compatibility_score
                    best_suggestion = suggestion
            
            # If no good alternatives found, create a fallback suggestion
            if not best_suggestion or best_score < 0.3:
                best_suggestion = ModelSuggestion(
                    suggested_model="mock_generation",
                    compatibility_score=0.2,
                    performance_difference=-0.8,
                    availability_status="available",
                    reason="No suitable model alternatives available",
                    estimated_quality_difference="significantly_lower"
                )
            
            # Cache the result
            self._compatibility_cache[cache_key] = best_suggestion
            
            logger.info(f"Best alternative for {requested_model}: {best_suggestion.suggested_model} (score: {best_suggestion.compatibility_score:.2f})")
            
            # End performance monitoring
            if PERFORMANCE_MONITORING_AVAILABLE and performance_id:
                try:
                    monitor = get_performance_monitor()
                    monitor.end_tracking(
                        performance_id,
                        success=True,
                        additional_metadata={
                            "cache_hit": False,
                            "suggested_model": best_suggestion.suggested_model,
                            "compatibility_score": best_suggestion.compatibility_score,
                            "alternatives_considered": len(available_models) - 1,
                            "fallback_strategy": "alternative_model" if best_suggestion.suggested_model != "mock_generation" else "mock_generation"
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to end performance monitoring: {e}")
            
            return best_suggestion
            
        except Exception as e:
            logger.error(f"Error suggesting alternative model: {e}")
            
            # End performance monitoring with error
            if PERFORMANCE_MONITORING_AVAILABLE and performance_id:
                try:
                    monitor = get_performance_monitor()
                    monitor.end_tracking(
                        performance_id,
                        success=False,
                        error_message=str(e),
                        additional_metadata={
                            "error_type": type(e).__name__,
                            "fallback_to_mock": True
                        }
                    )
                except Exception as monitor_e:
                    logger.warning(f"Failed to end performance monitoring: {monitor_e}")
            
            return ModelSuggestion(
                suggested_model="mock_generation",
                compatibility_score=0.1,
                performance_difference=-1.0,
                availability_status="available",
                reason=f"Error finding alternatives: {str(e)}",
                estimated_quality_difference="significantly_lower"
            )
    
    async def _get_available_models(self) -> List[str]:
        """Get list of currently available models"""
        try:
            if self.availability_manager:
                status_dict = await self.availability_manager.get_comprehensive_model_status()
                available = []
                for model_id, status in status_dict.items():
                    if status.availability_status.value == "available":
                        available.append(model_id)
                return available
            else:
                # Fallback to known models
                return ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
        except Exception as e:
            logger.warning(f"Error getting available models: {e}")
            return []
    
    async def _score_model_compatibility(
        self, 
        requested_model: str, 
        candidate_model: str, 
        requirements: GenerationRequirements
    ) -> ModelSuggestion:
        """Score compatibility between requested and candidate models"""
        try:
            # Base compatibility from matrix
            base_score = self._compatibility_matrix.get(requested_model, {}).get(candidate_model, 0.5)
            
            # Get model capabilities
            requested_caps = self._model_capabilities.get(requested_model, [])
            candidate_caps = self._model_capabilities.get(candidate_model, [])
            
            # Calculate capability overlap
            matching_caps = set(requested_caps) & set(candidate_caps)
            missing_caps = set(requested_caps) - set(candidate_caps)
            
            capability_score = len(matching_caps) / max(len(requested_caps), 1)
            
            # Adjust score based on requirements
            requirement_score = 1.0
            
            # Quality requirement adjustment
            if requirements.quality == "high":
                if ModelCapability.HIGH_QUALITY in candidate_caps:
                    requirement_score *= 1.2
                else:
                    requirement_score *= 0.7
            
            # Speed requirement adjustment
            if requirements.speed == "fast":
                if ModelCapability.FAST_GENERATION in candidate_caps:
                    requirement_score *= 1.2
                else:
                    requirement_score *= 0.8
            
            # Resolution requirement adjustment
            if requirements.resolution in ["1920x1080", "1920x1088"]:
                if ModelCapability.HIGH_RESOLUTION in candidate_caps:
                    requirement_score *= 1.1
                else:
                    requirement_score *= 0.6
            
            # Calculate final compatibility score
            final_score = (base_score * 0.4 + capability_score * 0.4 + requirement_score * 0.2)
            final_score = min(final_score, 1.0)
            
            # Estimate performance difference
            performance_diff = self._estimate_performance_difference(requested_model, candidate_model)
            
            # Determine quality difference description
            if final_score >= 0.9:
                quality_diff = "similar"
            elif final_score >= 0.7:
                quality_diff = "slightly_lower"
            else:
                quality_diff = "significantly_lower"
            
            # Get availability status
            availability_status = "unknown"
            vram_requirement = 8.0  # Default
            
            if self.availability_manager:
                status = await self.availability_manager._check_single_model_availability(candidate_model)
                availability_status = status.availability_status.value
                vram_requirement = self._estimate_vram_requirement(candidate_model, requirements.resolution)
            
            # Generate reason
            reason_parts = []
            if len(matching_caps) > 0:
                reason_parts.append(f"Shares {len(matching_caps)} key capabilities")
            if final_score >= 0.8:
                reason_parts.append("High compatibility score")
            if availability_status == "available":
                reason_parts.append("Currently available")
            
            reason = "; ".join(reason_parts) if reason_parts else "Basic compatibility"
            
            return ModelSuggestion(
                suggested_model=candidate_model,
                compatibility_score=final_score,
                performance_difference=performance_diff,
                availability_status=availability_status,
                reason=reason,
                estimated_quality_difference=quality_diff,
                capabilities_match=[cap.value for cap in matching_caps],
                capabilities_missing=[cap.value for cap in missing_caps],
                vram_requirement_gb=vram_requirement,
                estimated_generation_time=self._estimate_generation_time(candidate_model, requirements)
            )
            
        except Exception as e:
            logger.error(f"Error scoring model compatibility: {e}")
            return ModelSuggestion(
                suggested_model=candidate_model,
                compatibility_score=0.1,
                performance_difference=-0.5,
                availability_status="unknown",
                reason=f"Scoring error: {str(e)}",
                estimated_quality_difference="unknown"
            )
    
    def _estimate_performance_difference(self, requested_model: str, candidate_model: str) -> float:
        """Estimate performance difference between models (-1.0 to 1.0)"""
        # Model performance rankings (higher is better)
        performance_ranks = {
            "t2v-A14B": 0.9,
            "i2v-A14B": 0.95,
            "ti2v-5B": 0.8
        }
        
        requested_perf = performance_ranks.get(requested_model, 0.5)
        candidate_perf = performance_ranks.get(candidate_model, 0.5)
        
        # Return difference (-1.0 means much worse, 1.0 means much better)
        return (candidate_perf - requested_perf) / max(requested_perf, 0.1)
    
    def _estimate_vram_requirement(self, model_id: str, resolution: str) -> float:
        """Estimate VRAM requirement for model and resolution"""
        base_requirements = {
            "t2v-A14B": 8.0,
            "i2v-A14B": 9.0,
            "ti2v-5B": 6.0
        }
        
        resolution_multipliers = {
            "1280x720": 1.0,
            "1280x704": 1.0,
            "1920x1080": 1.4,
            "1920x1088": 1.4
        }
        
        base = base_requirements.get(model_id, 8.0)
        multiplier = resolution_multipliers.get(resolution, 1.2)
        
        return base * multiplier
    
    def _estimate_generation_time(self, model_id: str, requirements: GenerationRequirements) -> timedelta:
        """Estimate generation time for model and requirements"""
        # Base generation times (in seconds)
        base_times = {
            "t2v-A14B": 120,
            "i2v-A14B": 100,
            "ti2v-5B": 80
        }
        
        base_time = base_times.get(model_id, 100)
        
        # Adjust for resolution
        if requirements.resolution in ["1920x1080", "1920x1088"]:
            base_time *= 1.5
        
        # Adjust for quality
        if requirements.quality == "high":
            base_time *= 1.3
        elif requirements.quality == "low":
            base_time *= 0.7
        
        return timedelta(seconds=base_time)
    
    async def get_fallback_strategy(
        self, 
        failed_model: str, 
        error_context: Dict[str, Any]
    ) -> FallbackStrategy:
        """
        Get the best fallback strategy for a failed model.
        
        Args:
            failed_model: The model that failed to load/generate
            error_context: Context about the failure
            
        Returns:
            FallbackStrategy with recommended actions
        """
        try:
            logger.info(f"Determining fallback strategy for {failed_model}")
            
            # Extract requirements from context
            requirements = error_context.get("requirements")
            if not requirements:
                requirements = GenerationRequirements(model_type=failed_model)
            
            # Analyze the failure type
            failure_type = error_context.get("failure_type", "unknown")
            error_message = error_context.get("error_message", "")
            
            # Get model availability status
            model_status = None
            if self.availability_manager:
                model_status = await self.availability_manager._check_single_model_availability(failed_model)
            
            # Determine best strategy based on failure analysis
            if failure_type == "model_loading_failure" or "not found" in error_message.lower():
                return await self._handle_missing_model_strategy(failed_model, requirements, model_status)
            
            elif failure_type == "vram_exhaustion" or "memory" in error_message.lower():
                return await self._handle_vram_exhaustion_strategy(failed_model, requirements)
            
            elif failure_type == "network_error" or "download" in error_message.lower():
                return await self._handle_network_error_strategy(failed_model, requirements)
            
            else:
                return await self._handle_general_failure_strategy(failed_model, requirements)
            
        except Exception as e:
            logger.error(f"Error determining fallback strategy: {e}")
            return FallbackStrategy(
                strategy_type=FallbackType.MOCK_GENERATION,
                recommended_action="Use mock generation due to strategy error",
                user_message="Unable to determine optimal fallback strategy. Using mock generation.",
                confidence_score=0.1
            )
    
    async def _handle_missing_model_strategy(
        self, 
        failed_model: str, 
        requirements: GenerationRequirements,
        model_status: Optional[Any]
    ) -> FallbackStrategy:
        """Handle strategy for missing model"""
        # Check if model can be downloaded
        can_download = model_status and model_status.availability_status.value in ["missing", "corrupted"]
        
        # Get alternative model suggestion
        alternative = await self.suggest_alternative_model(failed_model, requirements)
        
        # Estimate download time if applicable
        download_time = None
        if can_download:
            download_time = await self.estimate_wait_time(failed_model)
        
        # Decide strategy based on alternatives and download feasibility
        if alternative.compatibility_score >= 0.7 and alternative.availability_status == "available":
            return FallbackStrategy(
                strategy_type=FallbackType.ALTERNATIVE_MODEL,
                recommended_action=f"Use alternative model {alternative.suggested_model}",
                alternative_model=alternative.suggested_model,
                user_message=f"Using {alternative.suggested_model} as alternative (compatibility: {alternative.compatibility_score:.1%})",
                confidence_score=alternative.compatibility_score,
                fallback_options=[{
                    "type": "alternative_model",
                    "model": alternative.suggested_model,
                    "compatibility": alternative.compatibility_score,
                    "quality_difference": alternative.estimated_quality_difference
                }]
            )
        
        elif can_download and download_time and download_time.total_wait_time and download_time.total_wait_time < timedelta(minutes=30):
            return FallbackStrategy(
                strategy_type=FallbackType.DOWNLOAD_AND_RETRY,
                recommended_action=f"Download {failed_model} and retry",
                estimated_wait_time=download_time.total_wait_time,
                user_message=f"Downloading {failed_model} (estimated time: {download_time.total_wait_time})",
                can_queue_request=True,
                confidence_score=0.8
            )
        
        elif requirements.allow_alternatives and alternative.compatibility_score >= 0.5:
            return FallbackStrategy(
                strategy_type=FallbackType.ALTERNATIVE_MODEL,
                recommended_action=f"Use alternative model {alternative.suggested_model}",
                alternative_model=alternative.suggested_model,
                user_message=f"Using {alternative.suggested_model} with reduced compatibility ({alternative.compatibility_score:.1%})",
                confidence_score=alternative.compatibility_score * 0.8
            )
        
        else:
            return FallbackStrategy(
                strategy_type=FallbackType.MOCK_GENERATION,
                recommended_action="Use mock generation",
                user_message="No suitable alternatives available. Using mock generation.",
                confidence_score=0.3
            )
    
    async def _handle_vram_exhaustion_strategy(
        self, 
        failed_model: str, 
        requirements: GenerationRequirements
    ) -> FallbackStrategy:
        """Handle strategy for VRAM exhaustion"""
        # Try to find a lighter alternative
        lighter_requirements = GenerationRequirements(
            model_type=failed_model,
            quality="medium" if requirements.quality == "high" else "low",
            speed=requirements.speed,
            resolution="1280x720" if requirements.resolution in ["1920x1080", "1920x1088"] else requirements.resolution,
            allow_alternatives=True,
            allow_quality_reduction=True
        )
        
        alternative = await self.suggest_alternative_model(failed_model, lighter_requirements)
        
        # Check if ti2v-5B is available (lighter model)
        if alternative.suggested_model == "ti2v-5B" and alternative.availability_status == "available":
            return FallbackStrategy(
                strategy_type=FallbackType.ALTERNATIVE_MODEL,
                recommended_action=f"Use lighter model {alternative.suggested_model}",
                alternative_model=alternative.suggested_model,
                user_message=f"Using {alternative.suggested_model} to reduce VRAM usage",
                confidence_score=0.8,
                requirements_adjustments={
                    "model": alternative.suggested_model,
                    "vram_optimized": True
                }
            )
        
        # Try reducing requirements
        elif requirements.allow_quality_reduction:
            return FallbackStrategy(
                strategy_type=FallbackType.REDUCE_REQUIREMENTS,
                recommended_action="Reduce generation parameters",
                user_message="Reducing resolution and quality to fit VRAM constraints",
                confidence_score=0.6,
                requirements_adjustments={
                    "resolution": "1280x720",
                    "quality": "medium",
                    "vram_optimized": True
                }
            )
        
        else:
            return FallbackStrategy(
                strategy_type=FallbackType.MOCK_GENERATION,
                recommended_action="Use mock generation",
                user_message="Insufficient VRAM for real generation. Using mock mode.",
                confidence_score=0.3
            )
    
    async def _handle_network_error_strategy(
        self, 
        failed_model: str, 
        requirements: GenerationRequirements
    ) -> FallbackStrategy:
        """Handle strategy for network errors"""
        # Check for available alternatives first
        alternative = await self.suggest_alternative_model(failed_model, requirements)
        
        if alternative.compatibility_score >= 0.7 and alternative.availability_status == "available":
            return FallbackStrategy(
                strategy_type=FallbackType.ALTERNATIVE_MODEL,
                recommended_action=f"Use available model {alternative.suggested_model}",
                alternative_model=alternative.suggested_model,
                user_message=f"Network issues detected. Using available {alternative.suggested_model}",
                confidence_score=alternative.compatibility_score
            )
        
        # Otherwise suggest queuing for retry
        else:
            return FallbackStrategy(
                strategy_type=FallbackType.QUEUE_AND_WAIT,
                recommended_action="Queue request for retry when network recovers",
                estimated_wait_time=timedelta(minutes=10),  # Conservative network recovery estimate
                user_message="Network issues detected. Request queued for automatic retry.",
                can_queue_request=True,
                confidence_score=0.7
            )
    
    async def _handle_general_failure_strategy(
        self, 
        failed_model: str, 
        requirements: GenerationRequirements
    ) -> FallbackStrategy:
        """Handle strategy for general failures"""
        # Try alternative model first
        alternative = await self.suggest_alternative_model(failed_model, requirements)
        
        if alternative.compatibility_score >= 0.6 and alternative.availability_status == "available":
            return FallbackStrategy(
                strategy_type=FallbackType.ALTERNATIVE_MODEL,
                recommended_action=f"Use alternative model {alternative.suggested_model}",
                alternative_model=alternative.suggested_model,
                user_message=f"System issue detected. Using {alternative.suggested_model} as fallback",
                confidence_score=alternative.compatibility_score * 0.9
            )
        
        # Fall back to mock generation
        else:
            return FallbackStrategy(
                strategy_type=FallbackType.MOCK_GENERATION,
                recommended_action="Use mock generation",
                user_message="System issues detected. Using mock generation mode.",
                confidence_score=0.4
            )
    
    async def estimate_wait_time(self, model_id: str) -> EstimatedWaitTime:
        """
        Estimate wait time for model to become available.
        
        Args:
            model_id: Model to estimate wait time for
            
        Returns:
            EstimatedWaitTime with breakdown of wait factors
        """
        try:
            logger.debug(f"Estimating wait time for {model_id}")
            
            result = EstimatedWaitTime(model_id=model_id)
            
            # Check current model status
            if self.availability_manager:
                status = await self.availability_manager._check_single_model_availability(model_id)
                
                # If already available, no wait time
                if status.availability_status.value == "available":
                    result.total_wait_time = timedelta(0)
                    result.confidence = "high"
                    result.factors = ["Model already available"]
                    return result
                
                # If downloading, use existing progress
                if status.availability_status.value == "downloading" and status.estimated_download_time:
                    result.download_time = status.estimated_download_time
                    result.factors.append(f"Download in progress: {status.download_progress:.1f}% complete")
                
                # If missing, estimate download time
                elif status.availability_status.value == "missing":
                    estimated_size_gb = self._estimate_model_size(model_id)
                    download_time_seconds = (estimated_size_gb * 1024) / self.default_download_speed_mbps
                    result.download_time = timedelta(seconds=download_time_seconds)
                    result.factors.append(f"Model download required: ~{estimated_size_gb:.1f}GB")
            
            # Check queue position
            async with self._queue_lock:
                queue_position = 0
                for i, queued_request in enumerate(self._request_queue):
                    if queued_request.model_id == model_id:
                        queue_position = i + 1
                        break
                
                result.queue_position = queue_position
                
                if queue_position > 0:
                    # Estimate queue wait time based on position and average processing time
                    avg_processing_time = timedelta(minutes=5)  # Conservative estimate
                    result.queue_wait_time = timedelta(seconds=avg_processing_time.total_seconds() * queue_position)
                    result.factors.append(f"Queue position: {queue_position}")
            
            # Calculate total wait time
            total_seconds = 0
            if result.download_time:
                total_seconds += result.download_time.total_seconds()
            if result.queue_wait_time:
                total_seconds += result.queue_wait_time.total_seconds()
            
            result.total_wait_time = timedelta(seconds=total_seconds)
            
            # Determine confidence based on available information
            if self.availability_manager and result.download_time:
                result.confidence = "medium"
            elif len(result.factors) > 0:
                result.confidence = "low"
            else:
                result.confidence = "low"
                result.factors.append("Limited information available")
            
            logger.debug(f"Estimated wait time for {model_id}: {result.total_wait_time}")
            return result
            
        except Exception as e:
            logger.error(f"Error estimating wait time for {model_id}: {e}")
            return EstimatedWaitTime(
                model_id=model_id,
                total_wait_time=timedelta(minutes=15),  # Conservative fallback
                confidence="low",
                factors=[f"Estimation error: {str(e)}"]
            )
    
    def _estimate_model_size(self, model_id: str) -> float:
        """Estimate model size in GB"""
        size_estimates = {
            "t2v-A14B": 12.0,
            "i2v-A14B": 14.0,
            "ti2v-5B": 8.0
        }
        return size_estimates.get(model_id, 10.0)
    
    async def queue_request_for_downloading_model(
        self, 
        model_id: str, 
        requirements: GenerationRequirements,
        callback: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> QueueResult:
        """
        Queue a generation request for when the model becomes available.
        
        Args:
            model_id: Model to wait for
            requirements: Generation requirements
            callback: Optional callback for when model is ready
            context: Additional context for the request
            
        Returns:
            QueueResult with queue information
        """
        try:
            async with self._queue_lock:
                # Check queue size limit
                if len(self._request_queue) >= self.max_queue_size:
                    return QueueResult(
                        success=False,
                        request_id="",
                        queue_position=0,
                        error="Queue is full. Please try again later."
                    )
                
                # Generate request ID
                request_id = f"req_{self._next_request_id}_{int(time.time())}"
                self._next_request_id += 1
                
                # Create queued request
                queued_request = QueuedRequest(
                    request_id=request_id,
                    model_id=model_id,
                    requirements=requirements,
                    queued_at=datetime.now(),
                    priority=requirements.priority,
                    callback=callback,
                    context=context or {}
                )
                
                # Insert based on priority
                insert_position = len(self._request_queue)
                for i, existing_request in enumerate(self._request_queue):
                    if self._get_priority_value(requirements.priority) > self._get_priority_value(existing_request.priority):
                        insert_position = i
                        break
                
                self._request_queue.insert(insert_position, queued_request)
                
                # Estimate wait time
                wait_time_estimate = await self.estimate_wait_time(model_id)
                
                logger.info(f"Queued request {request_id} for model {model_id} at position {insert_position + 1}")
                
                return QueueResult(
                    success=True,
                    request_id=request_id,
                    queue_position=insert_position + 1,
                    estimated_wait_time=wait_time_estimate.total_wait_time,
                    message=f"Request queued successfully. Position: {insert_position + 1}"
                )
                
        except Exception as e:
            logger.error(f"Error queuing request for {model_id}: {e}")
            return QueueResult(
                success=False,
                request_id="",
                queue_position=0,
                error=f"Failed to queue request: {str(e)}"
            )
    
    def _get_priority_value(self, priority: str) -> int:
        """Convert priority string to numeric value for sorting"""
        priority_values = {
            "critical": 4,
            "high": 3,
            "normal": 2,
            "low": 1
        }
        return priority_values.get(priority, 2)
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status and statistics"""
        try:
            async with self._queue_lock:
                queue_by_model = {}
                total_requests = len(self._request_queue)
                
                for request in self._request_queue:
                    if request.model_id not in queue_by_model:
                        queue_by_model[request.model_id] = []
                    queue_by_model[request.model_id].append({
                        "request_id": request.request_id,
                        "priority": request.priority,
                        "queued_at": request.queued_at.isoformat(),
                        "wait_time_minutes": (datetime.now() - request.queued_at).total_seconds() / 60
                    })
                
                return {
                    "total_queued_requests": total_requests,
                    "queue_by_model": queue_by_model,
                    "max_queue_size": self.max_queue_size,
                    "queue_utilization": total_requests / self.max_queue_size
                }
                
        except Exception as e:
            logger.error(f"Error getting queue status: {e}")
            return {"error": str(e)}
    
    async def process_queue(self) -> Dict[str, Any]:
        """Process queued requests for available models"""
        try:
            processed_requests = []
            errors = []
            
            async with self._queue_lock:
                remaining_queue = []
                
                for request in self._request_queue:
                    try:
                        # Check if model is now available
                        if self.availability_manager:
                            status = await self.availability_manager._check_single_model_availability(request.model_id)
                            
                            if status.availability_status.value == "available":
                                # Model is ready, process the request
                                if request.callback:
                                    try:
                                        await request.callback(request)
                                        processed_requests.append(request.request_id)
                                    except Exception as callback_error:
                                        errors.append(f"Callback error for {request.request_id}: {callback_error}")
                                        remaining_queue.append(request)
                                else:
                                    # No callback, just mark as processed
                                    processed_requests.append(request.request_id)
                            else:
                                # Model still not available, keep in queue
                                remaining_queue.append(request)
                        else:
                            # No availability manager, keep in queue
                            remaining_queue.append(request)
                            
                    except Exception as e:
                        errors.append(f"Error processing request {request.request_id}: {e}")
                        remaining_queue.append(request)
                
                # Update queue
                self._request_queue = remaining_queue
            
            result = {
                "processed_requests": len(processed_requests),
                "remaining_requests": len(self._request_queue),
                "errors": errors
            }
            
            if processed_requests:
                logger.info(f"Processed {len(processed_requests)} queued requests")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing queue: {e}")
            return {"error": str(e)}
    
    async def cleanup_expired_requests(self, max_age_hours: int = 24) -> int:
        """Clean up expired requests from the queue"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            removed_count = 0
            
            async with self._queue_lock:
                original_count = len(self._request_queue)
                self._request_queue = [
                    req for req in self._request_queue 
                    if req.queued_at > cutoff_time
                ]
                removed_count = original_count - len(self._request_queue)
            
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} expired requests from queue")
            
            return removed_count
            
        except Exception as e:
            logger.error(f"Error cleaning up expired requests: {e}")
            return 0


# Global instance management
_fallback_manager_instance: Optional[IntelligentFallbackManager] = None


def get_intelligent_fallback_manager(availability_manager=None) -> IntelligentFallbackManager:
    """Get or create the global IntelligentFallbackManager instance"""
    global _fallback_manager_instance
    
    if _fallback_manager_instance is None:
        _fallback_manager_instance = IntelligentFallbackManager(availability_manager)
    
    return _fallback_manager_instance


async def initialize_intelligent_fallback_manager(availability_manager=None) -> IntelligentFallbackManager:
    """Initialize the intelligent fallback manager with dependencies"""
    manager = get_intelligent_fallback_manager(availability_manager)
    
    # Start background queue processing
    async def queue_processor():
        while True:
            try:
                await manager.process_queue()
                await manager.cleanup_expired_requests()
                await asyncio.sleep(manager.queue_processing_interval)
            except Exception as e:
                logger.error(f"Error in queue processor: {e}")
                await asyncio.sleep(30)  # Wait longer on error
    
    # Start the background task
    asyncio.create_task(queue_processor())
    
    return manager