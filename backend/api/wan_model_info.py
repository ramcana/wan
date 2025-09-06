"""
WAN Model Information and Capabilities API
Provides comprehensive WAN model information, capabilities, health monitoring,
performance metrics, comparison system, and dashboard integration.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
import json

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field

# Import WAN model components
try:
    from core.models.wan_models.wan_base_model import WANModelType, WANModelStatus
    from core.models.wan_models.wan_model_config import get_wan_model_config, get_wan_model_info
    from core.models.wan_models.wan_pipeline_factory import WANPipelineFactory
    from core.models.wan_models.wan_hardware_optimizer import WANHardwareOptimizer
    from core.models.wan_models.wan_vram_monitor import WANVRAMMonitor
    from core.models.wan_models.wan_progress_tracker import WANProgressTracker
    WAN_MODELS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"WAN model components not available: {e}")
    WAN_MODELS_AVAILABLE = False

# Import existing infrastructure
try:
    from backend.core.model_integration_bridge import ModelIntegrationBridge
    from backend.core.model_health_monitor import ModelHealthMonitor
    from backend.core.performance_monitor import PerformanceMonitor
    from backend.websocket.manager import get_connection_manager
    INFRASTRUCTURE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Infrastructure components not available: {e}")
    INFRASTRUCTURE_AVAILABLE = False

logger = logging.getLogger(__name__)

# Pydantic models for API responses
class WANModelCapabilities(BaseModel):
    """WAN model capabilities information"""
    model_id: str
    model_type: str
    supported_resolutions: List[str]
    max_frames: int
    min_frames: int
    supported_fps: List[float]
    input_types: List[str]
    output_formats: List[str]
    quantization_support: List[str]
    lora_support: bool
    controlnet_support: bool
    inpainting_support: bool
    img2img_support: bool
    text2img_support: bool
    batch_processing: bool
    streaming_support: bool
    hardware_requirements: Dict[str, Any]

class WANModelHealthMetrics(BaseModel):
    """WAN model health monitoring metrics"""
    model_id: str
    health_status: str
    last_check: datetime
    integrity_score: float
    performance_score: float
    availability_score: float
    error_count_24h: int
    success_rate_24h: float
    average_response_time_ms: float
    memory_usage_mb: float
    gpu_utilization_percent: float
    temperature_celsius: Optional[float]
    issues: List[str]
    recommendations: List[str]

class WANModelPerformanceMetrics(BaseModel):
    """WAN model performance metrics"""
    model_id: str
    benchmark_date: datetime
    generation_time_avg_seconds: float
    generation_time_p95_seconds: float
    throughput_videos_per_hour: float
    memory_efficiency_score: float
    gpu_utilization_avg_percent: float
    power_consumption_watts: Optional[float]
    quality_score: float
    stability_score: float
    hardware_profile: str
    optimization_level: str

class WANModelComparison(BaseModel):
    """WAN model comparison data"""
    model_a: str
    model_b: str
    comparison_date: datetime
    performance_difference_percent: float
    quality_difference_score: float
    memory_usage_difference_mb: float
    speed_difference_percent: float
    recommendation: str
    use_cases: Dict[str, str]
    trade_offs: List[str]

class WANModelRecommendation(BaseModel):
    """WAN model recommendation"""
    recommended_model: str
    confidence_score: float
    reasoning: List[str]
    alternative_models: List[str]
    hardware_requirements: Dict[str, Any]
    expected_performance: Dict[str, Any]
    limitations: List[str]

class WANModelInfoAPI:
    """WAN Model Information and Capabilities API implementation"""
    
    def __init__(self):
        self.model_bridge: Optional[ModelIntegrationBridge] = None
        self.health_monitor: Optional[ModelHealthMonitor] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.websocket_manager = None
        self.wan_pipeline_factory: Optional[WANPipelineFactory] = None
        self.wan_hardware_optimizer: Optional[WANHardwareOptimizer] = None
        self.wan_vram_monitor: Optional[WANVRAMMonitor] = None
        self.wan_progress_tracker: Optional[WANProgressTracker] = None
        self._initialized = False
        self._model_cache: Dict[str, Any] = {}
        self._metrics_cache: Dict[str, Any] = {}
        self._cache_ttl = 300  # 5 minutes
    
    async def initialize(self) -> bool:
        """Initialize the WAN Model Info API"""
        try:
            # Initialize model integration bridge
            if INFRASTRUCTURE_AVAILABLE:
                self.model_bridge = ModelIntegrationBridge()
                await self.model_bridge.initialize()
                
                # Initialize health monitor
                self.health_monitor = ModelHealthMonitor()
                await self.health_monitor.initialize()
                
                # Initialize performance monitor
                self.performance_monitor = PerformanceMonitor()
                
                # Initialize WebSocket manager
                self.websocket_manager = get_connection_manager()
            
            # Initialize WAN-specific components
            if WAN_MODELS_AVAILABLE:
                self.wan_pipeline_factory = WANPipelineFactory()
                self.wan_hardware_optimizer = WANHardwareOptimizer()
                self.wan_vram_monitor = WANVRAMMonitor()
                self.wan_progress_tracker = WANProgressTracker()
                
                # Initialize WAN components
                await self.wan_hardware_optimizer.initialize()
                await self.wan_vram_monitor.initialize()
                await self.wan_progress_tracker.initialize()
            
            self._initialized = True
            logger.info("WAN Model Info API initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize WAN Model Info API: {e}")
            return False
    
    async def get_wan_model_capabilities(self, model_type: str) -> WANModelCapabilities:
        """Get comprehensive WAN model capabilities"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get model configuration
            if WAN_MODELS_AVAILABLE:
                model_config = get_wan_model_config(model_type)
                model_info = get_wan_model_info(model_type)
            else:
                # Fallback configuration
                model_config = self._get_fallback_model_config(model_type)
                model_info = self._get_fallback_model_info(model_type)
            
            # Determine capabilities based on model type
            capabilities = self._determine_model_capabilities(model_type, model_config, model_info)
            
            return WANModelCapabilities(
                model_id=f"WAN2.2-{model_type}",
                model_type=model_type,
                supported_resolutions=capabilities["resolutions"],
                max_frames=capabilities["max_frames"],
                min_frames=capabilities["min_frames"],
                supported_fps=capabilities["fps_options"],
                input_types=capabilities["input_types"],
                output_formats=capabilities["output_formats"],
                quantization_support=capabilities["quantization"],
                lora_support=capabilities["lora_support"],
                controlnet_support=capabilities["controlnet_support"],
                inpainting_support=capabilities["inpainting_support"],
                img2img_support=capabilities["img2img_support"],
                text2img_support=capabilities["text2img_support"],
                batch_processing=capabilities["batch_processing"],
                streaming_support=capabilities["streaming_support"],
                hardware_requirements=capabilities["hardware_requirements"]
            )
            
        except Exception as e:
            logger.error(f"Error getting WAN model capabilities for {model_type}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get model capabilities: {str(e)}")
    
    async def get_wan_model_health_metrics(self, model_type: str) -> WANModelHealthMetrics:
        """Get WAN model health monitoring metrics"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get health data from health monitor
            if self.health_monitor:
                health_result = await self.health_monitor.check_model_integrity(model_type)
                health_status = health_result.health_status.value if hasattr(health_result.health_status, 'value') else str(health_result.health_status)
                integrity_score = health_result.integrity_score
                issues = health_result.issues
                recommendations = health_result.repair_suggestions
            else:
                # Fallback health check
                health_status = "unknown"
                integrity_score = 0.8
                issues = []
                recommendations = []
            
            # Get performance metrics
            performance_score = await self._calculate_performance_score(model_type)
            availability_score = await self._calculate_availability_score(model_type)
            
            # Get recent error statistics
            error_stats = await self._get_error_statistics(model_type)
            
            # Get resource usage
            resource_usage = await self._get_resource_usage(model_type)
            
            return WANModelHealthMetrics(
                model_id=f"WAN2.2-{model_type}",
                health_status=health_status,
                last_check=datetime.now(),
                integrity_score=integrity_score,
                performance_score=performance_score,
                availability_score=availability_score,
                error_count_24h=error_stats["error_count"],
                success_rate_24h=error_stats["success_rate"],
                average_response_time_ms=error_stats["avg_response_time"],
                memory_usage_mb=resource_usage["memory_mb"],
                gpu_utilization_percent=resource_usage["gpu_utilization"],
                temperature_celsius=resource_usage.get("temperature"),
                issues=issues,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error getting WAN model health metrics for {model_type}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get health metrics: {str(e)}")
    
    async def get_wan_model_performance_metrics(self, model_type: str) -> WANModelPerformanceMetrics:
        """Get WAN model performance metrics"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get performance data from performance monitor
            if self.performance_monitor and hasattr(self.performance_monitor, 'get_model_performance_data'):
                perf_data = await self.performance_monitor.get_model_performance_data(model_type)
            else:
                perf_data = await self._get_fallback_performance_data(model_type)
            
            # Get hardware profile
            hardware_profile = "unknown"
            if self.model_bridge and self.model_bridge.hardware_profile:
                hardware_profile = f"{self.model_bridge.hardware_profile.gpu_name} ({self.model_bridge.hardware_profile.total_vram_gb:.1f}GB)"
            
            # Calculate optimization level
            optimization_level = await self._determine_optimization_level(model_type)
            
            return WANModelPerformanceMetrics(
                model_id=f"WAN2.2-{model_type}",
                benchmark_date=datetime.now(),
                generation_time_avg_seconds=perf_data["avg_generation_time"],
                generation_time_p95_seconds=perf_data["p95_generation_time"],
                throughput_videos_per_hour=perf_data["throughput"],
                memory_efficiency_score=perf_data["memory_efficiency"],
                gpu_utilization_avg_percent=perf_data["gpu_utilization"],
                power_consumption_watts=perf_data.get("power_consumption"),
                quality_score=perf_data["quality_score"],
                stability_score=perf_data["stability_score"],
                hardware_profile=hardware_profile,
                optimization_level=optimization_level
            )
            
        except Exception as e:
            logger.error(f"Error getting WAN model performance metrics for {model_type}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")
    
    async def compare_wan_models(self, model_a: str, model_b: str) -> WANModelComparison:
        """Compare two WAN models"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get performance metrics for both models
            perf_a = await self.get_wan_model_performance_metrics(model_a)
            perf_b = await self.get_wan_model_performance_metrics(model_b)
            
            # Calculate differences
            performance_diff = ((perf_b.generation_time_avg_seconds - perf_a.generation_time_avg_seconds) / perf_a.generation_time_avg_seconds) * 100
            quality_diff = perf_b.quality_score - perf_a.quality_score
            memory_diff = perf_b.memory_efficiency_score - perf_a.memory_efficiency_score
            speed_diff = ((perf_a.throughput_videos_per_hour - perf_b.throughput_videos_per_hour) / perf_b.throughput_videos_per_hour) * 100
            
            # Generate recommendation
            recommendation = self._generate_comparison_recommendation(model_a, model_b, perf_a, perf_b)
            
            # Define use cases
            use_cases = self._define_model_use_cases(model_a, model_b, perf_a, perf_b)
            
            # Identify trade-offs
            trade_offs = self._identify_trade_offs(model_a, model_b, perf_a, perf_b)
            
            return WANModelComparison(
                model_a=model_a,
                model_b=model_b,
                comparison_date=datetime.now(),
                performance_difference_percent=performance_diff,
                quality_difference_score=quality_diff,
                memory_usage_difference_mb=memory_diff,
                speed_difference_percent=speed_diff,
                recommendation=recommendation,
                use_cases=use_cases,
                trade_offs=trade_offs
            )
            
        except Exception as e:
            logger.error(f"Error comparing WAN models {model_a} vs {model_b}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to compare models: {str(e)}")
    
    async def get_wan_model_recommendation(self, 
                                         use_case: str,
                                         quality_priority: str = "medium",
                                         speed_priority: str = "medium",
                                         memory_constraint_gb: Optional[float] = None) -> WANModelRecommendation:
        """Get WAN model recommendation based on requirements"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get all available models
            available_models = ["T2V-A14B", "I2V-A14B", "TI2V-5B"]
            
            # Score models based on requirements
            model_scores = {}
            for model_type in available_models:
                try:
                    score = await self._score_model_for_requirements(
                        model_type, use_case, quality_priority, speed_priority, memory_constraint_gb
                    )
                    model_scores[model_type] = score
                except Exception as e:
                    logger.warning(f"Failed to score model {model_type}: {e}")
                    model_scores[model_type] = 0.0
            
            # Sort by score
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            
            if not sorted_models:
                raise HTTPException(status_code=404, detail="No suitable models found")
            
            recommended_model = sorted_models[0][0]
            confidence_score = sorted_models[0][1]
            alternative_models = [model for model, _ in sorted_models[1:3]]
            
            # Generate reasoning
            reasoning = await self._generate_recommendation_reasoning(
                recommended_model, use_case, quality_priority, speed_priority, memory_constraint_gb
            )
            
            # Get hardware requirements
            hardware_requirements = await self._get_model_hardware_requirements(recommended_model)
            
            # Get expected performance
            expected_performance = await self._get_expected_performance(recommended_model)
            
            # Get limitations
            limitations = await self._get_model_limitations(recommended_model)
            
            return WANModelRecommendation(
                recommended_model=recommended_model,
                confidence_score=confidence_score,
                reasoning=reasoning,
                alternative_models=alternative_models,
                hardware_requirements=hardware_requirements,
                expected_performance=expected_performance,
                limitations=limitations
            )
            
        except Exception as e:
            logger.error(f"Error getting WAN model recommendation: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get recommendation: {str(e)}")
    
    async def get_wan_model_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data for all WAN models"""
        if not self._initialized:
            await self.initialize()
        
        try:
            available_models = ["T2V-A14B", "I2V-A14B", "TI2V-5B"]
            dashboard_data = {
                "timestamp": datetime.now().isoformat(),
                "models": {},
                "system_overview": {},
                "alerts": [],
                "recommendations": []
            }
            
            # Get data for each model
            for model_type in available_models:
                try:
                    capabilities = await self.get_wan_model_capabilities(model_type)
                    health = await self.get_wan_model_health_metrics(model_type)
                    performance = await self.get_wan_model_performance_metrics(model_type)
                    
                    model_data = {
                        "capabilities": capabilities.dict(),
                        "health": health.dict(),
                        "performance": performance.dict()
                    }
                    dashboard_data["models"][model_type] = model_data
                    
                    # Check for alerts
                    health_status = health.health_status
                    if health_status in ["degraded", "critical"]:
                        dashboard_data["alerts"].append({
                            "model": model_type,
                            "severity": health_status,
                            "message": f"Model {model_type} health is {health_status}",
                            "timestamp": datetime.now().isoformat()
                        })
                    
                except Exception as e:
                    logger.warning(f"Failed to get dashboard data for {model_type}: {e}")
                    dashboard_data["models"][model_type] = {"error": str(e)}
            
            # Generate system overview
            dashboard_data["system_overview"] = await self._generate_system_overview(dashboard_data["models"])
            
            # Generate recommendations
            dashboard_data["recommendations"] = await self._generate_dashboard_recommendations(dashboard_data["models"])
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error getting WAN model dashboard data: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")
    
    # Helper methods
    def _get_fallback_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get fallback model configuration"""
        base_config = {
            "max_sequence_length": 77,
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "scheduler": "DPMSolverMultistepScheduler"
        }
        
        if model_type == "T2V-A14B":
            base_config.update({
                "model_size": "14B",
                "input_modalities": ["text"],
                "max_frames": 16,
                "base_resolution": "1280x720"
            })
        elif model_type == "I2V-A14B":
            base_config.update({
                "model_size": "14B",
                "input_modalities": ["image", "text"],
                "max_frames": 16,
                "base_resolution": "1280x720"
            })
        elif model_type == "TI2V-5B":
            base_config.update({
                "model_size": "5B",
                "input_modalities": ["text", "image"],
                "max_frames": 16,
                "base_resolution": "1280x720"
            })
        
        return base_config
    
    def _get_fallback_model_info(self, model_type: str) -> Dict[str, Any]:
        """Get fallback model information"""
        return {
            "name": f"WAN2.2-{model_type}",
            "version": "2.2.0",
            "description": f"WAN {model_type} video generation model",
            "architecture": "Diffusion Transformer",
            "training_data": "High-quality video dataset",
            "license": "Custom License"
        }
    
    def _determine_model_capabilities(self, model_type: str, config: Dict[str, Any], info: Dict[str, Any]) -> Dict[str, Any]:
        """Determine model capabilities based on type and configuration"""
        base_capabilities = {
            "resolutions": ["854x480", "1024x576", "1280x720", "1920x1080"],
            "max_frames": config.get("max_frames", 16),
            "min_frames": 8,
            "fps_options": [8.0, 12.0, 16.0, 24.0],
            "output_formats": ["mp4", "webm"],
            "quantization": ["fp16", "bf16", "int8"],
            "lora_support": True,
            "controlnet_support": False,
            "inpainting_support": False,
            "batch_processing": True,
            "streaming_support": False,
            "hardware_requirements": {
                "min_vram_gb": 8.0,
                "recommended_vram_gb": 12.0,
                "min_ram_gb": 16.0,
                "cuda_compute_capability": "7.0"
            }
        }
        
        # Model-specific capabilities
        if model_type == "T2V-A14B":
            base_capabilities.update({
                "input_types": ["text"],
                "text2img_support": False,
                "img2img_support": False,
                "hardware_requirements": {
                    "min_vram_gb": 10.0,
                    "recommended_vram_gb": 16.0,
                    "min_ram_gb": 32.0,
                    "cuda_compute_capability": "8.0"
                }
            })
        elif model_type == "I2V-A14B":
            base_capabilities.update({
                "input_types": ["image", "text"],
                "text2img_support": False,
                "img2img_support": True,
                "hardware_requirements": {
                    "min_vram_gb": 10.0,
                    "recommended_vram_gb": 16.0,
                    "min_ram_gb": 32.0,
                    "cuda_compute_capability": "8.0"
                }
            })
        elif model_type == "TI2V-5B":
            base_capabilities.update({
                "input_types": ["text", "image"],
                "text2img_support": False,
                "img2img_support": True,
                "hardware_requirements": {
                    "min_vram_gb": 6.0,
                    "recommended_vram_gb": 10.0,
                    "min_ram_gb": 16.0,
                    "cuda_compute_capability": "7.0"
                }
            })
        
        return base_capabilities
    
    async def _calculate_performance_score(self, model_type: str) -> float:
        """Calculate performance score for a model"""
        try:
            # This would integrate with actual performance monitoring
            # For now, return estimated scores based on model type
            scores = {
                "T2V-A14B": 0.85,
                "I2V-A14B": 0.82,
                "TI2V-5B": 0.88
            }
            return scores.get(model_type, 0.75)
        except Exception:
            return 0.75
    
    async def _calculate_availability_score(self, model_type: str) -> float:
        """Calculate availability score for a model"""
        try:
            # Check if model is available and loaded
            if self.model_bridge:
                status = await self.model_bridge.check_model_availability(model_type)
                if status.status.value == "loaded":
                    return 1.0
                elif status.status.value == "available":
                    return 0.8
                else:
                    return 0.0
            return 0.5
        except Exception:
            return 0.5
    
    async def _get_error_statistics(self, model_type: str) -> Dict[str, Any]:
        """Get error statistics for a model"""
        # This would integrate with actual error tracking
        return {
            "error_count": 2,
            "success_rate": 0.95,
            "avg_response_time": 1250.0
        }
    
    async def _get_resource_usage(self, model_type: str) -> Dict[str, Any]:
        """Get current resource usage for a model"""
        try:
            if self.wan_vram_monitor:
                vram_usage = await self.wan_vram_monitor.get_current_usage()
                return {
                    "memory_mb": vram_usage.get("used_mb", 0),
                    "gpu_utilization": vram_usage.get("utilization_percent", 0),
                    "temperature": vram_usage.get("temperature_celsius")
                }
            else:
                # Fallback estimates
                estimates = {
                    "T2V-A14B": {"memory_mb": 8500, "gpu_utilization": 75},
                    "I2V-A14B": {"memory_mb": 9000, "gpu_utilization": 78},
                    "TI2V-5B": {"memory_mb": 6000, "gpu_utilization": 65}
                }
                return estimates.get(model_type, {"memory_mb": 7000, "gpu_utilization": 70})
        except Exception:
            return {"memory_mb": 7000, "gpu_utilization": 70}
    
    async def _get_fallback_performance_data(self, model_type: str) -> Dict[str, Any]:
        """Get fallback performance data"""
        performance_data = {
            "T2V-A14B": {
                "avg_generation_time": 45.0,
                "p95_generation_time": 65.0,
                "throughput": 80.0,
                "memory_efficiency": 0.82,
                "gpu_utilization": 75.0,
                "quality_score": 0.88,
                "stability_score": 0.85
            },
            "I2V-A14B": {
                "avg_generation_time": 50.0,
                "p95_generation_time": 70.0,
                "throughput": 72.0,
                "memory_efficiency": 0.80,
                "gpu_utilization": 78.0,
                "quality_score": 0.90,
                "stability_score": 0.83
            },
            "TI2V-5B": {
                "avg_generation_time": 35.0,
                "p95_generation_time": 50.0,
                "throughput": 100.0,
                "memory_efficiency": 0.88,
                "gpu_utilization": 65.0,
                "quality_score": 0.85,
                "stability_score": 0.90
            }
        }
        
        return performance_data.get(model_type, {
            "avg_generation_time": 40.0,
            "p95_generation_time": 60.0,
            "throughput": 85.0,
            "memory_efficiency": 0.80,
            "gpu_utilization": 70.0,
            "quality_score": 0.85,
            "stability_score": 0.85
        })
    
    async def _determine_optimization_level(self, model_type: str) -> str:
        """Determine optimization level for a model"""
        try:
            if self.wan_hardware_optimizer:
                optimization_status = await self.wan_hardware_optimizer.get_optimization_status(model_type)
                return optimization_status.get("level", "standard")
            return "standard"
        except Exception:
            return "standard"
    
    def _generate_comparison_recommendation(self, model_a: str, model_b: str, perf_a: WANModelPerformanceMetrics, perf_b: WANModelPerformanceMetrics) -> str:
        """Generate comparison recommendation"""
        if perf_a.quality_score > perf_b.quality_score and perf_a.throughput_videos_per_hour > perf_b.throughput_videos_per_hour:
            return f"Use {model_a} for better quality and performance"
        elif perf_b.quality_score > perf_a.quality_score and perf_b.throughput_videos_per_hour > perf_a.throughput_videos_per_hour:
            return f"Use {model_b} for better quality and performance"
        elif perf_a.quality_score > perf_b.quality_score:
            return f"Use {model_a} for better quality, {model_b} for better speed"
        else:
            return f"Use {model_b} for better quality, {model_a} for better speed"
    
    def _define_model_use_cases(self, model_a: str, model_b: str, perf_a: WANModelPerformanceMetrics, perf_b: WANModelPerformanceMetrics) -> Dict[str, str]:
        """Define use cases for each model"""
        use_cases = {}
        
        if "T2V" in model_a:
            use_cases[model_a] = "Text-to-video generation, creative content creation"
        elif "I2V" in model_a:
            use_cases[model_a] = "Image-to-video animation, product demonstrations"
        elif "TI2V" in model_a:
            use_cases[model_a] = "Text+image-to-video, interpolation, storytelling"
        
        if "T2V" in model_b:
            use_cases[model_b] = "Text-to-video generation, creative content creation"
        elif "I2V" in model_b:
            use_cases[model_b] = "Image-to-video animation, product demonstrations"
        elif "TI2V" in model_b:
            use_cases[model_b] = "Text+image-to-video, interpolation, storytelling"
        
        return use_cases
    
    def _identify_trade_offs(self, model_a: str, model_b: str, perf_a: WANModelPerformanceMetrics, perf_b: WANModelPerformanceMetrics) -> List[str]:
        """Identify trade-offs between models"""
        trade_offs = []
        
        if perf_a.quality_score > perf_b.quality_score:
            trade_offs.append(f"{model_a} offers higher quality but may be slower")
        if perf_b.throughput_videos_per_hour > perf_a.throughput_videos_per_hour:
            trade_offs.append(f"{model_b} is faster but may use more memory")
        if perf_a.memory_efficiency_score > perf_b.memory_efficiency_score:
            trade_offs.append(f"{model_a} is more memory efficient")
        
        return trade_offs
    
    async def _score_model_for_requirements(self, model_type: str, use_case: str, quality_priority: str, speed_priority: str, memory_constraint_gb: Optional[float]) -> float:
        """Score a model based on requirements"""
        try:
            perf_metrics = await self.get_wan_model_performance_metrics(model_type)
            capabilities = await self.get_wan_model_capabilities(model_type)
            
            score = 0.0
            
            # Quality scoring
            quality_weight = {"low": 0.2, "medium": 0.5, "high": 0.8}.get(quality_priority, 0.5)
            score += perf_metrics.quality_score * quality_weight
            
            # Speed scoring
            speed_weight = {"slow": 0.2, "medium": 0.5, "fast": 0.8}.get(speed_priority, 0.5)
            speed_score = min(perf_metrics.throughput_videos_per_hour / 100.0, 1.0)
            score += speed_score * speed_weight
            
            # Memory constraint scoring
            if memory_constraint_gb:
                required_memory = capabilities.hardware_requirements.get("min_vram_gb", 8.0)
                if required_memory <= memory_constraint_gb:
                    score += 0.3
                else:
                    score -= 0.5  # Penalty for exceeding memory constraint
            
            # Use case matching
            use_case_bonus = self._calculate_use_case_bonus(model_type, use_case)
            score += use_case_bonus
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.warning(f"Error scoring model {model_type}: {e}")
            return 0.0
    
    def _calculate_use_case_bonus(self, model_type: str, use_case: str) -> float:
        """Calculate bonus score based on use case matching"""
        use_case_lower = use_case.lower()
        
        if "text" in use_case_lower and "T2V" in model_type:
            return 0.2
        elif "image" in use_case_lower and "I2V" in model_type:
            return 0.2
        elif ("text" in use_case_lower and "image" in use_case_lower) and "TI2V" in model_type:
            return 0.3
        elif "fast" in use_case_lower and "5B" in model_type:
            return 0.15
        elif "quality" in use_case_lower and "A14B" in model_type:
            return 0.15
        
        return 0.0
    
    async def _generate_recommendation_reasoning(self, model_type: str, use_case: str, quality_priority: str, speed_priority: str, memory_constraint_gb: Optional[float]) -> List[str]:
        """Generate reasoning for model recommendation"""
        reasoning = []
        
        if "T2V" in model_type:
            reasoning.append("Optimized for text-to-video generation")
        elif "I2V" in model_type:
            reasoning.append("Specialized for image-to-video animation")
        elif "TI2V" in model_type:
            reasoning.append("Supports both text and image inputs for versatile generation")
        
        if "5B" in model_type:
            reasoning.append("Smaller model size provides faster inference and lower memory usage")
        elif "A14B" in model_type:
            reasoning.append("Larger model size delivers higher quality results")
        
        if quality_priority == "high":
            reasoning.append("Selected for high quality output requirements")
        elif speed_priority == "fast":
            reasoning.append("Selected for fast generation requirements")
        
        if memory_constraint_gb:
            reasoning.append(f"Fits within {memory_constraint_gb}GB memory constraint")
        
        return reasoning
    
    async def _get_model_hardware_requirements(self, model_type: str) -> Dict[str, Any]:
        """Get hardware requirements for a model"""
        try:
            capabilities = await self.get_wan_model_capabilities(model_type)
            return capabilities.hardware_requirements
        except Exception:
            # Fallback requirements
            requirements = {
                "T2V-A14B": {"min_vram_gb": 10.0, "recommended_vram_gb": 16.0, "min_ram_gb": 32.0},
                "I2V-A14B": {"min_vram_gb": 10.0, "recommended_vram_gb": 16.0, "min_ram_gb": 32.0},
                "TI2V-5B": {"min_vram_gb": 6.0, "recommended_vram_gb": 10.0, "min_ram_gb": 16.0}
            }
            return requirements.get(model_type, {"min_vram_gb": 8.0, "recommended_vram_gb": 12.0, "min_ram_gb": 16.0})
    
    async def _get_expected_performance(self, model_type: str) -> Dict[str, Any]:
        """Get expected performance for a model"""
        try:
            perf_metrics = await self.get_wan_model_performance_metrics(model_type)
            return {
                "generation_time_seconds": perf_metrics.generation_time_avg_seconds,
                "quality_score": perf_metrics.quality_score,
                "throughput_videos_per_hour": perf_metrics.throughput_videos_per_hour,
                "memory_efficiency": perf_metrics.memory_efficiency_score
            }
        except Exception:
            # Fallback performance data
            performance = {
                "T2V-A14B": {"generation_time_seconds": 45.0, "quality_score": 0.88, "throughput_videos_per_hour": 80.0, "memory_efficiency": 0.82},
                "I2V-A14B": {"generation_time_seconds": 50.0, "quality_score": 0.90, "throughput_videos_per_hour": 72.0, "memory_efficiency": 0.80},
                "TI2V-5B": {"generation_time_seconds": 35.0, "quality_score": 0.85, "throughput_videos_per_hour": 100.0, "memory_efficiency": 0.88}
            }
            return performance.get(model_type, {"generation_time_seconds": 40.0, "quality_score": 0.85, "throughput_videos_per_hour": 85.0, "memory_efficiency": 0.80})
    
    async def _get_model_limitations(self, model_type: str) -> List[str]:
        """Get limitations for a model"""
        limitations = []
        
        if "A14B" in model_type:
            limitations.append("Requires high-end GPU with 16GB+ VRAM for optimal performance")
            limitations.append("Longer generation times due to model size")
        elif "5B" in model_type:
            limitations.append("Lower quality compared to larger models")
            limitations.append("Limited detail in complex scenes")
        
        if "T2V" in model_type:
            limitations.append("Text-only input, no image conditioning")
        elif "I2V" in model_type:
            limitations.append("Requires input image, cannot generate from text alone")
        
        limitations.append("Maximum 16 frames per generation")
        limitations.append("Limited to specific aspect ratios")
        
        return limitations
    
    async def _generate_system_overview(self, models_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate system overview from models data"""
        total_models = len(models_data)
        healthy_models = sum(1 for data in models_data.values() 
                           if isinstance(data, dict) and data.get("health", {}).get("health_status") == "healthy")
        
        avg_performance = 0.0
        if models_data:
            performance_scores = []
            for data in models_data.values():
                if isinstance(data, dict) and "performance" in data:
                    perf_data = data["performance"]
                    if isinstance(perf_data, dict):
                        performance_scores.append(perf_data.get("quality_score", 0.0))
                    else:
                        performance_scores.append(getattr(perf_data, "quality_score", 0.0))
            if performance_scores:
                avg_performance = sum(performance_scores) / len(performance_scores)
        
        return {
            "total_models": total_models,
            "healthy_models": healthy_models,
            "average_performance_score": avg_performance,
            "system_status": "healthy" if healthy_models == total_models else "degraded"
        }
    
    async def _generate_dashboard_recommendations(self, models_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate dashboard recommendations"""
        recommendations = []
        
        for model_type, data in models_data.items():
            if isinstance(data, dict) and "health" in data:
                health = data["health"]
                if health.health_status in ["degraded", "critical"]:
                    recommendations.append({
                        "type": "health_issue",
                        "model": model_type,
                        "priority": "high" if health.health_status == "critical" else "medium",
                        "message": f"Model {model_type} requires attention",
                        "action": "Check model integrity and consider redownloading"
                    })
                
                if health.gpu_utilization_percent > 90:
                    recommendations.append({
                        "type": "resource_optimization",
                        "model": model_type,
                        "priority": "medium",
                        "message": f"High GPU utilization detected for {model_type}",
                        "action": "Consider enabling quantization or CPU offloading"
                    })
        
        return recommendations


# Global instance
_wan_model_info_api = None

async def get_wan_model_info_api() -> WANModelInfoAPI:
    """Get the global WAN Model Info API instance"""
    global _wan_model_info_api
    if _wan_model_info_api is None:
        _wan_model_info_api = WANModelInfoAPI()
        await _wan_model_info_api.initialize()
    return _wan_model_info_api

# Create FastAPI router
router = APIRouter(prefix="/api/v1/wan-models", tags=["WAN Model Information"])

@router.get("/capabilities/{model_type}", response_model=WANModelCapabilities)
async def get_model_capabilities(
    model_type: str,
    api: WANModelInfoAPI = Depends(get_wan_model_info_api)
):
    """Get WAN model capabilities"""
    return await api.get_wan_model_capabilities(model_type)

@router.get("/health/{model_type}", response_model=WANModelHealthMetrics)
async def get_model_health(
    model_type: str,
    api: WANModelInfoAPI = Depends(get_wan_model_info_api)
):
    """Get WAN model health metrics"""
    return await api.get_wan_model_health_metrics(model_type)

@router.get("/performance/{model_type}", response_model=WANModelPerformanceMetrics)
async def get_model_performance(
    model_type: str,
    api: WANModelInfoAPI = Depends(get_wan_model_info_api)
):
    """Get WAN model performance metrics"""
    return await api.get_wan_model_performance_metrics(model_type)

@router.get("/compare/{model_a}/{model_b}", response_model=WANModelComparison)
async def compare_models(
    model_a: str,
    model_b: str,
    api: WANModelInfoAPI = Depends(get_wan_model_info_api)
):
    """Compare two WAN models"""
    return await api.compare_wan_models(model_a, model_b)

@router.get("/recommend", response_model=WANModelRecommendation)
async def get_model_recommendation(
    use_case: str = Query(..., description="Use case for the model"),
    quality_priority: str = Query("medium", description="Quality priority: low, medium, high"),
    speed_priority: str = Query("medium", description="Speed priority: slow, medium, fast"),
    memory_constraint_gb: Optional[float] = Query(None, description="Memory constraint in GB"),
    api: WANModelInfoAPI = Depends(get_wan_model_info_api)
):
    """Get WAN model recommendation based on requirements"""
    return await api.get_wan_model_recommendation(use_case, quality_priority, speed_priority, memory_constraint_gb)

@router.get("/dashboard")
async def get_dashboard_data(
    api: WANModelInfoAPI = Depends(get_wan_model_info_api)
):
    """Get comprehensive dashboard data for all WAN models"""
    return await api.get_wan_model_dashboard_data()

@router.get("/status")
async def get_wan_models_status():
    """Get overall WAN models system status"""
    return {
        "status": "operational",
        "available_models": ["T2V-A14B", "I2V-A14B", "TI2V-5B"],
        "api_version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }