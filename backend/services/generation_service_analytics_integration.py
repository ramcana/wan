"""
Integration module for adding usage analytics to the existing generation service.
This module provides hooks and utilities to track model usage without modifying the core generation service.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from functools import wraps

from core.model_usage_analytics import (
    track_generation_usage, UsageEventType, get_model_usage_analytics
)

logger = logging.getLogger(__name__)


class GenerationServiceAnalyticsIntegration:
    """
    Integration class to add analytics tracking to the generation service.
    This class provides methods to hook into generation events and track usage.
    """
    
    def __init__(self):
        self.analytics = None
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the analytics integration"""
        try:
            self.analytics = await get_model_usage_analytics()
            self._initialized = True
            logger.info("Generation service analytics integration initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize analytics integration: {e}")
            return False
    
    async def track_generation_request(self, model_id: str, generation_params: Dict[str, Any]) -> None:
        """Track a generation request"""
        try:
            if not self._initialized:
                await self.initialize()
            
            await track_generation_usage(
                model_id=model_id,
                event_type=UsageEventType.GENERATION_REQUEST,
                generation_params=generation_params
            )
        except Exception as e:
            logger.error(f"Failed to track generation request: {e}")
    
    async def track_generation_start(self, model_id: str, generation_params: Dict[str, Any]) -> None:
        """Track when generation starts"""
        try:
            if not self._initialized:
                await self.initialize()
            
            await track_generation_usage(
                model_id=model_id,
                event_type=UsageEventType.GENERATION_START,
                generation_params=generation_params
            )
        except Exception as e:
            logger.error(f"Failed to track generation start: {e}")
    
    async def track_generation_complete(self, model_id: str, duration_seconds: float, 
                                      generation_params: Dict[str, Any],
                                      performance_metrics: Optional[Dict[str, float]] = None) -> None:
        """Track successful generation completion"""
        try:
            if not self._initialized:
                await self.initialize()
            
            await track_generation_usage(
                model_id=model_id,
                event_type=UsageEventType.GENERATION_COMPLETE,
                duration_seconds=duration_seconds,
                success=True,
                generation_params=generation_params,
                performance_metrics=performance_metrics
            )
        except Exception as e:
            logger.error(f"Failed to track generation completion: {e}")
    
    async def track_generation_failed(self, model_id: str, duration_seconds: Optional[float],
                                    error_message: str, generation_params: Dict[str, Any]) -> None:
        """Track failed generation"""
        try:
            if not self._initialized:
                await self.initialize()
            
            await track_generation_usage(
                model_id=model_id,
                event_type=UsageEventType.GENERATION_FAILED,
                duration_seconds=duration_seconds,
                success=False,
                error_message=error_message,
                generation_params=generation_params
            )
        except Exception as e:
            logger.error(f"Failed to track generation failure: {e}")
    
    async def track_model_load(self, model_id: str) -> None:
        """Track model loading"""
        try:
            if not self._initialized:
                await self.initialize()
            
            await track_generation_usage(
                model_id=model_id,
                event_type=UsageEventType.MODEL_LOAD
            )
        except Exception as e:
            logger.error(f"Failed to track model load: {e}")
    
    async def track_model_unload(self, model_id: str) -> None:
        """Track model unloading"""
        try:
            if not self._initialized:
                await self.initialize()
            
            await track_generation_usage(
                model_id=model_id,
                event_type=UsageEventType.MODEL_UNLOAD
            )
        except Exception as e:
            logger.error(f"Failed to track model unload: {e}")


# Global integration instance
_integration_instance: Optional[GenerationServiceAnalyticsIntegration] = None


async def get_analytics_integration() -> GenerationServiceAnalyticsIntegration:
    """Get or create the global analytics integration instance"""
    global _integration_instance
    
    if _integration_instance is None:
        _integration_instance = GenerationServiceAnalyticsIntegration()
        await _integration_instance.initialize()
    
    return _integration_instance


def analytics_tracked(event_type: UsageEventType):
    """
    Decorator to automatically track analytics for generation service methods.
    
    Args:
        event_type: Type of usage event to track
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract model_id and other parameters from function arguments
            model_id = None
            generation_params = {}
            
            # Try to extract model_id from various argument patterns
            if len(args) > 1 and hasattr(args[1], 'model_type'):
                model_id = args[1].model_type
                generation_params = {
                    'prompt': getattr(args[1], 'prompt', ''),
                    'resolution': getattr(args[1], 'resolution', ''),
                    'steps': getattr(args[1], 'steps', 50),
                    'lora_strength': getattr(args[1], 'lora_strength', 1.0)
                }
            elif 'model_type' in kwargs:
                model_id = kwargs['model_type']
            elif 'model_id' in kwargs:
                model_id = kwargs['model_id']
            
            start_time = datetime.now()
            
            try:
                # Track start event if applicable
                if event_type in [UsageEventType.GENERATION_START, UsageEventType.GENERATION_REQUEST]:
                    integration = await get_analytics_integration()
                    if event_type == UsageEventType.GENERATION_REQUEST:
                        await integration.track_generation_request(model_id, generation_params)
                    elif event_type == UsageEventType.GENERATION_START:
                        await integration.track_generation_start(model_id, generation_params)
                
                # Execute the original function
                result = await func(*args, **kwargs)
                
                # Track completion event
                if event_type == UsageEventType.GENERATION_COMPLETE and model_id:
                    duration = (datetime.now() - start_time).total_seconds()
                    integration = await get_analytics_integration()
                    await integration.track_generation_complete(model_id, duration, generation_params)
                
                return result
                
            except Exception as e:
                # Track failure event
                if model_id:
                    duration = (datetime.now() - start_time).total_seconds()
                    integration = await get_analytics_integration()
                    await integration.track_generation_failed(model_id, duration, str(e), generation_params)
                
                # Re-raise the exception
                raise
        
        return wrapper
    return decorator


# Utility functions for manual tracking
async def track_model_usage_event(model_id: str, event_type: UsageEventType, 
                                 duration_seconds: Optional[float] = None,
                                 success: bool = True, error_message: Optional[str] = None,
                                 generation_params: Optional[Dict[str, Any]] = None,
                                 performance_metrics: Optional[Dict[str, float]] = None) -> None:
    """
    Utility function to manually track model usage events.
    
    Args:
        model_id: ID of the model
        event_type: Type of usage event
        duration_seconds: Duration of the operation
        success: Whether the operation was successful
        error_message: Error message if failed
        generation_params: Generation parameters
        performance_metrics: Performance metrics
    """
    try:
        await track_generation_usage(
            model_id=model_id,
            event_type=event_type,
            duration_seconds=duration_seconds,
            success=success,
            error_message=error_message,
            generation_params=generation_params,
            performance_metrics=performance_metrics
        )
    except Exception as e:
        logger.error(f"Failed to track model usage event: {e}")


async def get_usage_statistics_for_model(model_id: str) -> Dict[str, Any]:
    """
    Get usage statistics for a specific model.
    
    Args:
        model_id: ID of the model
        
    Returns:
        Dictionary containing usage statistics
    """
    try:
        analytics = await get_model_usage_analytics()
        stats = await analytics.get_usage_statistics(model_id)
        
        return {
            'model_id': stats.model_id,
            'total_uses': stats.total_uses,
            'uses_per_day': stats.uses_per_day,
            'average_generation_time': stats.average_generation_time,
            'success_rate': stats.success_rate,
            'last_used': stats.last_used.isoformat() if stats.last_used else None,
            'peak_usage_hours': stats.peak_usage_hours,
            'most_common_resolutions': stats.most_common_resolutions,
            'most_common_steps': stats.most_common_steps
        }
    except Exception as e:
        logger.error(f"Failed to get usage statistics for {model_id}: {e}")
        return {}


async def get_cleanup_recommendations() -> List[Dict[str, Any]]:
    """
    Get model cleanup recommendations.
    
    Returns:
        List of cleanup recommendations
    """
    try:
        analytics = await get_model_usage_analytics()
        recommendations = await analytics.recommend_model_cleanup()
        
        return [
            {
                'model_id': rec.model_id,
                'reason': rec.reason,
                'space_saved_mb': rec.space_saved_mb,
                'last_used': rec.last_used.isoformat() if rec.last_used else None,
                'usage_frequency': rec.usage_frequency,
                'priority': rec.priority,
                'confidence_score': rec.confidence_score
            }
            for rec in recommendations
        ]
    except Exception as e:
        logger.error(f"Failed to get cleanup recommendations: {e}")
        return []


async def get_preload_recommendations() -> List[Dict[str, Any]]:
    """
    Get model preload recommendations.
    
    Returns:
        List of preload recommendations
    """
    try:
        analytics = await get_model_usage_analytics()
        recommendations = await analytics.suggest_preload_models()
        
        return [
            {
                'model_id': rec.model_id,
                'reason': rec.reason,
                'usage_frequency': rec.usage_frequency,
                'predicted_next_use': rec.predicted_next_use.isoformat() if rec.predicted_next_use else None,
                'confidence_score': rec.confidence_score,
                'priority': rec.priority
            }
            for rec in recommendations
        ]
    except Exception as e:
        logger.error(f"Failed to get preload recommendations: {e}")
        return []


async def generate_analytics_report() -> Dict[str, Any]:
    """
    Generate a comprehensive analytics report.
    
    Returns:
        Dictionary containing the analytics report
    """
    try:
        analytics = await get_model_usage_analytics()
        report = await analytics.generate_usage_report()
        
        return {
            'report_date': report.report_date.isoformat(),
            'total_models_tracked': report.total_models_tracked,
            'total_usage_events': report.total_usage_events,
            'most_used_models': report.most_used_models,
            'least_used_models': report.least_used_models,
            'performance_trends': report.performance_trends,
            'cleanup_recommendations': [
                {
                    'model_id': rec.model_id,
                    'reason': rec.reason,
                    'space_saved_mb': rec.space_saved_mb,
                    'priority': rec.priority,
                    'confidence_score': rec.confidence_score
                }
                for rec in report.cleanup_recommendations
            ],
            'preload_recommendations': [
                {
                    'model_id': rec.model_id,
                    'reason': rec.reason,
                    'usage_frequency': rec.usage_frequency,
                    'priority': rec.priority,
                    'confidence_score': rec.confidence_score
                }
                for rec in report.preload_recommendations
            ],
            'performance_recommendations': [
                {
                    'model_id': rec.model_id,
                    'issue_type': rec.issue_type,
                    'recommendation': rec.recommendation,
                    'expected_improvement': rec.expected_improvement,
                    'confidence_score': rec.confidence_score
                }
                for rec in report.performance_recommendations
            ],
            'storage_usage_mb': report.storage_usage_mb,
            'estimated_savings_mb': report.estimated_savings_mb
        }
    except Exception as e:
        logger.error(f"Failed to generate analytics report: {e}")
        return {}