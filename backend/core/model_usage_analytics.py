"""
Model Usage Analytics System - Minimal Implementation
Tracks model usage patterns and provides basic recommendations.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, Counter

from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, Text
from sqlalchemy.orm import Session

from repositories.database import Base, engine, SessionLocal

logger = logging.getLogger(__name__)


class UsageEventType(Enum):
    """Types of usage events to track"""
    GENERATION_REQUEST = "generation_request"
    GENERATION_COMPLETE = "generation_complete"
    GENERATION_FAILED = "generation_failed"
    MODEL_LOAD = "model_load"


@dataclass
class UsageData:
    """Individual usage data point"""
    model_id: str
    event_type: UsageEventType
    timestamp: datetime
    duration_seconds: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    generation_params: Optional[Dict[str, Any]] = None


@dataclass
class UsageStatistics:
    """Usage statistics for a model"""
    model_id: str
    total_uses: int = 0
    uses_per_day: float = 0.0
    average_generation_time: float = 0.0
    success_rate: float = 1.0
    last_used: Optional[datetime] = None
    last_30_days_usage: List[Any] = field(default_factory=list)
    peak_usage_hours: List[int] = field(default_factory=list)
    success_rate: float = 1.0
    last_used: Optional[datetime] = None


@dataclass
class CleanupRecommendation:
    """Recommendation for model cleanup"""
    model_id: str
    reason: str
    space_saved_mb: float
    priority: str = "low"
    confidence_score: float = 0.0


@dataclass
class PreloadRecommendation:
    """Recommendation for model preloading"""
    model_id: str
    reason: str
    usage_frequency: float
    priority: str = "medium"
    confidence_score: float = 0.0


@dataclass
class CleanupAction:
    """Individual cleanup action"""
    action_type: str  # "remove_model", "clear_cache"
    model_id: str
    space_freed_gb: float
    reason: str
    last_used: Optional[datetime] = None


@dataclass
class CleanupRecommendations:
    """Comprehensive cleanup recommendations"""
    total_space_available_gb: float
    target_space_gb: Optional[float]
    space_to_free_gb: float
    cleanup_actions: List[CleanupAction] = field(default_factory=list)


class ModelUsageEventDB(Base):
    """Database model for usage events"""
    __tablename__ = "model_usage_events"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String, nullable=False, index=True)
    event_type = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    duration_seconds = Column(Float, nullable=True)
    success = Column(Boolean, nullable=False, default=True)
    error_message = Column(Text, nullable=True)
    generation_params = Column(Text, nullable=True)


class ModelUsageAnalytics:
    """Model Usage Analytics System"""
    
    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = Path(models_dir) if models_dir else Path("models")
        self.analytics_dir = self.models_dir / ".analytics"
        self.analytics_dir.mkdir(parents=True, exist_ok=True)
        
        self._usage_cache = defaultdict(list)
        self._stats_cache = {}
        self.cleanup_threshold_days = 30
        self.preload_threshold_frequency = 0.5
        
        logger.info(f"Analytics initialized: {self.analytics_dir}")
    
    async def initialize(self) -> bool:
        """Initialize the analytics system"""
        try:
            Base.metadata.create_all(bind=engine)
            logger.info("Analytics system initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize analytics: {e}")
            return False
    
    async def track_usage(self, model_id: str, event_type: UsageEventType, 
                         duration_seconds: Optional[float] = None,
                         success: bool = True, error_message: Optional[str] = None,
                         generation_params: Optional[Dict[str, Any]] = None) -> None:
        """Track a model usage event - simplified interface"""
        try:
            usage_data = UsageData(
                model_id=model_id,
                event_type=event_type,
                timestamp=datetime.now(),
                duration_seconds=duration_seconds,
                success=success,
                error_message=error_message,
                generation_params=generation_params
            )
            await self.track_model_usage(model_id, usage_data)
        except Exception as e:
            logger.error(f"Failed to track usage: {e}")

    async def track_model_usage(self, model_id: str, usage_data: UsageData) -> None:
        """Track a model usage event"""
        try:
            self._usage_cache[model_id].append(usage_data)
            await self._store_usage_event(usage_data)
            logger.debug(f"Tracked usage for {model_id}: {usage_data.event_type}")
        except Exception as e:
            logger.error(f"Failed to track usage: {e}")
    
    async def _store_usage_event(self, usage_data: UsageData) -> None:
        """Store usage event in database"""
        try:
            db = SessionLocal()
            try:
                params_json = json.dumps(usage_data.generation_params) if usage_data.generation_params else None
                
                event = ModelUsageEventDB(
                    model_id=usage_data.model_id,
                    event_type=usage_data.event_type.value,
                    timestamp=usage_data.timestamp,
                    duration_seconds=usage_data.duration_seconds,
                    success=usage_data.success,
                    error_message=usage_data.error_message,
                    generation_params=params_json
                )
                
                db.add(event)
                db.commit()
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Failed to store event: {e}")
    
    async def get_usage_statistics_simple(self, model_id: str) -> UsageStatistics:
        """Get usage statistics for a model"""
        try:
            if model_id in self._stats_cache:
                return self._stats_cache[model_id]
            
            db = SessionLocal()
            try:
                events = db.query(ModelUsageEventDB).filter(
                    ModelUsageEventDB.model_id == model_id
                ).all()
                
                stats = UsageStatistics(model_id=model_id)
                
                if events:
                    generation_events = [e for e in events if e.event_type == UsageEventType.GENERATION_REQUEST.value]
                    successful_events = [e for e in generation_events if e.success]
                    
                    stats.total_uses = len(generation_events)
                    stats.success_rate = len(successful_events) / len(generation_events) if generation_events else 1.0
                    stats.last_used = max(e.timestamp for e in events)
                    
                    # Calculate uses per day
                    if events:
                        first_use = min(e.timestamp for e in events)
                        days_active = max((datetime.now() - first_use).days, 1)
                        stats.uses_per_day = stats.total_uses / days_active
                    
                    # Calculate average generation time
                    completed_events = [e for e in events if e.duration_seconds]
                    if completed_events:
                        times = [e.duration_seconds for e in completed_events]
                        stats.average_generation_time = sum(times) / len(times)
                
                self._stats_cache[model_id] = stats
                return stats
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return UsageStatistics(model_id=model_id)
    
    async def recommend_model_cleanup(self, target_space_gb: Optional[float] = None, 
                                    keep_recent_days: int = 30) -> CleanupRecommendations:
        """Generate comprehensive cleanup recommendations"""
        try:
            # Calculate current space usage (simplified)
            total_space_gb = 100.0  # Assume 100GB total
            used_space_gb = 60.0    # Assume 60GB used
            available_space_gb = total_space_gb - used_space_gb
            
            # Determine space to free
            if target_space_gb:
                space_to_free_gb = max(0, target_space_gb - available_space_gb)
            else:
                space_to_free_gb = 10.0  # Default 10GB cleanup
            
            cleanup_actions = []
            tracked_models = await self._get_all_tracked_models()
            
            for model_id in tracked_models:
                stats = await self.get_usage_statistics(model_id)
                
                should_cleanup = False
                cleanup_reason = ""
                
                if stats.last_used:
                    days_since_use = (datetime.now() - stats.last_used).days
                    if days_since_use > keep_recent_days:
                        should_cleanup = True
                        cleanup_reason = f"Not used for {days_since_use} days"
                elif stats.total_uses == 0:
                    should_cleanup = True
                    cleanup_reason = "Never used"
                
                if should_cleanup:
                    action = CleanupAction(
                        action_type="remove_model",
                        model_id=model_id,
                        space_freed_gb=8.0,  # Estimated 8GB per model
                        reason=cleanup_reason,
                        last_used=stats.last_used
                    )
                    cleanup_actions.append(action)
            
            return CleanupRecommendations(
                total_space_available_gb=available_space_gb,
                target_space_gb=target_space_gb,
                space_to_free_gb=space_to_free_gb,
                cleanup_actions=cleanup_actions
            )
            
        except Exception as e:
            logger.error(f"Failed to generate cleanup recommendations: {e}")
            return CleanupRecommendations(
                total_space_available_gb=0.0,
                target_space_gb=target_space_gb,
                space_to_free_gb=0.0,
                cleanup_actions=[]
            )
    
    async def get_usage_statistics(self, model_id: str, start_time: Optional[datetime] = None, 
                                 end_time: Optional[datetime] = None) -> UsageStatistics:
        """Get usage statistics for a model with time range"""
        try:
            if model_id in self._stats_cache and not start_time and not end_time:
                return self._stats_cache[model_id]
            
            db = SessionLocal()
            try:
                query = db.query(ModelUsageEventDB).filter(
                    ModelUsageEventDB.model_id == model_id
                )
                
                if start_time:
                    query = query.filter(ModelUsageEventDB.timestamp >= start_time)
                if end_time:
                    query = query.filter(ModelUsageEventDB.timestamp <= end_time)
                
                events = query.all()
                
                stats = UsageStatistics(model_id=model_id)
                
                if events:
                    generation_events = [e for e in events if e.event_type == UsageEventType.GENERATION_REQUEST.value]
                    successful_events = [e for e in generation_events if e.success]
                    
                    stats.total_uses = len(generation_events)
                    stats.success_rate = len(successful_events) / len(generation_events) if generation_events else 1.0
                    stats.last_used = max(e.timestamp for e in events)
                    
                    # Calculate uses per day
                    if events and start_time and end_time:
                        days_in_period = max((end_time - start_time).days, 1)
                        stats.uses_per_day = stats.total_uses / days_in_period
                    elif events:
                        first_use = min(e.timestamp for e in events)
                        days_active = max((datetime.now() - first_use).days, 1)
                        stats.uses_per_day = stats.total_uses / days_active
                    
                    # Calculate average generation time
                    completed_events = [e for e in events if e.duration_seconds]
                    if completed_events:
                        times = [e.duration_seconds for e in completed_events]
                        stats.average_generation_time = sum(times) / len(times)
                    
                    # Add daily usage breakdown (simplified)
                    from collections import defaultdict
                    daily_usage = defaultdict(int)
                    for event in generation_events:
                        date_key = event.timestamp.date()
                        daily_usage[date_key] += 1
                    
                    # Convert to DailyUsage objects (simplified)
                    stats.last_30_days_usage = []
                    for date, uses in daily_usage.items():
                        stats.last_30_days_usage.append(type('DailyUsage', (), {
                            'date': date,
                            'uses': uses,
                            'avg_generation_time': stats.average_generation_time,
                            'success_rate': stats.success_rate
                        })())
                    
                    # Calculate peak usage hours (simplified)
                    hour_usage = defaultdict(int)
                    for event in generation_events:
                        hour_usage[event.timestamp.hour] += 1
                    
                    if hour_usage:
                        max_usage = max(hour_usage.values())
                        stats.peak_usage_hours = [hour for hour, count in hour_usage.items() 
                                                if count >= max_usage * 0.8]
                
                if not start_time and not end_time:
                    self._stats_cache[model_id] = stats
                return stats
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return UsageStatistics(model_id=model_id)
    
    async def suggest_preload_models(self) -> List[PreloadRecommendation]:
        """Suggest models for preloading"""
        try:
            recommendations = []
            tracked_models = await self._get_all_tracked_models()
            
            for model_id in tracked_models:
                stats = await self.get_usage_statistics(model_id)
                
                if stats.uses_per_day >= self.preload_threshold_frequency:
                    rec = PreloadRecommendation(
                        model_id=model_id,
                        reason=f"High usage frequency: {stats.uses_per_day:.2f} uses/day",
                        usage_frequency=stats.uses_per_day,
                        priority="high" if stats.uses_per_day > 1.0 else "medium",
                        confidence_score=min(stats.uses_per_day / 2.0, 1.0)
                    )
                    recommendations.append(rec)
            
            return sorted(recommendations, key=lambda r: r.confidence_score, reverse=True)
        except Exception as e:
            logger.error(f"Failed to generate preload recommendations: {e}")
            return []
    
    async def _get_all_tracked_models(self) -> List[str]:
        """Get all tracked model IDs"""
        try:
            db = SessionLocal()
            try:
                result = db.query(ModelUsageEventDB.model_id).distinct().all()
                return [row[0] for row in result]
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Failed to get tracked models: {e}")
            return []


# Global instance
_analytics_instance: Optional[ModelUsageAnalytics] = None


async def get_model_usage_analytics(models_dir: Optional[str] = None) -> ModelUsageAnalytics:
    """Get or create analytics instance"""
    global _analytics_instance
    
    if _analytics_instance is None:
        _analytics_instance = ModelUsageAnalytics(models_dir=models_dir)
        await _analytics_instance.initialize()
    
    return _analytics_instance


async def track_generation_usage(model_id: str, event_type: UsageEventType, 
                               duration_seconds: Optional[float] = None,
                               success: bool = True, error_message: Optional[str] = None,
                               generation_params: Optional[Dict[str, Any]] = None) -> None:
    """Track model usage"""
    try:
        analytics = await get_model_usage_analytics()
        
        usage_data = UsageData(
            model_id=model_id,
            event_type=event_type,
            timestamp=datetime.now(),
            duration_seconds=duration_seconds,
            success=success,
            error_message=error_message,
            generation_params=generation_params
        )
        
        await analytics.track_model_usage(model_id, usage_data)
    except Exception as e:
        logger.error(f"Failed to track usage: {e}")
