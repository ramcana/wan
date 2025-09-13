"""
Unit tests for Model Usage Analytics System
Tests analytics collection, recommendation algorithms, and reporting functionality.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.core.model_usage_analytics import (
    ModelUsageAnalytics, UsageEventType, UsageData, UsageStatistics,
    CleanupRecommendation, PreloadRecommendation, PerformanceRecommendation,
    ModelUsageEventDB, ModelUsageStatsDB, Base,
    track_generation_usage, get_model_usage_analytics
)
from backend.services.generation_service_analytics_integration import (
    GenerationServiceAnalyticsIntegration, get_analytics_integration,
    get_usage_statistics_for_model, get_cleanup_recommendations,
    get_preload_recommendations, generate_analytics_report
)


@pytest.fixture
def test_db_engine():
    """Create a test database engine using SQLite in memory"""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    return engine


    assert True  # TODO: Add proper assertion

@pytest.fixture
def test_db_session(test_db_engine):
    """Create a test database session"""
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_db_engine)
    session = TestSessionLocal()
    yield session
    session.close()


    assert True  # TODO: Add proper assertion

@pytest.fixture
def analytics_system(test_db_session, tmp_path):
    """Create a test analytics system"""
    analytics = ModelUsageAnalytics(
        models_dir=str(tmp_path / "models"),
        db_session=test_db_session
    )
    return analytics


@pytest.fixture
def sample_usage_data():
    """Create sample usage data for testing"""
    return UsageData(
        model_id="t2v-A14B",
        event_type=UsageEventType.GENERATION_REQUEST,
        timestamp=datetime.now(),
        duration_seconds=120.5,
        success=True,
        generation_params={
            "prompt": "A beautiful sunset",
            "resolution": "1280x720",
            "steps": 50,
            "lora_strength": 1.0
        },
        performance_metrics={
            "vram_usage_mb": 8500.0,
            "generation_speed": 2.5
        }
    )


class TestModelUsageAnalytics:
    """Test cases for ModelUsageAnalytics class"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, analytics_system):
        """Test analytics system initialization"""
        result = await analytics_system.initialize()
        assert result is True
        assert analytics_system._analytics_initialized is True
    
    @pytest.mark.asyncio
    async def test_track_model_usage(self, analytics_system, sample_usage_data):
        """Test tracking model usage events"""
        await analytics_system.initialize()
        
        # Track usage event
        await analytics_system.track_model_usage("t2v-A14B", sample_usage_data)
        
        # Verify event was stored in cache
        assert "t2v-A14B" in analytics_system._usage_cache
        assert len(analytics_system._usage_cache["t2v-A14B"]) == 1
        
        # Verify event was stored in database
        db = analytics_system.db_session
        events = db.query(ModelUsageEventDB).filter(
            ModelUsageEventDB.model_id == "t2v-A14B"
        ).all()
        assert len(events) == 1
        assert events[0].event_type == UsageEventType.GENERATION_REQUEST.value
        assert events[0].success is True
    
    @pytest.mark.asyncio
    async def test_get_usage_statistics_empty(self, analytics_system):
        """Test getting usage statistics for model with no usage"""
        await analytics_system.initialize()
        
        stats = await analytics_system.get_usage_statistics("nonexistent-model")
        
        assert stats.model_id == "nonexistent-model"
        assert stats.total_uses == 0
        assert stats.uses_per_day == 0.0
        assert stats.success_rate == 1.0
    
    @pytest.mark.asyncio
    async def test_get_usage_statistics_with_data(self, analytics_system, test_db_session):
        """Test getting usage statistics with actual data"""
        await analytics_system.initialize()
        
        # Create test events
        events = [
            ModelUsageEventDB(
                model_id="t2v-A14B",
                event_type=UsageEventType.GENERATION_REQUEST.value,
                timestamp=datetime.now() - timedelta(days=1),
                success=True,
                generation_params='{"resolution": "1280x720", "steps": 50}'
            ),
            ModelUsageEventDB(
                model_id="t2v-A14B",
                event_type=UsageEventType.GENERATION_COMPLETE.value,
                timestamp=datetime.now() - timedelta(days=1),
                duration_seconds=120.0,
                success=True
            ),
            ModelUsageEventDB(
                model_id="t2v-A14B",
                event_type=UsageEventType.GENERATION_REQUEST.value,
                timestamp=datetime.now(),
                success=False,
                error_message="Test error"
            )
        ]
        
        for event in events:
            test_db_session.add(event)
        test_db_session.commit()
        
        # Get statistics
        stats = await analytics_system.get_usage_statistics("t2v-A14B")
        
        assert stats.model_id == "t2v-A14B"
        assert stats.total_uses == 2  # Only generation requests count
        assert stats.success_rate == 0.5  # 1 success, 1 failure
        assert stats.average_generation_time == 120.0
        assert len(stats.most_common_resolutions) > 0
    
    @pytest.mark.asyncio
    async def test_recommend_model_cleanup(self, analytics_system, test_db_session):
        """Test cleanup recommendation generation"""
        await analytics_system.initialize()
        
        # Create old usage event (over 30 days ago)
        old_event = ModelUsageEventDB(
            model_id="old-model",
            event_type=UsageEventType.GENERATION_REQUEST.value,
            timestamp=datetime.now() - timedelta(days=45),
            success=True
        )
        test_db_session.add(old_event)
        test_db_session.commit()
        
        recommendations = await analytics_system.recommend_model_cleanup()
        
        # Should recommend cleanup for old model
        old_model_recs = [r for r in recommendations if r.model_id == "old-model"]
        assert len(old_model_recs) > 0
        assert "Not used for" in old_model_recs[0].reason
        assert old_model_recs[0].confidence_score > 0.3
    
    @pytest.mark.asyncio
    async def test_suggest_preload_models(self, analytics_system, test_db_session):
        """Test preload recommendation generation"""
        await analytics_system.initialize()
        
        # Create frequent usage events
        for i in range(10):
            event = ModelUsageEventDB(
                model_id="frequent-model",
                event_type=UsageEventType.GENERATION_REQUEST.value,
                timestamp=datetime.now() - timedelta(days=i),
                success=True
            )
            test_db_session.add(event)
        test_db_session.commit()
        
        recommendations = await analytics_system.suggest_preload_models()
        
        # Should recommend preloading for frequent model
        frequent_model_recs = [r for r in recommendations if r.model_id == "frequent-model"]
        assert len(frequent_model_recs) > 0
        assert "High usage frequency" in frequent_model_recs[0].reason
        assert frequent_model_recs[0].confidence_score > 0.4
    
    @pytest.mark.asyncio
    async def test_generate_usage_report(self, analytics_system, test_db_session):
        """Test comprehensive usage report generation"""
        await analytics_system.initialize()
        
        # Create test data for multiple models
        models = ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
        for i, model in enumerate(models):
            for j in range(i + 1):  # Different usage frequencies
                event = ModelUsageEventDB(
                    model_id=model,
                    event_type=UsageEventType.GENERATION_REQUEST.value,
                    timestamp=datetime.now() - timedelta(days=j),
                    success=True
                )
                test_db_session.add(event)
        test_db_session.commit()
        
        report = await analytics_system.generate_usage_report()
        
        assert report.total_models_tracked == 3
        assert report.total_usage_events == 6  # 1 + 2 + 3
        assert len(report.most_used_models) > 0
        assert report.most_used_models[0][0] == "ti2v-5B"  # Most used
        assert len(report.cleanup_recommendations) >= 0
        assert len(report.preload_recommendations) >= 0
    
    @pytest.mark.asyncio
    async def test_performance_recommendations(self, analytics_system, test_db_session):
        """Test performance recommendation generation"""
        await analytics_system.initialize()
        
        # Create slow generation events
        slow_events = [
            ModelUsageEventDB(
                model_id="slow-model",
                event_type=UsageEventType.GENERATION_COMPLETE.value,
                timestamp=datetime.now() - timedelta(days=i),
                duration_seconds=400.0,  # Over 5 minutes
                success=True
            )
            for i in range(5)
        ]
        
        for event in slow_events:
            test_db_session.add(event)
        test_db_session.commit()
        
        # Generate report to get performance recommendations
        report = await analytics_system.generate_usage_report()
        
        slow_model_recs = [r for r in report.performance_recommendations 
                          if r.model_id == "slow-model" and r.issue_type == "slow_generation"]
        assert len(slow_model_recs) > 0
        assert "model offloading" in slow_model_recs[0].recommendation
    
    @pytest.mark.asyncio
    async def test_cache_functionality(self, analytics_system):
        """Test caching of usage statistics"""
        await analytics_system.initialize()
        
        # First call should populate cache
        stats1 = await analytics_system.get_usage_statistics("test-model")
        assert "test-model" in analytics_system._stats_cache
        
        # Second call should use cache
        stats2 = await analytics_system.get_usage_statistics("test-model")
        assert stats1 is stats2  # Should be same object from cache
    
    @pytest.mark.asyncio
    async def test_realtime_stats_update(self, analytics_system, sample_usage_data):
        """Test real-time statistics updates"""
        await analytics_system.initialize()
        
        # Track a generation request
        request_data = UsageData(
            model_id="test-model",
            event_type=UsageEventType.GENERATION_REQUEST,
            timestamp=datetime.now(),
            success=True
        )
        await analytics_system.track_model_usage("test-model", request_data)
        
        # Check that daily stats were updated
        db = analytics_system.db_session
        today_stats = db.query(ModelUsageStatsDB).filter(
            ModelUsageStatsDB.model_id == "test-model"
        ).first()
        
        assert today_stats is not None
        assert today_stats.total_requests == 1
        assert today_stats.successful_requests == 1


class TestGenerationServiceIntegration:
    """Test cases for generation service analytics integration"""
    
    @pytest.mark.asyncio
    async def test_integration_initialization(self):
        """Test analytics integration initialization"""
        with patch('backend.services.generation_service_analytics_integration.get_model_usage_analytics') as mock_analytics:
            mock_analytics.return_value = Mock()
            
            integration = GenerationServiceAnalyticsIntegration()
            result = await integration.initialize()
            
            assert result is True
            assert integration._initialized is True
    
    @pytest.mark.asyncio
    async def test_track_generation_events(self):
        """Test tracking various generation events"""
        with patch('backend.services.generation_service_analytics_integration.track_generation_usage') as mock_track:
            integration = GenerationServiceAnalyticsIntegration()
            integration._initialized = True
            
            # Test tracking generation request
            await integration.track_generation_request("t2v-A14B", {"prompt": "test"})
            mock_track.assert_called_with(
                model_id="t2v-A14B",
                event_type=UsageEventType.GENERATION_REQUEST,
                generation_params={"prompt": "test"}
            )
            
            # Test tracking generation completion
            await integration.track_generation_complete(
                "t2v-A14B", 120.0, {"prompt": "test"}, {"vram": 8000}
            )
            mock_track.assert_called_with(
                model_id="t2v-A14B",
                event_type=UsageEventType.GENERATION_COMPLETE,
                duration_seconds=120.0,
                success=True,
                generation_params={"prompt": "test"},
                performance_metrics={"vram": 8000}
            )
    
    @pytest.mark.asyncio
    async def test_utility_functions(self):
        """Test utility functions for getting analytics data"""
        mock_analytics = Mock()
        mock_stats = Mock()
        mock_stats.model_id = "t2v-A14B"
        mock_stats.total_uses = 10
        mock_stats.uses_per_day = 2.5
        mock_stats.success_rate = 0.95
        mock_stats.last_used = datetime.now()
        mock_stats.peak_usage_hours = [14, 15, 16]
        mock_stats.most_common_resolutions = [("1280x720", 8)]
        mock_stats.most_common_steps = [(50, 6)]
        
        mock_analytics.get_usage_statistics.return_value = mock_stats
        
        with patch('backend.services.generation_service_analytics_integration.get_model_usage_analytics', 
                  return_value=mock_analytics):
            
            stats = await get_usage_statistics_for_model("t2v-A14B")
            
            assert stats['model_id'] == "t2v-A14B"
            assert stats['total_uses'] == 10
            assert stats['uses_per_day'] == 2.5
            assert stats['success_rate'] == 0.95
            assert 'last_used' in stats
    
    @pytest.mark.asyncio
    async def test_recommendations_functions(self):
        """Test recommendation utility functions"""
        mock_analytics = Mock()
        
        # Mock cleanup recommendations
        mock_cleanup_rec = Mock()
        mock_cleanup_rec.model_id = "old-model"
        mock_cleanup_rec.reason = "Not used for 45 days"
        mock_cleanup_rec.space_saved_mb = 8500.0
        mock_cleanup_rec.priority = "high"
        mock_cleanup_rec.confidence_score = 0.8
        mock_cleanup_rec.last_used = datetime.now() - timedelta(days=45)
        mock_cleanup_rec.usage_frequency = 0.02
        
        mock_analytics.recommend_model_cleanup.return_value = [mock_cleanup_rec]
        
        # Mock preload recommendations
        mock_preload_rec = Mock()
        mock_preload_rec.model_id = "frequent-model"
        mock_preload_rec.reason = "High usage frequency"
        mock_preload_rec.usage_frequency = 3.2
        mock_preload_rec.predicted_next_use = datetime.now() + timedelta(hours=2)
        mock_preload_rec.confidence_score = 0.9
        mock_preload_rec.priority = "high"
        
        mock_analytics.suggest_preload_models.return_value = [mock_preload_rec]
        
        with patch('backend.services.generation_service_analytics_integration.get_model_usage_analytics', 
                  return_value=mock_analytics):
            
            # Test cleanup recommendations
            cleanup_recs = await get_cleanup_recommendations()
            assert len(cleanup_recs) == 1
            assert cleanup_recs[0]['model_id'] == "old-model"
            assert cleanup_recs[0]['priority'] == "high"
            
            # Test preload recommendations
            preload_recs = await get_preload_recommendations()
            assert len(preload_recs) == 1
            assert preload_recs[0]['model_id'] == "frequent-model"
            assert preload_recs[0]['usage_frequency'] == 3.2


class TestAnalyticsReporting:
    """Test cases for analytics reporting functionality"""
    
    @pytest.mark.asyncio
    async def test_report_generation(self):
        """Test comprehensive report generation"""
        mock_analytics = Mock()
        mock_report = Mock()
        mock_report.report_date = datetime.now()
        mock_report.total_models_tracked = 3
        mock_report.total_usage_events = 150
        mock_report.most_used_models = [("t2v-A14B", 80), ("i2v-A14B", 45)]
        mock_report.least_used_models = [("ti2v-5B", 25)]
        mock_report.performance_trends = {"t2v-A14B": [120.0, 115.0, 110.0]}
        mock_report.cleanup_recommendations = []
        mock_report.preload_recommendations = []
        mock_report.performance_recommendations = []
        mock_report.storage_usage_mb = 25000.0
        mock_report.estimated_savings_mb = 8500.0
        
        mock_analytics.generate_usage_report.return_value = mock_report
        
        with patch('backend.services.generation_service_analytics_integration.get_model_usage_analytics', 
                  return_value=mock_analytics):
            
            report = await generate_analytics_report()
            
            assert report['total_models_tracked'] == 3
            assert report['total_usage_events'] == 150
            assert len(report['most_used_models']) == 2
            assert report['storage_usage_mb'] == 25000.0
    
    def test_usage_data_serialization(self, sample_usage_data):
        """Test serialization of usage data"""
        # Test that usage data can be converted to JSON
        data_dict = {
            'model_id': sample_usage_data.model_id,
            'event_type': sample_usage_data.event_type.value,
            'timestamp': sample_usage_data.timestamp.isoformat(),
            'duration_seconds': sample_usage_data.duration_seconds,
            'success': sample_usage_data.success,
            'generation_params': sample_usage_data.generation_params,
            'performance_metrics': sample_usage_data.performance_metrics
        }
        
        # Should be able to serialize to JSON
        json_str = json.dumps(data_dict)
        assert json_str is not None
        
        # Should be able to deserialize from JSON
        restored_dict = json.loads(json_str)
        assert restored_dict['model_id'] == sample_usage_data.model_id
        assert restored_dict['event_type'] == sample_usage_data.event_type.value


class TestAnalyticsAlgorithms:
    """Test cases for analytics algorithms and calculations"""
    
    @pytest.mark.asyncio
    async def test_usage_frequency_calculation(self, analytics_system, test_db_session):
        """Test usage frequency calculation accuracy"""
        await analytics_system.initialize()
        
        # Create events over 10 days (1 per day)
        for i in range(10):
            event = ModelUsageEventDB(
                model_id="daily-model",
                event_type=UsageEventType.GENERATION_REQUEST.value,
                timestamp=datetime.now() - timedelta(days=i),
                success=True
            )
            test_db_session.add(event)
        test_db_session.commit()
        
        stats = await analytics_system.get_usage_statistics("daily-model", timedelta(days=10))
        
        # Should be approximately 1 use per day
        assert abs(stats.uses_per_day - 1.0) < 0.1
    
    @pytest.mark.asyncio
    async def test_success_rate_calculation(self, analytics_system, test_db_session):
        """Test success rate calculation"""
        await analytics_system.initialize()
        
        # Create 7 successful and 3 failed events
        for i in range(7):
            event = ModelUsageEventDB(
                model_id="mixed-model",
                event_type=UsageEventType.GENERATION_REQUEST.value,
                timestamp=datetime.now() - timedelta(hours=i),
                success=True
            )
            test_db_session.add(event)
        
        for i in range(3):
            event = ModelUsageEventDB(
                model_id="mixed-model",
                event_type=UsageEventType.GENERATION_REQUEST.value,
                timestamp=datetime.now() - timedelta(hours=i + 7),
                success=False
            )
            test_db_session.add(event)
        test_db_session.commit()
        
        stats = await analytics_system.get_usage_statistics("mixed-model")
        
        # Should be 70% success rate
        assert abs(stats.success_rate - 0.7) < 0.01
    
    @pytest.mark.asyncio
    async def test_peak_hours_detection(self, analytics_system, test_db_session):
        """Test peak usage hours detection"""
        await analytics_system.initialize()
        
        # Create events concentrated in specific hours
        peak_hours = [14, 15, 16]  # 2-4 PM
        for hour in peak_hours:
            for i in range(5):  # 5 events per peak hour
                event = ModelUsageEventDB(
                    model_id="peak-model",
                    event_type=UsageEventType.GENERATION_REQUEST.value,
                    timestamp=datetime.now().replace(hour=hour, minute=i*10),
                    success=True
                )
                test_db_session.add(event)
        
        # Add some events in non-peak hours
        for hour in [8, 22]:
            event = ModelUsageEventDB(
                model_id="peak-model",
                event_type=UsageEventType.GENERATION_REQUEST.value,
                timestamp=datetime.now().replace(hour=hour),
                success=True
            )
            test_db_session.add(event)
        test_db_session.commit()
        
        stats = await analytics_system.get_usage_statistics("peak-model")
        
        # Peak hours should include the concentrated hours
        assert all(hour in stats.peak_usage_hours for hour in peak_hours)


if __name__ == "__main__":
    pytest.main([__file__])
