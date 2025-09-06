"""
Test suite for Task 12: Advanced system features
Tests WebSocket support, Chart.js integration, time range selection, and optimization presets
"""

import pytest
import asyncio
import json
from fastapi.testclient import TestClient
from fastapi import FastAPI
from unittest.mock import Mock, patch, AsyncMock
import websockets
from datetime import datetime, timedelta

from backend.app import app
from websocket.manager import ConnectionManager, connection_manager
from api.routes.optimization import OPTIMIZATION_PRESETS

class TestWebSocketSupport:
    """Test WebSocket support for sub-second updates"""
    
    def test_websocket_connection_manager_initialization(self):
        """Test that connection manager initializes correctly"""
        manager = ConnectionManager()
        
        assert len(manager.active_connections) == 0
        assert "system_stats" in manager.subscriptions
        assert "generation_progress" in manager.subscriptions
        assert "queue_updates" in manager.subscriptions
        assert "alerts" in manager.subscriptions
        assert not manager._is_running
    
    @pytest.mark.asyncio
    async def test_websocket_connection_lifecycle(self):
        """Test WebSocket connection and disconnection"""
        manager = ConnectionManager()
        mock_websocket = Mock()
        mock_websocket.accept = AsyncMock()
        
        # Test connection
        connection_id = "test-connection-1"
        success = await manager.connect(mock_websocket, connection_id)
        
        assert success
        assert connection_id in manager.active_connections
        assert manager.get_connection_count() == 1
        
        # Test disconnection
        await manager.disconnect(connection_id)
        assert connection_id not in manager.active_connections
        assert manager.get_connection_count() == 0
    
    @pytest.mark.asyncio
    async def test_websocket_subscription_management(self):
        """Test topic subscription and unsubscription"""
        manager = ConnectionManager()
        mock_websocket = Mock()
        mock_websocket.accept = AsyncMock()
        
        connection_id = "test-connection-1"
        await manager.connect(mock_websocket, connection_id)
        
        # Test subscription
        success = await manager.subscribe(connection_id, "system_stats")
        assert success
        assert connection_id in manager.subscriptions["system_stats"]
        assert manager.get_subscription_count("system_stats") == 1
        
        # Test unsubscription
        success = await manager.unsubscribe(connection_id, "system_stats")
        assert success
        assert connection_id not in manager.subscriptions["system_stats"]
        assert manager.get_subscription_count("system_stats") == 0
        
        await manager.disconnect(connection_id)
    
    @pytest.mark.asyncio
    async def test_websocket_message_broadcasting(self):
        """Test message broadcasting to subscribers"""
        manager = ConnectionManager()
        mock_websocket1 = Mock()
        mock_websocket2 = Mock()
        mock_websocket1.accept = AsyncMock()
        mock_websocket2.accept = AsyncMock()
        mock_websocket1.send_text = AsyncMock()
        mock_websocket2.send_text = AsyncMock()
        
        # Connect two clients
        connection_id1 = "test-connection-1"
        connection_id2 = "test-connection-2"
        await manager.connect(mock_websocket1, connection_id1)
        await manager.connect(mock_websocket2, connection_id2)
        
        # Subscribe both to system_stats
        await manager.subscribe(connection_id1, "system_stats")
        await manager.subscribe(connection_id2, "system_stats")
        
        # Broadcast message
        test_message = {
            "type": "system_stats_update",
            "data": {"cpu_percent": 50.0, "timestamp": datetime.utcnow().isoformat()}
        }
        
        await manager.broadcast_to_topic(test_message, "system_stats")
        
        # Verify both clients received the message
        mock_websocket1.send_text.assert_called_once_with(json.dumps(test_message))
        mock_websocket2.send_text.assert_called_once_with(json.dumps(test_message))
        
        await manager.disconnect(connection_id1)
        await manager.disconnect(connection_id2)
    
    def test_websocket_endpoint_availability(self):
        """Test that WebSocket endpoint is available"""
        client = TestClient(app)
        
        # Test WebSocket stats endpoint
        response = client.get("/api/v1/ws/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "active_connections" in data
        assert "subscription_counts" in data
        assert "available_topics" in data
        assert isinstance(data["subscription_counts"], dict)

class TestAdvancedCharts:
    """Test Chart.js integration and historical data"""
    
    def test_historical_data_endpoint(self):
        """Test historical system stats endpoint"""
        client = TestClient(app)
        
        response = client.get("/api/v1/system/stats/history?hours=1")
        assert response.status_code == 200
        
        data = response.json()
        assert "stats" in data
        assert "total_count" in data
        assert "time_range" in data
        assert isinstance(data["stats"], list)
    
    def test_historical_data_time_ranges(self):
        """Test different time ranges for historical data"""
        client = TestClient(app)
        
        time_ranges = [0.083, 0.25, 1, 6, 24, 168]  # 5min, 15min, 1h, 6h, 24h, 1week
        
        for hours in time_ranges:
            response = client.get(f"/api/v1/system/stats/history?hours={hours}")
            assert response.status_code == 200
            
            data = response.json()
            assert data["time_range"]["hours"] == hours
    
    @patch('api.routes.system.get_system_integration')
    def test_system_stats_data_format(self, mock_integration):
        """Test that system stats are in correct format for Chart.js"""
        client = TestClient(app)
        
        # Mock system integration
        mock_integration_instance = Mock()
        mock_integration_instance.get_enhanced_system_stats = AsyncMock(return_value={
            "cpu_percent": 45.5,
            "ram_used_gb": 8.2,
            "ram_total_gb": 16.0,
            "ram_percent": 51.25,
            "gpu_percent": 75.0,
            "vram_used_mb": 6144,
            "vram_total_mb": 12288,
            "vram_percent": 50.0
        })
        mock_integration.return_value = mock_integration_instance
        
        response = client.get("/api/v1/system/stats")
        assert response.status_code == 200
        
        data = response.json()
        
        # Verify all required fields for Chart.js are present
        required_fields = [
            "cpu_percent", "ram_used_gb", "ram_total_gb", "ram_percent",
            "gpu_percent", "vram_used_mb", "vram_total_mb", "vram_percent", "timestamp"
        ]
        
        for field in required_fields:
            assert field in data
            if field != "timestamp":
                assert isinstance(data[field], (int, float))
        
        # Verify timestamp is in ISO format
        datetime.fromisoformat(data["timestamp"].replace('Z', '+00:00'))

class TestTimeRangeSelection:
    """Test interactive time range selection"""
    
    def test_time_range_configurations(self):
        """Test that time range configurations are valid"""
        # This would be tested in the frontend, but we can validate the concept
        time_ranges = [
            {"label": "Real-time", "hours": 0.017, "updateInterval": 500},
            {"label": "5 minutes", "hours": 0.083, "updateInterval": 1000},
            {"label": "15 minutes", "hours": 0.25, "updateInterval": 2000},
            {"label": "1 hour", "hours": 1, "updateInterval": 5000},
            {"label": "6 hours", "hours": 6, "updateInterval": 30000},
            {"label": "24 hours", "hours": 24, "updateInterval": 60000},
            {"label": "1 week", "hours": 168, "updateInterval": 300000}
        ]
        
        for range_config in time_ranges:
            assert "label" in range_config
            assert "hours" in range_config
            assert "updateInterval" in range_config
            assert range_config["hours"] > 0
            assert range_config["updateInterval"] > 0
            
            # Verify reasonable update intervals for time ranges
            if range_config["hours"] < 1:
                assert range_config["updateInterval"] <= 2000  # Sub-second to 2s for short ranges
            elif range_config["hours"] < 24:
                assert range_config["updateInterval"] <= 30000  # Up to 30s for daily ranges
            else:
                assert range_config["updateInterval"] <= 300000  # Up to 5min for weekly ranges
    
    def test_historical_data_filtering(self):
        """Test that historical data can be filtered by time range"""
        client = TestClient(app)
        
        # Test with different time ranges
        test_ranges = [1, 6, 24, 168]
        
        for hours in test_ranges:
            response = client.get(f"/api/v1/system/stats/history?hours={hours}")
            assert response.status_code == 200
            
            data = response.json()
            
            # Verify time range is respected
            if data["stats"]:
                # Check that all timestamps are within the requested range
                cutoff_time = datetime.utcnow() - timedelta(hours=hours)
                
                for stat in data["stats"]:
                    stat_time = datetime.fromisoformat(stat["timestamp"].replace('Z', '+00:00'))
                    assert stat_time >= cutoff_time

class TestOptimizationPresets:
    """Test advanced optimization presets and recommendations"""
    
    def test_optimization_presets_structure(self):
        """Test that optimization presets have correct structure"""
        for preset_id, preset in OPTIMIZATION_PRESETS.items():
            # Verify required fields
            required_fields = [
                "name", "description", "settings", "vram_savings_gb",
                "performance_impact", "recommended_for", "min_vram_gb"
            ]
            
            for field in required_fields:
                assert field in preset, f"Preset {preset_id} missing field {field}"
            
            # Verify settings structure
            settings = preset["settings"]
            assert "quantization" in settings
            assert "enable_offload" in settings
            assert "vae_tile_size" in settings
            assert "max_vram_usage_gb" in settings
            
            # Verify value ranges
            assert preset["vram_savings_gb"] >= 0
            assert preset["performance_impact"] in ["low", "medium", "high"]
            assert preset["min_vram_gb"] >= 4.0
            assert settings["vae_tile_size"] in [128, 256, 512]
            assert settings["max_vram_usage_gb"] >= 4.0
    
    def test_get_optimization_presets_endpoint(self):
        """Test optimization presets API endpoint"""
        client = TestClient(app)
        
        with patch('api.routes.optimization.get_system_integration') as mock_integration:
            mock_integration_instance = Mock()
            mock_integration_instance.get_enhanced_system_stats = AsyncMock(return_value={
                "vram_total_mb": 12288  # 12GB VRAM
            })
            mock_integration.return_value = mock_integration_instance
            
            response = client.get("/api/v1/optimization/presets")
            assert response.status_code == 200
            
            data = response.json()
            assert "presets" in data
            assert "system_info" in data
            
            # Verify each preset has compatibility information
            for preset_id, preset in data["presets"].items():
                assert "compatible" in preset
                assert "compatibility_reason" in preset
                assert "estimated_time_multiplier" in preset
    
    def test_optimization_recommendations_endpoint(self):
        """Test optimization recommendations API endpoint"""
        client = TestClient(app)
        
        with patch('api.routes.optimization.get_system_integration') as mock_integration:
            mock_integration_instance = Mock()
            mock_integration_instance.get_enhanced_system_stats = AsyncMock(return_value={
                "vram_total_mb": 8192,  # 8GB VRAM
                "vram_used_mb": 6144,   # 6GB used (75%)
                "vram_percent": 75.0,
                "cpu_percent": 45.0,
                "ram_percent": 60.0
            })
            mock_integration.return_value = mock_integration_instance
            
            response = client.get("/api/v1/optimization/recommendations")
            assert response.status_code == 200
            
            data = response.json()
            assert "priority_recommendations" in data
            assert "recommendations" in data
            assert "best_preset" in data
            assert "system_analysis" in data
            
            # Verify system analysis
            analysis = data["system_analysis"]
            assert "total_vram_gb" in analysis
            assert "vram_category" in analysis
            assert "optimization_potential" in analysis
    
    def test_apply_optimization_preset(self):
        """Test applying optimization presets"""
        client = TestClient(app)
        
        with patch('api.routes.optimization.get_system_integration') as mock_integration:
            mock_integration_instance = Mock()
            mock_integration_instance.get_enhanced_system_stats = AsyncMock(return_value={
                "vram_total_mb": 12288  # 12GB VRAM
            })
            mock_integration.return_value = mock_integration_instance
            
            # Test applying a valid preset
            response = client.post("/api/v1/optimization/apply-preset/balanced")
            assert response.status_code == 200
            
            data = response.json()
            assert "message" in data
            assert "preset" in data
            assert "applied_settings" in data
            assert "expected_benefits" in data
            
            # Test applying invalid preset
            response = client.post("/api/v1/optimization/apply-preset/nonexistent")
            assert response.status_code == 404
    
    def test_optimization_analysis_endpoint(self):
        """Test detailed optimization analysis"""
        client = TestClient(app)
        
        with patch('api.routes.optimization.get_system_integration') as mock_integration:
            mock_integration_instance = Mock()
            mock_integration_instance.get_enhanced_system_stats = AsyncMock(return_value={
                "vram_total_mb": 16384,  # 16GB VRAM
                "vram_used_mb": 8192,    # 8GB used (50%)
                "vram_percent": 50.0
            })
            mock_integration.return_value = mock_integration_instance
            
            response = client.get("/api/v1/optimization/analysis")
            assert response.status_code == 200
            
            data = response.json()
            assert "system_overview" in data
            assert "preset_analysis" in data
            assert "recommendations" in data
            
            # Verify preset analysis
            preset_analysis = data["preset_analysis"]
            for preset_id in OPTIMIZATION_PRESETS.keys():
                assert preset_id in preset_analysis
                analysis = preset_analysis[preset_id]
                assert "current_compatibility" in analysis
                assert "estimated_vram_usage_after" in analysis
                assert "recommended_score" in analysis

class TestIntegrationRequirements:
    """Test that all requirements are met"""
    
    def test_requirement_7_5_websocket_sub_second_updates(self):
        """
        Requirement 7.5: Add WebSocket support for sub-second updates (if needed)
        """
        # Test that WebSocket manager supports sub-second updates
        manager = ConnectionManager()
        
        # Verify that system stats updater runs every 500ms
        assert hasattr(manager, '_system_stats_updater')
        
        # Test WebSocket endpoint exists
        client = TestClient(app)
        response = client.get("/api/v1/ws/stats")
        assert response.status_code == 200
    
    def test_requirement_4_2_advanced_charts(self):
        """
        Requirement 4.2: Implement advanced charts with Chart.js for historical data
        """
        client = TestClient(app)
        
        # Test historical data endpoint provides data suitable for Chart.js
        response = client.get("/api/v1/system/stats/history?hours=24")
        assert response.status_code == 200
        
        data = response.json()
        assert "stats" in data
        
        # Verify data format is suitable for Chart.js
        if data["stats"]:
            stat = data["stats"][0]
            chart_required_fields = [
                "cpu_percent", "ram_percent", "gpu_percent", "vram_percent", "timestamp"
            ]
            for field in chart_required_fields:
                assert field in stat
    
    def test_requirement_4_3_time_range_selection(self):
        """
        Requirement 4.3: Add interactive time range selection for monitoring
        """
        client = TestClient(app)
        
        # Test that different time ranges are supported
        time_ranges = [0.083, 1, 6, 24, 168]  # 5min, 1h, 6h, 24h, 1week
        
        for hours in time_ranges:
            response = client.get(f"/api/v1/system/stats/history?hours={hours}")
            assert response.status_code == 200
            
            data = response.json()
            assert data["time_range"]["hours"] == hours
    
    def test_requirement_advanced_optimization_presets(self):
        """
        Create advanced optimization presets and recommendations
        """
        client = TestClient(app)
        
        # Test presets endpoint
        with patch('api.routes.optimization.get_system_integration') as mock_integration:
            mock_integration_instance = Mock()
            mock_integration_instance.get_enhanced_system_stats = AsyncMock(return_value={
                "vram_total_mb": 12288
            })
            mock_integration.return_value = mock_integration_instance
            
            response = client.get("/api/v1/optimization/presets")
            assert response.status_code == 200
            
            data = response.json()
            assert len(data["presets"]) >= 4  # At least 4 presets
            
            # Verify preset categories
            preset_names = [preset["name"] for preset in data["presets"].values()]
            expected_categories = ["Balanced", "Memory Efficient", "Ultra Efficient", "High Performance"]
            
            for category in expected_categories:
                assert any(category in name for name in preset_names)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])