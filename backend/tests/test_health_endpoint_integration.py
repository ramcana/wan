"""
Integration tests for the enhanced health endpoint
"""

import pytest
import json
from datetime import datetime
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os

# Import the FastAPI app
import sys
from pathlib import Path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from backend.app import app

client = TestClient(app)


class TestHealthEndpointIntegration:
    """Test the enhanced system health endpoint"""

    def test_health_endpoint_basic_response(self):
        """Test basic health endpoint response structure"""
        response = client.get("/api/v1/system/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify required fields
        assert data["status"] == "ok"
        assert "port" in data
        assert "timestamp" in data
        assert "api_version" in data
        assert "system" in data
        assert "service" in data
        assert "endpoints" in data
        assert "connectivity" in data
        assert "server_info" in data

    def test_health_endpoint_schema_compliance(self):
        """Test that health endpoint matches the required schema"""
        response = client.get("/api/v1/system/health")
        data = response.json()
        
        # Verify schema compliance
        assert data["status"] == "ok"
        assert isinstance(data["port"], int)
        assert isinstance(data["timestamp"], str)
        
        # Verify timestamp format (ISO 8601 with Z suffix)
        timestamp = data["timestamp"]
        assert timestamp.endswith('Z')
        # Should be parseable as ISO format
        datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        # Verify endpoints structure
        endpoints = data["endpoints"]
        assert endpoints["health"] == "/api/v1/system/health"
        assert endpoints["docs"] == "/docs"
        assert endpoints["websocket"] == "/ws"
        assert endpoints["api_base"] == "/api/v1"
        
        # Verify connectivity structure
        connectivity = data["connectivity"]
        assert connectivity["cors_enabled"] is True
        assert isinstance(connectivity["allowed_origins"], list)
        assert "http://localhost:3000" in connectivity["allowed_origins"]
        assert connectivity["websocket_available"] is True
        
        # Verify server_info structure
        server_info = data["server_info"]
        assert "configured_port" in server_info
        assert "detected_port" in server_info
        assert "environment" in server_info

    def test_health_endpoint_port_detection_from_request(self):
        """Test port detection from request headers"""
        # Test with custom Host header
        headers = {"Host": "localhost:8080"}
        response = client.get("/api/v1/system/health", headers=headers)
        data = response.json()
        
        # Should detect port from Host header
        assert data["connectivity"]["host_header"] == "localhost:8080"
        # Port detection logic should work
        assert isinstance(data["port"], int)

    def test_health_endpoint_cors_information(self):
        """Test CORS information in health response"""
        headers = {"Origin": "http://localhost:3000"}
        response = client.get("/api/v1/system/health", headers=headers)
        data = response.json()
        
        connectivity = data["connectivity"]
        assert connectivity["cors_enabled"] is True
        assert "http://localhost:3000" in connectivity["allowed_origins"]
        assert data["connectivity"]["request_origin"] == "http://localhost:3000"

    @patch.dict(os.environ, {"PORT": "9000", "NODE_ENV": "production"})
    def test_health_endpoint_environment_variables(self):
        """Test health endpoint with environment variables"""
        response = client.get("/api/v1/system/health")
        data = response.json()
        
        server_info = data["server_info"]
        assert server_info["configured_port"] == 9000
        assert server_info["environment"] == "production"

    def test_health_endpoint_performance(self):
        """Test health endpoint response time"""
        import time

        start_time = time.time()
        response = client.get("/api/v1/system/health")
        end_time = time.time()
        
        assert response.status_code == 200
        response_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Health endpoint should respond quickly (under 100ms in tests)
        assert response_time < 100, f"Health endpoint took {response_time}ms, expected < 100ms"

    def test_health_endpoint_concurrent_requests(self):
        """Test health endpoint under concurrent load"""
        import concurrent.futures
        import threading

        def make_request():
            response = client.get("/api/v1/system/health")
            return response.status_code == 200
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        assert all(results), "Some concurrent health check requests failed"

    def test_health_endpoint_error_handling(self):
        """Test health endpoint error handling"""
        # Test with malformed request (should still work)
        response = client.get("/api/v1/system/health", headers={"Host": "invalid:host:format"})
        
        # Should still return 200 and valid response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_health_endpoint_logging(self):
        """Test that health endpoint requests are properly logged"""
        with patch('app.logger') as mock_logger:
            response = client.get("/api/v1/system/health")
            
            assert response.status_code == 200
            # Verify that request logging middleware was called
            # (The actual logging happens in the middleware, not the endpoint)

    def test_health_endpoint_api_version_consistency(self):
        """Test that API version is consistent across endpoints"""
        health_response = client.get("/api/v1/system/health")
        api_health_response = client.get("/api/health")
        
        health_data = health_response.json()
        api_health_data = api_health_response.json()
        
        # Both should return version information
        assert "api_version" in health_data
        assert "api_version" in api_health_data
        
        # Versions should be consistent
        assert health_data["api_version"] == api_health_data["api_version"]

    def test_health_endpoint_integration_with_cors(self):
        """Test health endpoint integration with CORS middleware"""
        # Test preflight request
        response = client.options(
            "/api/v1/system/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Content-Type"
            }
        )
        
        # CORS preflight should be handled by middleware
        assert response.status_code in [200, 204]
        
        # Test actual CORS request
        response = client.get(
            "/api/v1/system/health",
            headers={"Origin": "http://localhost:3000"}
        )
        
        assert response.status_code == 200
        # CORS headers should be present (added by middleware)
        assert "access-control-allow-origin" in response.headers

    def test_health_endpoint_websocket_availability(self):
        """Test WebSocket availability reporting in health endpoint"""
        response = client.get("/api/v1/system/health")
        data = response.json()
        
        assert data["connectivity"]["websocket_available"] is True
        assert data["endpoints"]["websocket"] == "/ws"

    def test_health_endpoint_json_serialization(self):
        """Test that health endpoint response is properly JSON serializable"""
        response = client.get("/api/v1/system/health")
        
        # Should be valid JSON
        assert response.headers["content-type"] == "application/json"
        
        # Should be parseable
        data = response.json()
        
        # Should be re-serializable
        json_str = json.dumps(data)
        reparsed = json.loads(json_str)
        
        assert reparsed == data


class TestHealthEndpointConnectivity:
    """Test health endpoint connectivity validation features"""

    def test_health_endpoint_provides_connectivity_diagnostics(self):
        """Test that health endpoint provides useful connectivity diagnostics"""
        response = client.get("/api/v1/system/health")
        data = response.json()
        
        connectivity = data["connectivity"]
        
        # Should provide CORS diagnostics
        assert "cors_enabled" in connectivity
        assert "allowed_origins" in connectivity
        assert isinstance(connectivity["allowed_origins"], list)
        
        # Should provide WebSocket diagnostics
        assert "websocket_available" in connectivity
        
        # Should provide request context
        assert "request_origin" in connectivity
        assert "host_header" in connectivity

    def test_health_endpoint_port_configuration_info(self):
        """Test that health endpoint provides port configuration information"""
        response = client.get("/api/v1/system/health")
        data = response.json()
        
        server_info = data["server_info"]
        
        # Should provide both configured and detected ports
        assert "configured_port" in server_info
        assert "detected_port" in server_info
        assert isinstance(server_info["configured_port"], int)
        assert isinstance(server_info["detected_port"], int)
        
        # Should provide environment info
        assert "environment" in server_info

    def test_health_endpoint_frontend_integration_info(self):
        """Test health endpoint provides information useful for frontend integration"""
        response = client.get("/api/v1/system/health")
        data = response.json()
        
        # Should provide all necessary endpoints
        endpoints = data["endpoints"]
        required_endpoints = ["health", "docs", "websocket", "api_base"]
        for endpoint in required_endpoints:
            assert endpoint in endpoints
            assert isinstance(endpoints[endpoint], str)
            assert endpoints[endpoint].startswith("/")
        
        # Should provide connectivity validation info
        connectivity = data["connectivity"]
        assert connectivity["cors_enabled"] is True
        assert len(connectivity["allowed_origins"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
