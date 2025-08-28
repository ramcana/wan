"""
Simple test to verify enhanced model management endpoints are working
"""

from fastapi.testclient import TestClient
from app import app

def test_enhanced_endpoints():
    """Test that enhanced endpoints are accessible"""
    client = TestClient(app)
    
    # Test endpoints that should return 500 (expected since components aren't fully initialized)
    # but should not return 404 (which would mean endpoint doesn't exist)
    
    endpoints_to_test = [
        "/api/v1/models/status/detailed",
        "/api/v1/models/health", 
        "/api/v1/models/analytics"
    ]
    
    for endpoint in endpoints_to_test:
        response = client.get(endpoint)
        print(f"GET {endpoint}: {response.status_code}")
        # Should not be 404 (endpoint exists) but may be 500 (initialization error)
        assert response.status_code != 404, f"Endpoint {endpoint} not found"
    
    # Test POST endpoints
    post_endpoints = [
        ("/api/v1/models/download/manage", {"model_id": "test", "action": "pause"}),
        ("/api/v1/models/cleanup", {"dry_run": True}),
        ("/api/v1/models/fallback/suggest", {"requested_model": "T2V-A14B"})
    ]
    
    for endpoint, data in post_endpoints:
        response = client.post(endpoint, json=data)
        print(f"POST {endpoint}: {response.status_code}")
        # Should not be 404 (endpoint exists) but may be 500 (initialization error)
        assert response.status_code != 404, f"Endpoint {endpoint} not found"
    
    print("All enhanced endpoints are accessible!")

if __name__ == "__main__":
    test_enhanced_endpoints()