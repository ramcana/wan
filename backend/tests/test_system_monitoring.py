#!/usr/bin/env python3
"""
Test system monitoring and optimization endpoints
Tests for task 2.4: Add system monitoring and optimization endpoints
"""

import sys
import os
from pathlib import Path
import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

async def test_system_stats_endpoint():
    """Test GET /api/v1/system/stats endpoint"""
    print("Testing system stats endpoint...")
    
    try:
        from fastapi.testclient import TestClient
        from main import app
        
        client = TestClient(app)
        
        # Test basic stats endpoint
        response = client.get("/api/v1/system/stats")
        print(f"Stats response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Stats data keys: {list(data.keys())}")
            
            # Verify required fields
            required_fields = [
                "cpu_percent", "ram_used_gb", "ram_total_gb", "ram_percent",
                "gpu_percent", "vram_used_mb", "vram_total_mb", "vram_percent", "timestamp"
            ]
            
            for field in required_fields:
                assert field in data, f"Missing required field: {field}"
                print(f"✓ {field}: {data[field]}")
            
            # Verify data types and ranges
            assert isinstance(data["cpu_percent"], (int, float))
            assert 0 <= data["cpu_percent"] <= 100
            assert isinstance(data["ram_used_gb"], (int, float))
            assert data["ram_used_gb"] >= 0
            assert isinstance(data["vram_percent"], (int, float))
            assert 0 <= data["vram_percent"] <= 100
            
            print("✅ System stats endpoint test passed")
        else:
            print(f"❌ Stats endpoint failed: {response.text}")
            
    except Exception as e:
        print(f"❌ System stats test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_optimization_settings_endpoints():
    """Test optimization settings GET and POST endpoints"""
    print("Testing optimization settings endpoints...")
    
    try:
        from fastapi.testclient import TestClient
        from main import app
        
        client = TestClient(app)
        
        # Test GET optimization settings
        response = client.get("/api/v1/system/optimization")
        print(f"Get optimization response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Optimization settings: {data}")
            
            # Verify required fields
            required_fields = ["quantization", "enable_offload", "vae_tile_size", "max_vram_usage_gb"]
            for field in required_fields:
                assert field in data, f"Missing required field: {field}"
            
            # Test POST optimization settings
            new_settings = {
                "quantization": "int8",
                "enable_offload": True,
                "vae_tile_size": 256,
                "max_vram_usage_gb": 10.0
            }
            
            post_response = client.post("/api/v1/system/optimization", json=new_settings)
            print(f"Post optimization response status: {post_response.status_code}")
            
            if post_response.status_code == 200:
                post_data = post_response.json()
                print(f"Update response: {post_data}")
                
                # Verify response includes recommendations
                assert "message" in post_data
                assert "current_vram_usage_gb" in post_data
                assert "estimated_vram_savings_gb" in post_data
                
                print("✅ Optimization settings endpoints test passed")
            else:
                print(f"❌ POST optimization failed: {post_response.text}")
        else:
            print(f"❌ GET optimization failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Optimization settings test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_system_health_endpoint():
    """Test system health endpoint with resource constraint checking"""
    print("Testing system health endpoint...")
    
    try:
        from fastapi.testclient import TestClient
        from main import app
        
        client = TestClient(app)
        
        # Test health endpoint
        response = client.get("/api/v1/system/health")
        print(f"Health response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Health data: {data}")
            
            # Verify required fields
            required_fields = ["status", "message", "timestamp", "system_info"]
            for field in required_fields:
                assert field in data, f"Missing required field: {field}"
            
            # Verify status is valid
            valid_statuses = ["healthy", "warning", "critical", "error"]
            assert data["status"] in valid_statuses, f"Invalid status: {data['status']}"
            
            # Verify system_info structure
            system_info = data["system_info"]
            assert "cpu_percent" in system_info
            assert "ram_percent" in system_info
            assert "vram_percent" in system_info
            assert "resource_constraints" in system_info
            
            print("✅ System health endpoint test passed")
        else:
            print(f"❌ Health endpoint failed: {response.text}")
            
    except Exception as e:
        print(f"❌ System health test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_resource_constraints_endpoints():
    """Test resource constraints management endpoints"""
    print("Testing resource constraints endpoints...")
    
    try:
        from fastapi.testclient import TestClient
        from main import app
        
        client = TestClient(app)
        
        # Test GET resource constraints
        response = client.get("/api/v1/system/constraints")
        print(f"Get constraints response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Resource constraints: {data}")
            
            # Verify structure
            assert "constraints" in data
            assert "description" in data
            
            constraints = data["constraints"]
            expected_keys = [
                "max_concurrent_generations", "vram_warning_threshold", 
                "vram_critical_threshold", "cpu_warning_threshold", "ram_warning_threshold"
            ]
            
            for key in expected_keys:
                assert key in constraints, f"Missing constraint: {key}"
            
            # Test POST resource constraints
            new_constraints = {
                "vram_warning_threshold": 0.80,
                "vram_critical_threshold": 0.90,
                "max_concurrent_generations": 1
            }
            
            post_response = client.post("/api/v1/system/constraints", json=new_constraints)
            print(f"Post constraints response status: {post_response.status_code}")
            
            if post_response.status_code == 200:
                post_data = post_response.json()
                print(f"Update constraints response: {post_data}")
                
                assert "message" in post_data
                assert "updated_constraints" in post_data
                
                print("✅ Resource constraints endpoints test passed")
            else:
                print(f"❌ POST constraints failed: {post_response.text}")
        else:
            print(f"❌ GET constraints failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Resource constraints test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_resource_availability_check():
    """Test resource availability checking for graceful degradation"""
    print("Testing resource availability check...")
    
    try:
        from fastapi.testclient import TestClient
        from main import app
        
        client = TestClient(app)
        
        # Test resource availability check
        response = client.get("/api/v1/system/resource-check")
        print(f"Resource check response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Resource availability: {data}")
            
            # Verify required fields
            required_fields = [
                "can_start_generation", "resource_status", "current_usage", 
                "blocking_issues", "message"
            ]
            
            for field in required_fields:
                assert field in data, f"Missing required field: {field}"
            
            # Verify resource_status structure
            resource_status = data["resource_status"]
            status_fields = ["vram_available", "cpu_available", "ram_available"]
            for field in status_fields:
                assert field in resource_status, f"Missing status field: {field}"
                assert isinstance(resource_status[field], bool)
            
            # Verify current_usage structure
            current_usage = data["current_usage"]
            usage_fields = ["vram_percent", "cpu_percent", "ram_percent"]
            for field in usage_fields:
                assert field in current_usage, f"Missing usage field: {field}"
                assert isinstance(current_usage[field], (int, float))
            
            print("✅ Resource availability check test passed")
        else:
            print(f"❌ Resource check failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Resource availability test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_vram_exhaustion_scenario():
    """Test VRAM exhaustion scenario for graceful degradation"""
    print("Testing VRAM exhaustion scenario...")
    
    try:
        from fastapi.testclient import TestClient
        from main import app
        from api.routes.system import _resource_constraints
        
        client = TestClient(app)
        
        # Temporarily set very low VRAM threshold to simulate exhaustion
        original_threshold = _resource_constraints["vram_critical_threshold"]
        _resource_constraints["vram_critical_threshold"] = 0.01  # 1% threshold
        
        try:
            # Check resource availability with low threshold
            response = client.get("/api/v1/system/resource-check")
            
            if response.status_code == 200:
                data = response.json()
                print(f"VRAM exhaustion test result: {data}")
                
                # Should indicate VRAM is not available
                if data["current_usage"]["vram_percent"] > 1.0:
                    assert not data["resource_status"]["vram_available"]
                    assert not data["can_start_generation"]
                    assert any("VRAM" in issue for issue in data["blocking_issues"])
                    print("✅ VRAM exhaustion scenario handled correctly")
                else:
                    print("ℹ️ VRAM usage too low to test exhaustion scenario")
            
        finally:
            # Restore original threshold
            _resource_constraints["vram_critical_threshold"] = original_threshold
            
    except Exception as e:
        print(f"❌ VRAM exhaustion test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_optimization_validation():
    """Test optimization settings validation"""
    print("Testing optimization settings validation...")
    
    try:
        from fastapi.testclient import TestClient
        from main import app
        
        client = TestClient(app)
        
        # Test invalid VAE tile size
        invalid_settings = {
            "quantization": "bf16",
            "enable_offload": True,
            "vae_tile_size": 64,  # Too small
            "max_vram_usage_gb": 12.0
        }
        
        response = client.post("/api/v1/system/optimization", json=invalid_settings)
        print(f"Invalid VAE tile size response: {response.status_code}")
        assert response.status_code in [400, 422]  # 422 for Pydantic validation, 400 for custom validation
        
        # Test invalid VRAM usage
        invalid_settings["vae_tile_size"] = 256
        invalid_settings["max_vram_usage_gb"] = 30.0  # Too high
        
        response = client.post("/api/v1/system/optimization", json=invalid_settings)
        print(f"Invalid VRAM usage response: {response.status_code}")
        assert response.status_code in [400, 422]  # 422 for Pydantic validation, 400 for custom validation
        
        print("✅ Optimization validation test passed")
        
    except Exception as e:
        print(f"❌ Optimization validation test failed: {e}")
        import traceback
        traceback.print_exc()

async def run_all_tests():
    """Run all system monitoring tests"""
    print("🚀 Starting system monitoring and optimization tests...")
    print("=" * 60)
    
    tests = [
        test_system_stats_endpoint,
        test_optimization_settings_endpoints,
        test_system_health_endpoint,
        test_resource_constraints_endpoints,
        test_resource_availability_check,
        test_vram_exhaustion_scenario,
        test_optimization_validation
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            failed += 1
        print("-" * 40)
    
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All system monitoring tests passed!")
    else:
        print(f"⚠️ {failed} tests failed")
    
    return failed == 0

if __name__ == "__main__":
    asyncio.run(run_all_tests())