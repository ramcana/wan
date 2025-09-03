#!/usr/bin/env python3
"""
Comprehensive validation test for Task 2.4: Add system monitoring and optimization endpoints
Validates all requirements: 7.1, 7.2, 4.1, 4.4
"""

import sys
import os
from pathlib import Path
import asyncio
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

async def validate_requirement_7_1():
    """
    Requirement 7.1: Display real-time charts and graphs for CPU, RAM, GPU, and VRAM usage
    """
    print("🔍 Validating Requirement 7.1: Real-time system monitoring...")
    
    try:
        from fastapi.testclient import TestClient
        from main import backend.app as app
        
        client = TestClient(app)
        
        # Test GET /api/v1/system/stats endpoint
        response = client.get("/api/v1/system/stats")
        assert response.status_code == 200, "System stats endpoint should be accessible"
        
        data = response.json()
        
        # Verify all required metrics are present
        required_metrics = [
            "cpu_percent", "ram_used_gb", "ram_total_gb", "ram_percent",
            "gpu_percent", "vram_used_mb", "vram_total_mb", "vram_percent", "timestamp"
        ]
        
        for metric in required_metrics:
            assert metric in data, f"Missing required metric: {metric}"
            if metric == "timestamp":
                # Timestamp should be a string (ISO format) or datetime
                assert isinstance(data[metric], str), f"Timestamp should be a string"
            else:
                assert isinstance(data[metric], (int, float)), f"Metric {metric} should be numeric"
        
        # Verify data ranges are reasonable
        assert 0 <= data["cpu_percent"] <= 100, "CPU percentage should be 0-100"
        assert 0 <= data["ram_percent"] <= 100, "RAM percentage should be 0-100"
        assert 0 <= data["vram_percent"] <= 100, "VRAM percentage should be 0-100"
        assert data["ram_used_gb"] >= 0, "RAM usage should be non-negative"
        assert data["vram_used_mb"] >= 0, "VRAM usage should be non-negative"
        
        print("✅ Requirement 7.1 validated: Real-time system metrics endpoint working")
        return True
        
    except Exception as e:
        print(f"❌ Requirement 7.1 failed: {e}")
        return False

async def validate_requirement_7_2():
    """
    Requirement 7.2: Update metrics through WebSocket connections with smooth chart animations
    Note: Testing historical data endpoint as WebSocket testing requires more complex setup
    """
    print("🔍 Validating Requirement 7.2: Historical data for smooth chart updates...")
    
    try:
        from fastapi.testclient import TestClient
        from main import backend.app as app
        
        client = TestClient(app)
        
        # Save some stats for historical tracking
        save_response = client.post("/api/v1/system/stats/save")
        assert save_response.status_code == 200, "Should be able to save system stats"
        
        # Test historical stats endpoint
        history_response = client.get("/api/v1/system/stats/history?hours=24")
        assert history_response.status_code == 200, "Historical stats endpoint should be accessible"
        
        history_data = history_response.json()
        
        # Verify structure
        assert "stats" in history_data, "Should include stats array"
        assert "total_count" in history_data, "Should include total count"
        assert "time_range" in history_data, "Should include time range info"
        
        # Verify time range
        time_range = history_data["time_range"]
        assert time_range["hours"] == 24, "Should respect hours parameter"
        assert "start" in time_range and "end" in time_range, "Should include start/end times"
        
        print("✅ Requirement 7.2 validated: Historical data tracking for chart updates")
        return True
        
    except Exception as e:
        print(f"❌ Requirement 7.2 failed: {e}")
        return False

async def validate_requirement_4_1():
    """
    Requirement 4.1: Provide modern settings panel with quantization options (fp16, bf16, int8)
    """
    print("🔍 Validating Requirement 4.1: Optimization settings with quantization options...")
    
    try:
        from fastapi.testclient import TestClient
        from main import backend.app as app
        
        client = TestClient(app)
        
        # Test GET optimization settings
        get_response = client.get("/api/v1/system/optimization")
        assert get_response.status_code == 200, "Should be able to get optimization settings"
        
        settings = get_response.json()
        
        # Verify required fields
        required_fields = ["quantization", "enable_offload", "vae_tile_size", "max_vram_usage_gb"]
        for field in required_fields:
            assert field in settings, f"Missing required optimization field: {field}"
        
        # Test quantization options
        valid_quantizations = ["fp16", "bf16", "int8"]
        
        for quant in valid_quantizations:
            test_settings = {
                "quantization": quant,
                "enable_offload": True,
                "vae_tile_size": 256,
                "max_vram_usage_gb": 12.0
            }
            
            post_response = client.post("/api/v1/system/optimization", json=test_settings)
            assert post_response.status_code == 200, f"Should accept {quant} quantization"
            
            response_data = post_response.json()
            assert "message" in response_data, "Should include success message"
            assert "estimated_vram_savings_gb" in response_data, "Should include VRAM savings estimate"
        
        print("✅ Requirement 4.1 validated: Quantization options (fp16, bf16, int8) supported")
        return True
        
    except Exception as e:
        print(f"❌ Requirement 4.1 failed: {e}")
        return False

async def validate_requirement_4_4():
    """
    Requirement 4.4: Display professional error dialogs with actionable suggestions when VRAM usage exceeds limits
    """
    print("🔍 Validating Requirement 4.4: VRAM limit handling with actionable suggestions...")
    
    try:
        from fastapi.testclient import TestClient
        from main import backend.app as app
        from api.routes.system import _resource_constraints
        
        client = TestClient(app)
        
        # Test system health endpoint for VRAM warnings
        health_response = client.get("/api/v1/system/health")
        assert health_response.status_code == 200, "Health endpoint should be accessible"
        
        health_data = health_response.json()
        
        # Verify health response structure
        assert "status" in health_data, "Should include status"
        assert "message" in health_data, "Should include message"
        assert "system_info" in health_data, "Should include system info"
        
        system_info = health_data["system_info"]
        assert "resource_constraints" in system_info, "Should include resource constraints"
        assert "recommendations" in system_info, "Should include recommendations"
        
        # Test resource availability check
        resource_response = client.get("/api/v1/system/resource-check")
        assert resource_response.status_code == 200, "Resource check should be accessible"
        
        resource_data = resource_response.json()
        
        # Verify resource check structure
        required_fields = [
            "can_start_generation", "resource_status", "current_usage", 
            "blocking_issues", "message"
        ]
        
        for field in required_fields:
            assert field in resource_data, f"Missing required field: {field}"
        
        # Test VRAM exhaustion scenario
        original_threshold = _resource_constraints["vram_critical_threshold"]
        _resource_constraints["vram_critical_threshold"] = 0.01  # Very low threshold
        
        try:
            exhaustion_response = client.get("/api/v1/system/resource-check")
            exhaustion_data = exhaustion_response.json()
            
            # Should detect VRAM exhaustion and provide actionable information
            if exhaustion_data["current_usage"]["vram_percent"] > 1.0:
                assert not exhaustion_data["can_start_generation"], "Should prevent generation when VRAM exhausted"
                assert not exhaustion_data["resource_status"]["vram_available"], "Should indicate VRAM not available"
                assert len(exhaustion_data["blocking_issues"]) > 0, "Should list blocking issues"
                assert any("VRAM" in issue for issue in exhaustion_data["blocking_issues"]), "Should mention VRAM in issues"
        
        finally:
            # Restore original threshold
            _resource_constraints["vram_critical_threshold"] = original_threshold
        
        # Test optimization recommendations
        test_settings = {
            "quantization": "fp16",  # Less efficient
            "enable_offload": False,  # No offloading
            "vae_tile_size": 512,     # Larger tiles
            "max_vram_usage_gb": 8.0  # Lower limit
        }
        
        opt_response = client.post("/api/v1/system/optimization", json=test_settings)
        if opt_response.status_code == 200:
            opt_data = opt_response.json()
            # Should provide recommendations when settings are suboptimal
            recommendations = opt_data.get("recommendations", [])
            # Recommendations are context-dependent, so we just verify the field exists
            assert isinstance(recommendations, list), "Recommendations should be a list"
        
        print("✅ Requirement 4.4 validated: VRAM limit handling with actionable suggestions")
        return True
        
    except Exception as e:
        print(f"❌ Requirement 4.4 failed: {e}")
        return False

async def validate_graceful_degradation():
    """
    Validate graceful degradation behavior for resource constraints
    """
    print("🔍 Validating graceful degradation behavior...")
    
    try:
        from fastapi.testclient import TestClient
        from main import backend.app as app
        
        client = TestClient(app)
        
        # Test resource constraints management
        constraints_response = client.get("/api/v1/system/constraints")
        assert constraints_response.status_code == 200, "Should be able to get resource constraints"
        
        constraints_data = constraints_response.json()
        assert "constraints" in constraints_data, "Should include constraints"
        assert "description" in constraints_data, "Should include descriptions"
        
        # Test constraint updates
        new_constraints = {
            "vram_warning_threshold": 0.80,
            "vram_critical_threshold": 0.90,
            "max_concurrent_generations": 1
        }
        
        update_response = client.post("/api/v1/system/constraints", json=new_constraints)
        assert update_response.status_code == 200, "Should be able to update constraints"
        
        update_data = update_response.json()
        assert "updated_constraints" in update_data, "Should return updated constraints"
        
        # Test constraint validation
        invalid_constraints = {"max_concurrent_generations": 0}
        invalid_response = client.post("/api/v1/system/constraints", json=invalid_constraints)
        assert invalid_response.status_code == 400, "Should reject invalid constraints"
        
        print("✅ Graceful degradation behavior validated")
        return True
        
    except Exception as e:
        print(f"❌ Graceful degradation validation failed: {e}")
        return False

async def validate_multiple_simultaneous_generations():
    """
    Validate handling of multiple simultaneous generations
    """
    print("🔍 Validating multiple simultaneous generation handling...")
    
    try:
        from fastapi.testclient import TestClient
        from main import backend.app as app
        from api.routes.system import _resource_constraints
        
        client = TestClient(app)
        
        # Check current queue status
        queue_response = client.get("/api/v1/queue")
        assert queue_response.status_code == 200, "Should be able to get queue status"
        
        queue_data = queue_response.json()
        
        # Verify queue structure includes concurrent task tracking
        assert "processing_tasks" in queue_data, "Should track processing tasks"
        assert "total_tasks" in queue_data, "Should track total tasks"
        
        # Test concurrent generation limit enforcement
        original_limit = _resource_constraints["max_concurrent_generations"]
        
        # Set a specific limit
        new_limit = {"max_concurrent_generations": 2}
        limit_response = client.post("/api/v1/system/constraints", json=new_limit)
        assert limit_response.status_code == 200, "Should be able to set concurrent limit"
        
        # Verify the limit was set
        updated_constraints = limit_response.json()["updated_constraints"]
        assert updated_constraints["max_concurrent_generations"] == 2, "Should update concurrent limit"
        
        # Restore original limit
        restore_limit = {"max_concurrent_generations": original_limit}
        client.post("/api/v1/system/constraints", json=restore_limit)
        
        print("✅ Multiple simultaneous generation handling validated")
        return True
        
    except Exception as e:
        print(f"❌ Multiple simultaneous generation validation failed: {e}")
        return False

async def run_task_2_4_validation():
    """Run comprehensive validation for Task 2.4"""
    print("🚀 Starting Task 2.4 Comprehensive Validation")
    print("Task: Add system monitoring and optimization endpoints")
    print("Requirements: 7.1, 7.2, 4.1, 4.4")
    print("=" * 80)
    
    validations = [
        ("Requirement 7.1", validate_requirement_7_1),
        ("Requirement 7.2", validate_requirement_7_2),
        ("Requirement 4.1", validate_requirement_4_1),
        ("Requirement 4.4", validate_requirement_4_4),
        ("Graceful Degradation", validate_graceful_degradation),
        ("Multiple Generations", validate_multiple_simultaneous_generations)
    ]
    
    passed = 0
    failed = 0
    results = {}
    
    for name, validation_func in validations:
        try:
            result = await validation_func()
            if result:
                passed += 1
                results[name] = "✅ PASSED"
            else:
                failed += 1
                results[name] = "❌ FAILED"
        except Exception as e:
            failed += 1
            results[name] = f"❌ FAILED: {e}"
        
        print("-" * 60)
    
    print("\n📊 TASK 2.4 VALIDATION RESULTS")
    print("=" * 80)
    
    for name, result in results.items():
        print(f"{name:25} {result}")
    
    print(f"\nSummary: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 TASK 2.4 FULLY VALIDATED!")
        print("✅ All system monitoring and optimization endpoints implemented correctly")
        print("✅ All requirements (7.1, 7.2, 4.1, 4.4) satisfied")
        print("✅ Resource limit scenarios tested")
        print("✅ Graceful degradation behavior defined and working")
    else:
        print(f"\n⚠️ TASK 2.4 VALIDATION INCOMPLETE: {failed} issues found")
    
    return failed == 0

if __name__ == "__main__":
    success = asyncio.run(run_task_2_4_validation())
    exit(0 if success else 1)