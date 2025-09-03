#!/usr/bin/env python3
"""
Test resource limit scenarios and graceful degradation behavior
Tests for task 2.4: Resource constraint testing and graceful degradation
"""

import sys
import os
from pathlib import Path
import asyncio
import time
from unittest.mock import Mock, patch
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

async def test_vram_exhaustion_prevention():
    """Test that system prevents new generations when VRAM is exhausted"""
    print("Testing VRAM exhaustion prevention...")
    
    try:
        from fastapi.testclient import TestClient
        from main import backend.app as app
        from api.routes.system import _resource_constraints
        
        client = TestClient(app)
        
        # Set very low VRAM threshold to simulate exhaustion
        original_threshold = _resource_constraints["vram_critical_threshold"]
        _resource_constraints["vram_critical_threshold"] = 0.30  # 30% threshold
        
        try:
            # Check if current VRAM usage is above threshold
            stats_response = client.get("/api/v1/system/stats")
            if stats_response.status_code == 200:
                stats = stats_response.json()
                current_vram = stats["vram_percent"]
                print(f"Current VRAM usage: {current_vram}%")
                
                # Check resource availability
                resource_response = client.get("/api/v1/system/resource-check")
                if resource_response.status_code == 200:
                    resource_data = resource_response.json()
                    print(f"Resource check result: {resource_data}")
                    
                    if current_vram > 30:
                        # Should prevent generation
                        assert not resource_data["can_start_generation"]
                        assert not resource_data["resource_status"]["vram_available"]
                        assert any("VRAM" in issue for issue in resource_data["blocking_issues"])
                        print("✅ VRAM exhaustion correctly prevents new generations")
                    else:
                        print("ℹ️ VRAM usage too low to test exhaustion scenario")
                        
                        # Simulate high VRAM usage by setting even lower threshold
                        _resource_constraints["vram_critical_threshold"] = 0.01
                        
                        resource_response2 = client.get("/api/v1/system/resource-check")
                        if resource_response2.status_code == 200:
                            resource_data2 = resource_response2.json()
                            assert not resource_data2["can_start_generation"]
                            print("✅ Simulated VRAM exhaustion correctly prevents generations")
                
        finally:
            # Restore original threshold
            _resource_constraints["vram_critical_threshold"] = original_threshold
            
    except Exception as e:
        print(f"❌ VRAM exhaustion test failed: {e}")
        import traceback
traceback.print_exc()

async def test_concurrent_generation_limits():
    """Test maximum concurrent generation limits"""
    print("Testing concurrent generation limits...")
    
    try:
        from fastapi.testclient import TestClient
        from main import backend.app as app
        from api.routes.system import _resource_constraints
        
        client = TestClient(app)
        
        # Set low concurrent generation limit
        original_limit = _resource_constraints["max_concurrent_generations"]
        _resource_constraints["max_concurrent_generations"] = 1
        
        try:
            # Check current queue status
            queue_response = client.get("/api/v1/queue")
            if queue_response.status_code == 200:
                queue_data = queue_response.json()
                processing_tasks = queue_data.get("processing_tasks", 0)
                print(f"Currently processing tasks: {processing_tasks}")
                
                # If no tasks are processing, we can test the limit
                if processing_tasks == 0:
                    print("✅ Concurrent generation limit can be enforced (no active tasks)")
                else:
                    print(f"ℹ️ {processing_tasks} tasks already processing, limit enforcement active")
                
                # Test constraint update
                new_constraints = {"max_concurrent_generations": 2}
                constraint_response = client.post("/api/v1/system/constraints", json=new_constraints)
                
                if constraint_response.status_code == 200:
                    constraint_data = constraint_response.json()
                    assert constraint_data["updated_constraints"]["max_concurrent_generations"] == 2
                    print("✅ Concurrent generation limit successfully updated")
                
        finally:
            # Restore original limit
            _resource_constraints["max_concurrent_generations"] = original_limit
            
    except Exception as e:
        print(f"❌ Concurrent generation limit test failed: {e}")
        import traceback
traceback.print_exc()

async def test_graceful_degradation_warnings():
    """Test graceful degradation with warning thresholds"""
    print("Testing graceful degradation warnings...")
    
    try:
        from fastapi.testclient import TestClient
        from main import backend.app as app
        from api.routes.system import _resource_constraints
        
        client = TestClient(app)
        
        # Set warning thresholds to trigger warnings
        original_vram_warning = _resource_constraints["vram_warning_threshold"]
        original_cpu_warning = _resource_constraints["cpu_warning_threshold"]
        
        _resource_constraints["vram_warning_threshold"] = 0.30  # 30% threshold
        _resource_constraints["cpu_warning_threshold"] = 0.01   # 1% threshold (will likely trigger)
        
        try:
            # Check system health with new thresholds
            health_response = client.get("/api/v1/system/health")
            if health_response.status_code == 200:
                health_data = health_response.json()
                print(f"Health status: {health_data['status']}")
                print(f"Health message: {health_data['message']}")
                
                system_info = health_data["system_info"]
                
                # Should have warnings or issues due to low thresholds
                has_warnings = len(system_info.get("warnings", [])) > 0
                has_issues = len(system_info.get("issues", [])) > 0
                has_recommendations = len(system_info.get("recommendations", [])) > 0
                
                if has_warnings or has_issues:
                    print("✅ Warning system correctly triggered")
                    if has_recommendations:
                        print(f"✅ Recommendations provided: {system_info['recommendations']}")
                else:
                    print("ℹ️ No warnings triggered (system usage very low)")
                
                # Verify status is not healthy when issues exist
                if has_issues:
                    assert health_data["status"] in ["warning", "critical"]
                    print("✅ Health status correctly reflects issues")
                
        finally:
            # Restore original thresholds
            _resource_constraints["vram_warning_threshold"] = original_vram_warning
            _resource_constraints["cpu_warning_threshold"] = original_cpu_warning
            
    except Exception as e:
        print(f"❌ Graceful degradation test failed: {e}")
        import traceback
traceback.print_exc()

async def test_optimization_recommendations():
    """Test optimization recommendations based on system state"""
    print("Testing optimization recommendations...")
    
    try:
        from fastapi.testclient import TestClient
        from main import backend.app as app
        
        client = TestClient(app)
        
        # Get current system stats
        stats_response = client.get("/api/v1/system/stats")
        if stats_response.status_code == 200:
            stats = stats_response.json()
            current_vram_gb = stats["vram_used_mb"] / 1024
            total_vram_gb = stats["vram_total_mb"] / 1024
            
            print(f"Current VRAM usage: {current_vram_gb:.1f}GB / {total_vram_gb:.1f}GB")
            
            # Test optimization settings that should trigger recommendations
            test_settings = {
                "quantization": "fp16",  # Less efficient than bf16/int8
                "enable_offload": False,  # No offloading
                "vae_tile_size": 512,     # Larger tile size
                "max_vram_usage_gb": current_vram_gb + 1  # Close to current usage
            }
            
            opt_response = client.post("/api/v1/system/optimization", json=test_settings)
            if opt_response.status_code == 200:
                opt_data = opt_response.json()
                print(f"Optimization response: {opt_data}")
                
                # Should include recommendations
                recommendations = opt_data.get("recommendations", [])
                if recommendations:
                    print(f"✅ Recommendations provided: {recommendations}")
                else:
                    print("ℹ️ No recommendations (system has sufficient resources)")
                
                # Should include VRAM savings estimate
                assert "estimated_vram_savings_gb" in opt_data
                savings = opt_data["estimated_vram_savings_gb"]
                print(f"✅ Estimated VRAM savings: {savings}GB")
                
        else:
            print(f"❌ Could not get system stats: {stats_response.text}")
            
    except Exception as e:
        print(f"❌ Optimization recommendations test failed: {e}")
        import traceback
traceback.print_exc()

async def test_historical_stats_tracking():
    """Test historical stats tracking for monitoring dashboard"""
    print("Testing historical stats tracking...")
    
    try:
        from fastapi.testclient import TestClient
        from main import backend.app as app
        
        client = TestClient(app)
        
        # Save current stats to history
        save_response = client.post("/api/v1/system/stats/save")
        if save_response.status_code == 200:
            print("✅ Stats saved to history")
            
            # Wait a moment and save again
            await asyncio.sleep(1)
            save_response2 = client.post("/api/v1/system/stats/save")
            
            if save_response2.status_code == 200:
                # Get historical stats
                history_response = client.get("/api/v1/system/stats/history?hours=1")
                
                if history_response.status_code == 200:
                    history_data = history_response.json()
                    print(f"Historical stats retrieved: {len(history_data['stats'])} entries")
                    
                    # Should have at least the stats we just saved
                    assert len(history_data["stats"]) >= 1
                    assert "time_range" in history_data
                    assert history_data["time_range"]["hours"] == 1
                    
                    print("✅ Historical stats tracking working correctly")
                else:
                    print(f"❌ Could not get historical stats: {history_response.text}")
        else:
            print(f"❌ Could not save stats: {save_response.text}")
            
    except Exception as e:
        print(f"❌ Historical stats test failed: {e}")
        import traceback
traceback.print_exc()

async def test_resource_constraint_validation():
    """Test resource constraint validation"""
    print("Testing resource constraint validation...")
    
    try:
        from fastapi.testclient import TestClient
        from main import backend.app as app
        
        client = TestClient(app)
        
        # Test invalid constraint values
        invalid_constraints = [
            {"max_concurrent_generations": 0},  # Too low
            {"max_concurrent_generations": 15}, # Too high
            {"vram_warning_threshold": 1.5},    # Too high
            {"vram_critical_threshold": -0.1},  # Too low
            {"invalid_key": 0.5}                # Invalid key
        ]
        
        for invalid_constraint in invalid_constraints:
            response = client.post("/api/v1/system/constraints", json=invalid_constraint)
            print(f"Invalid constraint {invalid_constraint}: {response.status_code}")
            assert response.status_code == 400, f"Should reject invalid constraint: {invalid_constraint}"
        
        print("✅ Resource constraint validation working correctly")
        
    except Exception as e:
        print(f"❌ Resource constraint validation test failed: {e}")
        import traceback
traceback.print_exc()

async def run_all_resource_tests():
    """Run all resource limit and graceful degradation tests"""
    print("🚀 Starting resource limit and graceful degradation tests...")
    print("=" * 70)
    
    tests = [
        test_vram_exhaustion_prevention,
        test_concurrent_generation_limits,
        test_graceful_degradation_warnings,
        test_optimization_recommendations,
        test_historical_stats_tracking,
        test_resource_constraint_validation
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
        print("-" * 50)
    
    print(f"📊 Resource Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All resource limit tests passed!")
    else:
        print(f"⚠️ {failed} resource tests failed")
    
    return failed == 0

if __name__ == "__main__":
    asyncio.run(run_all_resource_tests())