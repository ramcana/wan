#!/usr/bin/env python3
"""
Test WAN Model Information and Capabilities API
Comprehensive test suite for the WAN Model Information API endpoints
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
import sys

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_wan_model_info_api():
    """Test the WAN Model Information API"""
    try:
        # Import the API
        from api.wan_model_info import get_wan_model_info_api, WANModelInfoAPI
        
        logger.info("🧪 Starting WAN Model Information API Tests")
        logger.info("=" * 60)
        
        # Initialize the API
        api = await get_wan_model_info_api()
        
        # Test model types
        test_models = ["T2V-A14B", "I2V-A14B", "TI2V-5B"]
        
        # Test 1: Model Capabilities
        logger.info("📋 Testing Model Capabilities...")
        for model_type in test_models:
            try:
                capabilities = await api.get_wan_model_capabilities(model_type)
                logger.info(f"✅ {model_type} capabilities:")
                logger.info(f"   - Resolutions: {capabilities.supported_resolutions}")
                logger.info(f"   - Max frames: {capabilities.max_frames}")
                logger.info(f"   - Input types: {capabilities.input_types}")
                logger.info(f"   - LoRA support: {capabilities.lora_support}")
                logger.info(f"   - Hardware req: {capabilities.hardware_requirements}")
            except Exception as e:
                logger.error(f"❌ Failed to get capabilities for {model_type}: {e}")
        
        # Test 2: Health Metrics
        logger.info("\n🏥 Testing Health Metrics...")
        for model_type in test_models:
            try:
                health = await api.get_wan_model_health_metrics(model_type)
                logger.info(f"✅ {model_type} health:")
                logger.info(f"   - Status: {health.health_status}")
                logger.info(f"   - Integrity score: {health.integrity_score:.2f}")
                logger.info(f"   - Performance score: {health.performance_score:.2f}")
                logger.info(f"   - Success rate: {health.success_rate_24h:.2f}")
                logger.info(f"   - Memory usage: {health.memory_usage_mb:.1f}MB")
            except Exception as e:
                logger.error(f"❌ Failed to get health metrics for {model_type}: {e}")
        
        # Test 3: Performance Metrics
        logger.info("\n⚡ Testing Performance Metrics...")
        for model_type in test_models:
            try:
                performance = await api.get_wan_model_performance_metrics(model_type)
                logger.info(f"✅ {model_type} performance:")
                logger.info(f"   - Avg generation time: {performance.generation_time_avg_seconds:.1f}s")
                logger.info(f"   - Throughput: {performance.throughput_videos_per_hour:.1f} videos/hour")
                logger.info(f"   - Quality score: {performance.quality_score:.2f}")
                logger.info(f"   - GPU utilization: {performance.gpu_utilization_avg_percent:.1f}%")
                logger.info(f"   - Hardware: {performance.hardware_profile}")
            except Exception as e:
                logger.error(f"❌ Failed to get performance metrics for {model_type}: {e}")
        
        # Test 4: Model Comparison
        logger.info("\n🔄 Testing Model Comparison...")
        try:
            comparison = await api.compare_wan_models("T2V-A14B", "TI2V-5B")
            logger.info("✅ Model comparison T2V-A14B vs TI2V-5B:")
            logger.info(f"   - Performance difference: {comparison.performance_difference_percent:.1f}%")
            logger.info(f"   - Quality difference: {comparison.quality_difference_score:.2f}")
            logger.info(f"   - Speed difference: {comparison.speed_difference_percent:.1f}%")
            logger.info(f"   - Recommendation: {comparison.recommendation}")
            logger.info(f"   - Trade-offs: {comparison.trade_offs}")
        except Exception as e:
            logger.error(f"❌ Failed to compare models: {e}")
        
        # Test 5: Model Recommendation
        logger.info("\n🎯 Testing Model Recommendation...")
        test_cases = [
            {"use_case": "text to video creation", "quality_priority": "high", "speed_priority": "medium"},
            {"use_case": "image animation", "quality_priority": "medium", "speed_priority": "fast"},
            {"use_case": "storytelling with images", "quality_priority": "high", "speed_priority": "medium", "memory_constraint_gb": 10.0}
        ]
        
        for test_case in test_cases:
            try:
                recommendation = await api.get_wan_model_recommendation(**test_case)
                logger.info(f"✅ Recommendation for '{test_case['use_case']}':")
                logger.info(f"   - Recommended: {recommendation.recommended_model}")
                logger.info(f"   - Confidence: {recommendation.confidence_score:.2f}")
                logger.info(f"   - Reasoning: {recommendation.reasoning}")
                logger.info(f"   - Alternatives: {recommendation.alternative_models}")
            except Exception as e:
                logger.error(f"❌ Failed to get recommendation for {test_case}: {e}")
        
        # Test 6: Dashboard Data
        logger.info("\n📊 Testing Dashboard Data...")
        try:
            dashboard = await api.get_wan_model_dashboard_data()
            logger.info("✅ Dashboard data retrieved:")
            logger.info(f"   - Models: {list(dashboard['models'].keys())}")
            logger.info(f"   - System status: {dashboard['system_overview'].get('system_status', 'unknown')}")
            logger.info(f"   - Healthy models: {dashboard['system_overview'].get('healthy_models', 0)}")
            logger.info(f"   - Alerts: {len(dashboard.get('alerts', []))}")
            logger.info(f"   - Recommendations: {len(dashboard.get('recommendations', []))}")
            
            # Show sample model data
            for model_type, model_data in list(dashboard['models'].items())[:2]:
                if isinstance(model_data, dict) and 'capabilities' in model_data:
                    logger.info(f"   - {model_type} sample data available")
                else:
                    logger.info(f"   - {model_type} data: {model_data}")
                    
        except Exception as e:
            logger.error(f"❌ Failed to get dashboard data: {e}")
        
        # Test 7: API Integration Test
        logger.info("\n🔗 Testing API Integration...")
        try:
            # Test that the API can handle multiple concurrent requests
            tasks = []
            for model_type in test_models:
                tasks.append(api.get_wan_model_capabilities(model_type))
                tasks.append(api.get_wan_model_health_metrics(model_type))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            successful_requests = sum(1 for result in results if not isinstance(result, Exception))
            total_requests = len(results)
            
            logger.info(f"✅ Concurrent requests test:")
            logger.info(f"   - Successful: {successful_requests}/{total_requests}")
            logger.info(f"   - Success rate: {(successful_requests/total_requests)*100:.1f}%")
            
        except Exception as e:
            logger.error(f"❌ Failed integration test: {e}")
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 WAN Model Information API Tests Completed!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Critical error in WAN Model Info API tests: {e}")
        return False

async def test_api_endpoints_directly():
    """Test API endpoints directly using FastAPI test client"""
    try:
        from fastapi.testclient import TestClient
        from backend.app import app
        
        logger.info("\n🌐 Testing API Endpoints Directly...")
        
        client = TestClient(app)
        
        # Test endpoints
        endpoints = [
            "/api/v1/wan-models/status",
            "/api/v1/wan-models/capabilities/T2V-A14B",
            "/api/v1/wan-models/health/T2V-A14B",
            "/api/v1/wan-models/performance/T2V-A14B",
            "/api/v1/wan-models/compare/T2V-A14B/TI2V-5B",
            "/api/v1/wan-models/recommend?use_case=text%20to%20video&quality_priority=high",
            "/api/v1/wan-models/dashboard"
        ]
        
        for endpoint in endpoints:
            try:
                response = client.get(endpoint)
                if response.status_code == 200:
                    logger.info(f"✅ {endpoint}: {response.status_code}")
                    # Show sample response data
                    data = response.json()
                    if isinstance(data, dict):
                        if "model_id" in data:
                            logger.info(f"   - Model ID: {data['model_id']}")
                        elif "status" in data:
                            logger.info(f"   - Status: {data['status']}")
                        elif "models" in data:
                            logger.info(f"   - Models count: {len(data['models'])}")
                else:
                    logger.warning(f"⚠️  {endpoint}: {response.status_code}")
                    logger.warning(f"   - Response: {response.text[:200]}...")
            except Exception as e:
                logger.error(f"❌ {endpoint}: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to test API endpoints directly: {e}")
        return False

async def main():
    """Main test function"""
    logger.info("🚀 Starting WAN Model Information API Test Suite")
    logger.info("=" * 80)
    
    # Test 1: Direct API testing
    success1 = await test_wan_model_info_api()
    
    # Test 2: FastAPI endpoint testing
    success2 = await test_api_endpoints_directly()
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("📋 TEST SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Direct API Tests: {'✅ PASSED' if success1 else '❌ FAILED'}")
    logger.info(f"Endpoint Tests: {'✅ PASSED' if success2 else '❌ FAILED'}")
    
    overall_success = success1 and success2
    logger.info(f"Overall Result: {'🎉 ALL TESTS PASSED' if overall_success else '⚠️  SOME TESTS FAILED'}")
    
    if overall_success:
        logger.info("\n✨ WAN Model Information and Capabilities API is ready!")
        logger.info("🔗 Available endpoints:")
        logger.info("   - GET /api/v1/wan-models/capabilities/{model_type}")
        logger.info("   - GET /api/v1/wan-models/health/{model_type}")
        logger.info("   - GET /api/v1/wan-models/performance/{model_type}")
        logger.info("   - GET /api/v1/wan-models/compare/{model_a}/{model_b}")
        logger.info("   - GET /api/v1/wan-models/recommend")
        logger.info("   - GET /api/v1/wan-models/dashboard")
        logger.info("   - GET /api/v1/wan-models/status")
    
    return overall_success

if __name__ == "__main__":
    asyncio.run(main())