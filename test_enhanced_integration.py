#!/usr/bin/env python3
"""
Simple test script to verify enhanced generation service integration
"""

import asyncio
import sys
from pathlib import Path

# Add the backend directory to the path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

async def test_enhanced_integration():
    """Test the enhanced generation service integration"""
    try:
        print("Testing enhanced generation service integration...")
        
        # Test imports
        print("1. Testing imports...")
        from backend.services.generation_service import GenerationService
        from backend.core.model_availability_manager import ModelAvailabilityManager, ModelAvailabilityStatus
        from backend.core.enhanced_model_downloader import EnhancedModelDownloader
        from backend.core.intelligent_fallback_manager import IntelligentFallbackManager
        from backend.core.model_health_monitor import ModelHealthMonitor
        from backend.core.model_usage_analytics import ModelUsageAnalytics
        print("✓ All imports successful")
        
        # Test service creation
        print("2. Testing service creation...")
        service = GenerationService()
        print("✓ GenerationService created successfully")
        
        # Test enhanced components initialization
        print("3. Testing enhanced components...")
        
        # Create mock components for testing
        service.model_health_monitor = ModelHealthMonitor()
        service.enhanced_model_downloader = EnhancedModelDownloader()
        service.model_usage_analytics = ModelUsageAnalytics()
        service.model_availability_manager = ModelAvailabilityManager()
        service.intelligent_fallback_manager = IntelligentFallbackManager()
        
        print("✓ Enhanced components created successfully")
        
        # Test method existence
        print("4. Testing method existence...")
        assert hasattr(service, '_run_enhanced_generation'), "Missing _run_enhanced_generation method"
        assert hasattr(service, '_run_legacy_generation'), "Missing _run_legacy_generation method"
        assert hasattr(service, '_run_real_generation_with_monitoring'), "Missing _run_real_generation_with_monitoring method"
        assert hasattr(service, '_handle_downloading_model'), "Missing _handle_downloading_model method"
        assert hasattr(service, '_handle_missing_or_corrupted_model'), "Missing _handle_missing_or_corrupted_model method"
        assert hasattr(service, '_handle_model_unavailable'), "Missing _handle_model_unavailable method"
        print("✓ All required methods exist")
        
        print("\n🎉 Enhanced generation service integration test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Enhanced generation service integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_enhanced_integration())
    sys.exit(0 if success else 1)