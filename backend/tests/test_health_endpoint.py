#!/usr/bin/env python3
"""
Test the health endpoint directly
"""

import sys
import os
from pathlib import Path
import asyncio

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

async def test_health_endpoint():
    """Test the health endpoint logic directly"""
    try:
        from backend.core.system_integration import get_system_integration
        
        print("Testing system integration...")
        integration = await get_system_integration()
        
        print("Testing system info...")
        system_info = integration.get_system_info()
        print(f"System info: {system_info}")
        
        print("Testing GPU validation...")
        gpu_valid, gpu_message = await integration.validate_gpu_access()
        print(f"GPU validation: {gpu_valid} - {gpu_message}")
        
        print("Testing model loading validation...")
        model_valid, model_message = await integration.validate_model_loading()
        print(f"Model validation: {model_valid} - {model_message}")
        
        print("Testing system stats...")
        stats = await integration.get_enhanced_system_stats()
        print(f"System stats: {stats}")
        
        print("✅ All health checks completed successfully")
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        import traceback
traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_health_endpoint())