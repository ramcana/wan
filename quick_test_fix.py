#!/usr/bin/env python3
"""Quick test of the pipeline loader fix"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

async def test_fix():
    try:
        from backend.core.system_integration import get_system_integration
        
        print("Getting system integration...")
        integration = await get_system_integration()
        loader = integration.get_wan_pipeline_loader()
        
        print(f"Got loader: {type(loader).__name__}")
        
        # Test the method signature that was failing
        print("Testing method signature...")
        result = loader.load_wan_pipeline('test_path', True, True, {})
        print("✅ Method signature fixed - no TypeError!")
        return True
        
    except TypeError as e:
        print(f"❌ Still broken: {e}")
        return False
    except Exception as e:
        print(f"✅ Signature OK, got expected error: {e}")
        return True

if __name__ == "__main__":
    success = asyncio.run(test_fix())
    print("Fix successful!" if success else "Fix failed!")