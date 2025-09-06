#!/usr/bin/env python3
"""
Quick fix script to ensure all import paths are correct for running from backend directory
"""

import os
import sys
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

print("✅ Import paths fixed for backend directory")
print(f"📁 Current directory: {current_dir}")
print(f"🐍 Python path: {sys.path[:3]}...")

# Test critical imports
try:
    from core.system_integration import SystemIntegration
    print("✅ SystemIntegration import: OK")
except ImportError as e:
    print(f"❌ SystemIntegration import failed: {e}")

try:
    from services.generation_service import GenerationService
    print("✅ GenerationService import: OK")
except ImportError as e:
    print(f"❌ GenerationService import failed: {e}")

try:
    from core.performance_monitor import get_performance_monitor
    print("✅ PerformanceMonitor import: OK")
except ImportError as e:
    print(f"❌ PerformanceMonitor import failed: {e}")

try:
    from api.performance import router
    print("✅ Performance API import: OK")
except ImportError as e:
    print(f"❌ Performance API import failed: {e}")

print("\n🎯 Import validation complete!")