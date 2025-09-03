#!/usr/bin/env python3
"""
Simple test to check method existence
"""

import sys
from pathlib import Path

# Add the backend directory to the path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

try:
    from backend.services.generation_service import GenerationService
    service = GenerationService()
    
    print("Available methods:")
    methods = [method for method in dir(service) if method.startswith('_run')]
    for method in sorted(methods):
        print(f"  - {method}")
    
    print(f"\n_run_enhanced_generation exists: {hasattr(service, '_run_enhanced_generation')}")
    
    # Check if the method is callable
    if hasattr(service, '_run_enhanced_generation'):
        method = getattr(service, '_run_enhanced_generation')
        print(f"Method is callable: {callable(method)}")
        print(f"Method type: {type(method)}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
traceback.print_exc()