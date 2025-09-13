"""
Test script for hardware detection functionality.
"""

import sys
import os
import json
import logging

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

# Import the detection module
from scripts.detect_system import SystemDetector

def main():
    """Test hardware detection functionality."""
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("=== WAN2.2 Hardware Detection Test ===\n")
    
    try:
        # Create detector
        detector = SystemDetector(".")
        
        # Detect hardware
        print("Detecting hardware...")
        profile = detector.detect_hardware()
        
        # Get optimal settings
        print("\nGenerating optimal settings...")
        settings = detector.get_optimal_settings(profile)
        
        print("\n=== OPTIMAL SETTINGS ===")
        print(json.dumps(settings, indent=2))
        
        # Validate requirements
        print("\n=== VALIDATION RESULTS ===")
        validation = detector.validate_requirements(profile)
        print(f"Status: {'PASSED' if validation.success else 'FAILED'}")
        print(f"Message: {validation.message}")
        
        if validation.warnings:
            print("\nWarnings:")
            for warning in validation.warnings:
                print(f"  ⚠️  {warning}")
        
        if not validation.success and validation.details:
            print("\nIssues:")
            for issue in validation.details.get("issues", []):
                print(f"  ❌ {issue}")
        
        print(f"\nPerformance Tier: {validation.details.get('performance_tier', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ Hardware detection failed: {e}")
        import traceback
traceback.print_exc()

if __name__ == "__main__":
    main()
