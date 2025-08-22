#!/usr/bin/env python3
"""
Test the model override
"""

import sys
sys.path.insert(0, ".")

from model_override import get_local_model_path, patch_model_loading

def test_override():
    """Test the model override functionality"""
    
    print("Testing Model Override")
    print("=" * 25)
    
    # Test direct path lookup
    try:
        model_path = get_local_model_path("Wan-AI/Wan2.2-T2V-A14B-Diffusers")
        print(f"Found T2V model: {model_path}")
        return True
    except FileNotFoundError as e:
        print(f"T2V model not found: {e}")
        return False

if __name__ == "__main__":
    success = test_override()
    if success:
        print("\nModel override is working!")
        print("To use: import model_override; model_override.patch_model_loading()")
    else:
        print("\nModel override failed")