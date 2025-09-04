import pytest
"""
Simple tests for VAE Compatibility Handler (no pytest required)
"""

import json
import tempfile
import torch
import traceback
from pathlib import Path

from vae_compatibility_handler import (
    VAECompatibilityHandler,
    VAEDimensions,
    VAECompatibilityResult,
    create_vae_compatibility_handler
)


def test_vae_dimensions():
    """Test VAE dimensions data class"""
    print("Testing VAE dimensions...")
    
    # Test 2D VAE
    dims_2d = VAEDimensions(channels=4, height=64, width=64, is_3d=False)
    assert dims_2d.channels == 4
    assert dims_2d.shape == (4, 64, 64)
    print("✓ 2D VAE dimensions test passed")
    
    # Test 3D VAE
    dims_3d = VAEDimensions(channels=4, height=64, width=64, depth=16, is_3d=True)
    assert dims_3d.is_3d
    assert dims_3d.shape == (4, 16, 64, 64)
    print("✓ 3D VAE dimensions test passed")


def create_vae_config(temp_dir: Path, config_data: dict) -> Path:
    """Create a temporary VAE config file"""
    config_path = temp_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config_data, f)
    return config_path


def test_detect_2d_vae_architecture():
    """Test detection of 2D VAE architecture"""
    print("Testing 2D VAE architecture detection...")
    
    handler = VAECompatibilityHandler()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        config_data = {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 4,
            "sample_size": 64
        }
        
        config_path = create_vae_config(temp_path, config_data)
        result = handler.detect_vae_architecture(config_path)
        
        assert result.is_compatible
        assert not result.detected_dimensions.is_3d
        assert result.loading_strategy == "standard"
        
    print("✓ 2D VAE architecture detection test passed")


def test_detect_problematic_vae_shape():
    """Test detection of problematic VAE shape (384x384)"""
    print("Testing problematic VAE shape detection...")
    
    handler = VAECompatibilityHandler()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        config_data = {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 4,
            "sample_size": 384  # Problematic size
        }
        
        config_path = create_vae_config(temp_path, config_data)
        result = handler.detect_vae_architecture(config_path)
        
        assert result.detected_dimensions.height == 384
        assert result.loading_strategy == "reshape"
        assert any("384" in issue for issue in result.compatibility_issues)
        
    print("✓ Problematic VAE shape detection test passed")


def run_all_tests():
    """Run all tests"""
    print("Running VAE Compatibility Handler Tests")
    print("=" * 50)
    
    tests = [
        test_vae_dimensions,
        test_detect_2d_vae_architecture,
        test_detect_problematic_vae_shape
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Tests completed: {passed} passed, {failed} failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)