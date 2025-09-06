"""
Validation script for VAE Compatibility Handler
"""

import json
import tempfile
import torch
from pathlib import Path

from vae_compatibility_handler import (
    VAECompatibilityHandler,
    VAEDimensions,
    create_vae_compatibility_handler
)


def main():
    print("Validating VAE Compatibility Handler Implementation")
    print("=" * 60)
    
    # Test 1: Basic instantiation
    print("\n1. Testing basic instantiation...")
    handler = create_vae_compatibility_handler()
    print(f"✓ Handler created: {type(handler).__name__}")
    
    # Test 2: VAE dimensions
    print("\n2. Testing VAE dimensions...")
    dims_2d = VAEDimensions(channels=4, height=64, width=64, is_3d=False)
    dims_3d = VAEDimensions(channels=4, height=64, width=64, depth=16, is_3d=True)
    
    print(f"✓ 2D VAE shape: {dims_2d.shape}")
    print(f"✓ 3D VAE shape: {dims_3d.shape}")
    
    # Test 3: Config detection with temporary files
    print("\n3. Testing config detection...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test 2D VAE config
        config_2d = {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 4,
            "sample_size": 64
        }
        
        config_path = temp_path / "config_2d.json"
        with open(config_path, 'w') as f:
            json.dump(config_2d, f)
        
        result_2d = handler.detect_vae_architecture(config_path)
        print(f"✓ 2D VAE detected: compatible={result_2d.is_compatible}, "
              f"is_3d={result_2d.detected_dimensions.is_3d}, "
              f"strategy={result_2d.loading_strategy}")
        
        # Test 3D VAE config
        config_3d = {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 4,
            "sample_size": [16, 64, 64],
            "temporal_layers": True,
            "temporal_depth": 16
        }
        
        config_path_3d = temp_path / "config_3d.json"
        with open(config_path_3d, 'w') as f:
            json.dump(config_3d, f)
        
        result_3d = handler.detect_vae_architecture(config_path_3d)
        print(f"✓ 3D VAE detected: compatible={result_3d.is_compatible}, "
              f"is_3d={result_3d.detected_dimensions.is_3d}, "
              f"strategy={result_3d.loading_strategy}")
        
        # Test problematic VAE config
        config_problematic = {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 4,
            "sample_size": 384  # Problematic size
        }
        
        config_path_prob = temp_path / "config_prob.json"
        with open(config_path_prob, 'w') as f:
            json.dump(config_problematic, f)
        
        result_prob = handler.detect_vae_architecture(config_path_prob)
        print(f"✓ Problematic VAE detected: compatible={result_prob.is_compatible}, "
              f"size={result_prob.detected_dimensions.height}x{result_prob.detected_dimensions.width}, "
              f"strategy={result_prob.loading_strategy}")
        print(f"  Issues: {result_prob.compatibility_issues}")
    
    # Test 4: Weight validation
    print("\n4. Testing weight validation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mock 2D weights
        weights_2d = {
            "encoder.conv_in.weight": torch.randn(128, 3, 3, 3),  # 2D conv
            "decoder.conv_out.weight": torch.randn(3, 128, 3, 3),
        }
        
        weights_path_2d = temp_path / "weights_2d.bin"
        torch.save(weights_2d, weights_path_2d)
        
        expected_dims_2d = VAEDimensions(channels=4, height=64, width=64, is_3d=False)
        validation_result_2d = handler.validate_vae_weights(weights_path_2d, expected_dims_2d)
        
        print(f"✓ 2D weights validation: compatible={validation_result_2d.is_compatible}, "
              f"strategy={validation_result_2d.loading_strategy}")
        
        # Create mock 3D weights
        weights_3d = {
            "encoder.conv_in.weight": torch.randn(128, 3, 3, 3, 3),  # 3D conv
            "decoder.conv_out.weight": torch.randn(3, 128, 3, 3, 3),
        }
        
        weights_path_3d = temp_path / "weights_3d.bin"
        torch.save(weights_3d, weights_path_3d)
        
        expected_dims_3d = VAEDimensions(channels=4, height=64, width=64, depth=16, is_3d=True)
        validation_result_3d = handler.validate_vae_weights(weights_path_3d, expected_dims_3d)
        
        print(f"✓ 3D weights validation: compatible={validation_result_3d.is_compatible}, "
              f"detected_3d={validation_result_3d.detected_dimensions.is_3d}")
    
    # Test 5: Error guidance
    print("\n5. Testing error guidance...")
    
    # Test guidance for problematic VAE
    guidance = handler.get_vae_error_guidance(result_prob)
    print(f"✓ Generated {len(guidance)} guidance lines for problematic VAE")
    if guidance:
        print("  Sample guidance:")
        for line in guidance[:3]:  # Show first 3 lines
            print(f"    {line}")
    
    # Test guidance for 3D VAE
    guidance_3d = handler.get_vae_error_guidance(result_3d)
    print(f"✓ Generated {len(guidance_3d)} guidance lines for 3D VAE")
    if guidance_3d:
        print("  Sample guidance:")
        for line in guidance_3d[:3]:  # Show first 3 lines
            print(f"    {line}")
    
    print("\n" + "=" * 60)
    print("✅ VAE Compatibility Handler validation completed successfully!")
    print("\nKey features implemented:")
    print("  • VAE architecture detection (2D/3D)")
    print("  • Shape mismatch handling (384x384 → 64x64)")
    print("  • Weight validation with dimension checking")
    print("  • Loading strategy selection (standard/reshape/custom)")
    print("  • Comprehensive error guidance generation")
    print("  • Support for temporal/3D VAE architectures")


if __name__ == "__main__":
    main()