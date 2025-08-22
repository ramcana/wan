"""
Requirements validation test for VAE Compatibility Handler

This test validates that all requirements from the spec are properly implemented:
- Requirements 2.1, 2.2, 2.3, 2.4 from the wan22-model-compatibility spec
"""

import json
import tempfile
import torch
from pathlib import Path
from unittest.mock import Mock, patch

from vae_compatibility_handler import (
    VAECompatibilityHandler,
    VAEDimensions,
    VAECompatibilityResult,
    VAELoadingResult,
    create_vae_compatibility_handler
)


def test_requirement_2_1_recognize_3d_architecture():
    """
    Requirement 2.1: WHEN loading the Wan VAE THEN the system SHALL recognize 
    the 3D architecture and load weights correctly
    """
    print("Testing Requirement 2.1: 3D architecture recognition...")
    
    handler = VAECompatibilityHandler()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create 3D VAE config
        config_3d = {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 4,
            "sample_size": [16, 64, 64],  # 3D dimensions
            "temporal_layers": True,
            "temporal_depth": 16
        }
        
        config_path = temp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config_3d, f)
        
        # Test architecture detection
        result = handler.detect_vae_architecture(config_path)
        
        # Verify 3D architecture is recognized
        assert result.detected_dimensions.is_3d, "Failed to recognize 3D architecture"
        assert result.detected_dimensions.depth == 16, "Failed to detect correct temporal depth"
        assert result.detected_dimensions.channels == 4, "Failed to detect correct channels"
        
        # Create 3D weights
        weights_3d = {
            "encoder.conv_in.weight": torch.randn(128, 3, 3, 3, 3),  # 3D conv
            "decoder.conv_out.weight": torch.randn(3, 128, 3, 3, 3),
            "temporal_conv.weight": torch.randn(64, 64, 1, 3, 3)
        }
        
        weights_path = temp_path / "pytorch_model.bin"
        torch.save(weights_3d, weights_path)
        
        # Test weight validation
        validation_result = handler.validate_vae_weights(weights_path, result.detected_dimensions)
        
        # Verify 3D weights are recognized
        assert validation_result.detected_dimensions.is_3d, "Failed to recognize 3D weights"
        
    print("✓ Requirement 2.1 validated: 3D architecture recognition works")


def test_requirement_2_2_no_random_initialization():
    """
    Requirement 2.2: WHEN VAE shape mismatches occur THEN the system SHALL NOT 
    fall back to random initialization
    """
    print("Testing Requirement 2.2: No random initialization fallback...")
    
    handler = VAECompatibilityHandler()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create problematic VAE config (shape mismatch scenario)
        config_problematic = {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 4,
            "sample_size": 384  # Problematic size that could cause mismatch
        }
        
        config_path = temp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config_problematic, f)
        
        # Test detection
        result = handler.detect_vae_architecture(config_path)
        
        # Verify system detects the issue and provides handling strategy
        assert result.loading_strategy == "reshape", "Failed to detect shape mismatch"
        assert len(result.compatibility_issues) > 0, "Failed to identify compatibility issues"
        assert any("384" in issue for issue in result.compatibility_issues), "Failed to identify specific shape issue"
        
        # Verify the system provides a strategy instead of allowing random initialization
        assert result.loading_strategy != "standard", "Should not use standard loading for problematic shapes"
        
        # Test that loading strategy is provided (not random initialization)
        with patch('diffusers.AutoencoderKL') as mock_autoencoder:
            mock_vae = Mock()
            mock_vae.config = Mock()
            mock_vae.config.sample_size = [384, 384]
            mock_autoencoder.from_pretrained.return_value = mock_vae
            
            loading_result = handler.load_vae_with_compatibility(temp_path, result)
            
            # Verify reshape strategy was applied (not random initialization)
            assert loading_result.loading_strategy_used == "reshape", "Failed to apply reshape strategy"
            assert mock_vae.config.sample_size == [64, 64], "Failed to apply reshape correction"
    
    print("✓ Requirement 2.2 validated: No random initialization fallback")


def test_requirement_2_3_handle_dimension_differences():
    """
    Requirement 2.3: WHEN the VAE has shape [384, ...] instead of [64, ...] THEN 
    the system SHALL handle the dimensional difference appropriately
    """
    print("Testing Requirement 2.3: Handle dimensional differences...")
    
    handler = VAECompatibilityHandler()
    
    # Test various problematic dimensions
    problematic_sizes = [384, 512, [384, 384], [512, 512]]
    
    for size in problematic_sizes:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            config_data = {
                "in_channels": 3,
                "out_channels": 3,
                "latent_channels": 4,
                "sample_size": size
            }
            
            config_path = temp_path / "config.json"
            with open(config_path, 'w') as f:
                json.dump(config_data, f)
            
            result = handler.detect_vae_architecture(config_path)
            
            # Verify dimensional differences are handled
            if isinstance(size, int) and size > 64:
                assert result.loading_strategy in ["reshape", "custom"], f"Failed to handle size {size}"
                assert len(result.compatibility_issues) > 0, f"Failed to identify issues for size {size}"
            elif isinstance(size, list) and any(s > 64 for s in size):
                # For list sizes, check if problematic dimensions are detected
                detected_height = result.detected_dimensions.height
                detected_width = result.detected_dimensions.width
                if detected_height > 64 or detected_width > 64:
                    assert result.loading_strategy in ["reshape", "custom"], f"Failed to handle size {size}"
    
    print("✓ Requirement 2.3 validated: Dimensional differences handled appropriately")


def test_requirement_2_4_specific_error_messages():
    """
    Requirement 2.4: IF VAE loading fails THEN the system SHALL provide specific 
    error messages about VAE compatibility requirements
    """
    print("Testing Requirement 2.4: Specific error messages...")
    
    handler = VAECompatibilityHandler()
    
    # Test 1: Missing config file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        missing_config = temp_path / "nonexistent.json"
        
        result = handler.detect_vae_architecture(missing_config)
        
        assert not result.is_compatible, "Should detect incompatibility for missing file"
        assert result.error_message is not None, "Should provide error message"
        assert "not found" in result.error_message, "Should specify file not found"
    
    # Test 2: Invalid config file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        invalid_config = temp_path / "invalid.json"
        
        with open(invalid_config, 'w') as f:
            f.write("invalid json content")
        
        result = handler.detect_vae_architecture(invalid_config)
        
        assert not result.is_compatible, "Should detect incompatibility for invalid file"
        assert result.error_message is not None, "Should provide error message"
        assert "Failed to analyze" in result.error_message, "Should specify analysis failure"
    
    # Test 3: Missing weights file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        missing_weights = temp_path / "nonexistent_weights.bin"
        expected_dims = VAEDimensions(channels=4, height=64, width=64, is_3d=False)
        
        result = handler.validate_vae_weights(missing_weights, expected_dims)
        
        assert not result.is_compatible, "Should detect incompatibility for missing weights"
        assert result.error_message is not None, "Should provide error message"
        assert "not found" in result.error_message, "Should specify weights not found"
    
    # Test 4: Comprehensive error guidance
    problematic_result = VAECompatibilityResult(
        is_compatible=False,
        detected_dimensions=VAEDimensions(4, 384, 384, is_3d=False),
        compatibility_issues=["Detected problematic VAE shape (4, 384, 384), may need reshaping"],
        error_message="Shape mismatch detected"
    )
    
    guidance = handler.get_vae_error_guidance(problematic_result)
    
    assert len(guidance) > 0, "Should provide error guidance"
    assert any("VAE Compatibility Issues" in line for line in guidance), "Should identify compatibility issues"
    assert any("384x384 instead of 64x64" in line for line in guidance), "Should provide specific shape guidance"
    assert any("trust_remote_code=True" in line for line in guidance), "Should provide solution guidance"
    
    # Test 5: 3D VAE specific guidance
    vae_3d_result = VAECompatibilityResult(
        is_compatible=True,
        detected_dimensions=VAEDimensions(4, 64, 64, depth=16, is_3d=True),
        loading_strategy="custom"
    )
    
    guidance_3d = handler.get_vae_error_guidance(vae_3d_result)
    
    assert any("3D VAE Detected" in line for line in guidance_3d), "Should identify 3D VAE"
    assert any("video generation" in line for line in guidance_3d), "Should explain 3D purpose"
    assert any("WanPipeline" in line for line in guidance_3d), "Should mention required pipeline"
    assert any("VRAM" in line for line in guidance_3d), "Should mention resource requirements"
    
    print("✓ Requirement 2.4 validated: Specific error messages provided")


def test_integration_all_requirements():
    """
    Integration test: Verify all requirements work together in realistic scenarios
    """
    print("Testing integration of all VAE requirements...")
    
    handler = create_vae_compatibility_handler()
    
    # Scenario 1: Wan T2V model with 3D VAE
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create realistic Wan T2V VAE config
        wan_config = {
            "_class_name": "AutoencoderKLTemporalDecoder",
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 4,
            "sample_size": [16, 64, 64],
            "temporal_layers": True,
            "temporal_depth": 16,
            "block_out_channels": [128, 256, 512, 512],
            "layers_per_block": 2
        }
        
        config_path = temp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(wan_config, f)
        
        # Create realistic 3D weights
        wan_weights = {
            "encoder.conv_in.weight": torch.randn(128, 3, 3, 3, 3),
            "encoder.down_blocks.0.resnets.0.conv1.weight": torch.randn(128, 128, 3, 3, 3),
            "decoder.conv_out.weight": torch.randn(3, 128, 3, 3, 3),
            "decoder.up_blocks.0.resnets.0.conv1.weight": torch.randn(128, 128, 3, 3, 3),
            "quant_conv.weight": torch.randn(8, 8, 1, 1, 1)
        }
        
        weights_path = temp_path / "pytorch_model.bin"
        torch.save(wan_weights, weights_path)
        
        # Test complete workflow
        # Step 1: Architecture detection
        arch_result = handler.detect_vae_architecture(config_path)
        assert arch_result.detected_dimensions.is_3d, "Should detect 3D architecture"
        assert arch_result.detected_dimensions.depth == 16, "Should detect correct depth"
        
        # Step 2: Weight validation
        weight_result = handler.validate_vae_weights(weights_path, arch_result.detected_dimensions)
        assert weight_result.detected_dimensions.is_3d, "Should validate 3D weights"
        
        # Step 3: Error guidance
        guidance = handler.get_vae_error_guidance(arch_result)
        assert any("3D VAE" in line for line in guidance), "Should provide 3D VAE guidance"
    
    # Scenario 2: Problematic VAE with shape mismatch
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create problematic config
        prob_config = {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 4,
            "sample_size": 384  # Problematic
        }
        
        config_path = temp_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(prob_config, f)
        
        # Test detection and handling
        prob_result = handler.detect_vae_architecture(config_path)
        assert prob_result.loading_strategy == "reshape", "Should use reshape strategy"
        assert len(prob_result.compatibility_issues) > 0, "Should identify issues"
        
        # Test guidance
        prob_guidance = handler.get_vae_error_guidance(prob_result)
        assert any("Shape Mismatch" in line for line in prob_guidance), "Should provide shape guidance"
    
    print("✓ Integration test passed: All requirements work together")


def main():
    """Run all requirement validation tests"""
    print("VAE Compatibility Handler - Requirements Validation")
    print("=" * 60)
    print("Validating implementation against spec requirements 2.1-2.4")
    print()
    
    tests = [
        test_requirement_2_1_recognize_3d_architecture,
        test_requirement_2_2_no_random_initialization,
        test_requirement_2_3_handle_dimension_differences,
        test_requirement_2_4_specific_error_messages,
        test_integration_all_requirements
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Requirements validation completed: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 ALL REQUIREMENTS VALIDATED SUCCESSFULLY!")
        print("\nImplemented capabilities:")
        print("  ✓ 2.1: 3D VAE architecture recognition and weight loading")
        print("  ✓ 2.2: No random initialization fallback on shape mismatches")
        print("  ✓ 2.3: Proper handling of [384,...] vs [64,...] dimensions")
        print("  ✓ 2.4: Specific error messages for VAE compatibility issues")
        print("\nThe VAE compatibility handling system is ready for integration!")
        return True
    else:
        print(f"\n❌ {failed} requirements failed validation")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)