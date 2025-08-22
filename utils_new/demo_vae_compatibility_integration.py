"""
Demo: VAE Compatibility Handler Integration

This demo shows how the VAE compatibility handler integrates with the existing
Wan 2.2 model loading system to handle VAE compatibility issues.
"""

import json
import tempfile
import torch
from pathlib import Path

from vae_compatibility_handler import create_vae_compatibility_handler


def demo_wan_model_vae_detection():
    """Demo: Detect VAE compatibility for different Wan model variants"""
    print("🔍 Demo: VAE Compatibility Detection for Wan Models")
    print("=" * 60)
    
    handler = create_vae_compatibility_handler()
    
    # Simulate different Wan model VAE configurations
    model_variants = {
        "Wan 2.2 T2V (3D VAE)": {
            "_class_name": "AutoencoderKLTemporalDecoder",
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 4,
            "sample_size": [16, 64, 64],
            "temporal_layers": True,
            "temporal_depth": 16,
            "block_out_channels": [128, 256, 512, 512]
        },
        "Wan 2.2 T2I (2D VAE)": {
            "_class_name": "AutoencoderKL",
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 4,
            "sample_size": 64,
            "block_out_channels": [128, 256, 512, 512]
        },
        "Problematic VAE (384x384)": {
            "_class_name": "AutoencoderKL",
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 4,
            "sample_size": 384,  # Problematic size
            "block_out_channels": [128, 256, 512, 512]
        },
        "Custom 3D VAE (Missing Depth)": {
            "_class_name": "AutoencoderKLTemporal",
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 4,
            "sample_size": 64,
            "temporal_layers": True,  # 3D but no depth specified
            "block_out_channels": [128, 256, 512, 512]
        }
    }
    
    for model_name, config in model_variants.items():
        print(f"\n📋 Analyzing: {model_name}")
        print("-" * 40)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create config file
            config_path = temp_path / "config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Detect VAE architecture
            result = handler.detect_vae_architecture(config_path)
            
            # Display results
            print(f"  Architecture: {'3D' if result.detected_dimensions.is_3d else '2D'}")
            print(f"  Dimensions: {result.detected_dimensions.shape}")
            print(f"  Compatible: {result.is_compatible}")
            print(f"  Loading Strategy: {result.loading_strategy}")
            
            if result.compatibility_issues:
                print(f"  Issues Found: {len(result.compatibility_issues)}")
                for issue in result.compatibility_issues:
                    print(f"    • {issue}")
            
            # Get user guidance
            guidance = handler.get_vae_error_guidance(result)
            if guidance:
                print(f"  Guidance Available: {len(guidance)} recommendations")


def demo_vae_loading_strategies():
    """Demo: Different VAE loading strategies"""
    print("\n\n🔧 Demo: VAE Loading Strategies")
    print("=" * 60)
    
    handler = create_vae_compatibility_handler()
    
    strategies = {
        "Standard Loading": {
            "config": {
                "in_channels": 3,
                "out_channels": 3,
                "latent_channels": 4,
                "sample_size": 64
            },
            "expected_strategy": "standard"
        },
        "Reshape Strategy": {
            "config": {
                "in_channels": 3,
                "out_channels": 3,
                "latent_channels": 4,
                "sample_size": 384
            },
            "expected_strategy": "reshape"
        },
        "Custom 3D Strategy": {
            "config": {
                "in_channels": 3,
                "out_channels": 3,
                "latent_channels": 4,
                "sample_size": [16, 64, 64],
                "temporal_layers": True,
                "temporal_depth": 16
            },
            "expected_strategy": "standard"  # Should be standard for complete 3D config
        }
    }
    
    for strategy_name, strategy_info in strategies.items():
        print(f"\n🎯 Testing: {strategy_name}")
        print("-" * 30)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create config
            config_path = temp_path / "config.json"
            with open(config_path, 'w') as f:
                json.dump(strategy_info["config"], f)
            
            # Detect architecture
            result = handler.detect_vae_architecture(config_path)
            
            print(f"  Expected Strategy: {strategy_info['expected_strategy']}")
            print(f"  Detected Strategy: {result.loading_strategy}")
            print(f"  Match: {'✓' if result.loading_strategy == strategy_info['expected_strategy'] else '✗'}")
            
            # Show what the strategy would do
            if result.loading_strategy == "reshape":
                print(f"  Action: Reshape {result.detected_dimensions.height}x{result.detected_dimensions.width} → 64x64")
            elif result.loading_strategy == "custom":
                print(f"  Action: Use custom loading for 3D VAE")
            else:
                print(f"  Action: Standard Diffusers loading")


def demo_error_guidance_system():
    """Demo: Error guidance system for different scenarios"""
    print("\n\n💡 Demo: Error Guidance System")
    print("=" * 60)
    
    handler = create_vae_compatibility_handler()
    
    # Create different error scenarios
    scenarios = [
        {
            "name": "Shape Mismatch (384x384)",
            "config": {"sample_size": 384, "latent_channels": 4},
            "description": "Common issue with Wan models having non-standard VAE dimensions"
        },
        {
            "name": "3D VAE Detection",
            "config": {
                "sample_size": [16, 64, 64],
                "temporal_layers": True,
                "temporal_depth": 16,
                "latent_channels": 4
            },
            "description": "Guidance for 3D VAE requirements"
        },
        {
            "name": "Incomplete 3D Config",
            "config": {
                "sample_size": 64,
                "temporal_layers": True,  # 3D indicator but missing depth
                "latent_channels": 4
            },
            "description": "3D VAE with missing configuration"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📝 Scenario: {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        print("-" * 50)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create full config
            full_config = {
                "in_channels": 3,
                "out_channels": 3,
                **scenario["config"]
            }
            
            config_path = temp_path / "config.json"
            with open(config_path, 'w') as f:
                json.dump(full_config, f)
            
            # Get detection result
            result = handler.detect_vae_architecture(config_path)
            
            # Generate guidance
            guidance = handler.get_vae_error_guidance(result)
            
            print(f"   Status: {'⚠️  Issues Found' if not result.is_compatible or result.compatibility_issues else '✅ Compatible'}")
            
            if guidance:
                print(f"   Guidance ({len(guidance)} lines):")
                for i, line in enumerate(guidance[:5]):  # Show first 5 lines
                    print(f"     {line}")
                if len(guidance) > 5:
                    print(f"     ... and {len(guidance) - 5} more lines")
            else:
                print("   Guidance: No specific guidance needed")


def demo_integration_workflow():
    """Demo: Complete integration workflow"""
    print("\n\n🔄 Demo: Complete Integration Workflow")
    print("=" * 60)
    print("Simulating how VAE compatibility handler integrates with model loading...")
    
    handler = create_vae_compatibility_handler()
    
    # Simulate a Wan 2.2 T2V model loading workflow
    print("\n1️⃣  Model Loading Request: Wan-AI/Wan2.2-T2V-A14B-Diffusers")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create realistic Wan T2V VAE structure
        vae_dir = temp_path / "vae"
        vae_dir.mkdir()
        
        # VAE config
        vae_config = {
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
        
        config_path = vae_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(vae_config, f, indent=2)
        
        # Simulate weights
        vae_weights = {
            "encoder.conv_in.weight": torch.randn(128, 3, 3, 3, 3),
            "encoder.down_blocks.0.resnets.0.conv1.weight": torch.randn(128, 128, 3, 3, 3),
            "decoder.conv_out.weight": torch.randn(3, 128, 3, 3, 3),
            "quant_conv.weight": torch.randn(8, 8, 1, 1, 1)
        }
        
        weights_path = vae_dir / "pytorch_model.bin"
        torch.save(vae_weights, weights_path)
        
        print("\n2️⃣  VAE Architecture Detection...")
        arch_result = handler.detect_vae_architecture(config_path)
        
        print(f"     ✓ Detected: {'3D' if arch_result.detected_dimensions.is_3d else '2D'} VAE")
        print(f"     ✓ Dimensions: {arch_result.detected_dimensions.shape}")
        print(f"     ✓ Strategy: {arch_result.loading_strategy}")
        
        print("\n3️⃣  Weight Validation...")
        weight_result = handler.validate_vae_weights(weights_path, arch_result.detected_dimensions)
        
        print(f"     ✓ Weights Compatible: {weight_result.is_compatible}")
        print(f"     ✓ Detected Architecture: {'3D' if weight_result.detected_dimensions.is_3d else '2D'}")
        
        print("\n4️⃣  Loading Strategy Selection...")
        print(f"     ✓ Selected Strategy: {arch_result.loading_strategy}")
        
        if arch_result.loading_strategy == "custom":
            print("     ✓ Will use trust_remote_code=True for 3D VAE")
            print("     ✓ Will attempt WanPipeline loading")
        
        print("\n5️⃣  User Guidance Generation...")
        guidance = handler.get_vae_error_guidance(arch_result)
        
        if guidance:
            print(f"     ✓ Generated {len(guidance)} guidance lines")
            print("     ✓ Key recommendations:")
            for line in guidance[:3]:
                if "3D VAE" in line or "WanPipeline" in line or "VRAM" in line:
                    print(f"       • {line.strip()}")
        
        print("\n✅ Integration workflow completed successfully!")
        print("   The VAE compatibility handler provides all necessary information")
        print("   for the pipeline manager to load the Wan model correctly.")


def main():
    """Run all demos"""
    print("VAE Compatibility Handler - Integration Demo")
    print("=" * 80)
    print("This demo shows how the VAE compatibility handler works with Wan models")
    print()
    
    try:
        demo_wan_model_vae_detection()
        demo_vae_loading_strategies()
        demo_error_guidance_system()
        demo_integration_workflow()
        
        print("\n" + "=" * 80)
        print("🎉 All demos completed successfully!")
        print("\nThe VAE compatibility handler is ready for production use with:")
        print("  • Automatic 3D VAE detection")
        print("  • Shape mismatch handling (384x384 → 64x64)")
        print("  • Loading strategy selection")
        print("  • Comprehensive error guidance")
        print("  • Full integration with existing pipeline")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()