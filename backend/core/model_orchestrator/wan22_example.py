#!/usr/bin/env python3
"""
Example script demonstrating WAN2.2-specific model handling features.

This script shows how to:
1. Use selective downloading for development variants
2. Validate inputs for different model types
3. Configure VAE settings for memory optimization
4. Cache text embeddings for performance
5. Estimate memory usage for different configurations
"""

import tempfile
from pathlib import Path
from PIL import Image
import torch

from .model_registry import ModelRegistry, ModelSpec, FileSpec, ModelDefaults, VramEstimation
from .wan22_handler import WAN22ModelHandler
from .exceptions import InvalidInputError


def create_sample_model_spec() -> ModelSpec:
    """Create a sample WAN2.2 model specification for demonstration."""
    return ModelSpec(
        model_id="ti2v-5b@2.2.0",
        version="2.2.0",
        variants=["fp16", "bf16", "fp16-dev", "bf16-dev"],
        default_variant="fp16",
        files=[
            FileSpec(
                path="config.json",
                size=1024,
                sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                component="config"
            ),
            FileSpec(
                path="diffusion_pytorch_model.safetensors.index.json",
                size=4096,
                sha256="2c624232cdd221771294dfbb310aca000a0df6ac8b66b696b90ef72c9e9c323c",
                component="unet",
                shard_index=True
            ),
            FileSpec(
                path="diffusion_pytorch_model-00001-of-00003.safetensors",
                size=17179869184,
                sha256="d4735e3a265e16eee03f59718b9b5d03019c07d8b6c51f90da3a666eec13ab35",
                component="unet",
                shard_part=1
            ),
            FileSpec(
                path="diffusion_pytorch_model-00002-of-00003.safetensors",
                size=17179869184,
                sha256="4e07408562bedb8b60ce05c1decfe3ad16b72230967de01f640b7e4729b49fce",
                component="unet",
                shard_part=2
            ),
            FileSpec(
                path="diffusion_pytorch_model-00003-of-00003.safetensors",
                size=17179869184,
                sha256="4b227777d4dd1fc61c6f884f48641d02b4d121d3fd328cb08b5531fcacdabf8a",
                component="unet",
                shard_part=3
            ),
            FileSpec(
                path="text_encoder/pytorch_model.bin",
                size=4294967296,
                sha256="ef2d127de37b942baad06145e54b0c619a1f22327b2ebbcfbec78f5564afe39d",
                component="text_encoder"
            ),
            FileSpec(
                path="image_encoder/pytorch_model.bin",
                size=1073741824,
                sha256="1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b",
                component="image_encoder"
            ),
            FileSpec(
                path="vae/diffusion_pytorch_model.safetensors",
                size=335544320,
                sha256="e7f6c011776e8db7cd330b54174fd76f7d0216b612387a5ffcfb81e6f0919683",
                component="vae"
            ),
        ],
        sources=["local://ti2v-5b@2.2.0"],
        allow_patterns=["*.safetensors", "*.json", "*.bin"],
        resolution_caps=["720p24", "1080p24", "1440p24"],
        optional_components=[],
        lora_required=False,
        description="WAN2.2 Text+Image-to-Video 5B Model",
        required_components=["text_encoder", "image_encoder", "unet", "vae"],
        defaults=ModelDefaults(
            fps=24,
            frames=24,
            scheduler="ddim",
            precision="fp16",
            guidance_scale=7.5,
            image_guidance_scale=1.2,
            text_guidance_scale=7.5,
            vae_tile_size=512,
            vae_decode_chunk_size=8
        ),
        vram_estimation=VramEstimation(
            params_billion=5.0,
            family_size="medium",
            base_vram_gb=10.0,
            per_frame_vram_mb=200.0
        ),
        model_type="ti2v"
    )


def demonstrate_selective_downloading():
    """Demonstrate selective downloading for development variants."""
    print("=== Selective Downloading Demo ===")
    
    model_spec = create_sample_model_spec()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        handler = WAN22ModelHandler(model_spec, Path(temp_dir))
        
        # Production variant - all shards
        prod_shards = handler.get_required_shards("unet", "fp16")
        print(f"Production variant (fp16): {len(prod_shards)} UNet shards required")
        for shard in prod_shards:
            print(f"  - Shard {shard.shard_part}: {shard.file_path}")
        
        # Development variant - fewer shards
        dev_shards = handler.get_required_shards("unet", "fp16-dev")
        print(f"Development variant (fp16-dev): {len(dev_shards)} UNet shards required")
        for shard in dev_shards:
            print(f"  - Shard {shard.shard_part}: {shard.file_path}")
        
        print(f"Development variant saves {len(prod_shards) - len(dev_shards)} shards!")
        print()


def demonstrate_input_validation():
    """Demonstrate input validation for different model types."""
    print("=== Input Validation Demo ===")
    
    model_spec = create_sample_model_spec()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        handler = WAN22ModelHandler(model_spec, Path(temp_dir))
        
        # Valid TI2V input
        valid_input = {
            "prompt": "A beautiful sunset over mountains",
            "image": Image.new("RGB", (1024, 576), color="blue")
        }
        
        try:
            handler.validate_input_for_model_type(valid_input)
            print("✓ Valid TI2V input accepted")
        except InvalidInputError as e:
            print(f"✗ Unexpected validation error: {e}")
        
        # Invalid input - missing prompt
        invalid_input_1 = {
            "image": Image.new("RGB", (1024, 576), color="blue")
        }
        
        try:
            handler.validate_input_for_model_type(invalid_input_1)
            print("✗ Invalid input was incorrectly accepted")
        except InvalidInputError:
            print("✓ Missing prompt correctly rejected")
        
        # Invalid input - missing image
        invalid_input_2 = {
            "prompt": "A beautiful sunset over mountains"
        }
        
        try:
            handler.validate_input_for_model_type(invalid_input_2)
            print("✗ Invalid input was incorrectly accepted")
        except InvalidInputError:
            print("✓ Missing image correctly rejected")
        
        print()


def demonstrate_vae_configuration():
    """Demonstrate VAE configuration for memory optimization."""
    print("=== VAE Configuration Demo ===")
    
    model_spec = create_sample_model_spec()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        handler = WAN22ModelHandler(model_spec, Path(temp_dir))
        
        # Production configuration
        prod_config = handler.get_vae_config("fp16")
        print("Production VAE config (fp16):")
        print(f"  - Tile size: {prod_config['tile_size']}")
        print(f"  - Decode chunk size: {prod_config['decode_chunk_size']}")
        print(f"  - Data type: {prod_config['dtype']}")
        print(f"  - Tiling enabled: {prod_config['enable_tiling']}")
        
        # Development configuration (more aggressive optimization)
        dev_config = handler.get_vae_config("fp16-dev")
        print("Development VAE config (fp16-dev):")
        print(f"  - Tile size: {dev_config['tile_size']}")
        print(f"  - Decode chunk size: {dev_config['decode_chunk_size']}")
        print(f"  - Data type: {dev_config['dtype']}")
        print(f"  - Tiling enabled: {dev_config['enable_tiling']}")
        
        print()


def demonstrate_text_embedding_cache():
    """Demonstrate text embedding caching."""
    print("=== Text Embedding Cache Demo ===")
    
    model_spec = create_sample_model_spec()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        handler = WAN22ModelHandler(model_spec, Path(temp_dir))
        
        # Cache some embeddings
        prompts = [
            "A beautiful sunset",
            "Mountains in the distance",
            "Ocean waves crashing",
            "Forest in autumn"
        ]
        
        print("Caching text embeddings...")
        for i, prompt in enumerate(prompts):
            # Simulate text embedding
            embedding = torch.randn(1, 768)
            handler.cache_text_embedding(prompt, embedding)
            print(f"  - Cached: '{prompt}'")
        
        # Retrieve cached embeddings
        print("\nRetrieving cached embeddings...")
        for prompt in prompts[:2]:  # Test first two
            cached = handler.get_cached_text_embedding(prompt)
            if cached is not None:
                print(f"  ✓ Found cached embedding for: '{prompt}' (shape: {cached.shape})")
            else:
                print(f"  ✗ No cached embedding for: '{prompt}'")
        
        # Test cache miss
        new_prompt = "A completely new prompt"
        cached = handler.get_cached_text_embedding(new_prompt)
        if cached is None:
            print(f"  ✓ Cache miss for new prompt: '{new_prompt}'")
        
        print()


def demonstrate_memory_estimation():
    """Demonstrate memory usage estimation."""
    print("=== Memory Estimation Demo ===")
    
    model_spec = create_sample_model_spec()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        handler = WAN22ModelHandler(model_spec, Path(temp_dir))
        
        # Different input configurations
        configs = [
            {"frames": 16, "width": 720, "height": 480, "name": "720p, 16 frames"},
            {"frames": 24, "width": 1024, "height": 576, "name": "1024x576, 24 frames"},
            {"frames": 32, "width": 1440, "height": 810, "name": "1440p, 32 frames"},
        ]
        
        variants = ["fp32", "fp16", "bf16"]
        
        for config in configs:
            print(f"Configuration: {config['name']}")
            for variant in variants:
                memory_est = handler.estimate_memory_usage(config, variant)
                print(f"  {variant:>4}: {memory_est['total']:.1f} GB total "
                      f"({memory_est['base']:.1f} GB base + {memory_est['dynamic']:.1f} GB dynamic)")
            print()


def demonstrate_variant_switching():
    """Demonstrate development/production variant switching."""
    print("=== Variant Switching Demo ===")
    
    model_spec = create_sample_model_spec()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        handler = WAN22ModelHandler(model_spec, Path(temp_dir))
        
        base_variants = ["fp16", "bf16", "int8"]
        
        print("Variant conversions:")
        for base in base_variants:
            dev_variant = handler.get_development_variant(base)
            prod_variant = handler.get_production_variant(dev_variant)
            
            print(f"  {base} → {dev_variant} → {prod_variant}")
            print(f"    Is dev variant: {handler.is_development_variant(dev_variant)}")
            print(f"    Is prod variant: {handler.is_development_variant(base)}")
        
        print()


def main():
    """Run all demonstrations."""
    print("WAN2.2 Model Handler Demonstration")
    print("=" * 50)
    print()
    
    try:
        demonstrate_selective_downloading()
        demonstrate_input_validation()
        demonstrate_vae_configuration()
        demonstrate_text_embedding_cache()
        demonstrate_memory_estimation()
        demonstrate_variant_switching()
        
        print("All demonstrations completed successfully!")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()