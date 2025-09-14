"""
Tests for WAN2.2-specific model handling.

Tests cover:
- Sharded model support and selective downloading
- Model-specific input validation
- Text embedding caching
- VAE configuration and memory optimization
- Development/production variant switching
- Component validation and file management
"""

import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import torch
import numpy as np
from PIL import Image

from .wan22_handler import (
    WAN22ModelHandler, 
    ShardInfo, 
    ComponentInfo, 
    TextEmbeddingCache
)
from .model_registry import ModelSpec, FileSpec, ModelDefaults, VramEstimation
from .exceptions import ModelValidationError, InvalidInputError


class TestTextEmbeddingCache:
    """Test text embedding cache functionality."""
    
    def test_cache_basic_operations(self):
        """Test basic cache operations."""
        cache = TextEmbeddingCache(max_size=3)
        
        # Test empty cache
        assert cache.get("test") is None
        
        # Test caching
        embedding = torch.randn(1, 512)
        cache.put("test", embedding)
        
        cached = cache.get("test")
        assert cached is not None
        assert torch.equal(cached, embedding)
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = TextEmbeddingCache(max_size=2)
        
        # Fill cache
        emb1 = torch.randn(1, 512)
        emb2 = torch.randn(1, 512)
        cache.put("text1", emb1)
        cache.put("text2", emb2)
        
        # Access first item to make it more recent
        cache.get("text1")
        
        # Add third item, should evict text2
        emb3 = torch.randn(1, 512)
        cache.put("text3", emb3)
        
        assert cache.get("text1") is not None
        assert cache.get("text2") is None
        assert cache.get("text3") is not None
    
    def test_cache_clear(self):
        """Test cache clearing."""
        cache = TextEmbeddingCache()
        cache.put("test", torch.randn(1, 512))
        
        assert cache.get("test") is not None
        cache.clear()
        assert cache.get("test") is None


class TestWAN22ModelHandler:
    """Test WAN2.2 model handler functionality."""
    
    @pytest.fixture
    def sample_model_spec(self):
        """Create a sample model specification."""
        return ModelSpec(
            model_id="t2v-A14B@2.2.0",
            version="2.2.0",
            variants=["fp16", "bf16", "fp16-dev"],
            default_variant="fp16",
            files=[
                FileSpec(
                    path="config.json",
                    size=1024,
                    sha256="a" * 64,
                    component="config"
                ),
                FileSpec(
                    path="unet/diffusion_pytorch_model.safetensors.index.json",
                    size=4096,
                    sha256="b" * 64,
                    component="unet",
                    shard_index=True
                ),
                FileSpec(
                    path="unet/diffusion_pytorch_model-00001-of-00003.safetensors",
                    size=8589934592,
                    sha256="c" * 64,
                    component="unet",
                    shard_part=1
                ),
                FileSpec(
                    path="unet/diffusion_pytorch_model-00002-of-00003.safetensors",
                    size=8589934592,
                    sha256="d" * 64,
                    component="unet",
                    shard_part=2
                ),
                FileSpec(
                    path="unet/diffusion_pytorch_model-00003-of-00003.safetensors",
                    size=4294967296,
                    sha256="e" * 64,
                    component="unet",
                    shard_part=3
                ),
                FileSpec(
                    path="text_encoder/pytorch_model.bin",
                    size=4294967296,
                    sha256="f" * 64,
                    component="text_encoder"
                ),
                FileSpec(
                    path="vae/diffusion_pytorch_model.safetensors",
                    size=335544320,
                    sha256="g" * 64,
                    component="vae"
                ),
            ],
            sources=["local://test"],
            allow_patterns=["*.safetensors", "*.json"],
            resolution_caps=["720p24", "1080p24"],
            optional_components=[],
            lora_required=False,
            description="Test T2V model",
            required_components=["text_encoder", "unet", "vae"],
            defaults=ModelDefaults(
                fps=24,
                frames=16,
                scheduler="ddim",
                precision="fp16",
                guidance_scale=7.5,
                vae_tile_size=512,
                vae_decode_chunk_size=8
            ),
            vram_estimation=VramEstimation(
                params_billion=14.0,
                family_size="large",
                base_vram_gb=12.0,
                per_frame_vram_mb=256.0
            ),
            model_type="t2v"
        )
    
    @pytest.fixture
    def temp_model_dir(self):
        """Create a temporary model directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_component_parsing(self, sample_model_spec, temp_model_dir):
        """Test parsing of model components from file specs."""
        handler = WAN22ModelHandler(sample_model_spec, temp_model_dir)
        
        # Check that components were parsed correctly
        assert "config" in handler.components
        assert "unet" in handler.components
        assert "text_encoder" in handler.components
        assert "vae" in handler.components
        
        # Check UNet component has shards
        unet_component = handler.components["unet"]
        assert len(unet_component.shards) == 3
        assert unet_component.shards[0].shard_part == 1
        assert unet_component.shards[1].shard_part == 2
        assert unet_component.shards[2].shard_part == 3
    
    def test_required_shards_production(self, sample_model_spec, temp_model_dir):
        """Test getting required shards for production variant."""
        handler = WAN22ModelHandler(sample_model_spec, temp_model_dir)
        
        shards = handler.get_required_shards("unet", "fp16")
        assert len(shards) == 3  # All shards for production
        
        shard_parts = [s.shard_part for s in shards]
        assert shard_parts == [1, 2, 3]
    
    def test_required_shards_development(self, sample_model_spec, temp_model_dir):
        """Test getting required shards for development variant."""
        handler = WAN22ModelHandler(sample_model_spec, temp_model_dir)
        
        shards = handler.get_required_shards("unet", "fp16-dev")
        assert len(shards) == 2  # Only first 2 shards for development
        
        shard_parts = [s.shard_part for s in shards]
        assert shard_parts == [1, 2]
    
    def test_shard_index_parsing(self, sample_model_spec, temp_model_dir):
        """Test parsing of shard index files."""
        handler = WAN22ModelHandler(sample_model_spec, temp_model_dir)
        
        # Create a mock index file
        index_file = temp_model_dir / "index.json"
        index_data = {
            "metadata": {"total_size": 25769803776},
            "weight_map": {
                "layer1.weight": "diffusion_pytorch_model-00001-of-00003.safetensors",
                "layer2.weight": "diffusion_pytorch_model-00002-of-00003.safetensors",
                "layer3.weight": "diffusion_pytorch_model-00003-of-00003.safetensors"
            }
        }
        
        with open(index_file, 'w') as f:
            json.dump(index_data, f)
        
        parsed = handler.parse_shard_index(index_file)
        assert "weight_map" in parsed
        assert len(parsed["weight_map"]) == 3
    
    def test_input_validation_t2v(self, sample_model_spec, temp_model_dir):
        """Test input validation for T2V models."""
        handler = WAN22ModelHandler(sample_model_spec, temp_model_dir)
        
        # Valid input
        valid_inputs = {"prompt": "A beautiful sunset"}
        handler.validate_input_for_model_type(valid_inputs)  # Should not raise
        
        # Invalid: no prompt
        with pytest.raises(InvalidInputError, match="require a text prompt"):
            handler.validate_input_for_model_type({})
        
        # Invalid: has image
        with pytest.raises(InvalidInputError, match="do not accept image input"):
            handler.validate_input_for_model_type({
                "prompt": "test",
                "image": Image.new("RGB", (512, 512))
            })
    
    def test_input_validation_i2v(self, temp_model_dir):
        """Test input validation for I2V models."""
        # Create I2V model spec
        i2v_spec = ModelSpec(
            model_id="i2v-A14B@2.2.0",
            version="2.2.0",
            variants=["fp16"],
            default_variant="fp16",
            files=[],
            sources=[],
            allow_patterns=[],
            resolution_caps=[],
            optional_components=["text_encoder"],
            lora_required=False,
            model_type="i2v"
        )
        
        handler = WAN22ModelHandler(i2v_spec, temp_model_dir)
        
        # Valid input
        valid_inputs = {"image": Image.new("RGB", (512, 512))}
        handler.validate_input_for_model_type(valid_inputs)  # Should not raise
        
        # Valid with text
        valid_inputs_with_text = {
            "image": Image.new("RGB", (512, 512)),
            "prompt": "test"
        }
        handler.validate_input_for_model_type(valid_inputs_with_text)  # Should not raise
        
        # Invalid: no image
        with pytest.raises(InvalidInputError, match="require an image input"):
            handler.validate_input_for_model_type({"prompt": "test"})
    
    def test_input_validation_ti2v(self, temp_model_dir):
        """Test input validation for TI2V models."""
        # Create TI2V model spec
        ti2v_spec = ModelSpec(
            model_id="ti2v-5b@2.2.0",
            version="2.2.0",
            variants=["fp16"],
            default_variant="fp16",
            files=[],
            sources=[],
            allow_patterns=[],
            resolution_caps=[],
            optional_components=[],
            lora_required=False,
            model_type="ti2v"
        )
        
        handler = WAN22ModelHandler(ti2v_spec, temp_model_dir)
        
        # Valid input
        valid_inputs = {
            "prompt": "A beautiful scene",
            "image": Image.new("RGB", (512, 512))
        }
        handler.validate_input_for_model_type(valid_inputs)  # Should not raise
        
        # Invalid: no prompt
        with pytest.raises(InvalidInputError, match="require a text prompt"):
            handler.validate_input_for_model_type({"image": Image.new("RGB", (512, 512))})
        
        # Invalid: no image
        with pytest.raises(InvalidInputError, match="require an image input"):
            handler.validate_input_for_model_type({"prompt": "test"})
    
    def test_image_preprocessing(self, sample_model_spec, temp_model_dir):
        """Test image preprocessing for I2V/TI2V models."""
        handler = WAN22ModelHandler(sample_model_spec, temp_model_dir)
        
        # Test PIL Image input
        pil_image = Image.new("RGB", (512, 512), color="red")
        processed = handler.preprocess_image_input(pil_image)
        
        assert isinstance(processed, torch.Tensor)
        assert processed.shape == (1, 3, 512, 512)  # BCHW format
        assert processed.min() >= -1.0 and processed.max() <= 1.0  # Normalized range
        
        # Test numpy array input
        np_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        processed = handler.preprocess_image_input(np_image)
        
        assert isinstance(processed, torch.Tensor)
        assert processed.shape == (1, 3, 512, 512)
        
        # Test tensor input
        tensor_image = torch.rand(3, 512, 512)  # CHW format
        processed = handler.preprocess_image_input(tensor_image)
        
        assert isinstance(processed, torch.Tensor)
        assert processed.shape == (1, 3, 512, 512)
        
        # Test resizing
        processed_resized = handler.preprocess_image_input(pil_image, target_size=(256, 256))
        assert processed_resized.shape == (1, 3, 256, 256)
    
    def test_text_embedding_cache(self, sample_model_spec, temp_model_dir):
        """Test text embedding caching functionality."""
        handler = WAN22ModelHandler(sample_model_spec, temp_model_dir)
        
        # Test cache miss
        assert handler.get_cached_text_embedding("test prompt") is None
        
        # Test caching
        embedding = torch.randn(1, 512)
        handler.cache_text_embedding("test prompt", embedding)
        
        cached = handler.get_cached_text_embedding("test prompt")
        assert cached is not None
        assert torch.equal(cached, embedding)
        
        # Test cache clearing
        handler.clear_text_embedding_cache()
        assert handler.get_cached_text_embedding("test prompt") is None
    
    def test_vae_config_generation(self, sample_model_spec, temp_model_dir):
        """Test VAE configuration generation."""
        handler = WAN22ModelHandler(sample_model_spec, temp_model_dir)
        
        # Test default config
        config = handler.get_vae_config()
        assert config["tile_size"] == 512
        assert config["decode_chunk_size"] == 8
        assert config["enable_tiling"] is True
        assert config["enable_slicing"] is True
        
        # Test fp16 variant
        config_fp16 = handler.get_vae_config("fp16")
        assert config_fp16["dtype"] == torch.float16
        
        # Test development variant
        config_dev = handler.get_vae_config("fp16-dev")
        assert config_dev["tile_size"] <= 256
        assert config_dev["decode_chunk_size"] <= 4
    
    def test_memory_estimation(self, sample_model_spec, temp_model_dir):
        """Test memory usage estimation."""
        handler = WAN22ModelHandler(sample_model_spec, temp_model_dir)
        
        inputs = {
            "frames": 16,
            "width": 1024,
            "height": 576
        }
        
        # Test base estimation
        memory = handler.estimate_memory_usage(inputs)
        assert "total" in memory
        assert "base" in memory
        assert "dynamic" in memory
        assert memory["total"] > memory["base"]
        
        # Test fp16 variant (should use less memory)
        memory_fp16 = handler.estimate_memory_usage(inputs, "fp16")
        assert memory_fp16["variant_multiplier"] == 0.5
        assert memory_fp16["base"] < memory["base"]
    
    def test_variant_conversion(self, sample_model_spec, temp_model_dir):
        """Test development/production variant conversion."""
        handler = WAN22ModelHandler(sample_model_spec, temp_model_dir)
        
        # Test development variant generation
        dev_variant = handler.get_development_variant("fp16")
        assert dev_variant == "fp16-dev"
        
        # Test production variant extraction
        prod_variant = handler.get_production_variant("fp16-dev")
        assert prod_variant == "fp16"
        
        # Test variant detection
        assert handler.is_development_variant("fp16-dev") is True
        assert handler.is_development_variant("fp16") is False
    
    def test_component_validation(self, sample_model_spec, temp_model_dir):
        """Test component completeness validation."""
        handler = WAN22ModelHandler(sample_model_spec, temp_model_dir)
        
        # Test missing component
        is_complete, missing = handler.validate_component_completeness("unet")
        assert is_complete is False
        assert len(missing) > 0
        
        # Create some files
        unet_dir = temp_model_dir / "unet"
        unet_dir.mkdir(parents=True)
        
        # Create index file
        index_file = unet_dir / "diffusion_pytorch_model.safetensors.index.json"
        with open(index_file, 'w') as f:
            json.dump({"weight_map": {}}, f)
        
        # Create shard files
        for i in range(1, 4):
            shard_file = unet_dir / f"diffusion_pytorch_model-{i:05d}-of-00003.safetensors"
            shard_file.write_bytes(b"fake shard data")
        
        # Test complete component
        is_complete, missing = handler.validate_component_completeness("unet")
        assert is_complete is True
        assert len(missing) == 0
    
    def test_component_file_retrieval(self, sample_model_spec, temp_model_dir):
        """Test component file retrieval."""
        handler = WAN22ModelHandler(sample_model_spec, temp_model_dir)
        
        # Test getting UNet files
        unet_files = handler.get_component_files("unet")
        assert len(unet_files) > 0
        
        # Check that all returned files are UNet files
        for file_spec in unet_files:
            assert file_spec.component == "unet"
        
        # Test non-existent component
        missing_files = handler.get_component_files("nonexistent")
        assert len(missing_files) == 0


class TestWAN22Integration:
    """Integration tests for WAN2.2 model handling."""
    
    def test_full_model_workflow(self, tmp_path):
        """Test complete model workflow with WAN2.2 handler."""
        # Create a realistic model spec
        model_spec = ModelSpec(
            model_id="ti2v-5b@2.2.0",
            version="2.2.0",
            variants=["fp16", "bf16", "fp16-dev"],
            default_variant="fp16",
            files=[
                FileSpec(
                    path="config.json",
                    size=1024,
                    sha256="a" * 64,
                    component="config"
                ),
                FileSpec(
                    path="diffusion_pytorch_model.safetensors.index.json",
                    size=4096,
                    sha256="b" * 64,
                    component="unet",
                    shard_index=True
                ),
                FileSpec(
                    path="diffusion_pytorch_model-00001-of-00003.safetensors",
                    size=17179869184,
                    sha256="c" * 64,
                    component="unet",
                    shard_part=1
                ),
                FileSpec(
                    path="diffusion_pytorch_model-00002-of-00003.safetensors",
                    size=17179869184,
                    sha256="d" * 64,
                    component="unet",
                    shard_part=2
                ),
                FileSpec(
                    path="diffusion_pytorch_model-00003-of-00003.safetensors",
                    size=17179869184,
                    sha256="e" * 64,
                    component="unet",
                    shard_part=3
                ),
            ],
            sources=["local://test"],
            allow_patterns=["*.safetensors", "*.json"],
            resolution_caps=["720p24", "1080p24", "1440p24"],
            optional_components=[],
            lora_required=False,
            description="Test TI2V model",
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
        
        handler = WAN22ModelHandler(model_spec, tmp_path)
        
        # Test input validation
        valid_inputs = {
            "prompt": "A beautiful landscape",
            "image": Image.new("RGB", (1024, 576)),
            "frames": 24,
            "width": 1024,
            "height": 576
        }
        
        handler.validate_input_for_model_type(valid_inputs)
        
        # Test image preprocessing
        processed_image = handler.preprocess_image_input(valid_inputs["image"])
        assert processed_image.shape == (1, 3, 576, 1024)
        
        # Test memory estimation
        memory_est = handler.estimate_memory_usage(valid_inputs, "fp16")
        assert memory_est["total"] > 0
        
        # Test VAE config
        vae_config = handler.get_vae_config("fp16")
        assert vae_config["dtype"] == torch.float16
        
        # Test shard requirements
        prod_shards = handler.get_required_shards("unet", "fp16")
        dev_shards = handler.get_required_shards("unet", "fp16-dev")
        
        assert len(prod_shards) == 3  # All shards for production
        assert len(dev_shards) == 2   # Fewer shards for development
        
        # Test text embedding caching
        test_embedding = torch.randn(1, 768)
        handler.cache_text_embedding("test prompt", test_embedding)
        cached = handler.get_cached_text_embedding("test prompt")
        assert torch.equal(cached, test_embedding)


if __name__ == "__main__":
    pytest.main([__file__])