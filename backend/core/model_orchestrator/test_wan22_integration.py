"""
Integration tests for WAN2.2 model handling with the orchestrator.

Tests the complete integration between WAN2.2 handler and model orchestrator,
including selective downloading, variant switching, and component validation.
"""

import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np
from PIL import Image

from .model_ensurer import ModelEnsurer, ModelStatus
from .model_registry import ModelRegistry, ModelSpec, FileSpec, ModelDefaults, VramEstimation
from .model_resolver import ModelResolver
from .lock_manager import LockManager
from .storage_backends.base_store import StorageBackend, DownloadResult
from .wan22_handler import WAN22ModelHandler
from .exceptions import InvalidInputError, ModelValidationError


class MockStorageBackend(StorageBackend):
    """Mock storage backend for testing."""
    
    def __init__(self):
        self.downloaded_files = []
        self.should_fail = False
        
    def can_handle(self, source_url: str) -> bool:
        return source_url.startswith("mock://")
    
    def download(self, source_url: str, local_dir: Path, file_specs: list = None, 
                allow_patterns=None, progress_callback=None, **kwargs) -> DownloadResult:
        if self.should_fail:
            return DownloadResult(success=False, error_message="Mock failure")
        
        # Record which files were requested for download
        self.downloaded_files = [f.path for f in file_specs]
        
        # Create mock files
        for file_spec in file_specs:
            file_path = local_dir / file_spec.path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create mock file content
            if file_spec.path.endswith('.json'):
                if 'index.json' in file_spec.path:
                    # Create mock shard index
                    index_data = {
                        "metadata": {"total_size": file_spec.size},
                        "weight_map": {
                            f"layer_{i}.weight": f"diffusion_pytorch_model-{i:05d}-of-00003.safetensors"
                            for i in range(1, 4)
                        }
                    }
                    with open(file_path, 'w') as f:
                        json.dump(index_data, f)
                else:
                    # Create mock config
                    with open(file_path, 'w') as f:
                        json.dump({"model_type": "diffusion"}, f)
            else:
                # Create mock binary file
                file_path.write_bytes(b"mock_data" * (file_spec.size // 9 + 1)[:file_spec.size])
        
        return DownloadResult(success=True, metadata={"source": source_url})
    
    def verify_availability(self, source_url: str) -> bool:
        return True
    
    def estimate_download_size(self, source_url: str, file_specs: list) -> int:
        return sum(f.size for f in file_specs)


class TestWAN22Integration:
    """Test WAN2.2 integration with model orchestrator."""
    
    @pytest.fixture
    def temp_models_root(self):
        """Create temporary models root directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def sample_manifest(self, temp_models_root):
        """Create sample manifest file."""
        manifest_path = temp_models_root / "models.toml"
        manifest_content = '''
schema_version = 1

[models."t2v-A14B@2.2.0"]
description = "WAN2.2 Text-to-Video A14B Model"
version = "2.2.0"
variants = ["fp16", "bf16", "fp16-dev"]
default_variant = "fp16"
resolution_caps = ["720p24", "1080p24"]
optional_components = []
lora_required = false
allow_patterns = ["*.safetensors", "*.json", "*.pth"]
required_components = ["text_encoder", "unet", "vae"]

[models."t2v-A14B@2.2.0".defaults]
fps = 24
frames = 16
scheduler = "ddim"
precision = "fp16"
guidance_scale = 7.5
vae_tile_size = 512
vae_decode_chunk_size = 8

[models."t2v-A14B@2.2.0".vram_estimation]
params_billion = 14.0
family_size = "large"
base_vram_gb = 12.0
per_frame_vram_mb = 256.0

[[models."t2v-A14B@2.2.0".files]]
path = "config.json"
size = 1024
sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
component = "config"

[[models."t2v-A14B@2.2.0".files]]
path = "unet/diffusion_pytorch_model.safetensors.index.json"
size = 4096
sha256 = "2c624232cdd221771294dfbb310aca000a0df6ac8b66b696b90ef72c9e9c323c"
component = "unet"
shard_index = true

[[models."t2v-A14B@2.2.0".files]]
path = "unet/diffusion_pytorch_model-00001-of-00003.safetensors"
size = 8589934592
sha256 = "d4735e3a265e16eee03f59718b9b5d03019c07d8b6c51f90da3a666eec13ab35"
component = "unet"
shard_part = 1

[[models."t2v-A14B@2.2.0".files]]
path = "unet/diffusion_pytorch_model-00002-of-00003.safetensors"
size = 8589934592
sha256 = "4e07408562bedb8b60ce05c1decfe3ad16b72230967de01f640b7e4729b49fce"
component = "unet"
shard_part = 2

[[models."t2v-A14B@2.2.0".files]]
path = "unet/diffusion_pytorch_model-00003-of-00003.safetensors"
size = 4294967296
sha256 = "4b227777d4dd1fc61c6f884f48641d02b4d121d3fd328cb08b5531fcacdabf8a"
component = "unet"
shard_part = 3

[[models."t2v-A14B@2.2.0".files]]
path = "text_encoder/pytorch_model.bin"
size = 4294967296
sha256 = "ef2d127de37b942baad06145e54b0c619a1f22327b2ebbcfbec78f5564afe39d"
component = "text_encoder"

[[models."t2v-A14B@2.2.0".files]]
path = "vae/diffusion_pytorch_model.safetensors"
size = 335544320
sha256 = "e7f6c011776e8db7cd330b54174fd76f7d0216b612387a5ffcfb81e6f0919683"
component = "vae"

[models."t2v-A14B@2.2.0".sources]
priority = ["mock://t2v-A14B@2.2.0"]

[models."i2v-A14B@2.2.0"]
description = "WAN2.2 Image-to-Video A14B Model"
version = "2.2.0"
variants = ["fp16", "bf16", "fp16-dev"]
default_variant = "fp16"
resolution_caps = ["720p24", "1080p24"]
optional_components = ["text_encoder"]
lora_required = false
allow_patterns = ["*.safetensors", "*.json", "*.pth"]
required_components = ["image_encoder", "unet", "vae"]

[models."i2v-A14B@2.2.0".defaults]
fps = 24
frames = 16
scheduler = "ddim"
precision = "fp16"
guidance_scale = 5.0
image_guidance_scale = 1.5
vae_tile_size = 512
vae_decode_chunk_size = 8

[models."i2v-A14B@2.2.0".vram_estimation]
params_billion = 14.0
family_size = "large"
base_vram_gb = 14.0
per_frame_vram_mb = 320.0

[[models."i2v-A14B@2.2.0".files]]
path = "config.json"
size = 1024
sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
component = "config"

[[models."i2v-A14B@2.2.0".files]]
path = "image_encoder/pytorch_model.bin"
size = 1073741824
sha256 = "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b"
component = "image_encoder"

[[models."i2v-A14B@2.2.0".files]]
path = "unet/diffusion_pytorch_model-00001-of-00002.safetensors"
size = 8589934592
sha256 = "d4735e3a265e16eee03f59718b9b5d03019c07d8b6c51f90da3a666eec13ab35"
component = "unet"
shard_part = 1

[[models."i2v-A14B@2.2.0".files]]
path = "unet/diffusion_pytorch_model-00002-of-00002.safetensors"
size = 8589934592
sha256 = "4e07408562bedb8b60ce05c1decfe3ad16b72230967de01f640b7e4729b49fce"
component = "unet"
shard_part = 2

[[models."i2v-A14B@2.2.0".files]]
path = "vae/diffusion_pytorch_model.safetensors"
size = 335544320
sha256 = "e7f6c011776e8db7cd330b54174fd76f7d0216b612387a5ffcfb81e6f0919683"
component = "vae"

[models."i2v-A14B@2.2.0".sources]
priority = ["mock://i2v-A14B@2.2.0"]
'''
        
        with open(manifest_path, 'w') as f:
            f.write(manifest_content)
        
        return manifest_path
    
    @pytest.fixture
    def model_ensurer(self, temp_models_root, sample_manifest):
        """Create model ensurer with WAN2.2 support."""
        registry = ModelRegistry(str(sample_manifest))
        resolver = ModelResolver(str(temp_models_root))
        lock_manager = LockManager(str(temp_models_root / ".locks"))
        storage_backend = MockStorageBackend()
        
        ensurer = ModelEnsurer(
            registry=registry,
            resolver=resolver,
            lock_manager=lock_manager,
            storage_backends=[storage_backend],
            enable_deduplication=False  # Disable for simpler testing
        )
        
        return ensurer, storage_backend
    
    def test_wan22_handler_creation(self, model_ensurer):
        """Test WAN2.2 handler creation for different model types."""
        ensurer, _ = model_ensurer
        
        # Test T2V model
        t2v_spec = ensurer.registry.spec("t2v-A14B@2.2.0")
        handler = ensurer.get_wan22_handler(t2v_spec, Path("/tmp"))
        
        assert handler is not None
        assert handler.model_spec.model_type == "t2v"
        assert "unet" in handler.components
        assert "text_encoder" in handler.components
        assert "vae" in handler.components
        
        # Test I2V model
        i2v_spec = ensurer.registry.spec("i2v-A14B@2.2.0")
        handler = ensurer.get_wan22_handler(i2v_spec, Path("/tmp"))
        
        assert handler is not None
        assert handler.model_spec.model_type == "i2v"
        assert "image_encoder" in handler.components
    
    def test_selective_file_download_production(self, model_ensurer):
        """Test selective file download for production variants."""
        ensurer, _ = model_ensurer
        
        spec = ensurer.registry.spec("t2v-A14B@2.2.0")
        selective_files = ensurer.get_selective_files_for_variant(spec, "fp16")
        
        # For production variant, should get all files
        assert len(selective_files) == len(spec.files)
        
        # Check that all files are included
        file_paths = {f.path for f in selective_files}
        expected_paths = {f.path for f in spec.files}
        assert file_paths == expected_paths
    
    def test_selective_file_download_development(self, model_ensurer):
        """Test selective file download for development variants."""
        ensurer, _ = model_ensurer
        
        spec = ensurer.registry.spec("t2v-A14B@2.2.0")
        selective_files = ensurer.get_selective_files_for_variant(spec, "fp16-dev")
        
        # For development variant, should get fewer files
        assert len(selective_files) < len(spec.files)
        
        # Should include all non-shard files
        non_shard_files = [f for f in spec.files if f.shard_part is None]
        selective_non_shard = [f for f in selective_files if f.shard_part is None]
        assert len(selective_non_shard) == len(non_shard_files)
        
        # Should include only first 2 UNet shards for development
        unet_shards = [f for f in selective_files if f.component == "unet" and f.shard_part is not None]
        assert len(unet_shards) == 2
        assert all(f.shard_part <= 2 for f in unet_shards)
    
    @patch('backend.core.model_orchestrator.model_ensurer.get_logger')
    def test_full_download_workflow_production(self, mock_logger, model_ensurer, temp_models_root):
        """Test complete download workflow for production variant."""
        ensurer, storage_backend = model_ensurer
        
        # Ensure T2V model with production variant
        model_path = ensurer.ensure("t2v-A14B@2.2.0", variant="fp16")
        
        # Verify model was downloaded
        assert Path(model_path).exists()
        
        # Check that all files were downloaded
        expected_files = ensurer.registry.spec("t2v-A14B@2.2.0").files
        assert len(storage_backend.downloaded_files) == len(expected_files)
        
        # Verify WAN2.2 handler was created
        spec = ensurer.registry.spec("t2v-A14B@2.2.0")
        handler = ensurer.get_wan22_handler(spec, Path(model_path))
        assert handler is not None
    
    @patch('backend.core.model_orchestrator.model_ensurer.get_logger')
    def test_full_download_workflow_development(self, mock_logger, model_ensurer, temp_models_root):
        """Test complete download workflow for development variant."""
        ensurer, storage_backend = model_ensurer
        
        # Ensure T2V model with development variant
        model_path = ensurer.ensure("t2v-A14B@2.2.0", variant="fp16-dev")
        
        # Verify model was downloaded
        assert Path(model_path).exists()
        
        # Check that fewer files were downloaded for development
        all_files = ensurer.registry.spec("t2v-A14B@2.2.0").files
        assert len(storage_backend.downloaded_files) < len(all_files)
        
        # Verify that non-shard files were downloaded
        non_shard_count = len([f for f in all_files if f.shard_part is None])
        downloaded_non_shard = len([f for f in storage_backend.downloaded_files 
                                  if not any(str(i) in f for i in range(1, 10))])
        assert downloaded_non_shard >= non_shard_count
    
    def test_model_status_with_wan22(self, model_ensurer):
        """Test model status checking with WAN2.2 models."""
        ensurer, _ = model_ensurer
        
        # Check status of non-existent model
        status = ensurer.status("t2v-A14B@2.2.0")
        assert status == ModelStatus.NOT_PRESENT
        
        # Download model
        model_path = ensurer.ensure("t2v-A14B@2.2.0", variant="fp16")
        
        # Check status after download
        status = ensurer.status("t2v-A14B@2.2.0")
        assert status == ModelStatus.COMPLETE
    
    def test_input_validation_integration(self, model_ensurer, temp_models_root):
        """Test input validation integration with model ensurer."""
        ensurer, _ = model_ensurer
        
        # Ensure models are available
        t2v_path = ensurer.ensure("t2v-A14B@2.2.0")
        i2v_path = ensurer.ensure("i2v-A14B@2.2.0")
        
        # Get handlers
        t2v_spec = ensurer.registry.spec("t2v-A14B@2.2.0")
        i2v_spec = ensurer.registry.spec("i2v-A14B@2.2.0")
        
        t2v_handler = ensurer.get_wan22_handler(t2v_spec, Path(t2v_path))
        i2v_handler = ensurer.get_wan22_handler(i2v_spec, Path(i2v_path))
        
        # Test T2V validation
        assert t2v_handler is not None
        t2v_handler.validate_input_for_model_type({"prompt": "test"})
        
        with pytest.raises(InvalidInputError):
            t2v_handler.validate_input_for_model_type({"image": Image.new("RGB", (512, 512))})
        
        # Test I2V validation
        assert i2v_handler is not None
        i2v_handler.validate_input_for_model_type({"image": Image.new("RGB", (512, 512))})
        
        with pytest.raises(InvalidInputError):
            i2v_handler.validate_input_for_model_type({"prompt": "test"})
    
    def test_vae_config_integration(self, model_ensurer):
        """Test VAE configuration integration."""
        ensurer, _ = model_ensurer
        
        # Ensure model
        model_path = ensurer.ensure("t2v-A14B@2.2.0")
        
        # Get handler and test VAE config
        spec = ensurer.registry.spec("t2v-A14B@2.2.0")
        handler = ensurer.get_wan22_handler(spec, Path(model_path))
        
        assert handler is not None
        
        # Test different variant configs
        fp16_config = handler.get_vae_config("fp16")
        assert fp16_config["dtype"] == torch.float16
        assert fp16_config["tile_size"] == 512
        
        dev_config = handler.get_vae_config("fp16-dev")
        assert dev_config["tile_size"] <= 256  # More aggressive for dev
    
    def test_memory_estimation_integration(self, model_ensurer):
        """Test memory estimation integration."""
        ensurer, _ = model_ensurer
        
        # Ensure model
        model_path = ensurer.ensure("t2v-A14B@2.2.0")
        
        # Get handler and test memory estimation
        spec = ensurer.registry.spec("t2v-A14B@2.2.0")
        handler = ensurer.get_wan22_handler(spec, Path(model_path))
        
        assert handler is not None
        
        inputs = {"frames": 16, "width": 1024, "height": 576}
        memory_est = handler.estimate_memory_usage(inputs, "fp16")
        
        assert memory_est["total"] > 0
        assert memory_est["base"] > 0
        assert memory_est["dynamic"] >= 0
        assert memory_est["variant_multiplier"] == 0.5  # fp16 uses half memory
    
    def test_text_embedding_cache_integration(self, model_ensurer):
        """Test text embedding cache integration."""
        ensurer, _ = model_ensurer
        
        # Ensure model
        model_path = ensurer.ensure("t2v-A14B@2.2.0")
        
        # Get handler and test caching
        spec = ensurer.registry.spec("t2v-A14B@2.2.0")
        handler = ensurer.get_wan22_handler(spec, Path(model_path))
        
        assert handler is not None
        
        # Test caching
        test_embedding = torch.randn(1, 768)
        handler.cache_text_embedding("test prompt", test_embedding)
        
        cached = handler.get_cached_text_embedding("test prompt")
        assert cached is not None
        assert torch.equal(cached, test_embedding)
        
        # Test cache clearing
        handler.clear_text_embedding_cache()
        assert handler.get_cached_text_embedding("test prompt") is None
    
    def test_component_validation_integration(self, model_ensurer):
        """Test component validation integration."""
        ensurer, _ = model_ensurer
        
        # Ensure model
        model_path = ensurer.ensure("t2v-A14B@2.2.0")
        
        # Get handler and test component validation
        spec = ensurer.registry.spec("t2v-A14B@2.2.0")
        handler = ensurer.get_wan22_handler(spec, Path(model_path))
        
        assert handler is not None
        
        # Test component completeness
        for component_type in ["config", "unet", "text_encoder", "vae"]:
            is_complete, missing = handler.validate_component_completeness(component_type)
            assert is_complete, f"Component {component_type} should be complete, missing: {missing}"
    
    def test_variant_conversion_integration(self, model_ensurer):
        """Test variant conversion integration."""
        ensurer, _ = model_ensurer
        
        # Ensure model
        model_path = ensurer.ensure("t2v-A14B@2.2.0")
        
        # Get handler and test variant conversion
        spec = ensurer.registry.spec("t2v-A14B@2.2.0")
        handler = ensurer.get_wan22_handler(spec, Path(model_path))
        
        assert handler is not None
        
        # Test variant conversions
        assert handler.get_development_variant("fp16") == "fp16-dev"
        assert handler.get_production_variant("fp16-dev") == "fp16"
        assert handler.is_development_variant("fp16-dev") is True
        assert handler.is_development_variant("fp16") is False
    
    def test_error_handling_integration(self, model_ensurer):
        """Test error handling integration with WAN2.2 models."""
        ensurer, storage_backend = model_ensurer
        
        # Test download failure
        storage_backend.should_fail = True
        
        with pytest.raises(Exception):  # Should raise some orchestrator error
            ensurer.ensure("t2v-A14B@2.2.0")
        
        # Reset and test successful download
        storage_backend.should_fail = False
        model_path = ensurer.ensure("t2v-A14B@2.2.0")
        assert Path(model_path).exists()


if __name__ == "__main__":
    pytest.main([__file__])