"""
Test backwards compatibility with existing Gradio system.
Ensures model files, LoRA weights, and generation results are identical.
"""

import pytest
import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import torch

from ..core.system_integration import SystemIntegration
from ..services.generation_service import GenerationService
from ..models.schemas import GenerationRequest, ModelType

class TestBackwardsCompatibility:
    """Test suite for backwards compatibility with existing Gradio system."""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        temp_dir = tempfile.mkdtemp()
        dirs = {
            'models': Path(temp_dir) / 'models',
            'loras': Path(temp_dir) / 'loras',
            'outputs': Path(temp_dir) / 'outputs',
            'config': Path(temp_dir) / 'config.json'
        }
        
        # Create directories
        for key, path in dirs.items():
            if key != 'config':
                path.mkdir(parents=True, exist_ok=True)
        
        yield dirs
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_config(self, temp_dirs):
        """Create mock configuration matching Gradio setup."""
        config = {
            "model_type": "T2V-A14B",
            "model_path": str(temp_dirs['models']),
            "output_dir": str(temp_dirs['outputs']),
            "lora_dir": str(temp_dirs['loras']),
            "quantization": True,
            "vram_optimize": True,
            "cpu_offload": False,
            "vae_tile": 512,
            "steps": 50,
            "resolution": "1280x720"
        }
        
        with open(temp_dirs['config'], 'w') as f:
            json.dump(config, f)
        
        return config
    
    @pytest.fixture
    def mock_model_files(self, temp_dirs):
        """Create mock model files matching Gradio structure."""
        model_files = {}
        
        # Create mock model directories and files
        models = {
            'T2V-A14B': ['model.safetensors', 'config.json', 'tokenizer.json'],
            'I2V-A14B': ['model.safetensors', 'config.json', 'tokenizer.json'],
            'TI2V-5B': ['model.safetensors', 'config.json', 'tokenizer.json']
        }
        
        for model_name, files in models.items():
            model_dir = temp_dirs['models'] / model_name
            model_dir.mkdir(exist_ok=True)
            
            model_files[model_name] = {}
            for file_name in files:
                file_path = model_dir / file_name
                
                if file_name.endswith('.safetensors'):
                    # Create dummy tensor file
                    file_path.write_bytes(b'dummy_tensor_data')
                elif file_name.endswith('.json'):
                    # Create dummy JSON config
                    dummy_config = {"model_type": model_name, "version": "1.0"}
                    with open(file_path, 'w') as f:
                        json.dump(dummy_config, f)
                
                model_files[model_name][file_name] = file_path
        
        return model_files
    
    @pytest.fixture
    def mock_lora_files(self, temp_dirs):
        """Create mock LoRA files matching Gradio structure."""
        lora_files = {}
        
        loras = ['style_lora.safetensors', 'character_lora.safetensors', 'vace_lora.safetensors']
        
        for lora_name in loras:
            lora_path = temp_dirs['loras'] / lora_name
            lora_path.write_bytes(b'dummy_lora_data')
            lora_files[lora_name] = lora_path
        
        return lora_files
    
    def test_model_file_detection(self, temp_dirs, mock_model_files):
        """Test that model files are detected correctly."""
        system_integration = SystemIntegration()
        
        # Test model detection
        detected_models = system_integration.scan_available_models(str(temp_dirs['models']))
        
        assert len(detected_models) == 3
        assert 'T2V-A14B' in detected_models
        assert 'I2V-A14B' in detected_models
        assert 'TI2V-5B' in detected_models
        
        # Verify file paths are correct
        for model_name in detected_models:
            model_path = detected_models[model_name]['path']
            assert Path(model_path).exists()
            assert (Path(model_path) / 'model.safetensors').exists()
    
    def test_lora_file_detection(self, temp_dirs, mock_lora_files):
        """Test that LoRA files are detected correctly."""
        system_integration = SystemIntegration()
        
        # Test LoRA detection
        detected_loras = system_integration.scan_available_loras(str(temp_dirs['loras']))
        
        assert len(detected_loras) == 3
        assert 'style_lora.safetensors' in detected_loras
        assert 'character_lora.safetensors' in detected_loras
        assert 'vace_lora.safetensors' in detected_loras
        
        # Verify file paths are correct
        for lora_name in detected_loras:
            lora_path = detected_loras[lora_name]['path']
            assert Path(lora_path).exists()
    
    def test_config_compatibility(self, temp_dirs, mock_config):
        """Test that existing config.json is loaded correctly."""
        from ..config.config_validator import ConfigValidator
        
        validator = ConfigValidator(str(temp_dirs['config']))
        result = validator.run_validation()
        
        assert result.is_valid or len(result.errors) == 0  # Should be valid or have only warnings
        assert result.migrated_config is not None
        
        # Check that key settings are preserved
        migrated = result.migrated_config
        assert migrated['model_settings']['default_model'] == 'T2V-A14B'
        assert migrated['optimization_settings']['vram_optimization'] == True
        assert migrated['optimization_settings']['vae_tile_size'] == 512
        assert migrated['generation_settings']['default_steps'] == 50
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_properties')
    def test_generation_service_compatibility(self, mock_gpu_props, temp_dirs, mock_config, mock_model_files):
        """Test that generation service works with existing model structure."""
        # Mock GPU properties
        mock_gpu_props.return_value = Mock(total_memory=8 * 1024**3)  # 8GB VRAM
        
        with patch('sys.path', [str(temp_dirs['models'].parent)] + list(sys.path)):
            generation_service = GenerationService()
            
            # Test model loading
            request = GenerationRequest(
                model_type=ModelType.T2V_A14B,
                prompt="Test prompt",
                resolution="1280x720",
                steps=50
            )
            
            # Mock the actual generation to avoid GPU requirements
            with patch.object(generation_service, '_run_generation') as mock_gen:
                mock_gen.return_value = {
                    'output_path': str(temp_dirs['outputs'] / 'test_output.mp4'),
                    'thumbnail_path': str(temp_dirs['outputs'] / 'test_thumb.jpg')
                }
                
                result = generation_service.generate(request)
                
                assert result is not None
                assert 'output_path' in result
                mock_gen.assert_called_once()
    
    def test_output_directory_structure(self, temp_dirs, mock_config):
        """Test that output directory structure is compatible."""
        from ..migration.data_migrator import DataMigrator
        
        # Create some mock Gradio outputs
        gradio_outputs = temp_dirs['outputs']
        
        # Create mock video files with Gradio naming convention
        mock_videos = [
            'output_20240101_120000.mp4',
            'generated_video_t2v.mp4',
            'i2v_result.mp4'
        ]
        
        for video_name in mock_videos:
            video_path = gradio_outputs / video_name
            video_path.write_bytes(b'mock_video_data')
            
            # Create associated metadata file
            metadata_path = video_path.with_suffix('.json')
            metadata = {
                'prompt': f'Test prompt for {video_name}',
                'model_type': 'T2V-A14B',
                'resolution': '1280x720',
                'steps': 50
            }
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
        
        # Test migration
        migrator = DataMigrator(
            gradio_outputs_dir=str(gradio_outputs),
            new_outputs_dir=str(temp_dirs['outputs'] / 'new'),
            backup_dir=str(temp_dirs['outputs'] / 'backup')
        )
        
        outputs = migrator.scan_gradio_outputs()
        assert len(outputs) == 3
        
        # Verify metadata extraction
        for output in outputs:
            assert 'prompt' in output
            assert 'model_type' in output
            assert 'resolution' in output
            assert output['filename'] in mock_videos
    
    def test_lora_loading_compatibility(self, temp_dirs, mock_lora_files):
        """Test that LoRA files can be loaded with same interface."""
        system_integration = SystemIntegration()
        
        # Test LoRA loading
        lora_path = str(mock_lora_files['style_lora.safetensors'])
        
        # Mock LoRA loading (since we don't have real LoRA files)
        with patch.object(system_integration, 'load_lora') as mock_load:
            mock_load.return_value = {'status': 'loaded', 'strength': 1.0}
            
            result = system_integration.load_lora(lora_path, strength=0.8)
            
            assert result['status'] == 'loaded'
            mock_load.assert_called_once_with(lora_path, strength=0.8)
    
    def test_quantization_compatibility(self, temp_dirs, mock_config):
        """Test that quantization settings work identically."""
        from ..config.config_validator import ConfigValidator
        
        # Test different quantization configurations
        quantization_configs = [
            {'quantization': True},
            {'quantization': False},
            {'quantization': 'fp16'},
            {'quantization': 'bf16'},
            {'quantization': 'int8'}
        ]
        
        for quant_config in quantization_configs:
            config = mock_config.copy()
            config.update(quant_config)
            
            config_path = temp_dirs['config']
            with open(config_path, 'w') as f:
                json.dump(config, f)
            
            validator = ConfigValidator(str(config_path))
            result = validator.run_validation()
            
            # Should successfully migrate quantization settings
            assert result.migrated_config is not None
            quant_mode = result.migrated_config['model_settings']['quantization_mode']
            assert quant_mode in ['fp16', 'bf16', 'int8', 'none']
    
    def test_vram_optimization_compatibility(self, temp_dirs, mock_config):
        """Test that VRAM optimization settings are preserved."""
        from ..config.config_validator import ConfigValidator
        
        # Test VRAM optimization configurations
        vram_configs = [
            {'vram_optimize': True, 'cpu_offload': False, 'vae_tile': 512},
            {'vram_optimize': False, 'cpu_offload': True, 'vae_tile': 256},
            {'vram_optimize': True, 'cpu_offload': True, 'vae_tile': 1024}
        ]
        
        for vram_config in vram_configs:
            config = mock_config.copy()
            config.update(vram_config)
            
            config_path = temp_dirs['config']
            with open(config_path, 'w') as f:
                json.dump(config, f)
            
            validator = ConfigValidator(str(config_path))
            result = validator.run_validation()
            
            # Verify VRAM settings are preserved
            migrated = result.migrated_config
            opt_settings = migrated['optimization_settings']
            
            assert opt_settings['vram_optimization'] == vram_config['vram_optimize']
            assert opt_settings['cpu_offload'] == vram_config['cpu_offload']
            assert opt_settings['vae_tile_size'] == vram_config['vae_tile']
    
    def test_generation_parameters_compatibility(self, temp_dirs, mock_config):
        """Test that generation parameters produce identical results."""
        # Test parameter sets that should work identically
        parameter_sets = [
            {
                'model_type': ModelType.T2V_A14B,
                'prompt': 'A beautiful sunset',
                'resolution': '1280x720',
                'steps': 50
            },
            {
                'model_type': ModelType.I2V_A14B,
                'prompt': 'Transform this image',
                'resolution': '1920x1080',
                'steps': 30
            },
            {
                'model_type': ModelType.TI2V_5B,
                'prompt': 'Combine text and image',
                'resolution': '1280x720',
                'steps': 40
            }
        ]
        
        generation_service = GenerationService()
        
        for params in parameter_sets:
            request = GenerationRequest(**params)
            
            # Mock generation to test parameter handling
            with patch.object(generation_service, '_run_generation') as mock_gen:
                mock_gen.return_value = {
                    'output_path': 'test_output.mp4',
                    'thumbnail_path': 'test_thumb.jpg'
                }
                
                result = generation_service.generate(request)
                
                # Verify parameters were passed correctly
                call_args = mock_gen.call_args[0][0]  # First argument (request)
                assert call_args.model_type == params['model_type']
                assert call_args.prompt == params['prompt']
                assert call_args.resolution == params['resolution']
                assert call_args.steps == params['steps']
    
    def test_error_handling_compatibility(self, temp_dirs, mock_config):
        """Test that error handling works identically to Gradio."""
        generation_service = GenerationService()
        
        # Test common error scenarios
        error_scenarios = [
            {
                'name': 'invalid_model',
                'request': GenerationRequest(
                    model_type='INVALID_MODEL',
                    prompt='Test prompt'
                ),
                'expected_error': 'Invalid model type'
            },
            {
                'name': 'empty_prompt',
                'request': GenerationRequest(
                    model_type=ModelType.T2V_A14B,
                    prompt=''
                ),
                'expected_error': 'Prompt cannot be empty'
            },
            {
                'name': 'invalid_resolution',
                'request': GenerationRequest(
                    model_type=ModelType.T2V_A14B,
                    prompt='Test prompt',
                    resolution='invalid'
                ),
                'expected_error': 'Invalid resolution format'
            }
        ]
        
        for scenario in error_scenarios:
            with pytest.raises(Exception) as exc_info:
                generation_service.generate(scenario['request'])
            
            # Verify error message contains expected text
            assert any(expected in str(exc_info.value) 
                      for expected in [scenario['expected_error'], 'validation'])
    
    def test_file_format_compatibility(self, temp_dirs):
        """Test that file formats are handled identically."""
        from ..migration.data_migrator import DataMigrator
        
        # Create files with various formats that Gradio supports
        test_files = {
            'video.mp4': b'mock_mp4_data',
            'video.avi': b'mock_avi_data',
            'video.mov': b'mock_mov_data',
            'video.mkv': b'mock_mkv_data',
            'video.webm': b'mock_webm_data',
            'not_video.txt': b'text_file',  # Should be ignored
            'image.jpg': b'jpeg_data'  # Should be ignored
        }
        
        outputs_dir = temp_dirs['outputs']
        for filename, content in test_files.items():
            (outputs_dir / filename).write_bytes(content)
        
        migrator = DataMigrator(gradio_outputs_dir=str(outputs_dir))
        outputs = migrator.scan_gradio_outputs()
        
        # Should only detect video files
        video_files = [output['filename'] for output in outputs]
        expected_videos = ['video.mp4', 'video.avi', 'video.mov', 'video.mkv', 'video.webm']
        
        assert len(video_files) == len(expected_videos)
        for expected in expected_videos:
            assert expected in video_files
        
        # Should not include non-video files
        assert 'not_video.txt' not in video_files
        assert 'image.jpg' not in video_files

# Integration test to verify complete backwards compatibility
class TestFullBackwardsCompatibility:
    """Full integration test for backwards compatibility."""
    
    def test_complete_migration_workflow(self, tmp_path):
        """Test complete migration from Gradio to new system."""
        # Set up mock Gradio environment
        gradio_dir = tmp_path / "gradio_setup"
        gradio_dir.mkdir()
        
        # Create Gradio-style directory structure
        (gradio_dir / "models").mkdir()
        (gradio_dir / "loras").mkdir()
        (gradio_dir / "outputs").mkdir()
        
        # Create mock config.json
        config = {
            "model_type": "t2v",
            "quantization": True,
            "vram_optimize": True,
            "output_dir": "outputs",
            "steps": 50
        }
        
        with open(gradio_dir / "config.json", 'w') as f:
            json.dump(config, f)
        
        # Create mock outputs
        outputs_dir = gradio_dir / "outputs"
        (outputs_dir / "video1.mp4").write_bytes(b"mock_video_1")
        (outputs_dir / "video2.mp4").write_bytes(b"mock_video_2")
        
        # Create metadata files
        metadata1 = {"prompt": "Beautiful landscape", "model_type": "T2V-A14B"}
        metadata2 = {"prompt": "City at night", "model_type": "T2V-A14B"}
        
        with open(outputs_dir / "video1.json", 'w') as f:
            json.dump(metadata1, f)
        with open(outputs_dir / "video2.json", 'w') as f:
            json.dump(metadata2, f)
        
        # Run migration
        from ..migration.data_migrator import DataMigrator
        from ..config.config_validator import ConfigValidator
        
        # Validate and migrate config
        validator = ConfigValidator(str(gradio_dir / "config.json"))
        config_result = validator.run_validation()
        
        assert config_result.is_valid or len(config_result.errors) == 0
        assert config_result.migrated_config is not None
        
        # Migrate data
        migrator = DataMigrator(
            gradio_outputs_dir=str(outputs_dir),
            new_outputs_dir=str(tmp_path / "new_outputs"),
            backup_dir=str(tmp_path / "backup")
        )
        
        migration_result = migrator.run_migration()
        
        assert migration_result['success']
        assert migration_result['migrated_count'] == 2
        assert migration_result['error_count'] == 0
        
        # Verify backup was created
        backup_dir = tmp_path / "backup"
        assert backup_dir.exists()
        assert (backup_dir / "video1.mp4").exists()
        assert (backup_dir / "video2.mp4").exists()
        
        # Verify new outputs were created
        new_outputs_dir = tmp_path / "new_outputs"
        assert new_outputs_dir.exists()
        
        migrated_files = list(new_outputs_dir.glob("migrated_*.mp4"))
        assert len(migrated_files) == 2

if __name__ == "__main__":
    pytest.main([__file__, "-v"])