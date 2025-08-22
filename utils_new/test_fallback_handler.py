"""
Tests for Fallback Handler System

This module tests the comprehensive fallback strategies and graceful degradation
for model loading and pipeline initialization failures.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import torch

from fallback_handler import (
    FallbackHandler,
    ComponentAnalyzer,
    AlternativeModelSuggester,
    FallbackStrategy,
    FallbackStrategyType,
    UsableComponent,
    ComponentType,
    AlternativeModel,
    FallbackResult,
    handle_pipeline_failure,
    analyze_model_components,
    find_alternative_models
)


class TestComponentAnalyzer:
    """Test component analysis functionality"""
    
    @pytest.fixture
    def component_analyzer(self):
        return ComponentAnalyzer()
    
    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary model directory structure"""
        temp_dir = tempfile.mkdtemp()
        model_dir = Path(temp_dir) / "test_model"
        model_dir.mkdir(parents=True)
        
        # Create transformer component
        transformer_dir = model_dir / "transformer"
        transformer_dir.mkdir()
        transformer_config = {
            "_class_name": "Transformer2DModel",
            "hidden_size": 1024,
            "num_layers": 12,
            "boundary_ratio": 0.5
        }
        with open(transformer_dir / "config.json", 'w') as f:
            json.dump(transformer_config, f)
        
        # Create VAE component
        vae_dir = model_dir / "vae"
        vae_dir.mkdir()
        vae_config = {
            "_class_name": "AutoencoderKL",
            "in_channels": 3,
            "latent_channels": 4
        }
        with open(vae_dir / "config.json", 'w') as f:
            json.dump(vae_config, f)
        
        # Create scheduler component
        scheduler_dir = model_dir / "scheduler"
        scheduler_dir.mkdir()
        scheduler_config = {
            "_class_name": "DDIMScheduler",
            "num_train_timesteps": 1000
        }
        with open(scheduler_dir / "scheduler_config.json", 'w') as f:
            json.dump(scheduler_config, f)
        
        yield str(model_dir)
        shutil.rmtree(temp_dir)
    
    def test_analyze_transformer_component(self, component_analyzer, temp_model_dir):
        """Test transformer component analysis"""
        transformer_path = Path(temp_model_dir) / "transformer"
        component = component_analyzer.analyze_component(
            str(transformer_path), ComponentType.TRANSFORMER
        )
        
        assert component.component_type == ComponentType.TRANSFORMER
        assert component.is_functional == True
        assert component.class_name == "Transformer2DModel"
        assert component.memory_usage_mb > 0
        assert "torch" in component.required_dependencies
        assert "transformers" in component.required_dependencies
    
    def test_analyze_vae_component(self, component_analyzer, temp_model_dir):
        """Test VAE component analysis"""
        vae_path = Path(temp_model_dir) / "vae"
        component = component_analyzer.analyze_component(
            str(vae_path), ComponentType.VAE
        )
        
        assert component.component_type == ComponentType.VAE
        assert component.is_functional == True
        assert component.class_name == "AutoencoderKL"
        assert component.memory_usage_mb > 0
        assert "torch" in component.required_dependencies
        assert "diffusers" in component.required_dependencies
    
    def test_analyze_scheduler_component(self, component_analyzer, temp_model_dir):
        """Test scheduler component analysis"""
        scheduler_path = Path(temp_model_dir) / "scheduler"
        component = component_analyzer.analyze_component(
            str(scheduler_path), ComponentType.SCHEDULER
        )
        
        assert component.component_type == ComponentType.SCHEDULER
        assert component.is_functional == True
        assert component.class_name == "DDIMScheduler"
        assert component.memory_usage_mb == 10  # Schedulers are lightweight
        assert "diffusers" in component.required_dependencies
    
    def test_analyze_missing_component(self, component_analyzer):
        """Test analysis of missing component"""
        component = component_analyzer.analyze_component(
            "/nonexistent/path", ComponentType.TRANSFORMER
        )
        
        assert component.component_type == ComponentType.TRANSFORMER
        assert component.is_functional == False
        assert "Config file missing" in component.limitations[0]
    
    def test_analyze_corrupted_component(self, component_analyzer, temp_model_dir):
        """Test analysis of component with corrupted config"""
        transformer_path = Path(temp_model_dir) / "transformer"
        
        # Write invalid JSON
        with open(transformer_path / "config.json", 'w') as f:
            f.write("invalid json content")
        
        component = component_analyzer.analyze_component(
            str(transformer_path), ComponentType.TRANSFORMER
        )
        
        assert component.component_type == ComponentType.TRANSFORMER
        assert component.is_functional == False
        assert "Config validation failed" in component.limitations[0]
class
 TestAlternativeModelSuggester:
    """Test alternative model suggestion functionality"""
    
    @pytest.fixture
    def suggester(self):
        return AlternativeModelSuggester()
    
    @pytest.fixture
    def temp_models_dir(self):
        """Create temporary models directory with sample models"""
        temp_dir = tempfile.mkdtemp()
        models_dir = Path(temp_dir) / "models"
        models_dir.mkdir()
        
        # Create sample local model
        local_model_dir = models_dir / "local_wan_model"
        local_model_dir.mkdir()
        
        model_index = {
            "_class_name": "WanPipeline",
            "transformer": ["Transformer2DModel", "transformer"],
            "vae": ["AutoencoderKL", "vae"]
        }
        with open(local_model_dir / "model_index.json", 'w') as f:
            json.dump(model_index, f)
        
        # Create some dummy files to simulate model size
        (local_model_dir / "pytorch_model.bin").write_bytes(b"0" * 1024 * 1024)  # 1MB
        
        yield str(models_dir)
        shutil.rmtree(temp_dir)
    
    def test_suggest_wan_t2v_alternatives(self, suggester):
        """Test suggesting alternatives for Wan T2V models"""
        alternatives = suggester.suggest_alternatives("wan_t2v")
        
        assert len(alternatives) > 0
        assert all(alt.architecture_type == "wan_t2v" for alt in alternatives)
        assert alternatives[0].compatibility_score >= alternatives[-1].compatibility_score
    
    def test_suggest_alternatives_with_vram_filter(self, suggester):
        """Test VRAM filtering for alternative suggestions"""
        # Test with low VRAM
        low_vram_alternatives = suggester.suggest_alternatives("wan_t2v", available_vram_mb=4096)
        
        # Test with high VRAM
        high_vram_alternatives = suggester.suggest_alternatives("wan_t2v", available_vram_mb=16384)
        
        assert len(low_vram_alternatives) <= len(high_vram_alternatives)
        
        for alt in low_vram_alternatives:
            assert alt.resource_requirements.get("min_vram_mb", 0) <= 4096
    
    def test_get_local_alternatives(self, suggester, temp_models_dir):
        """Test finding local alternative models"""
        alternatives = suggester.get_local_alternatives(temp_models_dir)
        
        assert len(alternatives) > 0
        local_alt = alternatives[0]
        assert local_alt.model_name == "local_wan_model"
        assert local_alt.download_required == False
        assert local_alt.size_mb > 0
        assert "wan" in local_alt.architecture_type


class TestFallbackHandler:
    """Test main fallback handler functionality"""
    
    @pytest.fixture
    def fallback_handler(self):
        return FallbackHandler()
    
    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary model directory for testing"""
        temp_dir = tempfile.mkdtemp()
        model_dir = Path(temp_dir) / "test_model"
        model_dir.mkdir(parents=True)
        
        # Create model_index.json
        model_index = {
            "_class_name": "WanPipeline",
            "transformer": ["Transformer2DModel", "transformer"],
            "transformer_2": ["Transformer2DModel", "transformer_2"],
            "vae": ["AutoencoderKL", "vae"],
            "scheduler": ["DDIMScheduler", "scheduler"]
        }
        with open(model_dir / "model_index.json", 'w') as f:
            json.dump(model_index, f)
        
        # Create component directories with configs
        for component_name in ["transformer", "transformer_2", "vae", "scheduler"]:
            comp_dir = model_dir / component_name
            comp_dir.mkdir()
            
            if component_name == "scheduler":
                config_file = "scheduler_config.json"
            else:
                config_file = "config.json"
            
            config = {
                "_class_name": f"{component_name.title()}Model",
                "test_param": "test_value"
            }
            with open(comp_dir / config_file, 'w') as f:
                json.dump(config, f)
        
        yield str(model_dir)
        shutil.rmtree(temp_dir)
    
    def test_create_memory_fallback_strategy(self, fallback_handler):
        """Test creation of memory-related fallback strategy"""
        error = RuntimeError("CUDA out of memory")
        strategy = fallback_handler.create_fallback_strategy("WanPipeline", error)
        
        assert strategy.strategy_type == FallbackStrategyType.OPTIMIZATION_FALLBACK
        assert "memory optimizations" in strategy.description.lower()
        assert strategy.success_probability > 0.5
        assert "mixed precision" in " ".join(strategy.implementation_steps).lower()
    
    def test_create_pipeline_fallback_strategy(self, fallback_handler):
        """Test creation of pipeline-related fallback strategy"""
        error = RuntimeError("Pipeline class not found")
        strategy = fallback_handler.create_fallback_strategy("WanPipeline", error)
        
        assert strategy.strategy_type == FallbackStrategyType.PIPELINE_SUBSTITUTION
        assert "alternative pipeline" in strategy.description.lower()
        assert "trust_remote_code" in " ".join(strategy.implementation_steps).lower()
    
    def test_attempt_component_isolation(self, fallback_handler, temp_model_dir):
        """Test component isolation functionality"""
        components = fallback_handler.attempt_component_isolation(temp_model_dir)
        
        assert len(components) > 0
        
        # Check that we found the expected components
        component_types = [comp.component_type for comp in components]
        assert ComponentType.TRANSFORMER in component_types
        assert ComponentType.VAE in component_types
        assert ComponentType.SCHEDULER in component_types
        
        # Check that components have proper analysis
        for component in components:
            assert component.component_path is not None
            assert component.class_name is not None
            assert isinstance(component.is_functional, bool)
    
    def test_suggest_alternative_models(self, fallback_handler):
        """Test alternative model suggestion"""
        alternatives = fallback_handler.suggest_alternative_models("wan_t2v")
        
        assert len(alternatives) > 0
        
        # Check that alternatives are properly structured
        for alt in alternatives:
            assert alt.model_name is not None
            assert alt.architecture_type is not None
            assert 0 <= alt.compatibility_score <= 1
            assert isinstance(alt.feature_parity, dict)
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.empty_cache')
    def test_execute_optimization_strategy(self, mock_empty_cache, mock_cuda_available, 
                                         fallback_handler, temp_model_dir):
        """Test execution of optimization strategy"""
        mock_cuda_available.return_value = True
        
        strategy = FallbackStrategy(
            strategy_type=FallbackStrategyType.OPTIMIZATION_FALLBACK,
            description="Test optimization",
            implementation_steps=["Apply optimizations"],
            expected_limitations=["Slower generation"],
            success_probability=0.9,
            resource_requirements={}
        )
        
        result = fallback_handler.execute_fallback_strategy(strategy, temp_model_dir)
        
        assert result.success == True
        assert result.strategy_used == strategy
        assert "memory_reduction" in result.performance_impact
        mock_empty_cache.assert_called_once()


class TestConvenienceFunctions:
    """Test convenience functions for common scenarios"""
    
    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary model directory for testing"""
        temp_dir = tempfile.mkdtemp()
        model_dir = Path(temp_dir) / "test_model"
        model_dir.mkdir(parents=True)
        
        # Create minimal model structure
        model_index = {
            "_class_name": "WanPipeline",
            "transformer": ["Transformer2DModel", "transformer"]
        }
        with open(model_dir / "model_index.json", 'w') as f:
            json.dump(model_index, f)
        
        transformer_dir = model_dir / "transformer"
        transformer_dir.mkdir()
        config = {"_class_name": "Transformer2DModel"}
        with open(transformer_dir / "config.json", 'w') as f:
            json.dump(config, f)
        
        yield str(model_dir)
        shutil.rmtree(temp_dir)
    
    def test_handle_pipeline_failure(self, temp_model_dir):
        """Test pipeline failure handling convenience function"""
        error = RuntimeError("Pipeline loading failed")
        context = {"target_architecture": "wan_t2v"}
        
        result = handle_pipeline_failure(temp_model_dir, error, context)
        
        assert isinstance(result, FallbackResult)
        assert result.strategy_used is not None
    
    def test_analyze_model_components(self, temp_model_dir):
        """Test model component analysis convenience function"""
        components = analyze_model_components(temp_model_dir)
        
        assert len(components) > 0
        assert all(isinstance(comp, UsableComponent) for comp in components)
    
    def test_find_alternative_models(self):
        """Test alternative model finding convenience function"""
        alternatives = find_alternative_models("wan_t2v")
        
        assert len(alternatives) > 0
        assert all(isinstance(alt, AlternativeModel) for alt in alternatives)


class TestErrorRecoveryScenarios:
    """Test various error recovery scenarios"""
    
    @pytest.fixture
    def fallback_handler(self):
        return FallbackHandler()
    
    def test_memory_error_recovery(self, fallback_handler):
        """Test recovery from memory-related errors"""
        error = RuntimeError("CUDA out of memory: Tried to allocate 2.00 GiB")
        strategy = fallback_handler.create_fallback_strategy("WanPipeline", error)
        
        assert strategy.strategy_type == FallbackStrategyType.OPTIMIZATION_FALLBACK
        assert strategy.success_probability > 0.7
        assert any("mixed precision" in step.lower() for step in strategy.implementation_steps)
        assert any("cpu offload" in step.lower() for step in strategy.implementation_steps)
    
    def test_pipeline_not_found_recovery(self, fallback_handler):
        """Test recovery from pipeline not found errors"""
        error = ImportError("No module named 'wan_pipeline'")
        strategy = fallback_handler.create_fallback_strategy("WanPipeline", error)
        
        assert strategy.strategy_type == FallbackStrategyType.PIPELINE_SUBSTITUTION
        assert any("trust_remote_code" in step.lower() for step in strategy.implementation_steps)
    
    def test_component_missing_recovery(self, fallback_handler):
        """Test recovery from missing component errors"""
        error = FileNotFoundError("transformer component not found")
        strategy = fallback_handler.create_fallback_strategy("WanPipeline", error)
        
        assert strategy.strategy_type == FallbackStrategyType.COMPONENT_ISOLATION
        assert any("analyze each model component" in step.lower() for step in strategy.implementation_steps)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])