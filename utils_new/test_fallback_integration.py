"""
Integration tests for fallback handler system
"""

import tempfile
import json
import shutil
from pathlib import Path
from fallback_handler import (
    FallbackHandler, 
    FallbackStrategyType,
    ComponentType,
    handle_pipeline_failure,
    analyze_model_components,
    find_alternative_models
)


def test_memory_error_fallback():
    """Test memory error fallback strategy"""
    handler = FallbackHandler()
    error = RuntimeError("CUDA out of memory: Tried to allocate 2.00 GiB")
    
    strategy = handler.create_fallback_strategy("WanPipeline", error)
    
    assert strategy.strategy_type == FallbackStrategyType.OPTIMIZATION_FALLBACK
    assert strategy.success_probability > 0.7
    assert any("mixed precision" in step.lower() for step in strategy.implementation_steps)
    print("✓ Memory error fallback test passed")


def test_pipeline_error_fallback():
    """Test pipeline loading error fallback strategy"""
    handler = FallbackHandler()
    error = ImportError("No module named 'wan_pipeline'")
    
    strategy = handler.create_fallback_strategy("WanPipeline", error)
    
    assert strategy.strategy_type == FallbackStrategyType.PIPELINE_SUBSTITUTION
    assert any("trust_remote_code" in step.lower() for step in strategy.implementation_steps)
    print("✓ Pipeline error fallback test passed")


def test_component_isolation():
    """Test component isolation functionality"""
    # Create temporary model directory
    temp_dir = tempfile.mkdtemp()
    model_dir = Path(temp_dir) / "test_model"
    model_dir.mkdir(parents=True)
    
    try:
        # Create model_index.json
        model_index = {
            "_class_name": "WanPipeline",
            "transformer": ["Transformer2DModel", "transformer"],
            "vae": ["AutoencoderKL", "vae"],
            "scheduler": ["DDIMScheduler", "scheduler"]
        }
        with open(model_dir / "model_index.json", 'w') as f:
            json.dump(model_index, f)
        
        # Create transformer component
        transformer_dir = model_dir / "transformer"
        transformer_dir.mkdir()
        transformer_config = {
            "_class_name": "Transformer2DModel",
            "hidden_size": 1024,
            "num_layers": 12
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
        
        # Test component isolation
        handler = FallbackHandler()
        components = handler.attempt_component_isolation(str(model_dir))
        
        assert len(components) >= 3  # Should find at least transformer, vae, scheduler
        
        # Check component types
        component_types = [comp.component_type for comp in components]
        assert ComponentType.TRANSFORMER in component_types
        assert ComponentType.VAE in component_types
        assert ComponentType.SCHEDULER in component_types
        
        # Check that components are analyzed properly
        functional_components = [comp for comp in components if comp.is_functional]
        assert len(functional_components) > 0
        
        print(f"✓ Component isolation test passed - found {len(components)} components, {len(functional_components)} functional")
    
    finally:
        shutil.rmtree(temp_dir)


def test_alternative_model_suggestions():
    """Test alternative model suggestion system"""
    handler = FallbackHandler()
    
    # Test Wan T2V alternatives
    wan_alternatives = handler.suggest_alternative_models("wan_t2v")
    assert len(wan_alternatives) > 0
    assert all("wan" in alt.architecture_type for alt in wan_alternatives)
    
    # Test Stable Diffusion alternatives
    sd_alternatives = handler.suggest_alternative_models("stable_diffusion")
    assert len(sd_alternatives) > 0
    # At least one should be stable diffusion
    assert any(alt.architecture_type == "stable_diffusion" for alt in sd_alternatives)
    
    # Test unknown architecture - may include local models
    unknown_alternatives = handler.suggest_alternative_models("unknown_arch")
    # Should have fewer alternatives than known architectures
    assert len(unknown_alternatives) <= len(wan_alternatives)
    
    print(f"✓ Alternative model suggestions test passed - found {len(wan_alternatives)} Wan alternatives, {len(sd_alternatives)} SD alternatives")


def test_fallback_strategy_execution():
    """Test fallback strategy execution"""
    handler = FallbackHandler()
    
    # Create temporary model directory
    temp_dir = tempfile.mkdtemp()
    model_dir = Path(temp_dir) / "test_model"
    model_dir.mkdir(parents=True)
    
    try:
        # Create minimal model structure
        model_index = {"_class_name": "WanPipeline"}
        with open(model_dir / "model_index.json", 'w') as f:
            json.dump(model_index, f)
        
        # Test optimization fallback execution
        from fallback_handler import FallbackStrategy
        optimization_strategy = FallbackStrategy(
            strategy_type=FallbackStrategyType.OPTIMIZATION_FALLBACK,
            description="Test optimization",
            implementation_steps=["Apply optimizations"],
            expected_limitations=["Slower generation"],
            success_probability=0.9,
            resource_requirements={}
        )
        
        result = handler.execute_fallback_strategy(optimization_strategy, str(model_dir))
        assert result.success == True
        assert result.strategy_used == optimization_strategy
        
        # Test alternative model fallback execution
        alternative_strategy = FallbackStrategy(
            strategy_type=FallbackStrategyType.ALTERNATIVE_MODEL,
            description="Test alternative model",
            implementation_steps=["Find alternatives"],
            expected_limitations=["Different results"],
            success_probability=0.8,
            resource_requirements={}
        )
        
        context = {"target_architecture": "wan_t2v"}
        result = handler.execute_fallback_strategy(alternative_strategy, str(model_dir), context)
        assert result.success == True
        assert result.strategy_used == alternative_strategy
        
        print("✓ Fallback strategy execution test passed")
    
    finally:
        shutil.rmtree(temp_dir)


def test_convenience_functions():
    """Test convenience functions"""
    # Create temporary model directory
    temp_dir = tempfile.mkdtemp()
    model_dir = Path(temp_dir) / "test_model"
    model_dir.mkdir(parents=True)
    
    try:
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
        
        # Test handle_pipeline_failure
        error = RuntimeError("Pipeline loading failed")
        result = handle_pipeline_failure(str(model_dir), error)
        assert result.strategy_used is not None
        
        # Test analyze_model_components
        components = analyze_model_components(str(model_dir))
        assert len(components) > 0
        
        # Test find_alternative_models
        alternatives = find_alternative_models("wan_t2v")
        assert len(alternatives) > 0
        
        print("✓ Convenience functions test passed")
    
    finally:
        shutil.rmtree(temp_dir)


def test_error_categorization():
    """Test error categorization and strategy selection"""
    handler = FallbackHandler()
    
    # Test different error types
    test_cases = [
        ("CUDA out of memory", FallbackStrategyType.OPTIMIZATION_FALLBACK),
        ("Pipeline class not found", FallbackStrategyType.PIPELINE_SUBSTITUTION),
        ("Component missing", FallbackStrategyType.COMPONENT_ISOLATION),
        ("Unknown error", FallbackStrategyType.ALTERNATIVE_MODEL)
    ]
    
    for error_msg, expected_strategy in test_cases:
        error = RuntimeError(error_msg)
        strategy = handler.create_fallback_strategy("WanPipeline", error)
        assert strategy.strategy_type == expected_strategy
    
    print("✓ Error categorization test passed")


if __name__ == "__main__":
    print("Running fallback handler integration tests...")
    
    test_memory_error_fallback()
    test_pipeline_error_fallback()
    test_component_isolation()
    test_alternative_model_suggestions()
    test_fallback_strategy_execution()
    test_convenience_functions()
    test_error_categorization()
    
    print("\n🎉 All integration tests passed!")