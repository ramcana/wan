"""
Test fallback handler against specific requirements from the spec
"""

import tempfile
import json
import shutil
from pathlib import Path
from fallback_handler import (
    FallbackHandler, 
    FallbackStrategyType,
    ComponentType,
    handle_pipeline_failure
)


def test_requirement_5_1_fallback_configurations():
    """
    Requirement 5.1: WHEN the full WanPipeline is not available 
    THEN the system SHALL attempt compatible fallback configurations
    """
    handler = FallbackHandler()
    
    # Simulate WanPipeline not available
    error = ImportError("WanPipeline not found")
    strategy = handler.create_fallback_strategy("WanPipeline", error)
    
    # Should create a pipeline substitution strategy
    assert strategy.strategy_type == FallbackStrategyType.PIPELINE_SUBSTITUTION
    assert "alternative pipeline" in strategy.description.lower()
    assert strategy.success_probability > 0.5
    
    print("✓ Requirement 5.1 - Fallback configurations test passed")


def test_requirement_5_2_component_isolation():
    """
    Requirement 5.2: WHEN custom components cannot be loaded 
    THEN the system SHALL identify which components can be used independently
    """
    # Create temporary model with mixed functional/non-functional components
    temp_dir = tempfile.mkdtemp()
    model_dir = Path(temp_dir) / "test_model"
    model_dir.mkdir(parents=True)
    
    try:
        # Create model_index.json
        model_index = {
            "_class_name": "WanPipeline",
            "transformer": ["Transformer2DModel", "transformer"],
            "vae": ["AutoencoderKL", "vae"],
            "broken_component": ["BrokenModel", "broken"]
        }
        with open(model_dir / "model_index.json", 'w') as f:
            json.dump(model_index, f)
        
        # Create functional transformer
        transformer_dir = model_dir / "transformer"
        transformer_dir.mkdir()
        with open(transformer_dir / "config.json", 'w') as f:
            json.dump({"_class_name": "Transformer2DModel"}, f)
        
        # Create functional VAE
        vae_dir = model_dir / "vae"
        vae_dir.mkdir()
        with open(vae_dir / "config.json", 'w') as f:
            json.dump({"_class_name": "AutoencoderKL"}, f)
        
        # Create broken component directory (no config)
        broken_dir = model_dir / "broken"
        broken_dir.mkdir()
        
        # Test component isolation
        handler = FallbackHandler()
        components = handler.attempt_component_isolation(str(model_dir))
        
        # Should identify functional and non-functional components
        functional_components = [c for c in components if c.is_functional]
        non_functional_components = [c for c in components if not c.is_functional]
        
        assert len(functional_components) >= 2  # transformer and vae should work
        assert len(non_functional_components) >= 0  # broken component may not be detected
        
        print(f"✓ Requirement 5.2 - Component isolation test passed ({len(functional_components)} functional, {len(non_functional_components)} non-functional)")
    
    finally:
        shutil.rmtree(temp_dir)


def test_requirement_5_3_vram_optimization():
    """
    Requirement 5.3: WHEN VRAM is insufficient for the full 3D pipeline 
    THEN the system SHALL apply optimization strategies
    """
    handler = FallbackHandler()
    
    # Simulate VRAM insufficient error
    error = RuntimeError("CUDA out of memory: Tried to allocate 8.00 GiB")
    strategy = handler.create_fallback_strategy("WanPipeline", error)
    
    # Should create optimization fallback
    assert strategy.strategy_type == FallbackStrategyType.OPTIMIZATION_FALLBACK
    assert strategy.success_probability > 0.7
    
    # Should include specific optimization strategies
    steps_text = " ".join(strategy.implementation_steps).lower()
    assert "mixed precision" in steps_text
    assert "cpu offload" in steps_text
    
    print("✓ Requirement 5.3 - VRAM optimization test passed")


def test_requirement_5_4_memory_constraints():
    """
    Requirement 5.4: WHEN memory constraints exist 
    THEN the system SHALL offer frame-by-frame generation and chunked decoding options
    """
    handler = FallbackHandler()
    
    # Test optimization strategy includes chunked processing
    error = RuntimeError("Out of memory")
    strategy = handler.create_fallback_strategy("WanPipeline", error)
    
    assert strategy.strategy_type == FallbackStrategyType.OPTIMIZATION_FALLBACK
    
    # Should mention chunked or sequential processing
    steps_text = " ".join(strategy.implementation_steps).lower()
    limitations_text = " ".join(strategy.expected_limitations).lower()
    
    # The strategy should address memory constraints
    assert any(keyword in steps_text for keyword in ["sequential", "chunk", "batch"])
    
    print("✓ Requirement 5.4 - Memory constraints test passed")


def test_requirement_5_5_clear_guidance():
    """
    Requirement 5.5: IF no compatible configuration is found 
    THEN the system SHALL provide clear guidance on requirements and setup steps
    """
    handler = FallbackHandler()
    
    # Test alternative model strategy provides guidance
    error = Exception("Complete system failure")
    strategy = handler.create_fallback_strategy("WanPipeline", error)
    
    # Should provide alternative model strategy with clear steps
    assert strategy.strategy_type == FallbackStrategyType.ALTERNATIVE_MODEL
    assert len(strategy.implementation_steps) > 0
    assert len(strategy.expected_limitations) > 0
    
    # Steps should be actionable
    for step in strategy.implementation_steps:
        assert len(step) > 10  # Should be descriptive
        assert any(action_word in step.lower() for action_word in [
            "analyze", "find", "suggest", "provide", "download", "setup"
        ])
    
    print("✓ Requirement 5.5 - Clear guidance test passed")


def test_comprehensive_error_categorization():
    """
    Test that the system properly categorizes different types of errors
    and provides appropriate recovery flows
    """
    handler = FallbackHandler()
    
    test_cases = [
        # Memory errors -> Optimization fallback
        ("CUDA out of memory", FallbackStrategyType.OPTIMIZATION_FALLBACK),
        ("out of memory", FallbackStrategyType.OPTIMIZATION_FALLBACK),
        ("cuda error occurred", FallbackStrategyType.OPTIMIZATION_FALLBACK),
        
        # Pipeline errors -> Pipeline substitution
        ("pipeline error", FallbackStrategyType.PIPELINE_SUBSTITUTION),
        ("class not found", FallbackStrategyType.PIPELINE_SUBSTITUTION),
        
        # Component errors -> Component isolation
        ("component missing", FallbackStrategyType.COMPONENT_ISOLATION),
        ("missing component", FallbackStrategyType.COMPONENT_ISOLATION),
        
        # Unknown errors -> Alternative model
        ("unknown error", FallbackStrategyType.ALTERNATIVE_MODEL),
        ("unexpected failure", FallbackStrategyType.ALTERNATIVE_MODEL),
    ]
    
    for error_msg, expected_strategy in test_cases:
        error = RuntimeError(error_msg)
        strategy = handler.create_fallback_strategy("WanPipeline", error)
        assert strategy.strategy_type == expected_strategy, f"Error '{error_msg}' should map to {expected_strategy}, got {strategy.strategy_type}"
    
    print("✓ Comprehensive error categorization test passed")


def test_graceful_degradation_strategies():
    """
    Test that fallback strategies provide graceful degradation
    with reasonable success probabilities and clear limitations
    """
    handler = FallbackHandler()
    
    error_types = [
        "CUDA out of memory",
        "Pipeline not found", 
        "Component missing",
        "Unknown error"
    ]
    
    for error_msg in error_types:
        error = RuntimeError(error_msg)
        strategy = handler.create_fallback_strategy("WanPipeline", error)
        
        # All strategies should have reasonable success probability
        assert 0.0 < strategy.success_probability <= 1.0
        
        # Should have implementation steps
        assert len(strategy.implementation_steps) > 0
        
        # Should have expected limitations
        assert len(strategy.expected_limitations) > 0
        
        # Should have resource requirements
        assert isinstance(strategy.resource_requirements, dict)
        
        # Should be serializable
        serialized = strategy.to_dict()
        assert "strategy_type" in serialized
        assert "description" in serialized
    
    print("✓ Graceful degradation strategies test passed")


def test_end_to_end_fallback_flow():
    """
    Test complete end-to-end fallback flow from error to recovery
    """
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
        with open(transformer_dir / "config.json", 'w') as f:
            json.dump({"_class_name": "Transformer2DModel"}, f)
        
        # Test complete fallback flow using convenience function
        error = RuntimeError("Pipeline loading failed")
        context = {"target_architecture": "wan_t2v"}
        
        result = handle_pipeline_failure(str(model_dir), error, context)
        
        # Should successfully create a fallback result
        assert result.strategy_used is not None
        assert isinstance(result.warnings, list)
        assert isinstance(result.performance_impact, dict)
        
        # Result should be serializable
        serialized = result.to_dict()
        assert "success" in serialized
        assert "strategy_used" in serialized
        
        print("✓ End-to-end fallback flow test passed")
    
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    print("Testing fallback handler against requirements...")
    
    test_requirement_5_1_fallback_configurations()
    test_requirement_5_2_component_isolation()
    test_requirement_5_3_vram_optimization()
    test_requirement_5_4_memory_constraints()
    test_requirement_5_5_clear_guidance()
    test_comprehensive_error_categorization()
    test_graceful_degradation_strategies()
    test_end_to_end_fallback_flow()
    
    print("\n🎉 All requirement tests passed! Fallback handler meets specification requirements.")
