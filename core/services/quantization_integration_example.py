from unittest.mock import Mock, patch
"""
Integration example showing how to use the quantization management system.

This demonstrates the complete workflow from strategy determination through
quality validation with fallback handling.
"""

import logging
from typing import Any, Optional

from quantization_controller import (
    QuantizationController, QuantizationMethod, ModelInfo, HardwareProfile
)
from quantization_fallback_system import QuantizationFallbackSystem
from quantization_quality_validator import QuantizationQualityValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegratedQuantizationManager:
    """
    Integrated quantization manager that combines all quantization components
    for a complete quantization workflow.
    """
    
    def __init__(self):
        """Initialize the integrated quantization manager"""
        self.controller = QuantizationController()
        self.fallback_system = QuantizationFallbackSystem()
        self.quality_validator = QuantizationQualityValidator()
        
        logger.info("IntegratedQuantizationManager initialized")
    
    def quantize_model_with_validation(self, model: Any, model_info: ModelInfo,
                                     validate_quality: bool = True,
                                     max_fallback_attempts: int = 3) -> dict:
        """
        Complete quantization workflow with quality validation and fallback handling.
        
        Args:
            model: Model to quantize
            model_info: Information about the model
            validate_quality: Whether to perform quality validation
            max_fallback_attempts: Maximum number of fallback attempts
            
        Returns:
            Dictionary with quantization results and quality metrics
        """
        logger.info(f"Starting quantization workflow for {model_info.name}")
        
        # Step 1: Determine optimal quantization strategy
        strategy = self.controller.determine_optimal_strategy(model_info)
        logger.info(f"Selected quantization strategy: {strategy.method.value}")
        
        # Step 2: Validate compatibility
        compatibility = self.fallback_system.validate_quantization_compatibility(
            model_info, strategy.method, self.controller.hardware_profile
        )
        
        if not compatibility.is_compatible:
            logger.warning("Selected method not compatible, getting recommendation")
            recommended_method, confidence = self.fallback_system.get_recommended_method(
                model_info, self.controller.hardware_profile, self.controller.preferences
            )
            strategy.method = recommended_method
            logger.info(f"Using recommended method: {recommended_method.value} (confidence: {confidence:.2f})")
        
        # Step 3: Apply quantization with monitoring
        attempt = 0
        quantization_result = None
        
        while attempt < max_fallback_attempts:
            logger.info(f"Quantization attempt {attempt + 1}/{max_fallback_attempts}")
            
            # Apply quantization
            quantization_result = self.controller.apply_quantization_with_monitoring(model, strategy)
            
            if quantization_result.success:
                logger.info("Quantization successful")
                break
            
            # Check if fallback should be attempted
            should_fallback, fallback_method = self.fallback_system.should_attempt_fallback(
                quantization_result, model_info, self.controller.hardware_profile
            )
            
            if not should_fallback or not fallback_method:
                logger.warning("No fallback available or fallback not recommended")
                break
            
            # Record fallback attempt
            self.fallback_system.record_fallback_attempt(
                strategy.method, fallback_method, "quantization_failed",
                model_info.name, False
            )
            
            # Update strategy for fallback
            strategy.method = fallback_method
            logger.info(f"Attempting fallback to: {fallback_method.value}")
            attempt += 1
        
        # Step 4: Quality validation (if requested and quantization succeeded)
        quality_report = None
        if validate_quality and quantization_result and quantization_result.success:
            logger.info("Starting quality validation")
            try:
                # Create original model reference
                original_model = model  # In practice, you'd create a copy
                
                # Validate quality
                quality_report = self.quality_validator.validate_quantization_quality(
                    original_model, model, strategy.method, model_info
                )
                
                logger.info(f"Quality validation completed: {quality_report.overall_quality_level.value}")
                
                # Check if quality is acceptable
                if quality_report.overall_quality_score < 0.7:
                    logger.warning("Quality below acceptable threshold")
                    
                    # Record quality-based fallback if needed
                    if attempt < max_fallback_attempts - 1:
                        logger.info("Attempting quality-based fallback")
                        # This would trigger another quantization attempt with different settings
                
            except Exception as e:
                logger.error(f"Quality validation failed: {e}")
                quality_report = None
        
        # Step 5: Compile results
        results = {
            'quantization_result': quantization_result,
            'quality_report': quality_report,
            'strategy_used': strategy,
            'attempts_made': attempt + 1,
            'compatibility_check': compatibility,
            'success': quantization_result.success if quantization_result else False
        }
        
        # Step 6: Update preferences if successful
        if quantization_result and quantization_result.success:
            if self.controller.preferences.remember_model_settings:
                # Update model-specific preferences
                self.controller.preferences.model_specific_preferences[model_info.name] = strategy.method
                self.controller._save_preferences()
                logger.info(f"Updated preferences for {model_info.name}")
        
        logger.info("Quantization workflow completed")
        return results
    
    def compare_quantization_methods(self, model: Any, model_info: ModelInfo,
                                   methods: Optional[list] = None) -> dict:
        """
        Compare multiple quantization methods for the given model.
        
        Args:
            model: Model to test
            model_info: Information about the model
            methods: List of methods to compare (uses supported methods if None)
            
        Returns:
            Comparison results with recommendations
        """
        if methods is None:
            methods = self.controller.get_supported_methods()
        
        logger.info(f"Comparing {len(methods)} quantization methods")
        
        # Use quality validator for comparison
        comparison_reports = self.quality_validator.compare_quantization_methods(
            model, model_info, methods
        )
        
        # Analyze results and generate recommendations
        best_method = None
        best_score = 0.0
        
        for method, report in comparison_reports.items():
            if report.overall_quality_score > best_score:
                best_score = report.overall_quality_score
                best_method = method
        
        # Get fallback system recommendations
        recommended_method, confidence = self.fallback_system.get_recommended_method(
            model_info, self.controller.hardware_profile, self.controller.preferences
        )
        
        return {
            'comparison_reports': comparison_reports,
            'best_method_by_quality': best_method,
            'recommended_method': recommended_method,
            'recommendation_confidence': confidence,
            'summary': {
                method.value: {
                    'quality_score': report.overall_quality_score,
                    'quality_level': report.overall_quality_level.value,
                    'memory_savings': report.memory_savings,
                    'performance_impact': report.performance_impact
                }
                for method, report in comparison_reports.items()
            }
        }
    
    def get_quantization_status(self) -> dict:
        """Get current status of quantization system"""
        return {
            'hardware_profile': {
                'gpu_model': self.controller.hardware_profile.gpu_model,
                'vram_gb': self.controller.hardware_profile.vram_gb,
                'supports_bf16': self.controller.hardware_profile.supports_bf16,
                'supports_int8': self.controller.hardware_profile.supports_int8,
                'supports_fp8': self.controller.hardware_profile.supports_fp8
            },
            'supported_methods': [method.value for method in self.controller.get_supported_methods()],
            'current_preferences': {
                'preferred_method': self.controller.preferences.preferred_method.value,
                'auto_fallback_enabled': self.controller.preferences.auto_fallback_enabled,
                'timeout_seconds': self.controller.preferences.timeout_seconds
            },
            'fallback_statistics': self.fallback_system.get_fallback_statistics()
        }
    
    def update_quantization_preferences(self, **kwargs) -> None:
        """Update quantization preferences"""
        preferences = self.controller.get_preferences()
        
        if 'preferred_method' in kwargs:
            preferences.preferred_method = QuantizationMethod(kwargs['preferred_method'])
        
        if 'auto_fallback_enabled' in kwargs:
            preferences.auto_fallback_enabled = kwargs['auto_fallback_enabled']
        
        if 'timeout_seconds' in kwargs:
            preferences.timeout_seconds = kwargs['timeout_seconds']
        
        if 'skip_quality_check' in kwargs:
            preferences.skip_quality_check = kwargs['skip_quality_check']
        
        self.controller.update_preferences(preferences)
        logger.info("Quantization preferences updated")


# Example usage
def example_usage():
    """Example of how to use the integrated quantization manager"""
    
    # Initialize the manager
    manager = IntegratedQuantizationManager()
    
    # Example model info (this would come from actual model inspection)
    model_info = ModelInfo(
        name="Wan2.2-TI2V-5B",
        size_gb=5.2,
        architecture="wan_t2v",
        components=["transformer", "text_encoder", "vae"],
        estimated_vram_usage=8192  # MB
    )
    
    # Mock model object (in practice, this would be your actual model)
    class MockModel:
        def __init__(self):
            self.name = "Wan2.2-TI2V-5B"
    
    model = MockModel()
    
    # Get current system status
    status = manager.get_quantization_status()
    print("System Status:")
    print(f"GPU: {status['hardware_profile']['gpu_model']}")
    print(f"VRAM: {status['hardware_profile']['vram_gb']}GB")
    print(f"Supported methods: {status['supported_methods']}")
    
    # Update preferences if needed
    manager.update_quantization_preferences(
        preferred_method='bf16',
        auto_fallback_enabled=True,
        timeout_seconds=300
    )
    
    # Perform quantization with validation
    results = manager.quantize_model_with_validation(
        model=model,
        model_info=model_info,
        validate_quality=True,
        max_fallback_attempts=3
    )
    
    print(f"\nQuantization Results:")
    print(f"Success: {results['success']}")
    print(f"Method used: {results['strategy_used'].method.value}")
    print(f"Attempts made: {results['attempts_made']}")
    
    if results['quality_report']:
        print(f"Quality level: {results['quality_report'].overall_quality_level.value}")
        print(f"Quality score: {results['quality_report'].overall_quality_score:.3f}")
    
    # Compare methods (optional)
    comparison = manager.compare_quantization_methods(model, model_info)
    print(f"\nBest method by quality: {comparison['best_method_by_quality'].value if comparison['best_method_by_quality'] else 'None'}")
    print(f"Recommended method: {comparison['recommended_method'].value}")


if __name__ == "__main__":
    example_usage()