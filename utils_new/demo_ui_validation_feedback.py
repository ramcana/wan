"""
Demo: UI Validation and Feedback System

This script demonstrates the enhanced UI validation and feedback system
for the Wan2.2 Video Generation interface.
"""

import sys
import os
from typing import Dict, Any

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def demo_prompt_validation():
    """Demonstrate real-time prompt validation"""
    print("=== Prompt Validation Demo ===")
    
    from ui_validation import UIValidationManager
    
    manager = UIValidationManager({
        "max_prompt_length": 500,
        "enable_realtime_validation": True
    })
    
    test_prompts = [
        "",  # Empty prompt
        "Short",  # Very short prompt
        "A beautiful sunset over the ocean with gentle waves",  # Good prompt
        "A" * 600,  # Too long prompt
        "A video with nude content and violence",  # Problematic content
        "A flowing river through a dynamic forest scene with moving leaves"  # Optimized prompt
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}: {'Empty' if not prompt else prompt[:50] + ('...' if len(prompt) > 50 else '')}")
        
        validation_html, is_valid, char_count = manager.validate_prompt_realtime(prompt, "t2v-A14B")
        
        print(f"  Character Count: {char_count}")
        print(f"  Valid: {is_valid}")
        
        if validation_html:
            # Extract key information from HTML
            if "too long" in validation_html.lower():
                print("  Issue: Prompt too long")
            if "too short" in validation_html.lower():
                print("  Issue: Prompt too short")
            if "problematic" in validation_html.lower():
                print("  Issue: Potentially problematic content detected")
            if "suggestion" in validation_html.lower():
                print("  Has suggestions for improvement")

def demo_parameter_validation():
    """Demonstrate parameter validation"""
    print("\n=== Parameter Validation Demo ===")
    
    from ui_validation import UIValidationManager
    
    manager = UIValidationManager()
    
    test_params = [
        {
            "name": "Valid Parameters",
            "params": {
                "model_type": "t2v-A14B",
                "resolution": "1280x720",
                "steps": 50
            }
        },
        {
            "name": "Invalid Model",
            "params": {
                "model_type": "invalid-model",
                "resolution": "1280x720",
                "steps": 50
            }
        },
        {
            "name": "Invalid Resolution",
            "params": {
                "model_type": "t2v-A14B",
                "resolution": "100x100",
                "steps": 50
            }
        },
        {
            "name": "Too Many Steps",
            "params": {
                "model_type": "t2v-A14B",
                "resolution": "1280x720",
                "steps": 200
            }
        }
    ]
    
    for test_case in test_params:
        print(f"\nTest: {test_case['name']}")
        
        validation_html, is_valid = manager.validate_generation_params(
            test_case['params'], test_case['params']['model_type']
        )
        
        print(f"  Valid: {is_valid}")
        
        if validation_html:
            if "unknown model" in validation_html.lower():
                print("  Issue: Unknown model type")
            if "too low" in validation_html.lower():
                print("  Issue: Resolution too low")
            if "too high" in validation_html.lower():
                print("  Issue: Parameter value too high")

def demo_progress_indicators():
    """Demonstrate progress indicators"""
    print("\n=== Progress Indicators Demo ===")
    
    from ui_validation import UIValidationManager
    
    manager = UIValidationManager()
    
    stages = [
        ("validation", 0.1, "Validating inputs..."),
        ("model_loading", 0.3, "Loading AI model..."),
        ("generation", 0.7, "Generating video frames..."),
        ("post_processing", 0.9, "Processing output..."),
        ("complete", 1.0, "Generation complete!")
    ]
    
    for stage, progress, message in stages:
        print(f"\nStage: {stage.replace('_', ' ').title()}")
        print(f"Progress: {progress * 100:.1f}%")
        print(f"Message: {message}")
        
        progress_html = manager.create_progress_indicator(stage, progress, message)
        
        # Verify HTML contains expected elements
        assert stage.replace('_', ' ').title() in progress_html
        assert f"{progress * 100:.1f}%" in progress_html
        assert message in progress_html
        assert "progress-container" in progress_html
        
        print("  ✓ Progress indicator HTML generated successfully")

def demo_error_display():
    """Demonstrate error display with recovery suggestions"""
    print("\n=== Error Display Demo ===")
    
    from ui_validation import UIValidationManager
    from error_handler import UserFriendlyError, ErrorCategory, ErrorSeverity
    
    manager = UIValidationManager()
    
    test_errors = [
        {
            "name": "Validation Error",
            "error": UserFriendlyError(
                category=ErrorCategory.INPUT_VALIDATION,
                severity=ErrorSeverity.MEDIUM,
                title="Invalid Input",
                message="Your prompt is too long for the selected model",
                recovery_suggestions=[
                    "Shorten your prompt to under 500 characters",
                    "Split complex prompts into simpler descriptions",
                    "Remove unnecessary details"
                ],
                recovery_actions=[]
            )
        },
        {
            "name": "VRAM Error",
            "error": UserFriendlyError(
                category=ErrorCategory.VRAM_MEMORY,
                severity=ErrorSeverity.HIGH,
                title="Insufficient GPU Memory",
                message="Not enough VRAM available for this generation",
                recovery_suggestions=[
                    "Reduce resolution to 720p",
                    "Lower the number of inference steps",
                    "Close other GPU-intensive applications"
                ],
                recovery_actions=[]
            )
        },
        {
            "name": "Model Loading Error",
            "error": UserFriendlyError(
                category=ErrorCategory.MODEL_LOADING,
                severity=ErrorSeverity.HIGH,
                title="Model Loading Failed",
                message="Could not load the AI model files",
                recovery_suggestions=[
                    "Check if model files exist",
                    "Restart the application",
                    "Re-download the model if corrupted"
                ],
                recovery_actions=[]
            )
        }
    ]
    
    for test_case in test_errors:
        print(f"\nError Type: {test_case['name']}")
        print(f"Category: {test_case['error'].category.value}")
        print(f"Severity: {test_case['error'].severity.value}")
        print(f"Title: {test_case['error'].title}")
        print(f"Message: {test_case['error'].message}")
        print(f"Recovery Suggestions: {len(test_case['error'].recovery_suggestions)}")
        
        error_html, show_display = manager.create_error_display_with_recovery(
            test_case['error'], "demo"
        )
        
        print(f"  ✓ Error display HTML generated (show: {show_display})")
        
        # Verify HTML contains expected elements
        assert test_case['error'].title.lower() in error_html.lower()
        assert test_case['error'].message.lower() in error_html.lower()
        for suggestion in test_case['error'].recovery_suggestions[:3]:
            assert suggestion.lower() in error_html.lower()

def demo_comprehensive_validation():
    """Demonstrate comprehensive validation workflow"""
    print("\n=== Comprehensive Validation Demo ===")
    
    from ui_validation import UIValidationManager, UIValidationState
    
    manager = UIValidationManager()
    
    # Simulate validation states from different components
    print("\nScenario 1: All validations pass")
    manager.validation_states = {
        'prompt': UIValidationState(
            is_valid=True,
            errors=[],
            warnings=[],
            suggestions=["Consider adding motion terms for better video quality"]
        ),
        'image': UIValidationState(
            is_valid=True,
            errors=[],
            warnings=[],
            suggestions=[]
        ),
        'params': UIValidationState(
            is_valid=True,
            errors=[],
            warnings=[],
            suggestions=[]
        )
    }
    
    summary_html, all_valid = manager.create_comprehensive_validation_summary()
    print(f"All Valid: {all_valid}")
    print("✓ Success summary generated")
    
    print("\nScenario 2: Mixed validation results")
    manager.validation_states = {
        'prompt': UIValidationState(
            is_valid=False,
            errors=["Prompt too long (600 characters, max 500)"],
            warnings=["Contains repetitive content"],
            suggestions=["Shorten prompt", "Remove repetitive words"]
        ),
        'image': UIValidationState(
            is_valid=True,
            errors=[],
            warnings=["Image has low contrast"],
            suggestions=["Use higher contrast image"]
        ),
        'params': UIValidationState(
            is_valid=False,
            errors=["Steps too high (150, max 100)"],
            warnings=[],
            suggestions=["Use 50-80 steps for good quality"]
        )
    }
    
    summary_html, all_valid = manager.create_comprehensive_validation_summary()
    print(f"All Valid: {all_valid}")
    print("Issues found:")
    print("  - Prompt: 1 error, 1 warning")
    print("  - Image: 0 errors, 1 warning")
    print("  - Parameters: 1 error, 0 warnings")
    print("✓ Comprehensive validation summary generated")

def demo_real_time_feedback():
    """Demonstrate real-time feedback simulation"""
    print("\n=== Real-Time Feedback Demo ===")
    
    from ui_validation import UIValidationManager
    
    manager = UIValidationManager({
        "enable_realtime_validation": True,
        "validation_delay_ms": 100
    })
    
    # Simulate user typing a prompt
    typing_sequence = [
        "A",
        "A b",
        "A be",
        "A bea",
        "A beau",
        "A beaut",
        "A beauti",
        "A beautif",
        "A beautifu",
        "A beautiful",
        "A beautiful s",
        "A beautiful su",
        "A beautiful sun",
        "A beautiful suns",
        "A beautiful sunse",
        "A beautiful sunset",
        "A beautiful sunset over the ocean"
    ]
    
    print("Simulating user typing (real-time validation):")
    
    for i, partial_prompt in enumerate(typing_sequence):
        validation_html, is_valid, char_count = manager.validate_prompt_realtime(
            partial_prompt, "t2v-A14B"
        )
        
        if i % 5 == 0:  # Show every 5th step
            print(f"  '{partial_prompt}' -> {char_count} ({'✓' if is_valid else '✗'})")
    
    print("✓ Real-time validation feedback demonstrated")

def main():
    """Run all demos"""
    print("🎬 Wan2.2 UI Validation and Feedback System Demo")
    print("=" * 60)
    
    try:
        demo_prompt_validation()
        demo_parameter_validation()
        demo_progress_indicators()
        demo_error_display()
        demo_comprehensive_validation()
        demo_real_time_feedback()
        
        print("\n" + "=" * 60)
        print("✅ All demos completed successfully!")
        print("\nKey Features Demonstrated:")
        print("  • Real-time prompt validation with character counting")
        print("  • Parameter validation with specific error messages")
        print("  • Progress indicators with stage-specific styling")
        print("  • User-friendly error displays with recovery suggestions")
        print("  • Comprehensive validation summaries")
        print("  • Real-time feedback simulation")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()