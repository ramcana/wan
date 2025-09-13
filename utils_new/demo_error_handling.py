"""
Demonstration of the Enhanced Error Handling System

This script shows how to use the error handling system for various
video generation error scenarios.
"""

from error_handler import (
    GenerationErrorHandler,
    handle_validation_error,
    handle_model_loading_error,
    handle_vram_error,
    handle_generation_error
)


def demo_input_validation_error():
    """Demonstrate handling input validation errors"""
    print("=== Input Validation Error Demo ===")
    
    # Simulate a validation error
    error = ValueError("Invalid input provided: prompt too long")
    context = {
        "prompt": "A" * 600,  # Very long prompt
        "resolution": "1080p",
        "model_type": "wan22"
    }
    
    user_error = handle_validation_error(error, context)
    
    print(f"Error Category: {user_error.category.value}")
    print(f"Severity: {user_error.severity.value}")
    print(f"Title: {user_error.title}")
    print(f"Message: {user_error.message}")
    print("\nRecovery Suggestions:")
    for i, suggestion in enumerate(user_error.recovery_suggestions[:3], 1):
        print(f"  {i}. {suggestion}")
    
    # Try automatic recovery
    handler = GenerationErrorHandler()
    success, message = handler.attempt_automatic_recovery(user_error, context)
    print(f"\nAutomatic Recovery: {'Success' if success else 'Failed'}")
    print(f"Recovery Message: {message}")
    
    if success and "prompt" in context:
        print(f"Fixed Prompt Length: {len(context['prompt'])} characters")
    
    print("\n" + "="*50 + "\n")


def demo_vram_error():
    """Demonstrate handling VRAM/memory errors"""
    print("=== VRAM Error Demo ===")
    
    # Simulate a VRAM error
    error = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    context = {
        "resolution": "1080p",
        "steps": 50,
        "model": "wan22",
        "prompt": "A beautiful landscape video"
    }
    
    user_error = handle_vram_error(error, context)
    
    print(f"Error Category: {user_error.category.value}")
    print(f"Severity: {user_error.severity.value}")
    print(f"Title: {user_error.title}")
    print(f"Message: {user_error.message}")
    print("\nRecovery Suggestions:")
    for i, suggestion in enumerate(user_error.recovery_suggestions[:3], 1):
        print(f"  {i}. {suggestion}")
    
    # Show recovery actions
    print("\nAvailable Recovery Actions:")
    for action in user_error.recovery_actions:
        auto_text = " (Automatic)" if action.automatic else " (Manual)"
        print(f"  - {action.description}{auto_text} - Success Rate: {action.success_probability:.0%}")
    
    print("\n" + "="*50 + "\n")


def demo_model_loading_error():
    """Demonstrate handling model loading errors"""
    print("=== Model Loading Error Demo ===")
    
    # Simulate a model loading error
    error = FileNotFoundError("Model file not found: /models/wan22/model.safetensors")
    model_path = "/models/wan22/model.safetensors"
    
    user_error = handle_model_loading_error(error, model_path)
    
    print(f"Error Category: {user_error.category.value}")
    print(f"Severity: {user_error.severity.value}")
    print(f"Title: {user_error.title}")
    print(f"Message: {user_error.message}")
    print("\nRecovery Suggestions:")
    for i, suggestion in enumerate(user_error.recovery_suggestions[:3], 1):
        print(f"  {i}. {suggestion}")
    
    print(f"\nError Code: {user_error.error_code}")
    
    print("\n" + "="*50 + "\n")


def demo_generation_pipeline_error():
    """Demonstrate handling generation pipeline errors"""
    print("=== Generation Pipeline Error Demo ===")
    
    # Simulate a generation pipeline error
    error = RuntimeError("Generation failed: tensor dimension mismatch")
    context = {
        "prompt": "A beautiful landscape",
        "resolution": "720p",
        "steps": 25,
        "model": "wan22",
        "generation_mode": "T2V"
    }
    
    user_error = handle_generation_error(error, context)
    
    print(f"Error Category: {user_error.category.value}")
    print(f"Severity: {user_error.severity.value}")
    print(f"Title: {user_error.title}")
    print(f"Message: {user_error.message}")
    print("\nRecovery Suggestions:")
    for i, suggestion in enumerate(user_error.recovery_suggestions[:3], 1):
        print(f"  {i}. {suggestion}")
    
    print("\n" + "="*50 + "\n")


def demo_html_output():
    """Demonstrate HTML output for UI integration"""
    print("=== HTML Output Demo ===")
    
    # Create an error for HTML demonstration
    error = RuntimeError("CUDA out of memory")
    context = {"resolution": "1080p", "steps": 40}
    
    user_error = handle_vram_error(error, context)
    html_output = user_error.to_html()
    
    print("HTML Output for UI Integration:")
    print(html_output)
    
    print("\n" + "="*50 + "\n")


def demo_error_categorization():
    """Demonstrate error categorization capabilities"""
    print("=== Error Categorization Demo ===")
    
    handler = GenerationErrorHandler()
    
    test_errors = [
        ("Invalid input provided", "Input validation issue"),
        ("Model not found", "Model loading issue"),
        ("CUDA out of memory", "VRAM/memory issue"),
        ("Generation failed", "Pipeline issue"),
        ("Permission denied", "File system issue"),
        ("Network connection failed", "Network issue"),
        ("Some random error", "Unknown issue")
    ]
    
    print("Error Message → Category")
    print("-" * 40)
    
    for error_msg, description in test_errors:
        error = Exception(error_msg)
        user_error = handler.handle_error(error)
        print(f"{error_msg:<25} → {user_error.category.value}")
    
    print("\n" + "="*50 + "\n")


def main():
    """Run all error handling demonstrations"""
    print("Enhanced Error Handling System Demonstration")
    print("=" * 50)
    print()
    
    demo_input_validation_error()
    demo_vram_error()
    demo_model_loading_error()
    demo_generation_pipeline_error()
    demo_html_output()
    demo_error_categorization()
    
    print("Demonstration completed!")
    print("\nKey Features Demonstrated:")
    print("✓ Automatic error categorization")
    print("✓ User-friendly error messages")
    print("✓ Context-aware recovery suggestions")
    print("✓ Automatic recovery mechanisms")
    print("✓ HTML output for UI integration")
    print("✓ Comprehensive logging and debugging")


if __name__ == "__main__":
    main()
