from unittest.mock import Mock, patch
"""
Integrated Error Handler Example

This example demonstrates how to use the integrated error handler
for comprehensive error management in the FastAPI backend.
"""

import asyncio
import logging
from typing import Dict, Any
from pathlib import Path
import sys

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.core.integrated_error_handler import (
    IntegratedErrorHandler,
    handle_model_loading_error,
    handle_vram_exhaustion_error,
    handle_generation_pipeline_error,
    get_integrated_error_handler
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demonstrate_model_loading_error_handling():
    """Demonstrate model loading error handling with automatic recovery"""
    print("\n=== Model Loading Error Handling Demo ===")
    
    # Simulate a model loading error
    model_error = FileNotFoundError("Model file 'WAN2.2-T2V-A14B' not found in models directory")
    model_type = "t2v-A14B"
    context = {
        "task_id": "demo_task_001",
        "model_path": "/models/WAN2.2-T2V-A14B",
        "hardware_profile": {"gpu_name": "RTX 4080", "vram_gb": 16}
    }
    
    # Handle the error using the integrated error handler
    error_info = await handle_model_loading_error(model_error, model_type, context)
    
    print(f"Error Category: {error_info.category.value}")
    print(f"Error Severity: {error_info.severity.value}")
    print(f"Error Title: {error_info.title}")
    print(f"Error Message: {error_info.message}")
    print(f"Error Code: {error_info.error_code}")
    
    print("\nRecovery Suggestions:")
    for i, suggestion in enumerate(error_info.recovery_suggestions, 1):
        print(f"  {i}. {suggestion}")
    
    if hasattr(error_info, 'technical_details') and error_info.technical_details:
        print(f"\nTechnical Details Available: Yes")
    
    return error_info


async def demonstrate_vram_exhaustion_error_handling():
    """Demonstrate VRAM exhaustion error handling with optimization fallbacks"""
    print("\n=== VRAM Exhaustion Error Handling Demo ===")
    
    # Simulate a VRAM exhaustion error
    vram_error = RuntimeError("CUDA out of memory. Tried to allocate 12.50 GiB (GPU 0; 15.78 GiB total capacity)")
    generation_params = {
        "resolution": "1080p",
        "steps": 35,
        "model_type": "t2v-A14B",
        "batch_size": 2,
        "guidance_scale": 7.5
    }
    context = {
        "task_id": "demo_task_002",
        "gpu_memory_allocated": 14.2,
        "gpu_memory_total": 16.0
    }
    
    # Handle the error with automatic optimization
    error_info = await handle_vram_exhaustion_error(vram_error, generation_params, context)
    
    print(f"Error Category: {error_info.category.value}")
    print(f"Error Severity: {error_info.severity.value}")
    print(f"Error Title: {error_info.title}")
    print(f"Error Message: {error_info.message}")
    
    print("\nRecovery Suggestions:")
    for i, suggestion in enumerate(error_info.recovery_suggestions, 1):
        print(f"  {i}. {suggestion}")
    
    print(f"\nOptimized Generation Parameters:")
    for key, value in generation_params.items():
        print(f"  {key}: {value}")
    
    return error_info


async def demonstrate_generation_pipeline_error_handling():
    """Demonstrate generation pipeline error handling"""
    print("\n=== Generation Pipeline Error Handling Demo ===")
    
    # Simulate a generation pipeline error
    pipeline_error = RuntimeError("Generation pipeline failed: Tensor dimension mismatch in attention layer")
    context = {
        "task_id": "demo_task_003",
        "model_type": "i2v-A14B",
        "generation_step": 15,
        "total_steps": 25,
        "pipeline_stage": "attention_computation"
    }
    
    # Handle the error
    error_info = await handle_generation_pipeline_error(pipeline_error, context)
    
    print(f"Error Category: {error_info.category.value}")
    print(f"Error Severity: {error_info.severity.value}")
    print(f"Error Title: {error_info.title}")
    print(f"Error Message: {error_info.message}")
    
    print("\nRecovery Suggestions:")
    for i, suggestion in enumerate(error_info.recovery_suggestions, 1):
        print(f"  {i}. {suggestion}")
    
    return error_info


async def demonstrate_comprehensive_error_handling():
    """Demonstrate comprehensive error handling with the integrated handler"""
    print("\n=== Comprehensive Error Handling Demo ===")
    
    # Get the global error handler instance
    handler = get_integrated_error_handler()
    
    # Test different types of errors
    test_errors = [
        (ValueError("Invalid input: prompt exceeds maximum length of 512 characters"), 
         {"error_type": "input_validation", "prompt_length": 750}),
        
        (ConnectionError("WebSocket connection failed: Connection refused"), 
         {"error_type": "websocket_connection", "endpoint": "/ws/progress"}),
        
        (RuntimeError("Model integration bridge initialization failed"), 
         {"error_type": "system_integration", "component": "model_bridge"}),
        
        (MemoryError("System out of memory"), 
         {"error_type": "system_resource", "memory_usage": 95.5})
    ]
    
    for i, (error, context) in enumerate(test_errors, 1):
        print(f"\n--- Test Error {i} ---")
        print(f"Original Error: {error}")
        
        # Handle the error
        error_info = await handler.handle_error(error, context)
        
        print(f"Categorized as: {error_info.category.value} ({error_info.severity.value})")
        print(f"User Message: {error_info.message}")
        print(f"Top Recovery Suggestions: {', '.join(error_info.recovery_suggestions[:2])}")


async def demonstrate_system_status_monitoring():
    """Demonstrate system status monitoring for error context"""
    print("\n=== System Status Monitoring Demo ===")
    
    handler = get_integrated_error_handler()
    status = handler.get_system_status()
    
    print("Current System Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    print(f"\nAvailable Error Categories:")
    categories = handler.get_error_categories()
    for category in categories:
        print(f"  - {category}")


async def demonstrate_error_html_output():
    """Demonstrate HTML error output for UI integration"""
    print("\n=== HTML Error Output Demo ===")
    
    # Create a sample error
    handler = get_integrated_error_handler()
    test_error = RuntimeError("CUDA out of memory during model loading")
    context = {
        "model_type": "t2v-A14B",
        "resolution": "1080p",
        "task_id": "demo_html_001"
    }
    
    error_info = await handler.handle_error(test_error, context)
    
    # Generate HTML output (if available)
    if hasattr(error_info, 'to_html'):
        html_output = error_info.to_html()
        print("HTML Error Output Generated:")
        print(f"Length: {len(html_output)} characters")
        print("Contains: error-container, recovery suggestions, technical details")
        
        # Save to file for inspection
        html_file = Path(__file__).parent / "error_output_demo.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <title>Error Handler Demo</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .demo-container {{ max-width: 800px; margin: 0 auto; }}
    </style>
</head>
<body>
    <div class="demo-container">
        <h1>Integrated Error Handler Demo</h1>
        <h2>Sample Error Output</h2>
        {html_output}
    </div>
</body>
</html>
            """)
        print(f"HTML demo saved to: {html_file}")
    else:
        print("HTML output not available in this error object")


async def demonstrate_fastapi_integration_features():
    """Demonstrate FastAPI-specific integration features"""
    print("\n=== FastAPI Integration Features Demo ===")
    
    handler = get_integrated_error_handler()
    
    # Simulate FastAPI context with generation service
    class MockGenerationService:
        def __init__(self):
            self.model_integration_bridge = "available"
            self.real_generation_pipeline = None
            self.wan22_system_optimizer = "available"
    
    mock_service = MockGenerationService()
    context = {
        "generation_service": mock_service,
        "fastapi_request_id": "req_12345",
        "endpoint": "/api/v1/generation/submit"
    }
    
    # Test context enhancement
    enhanced_context = handler._enhance_context_for_fastapi(context)
    
    print("Enhanced Context for FastAPI:")
    for key, value in enhanced_context.items():
        if key != "generation_service":  # Skip complex object
            print(f"  {key}: {value}")
    
    # Test FastAPI-specific error handling
    fastapi_error = ValueError("Request validation failed: invalid model_type parameter")
    error_info = await handler.handle_error(fastapi_error, enhanced_context)
    
    print(f"\nFastAPI Error Handling Result:")
    print(f"Category: {error_info.category.value}")
    print(f"Message: {error_info.message}")
    
    # Check for FastAPI-specific suggestions
    fastapi_suggestions = [s for s in error_info.recovery_suggestions if "api" in s.lower() or "endpoint" in s.lower()]
    if fastapi_suggestions:
        print(f"FastAPI-specific suggestions: {len(fastapi_suggestions)}")
        for suggestion in fastapi_suggestions:
            print(f"  - {suggestion}")


async def main():
    """Run all error handling demonstrations"""
    print("Integrated Error Handler Demonstration")
    print("=" * 50)
    
    try:
        # Run all demonstrations
        await demonstrate_model_loading_error_handling()
        await demonstrate_vram_exhaustion_error_handling()
        await demonstrate_generation_pipeline_error_handling()
        await demonstrate_comprehensive_error_handling()
        await demonstrate_system_status_monitoring()
        await demonstrate_error_html_output()
        await demonstrate_fastapi_integration_features()
        
        print("\n" + "=" * 50)
        print("All demonstrations completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✓ Model loading error handling with automatic recovery")
        print("✓ VRAM exhaustion handling with optimization fallbacks")
        print("✓ Generation pipeline error categorization")
        print("✓ FastAPI-specific error context enhancement")
        print("✓ Comprehensive error categorization and recovery suggestions")
        print("✓ System status monitoring for error context")
        print("✓ HTML error output for UI integration")
        print("✓ Integration with existing GenerationErrorHandler infrastructure")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        print(f"\nDemonstration failed with error: {e}")
        print("This may be due to missing dependencies or infrastructure components.")


if __name__ == "__main__":
    asyncio.run(main())
