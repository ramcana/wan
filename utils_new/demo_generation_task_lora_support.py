"""
Demonstration of enhanced GenerationTask with LoRA support
Shows how to use the new LoRA-related functionality
"""

import sys
import json
from datetime import datetime

# Import the minimal version for demonstration
from test_generation_task_lora_support_minimal import GenerationTask, TaskStatus


def demo_basic_lora_functionality():
    """Demonstrate basic LoRA functionality"""
    print("=== Basic LoRA Functionality Demo ===\n")
    
    # Create a new generation task
    task = GenerationTask(
        model_type="t2v-A14B",
        prompt="A beautiful anime-style sunset over mountains with cinematic lighting",
        resolution="1280x720",
        steps=50
    )
    
    print(f"Created task: {task.id}")
    print(f"Prompt: {task.prompt}")
    print(f"Initial LoRA selections: {len(task.selected_loras)}")
    
    # Add some LoRA selections
    print("\n--- Adding LoRA selections ---")
    
    loras_to_add = [
        ("anime_style_v2", 0.8),
        ("detail_enhancer", 0.6),
        ("cinematic_lighting", 0.4),
        ("color_grading", 0.3)
    ]
    
    for lora_name, strength in loras_to_add:
        success = task.add_lora_selection(lora_name, strength)
        print(f"Added '{lora_name}' with strength {strength}: {'✓' if success else '✗'}")
    
    print(f"\nTotal LoRAs selected: {len(task.selected_loras)}")
    print(f"Selected LoRAs: {task.selected_loras}")
    
    # Validate selections
    print("\n--- Validating LoRA selections ---")
    is_valid, errors = task.validate_lora_selections()
    print(f"Validation result: {'✓ Valid' if is_valid else '✗ Invalid'}")
    if errors:
        for error in errors:
            print(f"  - Error: {error}")
    
    return task


def demo_lora_metrics():
    """Demonstrate LoRA metrics functionality"""
    print("\n=== LoRA Metrics Demo ===\n")
    
    task = GenerationTask(
        model_type="i2v-A14B",
        prompt="Transform this image with anime style"
    )
    
    # Add some LoRAs
    task.add_lora_selection("anime_transformer", 1.0)
    task.add_lora_selection("style_enhancer", 0.7)
    
    # Simulate LoRA metrics (normally this would come from the generation pipeline)
    memory_usage = 384.2  # MB
    load_time = 8.5  # seconds
    metadata = {
        "applied_loras": ["anime_transformer", "style_enhancer"],
        "total_parameters": 2500000,
        "model_compatibility": "i2v-A14B",
        "optimization_level": "bf16",
        "load_timestamp": datetime.now().isoformat()
    }
    
    # Update metrics
    task.update_lora_metrics(memory_usage, load_time, metadata)
    
    print(f"LoRA memory usage: {task.lora_memory_usage} MB")
    print(f"LoRA load time: {task.lora_load_time} seconds")
    print(f"LoRA metadata keys: {list(task.lora_metadata.keys())}")
    
    # Get summary
    summary = task.get_lora_summary()
    print(f"\n--- LoRA Summary ---")
    print(f"Selected count: {summary['selected_count']}")
    print(f"Memory usage: {summary['memory_usage_mb']} MB")
    print(f"Load time: {summary['load_time_seconds']} seconds")
    print(f"Is valid: {summary['is_valid']}")
    
    return task


def demo_validation_edge_cases():
    """Demonstrate validation edge cases"""
    print("\n=== Validation Edge Cases Demo ===\n")
    
    task = GenerationTask()
    
    # Test maximum LoRA limit
    print("--- Testing maximum LoRA limit (5) ---")
    for i in range(6):  # Try to add 6 LoRAs
        lora_name = f"test_lora_{i}"
        success = task.add_lora_selection(lora_name, 0.5)
        print(f"Adding LoRA {i+1}: {'✓' if success else '✗ (limit reached)'}")
    
    print(f"Final LoRA count: {len(task.selected_loras)}")
    
    # Test invalid strength values
    print("\n--- Testing invalid strength values ---")
    invalid_strengths = [-0.5, 2.5, "invalid", None]
    
    for i, strength in enumerate(invalid_strengths):
        success = task.add_lora_selection(f"invalid_test_{i}", strength)
        print(f"Adding LoRA with strength {strength}: {'✓' if success else '✗ (invalid)'}")
    
    # Test validation with invalid data
    print("\n--- Testing validation with manually set invalid data ---")
    task.selected_loras["invalid_strength"] = 3.0  # Manually set invalid strength
    task.selected_loras[""] = 0.5  # Empty name
    
    is_valid, errors = task.validate_lora_selections()
    print(f"Validation result: {'✓ Valid' if is_valid else '✗ Invalid'}")
    for error in errors:
        print(f"  - {error}")


def demo_serialization():
    """Demonstrate task serialization with LoRA data"""
    print("\n=== Serialization Demo ===\n")
    
    # Create a task with LoRA data
    task = GenerationTask(
        model_type="ti2v-5B",
        prompt="Create a video with anime style and cinematic effects",
        resolution="1920x1080",
        steps=75
    )
    
    # Add LoRAs and metrics
    task.add_lora_selection("anime_master_v3", 0.9)
    task.add_lora_selection("cinematic_fx", 0.6)
    task.update_lora_metrics(512.8, 15.2, {
        "optimization": "int8",
        "compatibility_score": 0.95
    })
    
    # Serialize to dictionary
    task_dict = task.to_dict()
    
    print("--- Serialized Task Data ---")
    print(f"Task ID: {task_dict['id']}")
    print(f"Model: {task_dict['model_type']}")
    print(f"Selected LoRAs: {task_dict['selected_loras']}")
    print(f"LoRA Memory: {task_dict['lora_memory_usage']} MB")
    print(f"LoRA Load Time: {task_dict['lora_load_time']} seconds")
    print(f"LoRA Metadata: {task_dict['lora_metadata']}")
    
    # Convert to JSON (for API/storage)
    try:
        json_str = json.dumps(task_dict, indent=2)
        print(f"\n--- JSON Serialization ---")
        print("✓ Successfully serialized to JSON")
        print(f"JSON size: {len(json_str)} characters")
        
        # Show a snippet
        lines = json_str.split('\n')
        print("JSON snippet (first 10 lines):")
        for line in lines[:10]:
            print(f"  {line}")
        if len(lines) > 10:
            print(f"  ... ({len(lines) - 10} more lines)")
            
    except Exception as e:
        print(f"✗ JSON serialization failed: {e}")


def demo_backward_compatibility():
    """Demonstrate backward compatibility with old LoRA fields"""
    print("\n=== Backward Compatibility Demo ===\n")
    
    # Create task with old-style LoRA fields
    old_style_task = GenerationTask(
        model_type="t2v-A14B",
        prompt="Old style LoRA usage",
        lora_path="/path/to/old_lora.safetensors",
        lora_strength=0.75
    )
    
    print("--- Old-style LoRA fields ---")
    print(f"LoRA path: {old_style_task.lora_path}")
    print(f"LoRA strength: {old_style_task.lora_strength}")
    
    # Show that new fields are also available
    print(f"New selected_loras field: {old_style_task.selected_loras}")
    print(f"New lora_memory_usage field: {old_style_task.lora_memory_usage}")
    
    # Add new-style LoRA selections
    old_style_task.add_lora_selection("new_style_lora", 0.8)
    
    print("\n--- After adding new-style LoRA ---")
    print(f"Old LoRA path: {old_style_task.lora_path}")
    print(f"Old LoRA strength: {old_style_task.lora_strength}")
    print(f"New LoRA selections: {old_style_task.selected_loras}")
    
    # Serialization includes both
    task_dict = old_style_task.to_dict()
    print(f"\n--- Serialization includes both styles ---")
    print(f"lora_path: {task_dict['lora_path']}")
    print(f"lora_strength: {task_dict['lora_strength']}")
    print(f"selected_loras: {task_dict['selected_loras']}")


def main():
    """Run all demonstrations"""
    print("🎨 GenerationTask LoRA Support Demonstration\n")
    print("This demo shows the enhanced GenerationTask class with LoRA support.")
    print("Features demonstrated:")
    print("  ✓ Multiple LoRA selection (up to 5 LoRAs)")
    print("  ✓ Strength validation (0.0-2.0 range)")
    print("  ✓ Memory usage and load time tracking")
    print("  ✓ LoRA metadata storage")
    print("  ✓ Enhanced serialization")
    print("  ✓ Backward compatibility")
    print("  ✓ Comprehensive validation")
    print()
    
    try:
        # Run all demos
        task1 = demo_basic_lora_functionality()
        task2 = demo_lora_metrics()
        demo_validation_edge_cases()
        demo_serialization()
        demo_backward_compatibility()
        
        print("\n=== Summary ===")
        print("✓ All demonstrations completed successfully!")
        print(f"✓ Task 1 has {len(task1.selected_loras)} LoRAs selected")
        print(f"✓ Task 2 uses {task2.lora_memory_usage} MB of LoRA memory")
        print("✓ Enhanced GenerationTask is ready for production use")
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
