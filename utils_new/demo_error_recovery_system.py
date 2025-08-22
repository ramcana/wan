"""
Demo script for the WAN22 Error Recovery System

This script demonstrates the comprehensive error recovery system capabilities
including error handler registration, automatic recovery attempts, system state
preservation, advanced logging, and user-guided recovery workflows.
"""

import time
from datetime import datetime
from error_recovery_system import ErrorRecoverySystem, RecoveryStrategy, RecoveryResult
from recovery_workflows import (
    AdvancedLogger, LogLevel, RecoveryWorkflowManager, 
    get_advanced_logger, get_workflow_manager
)


def demo_basic_error_recovery():
    """Demonstrate basic error recovery functionality"""
    print("=" * 60)
    print("DEMO: Basic Error Recovery System")
    print("=" * 60)
    
    # Initialize recovery system
    recovery_system = ErrorRecoverySystem(
        state_dir="demo_states",
        log_dir="demo_logs",
        max_recovery_attempts=3,
        enable_auto_recovery=True
    )
    
    print("✓ Error Recovery System initialized")
    
    # Register custom error handler
    def custom_memory_handler(error, context):
        print(f"  → Custom handler processing {type(error).__name__}")
        return RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.FALLBACK_CONFIG,
            actions_taken=["Applied memory optimization", "Reduced batch size"],
            time_taken=0.1,
            error_resolved=True,
            fallback_applied=True,
            user_intervention_required=False,
            recovery_message="Memory optimized successfully",
            warnings=[]
        )
    
    recovery_system.register_error_handler(
        MemoryError, custom_memory_handler, RecoveryStrategy.FALLBACK_CONFIG
    )
    print("✓ Custom memory error handler registered")
    
    # Simulate memory error and recovery
    print("\n📋 Simulating memory error...")
    test_error = MemoryError("CUDA out of memory")
    result = recovery_system.attempt_recovery(test_error, component="model_loader")
    
    print(f"✓ Recovery attempt completed:")
    print(f"  - Success: {result.success}")
    print(f"  - Strategy: {result.strategy_used.value}")
    print(f"  - Actions: {', '.join(result.actions_taken)}")
    print(f"  - Error resolved: {result.error_resolved}")
    print(f"  - Time taken: {result.time_taken:.3f}s")
    
    # Show recovery statistics
    stats = recovery_system.get_recovery_statistics()
    print(f"\n📊 Recovery Statistics:")
    print(f"  - Total attempts: {stats['total_recovery_attempts']}")
    print(f"  - Registered handlers: {len(stats['registered_handlers'])}")
    print(f"  - State files: {stats['state_files_count']}")


def demo_advanced_logging():
    """Demonstrate advanced logging capabilities"""
    print("\n" + "=" * 60)
    print("DEMO: Advanced Logging System")
    print("=" * 60)
    
    # Get advanced logger
    logger = get_advanced_logger()
    print("✓ Advanced Logger initialized")
    
    # Log different types of events
    print("\n📝 Logging various events...")
    
    # Log error with context
    test_error = ValueError("Invalid configuration parameter")
    logger.log_with_context(
        level=LogLevel.ERROR,
        component="config_validator",
        message="Configuration validation failed",
        error=test_error,
        user_context={"user_id": "demo_user", "action": "model_loading"},
        recovery_context={"attempt": 1, "strategy": "fallback_config"}
    )
    print("✓ Error logged with full context")
    
    # Log recovery attempt
    logger.log_recovery_attempt(
        component="quantization_controller",
        error=test_error,
        recovery_actions=["Disabled quantization", "Applied fallback settings"],
        success=True
    )
    print("✓ Recovery attempt logged")
    
    # Log user action
    logger.log_user_action(
        component="ui",
        action="model_selection",
        context={"model": "TI2V-5B", "quantization": "bf16", "vram": "16GB"}
    )
    print("✓ User action logged")
    
    # Show log statistics
    stats = logger.get_log_statistics()
    print(f"\n📊 Logging Statistics:")
    print(f"  - Total log files: {stats['total_files']}")
    print(f"  - Total size: {stats['total_size_mb']:.2f} MB")


def demo_recovery_workflows():
    """Demonstrate user-guided recovery workflows"""
    print("\n" + "=" * 60)
    print("DEMO: User-Guided Recovery Workflows")
    print("=" * 60)
    
    # Get workflow manager
    workflow_manager = get_workflow_manager()
    print("✓ Recovery Workflow Manager initialized")
    
    # Show available workflows
    print(f"\n📋 Available workflows: {len(workflow_manager.workflows)}")
    for workflow_id, workflow in workflow_manager.workflows.items():
        print(f"  - {workflow_id}: {workflow.title}")
        print(f"    Estimated time: {workflow.estimated_time} minutes")
        print(f"    Difficulty: {workflow.difficulty_level}")
    
    # Find applicable workflows for an error
    error_message = "VRAM detection failed on RTX 4080"
    applicable = workflow_manager.find_applicable_workflows(error_message)
    print(f"\n🔍 Workflows for '{error_message}':")
    for workflow_id in applicable:
        print(f"  - {workflow_id}")
    
    # Start a workflow
    if applicable:
        workflow_id = applicable[0]
        execution_id = workflow_manager.start_workflow(
            workflow_id, 
            {"error": error_message}
        )
        print(f"\n🚀 Started workflow: {execution_id}")
        
        # Show current step
        current_step = workflow_manager.get_current_step(execution_id)
        if current_step:
            print(f"📋 Current step: {current_step.title}")
            print(f"  Description: {current_step.description}")
            print(f"  Instructions:")
            for i, instruction in enumerate(current_step.instructions, 1):
                print(f"    {i}. {instruction}")
        
        # Simulate step completion
        print(f"\n✅ Simulating step completion...")
        continues = workflow_manager.complete_step(
            execution_id,
            success=True,
            user_response={"gpu_detected": True, "vram_amount": "16GB"},
            notes="GPU detected successfully via nvidia-smi"
        )
        
        # Show progress
        progress = workflow_manager.get_workflow_progress(execution_id)
        print(f"📊 Workflow Progress:")
        print(f"  - Status: {progress['status']}")
        print(f"  - Progress: {progress['progress_percent']:.1f}%")
        print(f"  - Current step: {progress['current_step']}/{progress['total_steps']}")
        print(f"  - Completed steps: {progress['completed_steps']}")


def demo_system_state_management():
    """Demonstrate system state preservation and restoration"""
    print("\n" + "=" * 60)
    print("DEMO: System State Management")
    print("=" * 60)
    
    recovery_system = ErrorRecoverySystem(
        state_dir="demo_states",
        log_dir="demo_logs"
    )
    
    # Create and save a system state
    from error_recovery_system import SystemState
    
    test_state = SystemState(
        timestamp=datetime.now(),
        active_model="TI2V-5B",
        configuration={
            "quantization": "bf16",
            "vram_limit": 16000,
            "batch_size": 1
        },
        memory_usage={
            "vram_used": 12000,
            "vram_total": 16000,
            "system_ram": 8000
        },
        gpu_state={
            "temperature": 72,
            "utilization": 85,
            "power_draw": 280
        },
        pipeline_state={
            "model_loaded": True,
            "quantization_applied": True,
            "optimization_level": "high"
        },
        user_preferences={
            "auto_quantization": True,
            "memory_optimization": True,
            "error_recovery": True
        }
    )
    
    print("📋 Creating system state snapshot...")
    state_path = recovery_system.save_system_state(test_state, "demo_state")
    print(f"✓ State saved to: {state_path}")
    
    # Restore the state
    print("\n🔄 Restoring system state...")
    result = recovery_system.restore_system_state(state_path)
    
    print(f"✓ State restoration:")
    print(f"  - Success: {result.success}")
    print(f"  - Actions taken: {', '.join(result.actions_taken)}")
    print(f"  - Time taken: {result.time_taken:.3f}s")
    if result.warnings:
        print(f"  - Warnings: {', '.join(result.warnings)}")


def demo_error_scenarios():
    """Demonstrate different error scenarios and recovery strategies"""
    print("\n" + "=" * 60)
    print("DEMO: Error Scenarios and Recovery Strategies")
    print("=" * 60)
    
    recovery_system = ErrorRecoverySystem(
        state_dir="demo_states",
        log_dir="demo_logs"
    )
    
    # Test different error types
    error_scenarios = [
        (MemoryError("CUDA out of memory"), "gpu_manager"),
        (FileNotFoundError("Model file not found"), "model_loader"),
        (ConnectionError("Download failed"), "model_downloader"),
        (ValueError("Invalid quantization setting"), "config_validator"),
        (RuntimeError("Pipeline initialization failed"), "pipeline_manager")
    ]
    
    print("🧪 Testing different error scenarios:")
    
    for i, (error, component) in enumerate(error_scenarios, 1):
        print(f"\n{i}. {type(error).__name__}: {error}")
        print(f"   Component: {component}")
        
        result = recovery_system.attempt_recovery(error, component=component)
        
        print(f"   → Strategy: {result.strategy_used.value}")
        print(f"   → Success: {result.success}")
        print(f"   → Actions: {', '.join(result.actions_taken)}")
        
        if result.user_intervention_required:
            print(f"   ⚠️  User intervention required")
        
        time.sleep(0.1)  # Small delay for demo purposes


def main():
    """Run all demos"""
    print("🚀 WAN22 Error Recovery System Demo")
    print("=" * 60)
    
    try:
        demo_basic_error_recovery()
        demo_advanced_logging()
        demo_recovery_workflows()
        demo_system_state_management()
        demo_error_scenarios()
        
        print("\n" + "=" * 60)
        print("✅ All demos completed successfully!")
        print("=" * 60)
        
        print("\n📁 Generated files:")
        print("  - demo_states/: System state snapshots")
        print("  - demo_logs/: Error recovery logs")
        print("  - logs/: Advanced logging output")
        
        print("\n🔧 Key Features Demonstrated:")
        print("  ✓ Error handler registration and management")
        print("  ✓ Automatic recovery with exponential backoff")
        print("  ✓ System state preservation and restoration")
        print("  ✓ Comprehensive logging with context")
        print("  ✓ Log rotation and cleanup")
        print("  ✓ User-guided recovery workflows")
        print("  ✓ Multiple recovery strategies")
        print("  ✓ Recovery statistics and monitoring")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()