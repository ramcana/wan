"""
Demonstration script for the comprehensive logging and diagnostics system.

This script shows how to use the logging system in a real video generation
workflow, including error handling, diagnostics collection, and log analysis.
"""

import time
import uuid
import random
from datetime import datetime, timedelta
from pathlib import Path

from generation_logger import (
    configure_logger, get_logger, GenerationContext, SystemDiagnostics
)
from diagnostic_collector import get_diagnostic_collector
from log_analyzer import get_log_analyzer


def simulate_generation_session(session_id: str, 
                               model_type: str = "wan22_t2v",
                               should_fail: bool = False):
    """
    Simulate a video generation session with comprehensive logging.
    
    Args:
        session_id: Unique session identifier
        model_type: Type of model to use
        should_fail: Whether to simulate a failure
    """
    logger = get_logger()
    
    # Create generation context
    context = GenerationContext(
        session_id=session_id,
        model_type=model_type,
        generation_mode="T2V",
        prompt="A beautiful sunset over mountains with flowing water",
        parameters={
            "resolution": "720p",
            "steps": 20,
            "guidance_scale": 7.5,
            "seed": random.randint(1, 1000000)
        },
        start_time=time.time()
    )
    
    try:
        with logger.generation_session(context):
            # Stage 1: Input validation
            logger.log_pipeline_stage("validation", "Starting input validation")
            time.sleep(0.5)  # Simulate processing time
            logger.log_pipeline_stage("validation", "Input validation completed successfully")
            
            # Stage 2: Model loading
            logger.log_pipeline_stage("model_loading", "Loading model checkpoint")
            model_load_time = random.uniform(10.0, 20.0)
            time.sleep(0.2)  # Simulate loading time
            
            if should_fail and random.random() < 0.3:
                # Simulate model loading failure
                error_msg = "Model checkpoint corrupted or missing"
                logger.log_model_loading(model_type, f"models/{model_type}", False, model_load_time, error_msg)
                raise FileNotFoundError(error_msg)
            else:
                logger.log_model_loading(model_type, f"models/{model_type}", True, model_load_time)
                logger.log_pipeline_stage("model_loading", "Model loaded successfully")
            
            # Stage 3: VRAM management
            logger.log_pipeline_stage("resource_management", "Checking VRAM availability")
            vram_used = random.uniform(6.0, 10.0)
            vram_total = 12.0
            vram_percentage = (vram_used / vram_total) * 100
            logger.log_vram_usage("pre_generation", vram_used, vram_total, vram_percentage)
            
            if vram_percentage > 85:
                # Simulate parameter optimization
                original_params = context.parameters.copy()
                optimized_params = original_params.copy()
                optimized_params["steps"] = max(15, optimized_params["steps"] - 5)
                logger.log_parameter_optimization(
                    original_params, optimized_params, "VRAM optimization"
                )
                context.parameters = optimized_params
            
            # Stage 4: Generation
            logger.log_pipeline_stage("generation", "Starting video generation")
            generation_time = random.uniform(30.0, 60.0)
            time.sleep(0.3)  # Simulate generation time
            
            # Log VRAM usage during generation
            peak_vram = min(vram_total, vram_used + random.uniform(1.0, 3.0))
            peak_percentage = (peak_vram / vram_total) * 100
            logger.log_vram_usage("generation", peak_vram, vram_total, peak_percentage)
            
            if should_fail and random.random() < 0.4:
                # Simulate generation failure
                if random.random() < 0.5:
                    # VRAM error
                    error_msg = "CUDA out of memory during generation"
                    logger.log_recovery_attempt("VRAMError", "reduce_batch_size", False)
                    raise RuntimeError(error_msg)
                else:
                    # Other generation error
                    error_msg = "Generation pipeline failed unexpectedly"
                    raise RuntimeError(error_msg)
            
            logger.log_pipeline_stage("generation", "Video generation completed")
            
            # Stage 5: Post-processing
            logger.log_pipeline_stage("postprocessing", "Starting post-processing")
            time.sleep(0.1)
            logger.log_pipeline_stage("postprocessing", "Post-processing completed")
            
            # Stage 6: Saving output
            logger.log_pipeline_stage("saving", "Saving generated video")
            time.sleep(0.1)
            logger.log_pipeline_stage("saving", "Video saved successfully")
            
            print(f"✅ Generation session {session_id[:8]} completed successfully")
            
    except Exception as e:
        print(f"❌ Generation session {session_id[:8]} failed: {str(e)}")
        # Error is automatically logged by the context manager


def demonstrate_logging_system():
    """Demonstrate the complete logging and diagnostics system."""
    print("🚀 WAN2.2 Video Generation Logging System Demo")
    print("=" * 50)
    
    # Configure logging system
    log_dir = "demo_logs"
    logger = configure_logger(
        log_dir=log_dir,
        max_log_size=5 * 1024 * 1024,  # 5MB
        backup_count=3,
        log_level="INFO"
    )
    
    print(f"📁 Logs will be stored in: {Path(log_dir).absolute()}")
    print()
    
    # Simulate multiple generation sessions
    print("🎬 Simulating video generation sessions...")
    session_ids = []
    
    for i in range(8):
        session_id = str(uuid.uuid4())
        session_ids.append(session_id)
        
        # Randomly decide if session should fail (30% chance)
        should_fail = random.random() < 0.3
        
        print(f"Session {i+1}/8: {session_id[:8]}...", end=" ")
        simulate_generation_session(session_id, should_fail=should_fail)
        
        # Small delay between sessions
        time.sleep(0.2)
    
    print()
    print("✨ Generation sessions completed!")
    print()
    
    # Collect system diagnostics
    print("🔍 Collecting system diagnostics...")
    collector = get_diagnostic_collector()
    
    # Generate diagnostic report for a specific session
    sample_session_id = session_ids[0]
    diagnostics = collector.collect_full_diagnostics(
        session_id=sample_session_id,
        include_logs=True,
        include_models=True
    )
    
    # Export diagnostics
    diag_report_path = collector.export_diagnostics(
        f"{log_dir}/diagnostic_report.json",
        session_id=sample_session_id,
        format='json'
    )
    
    diag_text_path = collector.export_diagnostics(
        f"{log_dir}/diagnostic_report.txt",
        session_id=sample_session_id,
        format='txt'
    )
    
    print(f"📊 Diagnostic reports exported:")
    print(f"   JSON: {diag_report_path}")
    print(f"   Text: {diag_text_path}")
    print()
    
    # Analyze logs
    print("📈 Analyzing logs...")
    analyzer = get_log_analyzer()
    
    # Analyze logs from the last hour
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=1)
    
    analysis_report = analyzer.analyze_logs(
        start_time=start_time,
        end_time=end_time
    )
    
    # Export analysis report
    analysis_json_path = analyzer.export_analysis_report(
        analysis_report,
        f"{log_dir}/analysis_report.json",
        format='json'
    )
    
    analysis_html_path = analyzer.export_analysis_report(
        analysis_report,
        f"{log_dir}/analysis_report.html",
        format='html'
    )
    
    print(f"📋 Analysis reports exported:")
    print(f"   JSON: {analysis_json_path}")
    print(f"   HTML: {analysis_html_path}")
    print()
    
    # Display summary statistics
    print("📊 Session Summary:")
    print(f"   Total sessions: {analysis_report.total_sessions}")
    print(f"   Successful: {analysis_report.successful_sessions}")
    print(f"   Failed: {analysis_report.failed_sessions}")
    print(f"   Incomplete: {analysis_report.incomplete_sessions}")
    print(f"   Average duration: {analysis_report.average_duration:.2f}s")
    print()
    
    if analysis_report.error_patterns:
        print("🚨 Error Patterns:")
        for error_type, count in analysis_report.error_patterns.items():
            print(f"   {error_type}: {count}")
        print()
    
    if analysis_report.model_usage:
        print("🤖 Model Usage:")
        for model_type, count in analysis_report.model_usage.items():
            print(f"   {model_type}: {count}")
        print()
    
    # Show system diagnostics summary
    print("💻 System Status:")
    sys_diag = diagnostics['system']
    print(f"   CPU Usage: {sys_diag['cpu_usage']:.1f}%")
    print(f"   Memory Usage: {sys_diag['memory_usage']:.1f}%")
    print(f"   CUDA Available: {sys_diag['cuda_available']}")
    if sys_diag.get('gpu_memory_used'):
        print(f"   GPU Memory: {sys_diag['gpu_memory_used']:.1f}GB / {sys_diag['gpu_memory_total']:.1f}GB")
    print()
    
    # Demonstrate log retrieval for specific session
    print(f"🔍 Session Details for {sample_session_id[:8]}:")
    session_logs = logger.get_session_logs(sample_session_id)
    
    for log_type, logs in session_logs.items():
        if logs:
            print(f"   {log_type.title()} logs: {len(logs)} entries")
    print()
    
    print("🎉 Logging system demonstration completed!")
    print(f"📁 Check the '{log_dir}' directory for all generated files.")


def demonstrate_error_scenarios():
    """Demonstrate logging of various error scenarios."""
    print("\n🚨 Demonstrating Error Scenario Logging")
    print("-" * 40)
    
    logger = get_logger()
    
    # Scenario 1: VRAM exhaustion with recovery
    print("Scenario 1: VRAM exhaustion with recovery attempt")
    session_id = str(uuid.uuid4())
    context = GenerationContext(
        session_id=session_id,
        model_type="wan22_t2v",
        generation_mode="T2V",
        prompt="High resolution video generation",
        parameters={"resolution": "1080p", "steps": 50},
        start_time=time.time()
    )
    
    try:
        with logger.generation_session(context):
            logger.log_pipeline_stage("validation", "Input validation completed")
            logger.log_vram_usage("pre_generation", 11.5, 12.0, 95.8)
            
            # Attempt recovery
            logger.log_recovery_attempt("VRAMError", "reduce_resolution", True)
            optimized_params = {"resolution": "720p", "steps": 30}
            logger.log_parameter_optimization(
                context.parameters, optimized_params, "VRAM exhaustion recovery"
            )
            
            logger.log_pipeline_stage("generation", "Generation completed with optimized parameters")
            print("   ✅ Recovery successful")
    except Exception as e:
        print(f"   ❌ Recovery failed: {e}")
    
    # Scenario 2: Model loading failure
    print("\nScenario 2: Model loading failure")
    session_id = str(uuid.uuid4())
    context = GenerationContext(
        session_id=session_id,
        model_type="missing_model",
        generation_mode="T2V",
        prompt="Test prompt",
        parameters={"resolution": "720p"},
        start_time=time.time()
    )
    
    try:
        with logger.generation_session(context):
            logger.log_pipeline_stage("validation", "Input validation completed")
            logger.log_model_loading("missing_model", "models/missing_model", False, 5.0, "Model file not found")
            raise FileNotFoundError("Model checkpoint not found")
    except FileNotFoundError:
        print("   ❌ Model loading failed as expected")
    
    print("\n✨ Error scenario demonstration completed!")


if __name__ == "__main__":
    # Set random seed for reproducible demo
    random.seed(42)
    
    # Run main demonstration
    demonstrate_logging_system()
    
    # Demonstrate error scenarios
    demonstrate_error_scenarios()
    
    print("\n" + "="*60)
    print("Demo completed! Check the generated log files and reports.")
    print("You can now integrate this logging system into your video generation pipeline.")
