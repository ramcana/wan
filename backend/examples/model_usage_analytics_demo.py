"""
Model Usage Analytics System Demo
Demonstrates the functionality of the model usage analytics system including
tracking, analysis, and recommendation generation.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from core.model_usage_analytics import (
    ModelUsageAnalytics, UsageEventType, UsageData,
    track_generation_usage, get_model_usage_analytics
)
from services.generation_service_analytics_integration import (
    GenerationServiceAnalyticsIntegration,
    get_usage_statistics_for_model,
    get_cleanup_recommendations,
    get_preload_recommendations,
    generate_analytics_report
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def simulate_model_usage():
    """Simulate realistic model usage patterns for demonstration"""
    logger.info("Starting model usage simulation...")
    
    # Initialize analytics system
    analytics = await get_model_usage_analytics(models_dir="demo_models")
    
    # Simulate usage for different models with different patterns
    models = {
        "t2v-A14B": {"frequency": 3.0, "success_rate": 0.95, "avg_time": 120.0},
        "i2v-A14B": {"frequency": 1.5, "success_rate": 0.90, "avg_time": 150.0},
        "ti2v-5B": {"frequency": 0.3, "success_rate": 0.85, "avg_time": 90.0}
    }
    
    # Simulate 30 days of usage
    for day in range(30):
        current_date = datetime.now() - timedelta(days=day)
        
        for model_id, config in models.items():
            # Determine if model is used today based on frequency
            daily_probability = config["frequency"] / 7.0  # Convert weekly to daily
            
            if day % int(1.0 / daily_probability) == 0:  # Simple pattern
                # Simulate generation request
                generation_params = {
                    "prompt": f"Demo prompt for {model_id} on day {day}",
                    "resolution": "1280x720" if day % 2 == 0 else "1920x1080",
                    "steps": 50 if day % 3 == 0 else 30,
                    "lora_strength": 1.0
                }
                
                # Track generation request
                await track_generation_usage(
                    model_id=model_id,
                    event_type=UsageEventType.GENERATION_REQUEST,
                    generation_params=generation_params
                )
                
                # Track generation start
                await track_generation_usage(
                    model_id=model_id,
                    event_type=UsageEventType.GENERATION_START,
                    generation_params=generation_params
                )
                
                # Simulate generation completion or failure
                success = day % int(1.0 / (1.0 - config["success_rate"])) != 0
                
                if success:
                    # Successful generation
                    generation_time = config["avg_time"] + (day % 20 - 10) * 5  # Add some variance
                    
                    await track_generation_usage(
                        model_id=model_id,
                        event_type=UsageEventType.GENERATION_COMPLETE,
                        duration_seconds=generation_time,
                        generation_params=generation_params,
                        performance_metrics={
                            "vram_usage_mb": 8000 + (day % 1000),
                            "generation_speed": 120.0 / generation_time
                        }
                    )
                else:
                    # Failed generation
                    await track_generation_usage(
                        model_id=model_id,
                        event_type=UsageEventType.GENERATION_FAILED,
                        duration_seconds=config["avg_time"] * 0.3,  # Partial time before failure
                        error_message="Simulated generation failure",
                        generation_params=generation_params
                    )
    
    logger.info("Model usage simulation completed")


async def demonstrate_analytics_features():
    """Demonstrate various analytics features"""
    logger.info("Demonstrating analytics features...")
    
    # Get usage statistics for each model
    models = ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
    
    print("\n" + "="*60)
    print("MODEL USAGE STATISTICS")
    print("="*60)
    
    for model_id in models:
        stats = await get_usage_statistics_for_model(model_id)
        
        print(f"\n{model_id}:")
        print(f"  Total Uses: {stats.get('total_uses', 0)}")
        print(f"  Uses per Day: {stats.get('uses_per_day', 0):.2f}")
        print(f"  Success Rate: {stats.get('success_rate', 0):.1%}")
        print(f"  Avg Generation Time: {stats.get('average_generation_time', 0):.1f}s")
        print(f"  Peak Hours: {stats.get('peak_usage_hours', [])}")
        print(f"  Common Resolutions: {stats.get('most_common_resolutions', [])}")
    
    # Get cleanup recommendations
    print("\n" + "="*60)
    print("CLEANUP RECOMMENDATIONS")
    print("="*60)
    
    cleanup_recs = await get_cleanup_recommendations()
    
    if cleanup_recs:
        for rec in cleanup_recs:
            print(f"\nModel: {rec['model_id']}")
            print(f"  Reason: {rec['reason']}")
            print(f"  Space Saved: {rec['space_saved_mb']:.1f} MB")
            print(f"  Priority: {rec['priority']}")
            print(f"  Confidence: {rec['confidence_score']:.2f}")
    else:
        print("No cleanup recommendations at this time.")
    
    # Get preload recommendations
    print("\n" + "="*60)
    print("PRELOAD RECOMMENDATIONS")
    print("="*60)
    
    preload_recs = await get_preload_recommendations()
    
    if preload_recs:
        for rec in preload_recs:
            print(f"\nModel: {rec['model_id']}")
            print(f"  Reason: {rec['reason']}")
            print(f"  Usage Frequency: {rec['usage_frequency']:.2f} uses/day")
            print(f"  Priority: {rec['priority']}")
            print(f"  Confidence: {rec['confidence_score']:.2f}")
    else:
        print("No preload recommendations at this time.")


async def demonstrate_comprehensive_report():
    """Demonstrate comprehensive analytics report generation"""
    logger.info("Generating comprehensive analytics report...")
    
    report = await generate_analytics_report()
    
    print("\n" + "="*60)
    print("COMPREHENSIVE ANALYTICS REPORT")
    print("="*60)
    
    print(f"\nReport Date: {report.get('report_date', 'N/A')}")
    print(f"Total Models Tracked: {report.get('total_models_tracked', 0)}")
    print(f"Total Usage Events: {report.get('total_usage_events', 0)}")
    print(f"Storage Usage: {report.get('storage_usage_mb', 0):.1f} MB")
    print(f"Potential Savings: {report.get('estimated_savings_mb', 0):.1f} MB")
    
    # Most used models
    most_used = report.get('most_used_models', [])
    if most_used:
        print(f"\nMost Used Models:")
        for model, count in most_used:
            print(f"  {model}: {count} uses")
    
    # Performance trends
    trends = report.get('performance_trends', {})
    if trends:
        print(f"\nPerformance Trends:")
        for model, trend in trends.items():
            if trend:
                avg_time = sum(trend) / len(trend)
                print(f"  {model}: {avg_time:.1f}s average")
    
    # Performance recommendations
    perf_recs = report.get('performance_recommendations', [])
    if perf_recs:
        print(f"\nPerformance Recommendations:")
        for rec in perf_recs:
            print(f"  {rec['model_id']}: {rec['recommendation']}")


async def demonstrate_integration_features():
    """Demonstrate generation service integration features"""
    logger.info("Demonstrating generation service integration...")
    
    # Initialize integration
    integration = GenerationServiceAnalyticsIntegration()
    await integration.initialize()
    
    print("\n" + "="*60)
    print("GENERATION SERVICE INTEGRATION DEMO")
    print("="*60)
    
    # Simulate a complete generation workflow
    model_id = "t2v-A14B"
    generation_params = {
        "prompt": "Integration demo generation",
        "resolution": "1280x720",
        "steps": 50,
        "lora_strength": 1.0
    }
    
    print(f"\nSimulating generation workflow for {model_id}...")
    
    # Track generation request
    await integration.track_generation_request(model_id, generation_params)
    print("✓ Generation request tracked")
    
    # Track generation start
    await integration.track_generation_start(model_id, generation_params)
    print("✓ Generation start tracked")
    
    # Simulate some processing time
    await asyncio.sleep(0.1)
    
    # Track successful completion
    await integration.track_generation_complete(
        model_id=model_id,
        duration_seconds=125.5,
        generation_params=generation_params,
        performance_metrics={
            "vram_usage_mb": 8200.0,
            "generation_speed": 2.3
        }
    )
    print("✓ Generation completion tracked")
    
    print("\nIntegration demo completed successfully!")


async def save_demo_results():
    """Save demo results to files for inspection"""
    logger.info("Saving demo results...")
    
    # Create demo output directory
    demo_dir = Path("demo_analytics_output")
    demo_dir.mkdir(exist_ok=True)
    
    # Generate and save comprehensive report
    report = await generate_analytics_report()
    
    report_file = demo_dir / f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nDemo results saved to: {report_file}")
    
    # Save individual model statistics
    models = ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
    for model_id in models:
        stats = await get_usage_statistics_for_model(model_id)
        
        stats_file = demo_dir / f"{model_id}_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
    
    print(f"Individual model statistics saved to: {demo_dir}")


async def main():
    """Main demo function"""
    print("Model Usage Analytics System Demo")
    print("=" * 50)
    
    try:
        # Step 1: Simulate model usage
        await simulate_model_usage()
        
        # Step 2: Demonstrate analytics features
        await demonstrate_analytics_features()
        
        # Step 3: Generate comprehensive report
        await demonstrate_comprehensive_report()
        
        # Step 4: Demonstrate integration features
        await demonstrate_integration_features()
        
        # Step 5: Save results
        await save_demo_results()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nThe Model Usage Analytics System provides:")
        print("• Comprehensive usage tracking and statistics")
        print("• Intelligent cleanup recommendations")
        print("• Smart preload suggestions")
        print("• Performance optimization recommendations")
        print("• Detailed analytics reports")
        print("• Seamless integration with generation services")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\nDemo failed with error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())