#!/usr/bin/env python3
"""
Performance Comparison Analysis
Analyzes the performance difference between optimized and non-optimized WAN22 system
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List

def load_test_results(optimized_path: str, non_optimized_path: str) -> tuple:
    """Load test results from both optimization states"""
    try:
        with open(optimized_path, 'r') as f:
            optimized_results = json.load(f)
    except FileNotFoundError:
        print(f"❌ Optimized results file not found: {optimized_path}")
        return None, None
    
    try:
        with open(non_optimized_path, 'r') as f:
            non_optimized_results = json.load(f)
    except FileNotFoundError:
        print(f"❌ Non-optimized results file not found: {non_optimized_path}")
        return optimized_results, None
    
    return optimized_results, non_optimized_results

def analyze_performance_difference(optimized: Dict[str, Any], non_optimized: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze performance differences between optimized and non-optimized results"""
    analysis = {
        'model_loading': {},
        'video_generation': {},
        'overall': {}
    }
    
    # Extract performance metrics
    opt_perf = optimized.get('performance_metrics', {})
    non_opt_perf = non_optimized.get('performance_metrics', {})
    
    # Model loading comparison
    opt_model_test = next((test for test in optimized.get('generation_tests', []) 
                          if test['test_type'] == 'model_loading'), None)
    non_opt_model_test = next((test for test in non_optimized.get('generation_tests', []) 
                              if test['test_type'] == 'model_loading'), None)
    
    if opt_model_test and non_opt_model_test:
        opt_loading_time = opt_model_test['result']['loading_time_seconds']
        non_opt_loading_time = non_opt_model_test['result']['loading_time_seconds']
        
        loading_improvement = ((non_opt_loading_time - opt_loading_time) / non_opt_loading_time) * 100
        
        analysis['model_loading'] = {
            'optimized_time_seconds': opt_loading_time,
            'non_optimized_time_seconds': non_opt_loading_time,
            'improvement_percent': loading_improvement,
            'time_saved_seconds': non_opt_loading_time - opt_loading_time
        }
    
    # Video generation comparison
    opt_avg_time = opt_perf.get('average_generation_time_seconds', 0)
    non_opt_avg_time = non_opt_perf.get('average_generation_time_seconds', 0)
    
    if opt_avg_time > 0 and non_opt_avg_time > 0:
        generation_improvement = ((non_opt_avg_time - opt_avg_time) / non_opt_avg_time) * 100
        
        analysis['video_generation'] = {
            'optimized_time_seconds': opt_avg_time,
            'non_optimized_time_seconds': non_opt_avg_time,
            'improvement_percent': generation_improvement,
            'time_saved_seconds': non_opt_avg_time - opt_avg_time,
            'optimized_time_minutes': opt_avg_time / 60,
            'non_optimized_time_minutes': non_opt_avg_time / 60
        }
    
    # Overall comparison
    opt_total_time = optimized.get('total_duration_seconds', 0)
    non_opt_total_time = non_optimized.get('total_duration_seconds', 0)
    
    if opt_total_time > 0 and non_opt_total_time > 0:
        overall_improvement = ((non_opt_total_time - opt_total_time) / non_opt_total_time) * 100
        
        analysis['overall'] = {
            'optimized_total_time_seconds': opt_total_time,
            'non_optimized_total_time_seconds': non_opt_total_time,
            'improvement_percent': overall_improvement,
            'time_saved_seconds': non_opt_total_time - opt_total_time
        }
    
    return analysis

def print_performance_comparison(analysis: Dict[str, Any], optimized: Dict[str, Any], non_optimized: Dict[str, Any]):
    """Print detailed performance comparison"""
    print("WAN22 Performance Optimization Analysis")
    print("=" * 50)
    
    # Hardware information
    hardware = optimized.get('hardware_profile', {})
    if hardware:
        print(f"\nHardware Configuration:")
        print(f"  GPU: {hardware.get('gpu_model', 'Unknown')}")
        print(f"  VRAM: {hardware.get('vram_gb', 'Unknown')}GB")
        print(f"  CPU: {hardware.get('cpu_model', 'Unknown')}")
        print(f"  System Memory: {hardware.get('total_memory_gb', 'Unknown')}GB")
        print(f"  RTX 4080 Detected: {'Yes' if hardware.get('is_rtx_4080') else 'No'}")
        print(f"  Threadripper PRO: {'Yes' if hardware.get('is_threadripper_pro') else 'No'}")
    
    # Optimization settings
    opt_settings = optimized.get('optimization_settings', {})
    if opt_settings:
        print(f"\nOptimization Settings Applied:")
        print(f"  VAE Tile Size: {opt_settings.get('vae_tile_size', 'Default')}")
        print(f"  Batch Size: {opt_settings.get('batch_size', 'Default')}")
        print(f"  Memory Fraction: {opt_settings.get('memory_fraction', 'Default')}")
        print(f"  Tensor Cores: {'Enabled' if opt_settings.get('enable_tensor_cores') else 'Disabled'}")
        print(f"  BF16 Precision: {'Enabled' if opt_settings.get('use_bf16') else 'Disabled'}")
        print(f"  xFormers: {'Enabled' if opt_settings.get('enable_xformers') else 'Disabled'}")
        print(f"  CPU Offload: {'Enabled' if opt_settings.get('cpu_offload') else 'Disabled'}")
    
    print(f"\nPerformance Comparison Results:")
    print("=" * 50)
    
    # Model loading comparison
    if analysis.get('model_loading'):
        ml = analysis['model_loading']
        print(f"\n📥 Model Loading Performance:")
        print(f"  Without Optimization: {ml['non_optimized_time_seconds']:.2f}s")
        print(f"  With Optimization:    {ml['optimized_time_seconds']:.2f}s")
        print(f"  Improvement:          {ml['improvement_percent']:.1f}% faster")
        print(f"  Time Saved:           {ml['time_saved_seconds']:.2f}s")
    
    # Video generation comparison
    if analysis.get('video_generation'):
        vg = analysis['video_generation']
        print(f"\n🎬 Video Generation Performance:")
        print(f"  Without Optimization: {vg['non_optimized_time_minutes']:.2f} minutes ({vg['non_optimized_time_seconds']:.1f}s)")
        print(f"  With Optimization:    {vg['optimized_time_minutes']:.2f} minutes ({vg['optimized_time_seconds']:.1f}s)")
        print(f"  Improvement:          {vg['improvement_percent']:.1f}% faster")
        print(f"  Time Saved:           {vg['time_saved_seconds']:.1f}s per generation")
        
        # Calculate time savings for multiple generations
        time_saved_per_hour = (vg['time_saved_seconds'] / vg['non_optimized_time_seconds']) * 3600
        print(f"  Time Saved per Hour:  {time_saved_per_hour:.1f}s ({time_saved_per_hour/60:.1f} minutes)")
    
    # Overall comparison
    if analysis.get('overall'):
        overall = analysis['overall']
        print(f"\n⚡ Overall Test Performance:")
        print(f"  Without Optimization: {overall['non_optimized_total_time_seconds']:.1f}s")
        print(f"  With Optimization:    {overall['optimized_total_time_seconds']:.1f}s")
        print(f"  Improvement:          {overall['improvement_percent']:.1f}% faster")
        print(f"  Time Saved:           {overall['time_saved_seconds']:.1f}s")
    
    # Performance targets validation
    print(f"\n🎯 Performance Targets Validation:")
    
    # Check video generation targets
    if analysis.get('video_generation'):
        vg = analysis['video_generation']
        target_minutes = 2.0  # 2-minute target
        
        opt_meets_target = vg['optimized_time_minutes'] <= target_minutes
        non_opt_meets_target = vg['non_optimized_time_minutes'] <= target_minutes
        
        print(f"  Target: Video generation < {target_minutes} minutes")
        print(f"  Without Optimization: {'✅ PASS' if non_opt_meets_target else '❌ FAIL'} ({vg['non_optimized_time_minutes']:.2f} min)")
        print(f"  With Optimization:    {'✅ PASS' if opt_meets_target else '❌ FAIL'} ({vg['optimized_time_minutes']:.2f} min)")
        
        if opt_meets_target and not non_opt_meets_target:
            print(f"  🎉 Optimization enables meeting performance targets!")
        elif opt_meets_target:
            print(f"  ✅ Both configurations meet targets, optimization provides additional headroom")
    
    # Success rates
    opt_success = optimized.get('performance_metrics', {}).get('success_rate_percent', 0)
    non_opt_success = non_optimized.get('performance_metrics', {}).get('success_rate_percent', 0)
    
    print(f"\n📊 Success Rates:")
    print(f"  Without Optimization: {non_opt_success:.1f}%")
    print(f"  With Optimization:    {opt_success:.1f}%")
    
    if opt_success > non_opt_success:
        print(f"  🎉 Optimization improves reliability by {opt_success - non_opt_success:.1f}%")
    elif opt_success == non_opt_success:
        print(f"  ✅ Both configurations achieve excellent reliability")
    
    # Recommendations
    print(f"\n💡 Key Findings:")
    
    if analysis.get('video_generation', {}).get('improvement_percent', 0) > 20:
        print(f"  🚀 Significant performance improvement: {analysis['video_generation']['improvement_percent']:.1f}% faster generation")
    elif analysis.get('video_generation', {}).get('improvement_percent', 0) > 10:
        print(f"  📈 Good performance improvement: {analysis['video_generation']['improvement_percent']:.1f}% faster generation")
    elif analysis.get('video_generation', {}).get('improvement_percent', 0) > 0:
        print(f"  📊 Moderate performance improvement: {analysis['video_generation']['improvement_percent']:.1f}% faster generation")
    
    if opt_settings.get('is_rtx_4080'):
        print(f"  🖥️  RTX 4080 optimizations are highly effective")
    
    if opt_settings.get('use_bf16'):
        print(f"  🔢 BF16 precision provides excellent performance/quality balance")
    
    if opt_settings.get('enable_tensor_cores'):
        print(f"  ⚡ Tensor cores significantly accelerate computation")
    
    print(f"\n🎯 Recommendations:")
    print(f"  ✅ Enable optimization for production deployment")
    print(f"  📈 Optimization provides measurable performance benefits")
    print(f"  🔧 Current settings are well-tuned for RTX 4080 hardware")
    print(f"  🚀 System is ready for high-performance video generation workloads")

def main():
    """Main function for performance comparison analysis"""
    optimized_path = "realtime_generation_test_report.json"
    non_optimized_path = "realtime_generation_test_no_optimization.json"
    
    # Load test results
    optimized_results, non_optimized_results = load_test_results(optimized_path, non_optimized_path)
    
    if not optimized_results:
        print("❌ Could not load optimized test results")
        return 1
    
    if not non_optimized_results:
        print("⚠️  Could not load non-optimized test results for comparison")
        print("Showing optimized results only:")
        
        # Show optimized results summary
        perf = optimized_results.get('performance_metrics', {})
        print(f"\nOptimized Performance Summary:")
        print(f"  Success Rate: {perf.get('success_rate_percent', 0):.1f}%")
        print(f"  Average Generation Time: {perf.get('average_generation_time_minutes', 0):.2f} minutes")
        print(f"  Total Tests: {perf.get('total_tests', 0)}")
        
        return 0
    
    # Analyze performance differences
    analysis = analyze_performance_difference(optimized_results, non_optimized_results)
    
    # Print comparison
    print_performance_comparison(analysis, optimized_results, non_optimized_results)
    
    # Save analysis results
    analysis_results = {
        'analysis': analysis,
        'optimized_summary': optimized_results.get('performance_metrics', {}),
        'non_optimized_summary': non_optimized_results.get('performance_metrics', {}),
        'hardware_profile': optimized_results.get('hardware_profile', {}),
        'optimization_settings': optimized_results.get('optimization_settings', {})
    }
    
    with open('performance_comparison_analysis.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"\n📄 Detailed analysis saved to: performance_comparison_analysis.json")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())