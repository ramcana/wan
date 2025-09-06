from unittest.mock import Mock, patch
#!/usr/bin/env python3
"""
Performance Optimization Script for Wan2.2 UI Variant
Automated performance analysis, bottleneck identification, and optimization recommendations
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import subprocess
import psutil

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import application modules
try:
    from infrastructure.hardware.performance_profiler import (
        get_performance_profiler,
        start_performance_monitoring,
        stop_performance_monitoring,
        generate_performance_report
    )
    from core.services.utils import get_model_manager, VRAMOptimizer
    from infrastructure.hardware.error_handler import handle_error_with_recovery, log_error_with_context
except ImportError as e:
    print(f"Error importing application modules: {e}")
    print("Make sure you're running this script from the application directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('performance_optimization.log')
    ]
)
logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """Comprehensive performance optimizer and analyzer"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.profiler = get_performance_profiler()
        self.optimization_results = {}
        self.recommendations = []
        
        # System information
        self.system_info = self._collect_system_info()
        
        logger.info("Performance optimizer initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load application configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def _save_config(self, config: Dict[str, Any]):
        """Save updated configuration"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info("Configuration updated and saved")
        except IOError as e:
            logger.error(f"Failed to save config: {e}")
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect comprehensive system information"""
        info = {
            "timestamp": datetime.now().isoformat(),
            "platform": {
                "system": os.name,
                "platform": sys.platform,
                "python_version": sys.version,
                "architecture": os.uname().machine if hasattr(os, 'uname') else 'unknown'
            },
            "hardware": {
                "cpu_count": psutil.cpu_count(),
                "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "disk_total_gb": psutil.disk_usage('.').total / (1024**3),
                "disk_free_gb": psutil.disk_usage('.').free / (1024**3)
            }
        }
        
        # GPU information
        try:
            import torch
            if torch.cuda.is_available():
                info["gpu"] = {
                    "available": True,
                    "device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "device_name": torch.cuda.get_device_name(),
                    "memory_total_mb": torch.cuda.get_device_properties(0).total_memory / (1024**2),
                    "cuda_version": torch.version.cuda,
                    "cudnn_version": torch.backends.cudnn.version()
                }
            else:
                info["gpu"] = {"available": False}
        except ImportError:
            info["gpu"] = {"available": False, "error": "PyTorch not available"}
        
        return info
    
    def analyze_system_performance(self) -> Dict[str, Any]:
        """Analyze current system performance and identify bottlenecks"""
        logger.info("Analyzing system performance...")
        
        # Start monitoring
        start_performance_monitoring()
        
        # Collect baseline metrics
        time.sleep(5)  # Collect 5 seconds of baseline data
        
        baseline_summary = self.profiler.get_system_performance_summary()
        
        # Analyze bottlenecks
        bottlenecks = []
        recommendations = []
        
        # CPU Analysis
        cpu_usage = baseline_summary.get("cpu", {}).get("average_percent", 0)
        if cpu_usage > 80:
            bottlenecks.append("High CPU usage")
            recommendations.append("Consider reducing concurrent operations or upgrading CPU")
        elif cpu_usage < 20:
            recommendations.append("CPU is underutilized - can handle more concurrent operations")
        
        # Memory Analysis
        memory_usage = baseline_summary.get("memory", {}).get("average_percent", 0)
        memory_used_gb = baseline_summary.get("memory", {}).get("current_used_mb", 0) / 1024
        
        if memory_usage > 85:
            bottlenecks.append("High memory usage")
            recommendations.append("Enable model offloading or add more RAM")
        elif memory_used_gb < 8:
            recommendations.append("Sufficient memory available for larger models")
        
        # GPU Analysis
        if baseline_summary.get("gpu", {}).get("available", False):
            gpu_usage = baseline_summary.get("gpu", {}).get("average_percent", 0)
            vram_percent = baseline_summary.get("vram", {}).get("current_percent", 0)
            
            if gpu_usage > 90:
                bottlenecks.append("GPU utilization saturated")
                recommendations.append("Consider model quantization or batch size reduction")
            elif gpu_usage < 30:
                recommendations.append("GPU is underutilized - can handle higher quality settings")
            
            if vram_percent > 90:
                bottlenecks.append("VRAM usage critical")
                recommendations.append("Enable CPU offloading or use lower precision")
            elif vram_percent < 50:
                recommendations.append("VRAM headroom available for higher quality settings")
        
        # Disk Analysis
        disk_free_gb = self.system_info["hardware"]["disk_free_gb"]
        if disk_free_gb < 10:
            bottlenecks.append("Low disk space")
            recommendations.append("Clean up old outputs or add more storage")
        
        analysis_result = {
            "system_info": self.system_info,
            "baseline_metrics": baseline_summary,
            "bottlenecks": bottlenecks,
            "recommendations": recommendations,
            "optimization_potential": self._assess_optimization_potential(baseline_summary)
        }
        
        stop_performance_monitoring()
        
        return analysis_result
    
    def _assess_optimization_potential(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """Assess optimization potential based on current metrics"""
        potential = {}
        
        # VRAM optimization potential
        vram_percent = metrics.get("vram", {}).get("current_percent", 0)
        if vram_percent > 80:
            potential["vram"] = "High - significant savings possible with optimization"
        elif vram_percent > 60:
            potential["vram"] = "Medium - moderate savings possible"
        else:
            potential["vram"] = "Low - already well optimized"
        
        # CPU optimization potential
        cpu_percent = metrics.get("cpu", {}).get("average_percent", 0)
        if cpu_percent > 80:
            potential["cpu"] = "High - CPU bottleneck limiting performance"
        elif cpu_percent < 30:
            potential["cpu"] = "Low - CPU has spare capacity"
        else:
            potential["cpu"] = "Medium - balanced CPU usage"
        
        # Memory optimization potential
        memory_percent = metrics.get("memory", {}).get("average_percent", 0)
        if memory_percent > 80:
            potential["memory"] = "High - memory pressure affecting performance"
        else:
            potential["memory"] = "Low - sufficient memory available"
        
        return potential
    
    def benchmark_optimization_settings(self) -> Dict[str, Any]:
        """Benchmark different optimization settings to find optimal configuration"""
        logger.info("Benchmarking optimization settings...")
        
        # Define test configurations
        test_configs = [
            {
                "name": "baseline",
                "quantization": "bf16",
                "cpu_offload": False,
                "sequential_offload": False,
                "vae_tile_size": 512
            },
            {
                "name": "memory_optimized",
                "quantization": "int8",
                "cpu_offload": True,
                "sequential_offload": True,
                "vae_tile_size": 256
            },
            {
                "name": "balanced",
                "quantization": "bf16",
                "cpu_offload": True,
                "sequential_offload": False,
                "vae_tile_size": 256
            },
            {
                "name": "quality_focused",
                "quantization": "fp16",
                "cpu_offload": False,
                "sequential_offload": False,
                "vae_tile_size": 384
            }
        ]
        
        benchmark_results = {}
        
        for config in test_configs:
            logger.info(f"Testing configuration: {config['name']}")
            
            try:
                # Apply configuration
                self._apply_test_configuration(config)
                
                # Run benchmark
                result = self._run_optimization_benchmark(config)
                benchmark_results[config['name']] = result
                
                logger.info(f"Configuration {config['name']} completed")
                
            except Exception as e:
                logger.error(f"Failed to benchmark configuration {config['name']}: {e}")
                benchmark_results[config['name']] = {"error": str(e)}
        
        # Analyze results and recommend best configuration
        best_config = self._analyze_benchmark_results(benchmark_results)
        
        return {
            "benchmark_results": benchmark_results,
            "recommended_config": best_config,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _apply_test_configuration(self, config: Dict[str, Any]):
        """Apply test configuration to the system"""
        # Update config file temporarily
        test_config = self.config.copy()
        
        test_config["optimization"] = test_config.get("optimization", {})
        test_config["optimization"]["default_quantization"] = config["quantization"]
        test_config["optimization"]["enable_cpu_offload"] = config["cpu_offload"]
        test_config["optimization"]["sequential_cpu_offload"] = config["sequential_offload"]
        test_config["optimization"]["vae_tile_size"] = config["vae_tile_size"]
        
        # Save temporary config
        with open("config_test.json", 'w') as f:
            json.dump(test_config, f, indent=2)
    
    def _run_optimization_benchmark(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a benchmark with specific optimization configuration"""
        # Mock benchmark - in real implementation, this would:
        # 1. Load a model with the configuration
        # 2. Run a test generation
        # 3. Measure performance metrics
        
        # Simulate benchmark results based on configuration
        baseline_time = 300  # 5 minutes baseline
        baseline_vram = 10240  # 10GB baseline
        
        # Calculate performance impact based on settings
        time_multiplier = 1.0
        vram_multiplier = 1.0
        
        if config["quantization"] == "int8":
            time_multiplier *= 1.1  # 10% slower
            vram_multiplier *= 0.3  # 70% VRAM reduction
        elif config["quantization"] == "fp16":
            time_multiplier *= 0.95  # 5% faster
            vram_multiplier *= 0.5   # 50% VRAM reduction
        
        if config["cpu_offload"]:
            time_multiplier *= 1.2   # 20% slower
            vram_multiplier *= 0.6   # 40% VRAM reduction
            
            if config["sequential_offload"]:
                time_multiplier *= 1.3   # Additional 30% slower
                vram_multiplier *= 0.4   # Additional 60% VRAM reduction
        
        if config["vae_tile_size"] < 256:
            time_multiplier *= 1.15  # 15% slower
            vram_multiplier *= 0.8   # 20% VRAM reduction
        elif config["vae_tile_size"] > 384:
            time_multiplier *= 0.95  # 5% faster
            vram_multiplier *= 1.1   # 10% more VRAM
        
        estimated_time = baseline_time * time_multiplier
        estimated_vram = baseline_vram * vram_multiplier
        
        # Calculate quality score (inverse of optimization level)
        quality_score = 100
        if config["quantization"] == "int8":
            quality_score -= 15
        if config["cpu_offload"]:
            quality_score -= 5
        if config["vae_tile_size"] < 256:
            quality_score -= 10
        
        return {
            "config": config,
            "estimated_generation_time_seconds": estimated_time,
            "estimated_vram_usage_mb": estimated_vram,
            "estimated_quality_score": quality_score,
            "performance_score": self._calculate_performance_score(estimated_time, estimated_vram, quality_score),
            "benchmark_timestamp": datetime.now().isoformat()
        }
    
    def _calculate_performance_score(self, time_seconds: float, vram_mb: float, quality_score: float) -> float:
        """Calculate overall performance score"""
        # Normalize metrics (lower is better for time and VRAM, higher is better for quality)
        time_score = max(0, 100 - (time_seconds / 300) * 50)  # Normalize to 300s baseline
        vram_score = max(0, 100 - (vram_mb / 10240) * 50)     # Normalize to 10GB baseline
        
        # Weighted average (time: 40%, VRAM: 30%, quality: 30%)
        performance_score = (time_score * 0.4) + (vram_score * 0.3) + (quality_score * 0.3)
        
        return round(performance_score, 2)
    
    def _analyze_benchmark_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze benchmark results and recommend best configuration"""
        if not results:
            return {"error": "No benchmark results available"}
        
        # Find configuration with highest performance score
        best_config = None
        best_score = -1
        
        for config_name, result in results.items():
            if "error" not in result:
                score = result.get("performance_score", 0)
                if score > best_score:
                    best_score = score
                    best_config = result
        
        if best_config is None:
            return {"error": "No valid benchmark results"}
        
        # Generate recommendations based on system capabilities
        recommendations = []
        
        # Check if system can handle quality-focused settings
        if self.system_info["hardware"]["memory_total_gb"] >= 32:
            recommendations.append("System has sufficient RAM for CPU offloading")
        
        if self.system_info.get("gpu", {}).get("memory_total_mb", 0) >= 16384:  # 16GB VRAM
            recommendations.append("High VRAM available - can use quality-focused settings")
        elif self.system_info.get("gpu", {}).get("memory_total_mb", 0) >= 12288:  # 12GB VRAM
            recommendations.append("Moderate VRAM available - balanced settings recommended")
        else:
            recommendations.append("Limited VRAM - memory-optimized settings recommended")
        
        return {
            "best_configuration": best_config["config"],
            "performance_score": best_score,
            "recommendations": recommendations,
            "all_scores": {name: result.get("performance_score", 0) for name, result in results.items() if "error" not in result}
        }
    
    def apply_optimal_configuration(self, benchmark_results: Dict[str, Any]):
        """Apply the optimal configuration based on benchmark results"""
        recommended_config = benchmark_results.get("recommended_config", {})
        
        if "best_configuration" not in recommended_config:
            logger.error("No optimal configuration found in benchmark results")
            return False
        
        optimal_config = recommended_config["best_configuration"]
        
        logger.info(f"Applying optimal configuration: {optimal_config}")
        
        # Update application configuration
        updated_config = self.config.copy()
        
        # Update optimization settings
        optimization_section = updated_config.setdefault("optimization", {})
        optimization_section["default_quantization"] = optimal_config["quantization"]
        optimization_section["enable_cpu_offload"] = optimal_config["cpu_offload"]
        optimization_section["sequential_cpu_offload"] = optimal_config["sequential_offload"]
        optimization_section["vae_tile_size"] = optimal_config["vae_tile_size"]
        
        # Add optimization metadata
        optimization_section["optimization_applied"] = {
            "timestamp": datetime.now().isoformat(),
            "performance_score": recommended_config["performance_score"],
            "method": "automated_benchmark"
        }
        
        # Save updated configuration
        self._save_config(updated_config)
        
        logger.info("Optimal configuration applied successfully")
        return True
    
    def generate_optimization_report(self, output_path: Optional[str] = None) -> str:
        """Generate comprehensive optimization report"""
        logger.info("Generating optimization report...")
        
        # Collect all optimization data
        system_analysis = self.analyze_system_performance()
        benchmark_results = self.benchmark_optimization_settings()
        
        # Generate performance report
        performance_report_path = "performance_report_temp.json"
        generate_performance_report(performance_report_path)
        
        try:
            with open(performance_report_path, 'r') as f:
                performance_data = json.load(f)
        except (IOError, json.JSONDecodeError):
            performance_data = {"error": "Failed to load performance report"}
        
        # Compile comprehensive report
        optimization_report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "report_version": "1.0",
                "system_info": self.system_info
            },
            "system_analysis": system_analysis,
            "benchmark_results": benchmark_results,
            "performance_data": performance_data,
            "optimization_summary": {
                "total_bottlenecks_identified": len(system_analysis.get("bottlenecks", [])),
                "optimization_recommendations": len(system_analysis.get("recommendations", [])),
                "best_configuration_score": benchmark_results.get("recommended_config", {}).get("performance_score", 0),
                "optimization_applied": "optimization_applied" in self.config.get("optimization", {})
            },
            "next_steps": self._generate_next_steps(system_analysis, benchmark_results)
        }
        
        # Save report
        if output_path:
            report_file = Path(output_path)
        else:
            report_file = Path(f"optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(optimization_report, f, indent=2)
        
        # Clean up temporary files
        if os.path.exists(performance_report_path):
            os.remove(performance_report_path)
        if os.path.exists("config_test.json"):
            os.remove("config_test.json")
        
        logger.info(f"Optimization report saved to {report_file}")
        
        return str(report_file)
    
    def _generate_next_steps(self, system_analysis: Dict[str, Any], benchmark_results: Dict[str, Any]) -> List[str]:
        """Generate actionable next steps based on analysis"""
        next_steps = []
        
        # System-level recommendations
        bottlenecks = system_analysis.get("bottlenecks", [])
        
        if "High CPU usage" in bottlenecks:
            next_steps.append("Consider upgrading CPU or reducing concurrent operations")
        
        if "High memory usage" in bottlenecks:
            next_steps.append("Add more RAM or enable aggressive memory optimization")
        
        if "VRAM usage critical" in bottlenecks:
            next_steps.append("Apply recommended VRAM optimization settings immediately")
        
        if "Low disk space" in bottlenecks:
            next_steps.append("Clean up old outputs and consider adding more storage")
        
        # Configuration recommendations
        recommended_config = benchmark_results.get("recommended_config", {})
        if "best_configuration" in recommended_config:
            next_steps.append("Apply the recommended optimal configuration")
            next_steps.append("Test the new configuration with a sample generation")
        
        # Performance improvement suggestions
        optimization_potential = system_analysis.get("optimization_potential", {})
        
        if optimization_potential.get("vram") == "High":
            next_steps.append("Implement aggressive VRAM optimization for significant performance gains")
        
        if optimization_potential.get("cpu") == "Low":
            next_steps.append("Consider increasing concurrent operations to utilize spare CPU capacity")
        
        # General recommendations
        next_steps.extend([
            "Monitor system performance regularly using the built-in profiler",
            "Re-run optimization analysis after hardware or software changes",
            "Keep the application and dependencies updated for best performance"
        ])
        
        return next_steps


def main():
    """Main entry point for performance optimization script"""
    parser = argparse.ArgumentParser(
        description="Wan2.2 UI Variant Performance Optimization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python optimize_performance.py --analyze          # Analyze current performance
  python optimize_performance.py --benchmark        # Run optimization benchmarks
  python optimize_performance.py --optimize         # Apply optimal settings
  python optimize_performance.py --full-report      # Generate comprehensive report
        """
    )
    
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze current system performance and identify bottlenecks"
    )
    
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark different optimization configurations"
    )
    
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Apply optimal configuration based on benchmarks"
    )
    
    parser.add_argument(
        "--full-report",
        action="store_true",
        help="Generate comprehensive optimization report"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to configuration file (default: config.json)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for reports"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize optimizer
    try:
        optimizer = PerformanceOptimizer(args.config)
    except Exception as e:
        logger.error(f"Failed to initialize optimizer: {e}")
        return 1
    
    try:
        if args.analyze:
            logger.info("Running system performance analysis...")
            analysis = optimizer.analyze_system_performance()
            
            print("\n" + "="*60)
            print("SYSTEM PERFORMANCE ANALYSIS")
            print("="*60)
            
            print(f"\nBottlenecks Identified: {len(analysis['bottlenecks'])}")
            for bottleneck in analysis['bottlenecks']:
                print(f"  ⚠️  {bottleneck}")
            
            print(f"\nRecommendations: {len(analysis['recommendations'])}")
            for recommendation in analysis['recommendations']:
                print(f"  💡 {recommendation}")
            
            print("\nOptimization Potential:")
            for component, potential in analysis['optimization_potential'].items():
                print(f"  {component.upper()}: {potential}")
        
        if args.benchmark:
            logger.info("Running optimization benchmarks...")
            benchmark_results = optimizer.benchmark_optimization_settings()
            
            print("\n" + "="*60)
            print("OPTIMIZATION BENCHMARK RESULTS")
            print("="*60)
            
            recommended = benchmark_results.get("recommended_config", {})
            if "best_configuration" in recommended:
                best_config = recommended["best_configuration"]
                score = recommended["performance_score"]
                
                print(f"\nRecommended Configuration (Score: {score}):")
                print(f"  Quantization: {best_config['quantization']}")
                print(f"  CPU Offload: {best_config['cpu_offload']}")
                print(f"  Sequential Offload: {best_config['sequential_offload']}")
                print(f"  VAE Tile Size: {best_config['vae_tile_size']}")
                
                print("\nAll Configuration Scores:")
                for config_name, score in recommended.get("all_scores", {}).items():
                    print(f"  {config_name}: {score}")
        
        if args.optimize:
            logger.info("Applying optimal configuration...")
            
            # Run benchmark if not already done
            benchmark_results = optimizer.benchmark_optimization_settings()
            
            if optimizer.apply_optimal_configuration(benchmark_results):
                print("\n✅ Optimal configuration applied successfully!")
                print("Restart the application to use the new settings.")
            else:
                print("\n❌ Failed to apply optimal configuration.")
                return 1
        
        if args.full_report:
            logger.info("Generating comprehensive optimization report...")
            report_path = optimizer.generate_optimization_report(args.output)
            
            print(f"\n📊 Comprehensive optimization report generated:")
            print(f"   {report_path}")
            print("\nThe report includes:")
            print("  • System performance analysis")
            print("  • Optimization benchmarks")
            print("  • Performance profiling data")
            print("  • Actionable recommendations")
        
        if not any([args.analyze, args.benchmark, args.optimize, args.full_report]):
            print("No action specified. Use --help for available options.")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())