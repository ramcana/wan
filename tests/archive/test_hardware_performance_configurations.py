#!/usr/bin/env python3
"""
Hardware Performance Configuration Test Suite

Tests WAN model performance under various hardware configurations including:
- RTX 4080 optimization
- Different VRAM scenarios
- CPU vs GPU processing
- Memory optimization strategies
- Quantization performance impact
"""

import asyncio
import sys
import logging
import json
import time
import psutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import requests

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    generation_time: float
    peak_vram_mb: float
    avg_vram_mb: float
    cpu_usage_percent: float
    memory_usage_percent: float
    gpu_utilization_percent: float
    model_loading_time: float
    inference_time: float
    post_processing_time: float

@dataclass
class HardwareConfig:
    """Hardware configuration"""
    name: str
    vram_limit_gb: Optional[float]
    quantization: str
    cpu_offload: bool
    optimization_level: str
    batch_size: int

class HardwarePerformanceTest:
    """Hardware performance testing suite"""
    
    def __init__(self):
        self.backend_url = "http://localhost:8000"
        self.test_results = {}
        self.hardware_info = {}
        
    async def detect_hardware_capabilities(self) -> Dict[str, Any]:
        """Detect current hardware capabilities"""
        logger.info("🔍 Detecting hardware capabilities...")
        
        hardware_info = {
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "platform": sys.platform
        }
        
        # Detect CUDA/GPU capabilities
        try:
            import torch
            hardware_info["cuda_available"] = torch.cuda.is_available()
            
            if torch.cuda.is_available():
                hardware_info["cuda_device_count"] = torch.cuda.device_count()
                hardware_info["cuda_device_name"] = torch.cuda.get_device_name(0)
                hardware_info["cuda_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                hardware_info["cuda_compute_capability"] = torch.cuda.get_device_capability(0)
            else:
                hardware_info["cuda_device_count"] = 0
                hardware_info["cuda_device_name"] = "None"
                hardware_info["cuda_memory_gb"] = 0
                
        except ImportError:
            hardware_info["cuda_available"] = False
            hardware_info["cuda_device_count"] = 0
            hardware_info["cuda_device_name"] = "PyTorch not available"
            hardware_info["cuda_memory_gb"] = 0
        
        # Detect specific GPU models
        if hardware_info.get("cuda_device_name"):
            gpu_name = hardware_info["cuda_device_name"].lower()
            if "rtx 4080" in gpu_name:
                hardware_info["gpu_type"] = "RTX_4080"
                hardware_info["optimized_for_rtx4080"] = True
            elif "rtx" in gpu_name:
                hardware_info["gpu_type"] = "RTX_OTHER"
                hardware_info["optimized_for_rtx4080"] = False
            else:
                hardware_info["gpu_type"] = "OTHER"
                hardware_info["optimized_for_rtx4080"] = False
        
        self.hardware_info = hardware_info
        
        logger.info(f"  CPU: {hardware_info['cpu_count']} cores")
        logger.info(f"  Memory: {hardware_info['memory_gb']:.1f}GB")
        logger.info(f"  CUDA Available: {hardware_info['cuda_available']}")
        if hardware_info['cuda_available']:
            logger.info(f"  GPU: {hardware_info['cuda_device_name']}")
            logger.info(f"  VRAM: {hardware_info['cuda_memory_gb']:.1f}GB")
        
        return hardware_info
    
    def get_test_configurations(self) -> List[HardwareConfig]:
        """Get test configurations based on detected hardware"""
        configs = []
        
        # Base configuration
        configs.append(HardwareConfig(
            name="Default",
            vram_limit_gb=None,
            quantization="fp16",
            cpu_offload=False,
            optimization_level="balanced",
            batch_size=1
        ))
        
        # Low VRAM configuration
        configs.append(HardwareConfig(
            name="Low VRAM",
            vram_limit_gb=6.0,
            quantization="int8",
            cpu_offload=True,
            optimization_level="memory_optimized",
            batch_size=1
        ))
        
        # High performance configuration (if sufficient VRAM)
        if self.hardware_info.get("cuda_memory_gb", 0) >= 12:
            configs.append(HardwareConfig(
                name="High Performance",
                vram_limit_gb=None,
                quantization="fp16",
                cpu_offload=False,
                optimization_level="performance",
                batch_size=1
            ))
        
        # RTX 4080 optimized configuration
        if self.hardware_info.get("optimized_for_rtx4080", False):
            configs.append(HardwareConfig(
                name="RTX 4080 Optimized",
                vram_limit_gb=16.0,
                quantization="bf16",
                cpu_offload=False,
                optimization_level="rtx4080_optimized",
                batch_size=1
            ))
        
        # CPU fallback configuration
        configs.append(HardwareConfig(
            name="CPU Fallback",
            vram_limit_gb=0.0,
            quantization="int8",
            cpu_offload=True,
            optimization_level="cpu_optimized",
            batch_size=1
        ))
        
        # Aggressive memory optimization
        configs.append(HardwareConfig(
            name="Aggressive Memory",
            vram_limit_gb=4.0,
            quantization="int4",
            cpu_offload=True,
            optimization_level="aggressive_memory",
            batch_size=1
        ))
        
        return configs
    
    async def test_configuration_performance(self, config: HardwareConfig, model_type: str = "t2v-A14B") -> Dict[str, Any]:
        """Test performance for a specific hardware configuration"""
        logger.info(f"🧪 Testing {config.name} configuration with {model_type}...")
        
        try:
            # Prepare generation request with configuration
            request_data = {
                "model_type": model_type,
                "prompt": f"Performance test for {config.name} configuration",
                "resolution": "1280x720",
                "steps": 20,  # Reduced steps for performance testing
                "guidance_scale": 7.5,
                "num_frames": 8,  # Reduced frames for faster testing
                "fps": 8.0,
                "seed": 42,
                # Hardware configuration parameters
                "hardware_config": {
                    "vram_limit_gb": config.vram_limit_gb,
                    "quantization": config.quantization,
                    "cpu_offload": config.cpu_offload,
                    "optimization_level": config.optimization_level,
                    "batch_size": config.batch_size
                }
            }
            
            # Record initial system state
            initial_stats = await self.get_system_stats()
            start_time = time.time()
            
            # Submit generation request
            response = requests.post(
                f"{self.backend_url}/api/v1/generation/submit",
                json=request_data,
                timeout=30
            )
            
            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"Request failed: {response.status_code} - {response.text}",
                    "config": config.__dict__
                }
            
            task_data = response.json()
            task_id = task_data["task_id"]
            
            # Monitor performance during generation
            performance_samples = []
            final_result = None
            
            # Monitor for up to 5 minutes
            timeout = 300
            poll_interval = 2
            
            while (time.time() - start_time) < timeout:
                await asyncio.sleep(poll_interval)
                
                # Get current system stats
                current_stats = await self.get_system_stats()
                performance_samples.append({
                    "timestamp": time.time() - start_time,
                    "stats": current_stats
                })
                
                # Check task status
                try:
                    status_response = requests.get(
                        f"{self.backend_url}/api/v1/queue/{task_id}",
                        timeout=10
                    )
                    
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        
                        if status_data["status"] in ["completed", "failed"]:
                            final_result = status_data
                            break
                            
                except Exception as e:
                    logger.warning(f"Error checking task status: {e}")
            
            total_time = time.time() - start_time
            
            # Analyze performance metrics
            if performance_samples:
                cpu_usage = [sample["stats"]["cpu_percent"] for sample in performance_samples]
                memory_usage = [sample["stats"]["memory_percent"] for sample in performance_samples]
                
                performance_metrics = {
                    "total_time": total_time,
                    "avg_cpu_usage": sum(cpu_usage) / len(cpu_usage),
                    "peak_cpu_usage": max(cpu_usage),
                    "avg_memory_usage": sum(memory_usage) / len(memory_usage),
                    "peak_memory_usage": max(memory_usage),
                    "sample_count": len(performance_samples)
                }
            else:
                performance_metrics = {
                    "total_time": total_time,
                    "avg_cpu_usage": 0,
                    "peak_cpu_usage": 0,
                    "avg_memory_usage": 0,
                    "peak_memory_usage": 0,
                    "sample_count": 0
                }
            
            # Process final result
            if final_result:
                if final_result["status"] == "completed":
                    result = {
                        "success": True,
                        "config": config.__dict__,
                        "performance_metrics": performance_metrics,
                        "generation_time": final_result.get("generation_time_seconds"),
                        "model_used": final_result.get("model_used"),
                        "vram_usage": {
                            "peak_mb": final_result.get("peak_vram_usage_mb"),
                            "average_mb": final_result.get("average_vram_usage_mb")
                        },
                        "optimizations_applied": final_result.get("optimizations_applied"),
                        "output_exists": Path(final_result.get("output_path", "")).exists()
                    }
                else:
                    result = {
                        "success": False,
                        "config": config.__dict__,
                        "performance_metrics": performance_metrics,
                        "error": final_result.get("error_message", "Generation failed"),
                        "final_status": final_result["status"]
                    }
            else:
                result = {
                    "success": False,
                    "config": config.__dict__,
                    "performance_metrics": performance_metrics,
                    "error": "Generation timed out",
                    "timeout": True
                }
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "config": config.__dict__,
                "error": str(e),
                "exception": True
            }
    
    async def get_system_stats(self) -> Dict[str, float]:
        """Get current system statistics"""
        try:
            # Try to get stats from backend API first
            response = requests.get(f"{self.backend_url}/api/v1/system/stats", timeout=5)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        
        # Fallback to local system stats
        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent if sys.platform != 'win32' else psutil.disk_usage('C:').percent
            }
        except:
            return {"cpu_percent": 0, "memory_percent": 0, "disk_usage": 0}
    
    async def test_model_loading_performance(self) -> Dict[str, Any]:
        """Test model loading performance for different WAN models"""
        logger.info("🧪 Testing model loading performance...")
        
        model_types = ["t2v-A14B", "i2v-A14B", "ti2v-5B"]
        loading_results = {}
        
        for model_type in model_types:
            try:
                logger.info(f"  Testing {model_type} loading...")
                
                start_time = time.time()
                
                # Test model availability/loading through API
                response = requests.get(
                    f"{self.backend_url}/api/v1/models/info/{model_type}",
                    timeout=30
                )
                
                loading_time = time.time() - start_time
                
                if response.status_code == 200:
                    model_info = response.json()
                    loading_results[model_type] = {
                        "success": True,
                        "loading_time": loading_time,
                        "model_info": model_info
                    }
                else:
                    loading_results[model_type] = {
                        "success": False,
                        "loading_time": loading_time,
                        "error": f"HTTP {response.status_code}: {response.text}"
                    }
                    
            except Exception as e:
                loading_results[model_type] = {
                    "success": False,
                    "loading_time": 0,
                    "error": str(e)
                }
        
        return loading_results
    
    async def test_vram_optimization_strategies(self) -> Dict[str, Any]:
        """Test different VRAM optimization strategies"""
        logger.info("🧪 Testing VRAM optimization strategies...")
        
        optimization_strategies = [
            {"name": "No Optimization", "quantization": "fp16", "cpu_offload": False},
            {"name": "Quantization Only", "quantization": "int8", "cpu_offload": False},
            {"name": "CPU Offload Only", "quantization": "fp16", "cpu_offload": True},
            {"name": "Full Optimization", "quantization": "int8", "cpu_offload": True},
            {"name": "Aggressive Optimization", "quantization": "int4", "cpu_offload": True}
        ]
        
        strategy_results = {}
        
        for strategy in optimization_strategies:
            try:
                logger.info(f"  Testing {strategy['name']}...")
                
                # Create a minimal test configuration
                test_config = HardwareConfig(
                    name=strategy["name"],
                    vram_limit_gb=8.0,  # Moderate VRAM limit
                    quantization=strategy["quantization"],
                    cpu_offload=strategy["cpu_offload"],
                    optimization_level="custom",
                    batch_size=1
                )
                
                # Test with a quick generation
                result = await self.test_configuration_performance(test_config, "t2v-A14B")
                strategy_results[strategy["name"]] = result
                
                # Wait between tests
                await asyncio.sleep(2)
                
            except Exception as e:
                strategy_results[strategy["name"]] = {
                    "success": False,
                    "error": str(e)
                }
        
        return strategy_results
    
    async def benchmark_generation_speeds(self) -> Dict[str, Any]:
        """Benchmark generation speeds across different configurations"""
        logger.info("🧪 Benchmarking generation speeds...")
        
        benchmark_configs = [
            {"name": "Fast", "steps": 10, "frames": 8, "resolution": "1280x720"},
            {"name": "Balanced", "steps": 20, "frames": 16, "resolution": "1280x720"},
            {"name": "Quality", "steps": 30, "frames": 16, "resolution": "1280x720"},
            {"name": "High Resolution", "steps": 20, "frames": 16, "resolution": "1920x1080"}
        ]
        
        benchmark_results = {}
        
        for bench_config in benchmark_configs:
            try:
                logger.info(f"  Benchmarking {bench_config['name']} preset...")
                
                request_data = {
                    "model_type": "t2v-A14B",
                    "prompt": f"Benchmark test for {bench_config['name']} preset",
                    "resolution": bench_config["resolution"],
                    "steps": bench_config["steps"],
                    "num_frames": bench_config["frames"],
                    "guidance_scale": 7.5,
                    "fps": 8.0,
                    "seed": 42
                }
                
                start_time = time.time()
                
                response = requests.post(
                    f"{self.backend_url}/api/v1/generation/submit",
                    json=request_data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    task_data = response.json()
                    task_id = task_data["task_id"]
                    
                    # Monitor for completion (with timeout)
                    final_result = await self.monitor_task_completion(task_id, timeout=180)
                    
                    total_time = time.time() - start_time
                    
                    if final_result and final_result["status"] == "completed":
                        benchmark_results[bench_config["name"]] = {
                            "success": True,
                            "total_time": total_time,
                            "generation_time": final_result.get("generation_time_seconds"),
                            "config": bench_config,
                            "vram_usage": final_result.get("peak_vram_usage_mb"),
                            "frames_per_second": bench_config["frames"] / final_result.get("generation_time_seconds", 1)
                        }
                    else:
                        benchmark_results[bench_config["name"]] = {
                            "success": False,
                            "total_time": total_time,
                            "config": bench_config,
                            "error": final_result.get("error_message") if final_result else "Timeout"
                        }
                else:
                    benchmark_results[bench_config["name"]] = {
                        "success": False,
                        "config": bench_config,
                        "error": f"Request failed: {response.status_code}"
                    }
                
                # Wait between benchmarks
                await asyncio.sleep(5)
                
            except Exception as e:
                benchmark_results[bench_config["name"]] = {
                    "success": False,
                    "config": bench_config,
                    "error": str(e)
                }
        
        return benchmark_results
    
    async def monitor_task_completion(self, task_id: str, timeout: int = 300) -> Optional[Dict]:
        """Monitor task completion with timeout"""
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            await asyncio.sleep(3)
            
            try:
                response = requests.get(
                    f"{self.backend_url}/api/v1/queue/{task_id}",
                    timeout=10
                )
                
                if response.status_code == 200:
                    status_data = response.json()
                    if status_data["status"] in ["completed", "failed"]:
                        return status_data
                        
            except Exception as e:
                logger.warning(f"Error monitoring task {task_id}: {e}")
        
        return None
    
    async def run_all_performance_tests(self) -> Dict[str, Any]:
        """Run all hardware performance tests"""
        logger.info("🚀 Starting Hardware Performance Configuration Tests...")
        
        # Detect hardware
        hardware_info = await self.detect_hardware_capabilities()
        
        # Get test configurations
        test_configs = self.get_test_configurations()
        
        logger.info(f"Testing {len(test_configs)} hardware configurations...")
        
        # Test each configuration
        config_results = {}
        for config in test_configs:
            try:
                result = await self.test_configuration_performance(config)
                config_results[config.name] = result
                
                status = "✅" if result.get("success", False) else "❌"
                logger.info(f"{status} {config.name}: {'PASS' if result.get('success') else 'FAIL'}")
                
            except Exception as e:
                config_results[config.name] = {
                    "success": False,
                    "error": str(e)
                }
                logger.error(f"❌ {config.name}: ERROR - {e}")
        
        # Test model loading performance
        loading_results = await self.test_model_loading_performance()
        
        # Test VRAM optimization strategies
        vram_results = await self.test_vram_optimization_strategies()
        
        # Benchmark generation speeds
        benchmark_results = await self.benchmark_generation_speeds()
        
        # Compile final report
        report = {
            "timestamp": time.time(),
            "hardware_info": hardware_info,
            "configuration_tests": config_results,
            "model_loading_tests": loading_results,
            "vram_optimization_tests": vram_results,
            "generation_benchmarks": benchmark_results,
            "summary": {
                "total_configs_tested": len(config_results),
                "successful_configs": len([r for r in config_results.values() if r.get("success", False)]),
                "hardware_optimized": hardware_info.get("optimized_for_rtx4080", False),
                "cuda_available": hardware_info.get("cuda_available", False),
                "vram_gb": hardware_info.get("cuda_memory_gb", 0)
            }
        }
        
        # Save report
        report_path = Path("hardware_performance_test_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        self.print_performance_summary(report)
        
        logger.info(f"📄 Performance report saved to: {report_path}")
        
        return report
    
    def print_performance_summary(self, report: Dict[str, Any]):
        """Print performance test summary"""
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 HARDWARE PERFORMANCE TEST SUMMARY")
        logger.info(f"{'='*60}")
        
        # Hardware info
        hw = report["hardware_info"]
        logger.info(f"🖥️ Hardware Configuration:")
        logger.info(f"  CPU: {hw['cpu_count']} cores")
        logger.info(f"  Memory: {hw['memory_gb']:.1f}GB")
        logger.info(f"  GPU: {hw.get('cuda_device_name', 'None')}")
        logger.info(f"  VRAM: {hw.get('cuda_memory_gb', 0):.1f}GB")
        logger.info(f"  CUDA Available: {hw.get('cuda_available', False)}")
        
        # Configuration test results
        config_tests = report["configuration_tests"]
        successful_configs = len([r for r in config_tests.values() if r.get("success", False)])
        total_configs = len(config_tests)
        
        logger.info(f"\n📊 Configuration Test Results:")
        logger.info(f"  Total Configurations: {total_configs}")
        logger.info(f"  ✅ Successful: {successful_configs}")
        logger.info(f"  ❌ Failed: {total_configs - successful_configs}")
        logger.info(f"  📈 Success Rate: {successful_configs/total_configs:.1%}")
        
        # Performance highlights
        logger.info(f"\n⚡ Performance Highlights:")
        for config_name, result in config_tests.items():
            if result.get("success", False):
                gen_time = result.get("generation_time", 0)
                vram_peak = result.get("vram_usage", {}).get("peak_mb", 0)
                logger.info(f"  {config_name}: {gen_time:.1f}s generation, {vram_peak:.0f}MB VRAM")
        
        # Recommendations
        logger.info(f"\n💡 Recommendations:")
        if hw.get("optimized_for_rtx4080", False):
            logger.info("  ✅ RTX 4080 optimizations available")
        elif hw.get("cuda_available", False):
            logger.info("  ⚠️ Consider RTX 4080 for optimal performance")
        else:
            logger.info("  ⚠️ GPU acceleration not available - CPU-only mode")
        
        if hw.get("cuda_memory_gb", 0) < 8:
            logger.info("  💾 Consider VRAM optimization for better performance")
        
        logger.info(f"{'='*60}")

async def main():
    """Main performance test function"""
    tester = HardwarePerformanceTest()
    
    try:
        report = await tester.run_all_performance_tests()
        
        # Determine success based on results
        successful_configs = report["summary"]["successful_configs"]
        total_configs = report["summary"]["total_configs_tested"]
        success_rate = successful_configs / total_configs if total_configs > 0 else 0
        
        if success_rate >= 0.5:  # At least 50% of configs should work
            logger.info("🎉 Hardware performance tests completed successfully!")
            return 0
        else:
            logger.error("💥 Hardware performance tests failed!")
            return 1
            
    except KeyboardInterrupt:
        logger.info("🛑 Performance tests interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"💥 Performance tests failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
