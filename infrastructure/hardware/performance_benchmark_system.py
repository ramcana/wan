"""
Performance Benchmarking System for WAN22 System Optimization
Implements before/after performance metrics collection, hardware validation, and settings recommendations
"""

import time
import json
import logging
import psutil
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Callable
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import nvidia_ml_py3 as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

from hardware_optimizer import HardwareProfile, OptimalSettings

@dataclass
class PerformanceMetrics:
    """Performance metrics for benchmarking"""
    timestamp: str
    # Timing metrics
    model_load_time: float = 0.0
    generation_time: float = 0.0
    preprocessing_time: float = 0.0
    postprocessing_time: float = 0.0
    total_time: float = 0.0
    
    # Memory metrics
    peak_vram_usage_mb: int = 0
    peak_ram_usage_mb: int = 0
    vram_efficiency: float = 0.0  # Used VRAM / Total VRAM
    
    # GPU metrics
    gpu_utilization_avg: float = 0.0
    gpu_temperature_max: float = 0.0
    gpu_power_draw_avg: float = 0.0
    
    # CPU metrics
    cpu_utilization_avg: float = 0.0
    cpu_temperature_max: float = 0.0
    
    # Quality metrics
    generation_speed_fps: float = 0.0  # For video generation
    throughput_items_per_second: float = 0.0
    
    # Configuration used
    settings_used: Optional[Dict[str, Any]] = None
    hardware_profile: Optional[Dict[str, Any]] = None

@dataclass
class BenchmarkResult:
    """Result of performance benchmark"""
    success: bool
    before_metrics: Optional[PerformanceMetrics]
    after_metrics: Optional[PerformanceMetrics]
    performance_improvement: float  # Percentage improvement
    memory_savings: int  # MB saved
    recommendations: List[str]
    warnings: List[str]
    errors: List[str]
    benchmark_type: str = "general"

@dataclass
class HardwareLimits:
    """Hardware performance limits and thresholds"""
    max_vram_usage_mb: int
    max_ram_usage_mb: int
    max_gpu_temperature: float = 83.0  # RTX 4080 safe limit
    max_cpu_temperature: float = 90.0  # Threadripper PRO safe limit
    max_gpu_power_draw: float = 320.0  # RTX 4080 TGP
    target_gpu_utilization: float = 95.0  # Target GPU utilization
    target_memory_efficiency: float = 0.85  # Target VRAM efficiency

class SystemMonitor:
    """Real-time system monitoring for benchmarks"""
    
    def __init__(self):
        self.monitoring = False
        self.metrics = []
        self.monitor_thread = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize NVML if available
        if NVML_AVAILABLE:
            try:
                nvml.nvmlInit()
                self.nvml_initialized = True
            except Exception as e:
                self.logger.warning(f"NVML initialization failed: {e}")
                self.nvml_initialized = False
        else:
            self.nvml_initialized = False
    
    def start_monitoring(self, interval: float = 1.0):
        """Start system monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.metrics = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.logger.info("Started system monitoring")
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return aggregated metrics"""
        if not self.monitoring:
            return {}
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        # Aggregate metrics
        if not self.metrics:
            return {}
        
        aggregated = {
            'gpu_utilization_avg': sum(m.get('gpu_utilization', 0) for m in self.metrics) / len(self.metrics),
            'gpu_temperature_max': max(m.get('gpu_temperature', 0) for m in self.metrics),
            'gpu_power_draw_avg': sum(m.get('gpu_power_draw', 0) for m in self.metrics) / len(self.metrics),
            'cpu_utilization_avg': sum(m.get('cpu_utilization', 0) for m in self.metrics) / len(self.metrics),
            'cpu_temperature_max': max(m.get('cpu_temperature', 0) for m in self.metrics),
            'peak_vram_usage_mb': max(m.get('vram_usage_mb', 0) for m in self.metrics),
            'peak_ram_usage_mb': max(m.get('ram_usage_mb', 0) for m in self.metrics)
        }
        
        self.logger.info("Stopped system monitoring")
        return aggregated
    
    def _monitor_loop(self, interval: float):
        """Monitoring loop"""
        while self.monitoring:
            try:
                metrics = self._collect_instant_metrics()
                self.metrics.append(metrics)
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(interval)
    
    def _collect_instant_metrics(self) -> Dict[str, float]:
        """Collect instantaneous system metrics"""
        metrics = {}
        
        # CPU metrics
        metrics['cpu_utilization'] = psutil.cpu_percent()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        metrics['ram_usage_mb'] = (memory.total - memory.available) / (1024 * 1024)
        
        # GPU metrics via NVML
        if self.nvml_initialized:
            try:
                handle = nvml.nvmlDeviceGetHandleByIndex(0)
                
                # GPU utilization
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                metrics['gpu_utilization'] = util.gpu
                
                # GPU temperature
                temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                metrics['gpu_temperature'] = temp
                
                # GPU power draw
                power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                metrics['gpu_power_draw'] = power
                
                # VRAM usage
                mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                metrics['vram_usage_mb'] = mem_info.used / (1024 * 1024)
                
            except Exception as e:
                self.logger.debug(f"NVML metrics collection failed: {e}")
        
        # GPU metrics via PyTorch (fallback)
        if TORCH_AVAILABLE and torch.cuda.is_available() and 'vram_usage_mb' not in metrics:
            try:
                metrics['vram_usage_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
            except Exception:
                pass
        
        return metrics

class PerformanceBenchmarkSystem:
    """Performance benchmarking system for WAN22 optimization"""
    
    def __init__(self, results_dir: str = "benchmark_results"):
        """Initialize performance benchmark system"""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.monitor = SystemMonitor()
        
        # Benchmark targets for different hardware
        self.benchmark_targets = {
            'rtx_4080': {
                'ti2v_5b_load_time_max': 300.0,  # 5 minutes max
                'video_2s_generation_max': 120.0,  # 2 minutes max for 2s video
                'vram_usage_max_mb': 12288,  # 12GB max for TI2V-5B
                'target_fps': 0.5  # Target generation speed
            },
            'threadripper_pro': {
                'preprocessing_speedup': 2.0,  # 2x speedup expected
                'parallel_efficiency': 0.8,  # 80% parallel efficiency
                'numa_optimization': 1.3  # 30% improvement with NUMA
            }
        }
    
    def create_hardware_limits(self, profile: HardwareProfile) -> HardwareLimits:
        """Create hardware limits based on profile"""
        # Threadripper PRO takes precedence over RTX 4080 for CPU-specific limits
        if profile.is_threadripper_pro:
            return HardwareLimits(
                max_vram_usage_mb=profile.vram_gb * 1024,
                max_ram_usage_mb=profile.total_memory_gb * 1024 // 2,  # Half of available RAM
                max_gpu_temperature=83.0,
                max_cpu_temperature=90.0,  # Threadripper PRO safe limit
                max_gpu_power_draw=320.0,  # RTX 4080 TGP (since it has RTX 4080)
                target_gpu_utilization=95.0,
                target_memory_efficiency=0.90  # Higher efficiency with powerful CPU
            )
        elif profile.is_rtx_4080:
            return HardwareLimits(
                max_vram_usage_mb=profile.vram_gb * 1024,  # Full VRAM
                max_ram_usage_mb=min(profile.total_memory_gb * 1024 // 2, 32768),  # Half RAM or 32GB max
                max_gpu_temperature=83.0,  # RTX 4080 safe limit
                max_cpu_temperature=90.0,
                max_gpu_power_draw=320.0,  # RTX 4080 TGP
                target_gpu_utilization=95.0,
                target_memory_efficiency=0.85
            )
        else:
            # Default limits
            return HardwareLimits(
                max_vram_usage_mb=profile.vram_gb * 1024,
                max_ram_usage_mb=min(profile.total_memory_gb * 1024 // 2, 16384),
                max_gpu_temperature=80.0,
                max_cpu_temperature=85.0,
                max_gpu_power_draw=250.0,
                target_gpu_utilization=90.0,
                target_memory_efficiency=0.80
            )
    
    def benchmark_model_loading(self, 
                              model_load_func: Callable,
                              model_name: str = "test_model",
                              settings: Optional[OptimalSettings] = None) -> PerformanceMetrics:
        """Benchmark model loading performance"""
        self.logger.info(f"Benchmarking model loading: {model_name}")
        
        # Start monitoring
        self.monitor.start_monitoring(interval=0.5)
        
        start_time = time.time()
        
        try:
            # Execute model loading
            result = model_load_func()
            load_time = time.time() - start_time
            
            # Stop monitoring and get metrics
            monitor_metrics = self.monitor.stop_monitoring()
            
            # Create performance metrics
            metrics = PerformanceMetrics(
                timestamp=datetime.now().isoformat(),
                model_load_time=load_time,
                total_time=load_time,
                peak_vram_usage_mb=monitor_metrics.get('peak_vram_usage_mb', 0),
                peak_ram_usage_mb=monitor_metrics.get('peak_ram_usage_mb', 0),
                gpu_utilization_avg=monitor_metrics.get('gpu_utilization_avg', 0),
                gpu_temperature_max=monitor_metrics.get('gpu_temperature_max', 0),
                gpu_power_draw_avg=monitor_metrics.get('gpu_power_draw_avg', 0),
                cpu_utilization_avg=monitor_metrics.get('cpu_utilization_avg', 0),
                cpu_temperature_max=monitor_metrics.get('cpu_temperature_max', 0),
                settings_used=asdict(settings) if settings else None
            )
            
            # Calculate efficiency
            if TORCH_AVAILABLE and torch.cuda.is_available():
                total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                metrics.vram_efficiency = metrics.peak_vram_usage_mb / total_vram
            
            self.logger.info(f"Model loading benchmark completed: {load_time:.2f}s")
            return metrics
            
        except Exception as e:
            self.monitor.stop_monitoring()
            self.logger.error(f"Model loading benchmark failed: {e}")
            raise
    
    def benchmark_generation(self,
                           generation_func: Callable,
                           generation_params: Dict[str, Any],
                           expected_output_count: int = 1,
                           settings: Optional[OptimalSettings] = None) -> PerformanceMetrics:
        """Benchmark generation performance"""
        self.logger.info("Benchmarking generation performance")
        
        # Start monitoring
        self.monitor.start_monitoring(interval=0.5)
        
        start_time = time.time()
        preprocess_start = time.time()
        
        try:
            # Execute generation
            result = generation_func(**generation_params)
            
            generation_time = time.time() - start_time
            
            # Stop monitoring and get metrics
            monitor_metrics = self.monitor.stop_monitoring()
            
            # Calculate throughput
            throughput = expected_output_count / generation_time if generation_time > 0 else 0
            
            # Create performance metrics
            metrics = PerformanceMetrics(
                timestamp=datetime.now().isoformat(),
                generation_time=generation_time,
                total_time=generation_time,
                peak_vram_usage_mb=monitor_metrics.get('peak_vram_usage_mb', 0),
                peak_ram_usage_mb=monitor_metrics.get('peak_ram_usage_mb', 0),
                gpu_utilization_avg=monitor_metrics.get('gpu_utilization_avg', 0),
                gpu_temperature_max=monitor_metrics.get('gpu_temperature_max', 0),
                gpu_power_draw_avg=monitor_metrics.get('gpu_power_draw_avg', 0),
                cpu_utilization_avg=monitor_metrics.get('cpu_utilization_avg', 0),
                cpu_temperature_max=monitor_metrics.get('cpu_temperature_max', 0),
                throughput_items_per_second=throughput,
                settings_used=asdict(settings) if settings else None
            )
            
            # Calculate efficiency
            if TORCH_AVAILABLE and torch.cuda.is_available():
                total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                metrics.vram_efficiency = metrics.peak_vram_usage_mb / total_vram
            
            self.logger.info(f"Generation benchmark completed: {generation_time:.2f}s")
            return metrics
            
        except Exception as e:
            self.monitor.stop_monitoring()
            self.logger.error(f"Generation benchmark failed: {e}")
            raise
    
    def validate_against_hardware_limits(self, 
                                       metrics: PerformanceMetrics,
                                       limits: HardwareLimits,
                                       profile: HardwareProfile) -> Tuple[bool, List[str], List[str]]:
        """Validate performance metrics against hardware limits"""
        warnings = []
        errors = []
        is_valid = True
        
        # VRAM validation
        if metrics.peak_vram_usage_mb > limits.max_vram_usage_mb:
            errors.append(f"VRAM usage ({metrics.peak_vram_usage_mb}MB) exceeds limit ({limits.max_vram_usage_mb}MB)")
            is_valid = False
        elif metrics.vram_efficiency > 0.95:
            warnings.append(f"VRAM efficiency very high ({metrics.vram_efficiency:.1%}), risk of OOM")
        
        # RAM validation
        if metrics.peak_ram_usage_mb > limits.max_ram_usage_mb:
            warnings.append(f"RAM usage ({metrics.peak_ram_usage_mb}MB) exceeds recommended limit ({limits.max_ram_usage_mb}MB)")
        
        # Temperature validation
        if metrics.gpu_temperature_max > limits.max_gpu_temperature:
            errors.append(f"GPU temperature ({metrics.gpu_temperature_max}°C) exceeds safe limit ({limits.max_gpu_temperature}°C)")
            is_valid = False
        elif metrics.gpu_temperature_max > limits.max_gpu_temperature - 5:
            warnings.append(f"GPU temperature ({metrics.gpu_temperature_max}°C) approaching limit")
        
        if metrics.cpu_temperature_max > limits.max_cpu_temperature:
            errors.append(f"CPU temperature ({metrics.cpu_temperature_max}°C) exceeds safe limit ({limits.max_cpu_temperature}°C)")
            is_valid = False
        
        # Power validation
        if metrics.gpu_power_draw_avg > limits.max_gpu_power_draw:
            warnings.append(f"GPU power draw ({metrics.gpu_power_draw_avg}W) exceeds TGP ({limits.max_gpu_power_draw}W)")
        
        # Utilization validation
        if metrics.gpu_utilization_avg < limits.target_gpu_utilization - 20:
            warnings.append(f"GPU utilization low ({metrics.gpu_utilization_avg:.1f}%), potential for optimization")
        
        # Hardware-specific validations
        if profile.is_rtx_4080:
            targets = self.benchmark_targets['rtx_4080']
            
            if metrics.model_load_time > targets['ti2v_5b_load_time_max']:
                warnings.append(f"Model load time ({metrics.model_load_time:.1f}s) exceeds target ({targets['ti2v_5b_load_time_max']}s)")
            
            if metrics.generation_time > targets['video_2s_generation_max']:
                warnings.append(f"Generation time ({metrics.generation_time:.1f}s) exceeds target for 2s video")
            
            if metrics.peak_vram_usage_mb > targets['vram_usage_max_mb']:
                errors.append(f"VRAM usage exceeds TI2V-5B target ({targets['vram_usage_max_mb']}MB)")
                is_valid = False
        
        return is_valid, warnings, errors
    
    def generate_performance_recommendations(self,
                                           metrics: PerformanceMetrics,
                                           profile: HardwareProfile,
                                           current_settings: Optional[OptimalSettings] = None) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # VRAM optimization recommendations
        if metrics.vram_efficiency > 0.90:
            recommendations.append("Enable CPU offloading for text encoder and VAE to reduce VRAM usage")
            recommendations.append("Consider reducing batch size or tile size")
            recommendations.append("Enable gradient checkpointing to save memory")
        
        # GPU utilization recommendations
        if metrics.gpu_utilization_avg < 80:
            recommendations.append("Increase batch size to improve GPU utilization")
            recommendations.append("Consider disabling CPU offloading if VRAM allows")
            recommendations.append("Enable mixed precision (FP16/BF16) for better performance")
        
        # Temperature recommendations
        if metrics.gpu_temperature_max > 75:
            recommendations.append("Improve GPU cooling or reduce power limit")
            recommendations.append("Consider undervolting GPU for better thermal performance")
        
        # Hardware-specific recommendations
        if profile.is_rtx_4080:
            if metrics.peak_vram_usage_mb < 8192:  # Less than 8GB used
                recommendations.append("VRAM usage is low - consider increasing batch size or tile size")
                recommendations.append("Disable CPU offloading for better performance")
            
            if not (current_settings and current_settings.enable_tensor_cores):
                recommendations.append("Enable Tensor Cores (TF32) for RTX 4080 optimization")
            
            if metrics.generation_time > 60:  # Slow generation
                recommendations.append("Enable xFormers for memory-efficient attention")
                recommendations.append("Use BF16 precision for RTX 4080 optimization")
        
        elif profile.is_threadripper_pro:
            if metrics.cpu_utilization_avg < 50:
                recommendations.append("Increase parallel processing workers to utilize more CPU cores")
                recommendations.append("Enable NUMA optimization for better memory bandwidth")
            
            if current_settings and current_settings.num_threads < profile.cpu_cores // 2:
                recommendations.append(f"Increase thread count to utilize more CPU cores (current: {current_settings.num_threads}, available: {profile.cpu_cores})")
        
        # Memory efficiency recommendations
        if metrics.peak_ram_usage_mb > profile.total_memory_gb * 1024 * 0.8:
            recommendations.append("High RAM usage detected - consider reducing preprocessing workers")
            recommendations.append("Enable memory-efficient attention mechanisms")
        
        # Performance recommendations based on timing
        if metrics.model_load_time > 180:  # > 3 minutes
            recommendations.append("Model loading is slow - consider model caching or faster storage")
            recommendations.append("Enable model loading optimizations")
        
        if metrics.generation_time > 0 and metrics.throughput_items_per_second < 0.1:
            recommendations.append("Low generation throughput - consider optimizing pipeline settings")
            recommendations.append("Review quantization settings for performance impact")
        
        return recommendations
    
    def generate_recommended_settings_for_hardware(self, 
                                                 profile: HardwareProfile,
                                                 fallback_mode: bool = False) -> OptimalSettings:
        """Generate recommended settings based on hardware detection (Requirement 5.4)"""
        self.logger.info(f"Generating recommended settings for hardware profile: {profile.gpu_model}")
        
        if fallback_mode:
            self.logger.warning("Hardware detection failed - using fallback configuration")
        
        # Import hardware optimizer to generate settings
        from hardware_optimizer import HardwareOptimizer
        optimizer = HardwareOptimizer()
        
        if profile.is_threadripper_pro:
            settings = optimizer.generate_threadripper_pro_settings(profile)
        elif profile.is_rtx_4080:
            settings = optimizer.generate_rtx_4080_settings(profile)
        else:
            # Generate conservative default settings for unknown hardware
            settings = OptimalSettings(
                tile_size=(256, 256),
                vae_tile_size=(256, 256),
                batch_size=1,
                enable_cpu_offload=True,
                text_encoder_offload=True,
                vae_offload=True,
                enable_tensor_cores=False,  # Conservative default
                use_fp16=True,
                use_bf16=False,  # Conservative default
                memory_fraction=0.8,  # Conservative memory usage
                gradient_checkpointing=True,
                num_threads=min(profile.cpu_cores, 4),  # Conservative threading
                enable_xformers=True,
                parallel_workers=1,
                enable_numa_optimization=False,
                preprocessing_threads=1,
                io_threads=1
            )
        
        return settings
    
    def validate_settings_against_hardware_limits(self,
                                                settings: OptimalSettings,
                                                profile: HardwareProfile) -> Tuple[bool, List[str], List[str]]:
        """Validate optimization settings against hardware limits (Requirement 5.5)"""
        warnings = []
        errors = []
        is_valid = True
        
        # Create hardware limits for validation
        limits = self.create_hardware_limits(profile)
        
        # Validate VRAM requirements
        estimated_vram_usage = self._estimate_vram_usage(settings, profile)
        if estimated_vram_usage > limits.max_vram_usage_mb:
            errors.append(f"Estimated VRAM usage ({estimated_vram_usage}MB) exceeds hardware limit ({limits.max_vram_usage_mb}MB)")
            is_valid = False
        elif estimated_vram_usage > limits.max_vram_usage_mb * 0.9:
            warnings.append(f"Estimated VRAM usage ({estimated_vram_usage}MB) is close to hardware limit")
        
        # Validate batch size
        if settings.batch_size > 4 and profile.vram_gb < 16:
            warnings.append(f"Batch size ({settings.batch_size}) may be too high for {profile.vram_gb}GB VRAM")
        
        # Validate tile sizes
        tile_area = settings.tile_size[0] * settings.tile_size[1]
        if tile_area > 512 * 512 and profile.vram_gb < 12:
            warnings.append(f"Tile size ({settings.tile_size}) may be too large for available VRAM")
        
        # Validate threading settings
        if settings.num_threads > profile.cpu_cores:
            errors.append(f"Thread count ({settings.num_threads}) exceeds available CPU cores ({profile.cpu_cores})")
            is_valid = False
        
        if settings.parallel_workers > profile.cpu_cores // 2:
            warnings.append(f"Parallel workers ({settings.parallel_workers}) may exceed optimal CPU utilization")
        
        # Validate memory fraction
        if settings.memory_fraction > 0.95:
            warnings.append(f"Memory fraction ({settings.memory_fraction}) is very high, risk of OOM")
        
        # Hardware-specific validations
        if profile.is_rtx_4080:
            if not settings.enable_tensor_cores:
                warnings.append("Tensor cores disabled - consider enabling for RTX 4080 optimization")
            
            if not settings.use_bf16:
                warnings.append("BF16 disabled - RTX 4080 supports BF16 for better performance")
        
        elif profile.is_threadripper_pro:
            if settings.num_threads < profile.cpu_cores // 4:
                warnings.append(f"Thread count ({settings.num_threads}) may be too low for Threadripper PRO")
            
            if not settings.enable_numa_optimization and profile.cpu_cores > 32:
                warnings.append("NUMA optimization disabled - consider enabling for Threadripper PRO")
        
        # Validate mixed precision settings
        if settings.use_fp16 and settings.use_bf16:
            warnings.append("Both FP16 and BF16 enabled - typically only one should be used")
        
        # Validate offloading settings
        if not settings.enable_cpu_offload and profile.vram_gb < 16:
            warnings.append("CPU offloading disabled with limited VRAM - consider enabling")
        
        return is_valid, warnings, errors
    
    def _estimate_vram_usage(self, settings: OptimalSettings, profile: HardwareProfile) -> int:
        """Estimate VRAM usage based on settings"""
        # Base model VRAM usage estimates (in MB)
        base_usage = {
            'ti2v_5b': 8192,  # 8GB base
            'default': 4096   # 4GB default
        }
        
        # Get base usage
        estimated_usage = base_usage.get('ti2v_5b', base_usage['default'])
        
        # Adjust for batch size
        estimated_usage *= settings.batch_size
        
        # Adjust for tile size
        tile_area = settings.tile_size[0] * settings.tile_size[1]
        tile_multiplier = tile_area / (256 * 256)  # Relative to 256x256 baseline
        estimated_usage *= tile_multiplier
        
        # Adjust for precision
        if settings.use_fp16:
            estimated_usage *= 0.7  # FP16 saves ~30% memory
        elif settings.use_bf16:
            estimated_usage *= 0.7  # BF16 saves ~30% memory
        
        # Adjust for gradient checkpointing
        if settings.gradient_checkpointing:
            estimated_usage *= 0.8  # Saves ~20% memory
        
        # Adjust for CPU offloading
        if settings.text_encoder_offload:
            estimated_usage *= 0.9  # Saves ~10% memory
        if settings.vae_offload:
            estimated_usage *= 0.85  # Saves ~15% memory
        
        return int(estimated_usage)
    
    def run_before_after_benchmark(self,
                                 before_func: Callable,
                                 after_func: Callable,
                                 profile: HardwareProfile,
                                 benchmark_name: str = "optimization") -> BenchmarkResult:
        """Run before/after performance benchmark (Requirement 5.3)"""
        self.logger.info(f"Running before/after benchmark: {benchmark_name}")
        
        errors = []
        warnings = []
        
        try:
            # Run before benchmark
            self.logger.info("Running 'before' benchmark...")
            before_metrics = before_func()
            
            # Small delay between benchmarks
            time.sleep(2)
            
            # Run after benchmark
            self.logger.info("Running 'after' benchmark...")
            after_metrics = after_func()
            
            # Calculate comprehensive performance improvements
            performance_improvement = self._calculate_performance_improvement(before_metrics, after_metrics)
            memory_savings = before_metrics.peak_vram_usage_mb - after_metrics.peak_vram_usage_mb
            
            # Generate detailed performance comparison
            performance_comparison = self._generate_detailed_performance_comparison(before_metrics, after_metrics)
            
            # Generate recommendations
            recommendations = self.generate_performance_recommendations(after_metrics, profile)
            
            # Validate against hardware limits
            limits = self.create_hardware_limits(profile)
            is_valid, val_warnings, val_errors = self.validate_against_hardware_limits(after_metrics, limits, profile)
            
            warnings.extend(val_warnings)
            errors.extend(val_errors)
            
            result = BenchmarkResult(
                success=True,
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                performance_improvement=performance_improvement,
                memory_savings=memory_savings,
                recommendations=recommendations,
                warnings=warnings,
                errors=errors,
                benchmark_type=benchmark_name
            )
            
            # Save results with detailed comparison
            self.save_benchmark_result(result, benchmark_name)
            self._save_performance_comparison(performance_comparison, benchmark_name)
            
            self.logger.info(f"Benchmark completed - Performance improvement: {performance_improvement:.1f}%, Memory savings: {memory_savings}MB")
            return result
            
        except Exception as e:
            error_msg = f"Benchmark failed: {e}"
            self.logger.error(error_msg)
            errors.append(error_msg)
            
            return BenchmarkResult(
                success=False,
                before_metrics=None,
                after_metrics=None,
                performance_improvement=0.0,
                memory_savings=0,
                recommendations=[],
                warnings=warnings,
                errors=errors,
                benchmark_type=benchmark_name
            )
    
    def _calculate_performance_improvement(self, 
                                         before_metrics: PerformanceMetrics, 
                                         after_metrics: PerformanceMetrics) -> float:
        """Calculate comprehensive performance improvement percentage"""
        improvements = []
        
        # Time-based improvements
        if before_metrics.total_time > 0 and after_metrics.total_time > 0:
            time_improvement = ((before_metrics.total_time - after_metrics.total_time) / before_metrics.total_time) * 100
            improvements.append(time_improvement)
        
        if before_metrics.model_load_time > 0 and after_metrics.model_load_time > 0:
            load_improvement = ((before_metrics.model_load_time - after_metrics.model_load_time) / before_metrics.model_load_time) * 100
            improvements.append(load_improvement)
        
        if before_metrics.generation_time > 0 and after_metrics.generation_time > 0:
            gen_improvement = ((before_metrics.generation_time - after_metrics.generation_time) / before_metrics.generation_time) * 100
            improvements.append(gen_improvement)
        
        # Throughput improvements
        if before_metrics.throughput_items_per_second > 0 and after_metrics.throughput_items_per_second > 0:
            throughput_improvement = ((after_metrics.throughput_items_per_second - before_metrics.throughput_items_per_second) / before_metrics.throughput_items_per_second) * 100
            improvements.append(throughput_improvement)
        
        # Return weighted average or best improvement
        return sum(improvements) / len(improvements) if improvements else 0.0
    
    def _generate_detailed_performance_comparison(self, 
                                                before_metrics: PerformanceMetrics, 
                                                after_metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Generate detailed performance comparison for requirement 5.3"""
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'timing_comparison': {
                'model_load_time': {
                    'before': before_metrics.model_load_time,
                    'after': after_metrics.model_load_time,
                    'improvement_seconds': before_metrics.model_load_time - after_metrics.model_load_time,
                    'improvement_percent': ((before_metrics.model_load_time - after_metrics.model_load_time) / before_metrics.model_load_time * 100) if before_metrics.model_load_time > 0 else 0
                },
                'generation_time': {
                    'before': before_metrics.generation_time,
                    'after': after_metrics.generation_time,
                    'improvement_seconds': before_metrics.generation_time - after_metrics.generation_time,
                    'improvement_percent': ((before_metrics.generation_time - after_metrics.generation_time) / before_metrics.generation_time * 100) if before_metrics.generation_time > 0 else 0
                },
                'total_time': {
                    'before': before_metrics.total_time,
                    'after': after_metrics.total_time,
                    'improvement_seconds': before_metrics.total_time - after_metrics.total_time,
                    'improvement_percent': ((before_metrics.total_time - after_metrics.total_time) / before_metrics.total_time * 100) if before_metrics.total_time > 0 else 0
                }
            },
            'memory_comparison': {
                'peak_vram_usage_mb': {
                    'before': before_metrics.peak_vram_usage_mb,
                    'after': after_metrics.peak_vram_usage_mb,
                    'savings_mb': before_metrics.peak_vram_usage_mb - after_metrics.peak_vram_usage_mb,
                    'savings_percent': ((before_metrics.peak_vram_usage_mb - after_metrics.peak_vram_usage_mb) / before_metrics.peak_vram_usage_mb * 100) if before_metrics.peak_vram_usage_mb > 0 else 0
                },
                'peak_ram_usage_mb': {
                    'before': before_metrics.peak_ram_usage_mb,
                    'after': after_metrics.peak_ram_usage_mb,
                    'savings_mb': before_metrics.peak_ram_usage_mb - after_metrics.peak_ram_usage_mb,
                    'savings_percent': ((before_metrics.peak_ram_usage_mb - after_metrics.peak_ram_usage_mb) / before_metrics.peak_ram_usage_mb * 100) if before_metrics.peak_ram_usage_mb > 0 else 0
                },
                'vram_efficiency': {
                    'before': before_metrics.vram_efficiency,
                    'after': after_metrics.vram_efficiency,
                    'improvement': after_metrics.vram_efficiency - before_metrics.vram_efficiency
                }
            },
            'performance_comparison': {
                'gpu_utilization_avg': {
                    'before': before_metrics.gpu_utilization_avg,
                    'after': after_metrics.gpu_utilization_avg,
                    'improvement': after_metrics.gpu_utilization_avg - before_metrics.gpu_utilization_avg
                },
                'throughput_items_per_second': {
                    'before': before_metrics.throughput_items_per_second,
                    'after': after_metrics.throughput_items_per_second,
                    'improvement': after_metrics.throughput_items_per_second - before_metrics.throughput_items_per_second,
                    'improvement_percent': ((after_metrics.throughput_items_per_second - before_metrics.throughput_items_per_second) / before_metrics.throughput_items_per_second * 100) if before_metrics.throughput_items_per_second > 0 else 0
                }
            },
            'thermal_comparison': {
                'gpu_temperature_max': {
                    'before': before_metrics.gpu_temperature_max,
                    'after': after_metrics.gpu_temperature_max,
                    'improvement': before_metrics.gpu_temperature_max - after_metrics.gpu_temperature_max
                },
                'cpu_temperature_max': {
                    'before': before_metrics.cpu_temperature_max,
                    'after': after_metrics.cpu_temperature_max,
                    'improvement': before_metrics.cpu_temperature_max - after_metrics.cpu_temperature_max
                },
                'gpu_power_draw_avg': {
                    'before': before_metrics.gpu_power_draw_avg,
                    'after': after_metrics.gpu_power_draw_avg,
                    'savings': before_metrics.gpu_power_draw_avg - after_metrics.gpu_power_draw_avg
                }
            }
        }
        
        return comparison
    
    def _save_performance_comparison(self, comparison: Dict[str, Any], benchmark_name: str):
        """Save detailed performance comparison to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_comparison_{benchmark_name}_{timestamp}.json"
            filepath = self.results_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(comparison, f, indent=2, default=str)
            
            self.logger.info(f"Saved performance comparison to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save performance comparison: {e}")
    
    def save_benchmark_result(self, result: BenchmarkResult, benchmark_name: str):
        """Save benchmark result to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_{benchmark_name}_{timestamp}.json"
            filepath = self.results_dir / filename
            
            # Convert to serializable format
            result_dict = asdict(result)
            
            with open(filepath, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)
            
            self.logger.info(f"Saved benchmark result to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save benchmark result: {e}")
    
    def load_benchmark_result(self, filepath: str) -> Optional[BenchmarkResult]:
        """Load benchmark result from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Reconstruct PerformanceMetrics objects
            if data.get('before_metrics'):
                data['before_metrics'] = PerformanceMetrics(**data['before_metrics'])
            if data.get('after_metrics'):
                data['after_metrics'] = PerformanceMetrics(**data['after_metrics'])
            
            return BenchmarkResult(**data)
            
        except Exception as e:
            self.logger.error(f"Failed to load benchmark result: {e}")
            return None
    
    def generate_benchmark_report(self, results: List[BenchmarkResult]) -> str:
        """Generate comprehensive benchmark report"""
        report_lines = [
            "# WAN22 Performance Benchmark Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            ""
        ]
        
        if not results:
            report_lines.append("No benchmark results available.")
            return "\n".join(report_lines)
        
        # Summary statistics
        successful_benchmarks = [r for r in results if r.success]
        avg_improvement = sum(r.performance_improvement for r in successful_benchmarks) / len(successful_benchmarks) if successful_benchmarks else 0
        total_memory_savings = sum(r.memory_savings for r in successful_benchmarks)
        
        report_lines.extend([
            f"- Total benchmarks: {len(results)}",
            f"- Successful benchmarks: {len(successful_benchmarks)}",
            f"- Average performance improvement: {avg_improvement:.1f}%",
            f"- Total memory savings: {total_memory_savings}MB",
            "",
            "## Detailed Results",
            ""
        ])
        
        # Detailed results
        for i, result in enumerate(results, 1):
            report_lines.extend([
                f"### Benchmark {i}: {result.benchmark_type}",
                f"- Success: {result.success}",
                f"- Performance improvement: {result.performance_improvement:.1f}%",
                f"- Memory savings: {result.memory_savings}MB",
                ""
            ])
            
            if result.before_metrics and result.after_metrics:
                report_lines.extend([
                    "#### Before/After Comparison",
                    f"- Total time: {result.before_metrics.total_time:.2f}s → {result.after_metrics.total_time:.2f}s",
                    f"- Peak VRAM: {result.before_metrics.peak_vram_usage_mb}MB → {result.after_metrics.peak_vram_usage_mb}MB",
                    f"- GPU utilization: {result.before_metrics.gpu_utilization_avg:.1f}% → {result.after_metrics.gpu_utilization_avg:.1f}%",
                    ""
                ])
            
            if result.recommendations:
                report_lines.extend([
                    "#### Recommendations",
                    *[f"- {rec}" for rec in result.recommendations],
                    ""
                ])
            
            if result.warnings:
                report_lines.extend([
                    "#### Warnings",
                    *[f"- {warn}" for warn in result.warnings],
                    ""
                ])
            
            if result.errors:
                report_lines.extend([
                    "#### Errors",
                    *[f"- {err}" for err in result.errors],
                    ""
                ])
        
        return "\n".join(report_lines)
    
    def handle_hardware_detection_failure(self, 
                                        partial_profile: Optional[HardwareProfile] = None) -> Tuple[HardwareProfile, OptimalSettings]:
        """Handle hardware detection failure and provide manual configuration (Requirement 5.4)"""
        self.logger.warning("Hardware detection failed - providing manual configuration options")
        
        # Create fallback hardware profile
        if partial_profile:
            fallback_profile = partial_profile
        else:
            # Create minimal fallback profile
            fallback_profile = HardwareProfile(
                cpu_model="Unknown CPU",
                cpu_cores=psutil.cpu_count(logical=False) or 4,
                total_memory_gb=int(psutil.virtual_memory().total / (1024**3)) or 16,
                gpu_model="Unknown GPU",
                vram_gb=8,  # Conservative default
                cuda_version="Unknown",
                driver_version="Unknown",
                is_rtx_4080=False,
                is_threadripper_pro=False
            )
        
        # Try to detect some hardware info if possible (only if not already provided)
        try:
            # Try to get more accurate CPU info only if not provided
            if not partial_profile or fallback_profile.cpu_cores <= 4:
                fallback_profile.cpu_cores = psutil.cpu_count(logical=False) or fallback_profile.cpu_cores
            if not partial_profile or fallback_profile.total_memory_gb <= 16:
                fallback_profile.total_memory_gb = int(psutil.virtual_memory().total / (1024**3))
            
            # Try to detect GPU VRAM if PyTorch is available (only if not already provided)
            if TORCH_AVAILABLE and torch.cuda.is_available() and (not partial_profile or fallback_profile.vram_gb <= 8):
                try:
                    gpu_props = torch.cuda.get_device_properties(0)
                    if not partial_profile or fallback_profile.gpu_model == "Unknown GPU":
                        fallback_profile.gpu_model = gpu_props.name
                    if not partial_profile or fallback_profile.vram_gb <= 8:
                        fallback_profile.vram_gb = int(gpu_props.total_memory / (1024**3))
                    
                    # Check if it's RTX 4080
                    if "RTX 4080" in gpu_props.name.upper():
                        fallback_profile.is_rtx_4080 = True
                        if not partial_profile or fallback_profile.vram_gb <= 8:
                            fallback_profile.vram_gb = 16
                except Exception as e:
                    self.logger.debug(f"GPU detection via PyTorch failed: {e}")
            
            # Try to detect Threadripper PRO (only if not already specified)
            if fallback_profile.cpu_cores >= 32 and not fallback_profile.is_threadripper_pro:
                try:
                    import platform
                    cpu_info = platform.processor()
                    if "Threadripper PRO" in cpu_info.upper():
                        fallback_profile.is_threadripper_pro = True
                        if not partial_profile or fallback_profile.cpu_model == "Unknown CPU":
                            fallback_profile.cpu_model = cpu_info
                except Exception as e:
                    self.logger.debug(f"CPU detection failed: {e}")
        
        except Exception as e:
            self.logger.warning(f"Partial hardware detection failed: {e}")
        
        # Generate recommended settings for fallback hardware
        recommended_settings = self.generate_recommended_settings_for_hardware(
            fallback_profile, fallback_mode=True
        )
        
        # Validate settings against detected hardware
        is_valid, warnings, errors = self.validate_settings_against_hardware_limits(
            recommended_settings, fallback_profile
        )
        
        if not is_valid:
            self.logger.warning("Generated settings may not be optimal for detected hardware")
            for error in errors:
                self.logger.warning(f"Settings validation error: {error}")
        
        # Log manual configuration options
        self.logger.info("Manual configuration options:")
        self.logger.info(f"  Detected CPU cores: {fallback_profile.cpu_cores}")
        self.logger.info(f"  Detected RAM: {fallback_profile.total_memory_gb}GB")
        self.logger.info(f"  Detected GPU: {fallback_profile.gpu_model}")
        self.logger.info(f"  Estimated VRAM: {fallback_profile.vram_gb}GB")
        self.logger.info(f"  Recommended batch size: {recommended_settings.batch_size}")
        self.logger.info(f"  Recommended tile size: {recommended_settings.tile_size}")
        self.logger.info(f"  CPU offloading: {'Enabled' if recommended_settings.enable_cpu_offload else 'Disabled'}")
        
        return fallback_profile, recommended_settings
    
    def create_manual_configuration_guide(self, profile: HardwareProfile) -> Dict[str, Any]:
        """Create manual configuration guide for users (Requirement 5.4)"""
        guide = {
            'hardware_profile': {
                'cpu_model': profile.cpu_model,
                'cpu_cores': profile.cpu_cores,
                'total_memory_gb': profile.total_memory_gb,
                'gpu_model': profile.gpu_model,
                'vram_gb': profile.vram_gb,
                'is_rtx_4080': profile.is_rtx_4080,
                'is_threadripper_pro': profile.is_threadripper_pro
            },
            'configuration_options': {
                'conservative': {
                    'description': 'Safe settings for unknown hardware',
                    'batch_size': 1,
                    'tile_size': [256, 256],
                    'enable_cpu_offload': True,
                    'memory_fraction': 0.7,
                    'use_fp16': True,
                    'use_bf16': False
                },
                'balanced': {
                    'description': 'Balanced performance and stability',
                    'batch_size': 2 if profile.vram_gb >= 12 else 1,
                    'tile_size': [384, 384] if profile.vram_gb >= 12 else [256, 256],
                    'enable_cpu_offload': profile.vram_gb < 16,
                    'memory_fraction': 0.85,
                    'use_fp16': True,
                    'use_bf16': profile.is_rtx_4080
                },
                'aggressive': {
                    'description': 'Maximum performance (requires sufficient VRAM)',
                    'batch_size': 4 if profile.vram_gb >= 16 else 2,
                    'tile_size': [512, 512] if profile.vram_gb >= 16 else [384, 384],
                    'enable_cpu_offload': False,
                    'memory_fraction': 0.95,
                    'use_fp16': False,
                    'use_bf16': profile.is_rtx_4080
                }
            },
            'hardware_specific_recommendations': [],
            'warnings': [],
            'manual_overrides': {
                'vram_gb': f'If VRAM detection is incorrect, manually set to your GPU\'s VRAM capacity',
                'cpu_cores': f'If CPU detection is incorrect, manually set to your CPU\'s physical core count',
                'batch_size': 'Increase for better GPU utilization, decrease if running out of VRAM',
                'tile_size': 'Increase for better quality, decrease if running out of VRAM'
            }
        }
        
        # Add hardware-specific recommendations
        if profile.is_rtx_4080:
            guide['hardware_specific_recommendations'].extend([
                'Enable Tensor Cores (TF32) for optimal RTX 4080 performance',
                'Use BF16 precision for best performance/quality balance',
                'Consider disabling CPU offloading if you have 16GB VRAM',
                'Optimal tile size for RTX 4080 is 256x256 for VAE operations'
            ])
        
        if profile.is_threadripper_pro:
            guide['hardware_specific_recommendations'].extend([
                'Utilize multiple CPU cores with parallel workers',
                'Enable NUMA optimization for better memory bandwidth',
                'Consider higher thread counts for preprocessing',
                'CPU offloading may be less beneficial with powerful CPU'
            ])
        
        # Add warnings based on detected hardware
        if profile.vram_gb < 8:
            guide['warnings'].append('Low VRAM detected - use conservative settings')
        
        if profile.cpu_cores < 4:
            guide['warnings'].append('Low CPU core count - limit parallel processing')
        
        if profile.total_memory_gb < 16:
            guide['warnings'].append('Low system RAM - enable CPU offloading carefully')
        
        return guide