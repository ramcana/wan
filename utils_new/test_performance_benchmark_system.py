"""
Test suite for Performance Benchmark System
"""

import pytest
import time
import json
import tempfile
import threading
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from performance_benchmark_system import (
    PerformanceMetrics, BenchmarkResult, HardwareLimits,
    SystemMonitor, PerformanceBenchmarkSystem
)
from hardware_optimizer import HardwareProfile, OptimalSettings

class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass"""
    
    def test_performance_metrics_creation(self):
        """Test creating PerformanceMetrics"""
        metrics = PerformanceMetrics(
            timestamp="2024-01-01T12:00:00",
            model_load_time=120.5,
            generation_time=45.2,
            peak_vram_usage_mb=8192,
            gpu_utilization_avg=85.5
        )
        
        assert metrics.timestamp == "2024-01-01T12:00:00"
        assert metrics.model_load_time == 120.5
        assert metrics.generation_time == 45.2
        assert metrics.peak_vram_usage_mb == 8192
        assert metrics.gpu_utilization_avg == 85.5
    
    def test_performance_metrics_defaults(self):
        """Test PerformanceMetrics with default values"""
        metrics = PerformanceMetrics(timestamp="2024-01-01T12:00:00")
        
        assert metrics.model_load_time == 0.0
        assert metrics.generation_time == 0.0
        assert metrics.peak_vram_usage_mb == 0
        assert metrics.settings_used is None

class TestBenchmarkResult:
    """Test BenchmarkResult dataclass"""
    
    def test_benchmark_result_creation(self):
        """Test creating BenchmarkResult"""
        before_metrics = PerformanceMetrics(
            timestamp="2024-01-01T12:00:00",
            total_time=100.0,
            peak_vram_usage_mb=10000
        )
        after_metrics = PerformanceMetrics(
            timestamp="2024-01-01T12:05:00",
            total_time=80.0,
            peak_vram_usage_mb=8000
        )
        
        result = BenchmarkResult(
            success=True,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            performance_improvement=20.0,
            memory_savings=2000,
            recommendations=["Enable tensor cores"],
            warnings=["High temperature"],
            errors=[]
        )
        
        assert result.success is True
        assert result.performance_improvement == 20.0
        assert result.memory_savings == 2000
        assert len(result.recommendations) == 1
        assert len(result.warnings) == 1

class TestHardwareLimits:
    """Test HardwareLimits dataclass"""
    
    def test_hardware_limits_creation(self):
        """Test creating HardwareLimits"""
        limits = HardwareLimits(
            max_vram_usage_mb=16384,
            max_ram_usage_mb=32768,
            max_gpu_temperature=83.0,
            max_cpu_temperature=90.0
        )
        
        assert limits.max_vram_usage_mb == 16384
        assert limits.max_ram_usage_mb == 32768
        assert limits.max_gpu_temperature == 83.0
        assert limits.target_gpu_utilization == 95.0  # Default value

class TestSystemMonitor:
    """Test SystemMonitor class"""
    
    def test_system_monitor_initialization(self):
        """Test SystemMonitor initialization"""
        monitor = SystemMonitor()
        
        assert monitor.monitoring is False
        assert monitor.metrics == []
        assert monitor.monitor_thread is None
    
    @patch('performance_benchmark_system.psutil.cpu_percent')
    @patch('performance_benchmark_system.psutil.virtual_memory')
    def test_collect_instant_metrics(self, mock_memory, mock_cpu):
        """Test collecting instant metrics"""
        # Mock system metrics
        mock_cpu.return_value = 45.5
        mock_memory.return_value = Mock(total=16*1024**3, available=8*1024**3)
        
        monitor = SystemMonitor()
        metrics = monitor._collect_instant_metrics()
        
        assert 'cpu_utilization' in metrics
        assert 'ram_usage_mb' in metrics
        assert metrics['cpu_utilization'] == 45.5
        assert metrics['ram_usage_mb'] == 8192.0  # 8GB in MB
    
    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring"""
        monitor = SystemMonitor()
        
        # Start monitoring
        monitor.start_monitoring(interval=0.1)
        assert monitor.monitoring is True
        assert monitor.monitor_thread is not None
        
        # Let it run briefly
        time.sleep(0.2)
        
        # Stop monitoring
        aggregated = monitor.stop_monitoring()
        assert monitor.monitoring is False
        assert isinstance(aggregated, dict)
    
    def test_nvml_metrics_collection(self):
        """Test NVML metrics collection"""
        with patch('performance_benchmark_system.NVML_AVAILABLE', True), \
             patch('performance_benchmark_system.nvml') as mock_nvml:
            
            # Mock NVML functions
            mock_handle = Mock()
            mock_nvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
            mock_nvml.nvmlDeviceGetUtilizationRates.return_value = Mock(gpu=85)
            mock_nvml.nvmlDeviceGetTemperature.return_value = 75
            mock_nvml.nvmlDeviceGetPowerUsage.return_value = 250000  # mW
            mock_nvml.nvmlDeviceGetMemoryInfo.return_value = Mock(used=8*1024**3)
            
            monitor = SystemMonitor()
            monitor.nvml_initialized = True
            
            metrics = monitor._collect_instant_metrics()
            
            assert metrics['gpu_utilization'] == 85
            assert metrics['gpu_temperature'] == 75
            assert metrics['gpu_power_draw'] == 250.0  # Converted to watts
            assert metrics['vram_usage_mb'] == 8192.0

class TestPerformanceBenchmarkSystem:
    """Test PerformanceBenchmarkSystem class"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.benchmark_system = PerformanceBenchmarkSystem(results_dir=self.temp_dir)
        
        # Create test hardware profile
        self.rtx_4080_profile = HardwareProfile(
            cpu_model="Intel i7-12700K",
            cpu_cores=12,
            total_memory_gb=32,
            gpu_model="RTX 4080",
            vram_gb=16,
            is_rtx_4080=True,
            is_threadripper_pro=False
        )
        
        self.threadripper_profile = HardwareProfile(
            cpu_model="AMD Threadripper PRO 5995WX",
            cpu_cores=64,
            total_memory_gb=128,
            gpu_model="RTX 4080",
            vram_gb=16,
            is_rtx_4080=True,
            is_threadripper_pro=True
        )
    
    def test_initialization(self):
        """Test PerformanceBenchmarkSystem initialization"""
        assert self.benchmark_system.results_dir.exists()
        assert isinstance(self.benchmark_system.monitor, SystemMonitor)
        assert 'rtx_4080' in self.benchmark_system.benchmark_targets
        assert 'threadripper_pro' in self.benchmark_system.benchmark_targets
    
    def test_create_hardware_limits_rtx_4080(self):
        """Test creating hardware limits for RTX 4080"""
        limits = self.benchmark_system.create_hardware_limits(self.rtx_4080_profile)
        
        assert limits.max_vram_usage_mb == 16384  # 16GB
        assert limits.max_gpu_temperature == 83.0
        assert limits.max_gpu_power_draw == 320.0
        assert limits.target_gpu_utilization == 95.0
        assert limits.target_memory_efficiency == 0.85
    
    def test_create_hardware_limits_threadripper_pro(self):
        """Test creating hardware limits for Threadripper PRO"""
        limits = self.benchmark_system.create_hardware_limits(self.threadripper_profile)
        
        assert limits.max_vram_usage_mb == 16384  # 16GB VRAM
        assert limits.max_ram_usage_mb == 65536  # Half of 128GB
        assert limits.max_cpu_temperature == 90.0
        assert limits.target_memory_efficiency == 0.90
    
    def test_benchmark_model_loading(self):
        """Test model loading benchmark"""
        def mock_model_load():
            time.sleep(0.1)  # Simulate loading time
            return "model_loaded"
        
        with patch.object(self.benchmark_system.monitor, 'start_monitoring'), \
             patch.object(self.benchmark_system.monitor, 'stop_monitoring') as mock_stop:
            
            mock_stop.return_value = {
                'peak_vram_usage_mb': 8192,
                'peak_ram_usage_mb': 4096,
                'gpu_utilization_avg': 75.0,
                'gpu_temperature_max': 70.0,
                'gpu_power_draw_avg': 200.0,
                'cpu_utilization_avg': 25.0,
                'cpu_temperature_max': 60.0
            }
            
            metrics = self.benchmark_system.benchmark_model_loading(
                mock_model_load,
                model_name="test_model"
            )
            
            assert metrics.model_load_time > 0.05  # At least 50ms
            assert metrics.peak_vram_usage_mb == 8192
            assert metrics.gpu_utilization_avg == 75.0
    
    def test_benchmark_generation(self):
        """Test generation benchmark"""
        def mock_generation(**kwargs):
            time.sleep(0.2)  # Simulate generation time
            return ["generated_output"]
        
        with patch.object(self.benchmark_system.monitor, 'start_monitoring'), \
             patch.object(self.benchmark_system.monitor, 'stop_monitoring') as mock_stop:
            
            mock_stop.return_value = {
                'peak_vram_usage_mb': 12288,
                'peak_ram_usage_mb': 6144,
                'gpu_utilization_avg': 95.0,
                'gpu_temperature_max': 80.0,
                'gpu_power_draw_avg': 300.0,
                'cpu_utilization_avg': 15.0,
                'cpu_temperature_max': 55.0
            }
            
            metrics = self.benchmark_system.benchmark_generation(
                mock_generation,
                {"prompt": "test prompt"},
                expected_output_count=1
            )
            
            assert metrics.generation_time > 0.15  # At least 150ms
            assert metrics.peak_vram_usage_mb == 12288
            assert metrics.throughput_items_per_second > 0
    
    def test_validate_against_hardware_limits(self):
        """Test hardware limits validation"""
        limits = HardwareLimits(
            max_vram_usage_mb=16384,
            max_ram_usage_mb=32768,
            max_gpu_temperature=83.0,
            max_cpu_temperature=90.0,
            max_gpu_power_draw=320.0
        )
        
        # Test valid metrics
        valid_metrics = PerformanceMetrics(
            timestamp="2024-01-01T12:00:00",
            peak_vram_usage_mb=12288,
            peak_ram_usage_mb=16384,
            gpu_temperature_max=75.0,
            cpu_temperature_max=65.0,
            gpu_power_draw_avg=250.0,
            vram_efficiency=0.75
        )
        
        is_valid, warnings, errors = self.benchmark_system.validate_against_hardware_limits(
            valid_metrics, limits, self.rtx_4080_profile
        )
        
        assert is_valid is True
        assert len(errors) == 0
        
        # Test invalid metrics
        invalid_metrics = PerformanceMetrics(
            timestamp="2024-01-01T12:00:00",
            peak_vram_usage_mb=18432,  # Exceeds limit
            gpu_temperature_max=85.0,  # Exceeds limit
            vram_efficiency=0.98  # Very high
        )
        
        is_valid, warnings, errors = self.benchmark_system.validate_against_hardware_limits(
            invalid_metrics, limits, self.rtx_4080_profile
        )
        
        assert is_valid is False
        assert len(errors) > 0
        assert len(warnings) > 0
    
    def test_generate_performance_recommendations(self):
        """Test performance recommendations generation"""
        # High VRAM usage scenario
        high_vram_metrics = PerformanceMetrics(
            timestamp="2024-01-01T12:00:00",
            vram_efficiency=0.95,
            gpu_utilization_avg=60.0,
            gpu_temperature_max=80.0,
            peak_vram_usage_mb=15360
        )
        
        recommendations = self.benchmark_system.generate_performance_recommendations(
            high_vram_metrics, self.rtx_4080_profile
        )
        
        assert len(recommendations) > 0
        assert any("CPU offloading" in rec for rec in recommendations)
        assert any("batch size" in rec for rec in recommendations)
    
    def test_run_before_after_benchmark(self):
        """Test before/after benchmark execution"""
        def mock_before_benchmark():
            return PerformanceMetrics(
                timestamp="2024-01-01T12:00:00",
                total_time=100.0,
                peak_vram_usage_mb=12288,
                gpu_utilization_avg=70.0
            )
        
        def mock_after_benchmark():
            return PerformanceMetrics(
                timestamp="2024-01-01T12:05:00",
                total_time=80.0,
                peak_vram_usage_mb=10240,
                gpu_utilization_avg=85.0
            )
        
        result = self.benchmark_system.run_before_after_benchmark(
            mock_before_benchmark,
            mock_after_benchmark,
            self.rtx_4080_profile,
            "test_optimization"
        )
        
        assert result.success is True
        assert result.performance_improvement == 20.0  # (100-80)/100 * 100
        assert result.memory_savings == 2048  # 12288 - 10240
        assert len(result.recommendations) > 0
    
    def test_save_and_load_benchmark_result(self):
        """Test saving and loading benchmark results"""
        # Create test result
        metrics = PerformanceMetrics(
            timestamp="2024-01-01T12:00:00",
            total_time=50.0,
            peak_vram_usage_mb=8192
        )
        
        result = BenchmarkResult(
            success=True,
            before_metrics=metrics,
            after_metrics=metrics,
            performance_improvement=15.0,
            memory_savings=1024,
            recommendations=["Test recommendation"],
            warnings=["Test warning"],
            errors=[]
        )
        
        # Save result
        self.benchmark_system.save_benchmark_result(result, "test_benchmark")
        
        # Check file was created
        result_files = list(self.benchmark_system.results_dir.glob("benchmark_test_benchmark_*.json"))
        assert len(result_files) == 1
        
        # Load result
        loaded_result = self.benchmark_system.load_benchmark_result(str(result_files[0]))
        
        assert loaded_result is not None
        assert loaded_result.success is True
        assert loaded_result.performance_improvement == 15.0
        assert loaded_result.memory_savings == 1024
    
    def test_generate_benchmark_report(self):
        """Test benchmark report generation"""
        # Create test results
        results = [
            BenchmarkResult(
                success=True,
                before_metrics=PerformanceMetrics(
                    timestamp="2024-01-01T12:00:00",
                    total_time=100.0,
                    peak_vram_usage_mb=12288
                ),
                after_metrics=PerformanceMetrics(
                    timestamp="2024-01-01T12:05:00",
                    total_time=80.0,
                    peak_vram_usage_mb=10240
                ),
                performance_improvement=20.0,
                memory_savings=2048,
                recommendations=["Enable tensor cores"],
                warnings=[],
                errors=[],
                benchmark_type="optimization_test"
            ),
            BenchmarkResult(
                success=True,
                before_metrics=None,
                after_metrics=None,
                performance_improvement=10.0,
                memory_savings=1024,
                recommendations=["Use mixed precision"],
                warnings=["High temperature"],
                errors=[],
                benchmark_type="memory_test"
            )
        ]
        
        report = self.benchmark_system.generate_benchmark_report(results)
        
        assert "WAN22 Performance Benchmark Report" in report
        assert "Total benchmarks: 2" in report
        assert "Successful benchmarks: 2" in report
        assert "Average performance improvement: 15.0%" in report
        assert "Total memory savings: 3072MB" in report
        assert "optimization_test" in report
        assert "memory_test" in report
    
    def test_generate_recommended_settings_for_hardware(self):
        """Test generating recommended settings based on hardware detection"""
        # Test RTX 4080 settings
        rtx_settings = self.benchmark_system.generate_recommended_settings_for_hardware(
            self.rtx_4080_profile
        )
        
        assert rtx_settings.batch_size >= 1
        assert rtx_settings.enable_tensor_cores is True
        assert rtx_settings.use_bf16 is True
        
        # Test Threadripper PRO settings
        threadripper_settings = self.benchmark_system.generate_recommended_settings_for_hardware(
            self.threadripper_profile
        )
        
        assert threadripper_settings.num_threads > 4
        assert threadripper_settings.parallel_workers >= 1
        
        # Test fallback mode
        fallback_settings = self.benchmark_system.generate_recommended_settings_for_hardware(
            self.rtx_4080_profile, fallback_mode=True
        )
        
        assert fallback_settings is not None
    
    def test_validate_settings_against_hardware_limits(self):
        """Test settings validation against hardware limits"""
        from hardware_optimizer import OptimalSettings
        
        # Test valid settings
        valid_settings = OptimalSettings(
            tile_size=(256, 256),
            vae_tile_size=(256, 256),
            batch_size=2,
            enable_cpu_offload=True,
            text_encoder_offload=True,
            vae_offload=True,
            enable_tensor_cores=True,
            use_fp16=True,
            use_bf16=False,
            memory_fraction=0.8,
            gradient_checkpointing=True,
            num_threads=8,
            enable_xformers=True,
            parallel_workers=2,
            enable_numa_optimization=False,
            preprocessing_threads=4,
            io_threads=2
        )
        
        is_valid, warnings, errors = self.benchmark_system.validate_settings_against_hardware_limits(
            valid_settings, self.rtx_4080_profile
        )
        
        assert isinstance(is_valid, bool)
        assert isinstance(warnings, list)
        assert isinstance(errors, list)
        
        # Test invalid settings (too many threads)
        invalid_settings = OptimalSettings(
            tile_size=(256, 256),
            vae_tile_size=(256, 256),
            batch_size=1,
            enable_cpu_offload=True,
            text_encoder_offload=True,
            vae_offload=True,
            enable_tensor_cores=True,
            use_fp16=True,
            use_bf16=False,
            memory_fraction=0.8,
            gradient_checkpointing=True,
            num_threads=100,  # Too many threads
            enable_xformers=True,
            parallel_workers=1,
            enable_numa_optimization=False,
            preprocessing_threads=1,
            io_threads=1
        )
        
        is_valid, warnings, errors = self.benchmark_system.validate_settings_against_hardware_limits(
            invalid_settings, self.rtx_4080_profile
        )
        
        assert is_valid is False
        assert len(errors) > 0
    
    def test_handle_hardware_detection_failure(self):
        """Test hardware detection failure handling"""
        # Test with no partial profile
        profile, settings = self.benchmark_system.handle_hardware_detection_failure()
        
        assert profile is not None
        assert settings is not None
        assert profile.cpu_cores > 0
        assert profile.total_memory_gb > 0
        assert profile.vram_gb > 0
        
        # Test with partial profile - use minimal values that won't trigger auto-detection
        partial_profile = HardwareProfile(
            cpu_model="Partial CPU",
            cpu_cores=8,
            total_memory_gb=32,  # Higher value to avoid override
            gpu_model="Test GPU",
            vram_gb=12,  # Higher value to avoid override
            is_rtx_4080=False,
            is_threadripper_pro=False
        )
        
        profile, settings = self.benchmark_system.handle_hardware_detection_failure(partial_profile)
        
        assert profile.cpu_model == "Partial CPU"
        assert profile.cpu_cores == 8
        assert profile.gpu_model == "Test GPU"
        assert settings is not None
    
    def test_create_manual_configuration_guide(self):
        """Test manual configuration guide creation"""
        guide = self.benchmark_system.create_manual_configuration_guide(self.rtx_4080_profile)
        
        assert 'hardware_profile' in guide
        assert 'configuration_options' in guide
        assert 'hardware_specific_recommendations' in guide
        assert 'warnings' in guide
        assert 'manual_overrides' in guide
        
        # Check configuration options
        assert 'conservative' in guide['configuration_options']
        assert 'balanced' in guide['configuration_options']
        assert 'aggressive' in guide['configuration_options']
        
        # Check RTX 4080 specific recommendations
        rtx_recommendations = guide['hardware_specific_recommendations']
        assert any('RTX 4080' in rec for rec in rtx_recommendations)
        
        # Test Threadripper PRO guide
        threadripper_guide = self.benchmark_system.create_manual_configuration_guide(self.threadripper_profile)
        threadripper_recommendations = threadripper_guide['hardware_specific_recommendations']
        assert any('CPU cores' in rec or 'NUMA' in rec for rec in threadripper_recommendations)
    
    def test_detailed_performance_comparison(self):
        """Test detailed performance comparison generation"""
        before_metrics = PerformanceMetrics(
            timestamp="2024-01-01T12:00:00",
            model_load_time=120.0,
            generation_time=60.0,
            total_time=180.0,
            peak_vram_usage_mb=12288,
            peak_ram_usage_mb=8192,
            gpu_utilization_avg=70.0,
            throughput_items_per_second=0.5,
            vram_efficiency=0.8
        )
        
        after_metrics = PerformanceMetrics(
            timestamp="2024-01-01T12:05:00",
            model_load_time=90.0,
            generation_time=45.0,
            total_time=135.0,
            peak_vram_usage_mb=10240,
            peak_ram_usage_mb=6144,
            gpu_utilization_avg=85.0,
            throughput_items_per_second=0.7,
            vram_efficiency=0.7
        )
        
        comparison = self.benchmark_system._generate_detailed_performance_comparison(
            before_metrics, after_metrics
        )
        
        assert 'timing_comparison' in comparison
        assert 'memory_comparison' in comparison
        assert 'performance_comparison' in comparison
        assert 'thermal_comparison' in comparison
        
        # Check timing improvements
        timing = comparison['timing_comparison']
        assert timing['model_load_time']['improvement_seconds'] == 30.0
        assert timing['generation_time']['improvement_seconds'] == 15.0
        assert timing['total_time']['improvement_seconds'] == 45.0
        
        # Check memory savings
        memory = comparison['memory_comparison']
        assert memory['peak_vram_usage_mb']['savings_mb'] == 2048
        assert memory['peak_ram_usage_mb']['savings_mb'] == 2048
    
    def test_empty_benchmark_report(self):
        """Test benchmark report with no results"""
        report = self.benchmark_system.generate_benchmark_report([])
        
        assert "WAN22 Performance Benchmark Report" in report
        assert "No benchmark results available." in report

class TestIntegration:
    """Integration tests for the performance benchmark system"""
    
    def test_full_benchmark_workflow(self):
        """Test complete benchmark workflow"""
        temp_dir = tempfile.mkdtemp()
        benchmark_system = PerformanceBenchmarkSystem(results_dir=temp_dir)
        
        # Create hardware profile
        profile = HardwareProfile(
            cpu_model="Test CPU",
            cpu_cores=8,
            total_memory_gb=16,
            gpu_model="Test GPU",
            vram_gb=8,
            is_rtx_4080=False,
            is_threadripper_pro=False
        )
        
        # Mock benchmark functions
        def mock_model_load():
            time.sleep(0.05)
            return "model"
        
        def mock_generation(**kwargs):
            time.sleep(0.1)
            return ["output"]
        
        # Run model loading benchmark
        with patch.object(benchmark_system.monitor, 'start_monitoring'), \
             patch.object(benchmark_system.monitor, 'stop_monitoring') as mock_stop:
            
            mock_stop.return_value = {
                'peak_vram_usage_mb': 4096,
                'peak_ram_usage_mb': 2048,
                'gpu_utilization_avg': 80.0,
                'gpu_temperature_max': 70.0,
                'gpu_power_draw_avg': 150.0,
                'cpu_utilization_avg': 30.0,
                'cpu_temperature_max': 55.0
            }
            
            load_metrics = benchmark_system.benchmark_model_loading(mock_model_load)
            gen_metrics = benchmark_system.benchmark_generation(
                mock_generation, {"prompt": "test"}, 1
            )
        
        # Validate metrics
        assert load_metrics.model_load_time > 0
        assert gen_metrics.generation_time > 0
        
        # Test hardware limits validation
        limits = benchmark_system.create_hardware_limits(profile)
        is_valid, warnings, errors = benchmark_system.validate_against_hardware_limits(
            load_metrics, limits, profile
        )
        
        assert isinstance(is_valid, bool)
        assert isinstance(warnings, list)
        assert isinstance(errors, list)
        
        # Test recommendations
        recommendations = benchmark_system.generate_performance_recommendations(
            gen_metrics, profile
        )
        
        assert isinstance(recommendations, list)

if __name__ == "__main__":
    pytest.main([__file__])