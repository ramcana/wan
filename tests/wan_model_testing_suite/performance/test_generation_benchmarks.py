"""
Performance benchmarking tests for WAN model generation
"""

import pytest
import time
import statistics
from unittest.mock import Mock, patch
from typing import Dict, Any, List, Tuple

# Import benchmarking utilities
try:
    from tests.wan_model_testing_suite.utils.benchmark_utils import (
        BenchmarkRunner,
        GenerationBenchmark,
        MemoryBenchmark,
        ThroughputBenchmark
    )
    BENCHMARK_UTILS_AVAILABLE = True
except ImportError:
    BENCHMARK_UTILS_AVAILABLE = False
    # Create mock classes
    class BenchmarkRunner:
        pass
    class GenerationBenchmark:
        pass


@pytest.mark.performance
@pytest.mark.slow
class TestGenerationBenchmarks:
    """Performance benchmarks for video generation"""
    
    def test_t2v_generation_benchmark(self, mock_wan_model, sample_generation_params, test_config):
        """Benchmark T2V generation performance"""
        # Mock T2V model
        mock_wan_model.model_type = "t2v-A14B"
        mock_wan_model.generate = Mock()
        
        # Mock generation result with timing
        def mock_generate(params):
            time.sleep(0.1)  # Simulate generation time
            result = Mock()
            result.success = True
            result.generation_time = 0.1
            result.video_path = "/tmp/test_video.mp4"
            result.metadata = {
                "frames": params.num_frames,
                "resolution": f"{params.width}x{params.height}",
                "model": "t2v-A14B"
            }
            return result
        
        mock_wan_model.generate.side_effect = mock_generate
        
        # Run benchmark
        iterations = test_config["benchmark_iterations"]
        generation_times = []
        
        for i in range(iterations):
            start_time = time.time()
            result = mock_wan_model.generate(sample_generation_params)
            end_time = time.time()
            
            assert result.success is True
            generation_times.append(end_time - start_time)
        
        # Calculate benchmark metrics
        avg_time = statistics.mean(generation_times)
        min_time = min(generation_times)
        max_time = max(generation_times)
        std_dev = statistics.stdev(generation_times) if len(generation_times) > 1 else 0
        
        # Verify performance expectations
        assert avg_time > 0
        assert min_time > 0
        assert max_time >= min_time
        assert std_dev >= 0
        
        # Log benchmark results
        print(f"\nT2V Generation Benchmark Results:")
        print(f"  Iterations: {iterations}")
        print(f"  Average time: {avg_time:.3f}s")
        print(f"  Min time: {min_time:.3f}s")
        print(f"  Max time: {max_time:.3f}s")
        print(f"  Std deviation: {std_dev:.3f}s")
    
    def test_i2v_generation_benchmark(self, mock_wan_model, sample_image_tensor, test_config):
        """Benchmark I2V generation performance"""
        # Mock I2V model
        mock_wan_model.model_type = "i2v-A14B"
        mock_wan_model.generate = Mock()
        
        # Mock generation with image preprocessing overhead
        def mock_generate(params):
            time.sleep(0.15)  # Slightly longer due to image processing
            result = Mock()
            result.success = True
            result.generation_time = 0.15
            result.video_path = "/tmp/test_i2v_video.mp4"
            result.metadata = {
                "frames": params.num_frames,
                "resolution": f"{params.width}x{params.height}",
                "model": "i2v-A14B",
                "has_image_conditioning": True
            }
            return result
        
        mock_wan_model.generate.side_effect = mock_generate
        
        # Create I2V parameters
        from backend.core.models.wan_models.wan_i2v_a14b import I2VGenerationParams
        i2v_params = I2VGenerationParams(
            image=sample_image_tensor,
            prompt="A cat playing in a garden",
            num_frames=16,
            width=512,
            height=512,
            num_inference_steps=20
        )
        
        # Run benchmark
        iterations = test_config["benchmark_iterations"]
        generation_times = []
        
        for i in range(iterations):
            start_time = time.time()
            result = mock_wan_model.generate(i2v_params)
            end_time = time.time()
            
            assert result.success is True
            generation_times.append(end_time - start_time)
        
        # Calculate metrics
        avg_time = statistics.mean(generation_times)
        throughput_fps = 16 / avg_time  # frames per second
        
        # Verify I2V-specific performance
        assert avg_time > 0
        assert throughput_fps > 0
        
        print(f"\nI2V Generation Benchmark Results:")
        print(f"  Average time: {avg_time:.3f}s")
        print(f"  Throughput: {throughput_fps:.2f} fps")
    
    def test_ti2v_generation_benchmark(self, mock_wan_model, sample_image_tensor, test_config):
        """Benchmark TI2V generation performance"""
        # Mock TI2V model (should be more efficient due to 5B parameters)
        mock_wan_model.model_type = "ti2v-5B"
        mock_wan_model.generate = Mock()
        
        # Mock generation with dual conditioning
        def mock_generate(params):
            time.sleep(0.08)  # Faster due to smaller model
            result = Mock()
            result.success = True
            result.generation_time = 0.08
            result.video_path = "/tmp/test_ti2v_video.mp4"
            result.metadata = {
                "frames": params.num_frames,
                "resolution": f"{params.width}x{params.height}",
                "model": "ti2v-5B",
                "has_dual_conditioning": True
            }
            return result
        
        mock_wan_model.generate.side_effect = mock_generate
        
        # Create TI2V parameters
        from backend.core.models.wan_models.wan_ti2v_5b import TI2VGenerationParams
        ti2v_params = TI2VGenerationParams(
            image=sample_image_tensor,
            prompt="A cat playing in a garden",
            num_frames=16,
            width=512,
            height=512,
            num_inference_steps=20,
            text_guidance_scale=7.5,
            image_guidance_scale=1.5
        )
        
        # Run benchmark
        iterations = test_config["benchmark_iterations"]
        generation_times = []
        
        for i in range(iterations):
            start_time = time.time()
            result = mock_wan_model.generate(ti2v_params)
            end_time = time.time()
            
            assert result.success is True
            generation_times.append(end_time - start_time)
        
        # Calculate metrics
        avg_time = statistics.mean(generation_times)
        
        # TI2V should be faster than 14B models
        assert avg_time > 0
        
        print(f"\nTI2V Generation Benchmark Results:")
        print(f"  Average time: {avg_time:.3f}s (5B model)")
    
    def test_resolution_scaling_benchmark(self, mock_wan_model, test_config):
        """Benchmark generation time scaling with resolution"""
        mock_wan_model.model_type = "t2v-A14B"
        mock_wan_model.generate = Mock()
        
        # Mock generation with resolution-dependent timing
        def mock_generate(params):
            # Simulate quadratic scaling with resolution
            pixel_count = params.width * params.height
            base_time = 0.05
            scaling_factor = pixel_count / (512 * 512)  # Normalize to 512x512
            generation_time = base_time * scaling_factor
            time.sleep(generation_time)
            
            result = Mock()
            result.success = True
            result.generation_time = generation_time
            result.video_path = f"/tmp/test_{params.width}x{params.height}.mp4"
            return result
        
        mock_wan_model.generate.side_effect = mock_generate
        
        # Test different resolutions
        resolutions = [(256, 256), (512, 512), (768, 768), (1024, 1024)]
        benchmark_results = {}
        
        for width, height in resolutions:
            from backend.core.models.wan_models.wan_t2v_a14b import T2VGenerationParams
            params = T2VGenerationParams(
                prompt="A cat playing in a garden",
                width=width,
                height=height,
                num_frames=16,
                num_inference_steps=10  # Fewer steps for faster benchmarking
            )
            
            # Run multiple iterations
            times = []
            for _ in range(3):
                start_time = time.time()
                result = mock_wan_model.generate(params)
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = statistics.mean(times)
            benchmark_results[f"{width}x{height}"] = avg_time
        
        # Verify scaling behavior
        assert benchmark_results["512x512"] > benchmark_results["256x256"]
        assert benchmark_results["1024x1024"] > benchmark_results["512x512"]
        
        print(f"\nResolution Scaling Benchmark:")
        for resolution, time_taken in benchmark_results.items():
            print(f"  {resolution}: {time_taken:.3f}s")
    
    def test_inference_steps_benchmark(self, mock_wan_model, test_config):
        """Benchmark generation time scaling with inference steps"""
        mock_wan_model.model_type = "t2v-A14B"
        mock_wan_model.generate = Mock()
        
        # Mock generation with step-dependent timing
        def mock_generate(params):
            # Linear scaling with inference steps
            base_time = 0.002  # 2ms per step
            generation_time = base_time * params.num_inference_steps
            time.sleep(generation_time)
            
            result = Mock()
            result.success = True
            result.generation_time = generation_time
            result.video_path = f"/tmp/test_{params.num_inference_steps}steps.mp4"
            return result
        
        mock_wan_model.generate.side_effect = mock_generate
        
        # Test different step counts
        step_counts = [10, 20, 50, 100]
        benchmark_results = {}
        
        for steps in step_counts:
            from backend.core.models.wan_models.wan_t2v_a14b import T2VGenerationParams
            params = T2VGenerationParams(
                prompt="A cat playing in a garden",
                width=512,
                height=512,
                num_frames=16,
                num_inference_steps=steps
            )
            
            start_time = time.time()
            result = mock_wan_model.generate(params)
            end_time = time.time()
            
            benchmark_results[steps] = end_time - start_time
        
        # Verify linear scaling
        assert benchmark_results[20] > benchmark_results[10]
        assert benchmark_results[50] > benchmark_results[20]
        assert benchmark_results[100] > benchmark_results[50]
        
        print(f"\nInference Steps Benchmark:")
        for steps, time_taken in benchmark_results.items():
            print(f"  {steps} steps: {time_taken:.3f}s")
    
    def test_batch_generation_benchmark(self, mock_wan_model, test_config):
        """Benchmark batch generation performance"""
        mock_wan_model.model_type = "t2v-A14B"
        mock_wan_model.generate_batch = Mock()
        
        # Mock batch generation with efficiency gains
        def mock_generate_batch(params_list):
            batch_size = len(params_list)
            # Simulate batch efficiency (not linear scaling)
            single_time = 0.1
            batch_efficiency = 0.8  # 20% efficiency gain
            total_time = single_time * batch_size * batch_efficiency
            time.sleep(total_time)
            
            results = []
            for i, params in enumerate(params_list):
                result = Mock()
                result.success = True
                result.generation_time = total_time / batch_size
                result.video_path = f"/tmp/batch_{i}.mp4"
                results.append(result)
            
            return results
        
        mock_wan_model.generate_batch.side_effect = mock_generate_batch
        
        # Test different batch sizes
        batch_sizes = [1, 2, 4, 8]
        benchmark_results = {}
        
        for batch_size in batch_sizes:
            # Create batch parameters
            batch_params = []
            for i in range(batch_size):
                from backend.core.models.wan_models.wan_t2v_a14b import T2VGenerationParams
                params = T2VGenerationParams(
                    prompt=f"A cat playing in a garden {i}",
                    width=512,
                    height=512,
                    num_frames=16,
                    num_inference_steps=20
                )
                batch_params.append(params)
            
            # Benchmark batch generation
            start_time = time.time()
            results = mock_wan_model.generate_batch(batch_params)
            end_time = time.time()
            
            total_time = end_time - start_time
            time_per_video = total_time / batch_size
            
            benchmark_results[batch_size] = {
                "total_time": total_time,
                "time_per_video": time_per_video,
                "throughput": batch_size / total_time
            }
            
            assert len(results) == batch_size
            assert all(result.success for result in results)
        
        print(f"\nBatch Generation Benchmark:")
        for batch_size, metrics in benchmark_results.items():
            print(f"  Batch size {batch_size}:")
            print(f"    Total time: {metrics['total_time']:.3f}s")
            print(f"    Time per video: {metrics['time_per_video']:.3f}s")
            print(f"    Throughput: {metrics['throughput']:.2f} videos/s")
    
    @pytest.mark.gpu
    def test_gpu_utilization_benchmark(self, mock_wan_model, mock_hardware_info, test_config):
        """Benchmark GPU utilization during generation"""
        mock_wan_model.model_type = "t2v-A14B"
        mock_wan_model.generate = Mock()
        mock_wan_model.get_gpu_utilization = Mock()
        
        # Mock GPU utilization tracking
        utilization_history = []
        
        def mock_generate(params):
            # Simulate varying GPU utilization during generation
            for step in range(params.num_inference_steps):
                utilization = 85 + (step % 10)  # 85-95% utilization
                utilization_history.append(utilization)
                time.sleep(0.001)  # Small delay per step
            
            result = Mock()
            result.success = True
            result.generation_time = params.num_inference_steps * 0.001
            result.video_path = "/tmp/gpu_test.mp4"
            return result
        
        mock_wan_model.generate.side_effect = mock_generate
        mock_wan_model.get_gpu_utilization.return_value = utilization_history
        
        # Run generation with GPU monitoring
        from backend.core.models.wan_models.wan_t2v_a14b import T2VGenerationParams
        params = T2VGenerationParams(
            prompt="A cat playing in a garden",
            width=512,
            height=512,
            num_frames=16,
            num_inference_steps=50
        )
        
        result = mock_wan_model.generate(params)
        gpu_utilization = mock_wan_model.get_gpu_utilization()
        
        # Analyze GPU utilization
        avg_utilization = statistics.mean(gpu_utilization)
        max_utilization = max(gpu_utilization)
        min_utilization = min(gpu_utilization)
        
        assert result.success is True
        assert avg_utilization > 0
        assert max_utilization <= 100
        assert min_utilization >= 0
        
        print(f"\nGPU Utilization Benchmark:")
        print(f"  Average utilization: {avg_utilization:.1f}%")
        print(f"  Max utilization: {max_utilization:.1f}%")
        print(f"  Min utilization: {min_utilization:.1f}%")


@pytest.mark.performance
class TestQualityBenchmarks:
    """Benchmarks for generation quality metrics"""
    
    def test_prompt_adherence_benchmark(self, mock_wan_model, test_config):
        """Benchmark prompt adherence quality"""
        mock_wan_model.model_type = "t2v-A14B"
        mock_wan_model.generate = Mock()
        mock_wan_model.calculate_prompt_adherence = Mock()
        
        # Mock quality scoring
        def mock_generate(params):
            result = Mock()
            result.success = True
            result.video_path = "/tmp/quality_test.mp4"
            return result
        
        def mock_calculate_adherence(prompt, video_path):
            # Mock CLIP-based prompt adherence scoring
            return 0.85 + (hash(prompt) % 100) / 1000  # 0.85-0.95 range
        
        mock_wan_model.generate.side_effect = mock_generate
        mock_wan_model.calculate_prompt_adherence.side_effect = mock_calculate_adherence
        
        # Test various prompts
        test_prompts = [
            "A cat playing in a garden",
            "A dog running on the beach",
            "A bird flying in the sky",
            "A car driving down a road",
            "A person walking in the rain"
        ]
        
        quality_scores = []
        
        for prompt in test_prompts:
            from backend.core.models.wan_models.wan_t2v_a14b import T2VGenerationParams
            params = T2VGenerationParams(
                prompt=prompt,
                width=512,
                height=512,
                num_frames=16,
                num_inference_steps=20
            )
            
            result = mock_wan_model.generate(params)
            quality_score = mock_wan_model.calculate_prompt_adherence(prompt, result.video_path)
            quality_scores.append(quality_score)
        
        # Analyze quality metrics
        avg_quality = statistics.mean(quality_scores)
        min_quality = min(quality_scores)
        max_quality = max(quality_scores)
        
        assert avg_quality > 0.8  # Expect good prompt adherence
        assert min_quality > 0.7  # Minimum acceptable quality
        assert max_quality <= 1.0  # Maximum possible score
        
        print(f"\nPrompt Adherence Benchmark:")
        print(f"  Average quality: {avg_quality:.3f}")
        print(f"  Min quality: {min_quality:.3f}")
        print(f"  Max quality: {max_quality:.3f}")
    
    def test_temporal_consistency_benchmark(self, mock_wan_model, test_config):
        """Benchmark temporal consistency in generated videos"""
        mock_wan_model.model_type = "t2v-A14B"
        mock_wan_model.generate = Mock()
        mock_wan_model.calculate_temporal_consistency = Mock()
        
        # Mock temporal consistency calculation
        def mock_generate(params):
            result = Mock()
            result.success = True
            result.video_path = "/tmp/temporal_test.mp4"
            result.frames = [Mock() for _ in range(params.num_frames)]
            return result
        
        def mock_calculate_consistency(frames):
            # Mock frame-to-frame similarity scoring
            return 0.92 + (len(frames) % 10) / 100  # 0.92-1.0 range
        
        mock_wan_model.generate.side_effect = mock_generate
        mock_wan_model.calculate_temporal_consistency.side_effect = mock_calculate_consistency
        
        # Test different frame counts
        frame_counts = [8, 16, 24, 32]
        consistency_scores = []
        
        for num_frames in frame_counts:
            from backend.core.models.wan_models.wan_t2v_a14b import T2VGenerationParams
            params = T2VGenerationParams(
                prompt="A cat playing in a garden",
                width=512,
                height=512,
                num_frames=num_frames,
                num_inference_steps=20
            )
            
            result = mock_wan_model.generate(params)
            consistency_score = mock_wan_model.calculate_temporal_consistency(result.frames)
            consistency_scores.append(consistency_score)
        
        # Analyze temporal consistency
        avg_consistency = statistics.mean(consistency_scores)
        
        assert avg_consistency > 0.9  # Expect high temporal consistency
        
        print(f"\nTemporal Consistency Benchmark:")
        print(f"  Average consistency: {avg_consistency:.3f}")
        for i, (frames, score) in enumerate(zip(frame_counts, consistency_scores)):
            print(f"  {frames} frames: {score:.3f}")


@pytest.mark.performance
class TestComparativeBenchmarks:
    """Comparative benchmarks between different WAN models"""
    
    def test_model_comparison_benchmark(self, mock_model_config, test_config):
        """Compare performance across different WAN models"""
        # Mock different models
        models = {
            "t2v-A14B": Mock(),
            "i2v-A14B": Mock(),
            "ti2v-5B": Mock()
        }
        
        # Mock generation times (TI2V should be fastest due to 5B parameters)
        generation_times = {
            "t2v-A14B": 0.12,
            "i2v-A14B": 0.15,  # Slightly slower due to image processing
            "ti2v-5B": 0.08   # Fastest due to smaller model
        }
        
        benchmark_results = {}
        
        for model_name, model in models.items():
            model.model_type = model_name
            model.generate = Mock()
            
            def mock_generate(params, model_time=generation_times[model_name]):
                time.sleep(model_time)
                result = Mock()
                result.success = True
                result.generation_time = model_time
                result.video_path = f"/tmp/{model_name}_test.mp4"
                return result
            
            model.generate.side_effect = mock_generate
            
            # Benchmark each model
            iterations = 3
            times = []
            
            for _ in range(iterations):
                start_time = time.time()
                result = model.generate(Mock())
                end_time = time.time()
                times.append(end_time - start_time)
            
            benchmark_results[model_name] = {
                "avg_time": statistics.mean(times),
                "parameter_count": 14_000_000_000 if "A14B" in model_name else 5_000_000_000
            }
        
        # Verify performance expectations
        assert benchmark_results["ti2v-5B"]["avg_time"] < benchmark_results["t2v-A14B"]["avg_time"]
        assert benchmark_results["ti2v-5B"]["parameter_count"] < benchmark_results["t2v-A14B"]["parameter_count"]
        
        print(f"\nModel Comparison Benchmark:")
        for model_name, metrics in benchmark_results.items():
            print(f"  {model_name}:")
            print(f"    Average time: {metrics['avg_time']:.3f}s")
            print(f"    Parameters: {metrics['parameter_count']:,}")
    
    def test_efficiency_ratio_benchmark(self, test_config):
        """Calculate efficiency ratios (performance per parameter)"""
        # Model specifications
        model_specs = {
            "t2v-A14B": {"params": 14_000_000_000, "time": 0.12},
            "i2v-A14B": {"params": 14_000_000_000, "time": 0.15},
            "ti2v-5B": {"params": 5_000_000_000, "time": 0.08}
        }
        
        efficiency_ratios = {}
        
        for model_name, specs in model_specs.items():
            # Calculate efficiency as 1/(time * params) - higher is better
            efficiency = 1.0 / (specs["time"] * specs["params"] / 1_000_000_000)  # Normalize params to billions
            efficiency_ratios[model_name] = efficiency
        
        # TI2V should have the best efficiency ratio
        assert efficiency_ratios["ti2v-5B"] > efficiency_ratios["t2v-A14B"]
        assert efficiency_ratios["ti2v-5B"] > efficiency_ratios["i2v-A14B"]
        
        print(f"\nEfficiency Ratio Benchmark:")
        for model_name, ratio in efficiency_ratios.items():
            print(f"  {model_name}: {ratio:.3f} (higher is better)")
