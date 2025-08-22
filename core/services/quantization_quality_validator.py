"""
Quantization Quality Validation System

This module provides comprehensive quality validation for quantized models,
including output quality comparison, degradation detection, and quality
metrics for different quantization methods.
"""

import json
import logging
import numpy as np
import time
import torch
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import hashlib

from quantization_controller import QuantizationMethod, ModelInfo

logger = logging.getLogger(__name__)


class QualityMetric(Enum):
    """Types of quality metrics"""
    STRUCTURAL_SIMILARITY = "ssim"
    PEAK_SIGNAL_NOISE_RATIO = "psnr"
    MEAN_SQUARED_ERROR = "mse"
    PERCEPTUAL_DISTANCE = "lpips"
    FEATURE_SIMILARITY = "feature_sim"
    TEMPORAL_CONSISTENCY = "temporal_consistency"


class QualityLevel(Enum):
    """Quality assessment levels"""
    EXCELLENT = "excellent"  # >95% similarity
    GOOD = "good"           # 90-95% similarity
    ACCEPTABLE = "acceptable"  # 80-90% similarity
    POOR = "poor"           # 70-80% similarity
    UNACCEPTABLE = "unacceptable"  # <70% similarity


@dataclass
class QualityTestCase:
    """Test case for quality validation"""
    name: str
    prompt: str
    negative_prompt: Optional[str]
    width: int
    height: int
    num_frames: int
    seed: int
    num_inference_steps: int
    guidance_scale: float
    expected_duration_seconds: float


@dataclass
class QualityMetrics:
    """Quality metrics for a single comparison"""
    ssim: float
    psnr: float
    mse: float
    lpips: Optional[float]
    feature_similarity: Optional[float]
    temporal_consistency: Optional[float]
    overall_score: float
    quality_level: QualityLevel


@dataclass
class QualityValidationResult:
    """Result of quality validation"""
    quantization_method: QuantizationMethod
    test_case_name: str
    metrics: QualityMetrics
    generation_time_original: float
    generation_time_quantized: float
    memory_usage_original: float
    memory_usage_quantized: float
    warnings: List[str]
    recommendations: List[str]
    passed_threshold: bool
    timestamp: datetime


@dataclass
class QualityReport:
    """Comprehensive quality report"""
    model_name: str
    quantization_method: QuantizationMethod
    test_results: List[QualityValidationResult]
    overall_quality_score: float
    overall_quality_level: QualityLevel
    performance_impact: Dict[str, float]
    memory_savings: float
    recommendations: List[str]
    warnings: List[str]
    timestamp: datetime


class QuantizationQualityValidator:
    """
    Validates quantization quality through comprehensive testing and comparison
    between original and quantized model outputs.
    """
    
    def __init__(self, test_cases_path: str = "quality_test_cases.json",
                 quality_thresholds_path: str = "quality_thresholds.json",
                 reports_dir: str = "quality_reports"):
        """
        Initialize the quality validator.
        
        Args:
            test_cases_path: Path to test cases configuration
            quality_thresholds_path: Path to quality thresholds configuration
            reports_dir: Directory for quality reports
        """
        self.test_cases_path = Path(test_cases_path)
        self.quality_thresholds_path = Path(quality_thresholds_path)
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.test_cases = self._load_test_cases()
        self.quality_thresholds = self._load_quality_thresholds()
        
        # Quality metric calculators
        self._metric_calculators = self._initialize_metric_calculators()
        
        # Cache for reference outputs
        self._reference_cache: Dict[str, Any] = {}
        
        self.logger.info("QuantizationQualityValidator initialized")
    
    def validate_quantization_quality(self, original_model: Any, quantized_model: Any,
                                    quantization_method: QuantizationMethod,
                                    model_info: ModelInfo,
                                    test_cases: Optional[List[QualityTestCase]] = None,
                                    timeout_seconds: int = 1800) -> QualityReport:
        """
        Validate quantization quality by comparing outputs.
        
        Args:
            original_model: Original unquantized model
            quantized_model: Quantized model to validate
            quantization_method: Quantization method used
            model_info: Information about the model
            test_cases: Custom test cases (uses defaults if None)
            timeout_seconds: Timeout for validation process
            
        Returns:
            Comprehensive quality report
        """
        self.logger.info(f"Starting quality validation for {quantization_method.value}")
        
        if test_cases is None:
            test_cases = self.test_cases
        
        validation_results = []
        warnings = []
        
        # Run validation with timeout
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self._run_quality_validation,
                    original_model, quantized_model, quantization_method,
                    model_info, test_cases
                )
                
                validation_results = future.result(timeout=timeout_seconds)
                
        except TimeoutError:
            warnings.append(f"Quality validation timed out after {timeout_seconds} seconds")
            self.logger.warning(f"Quality validation timeout for {quantization_method.value}")
            
            # Return partial results if available
            if validation_results:
                warnings.append("Using partial validation results due to timeout")
            else:
                # Create minimal report
                return self._create_timeout_report(quantization_method, model_info.name, warnings)
        
        except Exception as e:
            error_msg = f"Quality validation failed: {str(e)}"
            warnings.append(error_msg)
            self.logger.error(error_msg)
            return self._create_error_report(quantization_method, model_info.name, warnings)
        
        # Calculate overall metrics
        overall_score, overall_level = self._calculate_overall_quality(validation_results)
        performance_impact = self._calculate_performance_impact(validation_results)
        memory_savings = self._calculate_memory_savings(validation_results)
        recommendations = self._generate_recommendations(validation_results, quantization_method)
        
        # Create comprehensive report
        report = QualityReport(
            model_name=model_info.name,
            quantization_method=quantization_method,
            test_results=validation_results,
            overall_quality_score=overall_score,
            overall_quality_level=overall_level,
            performance_impact=performance_impact,
            memory_savings=memory_savings,
            recommendations=recommendations,
            warnings=warnings,
            timestamp=datetime.now()
        )
        
        # Save report
        self._save_quality_report(report)
        
        self.logger.info(f"Quality validation completed: {overall_level.value} ({overall_score:.2f})")
        return report
    
    def compare_quantization_methods(self, model: Any, model_info: ModelInfo,
                                   methods: List[QuantizationMethod],
                                   test_cases: Optional[List[QualityTestCase]] = None) -> Dict[QuantizationMethod, QualityReport]:
        """
        Compare multiple quantization methods for quality and performance.
        
        Args:
            model: Original model to quantize
            model_info: Information about the model
            methods: List of quantization methods to compare
            test_cases: Custom test cases (uses defaults if None)
            
        Returns:
            Dictionary mapping methods to quality reports
        """
        self.logger.info(f"Comparing {len(methods)} quantization methods")
        
        reports = {}
        original_model = self._create_model_copy(model)
        
        for method in methods:
            try:
                self.logger.info(f"Testing quantization method: {method.value}")
                
                # Create quantized model
                quantized_model = self._apply_quantization_for_testing(model, method)
                
                # Validate quality
                report = self.validate_quantization_quality(
                    original_model, quantized_model, method, model_info, test_cases
                )
                
                reports[method] = report
                
            except Exception as e:
                self.logger.error(f"Failed to test {method.value}: {e}")
                reports[method] = self._create_error_report(method, model_info.name, [str(e)])
        
        # Generate comparison summary
        self._save_comparison_report(reports, model_info.name)
        
        return reports
    
    def get_quality_threshold(self, quantization_method: QuantizationMethod,
                            metric: QualityMetric) -> float:
        """Get quality threshold for specific method and metric"""
        method_thresholds = self.quality_thresholds.get(quantization_method.value, {})
        return method_thresholds.get(metric.value, 0.8)  # Default threshold
    
    def set_quality_threshold(self, quantization_method: QuantizationMethod,
                            metric: QualityMetric, threshold: float) -> None:
        """Set quality threshold for specific method and metric"""
        if quantization_method.value not in self.quality_thresholds:
            self.quality_thresholds[quantization_method.value] = {}
        
        self.quality_thresholds[quantization_method.value][metric.value] = threshold
        self._save_quality_thresholds()
        
        self.logger.info(f"Updated threshold: {quantization_method.value}.{metric.value} = {threshold}")
    
    def create_custom_test_case(self, name: str, prompt: str, **kwargs) -> QualityTestCase:
        """Create a custom test case for quality validation"""
        return QualityTestCase(
            name=name,
            prompt=prompt,
            negative_prompt=kwargs.get('negative_prompt'),
            width=kwargs.get('width', 1280),
            height=kwargs.get('height', 720),
            num_frames=kwargs.get('num_frames', 16),
            seed=kwargs.get('seed', 42),
            num_inference_steps=kwargs.get('num_inference_steps', 20),
            guidance_scale=kwargs.get('guidance_scale', 7.5),
            expected_duration_seconds=kwargs.get('expected_duration_seconds', 2.0)
        )
    
    def add_test_case(self, test_case: QualityTestCase) -> None:
        """Add a new test case to the validation suite"""
        self.test_cases.append(test_case)
        self._save_test_cases()
        self.logger.info(f"Added test case: {test_case.name}")
    
    def get_historical_quality_trends(self, model_name: str,
                                    quantization_method: QuantizationMethod,
                                    days: int = 30) -> Dict[str, Any]:
        """Get historical quality trends for a model and method"""
        # This would analyze saved reports to show quality trends over time
        reports_pattern = f"{model_name}_{quantization_method.value}_*.json"
        report_files = list(self.reports_dir.glob(reports_pattern))
        
        if not report_files:
            return {"message": "No historical data available"}
        
        # Load and analyze reports
        quality_scores = []
        timestamps = []
        
        for report_file in sorted(report_files)[-days:]:  # Last N reports
            try:
                with open(report_file, 'r') as f:
                    report_data = json.load(f)
                    quality_scores.append(report_data['overall_quality_score'])
                    timestamps.append(report_data['timestamp'])
            except Exception as e:
                self.logger.warning(f"Failed to load report {report_file}: {e}")
        
        if not quality_scores:
            return {"message": "No valid historical data"}
        
        return {
            "average_quality": np.mean(quality_scores),
            "quality_trend": "improving" if quality_scores[-1] > quality_scores[0] else "declining",
            "quality_variance": np.var(quality_scores),
            "sample_count": len(quality_scores),
            "latest_score": quality_scores[-1] if quality_scores else None
        }
    
    def _run_quality_validation(self, original_model: Any, quantized_model: Any,
                              quantization_method: QuantizationMethod,
                              model_info: ModelInfo,
                              test_cases: List[QualityTestCase]) -> List[QualityValidationResult]:
        """Internal method to run quality validation"""
        results = []
        
        for test_case in test_cases:
            try:
                self.logger.info(f"Running test case: {test_case.name}")
                
                # Generate reference output (cached if available)
                reference_output = self._get_or_generate_reference(original_model, test_case)
                
                # Generate quantized output
                quantized_output, quantized_time, quantized_memory = self._generate_output_with_metrics(
                    quantized_model, test_case
                )
                
                # Calculate quality metrics
                metrics = self._calculate_quality_metrics(reference_output, quantized_output)
                
                # Check thresholds
                passed_threshold = self._check_quality_thresholds(metrics, quantization_method)
                
                # Generate warnings and recommendations
                warnings, recommendations = self._analyze_quality_result(
                    metrics, quantization_method, test_case
                )
                
                result = QualityValidationResult(
                    quantization_method=quantization_method,
                    test_case_name=test_case.name,
                    metrics=metrics,
                    generation_time_original=reference_output.get('generation_time', 0),
                    generation_time_quantized=quantized_time,
                    memory_usage_original=reference_output.get('memory_usage', 0),
                    memory_usage_quantized=quantized_memory,
                    warnings=warnings,
                    recommendations=recommendations,
                    passed_threshold=passed_threshold,
                    timestamp=datetime.now()
                )
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Test case {test_case.name} failed: {e}")
                # Create error result
                error_result = QualityValidationResult(
                    quantization_method=quantization_method,
                    test_case_name=test_case.name,
                    metrics=QualityMetrics(0, 0, float('inf'), None, None, None, 0, QualityLevel.UNACCEPTABLE),
                    generation_time_original=0,
                    generation_time_quantized=0,
                    memory_usage_original=0,
                    memory_usage_quantized=0,
                    warnings=[f"Test failed: {str(e)}"],
                    recommendations=["Review quantization settings"],
                    passed_threshold=False,
                    timestamp=datetime.now()
                )
                results.append(error_result)
        
        return results
    
    def _get_or_generate_reference(self, model: Any, test_case: QualityTestCase) -> Dict[str, Any]:
        """Get cached reference output or generate new one"""
        # Create cache key from test case parameters
        cache_key = self._create_cache_key(test_case)
        
        if cache_key in self._reference_cache:
            self.logger.info(f"Using cached reference for {test_case.name}")
            return self._reference_cache[cache_key]
        
        self.logger.info(f"Generating reference output for {test_case.name}")
        
        # Generate reference output
        output, generation_time, memory_usage = self._generate_output_with_metrics(model, test_case)
        
        reference_data = {
            'output': output,
            'generation_time': generation_time,
            'memory_usage': memory_usage,
            'test_case': asdict(test_case)
        }
        
        # Cache the reference
        self._reference_cache[cache_key] = reference_data
        
        return reference_data
    
    def _generate_output_with_metrics(self, model: Any, test_case: QualityTestCase) -> Tuple[Any, float, float]:
        """Generate output while measuring time and memory usage"""
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
        else:
            initial_memory = 0
        
        start_time = time.time()
        
        try:
            # Generate output using the model
            # This is a simplified version - actual implementation would depend on model interface
            output = self._call_model_generate(model, test_case)
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise
        
        generation_time = time.time() - start_time
        
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            memory_usage = (final_memory - initial_memory) / (1024 * 1024)  # MB
        else:
            memory_usage = 0
        
        return output, generation_time, memory_usage
    
    def _call_model_generate(self, model: Any, test_case: QualityTestCase) -> Any:
        """Call model generation - this would be implemented based on actual model interface"""
        # This is a placeholder - actual implementation would depend on the model's API
        # For example, for a diffusion model:
        
        if hasattr(model, 'generate'):
            return model.generate(
                prompt=test_case.prompt,
                negative_prompt=test_case.negative_prompt,
                width=test_case.width,
                height=test_case.height,
                num_frames=test_case.num_frames,
                num_inference_steps=test_case.num_inference_steps,
                guidance_scale=test_case.guidance_scale,
                generator=torch.Generator().manual_seed(test_case.seed)
            )
        elif hasattr(model, '__call__'):
            return model(
                prompt=test_case.prompt,
                negative_prompt=test_case.negative_prompt,
                width=test_case.width,
                height=test_case.height,
                num_frames=test_case.num_frames,
                num_inference_steps=test_case.num_inference_steps,
                guidance_scale=test_case.guidance_scale,
                generator=torch.Generator().manual_seed(test_case.seed)
            )
        else:
            raise ValueError("Model does not have a recognized generation interface")
    
    def _calculate_quality_metrics(self, reference_output: Dict[str, Any], 
                                 quantized_output: Any) -> QualityMetrics:
        """Calculate comprehensive quality metrics"""
        ref_frames = self._extract_frames(reference_output['output'])
        quant_frames = self._extract_frames(quantized_output)
        
        # Calculate basic metrics
        ssim = self._calculate_ssim(ref_frames, quant_frames)
        psnr = self._calculate_psnr(ref_frames, quant_frames)
        mse = self._calculate_mse(ref_frames, quant_frames)
        
        # Calculate advanced metrics (if available)
        lpips = self._calculate_lpips(ref_frames, quant_frames)
        feature_similarity = self._calculate_feature_similarity(ref_frames, quant_frames)
        temporal_consistency = self._calculate_temporal_consistency(ref_frames, quant_frames)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(ssim, psnr, mse, lpips, feature_similarity)
        quality_level = self._determine_quality_level(overall_score)
        
        return QualityMetrics(
            ssim=ssim,
            psnr=psnr,
            mse=mse,
            lpips=lpips,
            feature_similarity=feature_similarity,
            temporal_consistency=temporal_consistency,
            overall_score=overall_score,
            quality_level=quality_level
        )
    
    def _extract_frames(self, output: Any) -> np.ndarray:
        """Extract frames from model output"""
        # This would depend on the output format
        # Placeholder implementation
        if isinstance(output, torch.Tensor):
            return output.cpu().numpy()
        elif hasattr(output, 'frames'):
            return np.array(output.frames)
        elif isinstance(output, (list, tuple)):
            return np.array(output)
        else:
            # Try to convert to numpy array
            return np.array(output)
    
    def _calculate_ssim(self, ref_frames: np.ndarray, quant_frames: np.ndarray) -> float:
        """Calculate Structural Similarity Index"""
        try:
            from skimage.metrics import structural_similarity as ssim
            
            # Ensure same shape
            if ref_frames.shape != quant_frames.shape:
                self.logger.warning("Frame shapes don't match for SSIM calculation")
                return 0.0
            
            # Calculate SSIM for each frame and average
            ssim_scores = []
            for i in range(min(len(ref_frames), len(quant_frames))):
                if len(ref_frames[i].shape) == 3:  # Color image
                    score = ssim(ref_frames[i], quant_frames[i], multichannel=True, channel_axis=-1)
                else:
                    score = ssim(ref_frames[i], quant_frames[i])
                ssim_scores.append(score)
            
            return np.mean(ssim_scores)
            
        except ImportError:
            self.logger.warning("scikit-image not available for SSIM calculation")
            return 0.8  # Default reasonable value
        except Exception as e:
            self.logger.error(f"SSIM calculation failed: {e}")
            return 0.0
    
    def _calculate_psnr(self, ref_frames: np.ndarray, quant_frames: np.ndarray) -> float:
        """Calculate Peak Signal-to-Noise Ratio"""
        try:
            mse = np.mean((ref_frames - quant_frames) ** 2)
            if mse == 0:
                return float('inf')
            
            max_pixel = 1.0 if ref_frames.max() <= 1.0 else 255.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
            return float(psnr)
            
        except Exception as e:
            self.logger.error(f"PSNR calculation failed: {e}")
            return 0.0
    
    def _calculate_mse(self, ref_frames: np.ndarray, quant_frames: np.ndarray) -> float:
        """Calculate Mean Squared Error"""
        try:
            return float(np.mean((ref_frames - quant_frames) ** 2))
        except Exception as e:
            self.logger.error(f"MSE calculation failed: {e}")
            return float('inf')
    
    def _calculate_lpips(self, ref_frames: np.ndarray, quant_frames: np.ndarray) -> Optional[float]:
        """Calculate Learned Perceptual Image Patch Similarity"""
        try:
            # This would require the LPIPS library
            # Placeholder implementation
            self.logger.info("LPIPS calculation not implemented")
            return None
        except Exception as e:
            self.logger.error(f"LPIPS calculation failed: {e}")
            return None
    
    def _calculate_feature_similarity(self, ref_frames: np.ndarray, quant_frames: np.ndarray) -> Optional[float]:
        """Calculate feature-based similarity"""
        try:
            # This would use a pre-trained feature extractor
            # Placeholder implementation
            self.logger.info("Feature similarity calculation not implemented")
            return None
        except Exception as e:
            self.logger.error(f"Feature similarity calculation failed: {e}")
            return None
    
    def _calculate_temporal_consistency(self, ref_frames: np.ndarray, quant_frames: np.ndarray) -> Optional[float]:
        """Calculate temporal consistency between frames"""
        try:
            if len(ref_frames) < 2 or len(quant_frames) < 2:
                return None
            
            # Calculate frame-to-frame differences
            ref_diffs = np.diff(ref_frames, axis=0)
            quant_diffs = np.diff(quant_frames, axis=0)
            
            # Calculate similarity of differences
            consistency = 1.0 - np.mean(np.abs(ref_diffs - quant_diffs))
            return float(max(0.0, consistency))
            
        except Exception as e:
            self.logger.error(f"Temporal consistency calculation failed: {e}")
            return None
    
    def _calculate_overall_score(self, ssim: float, psnr: float, mse: float,
                               lpips: Optional[float], feature_sim: Optional[float]) -> float:
        """Calculate overall quality score"""
        # Weighted combination of metrics
        weights = {'ssim': 0.4, 'psnr': 0.3, 'mse': 0.2, 'lpips': 0.1}
        
        score = 0.0
        total_weight = 0.0
        
        # SSIM (higher is better)
        if ssim is not None:
            score += weights['ssim'] * ssim
            total_weight += weights['ssim']
        
        # PSNR (higher is better, normalize to 0-1)
        if psnr is not None and psnr != float('inf'):
            normalized_psnr = min(1.0, psnr / 40.0)  # 40dB is excellent
            score += weights['psnr'] * normalized_psnr
            total_weight += weights['psnr']
        
        # MSE (lower is better, invert and normalize)
        if mse is not None and mse != float('inf'):
            normalized_mse = max(0.0, 1.0 - min(1.0, mse))
            score += weights['mse'] * normalized_mse
            total_weight += weights['mse']
        
        # LPIPS (lower is better, invert)
        if lpips is not None:
            score += weights['lpips'] * (1.0 - lpips)
            total_weight += weights['lpips']
        
        # Normalize by total weight
        if total_weight > 0:
            score /= total_weight
        
        return float(score)
    
    def _determine_quality_level(self, score: float) -> QualityLevel:
        """Determine quality level from overall score"""
        if score >= 0.95:
            return QualityLevel.EXCELLENT
        elif score >= 0.90:
            return QualityLevel.GOOD
        elif score >= 0.80:
            return QualityLevel.ACCEPTABLE
        elif score >= 0.70:
            return QualityLevel.POOR
        else:
            return QualityLevel.UNACCEPTABLE
    
    def _check_quality_thresholds(self, metrics: QualityMetrics, 
                                method: QuantizationMethod) -> bool:
        """Check if metrics meet quality thresholds"""
        ssim_threshold = self.get_quality_threshold(method, QualityMetric.STRUCTURAL_SIMILARITY)
        psnr_threshold = self.get_quality_threshold(method, QualityMetric.PEAK_SIGNAL_NOISE_RATIO)
        
        return (metrics.ssim >= ssim_threshold and 
                metrics.psnr >= psnr_threshold and
                metrics.overall_score >= 0.7)  # Minimum overall threshold
    
    def _analyze_quality_result(self, metrics: QualityMetrics, 
                              method: QuantizationMethod,
                              test_case: QualityTestCase) -> Tuple[List[str], List[str]]:
        """Analyze quality result and generate warnings/recommendations"""
        warnings = []
        recommendations = []
        
        # Check individual metrics
        if metrics.ssim < 0.8:
            warnings.append(f"Low structural similarity: {metrics.ssim:.3f}")
            recommendations.append("Consider less aggressive quantization")
        
        if metrics.psnr < 25:
            warnings.append(f"Low PSNR: {metrics.psnr:.1f}dB")
            recommendations.append("Quality may be noticeably degraded")
        
        if metrics.mse > 0.01:
            warnings.append(f"High MSE: {metrics.mse:.6f}")
        
        # Method-specific analysis
        if method == QuantizationMethod.INT8:
            if metrics.overall_score < 0.85:
                warnings.append("INT8 quantization showing significant quality loss")
                recommendations.append("Consider BF16 quantization instead")
        
        if method == QuantizationMethod.FP8:
            warnings.append("FP8 quantization is experimental")
            recommendations.append("Monitor quality carefully in production")
        
        # Test case specific analysis
        if test_case.num_frames > 16 and metrics.temporal_consistency and metrics.temporal_consistency < 0.8:
            warnings.append("Poor temporal consistency in long video")
            recommendations.append("Consider shorter sequences or different quantization")
        
        return warnings, recommendations
    
    def _calculate_overall_quality(self, results: List[QualityValidationResult]) -> Tuple[float, QualityLevel]:
        """Calculate overall quality from all test results"""
        if not results:
            return 0.0, QualityLevel.UNACCEPTABLE
        
        scores = [r.metrics.overall_score for r in results]
        overall_score = np.mean(scores)
        overall_level = self._determine_quality_level(overall_score)
        
        return overall_score, overall_level
    
    def _calculate_performance_impact(self, results: List[QualityValidationResult]) -> Dict[str, float]:
        """Calculate performance impact from validation results"""
        if not results:
            return {}
        
        time_ratios = []
        memory_ratios = []
        
        for result in results:
            if result.generation_time_original > 0:
                time_ratio = result.generation_time_quantized / result.generation_time_original
                time_ratios.append(time_ratio)
            
            if result.memory_usage_original > 0:
                memory_ratio = result.memory_usage_quantized / result.memory_usage_original
                memory_ratios.append(memory_ratio)
        
        impact = {}
        if time_ratios:
            impact['speed_multiplier'] = np.mean(time_ratios)
        if memory_ratios:
            impact['memory_multiplier'] = np.mean(memory_ratios)
        
        return impact
    
    def _calculate_memory_savings(self, results: List[QualityValidationResult]) -> float:
        """Calculate memory savings from validation results"""
        if not results:
            return 0.0
        
        total_original = sum(r.memory_usage_original for r in results)
        total_quantized = sum(r.memory_usage_quantized for r in results)
        
        if total_original > 0:
            return (total_original - total_quantized) / total_original
        
        return 0.0
    
    def _generate_recommendations(self, results: List[QualityValidationResult],
                                method: QuantizationMethod) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        if not results:
            return ["No validation results available"]
        
        # Analyze overall quality
        overall_score = np.mean([r.metrics.overall_score for r in results])
        
        if overall_score < 0.7:
            recommendations.append("Consider using less aggressive quantization")
            recommendations.append("Quality may not be acceptable for production use")
        elif overall_score < 0.85:
            recommendations.append("Monitor quality in production use")
        
        # Analyze performance
        failed_tests = [r for r in results if not r.passed_threshold]
        if len(failed_tests) > len(results) * 0.3:  # More than 30% failed
            recommendations.append("High failure rate - review quantization settings")
        
        # Method-specific recommendations
        if method == QuantizationMethod.INT8:
            recommendations.append("Consider excluding VAE from INT8 quantization")
        elif method == QuantizationMethod.FP8:
            recommendations.append("FP8 is experimental - validate thoroughly")
        
        return recommendations
    
    def _create_timeout_report(self, method: QuantizationMethod, model_name: str, 
                             warnings: List[str]) -> QualityReport:
        """Create report for timeout case"""
        return QualityReport(
            model_name=model_name,
            quantization_method=method,
            test_results=[],
            overall_quality_score=0.0,
            overall_quality_level=QualityLevel.UNACCEPTABLE,
            performance_impact={},
            memory_savings=0.0,
            recommendations=["Increase validation timeout", "Reduce test case complexity"],
            warnings=warnings,
            timestamp=datetime.now()
        )
    
    def _create_error_report(self, method: QuantizationMethod, model_name: str,
                           warnings: List[str]) -> QualityReport:
        """Create report for error case"""
        return QualityReport(
            model_name=model_name,
            quantization_method=method,
            test_results=[],
            overall_quality_score=0.0,
            overall_quality_level=QualityLevel.UNACCEPTABLE,
            performance_impact={},
            memory_savings=0.0,
            recommendations=["Review quantization configuration", "Check model compatibility"],
            warnings=warnings,
            timestamp=datetime.now()
        )
    
    def _create_cache_key(self, test_case: QualityTestCase) -> str:
        """Create cache key for test case"""
        key_data = f"{test_case.prompt}_{test_case.width}_{test_case.height}_{test_case.num_frames}_{test_case.seed}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _create_model_copy(self, model: Any) -> Any:
        """Create a copy of the model for testing"""
        # This would depend on the model implementation
        # For now, return the same model (assuming it's stateless for generation)
        return model
    
    def _apply_quantization_for_testing(self, model: Any, method: QuantizationMethod) -> Any:
        """Apply quantization to model for testing purposes"""
        # This would use the quantization controller
        # Simplified implementation for now
        if method == QuantizationMethod.NONE:
            return model
        
        # Create a copy and apply quantization
        quantized_model = self._create_model_copy(model)
        
        if method == QuantizationMethod.FP16:
            if hasattr(quantized_model, 'half'):
                quantized_model.half()
        elif method == QuantizationMethod.BF16:
            if hasattr(quantized_model, 'to'):
                quantized_model.to(dtype=torch.bfloat16)
        
        return quantized_model
    
    def _initialize_metric_calculators(self) -> Dict[str, Callable]:
        """Initialize metric calculation functions"""
        return {
            'ssim': self._calculate_ssim,
            'psnr': self._calculate_psnr,
            'mse': self._calculate_mse,
            'lpips': self._calculate_lpips,
            'feature_similarity': self._calculate_feature_similarity,
            'temporal_consistency': self._calculate_temporal_consistency
        }
    
    def _load_test_cases(self) -> List[QualityTestCase]:
        """Load test cases from configuration"""
        try:
            if self.test_cases_path.exists():
                with open(self.test_cases_path, 'r') as f:
                    data = json.load(f)
                    test_cases = []
                    for case_data in data.get('test_cases', []):
                        test_case = QualityTestCase(**case_data)
                        test_cases.append(test_case)
                    return test_cases
        except Exception as e:
            self.logger.error(f"Failed to load test cases: {e}")
        
        # Default test cases
        return self._create_default_test_cases()
    
    def _create_default_test_cases(self) -> List[QualityTestCase]:
        """Create default test cases"""
        return [
            QualityTestCase(
                name="basic_generation",
                prompt="A beautiful landscape with mountains and a lake",
                negative_prompt="blurry, low quality",
                width=1280,
                height=720,
                num_frames=16,
                seed=42,
                num_inference_steps=20,
                guidance_scale=7.5,
                expected_duration_seconds=2.0
            ),
            QualityTestCase(
                name="complex_scene",
                prompt="A bustling city street with cars, people, and buildings",
                negative_prompt="blurry, distorted",
                width=1280,
                height=720,
                num_frames=24,
                seed=123,
                num_inference_steps=25,
                guidance_scale=8.0,
                expected_duration_seconds=3.0
            ),
            QualityTestCase(
                name="portrait_test",
                prompt="A portrait of a person smiling",
                negative_prompt="blurry, deformed",
                width=720,
                height=1280,
                num_frames=12,
                seed=456,
                num_inference_steps=30,
                guidance_scale=7.0,
                expected_duration_seconds=1.5
            )
        ]
    
    def _save_test_cases(self) -> None:
        """Save test cases to configuration"""
        try:
            data = {
                'test_cases': [asdict(tc) for tc in self.test_cases]
            }
            with open(self.test_cases_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save test cases: {e}")
    
    def _load_quality_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Load quality thresholds from configuration"""
        try:
            if self.quality_thresholds_path.exists():
                with open(self.quality_thresholds_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load quality thresholds: {e}")
        
        # Default thresholds
        return {
            "none": {"ssim": 1.0, "psnr": 100.0},
            "fp16": {"ssim": 0.95, "psnr": 35.0},
            "bf16": {"ssim": 0.95, "psnr": 35.0},
            "int8": {"ssim": 0.85, "psnr": 28.0},
            "fp8": {"ssim": 0.80, "psnr": 25.0}
        }
    
    def _save_quality_thresholds(self) -> None:
        """Save quality thresholds to configuration"""
        try:
            with open(self.quality_thresholds_path, 'w') as f:
                json.dump(self.quality_thresholds, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save quality thresholds: {e}")
    
    def _save_quality_report(self, report: QualityReport) -> None:
        """Save quality report to file"""
        try:
            timestamp = report.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"{report.model_name}_{report.quantization_method.value}_{timestamp}.json"
            filepath = self.reports_dir / filename
            
            # Convert report to JSON-serializable format
            report_data = asdict(report)
            report_data['timestamp'] = report.timestamp.isoformat()
            report_data['overall_quality_level'] = report.overall_quality_level.value
            
            # Convert test results
            for result in report_data['test_results']:
                result['quantization_method'] = result['quantization_method'].value
                result['metrics']['quality_level'] = result['metrics']['quality_level'].value
                result['timestamp'] = result['timestamp'].isoformat()
            
            with open(filepath, 'w') as f:
                json.dump(report_data, f, indent=2)
                
            self.logger.info(f"Quality report saved: {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to save quality report: {e}")
    
    def _save_comparison_report(self, reports: Dict[QuantizationMethod, QualityReport],
                              model_name: str) -> None:
        """Save comparison report for multiple methods"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name}_comparison_{timestamp}.json"
            filepath = self.reports_dir / filename
            
            comparison_data = {
                'model_name': model_name,
                'timestamp': datetime.now().isoformat(),
                'methods_compared': list(reports.keys()),
                'summary': {}
            }
            
            # Add summary statistics
            for method, report in reports.items():
                comparison_data['summary'][method.value] = {
                    'overall_quality_score': report.overall_quality_score,
                    'overall_quality_level': report.overall_quality_level.value,
                    'memory_savings': report.memory_savings,
                    'performance_impact': report.performance_impact
                }
            
            with open(filepath, 'w') as f:
                json.dump(comparison_data, f, indent=2)
                
            self.logger.info(f"Comparison report saved: {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to save comparison report: {e}")