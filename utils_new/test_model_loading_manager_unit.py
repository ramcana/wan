#!/usr/bin/env python3
"""
Unit tests for ModelLoadingManager component
Tests model loading optimization, progress tracking, and error handling
"""

import unittest
import tempfile
import json
import time
import threading
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
from datetime import datetime, timedelta

from model_loading_manager import (
    ModelLoadingManager, ModelLoadingPhase, ModelLoadingError,
    ModelLoadingProgress, LoadingParameters, ModelLoadingResult
)


class TestModelLoadingManager(unittest.TestCase):
    """Test cases for ModelLoadingManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir) / "cache"
        
        self.manager = ModelLoadingManager(
            cache_dir=str(self.cache_dir),
            enable_logging=False  # Disable logging for tests
        )
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test ModelLoadingManager initialization"""
        self.assertIsInstance(self.manager, ModelLoadingManager)
        self.assertTrue(self.cache_dir.exists())
        self.assertEqual(len(self.manager._progress_callbacks), 0)
        self.assertIsInstance(self.manager._parameter_cache, dict)
        self.assertIsInstance(self.manager._error_solutions, dict)
        self.assertEqual(self.manager._current_progress.phase, ModelLoadingPhase.INITIALIZATION)
    
    def test_init_with_logging(self):
        """Test ModelLoadingManager initialization with logging enabled"""
        manager = ModelLoadingManager(enable_logging=True)
        
        # Should have logger configured
        self.assertIsNotNone(manager.logger)
    
    def test_add_progress_callback(self):
        """Test adding progress callback"""
        callback = MagicMock()
        
        self.manager.add_progress_callback(callback)
        
        self.assertIn(callback, self.manager._progress_callbacks)
    
    def test_remove_progress_callback(self):
        """Test removing progress callback"""
        callback = MagicMock()
        
        self.manager.add_progress_callback(callback)
        self.assertIn(callback, self.manager._progress_callbacks)
        
        self.manager.remove_progress_callback(callback)
        self.assertNotIn(callback, self.manager._progress_callbacks)
    
    def test_update_progress(self):
        """Test progress update and callback notification"""
        callback = MagicMock()
        self.manager.add_progress_callback(callback)
        
        self.manager._update_progress(
            ModelLoadingPhase.LOADING,
            50.0,
            "Loading model components",
            estimated_time_remaining=120.0,
            memory_usage_mb=8192.0
        )
        
        # Check progress was updated
        progress = self.manager._current_progress
        self.assertEqual(progress.phase, ModelLoadingPhase.LOADING)
        self.assertEqual(progress.progress_percent, 50.0)
        self.assertEqual(progress.current_step, "Loading model components")
        self.assertEqual(progress.estimated_time_remaining, 120.0)
        self.assertEqual(progress.memory_usage_mb, 8192.0)
        
        # Check callback was called
        callback.assert_called_once()
        callback_progress = callback.call_args[0][0]
        self.assertEqual(callback_progress.phase, ModelLoadingPhase.LOADING)
    
    def test_update_progress_callback_exception(self):
        """Test progress update with callback exception"""
        failing_callback = MagicMock(side_effect=Exception("Callback failed"))
        working_callback = MagicMock()
        
        self.manager.add_progress_callback(failing_callback)
        self.manager.add_progress_callback(working_callback)
        
        # Should not raise exception despite failing callback
        self.manager._update_progress(
            ModelLoadingPhase.LOADING,
            25.0,
            "Test step"
        )
        
        # Working callback should still be called
        working_callback.assert_called_once()
    
    def test_load_parameter_cache_empty(self):
        """Test loading parameter cache when file doesn't exist"""
        self.manager._load_parameter_cache()
        
        self.assertEqual(len(self.manager._parameter_cache), 0)
    
    def test_load_parameter_cache_with_data(self):
        """Test loading parameter cache with existing data"""
        cache_data = {
            "test_key": {
                "parameters": {"model_path": "test/model", "torch_dtype": "float16"},
                "loading_time": 120.5,
                "memory_usage_mb": 8192.0,
                "last_used": "2024-01-01T12:00:00",
                "use_count": 3
            }
        }
        
        cache_file = self.cache_dir / "parameter_cache.json"
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        
        self.manager._load_parameter_cache()
        
        self.assertEqual(len(self.manager._parameter_cache), 1)
        self.assertIn("test_key", self.manager._parameter_cache)
        self.assertEqual(self.manager._parameter_cache["test_key"]["loading_time"], 120.5)
    
    def test_save_parameter_cache(self):
        """Test saving parameter cache"""
        self.manager._parameter_cache = {
            "test_key": {
                "parameters": {"model_path": "test/model"},
                "loading_time": 60.0,
                "memory_usage_mb": 4096.0,
                "last_used": "2024-01-01T12:00:00",
                "use_count": 1
            }
        }
        
        self.manager._save_parameter_cache()
        
        cache_file = self.cache_dir / "parameter_cache.json"
        self.assertTrue(cache_file.exists())
        
        with open(cache_file, 'r') as f:
            saved_data = json.load(f)
        
        self.assertEqual(len(saved_data), 1)
        self.assertIn("test_key", saved_data)
    
    def test_get_cached_parameters(self):
        """Test getting cached parameters"""
        self.manager._parameter_cache = {
            "existing_key": {"loading_time": 120.0},
            "another_key": {"loading_time": 60.0}
        }
        
        # Existing key
        result = self.manager._get_cached_parameters("existing_key")
        self.assertEqual(result["loading_time"], 120.0)
        
        # Non-existing key
        result = self.manager._get_cached_parameters("non_existing_key")
        self.assertIsNone(result)
    
    def test_cache_parameters(self):
        """Test caching parameters"""
        parameters = LoadingParameters(
            model_path="test/model",
            torch_dtype="float16"
        )
        
        self.manager._cache_parameters("test_key", parameters, 90.0, 6144.0)
        
        self.assertIn("test_key", self.manager._parameter_cache)
        cached = self.manager._parameter_cache["test_key"]
        self.assertEqual(cached["loading_time"], 90.0)
        self.assertEqual(cached["memory_usage_mb"], 6144.0)
        self.assertEqual(cached["use_count"], 1)
        self.assertIn("last_used", cached)
    
    def test_cache_parameters_update_existing(self):
        """Test caching parameters updates existing entry"""
        # Add initial cache entry
        self.manager._parameter_cache = {
            "test_key": {"use_count": 2}
        }
        
        parameters = LoadingParameters(
            model_path="test/model",
            torch_dtype="float16"
        )
        
        self.manager._cache_parameters("test_key", parameters, 90.0, 6144.0)
        
        # Should increment use count
        cached = self.manager._parameter_cache["test_key"]
        self.assertEqual(cached["use_count"], 3)
    
    def test_estimate_loading_time_cached(self):
        """Test loading time estimation with cached data"""
        self.manager._parameter_cache = {
            "test_key": {"loading_time": 150.0}
        }
        
        parameters = LoadingParameters(model_path="test/model", torch_dtype="float16")
        
        with patch.object(parameters, 'get_cache_key', return_value="test_key"):
            estimated_time = self.manager._estimate_loading_time("test/model", parameters)
        
        self.assertEqual(estimated_time, 150.0)
    
    def test_estimate_loading_time_fallback(self):
        """Test loading time estimation fallback for different model sizes"""
        parameters = LoadingParameters(model_path="test/model", torch_dtype="float16")
        
        with patch.object(parameters, 'get_cache_key', return_value="unknown_key"):
            # Large model (5B)
            time_5b = self.manager._estimate_loading_time("test/model-5B", parameters)
            self.assertEqual(time_5b, 300.0)
            
            # Medium model (1B)
            time_1b = self.manager._estimate_loading_time("test/model-1B", parameters)
            self.assertEqual(time_1b, 120.0)
            
            # Default model
            time_default = self.manager._estimate_loading_time("test/model", parameters)
            self.assertEqual(time_default, 60.0)
    
    @patch('model_loading_manager.psutil')
    @patch('model_loading_manager.DEPENDENCIES_AVAILABLE', True)
    def test_get_memory_usage(self, mock_psutil):
        """Test memory usage measurement"""
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 8 * 1024 * 1024 * 1024  # 8GB
        mock_psutil.Process.return_value = mock_process
        
        memory_usage = self.manager._get_memory_usage()
        
        self.assertEqual(memory_usage, 8 * 1024)  # Should return MB
    
    @patch('model_loading_manager.DEPENDENCIES_AVAILABLE', False)
    def test_get_memory_usage_no_dependencies(self):
        """Test memory usage measurement without dependencies"""
        memory_usage = self.manager._get_memory_usage()
        
        self.assertEqual(memory_usage, 0.0)
    
    def test_validate_model_path_existing_file(self):
        """Test model path validation for existing file"""
        test_file = Path(self.temp_dir) / "test_model"
        test_file.touch()
        
        is_valid = self.manager._validate_model_path(str(test_file))
        
        self.assertTrue(is_valid)
    
    def test_validate_model_path_huggingface_repo(self):
        """Test model path validation for HuggingFace repository"""
        is_valid = self.manager._validate_model_path("stabilityai/stable-diffusion-2-1")
        
        self.assertTrue(is_valid)
    
    def test_validate_model_path_invalid(self):
        """Test model path validation for invalid path"""
        is_valid = self.manager._validate_model_path("/nonexistent/path")
        
        self.assertFalse(is_valid)
    
    def test_handle_loading_error_cuda_out_of_memory(self):
        """Test error handling for CUDA out of memory"""
        error = RuntimeError("CUDA out of memory")
        parameters = LoadingParameters(model_path="test/model", torch_dtype="float16")
        
        result = self.manager._handle_loading_error(error, parameters)
        
        self.assertIsInstance(result, ModelLoadingResult)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "CUDA_OUT_OF_MEMORY")
        self.assertIn("Reduce model precision", result.suggestions[0])
    
    def test_handle_loading_error_model_not_found(self):
        """Test error handling for model not found"""
        error = FileNotFoundError("Model not found")
        parameters = LoadingParameters(model_path="test/model", torch_dtype="float16")
        
        result = self.manager._handle_loading_error(error, parameters)
        
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "MODEL_NOT_FOUND")
        self.assertIn("Verify the model path", result.suggestions[0])
    
    def test_handle_loading_error_trust_remote_code(self):
        """Test error handling for trust_remote_code error"""
        error = ValueError("trust_remote_code is required")
        parameters = LoadingParameters(model_path="test/model", torch_dtype="float16")
        
        result = self.manager._handle_loading_error(error, parameters)
        
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "TRUST_REMOTE_CODE_ERROR")
        self.assertIn("Set trust_remote_code=True", result.suggestions[0])
    
    def test_handle_loading_error_network_error(self):
        """Test error handling for network error"""
        error = ConnectionError("Network connection failed")
        parameters = LoadingParameters(model_path="test/model", torch_dtype="float16")
        
        result = self.manager._handle_loading_error(error, parameters)
        
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "NETWORK_ERROR")
        self.assertIn("Check internet connection", result.suggestions[0])
    
    def test_handle_loading_error_unknown(self):
        """Test error handling for unknown error"""
        error = ValueError("Unknown error")
        parameters = LoadingParameters(model_path="test/model", torch_dtype="float16")
        
        result = self.manager._handle_loading_error(error, parameters)
        
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "UNKNOWN")
        self.assertIn("Check the error message", result.suggestions[0])
    
    @patch('model_loading_manager.DEPENDENCIES_AVAILABLE', False)
    def test_load_model_no_dependencies(self):
        """Test model loading without required dependencies"""
        result = self.manager.load_model("test/model")
        
        self.assertIsInstance(result, ModelLoadingResult)
        self.assertFalse(result.success)
        self.assertEqual(result.error_code, "MISSING_DEPENDENCIES")
        self.assertIn("Install required packages", result.suggestions[0])
    
    @patch('model_loading_manager.DEPENDENCIES_AVAILABLE', True)
    @patch('model_loading_manager.DiffusionPipeline')
    @patch('model_loading_manager.torch')
    def test_load_model_success(self, mock_torch, mock_pipeline_class):
        """Test successful model loading"""
        # Mock dependencies
        mock_torch.float16 = MagicMock()
        
        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        
        # Mock memory usage
        with patch.object(self.manager, '_get_memory_usage', side_effect=[1024.0, 2048.0]):
            with patch.object(self.manager, '_validate_model_path', return_value=True):
                result = self.manager.load_model(
                    "test/model",
                    torch_dtype="float16",
                    device_map="auto"
                )
        
        self.assertIsInstance(result, ModelLoadingResult)
        self.assertTrue(result.success)
        self.assertEqual(result.model, mock_pipeline)
        self.assertGreater(result.loading_time, 0)
        self.assertEqual(result.memory_usage_mb, 1024.0)  # Difference
        self.assertFalse(result.cache_hit)
        self.assertIsInstance(result.parameters_used, LoadingParameters)
    
    @patch('model_loading_manager.DEPENDENCIES_AVAILABLE', True)
    def test_load_model_invalid_path(self):
        """Test model loading with invalid path"""
        with patch.object(self.manager, '_validate_model_path', return_value=False):
            result = self.manager.load_model("invalid/path")
        
        self.assertFalse(result.success)
        self.assertIn("Model path not found", result.error_message)
    
    @patch('model_loading_manager.DEPENDENCIES_AVAILABLE', True)
    @patch('model_loading_manager.DiffusionPipeline')
    def test_load_model_with_cache_hit(self, mock_pipeline_class):
        """Test model loading with cache hit"""
        # Set up cache
        cache_key = "test_cache_key"
        self.manager._parameter_cache = {
            cache_key: {
                "parameters": {
                    "model_path": "test/model",
                    "torch_dtype": "float16",
                    "device_map": None,
                    "low_cpu_mem_usage": True,
                    "trust_remote_code": False,
                    "variant": None,
                    "use_safetensors": True,
                    "load_in_8bit": False,
                    "load_in_4bit": False,
                    "custom_pipeline": None
                },
                "loading_time": 60.0,
                "memory_usage_mb": 4096.0,
                "last_used": "2024-01-01T12:00:00",
                "use_count": 1
            }
        }
        
        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        
        with patch.object(self.manager, '_validate_model_path', return_value=True):
            with patch.object(self.manager, '_get_memory_usage', side_effect=[1024.0, 2048.0]):
                # Mock LoadingParameters.get_cache_key to return our test key
                with patch('model_loading_manager.LoadingParameters.get_cache_key', return_value=cache_key):
                    result = self.manager.load_model("test/model", torch_dtype="float16")
        
        self.assertTrue(result.success)
        self.assertTrue(result.cache_hit)
    
    @patch('model_loading_manager.DEPENDENCIES_AVAILABLE', True)
    @patch('model_loading_manager.DiffusionPipeline')
    def test_load_model_with_custom_pipeline(self, mock_pipeline_class):
        """Test model loading with custom pipeline"""
        mock_pipeline = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline
        
        with patch.object(self.manager, '_validate_model_path', return_value=True):
            with patch.object(self.manager, '_get_memory_usage', side_effect=[1024.0, 2048.0]):
                result = self.manager.load_model(
                    "test/model",
                    custom_pipeline="custom_pipeline_name"
                )
        
        self.assertTrue(result.success)
        # Verify custom_pipeline was passed
        mock_pipeline_class.from_pretrained.assert_called_once()
        call_kwargs = mock_pipeline_class.from_pretrained.call_args[1]
        self.assertEqual(call_kwargs['custom_pipeline'], "custom_pipeline_name")
    
    @patch('model_loading_manager.DEPENDENCIES_AVAILABLE', True)
    @patch('model_loading_manager.DiffusionPipeline')
    def test_load_model_with_exception(self, mock_pipeline_class):
        """Test model loading with exception during loading"""
        mock_pipeline_class.from_pretrained.side_effect = RuntimeError("Loading failed")
        
        with patch.object(self.manager, '_validate_model_path', return_value=True):
            with patch.object(self.manager, '_get_memory_usage', return_value=1024.0):
                result = self.manager.load_model("test/model")
        
        self.assertFalse(result.success)
        self.assertIn("Loading failed", result.error_message)
    
    def test_get_loading_statistics_empty(self):
        """Test getting loading statistics with empty cache"""
        stats = self.manager.get_loading_statistics()
        
        self.assertEqual(stats["total_cached_parameters"], 0)
        self.assertEqual(stats["cache_hit_rate"], 0.0)
        self.assertEqual(stats["average_loading_times"], {})
        self.assertEqual(stats["memory_usage_stats"], {})
    
    def test_get_loading_statistics_with_data(self):
        """Test getting loading statistics with cached data"""
        self.manager._parameter_cache = {
            "key1": {
                "loading_time": 120.0,
                "memory_usage_mb": 8192.0,
                "use_count": 3
            },
            "key2": {
                "loading_time": 60.0,
                "memory_usage_mb": 4096.0,
                "use_count": 1
            }
        }
        
        stats = self.manager.get_loading_statistics()
        
        self.assertEqual(stats["total_cached_parameters"], 2)
        # Cache hit rate: (3-1 + 1-1) / (3 + 1) = 2/4 = 0.5
        self.assertEqual(stats["cache_hit_rate"], 0.5)
        self.assertEqual(stats["average_loading_times"]["overall"], 90.0)  # (120 + 60) / 2
        self.assertEqual(stats["memory_usage_stats"]["average_mb"], 6144.0)  # (8192 + 4096) / 2
        self.assertEqual(stats["memory_usage_stats"]["max_mb"], 8192.0)
        self.assertEqual(stats["memory_usage_stats"]["min_mb"], 4096.0)
    
    def test_clear_cache_all(self):
        """Test clearing all cache"""
        self.manager._parameter_cache = {
            "key1": {"loading_time": 120.0},
            "key2": {"loading_time": 60.0}
        }
        
        self.manager.clear_cache()
        
        self.assertEqual(len(self.manager._parameter_cache), 0)
    
    def test_clear_cache_by_age(self):
        """Test clearing cache by age"""
        now = datetime.now()
        old_date = (now - timedelta(days=10)).isoformat()
        recent_date = (now - timedelta(hours=1)).isoformat()
        
        self.manager._parameter_cache = {
            "old_key": {
                "loading_time": 120.0,
                "last_used": old_date
            },
            "recent_key": {
                "loading_time": 60.0,
                "last_used": recent_date
            },
            "no_date_key": {
                "loading_time": 90.0
                # No last_used field
            }
        }
        
        self.manager.clear_cache(older_than_days=7)
        
        # Should keep only recent_key
        self.assertEqual(len(self.manager._parameter_cache), 1)
        self.assertIn("recent_key", self.manager._parameter_cache)
        self.assertNotIn("old_key", self.manager._parameter_cache)
        self.assertNotIn("no_date_key", self.manager._parameter_cache)  # Invalid date removed
    
    def test_get_current_progress(self):
        """Test getting current progress"""
        # Update progress
        self.manager._update_progress(
            ModelLoadingPhase.LOADING,
            75.0,
            "Loading components"
        )
        
        progress = self.manager.get_current_progress()
        
        self.assertEqual(progress.phase, ModelLoadingPhase.LOADING)
        self.assertEqual(progress.progress_percent, 75.0)
        self.assertEqual(progress.current_step, "Loading components")


class TestModelLoadingPhase(unittest.TestCase):
    """Test cases for ModelLoadingPhase enum"""
    
    def test_model_loading_phase_values(self):
        """Test ModelLoadingPhase enum values"""
        self.assertEqual(ModelLoadingPhase.INITIALIZATION.value, "initialization")
        self.assertEqual(ModelLoadingPhase.VALIDATION.value, "validation")
        self.assertEqual(ModelLoadingPhase.CACHE_CHECK.value, "cache_check")
        self.assertEqual(ModelLoadingPhase.DOWNLOAD.value, "download")
        self.assertEqual(ModelLoadingPhase.LOADING.value, "loading")
        self.assertEqual(ModelLoadingPhase.OPTIMIZATION.value, "optimization")
        self.assertEqual(ModelLoadingPhase.FINALIZATION.value, "finalization")
        self.assertEqual(ModelLoadingPhase.COMPLETED.value, "completed")
        self.assertEqual(ModelLoadingPhase.FAILED.value, "failed")


class TestModelLoadingError(unittest.TestCase):
    """Test cases for ModelLoadingError exception"""
    
    def test_model_loading_error_basic(self):
        """Test ModelLoadingError with basic message"""
        error = ModelLoadingError("Test error message")
        
        self.assertEqual(str(error), "Test error message")
        self.assertEqual(error.error_code, "UNKNOWN")
        self.assertEqual(error.suggestions, [])
    
    def test_model_loading_error_with_code_and_suggestions(self):
        """Test ModelLoadingError with error code and suggestions"""
        suggestions = ["Try this", "Or try that"]
        error = ModelLoadingError(
            "Test error message",
            error_code="TEST_ERROR",
            suggestions=suggestions
        )
        
        self.assertEqual(str(error), "Test error message")
        self.assertEqual(error.error_code, "TEST_ERROR")
        self.assertEqual(error.suggestions, suggestions)


class TestModelLoadingProgress(unittest.TestCase):
    """Test cases for ModelLoadingProgress dataclass"""
    
    def test_model_loading_progress_creation(self):
        """Test ModelLoadingProgress creation"""
        progress = ModelLoadingProgress(
            phase=ModelLoadingPhase.LOADING,
            progress_percent=65.5,
            current_step="Loading model weights",
            estimated_time_remaining=45.0,
            memory_usage_mb=6144.0,
            download_speed_mbps=25.5,
            error_message=None
        )
        
        self.assertEqual(progress.phase, ModelLoadingPhase.LOADING)
        self.assertEqual(progress.progress_percent, 65.5)
        self.assertEqual(progress.current_step, "Loading model weights")
        self.assertEqual(progress.estimated_time_remaining, 45.0)
        self.assertEqual(progress.memory_usage_mb, 6144.0)
        self.assertEqual(progress.download_speed_mbps, 25.5)
        self.assertIsNone(progress.error_message)
    
    def test_model_loading_progress_to_dict(self):
        """Test ModelLoadingProgress to_dict conversion"""
        progress = ModelLoadingProgress(
            phase=ModelLoadingPhase.COMPLETED,
            progress_percent=100.0,
            current_step="Model loading completed"
        )
        
        progress_dict = progress.to_dict()
        
        self.assertIsInstance(progress_dict, dict)
        self.assertEqual(progress_dict['phase'], ModelLoadingPhase.COMPLETED)
        self.assertEqual(progress_dict['progress_percent'], 100.0)
        self.assertEqual(progress_dict['current_step'], "Model loading completed")


class TestLoadingParameters(unittest.TestCase):
    """Test cases for LoadingParameters dataclass"""
    
    def test_loading_parameters_creation(self):
        """Test LoadingParameters creation"""
        params = LoadingParameters(
            model_path="test/model",
            torch_dtype="float16",
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=False,
            variant="fp16",
            use_safetensors=True,
            load_in_8bit=False,
            load_in_4bit=False,
            custom_pipeline="custom_pipe"
        )
        
        self.assertEqual(params.model_path, "test/model")
        self.assertEqual(params.torch_dtype, "float16")
        self.assertEqual(params.device_map, "auto")
        self.assertTrue(params.low_cpu_mem_usage)
        self.assertFalse(params.trust_remote_code)
        self.assertEqual(params.variant, "fp16")
        self.assertTrue(params.use_safetensors)
        self.assertFalse(params.load_in_8bit)
        self.assertFalse(params.load_in_4bit)
        self.assertEqual(params.custom_pipeline, "custom_pipe")
    
    def test_loading_parameters_get_cache_key(self):
        """Test LoadingParameters cache key generation"""
        params1 = LoadingParameters(
            model_path="test/model",
            torch_dtype="float16"
        )
        
        params2 = LoadingParameters(
            model_path="test/model",
            torch_dtype="float16"
        )
        
        params3 = LoadingParameters(
            model_path="test/model",
            torch_dtype="bfloat16"  # Different dtype
        )
        
        key1 = params1.get_cache_key()
        key2 = params2.get_cache_key()
        key3 = params3.get_cache_key()
        
        # Same parameters should generate same key
        self.assertEqual(key1, key2)
        
        # Different parameters should generate different key
        self.assertNotEqual(key1, key3)
        
        # Keys should be strings
        self.assertIsInstance(key1, str)
        self.assertIsInstance(key3, str)


class TestModelLoadingResult(unittest.TestCase):
    """Test cases for ModelLoadingResult dataclass"""
    
    def test_model_loading_result_success(self):
        """Test ModelLoadingResult for successful loading"""
        mock_model = MagicMock()
        parameters = LoadingParameters(model_path="test/model", torch_dtype="float16")
        
        result = ModelLoadingResult(
            success=True,
            model=mock_model,
            loading_time=120.5,
            memory_usage_mb=8192.0,
            cache_hit=True,
            error_message=None,
            error_code=None,
            suggestions=None,
            parameters_used=parameters
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.model, mock_model)
        self.assertEqual(result.loading_time, 120.5)
        self.assertEqual(result.memory_usage_mb, 8192.0)
        self.assertTrue(result.cache_hit)
        self.assertIsNone(result.error_message)
        self.assertIsNone(result.error_code)
        self.assertEqual(result.suggestions, [])  # __post_init__ sets empty list
        self.assertEqual(result.parameters_used, parameters)
    
    def test_model_loading_result_failure(self):
        """Test ModelLoadingResult for failed loading"""
        parameters = LoadingParameters(model_path="test/model", torch_dtype="float16")
        suggestions = ["Check model path", "Verify permissions"]
        
        result = ModelLoadingResult(
            success=False,
            model=None,
            loading_time=5.0,
            memory_usage_mb=0.0,
            cache_hit=False,
            error_message="Model not found",
            error_code="MODEL_NOT_FOUND",
            suggestions=suggestions,
            parameters_used=parameters
        )
        
        self.assertFalse(result.success)
        self.assertIsNone(result.model)
        self.assertEqual(result.loading_time, 5.0)
        self.assertEqual(result.memory_usage_mb, 0.0)
        self.assertFalse(result.cache_hit)
        self.assertEqual(result.error_message, "Model not found")
        self.assertEqual(result.error_code, "MODEL_NOT_FOUND")
        self.assertEqual(result.suggestions, suggestions)
        self.assertEqual(result.parameters_used, parameters)
    
    def test_model_loading_result_post_init(self):
        """Test ModelLoadingResult __post_init__ method"""
        result = ModelLoadingResult(
            success=False,
            suggestions=None  # Should be converted to empty list
        )
        
        self.assertEqual(result.suggestions, [])


class TestModelLoadingManagerIntegration(unittest.TestCase):
    """Integration tests for ModelLoadingManager"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ModelLoadingManager(
            cache_dir=str(Path(self.temp_dir) / "cache"),
            enable_logging=False
        )
    
    def tearDown(self):
        """Clean up integration test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_progress_tracking(self):
        """Test complete progress tracking workflow"""
        progress_updates = []
        
        def progress_callback(progress):
            progress_updates.append((progress.phase, progress.progress_percent, progress.current_step))
        
        self.manager.add_progress_callback(progress_callback)
        
        # Simulate progress updates
        phases = [
            (ModelLoadingPhase.INITIALIZATION, 5.0, "Initializing"),
            (ModelLoadingPhase.VALIDATION, 15.0, "Validating"),
            (ModelLoadingPhase.LOADING, 50.0, "Loading model"),
            (ModelLoadingPhase.OPTIMIZATION, 80.0, "Optimizing"),
            (ModelLoadingPhase.COMPLETED, 100.0, "Completed")
        ]
        
        for phase, percent, step in phases:
            self.manager._update_progress(phase, percent, step)
        
        # Verify all progress updates were captured
        self.assertEqual(len(progress_updates), 5)
        self.assertEqual(progress_updates[0], (ModelLoadingPhase.INITIALIZATION, 5.0, "Initializing"))
        self.assertEqual(progress_updates[-1], (ModelLoadingPhase.COMPLETED, 100.0, "Completed"))
    
    def test_cache_persistence_workflow(self):
        """Test complete cache persistence workflow"""
        # Create parameters and cache them
        parameters = LoadingParameters(
            model_path="test/integration/model",
            torch_dtype="float16",
            device_map="auto"
        )
        
        cache_key = parameters.get_cache_key()
        self.manager._cache_parameters(cache_key, parameters, 150.0, 10240.0)
        
        # Verify cache was saved
        cache_file = self.manager.cache_dir / "parameter_cache.json"
        self.assertTrue(cache_file.exists())
        
        # Create new manager instance to test loading
        new_manager = ModelLoadingManager(
            cache_dir=str(self.manager.cache_dir),
            enable_logging=False
        )
        
        # Verify cache was loaded
        self.assertEqual(len(new_manager._parameter_cache), 1)
        self.assertIn(cache_key, new_manager._parameter_cache)
        
        cached_data = new_manager._parameter_cache[cache_key]
        self.assertEqual(cached_data["loading_time"], 150.0)
        self.assertEqual(cached_data["memory_usage_mb"], 10240.0)
    
    def test_error_handling_workflow(self):
        """Test complete error handling workflow"""
        # Test different error scenarios
        error_scenarios = [
            (RuntimeError("CUDA out of memory"), "CUDA_OUT_OF_MEMORY"),
            (FileNotFoundError("Model not found"), "MODEL_NOT_FOUND"),
            (ValueError("trust_remote_code required"), "TRUST_REMOTE_CODE_ERROR"),
            (ConnectionError("Network failed"), "NETWORK_ERROR"),
            (Exception("Unknown error"), "UNKNOWN")
        ]
        
        for error, expected_code in error_scenarios:
            parameters = LoadingParameters(model_path="test/model", torch_dtype="float16")
            result = self.manager._handle_loading_error(error, parameters)
            
            self.assertFalse(result.success)
            self.assertEqual(result.error_code, expected_code)
            self.assertGreater(len(result.suggestions), 0)
            self.assertEqual(result.parameters_used, parameters)


if __name__ == '__main__':
    unittest.main()