"""
Comprehensive tests for the generation logging system.

Tests cover logging functionality, diagnostic collection, log analysis,
and log management features.
"""

import pytest
import json
import tempfile
import shutil
import time
import uuid
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from generation_logger import (
    GenerationLogger, GenerationContext, SystemDiagnostics, 
    ErrorContext, get_logger, configure_logger
)
from diagnostic_collector import (
    DiagnosticCollector, ModelDiagnostics, EnvironmentDiagnostics,
    get_diagnostic_collector
)
from log_analyzer import LogAnalyzer, LogEntry, SessionAnalysis, get_log_analyzer


class TestGenerationLogger:
    """Test cases for GenerationLogger class."""
    
    @pytest.fixture
    def temp_log_dir(self):
        """Create a temporary directory for test logs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def logger(self, temp_log_dir):
        """Create a test logger instance."""
        return GenerationLogger(
            log_dir=temp_log_dir,
            max_log_size=1024,  # Small size for testing rotation
            backup_count=2,
            log_level="DEBUG"
        )
    
    @pytest.fixture
    def sample_context(self):
        """Create a sample generation context."""
        return GenerationContext(
            session_id=str(uuid.uuid4()),
            model_type="wan22_t2v",
            generation_mode="T2V",
            prompt="A beautiful sunset over mountains",
            parameters={
                "resolution": "720p",
                "steps": 20,
                "guidance_scale": 7.5
            },
            start_time=time.time()
        )
    
    def test_logger_initialization(self, temp_log_dir):
        """Test logger initialization and setup."""
        logger = GenerationLogger(log_dir=temp_log_dir)
        
        # Check log directory creation
        assert Path(temp_log_dir).exists()
        
        # Check logger setup
        assert logger.generation_logger is not None
        assert logger.error_logger is not None
        assert logger.performance_logger is not None
        assert logger.diagnostics_logger is not None
    
    def test_generation_session_context_manager(self, logger, sample_context):
        """Test generation session context manager."""
        with logger.generation_session(sample_context):
            logger.log_pipeline_stage("validation", "Input validation completed")
            logger.log_pipeline_stage("generation", "Video generation started")
        
        # Check that logs were written
        log_file = Path(logger.log_dir) / "generation.log"
        assert log_file.exists()
        
        with open(log_file, 'r') as f:
            content = f.read()
            assert sample_context.session_id in content
            assert "validation" in content
            assert "generation" in content
    
    def test_generation_session_with_error(self, logger, sample_context):
        """Test generation session context manager with error."""
        test_error = ValueError("Test error message")
        
        with pytest.raises(ValueError):
            with logger.generation_session(sample_context):
                logger.log_pipeline_stage("validation", "Input validation completed")
                raise test_error
        
        # Check error logging
        error_log_file = Path(logger.log_dir) / "errors.log"
        assert error_log_file.exists()
        
        with open(error_log_file, 'r') as f:
            content = f.read()
            assert sample_context.session_id in content
            assert "Test error message" in content
    
    def test_log_generation_start(self, logger, sample_context):
        """Test logging generation start."""
        logger.log_generation_start(sample_context)
        
        log_file = Path(logger.log_dir) / "generation.log"
        with open(log_file, 'r') as f:
            content = f.read()
            assert sample_context.session_id in content
            assert sample_context.model_type in content
            assert sample_context.generation_mode in content
    
    def test_log_generation_success(self, logger, sample_context):
        """Test logging successful generation."""
        duration = 45.5
        logger.log_generation_success(sample_context, duration)
        
        # Check generation log
        gen_log_file = Path(logger.log_dir) / "generation.log"
        with open(gen_log_file, 'r') as f:
            content = f.read()
            assert "completed successfully" in content
            assert str(duration) in content
        
        # Check performance log
        perf_log_file = Path(logger.log_dir) / "performance.log"
        with open(perf_log_file, 'r') as f:
            content = f.read()
            assert sample_context.session_id in content
            assert '"status": "success"' in content
    
    def test_log_generation_error(self, logger, sample_context):
        """Test logging generation error."""
        test_error = RuntimeError("Generation failed")
        duration = 30.0
        
        logger.log_generation_error(test_error, sample_context, duration)
        
        # Check error log
        error_log_file = Path(logger.log_dir) / "errors.log"
        with open(error_log_file, 'r') as f:
            content = f.read()
            assert sample_context.session_id in content
            assert "Generation failed" in content
    
    def test_log_pipeline_stage(self, logger, sample_context):
        """Test logging pipeline stages."""
        with logger.generation_session(sample_context):
            logger.log_pipeline_stage("model_loading", "Loading model checkpoint")
            logger.log_pipeline_stage("preprocessing", "Preprocessing input data")
        
        log_file = Path(logger.log_dir) / "generation.log"
        with open(log_file, 'r') as f:
            content = f.read()
            assert "model_loading" in content
            assert "preprocessing" in content
            assert sample_context.session_id in content
    
    def test_log_model_loading(self, logger):
        """Test logging model loading events."""
        model_type = "wan22_t2v"
        model_path = "/path/to/model"
        duration = 15.2
        
        # Test successful loading
        logger.log_model_loading(model_type, model_path, True, duration)
        
        # Test failed loading
        error_msg = "Model file not found"
        logger.log_model_loading(model_type, model_path, False, duration, error_msg)
        
        log_file = Path(logger.log_dir) / "generation.log"
        with open(log_file, 'r') as f:
            content = f.read()
            assert "Model loading success" in content
            assert "Model loading failed" in content
            assert error_msg in content
    
    def test_log_vram_usage(self, logger, sample_context):
        """Test logging VRAM usage."""
        with logger.generation_session(sample_context):
            logger.log_vram_usage("model_loading", 8.5, 12.0, 70.8)
            logger.log_vram_usage("generation", 11.2, 12.0, 93.3)
        
        perf_log_file = Path(logger.log_dir) / "performance.log"
        with open(perf_log_file, 'r') as f:
            content = f.read()
            assert "model_loading" in content
            assert "generation" in content
            assert "8.5" in content
            assert "11.2" in content
    
    def test_log_parameter_optimization(self, logger, sample_context):
        """Test logging parameter optimization."""
        original_params = {"steps": 50, "guidance_scale": 7.5}
        optimized_params = {"steps": 30, "guidance_scale": 6.0}
        reason = "VRAM optimization"
        
        with logger.generation_session(sample_context):
            logger.log_parameter_optimization(original_params, optimized_params, reason)
        
        log_file = Path(logger.log_dir) / "generation.log"
        with open(log_file, 'r') as f:
            content = f.read()
            assert "Parameter optimization applied" in content
            assert reason in content
    
    def test_log_recovery_attempt(self, logger, sample_context):
        """Test logging recovery attempts."""
        with logger.generation_session(sample_context):
            logger.log_recovery_attempt("VRAMError", "reduce_batch_size", True)
            logger.log_recovery_attempt("ModelLoadingError", "reload_model", False)
        
        log_file = Path(logger.log_dir) / "generation.log"
        with open(log_file, 'r') as f:
            content = f.read()
            assert "Recovery attempt successful" in content
            assert "Recovery attempt failed" in content
            assert "VRAMError" in content
            assert "ModelLoadingError" in content
    
    def test_get_session_logs(self, logger, sample_context):
        """Test retrieving logs for a specific session."""
        with logger.generation_session(sample_context):
            logger.log_pipeline_stage("validation", "Validation completed")
            logger.log_pipeline_stage("generation", "Generation started")
        
        session_logs = logger.get_session_logs(sample_context.session_id)
        
        assert 'generation' in session_logs
        assert 'errors' in session_logs
        assert 'performance' in session_logs
        assert 'diagnostics' in session_logs
        
        # Check that session logs contain expected entries
        gen_logs = session_logs['generation']
        assert any(sample_context.session_id in log for log in gen_logs)
        assert any("validation" in log for log in gen_logs)
    
    def test_generate_diagnostic_report(self, logger, sample_context):
        """Test generating diagnostic report."""
        with logger.generation_session(sample_context):
            logger.log_pipeline_stage("test", "Test stage")
        
        report = logger.generate_diagnostic_report(sample_context.session_id)
        
        assert 'timestamp' in report
        assert 'system_diagnostics' in report
        assert 'log_summary' in report
        assert 'session_logs' in report
        
        # Check system diagnostics structure
        sys_diag = report['system_diagnostics']
        assert 'cpu_usage' in sys_diag
        assert 'memory_usage' in sys_diag
        assert 'cuda_available' in sys_diag
    
    def test_log_rotation(self, temp_log_dir):
        """Test log file rotation."""
        logger = GenerationLogger(
            log_dir=temp_log_dir,
            max_log_size=100,  # Very small size to trigger rotation
            backup_count=2
        )
        
        # Generate enough log entries to trigger rotation
        for i in range(50):
            logger.generation_logger.info(f"Test log entry {i} with some additional text to increase size")
        
        # Check that rotation occurred
        log_files = list(Path(temp_log_dir).glob("generation.log*"))
        assert len(log_files) > 1  # Should have main log + rotated files
    
    def test_cleanup_old_logs(self, logger, temp_log_dir):
        """Test cleanup of old log files."""
        # Create some old log files
        old_log = Path(temp_log_dir) / "old.log"
        old_log.write_text("old log content")
        
        # Set modification time to be old
        old_time = time.time() - (31 * 24 * 60 * 60)  # 31 days ago
        os.utime(old_log, (old_time, old_time))
        
        # Create a recent log file
        recent_log = Path(temp_log_dir) / "recent.log"
        recent_log.write_text("recent log content")
        
        # Run cleanup
        logger.cleanup_old_logs(days_to_keep=30)
        
        # Check results
        assert not old_log.exists()
        assert recent_log.exists()
    
    def test_singleton_logger(self):
        """Test singleton pattern for global logger."""
        logger1 = get_logger()
        logger2 = get_logger()
        
        assert logger1 is logger2
    
    def test_configure_logger(self, temp_log_dir):
        """Test logger configuration."""
        logger = configure_logger(
            log_dir=temp_log_dir,
            log_level="DEBUG"
        )
        
        assert logger.log_dir == Path(temp_log_dir)
        assert logger.log_level == 10  # DEBUG level


class TestSystemDiagnostics:
    """Test cases for SystemDiagnostics class."""
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('torch.cuda.is_available')
    def test_collect_diagnostics(self, mock_cuda_available, mock_disk_usage, 
                                mock_virtual_memory, mock_cpu_percent):
        """Test collecting system diagnostics."""
        # Mock system information
        mock_cpu_percent.return_value = 45.2
        mock_virtual_memory.return_value = Mock(percent=67.8)
        mock_disk_usage.return_value = Mock(percent=23.4)
        mock_cuda_available.return_value = True
        
        with patch('torch.cuda.memory_allocated', return_value=8 * 1024**3):
            with patch('torch.cuda.get_device_properties') as mock_props:
                mock_props.return_value = Mock(total_memory=12 * 1024**3)
                
                diagnostics = SystemDiagnostics.collect()
        
        assert diagnostics.cpu_usage == 45.2
        assert diagnostics.memory_usage == 67.8
        assert diagnostics.disk_usage == 23.4
        assert diagnostics.cuda_available is True
        assert diagnostics.gpu_memory_used == 8.0
        assert diagnostics.gpu_memory_total == 12.0


class TestDiagnosticCollector:
    """Test cases for DiagnosticCollector class."""
    
    @pytest.fixture
    def temp_log_dir(self):
        """Create a temporary directory for test logs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def collector(self, temp_log_dir):
        """Create a test diagnostic collector."""
        # Configure logger to use temp directory
        configure_logger(log_dir=temp_log_dir)
        return DiagnosticCollector()
    
    def test_collect_full_diagnostics(self, collector):
        """Test collecting full diagnostic information."""
        diagnostics = collector.collect_full_diagnostics()
        
        assert 'collection_timestamp' in diagnostics
        assert 'system' in diagnostics
        assert 'environment' in diagnostics
        assert 'platform' in diagnostics
        assert 'models' in diagnostics
        assert 'logs' in diagnostics
        
        # Check system diagnostics structure
        system = diagnostics['system']
        assert 'cpu_usage' in system
        assert 'memory_usage' in system
        assert 'cuda_available' in system
        
        # Check environment diagnostics structure
        environment = diagnostics['environment']
        assert 'python_version' in environment
        assert 'torch_version' in environment
        assert 'installed_packages' in environment
    
    def test_collect_model_diagnostics(self, collector):
        """Test collecting model diagnostics."""
        model_diagnostics = collector._collect_model_diagnostics()
        
        assert isinstance(model_diagnostics, dict)
        # Should have entries for known model paths
        assert len(model_diagnostics) > 0
        
        for model_name, model_info in model_diagnostics.items():
            assert 'model_type' in model_info or 'error' in model_info
            assert 'model_path' in model_info
    
    def test_export_diagnostics_json(self, collector, temp_log_dir):
        """Test exporting diagnostics to JSON."""
        output_path = Path(temp_log_dir) / "diagnostics.json"
        
        exported_path = collector.export_diagnostics(str(output_path), format='json')
        
        assert Path(exported_path).exists()
        
        # Verify JSON content
        with open(exported_path, 'r') as f:
            data = json.load(f)
            assert 'collection_timestamp' in data
            assert 'system' in data
    
    def test_export_diagnostics_text(self, collector, temp_log_dir):
        """Test exporting diagnostics to text format."""
        output_path = Path(temp_log_dir) / "diagnostics.txt"
        
        exported_path = collector.export_diagnostics(str(output_path), format='txt')
        
        assert Path(exported_path).exists()
        
        # Verify text content
        with open(exported_path, 'r') as f:
            content = f.read()
            assert "WAN2.2 Video Generation Diagnostic Report" in content
            assert "SYSTEM INFORMATION" in content
    
    def test_singleton_collector(self):
        """Test singleton pattern for diagnostic collector."""
        collector1 = get_diagnostic_collector()
        collector2 = get_diagnostic_collector()
        
        assert collector1 is collector2


class TestModelDiagnostics:
    """Test cases for ModelDiagnostics class."""
    
    def test_collect_existing_file(self, tmp_path):
        """Test collecting diagnostics for existing model file."""
        model_file = tmp_path / "test_model.pt"
        model_file.write_text("fake model content")
        
        diagnostics = ModelDiagnostics.collect("test_model", str(model_file))
        
        assert diagnostics.model_type == "test_model"
        assert diagnostics.model_path == str(model_file)
        assert diagnostics.model_exists is True
        assert diagnostics.model_accessible is True
        assert diagnostics.model_format == "pytorch"
        assert diagnostics.model_size_gb is not None
    
    def test_collect_nonexistent_file(self):
        """Test collecting diagnostics for non-existent model."""
        diagnostics = ModelDiagnostics.collect("missing_model", "/nonexistent/path")
        
        assert diagnostics.model_type == "missing_model"
        assert diagnostics.model_exists is False
        assert diagnostics.model_accessible is False
        assert diagnostics.model_size_gb is None


class TestEnvironmentDiagnostics:
    """Test cases for EnvironmentDiagnostics class."""
    
    def test_collect_environment_info(self):
        """Test collecting environment diagnostic information."""
        diagnostics = EnvironmentDiagnostics.collect()
        
        assert diagnostics.python_version is not None
        assert diagnostics.python_executable is not None
        assert diagnostics.torch_version is not None
        assert isinstance(diagnostics.cuda_available, bool)
        assert isinstance(diagnostics.gpu_count, int)
        assert isinstance(diagnostics.gpu_names, list)
        assert isinstance(diagnostics.installed_packages, dict)
        assert isinstance(diagnostics.environment_variables, dict)


class TestLogAnalyzer:
    """Test cases for LogAnalyzer class."""
    
    @pytest.fixture
    def temp_log_dir(self):
        """Create a temporary directory for test logs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def analyzer(self, temp_log_dir):
        """Create a test log analyzer."""
        return LogAnalyzer(log_dir=temp_log_dir)
    
    @pytest.fixture
    def sample_log_entries(self, temp_log_dir):
        """Create sample log files for testing."""
        session_id = str(uuid.uuid4())
        
        # Create generation log
        gen_log = Path(temp_log_dir) / "generation.log"
        gen_log.write_text(f"""2024-01-15 10:30:00,123 - generation - INFO - Starting generation session {session_id} - Model: wan22_t2v, Mode: T2V
2024-01-15 10:30:05,456 - generation - INFO - [{session_id}] validation: Input validation completed
2024-01-15 10:30:10,789 - generation - INFO - [{session_id}] model_loading: Model loaded successfully
2024-01-15 10:30:45,012 - generation - INFO - Generation session {session_id} completed successfully in 45.2s
""")
        
        # Create error log
        error_log = Path(temp_log_dir) / "errors.log"
        error_log.write_text(f"""2024-01-15 11:00:00,123 - generation.errors - ERROR - Generation session {session_id} failed: CUDA out of memory
2024-01-15 11:00:00,124 - generation.errors - ERROR - Error context for session {session_id}: {{"error_type": "RuntimeError"}}
""")
        
        # Create performance log
        perf_log = Path(temp_log_dir) / "performance.log"
        perf_log.write_text(f"""2024-01-15 10:30:45,012 - {{"session_id": "{session_id}", "model_type": "wan22_t2v", "duration": 45.2, "status": "success"}}
2024-01-15 11:00:00,123 - {{"session_id": "{session_id}", "model_type": "wan22_t2v", "duration": 30.0, "status": "error", "error_type": "RuntimeError"}}
""")
        
        return session_id
    
    def test_log_entry_parsing(self):
        """Test parsing log entries."""
        log_line = "2024-01-15 10:30:00,123 - generation - INFO - Test message"
        entry = LogEntry.parse(log_line)
        
        assert entry is not None
        assert entry.level == "INFO"
        assert entry.logger_name == "generation"
        assert entry.message == "Test message"
        assert entry.timestamp.year == 2024
        assert entry.timestamp.month == 1
        assert entry.timestamp.day == 15
    
    def test_log_entry_with_session_id(self):
        """Test parsing log entries with session IDs."""
        session_id = str(uuid.uuid4())
        log_line = f"2024-01-15 10:30:00,123 - generation - INFO - [{session_id}] Test message"
        entry = LogEntry.parse(log_line)
        
        assert entry is not None
        assert entry.session_id == session_id
    
    def test_analyze_logs(self, analyzer, sample_log_entries):
        """Test comprehensive log analysis."""
        start_time = datetime(2024, 1, 15, 10, 0, 0)
        end_time = datetime(2024, 1, 15, 12, 0, 0)
        
        report = analyzer.analyze_logs(start_time, end_time)
        
        assert report.total_sessions > 0
        assert report.analysis_period == (start_time, end_time)
        assert isinstance(report.error_patterns, dict)
        assert isinstance(report.performance_trends, dict)
        assert isinstance(report.session_analyses, list)
    
    def test_session_analysis(self, analyzer, sample_log_entries):
        """Test individual session analysis."""
        # Read log entries
        entries = analyzer._read_log_entries(
            'generation', 
            datetime(2024, 1, 15, 10, 0, 0),
            datetime(2024, 1, 15, 12, 0, 0)
        )
        
        # Group by session
        sessions = analyzer._group_by_session(entries)
        
        # Analyze first session
        session_id = list(sessions.keys())[0]
        session_entries = sessions[session_id]
        analysis = analyzer._analyze_session(session_id, session_entries)
        
        assert analysis.session_id == session_id
        assert analysis.start_time is not None
        assert analysis.end_time is not None
        assert analysis.duration is not None
        assert analysis.model_type is not None
    
    def test_export_analysis_report_json(self, analyzer, sample_log_entries, temp_log_dir):
        """Test exporting analysis report to JSON."""
        report = analyzer.analyze_logs()
        output_path = Path(temp_log_dir) / "analysis_report.json"
        
        exported_path = analyzer.export_analysis_report(report, str(output_path), 'json')
        
        assert Path(exported_path).exists()
        
        # Verify JSON content
        with open(exported_path, 'r') as f:
            data = json.load(f)
            assert 'analysis_period' in data
            assert 'total_sessions' in data
    
    def test_export_analysis_report_html(self, analyzer, sample_log_entries, temp_log_dir):
        """Test exporting analysis report to HTML."""
        report = analyzer.analyze_logs()
        output_path = Path(temp_log_dir) / "analysis_report.html"
        
        exported_path = analyzer.export_analysis_report(report, str(output_path), 'html')
        
        assert Path(exported_path).exists()
        
        # Verify HTML content
        with open(exported_path, 'r') as f:
            content = f.read()
            assert "WAN2.2 Log Analysis Report" in content
            assert "<html>" in content
    
    def test_compress_old_logs(self, analyzer, temp_log_dir):
        """Test compressing old log files."""
        # Create an old log file
        old_log = Path(temp_log_dir) / "old.log"
        old_log.write_text("old log content")
        
        # Set modification time to be old
        old_time = time.time() - (8 * 24 * 60 * 60)  # 8 days ago
        os.utime(old_log, (old_time, old_time))
        
        # Create a recent log file
        recent_log = Path(temp_log_dir) / "recent.log"
        recent_log.write_text("recent log content")
        
        # Run compression
        analyzer.compress_old_logs(days_to_keep=7)
        
        # Check results
        assert not old_log.exists()
        assert (Path(temp_log_dir) / "old.log.gz").exists()
        assert recent_log.exists()
    
    def test_singleton_analyzer(self):
        """Test singleton pattern for log analyzer."""
        analyzer1 = get_log_analyzer()
        analyzer2 = get_log_analyzer()
        
        assert analyzer1 is analyzer2


class TestIntegration:
    """Integration tests for the complete logging system."""
    
    @pytest.fixture
    def temp_log_dir(self):
        """Create a temporary directory for test logs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_end_to_end_logging_workflow(self, temp_log_dir):
        """Test complete logging workflow from generation to analysis."""
        # Configure logger
        logger = configure_logger(log_dir=temp_log_dir, log_level="DEBUG")
        
        # Create sample generation context
        context = GenerationContext(
            session_id=str(uuid.uuid4()),
            model_type="wan22_t2v",
            generation_mode="T2V",
            prompt="Test prompt",
            parameters={"resolution": "720p", "steps": 20},
            start_time=time.time()
        )
        
        # Simulate generation workflow
        with logger.generation_session(context):
            logger.log_pipeline_stage("validation", "Input validation completed")
            logger.log_model_loading("wan22_t2v", "/path/to/model", True, 15.2)
            logger.log_vram_usage("generation", 8.5, 12.0, 70.8)
            logger.log_pipeline_stage("generation", "Video generation completed")
        
        # Collect diagnostics
        collector = DiagnosticCollector()
        diagnostics = collector.collect_full_diagnostics(context.session_id)
        
        # Analyze logs
        analyzer = LogAnalyzer(log_dir=temp_log_dir)
        report = analyzer.analyze_logs()
        
        # Verify results
        assert diagnostics['collection_timestamp'] is not None
        assert report.total_sessions > 0
        assert len(report.session_analyses) > 0
        
        # Check that session was analyzed correctly
        session_analysis = next(
            (s for s in report.session_analyses if s.session_id == context.session_id),
            None
        )
        assert session_analysis is not None
        assert session_analysis.status == 'success'
        assert session_analysis.model_type == 'wan22_t2v'
    
    def test_error_handling_and_recovery_logging(self, temp_log_dir):
        """Test logging of error handling and recovery scenarios."""
        logger = configure_logger(log_dir=temp_log_dir)
        
        context = GenerationContext(
            session_id=str(uuid.uuid4()),
            model_type="wan22_t2v",
            generation_mode="T2V",
            prompt="Test prompt",
            parameters={"resolution": "720p"},
            start_time=time.time()
        )
        
        # Simulate error and recovery
        try:
            with logger.generation_session(context):
                logger.log_pipeline_stage("validation", "Input validation completed")
                logger.log_recovery_attempt("VRAMError", "reduce_batch_size", True)
                logger.log_parameter_optimization(
                    {"batch_size": 4}, {"batch_size": 2}, "VRAM optimization"
                )
                raise RuntimeError("Simulated generation error")
        except RuntimeError:
            pass
        
        # Analyze the logs
        analyzer = LogAnalyzer(log_dir=temp_log_dir)
        report = analyzer.analyze_logs()
        
        # Verify error was captured and categorized
        assert report.failed_sessions > 0
        assert 'Other Error' in report.error_patterns or 'GPU/CUDA Error' in report.error_patterns
        
        # Check session analysis
        session_analysis = next(
            (s for s in report.session_analyses if s.session_id == context.session_id),
            None
        )
        assert session_analysis is not None
        assert session_analysis.status == 'error'
        assert session_analysis.error_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])