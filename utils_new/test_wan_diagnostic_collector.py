"""
Tests for the Wan model compatibility diagnostic collector.

This module tests the diagnostic collection, analysis, and reporting capabilities
for Wan model compatibility issues.
"""

import json
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

from wan_diagnostic_collector import (
    DiagnosticCollector,
    ModelDiagnostics,
    SystemInfo,
    ModelAnalysis,
    PipelineAttempt
)
from architecture_detector import ArchitectureType, ModelArchitecture, ComponentInfo


class TestSystemInfo:
    """Test SystemInfo collection and functionality."""

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.get_device_name')
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.memory_allocated')
    def test_collect_with_cuda(self, mock_memory_allocated, mock_device_props, 
                              mock_device_name, mock_device_count, mock_cuda_available):
        """Test system info collection with CUDA available."""
        # Setup mocks
        mock_cuda_available.return_value = True
        mock_device_count.return_value = 1
        mock_device_name.return_value = "NVIDIA RTX 4090"
        
        # Mock device properties
        mock_props = Mock()
        mock_props.total_memory = 24 * 1024 * 1024 * 1024  # 24GB in bytes
        mock_device_props.return_value = mock_props
        mock_memory_allocated.return_value = 2 * 1024 * 1024 * 1024  # 2GB allocated
        
        system_info = SystemInfo.collect()
        
        assert system_info.cuda_available is True
        assert system_info.gpu_name == "NVIDIA RTX 4090"
        assert system_info.vram_total == 24576  # 24GB in MB
        assert system_info.vram_available == 22528  # 22GB available in MB
        assert system_info.python_version is not None
        assert system_info.torch_version is not None

    @patch('torch.cuda.is_available')
    def test_collect_without_cuda(self, mock_cuda_available):
        """Test system info collection without CUDA."""
        mock_cuda_available.return_value = False
        
        system_info = SystemInfo.collect()
        
        assert system_info.cuda_available is False
        assert system_info.gpu_name is None
        assert system_info.vram_total is None
        assert system_info.vram_available is None
        assert system_info.python_version is not None
        assert system_info.torch_version is not None

    def test_to_dict(self):
        """Test SystemInfo to_dict conversion."""
        system_info = SystemInfo(
            gpu_name="Test GPU",
            vram_total=8192,
            vram_available=6144,
            python_version="3.10.0",
            torch_version="2.0.0",
            diffusers_version="0.21.0",
            cuda_available=True,
            cuda_version="11.8",
            platform_info={"system": "Linux"}
        )
        
        result = system_info.to_dict()
        assert isinstance(result, dict)
        assert result["gpu_name"] == "Test GPU"
        assert result["vram_total"] == 8192


class TestModelAnalysis:
    """Test ModelAnalysis functionality."""

    def test_analyze_model_with_valid_wan_model(self, tmp_path):
        """Test model analysis with a valid Wan model."""
        # Create mock model directory
        model_path = tmp_path / "wan_model"
        model_path.mkdir()
        
        # Create mock model_index.json
        model_index = {
            "_class_name": "WanPipeline",
            "_diffusers_version": "0.21.0",
            "transformer": ["diffusers", "Transformer2DModel"],
            "transformer_2": ["diffusers", "Transformer2DModel"],
            "vae": ["diffusers", "AutoencoderKL"],
            "scheduler": ["diffusers", "DDIMScheduler"],
            "boundary_ratio": 0.5
        }
        
        with open(model_path / "model_index.json", "w") as f:
            json.dump(model_index, f)
        
        # Create mock VAE config
        vae_dir = model_path / "vae"
        vae_dir.mkdir()
        vae_config = {
            "in_channels": 4,
            "out_channels": 4,
            "latent_channels": 4
        }
        with open(vae_dir / "config.json", "w") as f:
            json.dump(vae_config, f)
        
        # Mock architecture detector
        mock_detector = Mock()
        mock_architecture = ModelArchitecture(
            architecture_type=ArchitectureType.WAN_T2V,
            version="2.2",
            components={
                "transformer": ComponentInfo("Transformer2DModel", "transformer/config.json", "transformer/diffusion_pytorch_model.safetensors", True, []),
                "vae": ComponentInfo("AutoencoderKL", "vae/config.json", "vae/diffusion_pytorch_model.safetensors", False, [])
            },
            requirements=Mock(),
            capabilities=["text_to_video"]
        )
        mock_detector.detect_model_architecture.return_value = mock_architecture
        
        analysis = ModelAnalysis.analyze_model(str(model_path), mock_detector)
        
        assert analysis.architecture_detected == "wan_t2v"
        assert analysis.has_model_index is True
        assert analysis.model_index_valid is True
        assert analysis.is_wan_architecture is True
        assert analysis.pipeline_class_detected == "WanPipeline"
        assert "boundary_ratio" in analysis.custom_attributes
        assert "transformer" in analysis.components_found
        assert "vae" in analysis.components_found

    def test_analyze_model_without_model_index(self, tmp_path):
        """Test model analysis without model_index.json."""
        model_path = tmp_path / "model_no_index"
        model_path.mkdir()
        
        # Mock architecture detector
        mock_detector = Mock()
        mock_architecture = ModelArchitecture(
            architecture_type=ArchitectureType.STABLE_DIFFUSION,
            version="1.5",
            components={},
            requirements=Mock(),
            capabilities=["text_to_image"]
        )
        mock_detector.detect_model_architecture.return_value = mock_architecture
        
        analysis = ModelAnalysis.analyze_model(str(model_path), mock_detector)
        
        assert analysis.has_model_index is False
        assert analysis.model_index_valid is False
        assert analysis.is_wan_architecture is False

    def test_analyze_model_with_corrupted_index(self, tmp_path):
        """Test model analysis with corrupted model_index.json."""
        model_path = tmp_path / "model_corrupted"
        model_path.mkdir()
        
        # Create corrupted model_index.json
        with open(model_path / "model_index.json", "w") as f:
            f.write("{ invalid json")
        
        # Mock architecture detector
        mock_detector = Mock()
        mock_architecture = ModelArchitecture(
            architecture_type=ArchitectureType.UNKNOWN,
            version=None,
            components={},
            requirements=Mock(),
            capabilities=[]
        )
        mock_detector.detect_model_architecture.return_value = mock_architecture
        
        analysis = ModelAnalysis.analyze_model(str(model_path), mock_detector)
        
        assert analysis.has_model_index is True
        assert analysis.model_index_valid is False


class TestPipelineAttempt:
    """Test PipelineAttempt functionality."""

    def test_create_failed_attempt(self):
        """Test creating a failed pipeline attempt."""
        error = ValueError("Pipeline not found")
        attempt = PipelineAttempt.create_failed_attempt("WanPipeline", error, trust_remote_code=True)
        
        assert attempt.attempted_pipeline == "WanPipeline"
        assert attempt.pipeline_available is False
        assert attempt.trust_remote_code is True
        assert attempt.load_success is False
        assert attempt.error_type == "ValueError"
        assert attempt.error_message == "Pipeline not found"


class TestModelDiagnostics:
    """Test ModelDiagnostics functionality."""

    def test_to_json(self):
        """Test ModelDiagnostics JSON serialization."""
        diagnostics = ModelDiagnostics(
            model_path="/path/to/model",
            model_name="test_model",
            timestamp="2024-01-01T00:00:00",
            system_info={"gpu": "RTX 4090"},
            model_analysis={"architecture": "wan_t2v"},
            pipeline_attempt={"success": False},
            errors=["Error 1"],
            warnings=["Warning 1"],
            recommendations=["Recommendation 1"]
        )
        
        json_str = diagnostics.to_json()
        parsed = json.loads(json_str)
        
        assert parsed["model_name"] == "test_model"
        assert parsed["errors"] == ["Error 1"]
        assert parsed["warnings"] == ["Warning 1"]
        assert parsed["recommendations"] == ["Recommendation 1"]

    def test_to_dict(self):
        """Test ModelDiagnostics dictionary conversion."""
        diagnostics = ModelDiagnostics(
            model_path="/path/to/model",
            model_name="test_model",
            timestamp="2024-01-01T00:00:00",
            system_info={"gpu": "RTX 4090"},
            model_analysis={"architecture": "wan_t2v"},
            pipeline_attempt={"success": False},
            errors=["Error 1"],
            warnings=["Warning 1"],
            recommendations=["Recommendation 1"]
        )
        
        result = diagnostics.to_dict()
        assert isinstance(result, dict)
        assert result["model_name"] == "test_model"


class TestDiagnosticCollector:
    """Test DiagnosticCollector functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.collector = DiagnosticCollector(diagnostics_dir=self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init(self):
        """Test DiagnosticCollector initialization."""
        assert self.collector.diagnostics_dir.exists()
        assert self.collector.detector is not None

    @patch('wan_diagnostic_collector.SystemInfo.collect')
    @patch('wan_diagnostic_collector.ModelAnalysis.analyze_model')
    def test_collect_model_diagnostics(self, mock_analyze_model, mock_collect_system):
        """Test comprehensive model diagnostics collection."""
        # Setup mocks
        mock_system_info = SystemInfo(
            gpu_name="RTX 4090",
            vram_total=24576,
            vram_available=20480,
            python_version="3.10.0",
            torch_version="2.0.0",
            diffusers_version="0.21.0",
            cuda_available=True,
            cuda_version="11.8",
            platform_info={"system": "Linux"}
        )
        mock_collect_system.return_value = mock_system_info
        
        mock_model_analysis = ModelAnalysis(
            architecture_detected="wan_t2v",
            has_model_index=True,
            model_index_valid=True,
            components_found=["transformer", "vae"],
            vae_dimensions=3,
            custom_attributes=["boundary_ratio"],
            pipeline_class_detected="WanPipeline",
            is_wan_architecture=True,
            component_details={}
        )
        mock_analyze_model.return_value = mock_model_analysis
        
        # Test with no load attempt
        diagnostics = self.collector.collect_model_diagnostics("/path/to/model")
        
        assert diagnostics.model_path == "/path/to/model"
        assert diagnostics.model_name == "model"
        assert diagnostics.system_info["gpu_name"] == "RTX 4090"
        assert diagnostics.model_analysis["architecture_detected"] == "wan_t2v"
        assert diagnostics.pipeline_attempt["attempted_pipeline"] == "WanPipeline"
        assert len(diagnostics.errors) >= 0
        assert len(diagnostics.warnings) >= 0
        assert len(diagnostics.recommendations) >= 0

    def test_collect_model_diagnostics_with_exception(self):
        """Test diagnostics collection with pipeline loading exception."""
        error = ImportError("WanPipeline not found")
        
        with patch('wan_diagnostic_collector.SystemInfo.collect') as mock_collect_system, \
             patch('wan_diagnostic_collector.ModelAnalysis.analyze_model') as mock_analyze_model:
            
            mock_system_info = SystemInfo(
                gpu_name=None,
                vram_total=None,
                vram_available=None,
                python_version="3.10.0",
                torch_version="2.0.0",
                diffusers_version=None,
                cuda_available=False,
                cuda_version=None,
                platform_info={"system": "Linux"}
            )
            mock_collect_system.return_value = mock_system_info
            
            mock_model_analysis = ModelAnalysis(
                architecture_detected="wan_t2v",
                has_model_index=True,
                model_index_valid=True,
                components_found=["transformer"],
                vae_dimensions=3,
                custom_attributes=["boundary_ratio"],
                pipeline_class_detected="WanPipeline",
                is_wan_architecture=True,
                component_details={}
            )
            mock_analyze_model.return_value = mock_model_analysis
            
            diagnostics = self.collector.collect_model_diagnostics("/path/to/model", error)
            
            assert diagnostics.pipeline_attempt["load_success"] is False
            assert diagnostics.pipeline_attempt["error_type"] == "ImportError"
            assert "WanPipeline not found" in diagnostics.pipeline_attempt["error_message"]

    def test_write_compatibility_report(self):
        """Test writing compatibility report to file."""
        diagnostics = ModelDiagnostics(
            model_path="/path/to/model",
            model_name="test_model",
            timestamp="2024-01-01T00:00:00",
            system_info={"gpu": "RTX 4090"},
            model_analysis={"architecture": "wan_t2v"},
            pipeline_attempt={"success": False},
            errors=["Error 1"],
            warnings=["Warning 1"],
            recommendations=["Recommendation 1"]
        )
        
        report_path = self.collector.write_compatibility_report("test_model", diagnostics)
        
        assert Path(report_path).exists()
        assert "test_model_compat.json" in report_path
        
        # Verify file contents
        with open(report_path, 'r') as f:
            data = json.load(f)
        
        assert data["model_name"] == "test_model"
        assert data["errors"] == ["Error 1"]

    def test_write_compatibility_report_sanitized_name(self):
        """Test writing compatibility report with sanitized filename."""
        diagnostics = ModelDiagnostics(
            model_path="/path/to/model",
            model_name="test/model:with*special?chars",
            timestamp="2024-01-01T00:00:00",
            system_info={},
            model_analysis={},
            pipeline_attempt={},
            errors=[],
            warnings=[],
            recommendations=[]
        )
        
        report_path = self.collector.write_compatibility_report("test/model:with*special?chars", diagnostics)
        
        assert Path(report_path).exists()
        # Filename should be sanitized
        assert "testmodelwithspecialchars_compat.json" in Path(report_path).name

    def test_generate_diagnostic_summary(self):
        """Test generating human-readable diagnostic summary."""
        diagnostics = ModelDiagnostics(
            model_path="/path/to/model",
            model_name="test_model",
            timestamp="2024-01-01T00:00:00",
            system_info={
                "gpu_name": "RTX 4090",
                "vram_available": 20480,
                "vram_total": 24576,
                "cuda_available": True,
                "cuda_version": "11.8",
                "python_version": "3.10.0",
                "torch_version": "2.0.0",
                "diffusers_version": "0.21.0"
            },
            model_analysis={
                "architecture_detected": "wan_t2v",
                "is_wan_architecture": True,
                "pipeline_class_detected": "WanPipeline",
                "model_index_valid": True,
                "components_found": ["transformer", "vae"],
                "vae_dimensions": 3,
                "custom_attributes": ["boundary_ratio"]
            },
            pipeline_attempt={
                "attempted_pipeline": "WanPipeline",
                "load_success": False,
                "pipeline_available": False,
                "trust_remote_code": False,
                "error_type": "ImportError"
            },
            errors=["WanPipeline class not found"],
            warnings=["VRAM below recommended 12GB"],
            recommendations=["Install WanPipeline", "Enable CPU offloading"]
        )
        
        summary = self.collector.generate_diagnostic_summary(diagnostics)
        
        assert "WAN MODEL COMPATIBILITY DIAGNOSTIC REPORT" in summary
        assert "test_model" in summary
        assert "RTX 4090" in summary
        assert "wan_t2v" in summary
        assert "WanPipeline class not found" in summary
        assert "VRAM below recommended 12GB" in summary
        assert "Install WanPipeline" in summary
        assert "Enable CPU offloading" in summary

    def test_generate_diagnostics_missing_model_index(self):
        """Test diagnostic generation for missing model_index.json."""
        model_analysis = ModelAnalysis(
            architecture_detected="unknown",
            has_model_index=False,
            model_index_valid=False,
            components_found=[],
            vae_dimensions=None,
            custom_attributes=[],
            pipeline_class_detected=None,
            is_wan_architecture=False,
            component_details={}
        )
        
        pipeline_attempt = PipelineAttempt(
            attempted_pipeline="WanPipeline",
            pipeline_available=False,
            trust_remote_code=False,
            remote_code_fetched=False,
            load_success=False,
            error_type="no_attempt",
            error_message="No pipeline loading attempt was made",
            load_time_seconds=None
        )
        
        system_info = {"vram_available": 8192, "cuda_available": True}
        
        errors, warnings, recommendations = self.collector._generate_diagnostics(
            model_analysis, pipeline_attempt, system_info
        )
        
        assert "model_index.json file not found in model directory" in errors
        assert "Ensure the model directory contains a valid model_index.json file" in recommendations

    def test_generate_diagnostics_low_vram(self):
        """Test diagnostic generation for low VRAM scenarios."""
        model_analysis = ModelAnalysis(
            architecture_detected="wan_t2v",
            has_model_index=True,
            model_index_valid=True,
            components_found=["transformer", "vae"],
            vae_dimensions=3,
            custom_attributes=["boundary_ratio"],
            pipeline_class_detected="WanPipeline",
            is_wan_architecture=True,
            component_details={}
        )
        
        pipeline_attempt = PipelineAttempt(
            attempted_pipeline="WanPipeline",
            pipeline_available=True,
            trust_remote_code=True,
            remote_code_fetched=True,
            load_success=True,
            error_type=None,
            error_message=None,
            load_time_seconds=None
        )
        
        system_info = {"vram_available": 6144, "cuda_available": True}  # 6GB VRAM
        
        errors, warnings, recommendations = self.collector._generate_diagnostics(
            model_analysis, pipeline_attempt, system_info
        )
        
        assert any("VRAM" in warning and "insufficient" in warning for warning in warnings)
        assert any("CPU offloading" in rec for rec in recommendations)
        assert any("mixed precision" in rec for rec in recommendations)

    def test_generate_diagnostics_no_cuda(self):
        """Test diagnostic generation for systems without CUDA."""
        model_analysis = ModelAnalysis(
            architecture_detected="wan_t2v",
            has_model_index=True,
            model_index_valid=True,
            components_found=["transformer"],
            vae_dimensions=3,
            custom_attributes=[],
            pipeline_class_detected="WanPipeline",
            is_wan_architecture=True,
            component_details={}
        )
        
        pipeline_attempt = PipelineAttempt(
            attempted_pipeline="WanPipeline",
            pipeline_available=True,
            trust_remote_code=True,
            remote_code_fetched=True,
            load_success=True,
            error_type=None,
            error_message=None,
            load_time_seconds=None
        )
        
        system_info = {"cuda_available": False, "diffusers_version": "0.21.0"}
        
        errors, warnings, recommendations = self.collector._generate_diagnostics(
            model_analysis, pipeline_attempt, system_info
        )
        
        assert any("CUDA not available" in warning for warning in warnings)
        assert any("CUDA-compatible PyTorch" in rec for rec in recommendations)

    def test_generate_diagnostics_old_diffusers(self):
        """Test diagnostic generation for old diffusers version."""
        model_analysis = ModelAnalysis(
            architecture_detected="wan_t2v",
            has_model_index=True,
            model_index_valid=True,
            components_found=["transformer"],
            vae_dimensions=3,
            custom_attributes=[],
            pipeline_class_detected="WanPipeline",
            is_wan_architecture=True,
            component_details={}
        )
        
        pipeline_attempt = PipelineAttempt(
            attempted_pipeline="WanPipeline",
            pipeline_available=True,
            trust_remote_code=True,
            remote_code_fetched=True,
            load_success=True,
            error_type=None,
            error_message=None,
            load_time_seconds=None
        )
        
        system_info = {"cuda_available": True, "diffusers_version": "0.20.0"}
        
        errors, warnings, recommendations = self.collector._generate_diagnostics(
            model_analysis, pipeline_attempt, system_info
        )
        
        assert any("too old" in warning for warning in warnings)
        assert any("upgrade diffusers" in rec for rec in recommendations)

    def test_generate_diagnostics_missing_diffusers(self):
        """Test diagnostic generation for missing diffusers."""
        model_analysis = ModelAnalysis(
            architecture_detected="wan_t2v",
            has_model_index=True,
            model_index_valid=True,
            components_found=["transformer"],
            vae_dimensions=3,
            custom_attributes=[],
            pipeline_class_detected="WanPipeline",
            is_wan_architecture=True,
            component_details={}
        )
        
        pipeline_attempt = PipelineAttempt(
            attempted_pipeline="WanPipeline",
            pipeline_available=True,
            trust_remote_code=True,
            remote_code_fetched=True,
            load_success=True,
            error_type=None,
            error_message=None,
            load_time_seconds=None
        )
        
        system_info = {"cuda_available": True, "diffusers_version": None}
        
        errors, warnings, recommendations = self.collector._generate_diagnostics(
            model_analysis, pipeline_attempt, system_info
        )
        
        assert "Diffusers library not installed" in errors
        assert any("install diffusers" in rec for rec in recommendations)