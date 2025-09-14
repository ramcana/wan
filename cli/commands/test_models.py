#!/usr/bin/env python3
"""
Integration tests for Model Orchestrator CLI commands.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typer.testing import CliRunner

from cli.commands.models import app
from backend.core.model_orchestrator.model_ensurer import ModelStatus, ModelStatusInfo, VerificationResult
from backend.core.model_orchestrator.garbage_collector import GCResult, GCTrigger, DiskUsage
from backend.core.model_orchestrator.exceptions import ModelOrchestratorError, ErrorCode


@pytest.fixture
def runner():
    """Create CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_models_dir():
    """Create temporary models directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_orchestrator_components():
    """Mock orchestrator components."""
    mock_registry = Mock()
    mock_ensurer = Mock()
    mock_gc = Mock()
    
    # Mock registry methods
    mock_registry.list_models.return_value = ["t2v-A14B@2.2.0", "i2v-A14B@2.2.0", "ti2v-5b@2.2.0"]
    
    mock_spec = Mock()
    mock_spec.model_id = "t2v-A14B@2.2.0"
    mock_spec.description = "WAN2.2 Text-to-Video A14B Model"
    mock_spec.version = "2.2.0"
    mock_spec.variants = ["fp16", "bf16"]
    mock_spec.default_variant = "fp16"
    mock_spec.files = [Mock(path="model.safetensors", size=1000000000)]
    
    # Mock component deduplicator
    mock_deduplicator = Mock()
    mock_ensurer.component_deduplicator = mock_deduplicator
    mock_ensurer.enable_deduplication = True
    mock_registry.spec.return_value = mock_spec
    
    # Mock garbage collector
    mock_gc.collect.return_value = GCResult(
        trigger=GCTrigger.MANUAL,
        dry_run=True,
        models_removed=["old-model@fp16"],
        models_preserved=["t2v-A14B@2.2.0", "i2v-A14B@2.2.0"],
        bytes_reclaimed=500000000,
        bytes_preserved=2000000000,
        errors=[],
        duration_seconds=1.5
    )
    
    mock_gc.get_disk_usage.return_value = DiskUsage(
        total_bytes=1000000000000,  # 1TB
        used_bytes=800000000000,    # 800GB
        free_bytes=200000000000,    # 200GB
        models_bytes=50000000000,   # 50GB
        usage_percentage=80.0
    )
    
    mock_gc.estimate_reclaimable_space.return_value = 500000000  # 500MB
    mock_gc.is_pinned.return_value = False
    
    return mock_registry, mock_ensurer, mock_gc


class TestStatusCommand:
    """Test the status command."""
    
    @patch('cli.commands.models._get_orchestrator_components')
    def test_status_all_models(self, mock_get_components, runner, mock_orchestrator_components):
        """Test status command for all models."""
        mock_registry, mock_ensurer, mock_gc = mock_orchestrator_components
        mock_get_components.return_value = (mock_registry, mock_ensurer, mock_gc)
        
        # Mock status responses
        mock_ensurer.status.side_effect = [
            ModelStatusInfo(
                status=ModelStatus.COMPLETE,
                local_path="/models/t2v-A14B@2.2.0",
                missing_files=[],
                bytes_needed=0
            ),
            ModelStatusInfo(
                status=ModelStatus.NOT_PRESENT,
                local_path="/models/i2v-A14B@2.2.0",
                missing_files=["model.safetensors"],
                bytes_needed=1000000000
            ),
            ModelStatusInfo(
                status=ModelStatus.PARTIAL,
                local_path="/models/ti2v-5b@2.2.0",
                missing_files=["config.json"],
                bytes_needed=1024
            )
        ]
        
        result = runner.invoke(app, ["status"])
        
        assert result.exit_code == 0
        assert "WAN2.2 Model Status" in result.stdout
        assert "COMPLETE" in result.stdout
        assert "NOT_PRESENT" in result.stdout
        assert "PARTIAL" in result.stdout
        assert "Summary: 1/3 models complete" in result.stdout
    
    @patch('cli.commands.models._get_orchestrator_components')
    def test_status_specific_model(self, mock_get_components, runner, mock_orchestrator_components):
        """Test status command for specific model."""
        mock_registry, mock_ensurer, mock_gc = mock_orchestrator_components
        mock_get_components.return_value = (mock_registry, mock_ensurer, mock_gc)
        
        mock_ensurer.status.return_value = ModelStatusInfo(
            status=ModelStatus.COMPLETE,
            local_path="/models/t2v-A14B@2.2.0",
            missing_files=[],
            bytes_needed=0
        )
        
        result = runner.invoke(app, ["status", "--model", "t2v-A14B@2.2.0"])
        
        assert result.exit_code == 0
        assert "t2v-A14B@2.2.0" in result.stdout
        assert "COMPLETE" in result.stdout
        mock_ensurer.status.assert_called_once_with("t2v-A14B@2.2.0", None)
    
    @patch('cli.commands.models._get_orchestrator_components')
    def test_status_json_output(self, mock_get_components, runner, mock_orchestrator_components):
        """Test status command with JSON output."""
        mock_registry, mock_ensurer = mock_orchestrator_components
        mock_get_components.return_value = (mock_registry, mock_ensurer)
        
        mock_ensurer.status.return_value = ModelStatusInfo(
            status=ModelStatus.COMPLETE,
            local_path="/models/t2v-A14B@2.2.0",
            missing_files=[],
            bytes_needed=0
        )
        
        result = runner.invoke(app, ["status", "--model", "t2v-A14B@2.2.0", "--json"])
        
        assert result.exit_code == 0
        
        # Parse JSON output
        output = json.loads(result.stdout)
        assert "t2v-A14B@2.2.0" in output
        assert output["t2v-A14B@2.2.0"]["status"] == "COMPLETE"
        assert output["t2v-A14B@2.2.0"]["bytes_needed"] == 0
    
    @patch('cli.commands.models._get_orchestrator_components')
    def test_status_with_error(self, mock_get_components, runner, mock_orchestrator_components):
        """Test status command when model has error."""
        mock_registry, mock_ensurer = mock_orchestrator_components
        mock_get_components.return_value = (mock_registry, mock_ensurer)
        
        mock_ensurer.status.side_effect = Exception("Model not found")
        
        result = runner.invoke(app, ["status", "--model", "invalid-model"])
        
        assert result.exit_code == 0  # Status command doesn't fail on individual model errors
        assert "ERROR" in result.stdout


class TestEnsureCommand:
    """Test the ensure command."""
    
    @patch('cli.commands.models._get_orchestrator_components')
    def test_ensure_specific_model(self, mock_get_components, runner, mock_orchestrator_components):
        """Test ensure command for specific model."""
        mock_registry, mock_ensurer = mock_orchestrator_components
        mock_get_components.return_value = (mock_registry, mock_ensurer)
        
        mock_ensurer.ensure.return_value = "/models/t2v-A14B@2.2.0"
        
        result = runner.invoke(app, ["ensure", "--only", "t2v-A14B@2.2.0"])
        
        assert result.exit_code == 0
        assert "Model t2v-A14B@2.2.0 ready" in result.stdout
        # Verify the call was made with correct parameters (progress_callback will be a function)
        mock_ensurer.ensure.assert_called_once()
        args, kwargs = mock_ensurer.ensure.call_args
        assert kwargs["model_id"] == "t2v-A14B@2.2.0"
        assert kwargs["variant"] is None
        assert kwargs["force_redownload"] is False
        assert kwargs["progress_callback"] is not None  # Progress callback is set
    
    @patch('cli.commands.models._get_orchestrator_components')
    def test_ensure_all_models(self, mock_get_components, runner, mock_orchestrator_components):
        """Test ensure command for all models."""
        mock_registry, mock_ensurer = mock_orchestrator_components
        mock_get_components.return_value = (mock_registry, mock_ensurer)
        
        mock_ensurer.ensure.side_effect = [
            "/models/t2v-A14B@2.2.0",
            "/models/i2v-A14B@2.2.0",
            "/models/ti2v-5b@2.2.0"
        ]
        
        result = runner.invoke(app, ["ensure", "--all"])
        
        assert result.exit_code == 0
        assert "Successfully ensured 3 models" in result.stdout
        assert mock_ensurer.ensure.call_count == 3
    
    @patch('cli.commands.models._get_orchestrator_components')
    def test_ensure_with_force(self, mock_get_components, runner, mock_orchestrator_components):
        """Test ensure command with force flag."""
        mock_registry, mock_ensurer = mock_orchestrator_components
        mock_get_components.return_value = (mock_registry, mock_ensurer)
        
        mock_ensurer.ensure.return_value = "/models/t2v-A14B@2.2.0"
        
        result = runner.invoke(app, ["ensure", "--only", "t2v-A14B@2.2.0", "--force"])
        
        assert result.exit_code == 0
        mock_ensurer.ensure.assert_called_once()
        args, kwargs = mock_ensurer.ensure.call_args
        assert kwargs["force_redownload"] is True
    
    @patch('cli.commands.models._get_orchestrator_components')
    def test_ensure_json_output(self, mock_get_components, runner, mock_orchestrator_components):
        """Test ensure command with JSON output."""
        mock_registry, mock_ensurer = mock_orchestrator_components
        mock_get_components.return_value = (mock_registry, mock_ensurer)
        
        mock_ensurer.ensure.return_value = "/models/t2v-A14B@2.2.0"
        
        result = runner.invoke(app, ["ensure", "--only", "t2v-A14B@2.2.0", "--json", "--quiet"])
        
        assert result.exit_code == 0
        
        # Parse JSON output
        output = json.loads(result.stdout)
        assert "t2v-A14B@2.2.0" in output
        assert output["t2v-A14B@2.2.0"]["success"] is True
        assert output["t2v-A14B@2.2.0"]["local_path"] == "/models/t2v-A14B@2.2.0"
    
    @patch('cli.commands.models._get_orchestrator_components')
    def test_ensure_with_error(self, mock_get_components, runner, mock_orchestrator_components):
        """Test ensure command when download fails."""
        mock_registry, mock_ensurer = mock_orchestrator_components
        mock_get_components.return_value = (mock_registry, mock_ensurer)
        
        mock_ensurer.ensure.side_effect = ModelOrchestratorError(
            "Download failed", ErrorCode.SOURCE_UNAVAILABLE
        )
        
        result = runner.invoke(app, ["ensure", "--only", "t2v-A14B@2.2.0"])
        
        assert result.exit_code == 1
        assert "Failed to ensure" in result.stdout
    
    def test_ensure_no_model_specified(self, runner):
        """Test ensure command without specifying model or --all."""
        result = runner.invoke(app, ["ensure"])
        
        assert result.exit_code == 1
        assert "Must specify --only {model_id} or --all" in result.stdout


class TestVerifyCommand:
    """Test the verify command."""
    
    @patch('cli.commands.models._get_orchestrator_components')
    def test_verify_success(self, mock_get_components, runner, mock_orchestrator_components):
        """Test verify command with successful verification."""
        mock_registry, mock_ensurer = mock_orchestrator_components
        mock_get_components.return_value = (mock_registry, mock_ensurer)
        
        mock_ensurer.verify_integrity.return_value = VerificationResult(
            success=True,
            verified_files=["model.safetensors", "config.json"],
            failed_files=[],
            missing_files=[]
        )
        
        result = runner.invoke(app, ["verify", "t2v-A14B@2.2.0"])
        
        assert result.exit_code == 0
        assert "integrity verified" in result.stdout
        assert "Verified files: 2" in result.stdout
        mock_ensurer.verify_integrity.assert_called_once_with("t2v-A14B@2.2.0", None)
    
    @patch('cli.commands.models._get_orchestrator_components')
    def test_verify_failure(self, mock_get_components, runner, mock_orchestrator_components):
        """Test verify command with failed verification."""
        mock_registry, mock_ensurer = mock_orchestrator_components
        mock_get_components.return_value = (mock_registry, mock_ensurer)
        
        mock_ensurer.verify_integrity.return_value = VerificationResult(
            success=False,
            verified_files=["config.json"],
            failed_files=["model.safetensors"],
            missing_files=["tokenizer.json"],
            error_message="Checksum mismatch"
        )
        
        result = runner.invoke(app, ["verify", "t2v-A14B@2.2.0"])
        
        assert result.exit_code == 1
        assert "integrity check failed" in result.stdout
        assert "Failed files: 1" in result.stdout
        assert "Missing files: 1" in result.stdout
    
    @patch('cli.commands.models._get_orchestrator_components')
    def test_verify_json_output(self, mock_get_components, runner, mock_orchestrator_components):
        """Test verify command with JSON output."""
        mock_registry, mock_ensurer = mock_orchestrator_components
        mock_get_components.return_value = (mock_registry, mock_ensurer)
        
        mock_ensurer.verify_integrity.return_value = VerificationResult(
            success=True,
            verified_files=["model.safetensors"],
            failed_files=[],
            missing_files=[]
        )
        
        result = runner.invoke(app, ["verify", "t2v-A14B@2.2.0", "--json"])
        
        assert result.exit_code == 0
        
        # Parse JSON output
        output = json.loads(result.stdout)
        assert output["model_id"] == "t2v-A14B@2.2.0"
        assert output["success"] is True
        assert output["verified_files"] == ["model.safetensors"]


class TestListCommand:
    """Test the list command."""
    
    @patch('cli.commands.models._get_orchestrator_components')
    def test_list_models(self, mock_get_components, runner, mock_orchestrator_components):
        """Test list command."""
        mock_registry, mock_ensurer = mock_orchestrator_components
        mock_get_components.return_value = (mock_registry, mock_ensurer)
        
        result = runner.invoke(app, ["list"])
        
        assert result.exit_code == 0
        assert "Available WAN2.2 Models" in result.stdout
        assert "t2v-A14B@2.2.0" in result.stdout
        assert "i2v-A14B@2.2.0" in result.stdout
        assert "ti2v-5b@2.2.0" in result.stdout
        assert "Total models: 3" in result.stdout
    
    @patch('cli.commands.models._get_orchestrator_components')
    def test_list_models_detailed(self, mock_get_components, runner, mock_orchestrator_components):
        """Test list command with detailed output."""
        mock_registry, mock_ensurer = mock_orchestrator_components
        mock_get_components.return_value = (mock_registry, mock_ensurer)
        
        result = runner.invoke(app, ["list", "--detailed"])
        
        assert result.exit_code == 0
        # Check that the description appears in the output (may be wrapped)
        assert "Text-to-Video" in result.stdout
        assert "A14B Model" in result.stdout
        assert "953.7 MB" in result.stdout  # Size formatting (1000000000 bytes)
    
    @patch('cli.commands.models._get_orchestrator_components')
    def test_list_models_json(self, mock_get_components, runner, mock_orchestrator_components):
        """Test list command with JSON output."""
        mock_registry, mock_ensurer = mock_orchestrator_components
        mock_get_components.return_value = (mock_registry, mock_ensurer)
        
        result = runner.invoke(app, ["list", "--json"])
        
        assert result.exit_code == 0
        
        # Parse JSON output
        output = json.loads(result.stdout)
        assert "t2v-A14B@2.2.0" in output
        assert output["t2v-A14B@2.2.0"]["version"] == "2.2.0"
        assert output["t2v-A14B@2.2.0"]["variants"] == ["fp16", "bf16"]


class TestCLIIntegration:
    """Test CLI integration scenarios."""
    
    @patch('cli.commands.models._get_orchestrator_components')
    def test_workflow_status_ensure_verify(self, mock_get_components, runner, mock_orchestrator_components):
        """Test complete workflow: status -> ensure -> verify."""
        mock_registry, mock_ensurer = mock_orchestrator_components
        mock_get_components.return_value = (mock_registry, mock_ensurer)
        
        # Step 1: Check status (not present)
        mock_ensurer.status.return_value = ModelStatusInfo(
            status=ModelStatus.NOT_PRESENT,
            local_path="/models/t2v-A14B@2.2.0",
            missing_files=["model.safetensors"],
            bytes_needed=1000000000
        )
        
        status_result = runner.invoke(app, ["status", "--model", "t2v-A14B@2.2.0"])
        assert status_result.exit_code == 0
        assert "NOT_PRESENT" in status_result.stdout
        
        # Step 2: Ensure model
        mock_ensurer.ensure.return_value = "/models/t2v-A14B@2.2.0"
        
        ensure_result = runner.invoke(app, ["ensure", "--only", "t2v-A14B@2.2.0"])
        assert ensure_result.exit_code == 0
        assert "ready" in ensure_result.stdout
        
        # Step 3: Verify integrity
        mock_ensurer.verify_integrity.return_value = VerificationResult(
            success=True,
            verified_files=["model.safetensors"],
            failed_files=[],
            missing_files=[]
        )
        
        verify_result = runner.invoke(app, ["verify", "t2v-A14B@2.2.0"])
        assert verify_result.exit_code == 0
        assert "verified" in verify_result.stdout
    
    @patch('cli.commands.models._get_orchestrator_components')
    def test_error_handling_initialization(self, mock_get_components, runner):
        """Test error handling when orchestrator initialization fails."""
        mock_get_components.side_effect = Exception("Configuration error")
        
        result = runner.invoke(app, ["status"])
        
        assert result.exit_code == 1
        assert "Error checking model status" in result.stdout
    
    def test_help_output(self, runner):
        """Test help output for all commands."""
        # Main help
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Model Orchestrator - Unified model management for WAN2.2" in result.stdout
        
        # Command-specific help
        for command in ["status", "ensure", "verify", "list"]:
            result = runner.invoke(app, [command, "--help"])
            assert result.exit_code == 0
            assert "Examples:" in result.stdout


class TestProgressIndicators:
    """Test progress indicators during downloads."""
    
    @patch('cli.commands.models._get_orchestrator_components')
    def test_ensure_with_progress_callback(self, mock_get_components, runner, mock_orchestrator_components):
        """Test that progress callback is properly set up during ensure."""
        mock_registry, mock_ensurer = mock_orchestrator_components
        mock_get_components.return_value = (mock_registry, mock_ensurer)
        
        # Mock ensure to call progress callback
        def mock_ensure(*args, **kwargs):
            progress_callback = kwargs.get('progress_callback')
            if progress_callback:
                progress_callback(50, 100)  # 50% progress
                progress_callback(100, 100)  # Complete
            return "/models/t2v-A14B@2.2.0"
        
        mock_ensurer.ensure.side_effect = mock_ensure
        
        result = runner.invoke(app, ["ensure", "--only", "t2v-A14B@2.2.0"])
        
        assert result.exit_code == 0
        # Progress callback should have been called
        mock_ensurer.ensure.assert_called_once()
        args, kwargs = mock_ensurer.ensure.call_args
        assert 'progress_callback' in kwargs
        assert kwargs['progress_callback'] is not None


class TestGarbageCollectionCommand:
    """Test the garbage collection command."""
    
    @patch('cli.commands.models._get_orchestrator_components')
    def test_gc_dry_run(self, mock_get_components, runner, mock_orchestrator_components):
        """Test garbage collection dry run."""
        mock_registry, mock_ensurer, mock_gc = mock_orchestrator_components
        mock_get_components.return_value = (mock_registry, mock_ensurer, mock_gc)
        
        result = runner.invoke(app, ["gc"])
        
        assert result.exit_code == 0
        assert "Running garbage collection in dry-run mode" in result.stdout
        assert "Models would be removed:" in result.stdout
        assert "old-model@fp16" in result.stdout
        assert "476.8 MB reclaimed" in result.stdout
        mock_gc.collect.assert_called_once_with(dry_run=True, trigger=GCTrigger.MANUAL)
    
    @patch('cli.commands.models._get_orchestrator_components')
    def test_gc_execute(self, mock_get_components, runner, mock_orchestrator_components):
        """Test garbage collection execution."""
        mock_registry, mock_ensurer, mock_gc = mock_orchestrator_components
        mock_get_components.return_value = (mock_registry, mock_ensurer, mock_gc)
        
        # Update mock to return execute mode result
        mock_gc.collect.return_value = GCResult(
            trigger=GCTrigger.MANUAL,
            dry_run=False,
            models_removed=["old-model@fp16"],
            models_preserved=["t2v-A14B@2.2.0"],
            bytes_reclaimed=500000000,
            bytes_preserved=1000000000,
            errors=[],
            duration_seconds=2.1
        )
        
        result = runner.invoke(app, ["gc", "--execute"])
        
        assert result.exit_code == 0
        assert "Running garbage collection - models will be removed!" in result.stdout
        assert "Models removed:" in result.stdout
        assert "old-model@fp16" in result.stdout
        mock_gc.collect.assert_called_once_with(dry_run=False, trigger=GCTrigger.MANUAL)
    
    @patch('cli.commands.models._get_orchestrator_components')
    def test_gc_with_size_limit(self, mock_get_components, runner, mock_orchestrator_components):
        """Test garbage collection with size limit."""
        mock_registry, mock_ensurer, mock_gc = mock_orchestrator_components
        mock_get_components.return_value = (mock_registry, mock_ensurer, mock_gc)
        
        result = runner.invoke(app, ["gc", "--max-size", "10GB", "--execute"])
        
        if result.exit_code != 0:
            print(f"Error output: {result.stdout}")
            print(f"Exception: {result.exception}")
        
        assert result.exit_code == 0
        # Check that config was updated
        assert mock_gc.config.max_total_size == 10 * 1024**3  # 10GB in bytes
    
    @patch('cli.commands.models._get_orchestrator_components')
    def test_gc_with_age_limit(self, mock_get_components, runner, mock_orchestrator_components):
        """Test garbage collection with age limit."""
        mock_registry, mock_ensurer, mock_gc = mock_orchestrator_components
        mock_get_components.return_value = (mock_registry, mock_ensurer, mock_gc)
        
        result = runner.invoke(app, ["gc", "--max-age", "7d", "--execute"])
        
        assert result.exit_code == 0
        # Check that config was updated
        assert mock_gc.config.max_model_age == 7 * 24 * 3600  # 7 days in seconds
    
    @patch('cli.commands.models._get_orchestrator_components')
    def test_gc_json_output(self, mock_get_components, runner, mock_orchestrator_components):
        """Test garbage collection with JSON output."""
        mock_registry, mock_ensurer, mock_gc = mock_orchestrator_components
        mock_get_components.return_value = (mock_registry, mock_ensurer, mock_gc)
        
        result = runner.invoke(app, ["gc", "--json", "--quiet"])
        
        assert result.exit_code == 0
        
        # Parse JSON output
        output = json.loads(result.stdout)
        assert output["dry_run"] is True
        assert output["trigger"] == "manual"
        assert "old-model@fp16" in output["models_removed"]
        assert output["bytes_reclaimed"] == 500000000
    
    @patch('cli.commands.models._get_orchestrator_components')
    def test_gc_with_errors(self, mock_get_components, runner, mock_orchestrator_components):
        """Test garbage collection with errors."""
        mock_registry, mock_ensurer, mock_gc = mock_orchestrator_components
        mock_get_components.return_value = (mock_registry, mock_ensurer, mock_gc)
        
        # Mock GC result with errors
        mock_gc.collect.return_value = GCResult(
            trigger=GCTrigger.MANUAL,
            dry_run=False,
            models_removed=[],
            models_preserved=["t2v-A14B@2.2.0"],
            bytes_reclaimed=0,
            bytes_preserved=1000000000,
            errors=["Failed to remove model: Permission denied"],
            duration_seconds=0.5
        )
        
        result = runner.invoke(app, ["gc", "--execute"])
        
        assert result.exit_code == 1
        assert "Errors encountered:" in result.stdout
        assert "Permission denied" in result.stdout
    
    def test_gc_invalid_size_format(self, runner):
        """Test garbage collection with invalid size format."""
        result = runner.invoke(app, ["gc", "--max-size", "invalid"])
        
        assert result.exit_code == 1
        assert "Invalid parameter" in result.stdout
    
    def test_gc_invalid_age_format(self, runner):
        """Test garbage collection with invalid age format."""
        result = runner.invoke(app, ["gc", "--max-age", "invalid"])
        
        assert result.exit_code == 1
        assert "Invalid parameter" in result.stdout


class TestPinCommand:
    """Test the pin command."""
    
    @patch('cli.commands.models._get_orchestrator_components')
    def test_pin_model(self, mock_get_components, runner, mock_orchestrator_components):
        """Test pinning a model."""
        mock_registry, mock_ensurer, mock_gc = mock_orchestrator_components
        mock_get_components.return_value = (mock_registry, mock_ensurer, mock_gc)
        
        result = runner.invoke(app, ["pin", "t2v-A14B@2.2.0"])
        
        assert result.exit_code == 0
        assert "Model t2v-A14B@2.2.0 pinned" in result.stdout
        mock_gc.pin_model.assert_called_once_with("t2v-A14B@2.2.0", None)
    
    @patch('cli.commands.models._get_orchestrator_components')
    def test_unpin_model(self, mock_get_components, runner, mock_orchestrator_components):
        """Test unpinning a model."""
        mock_registry, mock_ensurer, mock_gc = mock_orchestrator_components
        mock_get_components.return_value = (mock_registry, mock_ensurer, mock_gc)
        
        result = runner.invoke(app, ["pin", "t2v-A14B@2.2.0", "--unpin"])
        
        assert result.exit_code == 0
        assert "Model t2v-A14B@2.2.0 unpinned" in result.stdout
        mock_gc.unpin_model.assert_called_once_with("t2v-A14B@2.2.0", None)
    
    @patch('cli.commands.models._get_orchestrator_components')
    def test_pin_model_with_variant(self, mock_get_components, runner, mock_orchestrator_components):
        """Test pinning a model with variant."""
        mock_registry, mock_ensurer, mock_gc = mock_orchestrator_components
        mock_get_components.return_value = (mock_registry, mock_ensurer, mock_gc)
        
        result = runner.invoke(app, ["pin", "t2v-A14B@2.2.0", "--variant", "fp16"])
        
        assert result.exit_code == 0
        assert "Model t2v-A14B@2.2.0 (variant: fp16) pinned" in result.stdout
        mock_gc.pin_model.assert_called_once_with("t2v-A14B@2.2.0", "fp16")
    
    @patch('cli.commands.models._get_orchestrator_components')
    def test_pin_json_output(self, mock_get_components, runner, mock_orchestrator_components):
        """Test pin command with JSON output."""
        mock_registry, mock_ensurer, mock_gc = mock_orchestrator_components
        mock_get_components.return_value = (mock_registry, mock_ensurer, mock_gc)
        
        mock_gc.is_pinned.return_value = True
        
        result = runner.invoke(app, ["pin", "t2v-A14B@2.2.0", "--json"])
        
        assert result.exit_code == 0
        
        # Parse JSON output
        output = json.loads(result.stdout)
        assert output["model_id"] == "t2v-A14B@2.2.0"
        assert output["action"] == "pinned"
        assert output["is_pinned"] is True


class TestDiskUsageCommand:
    """Test the disk usage command."""
    
    @patch('cli.commands.models._get_orchestrator_components')
    def test_disk_usage(self, mock_get_components, runner, mock_orchestrator_components):
        """Test disk usage command."""
        mock_registry, mock_ensurer, mock_gc = mock_orchestrator_components
        mock_get_components.return_value = (mock_registry, mock_ensurer, mock_gc)
        
        result = runner.invoke(app, ["disk-usage"])
        
        assert result.exit_code == 0
        assert "Disk Usage Information" in result.stdout
        assert "Total Space" in result.stdout
        assert "931.3 GB" in result.stdout  # 1TB formatted
        assert "46.6 GB" in result.stdout   # 50GB models space formatted
        assert "80.0%" in result.stdout     # Usage percentage
    
    @patch('cli.commands.models._get_orchestrator_components')
    def test_disk_usage_json(self, mock_get_components, runner, mock_orchestrator_components):
        """Test disk usage command with JSON output."""
        mock_registry, mock_ensurer, mock_gc = mock_orchestrator_components
        mock_get_components.return_value = (mock_registry, mock_ensurer, mock_gc)
        
        result = runner.invoke(app, ["disk-usage", "--json"])
        
        assert result.exit_code == 0
        
        # Parse JSON output
        output = json.loads(result.stdout)
        assert output["total_bytes"] == 1000000000000
        assert output["used_bytes"] == 800000000000
        assert output["free_bytes"] == 200000000000
        assert output["models_bytes"] == 50000000000
        assert output["usage_percentage"] == 80.0
        assert output["reclaimable_bytes"] == 500000000
    
    @patch('cli.commands.models._get_orchestrator_components')
    def test_disk_usage_with_warnings(self, mock_get_components, runner, mock_orchestrator_components):
        """Test disk usage command with high usage warnings."""
        mock_registry, mock_ensurer, mock_gc = mock_orchestrator_components
        mock_get_components.return_value = (mock_registry, mock_ensurer, mock_gc)
        
        # Mock high disk usage
        mock_gc.get_disk_usage.return_value = DiskUsage(
            total_bytes=1000000000000,
            used_bytes=950000000000,  # 95% used
            free_bytes=50000000000,
            models_bytes=50000000000,
            usage_percentage=95.0
        )
        
        result = runner.invoke(app, ["disk-usage"])
        
        assert result.exit_code == 0
        assert "Warning: Disk usage is high (95.0%)" in result.stdout
        assert "Consider running garbage collection" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__])


class TestComponentDeduplicationCommand:
    """Test the component deduplication commands."""

    @patch('cli.commands.models._get_orchestrator_components')
    def test_deduplicate_single_models(self, mock_get_components, runner, mock_orchestrator_components):
        """Test deduplication of individual models."""
        mock_registry, mock_ensurer, mock_gc = mock_orchestrator_components
        mock_get_components.return_value = (mock_registry, mock_ensurer, mock_gc)
        
        # Mock deduplication result
        from backend.core.model_orchestrator.component_deduplicator import DeduplicationResult
        mock_result = DeduplicationResult(
            total_files_processed=10,
            duplicates_found=3,
            bytes_saved=1024*1024*100,  # 100MB
            links_created=3,
            errors=[],
            processing_time=2.5
        )
        
        mock_ensurer.component_deduplicator.deduplicate_model.return_value = mock_result
        mock_ensurer.resolver.local_dir.return_value = "/models/t2v-A14B@2.2.0"
        
        # Mock Path.exists()
        with patch('pathlib.Path.exists', return_value=True):
            result = runner.invoke(app, ["deduplicate", "--model", "t2v-A14B@2.2.0"])
        
        assert result.exit_code == 0
        assert "Deduplication Results" in result.stdout
        assert "100.0 MB" in result.stdout  # Formatted bytes
        assert "3" in result.stdout  # Links created
        
        # Verify deduplication was called
        mock_ensurer.component_deduplicator.deduplicate_model.assert_called_once()

    @patch('cli.commands.models._get_orchestrator_components')
    def test_deduplicate_cross_model(self, mock_get_components, runner, mock_orchestrator_components):
        """Test cross-model deduplication."""
        mock_registry, mock_ensurer, mock_gc = mock_orchestrator_components
        mock_get_components.return_value = (mock_registry, mock_ensurer, mock_gc)
        
        # Mock cross-model deduplication result
        mock_result = {
            "files_processed": 50,
            "duplicates_found": 15,
            "bytes_saved": 1024*1024*500,  # 500MB
            "links_created": 15,
            "processing_time": 10.2,
            "errors": []
        }
        
        mock_ensurer.deduplicate_across_models.return_value = mock_result
        
        result = runner.invoke(app, ["deduplicate", "--cross-model"])
        
        assert result.exit_code == 0
        assert "cross-model deduplication" in result.stdout
        assert "500.0 MB" in result.stdout  # Total space saved
        assert "15" in result.stdout  # Total links created
        
        # Verify cross-model deduplication was called
        mock_ensurer.deduplicate_across_models.assert_called_once()

    @patch('cli.commands.models._get_orchestrator_components')
    def test_deduplicate_dry_run(self, mock_get_components, runner, mock_orchestrator_components):
        """Test deduplication dry run."""
        mock_registry, mock_ensurer, mock_gc = mock_orchestrator_components
        mock_get_components.return_value = (mock_registry, mock_ensurer, mock_gc)
        
        result = runner.invoke(app, ["deduplicate", "--dry-run"])
        
        assert result.exit_code == 0
        assert "DRY RUN" in result.stdout
        assert "No changes were made" in result.stdout

    @patch('cli.commands.models._get_orchestrator_components')
    def test_deduplicate_json_output(self, mock_get_components, runner, mock_orchestrator_components):
        """Test deduplication with JSON output."""
        mock_registry, mock_ensurer, mock_gc = mock_orchestrator_components
        mock_get_components.return_value = (mock_registry, mock_ensurer, mock_gc)
        
        # Mock cross-model deduplication result
        mock_result = {
            "files_processed": 30,
            "duplicates_found": 8,
            "bytes_saved": 1024*1024*200,  # 200MB
            "links_created": 8,
            "processing_time": 5.1,
            "errors": []
        }
        
        mock_ensurer.deduplicate_across_models.return_value = mock_result
        
        result = runner.invoke(app, ["deduplicate", "--cross-model", "--json"])
        
        assert result.exit_code == 0
        
        # Parse JSON output
        output_data = json.loads(result.stdout)
        assert "results" in output_data
        assert "summary" in output_data
        assert output_data["summary"]["total_bytes_saved"] == 1024*1024*200
        assert output_data["summary"]["cross_model"] is True

    @patch('cli.commands.models._get_orchestrator_components')
    def test_deduplicate_disabled(self, mock_get_components, runner, mock_orchestrator_components):
        """Test deduplication when disabled."""
        mock_registry, mock_ensurer, mock_gc = mock_orchestrator_components
        mock_get_components.return_value = (mock_registry, mock_ensurer, mock_gc)
        
        # Mock deduplication as disabled
        mock_ensurer.component_deduplicator = None
        
        result = runner.invoke(app, ["deduplicate"])
        
        assert result.exit_code == 1
        assert "Component deduplication is not enabled" in result.stdout

    @patch('cli.commands.models._get_orchestrator_components')
    def test_deduplicate_with_errors(self, mock_get_components, runner, mock_orchestrator_components):
        """Test deduplication with errors."""
        mock_registry, mock_ensurer, mock_gc = mock_orchestrator_components
        mock_get_components.return_value = (mock_registry, mock_ensurer, mock_gc)
        
        # Mock deduplication failure
        mock_ensurer.deduplicate_across_models.side_effect = Exception("Deduplication failed")
        
        result = runner.invoke(app, ["deduplicate", "--cross-model"])
        
        assert result.exit_code == 1
        assert "Deduplication failed" in result.stdout

    @patch('cli.commands.models._get_orchestrator_components')
    def test_component_stats(self, mock_get_components, runner, mock_orchestrator_components):
        """Test component statistics command."""
        mock_registry, mock_ensurer, mock_gc = mock_orchestrator_components
        mock_get_components.return_value = (mock_registry, mock_ensurer, mock_gc)
        
        # Mock component stats
        mock_stats = {
            "total_components": 15,
            "total_size_bytes": 1024*1024*1024*2,  # 2GB
            "total_references": 45,
            "component_types": {
                "tokenizer": 3,
                "text_encoder": 2,
                "image_encoder": 1
            },
            "components_root": "/models/components",
            "supports_hardlinks": True,
            "supports_symlinks": True
        }
        
        mock_ensurer.get_component_stats.return_value = mock_stats
        
        result = runner.invoke(app, ["component-stats"])
        
        assert result.exit_code == 0
        assert "Component Deduplication Statistics" in result.stdout
        assert "15" in result.stdout  # Total components
        assert "2.0 GB" in result.stdout  # Total size
        assert "tokenizer" in result.stdout
        assert "text_encoder" in result.stdout

    @patch('cli.commands.models._get_orchestrator_components')
    def test_component_stats_json(self, mock_get_components, runner, mock_orchestrator_components):
        """Test component statistics with JSON output."""
        mock_registry, mock_ensurer, mock_gc = mock_orchestrator_components
        mock_get_components.return_value = (mock_registry, mock_ensurer, mock_gc)
        
        # Mock component stats
        mock_stats = {
            "total_components": 10,
            "total_size_bytes": 1024*1024*500,  # 500MB
            "total_references": 30,
            "component_types": {"tokenizer": 2, "text_encoder": 1},
            "components_root": "/models/components",
            "supports_hardlinks": False,
            "supports_symlinks": True
        }
        
        mock_ensurer.get_component_stats.return_value = mock_stats
        
        result = runner.invoke(app, ["component-stats", "--json"])
        
        assert result.exit_code == 0
        
        # Parse JSON output
        output_data = json.loads(result.stdout)
        assert output_data["total_components"] == 10
        assert output_data["total_size_bytes"] == 1024*1024*500
        assert output_data["supports_hardlinks"] is False
        assert output_data["supports_symlinks"] is True

    @patch('cli.commands.models._get_orchestrator_components')
    def test_component_stats_disabled(self, mock_get_components, runner, mock_orchestrator_components):
        """Test component statistics when deduplication is disabled."""
        mock_registry, mock_ensurer, mock_gc = mock_orchestrator_components
        mock_get_components.return_value = (mock_registry, mock_ensurer, mock_gc)
        
        # Mock deduplication as disabled
        mock_ensurer.component_deduplicator = None
        
        result = runner.invoke(app, ["component-stats"])
        
        assert result.exit_code == 1
        assert "Component deduplication is not enabled" in result.stdout

    @patch('cli.commands.models._get_orchestrator_components')
    def test_component_stats_no_data(self, mock_get_components, runner, mock_orchestrator_components):
        """Test component statistics when no data is available."""
        mock_registry, mock_ensurer, mock_gc = mock_orchestrator_components
        mock_get_components.return_value = (mock_registry, mock_ensurer, mock_gc)
        
        # Mock no stats available
        mock_ensurer.get_component_stats.return_value = None
        
        result = runner.invoke(app, ["component-stats"])
        
        assert result.exit_code == 0
        assert "No component statistics available" in result.stdout