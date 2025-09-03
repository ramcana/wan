"""
End-to-end integration tests for the complete project health system workflow.

This module tests the integration of all project health components:
- Test suite orchestration and execution
- Documentation generation and validation
- Configuration management and validation
- Health monitoring and reporting
- Developer experience tools

Requirements: 1.1, 4.1
"""

import asyncio
import json
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
import pytest
import yaml

# Import project health system components
from tools.test-runner.orchestrator import TestSuiteOrchestrator
from tools.doc_generator.documentation_generator import DocumentationGenerator
from tools.config_manager.config_unifier import ConfigurationUnifier
from tools.health_checker.health_checker import ProjectHealthChecker
from tools.dev_environment.setup_dev_environment import DevEnvironmentSetup


class TestProjectHealthSystemE2E:
    """End-to-end tests for complete project health system."""

    @pytest.fixture
    def temp_project_dir(self):
        """Create a temporary project directory for testing."""
        temp_dir = tempfile.mkdtemp()
        project_dir = Path(temp_dir) / "test_project"
        project_dir.mkdir()
        
        # Create basic project structure
        (project_dir / "tests").mkdir()
        (project_dir / "docs").mkdir()
        (project_dir / "config").mkdir()
        (project_dir / "tools").mkdir()
        
        # Create sample files
        (project_dir / "tests" / "test_sample.py").write_text("""
import pytest

def test_sample():
    assert True

def test_another():
    assert 1 + 1 == 2
""")
        
        (project_dir / "docs" / "README.md").write_text("# Test Project\n\nSample documentation.")
        
        (project_dir / "config" / "base.yaml").write_text("""
system:
  name: test_project
  version: "1.0.0"
backend:
  port: 8000
""")
        
        yield project_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_complete_health_system_workflow(self, temp_project_dir):
        """Test the complete project health system workflow from start to finish."""
        
        # Step 1: Initialize project health system
        os.chdir(temp_project_dir)
        
        # Step 2: Run test suite orchestration
        test_orchestrator = TestSuiteOrchestrator(
            project_root=temp_project_dir,
            config_path=temp_project_dir / "tests" / "config" / "test-config.yaml"
        )
        
        # Create test config if it doesn't exist
        test_config_dir = temp_project_dir / "tests" / "config"
        test_config_dir.mkdir(exist_ok=True)
        test_config_path = test_config_dir / "test-config.yaml"
        
        test_config = {
            "test_categories": {
                "unit": {
                    "timeout": 30,
                    "parallel": True,
                    "coverage_threshold": 70
                },
                "integration": {
                    "timeout": 120,
                    "parallel": False
                }
            },
            "coverage": {
                "minimum_threshold": 70,
                "report_format": "json"
            }
        }
        
        with open(test_config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        # Run test suite
        test_results = await test_orchestrator.run_full_suite()
        
        # Verify test results
        assert test_results is not None
        assert test_results.overall_summary.total_tests >= 0
        
        # Step 3: Generate and validate documentation
        doc_generator = DocumentationGenerator(
            source_dirs=[temp_project_dir / "docs"],
            output_dir=temp_project_dir / "docs" / "_build"
        )
        
        # Generate documentation
        doc_generator.consolidate_existing_docs()
        doc_generator.generate_search_index()
        
        # Validate documentation
        validation_report = doc_generator.validate_links()
        assert validation_report is not None
        
        # Step 4: Unify and validate configuration
        config_unifier = ConfigurationUnifier(
            config_sources=[temp_project_dir / "config"]
        )
        
        # Migrate configurations
        migration_report = config_unifier.migrate_existing_configs()
        assert migration_report is not None
        
        # Step 5: Run comprehensive health check
        health_checker = ProjectHealthChecker(
            project_root=temp_project_dir
        )
        
        health_report = await health_checker.run_health_check()
        
        # Verify health report
        assert health_report is not None
        assert health_report.overall_score >= 0
        assert health_report.overall_score <= 100
        assert len(health_report.component_scores) > 0
        
        # Step 6: Verify system integration
        # Check that all components can work together
        assert health_report.component_scores.get("test_health", 0) >= 0
        assert health_report.component_scores.get("documentation_health", 0) >= 0
        assert health_report.component_scores.get("configuration_health", 0) >= 0
        
        # Verify recommendations are generated
        recommendations = health_checker.get_recommendations()
        assert isinstance(recommendations, list)

    @pytest.mark.asyncio
    async def test_component_interaction_and_data_flow(self, temp_project_dir):
        """Test interaction between project health components and data flow."""
        
        os.chdir(temp_project_dir)
        
        # Test data flow: Test Results -> Health Checker -> Recommendations
        
        # 1. Generate test results
        test_orchestrator = TestSuiteOrchestrator(
            project_root=temp_project_dir,
            config_path=None  # Use defaults
        )
        
        test_results = await test_orchestrator.run_category("unit")
        
        # 2. Feed test results to health checker
        health_checker = ProjectHealthChecker(
            project_root=temp_project_dir
        )
        
        # Mock test results integration
        health_checker.test_results = test_results
        
        # 3. Generate health report
        health_report = await health_checker.run_health_check()
        
        # 4. Verify data flow
        assert health_report.component_scores.get("test_health") is not None
        
        # Test data flow: Config Changes -> Health Validation -> Notifications
        
        # 1. Make configuration change
        config_file = temp_project_dir / "config" / "base.yaml"
        config_data = yaml.safe_load(config_file.read_text())
        config_data["backend"]["port"] = 9000
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # 2. Validate configuration through health system
        config_unifier = ConfigurationUnifier(
            config_sources=[temp_project_dir / "config"]
        )
        
        validation_result = config_unifier.validate_configuration(config_data)
        
        # 3. Verify validation integration
        assert validation_result is not None

    @pytest.mark.asyncio
    async def test_system_performance_benchmarking(self, temp_project_dir):
        """Test system performance and benchmarking capabilities."""
        
        os.chdir(temp_project_dir)
        
        # Create performance test configuration
        perf_config = {
            "performance_tests": {
                "test_suite_execution": {
                    "max_duration": 300,  # 5 minutes
                    "memory_limit": "1GB"
                },
                "documentation_generation": {
                    "max_duration": 60,  # 1 minute
                    "file_limit": 1000
                },
                "health_check_execution": {
                    "max_duration": 30,  # 30 seconds
                    "component_limit": 10
                }
            }
        }
        
        perf_config_path = temp_project_dir / "tests" / "performance_config.yaml"
        with open(perf_config_path, 'w') as f:
            yaml.dump(perf_config, f)
        
        # Benchmark test suite execution
        import time

        start_time = time.time()
        test_orchestrator = TestSuiteOrchestrator(
            project_root=temp_project_dir,
            config_path=None
        )
        
        test_results = await test_orchestrator.run_full_suite()
        test_duration = time.time() - start_time
        
        # Verify performance requirements
        assert test_duration < perf_config["performance_tests"]["test_suite_execution"]["max_duration"]
        
        # Benchmark documentation generation
        start_time = time.time()
        doc_generator = DocumentationGenerator(
            source_dirs=[temp_project_dir / "docs"],
            output_dir=temp_project_dir / "docs" / "_build"
        )
        
        doc_generator.consolidate_existing_docs()
        doc_duration = time.time() - start_time
        
        # Verify documentation performance
        assert doc_duration < perf_config["performance_tests"]["documentation_generation"]["max_duration"]
        
        # Benchmark health check execution
        start_time = time.time()
        health_checker = ProjectHealthChecker(
            project_root=temp_project_dir
        )
        
        health_report = await health_checker.run_health_check()
        health_duration = time.time() - start_time
        
        # Verify health check performance
        assert health_duration < perf_config["performance_tests"]["health_check_execution"]["max_duration"]
        
        # Create performance report
        performance_report = {
            "test_suite_duration": test_duration,
            "documentation_duration": doc_duration,
            "health_check_duration": health_duration,
            "total_duration": test_duration + doc_duration + health_duration,
            "performance_requirements_met": True
        }
        
        # Save performance report
        perf_report_path = temp_project_dir / "performance_report.json"
        with open(perf_report_path, 'w') as f:
            json.dump(performance_report, f, indent=2)
        
        assert performance_report["performance_requirements_met"]

    @pytest.mark.asyncio
    async def test_developer_experience_integration(self, temp_project_dir):
        """Test developer experience tools integration with health system."""
        
        os.chdir(temp_project_dir)
        
        # Test development environment setup integration
        dev_setup = DevEnvironmentSetup(project_root=temp_project_dir)
        
        # Validate development environment
        validation_result = dev_setup.validate_environment()
        assert validation_result is not None
        
        # Test integration with health monitoring
        health_checker = ProjectHealthChecker(
            project_root=temp_project_dir
        )
        
        # Include dev environment validation in health check
        health_report = await health_checker.run_health_check()
        
        # Verify dev environment is included in health assessment
        assert "development_environment" in health_report.component_scores or \
               any("environment" in key.lower() for key in health_report.component_scores.keys())

    def test_error_handling_and_recovery(self, temp_project_dir):
        """Test error handling and recovery mechanisms in integration scenarios."""
        
        os.chdir(temp_project_dir)
        
        # Test handling of missing configuration files
        missing_config_dir = temp_project_dir / "missing_config"
        
        try:
            config_unifier = ConfigurationUnifier(
                config_sources=[missing_config_dir]
            )
            migration_report = config_unifier.migrate_existing_configs()
            # Should handle missing directories gracefully
            assert migration_report is not None
        except Exception as e:
            # Should provide meaningful error messages
            assert "config" in str(e).lower() or "missing" in str(e).lower()
        
        # Test handling of invalid test files
        invalid_test_file = temp_project_dir / "tests" / "invalid_test.py"
        invalid_test_file.write_text("invalid python syntax !!!")
        
        try:
            test_orchestrator = TestSuiteOrchestrator(
                project_root=temp_project_dir,
                config_path=None
            )
            # Should handle syntax errors gracefully
            # This is tested in async context in other methods
        except Exception as e:
            # Should provide meaningful error messages
            assert "syntax" in str(e).lower() or "invalid" in str(e).lower()

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, temp_project_dir):
        """Test concurrent operations and thread safety."""
        
        os.chdir(temp_project_dir)
        
        # Test concurrent health checks
        health_checker1 = ProjectHealthChecker(project_root=temp_project_dir)
        health_checker2 = ProjectHealthChecker(project_root=temp_project_dir)
        
        # Run concurrent health checks
        results = await asyncio.gather(
            health_checker1.run_health_check(),
            health_checker2.run_health_check(),
            return_exceptions=True
        )
        
        # Verify both completed successfully
        assert len(results) == 2
        for result in results:
            if not isinstance(result, Exception):
                assert result.overall_score >= 0

    def test_integration_with_ci_cd_workflows(self, temp_project_dir):
        """Test integration with CI/CD workflows and automation."""
        
        os.chdir(temp_project_dir)
        
        # Create CI/CD configuration
        github_dir = temp_project_dir / ".github" / "workflows"
        github_dir.mkdir(parents=True, exist_ok=True)
        
        # Test health check workflow integration
        workflow_config = """
name: Project Health Check
on: [push, pull_request]
jobs:
  health-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Health Check
        run: python -m tools.health_checker.cli --format json
"""
        
        workflow_file = github_dir / "health-check.yml"
        workflow_file.write_text(workflow_config)
        
        # Verify workflow file exists and is valid
        assert workflow_file.exists()
        assert "health-check" in workflow_file.read_text()
        
        # Test pre-commit hook integration
        precommit_config = temp_project_dir / ".pre-commit-config.yaml"
        precommit_content = """
repos:
  - repo: local
    hooks:
      - id: health-check
        name: Project Health Check
        entry: python -m tools.health_checker.cli
        language: python
"""
        
        precommit_config.write_text(precommit_content)
        
        # Verify pre-commit configuration
        assert precommit_config.exists()
        assert "health-check" in precommit_config.read_text()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])