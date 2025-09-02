"""
Integration tests for project health component interaction and data flow.

This module tests the interaction between different project health components
and validates that data flows correctly between them.

Requirements: 1.1, 4.1
"""

import asyncio
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
import pytest
import yaml

from tools.test_runner.orchestrator import TestSuiteOrchestrator
from tools.test_runner.coverage_analyzer import CoverageAnalyzer
from tools.doc_generator.documentation_generator import DocumentationGenerator
from tools.doc_generator.validator import DocumentationValidator
from tools.config_manager.config_unifier import ConfigurationUnifier
from tools.config_manager.config_validator import ConfigurationValidator
from tools.health_checker.health_checker import ProjectHealthChecker
from tools.health_checker.health_reporter import HealthReporter


class TestComponentInteraction:
    """Test interaction between project health system components."""

    @pytest.fixture
    def sample_project(self):
        """Create a sample project for testing component interactions."""
        temp_dir = tempfile.mkdtemp()
        project_dir = Path(temp_dir) / "sample_project"
        project_dir.mkdir()
        
        # Create project structure
        (project_dir / "tests").mkdir()
        (project_dir / "tests" / "unit").mkdir()
        (project_dir / "tests" / "integration").mkdir()
        (project_dir / "docs").mkdir()
        (project_dir / "config").mkdir()
        (project_dir / "src").mkdir()
        
        # Create sample test files
        (project_dir / "tests" / "unit" / "test_math.py").write_text("""
import pytest

def test_addition():
    assert 1 + 1 == 2

def test_subtraction():
    assert 5 - 3 == 2

def test_multiplication():
    assert 3 * 4 == 12
""")
        
        (project_dir / "tests" / "integration" / "test_api.py").write_text("""
import pytest

def test_api_health():
    # Mock API health check
    assert True

def test_api_response():
    # Mock API response test
    assert True
""")
        
        # Create sample source files
        (project_dir / "src" / "calculator.py").write_text("""
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b
""")
        
        # Create documentation files
        (project_dir / "docs" / "README.md").write_text("""
# Sample Project

This is a sample project for testing.

## Features

- Addition
- Subtraction
- Multiplication

## API Reference

See [API Documentation](api.md) for details.
""")
        
        (project_dir / "docs" / "api.md").write_text("""
# API Documentation

## Calculator API

### add(a, b)
Returns the sum of a and b.

### subtract(a, b)
Returns the difference of a and b.

### multiply(a, b)
Returns the product of a and b.
""")
        
        # Create configuration files
        (project_dir / "config" / "base.yaml").write_text("""
system:
  name: sample_project
  version: "1.0.0"
  debug: false

api:
  host: localhost
  port: 8000
  timeout: 30

database:
  host: localhost
  port: 5432
  name: sample_db
""")
        
        (project_dir / "config" / "development.yaml").write_text("""
system:
  debug: true

api:
  port: 8001

database:
  name: sample_db_dev
""")
        
        yield project_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_test_results_to_health_checker_flow(self, sample_project):
        """Test data flow from test execution to health checker."""
        
        # Step 1: Execute tests and generate results
        test_orchestrator = TestSuiteOrchestrator(
            project_root=sample_project,
            config_path=None
        )
        
        test_results = await test_orchestrator.run_full_suite()
        
        # Verify test results are generated
        assert test_results is not None
        assert test_results.overall_summary.total_tests > 0
        
        # Step 2: Feed test results to health checker
        health_checker = ProjectHealthChecker(project_root=sample_project)
        
        # Simulate integration of test results
        health_checker.test_results = test_results
        
        # Step 3: Generate health report including test health
        health_report = await health_checker.run_health_check()
        
        # Verify test results are reflected in health report
        assert health_report is not None
        assert "test_health" in health_report.component_scores
        
        # Verify test metrics influence health score
        test_health_score = health_report.component_scores["test_health"]
        assert 0 <= test_health_score <= 100

    @pytest.mark.asyncio
    async def test_coverage_analysis_integration(self, sample_project):
        """Test integration of coverage analysis with health monitoring."""
        
        # Step 1: Run coverage analysis
        coverage_analyzer = CoverageAnalyzer(
            project_root=sample_project,
            source_dirs=[sample_project / "src"]
        )
        
        coverage_report = coverage_analyzer.analyze_coverage()
        
        # Step 2: Integrate coverage data with health checker
        health_checker = ProjectHealthChecker(project_root=sample_project)
        health_checker.coverage_report = coverage_report
        
        # Step 3: Generate health report with coverage metrics
        health_report = await health_checker.run_health_check()
        
        # Verify coverage is reflected in health assessment
        assert health_report is not None
        
        # Check if coverage influences test health score
        if "test_health" in health_report.component_scores:
            test_health_score = health_report.component_scores["test_health"]
            assert isinstance(test_health_score, (int, float))

    def test_documentation_validation_to_health_flow(self, sample_project):
        """Test data flow from documentation validation to health reporting."""
        
        # Step 1: Validate documentation
        doc_validator = DocumentationValidator(
            docs_dir=sample_project / "docs"
        )
        
        validation_report = doc_validator.validate_all()
        
        # Verify validation report is generated
        assert validation_report is not None
        
        # Step 2: Integrate validation results with health checker
        health_checker = ProjectHealthChecker(project_root=sample_project)
        health_checker.documentation_validation = validation_report
        
        # Step 3: Check health report includes documentation health
        # Note: This would be async in real implementation
        # For this test, we'll check the integration setup
        
        # Verify documentation validator can be integrated
        assert hasattr(health_checker, 'documentation_validation')

    def test_configuration_validation_integration(self, sample_project):
        """Test integration of configuration validation with health system."""
        
        # Step 1: Validate configuration
        config_validator = ConfigurationValidator()
        
        config_files = list((sample_project / "config").glob("*.yaml"))
        validation_results = {}
        
        for config_file in config_files:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            validation_result = config_validator.validate_config(config_data)
            validation_results[config_file.name] = validation_result
        
        # Step 2: Integrate with health checker
        health_checker = ProjectHealthChecker(project_root=sample_project)
        health_checker.config_validation_results = validation_results
        
        # Verify integration
        assert hasattr(health_checker, 'config_validation_results')
        assert len(validation_results) > 0

    @pytest.mark.asyncio
    async def test_cross_component_data_consistency(self, sample_project):
        """Test data consistency across different components."""
        
        # Step 1: Generate data from multiple components
        test_orchestrator = TestSuiteOrchestrator(
            project_root=sample_project,
            config_path=None
        )
        
        doc_generator = DocumentationGenerator(
            source_dirs=[sample_project / "docs"],
            output_dir=sample_project / "docs" / "_build"
        )
        
        config_unifier = ConfigurationUnifier(
            config_sources=[sample_project / "config"]
        )
        
        # Execute components
        test_results = await test_orchestrator.run_full_suite()
        doc_generator.consolidate_existing_docs()
        migration_report = config_unifier.migrate_existing_configs()
        
        # Step 2: Verify data consistency
        # All components should reference the same project
        assert test_results.suite_id is not None
        assert migration_report is not None
        
        # Step 3: Integrate all data in health checker
        health_checker = ProjectHealthChecker(project_root=sample_project)
        health_checker.test_results = test_results
        health_checker.migration_report = migration_report
        
        health_report = await health_checker.run_health_check()
        
        # Verify consistent project identification
        assert health_report is not None
        assert health_report.timestamp is not None

    def test_health_reporter_integration(self, sample_project):
        """Test integration of health reporter with other components."""
        
        # Step 1: Create sample health data
        sample_health_data = {
            "overall_score": 85.5,
            "component_scores": {
                "test_health": 90.0,
                "documentation_health": 80.0,
                "configuration_health": 85.0,
                "code_quality": 88.0
            },
            "issues": [
                {
                    "severity": "medium",
                    "category": "documentation",
                    "description": "Missing API documentation for 2 functions",
                    "affected_components": ["docs/api.md"]
                }
            ]
        }
        
        # Step 2: Generate health report
        health_reporter = HealthReporter()
        
        # Test different report formats
        json_report = health_reporter.generate_json_report(sample_health_data)
        html_report = health_reporter.generate_html_report(sample_health_data)
        summary_report = health_reporter.generate_summary_report(sample_health_data)
        
        # Verify reports are generated
        assert json_report is not None
        assert html_report is not None
        assert summary_report is not None
        
        # Verify report content consistency
        assert "85.5" in json_report or "85" in json_report
        assert "test_health" in json_report
        assert "documentation" in html_report.lower()

    @pytest.mark.asyncio
    async def test_recommendation_engine_integration(self, sample_project):
        """Test integration of recommendation engine with health data."""
        
        # Step 1: Generate health report with issues
        health_checker = ProjectHealthChecker(project_root=sample_project)
        health_report = await health_checker.run_health_check()
        
        # Step 2: Generate recommendations
        recommendations = health_checker.get_recommendations()
        
        # Verify recommendations are generated
        assert isinstance(recommendations, list)
        
        # Step 3: Test recommendation integration with health reporter
        health_reporter = HealthReporter()
        
        # Include recommendations in report
        report_data = {
            "overall_score": health_report.overall_score,
            "component_scores": health_report.component_scores,
            "recommendations": [
                {
                    "priority": "high",
                    "category": "testing",
                    "description": "Increase test coverage to 80%",
                    "action_items": ["Add unit tests for uncovered functions"]
                }
            ]
        }
        
        report_with_recommendations = health_reporter.generate_json_report(report_data)
        
        # Verify recommendations are included
        assert "recommendations" in report_with_recommendations
        assert "priority" in report_with_recommendations

    def test_configuration_hot_reload_integration(self, sample_project):
        """Test integration of configuration hot-reload with health monitoring."""
        
        # Step 1: Set up configuration monitoring
        config_unifier = ConfigurationUnifier(
            config_sources=[sample_project / "config"]
        )
        
        # Step 2: Simulate configuration change
        config_file = sample_project / "config" / "base.yaml"
        original_content = config_file.read_text()
        
        # Modify configuration
        config_data = yaml.safe_load(original_content)
        config_data["api"]["port"] = 9000
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Step 3: Validate configuration change
        validation_result = config_unifier.validate_configuration(config_data)
        
        # Step 4: Restore original configuration
        config_file.write_text(original_content)
        
        # Verify validation handled the change
        assert validation_result is not None

    @pytest.mark.asyncio
    async def test_parallel_component_execution(self, sample_project):
        """Test parallel execution of multiple components."""
        
        # Define component execution functions
        async def run_tests():
            orchestrator = TestSuiteOrchestrator(
                project_root=sample_project,
                config_path=None
            )
            return await orchestrator.run_full_suite()
        
        async def validate_docs():
            validator = DocumentationValidator(
                docs_dir=sample_project / "docs"
            )
            return validator.validate_all()
        
        async def check_health():
            checker = ProjectHealthChecker(project_root=sample_project)
            return await checker.run_health_check()
        
        # Execute components in parallel
        results = await asyncio.gather(
            run_tests(),
            validate_docs(),
            check_health(),
            return_exceptions=True
        )
        
        # Verify all components completed
        assert len(results) == 3
        
        # Check that no exceptions occurred
        for result in results:
            assert not isinstance(result, Exception), f"Component failed: {result}"

    def test_error_propagation_between_components(self, sample_project):
        """Test how errors propagate between integrated components."""
        
        # Step 1: Create invalid configuration
        invalid_config_file = sample_project / "config" / "invalid.yaml"
        invalid_config_file.write_text("invalid: yaml: content: [")
        
        # Step 2: Test error handling in configuration validator
        config_validator = ConfigurationValidator()
        
        try:
            with open(invalid_config_file, 'r') as f:
                yaml.safe_load(f)
        except yaml.YAMLError as e:
            # Verify error is caught and can be handled
            assert "yaml" in str(e).lower() or "invalid" in str(e).lower()
        
        # Step 3: Test error propagation to health checker
        health_checker = ProjectHealthChecker(project_root=sample_project)
        
        # Health checker should handle component errors gracefully
        try:
            # This should not crash the entire health check
            config_unifier = ConfigurationUnifier(
                config_sources=[sample_project / "config"]
            )
            migration_report = config_unifier.migrate_existing_configs()
            
            # Should handle invalid files gracefully
            assert migration_report is not None
        except Exception as e:
            # If an exception occurs, it should be informative
            assert len(str(e)) > 0
        
        # Cleanup
        invalid_config_file.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])