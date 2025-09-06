#!/usr/bin/env python3
"""
Integration tests for the comprehensive maintenance reporting system.
"""

import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from operation_logger import OperationLogger
from impact_analyzer import ImpactAnalyzer
from recommendation_engine import MaintenanceRecommendationEngine
from report_generator import MaintenanceReportGenerator
from models import MaintenanceOperationType, ImpactLevel, MaintenanceStatus


class TestMaintenanceReportingIntegration(unittest.TestCase):
    """Integration tests for the maintenance reporting system."""
    
    def setUp(self):
        """Set up test environment with temporary data directories."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir)
        
        # Initialize components with temporary data directories
        self.operation_logger = OperationLogger(str(self.data_dir / "operations"))
        self.impact_analyzer = ImpactAnalyzer(str(self.data_dir / "impact"))
        self.recommendation_engine = MaintenanceRecommendationEngine(str(self.data_dir / "recommendations"))
        self.report_generator = MaintenanceReportGenerator(str(self.data_dir / "reports"))
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_operation_lifecycle(self):
        """Test complete operation lifecycle with audit trails."""
        # Start operation
        operation_id = self.operation_logger.start_operation(
            operation_type=MaintenanceOperationType.TEST_REPAIR,
            title="Test operation lifecycle",
            description="Integration test for complete operation lifecycle",
            impact_level=ImpactLevel.MEDIUM,
            files_affected=["test_file.py"],
            components_affected=["test_component"]
        )
        
        self.assertIsNotNone(operation_id)
        
        # Update progress
        success = self.operation_logger.update_operation_progress(
            operation_id, {"progress": "50%", "tests_fixed": 5}
        )
        self.assertTrue(success)
        
        # Complete operation
        success = self.operation_logger.complete_operation(
            operation_id,
            success_metrics={"tests_fixed": 10, "coverage_improved": 5.2}
        )
        self.assertTrue(success)
        
        # Verify operation state
        operation = self.operation_logger.get_operation(operation_id)
        self.assertIsNotNone(operation)
        self.assertEqual(operation.status, MaintenanceStatus.COMPLETED)
        self.assertIsNotNone(operation.completed_at)
        self.assertIsNotNone(operation.duration_seconds)
        
        # Verify audit trail
        audit_trail = self.operation_logger.get_audit_trail_for_operation(operation_id)
        self.assertGreaterEqual(len(audit_trail), 3)  # start, progress, complete
        
        # Verify audit trail actions
        actions = [entry.action for entry in audit_trail]
        self.assertIn("operation_started", actions)
        self.assertIn("progress_update", actions)
        self.assertIn("operation_completed", actions)
    
    def test_impact_analysis_workflow(self):
        """Test impact analysis workflow."""
        # Start and complete an operation
        operation_id = self.operation_logger.start_operation(
            operation_type=MaintenanceOperationType.CODE_CLEANUP,
            title="Test impact analysis",
            description="Test operation for impact analysis",
            impact_level=ImpactLevel.HIGH
        )
        
        self.operation_logger.complete_operation(
            operation_id,
            success_metrics={"lines_removed": 100, "complexity_reduced": 2.5}
        )
        
        # Perform impact analysis
        before_metrics = {
            "code_complexity": 12.5,
            "duplicate_code": 18.7,
            "test_coverage": 72.0
        }
        
        after_metrics = {
            "code_complexity": 10.0,
            "duplicate_code": 15.2,
            "test_coverage": 74.5
        }
        
        analysis = self.impact_analyzer.analyze_operation_impact(
            operation_id, before_metrics, after_metrics
        )
        
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis.operation_id, operation_id)
        self.assertGreater(analysis.overall_impact_score, 0)  # Should be positive impact
        self.assertIn("improvement", analysis.impact_summary.lower())
        
        # Verify improvements were detected
        self.assertGreater(len(analysis.improvements), 0)
        self.assertIn("code_complexity", analysis.improvements)
        self.assertIn("duplicate_code", analysis.improvements)
    
    def test_recommendation_generation(self):
        """Test recommendation generation based on project metrics."""
        # Create some historical operations for context
        for i in range(3):
            op_id = self.operation_logger.start_operation(
                operation_type=MaintenanceOperationType.TEST_REPAIR,
                title=f"Historical operation {i}",
                description=f"Test operation {i}",
                impact_level=ImpactLevel.MEDIUM
            )
            self.operation_logger.complete_operation(op_id, {"tests_fixed": i + 1})
        
        # Generate recommendations
        project_metrics = {
            "test_coverage": 65.0,  # Below recommended threshold
            "code_complexity": 15.0,  # Above recommended threshold
            "documentation_coverage": 40.0,  # Below recommended threshold
            "duplicate_code": 20.0,  # Above recommended threshold
            "style_violations": 75.0  # Above recommended threshold
        }
        
        recommendations = self.recommendation_engine.generate_recommendations_from_analysis(
            self.operation_logger, self.impact_analyzer, project_metrics
        )
        
        self.assertGreater(len(recommendations), 0)
        
        # Verify recommendation properties
        for rec in recommendations:
            self.assertIsNotNone(rec.id)
            self.assertIsNotNone(rec.title)
            self.assertIsNotNone(rec.description)
            self.assertIsInstance(rec.operation_type, MaintenanceOperationType)
            self.assertIsInstance(rec.priority, ImpactLevel)
            self.assertGreater(rec.estimated_effort_hours, 0)
            self.assertIsNotNone(rec.estimated_impact_score)
    
    def test_schedule_optimization(self):
        """Test maintenance schedule optimization."""
        # Generate some recommendations
        project_metrics = {
            "test_coverage": 60.0,
            "code_complexity": 12.0,
            "documentation_coverage": 45.0
        }
        
        recommendations = self.recommendation_engine.generate_recommendations_from_analysis(
            self.operation_logger, self.impact_analyzer, project_metrics
        )
        
        if recommendations:
            # Optimize schedule
            schedule = self.recommendation_engine.optimize_maintenance_schedule(
                recommendations,
                available_hours_per_week=40,
                max_concurrent_operations=2
            )
            
            self.assertIsNotNone(schedule)
            self.assertGreater(len(schedule.recommended_schedule), 0)
            self.assertGreater(schedule.estimated_total_duration_hours, 0)
            self.assertIsInstance(schedule.resource_requirements, dict)
            self.assertIsInstance(schedule.risk_mitigation_plan, list)
            self.assertIsInstance(schedule.dependencies, dict)
            
            # Verify all recommendations are scheduled
            scheduled_ids = set(schedule.recommended_schedule)
            recommendation_ids = set(rec.id for rec in recommendations)
            self.assertEqual(scheduled_ids, recommendation_ids)
    
    def test_comprehensive_report_generation(self):
        """Test comprehensive report generation."""
        # Create test data
        operation_id = self.operation_logger.start_operation(
            operation_type=MaintenanceOperationType.QUALITY_IMPROVEMENT,
            title="Test report generation",
            description="Operation for testing report generation",
            impact_level=ImpactLevel.MEDIUM
        )
        
        self.operation_logger.complete_operation(
            operation_id,
            success_metrics={"quality_score_improved": 5.0}
        )
        
        # Add impact analysis
        self.impact_analyzer.analyze_operation_impact(
            operation_id,
            {"quality_score": 75.0},
            {"quality_score": 80.0}
        )
        
        # Generate comprehensive report
        report = self.report_generator.generate_comprehensive_report(
            self.operation_logger,
            self.impact_analyzer,
            self.recommendation_engine,
            period_days=1
        )
        
        self.assertIsNotNone(report)
        self.assertIsNotNone(report.report_id)
        self.assertEqual(report.report_type, "comprehensive")
        self.assertGreater(len(report.operations), 0)
        self.assertIsInstance(report.summary_statistics, dict)
        
        # Verify summary statistics
        stats = report.summary_statistics
        self.assertIn("operations", stats)
        self.assertIn("period", stats)
        
        op_stats = stats["operations"]
        self.assertGreater(op_stats["total"], 0)
        self.assertGreaterEqual(op_stats["success_rate"], 0)
    
    def test_html_report_export(self):
        """Test HTML report export functionality."""
        # Create a simple operation for the report
        operation_id = self.operation_logger.start_operation(
            operation_type=MaintenanceOperationType.DOCUMENTATION_UPDATE,
            title="Test HTML export",
            description="Operation for testing HTML export",
            impact_level=ImpactLevel.LOW
        )
        
        self.operation_logger.complete_operation(operation_id)
        
        # Generate report
        report = self.report_generator.generate_daily_report(
            self.operation_logger, self.impact_analyzer
        )
        
        # Export to HTML
        html_path = self.report_generator.export_report_to_html(report)
        
        self.assertTrue(html_path.exists())
        self.assertTrue(html_path.suffix == ".html")
        
        # Verify HTML content
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        self.assertIn("<!DOCTYPE html>", html_content)
        self.assertIn("Maintenance Report", html_content)
        self.assertIn(report.report_id, html_content)
        self.assertIn("Test HTML export", html_content)
    
    def test_json_serialization(self):
        """Test JSON serialization of reports."""
        # Create test operation
        operation_id = self.operation_logger.start_operation(
            operation_type=MaintenanceOperationType.PERFORMANCE_OPTIMIZATION,
            title="Test JSON serialization",
            description="Operation for testing JSON serialization",
            impact_level=ImpactLevel.HIGH
        )
        
        self.operation_logger.complete_operation(
            operation_id,
            success_metrics={"performance_improved": 15.5}
        )
        
        # Generate report
        report = self.report_generator.generate_daily_report(
            self.operation_logger, self.impact_analyzer
        )
        
        # Test JSON serialization
        json_str = report.to_json()
        self.assertIsInstance(json_str, str)
        
        # Verify JSON is valid
        parsed_json = json.loads(json_str)
        self.assertIsInstance(parsed_json, dict)
        self.assertIn("report_id", parsed_json)
        self.assertIn("operations", parsed_json)
        self.assertIn("summary_statistics", parsed_json)
    
    def test_failed_operation_handling(self):
        """Test handling of failed operations with rollback."""
        # Start operation
        operation_id = self.operation_logger.start_operation(
            operation_type=MaintenanceOperationType.CONFIGURATION_CONSOLIDATION,
            title="Test failure handling",
            description="Operation for testing failure handling",
            impact_level=ImpactLevel.CRITICAL
        )
        
        # Fail the operation
        rollback_info = {
            "backup_location": "/tmp/backup",
            "rollback_completed": True,
            "rollback_steps": ["restore_config", "restart_service"]
        }
        
        success = self.operation_logger.fail_operation(
            operation_id,
            error_details="Configuration validation failed",
            rollback_info=rollback_info
        )
        
        self.assertTrue(success)
        
        # Verify operation state
        operation = self.operation_logger.get_operation(operation_id)
        self.assertEqual(operation.status, MaintenanceStatus.FAILED)
        self.assertIsNotNone(operation.error_details)
        self.assertIsNotNone(operation.rollback_info)
        self.assertEqual(operation.rollback_info["rollback_completed"], True)
        
        # Verify audit trail includes failure
        audit_trail = self.operation_logger.get_audit_trail_for_operation(operation_id)
        actions = [entry.action for entry in audit_trail]
        self.assertIn("operation_failed", actions)
    
    def test_statistics_generation(self):
        """Test statistics generation."""
        # Create multiple operations with different outcomes
        operations = []
        
        # Successful operations
        for i in range(3):
            op_id = self.operation_logger.start_operation(
                operation_type=MaintenanceOperationType.TEST_REPAIR,
                title=f"Successful operation {i}",
                description=f"Test operation {i}",
                impact_level=ImpactLevel.MEDIUM
            )
            self.operation_logger.complete_operation(op_id)
            operations.append(op_id)
        
        # Failed operation
        failed_op_id = self.operation_logger.start_operation(
            operation_type=MaintenanceOperationType.CODE_CLEANUP,
            title="Failed operation",
            description="Test failed operation",
            impact_level=ImpactLevel.HIGH
        )
        self.operation_logger.fail_operation(failed_op_id, "Test failure")
        operations.append(failed_op_id)
        
        # Get statistics
        stats = self.operation_logger.get_operation_statistics()
        
        self.assertEqual(stats["total_operations"], 4)
        self.assertEqual(stats["by_status"]["completed"], 3)
        self.assertEqual(stats["by_status"]["failed"], 1)
        self.assertEqual(stats["success_rate"], 75.0)  # 3/4 = 75%
        
        # Test improvement opportunities
        opportunities = self.impact_analyzer.identify_improvement_opportunities(
            self.operation_logger, days=1
        )
        
        # Should identify high failure rate for code_cleanup
        self.assertIsInstance(opportunities, list)
    
    def test_data_cleanup(self):
        """Test data cleanup functionality."""
        # Create old operation (simulate by manipulating timestamp)
        operation_id = self.operation_logger.start_operation(
            operation_type=MaintenanceOperationType.DEPENDENCY_UPDATE,
            title="Old operation",
            description="Operation for testing cleanup",
            impact_level=ImpactLevel.LOW
        )
        
        # Manually set old timestamp for testing
        operation = self.operation_logger.get_operation(operation_id)
        old_date = datetime.now() - timedelta(days=100)
        operation.started_at = old_date
        operation.status = MaintenanceStatus.COMPLETED  # Make it completed so it can be cleaned up
        operation.completed_at = old_date + timedelta(hours=1)
        
        # Save the modified operation
        self.operation_logger.operations[operation_id] = operation
        self.operation_logger._save_data()
        
        # Perform cleanup
        cleaned_count = self.operation_logger.cleanup_old_operations(days=90)
        
        # Verify cleanup worked
        self.assertGreaterEqual(cleaned_count, 1)
        
        # Verify operation was removed
        remaining_operation = self.operation_logger.get_operation(operation_id)
        self.assertIsNone(remaining_operation)


def run_integration_tests():
    """Run all integration tests."""
    print("Running Maintenance Reporting System Integration Tests")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMaintenanceReportingIntegration)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("All integration tests passed successfully!")
        print(f"Tests run: {result.testsRun}")
    else:
        print(f"Integration tests failed!")
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  {test}: {traceback}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_integration_tests()
    exit(0 if success else 1)