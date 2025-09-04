"""
Comprehensive tests for the Health Reporting and Analytics System

Tests all functionality including installation reporting, error pattern tracking,
trend analysis, recovery method effectiveness logging, and dashboard generation.
"""

import unittest
import tempfile
import shutil
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import the modules to test
try:
    from scripts.health_reporter import (
        HealthReporter, InstallationReport, InstallationStatus, ErrorTrend,
        TrendAnalysis, RecoveryMethodStats, RecoveryEffectiveness,
        create_health_dashboard_html
    )
    from scripts.interfaces import ErrorCategory, InstallationPhase, HardwareProfile
    from scripts.diagnostic_monitor import ResourceMetrics, ComponentHealth, HealthStatus
except ImportError:
    import sys
    sys.path.append('scripts')
    from health_reporter import (
        HealthReporter, InstallationReport, InstallationStatus, ErrorTrend,
        TrendAnalysis, RecoveryMethodStats, RecoveryEffectiveness,
        create_health_dashboard_html
    )
    from interfaces import ErrorCategory, InstallationPhase, HardwareProfile
    from diagnostic_monitor import ResourceMetrics, ComponentHealth, HealthStatus


class TestHealthReporter(unittest.TestCase):
    """Test cases for HealthReporter class."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_path = Path(self.test_dir) / "test_config.json"
        
        # Create test configuration
        test_config = {
            "health_reporting": {
                "retention_days": 30,
                "analysis_window_days": 7,
                "min_data_points": 3
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(test_config, f)
        
        self.health_reporter = HealthReporter(
            installation_path=self.test_dir,
            config_path=str(self.config_path)
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test HealthReporter initialization."""
        self.assertIsInstance(self.health_reporter, HealthReporter)
        self.assertTrue(self.health_reporter.db_path.exists())
        self.assertEqual(self.health_reporter.retention_days, 30)
        self.assertEqual(self.health_reporter.analysis_window_days, 7)
        self.assertEqual(self.health_reporter.min_data_points, 3)
    
    def test_database_initialization(self):
        """Test database schema creation."""
        with sqlite3.connect(str(self.health_reporter.db_path)) as conn:
            cursor = conn.cursor()
            
            # Check if tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            expected_tables = [
                'installation_reports', 'error_patterns', 
                'recovery_stats', 'performance_metrics'
            ]
            
            for table in expected_tables:
                self.assertIn(table, tables)
    
    def test_generate_installation_report(self):
        """Test installation report generation."""
        # Create test data
        installation_id = "test-install-001"
        status = InstallationStatus.SUCCESS
        duration = 1800.0  # 30 minutes
        phases = [InstallationPhase.DETECTION, InstallationPhase.DEPENDENCIES]
        errors = [{"type": "network_error", "category": "network", "message": "Connection timeout"}]
        warnings = ["Low disk space warning"]
        recoveries = [{"method": "retry", "success": True, "execution_time": 5.0}]
        
        # Generate report
        report = self.health_reporter.generate_installation_report(
            installation_id=installation_id,
            status=status,
            duration_seconds=duration,
            phases_completed=phases,
            errors_encountered=errors,
            warnings_generated=warnings,
            successful_recoveries=recoveries
        )
        
        # Verify report
        self.assertEqual(report.installation_id, installation_id)
        self.assertEqual(report.status, status)
        self.assertEqual(report.duration_seconds, duration)
        self.assertEqual(report.phases_completed, phases)
        self.assertEqual(report.errors_encountered, errors)
        self.assertEqual(report.warnings_generated, warnings)
        self.assertGreater(report.final_health_score, 0)
        self.assertLessEqual(report.final_health_score, 100)
        
        # Verify database storage
        with sqlite3.connect(str(self.health_reporter.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM installation_reports WHERE id = ?", (installation_id,))
            count = cursor.fetchone()[0]
            self.assertEqual(count, 1)
    
    def test_health_score_calculation(self):
        """Test health score calculation logic."""
        # Test successful installation
        success_report = self.health_reporter.generate_installation_report(
            installation_id="success-test",
            status=InstallationStatus.SUCCESS,
            duration_seconds=1200.0,
            errors_encountered=[],
            warnings_generated=[],
            user_interventions=0
        )
        self.assertGreaterEqual(success_report.final_health_score, 90)
        
        # Test failed installation with errors
        failure_report = self.health_reporter.generate_installation_report(
            installation_id="failure-test",
            status=InstallationStatus.FAILURE,
            duration_seconds=3600.0,
            errors_encountered=[
                {"type": "error1", "category": "system"},
                {"type": "error2", "category": "network"}
            ],
            warnings_generated=["warning1", "warning2"],
            user_interventions=2
        )
        self.assertLess(failure_report.final_health_score, 70)
    
    def test_error_pattern_tracking(self):
        """Test error pattern tracking functionality."""
        errors = [
            {"type": "network_timeout", "category": "network", "installation_id": "test1"},
            {"type": "permission_denied", "category": "permission", "installation_id": "test2"},
            {"type": "network_timeout", "category": "network", "installation_id": "test3"}
        ]
        
        self.health_reporter.track_error_patterns(errors)
        
        # Verify database storage
        with sqlite3.connect(str(self.health_reporter.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM error_patterns")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 3)
            
            # Check specific error type count
            cursor.execute("SELECT COUNT(*) FROM error_patterns WHERE error_type = ?", ("network_timeout",))
            network_count = cursor.fetchone()[0]
            self.assertEqual(network_count, 2)
    
    def test_recovery_stats_tracking(self):
        """Test recovery method statistics tracking."""
        # Simulate recovery attempts
        recoveries = [
            {"method": "retry_with_backoff", "success": True, "execution_time": 5.0, "error_types_handled": ["network"]},
            {"method": "retry_with_backoff", "success": True, "execution_time": 3.0, "error_types_handled": ["network"]},
            {"method": "retry_with_backoff", "success": False, "execution_time": 10.0, "error_types_handled": ["network"]},
            {"method": "alternative_source", "success": True, "execution_time": 15.0, "error_types_handled": ["download"]}
        ]
        
        for recovery in recoveries:
            self.health_reporter._update_recovery_stats(recovery)
        
        # Check cache
        self.assertIn("retry_with_backoff", self.health_reporter.recovery_stats_cache)
        self.assertIn("alternative_source", self.health_reporter.recovery_stats_cache)
        
        retry_stats = self.health_reporter.recovery_stats_cache["retry_with_backoff"]
        self.assertEqual(retry_stats.total_attempts, 3)
        self.assertEqual(retry_stats.successful_attempts, 2)
        self.assertAlmostEqual(retry_stats.success_rate, 2/3, places=2)
        
        # Check database
        with sqlite3.connect(str(self.health_reporter.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM recovery_stats")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 2)
    
    def test_trend_analysis_insufficient_data(self):
        """Test trend analysis with insufficient data."""
        # Generate trend analysis with no data
        trend_analysis = self.health_reporter.generate_trend_analysis()
        
        self.assertEqual(trend_analysis.total_installations, 0)
        self.assertEqual(trend_analysis.success_rate, 0.0)
        self.assertIn("Insufficient data", trend_analysis.recommendations[0])
    
    def test_trend_analysis_with_data(self):
        """Test trend analysis with sufficient data."""
        # Generate test installation reports
        for i in range(10):
            status = InstallationStatus.SUCCESS if i < 8 else InstallationStatus.FAILURE
            self.health_reporter.generate_installation_report(
                installation_id=f"test-{i}",
                status=status,
                duration_seconds=1200.0 + i * 100,
                errors_encountered=[{"type": f"error_{i}", "category": "system"}] if status == InstallationStatus.FAILURE else []
            )
        
        # Generate trend analysis
        trend_analysis = self.health_reporter.generate_trend_analysis()
        
        self.assertEqual(trend_analysis.total_installations, 10)
        self.assertEqual(trend_analysis.success_rate, 0.8)  # 8 out of 10 successful
        self.assertGreater(trend_analysis.average_installation_time, 1200.0)
        self.assertIsInstance(trend_analysis.most_common_errors, list)
        self.assertIsInstance(trend_analysis.recommendations, list)
    
    def test_export_metrics_json(self):
        """Test metrics export in JSON format."""
        # Generate test data
        self.health_reporter.generate_installation_report(
            installation_id="export-test",
            status=InstallationStatus.SUCCESS,
            duration_seconds=1500.0
        )
        
        # Export metrics
        json_export = self.health_reporter.export_metrics(format="json")
        
        # Verify JSON structure
        export_data = json.loads(json_export)
        self.assertIn("export_timestamp", export_data)
        self.assertIn("installation_reports", export_data)
        self.assertIn("error_patterns", export_data)
        self.assertIn("recovery_stats", export_data)
        self.assertIn("performance_metrics", export_data)
        
        # Verify data content
        self.assertEqual(len(export_data["installation_reports"]), 1)
        self.assertEqual(export_data["installation_reports"][0]["id"], "export-test")
    
    def test_export_metrics_csv(self):
        """Test metrics export in CSV format."""
        # Generate test data
        self.health_reporter.generate_installation_report(
            installation_id="csv-test",
            status=InstallationStatus.SUCCESS,
            duration_seconds=1500.0
        )
        
        # Export metrics
        csv_export = self.health_reporter.export_metrics(format="csv")
        
        # Verify CSV structure
        lines = csv_export.strip().split('\n')
        self.assertGreater(len(lines), 1)  # Header + at least one data row
        self.assertIn("installation_id", lines[0])  # Header
        self.assertIn("csv-test", lines[1])  # Data row
    
    def test_dashboard_data_generation(self):
        """Test dashboard data generation."""
        # Generate test data
        for i in range(5):
            self.health_reporter.generate_installation_report(
                installation_id=f"dashboard-test-{i}",
                status=InstallationStatus.SUCCESS,
                duration_seconds=1200.0
            )
        
        # Get dashboard data
        dashboard_data = self.health_reporter.get_dashboard_data()
        
        # Verify structure
        self.assertIn("timestamp", dashboard_data)
        self.assertIn("summary", dashboard_data)
        self.assertIn("trend_analysis", dashboard_data)
        self.assertIn("recovery_effectiveness", dashboard_data)
        self.assertIn("recent_installations", dashboard_data)
        self.assertIn("recommendations", dashboard_data)
        self.assertIn("health_status", dashboard_data)
        
        # Verify summary data
        summary = dashboard_data["summary"]
        self.assertEqual(summary["total_installations_7d"], 5)
        self.assertEqual(summary["success_rate_7d"], 1.0)  # All successful
    
    def test_cleanup_old_data(self):
        """Test old data cleanup functionality."""
        # Generate old test data
        old_time = datetime.now() - timedelta(days=100)
        
        with sqlite3.connect(str(self.health_reporter.db_path)) as conn:
            cursor = conn.cursor()
            
            # Insert old installation report
            cursor.execute('''
                INSERT INTO installation_reports 
                (id, timestamp, status, duration_seconds, final_health_score, user_interventions)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', ("old-install", old_time.isoformat(), "success", 1200.0, 95.0, 0))
            
            # Insert old error pattern
            cursor.execute('''
                INSERT INTO error_patterns 
                (error_type, category, installation_id, timestamp)
                VALUES (?, ?, ?, ?)
            ''', ("old_error", "system", "old-install", old_time.isoformat()))
            
            conn.commit()
        
        # Verify data exists
        with sqlite3.connect(str(self.health_reporter.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM installation_reports WHERE id = ?", ("old-install",))
            self.assertEqual(cursor.fetchone()[0], 1)
        
        # Run cleanup
        self.health_reporter.cleanup_old_data()
        
        # Verify data was removed
        with sqlite3.connect(str(self.health_reporter.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM installation_reports WHERE id = ?", ("old-install",))
            self.assertEqual(cursor.fetchone()[0], 0)
    
    def test_callback_functionality(self):
        """Test callback registration and triggering."""
        report_callback_called = False
        trend_callback_called = False
        
        def report_callback(report):
            nonlocal report_callback_called
            report_callback_called = True
            self.assertIsInstance(report, InstallationReport)
        
        def trend_callback(trend):
            nonlocal trend_callback_called
            trend_callback_called = True
            self.assertIsInstance(trend, TrendAnalysis)
        
        # Register callbacks
        self.health_reporter.add_report_callback(report_callback)
        self.health_reporter.add_trend_callback(trend_callback)
        
        # Generate report (should trigger report callback)
        self.health_reporter.generate_installation_report(
            installation_id="callback-test",
            status=InstallationStatus.SUCCESS,
            duration_seconds=1200.0
        )
        
        # Generate trend analysis (should trigger trend callback)
        self.health_reporter.generate_trend_analysis()
        
        # Verify callbacks were called
        self.assertTrue(report_callback_called)
        self.assertTrue(trend_callback_called)
    
    def test_recovery_effectiveness_rating(self):
        """Test recovery effectiveness rating calculation."""
        # Test highly effective method (>90% success)
        for i in range(10):
            self.health_reporter._update_recovery_stats({
                "method": "highly_effective_method",
                "success": i < 9,  # 9 out of 10 successful
                "execution_time": 5.0,
                "error_types_handled": ["network"]
            })
        
        stats = self.health_reporter.recovery_stats_cache["highly_effective_method"]
        self.assertEqual(stats.effectiveness_rating, RecoveryEffectiveness.EFFECTIVE)  # 90% = EFFECTIVE
        
        # Test ineffective method (<50% success)
        for i in range(10):
            self.health_reporter._update_recovery_stats({
                "method": "ineffective_method",
                "success": i < 4,  # 4 out of 10 successful
                "execution_time": 10.0,
                "error_types_handled": ["system"]
            })
        
        stats = self.health_reporter.recovery_stats_cache["ineffective_method"]
        self.assertEqual(stats.effectiveness_rating, RecoveryEffectiveness.INEFFECTIVE)
    
    def test_hardware_correlation_analysis(self):
        """Test hardware correlation analysis."""
        # Create mock hardware profiles
        high_end_hw = {
            "gpu": {"model": "RTX4090"},
            "memory": {"total_gb": 32},
            "cpu": {"cores": 16}
        }
        
        low_end_hw = {
            "gpu": {"model": "GTX1060"},
            "memory": {"total_gb": 8},
            "cpu": {"cores": 4}
        }
        
        # Generate installations with different hardware
        installations = [
            ("high1", "2024-01-01T10:00:00", "success", 1200.0, json.dumps(high_end_hw)),
            ("high2", "2024-01-01T11:00:00", "success", 1300.0, json.dumps(high_end_hw)),
            ("low1", "2024-01-01T12:00:00", "failure", 2400.0, json.dumps(low_end_hw)),
            ("low2", "2024-01-01T13:00:00", "failure", 2500.0, json.dumps(low_end_hw))
        ]
        
        # Test correlation analysis
        correlations = self.health_reporter._analyze_hardware_correlation(installations)
        
        # High-end hardware should have better success rate
        self.assertIn("gpu_RTX4090", correlations)
        self.assertIn("gpu_GTX1060", correlations)
        self.assertGreater(correlations["gpu_RTX4090"], correlations["gpu_GTX1060"])


class TestHealthDashboard(unittest.TestCase):
    """Test cases for health dashboard functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.health_reporter = HealthReporter(installation_path=self.test_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def test_dashboard_html_generation(self):
        """Test HTML dashboard generation."""
        # Generate test data
        self.health_reporter.generate_installation_report(
            installation_id="dashboard-html-test",
            status=InstallationStatus.SUCCESS,
            duration_seconds=1200.0
        )
        
        # Generate HTML dashboard
        html_content = create_health_dashboard_html(self.health_reporter)
        
        # Verify HTML structure
        self.assertIn("<!DOCTYPE html>", html_content)
        self.assertIn("Installation Health Dashboard", html_content)
        self.assertIn("Success Rate", html_content)
        self.assertIn("Total Installations", html_content)
        self.assertIn("Recent Installations", html_content)
        self.assertIn("Recommendations", html_content)
    
    def test_dashboard_with_different_health_statuses(self):
        """Test dashboard generation with different health statuses."""
        # Generate mixed success/failure data
        for i in range(10):
            status = InstallationStatus.SUCCESS if i < 5 else InstallationStatus.FAILURE
            self.health_reporter.generate_installation_report(
                installation_id=f"mixed-{i}",
                status=status,
                duration_seconds=1200.0
            )
        
        dashboard_data = self.health_reporter.get_dashboard_data()
        health_status = dashboard_data["health_status"]
        
        # With 50% success rate, should be "poor"
        self.assertEqual(health_status, "poor")


class TestDataStructures(unittest.TestCase):
    """Test cases for data structure serialization and deserialization."""
    
    def test_installation_report_serialization(self):
        """Test InstallationReport to_dict method."""
        report = InstallationReport(
            installation_id="test-123",
            timestamp=datetime.now(),
            status=InstallationStatus.SUCCESS,
            duration_seconds=1500.0,
            hardware_profile=None,
            phases_completed=[InstallationPhase.DETECTION],
            errors_encountered=[],
            warnings_generated=[],
            recovery_attempts=[],
            successful_recoveries=[],
            final_health_score=95.0,
            resource_usage_peak={},
            performance_metrics={},
            user_interventions=0,
            configuration_used={}
        )
        
        report_dict = report.to_dict()
        
        self.assertEqual(report_dict["installation_id"], "test-123")
        self.assertEqual(report_dict["status"], "success")
        self.assertEqual(report_dict["final_health_score"], 95.0)
        self.assertIn("timestamp", report_dict)
    
    def test_trend_analysis_serialization(self):
        """Test TrendAnalysis to_dict method."""
        trend = TrendAnalysis(
            analysis_period=timedelta(days=7),
            total_installations=100,
            success_rate=0.85,
            average_installation_time=1800.0,
            most_common_errors=[],
            performance_trends={},
            hardware_correlation={},
            recovery_effectiveness={},
            recommendations=["Test recommendation"]
        )
        
        trend_dict = trend.to_dict()
        
        self.assertEqual(trend_dict["analysis_period_days"], 7)
        self.assertEqual(trend_dict["total_installations"], 100)
        self.assertEqual(trend_dict["success_rate"], 0.85)
        self.assertIn("generated_at", trend_dict)
    
    def test_recovery_method_stats_serialization(self):
        """Test RecoveryMethodStats to_dict method."""
        stats = RecoveryMethodStats(
            method_name="test_method",
            total_attempts=10,
            successful_attempts=8,
            success_rate=0.8,
            average_execution_time=5.5,
            error_types_handled=["network", "system"],
            last_used=datetime.now(),
            effectiveness_rating=RecoveryEffectiveness.EFFECTIVE
        )
        
        stats_dict = stats.to_dict()
        
        self.assertEqual(stats_dict["method_name"], "test_method")
        self.assertEqual(stats_dict["success_rate"], 0.8)
        self.assertEqual(stats_dict["effectiveness_rating"], "effective")
        self.assertIn("last_used", stats_dict)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestHealthReporter))
    test_suite.addTest(unittest.makeSuite(TestHealthDashboard))
    test_suite.addTest(unittest.makeSuite(TestDataStructures))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Health Reporter Test Summary")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            error_msg = traceback.split('AssertionError: ')[-1].split('\n')[0]
            print(f"- {test}: {error_msg}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            error_msg = traceback.split('\n')[-2]
            print(f"- {test}: {error_msg}")