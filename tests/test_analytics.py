"""
Unit tests for the analytics and optimization system.

Tests anonymous usage analytics, optimization suggestions,
performance benchmarking, and failure pattern identification.
"""

import pytest
import time
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from startup_manager.analytics import (
    AnalyticsEngine,
    SystemProfile,
    FailurePattern,
    OptimizationSuggestion,
    BenchmarkResult,
    UsageAnalytics,
    OptimizationCategory,
    OptimizationPriority,
    get_analytics_engine
)
from startup_manager.performance_monitor import (
    PerformanceMonitor,
    StartupSession,
    TimingMetric,
    ResourceSnapshot
)


class TestSystemProfile:
    """Test SystemProfile functionality."""
    
    @patch('platform.system')
    @patch('platform.version')
    @patch('platform.python_version')
    @patch('psutil.cpu_count')
    @patch('psutil.virtual_memory')
    @patch('psutil.cpu_freq')
    def test_create_anonymous_profile(
        self,
        mock_cpu_freq,
        mock_memory,
        mock_cpu_count,
        mock_python_version,
        mock_version,
        mock_system
    ):
        """Test creating anonymous system profile."""
        # Mock system information
        mock_system.return_value = "Windows"
        mock_version.return_value = "10.0.19041"
        mock_python_version.return_value = "3.9.7"
        mock_cpu_count.return_value = 8
        mock_memory.return_value = Mock(total=16 * 1024**3)  # 16GB
        mock_cpu_freq.return_value = Mock(max=3200.0)
        
        profile = SystemProfile.create_anonymous_profile()
        
        assert profile.os_type == "Windows"
        assert profile.os_version == "10.0.19041"
        assert profile.python_version == "3.9.7"
        assert profile.cpu_count == 8
        assert profile.memory_gb == 16.0
        assert profile.cpu_freq_mhz == 3200.0
        assert len(profile.profile_hash) == 16
        assert profile.disk_type in ["SSD", "HDD", "Unknown"]
    
    def test_profile_hash_consistency(self):
        """Test that profile hash is consistent for same system."""
        with patch.multiple(
            'platform',
            system=Mock(return_value="Linux"),
            python_version=Mock(return_value="3.9.0")
        ), patch.multiple(
            'psutil',
            cpu_count=Mock(return_value=4),
            virtual_memory=Mock(return_value=Mock(total=8 * 1024**3))
        ):
            profile1 = SystemProfile.create_anonymous_profile()
            profile2 = SystemProfile.create_anonymous_profile()
            
            assert profile1.profile_hash == profile2.profile_hash


class TestAnalyticsEngine:
    """Test AnalyticsEngine functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.performance_monitor = Mock(spec=PerformanceMonitor)
        self.performance_monitor.sessions = []
        
        # Mock performance stats
        mock_stats = Mock()
        mock_stats.total_sessions = 10
        mock_stats.success_rate = 0.8
        mock_stats.average_duration = 2.5
        mock_stats.phase_averages = {
            "environment_validation": 0.8,
            "port_management": 0.5,
            "process_startup": 1.0,
            "health_verification": 0.2
        }
        self.performance_monitor.get_performance_stats.return_value = mock_stats
        
        self.analytics = AnalyticsEngine(
            performance_monitor=self.performance_monitor,
            data_dir=self.temp_dir,
            enable_analytics=True
        )
    
    def test_analytics_engine_initialization(self):
        """Test analytics engine initialization."""
        assert self.analytics.performance_monitor == self.performance_monitor
        assert self.analytics.data_dir == Path(self.temp_dir)
        assert self.analytics.enable_analytics is True
        assert isinstance(self.analytics.system_profile, SystemProfile)
        assert len(self.analytics.failure_patterns) >= 0
        assert len(self.analytics.optimization_suggestions) >= 0
    
    def test_collect_session_analytics_disabled(self):
        """Test analytics collection when disabled."""
        self.analytics.enable_analytics = False
        
        session = StartupSession(
            session_id="test",
            start_time=time.time(),
            errors=["Test error"]
        )
        
        initial_patterns = len(self.analytics.failure_patterns)
        self.analytics.collect_session_analytics(session)
        
        # Should not create new patterns when disabled
        assert len(self.analytics.failure_patterns) == initial_patterns
    
    def test_analyze_error_patterns(self):
        """Test error pattern analysis."""
        session = StartupSession(
            session_id="test",
            start_time=time.time(),
            errors=["Port conflict: Address already in use", "Permission denied: Access forbidden"]
        )
        
        self.analytics._analyze_error_patterns(session)
        
        # Should create failure patterns for each error type
        assert len(self.analytics.failure_patterns) >= 2
        
        # Check that patterns contain expected data
        error_types = [pattern.error_type for pattern in self.analytics.failure_patterns.values()]
        assert "Port conflict" in error_types
        assert "Permission denied" in error_types
    
    def test_generate_optimization_suggestions_slow_phase(self):
        """Test optimization suggestion generation for slow phases."""
        # Create session with slow phase
        session = StartupSession(
            session_id="test",
            start_time=time.time(),
            phase_timings={
                "environment_validation": TimingMetric(
                    operation="environment_validation",
                    start_time=time.time(),
                    duration=3.0,  # Slow phase
                    success=True
                )
            }
        )
        
        self.analytics._generate_optimization_suggestions(session)
        
        # Should create optimization suggestion for slow phase
        suggestion_ids = list(self.analytics.optimization_suggestions.keys())
        assert any("optimize_environment_validation" in sid for sid in suggestion_ids)
    
    def test_generate_optimization_suggestions_high_cpu(self):
        """Test optimization suggestion generation for high CPU usage."""
        # Create session with high CPU usage
        session = StartupSession(
            session_id="test",
            start_time=time.time(),
            resource_snapshots=[
                Mock(cpu_percent=85.0, memory_percent=60.0),
                Mock(cpu_percent=90.0, memory_percent=65.0),
                Mock(cpu_percent=88.0, memory_percent=62.0)
            ]
        )
        
        self.analytics._generate_optimization_suggestions(session)
        
        # Should create CPU optimization suggestion
        assert "optimize_cpu_usage" in self.analytics.optimization_suggestions
        
        suggestion = self.analytics.optimization_suggestions["optimize_cpu_usage"]
        assert suggestion.category == OptimizationCategory.SYSTEM_RESOURCES
        assert suggestion.priority in [OptimizationPriority.HIGH, OptimizationPriority.MEDIUM]
    
    def test_generate_optimization_suggestions_high_memory(self):
        """Test optimization suggestion generation for high memory usage."""
        # Create session with high memory usage
        session = StartupSession(
            session_id="test",
            start_time=time.time(),
            resource_snapshots=[
                Mock(cpu_percent=50.0, memory_percent=90.0),
                Mock(cpu_percent=55.0, memory_percent=92.0),
                Mock(cpu_percent=52.0, memory_percent=91.0)
            ]
        )
        
        self.analytics._generate_optimization_suggestions(session)
        
        # Should create memory optimization suggestion
        assert "optimize_memory_usage" in self.analytics.optimization_suggestions
        
        suggestion = self.analytics.optimization_suggestions["optimize_memory_usage"]
        assert suggestion.category == OptimizationCategory.SYSTEM_RESOURCES
        assert suggestion.priority in [OptimizationPriority.HIGH, OptimizationPriority.MEDIUM]
    
    def test_get_optimization_suggestions_filtering(self):
        """Test optimization suggestion filtering."""
        # Add test suggestions with different priorities
        high_priority = OptimizationSuggestion(
            suggestion_id="high_priority",
            category=OptimizationCategory.SYSTEM_RESOURCES,
            priority=OptimizationPriority.HIGH,
            title="High Priority Suggestion",
            description="Test",
            expected_improvement="50%",
            implementation_steps=["Step 1"],
            applicable_systems=[self.analytics.system_profile.profile_hash],
            confidence_score=0.9
        )
        
        low_priority = OptimizationSuggestion(
            suggestion_id="low_priority",
            category=OptimizationCategory.CONFIGURATION,
            priority=OptimizationPriority.LOW,
            title="Low Priority Suggestion",
            description="Test",
            expected_improvement="10%",
            implementation_steps=["Step 1"],
            applicable_systems=[self.analytics.system_profile.profile_hash],
            confidence_score=0.5
        )
        
        self.analytics.optimization_suggestions["high_priority"] = high_priority
        self.analytics.optimization_suggestions["low_priority"] = low_priority
        
        # Get suggestions with medium minimum priority
        suggestions = self.analytics.get_optimization_suggestions(
            max_suggestions=10,
            min_priority=OptimizationPriority.MEDIUM
        )
        
        # Should only return high priority suggestion
        assert len(suggestions) == 1
        assert suggestions[0].suggestion_id == "high_priority"
    
    def test_get_optimization_suggestions_system_compatibility(self):
        """Test optimization suggestion system compatibility filtering."""
        # Add suggestion for different system
        incompatible = OptimizationSuggestion(
            suggestion_id="incompatible",
            category=OptimizationCategory.SYSTEM_RESOURCES,
            priority=OptimizationPriority.HIGH,
            title="Incompatible Suggestion",
            description="Test",
            expected_improvement="50%",
            implementation_steps=["Step 1"],
            applicable_systems=["different_system_hash"],
            confidence_score=0.9
        )
        
        self.analytics.optimization_suggestions["incompatible"] = incompatible
        
        suggestions = self.analytics.get_optimization_suggestions()
        
        # Should not include incompatible suggestion
        suggestion_ids = [s.suggestion_id for s in suggestions]
        assert "incompatible" not in suggestion_ids
    
    def test_run_performance_benchmark(self):
        """Test performance benchmarking."""
        benchmark = self.analytics.run_performance_benchmark()
        
        assert isinstance(benchmark, BenchmarkResult)
        assert benchmark.system_profile == self.analytics.system_profile
        assert benchmark.baseline_duration > 0
        assert benchmark.current_duration >= 0
        assert isinstance(benchmark.improvement_percentage, float)
        assert isinstance(benchmark.phase_comparisons, dict)
        assert benchmark.timestamp is not None
    
    def test_get_usage_analytics(self):
        """Test usage analytics generation."""
        # Add some test data
        self.analytics.failure_patterns["test_pattern"] = FailurePattern(
            pattern_id="test_pattern",
            error_type="Test Error",
            frequency=5,
            affected_systems=[self.analytics.system_profile.profile_hash],
            common_conditions={},
            suggested_fixes=["Fix 1"],
            confidence_score=0.8
        )
        
        analytics = self.analytics.get_usage_analytics()
        
        assert isinstance(analytics, UsageAnalytics)
        assert analytics.total_sessions == 10
        assert analytics.success_rate == 0.8
        assert analytics.average_duration == 2.5
        assert "Test Error" in analytics.common_errors
        assert analytics.common_errors["Test Error"] == 5
        assert len(analytics.system_profiles) > 0
        assert len(analytics.performance_trends) > 0
    
    def test_baseline_optimizations_low_memory(self):
        """Test baseline optimizations for low memory systems."""
        # Mock low memory system during profile creation
        with patch('startup_manager.analytics.SystemProfile.create_anonymous_profile') as mock_profile:
            mock_profile.return_value = SystemProfile(
                os_type="Windows",
                os_version="10.0",
                cpu_count=4,
                cpu_freq_mhz=2400.0,
                memory_gb=4.0,  # Low memory
                disk_type="SSD",
                python_version="3.9.0",
                node_version="v16.0.0",
                profile_hash="test_hash_low_mem"
            )
            
            analytics = AnalyticsEngine(
                performance_monitor=self.performance_monitor,
                data_dir=self.temp_dir
            )
            
            assert "low_memory_optimization" in analytics.optimization_suggestions
            suggestion = analytics.optimization_suggestions["low_memory_optimization"]
            assert suggestion.category == OptimizationCategory.SYSTEM_RESOURCES
    
    def test_baseline_optimizations_low_cpu(self):
        """Test baseline optimizations for low CPU systems."""
        # Mock low CPU system during profile creation
        with patch('startup_manager.analytics.SystemProfile.create_anonymous_profile') as mock_profile:
            mock_profile.return_value = SystemProfile(
                os_type="Linux",
                os_version="5.4.0",
                cpu_count=2,  # Low CPU count
                cpu_freq_mhz=2000.0,
                memory_gb=8.0,
                disk_type="SSD",
                python_version="3.9.0",
                node_version="v16.0.0",
                profile_hash="test_hash_low_cpu"
            )
            
            analytics = AnalyticsEngine(
                performance_monitor=self.performance_monitor,
                data_dir=self.temp_dir
            )
            
            assert "low_cpu_optimization" in analytics.optimization_suggestions
            suggestion = analytics.optimization_suggestions["low_cpu_optimization"]
            assert suggestion.category == OptimizationCategory.SYSTEM_RESOURCES
    
    def test_baseline_optimizations_windows(self):
        """Test baseline optimizations for Windows systems."""
        # Mock Windows system during profile creation
        with patch('startup_manager.analytics.SystemProfile.create_anonymous_profile') as mock_profile:
            mock_profile.return_value = SystemProfile(
                os_type="Windows",
                os_version="10.0.19041",
                cpu_count=8,
                cpu_freq_mhz=3200.0,
                memory_gb=16.0,
                disk_type="SSD",
                python_version="3.9.0",
                node_version="v16.0.0",
                profile_hash="test_hash_windows"
            )
            
            analytics = AnalyticsEngine(
                performance_monitor=self.performance_monitor,
                data_dir=self.temp_dir
            )
            
            assert "windows_optimization" in analytics.optimization_suggestions
            suggestion = analytics.optimization_suggestions["windows_optimization"]
            assert suggestion.category == OptimizationCategory.CONFIGURATION
    
    def test_error_fix_generation(self):
        """Test error fix generation."""
        # Test known error types
        port_fixes = self.analytics._generate_error_fixes("Port conflict detected")
        assert "Kill processes using the port" in port_fixes
        
        permission_fixes = self.analytics._generate_error_fixes("Permission denied")
        assert "Run as administrator" in permission_fixes
        
        module_fixes = self.analytics._generate_error_fixes("Module not found")
        assert "Install missing dependencies" in module_fixes
        
        # Test unknown error type
        unknown_fixes = self.analytics._generate_error_fixes("Unknown error type")
        assert "Check error logs for details" in unknown_fixes
    
    def test_data_persistence(self):
        """Test saving and loading analytics data."""
        # Add test data
        test_pattern = FailurePattern(
            pattern_id="test_pattern",
            error_type="Test Error",
            frequency=3,
            affected_systems=["system1"],
            common_conditions={"test": True},
            suggested_fixes=["Fix 1", "Fix 2"],
            confidence_score=0.7
        )
        
        test_suggestion = OptimizationSuggestion(
            suggestion_id="test_suggestion",
            category=OptimizationCategory.SYSTEM_RESOURCES,
            priority=OptimizationPriority.MEDIUM,
            title="Test Suggestion",
            description="Test description",
            expected_improvement="20%",
            implementation_steps=["Step 1", "Step 2"],
            applicable_systems=["system1"],
            confidence_score=0.8
        )
        
        self.analytics.failure_patterns["test_pattern"] = test_pattern
        self.analytics.optimization_suggestions["test_suggestion"] = test_suggestion
        
        # Save data
        self.analytics._save_analytics_data()
        
        # Create new analytics instance and load data
        new_analytics = AnalyticsEngine(
            performance_monitor=self.performance_monitor,
            data_dir=self.temp_dir
        )
        
        # Verify data was loaded
        assert "test_pattern" in new_analytics.failure_patterns
        assert "test_suggestion" in new_analytics.optimization_suggestions
        
        loaded_pattern = new_analytics.failure_patterns["test_pattern"]
        assert loaded_pattern.error_type == "Test Error"
        assert loaded_pattern.frequency == 3
        
        loaded_suggestion = new_analytics.optimization_suggestions["test_suggestion"]
        assert loaded_suggestion.title == "Test Suggestion"
        assert loaded_suggestion.category == OptimizationCategory.SYSTEM_RESOURCES


class TestGlobalAnalyticsEngine:
    """Test global analytics engine functionality."""
    
    def test_get_analytics_engine_singleton(self):
        """Test that get_analytics_engine returns singleton."""
        mock_monitor = Mock(spec=PerformanceMonitor)
        
        # Reset global analytics
        import startup_manager.analytics
        startup_manager.analytics._global_analytics = None
        
        analytics1 = get_analytics_engine(mock_monitor)
        analytics2 = get_analytics_engine(mock_monitor)
        
        assert analytics1 is analytics2
    
    def test_get_analytics_engine_with_args(self):
        """Test get_analytics_engine with custom arguments."""
        mock_monitor = Mock(spec=PerformanceMonitor)
        
        # Reset global analytics
        import startup_manager.analytics
        startup_manager.analytics._global_analytics = None
        
        analytics = get_analytics_engine(mock_monitor, enable_analytics=False)
        assert analytics.enable_analytics is False


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.performance_monitor = Mock(spec=PerformanceMonitor)
        self.performance_monitor.sessions = []
        
        # Mock performance stats
        mock_stats = Mock()
        mock_stats.total_sessions = 5
        mock_stats.success_rate = 0.6
        mock_stats.average_duration = 4.2
        mock_stats.phase_averages = {
            "environment_validation": 1.5,
            "port_management": 0.8,
            "process_startup": 1.7,
            "health_verification": 0.2
        }
        self.performance_monitor.get_performance_stats.return_value = mock_stats
        
        self.analytics = AnalyticsEngine(
            performance_monitor=self.performance_monitor,
            data_dir=self.temp_dir
        )
    
    def test_complete_analytics_workflow(self):
        """Test complete analytics workflow."""
        # Create session with various issues
        session = StartupSession(
            session_id="test_session",
            start_time=time.time() - 10,
            end_time=time.time(),
            success=False,
            total_duration=5.5,
            phase_timings={
                "environment_validation": TimingMetric(
                    operation="environment_validation",
                    start_time=time.time() - 10,
                    duration=2.5,  # Slow
                    success=True
                ),
                "port_management": TimingMetric(
                    operation="port_management",
                    start_time=time.time() - 7,
                    duration=0.5,
                    success=False,
                    error_message="Port conflict: Address already in use"
                )
            },
            resource_snapshots=[
                Mock(cpu_percent=85.0, memory_percent=90.0),
                Mock(cpu_percent=88.0, memory_percent=92.0),
                Mock(cpu_percent=90.0, memory_percent=89.0)
            ],
            errors=["Port conflict: Address already in use", "High resource usage detected"]
        )
        
        # Collect analytics
        self.analytics.collect_session_analytics(session)
        
        # Verify failure patterns were created
        assert len(self.analytics.failure_patterns) > 0
        error_types = [p.error_type for p in self.analytics.failure_patterns.values()]
        assert "Port conflict" in error_types
        
        # Verify optimization suggestions were generated
        suggestions = self.analytics.get_optimization_suggestions()
        assert len(suggestions) > 0
        
        # Should have suggestions for slow phase, high CPU, and high memory
        suggestion_ids = [s.suggestion_id for s in suggestions]
        assert any("optimize_environment_validation" in sid for sid in suggestion_ids)
        assert "optimize_cpu_usage" in suggestion_ids
        assert "optimize_memory_usage" in suggestion_ids
        
        # Run benchmark
        benchmark = self.analytics.run_performance_benchmark()
        assert benchmark.current_duration > benchmark.baseline_duration  # Performance is worse than baseline
        
        # Get usage analytics
        usage_analytics = self.analytics.get_usage_analytics()
        assert usage_analytics.total_sessions == 5
        assert usage_analytics.success_rate == 0.6
        assert "Port conflict" in usage_analytics.common_errors
    
    def test_optimization_suggestion_prioritization(self):
        """Test optimization suggestion prioritization."""
        # Create multiple sessions with different issues
        sessions = [
            # Session with critical memory issue
            StartupSession(
                session_id="critical_session",
                start_time=time.time(),
                resource_snapshots=[Mock(cpu_percent=50.0, memory_percent=98.0)] * 3,
                errors=["Out of memory error"] * 5  # Repeated error
            ),
            # Session with minor CPU issue
            StartupSession(
                session_id="minor_session",
                start_time=time.time(),
                resource_snapshots=[Mock(cpu_percent=75.0, memory_percent=60.0)] * 3,
                errors=["Minor warning"]
            )
        ]
        
        for session in sessions:
            self.analytics.collect_session_analytics(session)
        
        # Get high priority suggestions only
        high_priority_suggestions = self.analytics.get_optimization_suggestions(
            min_priority=OptimizationPriority.HIGH
        )
        
        # Should prioritize memory optimization over CPU optimization
        assert len(high_priority_suggestions) > 0
        assert any(s.suggestion_id == "optimize_memory_usage" for s in high_priority_suggestions)
        
        # Memory optimization should have higher priority than CPU
        memory_suggestion = next(
            (s for s in self.analytics.optimization_suggestions.values() 
             if s.suggestion_id == "optimize_memory_usage"), None
        )
        cpu_suggestion = next(
            (s for s in self.analytics.optimization_suggestions.values() 
             if s.suggestion_id == "optimize_cpu_usage"), None
        )
        
        if memory_suggestion and cpu_suggestion:
            priority_order = {
                OptimizationPriority.CRITICAL: 4,
                OptimizationPriority.HIGH: 3,
                OptimizationPriority.MEDIUM: 2,
                OptimizationPriority.LOW: 1
            }
            
            memory_priority = priority_order[memory_suggestion.priority]
            cpu_priority = priority_order[cpu_suggestion.priority]
            
            assert memory_priority >= cpu_priority


if __name__ == "__main__":
    pytest.main([__file__])