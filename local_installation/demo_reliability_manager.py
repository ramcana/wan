"""
Demonstration of ReliabilityManager Central Coordination System

This script demonstrates the key features of the ReliabilityManager:
- Component wrapping with reliability enhancements
- Failure handling coordination
- Recovery strategy selection and execution
- Reliability metrics collection and analysis
- Component health monitoring and tracking
"""

import logging
import time
import tempfile
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the ReliabilityManager and related components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from reliability_manager import ReliabilityManager, ComponentType, RecoveryStrategy


class DemoModelDownloader:
    """Demo component that simulates a model downloader with various failure modes."""
    
    def __init__(self, name="DemoModelDownloader"):
        self.name = name
        self.download_count = 0
        self.failure_mode = None
    
    def download_models(self):
        """Simulate model download with potential failures."""
        self.download_count += 1
        
        if self.failure_mode == "network_timeout":
            raise RuntimeError("connection timeout during model download")
        elif self.failure_mode == "missing_method":
            raise AttributeError("'DemoModelDownloader' object has no attribute 'get_required_models'")
        elif self.failure_mode == "model_validation":
            raise RuntimeError("model validation failed for wan2.2/t2v-a14b")
        elif self.failure_mode == "intermittent" and self.download_count % 3 == 0:
            raise RuntimeError("intermittent network failure")
        
        return f"Successfully downloaded models (attempt {self.download_count})"
    
    def verify_models(self):
        """Simulate model verification."""
        if self.failure_mode == "model_validation":
            raise RuntimeError("model integrity check failed")
        return "All models verified successfully"
    
    def set_failure_mode(self, mode):
        """Set the failure mode for testing."""
        self.failure_mode = mode


class DemoDependencyManager:
    """Demo component that simulates a dependency manager."""
    
    def __init__(self, name="DemoDependencyManager"):
        self.name = name
        self.install_count = 0
    
    def install_packages(self):
        """Simulate package installation."""
        self.install_count += 1
        if self.install_count <= 2:
            raise RuntimeError("package installation failed - network error")
        return f"Packages installed successfully (attempt {self.install_count})"
    
    def create_virtual_environment(self):
        """Simulate virtual environment creation."""
        return "Virtual environment created successfully"


def demonstrate_basic_wrapping():
    """Demonstrate basic component wrapping and health tracking."""
    print("\n" + "="*60)
    print("DEMONSTRATION: Basic Component Wrapping")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create ReliabilityManager
        manager = ReliabilityManager(temp_dir, logger)
        
        # Create demo components
        model_downloader = DemoModelDownloader()
        dependency_manager = DemoDependencyManager()
        
        # Wrap components
        print("Wrapping components with reliability enhancements...")
        wrapped_downloader = manager.wrap_component(
            model_downloader, 
            component_type="model_downloader",
            component_id="demo_model_downloader"
        )
        wrapped_dependency = manager.wrap_component(
            dependency_manager,
            component_type="dependency_manager", 
            component_id="demo_dependency_manager"
        )
        
        # Show initial metrics
        metrics = manager.get_reliability_metrics()
        print(f"Initial metrics: {metrics.total_components} components wrapped")
        
        # Perform some operations
        print("\nPerforming operations...")
        try:
            result = wrapped_downloader.download_models()
            print(f"✓ {result}")
        except Exception as e:
            print(f"✗ Download failed: {e}")
        
        try:
            result = wrapped_dependency.create_virtual_environment()
            print(f"✓ {result}")
        except Exception as e:
            print(f"✗ Environment creation failed: {e}")
        
        # Show updated metrics
        metrics = manager.get_reliability_metrics()
        print(f"\nUpdated metrics:")
        print(f"  Total method calls: {metrics.total_method_calls}")
        print(f"  Successful calls: {metrics.successful_calls}")
        print(f"  Failed calls: {metrics.failed_calls}")
        print(f"  Success rate: {metrics.successful_calls / max(metrics.total_method_calls, 1) * 100:.1f}%")
        
        manager.shutdown()


def demonstrate_failure_recovery():
    """Demonstrate failure handling and recovery strategies."""
    print("\n" + "="*60)
    print("DEMONSTRATION: Failure Recovery")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = ReliabilityManager(temp_dir, logger)
        
        # Create component with failure modes
        model_downloader = DemoModelDownloader()
        wrapped_downloader = manager.wrap_component(
            model_downloader,
            component_id="failing_downloader"
        )
        
        # Test different failure modes
        failure_modes = [
            ("network_timeout", "Network timeout failure"),
            ("missing_method", "Missing method failure"),
            ("model_validation", "Model validation failure"),
            ("intermittent", "Intermittent failures")
        ]
        
        for mode, description in failure_modes:
            print(f"\n--- Testing {description} ---")
            model_downloader.set_failure_mode(mode)
            
            # Attempt operations multiple times to trigger recovery
            for attempt in range(3):
                try:
                    result = wrapped_downloader.download_models()
                    print(f"  Attempt {attempt + 1}: ✓ {result}")
                    break
                except Exception as e:
                    print(f"  Attempt {attempt + 1}: ✗ {e}")
                    # The ReliabilityManager will handle the failure and attempt recovery
        
        # Show recovery statistics
        recovery_history = manager.get_recovery_history()
        print(f"\nRecovery attempts made: {len(recovery_history)}")
        
        for session in recovery_history[-3:]:  # Show last 3 recovery sessions
            print(f"  - {session.strategy.value}: {'✓' if session.success else '✗'} {session.details}")
        
        manager.shutdown()


def demonstrate_health_monitoring():
    """Demonstrate component health monitoring."""
    print("\n" + "="*60)
    print("DEMONSTRATION: Health Monitoring")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = ReliabilityManager(temp_dir, logger)
        
        # Create components with different reliability profiles
        reliable_component = DemoModelDownloader("ReliableDownloader")
        unreliable_component = DemoModelDownloader("UnreliableDownloader")
        unreliable_component.set_failure_mode("intermittent")
        
        # Wrap components
        reliable_wrapper = manager.wrap_component(reliable_component, component_id="reliable")
        unreliable_wrapper = manager.wrap_component(unreliable_component, component_id="unreliable")
        
        # Start monitoring
        manager.start_monitoring()
        
        print("Simulating operations over time...")
        
        # Simulate operations over time
        for i in range(10):
            print(f"  Operation batch {i + 1}")
            
            # Reliable component should mostly succeed
            try:
                reliable_wrapper.download_models()
                print("    Reliable: ✓")
            except Exception as e:
                print(f"    Reliable: ✗ {e}")
            
            # Unreliable component will fail intermittently
            try:
                unreliable_wrapper.download_models()
                print("    Unreliable: ✓")
            except Exception as e:
                print(f"    Unreliable: ✗ {e}")
            
            time.sleep(0.1)  # Small delay
        
        # Check component health
        print("\nComponent Health Status:")
        all_health = manager.get_all_component_health()
        
        for comp_id, health in all_health.items():
            status = "HEALTHY" if health.is_healthy else "UNHEALTHY"
            print(f"  {comp_id}: {status}")
            print(f"    Success rate: {health.success_rate:.2f}")
            print(f"    Total calls: {health.total_calls}")
            print(f"    Failed calls: {health.failed_calls}")
            print(f"    Consecutive failures: {health.consecutive_failures}")
        
        # Show overall system metrics
        metrics = manager.get_reliability_metrics()
        print(f"\nOverall System Health:")
        print(f"  Healthy components: {metrics.healthy_components}/{metrics.total_components}")
        print(f"  System uptime: {metrics.uptime_percentage:.1f}%")
        print(f"  Recovery attempts: {metrics.recovery_attempts}")
        print(f"  Successful recoveries: {metrics.successful_recoveries}")
        
        manager.stop_monitoring()
        manager.shutdown()


def demonstrate_metrics_and_reporting():
    """Demonstrate metrics collection and reporting."""
    print("\n" + "="*60)
    print("DEMONSTRATION: Metrics and Reporting")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = ReliabilityManager(temp_dir, logger)
        
        # Create multiple components
        components = []
        for i in range(3):
            comp = DemoModelDownloader(f"Component{i}")
            if i == 2:  # Make one component unreliable
                comp.set_failure_mode("intermittent")
            
            wrapper = manager.wrap_component(comp, component_id=f"comp_{i}")
            components.append((comp, wrapper))
        
        # Simulate various operations
        print("Simulating system activity...")
        for round_num in range(5):
            print(f"  Activity round {round_num + 1}")
            
            for i, (comp, wrapper) in enumerate(components):
                try:
                    wrapper.download_models()
                    wrapper.verify_models()
                except Exception as e:
                    pass  # Failures are handled by ReliabilityManager
        
        # Generate comprehensive report
        print("\nGenerating reliability report...")
        report_path = manager.export_reliability_report()
        
        if report_path and Path(report_path).exists():
            print(f"✓ Report generated: {report_path}")
            
            # Show some key metrics from the report
            import json
            with open(report_path, 'r') as f:
                report_data = json.load(f)
            
            metrics = report_data['metrics']
            print(f"\nKey Metrics from Report:")
            print(f"  Total components: {metrics['total_components']}")
            print(f"  Total method calls: {metrics['total_method_calls']}")
            print(f"  Success rate: {metrics['successful_calls'] / max(metrics['total_method_calls'], 1) * 100:.1f}%")
            print(f"  Average response time: {metrics['average_response_time']:.3f}s")
            print(f"  System uptime: {metrics['uptime_percentage']:.1f}%")
            
            print(f"\nComponent Health Summary:")
            for comp_id, health in report_data['component_health'].items():
                status = "HEALTHY" if health['is_healthy'] else "UNHEALTHY"
                print(f"  {comp_id}: {status} (Success rate: {health['success_rate']:.2f})")
        
        manager.shutdown()


def main():
    """Run all demonstrations."""
    print("ReliabilityManager Central Coordination System Demo")
    print("This demo shows the key capabilities of the ReliabilityManager")
    
    try:
        demonstrate_basic_wrapping()
        demonstrate_failure_recovery()
        demonstrate_health_monitoring()
        demonstrate_metrics_and_reporting()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nThe ReliabilityManager provides:")
        print("✓ Automatic component wrapping with reliability enhancements")
        print("✓ Intelligent failure detection and recovery coordination")
        print("✓ Multiple recovery strategies (retry, fallback, alternative sources)")
        print("✓ Real-time component health monitoring")
        print("✓ Comprehensive metrics collection and analysis")
        print("✓ Detailed reporting and analytics")
        print("✓ Thread-safe operation with background monitoring")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()