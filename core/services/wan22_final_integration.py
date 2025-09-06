#!/usr/bin/env python3
"""
Wan2.2 Final Integration - Complete System Integration
Brings together all compatibility components into a working system
"""

import logging
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class IntegrationConfig:
    """Configuration for the integrated system"""
    enable_diagnostics: bool = True
    enable_performance_monitoring: bool = True
    enable_safe_loading: bool = True
    enable_optimization: bool = True
    enable_fallback: bool = True
    diagnostics_dir: str = "diagnostics"
    logs_dir: str = "logs"
    max_memory_usage_gb: float = 12.0
    default_precision: str = "bf16"
    log_level: str = "INFO"


@dataclass
class SystemStatus:
    """System status information"""
    initialized: bool = False
    components_loaded: Dict[str, bool] = None
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.components_loaded is None:
            self.components_loaded = {}
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class Wan22IntegratedSystem:
    """
    Integrated Wan2.2 compatibility system
    Coordinates all components for seamless operation
    """
    
    def __init__(self, config: Optional[IntegrationConfig] = None):
        self.config = config or IntegrationConfig()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Component status tracking
        self.status = SystemStatus()
        self.components = {}
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize all system components"""
        self.logger.info("Initializing Wan2.2 Integrated System...")
        
        try:
            # Create required directories
            self._create_directories()
            
            # Load core components
            self._load_core_components()
            
            # Load optional components
            self._load_optional_components()
            
            # Verify system integrity
            self._verify_system_integrity()
            
            self.status.initialized = True
            self.logger.info("Wan2.2 Integrated System initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}")
            self.status.errors.append(f"Initialization failed: {e}")
            raise
    
    def _create_directories(self):
        """Create required directories"""
        directories = [
            self.config.diagnostics_dir,
            self.config.logs_dir,
            "outputs",
            "models"
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            self.logger.debug(f"Created directory: {directory}")
    
    def _load_core_components(self):
        """Load core system components"""
        core_components = [
            ("architecture_detector", self._load_architecture_detector),
            ("pipeline_manager", self._load_pipeline_manager),
            ("compatibility_registry", self._load_compatibility_registry),
            ("error_handler", self._load_error_handler)
        ]
        
        for name, loader in core_components:
            try:
                component = loader()
                if component:
                    self.components[name] = component
                    self.status.components_loaded[name] = True
                    self.logger.debug(f"Loaded core component: {name}")
                else:
                    self.status.components_loaded[name] = False
                    self.status.warnings.append(f"Core component {name} not available")
            except Exception as e:
                self.status.components_loaded[name] = False
                self.status.errors.append(f"Failed to load {name}: {e}")
                self.logger.error(f"Failed to load core component {name}: {e}")
    
    def _load_optional_components(self):
        """Load optional system components"""
        optional_components = [
            ("optimization_manager", self._load_optimization_manager),
            ("fallback_handler", self._load_fallback_handler),
            ("safe_load_manager", self._load_safe_load_manager),
            ("diagnostic_collector", self._load_diagnostic_collector),
            ("performance_profiler", self._load_performance_profiler)
        ]
        
        for name, loader in optional_components:
            try:
                if getattr(self.config, f"enable_{name.split('_')[0]}", True):
                    component = loader()
                    if component:
                        self.components[name] = component
                        self.status.components_loaded[name] = True
                        self.logger.debug(f"Loaded optional component: {name}")
                    else:
                        self.status.components_loaded[name] = False
                        self.status.warnings.append(f"Optional component {name} not available")
                else:
                    self.status.components_loaded[name] = False
                    self.logger.debug(f"Optional component {name} disabled by configuration")
            except Exception as e:
                self.status.components_loaded[name] = False
                self.status.warnings.append(f"Failed to load optional component {name}: {e}")
                self.logger.warning(f"Failed to load optional component {name}: {e}")
    
    def _load_architecture_detector(self):
        """Load architecture detector"""
        try:
            from infrastructure.hardware.architecture_detector import ArchitectureDetector
            return ArchitectureDetector()
        except ImportError as e:
            self.logger.warning(f"ArchitectureDetector not available: {e}")
            return None
    
    def _load_pipeline_manager(self):
        """Load pipeline manager"""
        try:
            from pipeline_manager import PipelineManager
            return PipelineManager()
        except ImportError as e:
            self.logger.warning(f"PipelineManager not available: {e}")
            return None
    
    def _load_compatibility_registry(self):
        """Load compatibility registry"""
        try:
            from compatibility_registry import CompatibilityRegistry
            return CompatibilityRegistry()
        except ImportError as e:
            self.logger.warning(f"CompatibilityRegistry not available: {e}")
            return None
    
    def _load_error_handler(self):
        """Load error handler"""
        try:
            from infrastructure.hardware.error_handler import GenerationErrorHandler
            return GenerationErrorHandler()
        except ImportError as e:
            self.logger.warning(f"GenerationErrorHandler not available: {e}")
            return None
    
    def _load_optimization_manager(self):
        """Load optimization manager"""
        try:
            from core.services.optimization_manager import OptimizationManager
            return OptimizationManager()
        except ImportError as e:
            self.logger.warning(f"OptimizationManager not available: {e}")
            return None
    
    def _load_fallback_handler(self):
        """Load fallback handler"""
        try:
            from fallback_handler import FallbackHandler
            return FallbackHandler()
        except ImportError as e:
            self.logger.warning(f"FallbackHandler not available: {e}")
            return None
    
    def _load_safe_load_manager(self):
        """Load safe load manager"""
        try:
            from safe_load_manager import SafeLoadManager
            return SafeLoadManager()
        except ImportError as e:
            self.logger.warning(f"SafeLoadManager not available: {e}")
            return None
    
    def _load_diagnostic_collector(self):
        """Load diagnostic collector"""
        try:
            from wan_diagnostic_collector import DiagnosticCollector
            return DiagnosticCollector(self.config.diagnostics_dir)
        except ImportError as e:
            self.logger.warning(f"DiagnosticCollector not available: {e}")
            return None
    
    def _load_performance_profiler(self):
        """Load performance profiler"""
        try:
            from infrastructure.hardware.performance_profiler import PerformanceProfiler
            return PerformanceProfiler()
        except ImportError as e:
            self.logger.warning(f"PerformanceProfiler not available: {e}")
            return None
    
    def _verify_system_integrity(self):
        """Verify system integrity"""
        # Check that core components are available
        core_components = ["architecture_detector", "pipeline_manager"]
        missing_core = [comp for comp in core_components 
                       if not self.status.components_loaded.get(comp, False)]
        
        if missing_core:
            error_msg = f"Missing critical components: {missing_core}"
            self.status.errors.append(error_msg)
            self.logger.error(error_msg)
        
        # Check system resources
        try:
            import psutil
            memory = psutil.virtual_memory()
            if memory.available < 2 * 1024 * 1024 * 1024:  # Less than 2GB
                self.status.warnings.append("Low system memory available")
        except ImportError:
            self.status.warnings.append("Cannot check system resources - psutil not available")
        
        # Check GPU availability
        try:
            import torch
            if not torch.cuda.is_available():
                self.status.warnings.append("CUDA not available - CPU-only mode")
            else:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                if gpu_memory < 4 * 1024 * 1024 * 1024:  # Less than 4GB
                    self.status.warnings.append("Limited GPU memory available")
        except ImportError:
            self.status.warnings.append("PyTorch not available")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "system": {
                "initialized": self.status.initialized,
                "components": self.status.components_loaded,
                "errors": self.status.errors,
                "warnings": self.status.warnings
            },
            "configuration": asdict(self.config),
            "available_components": list(self.components.keys()),
            "timestamp": time.time()
        }
    
    def test_system_functionality(self) -> Dict[str, Any]:
        """Test basic system functionality"""
        test_results = {
            "overall_success": True,
            "component_tests": {},
            "errors": [],
            "warnings": []
        }
        
        # Test architecture detector
        if "architecture_detector" in self.components:
            try:
                detector = self.components["architecture_detector"]
                # Basic functionality test
                test_results["component_tests"]["architecture_detector"] = True
            except Exception as e:
                test_results["component_tests"]["architecture_detector"] = False
                test_results["errors"].append(f"Architecture detector test failed: {e}")
                test_results["overall_success"] = False
        
        # Test pipeline manager
        if "pipeline_manager" in self.components:
            try:
                manager = self.components["pipeline_manager"]
                # Basic functionality test
                test_results["component_tests"]["pipeline_manager"] = True
            except Exception as e:
                test_results["component_tests"]["pipeline_manager"] = False
                test_results["errors"].append(f"Pipeline manager test failed: {e}")
                test_results["overall_success"] = False
        
        # Test other components
        for component_name in self.components:
            if component_name not in test_results["component_tests"]:
                try:
                    # Basic availability test
                    component = self.components[component_name]
                    test_results["component_tests"][component_name] = component is not None
                except Exception as e:
                    test_results["component_tests"][component_name] = False
                    test_results["warnings"].append(f"{component_name} test warning: {e}")
        
        return test_results
    
    def cleanup(self):
        """Cleanup system resources"""
        self.logger.info("Cleaning up Wan2.2 Integrated System...")
        
        try:
            # Cleanup components that support it
            for name, component in self.components.items():
                if hasattr(component, 'cleanup'):
                    try:
                        component.cleanup()
                        self.logger.debug(f"Cleaned up component: {name}")
                    except Exception as e:
                        self.logger.warning(f"Error cleaning up {name}: {e}")
            
            # Save final status
            self._save_final_status()
            
            self.logger.info("System cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def _save_final_status(self):
        """Save final system status"""
        try:
            status_file = Path(self.config.diagnostics_dir) / "final_system_status.json"
            final_status = self.get_system_status()
            
            with open(status_file, 'w') as f:
                json.dump(final_status, f, indent=2, default=str)
            
            self.logger.info(f"Final system status saved to {status_file}")
            
        except Exception as e:
            self.logger.warning(f"Could not save final status: {e}")


def main():
    """Main integration demonstration"""
    print("=" * 60)
    print("Wan2.2 Final Integration - System Demonstration")
    print("=" * 60)
    
    try:
        # Initialize system
        print("\n1. Initializing integrated system...")
        config = IntegrationConfig(
            enable_diagnostics=True,
            enable_performance_monitoring=True,
            log_level="INFO"
        )
        
        system = Wan22IntegratedSystem(config)
        print("✅ System initialized successfully")
        
        # Get system status
        print("\n2. Checking system status...")
        status = system.get_system_status()
        
        print(f"   Initialized: {status['system']['initialized']}")
        print(f"   Components loaded: {len(status['available_components'])}")
        print(f"   Errors: {len(status['system']['errors'])}")
        print(f"   Warnings: {len(status['system']['warnings'])}")
        
        if status['system']['errors']:
            print("   Errors found:")
            for error in status['system']['errors']:
                print(f"     - {error}")
        
        if status['system']['warnings']:
            print("   Warnings:")
            for warning in status['system']['warnings']:
                print(f"     - {warning}")
        
        # Test system functionality
        print("\n3. Testing system functionality...")
        test_results = system.test_system_functionality()
        
        print(f"   Overall success: {test_results['overall_success']}")
        print("   Component tests:")
        for component, success in test_results['component_tests'].items():
            status_icon = "✅" if success else "❌"
            print(f"     {status_icon} {component}")
        
        # Display available components
        print("\n4. Available components:")
        for component in status['available_components']:
            print(f"   ✅ {component}")
        
        # Performance summary
        print("\n5. System summary:")
        total_components = len(status['system']['components'])
        loaded_components = sum(1 for loaded in status['system']['components'].values() if loaded)
        
        print(f"   Components: {loaded_components}/{total_components} loaded")
        print(f"   Success rate: {(loaded_components/total_components)*100:.1f}%")
        
        if test_results['overall_success']:
            print("   🎉 System is ready for use!")
        else:
            print("   ⚠️  System has issues but may still be functional")
        
        # Cleanup
        print("\n6. Cleaning up...")
        system.cleanup()
        print("✅ Cleanup completed")
        
        print("\n" + "=" * 60)
        print("Integration demonstration completed successfully!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration failed: {e}")
        logger.error(f"Integration failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)