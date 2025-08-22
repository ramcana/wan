"""
Documentation Integration System for WAN22 System Optimization

This module integrates user documentation with error messages, provides contextual help,
and implements guided setup for first-time users with high-end hardware.

Requirements addressed:
- 12.2: Link relevant documentation sections in error messages
- 12.2: Create contextual help for optimization settings
- 12.2: Add guided setup for first-time users with high-end hardware
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import webbrowser
from urllib.parse import quote


class DocumentationType(Enum):
    """Types of documentation available"""
    USER_GUIDE = "user_guide"
    TROUBLESHOOTING = "troubleshooting"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    API_REFERENCE = "api_reference"
    QUICK_START = "quick_start"


class HardwareProfile(Enum):
    """Hardware profile categories for guided setup"""
    HIGH_END = "high_end"  # RTX 4080+, Threadripper PRO, 128GB+ RAM
    ENTHUSIAST = "enthusiast"  # RTX 3080+, High-end CPU, 64GB+ RAM
    MAINSTREAM = "mainstream"  # RTX 3060+, Mid-range CPU, 32GB+ RAM
    BUDGET = "budget"  # GTX 1660+, Budget CPU, 16GB+ RAM


@dataclass
class DocumentationLink:
    """Represents a link to documentation"""
    title: str
    url: str
    section: str
    description: str
    relevance_score: float
    doc_type: DocumentationType


@dataclass
class ContextualHelp:
    """Contextual help information for UI elements"""
    element_id: str
    title: str
    description: str
    documentation_links: List[DocumentationLink]
    quick_tips: List[str]
    related_settings: List[str]


@dataclass
class GuidedSetupStep:
    """Step in the guided setup process"""
    step_id: str
    title: str
    description: str
    instructions: List[str]
    validation_criteria: Dict[str, Any]
    documentation_links: List[DocumentationLink]
    estimated_time_minutes: int
    required: bool


class DocumentationIntegration:
    """
    Integrates documentation with error messages and provides contextual help
    """
    
    def __init__(self, docs_base_path: str = "docs"):
        """
        Initialize documentation integration system.
        
        Args:
            docs_base_path: Base path for documentation files
        """
        self.docs_base_path = Path(docs_base_path)
        self.logger = logging.getLogger(__name__)
        
        # Documentation mapping
        self._error_documentation_map = self._initialize_error_documentation_map()
        self._contextual_help_map = self._initialize_contextual_help_map()
        self._guided_setup_steps = self._initialize_guided_setup_steps()
        
        # Documentation URLs (can be local files or web URLs)
        self._documentation_urls = {
            DocumentationType.USER_GUIDE: "WAN22_SYSTEM_OPTIMIZATION_USER_GUIDE.md",
            DocumentationType.TROUBLESHOOTING: "WAN22_TROUBLESHOOTING_GUIDE.md",
            DocumentationType.PERFORMANCE_OPTIMIZATION: "WAN22_PERFORMANCE_OPTIMIZATION_GUIDE.md",
            DocumentationType.QUICK_START: "WAN22_SYSTEM_OPTIMIZATION_USER_GUIDE.md#quick-start",
            DocumentationType.API_REFERENCE: "api_reference.md"
        }
    
    def get_error_documentation_links(self, 
                                    error_category: str, 
                                    error_message: str,
                                    context: Optional[Dict[str, Any]] = None) -> List[DocumentationLink]:
        """
        Get relevant documentation links for an error.
        
        Args:
            error_category: Category of the error (e.g., "vram_memory", "model_loading")
            error_message: The error message text
            context: Additional context about the error
            
        Returns:
            List of relevant documentation links
        """
        links = []
        
        # Get base documentation for error category
        if error_category in self._error_documentation_map:
            links.extend(self._error_documentation_map[error_category])
        
        # Add context-specific links
        context_links = self._get_context_specific_links(error_category, error_message, context)
        links.extend(context_links)
        
        # Sort by relevance score
        links.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return links[:5]  # Return top 5 most relevant links
    
    def get_contextual_help(self, element_id: str) -> Optional[ContextualHelp]:
        """
        Get contextual help for a UI element.
        
        Args:
            element_id: Identifier for the UI element
            
        Returns:
            ContextualHelp object or None if not found
        """
        return self._contextual_help_map.get(element_id)
    
    def get_guided_setup_steps(self, hardware_profile: HardwareProfile) -> List[GuidedSetupStep]:
        """
        Get guided setup steps for a specific hardware profile.
        
        Args:
            hardware_profile: Hardware profile category
            
        Returns:
            List of setup steps tailored to the hardware profile
        """
        base_steps = self._guided_setup_steps.copy()
        
        # Customize steps based on hardware profile
        if hardware_profile == HardwareProfile.HIGH_END:
            return self._customize_steps_for_high_end(base_steps)
        elif hardware_profile == HardwareProfile.ENTHUSIAST:
            return self._customize_steps_for_enthusiast(base_steps)
        elif hardware_profile == HardwareProfile.MAINSTREAM:
            return self._customize_steps_for_mainstream(base_steps)
        else:
            return self._customize_steps_for_budget(base_steps)
    
    def create_enhanced_error_message(self, 
                                    original_message: str,
                                    error_category: str,
                                    context: Optional[Dict[str, Any]] = None) -> str:
        """
        Create an enhanced error message with documentation links.
        
        Args:
            original_message: Original error message
            error_category: Category of the error
            context: Additional context
            
        Returns:
            Enhanced error message with documentation links
        """
        # Get relevant documentation links
        doc_links = self.get_error_documentation_links(error_category, original_message, context)
        
        if not doc_links:
            return original_message
        
        # Create enhanced message
        enhanced_message = f"{original_message}\n\n"
        enhanced_message += "📚 **Helpful Resources:**\n"
        
        for link in doc_links:
            enhanced_message += f"• [{link.title}]({link.url}) - {link.description}\n"
        
        # Add quick action if available
        quick_action = self._get_quick_action_for_error(error_category, context)
        if quick_action:
            enhanced_message += f"\n🚀 **Quick Fix:** {quick_action}\n"
        
        return enhanced_message
    
    def open_documentation(self, doc_type: DocumentationType, section: Optional[str] = None):
        """
        Open documentation in the default browser or application.
        
        Args:
            doc_type: Type of documentation to open
            section: Specific section to navigate to
        """
        try:
            url = self._documentation_urls.get(doc_type)
            if not url:
                self.logger.warning(f"No URL configured for documentation type: {doc_type}")
                return
            
            # Add section anchor if specified
            if section:
                url += f"#{quote(section.lower().replace(' ', '-'))}"
            
            # Check if it's a local file
            if not url.startswith(('http://', 'https://')):
                local_path = self.docs_base_path / url
                if local_path.exists():
                    url = f"file://{local_path.absolute()}"
                else:
                    self.logger.warning(f"Documentation file not found: {local_path}")
                    return
            
            webbrowser.open(url)
            self.logger.info(f"Opened documentation: {url}")
            
        except Exception as e:
            self.logger.error(f"Failed to open documentation: {e}")
    
    def _initialize_error_documentation_map(self) -> Dict[str, List[DocumentationLink]]:
        """Initialize mapping of error categories to documentation links"""
        return {
            "vram_memory": [
                DocumentationLink(
                    title="VRAM Management Guide",
                    url="WAN22_SYSTEM_OPTIMIZATION_USER_GUIDE.md#vram-management",
                    section="VRAM Management",
                    description="Complete guide to VRAM detection and optimization",
                    relevance_score=1.0,
                    doc_type=DocumentationType.USER_GUIDE
                ),
                DocumentationLink(
                    title="Memory Optimization Strategies",
                    url="WAN22_PERFORMANCE_OPTIMIZATION_GUIDE.md#memory-management-optimizations",
                    section="Memory Management Optimizations",
                    description="Advanced memory optimization techniques",
                    relevance_score=0.9,
                    doc_type=DocumentationType.PERFORMANCE_OPTIMIZATION
                ),
                DocumentationLink(
                    title="VRAM Detection Failures",
                    url="WAN22_TROUBLESHOOTING_GUIDE.md#vram-detection-failures",
                    section="VRAM Detection Failures",
                    description="Troubleshooting VRAM detection issues",
                    relevance_score=0.8,
                    doc_type=DocumentationType.TROUBLESHOOTING
                )
            ],
            "model_loading": [
                DocumentationLink(
                    title="Model Loading Optimization",
                    url="WAN22_SYSTEM_OPTIMIZATION_USER_GUIDE.md#model-loading-optimization",
                    section="Model Loading Optimization",
                    description="Guide to optimizing model loading performance",
                    relevance_score=1.0,
                    doc_type=DocumentationType.USER_GUIDE
                ),
                DocumentationLink(
                    title="TI2V-5B Loading Issues",
                    url="WAN22_TROUBLESHOOTING_GUIDE.md#ti2v-5b-loading-failures",
                    section="TI2V-5B Loading Failures",
                    description="Specific troubleshooting for TI2V-5B model",
                    relevance_score=0.9,
                    doc_type=DocumentationType.TROUBLESHOOTING
                ),
                DocumentationLink(
                    title="Model-Specific Optimizations",
                    url="WAN22_PERFORMANCE_OPTIMIZATION_GUIDE.md#model-specific-optimizations",
                    section="Model-Specific Optimizations",
                    description="Performance optimizations for specific models",
                    relevance_score=0.8,
                    doc_type=DocumentationType.PERFORMANCE_OPTIMIZATION
                )
            ],
            "quantization": [
                DocumentationLink(
                    title="Quantization Settings",
                    url="WAN22_SYSTEM_OPTIMIZATION_USER_GUIDE.md#quantization-settings",
                    section="Quantization Settings",
                    description="Complete guide to quantization configuration",
                    relevance_score=1.0,
                    doc_type=DocumentationType.USER_GUIDE
                ),
                DocumentationLink(
                    title="Quantization Issues",
                    url="WAN22_TROUBLESHOOTING_GUIDE.md#quantization-issues",
                    section="Quantization Issues",
                    description="Troubleshooting quantization timeouts and failures",
                    relevance_score=0.9,
                    doc_type=DocumentationType.TROUBLESHOOTING
                )
            ],
            "configuration": [
                DocumentationLink(
                    title="Configuration Settings",
                    url="WAN22_SYSTEM_OPTIMIZATION_USER_GUIDE.md#configuration-settings",
                    section="Configuration Settings",
                    description="Complete configuration reference",
                    relevance_score=1.0,
                    doc_type=DocumentationType.USER_GUIDE
                ),
                DocumentationLink(
                    title="Configuration Errors",
                    url="WAN22_TROUBLESHOOTING_GUIDE.md#configuration-errors",
                    section="Configuration Errors",
                    description="Fixing configuration validation issues",
                    relevance_score=0.9,
                    doc_type=DocumentationType.TROUBLESHOOTING
                )
            ],
            "hardware_optimization": [
                DocumentationLink(
                    title="Hardware Optimization",
                    url="WAN22_SYSTEM_OPTIMIZATION_USER_GUIDE.md#hardware-optimization",
                    section="Hardware Optimization",
                    description="Hardware-specific optimization settings",
                    relevance_score=1.0,
                    doc_type=DocumentationType.USER_GUIDE
                ),
                DocumentationLink(
                    title="RTX 4080 Optimizations",
                    url="WAN22_PERFORMANCE_OPTIMIZATION_GUIDE.md#rtx-4080-optimizations",
                    section="RTX 4080 Optimizations",
                    description="Specific optimizations for RTX 4080",
                    relevance_score=0.9,
                    doc_type=DocumentationType.PERFORMANCE_OPTIMIZATION
                )
            ],
            "system_health": [
                DocumentationLink(
                    title="Performance Monitoring",
                    url="WAN22_SYSTEM_OPTIMIZATION_USER_GUIDE.md#performance-monitoring",
                    section="Performance Monitoring",
                    description="System health monitoring and alerts",
                    relevance_score=1.0,
                    doc_type=DocumentationType.USER_GUIDE
                ),
                DocumentationLink(
                    title="System Health Issues",
                    url="WAN22_TROUBLESHOOTING_GUIDE.md#system-health-issues",
                    section="System Health Issues",
                    description="Troubleshooting overheating and instability",
                    relevance_score=0.9,
                    doc_type=DocumentationType.TROUBLESHOOTING
                )
            ]
        }
    
    def _initialize_contextual_help_map(self) -> Dict[str, ContextualHelp]:
        """Initialize contextual help for UI elements"""
        return {
            "vram_optimization_toggle": ContextualHelp(
                element_id="vram_optimization_toggle",
                title="VRAM Optimization",
                description="Automatically optimizes VRAM usage when memory usage exceeds the threshold.",
                documentation_links=[
                    DocumentationLink(
                        title="VRAM Management Guide",
                        url="WAN22_SYSTEM_OPTIMIZATION_USER_GUIDE.md#vram-management",
                        section="VRAM Management",
                        description="Complete VRAM management guide",
                        relevance_score=1.0,
                        doc_type=DocumentationType.USER_GUIDE
                    )
                ],
                quick_tips=[
                    "Enable for automatic memory management",
                    "Recommended threshold: 90% for RTX 4080",
                    "Helps prevent out-of-memory errors"
                ],
                related_settings=["vram_threshold", "cpu_offload_enabled"]
            ),
            "quantization_strategy": ContextualHelp(
                element_id="quantization_strategy",
                title="Quantization Strategy",
                description="Reduces model precision to save VRAM with minimal quality impact.",
                documentation_links=[
                    DocumentationLink(
                        title="Quantization Settings",
                        url="WAN22_SYSTEM_OPTIMIZATION_USER_GUIDE.md#quantization-settings",
                        section="Quantization Settings",
                        description="Quantization configuration guide",
                        relevance_score=1.0,
                        doc_type=DocumentationType.USER_GUIDE
                    )
                ],
                quick_tips=[
                    "bf16: Best quality/performance balance",
                    "int8: Maximum VRAM savings",
                    "none: Highest quality, most VRAM usage"
                ],
                related_settings=["quantization_timeout", "fallback_enabled"]
            ),
            "hardware_optimization": ContextualHelp(
                element_id="hardware_optimization",
                title="Hardware Optimization",
                description="Applies optimizations specific to your detected hardware configuration.",
                documentation_links=[
                    DocumentationLink(
                        title="Hardware Optimization",
                        url="WAN22_SYSTEM_OPTIMIZATION_USER_GUIDE.md#hardware-optimization",
                        section="Hardware Optimization",
                        description="Hardware-specific optimization guide",
                        relevance_score=1.0,
                        doc_type=DocumentationType.USER_GUIDE
                    )
                ],
                quick_tips=[
                    "Automatically detects RTX 4080 and Threadripper PRO",
                    "Optimizes tensor core usage and memory allocation",
                    "Can improve performance by 40-60%"
                ],
                related_settings=["cpu_cores_utilized", "tensor_cores_enabled"]
            ),
            "health_monitoring": ContextualHelp(
                element_id="health_monitoring",
                title="System Health Monitoring",
                description="Continuously monitors system health and provides alerts for potential issues.",
                documentation_links=[
                    DocumentationLink(
                        title="Performance Monitoring",
                        url="WAN22_SYSTEM_OPTIMIZATION_USER_GUIDE.md#performance-monitoring",
                        section="Performance Monitoring",
                        description="System monitoring and health checks",
                        relevance_score=1.0,
                        doc_type=DocumentationType.USER_GUIDE
                    )
                ],
                quick_tips=[
                    "Monitors GPU temperature, VRAM, and CPU usage",
                    "Provides automatic alerts for safety thresholds",
                    "Helps prevent hardware damage"
                ],
                related_settings=["monitoring_interval", "alert_thresholds"]
            )
        }
    
    def _initialize_guided_setup_steps(self) -> List[GuidedSetupStep]:
        """Initialize guided setup steps"""
        return [
            GuidedSetupStep(
                step_id="hardware_detection",
                title="Hardware Detection",
                description="Detect and validate your system hardware configuration",
                instructions=[
                    "The system will automatically detect your hardware",
                    "Verify that your RTX 4080 is detected correctly",
                    "Confirm CPU and RAM specifications",
                    "Check CUDA version compatibility"
                ],
                validation_criteria={
                    "gpu_detected": True,
                    "vram_gb": {"min": 12, "recommended": 16},
                    "cuda_available": True
                },
                documentation_links=[
                    DocumentationLink(
                        title="System Requirements",
                        url="WAN22_SYSTEM_OPTIMIZATION_USER_GUIDE.md#system-requirements",
                        section="System Requirements",
                        description="Hardware requirements and compatibility",
                        relevance_score=1.0,
                        doc_type=DocumentationType.USER_GUIDE
                    )
                ],
                estimated_time_minutes=2,
                required=True
            ),
            GuidedSetupStep(
                step_id="vram_optimization",
                title="VRAM Optimization Setup",
                description="Configure VRAM management for optimal performance",
                instructions=[
                    "Enable automatic VRAM optimization",
                    "Set optimization threshold to 90% for RTX 4080",
                    "Enable CPU offloading for non-critical components",
                    "Test VRAM detection and monitoring"
                ],
                validation_criteria={
                    "vram_optimization_enabled": True,
                    "optimization_threshold": 0.9,
                    "cpu_offload_enabled": True
                },
                documentation_links=[
                    DocumentationLink(
                        title="VRAM Management",
                        url="WAN22_SYSTEM_OPTIMIZATION_USER_GUIDE.md#vram-management",
                        section="VRAM Management",
                        description="VRAM optimization configuration",
                        relevance_score=1.0,
                        doc_type=DocumentationType.USER_GUIDE
                    )
                ],
                estimated_time_minutes=3,
                required=True
            ),
            GuidedSetupStep(
                step_id="quantization_setup",
                title="Quantization Configuration",
                description="Configure model quantization for memory efficiency",
                instructions=[
                    "Select optimal quantization strategy (bf16 recommended)",
                    "Set appropriate timeout values",
                    "Enable fallback to non-quantized mode",
                    "Test quantization with a small model"
                ],
                validation_criteria={
                    "quantization_strategy": "bf16",
                    "timeout_seconds": {"min": 300, "max": 600},
                    "fallback_enabled": True
                },
                documentation_links=[
                    DocumentationLink(
                        title="Quantization Settings",
                        url="WAN22_SYSTEM_OPTIMIZATION_USER_GUIDE.md#quantization-settings",
                        section="Quantization Settings",
                        description="Quantization configuration guide",
                        relevance_score=1.0,
                        doc_type=DocumentationType.USER_GUIDE
                    )
                ],
                estimated_time_minutes=5,
                required=True
            ),
            GuidedSetupStep(
                step_id="hardware_optimization",
                title="Hardware-Specific Optimizations",
                description="Apply optimizations for your specific hardware",
                instructions=[
                    "Enable RTX 4080 tensor core optimizations",
                    "Configure optimal tile sizes for your GPU",
                    "Set up Threadripper PRO multi-core utilization",
                    "Apply memory bandwidth optimizations"
                ],
                validation_criteria={
                    "tensor_cores_enabled": True,
                    "tile_size_optimized": True,
                    "multi_core_enabled": True
                },
                documentation_links=[
                    DocumentationLink(
                        title="Hardware Optimization",
                        url="WAN22_PERFORMANCE_OPTIMIZATION_GUIDE.md#hardware-specific-optimizations",
                        section="Hardware-Specific Optimizations",
                        description="Hardware optimization guide",
                        relevance_score=1.0,
                        doc_type=DocumentationType.PERFORMANCE_OPTIMIZATION
                    )
                ],
                estimated_time_minutes=4,
                required=True
            ),
            GuidedSetupStep(
                step_id="health_monitoring",
                title="System Health Monitoring",
                description="Set up continuous system health monitoring",
                instructions=[
                    "Enable real-time health monitoring",
                    "Configure temperature and usage thresholds",
                    "Set up automatic alerts and notifications",
                    "Test monitoring dashboard"
                ],
                validation_criteria={
                    "monitoring_enabled": True,
                    "thresholds_configured": True,
                    "alerts_enabled": True
                },
                documentation_links=[
                    DocumentationLink(
                        title="Performance Monitoring",
                        url="WAN22_SYSTEM_OPTIMIZATION_USER_GUIDE.md#performance-monitoring",
                        section="Performance Monitoring",
                        description="System monitoring setup guide",
                        relevance_score=1.0,
                        doc_type=DocumentationType.USER_GUIDE
                    )
                ],
                estimated_time_minutes=3,
                required=False
            ),
            GuidedSetupStep(
                step_id="performance_validation",
                title="Performance Validation",
                description="Validate that optimizations are working correctly",
                instructions=[
                    "Run performance benchmarks",
                    "Test model loading with TI2V-5B",
                    "Verify VRAM usage optimization",
                    "Check generation speed improvements"
                ],
                validation_criteria={
                    "benchmarks_passed": True,
                    "model_loading_time": {"max": 300},  # 5 minutes
                    "vram_usage_optimized": True
                },
                documentation_links=[
                    DocumentationLink(
                        title="Performance Benchmarking",
                        url="WAN22_PERFORMANCE_OPTIMIZATION_GUIDE.md#performance-benchmarking",
                        section="Performance Benchmarking",
                        description="Performance validation guide",
                        relevance_score=1.0,
                        doc_type=DocumentationType.PERFORMANCE_OPTIMIZATION
                    )
                ],
                estimated_time_minutes=10,
                required=False
            )
        ]
    
    def _get_context_specific_links(self, 
                                  error_category: str, 
                                  error_message: str,
                                  context: Optional[Dict[str, Any]]) -> List[DocumentationLink]:
        """Get context-specific documentation links"""
        links = []
        
        # Add context-specific links based on error details
        if "rtx 4080" in error_message.lower() or (context and context.get("gpu_model") == "RTX 4080"):
            links.append(DocumentationLink(
                title="RTX 4080 Specific Guide",
                url="WAN22_PERFORMANCE_OPTIMIZATION_GUIDE.md#rtx-4080-optimizations",
                section="RTX 4080 Optimizations",
                description="Optimizations specific to RTX 4080",
                relevance_score=0.8,
                doc_type=DocumentationType.PERFORMANCE_OPTIMIZATION
            ))
        
        if "ti2v-5b" in error_message.lower() or (context and context.get("model_name") == "TI2V-5B"):
            links.append(DocumentationLink(
                title="TI2V-5B Model Guide",
                url="WAN22_PERFORMANCE_OPTIMIZATION_GUIDE.md#ti2v-5b-model-optimizations",
                section="TI2V-5B Model Optimizations",
                description="Specific optimizations for TI2V-5B model",
                relevance_score=0.9,
                doc_type=DocumentationType.PERFORMANCE_OPTIMIZATION
            ))
        
        return links
    
    def _get_quick_action_for_error(self, 
                                  error_category: str, 
                                  context: Optional[Dict[str, Any]]) -> Optional[str]:
        """Get quick action suggestion for an error"""
        quick_actions = {
            "vram_memory": "Try enabling VRAM optimization or reducing batch size",
            "model_loading": "Clear model cache and retry loading",
            "quantization": "Disable quantization or increase timeout",
            "configuration": "Reset configuration to defaults",
            "hardware_optimization": "Run hardware detection again"
        }
        
        return quick_actions.get(error_category)
    
    def _customize_steps_for_high_end(self, steps: List[GuidedSetupStep]) -> List[GuidedSetupStep]:
        """Customize setup steps for high-end hardware"""
        # Add high-end specific optimizations
        for step in steps:
            if step.step_id == "hardware_optimization":
                step.instructions.extend([
                    "Enable aggressive tensor core utilization",
                    "Configure large memory pools for 128GB+ RAM",
                    "Set up NUMA-aware memory allocation"
                ])
            elif step.step_id == "vram_optimization":
                step.instructions.extend([
                    "Configure for 16GB VRAM capacity",
                    "Enable advanced memory optimization techniques"
                ])
        
        return steps
    
    def _customize_steps_for_enthusiast(self, steps: List[GuidedSetupStep]) -> List[GuidedSetupStep]:
        """Customize setup steps for enthusiast hardware"""
        for step in steps:
            if step.step_id == "vram_optimization":
                step.validation_criteria["optimization_threshold"] = 0.85  # More conservative
        
        return steps
    
    def _customize_steps_for_mainstream(self, steps: List[GuidedSetupStep]) -> List[GuidedSetupStep]:
        """Customize setup steps for mainstream hardware"""
        for step in steps:
            if step.step_id == "quantization_setup":
                step.validation_criteria["quantization_strategy"] = "int8"  # More aggressive quantization
        
        return steps
    
    def _customize_steps_for_budget(self, steps: List[GuidedSetupStep]) -> List[GuidedSetupStep]:
        """Customize setup steps for budget hardware"""
        for step in steps:
            if step.step_id == "quantization_setup":
                step.validation_criteria["quantization_strategy"] = "int8"
            elif step.step_id == "vram_optimization":
                step.validation_criteria["optimization_threshold"] = 0.8  # More aggressive optimization
        
        return steps


class EnhancedErrorHandler:
    """
    Enhanced error handler that integrates with documentation system
    """
    
    def __init__(self, documentation_integration: DocumentationIntegration):
        """
        Initialize enhanced error handler.
        
        Args:
            documentation_integration: Documentation integration system
        """
        self.doc_integration = documentation_integration
        self.logger = logging.getLogger(__name__)
    
    def handle_error_with_documentation(self, 
                                      error: Exception,
                                      error_category: str,
                                      context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle error and provide enhanced error information with documentation links.
        
        Args:
            error: The exception that occurred
            error_category: Category of the error
            context: Additional context information
            
        Returns:
            Enhanced error information with documentation links
        """
        # Get documentation links
        doc_links = self.doc_integration.get_error_documentation_links(
            error_category, str(error), context
        )
        
        # Create enhanced error message
        enhanced_message = self.doc_integration.create_enhanced_error_message(
            str(error), error_category, context
        )
        
        # Get contextual help if available
        contextual_help = None
        if context and context.get("ui_element_id"):
            contextual_help = self.doc_integration.get_contextual_help(
                context["ui_element_id"]
            )
        
        return {
            "original_error": str(error),
            "enhanced_message": enhanced_message,
            "documentation_links": [
                {
                    "title": link.title,
                    "url": link.url,
                    "description": link.description,
                    "relevance_score": link.relevance_score
                }
                for link in doc_links
            ],
            "contextual_help": {
                "title": contextual_help.title,
                "description": contextual_help.description,
                "quick_tips": contextual_help.quick_tips
            } if contextual_help else None,
            "quick_action": self.doc_integration._get_quick_action_for_error(error_category, context)
        }


class GuidedSetupManager:
    """
    Manages guided setup process for first-time users
    """
    
    def __init__(self, documentation_integration: DocumentationIntegration):
        """
        Initialize guided setup manager.
        
        Args:
            documentation_integration: Documentation integration system
        """
        self.doc_integration = documentation_integration
        self.logger = logging.getLogger(__name__)
        self.current_step = 0
        self.setup_complete = False
    
    def detect_hardware_profile(self, system_info: Dict[str, Any]) -> HardwareProfile:
        """
        Detect hardware profile based on system information.
        
        Args:
            system_info: System hardware information
            
        Returns:
            Detected hardware profile
        """
        gpu_model = system_info.get("gpu_model", "").lower()
        cpu_model = system_info.get("cpu_model", "").lower()
        total_ram_gb = system_info.get("total_ram_gb", 0)
        vram_gb = system_info.get("vram_gb", 0)
        
        # High-end: RTX 4080+, Threadripper PRO, 128GB+ RAM
        if ("rtx 4080" in gpu_model or "rtx 4090" in gpu_model) and \
           "threadripper pro" in cpu_model and total_ram_gb >= 128:
            return HardwareProfile.HIGH_END
        
        # Enthusiast: RTX 3080+, High-end CPU, 64GB+ RAM
        elif ("rtx 3080" in gpu_model or "rtx 4080" in gpu_model) and \
             total_ram_gb >= 64:
            return HardwareProfile.ENTHUSIAST
        
        # Mainstream: RTX 3060+, Mid-range CPU, 32GB+ RAM
        elif ("rtx 3060" in gpu_model or "rtx 3070" in gpu_model) and \
             total_ram_gb >= 32:
            return HardwareProfile.MAINSTREAM
        
        # Budget: Everything else
        else:
            return HardwareProfile.BUDGET
    
    def start_guided_setup(self, system_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start guided setup process.
        
        Args:
            system_info: System hardware information
            
        Returns:
            Setup information and first step
        """
        hardware_profile = self.detect_hardware_profile(system_info)
        setup_steps = self.doc_integration.get_guided_setup_steps(hardware_profile)
        
        total_time = sum(step.estimated_time_minutes for step in setup_steps)
        required_steps = len([step for step in setup_steps if step.required])
        
        self.logger.info(f"Starting guided setup for {hardware_profile.value} hardware profile")
        
        return {
            "hardware_profile": hardware_profile.value,
            "total_steps": len(setup_steps),
            "required_steps": required_steps,
            "estimated_time_minutes": total_time,
            "current_step": 0,
            "steps": [
                {
                    "step_id": step.step_id,
                    "title": step.title,
                    "description": step.description,
                    "instructions": step.instructions,
                    "estimated_time_minutes": step.estimated_time_minutes,
                    "required": step.required,
                    "documentation_links": [
                        {
                            "title": link.title,
                            "url": link.url,
                            "description": link.description
                        }
                        for link in step.documentation_links
                    ]
                }
                for step in setup_steps
            ]
        }
    
    def validate_step_completion(self, 
                               step_id: str, 
                               user_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that a setup step has been completed correctly.
        
        Args:
            step_id: ID of the step to validate
            user_input: User's configuration input
            
        Returns:
            Validation result
        """
        # This would implement actual validation logic
        # For now, return a basic validation result
        return {
            "step_id": step_id,
            "valid": True,
            "warnings": [],
            "recommendations": []
        }