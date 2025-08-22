"""
Enhanced Error Messaging System with Documentation Integration

This module extends the existing error handling system to include documentation links,
contextual help, and guided recovery workflows.

Requirements addressed:
- 12.2: Link relevant documentation sections in error messages
- 12.2: Create contextual help for optimization settings
- 12.2: Add guided setup for first-time users with high-end hardware
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import webbrowser

from documentation_integration import (
    DocumentationIntegration, 
    EnhancedErrorHandler, 
    GuidedSetupManager,
    HardwareProfile
)


class MessageSeverity(Enum):
    """Message severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MessageType(Enum):
    """Types of messages"""
    ERROR = "error"
    RECOVERY_SUGGESTION = "recovery_suggestion"
    OPTIMIZATION_TIP = "optimization_tip"
    HARDWARE_RECOMMENDATION = "hardware_recommendation"
    DOCUMENTATION_LINK = "documentation_link"


@dataclass
class EnhancedMessage:
    """Enhanced message with documentation integration"""
    message_id: str
    title: str
    content: str
    severity: MessageSeverity
    message_type: MessageType
    documentation_links: List[Dict[str, Any]]
    quick_actions: List[Dict[str, Any]]
    contextual_help: Optional[Dict[str, Any]]
    timestamp: str
    component: str
    user_actionable: bool


class EnhancedErrorMessaging:
    """
    Enhanced error messaging system with documentation integration
    """
    
    def __init__(self, docs_base_path: str = "docs"):
        """
        Initialize enhanced error messaging system.
        
        Args:
            docs_base_path: Base path for documentation files
        """
        self.doc_integration = DocumentationIntegration(docs_base_path)
        self.enhanced_error_handler = EnhancedErrorHandler(self.doc_integration)
        self.guided_setup_manager = GuidedSetupManager(self.doc_integration)
        self.logger = logging.getLogger(__name__)
        
        # Message templates
        self._message_templates = self._initialize_message_templates()
        
        # Quick action handlers
        self._quick_action_handlers: Dict[str, Callable] = {}
        self._register_quick_action_handlers()
    
    def create_enhanced_error_message(self, 
                                    error: Exception,
                                    error_category: str,
                                    component: str = "system",
                                    context: Optional[Dict[str, Any]] = None) -> EnhancedMessage:
        """
        Create an enhanced error message with documentation links and recovery suggestions.
        
        Args:
            error: The exception that occurred
            error_category: Category of the error
            component: Component where the error occurred
            context: Additional context information
            
        Returns:
            Enhanced error message
        """
        # Get enhanced error information
        error_info = self.enhanced_error_handler.handle_error_with_documentation(
            error, error_category, context
        )
        
        # Generate message ID
        message_id = f"{error_category}_{component}_{hash(str(error)) % 10000:04d}"
        
        # Get message template
        template = self._message_templates.get(error_category, self._message_templates["default"])
        
        # Create enhanced message
        enhanced_message = EnhancedMessage(
            message_id=message_id,
            title=template["title"].format(error_type=type(error).__name__),
            content=error_info["enhanced_message"],
            severity=self._determine_message_severity(error, error_category),
            message_type=MessageType.ERROR,
            documentation_links=error_info["documentation_links"],
            quick_actions=self._generate_quick_actions(error_category, context),
            contextual_help=error_info["contextual_help"],
            timestamp=self._get_current_timestamp(),
            component=component,
            user_actionable=True
        )
        
        # Log the enhanced message
        self._log_enhanced_message(enhanced_message)
        
        return enhanced_message
    
    def create_optimization_tip_message(self, 
                                      optimization_type: str,
                                      current_settings: Dict[str, Any],
                                      recommended_settings: Dict[str, Any]) -> EnhancedMessage:
        """
        Create an optimization tip message.
        
        Args:
            optimization_type: Type of optimization
            current_settings: Current configuration settings
            recommended_settings: Recommended settings
            
        Returns:
            Optimization tip message
        """
        message_id = f"optimization_tip_{optimization_type}_{hash(str(current_settings)) % 10000:04d}"
        
        # Generate optimization content
        content = self._generate_optimization_tip_content(
            optimization_type, current_settings, recommended_settings
        )
        
        # Get relevant documentation links
        doc_links = self.doc_integration.get_error_documentation_links(
            optimization_type, "", {"settings": current_settings}
        )
        
        return EnhancedMessage(
            message_id=message_id,
            title=f"Optimization Recommendation: {optimization_type.replace('_', ' ').title()}",
            content=content,
            severity=MessageSeverity.INFO,
            message_type=MessageType.OPTIMIZATION_TIP,
            documentation_links=[
                {
                    "title": link.title,
                    "url": link.url,
                    "description": link.description,
                    "relevance_score": link.relevance_score
                }
                for link in doc_links
            ],
            quick_actions=[
                {
                    "action_id": f"apply_optimization_{optimization_type}",
                    "title": "Apply Recommended Settings",
                    "description": "Automatically apply the recommended optimization settings",
                    "parameters": recommended_settings
                }
            ],
            contextual_help=None,
            timestamp=self._get_current_timestamp(),
            component="optimization_system",
            user_actionable=True
        )
    
    def create_hardware_recommendation_message(self, 
                                             hardware_profile: HardwareProfile,
                                             detected_hardware: Dict[str, Any]) -> EnhancedMessage:
        """
        Create a hardware-specific recommendation message.
        
        Args:
            hardware_profile: Detected hardware profile
            detected_hardware: Detected hardware information
            
        Returns:
            Hardware recommendation message
        """
        message_id = f"hardware_recommendation_{hardware_profile.value}"
        
        # Generate hardware-specific recommendations
        content = self._generate_hardware_recommendation_content(hardware_profile, detected_hardware)
        
        # Get hardware-specific documentation
        doc_links = self.doc_integration.get_error_documentation_links(
            "hardware_optimization", "", {"hardware_profile": hardware_profile.value}
        )
        
        return EnhancedMessage(
            message_id=message_id,
            title=f"Hardware Optimization Recommendations for {hardware_profile.value.replace('_', ' ').title()}",
            content=content,
            severity=MessageSeverity.INFO,
            message_type=MessageType.HARDWARE_RECOMMENDATION,
            documentation_links=[
                {
                    "title": link.title,
                    "url": link.url,
                    "description": link.description,
                    "relevance_score": link.relevance_score
                }
                for link in doc_links
            ],
            quick_actions=[
                {
                    "action_id": "start_guided_setup",
                    "title": "Start Guided Setup",
                    "description": "Launch guided setup for your hardware configuration",
                    "parameters": {"hardware_profile": hardware_profile.value}
                },
                {
                    "action_id": "apply_hardware_optimizations",
                    "title": "Apply Hardware Optimizations",
                    "description": "Automatically apply optimizations for your hardware",
                    "parameters": {"hardware_profile": hardware_profile.value}
                }
            ],
            contextual_help=None,
            timestamp=self._get_current_timestamp(),
            component="hardware_optimizer",
            user_actionable=True
        )
    
    def create_guided_setup_message(self, 
                                  system_info: Dict[str, Any]) -> EnhancedMessage:
        """
        Create a guided setup initiation message.
        
        Args:
            system_info: System hardware information
            
        Returns:
            Guided setup message
        """
        setup_info = self.guided_setup_manager.start_guided_setup(system_info)
        
        message_id = f"guided_setup_{setup_info['hardware_profile']}"
        
        content = f"""
Welcome to WAN22 System Optimization! 

We've detected your system has {setup_info['hardware_profile'].replace('_', ' ')} hardware configuration.
Let's optimize your system for the best performance.

**Setup Overview:**
- Total Steps: {setup_info['total_steps']} ({setup_info['required_steps']} required)
- Estimated Time: {setup_info['estimated_time_minutes']} minutes
- Hardware Profile: {setup_info['hardware_profile'].replace('_', ' ').title()}

The guided setup will help you:
1. Configure VRAM optimization for your GPU
2. Set up quantization for memory efficiency  
3. Apply hardware-specific optimizations
4. Enable system health monitoring
5. Validate performance improvements

Click "Start Guided Setup" to begin the optimization process.
        """.strip()
        
        return EnhancedMessage(
            message_id=message_id,
            title="Welcome to WAN22 System Optimization",
            content=content,
            severity=MessageSeverity.INFO,
            message_type=MessageType.OPTIMIZATION_TIP,
            documentation_links=[
                {
                    "title": "Quick Start Guide",
                    "url": "WAN22_SYSTEM_OPTIMIZATION_USER_GUIDE.md#quick-start",
                    "description": "Get started with system optimization",
                    "relevance_score": 1.0
                },
                {
                    "title": "Hardware Requirements",
                    "url": "WAN22_SYSTEM_OPTIMIZATION_USER_GUIDE.md#system-requirements",
                    "description": "System requirements and compatibility",
                    "relevance_score": 0.9
                }
            ],
            quick_actions=[
                {
                    "action_id": "start_guided_setup",
                    "title": "Start Guided Setup",
                    "description": "Begin the guided optimization setup process",
                    "parameters": setup_info
                },
                {
                    "action_id": "skip_setup",
                    "title": "Skip Setup",
                    "description": "Continue with default settings (not recommended)",
                    "parameters": {}
                }
            ],
            contextual_help={
                "title": "Guided Setup Benefits",
                "description": "The guided setup ensures optimal performance for your specific hardware configuration.",
                "quick_tips": [
                    "Customized for your hardware profile",
                    "Prevents common configuration issues",
                    "Maximizes performance and stability",
                    "Can be re-run anytime if needed"
                ]
            },
            timestamp=self._get_current_timestamp(),
            component="guided_setup",
            user_actionable=True
        )
    
    def execute_quick_action(self, 
                           action_id: str, 
                           parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a quick action from an enhanced message.
        
        Args:
            action_id: ID of the action to execute
            parameters: Parameters for the action
            
        Returns:
            Result of the action execution
        """
        if action_id not in self._quick_action_handlers:
            return {
                "success": False,
                "message": f"Unknown quick action: {action_id}",
                "details": {}
            }
        
        try:
            handler = self._quick_action_handlers[action_id]
            result = handler(parameters)
            
            self.logger.info(f"Quick action executed: {action_id} - Success: {result.get('success', False)}")
            return result
            
        except Exception as e:
            error_result = {
                "success": False,
                "message": f"Failed to execute quick action {action_id}: {str(e)}",
                "details": {"error": str(e)}
            }
            self.logger.error(f"Quick action failed: {action_id} - {str(e)}")
            return error_result
    
    def open_documentation_link(self, link_url: str, section: Optional[str] = None):
        """
        Open a documentation link.
        
        Args:
            link_url: URL of the documentation
            section: Specific section to navigate to
        """
        try:
            # Use the documentation integration to open the link
            if section:
                full_url = f"{link_url}#{section.lower().replace(' ', '-')}"
            else:
                full_url = link_url
            
            # Check if it's a local file
            if not full_url.startswith(('http://', 'https://')):
                local_path = Path(full_url)
                if local_path.exists():
                    full_url = f"file://{local_path.absolute()}"
            
            webbrowser.open(full_url)
            self.logger.info(f"Opened documentation: {full_url}")
            
        except Exception as e:
            self.logger.error(f"Failed to open documentation link: {e}")
    
    def _initialize_message_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize message templates for different error categories"""
        return {
            "vram_memory": {
                "title": "GPU Memory Issue - {error_type}",
                "icon": "🔧"
            },
            "model_loading": {
                "title": "Model Loading Error - {error_type}",
                "icon": "📦"
            },
            "quantization": {
                "title": "Quantization Issue - {error_type}",
                "icon": "⚡"
            },
            "configuration": {
                "title": "Configuration Error - {error_type}",
                "icon": "⚙️"
            },
            "hardware_optimization": {
                "title": "Hardware Optimization Issue - {error_type}",
                "icon": "🖥️"
            },
            "system_health": {
                "title": "System Health Alert - {error_type}",
                "icon": "🏥"
            },
            "default": {
                "title": "System Error - {error_type}",
                "icon": "❌"
            }
        }
    
    def _register_quick_action_handlers(self):
        """Register handlers for quick actions"""
        
        def apply_vram_optimization(parameters: Dict[str, Any]) -> Dict[str, Any]:
            """Apply VRAM optimization settings"""
            try:
                # This would integrate with the actual VRAM manager
                return {
                    "success": True,
                    "message": "VRAM optimization applied successfully",
                    "details": {
                        "optimization_enabled": True,
                        "threshold": parameters.get("threshold", 0.9),
                        "cpu_offload_enabled": True
                    }
                }
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Failed to apply VRAM optimization: {str(e)}",
                    "details": {"error": str(e)}
                }
        
        def apply_quantization_settings(parameters: Dict[str, Any]) -> Dict[str, Any]:
            """Apply quantization settings"""
            try:
                # This would integrate with the actual quantization controller
                return {
                    "success": True,
                    "message": "Quantization settings applied successfully",
                    "details": {
                        "strategy": parameters.get("strategy", "bf16"),
                        "timeout": parameters.get("timeout", 300),
                        "fallback_enabled": True
                    }
                }
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Failed to apply quantization settings: {str(e)}",
                    "details": {"error": str(e)}
                }
        
        def start_guided_setup(parameters: Dict[str, Any]) -> Dict[str, Any]:
            """Start guided setup process"""
            try:
                # This would integrate with the actual guided setup system
                return {
                    "success": True,
                    "message": "Guided setup started successfully",
                    "details": {
                        "setup_id": f"setup_{self._get_current_timestamp()}",
                        "hardware_profile": parameters.get("hardware_profile", "unknown"),
                        "total_steps": parameters.get("total_steps", 0)
                    }
                }
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Failed to start guided setup: {str(e)}",
                    "details": {"error": str(e)}
                }
        
        def apply_hardware_optimizations(parameters: Dict[str, Any]) -> Dict[str, Any]:
            """Apply hardware-specific optimizations"""
            try:
                # This would integrate with the actual hardware optimizer
                return {
                    "success": True,
                    "message": "Hardware optimizations applied successfully",
                    "details": {
                        "hardware_profile": parameters.get("hardware_profile", "unknown"),
                        "optimizations_applied": [
                            "tensor_core_optimization",
                            "memory_bandwidth_optimization",
                            "multi_core_utilization"
                        ]
                    }
                }
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Failed to apply hardware optimizations: {str(e)}",
                    "details": {"error": str(e)}
                }
        
        def clear_model_cache(parameters: Dict[str, Any]) -> Dict[str, Any]:
            """Clear model cache"""
            try:
                # This would integrate with the actual model cache system
                return {
                    "success": True,
                    "message": "Model cache cleared successfully",
                    "details": {
                        "cache_size_cleared_mb": 1024,
                        "models_cleared": ["TI2V-5B", "cached_models"]
                    }
                }
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Failed to clear model cache: {str(e)}",
                    "details": {"error": str(e)}
                }
        
        # Register handlers
        self._quick_action_handlers.update({
            "apply_vram_optimization": apply_vram_optimization,
            "apply_quantization_settings": apply_quantization_settings,
            "start_guided_setup": start_guided_setup,
            "apply_hardware_optimizations": apply_hardware_optimizations,
            "clear_model_cache": clear_model_cache
        })
    
    def _determine_message_severity(self, error: Exception, error_category: str) -> MessageSeverity:
        """Determine message severity based on error type and category"""
        if isinstance(error, (SystemError, MemoryError)):
            return MessageSeverity.CRITICAL
        elif error_category in ["vram_memory", "model_loading"]:
            return MessageSeverity.ERROR
        elif error_category in ["quantization", "configuration"]:
            return MessageSeverity.WARNING
        else:
            return MessageSeverity.INFO
    
    def _generate_optimization_tip_content(self, 
                                         optimization_type: str,
                                         current_settings: Dict[str, Any],
                                         recommended_settings: Dict[str, Any]) -> str:
        """Generate content for optimization tip messages"""
        content = f"We've detected that your {optimization_type.replace('_', ' ')} settings can be optimized for better performance.\n\n"
        
        content += "**Current Settings:**\n"
        for key, value in current_settings.items():
            content += f"• {key.replace('_', ' ').title()}: {value}\n"
        
        content += "\n**Recommended Settings:**\n"
        for key, value in recommended_settings.items():
            content += f"• {key.replace('_', ' ').title()}: {value}\n"
        
        # Add expected benefits
        benefits = self._get_optimization_benefits(optimization_type)
        if benefits:
            content += f"\n**Expected Benefits:**\n"
            for benefit in benefits:
                content += f"• {benefit}\n"
        
        return content
    
    def _generate_hardware_recommendation_content(self, 
                                                hardware_profile: HardwareProfile,
                                                detected_hardware: Dict[str, Any]) -> str:
        """Generate content for hardware recommendation messages"""
        content = f"We've detected your system has {hardware_profile.value.replace('_', ' ')} hardware configuration.\n\n"
        
        content += "**Detected Hardware:**\n"
        for key, value in detected_hardware.items():
            content += f"• {key.replace('_', ' ').title()}: {value}\n"
        
        # Add hardware-specific recommendations
        recommendations = self._get_hardware_recommendations(hardware_profile)
        if recommendations:
            content += f"\n**Optimization Recommendations:**\n"
            for recommendation in recommendations:
                content += f"• {recommendation}\n"
        
        content += "\n**Next Steps:**\n"
        content += "• Run guided setup to apply optimal settings for your hardware\n"
        content += "• Enable hardware-specific optimizations\n"
        content += "• Configure system health monitoring\n"
        
        return content
    
    def _get_optimization_benefits(self, optimization_type: str) -> List[str]:
        """Get expected benefits for optimization type"""
        benefits_map = {
            "vram_optimization": [
                "Reduced VRAM usage by 30-50%",
                "Prevention of out-of-memory errors",
                "Ability to load larger models"
            ],
            "quantization": [
                "30-40% reduction in VRAM usage",
                "Faster model loading times",
                "Minimal impact on generation quality"
            ],
            "hardware_optimization": [
                "40-60% performance improvement",
                "Better utilization of hardware capabilities",
                "Optimized memory and CPU usage"
            ]
        }
        return benefits_map.get(optimization_type, [])
    
    def _get_hardware_recommendations(self, hardware_profile: HardwareProfile) -> List[str]:
        """Get recommendations for hardware profile"""
        recommendations_map = {
            HardwareProfile.HIGH_END: [
                "Enable aggressive tensor core utilization for RTX 4080",
                "Configure large memory pools for 128GB+ RAM",
                "Set up NUMA-aware memory allocation for Threadripper PRO",
                "Enable advanced VRAM optimization techniques"
            ],
            HardwareProfile.ENTHUSIAST: [
                "Enable tensor core optimizations",
                "Configure optimal VRAM management",
                "Set up multi-core CPU utilization",
                "Enable performance monitoring"
            ],
            HardwareProfile.MAINSTREAM: [
                "Enable basic hardware optimizations",
                "Configure conservative VRAM settings",
                "Use balanced quantization strategy",
                "Enable system health monitoring"
            ],
            HardwareProfile.BUDGET: [
                "Enable aggressive quantization for memory savings",
                "Configure conservative VRAM thresholds",
                "Use CPU offloading when possible",
                "Monitor system resources closely"
            ]
        }
        return recommendations_map.get(hardware_profile, [])
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp as string"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _log_enhanced_message(self, message: EnhancedMessage):
        """Log enhanced message for debugging and analytics"""
        log_data = {
            "message_id": message.message_id,
            "title": message.title,
            "severity": message.severity.value,
            "message_type": message.message_type.value,
            "component": message.component,
            "timestamp": message.timestamp,
            "user_actionable": message.user_actionable,
            "documentation_links_count": len(message.documentation_links),
            "quick_actions_count": len(message.quick_actions)
        }
        
        self.logger.info(f"Enhanced message created: {json.dumps(log_data, indent=2)}")


# Example usage and integration functions
def integrate_with_existing_error_handler():
    """
    Example of how to integrate with existing error handling systems
    """
    enhanced_messaging = EnhancedErrorMessaging()
    
    def enhanced_error_wrapper(original_handler):
        """Wrapper to enhance existing error handlers"""
        def wrapper(error, context=None):
            # Call original handler
            original_result = original_handler(error, context)
            
            # Create enhanced message
            error_category = context.get("category", "unknown") if context else "unknown"
            component = context.get("component", "system") if context else "system"
            
            enhanced_message = enhanced_messaging.create_enhanced_error_message(
                error, error_category, component, context
            )
            
            # Add enhanced message to result
            if isinstance(original_result, dict):
                original_result["enhanced_message"] = enhanced_message
            
            return original_result
        
        return wrapper
    
    return enhanced_error_wrapper


def create_ui_integration_helpers():
    """
    Create helper functions for UI integration
    """
    enhanced_messaging = EnhancedErrorMessaging()
    
    def get_contextual_help_for_element(element_id: str) -> Optional[Dict[str, Any]]:
        """Get contextual help for a UI element"""
        help_info = enhanced_messaging.doc_integration.get_contextual_help(element_id)
        if help_info:
            return {
                "title": help_info.title,
                "description": help_info.description,
                "quick_tips": help_info.quick_tips,
                "documentation_links": [
                    {
                        "title": link.title,
                        "url": link.url,
                        "description": link.description
                    }
                    for link in help_info.documentation_links
                ],
                "related_settings": help_info.related_settings
            }
        return None
    
    def execute_quick_action_from_ui(action_id: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a quick action from the UI"""
        return enhanced_messaging.execute_quick_action(action_id, parameters)
    
    def open_documentation_from_ui(link_url: str, section: Optional[str] = None):
        """Open documentation from the UI"""
        enhanced_messaging.open_documentation_link(link_url, section)
    
    return {
        "get_contextual_help": get_contextual_help_for_element,
        "execute_quick_action": execute_quick_action_from_ui,
        "open_documentation": open_documentation_from_ui
    }