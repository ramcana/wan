#!/usr/bin/env python3
"""
Recovery Guidance System for WAN22 UI
Provides intelligent recovery suggestions and user guidance for UI errors
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import re
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class GuidanceRule:
    """Rule for generating recovery guidance"""
    name: str
    pattern: str  # Regex pattern to match error
    title: str
    message: str
    suggestions: List[str]
    severity: str = "warning"
    priority: int = 1  # Higher priority rules are checked first
    conditions: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemContext:
    """System context information for guidance generation"""
    available_memory_gb: float = 0.0
    cpu_usage_percent: float = 0.0
    gpu_available: bool = False
    gpu_memory_gb: float = 0.0
    disk_space_gb: float = 0.0
    network_connected: bool = True
    python_version: str = ""
    gradio_version: str = ""
    component_count: int = 0
    failed_component_count: int = 0

class RecoveryGuidanceSystem:
    """Intelligent system for generating recovery guidance and suggestions"""
    
    def __init__(self):
        self.guidance_rules: List[GuidanceRule] = []
        self.system_context = SystemContext()
        self.logger = logging.getLogger(__name__)
        
        # Initialize built-in guidance rules
        self._initialize_guidance_rules()
        
        # Load custom rules if available
        self._load_custom_rules()
    
    def _initialize_guidance_rules(self):
        """Initialize built-in guidance rules"""
        
        # None/NoneType errors
        self.guidance_rules.append(GuidanceRule(
            name="none_component_error",
            pattern=r".*NoneType.*has no attribute.*_id.*|.*None.*component.*",
            title="Component Initialization Error",
            message="Some UI components failed to initialize properly, causing None component errors.",
            suggestions=[
                "Refresh the page to reinitialize all components",
                "Check if all required dependencies are properly installed",
                "Verify that component creation completed successfully",
                "Try restarting the application to clear any cached issues"
            ],
            severity="error",
            priority=10
        ))
        
        # Memory-related errors
        self.guidance_rules.append(GuidanceRule(
            name="memory_error",
            pattern=r".*out of memory.*|.*memory.*insufficient.*|.*CUDA out of memory.*",
            title="Insufficient Memory",
            message="The system is running low on memory, which may cause component failures.",
            suggestions=[
                "Close other applications to free up memory",
                "Try using smaller models or reduced settings",
                "Restart the application to clear memory leaks",
                "Consider upgrading system memory if this persists"
            ],
            severity="error",
            priority=9,
            conditions={"memory_threshold": 2.0}  # GB
        ))
        
        # Network/connection errors
        self.guidance_rules.append(GuidanceRule(
            name="network_error",
            pattern=r".*connection.*failed.*|.*network.*error.*|.*timeout.*|.*unreachable.*",
            title="Network Connection Issue",
            message="Network connectivity issues are affecting component functionality.",
            suggestions=[
                "Check your internet connection",
                "Verify that the server is running and accessible",
                "Try again in a few moments",
                "Check firewall settings if the issue persists"
            ],
            severity="warning",
            priority=8
        ))
        
        # File/path errors
        self.guidance_rules.append(GuidanceRule(
            name="file_error",
            pattern=r".*file not found.*|.*no such file.*|.*permission denied.*|.*path.*not.*exist.*",
            title="File Access Error",
            message="Unable to access required files or directories.",
            suggestions=[
                "Check that all required files are present",
                "Verify file permissions and access rights",
                "Ensure the file path is correct and accessible",
                "Try running with administrator privileges if needed"
            ],
            severity="error",
            priority=7
        ))
        
        # Model loading errors
        self.guidance_rules.append(GuidanceRule(
            name="model_loading_error",
            pattern=r".*model.*load.*failed.*|.*checkpoint.*error.*|.*safetensors.*|.*pytorch.*load.*",
            title="Model Loading Error",
            message="Failed to load AI model files, which may affect generation capabilities.",
            suggestions=[
                "Check that model files are properly downloaded",
                "Verify that you have sufficient VRAM/RAM for the model",
                "Try using a smaller or different model",
                "Re-download the model if files may be corrupted"
            ],
            severity="error",
            priority=8
        ))
        
        # GPU/CUDA errors
        self.guidance_rules.append(GuidanceRule(
            name="gpu_error",
            pattern=r".*CUDA.*error.*|.*GPU.*not.*available.*|.*device.*not.*found.*",
            title="GPU/CUDA Error",
            message="Issues with GPU acceleration may affect performance.",
            suggestions=[
                "Check that CUDA drivers are properly installed",
                "Verify GPU is detected by the system",
                "Try switching to CPU mode if GPU issues persist",
                "Update GPU drivers to the latest version"
            ],
            severity="warning",
            priority=6
        ))
        
        # Gradio-specific errors
        self.guidance_rules.append(GuidanceRule(
            name="gradio_component_error",
            pattern=r".*gradio.*component.*|.*blocks.*error.*|.*interface.*failed.*",
            title="Gradio Interface Error",
            message="Issues with the Gradio web interface components.",
            suggestions=[
                "Try refreshing the browser page",
                "Clear browser cache and cookies",
                "Check browser console for additional error details",
                "Try using a different browser if issues persist"
            ],
            severity="warning",
            priority=5
        ))
        
        # Import/dependency errors
        self.guidance_rules.append(GuidanceRule(
            name="import_error",
            pattern=r".*import.*error.*|.*module.*not.*found.*|.*dependency.*missing.*",
            title="Missing Dependencies",
            message="Required Python packages or modules are missing.",
            suggestions=[
                "Install missing dependencies using pip",
                "Check that all required packages are installed",
                "Verify Python environment is properly configured",
                "Try reinstalling the application dependencies"
            ],
            severity="error",
            priority=9
        ))
        
        # Generic fallback rule
        self.guidance_rules.append(GuidanceRule(
            name="generic_error",
            pattern=r".*",  # Matches everything
            title="Unexpected Error",
            message="An unexpected error occurred during UI operation.",
            suggestions=[
                "Try refreshing the page",
                "Restart the application if issues persist",
                "Check the console logs for more details",
                "Contact support if the problem continues"
            ],
            severity="warning",
            priority=1  # Lowest priority - fallback
        ))
    
    def _load_custom_rules(self):
        """Load custom guidance rules from configuration file"""
        try:
            rules_file = Path("config/recovery_guidance_rules.json")
            if rules_file.exists():
                with open(rules_file, 'r', encoding='utf-8') as f:
                    custom_rules_data = json.load(f)
                
                for rule_data in custom_rules_data.get('rules', []):
                    rule = GuidanceRule(
                        name=rule_data['name'],
                        pattern=rule_data['pattern'],
                        title=rule_data['title'],
                        message=rule_data['message'],
                        suggestions=rule_data['suggestions'],
                        severity=rule_data.get('severity', 'warning'),
                        priority=rule_data.get('priority', 5),
                        conditions=rule_data.get('conditions', {})
                    )
                    self.guidance_rules.append(rule)
                
                self.logger.info(f"Loaded {len(custom_rules_data.get('rules', []))} custom guidance rules")
                
        except Exception as e:
            self.logger.warning(f"Failed to load custom guidance rules: {e}")
    
    def update_system_context(self, context_data: Dict[str, Any]):
        """Update system context information"""
        try:
            if 'memory_gb' in context_data:
                self.system_context.available_memory_gb = context_data['memory_gb']
            if 'cpu_usage' in context_data:
                self.system_context.cpu_usage_percent = context_data['cpu_usage']
            if 'gpu_available' in context_data:
                self.system_context.gpu_available = context_data['gpu_available']
            if 'gpu_memory_gb' in context_data:
                self.system_context.gpu_memory_gb = context_data['gpu_memory_gb']
            if 'disk_space_gb' in context_data:
                self.system_context.disk_space_gb = context_data['disk_space_gb']
            if 'network_connected' in context_data:
                self.system_context.network_connected = context_data['network_connected']
            if 'component_count' in context_data:
                self.system_context.component_count = context_data['component_count']
            if 'failed_component_count' in context_data:
                self.system_context.failed_component_count = context_data['failed_component_count']
                
        except Exception as e:
            self.logger.error(f"Failed to update system context: {e}")
    
    def generate_guidance(self, 
                         error_message: str, 
                         component_name: str = "", 
                         error_type: str = "") -> Tuple[str, str, List[str], str]:
        """
        Generate recovery guidance based on error information
        
        Args:
            error_message: The error message to analyze
            component_name: Name of the affected component
            error_type: Type of error
            
        Returns:
            Tuple of (title, message, suggestions, severity)
        """
        # Sort rules by priority (highest first)
        sorted_rules = sorted(self.guidance_rules, key=lambda r: r.priority, reverse=True)
        
        for rule in sorted_rules:
            try:
                # Check if pattern matches
                if re.search(rule.pattern, error_message, re.IGNORECASE):
                    
                    # Check additional conditions
                    if self._check_rule_conditions(rule):
                        
                        # Customize suggestions based on context
                        customized_suggestions = self._customize_suggestions(
                            rule.suggestions, component_name, error_type
                        )
                        
                        self.logger.debug(f"Applied guidance rule: {rule.name}")
                        return rule.title, rule.message, customized_suggestions, rule.severity
                        
            except Exception as e:
                self.logger.error(f"Error applying guidance rule {rule.name}: {e}")
                continue
        
        # Fallback if no rules match (shouldn't happen due to generic rule)
        return "Error Occurred", "An error occurred during operation.", ["Try again later"], "warning"
    
    def _check_rule_conditions(self, rule: GuidanceRule) -> bool:
        """Check if rule conditions are met based on system context"""
        try:
            conditions = rule.conditions
            
            # Memory threshold check
            if 'memory_threshold' in conditions:
                if self.system_context.available_memory_gb < conditions['memory_threshold']:
                    return True
            
            # CPU usage threshold check
            if 'cpu_threshold' in conditions:
                if self.system_context.cpu_usage_percent > conditions['cpu_threshold']:
                    return True
            
            # GPU availability check
            if 'requires_gpu' in conditions:
                if conditions['requires_gpu'] and not self.system_context.gpu_available:
                    return True
            
            # Component failure rate check
            if 'failure_rate_threshold' in conditions:
                if self.system_context.component_count > 0:
                    failure_rate = self.system_context.failed_component_count / self.system_context.component_count
                    if failure_rate > conditions['failure_rate_threshold']:
                        return True
            
            # If no specific conditions, rule applies
            if not conditions:
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking rule conditions: {e}")
            return True  # Apply rule if condition check fails
    
    def _customize_suggestions(self, 
                             base_suggestions: List[str], 
                             component_name: str, 
                             error_type: str) -> List[str]:
        """Customize suggestions based on component and error context"""
        customized = base_suggestions.copy()
        
        # Add component-specific suggestions
        if component_name:
            if "image" in component_name.lower():
                customized.append("Try uploading a different image format (PNG, JPG, WebP)")
                customized.append("Ensure the image file is not corrupted or too large")
            
            elif "video" in component_name.lower():
                customized.append("Try a different video format (MP4, AVI, MOV)")
                customized.append("Check that the video file size is within limits")
            
            elif "model" in component_name.lower():
                customized.append("Verify that model files are completely downloaded")
                customized.append("Try using a smaller model if memory is limited")
            
            elif "lora" in component_name.lower():
                customized.append("Check that LoRA files are in the correct format")
                customized.append("Verify LoRA compatibility with the selected model")
        
        # Add system context-based suggestions
        if self.system_context.available_memory_gb < 4.0:
            customized.append("Consider closing other applications to free up memory")
        
        if not self.system_context.gpu_available:
            customized.append("GPU acceleration is not available - operations may be slower")
        
        if self.system_context.failed_component_count > 3:
            customized.append("Multiple components have failed - consider restarting the application")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for suggestion in customized:
            if suggestion not in seen:
                seen.add(suggestion)
                unique_suggestions.append(suggestion)
        
        return unique_suggestions[:6]  # Limit to 6 suggestions to avoid overwhelming
    
    def generate_proactive_guidance(self) -> List[Dict[str, Any]]:
        """Generate proactive guidance based on current system state"""
        guidance_items = []
        
        try:
            # Memory warning
            if self.system_context.available_memory_gb < 2.0:
                guidance_items.append({
                    'title': 'Low Memory Warning',
                    'message': f'Available memory is low ({self.system_context.available_memory_gb:.1f} GB). This may cause component failures.',
                    'suggestions': [
                        'Close unnecessary applications',
                        'Consider using smaller models',
                        'Restart the application to clear memory'
                    ],
                    'severity': 'warning',
                    'type': 'proactive'
                })
            
            # High CPU usage warning
            if self.system_context.cpu_usage_percent > 90:
                guidance_items.append({
                    'title': 'High CPU Usage',
                    'message': f'CPU usage is very high ({self.system_context.cpu_usage_percent:.1f}%). This may affect performance.',
                    'suggestions': [
                        'Wait for current operations to complete',
                        'Close other CPU-intensive applications',
                        'Consider reducing processing settings'
                    ],
                    'severity': 'warning',
                    'type': 'proactive'
                })
            
            # Component failure rate warning
            if (self.system_context.component_count > 0 and 
                self.system_context.failed_component_count / self.system_context.component_count > 0.3):
                guidance_items.append({
                    'title': 'High Component Failure Rate',
                    'message': f'{self.system_context.failed_component_count} out of {self.system_context.component_count} components failed to load.',
                    'suggestions': [
                        'Try refreshing the page',
                        'Check system resources and dependencies',
                        'Consider restarting the application'
                    ],
                    'severity': 'error',
                    'type': 'proactive'
                })
            
            # GPU not available info
            if not self.system_context.gpu_available:
                guidance_items.append({
                    'title': 'GPU Acceleration Unavailable',
                    'message': 'GPU acceleration is not available. Operations will use CPU processing.',
                    'suggestions': [
                        'Install CUDA drivers for GPU acceleration',
                        'Expect slower processing times',
                        'Consider using smaller models for better performance'
                    ],
                    'severity': 'info',
                    'type': 'proactive'
                })
            
        except Exception as e:
            self.logger.error(f"Error generating proactive guidance: {e}")
        
        return guidance_items
    
    def create_recovery_action_plan(self, 
                                  error_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a comprehensive recovery action plan
        
        Args:
            error_details: Dictionary containing error information
            
        Returns:
            Dictionary with recovery action plan
        """
        error_message = error_details.get('message', '')
        component_name = error_details.get('component_name', '')
        error_type = error_details.get('error_type', '')
        
        # Generate primary guidance
        title, message, suggestions, severity = self.generate_guidance(
            error_message, component_name, error_type
        )
        
        # Create action plan
        action_plan = {
            'primary_guidance': {
                'title': title,
                'message': message,
                'suggestions': suggestions,
                'severity': severity
            },
            'immediate_actions': self._get_immediate_actions(error_type, severity),
            'recovery_steps': self._get_recovery_steps(error_type, component_name),
            'prevention_tips': self._get_prevention_tips(error_type),
            'escalation_path': self._get_escalation_path(severity),
            'estimated_recovery_time': self._estimate_recovery_time(error_type, severity),
            'success_probability': self._estimate_success_probability(error_type, severity)
        }
        
        return action_plan
    
    def _get_immediate_actions(self, error_type: str, severity: str) -> List[str]:
        """Get immediate actions to take"""
        actions = []
        
        if severity in ['error', 'critical']:
            actions.append("Stop current operations if possible")
            actions.append("Save any important work")
        
        if error_type == 'memory_error':
            actions.extend([
                "Close unnecessary applications immediately",
                "Clear browser cache if using web interface"
            ])
        elif error_type == 'network_error':
            actions.extend([
                "Check internet connection",
                "Verify server status"
            ])
        elif error_type == 'component_error':
            actions.extend([
                "Refresh the page",
                "Clear browser cache"
            ])
        
        return actions
    
    def _get_recovery_steps(self, error_type: str, component_name: str) -> List[Dict[str, str]]:
        """Get detailed recovery steps"""
        steps = []
        
        if error_type == 'component_error':
            steps.extend([
                {'step': 1, 'action': 'Refresh the browser page', 'expected': 'Components should reload'},
                {'step': 2, 'action': 'Clear browser cache and cookies', 'expected': 'Fresh component loading'},
                {'step': 3, 'action': 'Restart the application', 'expected': 'Complete system reset'},
                {'step': 4, 'action': 'Check system resources', 'expected': 'Identify resource constraints'}
            ])
        elif error_type == 'memory_error':
            steps.extend([
                {'step': 1, 'action': 'Close other applications', 'expected': 'More memory available'},
                {'step': 2, 'action': 'Restart the application', 'expected': 'Memory leaks cleared'},
                {'step': 3, 'action': 'Use smaller models/settings', 'expected': 'Reduced memory usage'},
                {'step': 4, 'action': 'Consider system upgrade', 'expected': 'Long-term solution'}
            ])
        
        return steps
    
    def _get_prevention_tips(self, error_type: str) -> List[str]:
        """Get tips to prevent similar errors"""
        tips = []
        
        if error_type == 'memory_error':
            tips.extend([
                "Monitor system memory usage regularly",
                "Close unused applications before starting intensive tasks",
                "Consider upgrading system memory if errors persist"
            ])
        elif error_type == 'component_error':
            tips.extend([
                "Keep browser updated to latest version",
                "Regularly clear browser cache",
                "Ensure stable internet connection"
            ])
        
        return tips
    
    def _get_escalation_path(self, severity: str) -> List[str]:
        """Get escalation path for unresolved issues"""
        if severity == 'critical':
            return [
                "Contact technical support immediately",
                "Provide system logs and error details",
                "Consider emergency fallback procedures"
            ]
        elif severity == 'error':
            return [
                "Try all suggested recovery steps",
                "Contact support if issue persists after 30 minutes",
                "Document error patterns for analysis"
            ]
        else:
            return [
                "Monitor for recurring issues",
                "Contact support if problem becomes frequent",
                "Consider system optimization"
            ]
    
    def _estimate_recovery_time(self, error_type: str, severity: str) -> str:
        """Estimate recovery time"""
        if severity == 'critical':
            return "30-60 minutes"
        elif severity == 'error':
            return "5-15 minutes"
        else:
            return "1-5 minutes"
    
    def _estimate_success_probability(self, error_type: str, severity: str) -> str:
        """Estimate probability of successful recovery"""
        if error_type == 'component_error':
            return "85-95%"
        elif error_type == 'memory_error':
            return "70-85%"
        elif error_type == 'network_error':
            return "60-80%"
        else:
            return "70-90%"

# Global guidance system instance
global_guidance_system = RecoveryGuidanceSystem()

def get_recovery_guidance_system() -> RecoveryGuidanceSystem:
    """Get the global recovery guidance system instance"""
    return global_guidance_system

def generate_error_guidance(error_message: str, 
                          component_name: str = "", 
                          error_type: str = "") -> Tuple[str, str, List[str], str]:
    """
    Generate recovery guidance for an error
    
    Args:
        error_message: The error message
        component_name: Name of affected component
        error_type: Type of error
        
    Returns:
        Tuple of (title, message, suggestions, severity)
    """
    return global_guidance_system.generate_guidance(error_message, component_name, error_type)

def create_error_recovery_plan(error_details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create comprehensive recovery action plan
    
    Args:
        error_details: Error information dictionary
        
    Returns:
        Recovery action plan dictionary
    """
    return global_guidance_system.create_recovery_action_plan(error_details)

def update_system_context(context_data: Dict[str, Any]):
    """Update system context for better guidance generation"""
    global_guidance_system.update_system_context(context_data)