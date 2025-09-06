#!/usr/bin/env python3
"""
UI Error Recovery Integration Module
Integrates all error recovery and fallback systems into the main UI
"""

import logging
from typing import Dict, Any, Optional, List, Tuple, Callable
import gradio as gr
import traceback
import psutil
import platform
from datetime import datetime

from ui_error_recovery import UIErrorRecoveryManager, UIFallbackConfig, RecoveryGuidance
from enhanced_ui_creator import EnhancedUICreator
from recovery_guidance_system import RecoveryGuidanceSystem, update_system_context
from ui_creation_validator import UIComponentManager
from infrastructure.hardware.safe_event_handler import SafeEventHandler

logger = logging.getLogger(__name__)

class UIErrorRecoveryIntegration:
    """Main integration class for UI error recovery and fallback systems"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize all recovery systems
        self.recovery_manager = UIErrorRecoveryManager()
        self.ui_creator = EnhancedUICreator(config)
        self.guidance_system = RecoveryGuidanceSystem()
        self.component_manager = UIComponentManager()
        self.safe_event_handler = SafeEventHandler()
        
        # System monitoring
        self.system_monitor_enabled = self.config.get('enable_system_monitoring', True)
        self.last_system_update = datetime.now()
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize system context
        if self.system_monitor_enabled:
            self._update_system_context()
    
    def create_ui_with_comprehensive_recovery(self, 
                                            ui_definition: Dict[str, Any]) -> Tuple[gr.Blocks, Dict[str, Any]]:
        """
        Create UI with comprehensive error recovery and monitoring
        
        Args:
            ui_definition: UI definition dictionary
            
        Returns:
            Tuple of (gradio_interface, recovery_report)
        """
        self.logger.info("Starting UI creation with comprehensive error recovery")
        
        try:
            # Update system context before creation
            if self.system_monitor_enabled:
                self._update_system_context()
            
            # Create UI with enhanced error recovery
            interface, recovery_guidance = self.ui_creator.create_ui_with_fallbacks(ui_definition)
            
            # Generate comprehensive recovery report
            recovery_report = self._generate_recovery_report(recovery_guidance)
            
            # Add proactive guidance
            proactive_guidance = self.guidance_system.generate_proactive_guidance()
            recovery_report['proactive_guidance'] = proactive_guidance
            
            # Set up ongoing monitoring if enabled
            if self.system_monitor_enabled:
                self._setup_ongoing_monitoring(interface)
            
            self.logger.info("UI creation completed with error recovery integration")
            return interface, recovery_report
            
        except Exception as e:
            self.logger.error(f"Critical error in UI creation: {e}")
            
            # Create emergency recovery plan
            emergency_plan = self._create_emergency_recovery_plan(str(e))
            
            # Create minimal emergency interface
            emergency_interface = self._create_emergency_interface(str(e), emergency_plan)
            
            recovery_report = {
                'status': 'emergency_mode',
                'error': str(e),
                'emergency_plan': emergency_plan,
                'guidance': [],
                'proactive_guidance': []
            }
            
            return emergency_interface, recovery_report
    
    def _update_system_context(self):
        """Update system context information"""
        try:
            # Get system information
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            disk = psutil.disk_usage('/')
            
            # Check GPU availability
            gpu_available = False
            gpu_memory = 0.0
            try:
                import torch
                gpu_available = torch.cuda.is_available()
                if gpu_available:
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            except ImportError:
                pass
            
            # Update context
            context_data = {
                'memory_gb': memory.available / (1024**3),
                'cpu_usage': cpu_percent,
                'gpu_available': gpu_available,
                'gpu_memory_gb': gpu_memory,
                'disk_space_gb': disk.free / (1024**3),
                'network_connected': True,  # Assume connected if we're running
                'component_count': len(self.component_manager.components_registry),
                'failed_component_count': len(self.ui_creator.failed_components)
            }
            
            update_system_context(context_data)
            self.last_system_update = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Failed to update system context: {e}")
    
    def _generate_recovery_report(self, recovery_guidance: List[RecoveryGuidance]) -> Dict[str, Any]:
        """Generate comprehensive recovery report"""
        try:
            # Get statistics from all systems
            ui_creation_report = self.ui_creator.get_creation_report()
            recovery_stats = self.recovery_manager.get_recovery_statistics()
            event_handler_stats = self.safe_event_handler.get_setup_statistics()
            
            # Categorize guidance by severity
            guidance_by_severity = {
                'info': [],
                'warning': [],
                'error': [],
                'critical': []
            }
            
            for guidance in recovery_guidance:
                guidance_by_severity[guidance.severity].append({
                    'title': guidance.title,
                    'message': guidance.message,
                    'suggestions': guidance.suggestions,
                    'technical_details': guidance.technical_details if guidance.show_technical_details else None
                })
            
            # Determine overall status
            if guidance_by_severity['critical']:
                overall_status = 'critical'
            elif guidance_by_severity['error']:
                overall_status = 'error'
            elif guidance_by_severity['warning']:
                overall_status = 'warning'
            else:
                overall_status = 'healthy'
            
            return {
                'status': overall_status,
                'timestamp': datetime.now().isoformat(),
                'ui_creation': ui_creation_report,
                'recovery_statistics': recovery_stats,
                'event_handler_statistics': event_handler_stats,
                'guidance_by_severity': guidance_by_severity,
                'system_context': {
                    'memory_gb': getattr(self.guidance_system.system_context, 'available_memory_gb', 0),
                    'cpu_usage': getattr(self.guidance_system.system_context, 'cpu_usage_percent', 0),
                    'gpu_available': getattr(self.guidance_system.system_context, 'gpu_available', False)
                },
                'recommendations': self._generate_recommendations(overall_status, recovery_guidance)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate recovery report: {e}")
            return {
                'status': 'unknown',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_recommendations(self, 
                                status: str, 
                                recovery_guidance: List[RecoveryGuidance]) -> List[str]:
        """Generate high-level recommendations based on recovery status"""
        recommendations = []
        
        if status == 'critical':
            recommendations.extend([
                "Immediate attention required - multiple critical issues detected",
                "Consider restarting the application completely",
                "Check system resources and dependencies",
                "Contact technical support if issues persist"
            ])
        elif status == 'error':
            recommendations.extend([
                "Some components failed to load properly",
                "Try refreshing the page to reload components",
                "Monitor system resources during operation",
                "Consider using simplified settings if errors continue"
            ])
        elif status == 'warning':
            recommendations.extend([
                "Minor issues detected but system is functional",
                "Monitor for recurring problems",
                "Consider optimizing system settings",
                "Keep system updated to prevent issues"
            ])
        else:
            recommendations.extend([
                "System is operating normally",
                "Continue monitoring for any issues",
                "Regular maintenance recommended"
            ])
        
        # Add specific recommendations based on guidance
        error_types = set()
        for guidance in recovery_guidance:
            if 'memory' in guidance.message.lower():
                error_types.add('memory')
            if 'network' in guidance.message.lower():
                error_types.add('network')
            if 'component' in guidance.message.lower():
                error_types.add('component')
        
        if 'memory' in error_types:
            recommendations.append("Consider upgrading system memory for better performance")
        if 'network' in error_types:
            recommendations.append("Verify network stability and connection quality")
        if 'component' in error_types:
            recommendations.append("Check browser compatibility and update if needed")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _create_emergency_recovery_plan(self, error_details: str) -> Dict[str, Any]:
        """Create emergency recovery plan for critical failures"""
        return {
            'immediate_actions': [
                "Stop all current operations",
                "Save any important work",
                "Close other applications to free resources",
                "Prepare to restart the application"
            ],
            'recovery_steps': [
                {'step': 1, 'action': 'Close the application completely', 'duration': '30 seconds'},
                {'step': 2, 'action': 'Wait for system resources to clear', 'duration': '1 minute'},
                {'step': 3, 'action': 'Restart the application', 'duration': '2-3 minutes'},
                {'step': 4, 'action': 'Monitor for recurring issues', 'duration': 'Ongoing'}
            ],
            'escalation': [
                "If restart doesn't resolve the issue, check system logs",
                "Contact technical support with error details",
                "Consider system diagnostics if problems persist"
            ],
            'prevention': [
                "Ensure adequate system resources before starting",
                "Keep the application updated to latest version",
                "Regular system maintenance and cleanup"
            ]
        }
    
    def _create_emergency_interface(self, 
                                  error_details: str, 
                                  emergency_plan: Dict[str, Any]) -> gr.Blocks:
        """Create emergency interface when normal UI creation fails"""
        try:
            with gr.Blocks(title="WAN22 UI - Emergency Recovery Mode") as emergency_interface:
                gr.Markdown("# 🚨 WAN22 UI - Emergency Recovery Mode")
                
                # Error information
                gr.HTML(f"""
                <div style="background: #f8d7da; border: 2px solid #f5c6cb; border-radius: 8px; padding: 20px; margin: 20px 0;">
                    <h3 style="color: #721c24; margin-top: 0;">⚠️ Critical System Error</h3>
                    <p style="color: #721c24; margin-bottom: 15px;">
                        The WAN22 UI encountered a critical error and cannot start normally. 
                        This emergency interface provides recovery guidance and basic functionality.
                    </p>
                    <details style="margin-top: 15px;">
                        <summary style="cursor: pointer; font-weight: bold; color: #721c24;">
                            🔧 Technical Error Details
                        </summary>
                        <pre style="background: #e9ecef; padding: 15px; border-radius: 4px; margin-top: 10px; overflow-x: auto; font-size: 0.85em;">
{error_details}</pre>
                    </details>
                </div>
                """)
                
                # Recovery plan
                immediate_actions = "\n".join([f"• {action}" for action in emergency_plan['immediate_actions']])
                recovery_steps = "\n".join([
                    f"{step['step']}. {step['action']} (Est. {step['duration']})" 
                    for step in emergency_plan['recovery_steps']
                ])
                escalation_steps = "\n".join([f"• {step}" for step in emergency_plan['escalation']])
                
                gr.Markdown(f"""
                ## 🔧 Emergency Recovery Plan
                
                ### Immediate Actions Required:
                {immediate_actions}
                
                ### Recovery Steps:
                {recovery_steps}
                
                ### If Problems Persist:
                {escalation_steps}
                """)
                
                # System information
                try:
                    system_info = f"""
                    ## 📊 System Information
                    
                    - **Platform**: {platform.system()} {platform.release()}
                    - **Python Version**: {platform.python_version()}
                    - **Available Memory**: {psutil.virtual_memory().available / (1024**3):.1f} GB
                    - **CPU Usage**: {psutil.cpu_percent()}%
                    - **Disk Space**: {psutil.disk_usage('/').free / (1024**3):.1f} GB
                    - **Error Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    """
                    gr.Markdown(system_info)
                except Exception:
                    gr.Markdown("*System information unavailable*")
                
                # Recovery actions
                with gr.Row():
                    restart_btn = gr.Button("🔄 Restart Application", variant="primary")
                    logs_btn = gr.Button("📋 View Logs", variant="secondary")
                    support_btn = gr.Button("🆘 Contact Support", variant="secondary")
                
                # Status display
                status_display = gr.HTML(value="<p>Ready for recovery actions...</p>")
                
                # Simple event handlers for emergency interface
                def restart_action():
                    return "<p style='color: #28a745;'>✅ Restart initiated. Please close this window and restart the application.</p>"
                
                def logs_action():
                    return "<p style='color: #007bff;'>📋 Check the console logs and application log files for detailed error information.</p>"
                
                def support_action():
                    return f"""
                    <div style='background: #d1ecf1; padding: 15px; border-radius: 4px;'>
                        <p style='color: #0c5460; margin: 0;'>
                            <strong>📞 Support Information:</strong><br>
                            Please provide the following information when contacting support:<br>
                            • Error timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                            • System: {platform.system()} {platform.release()}<br>
                            • Python: {platform.python_version()}<br>
                            • Error details from the technical section above
                        </p>
                    </div>
                    """
                
                restart_btn.click(restart_action, outputs=status_display)
                logs_btn.click(logs_action, outputs=status_display)
                support_btn.click(support_action, outputs=status_display)
            
            return emergency_interface
            
        except Exception as e:
            self.logger.error(f"Failed to create emergency interface: {e}")
            # Absolute minimal fallback
            with gr.Blocks(title="WAN22 UI - Critical Error") as minimal_interface:
                gr.HTML(f"""
                <div style="padding: 40px; text-align: center; font-family: Arial, sans-serif;">
                    <h1 style="color: #dc3545;">🚨 Critical System Error</h1>
                    <p style="font-size: 1.2em; margin: 20px 0;">
                        The WAN22 UI cannot start due to a critical error.
                    </p>
                    <p style="background: #f8d7da; padding: 15px; border-radius: 4px; margin: 20px 0;">
                        <strong>Error:</strong> {str(e)}
                    </p>
                    <p style="font-size: 1.1em;">
                        Please restart the application and contact support if the issue persists.
                    </p>
                </div>
                """)
            return minimal_interface
    
    def _setup_ongoing_monitoring(self, interface: gr.Blocks):
        """Set up ongoing system monitoring for the UI"""
        try:
            # This would set up periodic monitoring
            # For now, we'll just log that monitoring is available
            self.logger.info("Ongoing system monitoring is available")
            
            # In a full implementation, this could:
            # - Set up periodic system resource checks
            # - Monitor component health
            # - Provide real-time recovery suggestions
            # - Alert on system issues
            
        except Exception as e:
            self.logger.error(f"Failed to set up ongoing monitoring: {e}")

# Global integration instance
global_recovery_integration = UIErrorRecoveryIntegration()

def create_ui_with_error_recovery(ui_definition: Dict[str, Any], 
                                config: Optional[Dict[str, Any]] = None) -> Tuple[gr.Blocks, Dict[str, Any]]:
    """
    Create UI with comprehensive error recovery integration
    
    Args:
        ui_definition: UI definition dictionary
        config: Optional configuration
        
    Returns:
        Tuple of (gradio_interface, recovery_report)
    """
    global global_recovery_integration
    if config:
        global_recovery_integration = UIErrorRecoveryIntegration(config)
    
    return global_recovery_integration.create_ui_with_comprehensive_recovery(ui_definition)

def get_recovery_integration() -> UIErrorRecoveryIntegration:
    """Get the global recovery integration instance"""
    return global_recovery_integration