"""
Error handling and user guidance system for the startup manager.

This module provides:
- Structured error display with clear messages
- Interactive error resolution
- Context-sensitive help and troubleshooting
- Error classification and recovery suggestions
"""

import traceback
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import re

from .cli import InteractiveCLI, VerbosityLevel


class ErrorSeverity(Enum):
    """Error severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of startup errors"""
    ENVIRONMENT = "environment"
    CONFIGURATION = "configuration"
    NETWORK = "network"
    PERMISSION = "permission"
    DEPENDENCY = "dependency"
    PROCESS = "process"
    SYSTEM = "system"
    UNKNOWN = "unknown"


@dataclass
class RecoveryAction:
    """Represents a recovery action that can be taken"""
    name: str
    description: str
    action: Callable[[], bool]
    auto_executable: bool = False
    requires_confirmation: bool = True


@dataclass
class StartupError:
    """Structured representation of a startup error"""
    message: str
    category: ErrorCategory
    severity: ErrorSeverity
    details: Optional[str] = None
    technical_details: Optional[str] = None
    recovery_actions: List[RecoveryAction] = field(default_factory=list)
    help_topics: List[str] = field(default_factory=list)
    related_files: List[Path] = field(default_factory=list)
    error_code: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization to set default recovery actions"""
        if not self.recovery_actions:
            self.recovery_actions = self._get_default_recovery_actions()
    
    def _get_default_recovery_actions(self) -> List[RecoveryAction]:
        """Get default recovery actions based on error category"""
        actions = []
        
        if self.category == ErrorCategory.NETWORK:
            actions.extend([
                RecoveryAction(
                    "retry",
                    "Retry the operation",
                    lambda: True,
                    auto_executable=True,
                    requires_confirmation=False
                ),
                RecoveryAction(
                    "check_firewall",
                    "Check Windows Firewall settings",
                    lambda: self._open_firewall_settings(),
                    requires_confirmation=True
                )
            ])
        elif self.category == ErrorCategory.PERMISSION:
            actions.extend([
                RecoveryAction(
                    "run_as_admin",
                    "Restart with administrator privileges",
                    lambda: self._suggest_admin_restart(),
                    requires_confirmation=True
                ),
                RecoveryAction(
                    "check_permissions",
                    "Check file and folder permissions",
                    lambda: self._check_permissions(),
                    requires_confirmation=True
                )
            ])
        elif self.category == ErrorCategory.DEPENDENCY:
            actions.extend([
                RecoveryAction(
                    "install_dependencies",
                    "Install missing dependencies",
                    lambda: self._install_dependencies(),
                    auto_executable=True,
                    requires_confirmation=True
                )
            ])
        
        # Always add generic actions
        actions.extend([
            RecoveryAction(
                "view_logs",
                "View detailed logs",
                lambda: self._view_logs(),
                requires_confirmation=False
            ),
            RecoveryAction(
                "get_help",
                "Get help and documentation",
                lambda: self._show_help(),
                requires_confirmation=False
            )
        ])
        
        return actions
    
    def _open_firewall_settings(self) -> bool:
        """Open Windows Firewall settings"""
        import subprocess
        try:
            subprocess.run(['control', 'firewall.cpl'], check=True)
            return True
        except Exception:
            return False
    
    def _suggest_admin_restart(self) -> bool:
        """Suggest restarting with admin privileges"""
        print("Please restart the application as administrator:")
        print("1. Right-click on Command Prompt")
        print("2. Select 'Run as administrator'")
        print("3. Navigate to the project directory")
        print("4. Run the startup command again")
        return True
    
    def _check_permissions(self) -> bool:
        """Check file and folder permissions"""
        print("Checking common permission issues...")
        # This would implement actual permission checking
        return True
    
    def _install_dependencies(self) -> bool:
        """Install missing dependencies"""
        print("Installing dependencies...")
        # This would implement actual dependency installation
        return True
    
    def _view_logs(self) -> bool:
        """View detailed logs"""
        print("Opening log files...")
        # This would implement log viewing
        return True
    
    def _show_help(self) -> bool:
        """Show help documentation"""
        print("Opening help documentation...")
        # This would implement help display
        return True


class ErrorClassifier:
    """Classifies errors and creates structured error objects"""
    
    def __init__(self):
        self.error_patterns = {
            # Network errors
            r"WinError 10013": (ErrorCategory.PERMISSION, "Socket access forbidden"),
            r"WinError 10048": (ErrorCategory.NETWORK, "Port conflict detected"),
            r"Address already in use": (ErrorCategory.NETWORK, "Port conflict detected"),
            r"Connection refused": (ErrorCategory.NETWORK, "Service not responding"),
            r"Network is unreachable": (ErrorCategory.NETWORK, "Network connectivity issue"),
            
            # Permission errors
            r"PermissionError": (ErrorCategory.PERMISSION, "File access denied"),
            r"Access is denied": (ErrorCategory.PERMISSION, "Windows access denied"),
            r"Operation not permitted": (ErrorCategory.PERMISSION, "Insufficient privileges"),
            
            # Dependency errors
            r"No module named": (ErrorCategory.DEPENDENCY, "Python module missing"),
            r"ModuleNotFoundError": (ErrorCategory.DEPENDENCY, "Python module missing"),
            r"ImportError": (ErrorCategory.DEPENDENCY, "Import failed"),
            r"command not found": (ErrorCategory.DEPENDENCY, "Command not available"),
            r"npm ERR!": (ErrorCategory.DEPENDENCY, "NPM error"),
            
            # Configuration errors
            r"JSONDecodeError": (ErrorCategory.CONFIGURATION, "Invalid JSON configuration"),
            r"Expecting.*delimiter": (ErrorCategory.CONFIGURATION, "Invalid JSON configuration"),
            r"KeyError": (ErrorCategory.CONFIGURATION, "Missing configuration key"),
            r"ValueError.*config": (ErrorCategory.CONFIGURATION, "Invalid configuration value"),
            
            # Process errors
            r"subprocess.*failed": (ErrorCategory.PROCESS, "Process execution failed"),
            r"TimeoutExpired": (ErrorCategory.PROCESS, "Process timeout"),
            r"CalledProcessError": (ErrorCategory.PROCESS, "Process returned error"),
            
            # System errors
            r"OSError": (ErrorCategory.SYSTEM, "Operating system error"),
            r"FileNotFoundError": (ErrorCategory.SYSTEM, "File not found"),
            r"IsADirectoryError": (ErrorCategory.SYSTEM, "Expected file, found directory"),
        }
    
    def classify_error(self, error: Exception, context: Dict[str, Any] = None) -> StartupError:
        """Classify an error and create a structured error object"""
        error_str = str(error)
        error_type = type(error).__name__
        
        # Try to match error patterns
        category = ErrorCategory.UNKNOWN
        base_message = error_str
        
        for pattern, (cat, msg) in self.error_patterns.items():
            if re.search(pattern, error_str, re.IGNORECASE):
                category = cat
                base_message = msg
                break
        
        # Determine severity
        severity = self._determine_severity(error, category)
        
        # Create structured error
        startup_error = StartupError(
            message=base_message,
            category=category,
            severity=severity,
            details=self._extract_details(error, context),
            technical_details=self._format_technical_details(error),
            error_code=self._generate_error_code(category, error_type)
        )
        
        # Add context-specific recovery actions
        startup_error.recovery_actions.extend(
            self._get_context_specific_actions(startup_error, context)
        )
        
        return startup_error
    
    def _determine_severity(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Determine error severity based on type and category"""
        if category in [ErrorCategory.DEPENDENCY, ErrorCategory.CONFIGURATION]:
            return ErrorSeverity.ERROR
        elif category in [ErrorCategory.NETWORK, ErrorCategory.PERMISSION]:
            return ErrorSeverity.WARNING
        elif category == ErrorCategory.SYSTEM:
            return ErrorSeverity.ERROR
        else:
            return ErrorSeverity.INFO
    
    def _extract_details(self, error: Exception, context: Dict[str, Any] = None) -> str:
        """Extract user-friendly details from error and context"""
        details = []
        
        if context:
            if 'operation' in context:
                details.append(f"Failed during: {context['operation']}")
            if 'file' in context:
                details.append(f"Related file: {context['file']}")
            if 'port' in context:
                details.append(f"Port involved: {context['port']}")
        
        # Add error-specific details
        error_str = str(error)
        if "port" in error_str.lower():
            port_match = re.search(r'port\s+(\d+)', error_str, re.IGNORECASE)
            if port_match:
                details.append(f"Port: {port_match.group(1)}")
        
        return "; ".join(details) if details else None
    
    def _format_technical_details(self, error: Exception) -> str:
        """Format technical details for debugging"""
        return f"{type(error).__name__}: {str(error)}"
    
    def _generate_error_code(self, category: ErrorCategory, error_type: str) -> str:
        """Generate a unique error code"""
        category_codes = {
            ErrorCategory.ENVIRONMENT: "ENV",
            ErrorCategory.CONFIGURATION: "CFG",
            ErrorCategory.NETWORK: "NET",
            ErrorCategory.PERMISSION: "PRM",
            ErrorCategory.DEPENDENCY: "DEP",
            ErrorCategory.PROCESS: "PRC",
            ErrorCategory.SYSTEM: "SYS",
            ErrorCategory.UNKNOWN: "UNK"
        }
        
        category_code = category_codes.get(category, "UNK")
        type_hash = abs(hash(error_type)) % 1000
        
        return f"{category_code}-{type_hash:03d}"
    
    def _get_context_specific_actions(self, error: StartupError, context: Dict[str, Any] = None) -> List[RecoveryAction]:
        """Get recovery actions specific to the context"""
        actions = []
        
        if not context:
            return actions
        
        if error.category == ErrorCategory.NETWORK and 'port' in context:
            port = context['port']
            actions.append(
                RecoveryAction(
                    "use_different_port",
                    f"Try a different port (current: {port})",
                    lambda: self._suggest_port_change(port),
                    auto_executable=True
                )
            )
        
        if error.category == ErrorCategory.CONFIGURATION and 'config_file' in context:
            config_file = context['config_file']
            actions.append(
                RecoveryAction(
                    "reset_config",
                    f"Reset configuration file: {config_file}",
                    lambda: self._reset_config_file(config_file),
                    requires_confirmation=True
                )
            )
        
        return actions
    
    def _suggest_port_change(self, current_port: int) -> bool:
        """Suggest using a different port"""
        suggested_ports = [current_port + 1, current_port + 10, current_port + 100]
        print(f"Current port {current_port} is in use.")
        print(f"Try these alternative ports: {', '.join(map(str, suggested_ports))}")
        return True
    
    def _reset_config_file(self, config_file: str) -> bool:
        """Reset a configuration file to defaults"""
        print(f"Resetting configuration file: {config_file}")
        # This would implement actual config reset
        return True


class ErrorDisplayManager:
    """Manages the display of errors and user interaction"""
    
    def __init__(self, cli: InteractiveCLI):
        self.cli = cli
        self.classifier = ErrorClassifier()
        self.help_system = HelpSystem()
        self.error_history = []
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> bool:
        """Handle an error with full user interaction"""
        # Classify the error
        structured_error = self.classifier.classify_error(error, context)
        
        # Add to error history for pattern detection
        self.error_history.append(structured_error)
        
        # Display the error with enhanced formatting
        self.display_error_enhanced(structured_error)
        
        # Check for recurring error patterns
        if self._is_recurring_error(structured_error):
            self._display_recurring_error_warning(structured_error)
        
        # Offer recovery options with improved interaction
        return self.offer_recovery_options_enhanced(structured_error)
    
    def display_error(self, error: StartupError):
        """Display a structured error with rich formatting"""
        # Determine display style based on severity
        severity_styles = {
            ErrorSeverity.INFO: "info",
            ErrorSeverity.WARNING: "warning",
            ErrorSeverity.ERROR: "error",
            ErrorSeverity.CRITICAL: "error"
        }
        
        style = severity_styles.get(error.severity, "error")
        
        # Display main error message
        self.cli.print_status(f"Error: {error.message}", style)
        
        # Display error code if available
        if error.error_code:
            self.cli.print_verbose(f"Error Code: {error.error_code}")
        
        # Display details if available
        if error.details:
            self.cli.print_status(f"Details: {error.details}", "info")
        
        # Display technical details in debug mode
        if error.technical_details:
            self.cli.print_debug(f"Technical: {error.technical_details}")
        
        # Display related files
        if error.related_files:
            files_str = ", ".join(str(f) for f in error.related_files)
            self.cli.print_verbose(f"Related files: {files_str}")
    
    def display_error_enhanced(self, error: StartupError):
        """Display error with enhanced formatting and context"""
        # Create error summary panel
        severity_icons = {
            ErrorSeverity.INFO: "ℹ️",
            ErrorSeverity.WARNING: "⚠️",
            ErrorSeverity.ERROR: "❌",
            ErrorSeverity.CRITICAL: "🚨"
        }
        
        severity_colors = {
            ErrorSeverity.INFO: "blue",
            ErrorSeverity.WARNING: "yellow",
            ErrorSeverity.ERROR: "red",
            ErrorSeverity.CRITICAL: "red"
        }
        
        icon = severity_icons.get(error.severity, "❓")
        color = severity_colors.get(error.severity, "red")
        
        # Build error content
        content_lines = [
            f"{icon} **{error.message}**",
            ""
        ]
        
        if error.details:
            content_lines.extend([
                f"**Details:** {error.details}",
                ""
            ])
        
        if error.error_code:
            content_lines.extend([
                f"**Error Code:** {error.error_code}",
                ""
            ])
        
        # Add category-specific guidance
        category_guidance = self._get_category_guidance(error.category)
        if category_guidance:
            content_lines.extend([
                "**Quick Tips:**",
                category_guidance,
                ""
            ])
        
        # Add technical details in verbose mode
        if error.technical_details and self.cli.options.verbosity in [VerbosityLevel.VERBOSE, VerbosityLevel.DEBUG]:
            content_lines.extend([
                f"**Technical Details:** {error.technical_details}",
                ""
            ])
        
        # Display the error panel
        self.cli.display_summary_panel(
            f"{error.category.value.title()} Error",
            "\n".join(content_lines).strip(),
            color
        )
    
    def _get_category_guidance(self, category: ErrorCategory) -> str:
        """Get quick guidance based on error category"""
        guidance_map = {
            ErrorCategory.NETWORK: "• Check if ports are available\n• Verify firewall settings\n• Try different port numbers",
            ErrorCategory.PERMISSION: "• Run as administrator\n• Check file permissions\n• Verify directory access",
            ErrorCategory.DEPENDENCY: "• Install missing packages\n• Activate virtual environment\n• Check version requirements",
            ErrorCategory.CONFIGURATION: "• Validate JSON syntax\n• Check required fields\n• Restore from backup",
            ErrorCategory.PROCESS: "• Check system resources\n• Verify command availability\n• Review process logs",
            ErrorCategory.SYSTEM: "• Check disk space\n• Verify file paths\n• Review system requirements"
        }
        
        return guidance_map.get(category, "• Check logs for details\n• Try restarting the operation")
    
    def _is_recurring_error(self, error: StartupError) -> bool:
        """Check if this is a recurring error pattern"""
        if len(self.error_history) < 2:
            return False
        
        # Check last 5 errors for similar patterns
        recent_errors = self.error_history[-5:]
        similar_count = sum(1 for e in recent_errors 
                          if e.category == error.category and e.message == error.message)
        
        return similar_count >= 2
    
    def _display_recurring_error_warning(self, error: StartupError):
        """Display warning about recurring errors"""
        warning_content = f"""
🔄 **Recurring Error Detected**

This {error.category.value} error has occurred multiple times.
Consider reviewing the troubleshooting guide or seeking additional help.

**Suggested Actions:**
• Review system configuration
• Check for underlying system issues
• Consider alternative approaches
• Consult documentation or support
        """
        
        self.cli.display_summary_panel(
            "Recurring Error Warning",
            warning_content.strip(),
            "warning"
        )
    
    def offer_recovery_options(self, error: StartupError) -> bool:
        """Offer recovery options to the user"""
        if not error.recovery_actions:
            self.cli.print_status("No automatic recovery options available", "warning")
            return False
        
        # Skip recovery options in quiet mode
        if self.cli.options.verbosity == VerbosityLevel.QUIET:
            return False
        
        self.cli.display_section_header("Recovery Options")
        
        # Display available actions
        action_choices = []
        for i, action in enumerate(error.recovery_actions, 1):
            action_choices.append(f"{i}")
            auto_indicator = " (automatic)" if action.auto_executable else ""
            self.cli.print_status(f"{i}. {action.description}{auto_indicator}", "info")
        
        action_choices.extend(["help", "skip"])
        
        while True:
            choice = self.cli.prompt_choice(
                "Choose a recovery action (or 'help' for more info, 'skip' to continue)",
                action_choices,
                "skip"
            )
            
            if choice == "skip":
                return False
            elif choice == "help":
                self.show_contextual_help(error)
                continue
            else:
                try:
                    action_index = int(choice) - 1
                    if 0 <= action_index < len(error.recovery_actions):
                        action = error.recovery_actions[action_index]
                        
                        # Confirm if required
                        if action.requires_confirmation:
                            if not self.cli.confirm_action(f"Execute: {action.description}?"):
                                continue
                        
                        # Execute the action
                        self.cli.print_status(f"Executing: {action.description}", "info")
                        
                        try:
                            success = action.action()
                            if success:
                                self.cli.print_status("Recovery action completed successfully", "success")
                                return True
                            else:
                                self.cli.print_status("Recovery action failed", "error")
                        except Exception as e:
                            self.cli.print_status(f"Recovery action error: {str(e)}", "error")
                            self.cli.print_debug(traceback.format_exc())
                    else:
                        self.cli.print_status("Invalid choice", "error")
                except ValueError:
                    self.cli.print_status("Invalid choice", "error")
    
    def offer_recovery_options_enhanced(self, error: StartupError) -> bool:
        """Enhanced recovery options with better user interaction"""
        if not error.recovery_actions:
            self._display_no_recovery_options(error)
            return False
        
        # Skip recovery options in quiet mode
        if self.cli.options.verbosity == VerbosityLevel.QUIET:
            return False
        
        # Display recovery options in a formatted table
        self._display_recovery_options_table(error.recovery_actions)
        
        # Group actions by type for better organization
        auto_actions = [a for a in error.recovery_actions if a.auto_executable]
        manual_actions = [a for a in error.recovery_actions if not a.auto_executable]
        
        # Offer to run all automatic actions first
        if auto_actions and self._offer_auto_recovery(auto_actions):
            return True
        
        # Interactive recovery selection
        return self._interactive_recovery_selection(error)
    
    def _display_no_recovery_options(self, error: StartupError):
        """Display message when no recovery options are available"""
        content = f"""
No automatic recovery options are available for this {error.category.value} error.

**Manual Steps:**
1. Review the error details above
2. Check the troubleshooting guide (use 'help' command)
3. Consult the documentation
4. Seek support if the issue persists

**Quick Actions:**
• View logs: Check recent log files for more details
• Get help: Use the built-in help system
• Report issue: Consider reporting this as a bug
        """
        
        self.cli.display_summary_panel(
            "Manual Resolution Required",
            content.strip(),
            "warning"
        )
    
    def _display_recovery_options_table(self, actions: List[RecoveryAction]):
        """Display recovery options in a formatted table"""
        headers = ["#", "Action", "Type", "Description"]
        rows = []
        
        for i, action in enumerate(actions, 1):
            action_type = "Auto" if action.auto_executable else "Manual"
            if action.requires_confirmation:
                action_type += " (Confirm)"
            
            rows.append([
                str(i),
                action.name.replace("_", " ").title(),
                action_type,
                action.description
            ])
        
        self.cli.display_table("Available Recovery Actions", headers, rows)
    
    def _offer_auto_recovery(self, auto_actions: List[RecoveryAction]) -> bool:
        """Offer to run all automatic actions"""
        if not auto_actions:
            return False
        
        action_names = [a.name.replace("_", " ").title() for a in auto_actions]
        actions_str = ", ".join(action_names)
        
        if self.cli.confirm_action(f"Run automatic recovery actions: {actions_str}?"):
            success_count = 0
            
            for action in auto_actions:
                self.cli.print_status(f"Running: {action.description}", "info")
                
                try:
                    if action.action():
                        self.cli.print_status(f"✓ {action.description} completed", "success")
                        success_count += 1
                    else:
                        self.cli.print_status(f"✗ {action.description} failed", "error")
                except Exception as e:
                    self.cli.print_status(f"✗ {action.description} error: {str(e)}", "error")
                    self.cli.print_debug(traceback.format_exc())
            
            if success_count > 0:
                self.cli.print_status(f"Completed {success_count}/{len(auto_actions)} automatic recovery actions", "info")
                return success_count == len(auto_actions)
        
        return False
    
    def _interactive_recovery_selection(self, error: StartupError) -> bool:
        """Interactive recovery action selection with enhanced UX"""
        action_choices = [str(i) for i in range(1, len(error.recovery_actions) + 1)]
        action_choices.extend(["all", "help", "skip"])
        
        while True:
            self.cli.console.print()  # Add spacing
            choice = self.cli.prompt_choice(
                "Choose recovery action ('all' for all actions, 'help' for guidance, 'skip' to continue)",
                action_choices,
                "skip"
            )
            
            if choice == "skip":
                return False
            elif choice == "help":
                self.show_enhanced_help(error)
                continue
            elif choice == "all":
                return self._execute_all_actions(error.recovery_actions)
            else:
                try:
                    action_index = int(choice) - 1
                    if 0 <= action_index < len(error.recovery_actions):
                        if self._execute_single_action(error.recovery_actions[action_index]):
                            return True
                    else:
                        self.cli.print_status("Invalid choice. Please try again.", "error")
                except ValueError:
                    self.cli.print_status("Invalid choice. Please enter a number.", "error")
    
    def _execute_single_action(self, action: RecoveryAction) -> bool:
        """Execute a single recovery action with enhanced feedback"""
        # Show action details
        self.cli.console.print()
        self.cli.print_status(f"Selected: {action.description}", "highlight")
        
        # Confirm if required
        if action.requires_confirmation:
            if not self.cli.confirm_action("Proceed with this action?"):
                return False
        
        # Execute with progress indication
        with self.cli.show_spinner(f"Executing {action.name}..."):
            try:
                success = action.action()
                
                if success:
                    self.cli.print_status("✓ Recovery action completed successfully", "success")
                    
                    # Ask if user wants to continue or stop here
                    if self.cli.confirm_action("Recovery successful. Continue with startup?", default=True):
                        return True
                else:
                    self.cli.print_status("✗ Recovery action failed", "error")
                    
                    # Offer to try another action
                    if self.cli.confirm_action("Try another recovery action?"):
                        return False  # Continue the loop
                    else:
                        return False  # Exit recovery
                        
            except Exception as e:
                self.cli.print_status(f"✗ Recovery action error: {str(e)}", "error")
                self.cli.print_debug(traceback.format_exc())
                
                # Offer to try another action
                if self.cli.confirm_action("Try another recovery action?"):
                    return False  # Continue the loop
                else:
                    return False  # Exit recovery
    
    def _execute_all_actions(self, actions: List[RecoveryAction]) -> bool:
        """Execute all recovery actions in sequence"""
        if not self.cli.confirm_action(f"Execute all {len(actions)} recovery actions?"):
            return False
        
        success_count = 0
        failed_actions = []
        
        for i, action in enumerate(actions, 1):
            self.cli.print_status(f"[{i}/{len(actions)}] {action.description}", "info")
            
            # Skip confirmation for batch execution unless critical
            skip_confirm = not action.requires_confirmation or self.cli.options.auto_confirm
            
            if not skip_confirm:
                if not self.cli.confirm_action(f"Execute: {action.description}?"):
                    self.cli.print_status(f"Skipped: {action.description}", "warning")
                    continue
            
            try:
                if action.action():
                    self.cli.print_status(f"✓ Completed: {action.description}", "success")
                    success_count += 1
                else:
                    self.cli.print_status(f"✗ Failed: {action.description}", "error")
                    failed_actions.append(action.name)
            except Exception as e:
                self.cli.print_status(f"✗ Error in {action.description}: {str(e)}", "error")
                failed_actions.append(action.name)
        
        # Display summary
        self._display_batch_recovery_summary(success_count, len(actions), failed_actions)
        
        return success_count > 0 and len(failed_actions) == 0
    
    def _display_batch_recovery_summary(self, success_count: int, total_count: int, failed_actions: List[str]):
        """Display summary of batch recovery execution"""
        if success_count == total_count:
            status = "success"
            title = "All Recovery Actions Completed"
            content = f"Successfully executed all {total_count} recovery actions."
        elif success_count > 0:
            status = "warning"
            title = "Partial Recovery Success"
            content = f"Completed {success_count}/{total_count} recovery actions.\n\nFailed actions: {', '.join(failed_actions)}"
        else:
            status = "error"
            title = "Recovery Failed"
            content = f"All {total_count} recovery actions failed.\n\nFailed actions: {', '.join(failed_actions)}"
        
        self.cli.display_summary_panel(title, content, status)
    
    def show_contextual_help(self, error: StartupError):
        """Show contextual help for the error"""
        help_content = self.help_system.get_help_for_error(error)
        
        if help_content:
            self.cli.display_summary_panel(
                f"Help: {error.category.value.title()} Error",
                help_content,
                "info"
            )
        else:
            self.cli.print_status("No specific help available for this error", "warning")
            self.show_general_troubleshooting()
    
    def show_enhanced_help(self, error: StartupError):
        """Show enhanced contextual help with interactive options"""
        # Display main help content
        self.show_contextual_help(error)
        
        # Offer additional help options
        help_choices = ["troubleshooting", "examples", "logs", "support", "back"]
        
        while True:
            choice = self.cli.prompt_choice(
                "Additional help options",
                help_choices,
                "back"
            )
            
            if choice == "back":
                break
            elif choice == "troubleshooting":
                self.show_detailed_troubleshooting(error)
            elif choice == "examples":
                self.show_error_examples(error)
            elif choice == "logs":
                self.show_log_guidance(error)
            elif choice == "support":
                self.show_support_information(error)
    
    def show_detailed_troubleshooting(self, error: StartupError):
        """Show detailed troubleshooting steps"""
        troubleshooting_steps = self.help_system.get_troubleshooting_steps(error.category)
        
        if troubleshooting_steps:
            self.cli.display_summary_panel(
                f"Detailed Troubleshooting: {error.category.value.title()}",
                troubleshooting_steps,
                "info"
            )
        else:
            self.show_general_troubleshooting()
    
    def show_error_examples(self, error: StartupError):
        """Show examples of similar errors and solutions"""
        examples = self.help_system.get_error_examples(error.category)
        
        if examples:
            self.cli.display_summary_panel(
                f"Common Examples: {error.category.value.title()} Errors",
                examples,
                "info"
            )
        else:
            self.cli.print_status("No examples available for this error type", "warning")
    
    def show_log_guidance(self, error: StartupError):
        """Show guidance on checking logs"""
        log_guidance = """
**Log File Locations:**
• Startup logs: logs/startup_*.log
• Application logs: logs/wan22_*.log
• Error logs: logs/errors.log

**What to Look For:**
• Timestamp of the error occurrence
• Full stack traces and error details
• Related warning messages before the error
• System resource information

**Log Analysis Tips:**
• Search for error keywords and timestamps
• Look for patterns in recurring errors
• Check for resource exhaustion indicators
• Review configuration loading messages

**Commands to Check Logs:**
• View recent errors: tail -n 50 logs/errors.log
• Search for specific error: grep "error_keyword" logs/*.log
• View startup sequence: cat logs/startup_*.log
        """
        
        self.cli.display_summary_panel(
            "Log Analysis Guidance",
            log_guidance.strip(),
            "info"
        )
    
    def show_support_information(self, error: StartupError):
        """Show support and reporting information"""
        support_info = f"""
**Getting Additional Support:**

**Error Information to Include:**
• Error Code: {error.error_code or 'N/A'}
• Category: {error.category.value}
• Message: {error.message}
• System: Windows {self._get_windows_version()}

**Before Seeking Support:**
1. Try all available recovery actions
2. Check the troubleshooting guide
3. Review recent log files
4. Document steps to reproduce the error

**Support Channels:**
• Documentation: Check project README and docs/
• Issue Tracker: Report bugs and feature requests
• Community: Join discussions and get help
• Logs: Include relevant log excerpts (remove sensitive data)

**Creating a Good Bug Report:**
• Clear description of what you were trying to do
• Exact error message and code
• Steps to reproduce the issue
• System information and environment details
• Log file excerpts (sanitized)
        """
        
        self.cli.display_summary_panel(
            "Support Information",
            support_info.strip(),
            "info"
        )
    
    def _get_windows_version(self) -> str:
        """Get Windows version information"""
        try:
            import platform
            return platform.platform()
        except Exception:
            return "Unknown"
    
    def show_general_troubleshooting(self):
        """Show general troubleshooting tips"""
        tips = [
            "1. Check that all required dependencies are installed",
            "2. Verify that no other applications are using the same ports",
            "3. Ensure you have proper permissions for the project directory",
            "4. Try running as administrator if permission issues persist",
            "5. Check Windows Firewall settings",
            "6. Review the log files for more detailed error information"
        ]
        
        self.cli.display_summary_panel(
            "General Troubleshooting Tips",
            "\n".join(tips),
            "info"
        )


class HelpSystem:
    """Provides context-sensitive help and troubleshooting guidance"""
    
    def __init__(self):
        self.help_topics = {
            ErrorCategory.ENVIRONMENT: self._get_environment_help,
            ErrorCategory.NETWORK: self._get_network_help,
            ErrorCategory.PERMISSION: self._get_permission_help,
            ErrorCategory.DEPENDENCY: self._get_dependency_help,
            ErrorCategory.CONFIGURATION: self._get_configuration_help,
            ErrorCategory.PROCESS: self._get_process_help,
            ErrorCategory.SYSTEM: self._get_system_help,
        }
        
        self.troubleshooting_steps = {
            ErrorCategory.ENVIRONMENT: self._get_environment_troubleshooting,
            ErrorCategory.NETWORK: self._get_network_troubleshooting,
            ErrorCategory.PERMISSION: self._get_permission_troubleshooting,
            ErrorCategory.DEPENDENCY: self._get_dependency_troubleshooting,
            ErrorCategory.CONFIGURATION: self._get_configuration_troubleshooting,
            ErrorCategory.PROCESS: self._get_process_troubleshooting,
            ErrorCategory.SYSTEM: self._get_system_troubleshooting,
        }
        
        self.error_examples = {
            ErrorCategory.ENVIRONMENT: self._get_environment_examples,
            ErrorCategory.NETWORK: self._get_network_examples,
            ErrorCategory.PERMISSION: self._get_permission_examples,
            ErrorCategory.DEPENDENCY: self._get_dependency_examples,
            ErrorCategory.CONFIGURATION: self._get_configuration_examples,
            ErrorCategory.PROCESS: self._get_process_examples,
            ErrorCategory.SYSTEM: self._get_system_examples,
        }
    
    def get_help_for_error(self, error: StartupError) -> Optional[str]:
        """Get help content for a specific error"""
        help_func = self.help_topics.get(error.category)
        if help_func:
            return help_func(error)
        return None
    
    def get_troubleshooting_steps(self, category: ErrorCategory) -> Optional[str]:
        """Get detailed troubleshooting steps for an error category"""
        troubleshooting_func = self.troubleshooting_steps.get(category)
        if troubleshooting_func:
            return troubleshooting_func()
        return None
    
    def get_error_examples(self, category: ErrorCategory) -> Optional[str]:
        """Get examples of common errors in a category"""
        examples_func = self.error_examples.get(category)
        if examples_func:
            return examples_func()
        return None
    
    def _get_environment_help(self, error: StartupError) -> str:
        """Get help for environment-related errors"""
        return """
Environment Error Help:

Common causes:
• Python or Node.js not installed or not in PATH
• Virtual environment not activated
• Incorrect versions of runtime environments
• Missing system dependencies

Solutions:
• Verify Python installation: python --version
• Verify Node.js installation: node --version
• Activate virtual environment: venv\\Scripts\\activate
• Check PATH environment variable
• Install missing system dependencies

Environment Setup:
1. Install Python 3.8+ from python.org
2. Install Node.js 16+ from nodejs.org
3. Create virtual environment: python -m venv venv
4. Activate virtual environment: venv\\Scripts\\activate
        """
    
    def _get_network_help(self, error: StartupError) -> str:
        """Get help for network-related errors"""
        return """
Network Error Help:

Common causes:
• Port already in use by another application
• Windows Firewall blocking the connection
• Network interface not available

Solutions:
• Use 'netstat -an' to check which ports are in use
• Add firewall exceptions for Python and Node.js
• Try using different port numbers
• Check if antivirus software is blocking connections

Windows Firewall:
1. Open Windows Defender Firewall
2. Click "Allow an app or feature through Windows Defender Firewall"
3. Add Python.exe and Node.js to the allowed list
        """
    
    def _get_permission_help(self, error: StartupError) -> str:
        """Get help for permission-related errors"""
        return """
Permission Error Help:

Common causes:
• Insufficient user privileges
• File or folder access restrictions
• Windows UAC (User Account Control) blocking access

Solutions:
• Run Command Prompt as Administrator
• Check file and folder permissions
• Ensure the project directory is not in a restricted location
• Temporarily disable UAC (not recommended for production)

Running as Administrator:
1. Right-click on Command Prompt
2. Select "Run as administrator"
3. Navigate to your project directory
4. Run the startup command again
        """
    
    def _get_dependency_help(self, error: StartupError) -> str:
        """Get help for dependency-related errors"""
        return """
Dependency Error Help:

Common causes:
• Missing Python packages
• Node.js modules not installed
• Incorrect Python or Node.js version
• Virtual environment not activated

Solutions:
• Install Python dependencies: pip install -r requirements.txt
• Install Node.js dependencies: npm install
• Activate virtual environment: venv\\Scripts\\activate
• Check Python version: python --version
• Check Node.js version: node --version

Virtual Environment:
1. Create: python -m venv venv
2. Activate: venv\\Scripts\\activate (Windows)
3. Install: pip install -r requirements.txt
        """
    
    def _get_configuration_help(self, error: StartupError) -> str:
        """Get help for configuration-related errors"""
        return """
Configuration Error Help:

Common causes:
• Invalid JSON syntax in config files
• Missing required configuration keys
• Incorrect configuration values
• Corrupted configuration files

Solutions:
• Validate JSON syntax using online tools
• Check for missing commas or brackets
• Restore from backup configuration
• Reset to default configuration

Configuration Files:
• startup_config.json - Main startup configuration
• config.json - Application configuration
• package.json - Node.js dependencies
• .env - Environment variables
        """
    
    def _get_process_help(self, error: StartupError) -> str:
        """Get help for process-related errors"""
        return """
Process Error Help:

Common causes:
• Process startup timeout
• Command not found
• Process crashed during startup
• Resource conflicts

Solutions:
• Increase timeout values in configuration
• Check that all required commands are available
• Review process logs for crash details
• Ensure sufficient system resources

Process Management:
• Use Task Manager to check running processes
• Kill conflicting processes if necessary
• Monitor system resource usage
• Check process exit codes for error details
        """
    
    def _get_system_help(self, error: StartupError) -> str:
        """Get help for system-related errors"""
        return """
System Error Help:

Common causes:
• File or directory not found
• Disk space issues
• System resource limitations
• Operating system compatibility

Solutions:
• Verify file and directory paths
• Check available disk space
• Monitor system resource usage
• Ensure Windows compatibility

System Checks:
• Disk space: dir (check available space)
• Memory usage: Task Manager > Performance
• File paths: Use absolute paths when possible
• Permissions: Check file properties > Security
        """
    
    # Troubleshooting steps methods
    def _get_environment_troubleshooting(self) -> str:
        """Get detailed environment troubleshooting steps"""
        return """
**Step-by-Step Environment Troubleshooting:**

**Step 1: Verify Runtime Installations**
1. Check Python: python --version (should be 3.8+)
2. Check Node.js: node --version (should be 16+)
3. Check npm: npm --version
4. If missing, download and install from official websites

**Step 2: Check PATH Configuration**
1. Check Python in PATH: where python
2. Check Node.js in PATH: where node
3. Add to PATH if missing: System Properties > Environment Variables
4. Restart command prompt after PATH changes

**Step 3: Virtual Environment Setup**
1. Create virtual environment: python -m venv venv
2. Activate virtual environment: venv\\Scripts\\activate
3. Verify activation: echo %VIRTUAL_ENV%
4. Install dependencies: pip install -r requirements.txt

**Step 4: System Dependencies**
1. Install Visual C++ Redistributable (for Python packages)
2. Install Git (if using git dependencies)
3. Check Windows SDK (for native modules)
4. Update Windows if needed

**Step 5: Environment Variables**
1. Check required environment variables
2. Set missing variables in .env file
3. Verify environment loading: echo %VARIABLE_NAME%
4. Restart application after changes
        """
    
    def _get_network_troubleshooting(self) -> str:
        """Get detailed network troubleshooting steps"""
        return """
**Step-by-Step Network Troubleshooting:**

**Step 1: Check Port Availability**
1. Open Command Prompt as Administrator
2. Run: netstat -an | findstr :8000
3. If port is in use, identify the process: netstat -ano | findstr :8000
4. Kill the process if safe: taskkill /PID <process_id> /F

**Step 2: Test Network Connectivity**
1. Test localhost: ping 127.0.0.1
2. Test network interface: ipconfig /all
3. Check for network adapter issues

**Step 3: Firewall Configuration**
1. Open Windows Defender Firewall
2. Click "Advanced settings"
3. Add inbound rule for Python.exe and Node.js
4. Test with firewall temporarily disabled (re-enable after testing)

**Step 4: Alternative Solutions**
1. Try different port ranges (8080-8090, 3001-3010)
2. Use 0.0.0.0 instead of localhost for binding
3. Check for VPN or proxy interference
4. Restart network services: ipconfig /release && ipconfig /renew
        """
    
    def _get_permission_troubleshooting(self) -> str:
        """Get detailed permission troubleshooting steps"""
        return """
**Step-by-Step Permission Troubleshooting:**

**Step 1: Check Current Permissions**
1. Right-click project folder → Properties → Security
2. Verify your user account has Full Control
3. Check if folder is in restricted location (Program Files, System32)

**Step 2: Run as Administrator**
1. Right-click Command Prompt → "Run as administrator"
2. Navigate to project directory
3. Try running the startup command again

**Step 3: UAC Configuration**
1. Open User Account Control settings
2. Lower UAC level temporarily for testing
3. Restart and test (remember to restore UAC level)

**Step 4: File/Folder Permissions**
1. Take ownership of project folder: takeown /f "project_path" /r /d y
2. Grant permissions: icacls "project_path" /grant %username%:F /t
3. Check for read-only attributes: attrib -r "project_path\\*" /s

**Step 5: Antivirus/Security Software**
1. Check if antivirus is blocking Python/Node.js
2. Add project folder to exclusions
3. Temporarily disable real-time protection for testing
        """
    
    def _get_dependency_troubleshooting(self) -> str:
        """Get detailed dependency troubleshooting steps"""
        return """
**Step-by-Step Dependency Troubleshooting:**

**Step 1: Verify Python Environment**
1. Check Python version: python --version
2. Check pip version: pip --version
3. Verify virtual environment: echo %VIRTUAL_ENV%
4. Activate if needed: venv\\Scripts\\activate

**Step 2: Install Python Dependencies**
1. Update pip: python -m pip install --upgrade pip
2. Install requirements: pip install -r requirements.txt
3. Check for conflicts: pip check
4. List installed packages: pip list

**Step 3: Verify Node.js Environment**
1. Check Node.js version: node --version
2. Check npm version: npm --version
3. Clear npm cache: npm cache clean --force
4. Install dependencies: npm install

**Step 4: Resolve Version Conflicts**
1. Check Python version requirements in requirements.txt
2. Check Node.js version requirements in package.json
3. Use version managers: pyenv (Python) or nvm (Node.js)
4. Create fresh virtual environment if needed

**Step 5: Alternative Installation Methods**
1. Use conda instead of pip: conda install package_name
2. Install from source: pip install git+https://github.com/...
3. Use different package index: pip install -i https://pypi.org/simple/
        """
    
    def _get_configuration_troubleshooting(self) -> str:
        """Get detailed configuration troubleshooting steps"""
        return """
**Step-by-Step Configuration Troubleshooting:**

**Step 1: Validate JSON Syntax**
1. Use online JSON validator: jsonlint.com
2. Check for common issues: missing commas, extra commas, unmatched brackets
3. Verify string quoting: use double quotes, not single quotes

**Step 2: Check Configuration Files**
1. startup_config.json: Main startup configuration
2. config.json: Application configuration
3. package.json: Node.js dependencies and scripts
4. .env files: Environment variables

**Step 3: Backup and Reset**
1. Create backup: copy config.json config.json.backup
2. Reset to defaults: use template or regenerate
3. Gradually restore custom settings

**Step 4: Validate Configuration Values**
1. Check port numbers: must be 1024-65535
2. Verify file paths: use forward slashes or escaped backslashes
3. Check boolean values: true/false (lowercase)
4. Validate URLs: proper protocol and format

**Step 5: Environment Variables**
1. Check .env file syntax: KEY=value (no spaces around =)
2. Verify environment variable loading
3. Use absolute paths for file references
4. Check for special characters that need escaping
        """
    
    def _get_process_troubleshooting(self) -> str:
        """Get detailed process troubleshooting steps"""
        return """
**Step-by-Step Process Troubleshooting:**

**Step 1: Check Process Status**
1. List running processes: tasklist | findstr python
2. Check for zombie processes: wmic process where "name='python.exe'" get processid,commandline
3. Kill stuck processes: taskkill /IM python.exe /F

**Step 2: Verify Commands**
1. Check Python availability: where python
2. Check Node.js availability: where node
3. Verify PATH environment variable
4. Test commands individually

**Step 3: Resource Monitoring**
1. Open Task Manager → Performance tab
2. Check CPU usage during startup
3. Monitor memory consumption
4. Check disk I/O activity

**Step 4: Process Debugging**
1. Run commands manually to isolate issues
2. Check exit codes: echo %ERRORLEVEL%
3. Capture output: command > output.log 2>&1
4. Use verbose/debug flags when available

**Step 5: Timeout and Retry Logic**
1. Increase timeout values in configuration
2. Check for network delays affecting startup
3. Monitor startup sequence timing
4. Implement exponential backoff for retries
        """
    
    def _get_system_troubleshooting(self) -> str:
        """Get detailed system troubleshooting steps"""
        return """
**Step-by-Step System Troubleshooting:**

**Step 1: Check System Resources**
1. Disk space: dir C:\\ (check available space)
2. Memory usage: wmic OS get TotalVisibleMemorySize,FreePhysicalMemory
3. CPU usage: wmic cpu get loadpercentage /value
4. Open file handles: handle.exe (if available)

**Step 2: Verify File System**
1. Check file/folder existence: dir "path"
2. Verify permissions: icacls "path"
3. Test file access: type "file" > nul
4. Check for file locks: handle.exe "file"

**Step 3: System Compatibility**
1. Check Windows version: ver
2. Verify architecture: wmic os get osarchitecture
3. Check .NET Framework: reg query "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\NET Framework Setup\\NDP"
4. Verify Visual C++ Redistributables

**Step 4: Path and Environment**
1. Check PATH variable: echo %PATH%
2. Verify PYTHONPATH: echo %PYTHONPATH%
3. Check working directory: cd
4. Test with absolute paths

**Step 5: System Cleanup**
1. Clear temporary files: del /q /f %TEMP%\\*
2. Restart Windows services if needed
3. Check Windows Event Viewer for system errors
4. Run system file checker: sfc /scannow
        """
    
    # Error examples methods
    def _get_environment_examples(self) -> str:
        """Get examples of common environment errors"""
        return """
**Common Environment Error Examples:**

**Example 1: Python Not Found**
Error: "'python' is not recognized as an internal or external command"
Cause: Python not installed or not in PATH
Solution: Install Python and add to PATH

**Example 2: Node.js Not Found**
Error: "'node' is not recognized as an internal or external command"
Cause: Node.js not installed or not in PATH
Solution: Install Node.js from nodejs.org

**Example 3: Virtual Environment Not Activated**
Error: "ModuleNotFoundError" (but packages are installed)
Cause: Virtual environment not activated
Solution: Run venv\\Scripts\\activate

**Example 4: Wrong Python Version**
Error: "SyntaxError: invalid syntax" (with modern Python syntax)
Cause: Using Python 2.x instead of Python 3.x
Solution: Install Python 3.8+ and update PATH

**Example 5: Missing System Dependencies**
Error: "Microsoft Visual C++ 14.0 is required"
Cause: Missing Visual C++ Build Tools
Solution: Install Visual Studio Build Tools or Visual C++ Redistributable
        """
    
    def _get_network_examples(self) -> str:
        """Get examples of common network errors"""
        return """
**Common Network Error Examples:**

**Example 1: Port Already in Use**
Error: "WinError 10048: Only one usage of each socket address is normally permitted"
Cause: Another application is using port 8000
Solution: Kill the process or use a different port

**Example 2: Permission Denied**
Error: "WinError 10013: An attempt was made to access a socket in a way forbidden"
Cause: Windows Firewall or insufficient privileges
Solution: Run as administrator or add firewall exception

**Example 3: Connection Refused**
Error: "ConnectionRefusedError: [WinError 10061] No connection could be made"
Cause: Target service is not running or unreachable
Solution: Verify service is running and accessible

**Example 4: Address Not Available**
Error: "OSError: [WinError 10049] The requested address is not valid"
Cause: Trying to bind to invalid IP address
Solution: Use 127.0.0.1 or 0.0.0.0 for local binding

**Example 5: Network Unreachable**
Error: "OSError: [WinError 10051] A socket operation was attempted to an unreachable network"
Cause: Network connectivity issues
Solution: Check network adapter and connectivity
        """
    
    def _get_permission_examples(self) -> str:
        """Get examples of common permission errors"""
        return """
**Common Permission Error Examples:**

**Example 1: File Access Denied**
Error: "PermissionError: [Errno 13] Permission denied: 'config.json'"
Cause: Insufficient file permissions
Solution: Run as administrator or change file permissions

**Example 2: Directory Access Denied**
Error: "PermissionError: [Errno 13] Permission denied: 'logs'"
Cause: Cannot create or access directory
Solution: Check folder permissions or create manually

**Example 3: Socket Permission Denied**
Error: "PermissionError: [Errno 13] Permission denied: ('127.0.0.1', 8000)"
Cause: Windows Firewall blocking socket access
Solution: Add firewall exception or run as administrator

**Example 4: Registry Access Denied**
Error: "PermissionError: [WinError 5] Access is denied"
Cause: Trying to access protected registry keys
Solution: Run as administrator or avoid registry operations

**Example 5: UAC Blocking Operation**
Error: "WindowsError: [Error 740] The requested operation requires elevation"
Cause: User Account Control blocking privileged operation
Solution: Right-click and "Run as administrator"
        """
    
    def _get_dependency_examples(self) -> str:
        """Get examples of common dependency errors"""
        return """
**Common Dependency Error Examples:**

**Example 1: Module Not Found**
Error: "ModuleNotFoundError: No module named 'fastapi'"
Cause: Package not installed in current environment
Solution: pip install fastapi

**Example 2: Version Conflict**
Error: "ImportError: cannot import name 'Literal' from 'typing'"
Cause: Python version too old for package requirements
Solution: Upgrade Python or use compatible package version

**Example 3: NPM Package Missing**
Error: "Error: Cannot find module 'react'"
Cause: Node.js dependencies not installed
Solution: npm install

**Example 4: Virtual Environment Not Activated**
Error: "ModuleNotFoundError" (packages installed but not found)
Cause: Virtual environment not activated
Solution: venv\\Scripts\\activate

**Example 5: Binary Dependency Missing**
Error: "ImportError: DLL load failed while importing _ssl"
Cause: Missing system libraries or Visual C++ redistributables
Solution: Install Visual C++ Redistributable packages
        """
    
    def _get_configuration_examples(self) -> str:
        """Get examples of common configuration errors"""
        return """
**Common Configuration Error Examples:**

**Example 1: Invalid JSON Syntax**
Error: "JSONDecodeError: Expecting ',' delimiter: line 5 column 10"
Cause: Missing comma in JSON file
Solution: Add missing comma between JSON elements

**Example 2: Missing Configuration Key**
Error: "KeyError: 'backend_port'"
Cause: Required configuration key not present
Solution: Add missing key to configuration file

**Example 3: Invalid Port Number**
Error: "ValueError: Port must be between 1024 and 65535"
Cause: Port number outside valid range
Solution: Use valid port number (1024-65535)

**Example 4: Invalid File Path**
Error: "FileNotFoundError: [Errno 2] No such file or directory: 'config\\app.json'"
Cause: Configuration file path doesn't exist
Solution: Create file or correct path in configuration

**Example 5: Environment Variable Not Set**
Error: "KeyError: 'DATABASE_URL'"
Cause: Required environment variable missing
Solution: Set environment variable in .env file
        """
    
    def _get_process_examples(self) -> str:
        """Get examples of common process errors"""
        return """
**Common Process Error Examples:**

**Example 1: Command Not Found**
Error: "'python' is not recognized as an internal or external command"
Cause: Python not in PATH or not installed
Solution: Install Python and add to PATH

**Example 2: Process Timeout**
Error: "TimeoutExpired: Command 'npm start' timed out after 30 seconds"
Cause: Process taking longer than expected to start
Solution: Increase timeout or check for blocking issues

**Example 3: Process Exit Code**
Error: "CalledProcessError: Command 'python app.py' returned non-zero exit status 1"
Cause: Application crashed during startup
Solution: Check application logs for specific error

**Example 4: Resource Exhaustion**
Error: "OSError: [WinError 8] Not enough storage is available to process this command"
Cause: Insufficient system resources
Solution: Close other applications or increase system resources

**Example 5: Process Already Running**
Error: "RuntimeError: Server already running on port 8000"
Cause: Previous instance still running
Solution: Kill existing process or use different port
        """
    
    def _get_system_examples(self) -> str:
        """Get examples of common system errors"""
        return """
**Common System Error Examples:**

**Example 1: File Not Found**
Error: "FileNotFoundError: [Errno 2] No such file or directory: 'startup.py'"
Cause: File doesn't exist at specified path
Solution: Verify file exists and path is correct

**Example 2: Disk Space Full**
Error: "OSError: [Errno 28] No space left on device"
Cause: Insufficient disk space
Solution: Free up disk space or use different location

**Example 3: Path Too Long**
Error: "OSError: [Errno 36] File name too long"
Cause: Windows path length limitation exceeded
Solution: Use shorter paths or enable long path support

**Example 4: Invalid Characters**
Error: "OSError: [Errno 22] Invalid argument"
Cause: Invalid characters in file path
Solution: Remove or escape invalid characters

**Example 5: System Resource Limit**
Error: "OSError: [WinError 87] The parameter is incorrect"
Cause: System resource limits exceeded
Solution: Reduce resource usage or increase system limits
        """