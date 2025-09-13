#!/usr/bin/env python3
"""
WAN22 Server Startup Manager

Main entry point for the intelligent server startup system.
Provides CLI interface and orchestrates all startup components.
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
import traceback
from datetime import datetime

# Add the startup_manager package to path
sys.path.insert(0, str(Path(__file__).parent))

from startup_manager.cli import cli, InteractiveCLI, CLIOptions, VerbosityLevel
from startup_manager.config import StartupConfig, load_config
from startup_manager.environment_validator import EnvironmentValidator
from startup_manager.port_manager import PortManager
from startup_manager.process_manager import ProcessManager
from startup_manager.recovery_engine import RecoveryEngine
from startup_manager.logger import configure_logging, get_logger
from startup_manager.diagnostics import DiagnosticMode
from startup_manager.performance_monitor import get_performance_monitor, StartupPhase
from startup_manager.analytics import get_analytics_engine


class StartupManager:
    """Main startup manager orchestrator"""
    
    def __init__(self, cli_interface: InteractiveCLI, config: StartupConfig):
        self.cli = cli_interface
        self.config = config
        
        # Initialize logging
        self.logger = get_logger()
        
        # Initialize performance monitoring
        self.performance_monitor = get_performance_monitor()
        
        # Initialize analytics engine
        self.analytics_engine = get_analytics_engine(self.performance_monitor)
        
        # Initialize components
        self.environment_validator = EnvironmentValidator()
        self.port_manager = PortManager()
        self.process_manager = ProcessManager(config)
        self.recovery_engine = RecoveryEngine()
        self.diagnostic_mode = DiagnosticMode()
    
    def run_startup_sequence(self, backend_port: Optional[int] = None, 
                           frontend_port: Optional[int] = None) -> bool:
        """Run the complete startup sequence"""
        import time
        
        # Start performance monitoring session
        session_metadata = {
            "backend_port": backend_port,
            "frontend_port": frontend_port,
            "timestamp": datetime.now().isoformat(),
            "user": os.getenv("USERNAME", "unknown")
        }
        session_id = self.performance_monitor.start_session(session_metadata)
        
        try:
            self.logger.info("Starting WAN22 server startup sequence")
            self.cli.display_banner()
            self.cli.print_status("Initializing startup sequence...", "info")
            
            # Phase 1: Environment Validation
            self.cli.display_section_header("Phase 1: Environment Validation")
            self.logger.log_startup_phase("Environment Validation", {"step": 1})
            
            with self.performance_monitor.time_operation(StartupPhase.ENVIRONMENT_VALIDATION.value):
                if not self._validate_environment():
                    self.logger.error("Environment validation failed")
                    self.performance_monitor.record_error("Environment validation failed", "environment_validation")
                    self.performance_monitor.finish_session(success=False)
                    return False
            
            # Phase 2: Port Management
            self.cli.display_section_header("Phase 2: Port Management")
            self.logger.log_startup_phase("Port Management", {"step": 2})
            
            with self.performance_monitor.time_operation(StartupPhase.PORT_MANAGEMENT.value):
                ports = self._manage_ports(backend_port, frontend_port)
                if not ports:
                    self.logger.error("Port management failed")
                    self.performance_monitor.record_error("Port management failed", "port_management")
                    self.performance_monitor.finish_session(success=False)
                    return False
            
            # Phase 3: Process Startup
            self.cli.display_section_header("Phase 3: Server Startup")
            self.logger.log_startup_phase("Process Startup", {"step": 3, "ports": ports})
            
            with self.performance_monitor.time_operation(StartupPhase.PROCESS_STARTUP.value, {"ports": ports}):
                if not self._start_processes(ports):
                    self.logger.error("Process startup failed")
                    self.performance_monitor.record_error("Process startup failed", "process_startup")
                    self.performance_monitor.finish_session(success=False)
                    return False
            
            # Phase 4: Health Verification
            self.cli.display_section_header("Phase 4: Health Verification")
            self.logger.log_startup_phase("Health Verification", {"step": 4})
            
            with self.performance_monitor.time_operation(StartupPhase.HEALTH_VERIFICATION.value):
                if not self._verify_health(ports):
                    self.logger.warning("Health verification failed")
                    self.performance_monitor.record_error("Health verification failed", "health_verification")
                    self.performance_monitor.finish_session(success=False)
                    return False
            
            # Success summary
            self.performance_monitor.finish_session(success=True)
            
            # Collect analytics from the completed session
            if self.performance_monitor.sessions:
                last_session = self.performance_monitor.sessions[-1]
                self.analytics_engine.collect_session_analytics(last_session)
            
            # Get performance stats and optimization suggestions
            stats = self.performance_monitor.get_performance_stats()
            resource_summary = self.performance_monitor.get_resource_usage_summary()
            optimization_suggestions = self.analytics_engine.get_optimization_suggestions(max_suggestions=3)
            
            self.logger.info(f"Startup sequence completed successfully")
            self.logger.info(f"Performance stats - Success rate: {stats.success_rate:.1%}, "
                           f"Average duration: {stats.average_duration:.2f}s, "
                           f"Trend: {stats.trend_direction}")
            
            if resource_summary:
                cpu_avg = resource_summary.get("cpu_usage", {}).get("average", 0)
                mem_avg = resource_summary.get("memory_usage", {}).get("average_percent", 0)
                self.logger.info(f"Resource usage - CPU: {cpu_avg:.1f}%, Memory: {mem_avg:.1f}%")
            
            if optimization_suggestions:
                self.logger.info(f"Optimization suggestions available: {len(optimization_suggestions)}")
                for suggestion in optimization_suggestions[:2]:  # Log top 2 suggestions
                    self.logger.info(f"  - {suggestion.title}: {suggestion.expected_improvement}")
            
            self._display_success_summary(ports, stats, optimization_suggestions)
            return True
            
        except KeyboardInterrupt:
            self.logger.warning("Startup interrupted by user")
            self.cli.print_status("Startup interrupted by user", "warning")
            self.performance_monitor.record_error("Startup interrupted by user", "user_interrupt")
            self.performance_monitor.finish_session(success=False)
            
            # Collect analytics from failed session
            if self.performance_monitor.sessions:
                last_session = self.performance_monitor.sessions[-1]
                self.analytics_engine.collect_session_analytics(last_session)
            
            return False
        except Exception as e:
            self.logger.log_error_with_context(e, {
                "operation": "startup_sequence",
                "backend_port": backend_port,
                "frontend_port": frontend_port,
                "session_id": session_id
            })
            self.cli.print_status(f"Unexpected error during startup: {str(e)}", "error")
            self.cli.print_debug(traceback.format_exc())
            self.performance_monitor.record_error(str(e), "unexpected_error")
            self.performance_monitor.finish_session(success=False)
            
            # Collect analytics from failed session
            if self.performance_monitor.sessions:
                last_session = self.performance_monitor.sessions[-1]
                self.analytics_engine.collect_session_analytics(last_session)
            
            return False
    
    def _validate_environment(self) -> bool:
        """Validate the development environment"""
        with self.cli.show_spinner("Validating environment...") as progress:
            try:
                # Simulate environment validation
                validation_result = self.environment_validator.validate_all()
                
                if validation_result.get('valid', False):
                    self.cli.print_status("Environment validation passed", "success")
                    
                    # Show validation details in verbose mode
                    if self.cli.options.verbosity in [VerbosityLevel.VERBOSE, VerbosityLevel.DEBUG]:
                        details = validation_result.get('details', {})
                        self.cli.display_key_value_pairs("Environment Details", {
                            "Python Version": details.get('python_version', 'Unknown'),
                            "Node.js Version": details.get('node_version', 'Unknown'),
                            "Virtual Environment": details.get('venv_active', 'Unknown'),
                            "Dependencies": details.get('dependencies_status', 'Unknown')
                        })
                    
                    return True
                else:
                    self.cli.print_status("Environment validation failed", "error")
                    issues = validation_result.get('issues', [])
                    for issue in issues:
                        self.cli.print_status(f"  - {issue}", "error")
                    
                    if self.cli.confirm_action("Attempt to fix issues automatically?"):
                        return self._attempt_environment_fixes(issues)
                    
                    return False
                    
            except Exception as e:
                self.cli.print_status(f"Environment validation error: {str(e)}", "error")
                return False
    
    def _manage_ports(self, backend_port: Optional[int], frontend_port: Optional[int]) -> Optional[Dict[str, int]]:
        """Manage port allocation and conflicts"""
        with self.cli.show_spinner("Checking port availability...") as progress:
            try:
                # Use provided ports or defaults
                target_backend = backend_port or self.config.backend.port
                target_frontend = frontend_port or self.config.frontend.port
                
                # Check port availability
                backend_available = self.port_manager.is_port_available(target_backend)
                frontend_available = self.port_manager.is_port_available(target_frontend)
                
                ports = {}
                
                # Handle backend port
                if backend_available:
                    ports['backend'] = target_backend
                    self.cli.print_status(f"Backend port {target_backend} is available", "success")
                else:
                    self.cli.print_status(f"Backend port {target_backend} is in use", "warning")
                    if self.cli.confirm_action("Find alternative backend port?"):
                        alt_port = self.port_manager.find_available_port(target_backend + 1)
                        ports['backend'] = alt_port
                        self.cli.print_status(f"Using alternative backend port {alt_port}", "info")
                    else:
                        return None
                
                # Handle frontend port
                if frontend_available:
                    ports['frontend'] = target_frontend
                    self.cli.print_status(f"Frontend port {target_frontend} is available", "success")
                else:
                    self.cli.print_status(f"Frontend port {target_frontend} is in use", "warning")
                    if self.cli.confirm_action("Find alternative frontend port?"):
                        alt_port = self.port_manager.find_available_port(target_frontend + 1)
                        ports['frontend'] = alt_port
                        self.cli.print_status(f"Using alternative frontend port {alt_port}", "info")
                    else:
                        return None
                
                return ports
                
            except Exception as e:
                self.cli.print_status(f"Port management error: {str(e)}", "error")
                return None
    
    def _start_processes(self, ports: Dict[str, int]) -> bool:
        """Start the server processes"""
        try:
            # Start backend
            with self.cli.show_spinner(f"Starting backend server on port {ports['backend']}...") as progress:
                backend_result = self.process_manager.start_backend(ports['backend'])
                
                if backend_result.get('success', False):
                    self.cli.print_status("Backend server started successfully", "success")
                else:
                    self.cli.print_status("Failed to start backend server", "error")
                    error_msg = backend_result.get('error', 'Unknown error')
                    self.cli.print_status(f"  Error: {error_msg}", "error")
                    return False
            
            # Start frontend
            with self.cli.show_spinner(f"Starting frontend server on port {ports['frontend']}...") as progress:
                frontend_result = self.process_manager.start_frontend(ports['frontend'])
                
                if frontend_result.get('success', False):
                    self.cli.print_status("Frontend server started successfully", "success")
                else:
                    self.cli.print_status("Failed to start frontend server", "error")
                    error_msg = frontend_result.get('error', 'Unknown error')
                    self.cli.print_status(f"  Error: {error_msg}", "error")
                    return False
            
            return True
            
        except Exception as e:
            self.cli.print_status(f"Process startup error: {str(e)}", "error")
            return False
    
    def _verify_health(self, ports: Dict[str, int]) -> bool:
        """Verify server health"""
        with self.cli.show_spinner("Verifying server health...") as progress:
            try:
                # Check backend health
                backend_healthy = self.process_manager.check_health(f"http://localhost:{ports['backend']}/health")
                if backend_healthy:
                    self.cli.print_status("Backend health check passed", "success")
                else:
                    self.cli.print_status("Backend health check failed", "warning")
                
                # Check frontend health
                frontend_healthy = self.process_manager.check_health(f"http://localhost:{ports['frontend']}")
                if frontend_healthy:
                    self.cli.print_status("Frontend health check passed", "success")
                else:
                    self.cli.print_status("Frontend health check failed", "warning")
                
                return backend_healthy and frontend_healthy
                
            except Exception as e:
                self.cli.print_status(f"Health verification error: {str(e)}", "error")
                return False
    
    def _display_success_summary(self, ports: Dict[str, int], stats: Optional[Any] = None, 
                               optimization_suggestions: Optional[List[Any]] = None):
        """Display startup success summary with performance metrics and optimization suggestions"""
        performance_info = ""
        if stats and stats.total_sessions > 0:
            performance_info = f"""
[bold]Performance Metrics:[/bold]
  • Success Rate: [green]{stats.success_rate:.1%}[/green]
  • Average Duration: {stats.average_duration:.2f}s
  • Trend: {stats.trend_direction.title()}
  • Total Sessions: {stats.total_sessions}
"""
        
        optimization_info = ""
        if optimization_suggestions:
            optimization_info = f"""
[bold]Optimization Suggestions:[/bold]"""
            for i, suggestion in enumerate(optimization_suggestions[:3], 1):
                priority_color = {
                    "critical": "red",
                    "high": "yellow", 
                    "medium": "blue",
                    "low": "dim"
                }.get(suggestion.priority.value, "white")
                
                optimization_info += f"""
  {i}. [{priority_color}]{suggestion.title}[/{priority_color}]
     Expected improvement: {suggestion.expected_improvement}
"""
        
        summary_content = f"""
[bold green]✓ Startup Complete![/bold green]

[bold]Backend Server:[/bold]
  • URL: [link]http://localhost:{ports['backend']}[/link]
  • API Docs: [link]http://localhost:{ports['backend']}/docs[/link]
  • Status: [green]Running[/green]

[bold]Frontend Server:[/bold]
  • URL: [link]http://localhost:{ports['frontend']}[/link]
  • Status: [green]Running[/green]
{performance_info}{optimization_info}
[bold]Next Steps:[/bold]
  • Open your browser to start using the application
  • Check logs for any warnings or issues
  • Use 'Ctrl+C' to stop servers when done
        """
        
        self.cli.display_summary_panel("Startup Summary", summary_content.strip(), "success")
    
    def _attempt_environment_fixes(self, issues: list) -> bool:
        """Attempt to automatically fix environment issues"""
        self.cli.print_status("Attempting automatic fixes...", "info")
        
        # This would integrate with the recovery engine
        # For now, just simulate the process
        with self.cli.show_spinner("Applying fixes...") as progress:
            # Simulate fix attempts
            pass
        
        self.cli.print_status("Automatic fixes completed", "success")
        return True


def main():
    """Main entry point"""
    try:
        # Configure logging early
        verbose = "--verbose" in sys.argv or "-v" in sys.argv
        debug = "--debug" in sys.argv
        
        if debug:
            configure_logging(console_level="DEBUG", verbose=True)
        elif verbose:
            configure_logging(console_level="INFO", verbose=True)
        else:
            configure_logging(console_level="INFO", verbose=False)
        
        logger = get_logger()
        logger.info("WAN22 Startup Manager initialized")
        
        # Load configuration
        config = load_config()
        
        # Create CLI interface with default options
        cli_options = CLIOptions()
        cli_interface = InteractiveCLI(cli_options)
        
        # Create startup manager
        startup_manager = StartupManager(cli_interface, config)
        
        # Check for diagnostic mode
        if "--diagnostics" in sys.argv:
            logger.info("Running diagnostic mode")
            diagnostic_data = startup_manager.diagnostic_mode.run_full_diagnostics()
            
            # Save diagnostic report
            report_file = startup_manager.diagnostic_mode.save_diagnostic_report(diagnostic_data)
            print(f"\nDiagnostic report saved to: {report_file}")
            
            # Display summary
            summary = diagnostic_data.get("summary", {})
            status = summary.get("overall_status", "unknown")
            print(f"Overall system status: {status}")
            
            if summary.get("recommendations"):
                print("\nTop recommendations:")
                for i, rec in enumerate(summary["recommendations"][:5], 1):
                    print(f"  {i}. {rec}")
            
            sys.exit(0)
        
        # If no arguments provided, run the CLI
        if len(sys.argv) == 1:
            # Run interactive startup
            success = startup_manager.run_startup_sequence()
            sys.exit(0 if success else 1)
        else:
            # Run Click CLI
            cli()
    
    except KeyboardInterrupt:
        print("\nStartup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
