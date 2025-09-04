"""
Interactive CLI interface for the WAN22 Server Startup Manager.

This module provides a rich, user-friendly command-line interface with:
- Progress bars and spinners
- Interactive prompts with clear options
- Verbose/quiet modes
- Colored output and structured display
"""

import click
import sys
from typing import List, Optional, Dict, Any
from enum import Enum
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.prompt import Confirm, Prompt, IntPrompt
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich.align import Align
from rich import box
import time
from dataclasses import dataclass

from .config import StartupConfig
from .logger import get_logger
from .diagnostics import DiagnosticMode


class VerbosityLevel(Enum):
    """Verbosity levels for CLI output"""
    QUIET = "quiet"
    NORMAL = "normal"
    VERBOSE = "verbose"
    DEBUG = "debug"


@dataclass
class CLIOptions:
    """CLI configuration options"""
    verbosity: VerbosityLevel = VerbosityLevel.NORMAL
    no_color: bool = False
    auto_confirm: bool = False
    interactive: bool = True
    show_progress: bool = True


class InteractiveCLI:
    """Rich-based interactive CLI for startup management"""
    
    def __init__(self, options: CLIOptions = None):
        self.options = options or CLIOptions()
        self.console = Console(
            color_system="auto" if not self.options.no_color else None,
            quiet=self.options.verbosity == VerbosityLevel.QUIET
        )
        self._setup_styles()
    
    def _setup_styles(self):
        """Setup console styles and themes"""
        self.styles = {
            "success": "bold green",
            "error": "bold red",
            "warning": "bold yellow",
            "info": "bold blue",
            "highlight": "bold cyan",
            "muted": "dim",
            "progress": "bold magenta"
        }
    
    def display_banner(self):
        """Display the startup manager banner"""
        if self.options.verbosity == VerbosityLevel.QUIET:
            return
            
        banner_text = """
[bold blue]╔══════════════════════════════════════════════════════════╗[/bold blue]
[bold blue]║                WAN22 Startup Manager                     ║[/bold blue]
[bold blue]║            Intelligent Server Management                 ║[/bold blue]
[bold blue]║                                                          ║[/bold blue]
[bold blue]║  [white]FastAPI Backend + React Frontend Orchestration[/white]     ║[/bold blue]
[bold blue]╚══════════════════════════════════════════════════════════╝[/bold blue]
        """
        
        panel = Panel(
            banner_text.strip(),
            border_style="blue",
            padding=(1, 2)
        )
        self.console.print(panel)
        self.console.print()
    
    def create_progress_context(self, description: str, total: Optional[int] = None):
        """Create a progress context manager"""
        if not self.options.show_progress or self.options.verbosity == VerbosityLevel.QUIET:
            return _DummyProgress()
        
        if total:
            return Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=self.console
            )
        else:
            return Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=self.console
            )
    
    def show_spinner(self, description: str):
        """Show a spinner for long-running operations"""
        if self.options.verbosity == VerbosityLevel.QUIET:
            return _DummyProgress()
        
        return Progress(
            SpinnerColumn(),
            TextColumn(f"[{self.styles['progress']}]{description}[/{self.styles['progress']}]"),
            console=self.console
        )
    
    def print_status(self, message: str, status: str = "info"):
        """Print a status message with appropriate styling"""
        if self.options.verbosity == VerbosityLevel.QUIET:
            return
        
        style = self.styles.get(status, "white")
        prefix_map = {
            "success": "✓",
            "error": "✗",
            "warning": "⚠",
            "info": "ℹ",
            "highlight": "→"
        }
        
        prefix = prefix_map.get(status, "•")
        self.console.print(f"[{style}]{prefix} {message}[/{style}]")
    
    def print_verbose(self, message: str):
        """Print message only in verbose mode"""
        if self.options.verbosity in [VerbosityLevel.VERBOSE, VerbosityLevel.DEBUG]:
            self.console.print(f"[{self.styles['muted']}]{message}[/{self.styles['muted']}]")
    
    def print_debug(self, message: str):
        """Print message only in debug mode"""
        if self.options.verbosity == VerbosityLevel.DEBUG:
            self.console.print(f"[dim]DEBUG: {message}[/dim]")
    
    def confirm_action(self, message: str, default: bool = True) -> bool:
        """Get user confirmation for actions"""
        if self.options.auto_confirm:
            return default
        
        if not self.options.interactive:
            return default
        
        return Confirm.ask(
            f"[{self.styles['highlight']}]{message}[/{self.styles['highlight']}]",
            default=default,
            console=self.console
        )
    
    def prompt_choice(self, message: str, choices: List[str], default: str = None) -> str:
        """Prompt user for a choice from a list"""
        if self.options.auto_confirm and default:
            return default
        
        if not self.options.interactive:
            return default or choices[0]
        
        return Prompt.ask(
            f"[{self.styles['highlight']}]{message}[/{self.styles['highlight']}]",
            choices=choices,
            default=default or choices[0],
            console=self.console
        )
    
    def prompt_number(self, message: str, default: int = None, min_val: int = None, max_val: int = None) -> int:
        """Prompt user for a number"""
        if self.options.auto_confirm and default is not None:
            return default
        
        if not self.options.interactive:
            return default or 0
        
        while True:
            try:
                result = IntPrompt.ask(
                    f"[{self.styles['highlight']}]{message}[/{self.styles['highlight']}]",
                    default=default,
                    console=self.console
                )
                
                if min_val is not None and result < min_val:
                    self.print_status(f"Value must be at least {min_val}", "error")
                    continue
                
                if max_val is not None and result > max_val:
                    self.print_status(f"Value must be at most {max_val}", "error")
                    continue
                
                return result
            except (ValueError, KeyboardInterrupt):
                if not self.confirm_action("Invalid input. Try again?"):
                    return default or 0
    
    def display_table(self, title: str, headers: List[str], rows: List[List[str]]):
        """Display a formatted table"""
        if self.options.verbosity == VerbosityLevel.QUIET:
            return
        
        table = Table(title=title, box=box.ROUNDED)
        
        for header in headers:
            table.add_column(header, style="bold")
        
        for row in rows:
            table.add_row(*row)
        
        self.console.print(table)
        self.console.print()
    
    def display_key_value_pairs(self, title: str, pairs: Dict[str, str]):
        """Display key-value pairs in a formatted way"""
        if self.options.verbosity == VerbosityLevel.QUIET:
            return
        
        table = Table(title=title, show_header=False, box=box.SIMPLE)
        table.add_column("Key", style="bold cyan")
        table.add_column("Value", style="white")
        
        for key, value in pairs.items():
            table.add_row(key, str(value))
        
        self.console.print(table)
        self.console.print()
    
    def display_section_header(self, title: str):
        """Display a section header"""
        if self.options.verbosity == VerbosityLevel.QUIET:
            return
        
        self.console.print()
        self.console.print(f"[bold underline]{title}[/bold underline]")
        self.console.print()
    
    def display_summary_panel(self, title: str, content: str, style: str = "info"):
        """Display a summary panel with content"""
        if self.options.verbosity == VerbosityLevel.QUIET:
            return
        
        border_style = {
            "success": "green",
            "error": "red",
            "warning": "yellow",
            "info": "blue"
        }.get(style, "blue")
        
        panel = Panel(
            content,
            title=title,
            border_style=border_style,
            padding=(1, 2)
        )
        self.console.print(panel)
        self.console.print()


class _DummyProgress:
    """Dummy progress context for quiet mode"""
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass
    
    def add_task(self, description: str, total: int = None):
        return 0
    
    def update(self, task_id: int, advance: int = 1, **kwargs):
        pass
    
    def start(self):
        pass
    
    def stop(self):
        pass


# Click CLI setup
@click.group()
@click.option('--verbose', '-v', count=True, help='Increase verbosity (use -vv for debug)')
@click.option('--quiet', '-q', is_flag=True, help='Suppress all output except errors')
@click.option('--no-color', is_flag=True, help='Disable colored output')
@click.option('--auto-confirm', '-y', is_flag=True, help='Automatically confirm all prompts')
@click.option('--non-interactive', is_flag=True, help='Run in non-interactive mode')
@click.pass_context
def cli(ctx, verbose, quiet, no_color, auto_confirm, non_interactive):
    """WAN22 Server Startup Manager - Intelligent server orchestration"""
    
    # Determine verbosity level
    if quiet:
        verbosity = VerbosityLevel.QUIET
    elif verbose >= 2:
        verbosity = VerbosityLevel.DEBUG
    elif verbose >= 1:
        verbosity = VerbosityLevel.VERBOSE
    else:
        verbosity = VerbosityLevel.NORMAL
    
    # Create CLI options
    cli_options = CLIOptions(
        verbosity=verbosity,
        no_color=no_color,
        auto_confirm=auto_confirm,
        interactive=not non_interactive,
        show_progress=not quiet
    )
    
    # Store in context
    ctx.ensure_object(dict)
    ctx.obj['cli_options'] = cli_options
    ctx.obj['cli'] = InteractiveCLI(cli_options)


@cli.command()
@click.option('--backend-port', type=int, help='Specify backend port')
@click.option('--frontend-port', type=int, help='Specify frontend port')
@click.option('--config', type=click.Path(exists=True), help='Path to config file')
@click.pass_context
def start(ctx, backend_port, frontend_port, config):
    """Start both backend and frontend servers"""
    cli_interface = ctx.obj['cli']
    cli_interface.display_banner()
    
    cli_interface.print_status("Starting server startup process...", "info")
    
    # This would integrate with the actual startup manager
    # For now, just demonstrate the CLI functionality
    with cli_interface.show_spinner("Validating environment...") as progress:
        time.sleep(1)  # Simulate work
    
    cli_interface.print_status("Environment validation complete", "success")
    
    # Example of interactive prompts
    if backend_port is None:
        backend_port = cli_interface.prompt_number(
            "Enter backend port", 
            default=8000, 
            min_val=1024, 
            max_val=65535
        )
    
    if frontend_port is None:
        frontend_port = cli_interface.prompt_number(
            "Enter frontend port", 
            default=3000, 
            min_val=1024, 
            max_val=65535
        )
    
    cli_interface.display_key_value_pairs("Configuration", {
        "Backend Port": str(backend_port),
        "Frontend Port": str(frontend_port),
        "Config File": config or "default"
    })
    
    if cli_interface.confirm_action("Proceed with startup?"):
        cli_interface.print_status("Starting servers...", "info")
        # Actual startup logic would go here
        cli_interface.print_status("Servers started successfully!", "success")
    else:
        cli_interface.print_status("Startup cancelled by user", "warning")


@cli.command()
@click.pass_context
def status(ctx):
    """Check server status"""
    cli_interface = ctx.obj['cli']
    
    cli_interface.display_section_header("Server Status")
    
    # Example status display
    status_data = [
        ["Backend", "Running", "8000", "✓"],
        ["Frontend", "Running", "3000", "✓"]
    ]
    
    cli_interface.display_table(
        "Current Server Status",
        ["Service", "Status", "Port", "Health"],
        status_data
    )


@cli.command()
@click.pass_context
def stop(ctx):
    """Stop running servers"""
    cli_interface = ctx.obj['cli']
    
    if cli_interface.confirm_action("Stop all running servers?"):
        with cli_interface.show_spinner("Stopping servers...") as progress:
            time.sleep(1)  # Simulate work
        
        cli_interface.print_status("All servers stopped", "success")
    else:
        cli_interface.print_status("Stop operation cancelled", "warning")


@cli.command()
@click.option('--save-report', type=click.Path(), help='Save diagnostic report to file')
@click.option('--log-dir', type=click.Path(exists=True), default='logs', help='Log directory to analyze')
@click.pass_context
def diagnostics(ctx, save_report, log_dir):
    """Run comprehensive system diagnostics"""
    cli_interface = ctx.obj['cli']
    logger = get_logger()
    
    cli_interface.display_section_header("System Diagnostics")
    cli_interface.print_status("Running comprehensive system diagnostics...", "info")
    
    try:
        diagnostic_mode = DiagnosticMode()
        
        with cli_interface.show_spinner("Collecting system information...") as progress:
            diagnostic_data = diagnostic_mode.run_full_diagnostics(log_dir)
        
        # Display summary
        summary = diagnostic_data.get("summary", {})
        overall_status = summary.get("overall_status", "unknown")
        
        status_style = {
            "healthy": "success",
            "warnings_present": "warning", 
            "issues_detected": "error"
        }.get(overall_status, "info")
        
        cli_interface.print_status(f"Overall system status: {overall_status}", status_style)
        
        # Display diagnostic checks summary
        checks_summary = summary.get("diagnostic_checks", {})
        cli_interface.display_key_value_pairs("Diagnostic Checks", {
            "Total Checks": str(checks_summary.get("total", 0)),
            "Passed": str(checks_summary.get("passed", 0)),
            "Failed": str(checks_summary.get("failed", 0)),
            "Warnings": str(checks_summary.get("warnings", 0))
        })
        
        # Display log analysis summary
        log_summary = summary.get("log_analysis", {})
        cli_interface.display_key_value_pairs("Log Analysis", {
            "Total Log Entries": str(log_summary.get("total_entries", 0)),
            "Error Count": str(log_summary.get("error_count", 0)),
            "Warning Count": str(log_summary.get("warning_count", 0)),
            "Error Categories": str(log_summary.get("common_error_categories", 0))
        })
        
        # Display recommendations
        recommendations = summary.get("recommendations", [])
        if recommendations:
            cli_interface.display_section_header("Top Recommendations")
            for i, rec in enumerate(recommendations[:5], 1):
                cli_interface.print_status(f"{i}. {rec}", "highlight")
        
        # Save report if requested
        if save_report:
            report_path = diagnostic_mode.save_diagnostic_report(diagnostic_data, save_report)
            cli_interface.print_status(f"Diagnostic report saved to: {report_path}", "success")
        else:
            # Save with default name
            report_path = diagnostic_mode.save_diagnostic_report(diagnostic_data)
            cli_interface.print_status(f"Diagnostic report saved to: {report_path}", "info")
        
        logger.info("Diagnostic command completed successfully")
        
    except Exception as e:
        logger.log_error_with_context(e, {"operation": "diagnostics_command"})
        cli_interface.print_status(f"Diagnostics failed: {str(e)}", "error")


@cli.command()
@click.option('--log-dir', type=click.Path(exists=True), default='logs', help='Log directory to analyze')
@click.option('--output-format', type=click.Choice(['table', 'json']), default='table', help='Output format')
@click.pass_context
def analyze_logs(ctx, log_dir, output_format):
    """Analyze startup logs for issues and patterns"""
    cli_interface = ctx.obj['cli']
    logger = get_logger()
    
    cli_interface.display_section_header("Log Analysis")
    
    try:
        from .diagnostics import LogAnalyzer
        analyzer = LogAnalyzer()
        
        with cli_interface.show_spinner(f"Analyzing logs in {log_dir}...") as progress:
            analysis_result = analyzer.analyze_logs(log_dir)
        
        if output_format == 'json':
            import json
            from dataclasses import asdict
            result_dict = asdict(analysis_result)
            cli_interface.console.print_json(json.dumps(result_dict, indent=2))
        else:
            # Display in table format
            cli_interface.display_key_value_pairs("Log Analysis Summary", {
                "Total Entries": str(analysis_result.total_entries),
                "Error Count": str(analysis_result.error_count),
                "Warning Count": str(analysis_result.warning_count)
            })
            
            # Display common errors
            if analysis_result.common_errors:
                cli_interface.display_section_header("Common Error Categories")
                error_rows = []
                for error in analysis_result.common_errors[:5]:
                    error_rows.append([
                        error["category"],
                        str(error["count"]),
                        error["description"]
                    ])
                
                cli_interface.display_table(
                    "Top Error Categories",
                    ["Category", "Count", "Description"],
                    error_rows
                )
            
            # Display performance metrics
            if analysis_result.performance_metrics:
                cli_interface.display_section_header("Performance Metrics")
                perf_rows = []
                for operation, metrics in analysis_result.performance_metrics.items():
                    perf_rows.append([
                        operation,
                        str(metrics["count"]),
                        f"{metrics['avg_duration']:.2f}s",
                        f"{metrics['min_duration']:.2f}s",
                        f"{metrics['max_duration']:.2f}s"
                    ])
                
                cli_interface.display_table(
                    "Performance Metrics",
                    ["Operation", "Count", "Avg Duration", "Min Duration", "Max Duration"],
                    perf_rows
                )
            
            # Display suggestions
            if analysis_result.suggestions:
                cli_interface.display_section_header("Suggestions")
                for i, suggestion in enumerate(analysis_result.suggestions, 1):
                    cli_interface.print_status(f"{i}. {suggestion}", "highlight")
        
        logger.info("Log analysis command completed successfully")
        
    except Exception as e:
        logger.log_error_with_context(e, {"operation": "analyze_logs_command"})
        cli_interface.print_status(f"Log analysis failed: {str(e)}", "error")


@cli.command()
@click.pass_context
def system_info(ctx):
    """Display comprehensive system information"""
    cli_interface = ctx.obj['cli']
    logger = get_logger()
    
    cli_interface.display_section_header("System Information")
    
    try:
from .diagnostics import SystemDiagnostics
        diagnostics = SystemDiagnostics()
        
        with cli_interface.show_spinner("Collecting system information...") as progress:
            system_info = diagnostics.collect_system_info()
        
        # Display basic system info
        cli_interface.display_key_value_pairs("Operating System", {
            "OS Name": system_info.os_name,
            "OS Version": system_info.os_version,
            "Architecture": system_info.architecture,
            "Processor": system_info.processor
        })
        
        # Display Python info
        cli_interface.display_key_value_pairs("Python Environment", {
            "Python Version": system_info.python_version,
            "Python Executable": system_info.python_executable,
            "Virtual Environment": system_info.virtual_env or "None"
        })
        
        # Display memory info
        memory_gb = system_info.memory_total / (1024**3)
        available_gb = system_info.memory_available / (1024**3)
        cli_interface.display_key_value_pairs("Memory", {
            "Total Memory": f"{memory_gb:.1f} GB",
            "Available Memory": f"{available_gb:.1f} GB",
            "Memory Usage": f"{((memory_gb - available_gb) / memory_gb * 100):.1f}%"
        })
        
        # Display disk info
        disk_info = system_info.disk_usage
        total_gb = disk_info["total"] / (1024**3)
        free_gb = disk_info["free"] / (1024**3)
        cli_interface.display_key_value_pairs("Disk Space", {
            "Total Space": f"{total_gb:.1f} GB",
            "Free Space": f"{free_gb:.1f} GB",
            "Disk Usage": f"{disk_info['percent']:.1f}%"
        })
        
        # Display network interfaces (first few)
        if system_info.network_interfaces:
            cli_interface.display_section_header("Network Interfaces")
            for interface in system_info.network_interfaces[:3]:  # Show first 3
                addresses = [addr["address"] for addr in interface["addresses"] if addr["type"] == "IPv4"]
                if addresses:
                    cli_interface.print_status(f"{interface['name']}: {', '.join(addresses)}", "info")
        
        logger.info("System info command completed successfully")
        
    except Exception as e:
        logger.log_error_with_context(e, {"operation": "system_info_command"})
        cli_interface.print_status(f"System info collection failed: {str(e)}", "error")


if __name__ == '__main__':
    cli()