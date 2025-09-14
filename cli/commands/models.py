#!/usr/bin/env python3
"""
Model Orchestrator CLI Commands
Provides CLI interface for WAN2.2 model management through the Model Orchestrator.
"""

import typer
import json
import sys
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich import print as rprint

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.core.model_orchestrator.model_registry import ModelRegistry
from backend.core.model_orchestrator.model_resolver import ModelResolver
from backend.core.model_orchestrator.model_ensurer import ModelEnsurer, ModelStatus
from backend.core.model_orchestrator.lock_manager import LockManager
from backend.core.model_orchestrator.storage_backends.hf_store import HFStore
from backend.core.model_orchestrator.garbage_collector import GarbageCollector, GCConfig, GCTrigger
from backend.core.model_orchestrator.exceptions import ModelOrchestratorError

app = typer.Typer(
    name="models",
    help="Model Orchestrator - Unified model management for WAN2.2",
    rich_markup_mode="rich"
)

console = Console()

def _get_orchestrator_components():
    """Initialize and return orchestrator components."""
    try:
        # Get configuration from environment or defaults
        models_root = Path.cwd() / "models"
        manifest_path = Path.cwd() / "config" / "models.toml"
        
        # Initialize components
        registry = ModelRegistry(str(manifest_path))
        resolver = ModelResolver(str(models_root))
        lock_manager = LockManager(str(models_root / ".locks"))
        
        # Initialize storage backends
        storage_backends = [HFStore()]
        
        ensurer = ModelEnsurer(
            registry=registry,
            resolver=resolver,
            lock_manager=lock_manager,
            storage_backends=storage_backends,
            enable_deduplication=True
        )
        
        # Initialize garbage collector with component deduplicator
        gc_config = GCConfig()
        garbage_collector = GarbageCollector(
            registry, 
            resolver, 
            gc_config,
            component_deduplicator=ensurer.component_deduplicator
        )
        
        # Connect garbage collector to ensurer
        ensurer.set_garbage_collector(garbage_collector)
        
        return registry, ensurer, garbage_collector
        
    except Exception as e:
        console.print(f"[red]Error initializing model orchestrator: {e}[/red]")
        raise typer.Exit(1)

def _format_bytes(bytes_value: int) -> str:
    """Format bytes in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"

def _get_status_color(status: ModelStatus) -> str:
    """Get color for model status."""
    status_colors = {
        ModelStatus.COMPLETE: "green",
        ModelStatus.NOT_PRESENT: "red",
        ModelStatus.PARTIAL: "yellow",
        ModelStatus.VERIFYING: "blue",
        ModelStatus.CORRUPT: "red"
    }
    return status_colors.get(status, "white")

@app.command("status")
def status_command(
    model_id: Optional[str] = typer.Option(None, "--model", "-m", help="Check specific model status"),
    variant: Optional[str] = typer.Option(None, "--variant", "-v", help="Check specific variant"),
    json_output: bool = typer.Option(False, "--json", help="Output in JSON format"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed information")
):
    """
    Show status of WAN2.2 models managed by the orchestrator.
    
    Examples:
      wan models status                    # Show all models
      wan models status --model t2v-A14B   # Show specific model
      wan models status --json             # JSON output for automation
    """
    try:
        registry, ensurer, _ = _get_orchestrator_components()
        
        # Get models to check
        if model_id:
            model_ids = [model_id]
        else:
            model_ids = registry.list_models()
        
        results = {}
        
        for mid in model_ids:
            try:
                status_info = ensurer.status(mid, variant)
                results[mid] = {
                    "status": status_info.status.value,
                    "local_path": status_info.local_path,
                    "missing_files": status_info.missing_files or [],
                    "bytes_needed": status_info.bytes_needed
                }
                
                if detailed and status_info.verification_result:
                    results[mid]["verification"] = {
                        "success": status_info.verification_result.success,
                        "verified_files": status_info.verification_result.verified_files,
                        "failed_files": status_info.verification_result.failed_files,
                        "missing_files": status_info.verification_result.missing_files
                    }
                    
            except Exception as e:
                results[mid] = {
                    "status": "ERROR",
                    "error": str(e)
                }
        
        if json_output:
            print(json.dumps(results, indent=2))
            return
        
        # Human-readable output
        console.print("[bold blue]WAN2.2 Model Status[/bold blue]")
        
        table = Table(title="Model Orchestrator Status")
        table.add_column("Model ID", style="cyan", no_wrap=True)
        table.add_column("Status", style="white")
        table.add_column("Local Path", style="dim")
        
        if detailed:
            table.add_column("Missing Files", style="yellow")
            table.add_column("Bytes Needed", style="magenta")
        
        for mid, info in results.items():
            if "error" in info:
                status_text = f"[red]ERROR: {info['error']}[/red]"
                row = [mid, status_text, "N/A"]
            else:
                status = info["status"]
                status_color = _get_status_color(ModelStatus(status))
                status_text = f"[{status_color}]{status}[/{status_color}]"
                
                local_path = info.get("local_path", "N/A")
                if local_path and len(local_path) > 50:
                    local_path = "..." + local_path[-47:]
                
                row = [mid, status_text, local_path or "N/A"]
                
                if detailed:
                    missing_count = len(info.get("missing_files", []))
                    bytes_needed = _format_bytes(info.get("bytes_needed", 0))
                    row.extend([str(missing_count), bytes_needed])
            
            table.add_row(*row)
        
        console.print(table)
        
        # Summary
        total_models = len(results)
        complete_models = sum(1 for info in results.values() if info.get("status") == "COMPLETE")
        error_models = sum(1 for info in results.values() if "error" in info)
        
        console.print(f"\n[dim]Summary: {complete_models}/{total_models} models complete")
        if error_models > 0:
            console.print(f"[red]{error_models} models have errors[/red]")
            
    except Exception as e:
        console.print(f"[red]Error checking model status: {e}[/red]")
        raise typer.Exit(1)

@app.command("ensure")
def ensure_command(
    model_id: Optional[str] = typer.Option(None, "--only", help="Ensure only specific model"),
    variant: Optional[str] = typer.Option(None, "--variant", "-v", help="Specific variant to ensure"),
    all_models: bool = typer.Option(False, "--all", help="Ensure all models in manifest"),
    force: bool = typer.Option(False, "--force", help="Force re-download even if complete"),
    json_output: bool = typer.Option(False, "--json", help="Output in JSON format"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output")
):
    """
    Ensure WAN2.2 models are downloaded and ready for use.
    
    Examples:
      wan models ensure --only t2v-A14B     # Download specific model
      wan models ensure --all               # Download all models
      wan models ensure --only t2v-A14B --force  # Force re-download
    """
    try:
        registry, ensurer, _ = _get_orchestrator_components()
        
        # Determine which models to ensure
        if model_id:
            model_ids = [model_id]
        elif all_models:
            model_ids = registry.list_models()
        else:
            console.print("[red]Error: Must specify --only {model_id} or --all[/red]")
            raise typer.Exit(1)
        
        results = {}
        
        for mid in model_ids:
            if not quiet:
                console.print(f"\n[blue]Ensuring model: {mid}[/blue]")
            
            try:
                # Progress callback for downloads
                current_progress = None
                current_task = None
                
                def progress_callback(downloaded: int, total: int):
                    nonlocal current_progress, current_task
                    if current_progress and current_task is not None:
                        current_progress.update(current_task, completed=downloaded, total=total)
                
                if not quiet:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        TaskProgressColumn(),
                        console=console,
                    ) as progress:
                        current_progress = progress
                        current_task = progress.add_task(f"Downloading {mid}...", total=100)
                        
                        local_path = ensurer.ensure(
                            model_id=mid,
                            variant=variant,
                            force_redownload=force,
                            progress_callback=progress_callback
                        )
                        
                        progress.update(current_task, description=f"[green]✓[/green] {mid} ready")
                else:
                    local_path = ensurer.ensure(
                        model_id=mid,
                        variant=variant,
                        force_redownload=force
                    )
                
                results[mid] = {
                    "success": True,
                    "local_path": local_path
                }
                
                if not quiet:
                    console.print(f"[green]✓ Model {mid} ready at: {local_path}[/green]")
                    
            except ModelOrchestratorError as e:
                results[mid] = {
                    "success": False,
                    "error": str(e),
                    "error_code": e.error_code.value
                }
                
                if not quiet:
                    console.print(f"[red]✗ Failed to ensure {mid}: {e}[/red]")
                    
            except Exception as e:
                results[mid] = {
                    "success": False,
                    "error": str(e)
                }
                
                if not quiet:
                    console.print(f"[red]✗ Unexpected error ensuring {mid}: {e}[/red]")
        
        if json_output:
            print(json.dumps(results, indent=2))
        
        # Set exit code based on results
        failed_models = [mid for mid, info in results.items() if not info["success"]]
        if failed_models:
            if not quiet:
                console.print(f"\n[red]Failed to ensure {len(failed_models)} models: {', '.join(failed_models)}[/red]")
            raise typer.Exit(1)
        else:
            if not quiet:
                console.print(f"\n[green]Successfully ensured {len(results)} models[/green]")
                
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error ensuring models: {e}[/red]")
        raise typer.Exit(1)

@app.command("verify")
def verify_command(
    model_id: str = typer.Argument(..., help="Model ID to verify"),
    variant: Optional[str] = typer.Option(None, "--variant", "-v", help="Specific variant to verify"),
    json_output: bool = typer.Option(False, "--json", help="Output in JSON format")
):
    """
    Verify integrity of a downloaded model using checksums.
    
    Examples:
      wan models verify t2v-A14B          # Verify model integrity
      wan models verify t2v-A14B --json   # JSON output
    """
    try:
        registry, ensurer, _ = _get_orchestrator_components()
        
        if not json_output:
            console.print(f"[blue]Verifying model: {model_id}[/blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            if not json_output:
                task = progress.add_task("Verifying integrity...", total=None)
            
            result = ensurer.verify_integrity(model_id, variant)
            
            if not json_output:
                progress.update(task, description="[green]✓[/green] Verification complete")
        
        if json_output:
            output = {
                "model_id": model_id,
                "variant": variant,
                "success": result.success,
                "verified_files": result.verified_files,
                "failed_files": result.failed_files,
                "missing_files": result.missing_files,
                "error_message": result.error_message
            }
            print(json.dumps(output, indent=2))
        else:
            if result.success:
                console.print(f"[green]✓ Model {model_id} integrity verified[/green]")
                console.print(f"  Verified files: {len(result.verified_files)}")
            else:
                console.print(f"[red]✗ Model {model_id} integrity check failed[/red]")
                if result.error_message:
                    console.print(f"  Error: {result.error_message}")
                if result.failed_files:
                    console.print(f"  Failed files: {len(result.failed_files)}")
                if result.missing_files:
                    console.print(f"  Missing files: {len(result.missing_files)}")
        
        # Set exit code based on verification result
        if not result.success:
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]Error verifying model: {e}[/red]")
        raise typer.Exit(1)

@app.command("list")
def list_command(
    json_output: bool = typer.Option(False, "--json", help="Output in JSON format"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed model information")
):
    """
    List all available models in the manifest.
    
    Examples:
      wan models list                # List all models
      wan models list --detailed     # Show detailed information
      wan models list --json         # JSON output
    """
    try:
        registry, _, _ = _get_orchestrator_components()
        model_ids = registry.list_models()
        
        if json_output:
            models_info = {}
            for model_id in model_ids:
                try:
                    spec = registry.spec(model_id)
                    models_info[model_id] = {
                        "description": spec.description,
                        "version": spec.version,
                        "variants": spec.variants,
                        "default_variant": spec.default_variant,
                        "file_count": len(spec.files),
                        "total_size": sum(f.size for f in spec.files)
                    }
                except Exception as e:
                    models_info[model_id] = {"error": str(e)}
            
            print(json.dumps(models_info, indent=2))
            return
        
        # Human-readable output
        console.print("[bold blue]Available WAN2.2 Models[/bold blue]")
        
        table = Table(title="Model Manifest")
        table.add_column("Model ID", style="cyan", no_wrap=True)
        table.add_column("Version", style="green")
        table.add_column("Variants", style="yellow")
        
        if detailed:
            table.add_column("Description", style="white")
            table.add_column("Files", style="magenta")
            table.add_column("Total Size", style="blue")
        
        for model_id in model_ids:
            try:
                spec = registry.spec(model_id)
                variants_str = ", ".join(spec.variants)
                
                row = [model_id, spec.version, variants_str]
                
                if detailed:
                    description = spec.description[:50] + "..." if len(spec.description) > 50 else spec.description
                    file_count = str(len(spec.files))
                    total_size = _format_bytes(sum(f.size for f in spec.files))
                    row.extend([description, file_count, total_size])
                
                table.add_row(*row)
                
            except Exception as e:
                error_row = [model_id, "ERROR", str(e)]
                if detailed:
                    error_row.extend(["N/A", "N/A", "N/A"])
                table.add_row(*error_row)
        
        console.print(table)
        console.print(f"\n[dim]Total models: {len(model_ids)}[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error listing models: {e}[/red]")
        raise typer.Exit(1)

@app.command("gc")
def gc_command(
    dry_run: bool = typer.Option(True, "--dry-run/--execute", help="Show what would be removed without actually removing"),
    max_size: Optional[str] = typer.Option(None, "--max-size", help="Maximum total size (e.g., '10GB', '500MB')"),
    max_age: Optional[str] = typer.Option(None, "--max-age", help="Maximum model age (e.g., '7d', '30d', '1w')"),
    json_output: bool = typer.Option(False, "--json", help="Output in JSON format"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output")
):
    """
    Garbage collection for model storage management.
    
    Removes old or unused models to free up disk space based on configurable policies.
    By default, runs in dry-run mode to show what would be removed.
    
    Examples:
      wan models gc                           # Dry run with default settings
      wan models gc --execute                 # Actually remove models
      wan models gc --max-size 10GB --execute # Limit total storage to 10GB
      wan models gc --max-age 7d --execute    # Remove models older than 7 days
    """
    try:
        registry, ensurer, garbage_collector = _get_orchestrator_components()
        
        # Parse size and age parameters
        def parse_size(size_str: str) -> int:
            """Parse size string like '10GB' to bytes."""
            if not size_str:
                return None
            
            size_str = size_str.upper()
            # Check units in order of length (longest first) to avoid partial matches
            units = [
                ('TB', 1024**4),
                ('GB', 1024**3),
                ('MB', 1024**2),
                ('KB', 1024),
                ('B', 1)
            ]
            
            for unit, multiplier in units:
                if size_str.endswith(unit):
                    try:
                        value = float(size_str[:-len(unit)])
                        return int(value * multiplier)
                    except ValueError:
                        raise ValueError(f"Invalid size format: {size_str}")
            
            # Try parsing as plain number (bytes)
            try:
                return int(size_str)
            except ValueError:
                raise ValueError(f"Invalid size format: {size_str}")
        
        def parse_age(age_str: str) -> float:
            """Parse age string like '7d' to seconds."""
            if not age_str:
                return None
            
            age_str = age_str.lower()
            multipliers = {
                's': 1,
                'm': 60,
                'h': 3600,
                'd': 86400,
                'w': 604800,  # 7 days
                'mo': 2592000,  # 30 days
                'y': 31536000   # 365 days
            }
            
            for unit, multiplier in multipliers.items():
                if age_str.endswith(unit):
                    try:
                        value = float(age_str[:-len(unit)])
                        return value * multiplier
                    except ValueError:
                        raise ValueError(f"Invalid age format: {age_str}")
            
            # Try parsing as plain number (seconds)
            try:
                return float(age_str)
            except ValueError:
                raise ValueError(f"Invalid age format: {age_str}")
        
        # Update garbage collector configuration
        if max_size:
            garbage_collector.config.max_total_size = parse_size(max_size)
        if max_age:
            garbage_collector.config.max_model_age = parse_age(max_age)
        
        if not quiet:
            if dry_run:
                console.print("[yellow]Running garbage collection in dry-run mode[/yellow]")
                console.print("[dim]Use --execute to actually remove models[/dim]")
            else:
                console.print("[red]Running garbage collection - models will be removed![/red]")
        
        # Show current disk usage
        if not quiet:
            disk_usage = garbage_collector.get_disk_usage()
            console.print(f"\n[blue]Current Disk Usage:[/blue]")
            console.print(f"  Total: {_format_bytes(disk_usage.total_bytes)}")
            console.print(f"  Used: {_format_bytes(disk_usage.used_bytes)} ({disk_usage.usage_percentage:.1f}%)")
            console.print(f"  Free: {_format_bytes(disk_usage.free_bytes)}")
            console.print(f"  Models: {_format_bytes(disk_usage.models_bytes)}")
        
        # Run garbage collection
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            if not quiet:
                task = progress.add_task("Analyzing models for garbage collection...", total=None)
            
            result = garbage_collector.collect(dry_run=dry_run, trigger=GCTrigger.MANUAL)
            
            if not quiet:
                progress.update(task, description="[green]✓[/green] Garbage collection complete")
        
        if json_output:
            output = {
                "dry_run": result.dry_run,
                "trigger": result.trigger.value,
                "models_removed": result.models_removed,
                "models_preserved": result.models_preserved,
                "bytes_reclaimed": result.bytes_reclaimed,
                "bytes_preserved": result.bytes_preserved,
                "errors": result.errors,
                "duration_seconds": result.duration_seconds
            }
            print(json.dumps(output, indent=2))
        else:
            # Human-readable output
            console.print(f"\n[bold blue]Garbage Collection Results[/bold blue]")
            
            if result.models_removed:
                console.print(f"\n[red]Models {'would be ' if dry_run else ''}removed:[/red]")
                for model in result.models_removed:
                    console.print(f"  • {model}")
                console.print(f"  Total: {len(result.models_removed)} models, {_format_bytes(result.bytes_reclaimed)} reclaimed")
            else:
                console.print(f"\n[green]No models selected for removal[/green]")
            
            if result.models_preserved:
                console.print(f"\n[green]Models preserved:[/green]")
                for model in result.models_preserved[:10]:  # Show first 10
                    console.print(f"  • {model}")
                if len(result.models_preserved) > 10:
                    console.print(f"  ... and {len(result.models_preserved) - 10} more")
                console.print(f"  Total: {len(result.models_preserved)} models, {_format_bytes(result.bytes_preserved)} preserved")
            
            if result.errors:
                console.print(f"\n[red]Errors encountered:[/red]")
                for error in result.errors:
                    console.print(f"  • {error}")
            
            console.print(f"\n[dim]Duration: {result.duration_seconds:.2f} seconds[/dim]")
            
            if dry_run and result.models_removed:
                console.print(f"\n[yellow]To actually remove these models, run with --execute[/yellow]")
        
        # Set exit code based on errors
        if result.errors:
            raise typer.Exit(1)
            
    except ValueError as e:
        console.print(f"[red]Invalid parameter: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error running garbage collection: {e}[/red]")
        raise typer.Exit(1)

@app.command("pin")
def pin_command(
    model_id: str = typer.Argument(..., help="Model ID to pin"),
    variant: Optional[str] = typer.Option(None, "--variant", "-v", help="Specific variant to pin"),
    unpin: bool = typer.Option(False, "--unpin", help="Unpin the model instead of pinning"),
    json_output: bool = typer.Option(False, "--json", help="Output in JSON format")
):
    """
    Pin or unpin models to protect them from garbage collection.
    
    Pinned models are never removed by garbage collection, regardless of age or size policies.
    
    Examples:
      wan models pin t2v-A14B              # Pin model
      wan models pin t2v-A14B --unpin      # Unpin model
      wan models pin t2v-A14B --variant fp16  # Pin specific variant
    """
    try:
        registry, ensurer, garbage_collector = _get_orchestrator_components()
        
        if unpin:
            garbage_collector.unpin_model(model_id, variant)
            action = "unpinned"
        else:
            garbage_collector.pin_model(model_id, variant)
            action = "pinned"
        
        if json_output:
            output = {
                "model_id": model_id,
                "variant": variant,
                "action": action,
                "is_pinned": garbage_collector.is_pinned(model_id, variant)
            }
            print(json.dumps(output, indent=2))
        else:
            variant_str = f" (variant: {variant})" if variant else ""
            console.print(f"[green]✓ Model {model_id}{variant_str} {action}[/green]")
            
    except Exception as e:
        console.print(f"[red]Error {'unpinning' if unpin else 'pinning'} model: {e}[/red]")
        raise typer.Exit(1)

@app.command("disk-usage")
def disk_usage_command(
    json_output: bool = typer.Option(False, "--json", help="Output in JSON format")
):
    """
    Show disk usage information for model storage.
    
    Examples:
      wan models disk-usage        # Show disk usage
      wan models disk-usage --json # JSON output
    """
    try:
        registry, ensurer, garbage_collector = _get_orchestrator_components()
        
        disk_usage = garbage_collector.get_disk_usage()
        reclaimable = garbage_collector.estimate_reclaimable_space()
        
        if json_output:
            output = {
                "total_bytes": disk_usage.total_bytes,
                "used_bytes": disk_usage.used_bytes,
                "free_bytes": disk_usage.free_bytes,
                "models_bytes": disk_usage.models_bytes,
                "usage_percentage": disk_usage.usage_percentage,
                "reclaimable_bytes": reclaimable
            }
            print(json.dumps(output, indent=2))
        else:
            console.print("[bold blue]Disk Usage Information[/bold blue]")
            
            table = Table(title="Storage Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")
            table.add_column("Percentage", style="yellow")
            
            table.add_row("Total Space", _format_bytes(disk_usage.total_bytes), "100.0%")
            table.add_row("Used Space", _format_bytes(disk_usage.used_bytes), f"{disk_usage.usage_percentage:.1f}%")
            table.add_row("Free Space", _format_bytes(disk_usage.free_bytes), f"{100 - disk_usage.usage_percentage:.1f}%")
            table.add_row("Models Space", _format_bytes(disk_usage.models_bytes), f"{(disk_usage.models_bytes / disk_usage.total_bytes) * 100:.1f}%")
            
            if reclaimable > 0:
                table.add_row("Reclaimable", _format_bytes(reclaimable), f"{(reclaimable / disk_usage.total_bytes) * 100:.1f}%")
            
            console.print(table)
            
            # Show warnings if needed
            if disk_usage.usage_percentage > 90:
                console.print(f"\n[red]⚠️  Warning: Disk usage is high ({disk_usage.usage_percentage:.1f}%)[/red]")
                console.print("[dim]Consider running garbage collection to free up space[/dim]")
            
            if reclaimable > 0:
                console.print(f"\n[yellow]💡 {_format_bytes(reclaimable)} could be reclaimed through garbage collection[/yellow]")
                console.print("[dim]Run 'wan models gc' to see what would be removed[/dim]")
            
    except Exception as e:
        console.print(f"[red]Error getting disk usage: {e}[/red]")
        raise typer.Exit(1)

@app.callback()
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """
    Model Orchestrator CLI for WAN2.2
    
    Unified model management system that handles:
    • Model discovery and manifest parsing
    • Atomic downloads with integrity verification
    • Cross-process locking and concurrency safety
    • Multiple storage backends (Local, S3, HuggingFace)
    • Deterministic path resolution
    
    Examples:
      wan models status                    # Check all model status
      wan models ensure --only t2v-A14B    # Download specific model
      wan models verify t2v-A14B          # Verify model integrity
      wan models list --detailed          # List available models
      wan models gc --dry-run              # Show what would be removed
      wan models gc --execute --max-size 10GB  # Limit storage to 10GB
      wan models pin t2v-A14B              # Pin model to prevent removal
      wan models disk-usage                # Show disk usage statistics
    """
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose

@app.command("deduplicate")
def deduplicate_command(
    model_ids: Optional[List[str]] = typer.Option(None, "--model", "-m", help="Specific models to deduplicate (default: all)"),
    cross_model: bool = typer.Option(False, "--cross-model", help="Perform cross-model deduplication"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be deduplicated without making changes"),
    json_output: bool = typer.Option(False, "--json", help="Output in JSON format")
):
    """
    Deduplicate shared components across WAN2.2 models to save disk space.
    
    Examples:
      wan models deduplicate                           # Deduplicate all models individually
      wan models deduplicate --cross-model             # Deduplicate across all models
      wan models deduplicate -m t2v-A14B -m i2v-A14B  # Deduplicate specific models
      wan models deduplicate --dry-run                 # Preview deduplication without changes
    """
    try:
        registry, ensurer, _ = _get_orchestrator_components()
        
        # Check if deduplication is enabled
        if not hasattr(ensurer, 'component_deduplicator') or ensurer.component_deduplicator is None:
            console.print("[red]Component deduplication is not enabled[/red]")
            raise typer.Exit(1)
        
        # Get models to process
        if model_ids:
            target_models = model_ids
        else:
            target_models = registry.list_models()
        
        if not target_models:
            console.print("[yellow]No models found to deduplicate[/yellow]")
            return
        
        results = {}
        total_bytes_saved = 0
        total_links_created = 0
        
        if cross_model:
            # Perform cross-model deduplication
            if not json_output:
                console.print("[blue]Starting cross-model deduplication...[/blue]")
            
            if not dry_run:
                result = ensurer.deduplicate_across_models(target_models)
                if result:
                    results["cross_model"] = result
                    total_bytes_saved = result["bytes_saved"]
                    total_links_created = result["links_created"]
            else:
                if not json_output:
                    console.print("[yellow]Dry run: Cross-model deduplication would be performed[/yellow]")
                results["cross_model"] = {"dry_run": True, "models": target_models}
        else:
            # Perform individual model deduplication
            if not json_output:
                console.print(f"[blue]Starting deduplication for {len(target_models)} models...[/blue]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Deduplicating models...", total=len(target_models))
                
                for model_id in target_models:
                    progress.update(task, description=f"Processing {model_id}")
                    
                    try:
                        if not dry_run:
                            # Get model path
                            resolver = ensurer.resolver
                            model_path = Path(resolver.local_dir(model_id))
                            
                            if model_path.exists():
                                dedup_result = ensurer.component_deduplicator.deduplicate_model(model_id, model_path)
                                results[model_id] = {
                                    "files_processed": dedup_result.total_files_processed,
                                    "duplicates_found": dedup_result.duplicates_found,
                                    "bytes_saved": dedup_result.bytes_saved,
                                    "links_created": dedup_result.links_created,
                                    "processing_time": dedup_result.processing_time,
                                    "errors": dedup_result.errors
                                }
                                total_bytes_saved += dedup_result.bytes_saved
                                total_links_created += dedup_result.links_created
                            else:
                                results[model_id] = {"error": "Model not found locally"}
                        else:
                            results[model_id] = {"dry_run": True, "status": "would_be_processed"}
                    
                    except Exception as e:
                        results[model_id] = {"error": str(e)}
                    
                    progress.advance(task)
        
        if json_output:
            output = {
                "results": results,
                "summary": {
                    "total_bytes_saved": total_bytes_saved,
                    "total_links_created": total_links_created,
                    "models_processed": len(target_models),
                    "cross_model": cross_model,
                    "dry_run": dry_run
                }
            }
            print(json.dumps(output, indent=2))
            return
        
        # Human-readable output
        console.print("\n[bold green]Deduplication Results[/bold green]")
        
        if dry_run:
            console.print("[yellow]DRY RUN - No changes were made[/yellow]")
        
        table = Table(title="Deduplication Summary")
        table.add_column("Model ID", style="cyan")
        table.add_column("Files Processed", justify="right")
        table.add_column("Duplicates Found", justify="right")
        table.add_column("Bytes Saved", justify="right")
        table.add_column("Links Created", justify="right")
        table.add_column("Status", style="green")
        
        for model_id, result in results.items():
            if "error" in result:
                table.add_row(model_id, "-", "-", "-", "-", f"[red]Error: {result['error']}[/red]")
            elif "dry_run" in result:
                table.add_row(model_id, "-", "-", "-", "-", "[yellow]Would process[/yellow]")
            else:
                table.add_row(
                    model_id,
                    str(result.get("files_processed", 0)),
                    str(result.get("duplicates_found", 0)),
                    _format_bytes(result.get("bytes_saved", 0)),
                    str(result.get("links_created", 0)),
                    "[green]Complete[/green]"
                )
        
        console.print(table)
        
        if not dry_run and (total_bytes_saved > 0 or total_links_created > 0):
            console.print(f"\n[bold green]Total space saved: {_format_bytes(total_bytes_saved)}[/bold green]")
            console.print(f"[bold green]Total links created: {total_links_created}[/bold green]")
        
    except Exception as e:
        console.print(f"[red]Deduplication failed: {e}[/red]")
        raise typer.Exit(1)

@app.command("component-stats")
def component_stats_command(
    json_output: bool = typer.Option(False, "--json", help="Output in JSON format")
):
    """
    Show statistics about shared components and deduplication.
    
    Examples:
      wan models component-stats        # Show component statistics
      wan models component-stats --json # JSON output for automation
    """
    try:
        registry, ensurer, _ = _get_orchestrator_components()
        
        # Check if deduplication is enabled
        if not hasattr(ensurer, 'component_deduplicator') or ensurer.component_deduplicator is None:
            console.print("[red]Component deduplication is not enabled[/red]")
            raise typer.Exit(1)
        
        stats = ensurer.get_component_stats()
        
        if not stats:
            console.print("[yellow]No component statistics available[/yellow]")
            return
        
        if json_output:
            print(json.dumps(stats, indent=2))
            return
        
        # Human-readable output
        console.print("[bold blue]Component Deduplication Statistics[/bold blue]")
        
        # Summary panel
        summary_text = f"""
[bold]Total Components:[/bold] {stats['total_components']}
[bold]Total Size:[/bold] {_format_bytes(stats['total_size_bytes'])}
[bold]Total References:[/bold] {stats['total_references']}
[bold]Components Root:[/bold] {stats['components_root']}
[bold]Hardlink Support:[/bold] {'✓' if stats['supports_hardlinks'] else '✗'}
[bold]Symlink Support:[/bold] {'✓' if stats['supports_symlinks'] else '✗'}
        """
        
        console.print(Panel(summary_text.strip(), title="Summary", border_style="blue"))
        
        # Component types table
        if stats['component_types']:
            console.print("\n[bold]Component Types[/bold]")
            
            table = Table()
            table.add_column("Component Type", style="cyan")
            table.add_column("Count", justify="right", style="green")
            
            for component_type, count in stats['component_types'].items():
                table.add_row(component_type, str(count))
            
            console.print(table)
        else:
            console.print("\n[yellow]No shared components found[/yellow]")
        
    except Exception as e:
        console.print(f"[red]Failed to get component statistics: {e}[/red]")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()