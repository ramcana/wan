"""Cleanup and maintenance commands"""

import typer
from pathlib import Path
from typing import Optional
import sys

app = typer.Typer()

@app.command()
def duplicates(
    remove: bool = typer.Option(False, "--remove", help="Actually remove duplicates (dry-run by default)"),
    threshold: float = typer.Option(0.95, "--threshold", help="Similarity threshold for duplicates")
):
    """🔍 Find and remove duplicate files"""
    
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from tools.codebase_cleanup.duplicate_detector import DuplicateDetector
    
    detector = DuplicateDetector()
    duplicates = detector.find_duplicates(threshold=threshold)
    
    if duplicates:
        typer.echo(f"🔍 Found {len(duplicates)} duplicate groups")
        
        if remove:
            typer.echo("🗑️ Removing duplicates...")
            detector.remove_duplicates(duplicates)
        else:
            typer.echo("📋 Dry run - use --remove to actually delete files")
            detector.show_duplicates(duplicates)
    else:
        typer.echo("✅ No duplicates found")

@app.command()
def dead_code(
    remove: bool = typer.Option(False, "--remove", help="Remove dead code (dry-run by default)"),
    aggressive: bool = typer.Option(False, "--aggressive", help="More aggressive dead code detection")
):
    """💀 Find and remove dead code"""
    
    from tools.codebase_cleanup.dead_code_analyzer import DeadCodeAnalyzer
    
    analyzer = DeadCodeAnalyzer()
    dead_code = analyzer.find_dead_code(aggressive=aggressive)
    
    if dead_code:
        typer.echo(f"💀 Found {len(dead_code)} dead code items")
        
        if remove:
            typer.echo("🗑️ Removing dead code...")
            analyzer.remove_dead_code(dead_code)
        else:
            typer.echo("📋 Dry run - use --remove to actually delete code")
            analyzer.show_dead_code(dead_code)
    else:
        typer.echo("✅ No dead code found")

@app.command()
def imports(
    fix: bool = typer.Option(False, "--fix", help="Fix import issues (dry-run by default)"),
    organize: bool = typer.Option(True, "--organize", help="Organize import order")
):
    """📦 Fix and organize imports"""
    
    from tests.utils.import_fixer import ImportFixer
    
    fixer = ImportFixer()
    issues = fixer.find_import_issues()
    
    if issues:
        typer.echo(f"📦 Found {len(issues)} import issues")
        
        if fix:
            typer.echo("🔧 Fixing imports...")
            fixer.fix_imports(issues, organize=organize)
        else:
            typer.echo("📋 Dry run - use --fix to actually fix imports")
            fixer.show_issues(issues)
    else:
        typer.echo("✅ No import issues found")

@app.command()
def naming(
    fix: bool = typer.Option(False, "--fix", help="Fix naming issues (dry-run by default)"),
    style: str = typer.Option("pep8", "--style", help="Naming style to enforce")
):
    """🏷️ Standardize naming conventions"""
    
    from tools.codebase_cleanup.naming_standardizer import NamingStandardizer
    
    standardizer = NamingStandardizer()
    issues = standardizer.find_naming_issues(style=style)
    
    if issues:
        typer.echo(f"🏷️ Found {len(issues)} naming issues")
        
        if fix:
            typer.echo("🔧 Fixing naming...")
            standardizer.fix_naming(issues)
        else:
            typer.echo("📋 Dry run - use --fix to actually fix naming")
            standardizer.show_issues(issues)
    else:
        typer.echo("✅ No naming issues found")

@app.command()
def all(
    dry_run: bool = typer.Option(True, "--dry-run/--execute", help="Dry run by default")
):
    """🧹 Run all cleanup operations"""
    
    typer.echo("🧹 Running comprehensive cleanup...")
    
    # Run all cleanup operations
    ctx = typer.Context(duplicates)
    ctx.invoke(duplicates, remove=not dry_run)
    
    ctx = typer.Context(dead_code)  
    ctx.invoke(dead_code, remove=not dry_run)
    
    ctx = typer.Context(imports)
    ctx.invoke(imports, fix=not dry_run)
    
    ctx = typer.Context(naming)
    ctx.invoke(naming, fix=not dry_run)
    
    if dry_run:
        typer.echo("📋 Dry run complete - use --execute to apply changes")