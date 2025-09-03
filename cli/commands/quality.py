"""Code quality and analysis commands"""

import typer
from pathlib import Path
from typing import Optional, List
import sys

app = typer.Typer()

@app.command()
def check(
    files: Optional[List[Path]] = typer.Option(None, "--files", help="Specific files to check"),
    fix: bool = typer.Option(False, "--fix", help="Auto-fix issues where possible"),
    strict: bool = typer.Option(False, "--strict", help="Use strict quality standards")
):
    """✨ Run comprehensive code quality checks"""
    
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from tools.code_quality.quality_checker import QualityChecker
    
    checker = QualityChecker()
    
    config = {
        'files': files,
        'auto_fix': fix,
        'strict_mode': strict
    }
    
    typer.echo("✨ Running code quality checks...")
    results = checker.run_quality_check(config)
    
    typer.echo(f"📊 Quality Score: {results.overall_score:.1%}")
    
    if results.has_issues:
        typer.echo(f"⚠️ Found {len(results.issues)} quality issues")
        
        if fix:
            typer.echo("🔧 Auto-fixing issues...")
            checker.fix_issues(results)
        else:
            checker.show_issues(results)
            typer.echo("Use --fix to attempt automatic fixes")
    else:
        typer.echo("✅ Code quality is excellent!")

@app.command()
def format(
    files: Optional[List[Path]] = typer.Option(None, "--files", help="Specific files to format"),
    style: str = typer.Option("black", "--style", help="Code style (black, autopep8, yapf)"),
    check_only: bool = typer.Option(False, "--check", help="Check formatting without making changes")
):
    """🎨 Format code according to style guidelines"""
    
    from tools.code_quality.formatters.code_formatter import CodeFormatter
    
    formatter = CodeFormatter()
    
    if check_only:
        typer.echo("🔍 Checking code formatting...")
        issues = formatter.check_formatting(files, style=style)
        
        if issues:
            typer.echo(f"❌ Found {len(issues)} formatting issues")
            formatter.show_formatting_issues(issues)
            raise typer.Exit(1)
        else:
            typer.echo("✅ Code formatting is correct")
    else:
        typer.echo("🎨 Formatting code...")
        formatter.format_code(files, style=style)
        typer.echo("✅ Code formatting complete")

@app.command()
def complexity(
    threshold: int = typer.Option(10, "--threshold", help="Complexity threshold"),
    show_details: bool = typer.Option(False, "--details", help="Show detailed complexity analysis")
):
    """📊 Analyze code complexity"""
    
    from tools.code_quality.analyzers.complexity_analyzer import ComplexityAnalyzer
    
    analyzer = ComplexityAnalyzer()
    results = analyzer.analyze_complexity(threshold=threshold)
    
    typer.echo("📊 Code Complexity Analysis:")
    typer.echo(f"Average Complexity: {results.average_complexity:.1f}")
    typer.echo(f"High Complexity Functions: {len(results.high_complexity_functions)}")
    
    if results.high_complexity_functions:
        typer.echo("⚠️ Functions exceeding complexity threshold:")
        for func in results.high_complexity_functions:
            typer.echo(f"  {func.name}: {func.complexity}")
    
    if show_details:
        analyzer.show_detailed_analysis(results)

@app.command()
def lint(
    files: Optional[List[Path]] = typer.Option(None, "--files", help="Specific files to lint"),
    linter: str = typer.Option("ruff", "--linter", help="Linter to use (ruff, flake8, pylint)"),
    fix: bool = typer.Option(False, "--fix", help="Auto-fix linting issues")
):
    """🔍 Run linting checks"""
    
    from tools.code_quality.quality_checker import QualityChecker
    
    checker = QualityChecker()
    results = checker.run_linting(files, linter=linter, auto_fix=fix)
    
    if results.has_issues:
        typer.echo(f"⚠️ Found {len(results.issues)} linting issues")
        
        if fix:
            typer.echo("🔧 Auto-fixing issues...")
            checker.fix_linting_issues(results)
        else:
            checker.show_linting_issues(results)
    else:
        typer.echo("✅ No linting issues found")

@app.command()
def types(
    files: Optional[List[Path]] = typer.Option(None, "--files", help="Specific files to check"),
    strict: bool = typer.Option(False, "--strict", help="Strict type checking"),
    add_hints: bool = typer.Option(False, "--add-hints", help="Add missing type hints")
):
    """🏷️ Check and improve type hints"""
    
    from tools.code_quality.validators.type_hint_validator import TypeHintValidator
    
    validator = TypeHintValidator()
    results = validator.validate_type_hints(files, strict=strict)
    
    if results.missing_hints:
        typer.echo(f"⚠️ Found {len(results.missing_hints)} missing type hints")
        
        if add_hints:
            typer.echo("🔧 Adding type hints...")
            validator.add_type_hints(results.missing_hints)
        else:
            validator.show_missing_hints(results.missing_hints)
            typer.echo("Use --add-hints to automatically add type hints")
    else:
        typer.echo("✅ All functions have proper type hints")

@app.command()
def docs_check(
    files: Optional[List[Path]] = typer.Option(None, "--files", help="Specific files to check"),
    add_missing: bool = typer.Option(False, "--add-missing", help="Add missing docstrings")
):
    """📝 Check and improve documentation"""
    
    from tools.code_quality.validators.documentation_validator import DocumentationValidator
    
    validator = DocumentationValidator()
    results = validator.validate_documentation(files)
    
    if results.missing_docs:
        typer.echo(f"📝 Found {len(results.missing_docs)} missing docstrings")
        
        if add_missing:
            typer.echo("🔧 Adding docstrings...")
            validator.add_docstrings(results.missing_docs)
        else:
            validator.show_missing_docs(results.missing_docs)
            typer.echo("Use --add-missing to automatically add docstrings")
    else:
        typer.echo("✅ All functions have proper documentation")

@app.command()
def review(
    target: str = typer.Option("HEAD~1", "--target", help="Git target for review (commit, branch)"),
    auto_fix: bool = typer.Option(False, "--auto-fix", help="Auto-fix issues where possible")
):
    """👀 Review code changes for quality issues"""
    
    from tools.code_review.code_reviewer import CodeReviewer
    
    reviewer = CodeReviewer()
    results = reviewer.review_changes(target, auto_fix=auto_fix)
    
    typer.echo("👀 Code Review Results:")
    typer.echo(f"Files Changed: {len(results.changed_files)}")
    typer.echo(f"Issues Found: {len(results.issues)}")
    typer.echo(f"Quality Score: {results.quality_score:.1%}")
    
    if results.issues:
        reviewer.show_review_results(results)