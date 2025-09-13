"""Documentation generation and validation commands"""

import typer
from pathlib import Path
from typing import Optional
import sys

app = typer.Typer()

@app.command()
def generate(
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    format: str = typer.Option("markdown", "--format", help="Output format (markdown, html, pdf)"),
    include_api: bool = typer.Option(True, "--api/--no-api", help="Include API documentation"),
    serve: bool = typer.Option(False, "--serve", help="Start documentation server after generation")
):
    """📚 Generate project documentation"""
    
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from tools.doc_generator.documentation_generator import DocumentationGenerator
    
    generator = DocumentationGenerator()
    
    config = {
        'output_dir': output_dir or Path('docs/generated'),
        'format': format,
        'include_api': include_api
    }
    
    typer.echo("📚 Generating documentation...")
    docs_path = generator.generate_docs(config)
    
    typer.echo(f"✅ Documentation generated: {docs_path}")
    
    if serve:
        typer.echo("🌐 Starting documentation server...")
        generator.serve_docs(docs_path)

@app.command()
def validate(
    check_links: bool = typer.Option(True, "--links/--no-links", help="Validate internal links"),
    check_images: bool = typer.Option(True, "--images/--no-images", help="Validate image references"),
    fix: bool = typer.Option(False, "--fix", help="Attempt to fix validation issues")
):
    """✅ Validate documentation integrity"""
    
    from tools.doc_generator.validator import DocumentationValidator
    
    validator = DocumentationValidator()
    results = validator.validate_docs(
        check_links=check_links,
        check_images=check_images
    )
    
    if results.has_issues:
        typer.echo(f"⚠️ Found {len(results.issues)} documentation issues")
        
        if fix:
            typer.echo("🔧 Attempting to fix issues...")
            validator.fix_issues(results)
        else:
            validator.show_issues(results)
            typer.echo("Use --fix to attempt automatic fixes")
    else:
        typer.echo("✅ All documentation is valid")

@app.command()
def serve(
    port: int = typer.Option(8000, "--port", "-p", help="Server port"),
    host: str = typer.Option("localhost", "--host", help="Server host"),
    watch: bool = typer.Option(True, "--watch/--no-watch", help="Watch for changes and reload")
):
    """🌐 Serve documentation locally"""
    
    from tools.doc_generator.server import DocumentationServer
    
    server = DocumentationServer()
    
    typer.echo(f"🌐 Starting documentation server at http://{host}:{port}")
    server.serve(host=host, port=port, watch=watch)

@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum results to show")
):
    """🔍 Search documentation content"""
    
    from tools.doc_generator.search_indexer import DocumentationSearcher
    
    searcher = DocumentationSearcher()
    results = searcher.search(query, limit=limit)
    
    if results:
        typer.echo(f"🔍 Found {len(results)} results for '{query}':")
        for result in results:
            typer.echo(f"📄 {result.title} ({result.file})")
            typer.echo(f"   {result.excerpt}")
    else:
        typer.echo(f"❌ No results found for '{query}'")

@app.command()
def structure():
    """📊 Show documentation structure"""
    
    from tools.doc_generator.navigation_generator import NavigationGenerator
    
    generator = NavigationGenerator()
    structure = generator.generate_structure()
    
    typer.echo("📊 Documentation Structure:")
    generator.print_structure(structure)

@app.command()
def migrate(
    from_format: str = typer.Argument(..., help="Source format"),
    to_format: str = typer.Argument(..., help="Target format"),
    source_dir: Path = typer.Option(Path("docs"), "--source", help="Source directory"),
    output_dir: Optional[Path] = typer.Option(None, "--output", help="Output directory")
):
    """🚀 Migrate documentation between formats"""
    
    from tools.doc_generator.migration_tool import DocumentationMigrator
    
    migrator = DocumentationMigrator()
    
    typer.echo(f"🚀 Migrating from {from_format} to {to_format}...")
    output_path = migrator.migrate(
        from_format=from_format,
        to_format=to_format,
        source_dir=source_dir,
        output_dir=output_dir
    )
    
    typer.echo(f"✅ Migration completed: {output_path}")
