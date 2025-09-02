#!/usr/bin/env python3
"""
Documentation Generator CLI

Unified command-line interface for all WAN22 documentation tools.
Provides commands for generation, validation, serving, and management.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tools.doc_generator.documentation_generator import DocumentationGenerator
from tools.doc_generator.server import DocumentationServer, ServerConfig
from tools.doc_generator.validator import DocumentationValidator
from tools.doc_generator.search_indexer import SearchIndexer
from tools.doc_generator.navigation_generator import NavigationGenerator, NavigationConfig
from tools.doc_generator.metadata_manager import MetadataManager
from tools.doc_generator.migration_tool import DocumentationMigrator


def cmd_generate(args):
    """Generate consolidated documentation"""
    print("🔄 Generating documentation...")
    
    # Initialize generator
    generator = DocumentationGenerator(args.source_dirs, args.output_dir)
    
    # Consolidate existing documentation
    if not args.api_only:
        print("📚 Consolidating existing documentation...")
        report = generator.consolidate_existing_docs()
        
        print(f"✅ Migration complete:")
        print(f"   📄 Files processed: {report.total_files}")
        print(f"   ✅ Successfully migrated: {report.migrated_files}")
        print(f"   ❌ Failed: {len(report.failed_files)}")
        print(f"   ⚠️  Duplicates: {len(report.duplicate_files)}")
        print(f"   🔗 Broken links: {len(report.broken_links)}")
    
    # Generate API documentation
    if args.generate_api or args.api_only:
        print("🔌 Generating API documentation...")
        code_dirs = [Path(d) for d in args.code_dirs if Path(d).exists()]
        api_docs = generator.generate_api_docs(code_dirs)
        print(f"✅ Generated API docs for {len(api_docs)} modules")
    
    # Save consolidated documentation
    print("💾 Saving consolidated documentation...")
    generator.save_consolidated_docs()
    
    print(f"🎉 Documentation generated successfully in: {args.output_dir}")


def cmd_migrate(args):
    """Migrate scattered documentation"""
    print("🔄 Migrating scattered documentation...")
    
    migrator = DocumentationMigrator(args.source, args.target)
    results = migrator.migrate_all(dry_run=args.dry_run)
    
    action = "Would migrate" if args.dry_run else "Migrated"
    print(f"✅ {action}:")
    print(f"   📄 Total files: {results['total_files']}")
    print(f"   ✅ Successful: {results['successful_migrations']}")
    print(f"   ❌ Failed: {results['failed_migrations']}")
    print(f"   📝 Renamed: {results['renamed_files']}")
    
    if args.report:
        report = migrator.generate_migration_report(results)
        with open(args.report, 'w') as f:
            f.write(report)
        print(f"📊 Report saved to: {args.report}")


def cmd_serve(args):
    """Start documentation server"""
    config = ServerConfig(
        docs_dir=args.docs_dir,
        site_dir=args.site_dir,
        port=args.port,
        dev_addr=f"{args.host}:{args.port}"
    )
    
    server = DocumentationServer(config, Path.cwd())
    
    if args.setup:
        print("🔧 Setting up documentation server...")
        success = server.setup_complete_server()
        if not success:
            print("❌ Server setup failed")
            return 1
    
    if args.build:
        print("🏗️  Building documentation site...")
        server.save_mkdocs_config()
        success = server.build_site()
        if not success:
            print("❌ Build failed")
            return 1
        print("✅ Site built successfully")
    
    if not args.build_only:
        print(f"🚀 Starting development server at http://{args.host}:{args.port}")
        server.save_mkdocs_config()
        server.serve_dev()


def cmd_validate(args):
    """Validate documentation"""
    print("🔍 Validating documentation...")
    
    # Load configuration
    config = None
    if args.config and Path(args.config).exists():
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    validator = DocumentationValidator(Path(args.docs_dir), config)
    
    # Run validation
    if args.links_only:
        print("🔗 Checking links only...")
        issues = validator.check_links_only(args.external_only)
        
        # Create minimal report
        from tools.doc_generator.validator import ValidationReport
        from datetime import datetime
        report = ValidationReport(
            total_files=0,
            issues=issues,
            summary={'total_issues': len(issues)},
            execution_time=0,
            timestamp=datetime.now().isoformat()
        )
    else:
        report = validator.validate_all()
    
    # Print results
    print(f"📊 Validation Results:")
    print(f"   📄 Files checked: {report.total_files}")
    print(f"   🔍 Total issues: {report.summary['total_issues']}")
    
    if 'errors' in report.summary:
        print(f"   ❌ Errors: {report.summary['errors']}")
    if 'warnings' in report.summary:
        print(f"   ⚠️  Warnings: {report.summary['warnings']}")
    if 'info' in report.summary:
        print(f"   ℹ️  Info: {report.summary['info']}")
    
    # Save report
    if args.output:
        output_path = Path(args.output)
        if args.format == 'html':
            validator.generate_report_html(report, output_path)
            print(f"📄 HTML report saved to: {output_path}")
        else:
            validator.save_report_json(report, output_path)
            print(f"📄 JSON report saved to: {output_path}")
    
    # Return appropriate exit code
    return 1 if report.summary.get('errors', 0) > 0 else 0


def cmd_search(args):
    """Search and index documentation"""
    indexer = SearchIndexer(Path(args.docs_dir))
    
    if args.command == 'index':
        print("🔍 Indexing documentation...")
        result = indexer.index_documentation()
        
        print(f"✅ Indexing complete:")
        print(f"   📄 New documents: {result['indexed']}")
        print(f"   🔄 Updated documents: {result['updated']}")
        print(f"   ❌ Errors: {result['errors']}")
        print(f"   📊 Total documents: {result['total_documents']}")
    
    elif args.command == 'search':
        if not args.query:
            print("❌ Search query required")
            return 1
        
        print(f"🔍 Searching for: '{args.query}'")
        results = indexer.search(args.query, args.category, args.tags, args.limit)
        
        if results:
            print(f"📄 Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result.document.title}")
                print(f"   📁 {result.document.path}")
                print(f"   ⭐ Score: {result.score:.2f}")
                print(f"   📝 {result.snippet}")
        else:
            print("❌ No results found")
    
    elif args.command == 'stats':
        stats = indexer.get_index_stats()
        print("📊 Search Index Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")


def cmd_navigation(args):
    """Generate navigation structure"""
    print("🧭 Generating navigation...")
    
    config = NavigationConfig()
    generator = NavigationGenerator(Path(args.docs_dir), config)
    
    root = generator.generate_navigation()
    
    if args.format in ['mkdocs', 'all']:
        mkdocs_nav = generator.generate_mkdocs_nav(root)
        print("📄 MkDocs Navigation:")
        import yaml
        print(yaml.dump({'nav': mkdocs_nav}, default_flow_style=False))
    
    if args.format in ['json', 'all']:
        sidebar_json = generator.generate_sidebar_json(root)
        print("📄 Sidebar JSON:")
        import json
        print(json.dumps(sidebar_json, indent=2))
    
    # Save files
    generator.save_navigation_files(root, Path(args.output_dir))
    print(f"💾 Navigation files saved to: {args.output_dir}")


def cmd_metadata(args):
    """Manage documentation metadata"""
    manager = MetadataManager(Path(args.docs_dir))
    
    if args.command == 'scan':
        print("🔍 Scanning documentation metadata...")
        index = manager.scan_documentation()
        manager.save_index(index)
        
        print(f"✅ Metadata scan complete:")
        print(f"   📄 Pages scanned: {len(index.pages)}")
        print(f"   🔗 Cross-references: {len(index.cross_references)}")
        print(f"   📁 Categories: {len(index.categories)}")
        print(f"   🏷️  Tags: {len(index.tags)}")
    
    elif args.command == 'check-refs':
        print("🔗 Checking cross-references...")
        index = manager.load_index()
        if index:
            broken_refs = manager.find_broken_references()
            if broken_refs:
                print(f"❌ Found {len(broken_refs)} broken references:")
                for ref in broken_refs:
                    print(f"   {ref.source_page} → {ref.target_page}")
            else:
                print("✅ No broken references found")
        else:
            print("❌ No metadata index found. Run 'scan' first.")
    
    elif args.command == 'generate-nav':
        print("🧭 Generating navigation menu...")
        index = manager.load_index()
        if index:
            nav_menu = manager.generate_navigation_menu()
            nav_file = Path(args.docs_dir) / '.metadata' / 'navigation.json'
            import json
            with open(nav_file, 'w') as f:
                json.dump(nav_menu, f, indent=2)
            print(f"💾 Navigation menu saved to: {nav_file}")
        else:
            print("❌ No metadata index found. Run 'scan' first.")


def cmd_all(args):
    """Run complete documentation pipeline"""
    print("🚀 Running complete documentation pipeline...")
    
    # Step 1: Migrate scattered docs
    print("\n📋 Step 1: Migrating scattered documentation...")
    migrate_args = argparse.Namespace(
        source=args.docs_dir,
        target=args.docs_dir,
        dry_run=False,
        report=None
    )
    cmd_migrate(migrate_args)
    
    # Step 2: Generate consolidated docs
    print("\n📚 Step 2: Generating consolidated documentation...")
    generate_args = argparse.Namespace(
        source_dirs=[args.docs_dir],
        output_dir=args.docs_dir,
        code_dirs=['backend', 'frontend/src', 'tools', 'scripts'],
        generate_api=True,
        api_only=False
    )
    cmd_generate(generate_args)
    
    # Step 3: Scan metadata
    print("\n🔍 Step 3: Scanning metadata...")
    metadata_args = argparse.Namespace(
        docs_dir=args.docs_dir,
        command='scan'
    )
    cmd_metadata(metadata_args)
    
    # Step 4: Generate navigation
    print("\n🧭 Step 4: Generating navigation...")
    nav_args = argparse.Namespace(
        docs_dir=args.docs_dir,
        output_dir=f"{args.docs_dir}/.metadata",
        format='all'
    )
    cmd_navigation(nav_args)
    
    # Step 5: Index for search
    print("\n🔍 Step 5: Building search index...")
    search_args = argparse.Namespace(
        docs_dir=args.docs_dir,
        command='index'
    )
    cmd_search(search_args)
    
    # Step 6: Setup server
    print("\n🚀 Step 6: Setting up documentation server...")
    serve_args = argparse.Namespace(
        docs_dir=args.docs_dir,
        site_dir='site',
        port=8000,
        host='127.0.0.1',
        setup=True,
        build=True,
        build_only=True
    )
    cmd_serve(serve_args)
    
    # Step 7: Validate
    print("\n🔍 Step 7: Validating documentation...")
    validate_args = argparse.Namespace(
        docs_dir=args.docs_dir,
        config=None,
        links_only=False,
        external_only=False,
        output=f"{args.docs_dir}/.metadata/validation_report.json",
        format='json'
    )
    validation_result = cmd_validate(validate_args)
    
    print("\n🎉 Documentation pipeline complete!")
    print(f"   📄 Documentation: {args.docs_dir}")
    print(f"   🌐 Built site: site/")
    print(f"   📊 Reports: {args.docs_dir}/.metadata/")
    
    if validation_result == 0:
        print("   ✅ All validation checks passed")
    else:
        print("   ⚠️  Some validation issues found - check the report")
    
    return validation_result


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='WAN22 Documentation Generator CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete documentation pipeline
  python cli.py all

  # Generate consolidated documentation
  python cli.py generate --generate-api

  # Start development server
  python cli.py serve --setup

  # Validate documentation
  python cli.py validate --output report.html --format html

  # Search documentation
  python cli.py search index
  python cli.py search search --query "installation"

  # Migrate scattered docs
  python cli.py migrate --source . --target docs
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate consolidated documentation')
    gen_parser.add_argument('--source-dirs', nargs='+', default=['.'], 
                           help='Source directories to scan')
    gen_parser.add_argument('--output-dir', default='docs', 
                           help='Output directory')
    gen_parser.add_argument('--code-dirs', nargs='+', 
                           default=['backend', 'frontend/src', 'tools', 'scripts'],
                           help='Code directories for API docs')
    gen_parser.add_argument('--generate-api', action='store_true', 
                           help='Generate API documentation')
    gen_parser.add_argument('--api-only', action='store_true', 
                           help='Generate only API docs')
    gen_parser.set_defaults(func=cmd_generate)
    
    # Migrate command
    migrate_parser = subparsers.add_parser('migrate', help='Migrate scattered documentation')
    migrate_parser.add_argument('--source', default='.', help='Source directory')
    migrate_parser.add_argument('--target', default='docs', help='Target directory')
    migrate_parser.add_argument('--dry-run', action='store_true', help='Show what would be done')
    migrate_parser.add_argument('--report', help='Save migration report')
    migrate_parser.set_defaults(func=cmd_migrate)
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start documentation server')
    serve_parser.add_argument('--docs-dir', default='docs', help='Documentation directory')
    serve_parser.add_argument('--site-dir', default='site', help='Built site directory')
    serve_parser.add_argument('--port', type=int, default=8000, help='Server port')
    serve_parser.add_argument('--host', default='127.0.0.1', help='Server host')
    serve_parser.add_argument('--setup', action='store_true', help='Setup server dependencies')
    serve_parser.add_argument('--build', action='store_true', help='Build site before serving')
    serve_parser.add_argument('--build-only', action='store_true', help='Build only, don\'t serve')
    serve_parser.set_defaults(func=cmd_serve)
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate documentation')
    validate_parser.add_argument('--docs-dir', default='docs', help='Documentation directory')
    validate_parser.add_argument('--config', help='Validation config file')
    validate_parser.add_argument('--output', help='Output report file')
    validate_parser.add_argument('--format', choices=['json', 'html'], default='json',
                                help='Report format')
    validate_parser.add_argument('--links-only', action='store_true', help='Check links only')
    validate_parser.add_argument('--external-only', action='store_true', help='Check external links only')
    validate_parser.set_defaults(func=cmd_validate)
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search and index documentation')
    search_subparsers = search_parser.add_subparsers(dest='command', help='Search commands')
    
    # Search index
    index_parser = search_subparsers.add_parser('index', help='Build search index')
    index_parser.add_argument('--docs-dir', default='docs', help='Documentation directory')
    
    # Search query
    query_parser = search_subparsers.add_parser('search', help='Search documentation')
    query_parser.add_argument('--docs-dir', default='docs', help='Documentation directory')
    query_parser.add_argument('--query', required=True, help='Search query')
    query_parser.add_argument('--category', help='Filter by category')
    query_parser.add_argument('--tags', nargs='+', help='Filter by tags')
    query_parser.add_argument('--limit', type=int, default=10, help='Result limit')
    
    # Search stats
    stats_parser = search_subparsers.add_parser('stats', help='Show search index statistics')
    stats_parser.add_argument('--docs-dir', default='docs', help='Documentation directory')
    
    search_parser.set_defaults(func=cmd_search)
    
    # Navigation command
    nav_parser = subparsers.add_parser('navigation', help='Generate navigation structure')
    nav_parser.add_argument('--docs-dir', default='docs', help='Documentation directory')
    nav_parser.add_argument('--output-dir', default='docs/.metadata', help='Output directory')
    nav_parser.add_argument('--format', choices=['mkdocs', 'json', 'all'], default='all',
                           help='Output format')
    nav_parser.set_defaults(func=cmd_navigation)
    
    # Metadata command
    meta_parser = subparsers.add_parser('metadata', help='Manage documentation metadata')
    meta_subparsers = meta_parser.add_subparsers(dest='command', help='Metadata commands')
    
    # Metadata scan
    scan_parser = meta_subparsers.add_parser('scan', help='Scan documentation metadata')
    scan_parser.add_argument('--docs-dir', default='docs', help='Documentation directory')
    
    # Metadata check-refs
    refs_parser = meta_subparsers.add_parser('check-refs', help='Check cross-references')
    refs_parser.add_argument('--docs-dir', default='docs', help='Documentation directory')
    
    # Metadata generate-nav
    gennav_parser = meta_subparsers.add_parser('generate-nav', help='Generate navigation menu')
    gennav_parser.add_argument('--docs-dir', default='docs', help='Documentation directory')
    
    meta_parser.set_defaults(func=cmd_metadata)
    
    # All command (complete pipeline)
    all_parser = subparsers.add_parser('all', help='Run complete documentation pipeline')
    all_parser.add_argument('--docs-dir', default='docs', help='Documentation directory')
    all_parser.set_defaults(func=cmd_all)
    
    # Parse arguments and run command
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        return args.func(args) or 0
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())