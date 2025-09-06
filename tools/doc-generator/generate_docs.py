#!/usr/bin/env python3
"""
Documentation Generation CLI

Command-line interface for generating consolidated documentation
and API documentation from code annotations.
"""

import sys
import argparse
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tools.doc_generator.documentation_generator import DocumentationGenerator


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description='Generate consolidated documentation for WAN22 project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic documentation consolidation
  python generate_docs.py

  # Include API documentation generation
  python generate_docs.py --generate-api

  # Custom source and output directories
  python generate_docs.py --source-dirs . docs backend --output-dir consolidated-docs

  # Generate only API docs
  python generate_docs.py --api-only --code-dirs backend frontend/src
        """
    )
    
    parser.add_argument(
        '--source-dirs', 
        nargs='+', 
        default=['.'], 
        help='Source directories to scan for documentation (default: current directory)'
    )
    
    parser.add_argument(
        '--output-dir', 
        default='docs', 
        help='Output directory for consolidated documentation (default: docs)'
    )
    
    parser.add_argument(
        '--code-dirs', 
        nargs='+', 
        default=['backend', 'frontend/src', 'tools', 'scripts'], 
        help='Code directories to scan for API documentation'
    )
    
    parser.add_argument(
        '--generate-api', 
        action='store_true', 
        help='Generate API documentation from code annotations'
    )
    
    parser.add_argument(
        '--api-only', 
        action='store_true', 
        help='Generate only API documentation, skip consolidation'
    )
    
    parser.add_argument(
        '--verbose', '-v', 
        action='store_true', 
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--dry-run', 
        action='store_true', 
        help='Show what would be done without actually doing it'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    output_dir = Path(args.output_dir)
    source_dirs = [Path(d) for d in args.source_dirs]
    code_dirs = [Path(d) for d in args.code_dirs]
    
    # Check if source directories exist
    for source_dir in source_dirs:
        if not source_dir.exists():
            print(f"Warning: Source directory {source_dir} does not exist")
    
    # Check if code directories exist
    existing_code_dirs = [d for d in code_dirs if d.exists()]
    if args.generate_api or args.api_only:
        if not existing_code_dirs:
            print("Error: No valid code directories found for API generation")
            return 1
    
    if args.verbose:
        print(f"Source directories: {[str(d) for d in source_dirs]}")
        print(f"Output directory: {output_dir}")
        print(f"Code directories: {[str(d) for d in existing_code_dirs]}")
        print(f"Generate API: {args.generate_api or args.api_only}")
        print(f"Dry run: {args.dry_run}")
    
    if args.dry_run:
        print("\nDry run mode - no files will be modified")
        return 0
    
    try:
        # Initialize generator
        generator = DocumentationGenerator(source_dirs, output_dir)
        
        if not args.api_only:
            # Consolidate existing documentation
            print("Consolidating existing documentation...")
            report = generator.consolidate_existing_docs()
            
            print(f"\nMigration Report:")
            print(f"  Total files found: {report.total_files}")
            print(f"  Successfully migrated: {report.migrated_files}")
            print(f"  Failed migrations: {len(report.failed_files)}")
            print(f"  Duplicate files skipped: {len(report.duplicate_files)}")
            print(f"  Broken links found: {len(report.broken_links)}")
            
            if args.verbose and report.failed_files:
                print(f"\nFailed files:")
                for failed in report.failed_files:
                    print(f"  - {failed}")
            
            if args.verbose and report.duplicate_files:
                print(f"\nDuplicate files:")
                for duplicate in report.duplicate_files:
                    print(f"  - {duplicate}")
            
            if args.verbose and report.broken_links:
                print(f"\nBroken links:")
                for broken in report.broken_links:
                    print(f"  - {broken}")
        
        # Generate API documentation if requested
        if args.generate_api or args.api_only:
            print("\nGenerating API documentation...")
            api_docs = generator.generate_api_docs(existing_code_dirs)
            print(f"Generated API documentation for {len(api_docs)} modules")
            
            if args.verbose:
                print("API modules documented:")
                for api_doc in api_docs:
                    print(f"  - {api_doc.module_name} ({len(api_doc.classes)} classes, {len(api_doc.functions)} functions)")
        
        # Save consolidated documentation
        print("\nSaving consolidated documentation...")
        generator.save_consolidated_docs()
        
        print(f"\n✅ Documentation successfully generated in: {output_dir}")
        
        # Provide next steps
        print(f"\nNext steps:")
        print(f"  1. Review the generated documentation in {output_dir}")
        print(f"  2. Set up a documentation server (see task 4.3)")
        print(f"  3. Configure documentation validation (see task 4.4)")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())