"""
Command Line Interface for Codebase Cleanup Tools

Provides unified CLI access to all codebase cleanup functionality:
- Duplicate detection and removal
- Dead code analysis and removal
- Naming standardization and file organization
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

try:
    from tools.codebase-cleanup.duplicate_detector import DuplicateDetector
    from tools.codebase-cleanup.dead_code_analyzer import DeadCodeAnalyzer
    from tools.codebase-cleanup.naming_standardizer import NamingStandardizer
except ImportError:
    # Fallback for direct execution
    from duplicate_detector import DuplicateDetector
    from dead_code_analyzer import DeadCodeAnalyzer
    from naming_standardizer import NamingStandardizer


def setup_duplicate_parser(subparsers):
    """Setup duplicate detection command parser"""
    duplicate_parser = subparsers.add_parser(
        'duplicates', 
        help='Detect and remove duplicate files'
    )
    
    duplicate_parser.add_argument(
        'path',
        nargs='?',
        default='.',
        help='Root path to scan for duplicates (default: current directory)'
    )
    
    duplicate_parser.add_argument(
        '--output', '-o',
        default='duplicate_report.json',
        help='Output file for duplicate report (default: duplicate_report.json)'
    )
    
    duplicate_parser.add_argument(
        '--backup-dir',
        default='backups/duplicates',
        help='Directory for backups (default: backups/duplicates)'
    )
    
    duplicate_parser.add_argument(
        '--remove',
        action='store_true',
        help='Automatically remove exact duplicates (creates backup first)'
    )
    
    duplicate_parser.add_argument(
        '--rollback',
        help='Rollback duplicate removal using backup path'
    )
    
    duplicate_parser.add_argument(
        '--similarity-threshold',
        type=float,
        default=0.8,
        help='Similarity threshold for near-duplicates (default: 0.8)'
    )


def setup_dead_code_parser(subparsers):
    """Setup dead code analysis command parser"""
    dead_code_parser = subparsers.add_parser(
        'dead-code',
        help='Analyze and remove dead code'
    )
    
    dead_code_parser.add_argument(
        'path',
        nargs='?',
        default='.',
        help='Root path to analyze (default: current directory)'
    )
    
    dead_code_parser.add_argument(
        '--output', '-o',
        default='dead_code_report.json',
        help='Output file for dead code report'
    )
    
    dead_code_parser.add_argument(
        '--remove',
        action='store_true',
        help='Remove identified dead code (creates backup first)'
    )
    
    dead_code_parser.add_argument(
        '--include-tests',
        action='store_true',
        help='Include test files in analysis'
    )


def setup_naming_parser(subparsers):
    """Setup naming standardization command parser"""
    naming_parser = subparsers.add_parser(
        'naming',
        help='Standardize naming conventions'
    )
    
    naming_parser.add_argument(
        'path',
        nargs='?',
        default='.',
        help='Root path to standardize (default: current directory)'
    )
    
    naming_parser.add_argument(
        '--output', '-o',
        default='naming_report.json',
        help='Output file for naming analysis report'
    )
    
    naming_parser.add_argument(
        '--fix',
        action='store_true',
        help='Apply naming fixes (creates backup first)'
    )
    
    naming_parser.add_argument(
        '--convention',
        choices=['snake_case', 'camelCase', 'PascalCase', 'kebab-case'],
        default='snake_case',
        help='Naming convention to enforce (default: snake_case)'
    )


def handle_duplicates_command(args):
    """Handle duplicate detection command"""
    if args.rollback:
        detector = DuplicateDetector(args.path, args.backup_dir)
        success = detector.rollback_removal(args.rollback)
        if success:
            print("Rollback completed successfully")
            return 0
        else:
            print("Rollback failed")
            return 1
    
    print(f"Scanning for duplicates in: {args.path}")
    detector = DuplicateDetector(args.path, args.backup_dir)
    
    # Scan for duplicates
    report = detector.scan_for_duplicates()
    
    # Save report
    detector.save_report(report, args.output)
    
    # Print summary
    print(f"\nDuplicate Detection Results:")
    print(f"Files scanned: {report.total_files_scanned}")
    print(f"Duplicate files found: {len(report.duplicate_files)}")
    print(f"Potential savings: {report.potential_savings / 1024:.1f} KB")
    print(f"Duplicate groups: {len(report.duplicate_groups)}")
    
    if report.recommendations:
        print("\nRecommendations:")
        for rec in report.recommendations:
            print(f"- {rec}")
    
    # Remove duplicates if requested
    if args.remove and report.duplicate_groups:
        print("\nRemoving exact duplicates...")
        results = detector.safe_remove_duplicates(report.duplicate_groups, auto_remove_exact=True)
        
        for operation, result in results.items():
            print(f"{operation}: {result}")
    
    return 0


def handle_dead_code_command(args):
    """Handle dead code analysis command"""
    print(f"Analyzing dead code in: {args.path}")
    
    try:
        from tools.codebase-cleanup.dead_code_analyzer import DeadCodeAnalyzer
    except ImportError:
        from dead_code_analyzer import DeadCodeAnalyzer
    analyzer = DeadCodeAnalyzer(args.path)
    
    # Analyze dead code
    report = analyzer.analyze_dead_code(include_tests=args.include_tests)
    
    # Save report
    analyzer.save_report(report, args.output)
    
    # Print summary
    print(f"\nDead Code Analysis Results:")
    print(f"Files analyzed: {report.total_files_analyzed}")
    print(f"Dead functions: {len(report.dead_functions)}")
    print(f"Dead classes: {len(report.dead_classes)}")
    print(f"Unused imports: {len(report.unused_imports)}")
    print(f"Dead files: {len(report.dead_files)}")
    print(f"Potential lines removed: {report.potential_lines_removed}")
    
    if report.recommendations:
        print("\nRecommendations:")
        for rec in report.recommendations:
            print(f"- {rec}")
    
    if args.remove:
        print("\nRemoving dead code...")
        results = analyzer.safe_remove_dead_code(report)
        for operation, result in results.items():
            print(f"{operation}: {result}")
    
    return 0


def handle_naming_command(args):
    """Handle naming standardization command"""
    print(f"Analyzing naming conventions in: {args.path}")
    
    try:
        from tools.codebase-cleanup.naming_standardizer import NamingStandardizer
    except ImportError:
        from naming_standardizer import NamingStandardizer
    standardizer = NamingStandardizer(args.path)
    
    # Analyze naming
    report = standardizer.analyze_naming_conventions()
    
    # Save report
    standardizer.save_report(report, args.output)
    
    # Print summary
    print(f"\nNaming Analysis Results:")
    print(f"Files analyzed: {report.total_files_analyzed}")
    print(f"Naming violations: {len(report.violations)}")
    print(f"Inconsistent patterns: {len(report.inconsistent_patterns)}")
    print(f"Organization suggestions: {len(report.organization_suggestions)}")
    
    if report.convention_summary:
        print(f"\nConvention Summary:")
        for convention, count in report.convention_summary.items():
            print(f"  {convention}: {count}")
    
    if report.recommendations:
        print(f"\nRecommendations:")
        for rec in report.recommendations:
            print(f"- {rec}")
    
    if args.fix:
        print(f"\nApplying {args.convention} naming convention...")
        results = standardizer.apply_naming_fixes(report, args.convention)
        for operation, result in results.items():
            print(f"{operation}: {result}")
    
    return 0


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Codebase Cleanup and Organization Tools',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan for duplicates
  python -m tools.codebase-cleanup duplicates

  # Remove exact duplicates automatically
  python -m tools.codebase-cleanup duplicates --remove

  # Analyze dead code
  python -m tools.codebase-cleanup dead-code

  # Standardize naming conventions
  python -m tools.codebase-cleanup naming --convention snake_case

  # Rollback duplicate removal
  python -m tools.codebase-cleanup duplicates --rollback backups/duplicates/duplicate_removal_20240101_120000
        """
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Codebase Cleanup Tools 1.0.0'
    )
    
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        metavar='COMMAND'
    )
    
    # Setup command parsers
    setup_duplicate_parser(subparsers)
    setup_dead_code_parser(subparsers)
    setup_naming_parser(subparsers)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Handle commands
    try:
        if args.command == 'duplicates':
            return handle_duplicates_command(args)
        elif args.command == 'dead-code':
            return handle_dead_code_command(args)
        elif args.command == 'naming':
            return handle_naming_command(args)
        else:
            print(f"Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())