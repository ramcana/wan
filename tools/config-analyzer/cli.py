#!/usr/bin/env python3
"""
Configuration Analysis CLI

Command-line interface for configuration landscape analysis.
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, Any

from config_landscape_analyzer import ConfigLandscapeAnalyzer, ConfigAnalysisReport


def print_summary(report: ConfigAnalysisReport):
    """Print a summary of the configuration analysis."""
    print("=" * 60)
    print("CONFIGURATION LANDSCAPE ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"\n📊 OVERVIEW:")
    print(f"  Total configuration files found: {report.total_files}")
    
    # File types breakdown
    file_types = {}
    for config_file in report.config_files:
        file_types[config_file.file_type] = file_types.get(config_file.file_type, 0) + 1
    
    print(f"  File types:")
    for file_type, count in sorted(file_types.items()):
        print(f"    {file_type.upper()}: {count} files")
    
    print(f"\n⚠️  ISSUES DETECTED:")
    print(f"  Configuration conflicts: {len(report.conflicts)}")
    print(f"  Duplicate settings: {len(report.duplicate_settings)}")
    print(f"  Dependencies: {len(report.dependencies)}")
    
    if report.conflicts:
        print(f"\n🔥 HIGH PRIORITY CONFLICTS:")
        high_priority = [c for c in report.conflicts if c.severity == 'high']
        for conflict in high_priority[:5]:  # Show top 5
            print(f"    {conflict.setting_name}: {len(conflict.files)} files with different values")
    
    print(f"\n💡 RECOMMENDATIONS:")
    for i, recommendation in enumerate(report.recommendations[:5], 1):
        print(f"  {i}. {recommendation}")
    
    print(f"\n📋 MIGRATION PLAN:")
    phases = report.migration_plan.get('phases', [])
    for phase in phases:
        print(f"  Phase {phase['phase']}: {phase['name']}")
    
    print("\n" + "=" * 60)


def print_detailed_conflicts(report: ConfigAnalysisReport):
    """Print detailed information about configuration conflicts."""
    if not report.conflicts:
        print("No configuration conflicts detected.")
        return
    
    print("CONFIGURATION CONFLICTS")
    print("=" * 40)
    
    # Group by severity
    by_severity = {'high': [], 'medium': [], 'low': []}
    for conflict in report.conflicts:
        by_severity[conflict.severity].append(conflict)
    
    for severity in ['high', 'medium', 'low']:
        conflicts = by_severity[severity]
        if not conflicts:
            continue
        
        print(f"\n{severity.upper()} SEVERITY ({len(conflicts)} conflicts):")
        print("-" * 30)
        
        for conflict in conflicts:
            print(f"\nSetting: {conflict.setting_name}")
            print(f"Files involved: {len(conflict.files)}")
            for i, (file_path, value) in enumerate(zip(conflict.files, conflict.values)):
                print(f"  {i+1}. {file_path}: {value}")


def print_file_details(report: ConfigAnalysisReport):
    """Print detailed information about each configuration file."""
    print("CONFIGURATION FILES DETAILS")
    print("=" * 40)
    
    # Group by directory
    by_directory = {}
    for config_file in report.config_files:
        directory = str(config_file.path.parent)
        if directory not in by_directory:
            by_directory[directory] = []
        by_directory[directory].append(config_file)
    
    for directory, files in sorted(by_directory.items()):
        print(f"\n📁 {directory}/")
        print("-" * (len(directory) + 4))
        
        for config_file in sorted(files, key=lambda x: x.path.name):
            size_kb = config_file.size / 1024
            settings_count = len(config_file.settings) if config_file.settings else 0
            
            print(f"  📄 {config_file.path.name}")
            print(f"      Type: {config_file.file_type.upper()}")
            print(f"      Size: {size_kb:.1f} KB")
            print(f"      Settings: {settings_count}")
            
            if config_file.dependencies:
                print(f"      Dependencies: {', '.join(config_file.dependencies)}")


def print_migration_plan(report: ConfigAnalysisReport):
    """Print the detailed migration plan."""
    plan = report.migration_plan
    
    print("CONFIGURATION MIGRATION PLAN")
    print("=" * 40)
    
    print("\n📋 MIGRATION PHASES:")
    for phase in plan.get('phases', []):
        print(f"\nPhase {phase['phase']}: {phase['name']}")
        print(f"Description: {phase['description']}")
        print("Tasks:")
        for task in phase['tasks']:
            print(f"  • {task}")
    
    print(f"\n📂 FILE MAPPING:")
    file_mapping = plan.get('file_mapping', {})
    
    # Group by target section
    by_section = {}
    for file_path, mapping in file_mapping.items():
        section = mapping['target_section']
        if section not in by_section:
            by_section[section] = []
        by_section[section].append((file_path, mapping))
    
    for section, files in sorted(by_section.items()):
        print(f"\n{section.upper()} Section:")
        for file_path, mapping in files:
            priority = mapping['priority']
            manual_review = "⚠️ Manual review required" if mapping['requires_manual_review'] else "✅ Auto-migrate"
            print(f"  {Path(file_path).name} ({priority} priority) - {manual_review}")


def export_report(report: ConfigAnalysisReport, output_path: str, format_type: str):
    """Export the analysis report to a file."""
    # Convert report to dictionary
    report_dict = {
        'total_files': report.total_files,
        'config_files': [
            {
                'path': str(cf.path),
                'file_type': cf.file_type,
                'size': cf.size,
                'settings_count': len(cf.settings) if cf.settings else 0,
                'settings': list(cf.settings) if cf.settings else [],
                'dependencies': cf.dependencies,
                'referenced_by': cf.referenced_by
            }
            for cf in report.config_files
        ],
        'conflicts': [
            {
                'setting_name': c.setting_name,
                'files': c.files,
                'values': c.values,
                'severity': c.severity
            }
            for c in report.conflicts
        ],
        'dependencies': [
            {
                'source_file': d.source_file,
                'target_file': d.target_file,
                'dependency_type': d.dependency_type
            }
            for d in report.dependencies
        ],
        'duplicate_settings': report.duplicate_settings,
        'recommendations': report.recommendations,
        'migration_plan': report.migration_plan
    }
    
    with open(output_path, 'w') as f:
        if format_type == 'yaml':
            yaml.dump(report_dict, f, default_flow_style=False, indent=2)
        else:
            json.dump(report_dict, f, indent=2, default=str)
    
    print(f"Report exported to {output_path}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description='Analyze configuration landscape and generate consolidation recommendations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with summary
  python cli.py analyze --project-root .
  
  # Detailed analysis with conflicts
  python cli.py analyze --project-root . --show-conflicts
  
  # Export full report
  python cli.py analyze --project-root . --export report.json
  
  # Show migration plan
  python cli.py analyze --project-root . --show-migration-plan
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze configuration landscape')
    analyze_parser.add_argument('--project-root', default='.', help='Project root directory')
    analyze_parser.add_argument('--show-conflicts', action='store_true', help='Show detailed conflict information')
    analyze_parser.add_argument('--show-files', action='store_true', help='Show detailed file information')
    analyze_parser.add_argument('--show-migration-plan', action='store_true', help='Show migration plan')
    analyze_parser.add_argument('--export', help='Export report to file')
    analyze_parser.add_argument('--format', choices=['json', 'yaml'], default='json', help='Export format')
    
    # Summary command
    summary_parser = subparsers.add_parser('summary', help='Show summary from existing report')
    summary_parser.add_argument('report_file', help='Path to existing analysis report')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'analyze':
        # Run analysis
        analyzer = ConfigLandscapeAnalyzer(args.project_root)
        report = analyzer.generate_report()
        
        # Always show summary
        print_summary(report)
        
        # Show additional details if requested
        if args.show_conflicts:
            print("\n")
            print_detailed_conflicts(report)
        
        if args.show_files:
            print("\n")
            print_file_details(report)
        
        if args.show_migration_plan:
            print("\n")
            print_migration_plan(report)
        
        # Export if requested
        if args.export:
            export_report(report, args.export, args.format)
    
    elif args.command == 'summary':
        # Load existing report and show summary
        with open(args.report_file, 'r') as f:
            if args.report_file.endswith('.yaml') or args.report_file.endswith('.yml'):
                report_dict = yaml.safe_load(f)
            else:
                report_dict = json.load(f)
        
        # Convert back to report object (simplified)
        print("CONFIGURATION ANALYSIS SUMMARY")
        print("=" * 40)
        print(f"Total files: {report_dict.get('total_files', 0)}")
        print(f"Conflicts: {len(report_dict.get('conflicts', []))}")
        print(f"Duplicate settings: {len(report_dict.get('duplicate_settings', {}))}")
        
        print("\nRecommendations:")
        for i, rec in enumerate(report_dict.get('recommendations', []), 1):
            print(f"  {i}. {rec}")


if __name__ == '__main__':
    main()
