"""
Example usage of the Codebase Cleanup Tools

This script demonstrates how to use the duplicate detection system
to clean up a codebase.
"""

import json
from pathlib import Path
from duplicate_detector import DuplicateDetector


def demonstrate_duplicate_detection():
    """Demonstrate duplicate detection functionality"""
    print("=== Duplicate Detection Demo ===\n")
    
    # Initialize detector for current project
    detector = DuplicateDetector(".", backup_dir="demo_backups/duplicates")
    
    print("Scanning project for duplicates...")
    report = detector.scan_for_duplicates()
    
    # Display results
    print(f"\n📊 Scan Results:")
    print(f"Files scanned: {report.total_files_scanned}")
    print(f"Duplicate files found: {len(report.duplicate_files)}")
    print(f"Duplicate groups: {len(report.duplicate_groups)}")
    print(f"Potential disk space savings: {report.potential_savings / 1024:.1f} KB")
    
    # Show duplicate groups
    if report.duplicate_groups:
        print(f"\n📁 Duplicate Groups:")
        for group_id, files in report.duplicate_groups.items():
            print(f"\n{group_id}:")
            for file_path in files:
                file_size = Path(file_path).stat().st_size if Path(file_path).exists() else 0
                print(f"  - {file_path} ({file_size} bytes)")
    
    # Show recommendations
    if report.recommendations:
        print(f"\n💡 Recommendations:")
        for rec in report.recommendations:
            print(f"  - {rec}")
    
    # Save detailed report
    report_path = "duplicate_analysis_report.json"
    detector.save_report(report, report_path)
    print(f"\n📄 Detailed report saved to: {report_path}")
    
    return report, detector


def demonstrate_dead_code_analysis():
    """Demonstrate dead code analysis functionality"""
    print("\n=== Dead Code Analysis Demo ===\n")
    
    from dead_code_analyzer import DeadCodeAnalyzer
    
    # Initialize analyzer
    analyzer = DeadCodeAnalyzer(".", backup_dir="demo_backups/dead_code")
    
    print("Analyzing dead code...")
    report = analyzer.analyze_dead_code(include_tests=False)
    
    # Display results
    print(f"\n📊 Analysis Results:")
    print(f"Files analyzed: {report.total_files_analyzed}")
    print(f"Dead functions: {len(report.dead_functions)}")
    print(f"Dead classes: {len(report.dead_classes)}")
    print(f"Unused imports: {len(report.unused_imports)}")
    print(f"Dead files: {len(report.dead_files)}")
    print(f"Potential lines removed: {report.potential_lines_removed}")
    
    # Show some examples
    if report.dead_functions:
        print(f"\n🔍 Dead Functions (showing first 5):")
        for func in report.dead_functions[:5]:
            print(f"  - {func.name} in {func.file_path}:{func.line_number}")
    
    if report.unused_imports:
        print(f"\n📦 Unused Imports (showing first 5):")
        for imp in report.unused_imports[:5]:
            print(f"  - {imp.import_name} from {imp.module_name} in {Path(imp.file_path).name}")
    
    # Show recommendations
    if report.recommendations:
        print(f"\n💡 Recommendations:")
        for rec in report.recommendations:
            print(f"  - {rec}")
    
    # Save detailed report
    report_path = "dead_code_analysis_report.json"
    analyzer.save_report(report, report_path)
    print(f"\n📄 Detailed report saved to: {report_path}")
    
    return report, analyzer


def demonstrate_naming_standardization():
    """Demonstrate naming standardization functionality"""
    print("\n=== Naming Standardization Demo ===\n")
    
    from naming_standardizer import NamingStandardizer
    
    # Initialize standardizer
    standardizer = NamingStandardizer(".", backup_dir="demo_backups/naming")
    
    print("Analyzing naming conventions...")
    report = standardizer.analyze_naming_conventions()
    
    # Display results
    print(f"\n📊 Analysis Results:")
    print(f"Files analyzed: {report.total_files_analyzed}")
    print(f"Naming violations: {len(report.violations)}")
    print(f"Inconsistent patterns: {len(report.inconsistent_patterns)}")
    print(f"Organization suggestions: {len(report.organization_suggestions)}")
    
    # Show convention summary
    if report.convention_summary:
        print(f"\n📈 Convention Summary:")
        for convention, count in report.convention_summary.items():
            print(f"  {convention}: {count} violations")
    
    # Show some violations
    if report.violations:
        print(f"\n🔍 Naming Violations (showing first 5):")
        for violation in report.violations[:5]:
            print(f"  - {violation.element_type} '{violation.name}' -> '{violation.suggested_name}'")
            print(f"    in {Path(violation.file_path).name}:{violation.line_number}")
    
    # Show organization suggestions
    if report.organization_suggestions:
        print(f"\n📁 Organization Suggestions (showing first 3):")
        for suggestion in report.organization_suggestions[:3]:
            print(f"  - Move {Path(suggestion.current_path).name}")
            print(f"    to {suggestion.suggested_path}")
            print(f"    Reason: {suggestion.reason}")
    
    # Show recommendations
    if report.recommendations:
        print(f"\n💡 Recommendations:")
        for rec in report.recommendations:
            print(f"  - {rec}")
    
    # Save detailed report
    report_path = "naming_analysis_report.json"
    standardizer.save_report(report, report_path)
    print(f"\n📄 Detailed report saved to: {report_path}")
    
    return report, standardizer


def demonstrate_safe_removal():
    """Demonstrate safe duplicate removal with backup"""
    print("\n=== Safe Duplicate Removal Demo ===\n")
    
    report, detector = demonstrate_duplicate_detection()
    
    if not report.duplicate_groups:
        print("No duplicates found to remove.")
        return
    
    # Show what would be removed
    exact_groups = {k: v for k, v in report.duplicate_groups.items() if k.startswith('exact_')}
    similar_groups = {k: v for k, v in report.duplicate_groups.items() if k.startswith('similar_')}
    
    if exact_groups:
        print(f"🔍 Found {len(exact_groups)} groups of exact duplicates:")
        for group_id, files in exact_groups.items():
            if len(files) > 1:
                keep_file = min(files, key=len)  # Keep shortest path
                remove_files = [f for f in files if f != keep_file]
                print(f"\n{group_id}:")
                print(f"  Keep: {keep_file}")
                for remove_file in remove_files:
                    print(f"  Remove: {remove_file}")
    
    if similar_groups:
        print(f"\n🔍 Found {len(similar_groups)} groups of similar files (manual review recommended):")
        for group_id, files in similar_groups.items():
            print(f"\n{group_id}:")
            for file_path in files:
                print(f"  - {file_path}")
    
    # Ask for confirmation (in real usage)
    print(f"\n⚠️  This is a demo - no files will actually be removed.")
    print(f"In real usage, you would be prompted to confirm removal.")
    
    # Demonstrate the removal process (dry run)
    print(f"\n🛡️  Safe removal process:")
    print(f"1. Create backup of all files to be removed")
    print(f"2. Remove duplicate files (keeping one copy of each)")
    print(f"3. Generate removal report")
    print(f"4. Provide rollback capability")
    
    # Show how to use the removal function
    print(f"\n💻 Code example:")
    print(f"results = detector.safe_remove_duplicates(")
    print(f"    report.duplicate_groups,")
    print(f"    auto_remove_exact=True")
    print(f")")
    print(f"print(results)")


def demonstrate_rollback():
    """Demonstrate rollback functionality"""
    print("\n=== Rollback Demo ===\n")
    
    print("🔄 Rollback functionality allows you to:")
    print("  - Restore all removed files from backup")
    print("  - Maintain original directory structure")
    print("  - Verify backup integrity before rollback")
    
    print(f"\n💻 Code example:")
    print(f"success = detector.rollback_removal('backups/duplicates/duplicate_removal_20240101_120000')")
    print(f"if success:")
    print(f"    print('All files restored successfully')")


def analyze_project_duplicates():
    """Analyze the current project for real duplicates"""
    print("\n=== Real Project Analysis ===\n")
    
    # Focus on specific areas that commonly have duplicates
    areas_to_check = [
        "backend",
        "frontend", 
        "tools",
        "tests",
        "docs"
    ]
    
    for area in areas_to_check:
        area_path = Path(area)
        if area_path.exists():
            print(f"\n📂 Analyzing {area}/")
            detector = DuplicateDetector(str(area_path))
            report = detector.scan_for_duplicates()
            
            if report.duplicate_files:
                print(f"  Found {len(report.duplicate_files)} duplicate files")
                print(f"  Potential savings: {report.potential_savings / 1024:.1f} KB")
                
                # Show top duplicate groups
                for group_id, files in list(report.duplicate_groups.items())[:3]:
                    if len(files) > 1:
                        print(f"  Group {group_id}: {len(files)} files")
                        for file_path in files[:2]:  # Show first 2 files
                            print(f"    - {file_path}")
                        if len(files) > 2:
                            print(f"    ... and {len(files) - 2} more")
            else:
                print(f"  ✅ No duplicates found")


def main():
    """Main demonstration function"""
    print("🧹 Codebase Cleanup Tools - Comprehensive Demo")
    print("=" * 60)
    
    try:
        # Run all demonstrations
        print("Running comprehensive codebase analysis...")
        
        duplicate_report, duplicate_detector = demonstrate_duplicate_detection()
        dead_code_report, dead_code_analyzer = demonstrate_dead_code_analysis()
        naming_report, naming_standardizer = demonstrate_naming_standardization()
        
        # Summary
        print(f"\n" + "=" * 60)
        print(f"📋 COMPREHENSIVE CLEANUP SUMMARY")
        print(f"=" * 60)
        
        print(f"\n🔍 Duplicate Analysis:")
        print(f"  - {len(duplicate_report.duplicate_files)} duplicate files found")
        print(f"  - {duplicate_report.potential_savings / 1024:.1f} KB potential savings")
        
        print(f"\n💀 Dead Code Analysis:")
        print(f"  - {len(dead_code_report.dead_functions)} dead functions")
        print(f"  - {len(dead_code_report.dead_classes)} dead classes")
        print(f"  - {len(dead_code_report.unused_imports)} unused imports")
        print(f"  - {dead_code_report.potential_lines_removed} potential lines to remove")
        
        print(f"\n📝 Naming Analysis:")
        print(f"  - {len(naming_report.violations)} naming violations")
        print(f"  - {len(naming_report.inconsistent_patterns)} inconsistent patterns")
        print(f"  - {len(naming_report.organization_suggestions)} organization suggestions")
        
        # Overall recommendations
        print(f"\n🎯 PRIORITY ACTIONS:")
        print(f"1. Fix critical naming violations (classes, functions)")
        print(f"2. Remove unused imports (quick wins)")
        print(f"3. Clean up duplicate files")
        print(f"4. Remove dead code after thorough testing")
        print(f"5. Reorganize files for better structure")
        
        print(f"\n🛠️  NEXT STEPS:")
        print(f"To apply fixes, use:")
        print(f"  python -m tools.codebase-cleanup duplicates --remove")
        print(f"  python -m tools.codebase-cleanup dead-code --remove")
        print(f"  python -m tools.codebase-cleanup naming --fix --convention snake_case")
        
        print(f"\n✅ Comprehensive analysis completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
