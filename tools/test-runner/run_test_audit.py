#!/usr/bin/env python3
"""
Test Audit Runner - Script to audit and fix existing tests
"""

import asyncio
import logging
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from test_auditor import TestAuditor
from orchestrator import TestConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Main audit and fix process"""
    logger.info("Starting test suite audit and fix process")
    
    try:
        # Load configuration
        config_path = project_root / "tests" / "config" / "test-config.yaml"
        if not config_path.exists():
            logger.error(f"Test configuration not found: {config_path}")
            return 1
        
        config = TestConfig.load_from_file(config_path)
        logger.info(f"Loaded test configuration from {config_path}")
        
        # Create auditor
        auditor = TestAuditor(config)
        
        # Run comprehensive audit
        logger.info("Running comprehensive test audit...")
        report = auditor.audit_all_tests()
        
        # Print summary
        print("\n" + "="*60)
        print("TEST AUDIT SUMMARY")
        print("="*60)
        print(f"Total Files:        {report.summary['total_files']}")
        print(f"Broken Files:       {report.summary['broken_files']}")
        print(f"Incomplete Files:   {report.summary['incomplete_files']}")
        print(f"Miscategorized:     {report.summary['miscategorized_files']}")
        print(f"Healthy Files:      {report.summary['healthy_files']}")
        print(f"Total Issues:       {report.summary['issues_found']}")
        print("="*60)
        
        # Generate detailed audit report
        report_path = project_root / "test_results" / f"audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        auditor.generate_audit_report(report, report_path)
        print(f"Detailed report: {report_path}")
        
        # Show broken files details
        if report.broken_files:
            print(f"\nBROKEN FILES ({len(report.broken_files)}):")
            for audit in report.broken_files:
                print(f"  - {audit.file_path}")
                for issue in audit.issues:
                    if issue.severity == 'critical':
                        print(f"    ❌ {issue.description}")
        
        # Show incomplete files
        if report.incomplete_files:
            print(f"\nINCOMPLETE FILES ({len(report.incomplete_files)}):")
            for audit in report.incomplete_files:
                print(f"  - {audit.file_path}")
                incomplete_issues = [i for i in audit.issues if i.issue_type.value == 'incomplete']
                for issue in incomplete_issues:
                    print(f"    ⚠️  {issue.description}")
        
        # Show miscategorized files
        if report.miscategorized_files:
            print(f"\nMISCATEGORIZED FILES ({len(report.miscategorized_files)}):")
            for audit in report.miscategorized_files:
                current = audit.current_category.value if audit.current_category else 'None'
                suggested = audit.suggested_category.value if audit.suggested_category else 'None'
                print(f"  - {audit.file_path}")
                print(f"    📁 {current} → {suggested}")
        
        # Ask user for actions
        print("\nRECOMMENDED ACTIONS:")
        
        if report.broken_files:
            print("\n1. Fix broken tests:")
            response = input("   Apply automatic fixes where possible? (y/N): ").lower().strip()
            if response == 'y':
                fix_results = auditor.fix_broken_tests(report, auto_fix=True)
                print(f"   ✅ Fixed: {fix_results['files_fixed']} files")
                print(f"   ⚠️  Manual fixes needed: {fix_results['manual_fixes_needed']} files")
            else:
                # Generate fix suggestions
                fix_results = auditor.fix_broken_tests(report, auto_fix=False)
                print(f"   📝 Fix suggestions generated for {fix_results['manual_fixes_needed']} files")
        
        if report.miscategorized_files:
            print("\n2. Categorize tests:")
            response = input("   Move files to correct categories? (y/N): ").lower().strip()
            if response == 'y':
                categorization_results = auditor.categorize_tests(report, apply_moves=True)
                print(f"   ✅ Moved: {categorization_results['files_moved']} files")
            else:
                categorization_results = auditor.categorize_tests(report, apply_moves=False)
                print(f"   📋 Move plan created for {categorization_results['moves_planned']} files")
                
                # Show move plan
                if categorization_results['move_plan']:
                    print("   Move plan:")
                    for move in categorization_results['move_plan']:
                        print(f"     {Path(move['source']).name} → {move['category']}/")
        
        # Create summary of actions taken
        actions_summary = {
            'audit_completed': True,
            'report_generated': str(report_path),
            'broken_files_found': len(report.broken_files),
            'incomplete_files_found': len(report.incomplete_files),
            'miscategorized_files_found': len(report.miscategorized_files),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save actions summary
        summary_path = project_root / "test_results" / "audit_actions_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        

        with open(summary_path, 'w') as f:
            json.dump(actions_summary, f, indent=2)
        
        print(f"\n✅ Audit complete! Summary saved to: {summary_path}")
        
        # Return appropriate exit code
        if report.broken_files:
            print("\n⚠️  Some tests still need manual fixes")
            return 1
        else:
            print("\n🎉 All tests are in good shape!")
            return 0
        
    except Exception as e:
        logger.error(f"Audit process failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
