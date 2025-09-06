#!/usr/bin/env python3
"""
Automated Syntax Checking Script for WAN22 Critical Files
Performs regular syntax validation on critical Python files and reports issues
"""

import sys
import os
import logging
from datetime import datetime
from typing import Dict, List
from syntax_validator import SyntaxValidator, ValidationResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('syntax_check.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class AutomatedSyntaxChecker:
    """
    Automated syntax checker for critical WAN22 files
    """
    
    def __init__(self):
        """Initialize the automated syntax checker"""
        self.validator = SyntaxValidator()
        self.critical_files = [
            'ui_event_handlers_enhanced.py',
            'ui_event_handlers.py',
            'main.py',
            'ui.py',
            'wan_pipeline_loader.py',
            'utils.py',
            'syntax_validator.py',
            'enhanced_image_validation.py',
            'enhanced_image_preview_manager.py',
            'resolution_manager.py',
            'progress_tracker.py',
            'help_text_system.py',
            'error_handler.py',
            'input_validation.py',
            'ui_validation.py'
        ]
        
        # Files that are expected to exist
        self.required_files = [
            'main.py',
            'ui.py',
            'utils.py'
        ]
        
        logger.info("Automated syntax checker initialized")
    
    def check_all_files(self) -> Dict[str, ValidationResult]:
        """
        Check syntax of all critical files
        
        Returns:
            Dictionary mapping file paths to validation results
        """
        logger.info("Starting automated syntax check of critical files")
        
        results = {}
        files_checked = 0
        files_with_errors = 0
        
        for file_path in self.critical_files:
            if os.path.exists(file_path):
                logger.info(f"Checking {file_path}...")
                result = self.validator.validate_file(file_path)
                results[file_path] = result
                files_checked += 1
                
                if not result.is_valid:
                    files_with_errors += 1
                    logger.error(f"❌ Syntax errors found in {file_path}")
                    for error in result.errors:
                        logger.error(f"   Line {error.line_number}: {error.message}")
                else:
                    logger.info(f"✅ {file_path} passed syntax validation")
            else:
                if file_path in self.required_files:
                    logger.warning(f"⚠️ Required file missing: {file_path}")
                else:
                    logger.debug(f"Optional file not found: {file_path}")
        
        logger.info(f"Syntax check completed: {files_checked} files checked, {files_with_errors} with errors")
        return results
    
    def generate_report(self, results: Dict[str, ValidationResult]) -> str:
        """
        Generate a detailed syntax check report
        
        Args:
            results: Dictionary of validation results
            
        Returns:
            Formatted report string
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        total_files = len(results)
        valid_files = sum(1 for result in results.values() if result.is_valid)
        invalid_files = total_files - valid_files
        
        report = f"""
WAN22 Automated Syntax Check Report
==================================
Generated: {timestamp}

Summary:
--------
Total files checked: {total_files}
Files with valid syntax: {valid_files}
Files with syntax errors: {invalid_files}
Success rate: {(valid_files/total_files*100):.1f}% if total_files > 0 else 0

"""
        
        if invalid_files > 0:
            report += "Files with syntax errors:\n"
            report += "-------------------------\n"
            
            for file_path, result in results.items():
                if not result.is_valid:
                    report += f"\n❌ {file_path}:\n"
                    for error in result.errors:
                        report += f"   Line {error.line_number}: {error.message}\n"
                        if error.suggested_fix:
                            report += f"   Suggested fix: {error.suggested_fix}\n"
        
        if valid_files > 0:
            report += "\nFiles with valid syntax:\n"
            report += "------------------------\n"
            
            for file_path, result in results.items():
                if result.is_valid:
                    report += f"✅ {file_path}\n"
        
        # Check for missing required files
        missing_required = []
        for required_file in self.required_files:
            if not os.path.exists(required_file):
                missing_required.append(required_file)
        
        if missing_required:
            report += "\nMissing required files:\n"
            report += "----------------------\n"
            for file_path in missing_required:
                report += f"⚠️ {file_path}\n"
        
        report += "\nRecommendations:\n"
        report += "---------------\n"
        
        if invalid_files > 0:
            report += "- Fix syntax errors in the files listed above\n"
            report += "- Use the SyntaxValidator.repair_syntax_errors() method for automated fixes\n"
            report += "- Review and test any automated repairs before committing\n"
        
        if missing_required:
            report += "- Ensure all required files are present in the project\n"
        
        if invalid_files == 0 and not missing_required:
            report += "- All critical files have valid syntax! ✅\n"
            report += "- Continue regular syntax checking to maintain code quality\n"
        
        return report
    
    def save_report(self, report: str, filename: str = None) -> str:
        """
        Save the syntax check report to a file
        
        Args:
            report: The report content
            filename: Optional filename (defaults to timestamped name)
            
        Returns:
            Path to the saved report file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"syntax_check_report_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info(f"📄 Syntax check report saved to: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to save report to {filename}: {e}")
            raise
    
    def run_check_and_report(self, save_report: bool = True) -> bool:
        """
        Run complete syntax check and generate report
        
        Args:
            save_report: Whether to save the report to a file
            
        Returns:
            True if all files passed syntax validation, False otherwise
        """
        try:
            # Run syntax check
            results = self.check_all_files()
            
            # Generate report
            report = self.generate_report(results)
            
            # Print report to console
            print(report)
            
            # Save report if requested
            if save_report:
                self.save_report(report)
            
            # Return success status
            all_valid = all(result.is_valid for result in results.values())
            
            if all_valid:
                logger.info("🎉 All critical files passed syntax validation!")
            else:
                logger.warning("⚠️ Some files have syntax errors that need attention")
            
            return all_valid
            
        except Exception as e:
            logger.error(f"Syntax check failed: {e}")
            return False

def main():
    """Main entry point for automated syntax checking"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated syntax checker for WAN22 critical files")
    parser.add_argument('--no-report', action='store_true', 
                       help="Don't save report to file")
    parser.add_argument('--quiet', action='store_true',
                       help="Reduce output verbosity")
    parser.add_argument('--files', nargs='+',
                       help="Specific files to check (overrides default critical files)")
    
    args = parser.parse_args()
    
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    checker = AutomatedSyntaxChecker()
    
    # Override critical files if specified
    if args.files:
        checker.critical_files = args.files
        logger.info(f"Checking specified files: {args.files}")
    
    # Run the check
    success = checker.run_check_and_report(save_report=not args.no_report)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()