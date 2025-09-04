import pytest
#!/usr/bin/env python3
"""
Test Suite Validation Script
Validates that all comprehensive test files are properly structured and importable
"""

import sys
import importlib.util
from pathlib import Path
from typing import List, Dict, Any

class TestSuiteValidator:
    """Validator for comprehensive test suite"""
    
    def __init__(self):
        self.validation_results = {}
        self.test_files = [
            'test_comprehensive_integration_suite.py',
            'test_model_integration_comprehensive.py',
            'test_end_to_end_comprehensive.py',
            'test_performance_benchmarks.py'
        ]
    
    def validate_file_structure(self, file_path: Path) -> Dict[str, Any]:
        """Validate a test file's structure"""
        result = {
            'file_exists': file_path.exists(),
            'is_importable': False,
            'has_test_classes': False,
            'has_test_methods': False,
            'test_classes': [],
            'test_methods': [],
            'errors': []
        }
        
        if not result['file_exists']:
            result['errors'].append(f"File does not exist: {file_path}")
            return result
        
        try:
            # Try to import the module
            spec = importlib.util.spec_from_file_location("test_module", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            result['is_importable'] = True
            
            # Find test classes and methods
            for name in dir(module):
                obj = getattr(module, name)
                
                if isinstance(obj, type) and name.startswith('Test'):
                    result['test_classes'].append(name)
                    
                    # Find test methods in the class
                    test_methods = [method for method in dir(obj) 
                                  if method.startswith('test_') and callable(getattr(obj, method))]
                    result['test_methods'].extend([f"{name}.{method}" for method in test_methods])
            
            result['has_test_classes'] = len(result['test_classes']) > 0
            result['has_test_methods'] = len(result['test_methods']) > 0
            
        except Exception as e:
            result['errors'].append(f"Import error: {str(e)}")
        
        return result
    
    def validate_all_files(self) -> bool:
        """Validate all test files"""
        print("🔍 Validating Comprehensive Test Suite")
        print("=" * 50)
        
        all_valid = True
        
        for test_file in self.test_files:
            file_path = Path(__file__).parent / test_file
            print(f"\n📄 Validating {test_file}")
            
            result = self.validate_file_structure(file_path)
            self.validation_results[test_file] = result
            
            if result['file_exists']:
                print(f"  ✅ File exists")
            else:
                print(f"  ❌ File missing")
                all_valid = False
            
            if result['is_importable']:
                print(f"  ✅ File is importable")
            else:
                print(f"  ❌ File has import errors")
                all_valid = False
            
            if result['has_test_classes']:
                print(f"  ✅ Has test classes: {len(result['test_classes'])}")
                for class_name in result['test_classes']:
                    print(f"    - {class_name}")
            else:
                print(f"  ❌ No test classes found")
                all_valid = False
            
            if result['has_test_methods']:
                print(f"  ✅ Has test methods: {len(result['test_methods'])}")
            else:
                print(f"  ❌ No test methods found")
                all_valid = False
            
            if result['errors']:
                print(f"  ❌ Errors:")
                for error in result['errors']:
                    print(f"    - {error}")
                all_valid = False
        
        return all_valid
    
    def print_summary(self):
        """Print validation summary"""
        print("\n" + "=" * 50)
        print("📊 VALIDATION SUMMARY")
        print("=" * 50)
        
        total_files = len(self.test_files)
        valid_files = sum(1 for result in self.validation_results.values() 
                         if result['file_exists'] and result['is_importable'] 
                         and result['has_test_classes'] and result['has_test_methods'])
        
        total_classes = sum(len(result['test_classes']) for result in self.validation_results.values())
        total_methods = sum(len(result['test_methods']) for result in self.validation_results.values())
        
        print(f"📁 Files: {valid_files}/{total_files} valid")
        print(f"🏗️  Test Classes: {total_classes}")
        print(f"🧪 Test Methods: {total_methods}")
        
        if valid_files == total_files:
            print(f"\n🎉 ALL TEST FILES ARE VALID!")
            return True
        else:
            print(f"\n💥 {total_files - valid_files} TEST FILES HAVE ISSUES!")
            return False
    
    def check_dependencies(self):
        """Check if required dependencies are available"""
        print("\n🔧 Checking Dependencies")
        print("-" * 30)
        
        required_packages = [
            'pytest',
            'asyncio',
            'httpx',
            'fastapi',
            'psutil',
            'torch',
            'pathlib'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"  ✅ {package}")
            except ImportError:
                print(f"  ❌ {package} (missing)")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
            print("Install with: pip install " + " ".join(missing_packages))
            return False
        else:
            print(f"\n✅ All dependencies available")
            return True
    
    def validate_test_requirements(self):
        """Validate that tests meet the requirements"""
        print("\n📋 Checking Requirements Coverage")
        print("-" * 40)
        
        requirements_coverage = {
            'model_integration_bridge_tests': False,
            'real_generation_pipeline_tests': False,
            'end_to_end_integration_tests': False,
            'performance_benchmark_tests': False
        }
        
        # Check for specific test patterns
        for file_name, result in self.validation_results.items():
            if 'model_integration' in file_name:
                requirements_coverage['model_integration_bridge_tests'] = result['has_test_methods']
            elif 'real_generation_pipeline' in file_name:
                requirements_coverage['real_generation_pipeline_tests'] = result['has_test_methods']
            elif 'comprehensive_integration_suite' in file_name:
                # The comprehensive suite covers real generation pipeline tests
                if result['has_test_methods']:
                    requirements_coverage['real_generation_pipeline_tests'] = True
            elif 'end_to_end' in file_name:
                requirements_coverage['end_to_end_integration_tests'] = result['has_test_methods']
            elif 'performance' in file_name:
                requirements_coverage['performance_benchmark_tests'] = result['has_test_methods']
        
        all_requirements_met = True
        
        for requirement, covered in requirements_coverage.items():
            if covered:
                print(f"  ✅ {requirement.replace('_', ' ').title()}")
            else:
                print(f"  ❌ {requirement.replace('_', ' ').title()}")
                all_requirements_met = False
        
        return all_requirements_met

def main():
    """Main validation function"""
    validator = TestSuiteValidator()
    
    # Run all validations
    files_valid = validator.validate_all_files()
    deps_available = validator.check_dependencies()
    requirements_met = validator.validate_test_requirements()
    
    # Print final summary
    overall_valid = validator.print_summary()
    
    print(f"\n🔍 OVERALL VALIDATION RESULT:")
    if files_valid and deps_available and requirements_met and overall_valid:
        print(f"✅ Test suite is ready for execution!")
        return True
    else:
        print(f"❌ Test suite has issues that need to be resolved.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)