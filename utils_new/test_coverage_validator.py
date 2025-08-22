#!/usr/bin/env python3
"""
Test Coverage Validator for Wan Model Compatibility System
Provides comprehensive test coverage analysis and validation
"""

import ast
import inspect
import importlib
import sys
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FunctionCoverage:
    """Coverage information for a single function"""
    function_name: str
    module_name: str
    is_tested: bool
    test_files: List[str] = field(default_factory=list)
    test_functions: List[str] = field(default_factory=list)
    coverage_percentage: float = 0.0
    missing_test_scenarios: List[str] = field(default_factory=list)

@dataclass
class ModuleCoverage:
    """Coverage information for a module"""
    module_name: str
    total_functions: int
    tested_functions: int
    coverage_percentage: float
    function_coverage: List[FunctionCoverage] = field(default_factory=list)
    missing_tests: List[str] = field(default_factory=list)

@dataclass
class CoverageReport:
    """Comprehensive coverage report"""
    total_modules: int
    total_functions: int
    tested_functions: int
    overall_coverage_percentage: float
    module_coverage: List[ModuleCoverage] = field(default_factory=list)
    coverage_gaps: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

class TestCoverageValidator:
    """
    Comprehensive test coverage validator for Wan model compatibility system
    Analyzes code coverage and identifies testing gaps
    """
    
    def __init__(self, source_dir: str = ".", test_dir: str = "."):
        """
        Initialize test coverage validator
        
        Args:
            source_dir: Directory containing source code
            test_dir: Directory containing test files
        """
        self.source_dir = Path(source_dir)
        self.test_dir = Path(test_dir)
        self.coverage_data = {}
        self.test_mapping = {}
        
        # Core components to analyze
        self.core_components = [
            "architecture_detector",
            "pipeline_manager", 
            "dependency_manager",
            "wan_pipeline_loader",
            "fallback_handler",
            "optimization_manager",
            "frame_tensor_handler",
            "video_encoder",
            "smoke_test_runner"
        ]
        
        # Required test scenarios for each component type
        self.required_test_scenarios = {
            "detector": [
                "valid_input_detection",
                "invalid_input_handling", 
                "edge_case_handling",
                "error_recovery"
            ],
            "manager": [
                "successful_operation",
                "failure_handling",
                "resource_management",
                "concurrent_access"
            ],
            "loader": [
                "successful_loading",
                "loading_failure",
                "optimization_application",
                "memory_management"
            ],
            "handler": [
                "normal_processing",
                "error_handling",
                "edge_cases",
                "resource_cleanup"
            ],
            "encoder": [
                "successful_encoding",
                "encoding_failure",
                "format_support",
                "dependency_handling"
            ]
        }
    
    def analyze_coverage(self) -> CoverageReport:
        """
        Analyze test coverage for all components
        
        Returns:
            CoverageReport with comprehensive coverage analysis
        """
        logger.info("Starting comprehensive test coverage analysis")
        
        # Discover source files and test files
        source_files = self._discover_source_files()
        test_files = self._discover_test_files()
        
        # Build test mapping
        self._build_test_mapping(test_files)
        
        # Analyze coverage for each module
        module_coverage_list = []
        total_functions = 0
        tested_functions = 0
        
        for source_file in source_files:
            module_coverage = self._analyze_module_coverage(source_file)
            if module_coverage:
                module_coverage_list.append(module_coverage)
                total_functions += module_coverage.total_functions
                tested_functions += module_coverage.tested_functions
        
        # Calculate overall coverage
        overall_coverage = (tested_functions / total_functions * 100) if total_functions > 0 else 0
        
        # Generate coverage report
        report = CoverageReport(
            total_modules=len(module_coverage_list),
            total_functions=total_functions,
            tested_functions=tested_functions,
            overall_coverage_percentage=overall_coverage,
            module_coverage=module_coverage_list
        )
        
        # Identify coverage gaps and generate recommendations
        report.coverage_gaps = self._identify_coverage_gaps(module_coverage_list)
        report.recommendations = self._generate_recommendations(report)
        
        # Save coverage report
        self._save_coverage_report(report)
        
        logger.info(f"Coverage analysis completed: {overall_coverage:.1f}% overall coverage")
        return report
    
    def _discover_source_files(self) -> List[Path]:
        """Discover source files to analyze"""
        source_files = []
        
        # Look for core component files
        for component in self.core_components:
            component_file = self.source_dir / f"{component}.py"
            if component_file.exists():
                source_files.append(component_file)
            else:
                logger.warning(f"Core component not found: {component_file}")
        
        # Look for additional Python files
        for py_file in self.source_dir.glob("*.py"):
            if (py_file.name.startswith("test_") or 
                py_file.name in ["__init__.py", "setup.py"] or
                py_file in source_files):
                continue
            
            # Check if it's a relevant module
            if self._is_relevant_module(py_file):
                source_files.append(py_file)
        
        logger.info(f"Discovered {len(source_files)} source files")
        return source_files
    
    def _discover_test_files(self) -> List[Path]:
        """Discover test files"""
        test_files = []
        
        # Look for test files in current directory
        for test_file in self.test_dir.glob("test_*.py"):
            test_files.append(test_file)
        
        # Look for test files in subdirectories
        for test_file in self.test_dir.glob("**/test_*.py"):
            test_files.append(test_file)
        
        logger.info(f"Discovered {len(test_files)} test files")
        return test_files
    
    def _build_test_mapping(self, test_files: List[Path]):
        """Build mapping between source functions and test functions"""
        
        for test_file in test_files:
            try:
                # Parse test file
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                # Extract test functions
                test_functions = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                        test_functions.append(node.name)
                
                # Map to source modules based on imports and function names
                imports = self._extract_imports(tree)
                
                for test_func in test_functions:
                    # Try to determine which source function this tests
                    source_func = self._infer_source_function(test_func, imports)
                    if source_func:
                        if source_func not in self.test_mapping:
                            self.test_mapping[source_func] = []
                        
                        self.test_mapping[source_func].append({
                            "test_file": str(test_file),
                            "test_function": test_func
                        })
                
            except Exception as e:
                logger.warning(f"Failed to parse test file {test_file}: {e}")
    
    def _analyze_module_coverage(self, source_file: Path) -> Optional[ModuleCoverage]:
        """Analyze coverage for a single module"""
        
        try:
            # Parse source file
            with open(source_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Extract functions and classes
            functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Skip private functions and special methods
                    if not node.name.startswith('_'):
                        functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    # Extract public methods from classes
                    for item in node.body:
                        if (isinstance(item, ast.FunctionDef) and 
                            not item.name.startswith('_')):
                            functions.append(f"{node.name}.{item.name}")
            
            if not functions:
                return None
            
            # Analyze coverage for each function
            function_coverage_list = []
            tested_count = 0
            
            for func_name in functions:
                func_coverage = self._analyze_function_coverage(
                    func_name, source_file.stem, functions
                )
                function_coverage_list.append(func_coverage)
                
                if func_coverage.is_tested:
                    tested_count += 1
            
            # Calculate module coverage
            coverage_percentage = (tested_count / len(functions) * 100) if functions else 0
            
            # Identify missing tests
            missing_tests = [f.function_name for f in function_coverage_list if not f.is_tested]
            
            module_coverage = ModuleCoverage(
                module_name=source_file.stem,
                total_functions=len(functions),
                tested_functions=tested_count,
                coverage_percentage=coverage_percentage,
                function_coverage=function_coverage_list,
                missing_tests=missing_tests
            )
            
            return module_coverage
            
        except Exception as e:
            logger.error(f"Failed to analyze module {source_file}: {e}")
            return None
    
    def _analyze_function_coverage(self, func_name: str, module_name: str, 
                                 all_functions: List[str]) -> FunctionCoverage:
        """Analyze coverage for a single function"""
        
        # Check if function has tests
        test_info = self.test_mapping.get(func_name, [])
        is_tested = len(test_info) > 0
        
        # Extract test files and functions
        test_files = [info["test_file"] for info in test_info]
        test_functions = [info["test_function"] for info in test_info]
        
        # Calculate coverage percentage based on test scenarios
        coverage_percentage = self._calculate_function_coverage_percentage(
            func_name, module_name, test_functions
        )
        
        # Identify missing test scenarios
        missing_scenarios = self._identify_missing_test_scenarios(
            func_name, module_name, test_functions
        )
        
        return FunctionCoverage(
            function_name=func_name,
            module_name=module_name,
            is_tested=is_tested,
            test_files=test_files,
            test_functions=test_functions,
            coverage_percentage=coverage_percentage,
            missing_test_scenarios=missing_scenarios
        )
    
    def _calculate_function_coverage_percentage(self, func_name: str, 
                                              module_name: str, 
                                              test_functions: List[str]) -> float:
        """Calculate coverage percentage for a function based on test scenarios"""
        
        if not test_functions:
            return 0.0
        
        # Determine component type
        component_type = self._determine_component_type(module_name)
        required_scenarios = self.required_test_scenarios.get(component_type, [])
        
        if not required_scenarios:
            # If no specific scenarios required, base on number of tests
            return min(100.0, len(test_functions) * 25.0)  # 25% per test, max 100%
        
        # Check which scenarios are covered
        covered_scenarios = 0
        for scenario in required_scenarios:
            if any(scenario.replace('_', '') in test_func.lower() for test_func in test_functions):
                covered_scenarios += 1
        
        return (covered_scenarios / len(required_scenarios)) * 100.0
    
    def _identify_missing_test_scenarios(self, func_name: str, module_name: str,
                                       test_functions: List[str]) -> List[str]:
        """Identify missing test scenarios for a function"""
        
        component_type = self._determine_component_type(module_name)
        required_scenarios = self.required_test_scenarios.get(component_type, [])
        
        missing_scenarios = []
        for scenario in required_scenarios:
            scenario_covered = any(
                scenario.replace('_', '') in test_func.lower() 
                for test_func in test_functions
            )
            
            if not scenario_covered:
                missing_scenarios.append(scenario)
        
        return missing_scenarios
    
    def _identify_coverage_gaps(self, module_coverage_list: List[ModuleCoverage]) -> List[str]:
        """Identify major coverage gaps"""
        
        gaps = []
        
        # Check for modules with low coverage
        for module in module_coverage_list:
            if module.coverage_percentage < 50:
                gaps.append(f"Low coverage in {module.module_name}: {module.coverage_percentage:.1f}%")
            
            # Check for critical functions without tests
            critical_functions = [
                "detect_model_architecture",
                "load_custom_pipeline", 
                "load_wan_pipeline",
                "process_output_tensors",
                "encode_frames_to_video"
            ]
            
            for func_coverage in module.function_coverage:
                if (any(critical in func_coverage.function_name for critical in critical_functions) and
                    not func_coverage.is_tested):
                    gaps.append(f"Critical function not tested: {module.module_name}.{func_coverage.function_name}")
        
        # Check for missing integration tests
        integration_test_files = [
            "test_end_to_end_integration.py",
            "test_pipeline_integration.py",
            "test_video_processing_integration.py"
        ]
        
        for test_file in integration_test_files:
            if not (self.test_dir / test_file).exists():
                gaps.append(f"Missing integration test file: {test_file}")
        
        return gaps
    
    def _generate_recommendations(self, report: CoverageReport) -> List[str]:
        """Generate recommendations for improving test coverage"""
        
        recommendations = []
        
        # Overall coverage recommendations
        if report.overall_coverage_percentage < 70:
            recommendations.append(
                f"Overall coverage is {report.overall_coverage_percentage:.1f}%. "
                "Target should be at least 70% for production readiness."
            )
        
        # Module-specific recommendations
        for module in report.module_coverage:
            if module.coverage_percentage < 60:
                recommendations.append(
                    f"Increase test coverage for {module.module_name} "
                    f"(currently {module.coverage_percentage:.1f}%)"
                )
            
            # Function-specific recommendations
            untested_critical = [
                f for f in module.function_coverage 
                if not f.is_tested and self._is_critical_function(f.function_name)
            ]
            
            if untested_critical:
                func_names = [f.function_name for f in untested_critical]
                recommendations.append(
                    f"Add tests for critical functions in {module.module_name}: "
                    f"{', '.join(func_names)}"
                )
        
        # Test scenario recommendations
        for module in report.module_coverage:
            for func in module.function_coverage:
                if func.missing_test_scenarios:
                    recommendations.append(
                        f"Add test scenarios for {module.module_name}.{func.function_name}: "
                        f"{', '.join(func.missing_test_scenarios)}"
                    )
        
        # Integration test recommendations
        if not any("integration" in m.module_name for m in report.module_coverage):
            recommendations.append(
                "Add comprehensive integration tests to verify component interactions"
            )
        
        return recommendations
    
    def _save_coverage_report(self, report: CoverageReport):
        """Save coverage report to file"""
        
        # Create reports directory
        reports_dir = Path("coverage_reports")
        reports_dir.mkdir(exist_ok=True)
        
        # Save detailed JSON report
        timestamp = Path(__file__).stat().st_mtime
        json_file = reports_dir / f"coverage_report_{int(timestamp)}.json"
        
        report_dict = {
            "summary": {
                "total_modules": report.total_modules,
                "total_functions": report.total_functions,
                "tested_functions": report.tested_functions,
                "overall_coverage_percentage": report.overall_coverage_percentage
            },
            "module_coverage": [
                {
                    "module_name": m.module_name,
                    "total_functions": m.total_functions,
                    "tested_functions": m.tested_functions,
                    "coverage_percentage": m.coverage_percentage,
                    "missing_tests": m.missing_tests,
                    "function_coverage": [
                        {
                            "function_name": f.function_name,
                            "is_tested": f.is_tested,
                            "test_files": f.test_files,
                            "test_functions": f.test_functions,
                            "coverage_percentage": f.coverage_percentage,
                            "missing_test_scenarios": f.missing_test_scenarios
                        }
                        for f in m.function_coverage
                    ]
                }
                for m in report.module_coverage
            ],
            "coverage_gaps": report.coverage_gaps,
            "recommendations": report.recommendations
        }
        
        with open(json_file, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        # Save human-readable report
        text_file = reports_dir / f"coverage_summary_{int(timestamp)}.txt"
        with open(text_file, 'w') as f:
            f.write("TEST COVERAGE ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Overall Coverage: {report.overall_coverage_percentage:.1f}%\n")
            f.write(f"Total Modules: {report.total_modules}\n")
            f.write(f"Total Functions: {report.total_functions}\n")
            f.write(f"Tested Functions: {report.tested_functions}\n\n")
            
            f.write("MODULE COVERAGE:\n")
            f.write("-" * 30 + "\n")
            for module in sorted(report.module_coverage, key=lambda x: x.coverage_percentage):
                f.write(f"{module.module_name}: {module.coverage_percentage:.1f}% "
                       f"({module.tested_functions}/{module.total_functions})\n")
            
            if report.coverage_gaps:
                f.write("\nCOVERAGE GAPS:\n")
                f.write("-" * 30 + "\n")
                for gap in report.coverage_gaps:
                    f.write(f"• {gap}\n")
            
            if report.recommendations:
                f.write("\nRECOMMENDATIONS:\n")
                f.write("-" * 30 + "\n")
                for i, rec in enumerate(report.recommendations, 1):
                    f.write(f"{i}. {rec}\n")
        
        logger.info(f"Coverage report saved to {json_file} and {text_file}")
    
    def _is_relevant_module(self, py_file: Path) -> bool:
        """Check if a Python file is relevant for coverage analysis"""
        
        # Skip test files, examples, and utilities
        skip_patterns = [
            "test_", "example_", "demo_", "debug_", "temp_",
            "setup.py", "__init__.py", "conftest.py"
        ]
        
        if any(py_file.name.startswith(pattern) for pattern in skip_patterns):
            return False
        
        # Check if file contains classes or functions (not just scripts)
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Look for class or function definitions
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    if not node.name.startswith('_'):
                        return True
            
            return False
            
        except Exception:
            return False
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements from AST"""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        imports.append(f"{node.module}.{alias.name}")
        
        return imports
    
    def _infer_source_function(self, test_func_name: str, imports: List[str]) -> Optional[str]:
        """Infer which source function a test function is testing"""
        
        # Remove 'test_' prefix
        if test_func_name.startswith('test_'):
            base_name = test_func_name[5:]
        else:
            return None
        
        # Try direct mapping
        if base_name in imports:
            return base_name
        
        # Try to find in imported modules
        for imp in imports:
            if base_name in imp:
                return base_name
        
        # Try common patterns
        common_mappings = {
            "detect": "detect_model_architecture",
            "load": "load_custom_pipeline", 
            "process": "process_output_tensors",
            "encode": "encode_frames_to_video"
        }
        
        for pattern, func_name in common_mappings.items():
            if pattern in base_name.lower():
                return func_name
        
        return base_name
    
    def _determine_component_type(self, module_name: str) -> str:
        """Determine component type from module name"""
        
        type_mappings = {
            "detector": "detector",
            "manager": "manager", 
            "loader": "loader",
            "handler": "handler",
            "encoder": "encoder"
        }
        
        for pattern, comp_type in type_mappings.items():
            if pattern in module_name.lower():
                return comp_type
        
        return "manager"  # Default
    
    def _is_critical_function(self, func_name: str) -> bool:
        """Check if a function is considered critical"""
        
        critical_patterns = [
            "detect", "load", "process", "encode", "generate",
            "validate", "optimize", "handle", "manage"
        ]
        
        return any(pattern in func_name.lower() for pattern in critical_patterns)


if __name__ == "__main__":
    # Run test coverage analysis
    print("Test Coverage Validator - Analyzing Coverage")
    
    validator = TestCoverageValidator()
    report = validator.analyze_coverage()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST COVERAGE ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"Overall Coverage: {report.overall_coverage_percentage:.1f}%")
    print(f"Total Modules: {report.total_modules}")
    print(f"Total Functions: {report.total_functions}")
    print(f"Tested Functions: {report.tested_functions}")
    
    # Show module coverage
    print(f"\nModule Coverage:")
    for module in sorted(report.module_coverage, key=lambda x: x.coverage_percentage, reverse=True):
        status = "✅" if module.coverage_percentage >= 70 else "⚠️" if module.coverage_percentage >= 50 else "❌"
        print(f"  {status} {module.module_name}: {module.coverage_percentage:.1f}% "
              f"({module.tested_functions}/{module.total_functions})")
    
    # Show coverage gaps
    if report.coverage_gaps:
        print(f"\nCoverage Gaps:")
        for gap in report.coverage_gaps[:5]:  # Show first 5
            print(f"  ⚠️  {gap}")
    
    # Show recommendations
    if report.recommendations:
        print(f"\nTop Recommendations:")
        for i, rec in enumerate(report.recommendations[:3], 1):  # Show first 3
            print(f"  {i}. {rec}")
    
    print(f"\nCoverage analysis completed!")