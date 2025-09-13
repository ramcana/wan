"""Testing and validation commands"""

import typer
from pathlib import Path
from typing import Optional, List
import sys

app = typer.Typer()

@app.command()
def run(
    pattern: Optional[str] = typer.Option(None, "--pattern", "-p", help="Test pattern to match"),
    fast: bool = typer.Option(False, "--fast", help="Run only fast tests"),
    coverage: bool = typer.Option(False, "--coverage", help="Generate coverage report"),
    parallel: bool = typer.Option(True, "--parallel/--sequential", help="Run tests in parallel"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Run the test suite with smart defaults"""
    
    # Import test execution engine
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from tests.utils.test_execution_engine import TestExecutionEngine
    
    engine = TestExecutionEngine()
    
    config = {
        'pattern': pattern,
        'fast_only': fast,
        'coverage': coverage,
        'parallel': parallel,
        'verbose': verbose
    }
    
    typer.echo("Running test suite...")
    success = engine.run_tests(config)
    
    if not success:
        raise typer.Exit(1)

@app.command()
def audit(
    detailed: bool = typer.Option(False, "--detailed", help="Show detailed analysis"),
    save_report: bool = typer.Option(True, "--save-report", help="Save audit report to file")
):
    """Run comprehensive test suite audit and analysis"""
    
    sys.path.append(str(Path(__file__).parent.parent.parent))
    # Add tools directory to path for imports
    tools_path = Path(__file__).parent.parent.parent / "tools"
    sys.path.insert(0, str(tools_path))
    
    # Import with correct path structure
    sys.path.insert(0, str(tools_path / "test-auditor"))
    from test_auditor import TestAuditor
    
    typer.echo("Running comprehensive test suite audit...")
    
    project_root = Path.cwd()
    auditor = TestAuditor(project_root)
    report = auditor.audit_test_suite()
    
    # Display summary
    typer.echo("\nTest Suite Health Report:")
    typer.echo("=" * 40)
    
    # Overall stats
    typer.echo(f"Total test files: {report.total_files}")
    typer.echo(f"Total tests: {report.total_tests}")
    
    if report.total_tests > 0:
        pass_rate = (report.passing_tests / report.total_tests) * 100
        typer.echo(f"Passing tests: {report.passing_tests} ({pass_rate:.1f}%)")
        typer.echo(f"Failing tests: {report.failing_tests}")
        typer.echo(f"Skipped tests: {report.skipped_tests}")
    
    # Issues summary
    if report.broken_files:
        typer.echo(f"\nBroken test files: {len(report.broken_files)}")
        for broken_file in report.broken_files[:5]:  # Show first 5
            typer.echo(f"  - {broken_file}")
        if len(report.broken_files) > 5:
            typer.echo(f"  ... and {len(report.broken_files) - 5} more")
    
    if report.critical_issues:
        typer.echo(f"\nCritical issues: {len(report.critical_issues)}")
        for issue in report.critical_issues[:3]:  # Show first 3
            typer.echo(f"  - {issue.description} ({issue.test_file})")
        if len(report.critical_issues) > 3:
            typer.echo(f"  ... and {len(report.critical_issues) - 3} more")
    
    # Performance summary
    exec_summary = report.execution_summary
    typer.echo(f"\nPerformance:")
    typer.echo(f"  Total execution time: {exec_summary['total_execution_time']:.1f}s")
    typer.echo(f"  Average per file: {exec_summary['average_execution_time']:.1f}s")
    typer.echo(f"  Files with import errors: {exec_summary['import_error_files']}")
    typer.echo(f"  Files with syntax errors: {exec_summary['syntax_error_files']}")
    
    # Recommendations
    if report.recommendations:
        typer.echo(f"\nRecommendations:")
        for i, rec in enumerate(report.recommendations[:5], 1):
            typer.echo(f"  {i}. {rec}")
        if len(report.recommendations) > 5:
            typer.echo(f"  ... and {len(report.recommendations) - 5} more")
    
    # Detailed analysis
    if detailed:
        typer.echo(f"\nDetailed Analysis:")
        typer.echo("-" * 20)
        
        for analysis in report.file_analyses[:10]:  # Show first 10 files
            rel_path = Path(analysis.file_path).relative_to(project_root)
            typer.echo(f"\n{rel_path}:")
            typer.echo(f"  Tests: {analysis.total_tests} (P:{analysis.passing_tests} F:{analysis.failing_tests} S:{analysis.skipped_tests})")
            typer.echo(f"  Execution time: {analysis.execution_time:.2f}s")
            
            if analysis.missing_imports:
                typer.echo(f"  Missing imports: {', '.join(analysis.missing_imports[:3])}")
            
            if analysis.missing_fixtures:
                typer.echo(f"  Missing fixtures: {', '.join(analysis.missing_fixtures[:3])}")
            
            if analysis.issues:
                typer.echo(f"  Issues: {len(analysis.issues)} ({', '.join(set(i.issue_type for i in analysis.issues))})")
    
    # Save report
    if save_report:
        import json
        from dataclasses import asdict
        
        report_file = project_root / 'test_audit_report.json'
        with open(report_file, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        typer.echo(f"\nDetailed report saved to: {report_file}")
    
    # Exit with error if critical issues found
    if report.critical_issues or report.broken_files:
        typer.echo("\nCritical issues found - run 'wan-cli test fix' to attempt repairs")
        raise typer.Exit(1)

@app.command()
def fix(
    imports: bool = typer.Option(True, "--imports/--no-imports", help="Fix import issues"),
    fixtures: bool = typer.Option(True, "--fixtures/--no-fixtures", help="Generate missing fixtures"),
    syntax: bool = typer.Option(False, "--syntax", help="Attempt syntax fixes (experimental)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be fixed without applying changes")
):
    """Run test infrastructure repair tools"""
    
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    typer.echo("Running test infrastructure repair...")
    
    fixes_applied = []
    
    # Fix import issues
    if imports:
        typer.echo("\n1. Fixing import issues...")
        from tests.utils.import_fixer import TestImportFixer
        
        fixer = TestImportFixer()
        
        if dry_run:
            # Just analyze without fixing
            test_files = fixer.scan_test_files()
            total_issues = 0
            
            for file_path in test_files:
                issues = fixer.analyze_imports(file_path)
                if issues:
                    rel_path = file_path.relative_to(Path.cwd())
                    typer.echo(f"  {rel_path}: {len(issues)} import issues")
                    for issue in issues[:3]:  # Show first 3
                        typer.echo(f"    - Line {issue.line_number}: {issue.suggested_fix}")
                    total_issues += len(issues)
            
            typer.echo(f"  Total import issues found: {total_issues}")
        else:
            # Actually fix imports
            results = fixer.fix_all_test_files()
            
            successful_fixes = sum(1 for r in results.values() if r.success and r.fixes_applied)
            total_fixes = sum(len(r.fixes_applied) for r in results.values())
            
            if successful_fixes > 0:
                typer.echo(f"  Fixed imports in {successful_fixes} files ({total_fixes} total fixes)")
                fixes_applied.append(f"Import fixes: {successful_fixes} files")
                
                # Show some examples
                for file_path, result in list(results.items())[:3]:
                    if result.fixes_applied:
                        rel_path = file_path.relative_to(Path.cwd())
                        typer.echo(f"    {rel_path}: {len(result.fixes_applied)} fixes")
            else:
                typer.echo("  No import issues found")
    
    # Generate missing fixtures
    if fixtures:
        typer.echo("\n2. Generating missing fixtures...")
        
        if dry_run:
            typer.echo("  [DRY RUN] Would analyze and generate missing fixtures")
        else:
            # This would integrate with fixture management system
            from tests.fixtures.fixture_manager import FixtureManager
            
            try:
                fixture_manager = FixtureManager()
                missing_fixtures = fixture_manager.find_missing_fixtures()
                
                if missing_fixtures:
                    generated = fixture_manager.generate_missing_fixtures(missing_fixtures)
                    typer.echo(f"  Generated {len(generated)} missing fixtures")
                    fixes_applied.append(f"Generated fixtures: {len(generated)}")
                else:
                    typer.echo("  No missing fixtures found")
            except Exception as e:
                typer.echo(f"  Warning: Could not analyze fixtures: {e}")
    
    # Syntax fixes (experimental)
    if syntax:
        typer.echo("\n3. Attempting syntax fixes...")
        
        if dry_run:
            typer.echo("  [DRY RUN] Would attempt to fix syntax errors")
        else:
            typer.echo("  Syntax fixing is experimental - manual review recommended")
            # This would be a more advanced feature
    
    # Summary
    if fixes_applied:
        typer.echo(f"\nRepair Summary:")
        for fix in fixes_applied:
            typer.echo(f"  - {fix}")
        typer.echo("\nTest infrastructure repair completed!")
    elif dry_run:
        typer.echo("\nDry run completed - use without --dry-run to apply fixes")
    else:
        typer.echo("\nNo repairs needed - test infrastructure is healthy!")

@app.command()
def flaky(
    runs: int = typer.Option(5, "--runs", "-r", help="Number of test runs for flaky detection"),
    threshold: float = typer.Option(0.1, "--threshold", help="Flakiness threshold (0.0-1.0)"),
    fix: bool = typer.Option(False, "--fix", help="Attempt to fix flaky tests"),
    report: bool = typer.Option(True, "--report", help="Generate flaky test report"),
    quarantine: bool = typer.Option(False, "--quarantine", help="Quarantine consistently flaky tests")
):
    """Detect and handle flaky tests using statistical analysis"""
    
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    typer.echo(f"Running flaky test detection ({runs} runs, threshold: {threshold})...")
    
    try:
        # Add tools directory to path for imports
        tools_path = Path(__file__).parent.parent.parent / "tools"
        sys.path.insert(0, str(tools_path / "test-quality"))
        
        from flaky_test_detector import (
            FlakyTestDetector, FlakyTestRunner, FlakyTestStatisticalAnalyzer,
            FlakyTestTracker, FlakyTestRecommendationEngine
        )
        
        project_root = Path.cwd()
        
        # Initialize components
        runner = FlakyTestRunner(project_root)
        analyzer = FlakyTestStatisticalAnalyzer()
        tracker = FlakyTestTracker(project_root)
        recommendation_engine = FlakyTestRecommendationEngine()
        
        # Run tests multiple times to detect flakiness
        typer.echo("Running test suite multiple times to detect flaky behavior...")
        
        all_executions = []
        for run_num in range(runs):
            typer.echo(f"  Run {run_num + 1}/{runs}...")
            executions = runner.run_test_suite_with_tracking()
            all_executions.extend(executions)
            
            # Record executions in tracker
            tracker.record_test_executions(executions)
        
        # Analyze for flaky patterns
        typer.echo("Analyzing test execution patterns...")
        flaky_patterns = analyzer.analyze_test_flakiness(all_executions)
        
        # Filter by threshold
        significant_flaky = [p for p in flaky_patterns if p.flakiness_score >= threshold]
        
        if not significant_flaky:
            typer.echo("No flaky tests detected!")
            return
        
        # Display results
        typer.echo(f"\nFlaky Test Detection Results:")
        typer.echo("=" * 40)
        typer.echo(f"Found {len(significant_flaky)} flaky tests:")
        
        for i, pattern in enumerate(significant_flaky[:10], 1):  # Show top 10
            test_name = pattern.test_id.split("::")[-1] if "::" in pattern.test_id else pattern.test_id
            typer.echo(f"\n{i}. {test_name}")
            typer.echo(f"   Flakiness Score: {pattern.flakiness_score:.2f}")
            typer.echo(f"   Failure Rate: {pattern.failure_rate:.1%}")
            typer.echo(f"   Runs: {pattern.total_runs} (P:{pattern.passed_runs} F:{pattern.failed_runs})")
            
            if pattern.common_errors:
                error_types = [f"{error}({count})" for error, count in pattern.common_errors[:2]]
                typer.echo(f"   Common Errors: {', '.join(error_types)}")
        
        if len(significant_flaky) > 10:
            typer.echo(f"\n... and {len(significant_flaky) - 10} more flaky tests")
        
        # Generate recommendations
        if fix:
            typer.echo("\nGenerating fix recommendations...")
            recommendations = recommendation_engine.generate_recommendations(significant_flaky)
            
            if recommendations:
                typer.echo(f"\nFix Recommendations:")
                typer.echo("-" * 20)
                
                for rec in recommendations[:5]:  # Show top 5 recommendations
                    test_name = rec.test_id.split("::")[-1] if "::" in rec.test_id else rec.test_id
                    typer.echo(f"\n{test_name}:")
                    typer.echo(f"  Issue: {rec.recommendation_type}")
                    typer.echo(f"  Fix: {rec.description}")
                    typer.echo(f"  Effort: {rec.implementation_effort}")
                    typer.echo(f"  Priority: {rec.priority}")
                    
                    if rec.code_examples:
                        typer.echo(f"  Example:")
                        for example in rec.code_examples[:2]:
                            typer.echo(f"    {example}")
        
        # Quarantine highly flaky tests
        if quarantine:
            highly_flaky = [p for p in significant_flaky if p.flakiness_score >= 0.5]
            
            if highly_flaky:
                typer.echo(f"\nQuarantining {len(highly_flaky)} highly flaky tests...")
                
                for pattern in highly_flaky:
                    # Update pattern as quarantined in database
                    tracker.update_flaky_pattern(pattern)
                    typer.echo(f"  Quarantined: {pattern.test_id}")
                
                typer.echo("Quarantined tests will be skipped in future runs")
        
        # Save report
        if report:
            report_file = project_root / 'flaky_test_report.json'
            
            import json
            from dataclasses import asdict
            
            report_data = {
                'detection_summary': {
                    'total_runs': runs,
                    'threshold': threshold,
                    'flaky_tests_found': len(significant_flaky),
                    'total_executions': len(all_executions)
                },
                'flaky_patterns': [asdict(p) for p in significant_flaky],
                'recommendations': [asdict(r) for r in recommendations] if fix else []
            }
            
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            typer.echo(f"\nDetailed report saved to: {report_file}")
    
    except ImportError as e:
        typer.echo(f"Error: Could not import flaky test detector: {e}")
        typer.echo("Make sure all required dependencies are installed")
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error during flaky test detection: {e}")
        raise typer.Exit(1)

@app.command()
def validate():
    """Validate test suite integrity and health"""
    
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from tests.comprehensive.test_suite_validation import TestSuiteValidator
    
    typer.echo("Validating test suite integrity...")
    
    validator = TestSuiteValidator()
    is_valid = validator.validate_test_suite()
    
    if is_valid:
        typer.echo("Test suite is valid and healthy")
    else:
        typer.echo("Test suite has issues - check the validation report")
        raise typer.Exit(1)

@app.command()
def coverage(
    html: bool = typer.Option(False, "--html", help="Generate HTML coverage report"),
    min_coverage: float = typer.Option(80.0, "--min", help="Minimum coverage threshold"),
    exclude: Optional[List[str]] = typer.Option(None, "--exclude", help="Exclude patterns from coverage")
):
    """Run coverage analysis with detailed reporting"""
    
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    typer.echo("Running coverage analysis...")
    
    try:
        # Add tools directory to path for imports
        tools_path = Path(__file__).parent.parent.parent / "tools"
        sys.path.insert(0, str(tools_path / "test-quality"))
        
        from coverage_system import CoverageAnalyzer
        
        analyzer = CoverageAnalyzer()
        
        # Configure exclusions
        if exclude:
            analyzer.set_exclusions(exclude)
        
        # Run coverage analysis
        results = analyzer.run_coverage_analysis(html_report=html)
        
        # Display results
        typer.echo(f"\nCoverage Results:")
        typer.echo("=" * 20)
        typer.echo(f"Overall Coverage: {results.overall_coverage:.1f}%")
        
        if results.overall_coverage >= min_coverage:
            typer.echo(f"Coverage meets minimum threshold ({min_coverage}%)")
        else:
            typer.echo(f"Coverage below minimum threshold ({min_coverage}%)")
        
        # Show file-level coverage
        if results.file_coverage:
            typer.echo(f"\nFile Coverage (showing files below {min_coverage}%):")
            low_coverage_files = [(f, c) for f, c in results.file_coverage.items() if c < min_coverage]
            
            for file_path, coverage in sorted(low_coverage_files, key=lambda x: x[1])[:10]:
                typer.echo(f"  {file_path}: {coverage:.1f}%")
        
        if html:
            typer.echo(f"\nHTML report generated: {results.html_report_path}")
        
        # Exit with error if coverage is too low
        if results.overall_coverage < min_coverage:
            raise typer.Exit(1)
    
    except ImportError:
        typer.echo("Coverage analysis requires 'coverage' package")
        typer.echo("Install with: pip install coverage")
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error during coverage analysis: {e}")
        raise typer.Exit(1)
