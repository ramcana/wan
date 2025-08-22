#!/usr/bin/env python3
"""
Command-line interface for the Local Testing Framework
"""

import argparse
import sys
import json
import os
from typing import List, Optional
from datetime import datetime

from ..test_manager import LocalTestManager
from ..models.test_results import TestStatus


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    import logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def validate_env_command(args) -> int:
    """Handle environment validation command"""
    try:
        print("🔍 Running environment validation...")
        manager = LocalTestManager(args.config)
        
        results = manager.run_environment_validation()
        
        # Print results
        print("\nEnvironment Validation Results:")
        print("=" * 50)
        
        validation_checks = [
            ("Python Version", results.python_version),
            ("CUDA Availability", results.cuda_availability),
            ("Dependencies", results.dependencies),
            ("Configuration", results.configuration),
            ("Environment Variables", results.environment_variables)
        ]
        
        for name, result in validation_checks:
            status_icon = "✅" if result.status.value == "passed" else "❌"
            print(f"{status_icon} {name}: {result.message}")
            
            if result.status.value != "passed" and result.remediation_steps:
                print(f"   Remediation steps:")
                for step in result.remediation_steps:
                    print(f"   - {step}")
        
        print(f"\nOverall Status: {results.overall_status.value.upper()}")
        
        if results.remediation_steps:
            print("\nGeneral Remediation Steps:")
            for step in results.remediation_steps:
                print(f"- {step}")
        
        if args.report:
            report_path = f"environment_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(results.to_dict(), f, indent=2)
            print(f"\n📄 Report saved to: {report_path}")
        
        return 0 if results.overall_status.value == "passed" else 1
        
    except Exception as e:
        print(f"❌ Error during environment validation: {str(e)}")
        return 1


def test_performance_command(args) -> int:
    """Handle performance testing command"""
    try:
        print("🚀 Running performance tests...")
        manager = LocalTestManager(args.config)
        
        results = manager.run_performance_tests()
        
        # Print results
        print("\nPerformance Test Results:")
        print("=" * 50)
        
        if results.benchmark_720p:
            benchmark = results.benchmark_720p
            status_icon = "✅" if benchmark.meets_target else "❌"
            print(f"{status_icon} 720p Generation: {benchmark.generation_time:.1f} min "
                  f"(Target: <{benchmark.target_time:.1f} min)")
            print(f"   VRAM Usage: {benchmark.vram_usage:.1f} GB")
            print(f"   CPU Usage: {benchmark.cpu_usage:.1f}%")
        
        if results.benchmark_1080p:
            benchmark = results.benchmark_1080p
            status_icon = "✅" if benchmark.meets_target else "❌"
            print(f"{status_icon} 1080p Generation: {benchmark.generation_time:.1f} min "
                  f"(Target: <{benchmark.target_time:.1f} min)")
            print(f"   VRAM Usage: {benchmark.vram_usage:.1f} GB")
            print(f"   CPU Usage: {benchmark.cpu_usage:.1f}%")
        
        if results.vram_optimization:
            opt = results.vram_optimization
            status_icon = "✅" if opt.meets_target else "❌"
            print(f"{status_icon} VRAM Optimization: {opt.reduction_percent:.1f}% reduction "
                  f"(Target: {opt.target_reduction_percent:.1f}%)")
            print(f"   Baseline: {opt.baseline_vram_mb} MB → Optimized: {opt.optimized_vram_mb} MB")
        
        print(f"\nOverall Status: {results.overall_status.value.upper()}")
        
        if results.recommendations:
            print("\nRecommendations:")
            for rec in results.recommendations:
                print(f"- {rec}")
        
        if args.benchmark:
            print("\n📊 Detailed benchmark data available in performance logs")
        
        return 0 if results.overall_status == TestStatus.PASSED else 1
        
    except Exception as e:
        print(f"❌ Error during performance testing: {str(e)}")
        return 1


def test_integration_command(args) -> int:
    """Handle integration testing command"""
    try:
        print("🔗 Running integration tests...")
        manager = LocalTestManager(args.config)
        
        results = manager.run_integration_tests()
        
        # Print results
        print("\nIntegration Test Results:")
        print("=" * 50)
        
        status_icon = "✅" if results.overall_status == TestStatus.PASSED else "❌"
        print(f"{status_icon} Overall Status: {results.overall_status.value.upper()}")
        
        if results.error_handling_result:
            error_status = "✅" if results.error_handling_result.status.value == "passed" else "❌"
            print(f"{error_status} Error Handling: {results.error_handling_result.message}")
        
        if results.ui_results:
            ui_status = "✅" if results.ui_results.overall_status == TestStatus.PASSED else "❌"
            print(f"{ui_status} UI Tests: {results.ui_results.overall_status.value}")
            
            if results.ui_results.component_test_results:
                for component_result in results.ui_results.component_test_results:
                    comp_status = "✅" if component_result.status.value == "passed" else "❌"
                    print(f"   {comp_status} {component_result.component}: {component_result.message}")
        
        if results.api_results:
            api_status = "✅" if results.api_results.overall_status == TestStatus.PASSED else "❌"
            print(f"{api_status} API Tests: {results.api_results.overall_status.value}")
            
            if results.api_results.endpoint_test_results:
                for endpoint_result in results.api_results.endpoint_test_results:
                    end_status = "✅" if endpoint_result.status.value == "passed" else "❌"
                    print(f"   {end_status} {endpoint_result.component}: {endpoint_result.message}")
        
        if results.resource_monitoring_result:
            res_status = "✅" if results.resource_monitoring_result.status.value == "passed" else "❌"
            print(f"{res_status} Resource Monitoring: {results.resource_monitoring_result.message}")
        
        duration = (results.end_time - results.start_time).total_seconds()
        print(f"\nTest Duration: {duration:.1f} seconds")
        
        return 0 if results.overall_status == TestStatus.PASSED else 1
        
    except Exception as e:
        print(f"❌ Error during integration testing: {str(e)}")
        return 1


def diagnose_command(args) -> int:
    """Handle diagnostic command"""
    try:
        print("🔧 Running system diagnostics...")
        manager = LocalTestManager(args.config)
        
        results = manager.run_diagnostics()
        
        # Print results
        print("\nDiagnostic Results:")
        print("=" * 50)
        
        if "system_status" in results:
            status = results["system_status"]
            status_icon = "✅" if status == "healthy" else "⚠️"
            print(f"{status_icon} System Status: {status}")
        
        if "issues" in results:
            issues = results["issues"]
            if issues:
                print(f"\n⚠️  Found {len(issues)} issues:")
                for i, issue in enumerate(issues, 1):
                    print(f"{i}. {issue.get('description', 'Unknown issue')}")
                    if 'recommendations' in issue:
                        for rec in issue['recommendations']:
                            print(f"   → {rec}")
            else:
                print("✅ No issues detected")
        
        if "resource_usage" in results:
            resources = results["resource_usage"]
            print(f"\n📊 Resource Usage:")
            print(f"   CPU: {resources.get('cpu_percent', 0):.1f}%")
            print(f"   Memory: {resources.get('memory_percent', 0):.1f}%")
            print(f"   VRAM: {resources.get('vram_percent', 0):.1f}%")
        
        return 0 if results.get("system_status") == "healthy" else 1
        
    except Exception as e:
        print(f"❌ Error during diagnostics: {str(e)}")
        return 1


def generate_samples_command(args) -> int:
    """Handle sample generation command"""
    try:
        print("📝 Generating sample data and configurations...")
        manager = LocalTestManager(args.config)
        
        sample_types = []
        if args.config_samples:
            sample_types.append("config")
        if args.data_samples:
            sample_types.append("data")
        if args.env_samples:
            sample_types.append("env")
        if args.all_samples or not sample_types:
            sample_types = ["config", "data", "env"]
        
        results = manager.generate_samples(sample_types)
        
        # Print results
        print("\nSample Generation Results:")
        print("=" * 50)
        
        for sample_type, result in results.items():
            if result:
                print(f"✅ {sample_type.title()} samples generated successfully")
                if isinstance(result, dict) and "path" in result:
                    print(f"   Saved to: {result['path']}")
                elif isinstance(result, list):
                    print(f"   Generated {len(result)} sample(s)")
            else:
                print(f"❌ Failed to generate {sample_type} samples")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error generating samples: {str(e)}")
        return 1


def run_all_command(args) -> int:
    """Handle run-all command"""
    try:
        print("🚀 Running complete test suite...")
        manager = LocalTestManager(args.config)
        
        results = manager.run_full_test_suite()
        
        # Print summary
        print("\nComplete Test Suite Results:")
        print("=" * 50)
        
        if results.environment_results:
            env_status = "✅" if results.environment_results.overall_status.value == "passed" else "❌"
            print(f"{env_status} Environment Validation: {results.environment_results.overall_status.value}")
        
        if results.performance_results:
            perf_status = "✅" if results.performance_results.overall_status == TestStatus.PASSED else "❌"
            print(f"{perf_status} Performance Tests: {results.performance_results.overall_status.value}")
        
        if results.integration_results:
            int_status = "✅" if results.integration_results.overall_status == TestStatus.PASSED else "❌"
            print(f"{int_status} Integration Tests: {results.integration_results.overall_status.value}")
        
        print(f"\n🎯 Overall Status: {results.overall_status.value.upper()}")
        
        if results.recommendations:
            print("\n💡 Recommendations:")
            for rec in results.recommendations:
                print(f"- {rec}")
        
        # Generate report
        if args.report_format:
            try:
                report_content = manager.generate_reports(results, args.report_format)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                report_filename = f"test_report_{timestamp}.{args.report_format}"
                
                with open(report_filename, 'w') as f:
                    f.write(report_content)
                
                print(f"\n📄 {args.report_format.upper()} report saved to: {report_filename}")
            except Exception as e:
                print(f"⚠️  Warning: Could not generate report: {str(e)}")
        
        duration = (results.end_time - results.start_time).total_seconds() if results.end_time else 0
        print(f"\nTotal Duration: {duration:.1f} seconds")
        
        return 0 if results.overall_status == TestStatus.PASSED else 1
        
    except Exception as e:
        print(f"❌ Error during full test suite: {str(e)}")
        return 1


def monitor_command(args) -> int:
    """Handle monitoring command"""
    try:
        print(f"📊 Starting continuous monitoring for {args.duration} seconds...")
        manager = LocalTestManager(args.config)
        
        monitor_id = manager.start_monitoring(args.duration, args.alerts)
        
        print(f"✅ Monitoring started with ID: {monitor_id}")
        print("   Use Ctrl+C to stop monitoring early")
        
        # Note: In a real implementation, this would handle the monitoring loop
        # For now, we just indicate that monitoring has started
        print(f"📈 Monitoring will run for {args.duration} seconds...")
        print("   Check logs for real-time updates")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error starting monitoring: {str(e)}")
        return 1


def validate_production_command(args) -> int:
    """Handle production readiness validation command"""
    try:
        print("🏭 Running production readiness validation...")
        manager = LocalTestManager(args.config)
        
        results = manager.run_production_validation()
        
        # Print results summary
        print("\nProduction Readiness Validation Results:")
        print("=" * 60)
        
        # Consistency validation
        if results.consistency_validation:
            status_icon = "✅" if results.consistency_validation.overall_status.value == "passed" else "❌"
            print(f"{status_icon} Performance Consistency: {results.consistency_validation.overall_status.value}")
            print(f"   Runs completed: {results.consistency_validation.runs_completed}/{results.consistency_validation.target_runs}")
            if results.consistency_validation.consistency_metrics:
                score = results.consistency_validation.consistency_metrics.get("consistency_score", 0)
                print(f"   Consistency score: {score:.1f}/100")
        
        # Configuration validation
        if results.configuration_validation:
            status_icon = "✅" if results.configuration_validation.overall_status.value == "passed" else "❌"
            print(f"{status_icon} Configuration: {results.configuration_validation.overall_status.value}")
        
        # Security validation
        if results.security_validation:
            status_icon = "✅" if results.security_validation.overall_status.value == "passed" else "❌"
            print(f"{status_icon} Security Compliance: {results.security_validation.overall_status.value}")
        
        # Scalability validation
        if results.scalability_validation:
            status_icon = "✅" if results.scalability_validation.overall_status.value == "passed" else "❌"
            print(f"{status_icon} Scalability: {results.scalability_validation.overall_status.value}")
            print(f"   Concurrent users tested: {results.scalability_validation.concurrent_users_tested}")
            print(f"   Load test duration: {results.scalability_validation.load_test_duration_minutes} minutes")
        
        # Overall status
        print(f"\n🎯 Overall Status: {results.overall_status.value.upper()}")
        
        # Certificate information
        if results.certificate:
            print(f"\n📜 Production Certificate Generated:")
            print(f"   Certificate ID: {results.certificate.certificate_id}")
            print(f"   Valid until: {results.certificate.valid_until.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Security compliance: {'✅' if results.certificate.security_compliance else '❌'}")
            print(f"   Scalability validated: {'✅' if results.certificate.scalability_validated else '❌'}")
        
        # Recommendations
        if results.recommendations:
            print(f"\n💡 Recommendations:")
            for rec in results.recommendations:
                print(f"   • {rec}")
        
        # Save detailed results if requested
        if args.report:
            output_file = f"production_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(results.to_dict(), f, indent=2)
            print(f"\n📄 Detailed report saved to: {output_file}")
        
        # Return appropriate exit code
        if results.overall_status == TestStatus.PASSED:
            return 0
        elif results.overall_status == TestStatus.PARTIAL:
            return 2  # Warning exit code
        else:
            return 1  # Error exit code
        
    except Exception as e:
        print(f"❌ Error during production validation: {e}")
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser"""
    parser = argparse.ArgumentParser(
        prog='local-testing-framework',
        description='Local Testing Framework for Wan2.2 UI Variant',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s validate-env --report
  %(prog)s test-performance --resolution 720p --benchmark
  %(prog)s test-integration --ui --api
  %(prog)s diagnose --cuda --memory
  %(prog)s generate-samples --config --data
  %(prog)s run-all --report-format html
  %(prog)s monitor --duration 3600 --alerts
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        default='config.json',
        help='Path to configuration file (default: config.json)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Environment validation command
    env_parser = subparsers.add_parser(
        'validate-env',
        help='Validate environment setup and prerequisites'
    )
    env_parser.add_argument(
        '--report',
        action='store_true',
        help='Generate detailed validation report'
    )
    env_parser.set_defaults(func=validate_env_command)
    
    # Performance testing command
    perf_parser = subparsers.add_parser(
        'test-performance',
        help='Run performance benchmarks and optimization tests'
    )
    perf_parser.add_argument(
        '--resolution',
        choices=['720p', '1080p', 'both'],
        default='both',
        help='Resolution to test (default: both)'
    )
    perf_parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run detailed benchmark analysis'
    )
    perf_parser.set_defaults(func=test_performance_command)
    
    # Integration testing command
    int_parser = subparsers.add_parser(
        'test-integration',
        help='Run integration tests and component validation'
    )
    int_parser.add_argument(
        '--ui',
        action='store_true',
        help='Include UI testing'
    )
    int_parser.add_argument(
        '--api',
        action='store_true',
        help='Include API testing'
    )
    int_parser.add_argument(
        '--full',
        action='store_true',
        help='Run all integration tests'
    )
    int_parser.set_defaults(func=test_integration_command)
    
    # Diagnostic command
    diag_parser = subparsers.add_parser(
        'diagnose',
        help='Run system diagnostics and troubleshooting'
    )
    diag_parser.add_argument(
        '--system',
        action='store_true',
        help='Include system diagnostics'
    )
    diag_parser.add_argument(
        '--cuda',
        action='store_true',
        help='Include CUDA diagnostics'
    )
    diag_parser.add_argument(
        '--memory',
        action='store_true',
        help='Include memory diagnostics'
    )
    diag_parser.set_defaults(func=diagnose_command)
    
    # Sample generation command
    sample_parser = subparsers.add_parser(
        'generate-samples',
        help='Generate sample data and configuration files'
    )
    sample_parser.add_argument(
        '--config',
        dest='config_samples',
        action='store_true',
        help='Generate configuration samples'
    )
    sample_parser.add_argument(
        '--data',
        dest='data_samples',
        action='store_true',
        help='Generate test data samples'
    )
    sample_parser.add_argument(
        '--env',
        dest='env_samples',
        action='store_true',
        help='Generate environment file samples'
    )
    sample_parser.add_argument(
        '--all',
        dest='all_samples',
        action='store_true',
        help='Generate all sample types'
    )
    sample_parser.set_defaults(func=generate_samples_command)
    
    # Run all command
    all_parser = subparsers.add_parser(
        'run-all',
        help='Run complete test suite'
    )
    all_parser.add_argument(
        '--report-format',
        choices=['html', 'json'],
        help='Generate report in specified format'
    )
    all_parser.set_defaults(func=run_all_command)
    
    # Monitor command
    monitor_parser = subparsers.add_parser(
        'monitor',
        help='Start continuous monitoring'
    )
    monitor_parser.add_argument(
        '--duration',
        type=int,
        default=3600,
        help='Monitoring duration in seconds (default: 3600)'
    )
    monitor_parser.add_argument(
        '--alerts',
        action='store_true',
        help='Enable threshold alerts'
    )
    monitor_parser.set_defaults(func=monitor_command)
    
    # Production validation command
    prod_parser = subparsers.add_parser(
        'validate-production',
        help='Validate production readiness'
    )
    prod_parser.add_argument(
        '--report',
        action='store_true',
        help='Generate detailed validation report'
    )
    prod_parser.set_defaults(func=validate_production_command)
    
    return parser


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Handle no command case
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        return 130
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())