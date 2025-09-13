#!/usr/bin/env python3
"""
WAN Model Deployment Validation Script

Standalone validation utility for WAN model deployments with comprehensive
pre and post-deployment checks.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.deployment import ValidationService, DeploymentConfig


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/validation.log')
        ]
    )


async def validate_pre_deployment(args):
    """Run pre-deployment validation"""
    print(f"🔍 Running pre-deployment validation...")
    print(f"Models to validate: {', '.join(args.models)}")
    
    config = DeploymentConfig(
        source_models_path=args.source_path,
        target_models_path=args.target_path,
        backup_path=args.backup_path
    )
    
    validation_service = ValidationService(config)
    
    try:
        result = await validation_service.validate_pre_deployment(args.models)
        
        # Print validation results
        print(f"\n📊 Pre-Deployment Validation Results:")
        print(f"Overall Status: {'✅ PASSED' if result.is_valid else '❌ FAILED'}")
        print(f"Validation Type: {result.validation_type}")
        print(f"Timestamp: {result.timestamp}")
        
        if result.checks_performed:
            print(f"\n🔍 Checks Performed ({len(result.checks_performed)}):")
            for check in result.checks_performed:
                status = "✅" if check in result.passed_checks else "❌" if check in result.failed_checks else "⏳"
                print(f"  {status} {check}")
        
        if result.warnings:
            print(f"\n⚠️  Warnings ({len(result.warnings)}):")
            for warning in result.warnings:
                print(f"  • {warning}")
        
        if result.errors:
            print(f"\n❌ Errors ({len(result.errors)}):")
            for error in result.errors:
                print(f"  • {error}")
        
        if result.performance_metrics:
            print(f"\n📈 Performance Metrics:")
            for metric, value in result.performance_metrics.items():
                if isinstance(value, float):
                    print(f"  • {metric}: {value:.2f}")
                else:
                    print(f"  • {metric}: {value}")
        
        # Save validation report
        if args.output_report:
            report_data = {
                "validation_type": result.validation_type,
                "is_valid": result.is_valid,
                "model_name": result.model_name,
                "timestamp": result.timestamp.isoformat(),
                "checks_performed": result.checks_performed,
                "passed_checks": result.passed_checks,
                "failed_checks": result.failed_checks,
                "warnings": result.warnings,
                "errors": result.errors,
                "performance_metrics": result.performance_metrics
            }
            
            with open(args.output_report, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            print(f"\n📄 Validation report saved to: {args.output_report}")
        
        return 0 if result.is_valid else 1
        
    except Exception as e:
        print(f"❌ Validation failed with exception: {str(e)}")
        logging.error(f"Validation exception: {str(e)}", exc_info=True)
        return 1


async def validate_post_deployment(args):
    """Run post-deployment validation"""
    print(f"🔍 Running post-deployment validation...")
    print(f"Models to validate: {', '.join(args.models)}")
    
    config = DeploymentConfig(
        source_models_path=args.source_path,
        target_models_path=args.target_path,
        backup_path=args.backup_path
    )
    
    validation_service = ValidationService(config)
    
    try:
        result = await validation_service.validate_post_deployment(args.models)
        
        # Print validation results
        print(f"\n📊 Post-Deployment Validation Results:")
        print(f"Overall Status: {'✅ PASSED' if result.is_valid else '❌ FAILED'}")
        print(f"Validation Type: {result.validation_type}")
        print(f"Timestamp: {result.timestamp}")
        
        if result.checks_performed:
            print(f"\n🔍 Checks Performed ({len(result.checks_performed)}):")
            for check in result.checks_performed:
                status = "✅" if check in result.passed_checks else "❌" if check in result.failed_checks else "⏳"
                print(f"  {status} {check}")
        
        if result.warnings:
            print(f"\n⚠️  Warnings ({len(result.warnings)}):")
            for warning in result.warnings:
                print(f"  • {warning}")
        
        if result.errors:
            print(f"\n❌ Errors ({len(result.errors)}):")
            for error in result.errors:
                print(f"  • {error}")
        
        if result.performance_metrics:
            print(f"\n📈 Performance Metrics:")
            for metric, value in result.performance_metrics.items():
                if isinstance(value, (int, float)):
                    if isinstance(value, float):
                        print(f"  • {metric}: {value:.2f}")
                    else:
                        print(f"  • {metric}: {value}")
                else:
                    print(f"  • {metric}: {value}")
        
        # Save validation report
        if args.output_report:
            report_data = {
                "validation_type": result.validation_type,
                "is_valid": result.is_valid,
                "model_name": result.model_name,
                "timestamp": result.timestamp.isoformat(),
                "checks_performed": result.checks_performed,
                "passed_checks": result.passed_checks,
                "failed_checks": result.failed_checks,
                "warnings": result.warnings,
                "errors": result.errors,
                "performance_metrics": result.performance_metrics
            }
            
            with open(args.output_report, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            print(f"\n📄 Validation report saved to: {args.output_report}")
        
        return 0 if result.is_valid else 1
        
    except Exception as e:
        print(f"❌ Validation failed with exception: {str(e)}")
        logging.error(f"Validation exception: {str(e)}", exc_info=True)
        return 1


async def validate_model_health(args):
    """Run model health validation"""
    print(f"🏥 Running health validation for model: {args.model}")
    
    config = DeploymentConfig(
        source_models_path=args.source_path,
        target_models_path=args.target_path,
        backup_path=args.backup_path
    )
    
    validation_service = ValidationService(config)
    
    try:
        result = await validation_service.validate_model_health(args.model)
        
        # Print validation results
        print(f"\n📊 Model Health Validation Results:")
        print(f"Model: {result.model_name}")
        print(f"Overall Status: {'✅ HEALTHY' if result.is_valid else '❌ UNHEALTHY'}")
        print(f"Timestamp: {result.timestamp}")
        
        if result.checks_performed:
            print(f"\n🔍 Health Checks ({len(result.checks_performed)}):")
            for check in result.checks_performed:
                status = "✅" if check in result.passed_checks else "❌" if check in result.failed_checks else "⏳"
                print(f"  {status} {check}")
        
        if result.warnings:
            print(f"\n⚠️  Warnings ({len(result.warnings)}):")
            for warning in result.warnings:
                print(f"  • {warning}")
        
        if result.errors:
            print(f"\n❌ Errors ({len(result.errors)}):")
            for error in result.errors:
                print(f"  • {error}")
        
        if result.performance_metrics:
            print(f"\n📈 Performance Metrics:")
            for metric, value in result.performance_metrics.items():
                if isinstance(value, (int, float)):
                    if isinstance(value, float):
                        print(f"  • {metric}: {value:.2f}")
                    else:
                        print(f"  • {metric}: {value}")
                else:
                    print(f"  • {metric}: {value}")
        
        # Save validation report
        if args.output_report:
            report_data = {
                "validation_type": result.validation_type,
                "is_valid": result.is_valid,
                "model_name": result.model_name,
                "timestamp": result.timestamp.isoformat(),
                "checks_performed": result.checks_performed,
                "passed_checks": result.passed_checks,
                "failed_checks": result.failed_checks,
                "warnings": result.warnings,
                "errors": result.errors,
                "performance_metrics": result.performance_metrics
            }
            
            with open(args.output_report, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            print(f"\n📄 Health report saved to: {args.output_report}")
        
        return 0 if result.is_valid else 1
        
    except Exception as e:
        print(f"❌ Health validation failed with exception: {str(e)}")
        logging.error(f"Health validation exception: {str(e)}", exc_info=True)
        return 1


async def export_validation_history(args):
    """Export validation history"""
    print(f"📄 Exporting validation history...")
    
    config = DeploymentConfig(
        source_models_path=args.source_path,
        target_models_path=args.target_path,
        backup_path=args.backup_path
    )
    
    validation_service = ValidationService(config)
    
    try:
        await validation_service.export_validation_report(args.output_file)
        print(f"✅ Validation history exported to: {args.output_file}")
        return 0
        
    except Exception as e:
        print(f"❌ Export failed with exception: {str(e)}")
        logging.error(f"Export exception: {str(e)}", exc_info=True)
        return 1


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="WAN Model Deployment Validation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pre-deployment validation
  python validate_wan_deployment.py pre --models t2v-A14B i2v-A14B
  
  # Post-deployment validation
  python validate_wan_deployment.py post --models t2v-A14B --output-report validation_report.json
  
  # Health check for specific model
  python validate_wan_deployment.py health --model ti2v-5B
  
  # Export validation history
  python validate_wan_deployment.py export --output-file validation_history.json
        """
    )
    
    # Global arguments
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--source-path', default='models', help='Source models directory')
    parser.add_argument('--target-path', default='models', help='Target models directory')
    parser.add_argument('--backup-path', default='backups/models', help='Backup directory')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Pre-deployment validation
    pre_parser = subparsers.add_parser('pre', help='Run pre-deployment validation')
    pre_parser.add_argument('--models', nargs='+', required=True, 
                           help='Model names to validate')
    pre_parser.add_argument('--output-report', help='Output validation report file')
    
    # Post-deployment validation
    post_parser = subparsers.add_parser('post', help='Run post-deployment validation')
    post_parser.add_argument('--models', nargs='+', required=True,
                            help='Model names to validate')
    post_parser.add_argument('--output-report', help='Output validation report file')
    
    # Health validation
    health_parser = subparsers.add_parser('health', help='Run model health validation')
    health_parser.add_argument('--model', required=True, help='Model name to check')
    health_parser.add_argument('--output-report', help='Output health report file')
    
    # Export history
    export_parser = subparsers.add_parser('export', help='Export validation history')
    export_parser.add_argument('--output-file', required=True, help='Output file for validation history')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Ensure logs directory exists
    Path('logs').mkdir(exist_ok=True)
    
    # Run the appropriate command
    try:
        if args.command == 'pre':
            return asyncio.run(validate_pre_deployment(args))
        elif args.command == 'post':
            return asyncio.run(validate_post_deployment(args))
        elif args.command == 'health':
            return asyncio.run(validate_model_health(args))
        elif args.command == 'export':
            return asyncio.run(export_validation_history(args))
        else:
            print(f"Unknown command: {args.command}")
            return 1
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
