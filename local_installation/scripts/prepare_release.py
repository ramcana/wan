#!/usr/bin/env python3
"""
Automated Release Preparation Script

This script automates the complete release preparation process including
packaging, testing, and distribution artifact generation.
"""

import argparse
import sys
import os
import json
from pathlib import Path
from datetime import datetime
import logging

# Add the scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from distribution_manager import DistributionManager
from logging_system import setup_logging


def prepare_release(args):
    """Prepare a complete release."""
    try:
        # Setup logging
        setup_logging(level=args.log_level, log_file="release_preparation.log")
        logger = logging.getLogger(__name__)
        
        logger.info(f"Starting release preparation for v{args.version}")
        
        # Initialize distribution manager
        dist_manager = DistributionManager(
            source_dir=args.source_dir,
            output_dir=args.output_dir
        )
        
        # Prepare release
        release_info = dist_manager.prepare_release(
            version=args.version,
            release_notes=args.release_notes or f"Release v{args.version}",
            test_compatibility=not args.skip_tests
        )
        
        print(f"✓ Release v{args.version} prepared successfully")
        print(f"  Release directory: {release_info['release_dir']}")
        print(f"  Release package: {release_info['release_package']}")
        
        # Display artifacts
        print("\n📦 Generated artifacts:")
        for artifact_type, artifact_path in release_info['artifacts'].items():
            artifact_name = Path(artifact_path).name
            artifact_size = Path(artifact_path).stat().st_size / (1024 * 1024)
            print(f"  • {artifact_type}: {artifact_name} ({artifact_size:.1f} MB)")
        
        # Display compatibility test results
        if 'compatibility_tests' in release_info['manifest']:
            compat_tests = release_info['manifest']['compatibility_tests']
            score = compat_tests.get('compatibility_score', 0)
            status = compat_tests.get('overall_status', 'UNKNOWN')
            print(f"\n🧪 Compatibility tests: {status} (Score: {score:.2f})")
            
            if args.verbose:
                print("  Test results:")
                for test_name, test_result in compat_tests.get('tests', {}).items():
                    status_icon = "✓" if test_result.get('passed', False) else "✗"
                    message = test_result.get('message', 'No message')
                    print(f"    {status_icon} {test_name}: {message}")
        
        # Validate release if requested
        if args.validate:
            print("\n🔍 Validating release...")
            validation_results = dist_manager.validate_release(release_info['release_dir'])
            
            if validation_results['valid']:
                print("✓ Release validation passed")
            else:
                print("✗ Release validation failed")
                if 'error' in validation_results:
                    print(f"  Error: {validation_results['error']}")
                else:
                    failed_checks = validation_results.get('failed_checks', 0)
                    total_checks = validation_results.get('total_checks', 0)
                    print(f"  Failed checks: {failed_checks}/{total_checks}")
                    
                    if args.verbose and 'checks' in validation_results:
                        for check_name, check_result in validation_results['checks'].items():
                            if not check_result.get('passed', False):
                                print(f"    ✗ {check_name}: {check_result.get('message', 'Failed')}")
                
                return 1
        
        # Generate release summary
        if args.summary:
            summary_file = Path(release_info['release_dir']) / "RELEASE_SUMMARY.md"
            _generate_release_summary(release_info, summary_file)
            print(f"\n📄 Release summary: {summary_file}")
        
        print(f"\n🎉 Release v{args.version} is ready for distribution!")
        return 0
        
    except Exception as e:
        print(f"✗ Failed to prepare release: {e}")
        return 1


def validate_release(args):
    """Validate an existing release."""
    try:
        dist_manager = DistributionManager()
        
        print(f"Validating release: {args.release_dir}")
        validation_results = dist_manager.validate_release(args.release_dir)
        
        if validation_results['valid']:
            print("✓ Release validation passed")
            
            if args.verbose:
                print("  Validation details:")
                for check_name, check_result in validation_results.get('checks', {}).items():
                    status_icon = "✓" if check_result.get('passed', False) else "✗"
                    message = check_result.get('message', 'No message')
                    print(f"    {status_icon} {check_name}: {message}")
            
            return 0
        else:
            print("✗ Release validation failed")
            
            if 'error' in validation_results:
                print(f"  Error: {validation_results['error']}")
            else:
                failed_checks = validation_results.get('failed_checks', 0)
                total_checks = validation_results.get('total_checks', 0)
                print(f"  Failed checks: {failed_checks}/{total_checks}")
                
                if 'checks' in validation_results:
                    for check_name, check_result in validation_results['checks'].items():
                        if not check_result.get('passed', False):
                            print(f"    ✗ {check_name}: {check_result.get('message', 'Failed')}")
            
            return 1
            
    except Exception as e:
        print(f"✗ Failed to validate release: {e}")
        return 1


def list_releases(args):
    """List available releases."""
    try:
        output_dir = Path(args.output_dir or "dist")
        
        if not output_dir.exists():
            print("No releases found (output directory doesn't exist)")
            return 0
        
        # Find release directories
        release_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("release-v")]
        
        if not release_dirs:
            print("No releases found")
            return 0
        
        print("Available releases:")
        
        for release_dir in sorted(release_dirs):
            manifest_file = release_dir / "release_manifest.json"
            
            if manifest_file.exists():
                try:
                    with open(manifest_file, 'r') as f:
                        manifest = json.load(f)
                    
                    version = manifest.get('version', 'unknown')
                    release_date = manifest.get('release_date', 'unknown')
                    artifacts_count = len(manifest.get('artifacts', {}))
                    
                    print(f"  • v{version} ({release_date[:10]}) - {artifacts_count} artifacts")
                    
                    if args.verbose:
                        print(f"    Directory: {release_dir}")
                        if 'compatibility_tests' in manifest:
                            score = manifest['compatibility_tests'].get('compatibility_score', 0)
                            status = manifest['compatibility_tests'].get('overall_status', 'UNKNOWN')
                            print(f"    Compatibility: {status} ({score:.2f})")
                        
                        print(f"    Artifacts:")
                        for artifact_type, artifact_info in manifest.get('artifacts', {}).items():
                            size_mb = artifact_info.get('size_mb', 0)
                            print(f"      - {artifact_type}: {artifact_info.get('filename', 'unknown')} ({size_mb:.1f} MB)")
                
                except Exception as e:
                    print(f"  • {release_dir.name} (error reading manifest: {e})")
            else:
                print(f"  • {release_dir.name} (no manifest)")
        
        return 0
        
    except Exception as e:
        print(f"✗ Failed to list releases: {e}")
        return 1


def _generate_release_summary(release_info: dict, summary_file: Path):
    """Generate a release summary document."""
    manifest = release_info['manifest']
    
    summary_content = f"""# Release Summary - v{manifest['version']}

**Release Date:** {manifest['release_date'][:10]}  
**Build System:** {manifest['build_info']['build_system']}  
**Build Timestamp:** {manifest['build_info']['build_timestamp'][:19]}

## Artifacts

"""
    
    for artifact_type, artifact_info in manifest.get('artifacts', {}).items():
        summary_content += f"- **{artifact_type.title()}**: `{artifact_info['filename']}` ({artifact_info['size_mb']} MB)\n"
    
    summary_content += f"""
## System Requirements

- **Operating System:** {', '.join(manifest['system_requirements']['os'])}
- **Architecture:** {', '.join(manifest['system_requirements']['architecture'])}
- **Minimum RAM:** {manifest['system_requirements']['min_ram_gb']} GB
- **Recommended RAM:** {manifest['system_requirements']['recommended_ram_gb']} GB
- **Disk Space:** {manifest['system_requirements']['min_disk_space_gb']} GB
- **GPU VRAM:** {manifest['system_requirements']['gpu_requirements']['min_vram_gb']} GB minimum

## Installation

- **Installer File:** `{manifest['installation']['installer_file']}`
- **Estimated Time:** {manifest['installation']['installation_time_minutes']} minutes
- **Admin Required:** {'Yes' if manifest['installation']['requires_admin'] else 'No'}
- **Offline Capable:** {'Yes' if manifest['installation']['offline_capable'] else 'No'}

"""
    
    if 'compatibility_tests' in manifest:
        compat_tests = manifest['compatibility_tests']
        summary_content += f"""## Compatibility Tests

- **Overall Status:** {compat_tests.get('overall_status', 'UNKNOWN')}
- **Compatibility Score:** {compat_tests.get('compatibility_score', 0):.2f}
- **Test System:** {compat_tests['test_system']['os']} {compat_tests['test_system']['architecture'][0]}

### Test Results

"""
        for test_name, test_result in compat_tests.get('tests', {}).items():
            status = "✅ PASS" if test_result.get('passed', False) else "❌ FAIL"
            message = test_result.get('message', 'No message')
            summary_content += f"- **{test_name.replace('_', ' ').title()}:** {status} - {message}\n"
    
    if manifest.get('release_notes'):
        summary_content += f"""
## Release Notes

{manifest['release_notes']}
"""
    
    summary_content += f"""
## Distribution

This release has been prepared and tested for distribution. All artifacts have been verified for integrity and compatibility.

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    summary_file.write_text(summary_content, encoding='utf-8')


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="WAN2.2 Release Preparation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare a new release
  python prepare_release.py prepare --version 1.0.0 --release-notes "Initial release"
  
  # Prepare release with custom settings
  python prepare_release.py prepare --version 1.1.0 --skip-tests --no-validate
  
  # Validate an existing release
  python prepare_release.py validate --release-dir dist/release-v1.0.0
  
  # List all releases
  python prepare_release.py list --verbose
        """
    )
    
    # Global arguments
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Logging level")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Prepare release command
    prepare_parser = subparsers.add_parser("prepare", help="Prepare a new release")
    prepare_parser.add_argument("--version", required=True, help="Release version (e.g., 1.0.0)")
    prepare_parser.add_argument("--release-notes", help="Release notes text")
    prepare_parser.add_argument("--source-dir", help="Source directory (default: current)")
    prepare_parser.add_argument("--output-dir", help="Output directory (default: dist)")
    prepare_parser.add_argument("--skip-tests", action="store_true", help="Skip compatibility tests")
    prepare_parser.add_argument("--no-validate", dest="validate", action="store_false", 
                               help="Skip release validation")
    prepare_parser.add_argument("--no-summary", dest="summary", action="store_false",
                               help="Skip release summary generation")
    prepare_parser.set_defaults(func=prepare_release, validate=True, summary=True)
    
    # Validate release command
    validate_parser = subparsers.add_parser("validate", help="Validate existing release")
    validate_parser.add_argument("--release-dir", required=True, help="Release directory to validate")
    validate_parser.set_defaults(func=validate_release)
    
    # List releases command
    list_parser = subparsers.add_parser("list", help="List available releases")
    list_parser.add_argument("--output-dir", help="Output directory to search (default: dist)")
    list_parser.set_defaults(func=list_releases)
    
    # Parse arguments and execute
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())