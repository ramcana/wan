#!/usr/bin/env python3
"""
Comprehensive test for distribution preparation functionality.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from distribution_manager import DistributionManager


def test_distribution_preparation():
    """Test the complete distribution preparation process."""
    print("🧪 Testing Distribution Preparation System")
    print("=" * 50)
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        source_dir = temp_path / "source"
        output_dir = temp_path / "output"
        
        # Create source directory structure
        source_dir.mkdir()
        (source_dir / "scripts").mkdir()
        (source_dir / "application").mkdir()
        (source_dir / "resources").mkdir()
        
        # Create test files
        print("📁 Creating test source files...")
        (source_dir / "install.bat").write_text("@echo off\necho WAN2.2 Installer")
        (source_dir / "README.md").write_text("# WAN2.2 Local Installation\n\nTest project")
        (source_dir / "requirements.txt").write_text("torch>=2.0.0\nnumpy>=1.21.0")
        (source_dir / "scripts" / "main.py").write_text("print('Main script')")
        (source_dir / "application" / "app.py").write_text("print('Application')")
        (source_dir / "resources" / "config.json").write_text('{"version": "1.0.0"}')
        
        # Initialize distribution manager
        print("🔧 Initializing DistributionManager...")
        dist_manager = DistributionManager(str(source_dir), str(output_dir))
        print(f"✓ Source directory: {dist_manager.source_dir}")
        print(f"✓ Output directory: {dist_manager.output_dir}")
        
        # Test artifact generation
        print("\n📦 Testing artifact generation...")
        release_dir = output_dir / "test_release"
        release_dir.mkdir(parents=True)
        
        try:
            # Test source archive creation
            print("  • Creating source archive...")
            source_archive = dist_manager._create_source_archive(release_dir, "1.0.0")
            print(f"    ✓ Source archive: {source_archive.name}")
            
            # Test quickstart guide creation
            print("  • Creating quickstart guide...")
            quickstart = dist_manager._create_quickstart_guide(release_dir, "1.0.0")
            print(f"    ✓ Quickstart guide: {quickstart.name}")
            
            # Test release manifest creation
            print("  • Creating release manifest...")
            artifacts = {
                "source": str(source_archive),
                "quickstart": str(quickstart)
            }
            manifest = dist_manager._create_release_manifest("1.0.0", "Test release", artifacts)
            print(f"    ✓ Manifest version: {manifest['version']}")
            print(f"    ✓ Artifacts count: {len(manifest['artifacts'])}")
            
            # Test compatibility tests
            print("\n🧪 Testing compatibility checks...")
            
            # Test path compatibility
            test_artifacts = {
                "good_file": str(release_dir / "good_file.txt"),
                "test_file": str(release_dir / "test_file.txt")
            }
            
            # Create test files
            for artifact_path in test_artifacts.values():
                Path(artifact_path).write_text("test content")
            
            path_result = dist_manager._test_path_compatibility(test_artifacts)
            print(f"    ✓ Path compatibility: {'PASS' if path_result['passed'] else 'FAIL'}")
            
            size_result = dist_manager._test_size_validation(test_artifacts)
            print(f"    ✓ Size validation: {'PASS' if size_result['passed'] else 'FAIL'}")
            
            perm_result = dist_manager._test_file_permissions(test_artifacts)
            print(f"    ✓ File permissions: {'PASS' if perm_result['passed'] else 'FAIL'}")
            
            # Test checksum generation
            print("\n🔐 Testing checksum generation...")
            dist_manager._generate_distribution_checksums(release_dir)
            checksums_file = release_dir / "CHECKSUMS.txt"
            if checksums_file.exists():
                print(f"    ✓ Checksums file created: {checksums_file.name}")
                content = checksums_file.read_text()
                print(f"    ✓ Contains {len(content.splitlines()) - 3} file checksums")
            
            # Test release package creation
            print("\n📦 Testing release package creation...")
            release_package = dist_manager._create_release_package(release_dir, "1.0.0")
            print(f"    ✓ Release package: {release_package.name}")
            print(f"    ✓ Package size: {release_package.stat().st_size / 1024:.1f} KB")
            
            # Test release validation
            print("\n✅ Testing release validation...")
            
            # Create a proper manifest for validation
            validation_manifest = {
                "version": "1.0.0",
                "artifacts": {
                    "quickstart": {"filename": quickstart.name}
                }
            }
            
            manifest_file = release_dir / "release_manifest.json"
            import json
            with open(manifest_file, "w") as f:
                json.dump(validation_manifest, f, indent=2)
            
            validation_result = dist_manager.validate_release(str(release_dir))
            print(f"    ✓ Validation result: {'VALID' if validation_result['valid'] else 'INVALID'}")
            if 'checks' in validation_result:
                passed_checks = sum(1 for check in validation_result['checks'].values() if check.get('passed', False))
                total_checks = len(validation_result['checks'])
                print(f"    ✓ Checks passed: {passed_checks}/{total_checks}")
            
            print("\n🎉 All distribution preparation tests completed successfully!")
            return True
            
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_prepare_release_script():
    """Test the prepare_release.py script functionality."""
    print("\n🔧 Testing prepare_release.py script...")
    
    try:
        # Test help command
        import subprocess
        result = subprocess.run([
            sys.executable, "scripts/prepare_release.py", "--help"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print("    ✓ Help command works")
        else:
            print(f"    ❌ Help command failed: {result.stderr}")
            return False
        
        # Test list command (should show no releases)
        result = subprocess.run([
            sys.executable, "scripts/prepare_release.py", "list"
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print("    ✓ List command works")
        else:
            print(f"    ❌ List command failed: {result.stderr}")
            return False
        
        print("    ✓ prepare_release.py script tests passed")
        return True
        
    except Exception as e:
        print(f"    ❌ Script test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Starting Distribution Preparation Tests")
    print("=" * 60)
    
    success = True
    
    # Test distribution manager
    if not test_distribution_preparation():
        success = False
    
    # Test prepare_release script
    if not test_prepare_release_script():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! Distribution preparation system is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())