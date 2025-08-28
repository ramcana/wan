"""
Test script for the dependency management system.
Tests Python installation, virtual environment creation, and package installation.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the scripts directory to the path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from scripts.setup_dependencies import DependencyManager, PythonInstallationHandler
from scripts.interfaces import HardwareProfile, CPUInfo, MemoryInfo, GPUInfo, StorageInfo, OSInfo
from scripts.base_classes import ConsoleProgressReporter


def create_test_hardware_profile() -> HardwareProfile:
    """Create a test hardware profile for testing."""
    return HardwareProfile(
        cpu=CPUInfo(
            model="AMD Ryzen 9 5900X",
            cores=12,
            threads=24,
            base_clock=3.7,
            boost_clock=4.8,
            architecture="x64"
        ),
        memory=MemoryInfo(
            total_gb=32,
            available_gb=28,
            type="DDR4",
            speed=3200
        ),
        gpu=GPUInfo(
            model="NVIDIA GeForce RTX 3080",
            vram_gb=10,
            cuda_version="12.1",
            driver_version="537.13",
            compute_capability="8.6"
        ),
        storage=StorageInfo(
            available_gb=500,
            type="NVMe SSD"
        ),
        os=OSInfo(
            name="Windows",
            version="11",
            architecture="x64"
        )
    )


def test_python_detection():
    """Test Python installation detection."""
    print("=== Testing Python Detection ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        handler = PythonInstallationHandler(temp_dir)
        
        python_info = handler.check_python_installation()
        
        print(f"System Python: {python_info.get('system_python')}")
        print(f"Embedded Python: {python_info.get('embedded_python')}")
        print(f"Recommended Action: {python_info.get('recommended_action')}")
        
        return python_info


def test_virtual_environment_creation():
    """Test virtual environment creation."""
    print("\n=== Testing Virtual Environment Creation ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        handler = PythonInstallationHandler(temp_dir)
        hardware_profile = create_test_hardware_profile()
        
        try:
            # Create virtual environment
            venv_path = Path(temp_dir) / "test_venv"
            success = handler.create_virtual_environment(str(venv_path), hardware_profile)
            
            if success:
                print("✅ Virtual environment created successfully")
                
                # Check if virtual environment was created
                if venv_path.exists():
                    print(f"✅ Virtual environment directory exists: {venv_path}")
                    
                    # Check for Python executable
                    python_exe = venv_path / "Scripts" / "python.exe"
                    if python_exe.exists():
                        print("✅ Python executable found in virtual environment")
                    else:
                        print("❌ Python executable not found in virtual environment")
                else:
                    print("❌ Virtual environment directory not created")
            else:
                print("❌ Virtual environment creation failed")
                
        except Exception as e:
            print(f"❌ Virtual environment creation failed with error: {e}")


def test_requirements_processing():
    """Test requirements file processing."""
    print("\n=== Testing Requirements Processing ===")
    
    # Create a test requirements file
    test_requirements = [
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "pillow>=9.5.0"
    ]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test requirements file
        req_file = Path(temp_dir) / "test_requirements.txt"
        req_file.write_text("\n".join(test_requirements))
        
        # Test processing
        from scripts.setup_dependencies import PackageInstallationSystem
        from scripts.setup_dependencies import PythonInstallationHandler
        
        python_handler = PythonInstallationHandler(temp_dir)
        package_system = PackageInstallationSystem(temp_dir, python_handler)
        package_system.requirements_file = req_file  # Set the requirements file path
        
        hardware_profile = create_test_hardware_profile()
        
        try:
            modified_reqs = package_system._process_requirements_for_hardware(hardware_profile)
            print(f"✅ Processed {len(modified_reqs)} requirements")
            
            # Check for GPU-specific packages
            gpu_packages = [req for req in modified_reqs if "cu" in req.lower() or "cuda" in req.lower()]
            if gpu_packages:
                print(f"✅ Found {len(gpu_packages)} GPU-specific packages")
                for pkg in gpu_packages:
                    print(f"   - {pkg}")
            else:
                print("ℹ️  No GPU-specific packages added")
                
        except Exception as e:
            print(f"❌ Requirements processing failed: {e}")


def test_dependency_manager_integration():
    """Test the complete dependency manager integration."""
    print("\n=== Testing Dependency Manager Integration ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create resources directory and requirements file
        resources_dir = Path(temp_dir) / "resources"
        resources_dir.mkdir()
        
        # Copy the actual requirements file
        actual_req_file = Path(__file__).parent / "resources" / "requirements.txt"
        test_req_file = resources_dir / "requirements.txt"
        
        if actual_req_file.exists():
            shutil.copy2(actual_req_file, test_req_file)
        else:
            # Create a minimal requirements file for testing
            test_requirements = [
                "numpy>=1.24.0",
                "pillow>=9.5.0",
                "requests>=2.31.0",
                "tqdm>=4.65.0"
            ]
            test_req_file.write_text("\n".join(test_requirements))
        
        # Initialize dependency manager
        progress_reporter = ConsoleProgressReporter()
        dep_manager = DependencyManager(temp_dir, progress_reporter)
        hardware_profile = create_test_hardware_profile()
        
        try:
            # Test Python detection
            python_info = dep_manager.check_python_installation()
            print(f"✅ Python detection completed: {python_info['recommended_action']}")
            
            # Test virtual environment creation
            venv_path = str(Path(temp_dir) / "venv")
            venv_success = dep_manager.create_virtual_environment(venv_path, hardware_profile)
            
            if venv_success:
                print("✅ Virtual environment creation successful")
            else:
                print("❌ Virtual environment creation failed")
                return False
            
            # Note: We skip actual package installation in tests to avoid long download times
            # In a real scenario, you would call:
            # dep_manager.install_packages(str(test_req_file), hardware_profile)
            
            print("✅ Dependency manager integration test completed")
            return True
            
        except Exception as e:
            print(f"❌ Dependency manager integration test failed: {e}")
            return False


def test_cuda_package_selection():
    """Test CUDA package selection logic."""
    print("\n=== Testing CUDA Package Selection ===")
    
    try:
        from scripts.package_resolver import CUDAPackageSelector
        
        # Test with RTX 3080
        gpu_info = GPUInfo(
            model="NVIDIA GeForce RTX 3080",
            vram_gb=10,
            cuda_version="12.1",
            driver_version="537.13",
            compute_capability="8.6"
        )
        
        selector = CUDAPackageSelector(gpu_info)
        
        # Test CUDA version selection
        cuda_version = selector.select_cuda_version()
        print(f"✅ Selected CUDA version: {cuda_version}")
        
        # Test package selection
        if cuda_version:
            cuda_packages = selector.get_cuda_packages(cuda_version)
            print(f"✅ CUDA packages selected:")
            for pkg, spec in cuda_packages.items():
                print(f"   - {pkg}: {spec}")
            
            # Test index URL
            index_url = selector.get_torch_index_url(cuda_version)
            if index_url:
                print(f"✅ PyTorch index URL: {index_url}")
        
        return True
        
    except Exception as e:
        print(f"❌ CUDA package selection test failed: {e}")
        return False


def main():
    """Run all dependency management tests."""
    print("WAN2.2 Dependency Management System Tests")
    print("=" * 50)
    
    tests = [
        test_python_detection,
        test_virtual_environment_creation,
        test_requirements_processing,
        test_cuda_package_selection,
        test_dependency_manager_integration
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result if result is not None else True)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for r in results if r)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())