#!/usr/bin/env python3
"""
Debug script to check dependency validation logic
"""

import sys
from pathlib import Path
import importlib.util

# Add local_testing_framework to path
sys.path.insert(0, str(Path(__file__).parent))

from local_testing_framework.environment_validator import EnvironmentValidator

def debug_dependencies():
    """Debug dependency validation"""
    validator = EnvironmentValidator()
    
    print("Debugging dependency validation...")
    print("=" * 50)
    
    # Read requirements file
    requirements_path = Path(validator.config.requirements_path)
    print(f"Requirements file: {requirements_path}")
    print(f"File exists: {requirements_path.exists()}")
    
    if requirements_path.exists():
        with open(requirements_path, 'r') as f:
            requirements = f.read().strip().split('\n')
        
        # Filter out empty lines and comments
        requirements = [req.strip() for req in requirements 
                      if req.strip() and not req.strip().startswith('#')]
        
        print(f"Total requirements found: {len(requirements)}")
        print("\nChecking each requirement:")
        
        missing_packages = []
        installed_packages = []
        
        for requirement in requirements:
            # Parse package name (handle version specifiers)
            package_name = requirement.split('==')[0].split('>=')[0].split('<=')[0].split('>')[0].split('<')[0].strip()
            
            # Map package names to their import names
            import_name_map = {
                'Pillow': 'PIL',
                'opencv-python': 'cv2',
                'pyyaml': 'yaml',
                'huggingface-hub': 'huggingface_hub',
                'imageio-ffmpeg': 'imageio_ffmpeg',
                'python-multipart': 'multipart',
                'nvidia-ml-py': 'pynvml'
            }
            
            import_name = import_name_map.get(package_name, package_name.replace('-', '_'))
            
            try:
                # Try to import the package
                spec = importlib.util.find_spec(import_name)
                if spec is not None:
                    installed_packages.append(requirement)
                    print(f"✓ {package_name} ({import_name}) - INSTALLED")
                else:
                    missing_packages.append(requirement)
                    print(f"✗ {package_name} ({import_name}) - MISSING")
            except (ImportError, ModuleNotFoundError):
                missing_packages.append(requirement)
                print(f"✗ {package_name} ({import_name}) - MISSING (ImportError)")
        
        print(f"\nSummary:")
        print(f"Installed: {len(installed_packages)}")
        print(f"Missing: {len(missing_packages)}")
        
        if missing_packages:
            print(f"\nMissing packages:")
            for pkg in missing_packages:
                print(f"  - {pkg}")

if __name__ == '__main__':
    debug_dependencies()
