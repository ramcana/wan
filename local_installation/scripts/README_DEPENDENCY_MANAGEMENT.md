# WAN2.2 Dependency Management System

This document describes the dependency management system for the WAN2.2 local installation package.

## Overview

The dependency management system handles:

- Python installation detection and embedded Python deployment
- Virtual environment creation with hardware-optimized settings
- Package installation with CUDA-aware selection
- Dependency conflict resolution and automatic retry mechanisms

## Components

### 1. PythonInstallationHandler

Handles Python detection, download, and installation.

**Key Features:**

- Detects existing Python installations (system and embedded)
- Downloads and installs embedded Python for portable deployment
- Configures embedded Python for package installation
- Creates virtual environments with hardware-optimized environment variables

**Usage:**

```python
from scripts.setup_dependencies import PythonInstallationHandler

handler = PythonInstallationHandler("/path/to/installation")

# Check existing Python
python_info = handler.check_python_installation()

# Install embedded Python if needed
if python_info["recommended_action"] == "install_embedded":
    handler.install_python()

# Create virtual environment
handler.create_virtual_environment("/path/to/venv", hardware_profile)
```

### 2. PackageInstallationSystem

Handles package installation with hardware-specific optimizations.

**Key Features:**

- Processes requirements.txt with hardware-specific modifications
- Installs packages in batches to handle dependencies
- Adds CUDA-specific packages based on GPU detection
- Provides fallback strategies for failed installations

**Usage:**

```python
from scripts.setup_dependencies import PackageInstallationSystem

package_system = PackageInstallationSystem("/path/to/installation", python_handler)

# Install packages with hardware optimization
package_system.install_packages("/path/to/requirements.txt", hardware_profile)
```

### 3. PackageInstallationOrchestrator

Advanced package resolver with conflict detection and resolution.

**Key Features:**

- CUDA-aware package selection based on GPU capabilities
- Dependency conflict detection and resolution
- Automatic retry mechanisms with fallback strategies
- Package grouping for optimal installation order

**Usage:**

```python
from scripts.package_resolver import PackageInstallationOrchestrator

orchestrator = PackageInstallationOrchestrator("/path/to/installation", python_exe)

# Install with advanced resolution
orchestrator.install_packages_with_resolution(requirements, hardware_profile)
```

### 4. DependencyManager

Main orchestrator that coordinates all dependency management operations.

**Key Features:**

- Unified interface for all dependency operations
- Progress reporting and error handling
- Installation validation and summary generation
- Integration with hardware detection

**Usage:**

```python
from scripts.setup_dependencies import DependencyManager

dep_manager = DependencyManager("/path/to/installation", progress_reporter)

# Complete dependency setup
dep_manager.check_python_installation()
dep_manager.create_virtual_environment("/path/to/venv", hardware_profile)
dep_manager.install_packages("/path/to/requirements.txt", hardware_profile)

# Validate installation
result = dep_manager.validate_installation()
```

## Hardware Optimization

The system automatically optimizes installations based on detected hardware:

### CPU Optimization

- Sets `OMP_NUM_THREADS` and `MKL_NUM_THREADS` based on CPU cores
- Configures thread allocation for high-core systems (32+ cores)

### Memory Optimization

- Sets `PYTORCH_CUDA_ALLOC_CONF` based on available RAM
- Configures memory pool sizes for different RAM tiers

### GPU Optimization

- Selects appropriate CUDA versions based on GPU model and driver
- Installs CUDA-specific PyTorch packages
- Adds xformers for high-end GPUs (8GB+ VRAM)
- Configures GPU memory allocation settings

## CUDA Package Selection

The system includes a sophisticated CUDA package selector:

### Supported CUDA Versions

- **CUDA 11.8**: RTX 30/40 series, GTX 16 series, Tesla, Quadro
- **CUDA 12.1**: RTX 40/30 series, Tesla H100, A100
- **CUDA 12.4**: RTX 40/50 series, H100, A100

### Package Selection Logic

1. Detects GPU model and CUDA version
2. Matches against compatibility matrix
3. Selects highest compatible CUDA version
4. Downloads packages from appropriate PyTorch index

### Fallback Strategies

- Falls back to CPU versions if CUDA installation fails
- Uses closest compatible CUDA version if exact match unavailable
- Removes version constraints if specific versions fail

## Dependency Conflict Resolution

The system detects and resolves common package conflicts:

### Known Conflicts

- **torch vs tensorflow**: Recommends choosing one framework
- **opencv-python vs opencv-contrib-python**: Uses contrib version
- **pillow vs PIL**: Uses modern Pillow package

### Version Compatibility

- Ensures PyTorch and Transformers version compatibility
- Validates CUDA package version alignment
- Resolves dependency version conflicts automatically

## Error Handling

Comprehensive error handling with recovery suggestions:

### Error Categories

- **System Errors**: Hardware detection, insufficient resources
- **Network Errors**: Download failures, connectivity issues
- **Permission Errors**: Insufficient privileges, file access
- **Configuration Errors**: Invalid settings, compatibility issues

### Recovery Strategies

- **Automatic Retry**: Network downloads, transient failures
- **Fallback Options**: Alternative packages, reduced settings
- **User Guidance**: Clear error messages with solutions

## Testing

The system includes comprehensive tests:

```bash
# Run dependency management tests
python test_dependency_management.py

# Run example workflow
python examples/dependency_management_example.py
```

### Test Coverage

- Python installation detection
- Virtual environment creation
- Requirements processing
- CUDA package selection
- Complete integration workflow

## Configuration Files

### requirements.txt

Located at `resources/requirements.txt`, contains base package requirements:

```
# Core dependencies
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
diffusers>=0.20.0
# ... additional packages
```

### Hardware-Specific Additions

The system automatically adds hardware-specific packages:

**For NVIDIA GPUs:**

```
torch --index-url https://download.pytorch.org/whl/cu118
torchvision --index-url https://download.pytorch.org/whl/cu118
xformers  # For high-end GPUs
```

**For AMD GPUs:**

```
torch --index-url https://download.pytorch.org/whl/rocm5.4.2
torchvision --index-url https://download.pytorch.org/whl/rocm5.4.2
```

## Integration with Installation System

The dependency management system integrates with the broader installation system:

1. **Hardware Detection**: Uses detected hardware profile for optimization
2. **Progress Reporting**: Reports progress to batch file interface
3. **Error Handling**: Provides user-friendly error messages
4. **State Management**: Saves installation state for recovery
5. **Validation**: Validates installation before proceeding

## Best Practices

### For Developers

- Always use the `DependencyManager` class for high-level operations
- Handle `InstallationError` exceptions with appropriate recovery
- Use progress reporters for user feedback
- Test with different hardware profiles

### For Users

- Ensure stable internet connection during installation
- Run installation with administrator privileges if needed
- Allow sufficient disk space for packages and models
- Check system requirements before installation

## Troubleshooting

### Common Issues

**Python Installation Fails:**

- Check internet connection
- Verify disk space availability
- Try running as administrator

**Package Installation Fails:**

- Check PyPI connectivity
- Verify CUDA driver installation
- Try CPU-only installation as fallback

**Virtual Environment Creation Fails:**

- Check Python installation
- Verify file permissions
- Ensure sufficient disk space

**CUDA Packages Not Found:**

- Update GPU drivers
- Check CUDA toolkit installation
- Verify GPU compatibility

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

Planned improvements for the dependency management system:

1. **Conda Support**: Add conda environment management
2. **Offline Installation**: Support for offline package installation
3. **Package Caching**: Cache downloaded packages for faster reinstallation
4. **Version Pinning**: Pin specific package versions for reproducibility
5. **Health Checks**: Periodic dependency health validation
6. **Update Management**: Automatic dependency updates and migration
