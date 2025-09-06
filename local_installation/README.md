# WAN2.2 Local Installation Package

This package provides a one-click installation solution for the WAN2.2 video generation system. The installer automatically detects your hardware specifications and configures the system for optimal performance.

## Quick Start

1. Double-click `install.bat` to start the installation
2. The installer will automatically:
   - Detect your hardware (CPU, RAM, GPU)
   - Install Python and dependencies
   - Download WAN2.2 models
   - Configure optimal settings
   - Validate the installation

## System Requirements

- Windows 10/11 (64-bit)
- 8GB RAM minimum (16GB+ recommended)
- 50GB free disk space
- NVIDIA GPU with 6GB+ VRAM (recommended)

## Installation Options

```batch
install.bat                 # Standard installation
install.bat --silent       # Silent installation mode
install.bat --dev-mode     # Install development dependencies
install.bat --skip-models  # Skip model download (for testing)
```

## Troubleshooting

If installation fails, check `logs/installation.log` for detailed error information.

For support, see the troubleshooting guide in the documentation.
