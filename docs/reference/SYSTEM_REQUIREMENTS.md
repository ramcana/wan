---
category: reference
last_updated: '2025-09-15T22:50:00.490365'
original_path: reports\SYSTEM_REQUIREMENTS.md
tags:
- configuration
- api
- troubleshooting
- installation
- security
- performance
title: "\U0001F5A5\uFE0F WAN22 System Requirements"
---

# 🖥️ WAN22 System Requirements

## Minimum Requirements

### Operating System

- **Windows 10** (64-bit) or newer
- **Windows 11** (recommended)
- **Linux** (Ubuntu 20.04+ or equivalent)
- **macOS** (10.15+ with Intel or Apple Silicon)

### Hardware Requirements

#### CPU

- **Minimum:** Intel i5-8400 / AMD Ryzen 5 2600 or equivalent
- **Recommended:** Intel i7-10700K / AMD Ryzen 7 3700X or better
- **Cores:** 4+ cores (8+ recommended)
- **Architecture:** x64 (64-bit)

#### Memory (RAM)

- **Minimum:** 8 GB RAM
- **Recommended:** 16 GB RAM
- **Optimal:** 32 GB RAM (for large model processing)

#### Storage

- **Minimum:** 50 GB free space
- **Recommended:** 100 GB+ free space (SSD preferred)
- **Model Storage:** Additional 20-50 GB per AI model
- **Type:** SSD strongly recommended for performance

#### Graphics Card (GPU)

##### NVIDIA GPUs (Recommended)

- **Minimum:** GTX 1060 6GB / RTX 2060
- **Recommended:** RTX 3070 / RTX 4070 or better
- **Optimal:** RTX 4080 / RTX 4090
- **VRAM:** 6 GB minimum, 12+ GB recommended
- **CUDA:** Version 11.8+ required

##### AMD GPUs (Limited Support)

- **Minimum:** RX 6600 XT
- **Recommended:** RX 7800 XT or better
- **Note:** Some features may be limited compared to NVIDIA

##### Intel GPUs (Experimental)

- **Arc A770** or better
- **Note:** Limited compatibility, use at your own risk

##### CPU-Only Mode

- **Supported:** Yes, but significantly slower
- **RAM:** 32 GB+ recommended for CPU inference
- **Performance:** 10-50x slower than GPU acceleration

### Software Requirements

#### Python Environment

- **Version:** Python 3.8, 3.9, 3.10, or 3.11
- **Not Supported:** Python 3.12+ (dependency compatibility issues)
- **Package Manager:** pip (included with Python)
- **Virtual Environment:** Recommended but not required

#### Node.js Environment

- **Version:** Node.js 16.x, 18.x, or 20.x LTS
- **Package Manager:** npm (included) or yarn
- **Not Supported:** Node.js 14.x or older

#### Additional Dependencies

- **Git:** For cloning repository and version control
- **Visual C++ Redistributable:** (Windows only, usually pre-installed)
- **CUDA Toolkit:** 11.8+ (for NVIDIA GPU acceleration)

## Performance Tiers

### Tier 1: Basic Usage (Minimum Requirements)

- **CPU:** Intel i5-8400 / AMD Ryzen 5 2600
- **RAM:** 8 GB
- **GPU:** GTX 1060 6GB / CPU-only
- **Storage:** 50 GB HDD
- **Performance:** Basic video generation, longer processing times

### Tier 2: Standard Usage (Recommended)

- **CPU:** Intel i7-10700K / AMD Ryzen 7 3700X
- **RAM:** 16 GB
- **GPU:** RTX 3070 / RTX 4070
- **Storage:** 100 GB SSD
- **Performance:** Good video generation speed, multiple concurrent tasks

### Tier 3: Professional Usage (Optimal)

- **CPU:** Intel i9-12900K / AMD Ryzen 9 5900X
- **RAM:** 32 GB
- **GPU:** RTX 4080 / RTX 4090
- **Storage:** 500 GB NVMe SSD
- **Performance:** Fast video generation, batch processing, development work

### Tier 4: Enterprise/Research (Maximum)

- **CPU:** Intel i9-13900K / AMD Ryzen 9 7950X
- **RAM:** 64 GB+
- **GPU:** Multiple RTX 4090s or A100/H100
- **Storage:** 1 TB+ NVMe SSD
- **Performance:** Maximum throughput, large-scale processing

## Network Requirements

### Internet Connection

- **Required for:** Initial setup, model downloads, updates
- **Speed:** 10 Mbps+ recommended for model downloads
- **Data Usage:** 5-20 GB for initial model downloads
- **Offline Usage:** Supported after initial setup

### Firewall/Security

- **Ports:** 3000 (frontend), 8000 (backend), 7860 (Gradio UI)
- **Firewall:** Allow Python and Node.js through Windows Firewall
- **Antivirus:** May need to whitelist project directory

## Browser Requirements

### Supported Browsers

- **Chrome:** Version 90+ (recommended)
- **Firefox:** Version 88+
- **Edge:** Version 90+
- **Safari:** Version 14+ (macOS only)

### Browser Features Required

- **JavaScript:** ES2020+ support
- **WebSockets:** For real-time updates
- **Local Storage:** For settings and cache
- **WebGL:** For advanced UI features (optional)

## Development Requirements (Additional)

### Code Editor

- **VS Code:** Recommended with Python and TypeScript extensions
- **PyCharm:** Professional or Community edition
- **Any editor:** With Python and JavaScript/TypeScript support

### Development Tools

- **Git:** Version control
- **Docker:** For containerized development (optional)
- **Postman/Insomnia:** API testing (optional)

## Platform-Specific Notes

### Windows

- **Windows Defender:** May slow down file operations, consider exclusions
- **PowerShell:** Version 5.1+ or PowerShell Core 7+
- **WSL2:** Supported for Linux-like development environment
- **Long Path Support:** Enable for deep directory structures

### Linux

- **Distribution:** Ubuntu 20.04+, Debian 11+, CentOS 8+, or equivalent
- **Packages:** `python3-dev`, `nodejs`, `npm`, `git`, `build-essential`
- **CUDA:** Install NVIDIA drivers and CUDA toolkit separately
- **Permissions:** User should be in `docker` group if using Docker

### macOS

- **Xcode Command Line Tools:** Required for compilation
- **Homebrew:** Recommended package manager
- **Rosetta 2:** Required for Intel-based software on Apple Silicon
- **Metal:** Used for GPU acceleration on Apple Silicon

## Compatibility Matrix

| Component       | Windows 10 | Windows 11 | Ubuntu 20.04+ | macOS 10.15+ |
| --------------- | ---------- | ---------- | ------------- | ------------ |
| Python 3.8-3.11 | ✅         | ✅         | ✅            | ✅           |
| Node.js 16-20   | ✅         | ✅         | ✅            | ✅           |
| NVIDIA GPU      | ✅         | ✅         | ✅            | ❌           |
| AMD GPU         | ⚠️         | ⚠️         | ⚠️            | ❌           |
| Intel GPU       | ⚠️         | ⚠️         | ⚠️            | ❌           |
| Apple Silicon   | ❌         | ❌         | ❌            | ⚠️           |
| CPU-only        | ✅         | ✅         | ✅            | ✅           |

**Legend:**

- ✅ Fully supported
- ⚠️ Limited support or experimental
- ❌ Not supported

## Performance Benchmarks

### Video Generation Times (1024x576, 16 frames)

| Hardware Configuration | Generation Time | Relative Performance |
| ---------------------- | --------------- | -------------------- |
| RTX 4090 + i9-13900K   | 15-30 seconds   | 100% (baseline)      |
| RTX 4080 + i7-12700K   | 20-40 seconds   | 75%                  |
| RTX 3080 + i7-10700K   | 30-60 seconds   | 50%                  |
| RTX 3070 + i5-11400F   | 45-90 seconds   | 35%                  |
| CPU-only (32GB RAM)    | 10-30 minutes   | 5%                   |

### Memory Usage Patterns

| Model Type | VRAM Usage | System RAM | Storage |
| ---------- | ---------- | ---------- | ------- |
| T2V-A14B   | 8-12 GB    | 4-8 GB     | 15 GB   |
| I2V-A14B   | 6-10 GB    | 3-6 GB     | 12 GB   |
| TI2V-5B    | 4-8 GB     | 2-4 GB     | 8 GB    |

## Optimization Recommendations

### For RTX 4080 Users

- Enable memory optimization in settings
- Use batch size 1 for large models
- Monitor VRAM usage during generation

### For Limited VRAM (< 8GB)

- Use smaller models (TI2V-5B)
- Enable CPU offloading
- Reduce batch sizes
- Close other GPU-intensive applications

### For CPU-Only Systems

- Increase system RAM to 32GB+
- Use fast NVMe SSD for swap
- Enable all CPU cores
- Consider cloud GPU services for heavy workloads

### For Development

- Use SSD for project directory
- Enable Windows long path support
- Configure antivirus exclusions
- Use virtual environments for Python

## Troubleshooting Hardware Issues

### GPU Not Detected

1. Check NVIDIA drivers are installed and up to date
2. Verify CUDA toolkit installation
3. Run `nvidia-smi` to check GPU status
4. Restart system after driver installation

### Out of Memory Errors

1. Close other applications using GPU/RAM
2. Reduce model batch size
3. Enable memory optimization settings
4. Consider upgrading hardware

### Slow Performance

1. Check if using integrated graphics instead of dedicated GPU
2. Verify SSD vs HDD usage
3. Monitor CPU/GPU temperatures for thermal throttling
4. Check for background processes consuming resources

### Installation Issues

1. Verify system meets minimum requirements
2. Check Python and Node.js versions
3. Run as administrator if permission errors
4. Disable antivirus temporarily during installation

For more detailed troubleshooting, see [COMPREHENSIVE_TROUBLESHOOTING.md](COMPREHENSIVE_TROUBLESHOOTING.md).
