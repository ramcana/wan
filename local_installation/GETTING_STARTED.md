# WAN2.2 Getting Started Guide

Welcome to WAN2.2, the advanced video generation system! This guide will help you get started with generating high-quality videos from text prompts and images.

## Quick Start

### First Launch

1. **Run First-Time Setup**: Double-click `run_first_setup.bat` to configure your preferences
2. **Launch WAN2.2**: Use the desktop shortcut or Start Menu entry
3. **Start Generating**: Follow the in-app instructions to create your first video

### Launch Options

- **Desktop Shortcuts**: Double-click "WAN2.2 Video Generator" or "WAN2.2 UI"
- **Start Menu**: Navigate to Start → WAN2.2 → Choose your preferred application
- **Manual Launch**: Run `launch_wan22.bat` or `launch_wan22_ui.bat` from the installation folder

## Generation Modes

### Text-to-Video (T2V)

Generate videos directly from text descriptions.

**Example Prompts:**

- "A serene lake at sunset with gentle ripples"
- "A bustling city street with cars and pedestrians"
- "Ocean waves crashing against rocky cliffs"

**Tips:**

- Be descriptive and specific
- Include details about lighting, movement, and atmosphere
- Start with shorter prompts and expand as needed

### Image-to-Video (I2V)

Transform static images into dynamic videos.

**Supported Formats:**

- PNG, JPG, JPEG
- Recommended resolution: 512x512 or 1024x1024
- File size: Under 10MB for best performance

**Tips:**

- Use high-quality source images
- Images with clear subjects work best
- Add text prompts to guide the animation style

### Text+Image-to-Video (TI2V)

Combine text prompts with source images for guided generation.

**Best Practices:**

- Ensure text prompt complements the image
- Use prompts to describe desired motion or changes
- Experiment with different prompt styles

## Configuration

### Performance Settings

Access via Start Menu → WAN2.2 → WAN2.2 Configuration

**Key Settings:**

- **VRAM Usage**: Adjust based on your GPU memory (4-16GB)
- **CPU Threads**: Set to match your CPU cores for optimal performance
- **Resolution**: Choose 720p for speed or 1080p for quality

### Quality vs Speed

- **Fast Generation**: 720p, lower quality settings, reduced VRAM usage
- **High Quality**: 1080p, high quality settings, maximum VRAM usage
- **Balanced**: 720p, high quality settings, moderate VRAM usage

## File Locations

### Important Directories

- **Installation**: `[Installation Path]`
- **Configuration**: `config.json`
- **Generated Videos**: `outputs/`
- **Log Files**: `logs/`
- **Models**: `models/`

### Output Organization

Generated videos are automatically organized by:

- Date and time of generation
- Generation mode (T2V, I2V, TI2V)
- Resolution and quality settings

## Troubleshooting

### Common Issues

**Application Won't Start**

- Check `logs/installation.log` for errors
- Ensure virtual environment is intact
- Try running `run_first_setup.bat` again

**Generation is Slow**

- Reduce resolution to 720p
- Lower VRAM usage limit
- Adjust CPU thread count
- Close other GPU-intensive applications

**Out of Memory Errors**

- Reduce VRAM usage in configuration
- Lower resolution settings
- Ensure sufficient system RAM (8GB+ recommended)
- Close unnecessary applications

**Poor Quality Results**

- Increase resolution to 1080p
- Adjust quality settings to "high"
- Refine your text prompts
- Use higher quality source images

### Performance Optimization

**For High-End Systems (RTX 4080+, 32GB+ RAM):**

- VRAM Usage: 12-16GB
- CPU Threads: 16-32
- Resolution: 1080p
- Quality: High

**For Mid-Range Systems (RTX 3070+, 16GB+ RAM):**

- VRAM Usage: 8-10GB
- CPU Threads: 8-16
- Resolution: 720p-1080p
- Quality: Medium-High

**For Budget Systems (GTX 1660+, 8GB+ RAM):**

- VRAM Usage: 4-6GB
- CPU Threads: 4-8
- Resolution: 720p
- Quality: Medium

## Advanced Features

### Batch Processing

- Queue multiple generations
- Process overnight for large batches
- Monitor progress in the UI

### Custom Models

- Additional models can be placed in `models/` directory
- Restart application after adding new models
- Check compatibility before downloading

### Configuration Backup

- Configuration files are automatically backed up
- Manual backups can be created by copying `config.json`
- Restore previous settings if needed

## Getting Help

### Resources

- **Documentation**: Check the `docs/` folder for detailed guides
- **Log Files**: Review `logs/` for error details
- **Configuration**: Use the built-in configuration wizard

### Support

- Check the GitHub repository for known issues
- Review the troubleshooting section above
- Ensure your system meets minimum requirements

### System Requirements

- **OS**: Windows 10/11 (64-bit)
- **RAM**: 8GB minimum, 16GB+ recommended
- **GPU**: NVIDIA GTX 1060+ or AMD equivalent
- **Storage**: 50GB+ free space
- **Python**: Automatically installed by the installer

## Tips for Best Results

### Prompt Engineering

- Use descriptive, specific language
- Include lighting and atmosphere details
- Specify camera movements or perspectives
- Experiment with artistic styles

### Image Preparation

- Use high-resolution source images
- Ensure good contrast and clarity
- Avoid heavily compressed images
- Consider the aspect ratio for video output

### Performance Monitoring

- Monitor GPU usage in Task Manager
- Watch for memory warnings in logs
- Adjust settings based on system performance
- Keep drivers updated for best compatibility

---

**Enjoy creating with WAN2.2!**

For additional help, run the first-time setup wizard again or check the configuration options in the Start Menu.
