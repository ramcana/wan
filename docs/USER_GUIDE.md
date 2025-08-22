# Wan2.2 UI Variant - User Guide

## Welcome to Wan2.2 Video Generation

This comprehensive user guide will help you get the most out of the Wan2.2 UI Variant, a powerful web-based interface for generating high-quality videos using advanced AI models.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Interface Overview](#interface-overview)
3. [Generation Modes](#generation-modes)
4. [Optimization Settings](#optimization-settings)
5. [Queue Management](#queue-management)
6. [Output Management](#output-management)
7. [Advanced Features](#advanced-features)
8. [Tips and Best Practices](#tips-and-best-practices)
9. [Troubleshooting](#troubleshooting)
10. [FAQ](#faq)

## Getting Started

### First Launch

1. **Start the Application**

   ```bash
   python main.py
   ```

   The interface will open in your default web browser at `http://localhost:7860`

2. **System Check**

   - The application will automatically check your system requirements
   - GPU availability and VRAM will be detected
   - Models will be downloaded automatically on first use

3. **Initial Setup**
   - Review the default settings in the Optimizations tab
   - Adjust VRAM settings based on your GPU
   - Test with a simple generation to verify everything works

### Quick Start Tutorial

1. **Navigate to the Generation tab**
2. **Select "t2v-A14B" model type**
3. **Enter a simple prompt**: "A beautiful sunset over mountains"
4. **Click "Generate Now"**
5. **Wait for the video to generate** (typically 5-9 minutes for 720p)
6. **View your result** in the output area

## Interface Overview

The Wan2.2 UI is organized into four main tabs:

### 🎥 Generation Tab

- **Primary interface** for creating videos
- **Model selection** and input configuration
- **Real-time generation** status and progress
- **Immediate output** display

### ⚙️ Optimizations Tab

- **VRAM management** settings
- **Performance optimization** controls
- **Quick preset** configurations
- **Resource monitoring**

### 📊 Queue & Stats Tab

- **Batch processing** queue management
- **Real-time system** statistics
- **Performance monitoring**
- **Error tracking**

### 📁 Outputs Tab

- **Video gallery** with thumbnails
- **Metadata viewing**
- **File management** tools
- **Export options**

## Generation Modes

### Text-to-Video (T2V)

**Best for**: Creating videos from text descriptions alone

**Model**: t2v-A14B

- **Input**: Text prompt only
- **Resolution**: Up to 1920x1080
- **Generation Time**: 5-9 minutes (720p), 12-17 minutes (1080p)

**Example Prompts**:

```
"A majestic eagle soaring through cloudy skies"
"Waves crashing against rocky cliffs at sunset"
"A bustling city street with neon lights at night"
```

### Image-to-Video (I2V)

**Best for**: Animating existing images or extending static content

**Model**: i2v-A14B

- **Input**: Image + optional text prompt
- **Supported Formats**: PNG, JPG, JPEG, WebP
- **Max File Size**: 10MB
- **Resolution**: Matches input image (up to 1920x1080)

**Tips**:

- Use high-quality input images for best results
- Text prompts can guide the animation style
- Works well with portraits, landscapes, and objects

### Text-Image-to-Video (TI2V)

**Best for**: Precise control using both text and visual references

**Model**: ti2v-5B

- **Input**: Text prompt + reference image
- **Advanced Features**: Cross-attention between text and image
- **Best Quality**: Highest fidelity output
- **Generation Time**: Longer due to complexity

**Use Cases**:

- Style transfer from reference images
- Character consistency across scenes
- Complex scene composition

## Optimization Settings

### Quantization Levels

**FP16 (Half Precision)**

- **VRAM Usage**: ~50% reduction
- **Quality**: Minimal impact
- **Speed**: Faster inference
- **Recommended**: For most users

**BF16 (Brain Float 16)**

- **VRAM Usage**: ~45% reduction
- **Quality**: Better than FP16
- **Compatibility**: Best with modern GPUs
- **Recommended**: Default setting

**INT8 (8-bit Integer)**

- **VRAM Usage**: ~70% reduction
- **Quality**: Slight reduction
- **Speed**: May be slower
- **Use Case**: Very limited VRAM

### Model Offloading

**Standard CPU Offload**

- **VRAM Savings**: ~40%
- **Performance Impact**: ~20% slower
- **RAM Requirement**: Additional 8-16GB

**Sequential CPU Offload**

- **VRAM Savings**: ~60%
- **Performance Impact**: ~30% slower
- **Best For**: Limited VRAM systems

### VAE Tiling

**Tile Size Settings**:

- **128px**: Maximum VRAM savings, slower
- **256px**: Balanced (recommended)
- **384px**: Less savings, faster
- **512px**: Minimal savings, fastest

### Quick Presets

**🔋 Low VRAM (8GB)**

```json
{
  "quantization": "int8",
  "cpu_offload": true,
  "sequential_offload": true,
  "vae_tile_size": 128
}
```

**⚖️ Balanced (12GB)**

```json
{
  "quantization": "bf16",
  "cpu_offload": true,
  "sequential_offload": false,
  "vae_tile_size": 256
}
```

**🎯 High Quality (16GB+)**

```json
{
  "quantization": "fp16",
  "cpu_offload": false,
  "sequential_offload": false,
  "vae_tile_size": 512
}
```

## Queue Management

### Adding to Queue

1. **Configure your generation** settings
2. **Click "Add to Queue"** instead of "Generate Now"
3. **Repeat** for multiple generations
4. **Monitor progress** in Queue & Stats tab

### Queue Features

- **FIFO Processing**: First in, first out
- **Automatic Progression**: No manual intervention needed
- **Status Tracking**: Real-time progress updates
- **Error Handling**: Failed tasks don't stop the queue

### Queue Controls

- **⏸️ Pause Queue**: Stop processing new tasks
- **▶️ Resume Queue**: Continue processing
- **🗑️ Clear Queue**: Remove all pending tasks

## Output Management

### Video Gallery

- **Thumbnail View**: Quick preview of all generated videos
- **Metadata Display**: Prompt, settings, and generation info
- **Sorting Options**: By date, name, or generation time

### File Operations

- **Download**: Save videos to your device
- **Delete**: Remove unwanted outputs
- **Rename**: Organize your content
- **Share**: Generate shareable links

### Metadata Information

Each video includes:

- **Original Prompt**: Text used for generation
- **Model Type**: T2V, I2V, or TI2V
- **Resolution**: Output dimensions
- **Generation Time**: How long it took to create
- **Settings Used**: Optimization parameters
- **Timestamp**: When it was created

## Advanced Features

### Prompt Enhancement

**Automatic Enhancement**:

- Click "✨ Enhance Prompt" to improve your text
- Adds quality keywords and cinematic terms
- Preserves original creative intent
- Shows enhanced version for review

**VACE Aesthetics**:

- Automatically detected in prompts
- Adds cinematic style improvements
- Enhances visual quality and coherence

### LoRA Support

**Loading LoRA Weights**:

1. **Place LoRA files** in the `loras/` directory
2. **Enter the path** in the LoRA settings
3. **Adjust strength** (0.0 to 2.0)
4. **Generate** with enhanced styling

**LoRA Strength Guidelines**:

- **0.0**: No effect
- **0.5-0.8**: Subtle influence
- **1.0**: Full strength (recommended)
- **1.2-1.5**: Strong influence
- **2.0**: Maximum effect (may cause artifacts)

### Performance Monitoring

**Real-time Stats**:

- **CPU Usage**: Current processor load
- **RAM Usage**: System memory consumption
- **GPU Usage**: Graphics card utilization
- **VRAM Usage**: Video memory with warnings

**Performance Warnings**:

- **High CPU**: >80% usage
- **High Memory**: >85% usage
- **High VRAM**: >90% usage

## Tips and Best Practices

### Writing Effective Prompts

**Structure Your Prompts**:

```
[Subject] [Action] [Setting] [Style] [Quality Terms]
```

**Example**:

```
"A majestic dragon flying through stormy clouds over a medieval castle, cinematic lighting, high quality, detailed"
```

**Quality Keywords**:

- "high quality", "detailed", "sharp"
- "cinematic", "professional", "masterpiece"
- "4k", "ultra detailed", "photorealistic"

**Style Keywords**:

- "anime style", "oil painting", "watercolor"
- "cyberpunk", "steampunk", "fantasy"
- "noir", "vintage", "modern"

### Optimization Strategies

**For Speed**:

- Use 720p resolution
- Reduce generation steps (30-40)
- Enable CPU offloading
- Use FP16 quantization

**For Quality**:

- Use 1080p resolution
- Increase generation steps (60-80)
- Disable offloading if VRAM allows
- Use BF16 quantization

**For VRAM Efficiency**:

- Enable sequential CPU offload
- Use smaller VAE tile sizes
- Try INT8 quantization
- Close other GPU applications

### Batch Processing Tips

1. **Plan Your Queue**: Organize similar generations together
2. **Monitor Resources**: Check system stats regularly
3. **Vary Settings**: Test different optimizations
4. **Save Configurations**: Note successful settings

### File Management

- **Organize Outputs**: Create folders by project or date
- **Regular Cleanup**: Remove unwanted generations
- **Backup Important**: Save your best results
- **Monitor Disk Space**: Large video files accumulate quickly

## Troubleshooting

### Common Issues

#### Generation Fails to Start

**Symptoms**: Error message, no progress
**Solutions**:

1. Check GPU availability
2. Verify model downloads
3. Restart application
4. Check disk space

#### Out of Memory Errors

**Symptoms**: "CUDA out of memory"
**Solutions**:

1. Enable CPU offloading
2. Reduce resolution
3. Use INT8 quantization
4. Close other applications

#### Slow Generation Times

**Symptoms**: Takes much longer than expected
**Solutions**:

1. Check system resources
2. Optimize VRAM settings
3. Update GPU drivers
4. Use SSD storage

#### Poor Video Quality

**Symptoms**: Blurry, artifacts, inconsistent
**Solutions**:

1. Improve prompt quality
2. Increase generation steps
3. Use higher resolution
4. Try different model types

### Error Messages

#### "Model not found"

- **Cause**: Download failed or interrupted
- **Solution**: Clear model cache and retry

#### "Invalid image format"

- **Cause**: Unsupported file type
- **Solution**: Convert to PNG, JPG, or WebP

#### "Queue processing stopped"

- **Cause**: System error or resource exhaustion
- **Solution**: Check logs, restart queue

### Performance Issues

#### High CPU Usage

- Close unnecessary applications
- Reduce concurrent operations
- Check for background processes

#### High Memory Usage

- Restart application periodically
- Clear browser cache
- Monitor system resources

#### GPU Not Detected

- Update GPU drivers
- Check CUDA installation
- Verify GPU compatibility

## FAQ

### General Questions

**Q: How long does it take to generate a video?**
A: 720p videos typically take 5-9 minutes, 1080p takes 12-17 minutes on RTX 4080.

**Q: Can I use multiple GPUs?**
A: Currently, the application uses a single GPU. Multi-GPU support may be added in future versions.

**Q: What video formats are supported?**
A: Output videos are in MP4 format with H.264 encoding.

**Q: Can I cancel a generation in progress?**
A: Yes, you can stop the current generation, though partial progress will be lost.

### Technical Questions

**Q: How much VRAM do I need?**
A: Minimum 8GB, recommended 12GB for optimal performance.

**Q: Can I run this on CPU only?**
A: GPU is required for reasonable generation times. CPU-only mode is not supported.

**Q: What's the maximum video resolution?**
A: Up to 1920x1080 (Full HD) depending on the model used.

**Q: How do I update the application?**
A: Pull the latest code from the repository and restart the application.

### Troubleshooting Questions

**Q: Why is my generation taking so long?**
A: Check system resources, enable optimizations, and ensure no other GPU applications are running.

**Q: The interface won't load, what should I do?**
A: Check if the port is available, verify firewall settings, and try a different browser.

**Q: How do I report bugs or request features?**
A: Use the GitHub issues page or contact the development team.

### Advanced Questions

**Q: Can I fine-tune the models?**
A: The interface doesn't support fine-tuning, but you can use LoRA weights for style adaptation.

**Q: How do I backup my generated videos?**
A: Videos are stored in the `outputs/` directory. Copy this folder to backup your content.

**Q: Can I use custom models?**
A: Currently, only the built-in Wan2.2 models are supported.

## Getting Help

### Support Resources

1. **Documentation**: This user guide and deployment guide
2. **Logs**: Check `wan22_ui.log` for detailed information
3. **Performance Reports**: Generate reports from the Stats tab
4. **Community**: Join the user community for tips and support

### Reporting Issues

When reporting issues, please include:

- System specifications (GPU, RAM, OS)
- Application version
- Error messages or logs
- Steps to reproduce the problem
- Screenshots if applicable

### Feature Requests

We welcome suggestions for new features:

- Describe the desired functionality
- Explain the use case
- Provide examples if possible

---

**Happy video generating!** 🎬

For technical support and deployment information, see the [Deployment Guide](DEPLOYMENT_GUIDE.md).
