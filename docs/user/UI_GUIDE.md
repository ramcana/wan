---
category: user
last_updated: '2025-09-15T22:50:00.287462'
original_path: local_installation\UI_GUIDE.md
tags:
- configuration
- troubleshooting
- installation
- performance
title: WAN2.2 User Interface Guide
---

# WAN2.2 User Interface Guide

## Overview

WAN2.2 provides two user interface options to suit different preferences and use cases:

1. **Desktop UI** - A native desktop application using Tkinter
2. **Web UI** - A browser-based interface using Flask

Both interfaces provide the same core functionality for video generation using the WAN2.2 models.

## Desktop UI

### Features

- **Native Windows Application**: Runs as a standalone desktop application
- **Three Generation Modes**:
  - Text-to-Video (T2V)
  - Image-to-Video (I2V)
  - Text+Image-to-Video (TI2V)
- **Real-time Progress Tracking**: Visual progress bars and status updates
- **Queue Management**: View and manage generation tasks
- **Output Gallery**: Browse and preview generated videos
- **System Monitoring**: Hardware status and model loading indicators

### Launching the Desktop UI

1. **From Desktop**: Double-click the "WAN2.2 Desktop UI" shortcut
2. **From Start Menu**: Navigate to Start Menu → WAN2.2 → WAN2.2 Desktop UI
3. **Manual Launch**: Run `launch_wan22.bat` from the installation directory

### Desktop UI Interface

#### Main Window Layout

- **Left Panel**: Generation controls and parameters
- **Right Panel**: Preview, output gallery, and queue management
- **Status Bar**: System status, GPU info, and progress indicators
- **Menu Bar**: File operations, model management, and settings

#### Generation Modes

**Text-to-Video (T2V)**

- Enter a text prompt describing the desired video
- Optional negative prompt to exclude unwanted elements
- Adjust duration (1-10 seconds)
- Select resolution (512x512 to 1920x1080)
- Set frame rate (24, 30, or 60 FPS)
- Configure inference steps and guidance scale

**Image-to-Video (I2V)**

- Upload an input image
- Describe the desired motion or animation
- Set duration (1-8 seconds)
- Adjust motion strength

**Text+Image-to-Video (TI2V)**

- Upload an input image
- Enter a text prompt describing the video
- Set duration (1-6 seconds)
- Adjust text influence on the generation

### System Requirements for Desktop UI

- Windows 10/11
- Python 3.9+ (installed automatically)
- 8GB+ RAM recommended
- GPU with 4GB+ VRAM (optional but recommended)
- Required Python packages:
  - tkinter (usually built-in)
  - Pillow
  - opencv-python

## Web UI

### Features

- **Browser-Based Interface**: Access through any modern web browser
- **Responsive Design**: Works on desktop and mobile devices
- **Same Generation Modes**: T2V, I2V, and TI2V support
- **Real-time Updates**: Live progress tracking and queue management
- **File Upload/Download**: Easy file management through the browser
- **Cross-Platform**: Works on any system with a web browser

### Launching the Web UI

1. **From Desktop**: Double-click the "WAN2.2 Web UI" shortcut
2. **From Start Menu**: Navigate to Start Menu → WAN2.2 → WAN2.2 Web UI
3. **Manual Launch**: Run `launch_web_ui.bat` from the installation directory
4. **Command Line**: `python application/web_ui.py`

### Web UI Interface

#### Accessing the Interface

- Default URL: `http://127.0.0.1:7860`
- The browser should open automatically when launched
- If not, manually navigate to the URL above

#### Web Interface Layout

- **Tab-Based Navigation**: Switch between T2V, I2V, TI2V, Queue, and Outputs
- **Status Bar**: Real-time system status and model information
- **Generation Forms**: Parameter controls for each generation mode
- **Progress Tracking**: Visual progress bars and queue status
- **Output Gallery**: Grid view of generated videos with download links

### System Requirements for Web UI

- Windows 10/11
- Python 3.9+ (installed automatically)
- Modern web browser (Chrome, Firefox, Edge, Safari)
- Required Python packages:
  - Flask
  - Werkzeug

## Generation Parameters

### Common Parameters

- **Duration**: Length of the generated video in seconds
- **Resolution**: Output video resolution (higher = better quality, slower generation)
- **Prompt**: Text description of the desired video content
- **Negative Prompt**: Text describing what to avoid in the generation

### Advanced Parameters (Desktop UI)

- **Inference Steps**: Number of denoising steps (more = higher quality, slower)
- **Guidance Scale**: How closely to follow the prompt (7.5 is typical)
- **Frame Rate**: Output video frame rate (24/30/60 FPS)
- **Motion Strength**: For I2V, controls how much the image should move
- **Text Influence**: For TI2V, balances text prompt vs. image content

## File Management

### Input Files

- **Supported Image Formats**: JPG, JPEG, PNG, BMP, TIFF
- **Recommended Image Size**: 512x512 to 1920x1080 pixels
- **File Size Limit**: 50MB per image (web UI)

### Output Files

- **Format**: MP4 video files
- **Location**: `outputs/` directory in the installation folder
- **Naming**: Automatic timestamped naming (e.g., `t2v_20250801_143022.mp4`)
- **Download**: Available through both UIs

## Troubleshooting

### Common Issues

**Desktop UI Won't Start**

- Ensure all dependencies are installed
- Check that the virtual environment is properly activated
- Verify Python and tkinter are available
- Check logs in the `logs/` directory

**Web UI Won't Start**

- Ensure Flask is installed: `pip install flask werkzeug`
- Check if port 7860 is available
- Try a different port: `python application/web_ui.py --port 8080`
- Check firewall settings

**Models Not Loading**

- Verify models are downloaded in the `models/` directory
- Check the Models menu in Desktop UI or status in Web UI
- Re-run the installation if models are missing
- Check available disk space

**Generation Fails**

- Ensure sufficient GPU memory (4GB+ recommended)
- Try reducing resolution or duration
- Check that all required models are present
- Monitor system resources during generation

**Slow Generation**

- Enable GPU acceleration if available
- Reduce inference steps for faster generation
- Lower resolution for quicker results
- Close other applications to free up resources

### Performance Optimization

**For Better Performance**:

- Use GPU acceleration when available
- Ensure adequate system RAM (16GB+ recommended)
- Use SSD storage for faster model loading
- Close unnecessary applications during generation
- Use lower resolutions for testing, higher for final output

**Hardware Recommendations**:

- **Minimum**: 8GB RAM, GTX 1060 6GB or equivalent
- **Recommended**: 16GB+ RAM, RTX 3070 8GB or better
- **Optimal**: 32GB+ RAM, RTX 4080 16GB or better

## Getting Help

### Resources

- **User Guide**: Check `GETTING_STARTED.md` in the installation directory
- **Troubleshooting**: See `docs/TROUBLESHOOTING.md` (if available)
- **Logs**: Check the `logs/` directory for detailed error information
- **System Info**: Use the "System Information" option in the Desktop UI

### Support

- Check the installation logs for specific error messages
- Ensure your system meets the minimum requirements
- Try reinstalling if persistent issues occur
- Verify all dependencies are properly installed

## Tips and Best Practices

### Prompt Writing

- **Be Specific**: Detailed prompts generally produce better results
- **Use Descriptive Language**: Include style, mood, and visual details
- **Avoid Contradictions**: Don't include conflicting instructions
- **Experiment**: Try different phrasings for the same concept

### Generation Settings

- **Start Simple**: Begin with shorter durations and lower resolutions
- **Iterate**: Refine prompts based on initial results
- **Balance Quality vs. Speed**: Higher settings = better quality but slower generation
- **Save Good Settings**: Note parameter combinations that work well

### File Organization

- **Organize Outputs**: Regularly clean up the outputs directory
- **Backup Important Results**: Copy good generations to a separate folder
- **Name Consistently**: Use descriptive names for your projects
- **Monitor Disk Space**: Video files can be large, especially at high resolutions

---

This guide covers the basic usage of both WAN2.2 user interfaces. For more advanced features and detailed technical information, refer to the additional documentation files in your installation directory.
