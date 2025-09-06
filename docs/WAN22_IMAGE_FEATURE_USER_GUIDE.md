# WAN22 Image Feature User Guide

## Overview

This guide covers the enhanced start and end image functionality in WAN22, including image upload, validation, progress tracking, and performance optimization features.

## Getting Started

### Model Types and Image Support

WAN22 supports three generation modes with different image requirements:

1. **Text-to-Video (T2V)**: Text prompts only, no image inputs
2. **Image-to-Video (I2V)**: Requires start image, text prompt optional
3. **Text-Image-to-Video (TI2V)**: Requires start image and text prompt, end image optional

### Supported Image Formats

- **PNG**: Recommended for best quality
- **JPEG/JPG**: Good for photographs
- **WebP**: Efficient compression, smaller file sizes

### Image Requirements

- **Minimum Size**: 256x256 pixels
- **Maximum Size**: 4096x4096 pixels (depending on available memory)
- **Aspect Ratios**: Any ratio supported, but consistent ratios recommended for start/end images
- **File Size**: Up to 50MB per image

## Using the Image Upload Feature

### Selecting Model Type

1. Choose your desired model type from the dropdown:

   - **t2v-A14B**: Text-to-video generation
   - **i2v-A14B**: Image-to-video generation
   - **ti2v-5B**: Text-image-to-video generation

2. The interface will automatically show/hide image upload controls based on your selection

### Uploading Images

#### Start Image (Required for I2V and TI2V)

1. Click the "Start Frame Image" upload area
2. Select your image file from the file browser
3. Wait for validation to complete
4. Review the thumbnail preview and validation status

#### End Image (Optional for TI2V)

1. Click the "End Frame Image" upload area
2. Select your image file from the file browser
3. The system will validate compatibility with the start image
4. Review the thumbnail preview and validation status

### Image Validation

The system automatically validates uploaded images:

#### Validation Checks

- **Format Validation**: Ensures supported file format
- **Size Validation**: Checks minimum and maximum dimensions
- **Compatibility Check**: Verifies start/end image compatibility
- **Memory Check**: Ensures image can be processed with available memory

#### Validation Messages

- **✅ Success**: Green checkmark with image dimensions and file size
- **⚠️ Warning**: Yellow warning with suggestions for optimization
- **❌ Error**: Red error with specific requirements and solutions

### Image Preview and Management

#### Thumbnail Previews

- Automatically generated for uploaded images
- Maintains original aspect ratio
- Shows image dimensions and file size on hover
- Click thumbnail for larger preview

#### Image Management

- **Clear Image**: Click the "×" button to remove an uploaded image
- **Replace Image**: Upload a new image to automatically replace the current one
- **Image Persistence**: Images are preserved when switching between compatible model types

## Resolution Settings

### Available Resolutions by Model Type

#### T2V and I2V Models (t2v-A14B, i2v-A14B)

- 1280×720 (HD)
- 1280×704 (HD Wide)
- 1920×1080 (Full HD)

#### TI2V Model (ti2v-5B)

- 1280×720 (HD)
- 1280×704 (HD Wide)
- 1920×1080 (Full HD)
- 1024×1024 (Square)

### Resolution Selection

1. The resolution dropdown automatically updates when you change model types
2. Select your desired output resolution
3. The system will validate compatibility with your uploaded images
4. If an incompatible resolution is selected, the system will suggest alternatives

## Progress Tracking

### Generation Progress

When you start video generation, you'll see:

#### Progress Bar

- Visual progress indicator (0-100%)
- Current step and total steps
- Estimated time remaining

#### Generation Statistics

- **Current Phase**: Initialization, Processing, or Encoding
- **Frames Processed**: Number of frames completed
- **Processing Speed**: Frames per second
- **Memory Usage**: Current memory consumption
- **GPU Utilization**: Graphics card usage (if available)

#### Performance Monitoring

The system continuously monitors performance and will:

- Adjust update frequency based on system load
- Display warnings for performance issues
- Provide optimization suggestions when needed

## Troubleshooting

### Common Image Issues

#### "Invalid Image Format"

**Problem**: Uploaded file is not a supported image format
**Solution**: Convert your image to PNG, JPEG, or WebP format

#### "Image Too Small"

**Problem**: Image dimensions are below 256×256 pixels
**Solution**: Resize your image to at least 256×256 pixels

#### "Image Too Large"

**Problem**: Image file size or dimensions exceed limits
**Solution**:

- Reduce image dimensions
- Compress the image file
- Use a more efficient format (WebP)

#### "Incompatible Aspect Ratios"

**Problem**: Start and end images have different aspect ratios
**Solution**:

- Crop images to match aspect ratios
- Use image editing software to pad images to same ratio
- Upload only a start image (end image is optional)

### Performance Issues

#### Slow Image Upload/Validation

**Symptoms**: Long delays when uploading images
**Solutions**:

- Use smaller image files
- Ensure stable internet connection
- Close other resource-intensive applications
- Try uploading images one at a time

#### High Memory Usage

**Symptoms**: System becomes slow, memory warnings
**Solutions**:

- Use smaller images
- Close other applications
- Restart the application
- Reduce cache size in settings

#### Slow Progress Updates

**Symptoms**: Progress bar updates infrequently or freezes
**Solutions**:

- The system automatically adjusts update frequency
- Check system resources (CPU, memory, GPU)
- Ensure adequate cooling for hardware
- Consider reducing generation resolution

### Error Messages and Solutions

#### "Failed to Load Image"

- Check file permissions
- Ensure file is not corrupted
- Try a different image file
- Restart the application

#### "Validation Timeout"

- Image file may be too large
- System may be under heavy load
- Try uploading a smaller image
- Wait and try again

#### "Memory Allocation Error"

- Insufficient system memory
- Close other applications
- Use smaller images
- Restart the application

## Advanced Features

### Performance Optimization

#### Automatic Caching

- The system automatically caches validation results and thumbnails
- Repeated operations are significantly faster (up to 25x speedup)
- Cache is managed automatically with LRU eviction
- Memory-aware caching prevents system overload

#### Enhanced Memory Management

- Real-time memory usage monitoring and optimization
- Automatic cleanup of unused data and cache entries
- Emergency optimizations for critical memory situations
- GPU memory tracking and optimization (when available)
- Adaptive cache sizing based on available system memory

#### Progress Optimization

- Update frequency adjusts dynamically to system performance
- Reduced overhead during high system load
- Emergency optimizations for critical performance issues
- Real-time performance monitoring with alerts
- Automatic optimization recommendations

#### Performance Monitoring

- Comprehensive performance profiling of image operations
- Bottleneck detection and optimization recommendations
- Real-time performance alerts and suggestions
- Detailed performance reports and metrics export
- Integration with system resource monitoring

### Customization Options

#### Cache Settings

Advanced users can adjust cache settings:

- Maximum cache memory usage (default: 256MB)
- Number of cached images (default: 50 entries)
- Cleanup intervals (default: 5 minutes)
- LRU eviction policy configuration

#### Performance Thresholds

Customize performance monitoring:

- Memory usage alert thresholds (warning: 75%, critical: 85%, emergency: 95%)
- CPU usage limits and monitoring intervals
- Update frequency preferences and adaptive intervals
- Performance profiling granularity settings

#### Memory Management

Configure memory optimization:

- Automatic memory threshold detection
- Emergency optimization triggers
- GPU memory monitoring (when available)
- Cache size optimization based on available memory
- Memory trend analysis and reporting

## Best Practices

### Image Preparation

1. **Use High-Quality Images**: Better input images produce better results
2. **Consistent Aspect Ratios**: Use matching ratios for start/end images
3. **Appropriate Resolution**: Higher resolution images may require more processing time
4. **Optimize File Size**: Use efficient formats and compression

### Performance Optimization

1. **Close Unnecessary Applications**: Free up system resources
2. **Use Adequate Hardware**: Ensure sufficient RAM and GPU memory
3. **Monitor System Performance**: Watch for performance warnings
4. **Regular Maintenance**: Restart application periodically to clear cache

### Generation Settings

1. **Start with Lower Resolutions**: Test with 720p before using 1080p
2. **Use End Images Sparingly**: Only when specific ending is required
3. **Monitor Progress**: Watch for performance issues during generation
4. **Save Frequently**: Save your work regularly

## FAQ

### Q: Can I use the same image for start and end frames?

A: Yes, but this may result in a static video. Use different images for dynamic results.

### Q: What happens if I upload incompatible images?

A: The system will show validation errors and suggest solutions.

### Q: Can I change images after starting generation?

A: No, images are locked once generation begins. You can change them for the next generation.

### Q: Why are my images taking long to validate?

A: Large images require more processing time. The system may also be optimizing performance.

### Q: Can I use images with different aspect ratios?

A: Yes, but the system will warn you about potential issues and may suggest adjustments.

### Q: How do I know if my system can handle large images?

A: The system monitors performance and will warn you if resources are insufficient.

## Support and Resources

### Getting Help

1. **Check Error Messages**: Most issues include specific solutions
2. **Review System Requirements**: Ensure your hardware meets minimum requirements
3. **Monitor Performance**: Use built-in performance monitoring
4. **Consult Logs**: Check application logs for detailed error information

### Performance Monitoring

Access performance information through:

- Real-time progress statistics
- Performance alerts and warnings
- Detailed performance reports
- System resource monitoring

### Additional Resources

- **WAN22 Performance Optimization Guide**: Detailed technical information
- **Troubleshooting Guide**: Comprehensive problem-solving resource
- **System Requirements**: Hardware and software requirements
- **Release Notes**: Latest features and improvements

## Conclusion

The WAN22 image feature provides powerful tools for creating videos from images with comprehensive validation, progress tracking, and performance optimization. By following this guide and best practices, you can achieve optimal results while maintaining good system performance.

For technical issues or advanced configuration, refer to the Performance Optimization Guide or consult the system documentation.
