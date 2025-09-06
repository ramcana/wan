# Enhanced Model Availability - User Guide

## Overview

The Enhanced Model Availability system provides intelligent model management, automatic download retry mechanisms, and improved user experience when working with AI models. This guide covers all user-facing features and how to use them effectively.

## Key Features

### 1. Automatic Model Download Management

- **Intelligent Retry**: Failed downloads automatically retry up to 3 times with exponential backoff
- **Partial Download Recovery**: Resume interrupted downloads from where they left off
- **Download Controls**: Pause, resume, or cancel downloads as needed
- **Bandwidth Management**: Set download speed limits to manage network usage

### 2. Model Health Monitoring

- **Integrity Checking**: Automatic verification of model file integrity
- **Corruption Detection**: Identifies and repairs corrupted model files
- **Performance Monitoring**: Tracks model generation performance over time
- **Proactive Maintenance**: Scheduled health checks and automatic repairs

### 3. Intelligent Fallback System

- **Smart Alternatives**: Suggests compatible models when preferred ones are unavailable
- **Queue Management**: Queue generation requests while models download
- **Wait Time Estimates**: Provides accurate download completion estimates
- **Graceful Degradation**: Clear guidance when falling back to mock generation

### 4. Usage Analytics

- **Usage Tracking**: Monitor which models you use most frequently
- **Storage Optimization**: Recommendations for freeing up disk space
- **Performance Insights**: Identify models with performance issues
- **Preload Suggestions**: Automatically prepare frequently used models

## Getting Started

### Checking Model Status

1. **Via Web Interface**:

   - Navigate to the Model Management section
   - View comprehensive status for all models
   - See download progress, integrity status, and availability

2. **Via API**:
   ```bash
   curl http://localhost:8000/api/v1/models/status/detailed
   ```

### Managing Downloads

#### Starting a Download

- Models download automatically when first requested
- Manual downloads can be initiated from the Model Management interface
- Priority can be set for multiple simultaneous downloads

#### Controlling Downloads

- **Pause**: Click the pause button or use the API endpoint
- **Resume**: Resume paused downloads with a single click
- **Cancel**: Stop unwanted downloads and free up bandwidth
- **Set Priority**: Reorder download queue based on your needs

#### Bandwidth Management

1. Go to Settings → Model Management
2. Set "Download Speed Limit" (in MB/s)
3. Enable "Pause downloads during generation" if needed

### Understanding Model Health

#### Health Indicators

- **Green**: Model is healthy and ready for use
- **Yellow**: Minor issues detected, automatic repair in progress
- **Red**: Significant problems, manual intervention may be needed

#### Health Actions

- **Auto-Repair**: System automatically fixes detected issues
- **Manual Repair**: Force integrity check and repair
- **Re-download**: Download fresh copy if corruption is severe

### Using Intelligent Fallbacks

#### When Models Are Unavailable

1. System suggests compatible alternatives
2. Shows compatibility score and expected quality difference
3. Provides options to:
   - Use suggested alternative immediately
   - Queue request and wait for preferred model
   - Download preferred model in background

#### Fallback Strategies

- **Alternative Model**: Use similar available model
- **Queue and Wait**: Wait for model to download
- **Mock Generation**: Use placeholder generation with upgrade path
- **Download and Retry**: Trigger download and retry automatically

## Advanced Features

### Model Update Management

#### Automatic Updates

- System checks for model updates periodically
- Notifications appear when updates are available
- Updates download in background without interrupting current models

#### Manual Updates

1. Go to Model Management
2. Click "Check for Updates"
3. Review available updates
4. Select models to update
5. Updates download and install automatically

#### Update Rollback

- Previous versions kept until new version verified
- Automatic rollback if new version fails
- Manual rollback option in Model Management

### Usage Analytics Dashboard

#### Viewing Analytics

1. Navigate to Analytics → Model Usage
2. View usage statistics by time period
3. See performance trends and recommendations

#### Key Metrics

- **Usage Frequency**: How often each model is used
- **Performance Scores**: Generation speed and quality metrics
- **Storage Impact**: Disk space used by each model
- **Success Rates**: Reliability statistics for each model

#### Optimization Recommendations

- **Cleanup Suggestions**: Models safe to remove
- **Preload Recommendations**: Models to keep ready
- **Performance Alerts**: Models with degrading performance

### Storage Management

#### Automatic Cleanup

- System suggests unused models for removal
- Configurable retention policies
- Safe cleanup with usage history consideration

#### Manual Management

1. Go to Model Management → Storage
2. View storage usage by model
3. Select models to remove
4. Confirm cleanup operation

#### Storage Policies

- **Keep Recent**: Retain models used in last N days
- **Keep Frequent**: Retain models used more than N times
- **Size Limits**: Remove largest models when space needed

## Troubleshooting

### Common Issues

#### Download Failures

**Problem**: Model download keeps failing
**Solutions**:

1. Check internet connection stability
2. Verify sufficient disk space
3. Try downloading during off-peak hours
4. Contact support if issue persists

#### Model Corruption

**Problem**: Model generates poor quality or fails
**Solutions**:

1. Run integrity check from Model Management
2. Use auto-repair function
3. Re-download model if corruption is severe
4. Check system logs for hardware issues

#### Performance Issues

**Problem**: Model generation is slow or unreliable
**Solutions**:

1. Check model health status
2. Review system resource usage
3. Consider using alternative model temporarily
4. Update to latest model version

#### Storage Full

**Problem**: Not enough space for new models
**Solutions**:

1. Use cleanup recommendations
2. Remove unused models manually
3. Move models to external storage
4. Upgrade storage capacity

### Getting Help

#### Built-in Help

- Hover over any interface element for tooltips
- Click "?" icons for contextual help
- Check status messages for guidance

#### Log Files

- Model operations logged to `logs/model_management.log`
- Download progress in `logs/downloads.log`
- Health monitoring in `logs/health_monitor.log`

#### Support Resources

- Check troubleshooting guide for common issues
- Review API documentation for advanced usage
- Contact support with log files for complex problems

## Best Practices

### Download Management

1. **Prioritize**: Download most-used models first
2. **Schedule**: Use off-peak hours for large downloads
3. **Monitor**: Keep an eye on download progress
4. **Bandwidth**: Set appropriate speed limits

### Model Health

1. **Regular Checks**: Enable automatic health monitoring
2. **Quick Response**: Address health issues promptly
3. **Preventive**: Keep models updated to latest versions
4. **Backup**: Consider backing up frequently used models

### Storage Optimization

1. **Regular Cleanup**: Remove unused models monthly
2. **Usage Tracking**: Monitor which models you actually use
3. **Size Awareness**: Consider model sizes when downloading
4. **External Storage**: Use for infrequently used models

### Performance Optimization

1. **Model Selection**: Choose appropriate models for tasks
2. **Resource Monitoring**: Watch system resource usage
3. **Update Management**: Keep models updated
4. **Alternative Planning**: Have backup models ready

## Configuration Options

### User Preferences

- **Auto-download**: Enable automatic model downloads
- **Retry Settings**: Configure retry attempts and delays
- **Notification Level**: Set alert preferences
- **Cleanup Policy**: Define automatic cleanup rules

### Advanced Settings

- **Download Concurrency**: Number of simultaneous downloads
- **Health Check Frequency**: How often to check model health
- **Analytics Collection**: Enable/disable usage tracking
- **Fallback Behavior**: Configure fallback preferences

## API Integration

### Basic Usage

```python
import requests

# Check model status
response = requests.get('http://localhost:8000/api/v1/models/status/detailed')
models = response.json()

# Start download with retry
requests.post('http://localhost:8000/api/v1/models/download/manage', {
    'model_id': 'my-model',
    'action': 'start',
    'max_retries': 3
})

# Get usage analytics
analytics = requests.get('http://localhost:8000/api/v1/models/analytics')
```

### WebSocket Integration

```javascript
const ws = new WebSocket("ws://localhost:8000/ws");

ws.onmessage = function (event) {
  const data = JSON.parse(event.data);
  if (data.type === "model_download_progress") {
    updateProgressBar(data.model_id, data.progress);
  }
};
```

## Conclusion

The Enhanced Model Availability system provides comprehensive model management with intelligent automation and user-friendly controls. By following this guide and best practices, you can ensure optimal model availability and performance for your AI generation needs.

For additional help, consult the troubleshooting guide, API documentation, or contact support.
