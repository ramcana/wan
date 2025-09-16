---
category: deployment
last_updated: '2025-09-15T22:49:59.673957'
original_path: backend\core\README_PERFORMANCE_MONITORING.md
tags:
- configuration
- api
- troubleshooting
- installation
- security
- performance
title: Performance Monitoring System
---

# Performance Monitoring System

The performance monitoring system provides comprehensive tracking, analysis, and optimization recommendations for the real AI model integration.

## Overview

The system consists of several components:

1. **PerformanceMonitor**: Core monitoring engine
2. **Performance API**: REST endpoints for accessing metrics
3. **Performance Dashboard**: Interactive monitoring interface
4. **Integration**: Automatic tracking in generation service

## Features

### 📊 Comprehensive Metrics Collection

- **Generation Performance**: Time, success rate, model loading time
- **Resource Usage**: CPU, RAM, VRAM, disk usage
- **System Health**: Temperature, hardware utilization
- **Optimization Tracking**: Applied optimizations, quantization, offloading

### 🔍 Real-time Analysis

- **Bottleneck Detection**: Identifies VRAM, RAM, CPU, and model loading bottlenecks
- **Performance Trends**: Tracks performance changes over time
- **Resource Efficiency**: Calculates overall system efficiency
- **Success Rate Monitoring**: Tracks generation success rates

### 💡 Optimization Recommendations

- **Hardware Optimization**: GPU memory management, CPU utilization
- **Configuration Tuning**: Quantization, offloading, batch size recommendations
- **Workflow Improvements**: Model caching, loading optimization
- **Performance Targets**: Benchmarks against industry standards

### 📈 Performance Benchmarks

- **Generation Time Targets**:

  - 720p: 5 minutes (target), 7.5 minutes (acceptable)
  - 1080p: 15 minutes (target), 22.5 minutes (acceptable)

- **Resource Usage Targets**:

  - VRAM: 70% optimal, 85% acceptable, 95% maximum
  - RAM: 60% optimal, 75% acceptable, 90% maximum
  - CPU: 70% optimal, 85% acceptable, 95% maximum

- **Quality Targets**:
  - Success Rate: 98% target, 95% acceptable, 90% minimum
  - Resource Efficiency: 85% target, 70% acceptable, 50% minimum

## API Endpoints

### System Status

```http
GET /api/v1/performance/status
```

Returns current system performance status including CPU, RAM, GPU usage.

### Performance Analysis

```http
GET /api/v1/performance/analysis?time_window_hours=24
```

Returns comprehensive performance analysis for the specified time window.

### Performance Metrics

```http
GET /api/v1/performance/metrics?time_window_hours=24&model_type=t2v-A14B&success_only=true
```

Returns detailed performance metrics with optional filtering.

### Optimization Recommendations

```http
GET /api/v1/performance/recommendations?time_window_hours=24
```

Returns categorized optimization recommendations based on recent performance.

### Performance Benchmarks

```http
GET /api/v1/performance/benchmarks
```

Returns performance benchmarks and current performance against targets.

### Export Performance Data

```http
POST /api/v1/performance/export?time_window_hours=24&format=json
```

Exports performance data to JSON or CSV format.

## Performance Dashboard

### Interactive Mode

```bash
cd backend
python scripts/performance_dashboard.py
```

### Command Line Usage

```bash
# System status
python scripts/performance_dashboard.py status

# Performance analysis (24 hours)
python scripts/performance_dashboard.py analysis 24

# Recent tasks (last 10)
python scripts/performance_dashboard.py tasks 10

# Benchmarks
python scripts/performance_dashboard.py benchmarks

# Export data
python scripts/performance_dashboard.py export

# Continuous monitoring (10 second intervals)
python scripts/performance_dashboard.py monitor 10
```

## Integration with Generation Service

The performance monitoring is automatically integrated into the generation service:

### Automatic Task Tracking

```python
# Performance monitoring starts automatically when generation begins
performance_metrics = self.performance_monitor.start_task_monitoring(
    task_id=str(task.id),
    model_type=model_type,
    resolution=getattr(task, 'resolution', '720p'),
    steps=getattr(task, 'steps', 20)
)

# Metrics are updated during generation
self.performance_monitor.update_task_metrics(
    task_id, model_load_time_seconds=load_time
)

# Monitoring completes automatically when generation finishes
completed_metrics = self.performance_monitor.complete_task_monitoring(
    str(task.id), success=True
)
```

### Database Integration

Performance metrics are automatically stored in the database:

```sql
-- New columns added to generation_tasks table
ALTER TABLE generation_tasks ADD COLUMN model_used VARCHAR(100);
ALTER TABLE generation_tasks ADD COLUMN generation_time_seconds FLOAT DEFAULT 0.0;
ALTER TABLE generation_tasks ADD COLUMN peak_vram_usage_mb FLOAT DEFAULT 0.0;
ALTER TABLE generation_tasks ADD COLUMN optimizations_applied TEXT;
ALTER TABLE generation_tasks ADD COLUMN error_category VARCHAR(50);
ALTER TABLE generation_tasks ADD COLUMN recovery_suggestions TEXT;
```

## Configuration

### Performance Monitor Configuration

```json
{
  "performance_monitoring": {
    "monitoring_interval_seconds": 5.0,
    "max_history_size": 1000,
    "enable_gpu_monitoring": true,
    "performance_thresholds": {
      "max_generation_time_720p": 300,
      "max_generation_time_1080p": 900,
      "max_vram_usage_percent": 90,
      "max_ram_usage_percent": 85,
      "min_success_rate": 0.95
    }
  }
}
```

### Hardware Requirements

- **Minimum RAM**: 8GB (16GB+ recommended)
- **Minimum Disk**: 50GB free space for models
- **Minimum CPU**: 4 cores (8+ recommended)
- **GPU**: CUDA-compatible GPU recommended (optional)

## Optimization Recommendations

### VRAM Optimization

- Enable model quantization (fp16, int8)
- Use model offloading to CPU when not in use
- Reduce VAE tile size for lower memory usage
- Consider GPU memory upgrade if consistently hitting limits

### Performance Optimization

- Use SSD storage for faster model loading
- Keep frequently used models loaded in memory
- Enable model caching to avoid repeated loading
- Optimize CPU thread count for your hardware

### System Optimization

- Close unnecessary applications during generation
- Monitor system temperature to prevent throttling
- Use appropriate power management settings
- Regular system maintenance and updates

## Monitoring Best Practices

### Regular Monitoring

1. **Daily**: Check system status and recent performance
2. **Weekly**: Review performance analysis and trends
3. **Monthly**: Export and archive performance data

### Performance Tuning

1. **Baseline**: Establish performance baselines after setup
2. **Optimization**: Apply recommendations gradually
3. **Validation**: Measure impact of each optimization
4. **Documentation**: Keep records of configuration changes

### Troubleshooting

1. **Performance Issues**: Check bottleneck analysis
2. **Resource Problems**: Monitor resource usage trends
3. **Generation Failures**: Review error categories and recovery suggestions
4. **System Health**: Monitor temperature and hardware status

## Data Export and Analysis

### JSON Export Format

```json
{
  "export_timestamp": "2024-01-01T12:00:00",
  "time_window_hours": 24,
  "metrics_count": 150,
  "performance_analysis": {
    "average_generation_time": 245.5,
    "success_rate": 0.97,
    "resource_efficiency": 0.82,
    "bottleneck_analysis": {...},
    "optimization_recommendations": [...],
    "performance_trends": {...}
  },
  "metrics": [...]
}
```

### CSV Export Format

Includes all performance metrics in tabular format for analysis in spreadsheet applications or data analysis tools.

## Integration with External Tools

### Grafana Dashboard

The performance data can be integrated with Grafana for advanced visualization:

1. Export performance data to time-series database
2. Create Grafana dashboards for real-time monitoring
3. Set up alerts for performance thresholds

### Monitoring Systems

Integration with system monitoring tools:

1. **Prometheus**: Export metrics in Prometheus format
2. **InfluxDB**: Store time-series performance data
3. **Elasticsearch**: Index performance logs for analysis

## Security Considerations

### Data Privacy

- Performance metrics contain no user-generated content
- System information is limited to performance-relevant data
- Export files should be secured appropriately

### Access Control

- Performance API endpoints should be secured in production
- Dashboard access should be restricted to authorized users
- Export functionality should have appropriate permissions

## Troubleshooting

### Common Issues

1. **GPU Monitoring Not Working**

   - Ensure PyTorch is installed with CUDA support
   - Check CUDA drivers and compatibility
   - Verify GPU is accessible to the application

2. **High Memory Usage**

   - Check for memory leaks in monitoring code
   - Adjust history size limits
   - Monitor system memory availability

3. **Performance Data Missing**
   - Verify performance monitor initialization
   - Check for errors in generation service integration
   - Ensure database schema is up to date

### Debug Mode

Enable debug logging for detailed monitoring information:

```python
import logging
logging.getLogger('backend.core.performance_monitor').setLevel(logging.DEBUG)
```

## Future Enhancements

### Planned Features

1. **Machine Learning**: Predictive performance analysis
2. **Auto-tuning**: Automatic optimization parameter adjustment
3. **Distributed Monitoring**: Multi-node performance tracking
4. **Advanced Analytics**: Statistical analysis and forecasting

### Integration Opportunities

1. **Cloud Monitoring**: AWS CloudWatch, Azure Monitor integration
2. **APM Tools**: New Relic, Datadog integration
3. **Custom Dashboards**: React-based performance dashboard
4. **Mobile Monitoring**: Mobile app for performance monitoring
