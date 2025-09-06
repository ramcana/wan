interface PerformanceMetric {
  name: string;
  value: number;
  timestamp: number;
  metadata?: Record<string, any>;
}

interface PerformanceThreshold {
  metric: string;
  warning: number;
  critical: number;
}

class PerformanceMonitor {
  private metrics: PerformanceMetric[] = [];
  private observers: PerformanceObserver[] = [];
  private thresholds: PerformanceThreshold[] = [
    { metric: 'FCP', warning: 2000, critical: 4000 }, // First Contentful Paint
    { metric: 'LCP', warning: 2500, critical: 4000 }, // Largest Contentful Paint
    { metric: 'FID', warning: 100, critical: 300 },   // First Input Delay
    { metric: 'CLS', warning: 0.1, critical: 0.25 },  // Cumulative Layout Shift
    { metric: 'TTFB', warning: 600, critical: 1000 }, // Time to First Byte
  ];

  constructor() {
    this.initializeObservers();
    this.monitorResourceUsage();
  }

  private initializeObservers() {
    // Web Vitals Observer
    if ('PerformanceObserver' in window) {
      // First Contentful Paint
      const fcpObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach((entry) => {
          if (entry.name === 'first-contentful-paint') {
            this.recordMetric('FCP', entry.startTime);
          }
        });
      });
      fcpObserver.observe({ entryTypes: ['paint'] });
      this.observers.push(fcpObserver);

      // Largest Contentful Paint
      const lcpObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        const lastEntry = entries[entries.length - 1];
        if (lastEntry) {
          this.recordMetric('LCP', lastEntry.startTime);
        }
      });
      lcpObserver.observe({ entryTypes: ['largest-contentful-paint'] });
      this.observers.push(lcpObserver);

      // First Input Delay
      const fidObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach((entry: any) => {
          this.recordMetric('FID', entry.processingStart - entry.startTime);
        });
      });
      fidObserver.observe({ entryTypes: ['first-input'] });
      this.observers.push(fidObserver);

      // Layout Shift
      const clsObserver = new PerformanceObserver((list) => {
        let clsValue = 0;
        const entries = list.getEntries();
        entries.forEach((entry: any) => {
          if (!entry.hadRecentInput) {
            clsValue += entry.value;
          }
        });
        this.recordMetric('CLS', clsValue);
      });
      clsObserver.observe({ entryTypes: ['layout-shift'] });
      this.observers.push(clsObserver);

      // Navigation Timing
      const navigationObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach((entry: any) => {
          this.recordMetric('TTFB', entry.responseStart - entry.requestStart);
          this.recordMetric('DOM_LOAD', entry.domContentLoadedEventEnd - entry.navigationStart);
          this.recordMetric('LOAD_COMPLETE', entry.loadEventEnd - entry.navigationStart);
        });
      });
      navigationObserver.observe({ entryTypes: ['navigation'] });
      this.observers.push(navigationObserver);
    }
  }

  private monitorResourceUsage() {
    // Monitor memory usage
    if ('memory' in performance) {
      setInterval(() => {
        const memory = (performance as any).memory;
        this.recordMetric('HEAP_USED', memory.usedJSHeapSize / 1024 / 1024); // MB
        this.recordMetric('HEAP_TOTAL', memory.totalJSHeapSize / 1024 / 1024); // MB
        this.recordMetric('HEAP_LIMIT', memory.jsHeapSizeLimit / 1024 / 1024); // MB
      }, 5000);
    }

    // Monitor frame rate
    let lastTime = performance.now();
    let frameCount = 0;

    const measureFPS = () => {
      frameCount++;
      const currentTime = performance.now();
      
      if (currentTime - lastTime >= 1000) {
        this.recordMetric('FPS', frameCount);
        frameCount = 0;
        lastTime = currentTime;
      }
      
      requestAnimationFrame(measureFPS);
    };
    requestAnimationFrame(measureFPS);
  }

  recordMetric(name: string, value: number, metadata?: Record<string, any>) {
    const metric: PerformanceMetric = {
      name,
      value,
      timestamp: Date.now(),
      metadata,
    };

    this.metrics.push(metric);

    // Keep only last 1000 metrics
    if (this.metrics.length > 1000) {
      this.metrics = this.metrics.slice(-1000);
    }

    // Check thresholds
    this.checkThresholds(metric);

    // Report to analytics
    this.reportToAnalytics(metric);
  }

  private checkThresholds(metric: PerformanceMetric) {
    const threshold = this.thresholds.find(t => t.metric === metric.name);
    if (!threshold) return;

    if (metric.value > threshold.critical) {
      console.error(`Critical performance issue: ${metric.name} = ${metric.value}ms (threshold: ${threshold.critical}ms)`);
      this.reportPerformanceIssue(metric, 'critical');
    } else if (metric.value > threshold.warning) {
      console.warn(`Performance warning: ${metric.name} = ${metric.value}ms (threshold: ${threshold.warning}ms)`);
      this.reportPerformanceIssue(metric, 'warning');
    }
  }

  private reportPerformanceIssue(metric: PerformanceMetric, severity: 'warning' | 'critical') {
    // Send to error reporting service
    if (window.gtag) {
      window.gtag('event', 'performance_issue', {
        metric_name: metric.name,
        metric_value: metric.value,
        severity,
        timestamp: metric.timestamp,
      });
    }

    // Custom event for internal handling
    window.dispatchEvent(new CustomEvent('performance-issue', {
      detail: { metric, severity }
    }));
  }

  private reportToAnalytics(metric: PerformanceMetric) {
    // Report to Google Analytics
    if (window.gtag) {
      window.gtag('event', 'performance_metric', {
        metric_name: metric.name,
        metric_value: metric.value,
        custom_map: metric.metadata,
      });
    }

    // Report to custom analytics endpoint
    if (process.env.NODE_ENV === 'production') {
      fetch('/api/v1/analytics/performance', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(metric),
      }).catch(console.error);
    }
  }

  getMetrics(name?: string, timeRange?: { start: number; end: number }): PerformanceMetric[] {
    let filtered = this.metrics;

    if (name) {
      filtered = filtered.filter(m => m.name === name);
    }

    if (timeRange) {
      filtered = filtered.filter(m => 
        m.timestamp >= timeRange.start && m.timestamp <= timeRange.end
      );
    }

    return filtered;
  }

  getAverageMetric(name: string, timeRange?: { start: number; end: number }): number {
    const metrics = this.getMetrics(name, timeRange);
    if (metrics.length === 0) return 0;

    const sum = metrics.reduce((acc, m) => acc + m.value, 0);
    return sum / metrics.length;
  }

  getPercentile(name: string, percentile: number, timeRange?: { start: number; end: number }): number {
    const metrics = this.getMetrics(name, timeRange);
    if (metrics.length === 0) return 0;

    const sorted = metrics.map(m => m.value).sort((a, b) => a - b);
    const index = Math.ceil((percentile / 100) * sorted.length) - 1;
    return sorted[index];
  }

  measureUserTiming(name: string, fn: () => void | Promise<void>): Promise<number> {
    return new Promise(async (resolve) => {
      const startTime = performance.now();
      
      try {
        await fn();
      } finally {
        const duration = performance.now() - startTime;
        this.recordMetric(`USER_TIMING_${name}`, duration);
        resolve(duration);
      }
    });
  }

  startMark(name: string) {
    performance.mark(`${name}_start`);
  }

  endMark(name: string) {
    performance.mark(`${name}_end`);
    performance.measure(name, `${name}_start`, `${name}_end`);
    
    const measure = performance.getEntriesByName(name, 'measure')[0];
    if (measure) {
      this.recordMetric(`MARK_${name}`, measure.duration);
    }
  }

  generateReport(): {
    summary: Record<string, number>;
    issues: Array<{ metric: string; severity: string; value: number }>;
    recommendations: string[];
  } {
    const now = Date.now();
    const oneHourAgo = now - 60 * 60 * 1000;

    const summary: Record<string, number> = {};
    const issues: Array<{ metric: string; severity: string; value: number }> = [];
    const recommendations: string[] = [];

    // Calculate summary metrics
    this.thresholds.forEach(threshold => {
      const avg = this.getAverageMetric(threshold.metric, { start: oneHourAgo, end: now });
      const p95 = this.getPercentile(threshold.metric, 95, { start: oneHourAgo, end: now });
      
      summary[`${threshold.metric}_avg`] = avg;
      summary[`${threshold.metric}_p95`] = p95;

      // Check for issues
      if (p95 > threshold.critical) {
        issues.push({ metric: threshold.metric, severity: 'critical', value: p95 });
      } else if (p95 > threshold.warning) {
        issues.push({ metric: threshold.metric, severity: 'warning', value: p95 });
      }
    });

    // Generate recommendations
    if (issues.some(i => i.metric === 'FCP' || i.metric === 'LCP')) {
      recommendations.push('Consider code splitting and lazy loading to improve initial load times');
    }
    if (issues.some(i => i.metric === 'FID')) {
      recommendations.push('Optimize JavaScript execution and consider using web workers for heavy computations');
    }
    if (issues.some(i => i.metric === 'CLS')) {
      recommendations.push('Ensure images and ads have defined dimensions to prevent layout shifts');
    }

    const avgHeapUsed = this.getAverageMetric('HEAP_USED', { start: oneHourAgo, end: now });
    if (avgHeapUsed > 100) { // 100MB
      recommendations.push('Memory usage is high. Consider optimizing data structures and cleaning up unused objects');
    }

    const avgFPS = this.getAverageMetric('FPS', { start: oneHourAgo, end: now });
    if (avgFPS < 30) {
      recommendations.push('Frame rate is low. Consider optimizing animations and reducing DOM manipulations');
    }

    return { summary, issues, recommendations };
  }

  destroy() {
    this.observers.forEach(observer => observer.disconnect());
    this.observers = [];
    this.metrics = [];
  }
}

export const performanceMonitor = new PerformanceMonitor();

// Global performance monitoring utilities
export const measureAsync = async <T>(name: string, fn: () => Promise<T>): Promise<T> => {
  const startTime = performance.now();
  try {
    const result = await fn();
    const duration = performance.now() - startTime;
    performanceMonitor.recordMetric(`ASYNC_${name}`, duration);
    return result;
  } catch (error) {
    const duration = performance.now() - startTime;
    performanceMonitor.recordMetric(`ASYNC_${name}_ERROR`, duration);
    throw error;
  }
};

export const measureSync = <T>(name: string, fn: () => T): T => {
  const startTime = performance.now();
  try {
    const result = fn();
    const duration = performance.now() - startTime;
    performanceMonitor.recordMetric(`SYNC_${name}`, duration);
    return result;
  } catch (error) {
    const duration = performance.now() - startTime;
    performanceMonitor.recordMetric(`SYNC_${name}_ERROR`, duration);
    throw error;
  }
};

// React hook for performance monitoring
export const usePerformanceMonitor = () => {
  return {
    recordMetric: performanceMonitor.recordMetric.bind(performanceMonitor),
    measureAsync,
    measureSync,
    startMark: performanceMonitor.startMark.bind(performanceMonitor),
    endMark: performanceMonitor.endMark.bind(performanceMonitor),
  };
};