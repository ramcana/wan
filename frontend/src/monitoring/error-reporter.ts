interface ErrorReport {
  id: string;
  message: string;
  stack?: string;
  timestamp: number;
  url: string;
  userAgent: string;
  userId?: string;
  sessionId: string;
  context: {
    component?: string;
    action?: string;
    props?: Record<string, any>;
    state?: Record<string, any>;
  };
  severity: 'low' | 'medium' | 'high' | 'critical';
  tags: string[];
  breadcrumbs: Breadcrumb[];
}

interface Breadcrumb {
  timestamp: number;
  category: 'navigation' | 'user' | 'api' | 'console' | 'error';
  message: string;
  data?: Record<string, any>;
  level: 'info' | 'warning' | 'error';
}

class ErrorReporter {
  private breadcrumbs: Breadcrumb[] = [];
  private sessionId: string;
  private userId?: string;
  private maxBreadcrumbs = 50;
  private reportQueue: ErrorReport[] = [];
  private isOnline = navigator.onLine;

  constructor() {
    this.sessionId = this.generateSessionId();
    this.setupGlobalErrorHandlers();
    this.setupNetworkMonitoring();
    this.startQueueProcessor();
  }

  private generateSessionId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private setupGlobalErrorHandlers() {
    // Unhandled JavaScript errors
    window.addEventListener('error', (event) => {
      this.reportError(event.error || new Error(event.message), {
        severity: 'high',
        context: {
          action: 'global_error',
        },
        tags: ['unhandled', 'javascript'],
      });
    });

    // Unhandled promise rejections
    window.addEventListener('unhandledrejection', (event) => {
      this.reportError(new Error(event.reason), {
        severity: 'high',
        context: {
          action: 'unhandled_promise_rejection',
        },
        tags: ['unhandled', 'promise'],
      });
    });

    // Console errors
    const originalConsoleError = console.error;
    console.error = (...args) => {
      this.addBreadcrumb({
        category: 'console',
        message: args.join(' '),
        level: 'error',
      });
      originalConsoleError.apply(console, args);
    };
  }

  private setupNetworkMonitoring() {
    // Monitor online/offline status
    window.addEventListener('online', () => {
      this.isOnline = true;
      this.addBreadcrumb({
        category: 'navigation',
        message: 'Connection restored',
        level: 'info',
      });
    });

    window.addEventListener('offline', () => {
      this.isOnline = false;
      this.addBreadcrumb({
        category: 'navigation',
        message: 'Connection lost',
        level: 'warning',
      });
    });

    // Intercept fetch requests for API monitoring
    const originalFetch = window.fetch;
    window.fetch = async (...args) => {
      const startTime = performance.now();
      const url = args[0] instanceof Request ? args[0].url : args[0];
      
      try {
        const response = await originalFetch(...args);
        const duration = performance.now() - startTime;
        
        this.addBreadcrumb({
          category: 'api',
          message: `${response.status} ${url}`,
          level: response.ok ? 'info' : 'warning',
          data: {
            method: args[1]?.method || 'GET',
            status: response.status,
            duration,
          },
        });

        // Report API errors
        if (!response.ok) {
          this.reportError(new Error(`API Error: ${response.status} ${response.statusText}`), {
            severity: response.status >= 500 ? 'high' : 'medium',
            context: {
              action: 'api_error',
              props: {
                url,
                method: args[1]?.method || 'GET',
                status: response.status,
                duration,
              },
            },
            tags: ['api', 'http_error'],
          });
        }

        return response;
      } catch (error) {
        const duration = performance.now() - startTime;
        
        this.addBreadcrumb({
          category: 'api',
          message: `Network error: ${url}`,
          level: 'error',
          data: {
            method: args[1]?.method || 'GET',
            duration,
            error: error.message,
          },
        });

        this.reportError(error as Error, {
          severity: 'high',
          context: {
            action: 'network_error',
            props: {
              url,
              method: args[1]?.method || 'GET',
              duration,
            },
          },
          tags: ['api', 'network_error'],
        });

        throw error;
      }
    };
  }

  private startQueueProcessor() {
    setInterval(() => {
      if (this.isOnline && this.reportQueue.length > 0) {
        this.processQueue();
      }
    }, 5000);
  }

  private async processQueue() {
    const reports = this.reportQueue.splice(0, 10); // Process up to 10 reports at a time
    
    try {
      await fetch('/api/v1/analytics/errors', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reports }),
      });
    } catch (error) {
      // Put reports back in queue if sending failed
      this.reportQueue.unshift(...reports);
      console.error('Failed to send error reports:', error);
    }
  }

  setUserId(userId: string) {
    this.userId = userId;
  }

  addBreadcrumb(breadcrumb: Omit<Breadcrumb, 'timestamp'>) {
    const fullBreadcrumb: Breadcrumb = {
      ...breadcrumb,
      timestamp: Date.now(),
    };

    this.breadcrumbs.push(fullBreadcrumb);

    // Keep only the most recent breadcrumbs
    if (this.breadcrumbs.length > this.maxBreadcrumbs) {
      this.breadcrumbs = this.breadcrumbs.slice(-this.maxBreadcrumbs);
    }
  }

  reportError(
    error: Error,
    options: {
      severity?: 'low' | 'medium' | 'high' | 'critical';
      context?: ErrorReport['context'];
      tags?: string[];
      userId?: string;
    } = {}
  ) {
    const report: ErrorReport = {
      id: `error_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      message: error.message,
      stack: error.stack,
      timestamp: Date.now(),
      url: window.location.href,
      userAgent: navigator.userAgent,
      userId: options.userId || this.userId,
      sessionId: this.sessionId,
      context: options.context || {},
      severity: options.severity || 'medium',
      tags: options.tags || [],
      breadcrumbs: [...this.breadcrumbs], // Copy breadcrumbs
    };

    // Add to queue for processing
    this.reportQueue.push(report);

    // Log to console in development
    if (process.env.NODE_ENV === 'development') {
      console.group(`🚨 Error Report (${report.severity})`);
      console.error('Error:', error);
      console.log('Context:', report.context);
      console.log('Tags:', report.tags);
      console.log('Breadcrumbs:', report.breadcrumbs);
      console.groupEnd();
    }

    // Send to external services
    this.sendToExternalServices(report);

    return report.id;
  }

  private sendToExternalServices(report: ErrorReport) {
    // Send to Google Analytics
    if (window.gtag) {
      window.gtag('event', 'exception', {
        description: report.message,
        fatal: report.severity === 'critical',
        custom_map: {
          error_id: report.id,
          component: report.context.component,
          action: report.context.action,
        },
      });
    }

    // Send to Sentry (if configured)
    if (window.Sentry) {
      window.Sentry.withScope((scope) => {
        scope.setTag('severity', report.severity);
        scope.setContext('error_context', report.context);
        scope.setUser({ id: report.userId });
        
        report.tags.forEach(tag => scope.setTag('custom_tag', tag));
        
        report.breadcrumbs.forEach(breadcrumb => {
          scope.addBreadcrumb({
            message: breadcrumb.message,
            category: breadcrumb.category,
            level: breadcrumb.level as any,
            data: breadcrumb.data,
            timestamp: breadcrumb.timestamp / 1000,
          });
        });

        window.Sentry.captureException(new Error(report.message));
      });
    }
  }

  // React-specific error reporting
  reportReactError(
    error: Error,
    errorInfo: { componentStack: string },
    component: string
  ) {
    this.reportError(error, {
      severity: 'high',
      context: {
        component,
        action: 'react_error',
        props: {
          componentStack: errorInfo.componentStack,
        },
      },
      tags: ['react', 'component_error'],
    });
  }

  // User action tracking
  trackUserAction(action: string, data?: Record<string, any>) {
    this.addBreadcrumb({
      category: 'user',
      message: `User action: ${action}`,
      level: 'info',
      data,
    });
  }

  // Performance issue reporting
  reportPerformanceIssue(metric: string, value: number, threshold: number) {
    this.reportError(new Error(`Performance issue: ${metric} = ${value}ms (threshold: ${threshold}ms)`), {
      severity: value > threshold * 2 ? 'high' : 'medium',
      context: {
        action: 'performance_issue',
        props: {
          metric,
          value,
          threshold,
          ratio: value / threshold,
        },
      },
      tags: ['performance', metric.toLowerCase()],
    });
  }

  // Get error statistics
  getErrorStats(timeRange?: { start: number; end: number }) {
    let reports = this.reportQueue;

    if (timeRange) {
      reports = reports.filter(r => 
        r.timestamp >= timeRange.start && r.timestamp <= timeRange.end
      );
    }

    const stats = {
      total: reports.length,
      bySeverity: {
        low: reports.filter(r => r.severity === 'low').length,
        medium: reports.filter(r => r.severity === 'medium').length,
        high: reports.filter(r => r.severity === 'high').length,
        critical: reports.filter(r => r.severity === 'critical').length,
      },
      byTag: {} as Record<string, number>,
      byComponent: {} as Record<string, number>,
    };

    // Count by tags
    reports.forEach(report => {
      report.tags.forEach(tag => {
        stats.byTag[tag] = (stats.byTag[tag] || 0) + 1;
      });

      if (report.context.component) {
        stats.byComponent[report.context.component] = 
          (stats.byComponent[report.context.component] || 0) + 1;
      }
    });

    return stats;
  }

  // Clear old data
  clearOldData(olderThan: number = 24 * 60 * 60 * 1000) { // 24 hours
    const cutoff = Date.now() - olderThan;
    this.reportQueue = this.reportQueue.filter(r => r.timestamp > cutoff);
    this.breadcrumbs = this.breadcrumbs.filter(b => b.timestamp > cutoff);
  }
}

export const errorReporter = new ErrorReporter();

// React Error Boundary integration
export const reportReactError = (error: Error, errorInfo: any, component: string) => {
  errorReporter.reportReactError(error, errorInfo, component);
};

// User action tracking
export const trackUserAction = (action: string, data?: Record<string, any>) => {
  errorReporter.trackUserAction(action, data);
};

// Performance issue reporting
export const reportPerformanceIssue = (metric: string, value: number, threshold: number) => {
  errorReporter.reportPerformanceIssue(metric, value, threshold);
};

// Hook for React components
export const useErrorReporter = () => {
  return {
    reportError: errorReporter.reportError.bind(errorReporter),
    addBreadcrumb: errorReporter.addBreadcrumb.bind(errorReporter),
    trackUserAction,
  };
};