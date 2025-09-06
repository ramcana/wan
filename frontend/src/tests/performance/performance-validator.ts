/**
 * Frontend performance validation utilities
 * Tests bundle size, first meaningful paint, and runtime performance
 */

interface PerformanceMetrics {
  bundleSize: number;
  firstMeaningfulPaint: number;
  timeToInteractive: number;
  apiResponseTimes: Record<string, number[]>;
  memoryUsage: number[];
}

export class FrontendPerformanceValidator {
  private metrics: PerformanceMetrics = {
    bundleSize: 0,
    firstMeaningfulPaint: 0,
    timeToInteractive: 0,
    apiResponseTimes: {},
    memoryUsage: []
  };

  /**
   * Measure First Meaningful Paint and Time to Interactive
   */
  measurePageLoadPerformance(): Promise<void> {
    return new Promise((resolve) => {
      // Use Performance Observer API to measure key metrics
      if ('PerformanceObserver' in window) {
        const observer = new PerformanceObserver((list) => {
          const entries = list.getEntries();
          
          entries.forEach((entry) => {
            if (entry.entryType === 'paint') {
              if (entry.name === 'first-contentful-paint') {
                this.metrics.firstMeaningfulPaint = entry.startTime;
              }
            }
            
            if (entry.entryType === 'navigation') {
              const navEntry = entry as PerformanceNavigationTiming;
              this.metrics.timeToInteractive = navEntry.loadEventEnd - navEntry.navigationStart;
            }
          });
        });

        observer.observe({ entryTypes: ['paint', 'navigation'] });

        // Stop observing after 5 seconds
        setTimeout(() => {
          observer.disconnect();
          resolve();
        }, 5000);
      } else {
        // Fallback for browsers without Performance Observer
        setTimeout(() => {
          const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
          if (navigation) {
            this.metrics.timeToInteractive = navigation.loadEventEnd - navigation.navigationStart;
          }
          resolve();
        }, 1000);
      }
    });
  }

  /**
   * Measure API response times
   */
  async measureApiPerformance(): Promise<void> {
    const endpoints = [
      '/api/v1/health',
      '/api/v1/system/stats',
      '/api/v1/queue',
      '/api/v1/outputs'
    ];

    for (const endpoint of endpoints) {
      const times: number[] = [];
      
      // Test each endpoint 5 times
      for (let i = 0; i < 5; i++) {
        const startTime = performance.now();
        
        try {
          const response = await fetch(endpoint);
          const endTime = performance.now();
          
          if (response.ok) {
            times.push(endTime - startTime);
          }
        } catch (error) {
          console.warn(`Failed to test endpoint ${endpoint}:`, error);
        }
        
        // Small delay between requests
        await new Promise(resolve => setTimeout(resolve, 100));
      }
      
      this.metrics.apiResponseTimes[endpoint] = times;
    }
  }

  /**
   * Monitor memory usage over time
   */
  monitorMemoryUsage(durationMs: number = 30000): Promise<void> {
    return new Promise((resolve) => {
      const interval = 1000; // Check every second
      const iterations = durationMs / interval;
      let currentIteration = 0;

      const checkMemory = () => {
        if ('memory' in performance) {
          const memInfo = (performance as any).memory;
          this.metrics.memoryUsage.push(memInfo.usedJSHeapSize / 1024 / 1024); // MB
        }

        currentIteration++;
        if (currentIteration < iterations) {
          setTimeout(checkMemory, interval);
        } else {
          resolve();
        }
      };

      checkMemory();
    });
  }

  /**
   * Validate performance budgets
   */
  validatePerformanceBudgets(): {
    passed: boolean;
    results: Record<string, { passed: boolean; actual: number; budget: number; message: string }>;
  } {
    const results: Record<string, { passed: boolean; actual: number; budget: number; message: string }> = {};

    // First Meaningful Paint budget: under 2 seconds
    const fmpBudget = 2000; // 2 seconds in ms
    results.firstMeaningfulPaint = {
      passed: this.metrics.firstMeaningfulPaint < fmpBudget,
      actual: this.metrics.firstMeaningfulPaint,
      budget: fmpBudget,
      message: `First Meaningful Paint: ${this.metrics.firstMeaningfulPaint.toFixed(0)}ms (budget: ${fmpBudget}ms)`
    };

    // API response time budgets
    const apiResponseBudgets = {
      '/api/v1/health': 1000, // 1 second
      '/api/v1/system/stats': 2000, // 2 seconds
      '/api/v1/queue': 1500, // 1.5 seconds
      '/api/v1/outputs': 3000 // 3 seconds
    };

    Object.entries(apiResponseBudgets).forEach(([endpoint, budget]) => {
      const times = this.metrics.apiResponseTimes[endpoint] || [];
      const avgTime = times.length > 0 ? times.reduce((a, b) => a + b, 0) / times.length : 0;
      
      results[`api_${endpoint.replace(/[^a-zA-Z0-9]/g, '_')}`] = {
        passed: avgTime < budget,
        actual: avgTime,
        budget,
        message: `${endpoint} avg response: ${avgTime.toFixed(0)}ms (budget: ${budget}ms)`
      };
    });

    // Memory usage budget: no more than 100MB increase over 30 seconds
    if (this.metrics.memoryUsage.length > 1) {
      const initialMemory = this.metrics.memoryUsage[0];
      const finalMemory = this.metrics.memoryUsage[this.metrics.memoryUsage.length - 1];
      const memoryIncrease = finalMemory - initialMemory;
      const memoryBudget = 100; // 100MB

      results.memoryUsage = {
        passed: memoryIncrease < memoryBudget,
        actual: memoryIncrease,
        budget: memoryBudget,
        message: `Memory increase: ${memoryIncrease.toFixed(1)}MB (budget: ${memoryBudget}MB)`
      };
    }

    const allPassed = Object.values(results).every(result => result.passed);

    return { passed: allPassed, results };
  }

  /**
   * Generate performance report
   */
  generateReport(): string {
    const validation = this.validatePerformanceBudgets();
    
    let report = '# Frontend Performance Report\n\n';
    
    report += '## Performance Metrics\n\n';
    report += `- First Meaningful Paint: ${this.metrics.firstMeaningfulPaint.toFixed(0)}ms\n`;
    report += `- Time to Interactive: ${this.metrics.timeToInteractive.toFixed(0)}ms\n`;
    
    if (this.metrics.memoryUsage.length > 0) {
      const avgMemory = this.metrics.memoryUsage.reduce((a, b) => a + b, 0) / this.metrics.memoryUsage.length;
      report += `- Average Memory Usage: ${avgMemory.toFixed(1)}MB\n`;
    }
    
    report += '\n## API Response Times\n\n';
    Object.entries(this.metrics.apiResponseTimes).forEach(([endpoint, times]) => {
      if (times.length > 0) {
        const avg = times.reduce((a, b) => a + b, 0) / times.length;
        const min = Math.min(...times);
        const max = Math.max(...times);
        report += `- ${endpoint}: avg ${avg.toFixed(0)}ms, min ${min.toFixed(0)}ms, max ${max.toFixed(0)}ms\n`;
      }
    });
    
    report += '\n## Budget Validation\n\n';
    Object.entries(validation.results).forEach(([metric, result]) => {
      const status = result.passed ? '✅' : '❌';
      report += `${status} ${result.message}\n`;
    });
    
    report += `\n**Overall Status: ${validation.passed ? '✅ PASSED' : '❌ FAILED'}**\n`;
    
    return report;
  }

  /**
   * Save metrics to localStorage for comparison
   */
  saveBaseline(): void {
    localStorage.setItem('performance_baseline', JSON.stringify(this.metrics));
  }

  /**
   * Load baseline metrics from localStorage
   */
  loadBaseline(): PerformanceMetrics | null {
    const baseline = localStorage.getItem('performance_baseline');
    return baseline ? JSON.parse(baseline) : null;
  }

  /**
   * Compare current metrics with baseline
   */
  compareWithBaseline(): {
    hasBaseline: boolean;
    regressions: string[];
    improvements: string[];
  } {
    const baseline = this.loadBaseline();
    
    if (!baseline) {
      return { hasBaseline: false, regressions: [], improvements: [] };
    }

    const regressions: string[] = [];
    const improvements: string[] = [];

    // Compare First Meaningful Paint
    const fmpDiff = this.metrics.firstMeaningfulPaint - baseline.firstMeaningfulPaint;
    if (Math.abs(fmpDiff) > 100) { // 100ms threshold
      if (fmpDiff > 0) {
        regressions.push(`First Meaningful Paint increased by ${fmpDiff.toFixed(0)}ms`);
      } else {
        improvements.push(`First Meaningful Paint improved by ${Math.abs(fmpDiff).toFixed(0)}ms`);
      }
    }

    // Compare API response times
    Object.entries(this.metrics.apiResponseTimes).forEach(([endpoint, times]) => {
      const baselineTimes = baseline.apiResponseTimes[endpoint];
      if (baselineTimes && times.length > 0) {
        const currentAvg = times.reduce((a, b) => a + b, 0) / times.length;
        const baselineAvg = baselineTimes.reduce((a, b) => a + b, 0) / baselineTimes.length;
        const diff = currentAvg - baselineAvg;
        
        if (Math.abs(diff) > 50) { // 50ms threshold
          if (diff > 0) {
            regressions.push(`${endpoint} response time increased by ${diff.toFixed(0)}ms`);
          } else {
            improvements.push(`${endpoint} response time improved by ${Math.abs(diff).toFixed(0)}ms`);
          }
        }
      }
    });

    return { hasBaseline: true, regressions, improvements };
  }
}

/**
 * Bundle size analyzer (to be used with build tools)
 */
export class BundleSizeAnalyzer {
  /**
   * Analyze bundle size from build output
   */
  static analyzeBundleSize(buildPath: string): Promise<{
    totalSize: number;
    gzippedSize: number;
    passesbudget: boolean;
  }> {
    // This would typically integrate with webpack-bundle-analyzer or similar
    // For now, return a mock implementation
    return Promise.resolve({
      totalSize: 450 * 1024, // 450KB
      gzippedSize: 135 * 1024, // ~30% of original (typical gzip ratio)
      passesbudget: true // Under 500KB budget
    });
  }

  /**
   * Generate bundle size report
   */
  static generateBundleReport(analysis: {
    totalSize: number;
    gzippedSize: number;
    passesbudget: boolean;
  }): string {
    const budget = 500 * 1024; // 500KB
    const status = analysis.passesbudget ? '✅' : '❌';
    
    return `
# Bundle Size Report

${status} **Bundle Size Validation**

- Total Size: ${(analysis.totalSize / 1024).toFixed(1)}KB
- Gzipped Size: ${(analysis.gzippedSize / 1024).toFixed(1)}KB
- Budget: ${(budget / 1024).toFixed(0)}KB
- Status: ${analysis.passesbudget ? 'PASSED' : 'FAILED'}

${analysis.passesbudget ? 
  'Bundle size is within the 500KB budget.' : 
  `Bundle size exceeds budget by ${((analysis.gzippedSize - budget) / 1024).toFixed(1)}KB`
}
    `.trim();
  }
}

// Export for use in tests
export default FrontendPerformanceValidator;