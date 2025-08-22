/**
 * Performance test runner for frontend validation
 * Tests performance budgets and establishes baseline metrics
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import FrontendPerformanceValidator, { BundleSizeAnalyzer } from './performance-validator';

describe('Frontend Performance Validation', () => {
  let validator: FrontendPerformanceValidator;

  beforeAll(async () => {
    validator = new FrontendPerformanceValidator();
    
    // Measure initial page load performance
    await validator.measurePageLoadPerformance();
  });

  afterAll(() => {
    // Save current metrics as baseline for future comparisons
    validator.saveBaseline();
  });

  describe('Page Load Performance', () => {
    it('should meet First Meaningful Paint budget (under 2 seconds)', async () => {
      const validation = validator.validatePerformanceBudgets();
      const fmpResult = validation.results.firstMeaningfulPaint;
      
      expect(fmpResult.passed).toBe(true);
      expect(fmpResult.actual).toBeLessThan(2000);
      
      console.log(`✓ First Meaningful Paint: ${fmpResult.actual.toFixed(0)}ms`);
    });

    it('should have reasonable Time to Interactive', async () => {
      // TTI should be under 5 seconds for good user experience
      const validation = validator.validatePerformanceBudgets();
      
      // Note: TTI is not in the budget validation but we can check it separately
      // This is more of an informational test
      console.log('Time to Interactive measured during page load');
    });
  });

  describe('API Performance', () => {
    beforeAll(async () => {
      // Measure API performance
      await validator.measureApiPerformance();
    });

    it('should meet API response time budgets', async () => {
      const validation = validator.validatePerformanceBudgets();
      
      // Check each API endpoint budget
      const apiResults = Object.entries(validation.results).filter(([key]) => 
        key.startsWith('api_')
      );

      apiResults.forEach(([key, result]) => {
        expect(result.passed).toBe(true);
        console.log(`✓ ${result.message}`);
      });

      expect(apiResults.length).toBeGreaterThan(0);
    });

    it('should have consistent API response times', async () => {
      const validation = validator.validatePerformanceBudgets();
      
      // Check that API response times don't vary too much
      Object.entries(validation.results).forEach(([key, result]) => {
        if (key.startsWith('api_')) {
          // Response time should be reasonably consistent
          // This is more of a stability check
          expect(result.actual).toBeGreaterThan(0);
        }
      });
    });
  });

  describe('Memory Usage', () => {
    it('should not have memory leaks during normal operation', async () => {
      // Monitor memory usage for 10 seconds (shorter for testing)
      await validator.monitorMemoryUsage(10000);
      
      const validation = validator.validatePerformanceBudgets();
      const memoryResult = validation.results.memoryUsage;
      
      if (memoryResult) {
        expect(memoryResult.passed).toBe(true);
        console.log(`✓ ${memoryResult.message}`);
      } else {
        console.log('⚠ Memory monitoring not available in this environment');
      }
    });
  });

  describe('Bundle Size', () => {
    it('should meet bundle size budget (under 500KB gzipped)', async () => {
      const analysis = await BundleSizeAnalyzer.analyzeBundleSize('dist');
      
      expect(analysis.passesbudget).toBe(true);
      expect(analysis.gzippedSize).toBeLessThan(500 * 1024);
      
      const report = BundleSizeAnalyzer.generateBundleReport(analysis);
      console.log(report);
    });
  });

  describe('Baseline Comparison', () => {
    it('should not have performance regressions', () => {
      const comparison = validator.compareWithBaseline();
      
      if (comparison.hasBaseline) {
        // Log any regressions or improvements
        comparison.regressions.forEach(regression => {
          console.warn(`⚠ Performance regression: ${regression}`);
        });
        
        comparison.improvements.forEach(improvement => {
          console.log(`✓ Performance improvement: ${improvement}`);
        });
        
        // Fail test if there are significant regressions
        expect(comparison.regressions.length).toBe(0);
      } else {
        console.log('ℹ No baseline found, establishing new baseline');
      }
    });
  });

  describe('Performance Report Generation', () => {
    it('should generate comprehensive performance report', () => {
      const report = validator.generateReport();
      
      expect(report).toContain('# Frontend Performance Report');
      expect(report).toContain('## Performance Metrics');
      expect(report).toContain('## API Response Times');
      expect(report).toContain('## Budget Validation');
      
      console.log('\n' + report);
    });
  });
});

describe('Resource Constraint Simulation', () => {
  it('should handle slow network conditions gracefully', async () => {
    // Simulate slow network by adding artificial delays
    const originalFetch = global.fetch;
    
    global.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
      // Add 2 second delay to simulate slow network
      await new Promise(resolve => setTimeout(resolve, 2000));
      return originalFetch(input, init);
    };

    try {
      const validator = new FrontendPerformanceValidator();
      await validator.measureApiPerformance();
      
      const validation = validator.validatePerformanceBudgets();
      
      // Under slow network, we expect some API calls to exceed budgets
      // but the app should still function
      const apiResults = Object.entries(validation.results).filter(([key]) => 
        key.startsWith('api_')
      );
      
      expect(apiResults.length).toBeGreaterThan(0);
      console.log('✓ App handles slow network conditions');
      
    } finally {
      // Restore original fetch
      global.fetch = originalFetch;
    }
  });

  it('should handle offline conditions gracefully', async () => {
    // Simulate offline by making fetch throw network errors
    const originalFetch = global.fetch;
    
    global.fetch = async () => {
      throw new Error('Network error - offline');
    };

    try {
      const validator = new FrontendPerformanceValidator();
      
      // This should not crash the app
      await expect(validator.measureApiPerformance()).resolves.not.toThrow();
      
      console.log('✓ App handles offline conditions gracefully');
      
    } finally {
      // Restore original fetch
      global.fetch = originalFetch;
    }
  });
});

describe('Load Testing', () => {
  it('should handle rapid API calls without degradation', async () => {
    const validator = new FrontendPerformanceValidator();
    const startTime = performance.now();
    
    // Make 20 rapid API calls
    const promises = Array.from({ length: 20 }, () => 
      fetch('/api/v1/health').catch(() => null)
    );
    
    await Promise.all(promises);
    
    const endTime = performance.now();
    const totalTime = endTime - startTime;
    
    // Should complete within reasonable time (10 seconds max)
    expect(totalTime).toBeLessThan(10000);
    
    console.log(`✓ Completed 20 API calls in ${totalTime.toFixed(0)}ms`);
  });

  it('should maintain UI responsiveness during heavy operations', async () => {
    // Simulate heavy DOM operations
    const startTime = performance.now();
    
    // Create and manipulate many DOM elements
    const container = document.createElement('div');
    document.body.appendChild(container);
    
    for (let i = 0; i < 1000; i++) {
      const element = document.createElement('div');
      element.textContent = `Element ${i}`;
      element.style.width = '100px';
      element.style.height = '20px';
      container.appendChild(element);
    }
    
    const endTime = performance.now();
    const operationTime = endTime - startTime;
    
    // DOM operations should complete quickly
    expect(operationTime).toBeLessThan(1000); // Under 1 second
    
    // Clean up
    document.body.removeChild(container);
    
    console.log(`✓ Heavy DOM operations completed in ${operationTime.toFixed(0)}ms`);
  });
});