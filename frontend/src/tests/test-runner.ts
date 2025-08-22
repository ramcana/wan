import { describe, it, expect, beforeAll, afterAll, beforeEach, afterEach } from 'vitest';
import { performanceMonitor } from '../monitoring/performance-monitor';
import { errorReporter } from '../monitoring/error-reporter';
import { userJourneyTracker } from '../monitoring/user-journey-tracker';

interface TestSuite {
  name: string;
  tests: TestCase[];
  setup?: () => void | Promise<void>;
  teardown?: () => void | Promise<void>;
}

interface TestCase {
  name: string;
  fn: () => void | Promise<void>;
  timeout?: number;
  skip?: boolean;
  only?: boolean;
}

interface TestResult {
  suite: string;
  test: string;
  status: 'passed' | 'failed' | 'skipped';
  duration: number;
  error?: Error;
  performance?: {
    memory: number;
    timing: number;
  };
}

class ComprehensiveTestRunner {
  private results: TestResult[] = [];
  private suites: TestSuite[] = [];
  private startTime: number = 0;
  private endTime: number = 0;

  constructor() {
    this.setupGlobalTestEnvironment();
  }

  private setupGlobalTestEnvironment() {
    // Setup global test utilities
    (global as any).testUtils = {
      performanceMonitor,
      errorReporter,
      userJourneyTracker,
      measurePerformance: this.measurePerformance.bind(this),
      expectNoMemoryLeaks: this.expectNoMemoryLeaks.bind(this),
      expectPerformanceWithin: this.expectPerformanceWithin.bind(this),
    };

    // Setup test-specific error handling
    errorReporter.setUserId('test-user');
    
    // Mock browser APIs for testing
    this.mockBrowserAPIs();
  }

  private mockBrowserAPIs() {
    // Mock IntersectionObserver
    (global as any).IntersectionObserver = class MockIntersectionObserver {
      observe = vi.fn();
      unobserve = vi.fn();
      disconnect = vi.fn();
    };

    // Mock ResizeObserver
    (global as any).ResizeObserver = class MockResizeObserver {
      observe = vi.fn();
      unobserve = vi.fn();
      disconnect = vi.fn();
    };

    // Mock PerformanceObserver
    (global as any).PerformanceObserver = class MockPerformanceObserver {
      observe = vi.fn();
      disconnect = vi.fn();
    };

    // Mock WebSocket
    (global as any).WebSocket = class MockWebSocket {
      readyState = 1; // OPEN
      send = vi.fn();
      close = vi.fn();
      addEventListener = vi.fn();
      removeEventListener = vi.fn();
    };

    // Mock Notification API
    (global as any).Notification = class MockNotification {
      static permission = 'granted';
      static requestPermission = vi.fn().mockResolvedValue('granted');
      constructor(title: string, options?: NotificationOptions) {
        // Mock notification
      }
    };
  }

  addSuite(suite: TestSuite) {
    this.suites.push(suite);
  }

  async runAllTests(): Promise<{
    results: TestResult[];
    summary: {
      total: number;
      passed: number;
      failed: number;
      skipped: number;
      duration: number;
    };
    performance: {
      averageTestTime: number;
      slowestTest: TestResult;
      memoryUsage: number;
    };
  }> {
    this.startTime = performance.now();
    this.results = [];

    console.log('🚀 Starting comprehensive test suite...\n');

    for (const suite of this.suites) {
      await this.runSuite(suite);
    }

    this.endTime = performance.now();
    const totalDuration = this.endTime - this.startTime;

    const summary = this.generateSummary(totalDuration);
    const performanceReport = this.generatePerformanceReport();

    this.printResults(summary, performanceReport);

    return {
      results: this.results,
      summary,
      performance: performanceReport,
    };
  }

  private async runSuite(suite: TestSuite) {
    console.log(`📦 Running suite: ${suite.name}`);

    try {
      // Run suite setup
      if (suite.setup) {
        await suite.setup();
      }

      // Run each test in the suite
      for (const test of suite.tests) {
        if (test.skip) {
          this.results.push({
            suite: suite.name,
            test: test.name,
            status: 'skipped',
            duration: 0,
          });
          console.log(`  ⏭️  ${test.name} (skipped)`);
          continue;
        }

        await this.runTest(suite.name, test);
      }

      // Run suite teardown
      if (suite.teardown) {
        await suite.teardown();
      }

    } catch (error) {
      console.error(`❌ Suite setup/teardown failed: ${suite.name}`, error);
    }

    console.log(''); // Empty line between suites
  }

  private async runTest(suiteName: string, test: TestCase) {
    const startTime = performance.now();
    const initialMemory = (performance as any).memory?.usedJSHeapSize || 0;

    try {
      // Set timeout if specified
      const timeout = test.timeout || 5000;
      const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => reject(new Error(`Test timeout: ${timeout}ms`)), timeout);
      });

      // Run the test
      await Promise.race([test.fn(), timeoutPromise]);

      const endTime = performance.now();
      const finalMemory = (performance as any).memory?.usedJSHeapSize || 0;
      const duration = endTime - startTime;

      this.results.push({
        suite: suiteName,
        test: test.name,
        status: 'passed',
        duration,
        performance: {
          memory: finalMemory - initialMemory,
          timing: duration,
        },
      });

      console.log(`  ✅ ${test.name} (${duration.toFixed(2)}ms)`);

    } catch (error) {
      const endTime = performance.now();
      const duration = endTime - startTime;

      this.results.push({
        suite: suiteName,
        test: test.name,
        status: 'failed',
        duration,
        error: error as Error,
      });

      console.log(`  ❌ ${test.name} (${duration.toFixed(2)}ms)`);
      console.log(`     Error: ${(error as Error).message}`);

      // Report error to monitoring system
      errorReporter.reportError(error as Error, {
        severity: 'medium',
        context: {
          component: 'test-runner',
          action: 'test-execution',
          props: { suite: suiteName, test: test.name },
        },
        tags: ['test', 'failure'],
      });
    }
  }

  private generateSummary(totalDuration: number) {
    const total = this.results.length;
    const passed = this.results.filter(r => r.status === 'passed').length;
    const failed = this.results.filter(r => r.status === 'failed').length;
    const skipped = this.results.filter(r => r.status === 'skipped').length;

    return {
      total,
      passed,
      failed,
      skipped,
      duration: totalDuration,
    };
  }

  private generatePerformanceReport() {
    const completedTests = this.results.filter(r => r.status !== 'skipped');
    const totalTestTime = completedTests.reduce((sum, r) => sum + r.duration, 0);
    const averageTestTime = totalTestTime / completedTests.length;
    
    const slowestTest = completedTests.reduce((slowest, current) => 
      current.duration > slowest.duration ? current : slowest
    );

    const totalMemoryUsage = completedTests.reduce((sum, r) => 
      sum + (r.performance?.memory || 0), 0
    );

    return {
      averageTestTime,
      slowestTest,
      memoryUsage: totalMemoryUsage,
    };
  }

  private printResults(summary: any, performance: any) {
    console.log('\n📊 Test Results Summary');
    console.log('========================');
    console.log(`Total Tests: ${summary.total}`);
    console.log(`✅ Passed: ${summary.passed}`);
    console.log(`❌ Failed: ${summary.failed}`);
    console.log(`⏭️  Skipped: ${summary.skipped}`);
    console.log(`⏱️  Duration: ${summary.duration.toFixed(2)}ms`);
    
    console.log('\n🚀 Performance Report');
    console.log('=====================');
    console.log(`Average Test Time: ${performance.averageTestTime.toFixed(2)}ms`);
    console.log(`Slowest Test: ${performance.slowestTest.test} (${performance.slowestTest.duration.toFixed(2)}ms)`);
    console.log(`Total Memory Usage: ${(performance.memoryUsage / 1024 / 1024).toFixed(2)}MB`);

    // Print failed tests
    const failedTests = this.results.filter(r => r.status === 'failed');
    if (failedTests.length > 0) {
      console.log('\n❌ Failed Tests');
      console.log('===============');
      failedTests.forEach(test => {
        console.log(`${test.suite} > ${test.test}`);
        console.log(`  Error: ${test.error?.message}`);
      });
    }

    // Performance warnings
    if (performance.averageTestTime > 1000) {
      console.log('\n⚠️  Performance Warning: Average test time is high');
    }

    if (performance.memoryUsage > 100 * 1024 * 1024) { // 100MB
      console.log('\n⚠️  Memory Warning: High memory usage detected');
    }
  }

  // Utility methods for tests
  private async measurePerformance<T>(
    name: string,
    fn: () => Promise<T> | T
  ): Promise<{ result: T; duration: number; memory: number }> {
    const startTime = performance.now();
    const startMemory = (performance as any).memory?.usedJSHeapSize || 0;

    const result = await fn();

    const endTime = performance.now();
    const endMemory = (performance as any).memory?.usedJSHeapSize || 0;

    const duration = endTime - startTime;
    const memory = endMemory - startMemory;

    performanceMonitor.recordMetric(`TEST_${name}`, duration);

    return { result, duration, memory };
  }

  private expectNoMemoryLeaks(threshold: number = 10 * 1024 * 1024) { // 10MB
    return (memoryUsage: number) => {
      if (memoryUsage > threshold) {
        throw new Error(`Memory leak detected: ${memoryUsage} bytes (threshold: ${threshold})`);
      }
    };
  }

  private expectPerformanceWithin(maxDuration: number) {
    return (duration: number) => {
      if (duration > maxDuration) {
        throw new Error(`Performance expectation failed: ${duration}ms > ${maxDuration}ms`);
      }
    };
  }

  // Generate test report for CI/CD
  generateJUnitReport(): string {
    const xml = `<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  ${this.suites.map(suite => `
  <testsuite name="${suite.name}" tests="${this.results.filter(r => r.suite === suite.name).length}">
    ${this.results.filter(r => r.suite === suite.name).map(result => `
    <testcase name="${result.test}" time="${result.duration / 1000}">
      ${result.status === 'failed' ? `<failure message="${result.error?.message}">${result.error?.stack}</failure>` : ''}
      ${result.status === 'skipped' ? '<skipped/>' : ''}
    </testcase>`).join('')}
  </testsuite>`).join('')}
</testsuites>`;

    return xml;
  }

  // Generate coverage report
  generateCoverageReport(): {
    components: Record<string, number>;
    overall: number;
    uncoveredFiles: string[];
  } {
    // This would integrate with a coverage tool like Istanbul/NYC
    // For now, return mock data
    return {
      components: {
        'GenerationPanel': 95,
        'QueueManager': 88,
        'MediaGallery': 92,
        'SystemMonitor': 85,
      },
      overall: 90,
      uncoveredFiles: [
        'src/utils/legacy-helper.ts',
        'src/components/deprecated/OldComponent.tsx',
      ],
    };
  }
}

// Export singleton instance
export const testRunner = new ComprehensiveTestRunner();

// Export test utilities
export const testUtils = {
  measurePerformance: (global as any).testUtils?.measurePerformance,
  expectNoMemoryLeaks: (global as any).testUtils?.expectNoMemoryLeaks,
  expectPerformanceWithin: (global as any).testUtils?.expectPerformanceWithin,
};

// Helper function to create test suites
export const createTestSuite = (name: string, tests: TestCase[], options?: {
  setup?: () => void | Promise<void>;
  teardown?: () => void | Promise<void>;
}): TestSuite => ({
  name,
  tests,
  setup: options?.setup,
  teardown: options?.teardown,
});

// Helper function to create test cases
export const createTestCase = (
  name: string,
  fn: () => void | Promise<void>,
  options?: {
    timeout?: number;
    skip?: boolean;
    only?: boolean;
  }
): TestCase => ({
  name,
  fn,
  timeout: options?.timeout,
  skip: options?.skip,
  only: options?.only,
});

export default testRunner;