/**
 * E2E Test Runner
 * Orchestrates end-to-end testing with proper setup and teardown
 */

import { execSync } from 'child_process';
import { chromium, Browser } from 'playwright';

interface TestConfig {
  frontendUrl: string;
  backendUrl: string;
  timeout: number;
  retries: number;
  headless: boolean;
}

class E2ETestRunner {
  private config: TestConfig;
  private browser?: Browser;

  constructor(config: Partial<TestConfig> = {}) {
    this.config = {
      frontendUrl: 'http://localhost:5173',
      backendUrl: 'http://localhost:8000',
      timeout: 30000,
      retries: 2,
      headless: true,
      ...config
    };
  }

  async setup(): Promise<void> {
    console.log('🚀 Setting up E2E test environment...');

    // Check if backend is running
    try {
      const response = await fetch(`${this.config.backendUrl}/api/v1/health`);
      if (!response.ok) {
        throw new Error('Backend health check failed');
      }
      console.log('✅ Backend is running');
    } catch (error) {
      console.error('❌ Backend is not running. Please start the backend server.');
      throw error;
    }

    // Check if frontend is running
    try {
      const response = await fetch(this.config.frontendUrl);
      if (!response.ok) {
        throw new Error('Frontend is not accessible');
      }
      console.log('✅ Frontend is running');
    } catch (error) {
      console.error('❌ Frontend is not running. Please start the frontend server.');
      throw error;
    }

    // Setup browser
    this.browser = await chromium.launch({ 
      headless: this.config.headless,
      args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
    console.log('✅ Browser launched');
  }

  async teardown(): Promise<void> {
    if (this.browser) {
      await this.browser.close();
      console.log('✅ Browser closed');
    }
  }

  async runTests(): Promise<void> {
    try {
      await this.setup();
      
      console.log('🧪 Running E2E tests...');
      
      // Run the actual tests using vitest
      execSync('npm run test:e2e', { 
        stdio: 'inherit',
        cwd: process.cwd()
      });
      
      console.log('✅ All E2E tests passed!');
    } catch (error) {
      console.error('❌ E2E tests failed:', error);
      throw error;
    } finally {
      await this.teardown();
    }
  }

  async waitForBackend(maxAttempts: number = 30): Promise<void> {
    console.log('⏳ Waiting for backend to be ready...');
    
    for (let i = 0; i < maxAttempts; i++) {
      try {
        const response = await fetch(`${this.config.backendUrl}/api/v1/health`);
        if (response.ok) {
          console.log('✅ Backend is ready');
          return;
        }
      } catch (error) {
        // Backend not ready yet
      }
      
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    
    throw new Error('Backend failed to start within timeout');
  }

  async waitForFrontend(maxAttempts: number = 30): Promise<void> {
    console.log('⏳ Waiting for frontend to be ready...');
    
    for (let i = 0; i < maxAttempts; i++) {
      try {
        const response = await fetch(this.config.frontendUrl);
        if (response.ok) {
          console.log('✅ Frontend is ready');
          return;
        }
      } catch (error) {
        // Frontend not ready yet
      }
      
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    
    throw new Error('Frontend failed to start within timeout');
  }
}

// CLI runner
if (require.main === module) {
  const runner = new E2ETestRunner({
    headless: process.env.CI === 'true' || process.argv.includes('--headless'),
  });

  runner.runTests().catch((error) => {
    console.error('E2E test runner failed:', error);
    process.exit(1);
  });
}

export { E2ETestRunner };