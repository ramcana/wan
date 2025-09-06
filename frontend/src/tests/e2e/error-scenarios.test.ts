/**
 * End-to-End Error Scenarios Test
 * Tests error handling and recovery scenarios
 * Requirements: 1.4, 3.4, 6.5
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'vitest';
import { chromium, Browser, Page, BrowserContext } from 'playwright';

describe('E2E Error Scenarios Test', () => {
  let browser: Browser;
  let context: BrowserContext;
  let page: Page;

  const FRONTEND_URL = 'http://localhost:5173';

  beforeAll(async () => {
    browser = await chromium.launch({ headless: true });
  });

  afterAll(async () => {
    await browser.close();
  });

  beforeEach(async () => {
    context = await browser.newContext();
    page = await context.newPage();
    await page.goto(FRONTEND_URL);
    await page.waitForLoadState('networkidle');
  });

  afterEach(async () => {
    await context.close();
  });

  it('should handle form validation errors', async () => {
    // Test empty prompt
    await page.locator('[data-testid="generate-button"]').click();
    await expect(page.locator('[data-testid="error-message"]')).toContainText('Prompt is required');

    // Test prompt too long
    const longPrompt = 'a'.repeat(501);
    await page.locator('[data-testid="prompt-input"]').fill(longPrompt);
    await expect(page.locator('[data-testid="error-message"]')).toContainText('Prompt must be 500 characters or less');

    // Verify error message appears within 1 second
    const startTime = Date.now();
    await page.locator('[data-testid="prompt-input"]').fill('');
    await page.locator('[data-testid="generate-button"]').click();
    await expect(page.locator('[data-testid="error-message"]')).toBeVisible();
    const endTime = Date.now();
    expect(endTime - startTime).toBeLessThan(1000);
  });

  it('should handle image upload validation errors', async () => {
    await page.locator('[data-testid="model-select"]').selectOption('I2V-A14B');

    // Test invalid file type
    const fileInput = page.locator('[data-testid="image-upload-input"]');
    await fileInput.setInputFiles('./src/tests/fixtures/invalid-file.txt');
    
    await expect(page.locator('[data-testid="error-message"]')).toContainText('Invalid file type');

    // Test file too large (if we have such validation)
    // This would require a large test file
  });

  it('should handle backend connection errors gracefully', async () => {
    // Intercept API calls to simulate backend down
    await page.route('**/api/**', route => {
      route.abort('failed');
    });

    await page.locator('[data-testid="model-select"]').selectOption('T2V-A14B');
    await page.locator('[data-testid="prompt-input"]').fill('Test prompt');
    await page.locator('[data-testid="generate-button"]').click();

    // Should show connection error
    await expect(page.locator('[data-testid="error-message"]')).toContainText('Connection error');
    
    // Should show retry option
    await expect(page.locator('[data-testid="retry-button"]')).toBeVisible();
  });

  it('should handle generation failures gracefully', async () => {
    // Mock API to return failure response
    await page.route('**/api/v1/generate', route => {
      route.fulfill({
        status: 400,
        contentType: 'application/json',
        body: JSON.stringify({
          error: 'generation_error',
          message: 'VRAM exhausted',
          suggestions: ['Try lower resolution', 'Enable model offloading']
        })
      });
    });

    await page.locator('[data-testid="model-select"]').selectOption('T2V-A14B');
    await page.locator('[data-testid="prompt-input"]').fill('Test prompt');
    await page.locator('[data-testid="generate-button"]').click();

    // Should show specific error message
    await expect(page.locator('[data-testid="error-message"]')).toContainText('VRAM exhausted');
    
    // Should show suggestions
    await expect(page.locator('[data-testid="error-suggestions"]')).toContainText('Try lower resolution');
    await expect(page.locator('[data-testid="error-suggestions"]')).toContainText('Enable model offloading');
  });

  it('should handle queue errors and recovery', async () => {
    // Add task to queue first
    await page.locator('[data-testid="model-select"]').selectOption('T2V-A14B');
    await page.locator('[data-testid="prompt-input"]').fill('Test prompt');
    await page.locator('[data-testid="generate-button"]').click();

    // Wait for task to appear
    await expect(page.locator('[data-testid="task-card"]').first()).toBeVisible();

    // Mock queue API to return error
    await page.route('**/api/v1/queue/**', route => {
      if (route.request().method() === 'DELETE') {
        route.fulfill({
          status: 500,
          contentType: 'application/json',
          body: JSON.stringify({
            error: 'server_error',
            message: 'Failed to cancel task'
          })
        });
      } else {
        route.continue();
      }
    });

    // Try to cancel task
    await page.locator('[data-testid="task-card"]').first().locator('[data-testid="cancel-button"]').click();

    // Should show error message
    await expect(page.locator('[data-testid="error-message"]')).toContainText('Failed to cancel task');
    
    // Task should remain in original state
    await expect(page.locator('[data-testid="task-card"]').first().locator('[data-testid="task-status"]')).not.toContainText('cancelled');
  });

  it('should handle offline scenarios', async () => {
    // Simulate offline
    await context.setOffline(true);

    await page.locator('[data-testid="model-select"]').selectOption('T2V-A14B');
    await page.locator('[data-testid="prompt-input"]').fill('Test prompt');
    await page.locator('[data-testid="generate-button"]').click();

    // Should show offline indicator
    await expect(page.locator('[data-testid="offline-indicator"]')).toBeVisible();
    
    // Should queue request for later
    await expect(page.locator('[data-testid="queued-for-sync"]')).toBeVisible();

    // Simulate coming back online
    await context.setOffline(false);
    
    // Should attempt to sync
    await expect(page.locator('[data-testid="syncing-indicator"]')).toBeVisible();
  });

  it('should handle resource constraint warnings', async () => {
    // Mock system stats to show high VRAM usage
    await page.route('**/api/v1/system/stats', route => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          cpu_percent: 45.2,
          ram_used_gb: 12.5,
          ram_total_gb: 32.0,
          gpu_percent: 85.3,
          vram_used_mb: 7200,
          vram_total_mb: 8192,
          timestamp: new Date().toISOString()
        })
      });
    });

    await page.locator('[data-testid="monitoring-tab"]').click();

    // Should show VRAM warning
    await expect(page.locator('[data-testid="vram-warning"]')).toBeVisible();
    await expect(page.locator('[data-testid="vram-warning"]')).toContainText('VRAM usage high');
    
    // Should show optimization suggestions
    await expect(page.locator('[data-testid="optimization-suggestions"]')).toBeVisible();
  });

  it('should provide error recovery options', async () => {
    // Simulate a failed generation
    await page.route('**/api/v1/generate', route => {
      route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({
          error: 'server_error',
          message: 'Internal server error'
        })
      });
    });

    await page.locator('[data-testid="model-select"]').selectOption('T2V-A14B');
    await page.locator('[data-testid="prompt-input"]').fill('Test prompt');
    await page.locator('[data-testid="generate-button"]').click();

    // Should show error with recovery options
    await expect(page.locator('[data-testid="error-message"]')).toBeVisible();
    await expect(page.locator('[data-testid="retry-button"]')).toBeVisible();
    await expect(page.locator('[data-testid="report-issue-button"]')).toBeVisible();

    // Test retry functionality
    await page.unroute('**/api/v1/generate');
    await page.locator('[data-testid="retry-button"]').click();
    
    // Should attempt the request again
    await expect(page.locator('[data-testid="task-card"]').first()).toBeVisible();
  });
});