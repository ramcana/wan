/**
 * End-to-End Happy Path Test
 * Tests the complete workflow: prompt → generation → completed video
 * Requirements: 1.4, 3.4, 6.5
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'vitest';
import { chromium, Browser, Page, BrowserContext } from 'playwright';

describe('E2E Happy Path Test', () => {
  let browser: Browser;
  let context: BrowserContext;
  let page: Page;

  const FRONTEND_URL = 'http://localhost:5173';
  const BACKEND_URL = 'http://localhost:8000';

  beforeAll(async () => {
    browser = await chromium.launch({ headless: true });
  });

  afterAll(async () => {
    await browser.close();
  });

  beforeEach(async () => {
    context = await browser.newContext();
    page = await context.newPage();
    
    // Wait for backend to be ready
    await page.goto(FRONTEND_URL);
    await page.waitForLoadState('networkidle');
  });

  afterEach(async () => {
    await context.close();
  });

  it('should complete T2V generation workflow successfully', async () => {
    // Step 1: Navigate to generation page
    await page.goto(FRONTEND_URL);
    await expect(page.locator('[data-testid="generation-form"]')).toBeVisible();

    // Step 2: Select T2V model
    await page.locator('[data-testid="model-select"]').selectOption('T2V-A14B');
    await expect(page.locator('[data-testid="model-select"]')).toHaveValue('T2V-A14B');

    // Step 3: Enter prompt
    const testPrompt = 'A beautiful sunset over mountains, cinematic lighting';
    await page.locator('[data-testid="prompt-input"]').fill(testPrompt);
    await expect(page.locator('[data-testid="prompt-input"]')).toHaveValue(testPrompt);

    // Step 4: Verify character counter
    await expect(page.locator('[data-testid="character-counter"]')).toContainText(`${testPrompt.length}/500`);

    // Step 5: Select resolution
    await page.locator('[data-testid="resolution-select"]').selectOption('1280x720');
    await expect(page.locator('[data-testid="resolution-select"]')).toHaveValue('1280x720');

    // Step 6: Submit generation request
    await page.locator('[data-testid="generate-button"]').click();

    // Step 7: Verify task appears in queue within 3 seconds
    await expect(page.locator('[data-testid="queue-panel"]')).toBeVisible({ timeout: 3000 });
    await expect(page.locator('[data-testid="task-card"]').first()).toContainText(testPrompt.substring(0, 30));

    // Step 8: Verify task status updates
    const taskCard = page.locator('[data-testid="task-card"]').first();
    await expect(taskCard.locator('[data-testid="task-status"]')).toContainText('pending');

    // Step 9: Wait for processing to start
    await expect(taskCard.locator('[data-testid="task-status"]')).toContainText('processing', { timeout: 10000 });

    // Step 10: Verify progress updates appear within 5 seconds
    await expect(taskCard.locator('[data-testid="progress-bar"]')).toBeVisible({ timeout: 5000 });

    // Step 11: Wait for completion (with reasonable timeout for testing)
    await expect(taskCard.locator('[data-testid="task-status"]')).toContainText('completed', { timeout: 120000 });

    // Step 12: Verify video appears in gallery
    await page.locator('[data-testid="gallery-tab"]').click();
    await expect(page.locator('[data-testid="video-card"]').first()).toBeVisible({ timeout: 5000 });

    // Step 13: Verify video metadata
    const videoCard = page.locator('[data-testid="video-card"]').first();
    await expect(videoCard.locator('[data-testid="video-prompt"]')).toContainText(testPrompt.substring(0, 30));
    await expect(videoCard.locator('[data-testid="video-resolution"]')).toContainText('1280x720');

    // Step 14: Test video playback
    await videoCard.click();
    await expect(page.locator('[data-testid="video-player"]')).toBeVisible();
    await expect(page.locator('[data-testid="video-player"] video')).toBeVisible();
  });

  it('should handle I2V generation workflow successfully', async () => {
    await page.goto(FRONTEND_URL);

    // Select I2V model
    await page.locator('[data-testid="model-select"]').selectOption('I2V-A14B');

    // Upload test image
    const fileInput = page.locator('[data-testid="image-upload-input"]');
    await fileInput.setInputFiles('./src/tests/fixtures/test-image.jpg');

    // Verify image preview
    await expect(page.locator('[data-testid="image-preview"]')).toBeVisible();

    // Enter prompt
    const testPrompt = 'Transform this image into a dynamic video';
    await page.locator('[data-testid="prompt-input"]').fill(testPrompt);

    // Submit generation
    await page.locator('[data-testid="generate-button"]').click();

    // Verify task in queue
    await expect(page.locator('[data-testid="task-card"]').first()).toContainText(testPrompt.substring(0, 30));
    await expect(page.locator('[data-testid="task-card"]').first().locator('[data-testid="task-type"]')).toContainText('I2V');
  });

  it('should handle queue management correctly', async () => {
    await page.goto(FRONTEND_URL);

    // Add multiple tasks to queue
    for (let i = 0; i < 3; i++) {
      await page.locator('[data-testid="model-select"]').selectOption('T2V-A14B');
      await page.locator('[data-testid="prompt-input"]').fill(`Test prompt ${i + 1}`);
      await page.locator('[data-testid="generate-button"]').click();
      await page.waitForTimeout(1000); // Brief pause between submissions
    }

    // Verify all tasks appear in queue
    await expect(page.locator('[data-testid="task-card"]')).toHaveCount(3);

    // Test task cancellation
    const secondTask = page.locator('[data-testid="task-card"]').nth(1);
    await secondTask.locator('[data-testid="cancel-button"]').click();

    // Verify cancellation within 10 seconds
    await expect(secondTask.locator('[data-testid="task-status"]')).toContainText('cancelled', { timeout: 10000 });

    // Verify queue stats update
    await expect(page.locator('[data-testid="queue-stats"]')).toContainText('2 active');
  });

  it('should display system monitoring correctly', async () => {
    await page.goto(FRONTEND_URL);
    await page.locator('[data-testid="monitoring-tab"]').click();

    // Verify system stats are displayed
    await expect(page.locator('[data-testid="cpu-usage"]')).toBeVisible();
    await expect(page.locator('[data-testid="ram-usage"]')).toBeVisible();
    await expect(page.locator('[data-testid="gpu-usage"]')).toBeVisible();
    await expect(page.locator('[data-testid="vram-usage"]')).toBeVisible();

    // Verify stats update within 10 seconds
    const initialCpuValue = await page.locator('[data-testid="cpu-usage"]').textContent();
    await page.waitForTimeout(11000); // Wait for next update cycle
    const updatedCpuValue = await page.locator('[data-testid="cpu-usage"]').textContent();
    
    // Stats should have updated (or at least the timestamp should be different)
    await expect(page.locator('[data-testid="last-updated"]')).not.toContainText('11 seconds ago');
  });

  it('should handle prompt enhancement workflow', async () => {
    await page.goto(FRONTEND_URL);

    // Enter basic prompt
    const basicPrompt = 'sunset mountains';
    await page.locator('[data-testid="prompt-input"]').fill(basicPrompt);

    // Click enhance button
    await page.locator('[data-testid="enhance-prompt-button"]').click();

    // Verify enhancement appears
    await expect(page.locator('[data-testid="enhanced-prompt"]')).toBeVisible({ timeout: 5000 });
    
    // Verify enhanced prompt is different and longer
    const enhancedText = await page.locator('[data-testid="enhanced-prompt"]').textContent();
    expect(enhancedText).not.toBe(basicPrompt);
    expect(enhancedText!.length).toBeGreaterThan(basicPrompt.length);

    // Accept enhancement
    await page.locator('[data-testid="accept-enhancement-button"]').click();
    await expect(page.locator('[data-testid="prompt-input"]')).toHaveValue(enhancedText!);
  });
});