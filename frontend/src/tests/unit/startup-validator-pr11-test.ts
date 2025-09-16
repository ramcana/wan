/**
 * Test file to verify PR #11 changes to startup-validator.ts
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { startupValidator } from '@/lib/startup-validator';
import { getSystemHealth } from '@/lib/api-client';
import { SystemHealth } from '@/lib/api-schemas';

// Mock the API client
vi.mock('@/lib/api-client', () => ({
  getSystemHealth: vi.fn(),
  testConnection: vi.fn().mockResolvedValue(true)
}));

describe('PR #11 - Prefetch system health for startup validation checks', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should pass system health to validation checks', async () => {
    // Mock system health response
    const mockHealth: SystemHealth = {
      status: 'healthy',
      backend_online: true,
      database_connected: true,
      gpu_available: true
    };

    vi.mocked(getSystemHealth).mockResolvedValue(mockHealth);

    // Run validation
    const result = await startupValidator.validateStartup();

    // Verify that system health was fetched once
    expect(getSystemHealth).toHaveBeenCalledTimes(1);
    
    // Verify that the result includes system health
    expect(result.systemHealth).toEqual(mockHealth);
    
    // Verify validation passed
    expect(result.isValid).toBe(true);
    expect(result.errors).toHaveLength(0);
  });

  it('should handle unavailable system health gracefully', async () => {
    // Mock system health failure
    vi.mocked(getSystemHealth).mockRejectedValue(new Error('Network error'));

    // Run validation
    const result = await startupValidator.validateStartup();

    // Verify that system health was attempted to be fetched
    expect(getSystemHealth).toHaveBeenCalledTimes(1);
    
    // Verify that a warning was added
    expect(result.warnings).toContain('Could not retrieve system health information');
    
    // Verify validation still passes (since checks are not required)
    expect(result.isValid).toBe(true);
  });

  it('should provide appropriate messages when system health is unavailable', async () => {
    // Mock system health failure
    vi.mocked(getSystemHealth).mockRejectedValue(new Error('Network error'));

    // Get the validation checks
    const validator = startupValidator as any;
    const checks = validator.checks;

    // Test each check with undefined system health
    for (const check of checks) {
      if (check.name !== 'Backend Connection') { // Skip backend connection as it doesn't use system health
        const result = await check.check(undefined);
        expect(result.success).toBe(false);
        expect(result.message).toContain('System health unavailable');
      }
    }
  });
});