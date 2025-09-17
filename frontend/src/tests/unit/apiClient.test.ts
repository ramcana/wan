import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import apiClient from '../../services/apiClient';

describe('ApiClient', () => {
  beforeEach(() => {
    // Clear localStorage before each test
    localStorage.clear();
    
    // Reset fetch mock
    global.fetch = vi.fn();
  });

  afterEach(() => {
    // Clear localStorage after each test
    localStorage.clear();
    
    // Clear mocks
    vi.clearAllMocks();
  });

  it('should make a successful login request', async () => {
    // Mock successful login response
    (global.fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        user: {
          id: '1',
          username: 'testuser',
          email: 'test@example.com',
          isAdmin: false,
          createdAt: '2023-01-01T00:00:00Z'
        },
        access_token: 'access-token',
        refresh_token: 'refresh-token'
      })
    });

    const result = await apiClient.login('testuser', 'password');
    
    expect(result.user.username).toBe('testuser');
    expect(result.access_token).toBe('access-token');
    expect(result.refresh_token).toBe('refresh-token');
    
    // Check that fetch was called with correct parameters
    expect(global.fetch).toHaveBeenCalledWith(
      '/api/v1/auth/login',
      expect.objectContaining({
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username: 'testuser', password: 'password' })
      })
    );
  });

  it('should handle login failure', async () => {
    // Mock failed login response
    (global.fetch as any).mockResolvedValueOnce({
      ok: false,
      status: 401,
      json: async () => ({ detail: 'Invalid credentials' })
    });

    await expect(apiClient.login('testuser', 'wrongpassword'))
      .rejects
      .toThrow('Invalid credentials');
  });

  it('should include authorization header when access token is present', async () => {
    // Set access token in localStorage
    localStorage.setItem('accessToken', 'test-token');
    
    // Mock successful response
    (global.fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ data: 'test' })
    });

    await apiClient.getProfile();
    
    // Check that fetch was called with authorization header
    expect(global.fetch).toHaveBeenCalledWith(
      '/api/v1/auth/profile',
      expect.objectContaining({
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer test-token'
        }
      })
    );
  });

  it('should handle 401 responses by throwing an error', async () => {
    // Mock 401 response
    (global.fetch as any).mockResolvedValueOnce({
      ok: false,
      status: 401,
      json: async () => ({ detail: 'Authentication required' })
    });

    await expect(apiClient.getProfile())
      .rejects
      .toThrow('Authentication required');
  });
});