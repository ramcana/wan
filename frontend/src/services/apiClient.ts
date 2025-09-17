import { useAuth } from '../contexts/AuthContext';

class ApiClient {
  private baseUrl: string;
  private refreshPromise: Promise<boolean> | null = null;

  constructor(baseUrl: string = '/api/v1') {
    this.baseUrl = baseUrl;
  }

  private async refreshAccessTokenInternal(): Promise<boolean> {
    // This would typically use a global auth service
    // For now, we'll simulate the refresh
    try {
      const refreshToken = localStorage.getItem('refreshToken');
      if (!refreshToken) return false;

      const response = await fetch(`${this.baseUrl}/auth/refresh`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ refresh_token: refreshToken }),
      });

      if (response.ok) {
        const data = await response.json();
        localStorage.setItem('accessToken', data.access_token);
        return true;
      }
      return false;
    } catch (error) {
      console.error('Token refresh error:', error);
      return false;
    }
  }

  private async getAccessToken(): Promise<string | null> {
    return localStorage.getItem('accessToken');
  }

  private async handleResponse(response: Response): Promise<any> {
    if (response.status === 401) {
      // Try to refresh token
      if (!this.refreshPromise) {
        this.refreshPromise = this.refreshAccessTokenInternal();
      }

      const refreshSuccess = await this.refreshPromise;
      this.refreshPromise = null;

      if (refreshSuccess) {
        // Retry the original request
        // Note: In a real implementation, you'd need to pass the original request details
        throw new Error('Authentication required - please retry the request');
      } else {
        // Refresh failed, logout user
        localStorage.removeItem('user');
        localStorage.removeItem('accessToken');
        localStorage.removeItem('refreshToken');
        throw new Error('Authentication failed - please log in again');
      }
    }

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
    }

    return response.json();
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const accessToken = await this.getAccessToken();

    const defaultHeaders = {
      'Content-Type': 'application/json',
      ...(accessToken && { 'Authorization': `Bearer ${accessToken}` }),
    };

    const config: RequestInit = {
      ...options,
      headers: {
        ...defaultHeaders,
        ...options.headers,
      },
    };

    // Add CSRF protection if needed
    if (['POST', 'PUT', 'DELETE'].includes(config.method || '')) {
      const csrfToken = localStorage.getItem('csrfToken');
      if (csrfToken) {
        (config.headers as Record<string, string>)['X-CSRF-Token'] = csrfToken;
      }
    }

    const response = await fetch(url, config);
    return this.handleResponse(response);
  }

  // Authentication endpoints
  async login(username: string, password: string): Promise<any> {
    return this.request('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    });
  }

  async register(username: string, email: string, password: string): Promise<any> {
    return this.request('/auth/register', {
      method: 'POST',
      body: JSON.stringify({ username, email, password }),
    });
  }

  async refreshAccessToken(refreshToken: string): Promise<any> {
    return this.request('/auth/refresh', {
      method: 'POST',
      body: JSON.stringify({ refresh_token: refreshToken }),
    });
  }

  async getProfile(): Promise<any> {
    return this.request('/auth/profile');
  }

  async updateProfile(userData: any): Promise<any> {
    return this.request('/auth/profile', {
      method: 'PUT',
      body: JSON.stringify(userData),
    });
  }

  async createApiKey(name: string): Promise<any> {
    return this.request('/auth/api-keys', {
      method: 'POST',
      body: JSON.stringify({ name }),
    });
  }

  async listApiKeys(): Promise<any> {
    return this.request('/auth/api-keys');
  }

  async revokeApiKey(keyId: string): Promise<any> {
    return this.request(`/auth/api-keys/${keyId}`, {
      method: 'DELETE',
    });
  }

  // Video generation endpoints
  async generateVideo(requestData: any): Promise<any> {
    return this.request('/video/generate', {
      method: 'POST',
      body: JSON.stringify(requestData),
    });
  }

  async uploadImage(file: File): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);

    return this.request('/video/upload', {
      method: 'POST',
      body: formData,
      headers: {
        // Remove Content-Type to let browser set it with boundary
      },
    });
  }

  async getTaskStatus(taskId: string): Promise<any> {
    return this.request(`/video/tasks/${taskId}`);
  }

  async listTasks(): Promise<any> {
    return this.request('/video/tasks');
  }

  // Rate limiting info
  async getRateLimits(): Promise<any> {
    return this.request('/auth/rate-limits');
  }
}

// Create a singleton instance
const apiClient = new ApiClient();

export default apiClient;

// Export types for better type safety
export interface User {
  id: string;
  username: string;
  email: string;
  isAdmin: boolean;
  createdAt: string;
}

export interface VideoRequest {
  prompt: string;
  negativePrompt?: string;
  model?: string;
  width?: number;
  height?: number;
  steps?: number;
  seed?: number;
}

export interface VideoTask {
  id: string;
  userId: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  resultUrl?: string;
  errorMessage?: string;
  createdAt: string;
  updatedAt: string;
}

export interface ApiKey {
  id: string;
  name: string;
  lastUsed?: string;
  createdAt: string;
  expiresAt?: string;
}