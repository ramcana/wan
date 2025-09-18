import axios, { AxiosInstance, AxiosResponse, AxiosError } from 'axios'
import { 
  SystemHealth, 
  PortDetectionResponse, 
  PortDetectionResponseSchema,
  SystemHealthSchema,
  LoRAListResponse,
  LoRAUploadResponse,
  LoRAStatusResponse,
  LoRAPreviewResponse,
  LoRAMemoryImpactResponse,
  LoRAListResponseSchema,
  LoRAUploadResponseSchema,
  LoRAStatusResponseSchema,
  LoRAPreviewResponseSchema,
  LoRAMemoryImpactResponseSchema
} from './api-schemas'

// Port configuration
const BACKEND_PORT = import.meta.env.BACKEND_PORT || '8000'
const API_PREFIX = '/api/v1'

export class ApiError extends Error {
  constructor(
    message: string,
    public status?: number,
    public code?: string
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

class ApiClient {
  private client: AxiosInstance
  private baseURL: string
  private configuredBaseURL: string

  constructor() {
    // Initialize base URL from environment variable with relative path fallback for proxy setups
    this.configuredBaseURL = import.meta.env.VITE_API_URL || ''
    this.baseURL = this.configuredBaseURL

    console.log('API Client initialized with configured URL:', this.configuredBaseURL)
    console.log('API Client using base URL:', this.baseURL)

    this.client = axios.create({
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    })

    this.setupInterceptors()
  }

  private setupInterceptors() {
    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        console.log('Making request to:', config.baseURL, config.url);
        // Use the configured base URL or default to relative paths for proxy setups
        if (this.baseURL) {
          config.baseURL = this.baseURL;
        } else {
          // When baseURL is empty, use relative paths (works with reverse proxy)
          config.baseURL = '';
        }
        console.log('Request config:', config);
        return config;
      },
      (error) => {
        console.error('Request interceptor error:', error);
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => {
        console.log('Response received:', response.status, response.config.url);
        return response;
      },
      (error: AxiosError) => {
        console.error('Response interceptor error:', error);
        const message = (error.response?.data as any)?.detail || error.message || 'An error occurred';
        const status = error.response?.status;
        const code = error.code;
        
        throw new ApiError(message, status, code);
      }
    );
  }

  public setBaseURL(url: string) {
    console.log('Setting API base URL from', this.baseURL, 'to', url)
    this.baseURL = url
  }

  public getBaseURL(): string {
    return this.baseURL
  }

  public getConfiguredBaseURL(): string {
    return this.configuredBaseURL
  }

  // Generic HTTP methods
  async get<T>(url: string, params?: Record<string, any>): Promise<T> {
    const response: AxiosResponse<T> = await this.client.get(url, { params })
    return response.data
  }

  async post<T>(url: string, data?: any, config?: any): Promise<T> {
    const response: AxiosResponse<T> = await this.client.post(url, data, config)
    return response.data
  }

  async put<T>(url: string, data?: any): Promise<T> {
    const response: AxiosResponse<T> = await this.client.put(url, data)
    return response.data
  }

  async patch<T>(url: string, data?: any): Promise<T> {
    const response: AxiosResponse<T> = await this.client.patch(url, data)
    return response.data
  }

  async del<T>(url: string): Promise<T> {
    const response: AxiosResponse<T> = await this.client.delete(url)
    return response.data
  }

  // Port detection and initialization
  async detectPorts(): Promise<PortDetectionResponse> {
    try {
      console.log('Detecting ports...')
      const response = await this.get<PortDetectionResponse>('/api/v1/system/ports')
      console.log('Port detection response:', response)
      return PortDetectionResponseSchema.parse(response)
    } catch (error) {
      console.error('Port detection failed:', error)
      // Fallback to default ports if detection fails
      return {
        backend_port: parseInt(BACKEND_PORT),
        frontend_port: 3000,
        status: 'fallback'
      }
    }
  }

  async initializePortDetection(): Promise<void> {
    try {
      console.log('Initializing port detection...')
      console.log('Configured base URL:', this.configuredBaseURL)
      // Only perform port detection if using the default localhost configuration
      if (this.configuredBaseURL === `http://localhost:${BACKEND_PORT}`) {
        console.log('Using default configuration, detecting ports...')
        const ports = await this.detectPorts()
        console.log('Detected ports:', ports)
        if (ports.backend_port !== parseInt(BACKEND_PORT)) {
          console.log('Setting base URL to detected port:', ports.backend_port)
          this.setBaseURL(`http://localhost:${ports.backend_port}`)
        }
      } else {
        console.log('Using custom configuration, skipping port detection')
      }
      // If a custom VITE_API_URL is configured, respect it and skip port detection
    } catch (error) {
      console.warn('Port detection failed, using configured base URL:', this.configuredBaseURL, error)
    }
  }

  // System health check
  async getSystemHealth(): Promise<SystemHealth> {
    try {
      console.log('Fetching system health from:', this.baseURL);
      const response = await this.get<SystemHealth>('/api/v1/system/health');
      console.log('System health response:', response);
      return SystemHealthSchema.parse(response);
    } catch (error: any) {
      console.error('System health check failed:', error);
      console.error('Error details:', {
        message: error.message,
        code: error.code,
        status: error.status,
        response: error.response
      });
      throw error;
    }
  }

  // Test connection
  async testConnection(): Promise<boolean> {
    try {
      console.log('Testing connection to:', this.baseURL);
      const response = await this.get('/api/v1/system/health');
      console.log('Backend connection test response:', response);
      return true;
    } catch (error: any) {
      console.error('Backend connection test failed:', error);
      console.error('Error details:', {
        message: error.message,
        code: error.code,
        status: error.status,
        response: error.response
      });
      return false;
    }
  }
  
  // LoRA API methods
  async listLoRAs(): Promise<LoRAListResponse> {
    const response = await this.get<LoRAListResponse>('/api/v1/lora')
    return LoRAListResponseSchema.parse(response)
  }
  
  async uploadLoRA(file: File, name?: string, onProgress?: (progress: number) => void): Promise<LoRAUploadResponse> {
    const formData = new FormData()
    formData.append('file', file)
    if (name) {
      formData.append('name', name)
    }
    
    const config = onProgress ? {
      onUploadProgress: (progressEvent: ProgressEvent) => {
        const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total!)
        onProgress(percentCompleted)
      }
    } : {}
    
    const response = await this.post<LoRAUploadResponse>('/api/v1/lora/upload', formData, config)
    return LoRAUploadResponseSchema.parse(response)
  }
  
  async getLoRAStatus(name: string): Promise<LoRAStatusResponse> {
    const response = await this.get<LoRAStatusResponse>(`/api/v1/lora/${name}/status`)
    return LoRAStatusResponseSchema.parse(response)
  }
  
  async generateLoRAPreview(name: string, prompt: string): Promise<LoRAPreviewResponse> {
    const response = await this.post<LoRAPreviewResponse>(`/api/v1/lora/${name}/preview`, { prompt })
    return LoRAPreviewResponseSchema.parse(response)
  }
  
  async estimateLoRAMemoryImpact(name: string): Promise<LoRAMemoryImpactResponse> {
    const response = await this.get<LoRAMemoryImpactResponse>(`/api/v1/lora/${name}/memory-impact`)
    return LoRAMemoryImpactResponseSchema.parse(response)
  }
  
  async deleteLoRA(name: string): Promise<any> {
    return await this.del<any>(`/api/v1/lora/${name}`)
  }
}

// Create singleton instance
const apiClient = new ApiClient()

// Export convenience methods
export const get = <T>(url: string, params?: Record<string, any>) => apiClient.get<T>(url, params)
export const post = <T>(url: string, data?: any, config?: any) => apiClient.post<T>(url, data, config)
export const put = <T>(url: string, data?: any) => apiClient.put<T>(url, data)
export const patch = <T>(url: string, data?: any) => apiClient.patch<T>(url, data)
export const del = <T>(url: string) => apiClient.del<T>(url)

// Export specific API functions
export const portDetectionApi = {
  detect: () => apiClient.detectPorts(),
}

export const loraApi = {
  list: () => apiClient.listLoRAs(),
  upload: (file: File, name?: string, onProgress?: (progress: number) => void) => 
    apiClient.uploadLoRA(file, name, onProgress),
  getStatus: (name: string) => apiClient.getLoRAStatus(name),
  generatePreview: (name: string, prompt: string) => apiClient.generateLoRAPreview(name, prompt),
  estimateMemoryImpact: (name: string) => apiClient.estimateLoRAMemoryImpact(name),
  delete: (name: string) => apiClient.deleteLoRA(name),
}

export const getSystemHealth = () => apiClient.getSystemHealth()
export const initializePortDetection = () => apiClient.initializePortDetection()
export const testConnection = () => apiClient.testConnection()

// Export configuration constants
export const BACKEND_PORT_CONFIG = BACKEND_PORT
export const API_PREFIX_CONFIG = API_PREFIX

// Export the client instance
export { apiClient }
export default apiClient