import axios, { AxiosInstance, AxiosResponse, AxiosError } from 'axios'
import { 
  SystemHealth, 
  PortDetectionResponse, 
  PortDetectionResponseSchema,
  SystemHealthSchema 
} from './api-schemas'

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
  private baseURL: string = 'http://localhost:9000'

  constructor() {
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
        config.baseURL = this.baseURL
        return config
      },
      (error) => Promise.reject(error)
    )

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        const message = (error.response?.data as any)?.detail || error.message || 'An error occurred'
        const status = error.response?.status
        const code = error.code
        
        throw new ApiError(message, status, code)
      }
    )
  }

  public setBaseURL(url: string) {
    this.baseURL = url
  }

  public getBaseURL(): string {
    return this.baseURL
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
      const response = await this.get<PortDetectionResponse>('/api/v1/system/ports')
      return PortDetectionResponseSchema.parse(response)
    } catch (error) {
      // Fallback to default ports if detection fails
      return {
        backend_port: 9000,
        frontend_port: 3000,
        status: 'fallback'
      }
    }
  }

  async initializePortDetection(): Promise<void> {
    try {
      const ports = await this.detectPorts()
      if (ports.backend_port !== 9000) {
        this.setBaseURL(`http://localhost:${ports.backend_port}`)
      }
    } catch (error) {
      console.warn('Port detection failed, using default backend port 9000')
    }
  }

  // System health check
  async getSystemHealth(): Promise<SystemHealth> {
    const response = await this.get<SystemHealth>('/api/v1/system/health')
    return SystemHealthSchema.parse(response)
  }

  // Test connection
  async testConnection(): Promise<boolean> {
    try {
      await this.get('/api/v1/system/health')
      return true
    } catch {
      return false
    }
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

export const getSystemHealth = () => apiClient.getSystemHealth()
export const initializePortDetection = () => apiClient.initializePortDetection()
export const testConnection = () => apiClient.testConnection()

// Export the client instance
export { apiClient }
export default apiClient
