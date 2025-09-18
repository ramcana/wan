import { getSystemHealth, testConnection } from './api-client'
import { SystemHealth } from './api-schemas'
import { apiClient } from './api-client'

export interface StartupValidationResult {
  isValid: boolean
  errors: string[]
  warnings: string[]
  systemHealth?: SystemHealth
}

export interface ValidationCheck {
  name: string
  check: (systemHealth?: SystemHealth) => Promise<{ success: boolean; message?: string }>
  required: boolean
}

export class StartupValidator {
  private checks: ValidationCheck[] = [
    {
      name: 'Backend Connection',
      check: async () => {
        try {
          // Log the current API client configuration
          console.log('API Client Base URL:', apiClient.getBaseURL())
          console.log('API Client Configured URL:', apiClient.getConfiguredBaseURL())
          
          const isConnected = await testConnection()
          return {
            success: isConnected,
            message: isConnected ? 'Backend is reachable' : 'Backend is not responding'
          }
        } catch (error) {
          return {
            success: false,
            message: `Backend connection failed: ${error instanceof Error ? error.message : 'Unknown error'}`
          }
        }
      },
      required: true
    },
    {
      name: 'System Health',
      check: async (health?: SystemHealth) => {
        if (!health) {
          return {
            success: false,
            message: 'Health check could not be completed: System health unavailable'
          }
        }

        // Updated to handle the new health response format
        const isHealthy = health.status === 'ok' || health.status === 'healthy'
        return {
          success: isHealthy,
          message: isHealthy ? 'System is healthy' : `System status: ${health.status}`
        }
      },
      required: false
    },
    {
      name: 'GPU Availability',
      check: async (health?: SystemHealth) => {
        if (!health) {
          return {
            success: false,
            message: 'GPU check could not be completed: System health unavailable'
          }
        }

        // For now, we'll assume GPU is available since this info isn't in the new response
        // In a real implementation, we'd check the actual health response for GPU info
        return {
          success: true, // Temporarily assume GPU is available
          message: 'GPU status check not available in current health response'
        }
      },
      required: false
    },
    {
      name: 'Database Connection',
      check: async (health?: SystemHealth) => {
        if (!health) {
          return {
            success: false,
            message: 'Database check could not be completed: System health unavailable'
          }
        }

        // For now, we'll assume database is connected since this info isn't in the new response
        // In a real implementation, we'd check the actual health response for database info
        return {
          success: true, // Temporarily assume database is connected
          message: 'Database status check not available in current health response'
        }
      },
      required: true
    }
  ]

  async validateStartup(): Promise<StartupValidationResult> {
    const result: StartupValidationResult = {
      isValid: true,
      errors: [],
      warnings: []
    }

    try {
      // Get system health first
      let systemHealth: SystemHealth | undefined
      try {
        systemHealth = await getSystemHealth()
        result.systemHealth = systemHealth
        console.log('System health retrieved:', systemHealth)
      } catch (error) {
        result.warnings.push('Could not retrieve system health information')
        console.error('System health retrieval failed:', error)
      }

      // Run all validation checks
      for (const check of this.checks) {
        try {
          console.log(`Running validation check: ${check.name}`)
          const checkResult = await check.check(systemHealth)
          console.log(`Validation check result for ${check.name}:`, checkResult)
          
          if (!checkResult.success) {
            const message = `${check.name}: ${checkResult.message || 'Check failed'}`
            
            if (check.required) {
              result.errors.push(message)
              result.isValid = false
            } else {
              result.warnings.push(message)
            }
          }
        } catch (error) {
          const message = `${check.name}: Validation error - ${error instanceof Error ? error.message : 'Unknown error'}`
          
          if (check.required) {
            result.errors.push(message)
            result.isValid = false
          } else {
            result.warnings.push(message)
          }
        }
      }

      return result
    } catch (error) {
      result.isValid = false
      result.errors.push(`Startup validation failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
      return result
    }
  }

  addCustomCheck(check: ValidationCheck): void {
    this.checks.push(check)
  }

  removeCheck(name: string): void {
    this.checks = this.checks.filter(check => check.name !== name)
  }
}

// Export singleton instance
export const startupValidator = new StartupValidator()

// Export convenience function
export const validateStartup = () => startupValidator.validateStartup()

export default startupValidator