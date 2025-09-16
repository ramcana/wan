import { getSystemHealth, testConnection } from './api-client'
import { SystemHealth } from './api-schemas'

export interface StartupValidationResult {
  isValid: boolean
  errors: string[]
  warnings: string[]
  systemHealth?: SystemHealth
}

export interface ValidationCheck {
  name: string
  check: () => Promise<{ success: boolean; message?: string }>
  required: boolean
}

export class StartupValidator {
  private checks: ValidationCheck[] = [
    {
      name: 'Backend Connection',
      check: async () => {
        try {
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
      check: async () => {
        try {
          const health = await getSystemHealth()
          const isHealthy = health.status === 'healthy'
          return {
            success: isHealthy,
            message: isHealthy 
              ? 'System is healthy' 
              : `System status: ${health.status}`
          }
        } catch (error) {
          return {
            success: false,
            message: `Health check failed: ${error instanceof Error ? error.message : 'Unknown error'}`
          }
        }
      },
      required: false
    },
    {
      name: 'GPU Availability',
      check: async () => {
        try {
          const health = await getSystemHealth()
          return {
            success: health.gpu_available || false,
            message: health.gpu_available 
              ? 'GPU is available' 
              : 'GPU not detected or unavailable'
          }
        } catch (error) {
          return {
            success: false,
            message: `GPU check failed: ${error instanceof Error ? error.message : 'Unknown error'}`
          }
        }
      },
      required: false
    },
    {
      name: 'Database Connection',
      check: async () => {
        try {
          const health = await getSystemHealth()
          return {
            success: health.database_connected || false,
            message: health.database_connected 
              ? 'Database is connected' 
              : 'Database connection failed'
          }
        } catch (error) {
          return {
            success: false,
            message: `Database check failed: ${error instanceof Error ? error.message : 'Unknown error'}`
          }
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
      try {
        result.systemHealth = await getSystemHealth()
      } catch (error) {
        result.warnings.push('Could not retrieve system health information')
      }

      // Run all validation checks
      for (const check of this.checks) {
        try {
          const checkResult = await check.check()
          
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
