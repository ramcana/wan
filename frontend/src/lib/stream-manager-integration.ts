import { streamManager, StreamRequest, StreamOptions } from './stream-manager'

export interface IntegrationConfig {
  baseUrl: string
  apiKey?: string
  timeout?: number
}

export interface StreamIntegrationOptions extends StreamOptions {
  config: IntegrationConfig
}

export class StreamManagerIntegration {
  private config: IntegrationConfig

  constructor(options: StreamIntegrationOptions) {
    this.config = options.config
  }

  async startVideoStream(taskId: string): Promise<ReadableStream> {
    const request: StreamRequest = {
      url: `${this.config.baseUrl}/api/v1/stream/video/${taskId}`,
      method: 'GET',
      headers: {
        'Accept': 'application/octet-stream',
        ...(this.config.apiKey && { 'Authorization': `Bearer ${this.config.apiKey}` })
      }
    }

    return streamManager.createStream(`video-${taskId}`, request)
  }

  async startProgressStream(taskId: string): Promise<ReadableStream> {
    const request: StreamRequest = {
      url: `${this.config.baseUrl}/api/v1/stream/progress/${taskId}`,
      method: 'GET',
      headers: {
        'Accept': 'text/event-stream',
        'Cache-Control': 'no-cache',
        ...(this.config.apiKey && { 'Authorization': `Bearer ${this.config.apiKey}` })
      }
    }

    return streamManager.createStream(`progress-${taskId}`, request)
  }

  stopVideoStream(taskId: string): void {
    streamManager.stopStream(`video-${taskId}`)
  }

  stopProgressStream(taskId: string): void {
    streamManager.stopStream(`progress-${taskId}`)
  }

  stopAllStreams(): void {
    streamManager.stopAllStreams()
  }

  updateConfig(config: Partial<IntegrationConfig>): void {
    this.config = { ...this.config, ...config }
  }
}

export function createStreamIntegration(config: IntegrationConfig): StreamManagerIntegration {
  return new StreamManagerIntegration({ config })
}
