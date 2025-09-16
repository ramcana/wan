export interface StreamOptions {
  timeout?: number
  retryAttempts?: number
  bufferSize?: number
}

export interface StreamRequest {
  url: string
  method: string
  headers?: Record<string, string>
  body?: any
}

export interface StreamResponse {
  data: any
  status: number
  headers: Record<string, string>
}

export class StreamManager {
  private options: StreamOptions
  private activeStreams = new Map<string, AbortController>()

  constructor(options: StreamOptions = {}) {
    this.options = {
      timeout: 30000,
      retryAttempts: 3,
      bufferSize: 1024,
      ...options
    }
  }

  async createStream(id: string, request: StreamRequest): Promise<ReadableStream> {
    const controller = new AbortController()
    this.activeStreams.set(id, controller)

    try {
      const response = await fetch(request.url, {
        method: request.method,
        headers: request.headers,
        body: request.body,
        signal: controller.signal
      })

      if (!response.ok) {
        throw new Error(`Stream request failed: ${response.status}`)
      }

      return response.body || new ReadableStream()
    } catch (error) {
      this.activeStreams.delete(id)
      throw error
    }
  }

  stopStream(id: string): void {
    const controller = this.activeStreams.get(id)
    if (controller) {
      controller.abort()
      this.activeStreams.delete(id)
    }
  }

  stopAllStreams(): void {
    for (const [id] of this.activeStreams) {
      this.stopStream(id)
    }
  }

  recreateRequest(originalRequest: StreamRequest, body: any): StreamRequest {
    return {
      ...originalRequest,
      body
    }
  }

  isStreamActive(id: string): boolean {
    return this.activeStreams.has(id)
  }

  getActiveStreamCount(): number {
    return this.activeStreams.size
  }
}

export const streamManager = new StreamManager()
