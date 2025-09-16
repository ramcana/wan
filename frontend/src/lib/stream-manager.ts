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
  private activeStreams = new Map<string, AbortController>()

  constructor(options: StreamOptions = {}) {
    // Options are validated but not stored as they're not used elsewhere
    const validatedOptions = {
      timeout: 30000,
      retryAttempts: 3,
      bufferSize: 1024,
      ...options
    }
    // Could be used for future timeout/retry logic
    void validatedOptions
  }

  async createStream(id: string, request: StreamRequest): Promise<ReadableStream> {
    const manualController = new AbortController()
    this.activeStreams.set(id, manualController)

    const { timeout = 30000, retryAttempts = 3, bufferSize } = this.options
    const attempts = Math.max(1, retryAttempts ?? 1)
    let lastError: unknown

    for (let attempt = 0; attempt < attempts; attempt++) {
      if (manualController.signal.aborted) {
        this.activeStreams.delete(id)
        const reason = manualController.signal.reason
        if (reason instanceof Error) {
          throw reason
        }
        throw new Error(typeof reason === 'string' ? reason : 'Stream aborted')
      }

      const attemptController = new AbortController()
      const abortAttempt = () => {
        if (!attemptController.signal.aborted) {
          attemptController.abort(manualController.signal.reason)
        }
      }
      manualController.signal.addEventListener('abort', abortAttempt)

      let timeoutId: ReturnType<typeof setTimeout> | undefined
      if (typeof timeout === 'number' && timeout > 0) {
        timeoutId = setTimeout(() => {
          if (!attemptController.signal.aborted) {
            attemptController.abort(new Error('Stream request timed out'))
          }
        }, timeout)
      }

      try {
        const response = await fetch(request.url, {
          method: request.method,
          headers: request.headers,
          body: request.body,
          signal: attemptController.signal
        })

        if (!response.ok) {
          throw new Error(`Stream request failed: ${response.status}`)
        }

        const responseStream = response.body

        if (!responseStream) {
          this.activeStreams.delete(id)
          return new ReadableStream()
        }

        this.activeStreams.set(id, attemptController)

        if (bufferSize && bufferSize > 0) {
          return this.wrapStreamWithBuffer(responseStream, bufferSize)
        }

        return responseStream
      } catch (error) {
        lastError = error

        if (manualController.signal.aborted) {
          this.activeStreams.delete(id)
          throw error
        }

        if (attempt === attempts - 1) {
          this.activeStreams.delete(id)
          if (error instanceof Error) {
            throw error
          }
          throw new Error('Stream request failed')
        }
      } finally {
        if (timeoutId !== undefined) {
          clearTimeout(timeoutId)
        }
        manualController.signal.removeEventListener('abort', abortAttempt)
      }
    }

    this.activeStreams.delete(id)
    if (lastError instanceof Error) {
      throw lastError
    }
    throw new Error('Stream request failed')
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

  private wrapStreamWithBuffer(stream: ReadableStream<Uint8Array>, bufferSize: number): ReadableStream<Uint8Array> {
    const effectiveBufferSize = Math.max(1, Math.floor(bufferSize))
    const reader = stream.getReader()
    let pendingChunks: Uint8Array[] = []
    let pendingLength = 0

    return new ReadableStream<Uint8Array>({
      async pull(controller) {
        while (pendingLength < effectiveBufferSize) {
          const { done, value } = await reader.read()

          if (done) {
            if (pendingLength > 0) {
              const remaining = StreamManager.consumeFromBuffer(pendingChunks, pendingLength)
              pendingChunks = []
              pendingLength = 0
              controller.enqueue(remaining)
            }
            controller.close()
            return
          }

          if (value && value.byteLength > 0) {
            pendingChunks.push(value)
            pendingLength += value.byteLength
          }
        }

        const chunk = StreamManager.consumeFromBuffer(pendingChunks, effectiveBufferSize)
        pendingLength -= chunk.byteLength
        controller.enqueue(chunk)
      },
      async cancel(reason) {
        pendingChunks = []
        pendingLength = 0
        await reader.cancel(reason)
      }
    })
  }

  private static consumeFromBuffer(chunks: Uint8Array[], size: number): Uint8Array {
    if (size <= 0 || chunks.length === 0) {
      return new Uint8Array(0)
    }

    const first = chunks[0]

    if (first.byteLength === size) {
      chunks.shift()
      return first
    }

    if (first.byteLength > size) {
      const result = first.slice(0, size)
      chunks[0] = first.slice(size)
      return result
    }

    const result = new Uint8Array(size)
    let offset = 0

    while (offset < size && chunks.length > 0) {
      const chunk = chunks[0]
      const remaining = size - offset

      if (chunk.byteLength <= remaining) {
        result.set(chunk, offset)
        offset += chunk.byteLength
        chunks.shift()
      } else {
        result.set(chunk.subarray(0, remaining), offset)
        chunks[0] = chunk.subarray(remaining)
        offset += remaining
      }
    }

    return offset === size ? result : result.slice(0, offset)
  }
}

export const streamManager = new StreamManager()
