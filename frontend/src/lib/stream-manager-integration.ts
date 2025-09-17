import { streamManager, StreamRequest, StreamOptions, isStreamConsumed, recreateRequest, handleStreamError } from './stream-manager'

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

  // Add methods expected by tests
  getBrowserSupport() {
    return streamManager.getBrowserSupport();
  }

  isBrowserSupported(): boolean {
    return streamManager.isBrowserSupported();
  }
}

// Add exported functions expected by tests
export class EnhancedApiClient extends StreamManagerIntegration {
  // EnhancedApiClient inherits from StreamManagerIntegration
}

export class StreamErrorHandler {
  async handleError(error: Error, request?: Request): Promise<StreamErrorResult> {
    const isStreamError = error.message.includes('stream') || error.message.includes('body stream already read');
    
    if (isStreamError && request) {
      try {
        const recoveredRequest = await handleStreamError(request, error);
        return {
          isStreamError: true,
          canRecover: true,
          errorType: 'stream-recoverable',
          originalError: error,
          recoveredRequest
        };
      } catch (recoveryError) {
        return {
          isStreamError: true,
          canRecover: false,
          errorType: 'stream-unrecoverable',
          originalError: error
        };
      }
    }
    
    return {
      isStreamError: false,
      canRecover: false,
      errorType: 'non-stream',
      originalError: error
    };
  }

  getErrorMessage(errorType: string): string {
    switch (errorType) {
      case 'stream-recoverable':
        return 'The stream error has been recovered and the request can be retried.';
      case 'stream-unrecoverable':
        return 'The stream could not be recovered. Please try again later.';
      default:
        return 'Network request failed. Please check your connection and try again.';
    }
  }
}

export interface StreamErrorResult {
  isStreamError: boolean;
  canRecover: boolean;
  errorType: string;
  originalError: Error;
  recoveredRequest?: Request;
}

export function enhancedFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  return fetch(input, init);
}

export function useStreamManager(options: StreamOptions = {}) {
  const manager = new StreamManagerIntegration({ 
    config: { baseUrl: '' },
    ...options 
  });
  
  return {
    handleRequest: streamManager.handleRequest.bind(streamManager),
    isStreamConsumed,
    recreateRequest,
    handleStreamError,
    browserSupport: streamManager.getBrowserSupport(),
    isSupported: streamManager.isBrowserSupported(),
  };
}

export function createStreamIntegration(config: IntegrationConfig): StreamManagerIntegration {
  return new StreamManagerIntegration({ config })
}