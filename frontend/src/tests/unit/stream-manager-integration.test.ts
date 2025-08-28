import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { 
  EnhancedApiClient, 
  enhancedFetch, 
  StreamErrorHandler,
  useStreamManager 
} from '../../lib/stream-manager-integration';

// Mock fetch
global.fetch = vi.fn();

describe('StreamManager Integration', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('EnhancedApiClient', () => {
    it('should create an instance with default options', () => {
      const client = new EnhancedApiClient();
      expect(client).toBeDefined();
      expect(client.isBrowserSupported()).toBeDefined();
    });

    it('should handle successful requests', async () => {
      const mockResponse = new Response('{"success": true}', { status: 200 });
      (global.fetch as any).mockResolvedValueOnce(mockResponse);

      const client = new EnhancedApiClient();
      const response = await client.makeRequest('https://api.example.com/test');

      expect(response).toBe(mockResponse);
      expect(global.fetch).toHaveBeenCalledOnce();
    });

    it('should provide browser support information', () => {
      const client = new EnhancedApiClient();
      const support = client.getBrowserSupport();
      
      expect(support).toHaveProperty('requestClone');
      expect(support).toHaveProperty('readableStream');
      expect(support).toHaveProperty('textEncoder');
    });
  });

  describe('enhancedFetch', () => {
    it('should work as a drop-in replacement for fetch', async () => {
      const mockResponse = new Response('test data', { status: 200 });
      (global.fetch as any).mockResolvedValueOnce(mockResponse);

      const response = await enhancedFetch('https://api.example.com/test');

      expect(response).toBe(mockResponse);
      expect(global.fetch).toHaveBeenCalledOnce();
    });

    it('should handle RequestInit options', async () => {
      const mockResponse = new Response('test data', { status: 200 });
      (global.fetch as any).mockResolvedValueOnce(mockResponse);

      const options = {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: '{"test": true}',
      };

      await enhancedFetch('https://api.example.com/test', options);

      expect(global.fetch).toHaveBeenCalledOnce();
    });
  });

  describe('StreamErrorHandler', () => {
    it('should create an instance', () => {
      const handler = new StreamErrorHandler();
      expect(handler).toBeDefined();
    });

    it('should identify non-stream errors', async () => {
      const handler = new StreamErrorHandler();
      const error = new Error('Network timeout');
      
      const result = await handler.handleError(error);
      
      expect(result.isStreamError).toBe(false);
      expect(result.canRecover).toBe(false);
      expect(result.errorType).toBe('non-stream');
    });

    it('should identify stream errors', async () => {
      const handler = new StreamErrorHandler();
      const error = new Error('body stream already read');
      const mockRequest = new Request('https://api.example.com/test');
      
      const result = await handler.handleError(error, mockRequest);
      
      expect(result.isStreamError).toBe(true);
      expect(result.errorType).toContain('stream');
    });

    it('should provide user-friendly error messages', () => {
      const handler = new StreamErrorHandler();
      
      expect(handler.getErrorMessage('stream-recoverable')).toContain('recovered');
      expect(handler.getErrorMessage('stream-unrecoverable')).toContain('could not be recovered');
      expect(handler.getErrorMessage('non-stream')).toContain('Network request failed');
    });
  });

  describe('useStreamManager hook', () => {
    it('should return stream manager functions', () => {
      const {
        handleRequest,
        isStreamConsumed,
        recreateRequest,
        handleStreamError,
        browserSupport,
        isSupported,
      } = useStreamManager();

      expect(typeof handleRequest).toBe('function');
      expect(typeof isStreamConsumed).toBe('function');
      expect(typeof recreateRequest).toBe('function');
      expect(typeof handleStreamError).toBe('function');
      expect(typeof browserSupport).toBe('object');
      expect(typeof isSupported).toBe('boolean');
    });

    it('should accept options', () => {
      const { isSupported } = useStreamManager({ enableCompatibilityMode: false });
      expect(typeof isSupported).toBe('boolean');
    });
  });
});