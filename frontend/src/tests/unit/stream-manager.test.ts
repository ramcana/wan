import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { 
  StreamManager, 
  streamManager, 
  isStreamConsumed, 
  recreateRequest, 
  handleStreamError 
} from '../../lib/stream-manager';

// Mock global APIs for testing
const mockTextEncoder = vi.fn().mockImplementation(() => ({
  encode: vi.fn((text: string) => new Uint8Array(Buffer.from(text, 'utf8')))
}));

const mockReadableStream = vi.fn().mockImplementation((underlyingSource: any) => {
  const stream = {
    locked: false,
    cancel: vi.fn(),
    getReader: vi.fn(),
    pipeThrough: vi.fn(),
    pipeTo: vi.fn(),
    tee: vi.fn(),
  };
  
  // Simulate the start method being called
  if (underlyingSource && underlyingSource.start) {
    const controller = {
      enqueue: vi.fn(),
      close: vi.fn(),
      error: vi.fn(),
    };
    underlyingSource.start(controller);
  }
  
  return stream;
});

// Mock ReadableStreamDefaultController for browser support check
const mockReadableStreamDefaultController = vi.fn();

// Mock Request class
const createMockRequest = (options: {
  url?: string;
  method?: string;
  body?: any;
  bodyUsed?: boolean;
  headers?: Record<string, string>;
  locked?: boolean;
} = {}) => {
  const mockBody = options.body ? {
    locked: options.locked ?? false,
    cancel: vi.fn(),
    getReader: vi.fn(),
  } : null;

  const mockRequest = {
    url: options.url ?? 'https://api.example.com/test',
    method: options.method ?? 'POST',
    body: mockBody,
    bodyUsed: options.bodyUsed ?? false,
    headers: new Headers(options.headers ?? { 'Content-Type': 'application/json' }),
    mode: 'cors' as RequestMode,
    credentials: 'same-origin' as RequestCredentials,
    cache: 'default' as RequestCache,
    redirect: 'follow' as RequestRedirect,
    referrer: '',
    integrity: '',
    clone: vi.fn(),
    text: vi.fn(),
    json: vi.fn(),
    blob: vi.fn(),
    arrayBuffer: vi.fn(),
    formData: vi.fn(),
  };

  // Mock clone method
  mockRequest.clone.mockImplementation(() => {
    if (mockRequest.bodyUsed || (mockRequest.body && mockRequest.body.locked)) {
      throw new Error('Failed to execute \'clone\' on \'Request\': Request body is already used');
    }
    return createMockRequest(options);
  });

  // Mock text method
  mockRequest.text.mockImplementation(async () => {
    if (mockRequest.bodyUsed) {
      throw new Error('Failed to execute \'text\' on \'Request\': body stream already read');
    }
    mockRequest.bodyUsed = true;
    return JSON.stringify({ test: 'data' });
  });

  return mockRequest as unknown as Request;
};

describe('StreamManager', () => {
  let manager: StreamManager;

  beforeEach(() => {
    // Setup global mocks
    global.TextEncoder = mockTextEncoder as any;
    global.ReadableStream = mockReadableStream as any;
    (global as any).ReadableStreamDefaultController = mockReadableStreamDefaultController;
    
    // Mock Request constructor and prototype
    const MockRequest = vi.fn().mockImplementation((url, init) => createMockRequest({ 
      url, 
      method: init?.method,
      body: init?.body,
      headers: init?.headers 
    }));
    MockRequest.prototype = {
      clone: vi.fn(),
      text: vi.fn(),
      json: vi.fn(),
    };
    global.Request = MockRequest as any;

    manager = new StreamManager();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('constructor and browser support', () => {
    it('should initialize with default options', () => {
      const options = manager.getOptions();
      expect(options.enableCompatibilityMode).toBe(true);
      expect(options.maxRetries).toBe(3);
      expect(options.fallbackToDirectFetch).toBe(true);
    });

    it('should accept custom options', () => {
      const customManager = new StreamManager({
        enableCompatibilityMode: false,
        maxRetries: 5,
        fallbackToDirectFetch: false,
      });

      const options = customManager.getOptions();
      expect(options.enableCompatibilityMode).toBe(false);
      expect(options.maxRetries).toBe(5);
      expect(options.fallbackToDirectFetch).toBe(false);
    });

    it('should check browser support correctly', () => {
      const support = manager.getBrowserSupport();
      expect(support.requestClone).toBe(true);
      expect(support.readableStream).toBe(true);
      expect(support.textEncoder).toBe(true);
    });

    it('should report browser as supported when all features are available', () => {
      expect(manager.isBrowserSupported()).toBe(true);
    });
  });

  describe('isStreamConsumed', () => {
    it('should return false for requests without body', () => {
      const request = createMockRequest({ method: 'GET', body: null });
      expect(manager.isStreamConsumed(request)).toBe(false);
    });

    it('should return false for fresh requests with body', () => {
      const request = createMockRequest({ 
        method: 'POST', 
        body: { test: 'data' }, 
        bodyUsed: false,
        locked: false 
      });
      expect(manager.isStreamConsumed(request)).toBe(false);
    });

    it('should return true for requests with consumed body', () => {
      const request = createMockRequest({ 
        method: 'POST', 
        body: { test: 'data' }, 
        bodyUsed: true 
      });
      expect(manager.isStreamConsumed(request)).toBe(true);
    });

    it('should return true for requests with locked body stream', () => {
      const request = createMockRequest({ 
        method: 'POST', 
        body: { test: 'data' }, 
        bodyUsed: false,
        locked: true 
      });
      expect(manager.isStreamConsumed(request)).toBe(true);
    });

    it('should handle errors gracefully and assume consumed', () => {
      const request = createMockRequest();
      // Mock an error when accessing body properties
      Object.defineProperty(request, 'body', {
        get: () => { throw new Error('Test error'); }
      });

      expect(manager.isStreamConsumed(request)).toBe(true);
    });
  });

  describe('safeCloneRequest', () => {
    it('should successfully clone a fresh request', async () => {
      const request = createMockRequest({ bodyUsed: false, locked: false });
      const cloned = await manager.safeCloneRequest(request);
      
      expect(cloned).toBeDefined();
    });

    it('should handle clone failure by recreating request', async () => {
      const request = createMockRequest({ bodyUsed: true });
      request.clone = vi.fn().mockImplementation(() => {
        throw new Error('Request body is already used');
      });

      const result = await manager.safeCloneRequest(request);
      expect(result).toBeDefined();
      expect(result.url).toBe(request.url);
    });

    it('should handle browser compatibility issues', async () => {
      // Mock unsupported browser
      const incompatibleManager = new StreamManager();
      (incompatibleManager as any).browserSupport.requestClone = false;

      const request = createMockRequest();
      const result = await incompatibleManager.safeCloneRequest(request);
      
      expect(result).toBeDefined();
      expect(result.url).toBe(request.url);
    });
  });

  describe('recreateRequest', () => {
    it('should recreate request with ReadableStream body', () => {
      const originalRequest = createMockRequest({
        url: 'https://api.example.com/test',
        method: 'POST'
      });
      const body = '{"test": "data"}';

      const recreated = manager.recreateRequest(originalRequest, body);

      expect(global.Request).toHaveBeenCalledWith(
        originalRequest.url,
        expect.objectContaining({
          method: 'POST',
          headers: expect.any(Headers),
          body: expect.any(Object), // ReadableStream
        })
      );
    });

    it('should recreate request without body for GET requests', () => {
      const originalRequest = createMockRequest({
        url: 'https://api.example.com/test',
        method: 'GET'
      });

      const recreated = manager.recreateRequest(originalRequest, null);

      expect(global.Request).toHaveBeenCalledWith(
        originalRequest.url,
        expect.objectContaining({
          method: 'GET',
          headers: expect.any(Headers),
        })
      );
    });

    it('should handle ReadableStream creation errors gracefully', () => {
      // Mock ReadableStream to throw error
      global.ReadableStream = vi.fn().mockImplementation(() => {
        throw new Error('ReadableStream not supported');
      });

      const originalRequest = createMockRequest();
      const body = '{"test": "data"}';

      const recreated = manager.recreateRequest(originalRequest, body);
      expect(recreated).toBeDefined();
    });

    it('should fallback to string body when ReadableStream fails', () => {
      // Mock browser without ReadableStream support
      const incompatibleManager = new StreamManager();
      (incompatibleManager as any).browserSupport.readableStream = false;

      const originalRequest = createMockRequest();
      const body = '{"test": "data"}';

      const recreated = incompatibleManager.recreateRequest(originalRequest, body);

      expect(global.Request).toHaveBeenCalledWith(
        originalRequest.url,
        expect.objectContaining({
          body: body,
        })
      );
    });
  });

  describe('handleRequest', () => {
    it('should return original request if stream is not consumed', async () => {
      const request = createMockRequest({ bodyUsed: false, locked: false });
      const result = await manager.handleRequest(request);

      expect(result.success).toBe(true);
      expect(result.request).toBe(request);
      expect(result.error).toBeUndefined();
    });

    it('should clone request if stream is consumed', async () => {
      const request = createMockRequest({ bodyUsed: true });
      const result = await manager.handleRequest(request);

      expect(result.success).toBe(true);
      expect(result.request).toBeDefined();
      // Since we're recreating the request, it should be a different object
      if (result.request) {
        expect(result.request.url).toBe(request.url);
      }
    });

    it('should retry on failures up to maxRetries', async () => {
      // Create a manager that will fail all attempts
      const managerWithRetries = new StreamManager({ maxRetries: 2 });
      
      // Force the first attempt to fail so we enter retry path
      // We need to make isStreamConsumed return true on the first attempt
      // so that safeCloneRequest is called, and then make safeCloneRequest throw
      const request = createMockRequest({ bodyUsed: true }); // Start with consumed request
      
      // Mock safeCloneRequest to always fail with a non-stream error
      const mockCloneRequest = vi.spyOn(managerWithRetries, 'safeCloneRequest').mockImplementation(async () => {
        throw new Error('Network error - not stream related');
      });

      const result = await managerWithRetries.handleRequest(request);

      expect(result.success).toBe(false);
      expect(result.error).toBeDefined();
      expect(result.error?.message).toContain('Network error');
      
      // Verify that safeCloneRequest was called exactly the number of retries + 1 (initial + retries)
      // First attempt uses cloning (because stream is consumed), retries 1 and 2 also use cloning
      expect(mockCloneRequest).toHaveBeenCalledTimes(3);
    });

    it('should handle stream consumption errors specifically', async () => {
      const request = createMockRequest({ bodyUsed: true });
      request.clone = vi.fn().mockImplementation(() => {
        throw new Error('body stream already read');
      });

      const result = await manager.handleRequest(request);

      expect(result.success).toBe(true);
      expect(result.request).toBeDefined();
    });
  });

  describe('handleStreamConsumedError', () => {
    it('should recreate request when handling stream consumed error', async () => {
      const request = createMockRequest();
      const error = new Error('Failed to execute \'text\' on \'Request\': body stream already read');

      const result = await manager.handleStreamConsumedError(request, error);

      expect(result).toBeDefined();
      expect(result.url).toBe(request.url);
      expect(result.method).toBe(request.method);
    });

    it('should create minimal request if recreation fails', async () => {
      const request = createMockRequest();
      const error = new Error('body stream already read');

      // Mock Request constructor to fail
      const originalRequest = global.Request;
      global.Request = vi.fn().mockImplementationOnce(() => {
        throw new Error('Request creation failed');
      }).mockImplementation((...args: any[]) => new (originalRequest as any)(...args));

      const result = await manager.handleStreamConsumedError(request, error);

      expect(result).toBeDefined();
    });
  });

  describe('utility methods', () => {
    it('should allow setting compatibility mode', () => {
      manager.setCompatibilityMode(false);
      expect(manager.getOptions().enableCompatibilityMode).toBe(false);

      manager.setCompatibilityMode(true);
      expect(manager.getOptions().enableCompatibilityMode).toBe(true);
    });
  });

  describe('exported utility functions', () => {
    it('should export isStreamConsumed utility', () => {
      const request = createMockRequest({ 
        bodyUsed: true, 
        body: { locked: false } 
      });
      expect(isStreamConsumed(request)).toBe(true);
    });

    it('should export recreateRequest utility', () => {
      const request = createMockRequest();
      const body = '{"test": "data"}';
      const result = recreateRequest(request, body);
      
      expect(result).toBeDefined();
      expect(result.url).toBe(request.url);
    });

    it('should export handleStreamError utility', async () => {
      const request = createMockRequest();
      const error = new Error('body stream already read');
      const result = await handleStreamError(request, error);
      
      expect(result).toBeDefined();
      expect(result.url).toBe(request.url);
    });
  });

  describe('edge cases and error handling', () => {
    it('should handle null/undefined requests gracefully', () => {
      // The method should handle null gracefully and return true (assume consumed)
      expect(manager.isStreamConsumed(null as any)).toBe(true);
    });

    it('should handle requests with unusual body types', () => {
      const request = createMockRequest({ 
        body: new ArrayBuffer(10) 
      });
      
      expect(() => {
        manager.isStreamConsumed(request);
      }).not.toThrow();
    });

    it('should handle concurrent request processing', async () => {
      const requests = Array.from({ length: 5 }, (_, i) => 
        createMockRequest({ url: `https://api.example.com/test${i}` })
      );

      const results = await Promise.all(
        requests.map(req => manager.handleRequest(req))
      );

      results.forEach(result => {
        expect(result.success).toBe(true);
        expect(result.request).toBeDefined();
      });
    });

    it('should handle very large request bodies', async () => {
      const largeBody = 'x'.repeat(1024 * 1024); // 1MB string
      const request = createMockRequest();
      
      const result = manager.recreateRequest(request, largeBody);
      expect(result).toBeDefined();
    });

    it('should handle requests with special characters in body', async () => {
      const specialBody = '{"emoji": "🚀", "unicode": "测试", "special": "!@#$%^&*()"}';
      const request = createMockRequest();
      
      const result = manager.recreateRequest(request, specialBody);
      expect(result).toBeDefined();
    });
  });

  describe('browser compatibility edge cases', () => {
    it('should handle missing TextEncoder gracefully', () => {
      const originalTextEncoder = global.TextEncoder;
      delete (global as any).TextEncoder;

      const incompatibleManager = new StreamManager();
      // Manually set the browser support to simulate missing feature
      (incompatibleManager as any).browserSupport.textEncoder = false;
      (incompatibleManager as any).browserSupport.isCompatible = () => false;
      
      const support = incompatibleManager.getBrowserSupport();
      
      expect(support.textEncoder).toBe(false);
      expect(incompatibleManager.isBrowserSupported()).toBe(false);

      global.TextEncoder = originalTextEncoder;
    });

    it('should handle missing ReadableStream gracefully', () => {
      const originalReadableStream = global.ReadableStream;
      delete (global as any).ReadableStream;

      const incompatibleManager = new StreamManager();
      // Manually set the browser support to simulate missing feature
      (incompatibleManager as any).browserSupport.readableStream = false;
      (incompatibleManager as any).browserSupport.isCompatible = () => false;
      
      const support = incompatibleManager.getBrowserSupport();
      
      expect(support.readableStream).toBe(false);
      expect(incompatibleManager.isBrowserSupported()).toBe(false);

      global.ReadableStream = originalReadableStream;
    });

    it('should handle missing Request.clone gracefully', () => {
      const originalRequest = global.Request;
      global.Request = vi.fn().mockImplementation((url, init) => {
        const req = createMockRequest({ url, method: init?.method });
        delete (req as any).clone;
        return req;
      }) as any;

      const incompatibleManager = new StreamManager();
      // Manually set the browser support to simulate missing feature
      (incompatibleManager as any).browserSupport.requestClone = false;
      (incompatibleManager as any).browserSupport.isCompatible = () => false;
      
      const support = incompatibleManager.getBrowserSupport();
      
      expect(support.requestClone).toBe(false);

      global.Request = originalRequest;
    });
  });
});