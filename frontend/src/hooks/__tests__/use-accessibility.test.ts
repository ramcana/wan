import { renderHook, act } from '@testing-library/react';
import { vi } from 'vitest';
import { useAccessibility, useAriaAttributes, useScreenReaderAnnouncements } from '../use-accessibility';

// Mock DOM methods
const mockFocus = vi.fn();
const mockScrollIntoView = vi.fn();
const mockQuerySelectorAll = vi.fn();
const mockGetElementById = vi.fn();
const mockCreateElement = vi.fn();
const mockAppendChild = vi.fn();
const mockRemoveChild = vi.fn();
const mockAddEventListener = vi.fn();
const mockRemoveEventListener = vi.fn();

// Setup DOM mocks
beforeEach(() => {
  Object.defineProperty(document, 'querySelectorAll', {
    value: mockQuerySelectorAll,
    writable: true,
  });
  
  Object.defineProperty(document, 'getElementById', {
    value: mockGetElementById,
    writable: true,
  });
  
  Object.defineProperty(document, 'createElement', {
    value: mockCreateElement,
    writable: true,
  });
  
  Object.defineProperty(document.body, 'appendChild', {
    value: mockAppendChild,
    writable: true,
  });
  
  Object.defineProperty(document.body, 'removeChild', {
    value: mockRemoveChild,
    writable: true,
  });
  
  Object.defineProperty(document, 'addEventListener', {
    value: mockAddEventListener,
    writable: true,
  });
  
  Object.defineProperty(document, 'removeEventListener', {
    value: mockRemoveEventListener,
    writable: true,
  });

  // Mock HTML elements
  const mockElement = {
    focus: mockFocus,
    scrollIntoView: mockScrollIntoView,
    setAttribute: vi.fn(),
    getAttribute: vi.fn(),
    contains: vi.fn(),
    textContent: '',
    tagName: 'DIV',
    classList: {
      contains: vi.fn(),
      add: vi.fn(),
      remove: vi.fn(),
    },
  };

  mockCreateElement.mockReturnValue(mockElement);
  mockQuerySelectorAll.mockReturnValue([mockElement]);
  mockGetElementById.mockReturnValue(mockElement);
});

afterEach(() => {
  vi.clearAllMocks();
});

describe('useAccessibility', () => {
  it('should initialize with default options', () => {
    const { result } = renderHook(() => useAccessibility());
    
    expect(result.current.announce).toBeDefined();
    expect(result.current.toggleHighContrast).toBeDefined();
    expect(result.current.manageFocusForModal).toBeDefined();
    expect(result.current.announceLoadingState).toBeDefined();
    expect(result.current.announceFormErrors).toBeDefined();
  });

  it('should create live region for announcements', () => {
    renderHook(() => useAccessibility({ announcePageChanges: true }));
    
    expect(mockCreateElement).toHaveBeenCalledWith('div');
    expect(mockAppendChild).toHaveBeenCalled();
  });

  it('should announce messages', () => {
    const { result } = renderHook(() => useAccessibility());
    
    act(() => {
      result.current.announce('Test message', 'polite');
    });

    // Should set textContent on the live region
    expect(mockCreateElement).toHaveBeenCalled();
  });

  it('should toggle high contrast mode', () => {
    const { result } = renderHook(() => useAccessibility({ enableHighContrast: true }));
    
    act(() => {
      result.current.toggleHighContrast();
    });

    // Should add/remove high-contrast class
    expect(document.body.classList.add).toHaveBeenCalledWith('high-contrast');
  });

  it('should announce loading states', () => {
    const { result } = renderHook(() => useAccessibility());
    
    act(() => {
      result.current.announceLoadingState(true, 'data');
    });

    // Should announce loading message
    expect(result.current.announce).toBeDefined();
  });

  it('should announce form errors', () => {
    const { result } = renderHook(() => useAccessibility());
    
    const errors = { email: 'Invalid email', password: 'Too short' };
    
    act(() => {
      result.current.announceFormErrors(errors);
    });

    // Should announce error count and messages
    expect(result.current.announce).toBeDefined();
  });

  it('should handle landmark navigation', () => {
    const { result } = renderHook(() => useAccessibility({ enableLandmarkNavigation: true }));
    
    // Mock keyboard event
    const mockEvent = new KeyboardEvent('keydown', { 
      key: 'r', 
      altKey: true 
    });
    
    act(() => {
      document.dispatchEvent(mockEvent);
    });

    expect(mockQuerySelectorAll).toHaveBeenCalledWith(
      '[role="main"], [role="navigation"], [role="banner"], [role="contentinfo"], [role="complementary"], main, nav, header, footer, aside'
    );
  });

  it('should create skip links when enabled', () => {
    renderHook(() => useAccessibility({ enableSkipLinks: true }));
    
    expect(mockCreateElement).toHaveBeenCalled();
    expect(mockAppendChild).toHaveBeenCalled();
  });

  it('should manage focus for modals', () => {
    const { result } = renderHook(() => useAccessibility());
    
    const mockModal = document.createElement('div');
    
    act(() => {
      const cleanup = result.current.manageFocusForModal(mockModal, true);
      expect(cleanup).toBeDefined();
    });
  });
});

describe('useAriaAttributes', () => {
  it('should provide ARIA attribute setters', () => {
    const { result } = renderHook(() => useAriaAttributes());
    
    expect(result.current.setAriaLabel).toBeDefined();
    expect(result.current.setAriaDescribedBy).toBeDefined();
    expect(result.current.setAriaExpanded).toBeDefined();
    expect(result.current.setAriaSelected).toBeDefined();
    expect(result.current.setAriaPressed).toBeDefined();
    expect(result.current.setAriaDisabled).toBeDefined();
    expect(result.current.setAriaLive).toBeDefined();
  });

  it('should set aria-label attribute', () => {
    const { result } = renderHook(() => useAriaAttributes());
    const mockElement = document.createElement('div');
    const mockSetAttribute = vi.fn();
    mockElement.setAttribute = mockSetAttribute;
    
    act(() => {
      result.current.setAriaLabel(mockElement, 'Test label');
    });

    expect(mockSetAttribute).toHaveBeenCalledWith('aria-label', 'Test label');
  });

  it('should set aria-expanded attribute', () => {
    const { result } = renderHook(() => useAriaAttributes());
    const mockElement = document.createElement('div');
    const mockSetAttribute = vi.fn();
    mockElement.setAttribute = mockSetAttribute;
    
    act(() => {
      result.current.setAriaExpanded(mockElement, true);
    });

    expect(mockSetAttribute).toHaveBeenCalledWith('aria-expanded', 'true');
  });

  it('should handle disabled state with tabindex', () => {
    const { result } = renderHook(() => useAriaAttributes());
    const mockElement = document.createElement('div');
    const mockSetAttribute = vi.fn();
    const mockRemoveAttribute = vi.fn();
    mockElement.setAttribute = mockSetAttribute;
    mockElement.removeAttribute = mockRemoveAttribute;
    
    act(() => {
      result.current.setAriaDisabled(mockElement, true);
    });

    expect(mockSetAttribute).toHaveBeenCalledWith('aria-disabled', 'true');
    expect(mockSetAttribute).toHaveBeenCalledWith('tabindex', '-1');
  });
});

describe('useScreenReaderAnnouncements', () => {
  it('should provide announcement functions', () => {
    const { result } = renderHook(() => useScreenReaderAnnouncements());
    
    expect(result.current.announceNavigation).toBeDefined();
    expect(result.current.announceAction).toBeDefined();
    expect(result.current.announceError).toBeDefined();
    expect(result.current.announceSuccess).toBeDefined();
  });

  it('should dispatch custom events for announcements', () => {
    const { result } = renderHook(() => useScreenReaderAnnouncements());
    const mockDispatchEvent = vi.fn();
    document.dispatchEvent = mockDispatchEvent;
    
    act(() => {
      result.current.announceNavigation('Home');
    });

    expect(mockDispatchEvent).toHaveBeenCalledWith(
      expect.objectContaining({
        type: 'accessibility-announcement',
        detail: {
          message: 'Navigated to Home page',
          priority: 'polite'
        }
      })
    );
  });

  it('should announce errors with assertive priority', () => {
    const { result } = renderHook(() => useScreenReaderAnnouncements());
    const mockDispatchEvent = vi.fn();
    document.dispatchEvent = mockDispatchEvent;
    
    act(() => {
      result.current.announceError('Something went wrong');
    });

    expect(mockDispatchEvent).toHaveBeenCalledWith(
      expect.objectContaining({
        detail: {
          message: 'Error: Something went wrong',
          priority: 'assertive'
        }
      })
    );
  });

  it('should announce actions with results', () => {
    const { result } = renderHook(() => useScreenReaderAnnouncements());
    const mockDispatchEvent = vi.fn();
    document.dispatchEvent = mockDispatchEvent;
    
    act(() => {
      result.current.announceAction('Save', 'completed successfully');
    });

    expect(mockDispatchEvent).toHaveBeenCalledWith(
      expect.objectContaining({
        detail: {
          message: 'Save: completed successfully',
          priority: 'polite'
        }
      })
    );
  });
});