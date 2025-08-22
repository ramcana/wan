import { useEffect, useCallback, useRef } from 'react';
import { useKeyboardNavigation, useFocusManagement } from './use-keyboard-navigation';

interface AccessibilityOptions {
  announcePageChanges?: boolean;
  enableSkipLinks?: boolean;
  enableLandmarkNavigation?: boolean;
  enableHighContrast?: boolean;
}

export const useAccessibility = (options: AccessibilityOptions = {}) => {
  const {
    announcePageChanges = true,
    enableSkipLinks = true,
    enableLandmarkNavigation = true,
    enableHighContrast = true,
  } = options;

  const { shortcuts } = useKeyboardNavigation();
  const { focusFirstElement, focusLastElement, trapFocus } = useFocusManagement();
  const announcementRef = useRef<HTMLDivElement | null>(null);

  // Create live region for announcements
  useEffect(() => {
    if (!announcePageChanges) return;

    const liveRegion = document.createElement('div');
    liveRegion.setAttribute('aria-live', 'polite');
    liveRegion.setAttribute('aria-atomic', 'true');
    liveRegion.setAttribute('class', 'sr-only');
    liveRegion.setAttribute('id', 'accessibility-announcements');
    document.body.appendChild(liveRegion);
    announcementRef.current = liveRegion;

    return () => {
      if (announcementRef.current) {
        document.body.removeChild(announcementRef.current);
      }
    };
  }, [announcePageChanges]);

  // Announce messages to screen readers
  const announce = useCallback((message: string, priority: 'polite' | 'assertive' = 'polite') => {
    if (!announcementRef.current) return;

    announcementRef.current.setAttribute('aria-live', priority);
    announcementRef.current.textContent = message;

    // Clear after announcement
    setTimeout(() => {
      if (announcementRef.current) {
        announcementRef.current.textContent = '';
      }
    }, 1000);
  }, []);

  // Enhanced keyboard navigation with landmark support
  const handleLandmarkNavigation = useCallback((event: KeyboardEvent) => {
    if (!enableLandmarkNavigation) return;

    // Alt + R for regions/landmarks
    if (event.altKey && event.key === 'r') {
      event.preventDefault();
      const landmarks = document.querySelectorAll('[role="main"], [role="navigation"], [role="banner"], [role="contentinfo"], [role="complementary"], main, nav, header, footer, aside');
      const landmarkArray = Array.from(landmarks) as HTMLElement[];
      
      if (landmarkArray.length === 0) return;

      const currentIndex = landmarkArray.findIndex(el => el.contains(document.activeElement));
      const nextIndex = (currentIndex + 1) % landmarkArray.length;
      const nextLandmark = landmarkArray[nextIndex];
      
      nextLandmark.focus();
      nextLandmark.scrollIntoView({ behavior: 'smooth', block: 'center' });
      
      const landmarkName = nextLandmark.getAttribute('aria-label') || 
                           nextLandmark.tagName.toLowerCase() || 
                           'landmark';
      announce(`Navigated to ${landmarkName}`);
    }

    // Alt + H for headings
    if (event.altKey && event.key === 'h') {
      event.preventDefault();
      const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
      const headingArray = Array.from(headings) as HTMLElement[];
      
      if (headingArray.length === 0) return;

      const currentIndex = headingArray.findIndex(el => el.contains(document.activeElement));
      const nextIndex = (currentIndex + 1) % headingArray.length;
      const nextHeading = headingArray[nextIndex];
      
      nextHeading.focus();
      nextHeading.scrollIntoView({ behavior: 'smooth', block: 'center' });
      announce(`Navigated to heading: ${nextHeading.textContent}`);
    }

    // Alt + L for links
    if (event.altKey && event.key === 'l') {
      event.preventDefault();
      const links = document.querySelectorAll('a[href], button');
      const linkArray = Array.from(links) as HTMLElement[];
      
      if (linkArray.length === 0) return;

      const currentIndex = linkArray.findIndex(el => el === document.activeElement);
      const nextIndex = (currentIndex + 1) % linkArray.length;
      const nextLink = linkArray[nextIndex];
      
      nextLink.focus();
      nextLink.scrollIntoView({ behavior: 'smooth', block: 'center' });
      
      const linkText = nextLink.textContent || nextLink.getAttribute('aria-label') || 'interactive element';
      announce(`Navigated to ${linkText}`);
    }
  }, [enableLandmarkNavigation, announce]);

  // Skip links functionality
  const createSkipLinks = useCallback(() => {
    if (!enableSkipLinks) return;

    const skipLinksContainer = document.createElement('div');
    skipLinksContainer.className = 'skip-links';
    skipLinksContainer.innerHTML = `
      <a href="#main-content" class="skip-link">Skip to main content</a>
      <a href="#navigation" class="skip-link">Skip to navigation</a>
      <a href="#search" class="skip-link">Skip to search</a>
    `;

    // Add skip link styles
    const style = document.createElement('style');
    style.textContent = `
      .skip-links {
        position: absolute;
        top: -40px;
        left: 6px;
        z-index: 1000;
      }
      .skip-link {
        position: absolute;
        top: -40px;
        left: 6px;
        background: #000;
        color: #fff;
        padding: 8px;
        text-decoration: none;
        border-radius: 4px;
        font-size: 14px;
        font-weight: bold;
        z-index: 1001;
        transition: top 0.3s;
      }
      .skip-link:focus {
        top: 6px;
      }
    `;

    document.head.appendChild(style);
    document.body.insertBefore(skipLinksContainer, document.body.firstChild);

    return () => {
      document.head.removeChild(style);
      document.body.removeChild(skipLinksContainer);
    };
  }, [enableSkipLinks]);

  // High contrast mode toggle
  const toggleHighContrast = useCallback(() => {
    if (!enableHighContrast) return;

    const body = document.body;
    const isHighContrast = body.classList.contains('high-contrast');
    
    if (isHighContrast) {
      body.classList.remove('high-contrast');
      announce('High contrast mode disabled');
    } else {
      body.classList.add('high-contrast');
      announce('High contrast mode enabled');
    }
  }, [enableHighContrast, announce]);

  // Enhanced focus management for modals and dialogs
  const manageFocusForModal = useCallback((modalElement: HTMLElement, isOpen: boolean) => {
    if (isOpen) {
      const previouslyFocused = document.activeElement as HTMLElement;
      
      // Focus first element in modal
      setTimeout(() => {
        focusFirstElement(modalElement);
      }, 100);

      // Trap focus within modal
      const cleanup = trapFocus(modalElement);

      return () => {
        cleanup();
        // Return focus to previously focused element
        if (previouslyFocused && previouslyFocused.focus) {
          previouslyFocused.focus();
        }
      };
    }
  }, [focusFirstElement, trapFocus]);

  // Announce loading states
  const announceLoadingState = useCallback((isLoading: boolean, context: string) => {
    if (isLoading) {
      announce(`Loading ${context}`, 'polite');
    } else {
      announce(`${context} loaded`, 'polite');
    }
  }, [announce]);

  // Announce form validation errors
  const announceFormErrors = useCallback((errors: Record<string, string>) => {
    const errorCount = Object.keys(errors).length;
    if (errorCount > 0) {
      const errorMessage = errorCount === 1 
        ? `1 form error: ${Object.values(errors)[0]}`
        : `${errorCount} form errors found`;
      announce(errorMessage, 'assertive');
    }
  }, [announce]);

  // Set up event listeners
  useEffect(() => {
    document.addEventListener('keydown', handleLandmarkNavigation);
    const skipLinksCleanup = createSkipLinks();

    // Add high contrast styles
    if (enableHighContrast) {
      const style = document.createElement('style');
      style.textContent = `
        .high-contrast {
          filter: contrast(150%) brightness(150%);
        }
        .high-contrast * {
          background-color: white !important;
          color: black !important;
          border-color: black !important;
        }
        .high-contrast button, .high-contrast a {
          background-color: black !important;
          color: white !important;
          border: 2px solid white !important;
        }
        .high-contrast button:hover, .high-contrast a:hover {
          background-color: white !important;
          color: black !important;
          border: 2px solid black !important;
        }
      `;
      document.head.appendChild(style);

      return () => {
        document.removeEventListener('keydown', handleLandmarkNavigation);
        if (skipLinksCleanup) skipLinksCleanup();
        document.head.removeChild(style);
      };
    }

    return () => {
      document.removeEventListener('keydown', handleLandmarkNavigation);
      if (skipLinksCleanup) skipLinksCleanup();
    };
  }, [handleLandmarkNavigation, createSkipLinks, enableHighContrast]);

  return {
    announce,
    toggleHighContrast,
    manageFocusForModal,
    announceLoadingState,
    announceFormErrors,
    shortcuts,
    focusFirstElement,
    focusLastElement,
    trapFocus,
  };
};

// Hook for managing ARIA attributes dynamically
export const useAriaAttributes = () => {
  const setAriaLabel = useCallback((element: HTMLElement, label: string) => {
    element.setAttribute('aria-label', label);
  }, []);

  const setAriaDescribedBy = useCallback((element: HTMLElement, describedById: string) => {
    element.setAttribute('aria-describedby', describedById);
  }, []);

  const setAriaExpanded = useCallback((element: HTMLElement, expanded: boolean) => {
    element.setAttribute('aria-expanded', expanded.toString());
  }, []);

  const setAriaSelected = useCallback((element: HTMLElement, selected: boolean) => {
    element.setAttribute('aria-selected', selected.toString());
  }, []);

  const setAriaPressed = useCallback((element: HTMLElement, pressed: boolean) => {
    element.setAttribute('aria-pressed', pressed.toString());
  }, []);

  const setAriaDisabled = useCallback((element: HTMLElement, disabled: boolean) => {
    element.setAttribute('aria-disabled', disabled.toString());
    if (disabled) {
      element.setAttribute('tabindex', '-1');
    } else {
      element.removeAttribute('tabindex');
    }
  }, []);

  const setAriaLive = useCallback((element: HTMLElement, live: 'off' | 'polite' | 'assertive') => {
    element.setAttribute('aria-live', live);
  }, []);

  return {
    setAriaLabel,
    setAriaDescribedBy,
    setAriaExpanded,
    setAriaSelected,
    setAriaPressed,
    setAriaDisabled,
    setAriaLive,
  };
};

// Hook for screen reader announcements with context
export const useScreenReaderAnnouncements = () => {
  const announceNavigation = useCallback((pageName: string) => {
    const announcement = `Navigated to ${pageName} page`;
    const event = new CustomEvent('accessibility-announcement', {
      detail: { message: announcement, priority: 'polite' }
    });
    document.dispatchEvent(event);
  }, []);

  const announceAction = useCallback((action: string, result?: string) => {
    const message = result ? `${action}: ${result}` : action;
    const event = new CustomEvent('accessibility-announcement', {
      detail: { message, priority: 'polite' }
    });
    document.dispatchEvent(event);
  }, []);

  const announceError = useCallback((error: string) => {
    const event = new CustomEvent('accessibility-announcement', {
      detail: { message: `Error: ${error}`, priority: 'assertive' }
    });
    document.dispatchEvent(event);
  }, []);

  const announceSuccess = useCallback((message: string) => {
    const event = new CustomEvent('accessibility-announcement', {
      detail: { message: `Success: ${message}`, priority: 'polite' }
    });
    document.dispatchEvent(event);
  }, []);

  return {
    announceNavigation,
    announceAction,
    announceError,
    announceSuccess,
  };
};