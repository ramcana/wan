import React, { createContext, useContext, useEffect, useState } from "react";
import {
  useAccessibility,
  useScreenReaderAnnouncements,
} from "../../hooks/use-accessibility";
import { useLocation } from "react-router-dom";

interface AccessibilityContextType {
  announce: (message: string, priority?: "polite" | "assertive") => void;
  toggleHighContrast: () => void;
  manageFocusForModal: (
    modalElement: HTMLElement,
    isOpen: boolean
  ) => (() => void) | undefined;
  announceLoadingState: (isLoading: boolean, context: string) => void;
  announceFormErrors: (errors: Record<string, string>) => void;
  isHighContrast: boolean;
  announceNavigation: (pageName: string) => void;
  announceAction: (action: string, result?: string) => void;
  announceError: (error: string) => void;
  announceSuccess: (message: string) => void;
}

const AccessibilityContext = createContext<
  AccessibilityContextType | undefined
>(undefined);

export const useAccessibilityContext = () => {
  const context = useContext(AccessibilityContext);
  if (!context) {
    throw new Error(
      "useAccessibilityContext must be used within AccessibilityProvider"
    );
  }
  return context;
};

interface AccessibilityProviderProps {
  children: React.ReactNode;
}

export const AccessibilityProvider: React.FC<AccessibilityProviderProps> = ({
  children,
}) => {
  const [isHighContrast, setIsHighContrast] = useState(false);
  const location = useLocation();

  const {
    announce,
    toggleHighContrast: toggleHighContrastHook,
    manageFocusForModal,
    announceLoadingState,
    announceFormErrors,
  } = useAccessibility({
    announcePageChanges: true,
    enableSkipLinks: true,
    enableLandmarkNavigation: true,
    enableHighContrast: true,
  });

  const { announceNavigation, announceAction, announceError, announceSuccess } =
    useScreenReaderAnnouncements();

  // Enhanced high contrast toggle
  const toggleHighContrast = () => {
    setIsHighContrast((prev) => !prev);
    toggleHighContrastHook();
  };

  // Announce page changes
  useEffect(() => {
    const getPageName = (pathname: string) => {
      switch (pathname) {
        case "/":
        case "/generation":
          return "Generation";
        case "/queue":
          return "Queue";
        case "/outputs":
          return "Outputs";
        case "/system":
          return "System";
        case "/lora":
          return "LoRA";
        default:
          return "Page";
      }
    };

    const pageName = getPageName(location.pathname);
    announceNavigation(pageName);
  }, [location.pathname, announceNavigation]);

  // Listen for custom accessibility events
  useEffect(() => {
    const handleAccessibilityAnnouncement = (event: CustomEvent) => {
      const { message, priority } = event.detail;
      announce(message, priority);
    };

    document.addEventListener(
      "accessibility-announcement",
      handleAccessibilityAnnouncement as EventListener
    );

    return () => {
      document.removeEventListener(
        "accessibility-announcement",
        handleAccessibilityAnnouncement as EventListener
      );
    };
  }, [announce]);

  // Set up global accessibility attributes
  useEffect(() => {
    // Set language
    document.documentElement.lang = "en";

    // Add accessibility metadata
    const metaDescription = document.querySelector('meta[name="description"]');
    if (!metaDescription) {
      const meta = document.createElement("meta");
      meta.name = "description";
      meta.content =
        "Wan2.2 Video Generation - Professional AI-powered video creation tool with accessibility features";
      document.head.appendChild(meta);
    }

    // Add viewport meta for mobile accessibility
    const metaViewport = document.querySelector('meta[name="viewport"]');
    if (!metaViewport) {
      const meta = document.createElement("meta");
      meta.name = "viewport";
      meta.content = "width=device-width, initial-scale=1.0, maximum-scale=5.0";
      document.head.appendChild(meta);
    }

    // Add theme color for better mobile experience
    const metaTheme = document.querySelector('meta[name="theme-color"]');
    if (!metaTheme) {
      const meta = document.createElement("meta");
      meta.name = "theme-color";
      meta.content = "#667eea";
      document.head.appendChild(meta);
    }
  }, []);

  // Add global keyboard shortcuts help
  useEffect(() => {
    const handleKeyboardHelp = (event: KeyboardEvent) => {
      if (event.key === "?" && event.shiftKey) {
        event.preventDefault();

        const helpText = `
          Keyboard Shortcuts:
          Alt + 1: Go to Generation page
          Alt + 2: Go to Queue page
          Alt + 3: Go to System page
          Alt + 4: Go to Outputs page
          Alt + R: Navigate between page regions
          Alt + H: Navigate between headings
          Alt + L: Navigate between links
          Escape: Close modals and dialogs
          Shift + ?: Show this help
        `;

        announce(helpText, "polite");
      }
    };

    document.addEventListener("keydown", handleKeyboardHelp);
    return () => document.removeEventListener("keydown", handleKeyboardHelp);
  }, [announce]);

  // Manage focus for route changes
  useEffect(() => {
    // Focus main content area after route change
    const mainContent = document.getElementById("main-content");
    if (mainContent) {
      mainContent.focus();
    }
  }, [location.pathname]);

  const contextValue: AccessibilityContextType = {
    announce,
    toggleHighContrast,
    manageFocusForModal,
    announceLoadingState,
    announceFormErrors,
    isHighContrast,
    announceNavigation,
    announceAction,
    announceError,
    announceSuccess,
  };

  return (
    <AccessibilityContext.Provider value={contextValue}>
      {children}
    </AccessibilityContext.Provider>
  );
};

// Component for managing focus on route changes
export const RouteFocusManager: React.FC = () => {
  const location = useLocation();

  useEffect(() => {
    // Skip focus management for hash changes
    if (location.hash) {
      const element = document.getElementById(location.hash.slice(1));
      if (element) {
        element.focus();
        return;
      }
    }

    // Focus the main heading or main content
    const mainHeading = document.querySelector("h1");
    const mainContent = document.getElementById("main-content");

    if (mainHeading) {
      (mainHeading as HTMLElement).focus();
    } else if (mainContent) {
      mainContent.focus();
    }
  }, [location]);

  return null;
};

// Component for screen reader only content
export const ScreenReaderOnly: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  return <span className="sr-only">{children}</span>;
};

// Component for accessible loading states
export const AccessibleLoading: React.FC<{
  isLoading: boolean;
  loadingText?: string;
  children: React.ReactNode;
}> = ({ isLoading, loadingText = "Loading", children }) => {
  const { announceLoadingState } = useAccessibilityContext();

  useEffect(() => {
    announceLoadingState(isLoading, loadingText);
  }, [isLoading, loadingText, announceLoadingState]);

  return (
    <div role="region" aria-live="polite" aria-busy={isLoading}>
      {isLoading ? (
        <div role="status" aria-label={`${loadingText}...`}>
          <ScreenReaderOnly>{loadingText}...</ScreenReaderOnly>
          {/* Loading spinner or skeleton */}
        </div>
      ) : (
        children
      )}
    </div>
  );
};

// Component for accessible form errors
export const AccessibleFormErrors: React.FC<{
  errors: Record<string, string>;
  fieldId?: string;
}> = ({ errors, fieldId }) => {
  const { announceFormErrors } = useAccessibilityContext();

  useEffect(() => {
    if (Object.keys(errors).length > 0) {
      announceFormErrors(errors);
    }
  }, [errors, announceFormErrors]);

  if (Object.keys(errors).length === 0) return null;

  return (
    <div
      role="alert"
      aria-live="assertive"
      id={fieldId ? `${fieldId}-error` : undefined}
      className="text-red-600 text-sm mt-1"
    >
      {Object.entries(errors).map(([field, message]) => (
        <div key={field}>
          <ScreenReaderOnly>Error for {field}: </ScreenReaderOnly>
          {message}
        </div>
      ))}
    </div>
  );
};
