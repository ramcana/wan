import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { BrowserRouter } from "react-router-dom";
import { vi } from "vitest";
import {
  AccessibilityProvider,
  useAccessibilityContext,
  RouteFocusManager,
  ScreenReaderOnly,
  AccessibleLoading,
  AccessibleFormErrors,
} from "../AccessibilityProvider";

// Mock the hooks
vi.mock("../../../hooks/use-accessibility", () => ({
  useAccessibility: () => ({
    announce: vi.fn(),
    toggleHighContrast: vi.fn(),
    manageFocusForModal: vi.fn(),
    announceLoadingState: vi.fn(),
    announceFormErrors: vi.fn(),
  }),
  useScreenReaderAnnouncements: () => ({
    announceNavigation: vi.fn(),
    announceAction: vi.fn(),
    announceError: vi.fn(),
    announceSuccess: vi.fn(),
  }),
}));

// Test component that uses the context
const TestComponent: React.FC = () => {
  const {
    announce,
    toggleHighContrast,
    isHighContrast,
    announceError,
    announceSuccess,
  } = useAccessibilityContext();

  return (
    <div>
      <button onClick={() => announce("Test message")}>Announce</button>
      <button onClick={toggleHighContrast}>Toggle High Contrast</button>
      <button onClick={() => announceError("Test error")}>
        Announce Error
      </button>
      <button onClick={() => announceSuccess("Test success")}>
        Announce Success
      </button>
      <span data-testid="high-contrast">{isHighContrast.toString()}</span>
    </div>
  );
};

const renderWithProviders = (component: React.ReactElement) => {
  return render(
    <BrowserRouter>
      <AccessibilityProvider>{component}</AccessibilityProvider>
    </BrowserRouter>
  );
};

describe("AccessibilityProvider", () => {
  beforeEach(() => {
    // Mock document methods
    document.documentElement.lang = "";
    document.head.appendChild = vi.fn();
    document.addEventListener = vi.fn();
    document.removeEventListener = vi.fn();
  });

  it("should provide accessibility context", () => {
    renderWithProviders(<TestComponent />);

    expect(screen.getByText("Announce")).toBeInTheDocument();
    expect(screen.getByText("Toggle High Contrast")).toBeInTheDocument();
    expect(screen.getByText("Announce Error")).toBeInTheDocument();
    expect(screen.getByText("Announce Success")).toBeInTheDocument();
  });

  it("should set document language", () => {
    renderWithProviders(<TestComponent />);

    expect(document.documentElement.lang).toBe("en");
  });

  it("should throw error when used outside provider", () => {
    // Suppress console.error for this test
    const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    expect(() => {
      render(<TestComponent />);
    }).toThrow(
      "useAccessibilityContext must be used within AccessibilityProvider"
    );

    consoleSpy.mockRestore();
  });
});

describe("ScreenReaderOnly", () => {
  it("should render content with sr-only class", () => {
    render(<ScreenReaderOnly>Hidden content</ScreenReaderOnly>);

    const element = screen.getByText("Hidden content");
    expect(element).toHaveClass("sr-only");
  });
});

describe("AccessibleLoading", () => {
  it("should show loading state", () => {
    render(
      <AccessibilityProvider>
        <AccessibleLoading isLoading={true} loadingText="Loading data">
          <div>Content</div>
        </AccessibleLoading>
      </AccessibilityProvider>
    );

    expect(screen.getByRole("status")).toBeInTheDocument();
    expect(screen.getByLabelText("Loading data...")).toBeInTheDocument();
    expect(screen.queryByText("Content")).not.toBeInTheDocument();
  });

  it("should show content when not loading", () => {
    render(
      <AccessibilityProvider>
        <AccessibleLoading isLoading={false}>
          <div>Content</div>
        </AccessibleLoading>
      </AccessibilityProvider>
    );

    expect(screen.getByText("Content")).toBeInTheDocument();
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });
});

describe("AccessibleFormErrors", () => {
  it("should not render when no errors", () => {
    render(
      <AccessibilityProvider>
        <AccessibleFormErrors errors={{}} />
      </AccessibilityProvider>
    );

    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("should render errors with proper ARIA attributes", () => {
    const errors = {
      email: "Invalid email address",
      password: "Password too short",
    };

    render(
      <AccessibilityProvider>
        <AccessibleFormErrors errors={errors} fieldId="test-field" />
      </AccessibilityProvider>
    );

    const alert = screen.getByRole("alert");
    expect(alert).toHaveAttribute("aria-live", "assertive");
    expect(alert).toHaveAttribute("id", "test-field-error");

    expect(screen.getByText("Invalid email address")).toBeInTheDocument();
    expect(screen.getByText("Password too short")).toBeInTheDocument();
  });
});
