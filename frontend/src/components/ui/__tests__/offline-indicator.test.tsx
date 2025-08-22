import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { vi } from "vitest";
import {
  OfflineIndicator,
  OfflineIndicatorCompact,
  useOfflineAware,
} from "../offline-indicator";
import { AccessibilityProvider } from "../../providers/AccessibilityProvider";

// Mock the hooks
const mockUseOffline = {
  isOnline: true,
  queuedRequestsCount: 0,
  syncInProgress: false,
  lastSyncAttempt: undefined,
  processOfflineQueue: vi.fn(),
  clearOfflineQueue: vi.fn(),
  queueRequest: vi.fn(),
  getOfflineData: vi.fn(),
};

const mockUseAccessibilityContext = {
  announce: vi.fn(),
  announceError: vi.fn(),
  announceSuccess: vi.fn(),
};

vi.mock("../../../hooks/use-offline", () => ({
  useOffline: () => mockUseOffline,
}));

vi.mock("../../providers/AccessibilityProvider", () => ({
  ...vi.importActual("../../providers/AccessibilityProvider"),
  useAccessibilityContext: () => mockUseAccessibilityContext,
}));

// Mock framer-motion
vi.mock("framer-motion", () => ({
  motion: {
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  },
  AnimatePresence: ({ children }: any) => <>{children}</>,
}));

const renderWithProviders = (component: React.ReactElement) => {
  return render(<AccessibilityProvider>{component}</AccessibilityProvider>);
};

describe("OfflineIndicator", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseOffline.isOnline = true;
    mockUseOffline.queuedRequestsCount = 0;
    mockUseOffline.syncInProgress = false;
    mockUseOffline.lastSyncAttempt = undefined;
  });

  it("should not render when online with no queued requests", () => {
    renderWithProviders(<OfflineIndicator />);

    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("should render when offline", () => {
    mockUseOffline.isOnline = false;

    renderWithProviders(<OfflineIndicator />);

    expect(screen.getByRole("status")).toBeInTheDocument();
    expect(screen.getByText("Offline Mode")).toBeInTheDocument();
    expect(screen.getByText("Limited functionality")).toBeInTheDocument();
  });

  it("should render when online with queued requests", () => {
    mockUseOffline.queuedRequestsCount = 3;

    renderWithProviders(<OfflineIndicator />);

    expect(screen.getByRole("status")).toBeInTheDocument();
    expect(screen.getByText("Online")).toBeInTheDocument();
    expect(screen.getByText("3")).toBeInTheDocument();
    expect(screen.getByText("Queued Requests")).toBeInTheDocument();
  });

  it("should handle sync now button click", async () => {
    mockUseOffline.queuedRequestsCount = 1;
    mockUseOffline.processOfflineQueue.mockResolvedValue(true);

    renderWithProviders(<OfflineIndicator />);

    const syncButton = screen.getByText("Sync Now");
    fireEvent.click(syncButton);

    expect(mockUseOffline.processOfflineQueue).toHaveBeenCalled();
    expect(mockUseAccessibilityContext.announce).toHaveBeenCalledWith(
      "Syncing queued requests",
      "polite"
    );

    await waitFor(() => {
      expect(mockUseAccessibilityContext.announce).toHaveBeenCalledWith(
        "Sync completed successfully",
        "polite"
      );
    });
  });

  it("should handle clear queue button click", async () => {
    mockUseOffline.queuedRequestsCount = 2;
    mockUseOffline.clearOfflineQueue.mockResolvedValue(true);

    // Mock window.confirm
    window.confirm = vi.fn().mockReturnValue(true);

    renderWithProviders(<OfflineIndicator />);

    const clearButton = screen.getByText("Clear");
    fireEvent.click(clearButton);

    expect(window.confirm).toHaveBeenCalledWith(
      "Are you sure you want to clear all queued requests? This action cannot be undone."
    );
    expect(mockUseOffline.clearOfflineQueue).toHaveBeenCalled();

    await waitFor(() => {
      expect(mockUseAccessibilityContext.announce).toHaveBeenCalledWith(
        "Offline queue cleared",
        "polite"
      );
    });
  });
});

describe("OfflineIndicatorCompact", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseOffline.isOnline = true;
    mockUseOffline.queuedRequestsCount = 0;
  });

  it("should not render when online with no queued requests", () => {
    render(<OfflineIndicatorCompact />);

    expect(screen.queryByText("Offline")).not.toBeInTheDocument();
  });

  it("should render offline badge when offline", () => {
    mockUseOffline.isOnline = false;

    render(<OfflineIndicatorCompact />);

    expect(screen.getByText("Offline")).toBeInTheDocument();
  });

  it("should render queue count badge when there are queued requests", () => {
    mockUseOffline.queuedRequestsCount = 5;

    render(<OfflineIndicatorCompact />);

    expect(screen.getByText("5")).toBeInTheDocument();
  });
});
