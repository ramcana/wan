import React from "react";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "react-query";
import { vi, describe, it, expect, beforeEach } from "vitest";
import SystemHealthIndicator from "../SystemHealthIndicator";
import * as useSystemHook from "@/hooks/api/use-system";

// Mock the system hook
vi.mock("@/hooks/api/use-system");

const mockUseSystemHealth = vi.mocked(useSystemHook.useSystemHealth);

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe("SystemHealthIndicator", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("shows loading state", () => {
    mockUseSystemHealth.mockReturnValue({
      data: null,
      isLoading: true,
      error: null,
      refetch: vi.fn(),
      isError: false,
      isSuccess: false,
      status: "loading",
    } as any);

    render(<SystemHealthIndicator />, { wrapper: createWrapper() });

    expect(screen.getByText("Checking...")).toBeInTheDocument();
  });

  it("shows offline state on error", () => {
    mockUseSystemHealth.mockReturnValue({
      data: null,
      isLoading: false,
      error: new Error("Network error"),
      refetch: vi.fn(),
      isError: true,
      isSuccess: false,
      status: "error",
    } as any);

    render(<SystemHealthIndicator />, { wrapper: createWrapper() });

    expect(screen.getByText("Offline")).toBeInTheDocument();
  });

  it("shows offline state with details when showDetails is true", () => {
    mockUseSystemHealth.mockReturnValue({
      data: null,
      isLoading: false,
      error: new Error("Network error"),
      refetch: vi.fn(),
      isError: true,
      isSuccess: false,
      status: "error",
    } as any);

    render(<SystemHealthIndicator showDetails />, { wrapper: createWrapper() });

    expect(screen.getByText("Offline")).toBeInTheDocument();
    expect(screen.getByText("Unable to connect to system")).toBeInTheDocument();
  });

  it("shows healthy status", () => {
    const mockHealth = {
      status: "healthy",
      message: "System is running normally",
      timestamp: "2024-01-01T12:00:00Z",
      system_info: {
        warnings: [],
        issues: [],
      },
    };

    mockUseSystemHealth.mockReturnValue({
      data: mockHealth,
      isLoading: false,
      error: null,
      refetch: vi.fn(),
      isError: false,
      isSuccess: true,
      status: "success",
    } as any);

    render(<SystemHealthIndicator />, { wrapper: createWrapper() });

    expect(screen.getByText("Healthy")).toBeInTheDocument();
  });

  it("shows warning status", () => {
    const mockHealth = {
      status: "warning",
      message: "High resource usage detected",
      timestamp: "2024-01-01T12:00:00Z",
      system_info: {
        warnings: ["High VRAM usage"],
        issues: [],
      },
    };

    mockUseSystemHealth.mockReturnValue({
      data: mockHealth,
      isLoading: false,
      error: null,
      refetch: vi.fn(),
      isError: false,
      isSuccess: true,
      status: "success",
    } as any);

    render(<SystemHealthIndicator />, { wrapper: createWrapper() });

    expect(screen.getByText("Warning")).toBeInTheDocument();
  });

  it("shows critical status", () => {
    const mockHealth = {
      status: "critical",
      message: "Critical system issues detected",
      timestamp: "2024-01-01T12:00:00Z",
      system_info: {
        warnings: [],
        issues: ["VRAM exhausted"],
      },
    };

    mockUseSystemHealth.mockReturnValue({
      data: mockHealth,
      isLoading: false,
      error: null,
      refetch: vi.fn(),
      isError: false,
      isSuccess: true,
      status: "success",
    } as any);

    render(<SystemHealthIndicator />, { wrapper: createWrapper() });

    expect(screen.getByText("Critical")).toBeInTheDocument();
  });

  it("shows details when showDetails is true", () => {
    const mockHealth = {
      status: "warning",
      message: "High resource usage detected",
      timestamp: "2024-01-01T12:00:00Z",
      system_info: {
        warnings: ["High VRAM usage", "High RAM usage"],
        issues: ["Critical CPU usage"],
      },
    };

    mockUseSystemHealth.mockReturnValue({
      data: mockHealth,
      isLoading: false,
      error: null,
      refetch: vi.fn(),
      isError: false,
      isSuccess: true,
      status: "success",
    } as any);

    render(<SystemHealthIndicator showDetails />, { wrapper: createWrapper() });

    expect(screen.getByText("Warning")).toBeInTheDocument();
    expect(
      screen.getByText("High resource usage detected")
    ).toBeInTheDocument();
    expect(screen.getByText("2 warnings")).toBeInTheDocument();
    expect(screen.getByText("1 issue")).toBeInTheDocument();
  });

  it("shows error status", () => {
    const mockHealth = {
      status: "error",
      message: "System error occurred",
      timestamp: "2024-01-01T12:00:00Z",
      system_info: {},
    };

    mockUseSystemHealth.mockReturnValue({
      data: mockHealth,
      isLoading: false,
      error: null,
      refetch: vi.fn(),
      isError: false,
      isSuccess: true,
      status: "success",
    } as any);

    render(<SystemHealthIndicator />, { wrapper: createWrapper() });

    expect(screen.getByText("Error")).toBeInTheDocument();
  });

  it("applies custom className", () => {
    const mockHealth = {
      status: "healthy",
      message: "System is running normally",
      timestamp: "2024-01-01T12:00:00Z",
    };

    mockUseSystemHealth.mockReturnValue({
      data: mockHealth,
      isLoading: false,
      error: null,
      refetch: vi.fn(),
      isError: false,
      isSuccess: true,
      status: "success",
    } as any);

    const { container } = render(
      <SystemHealthIndicator className="custom-class" />,
      { wrapper: createWrapper() }
    );

    expect(container.firstChild).toHaveClass("custom-class");
  });
});
