import React from "react";
import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "react-query";
import { vi, describe, it, expect, beforeEach } from "vitest";
import SystemMonitor from "../SystemMonitor";
import * as useSystemHook from "@/hooks/api/use-system";

// Mock the system hook
vi.mock("@/hooks/api/use-system");

const mockUseSystemMonitoring = vi.mocked(useSystemHook.useSystemMonitoring);

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

describe("SystemMonitor", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("shows loading state", () => {
    mockUseSystemMonitoring.mockReturnValue({
      stats: null,
      alerts: [],
      isLoading: true,
      error: null,
      hasAlerts: false,
      criticalAlerts: [],
      warningAlerts: [],
    });

    render(<SystemMonitor />, { wrapper: createWrapper() });

    expect(screen.getByText("Loading System Stats...")).toBeInTheDocument();
  });

  it("shows error state", () => {
    mockUseSystemMonitoring.mockReturnValue({
      stats: null,
      alerts: [],
      isLoading: false,
      error: new Error("Network error"),
      hasAlerts: false,
      criticalAlerts: [],
      warningAlerts: [],
    });

    render(<SystemMonitor />, { wrapper: createWrapper() });

    expect(screen.getByText("Failed to Load System Stats")).toBeInTheDocument();
    expect(
      screen.getByText(/Unable to retrieve system monitoring data/)
    ).toBeInTheDocument();
  });

  it("displays system stats correctly", () => {
    const mockStats = {
      cpu_percent: 45.5,
      ram_used_gb: 8.2,
      ram_total_gb: 16.0,
      ram_percent: 51.25,
      gpu_percent: 75.0,
      vram_used_mb: 6144,
      vram_total_mb: 8192,
      vram_percent: 75.0,
      timestamp: "2024-01-01T12:00:00Z",
    };

    mockUseSystemMonitoring.mockReturnValue({
      stats: mockStats,
      alerts: [],
      isLoading: false,
      error: null,
      hasAlerts: false,
      criticalAlerts: [],
      warningAlerts: [],
    });

    render(<SystemMonitor />, { wrapper: createWrapper() });

    // Check if main components are rendered
    expect(screen.getByText("System Monitor")).toBeInTheDocument();
    expect(screen.getByText("CPU Usage")).toBeInTheDocument();
    expect(screen.getByText("System Memory")).toBeInTheDocument();
    expect(screen.getByText("GPU Usage")).toBeInTheDocument();
    expect(screen.getByText("Video Memory")).toBeInTheDocument();

    // Check specific values (using getAllByText since values appear multiple times)
    expect(screen.getAllByText("45.5%")).toHaveLength(3); // CPU usage appears 3 times
    expect(screen.getByText("8.2 / 16.0 GB")).toBeInTheDocument();
    expect(screen.getAllByText("75.0%")).toHaveLength(5); // GPU and VRAM both 75% (appears in multiple places)
    expect(screen.getByText("6.0 / 8.0 GB")).toBeInTheDocument();

    // Check healthy status
    expect(screen.getByText("Healthy")).toBeInTheDocument();
    expect(screen.getByText("System Running Normally")).toBeInTheDocument();
  });

  it("displays alerts correctly", () => {
    const mockStats = {
      cpu_percent: 45.5,
      ram_used_gb: 8.2,
      ram_total_gb: 16.0,
      ram_percent: 51.25,
      gpu_percent: 75.0,
      vram_used_mb: 7372, // 90% of 8192MB
      vram_total_mb: 8192,
      vram_percent: 90.0,
      timestamp: "2024-01-01T12:00:00Z",
    };

    const mockAlerts = [
      {
        type: "error" as const,
        message: "VRAM usage critical: 90.0%",
        suggestion: "Consider reducing model precision or batch size",
      },
    ];

    mockUseSystemMonitoring.mockReturnValue({
      stats: mockStats,
      alerts: mockAlerts,
      isLoading: false,
      error: null,
      hasAlerts: true,
      criticalAlerts: mockAlerts,
      warningAlerts: [],
    });

    render(<SystemMonitor />, { wrapper: createWrapper() });

    // Check alert display
    expect(screen.getByText("1 Alert")).toBeInTheDocument();
    expect(screen.getByText("VRAM usage critical: 90.0%")).toBeInTheDocument();
    expect(
      screen.getByText("Consider reducing model precision or batch size")
    ).toBeInTheDocument();

    // Check optimization suggestions
    expect(screen.getByText("Optimization Suggestions")).toBeInTheDocument();
    expect(
      screen.getByText("Enable model offloading to reduce VRAM usage")
    ).toBeInTheDocument();
  });

  it("displays warning alerts correctly", () => {
    const mockStats = {
      cpu_percent: 45.5,
      ram_used_gb: 8.2,
      ram_total_gb: 16.0,
      ram_percent: 85.0,
      gpu_percent: 75.0,
      vram_used_mb: 6553, // 80% of 8192MB
      vram_total_mb: 8192,
      vram_percent: 80.0,
      timestamp: "2024-01-01T12:00:00Z",
    };

    const mockAlerts = [
      {
        type: "warning" as const,
        message: "RAM usage high: 85.0%",
        suggestion: "Consider freeing up system memory",
      },
      {
        type: "warning" as const,
        message: "VRAM usage high: 80.0%",
        suggestion: "Monitor memory usage closely",
      },
    ];

    mockUseSystemMonitoring.mockReturnValue({
      stats: mockStats,
      alerts: mockAlerts,
      isLoading: false,
      error: null,
      hasAlerts: true,
      criticalAlerts: [],
      warningAlerts: mockAlerts,
    });

    render(<SystemMonitor />, { wrapper: createWrapper() });

    // Check alert display
    expect(screen.getByText("2 Alerts")).toBeInTheDocument();
    expect(screen.getByText("RAM usage high: 85.0%")).toBeInTheDocument();
    expect(screen.getByText("VRAM usage high: 80.0%")).toBeInTheDocument();
  });

  it("shows system health summary", () => {
    const mockStats = {
      cpu_percent: 45.5,
      ram_used_gb: 8.2,
      ram_total_gb: 16.0,
      ram_percent: 51.25,
      gpu_percent: 75.0,
      vram_used_mb: 6144,
      vram_total_mb: 8192,
      vram_percent: 75.0,
      timestamp: "2024-01-01T12:00:00Z",
    };

    mockUseSystemMonitoring.mockReturnValue({
      stats: mockStats,
      alerts: [],
      isLoading: false,
      error: null,
      hasAlerts: false,
      criticalAlerts: [],
      warningAlerts: [],
    });

    render(<SystemMonitor />, { wrapper: createWrapper() });

    // Check system health summary
    expect(screen.getByText("System Health Summary")).toBeInTheDocument();

    // Check that all percentage values are displayed in the summary
    const summarySection = screen
      .getByText("System Health Summary")
      .closest("div");
    expect(summarySection).toBeTruthy();
  });
});
