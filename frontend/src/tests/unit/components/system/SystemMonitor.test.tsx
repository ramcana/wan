import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import { vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { SystemMonitor } from "../../../../components/system/SystemMonitor";
import { useWebSocket } from "../../../../hooks/use-websocket";

vi.mock("../../../../hooks/use-websocket");

const mockUseWebSocket = vi.mocked(useWebSocket);

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

const mockSystemStats = {
  cpu: 45.5,
  ram: { used: 8.2, total: 16.0 },
  gpu: 78.3,
  vram: { used: 6.4, total: 8.0 },
  timestamp: new Date(),
};

describe("SystemMonitor", () => {
  beforeEach(() => {
    mockUseWebSocket.mockReturnValue({
      data: mockSystemStats,
      isConnected: true,
      error: null,
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  test("renders system metrics correctly", () => {
    render(<SystemMonitor />, { wrapper: createWrapper() });

    expect(screen.getByText("CPU")).toBeInTheDocument();
    expect(screen.getByText("RAM")).toBeInTheDocument();
    expect(screen.getByText("GPU")).toBeInTheDocument();
    expect(screen.getByText("VRAM")).toBeInTheDocument();

    expect(screen.getByText("45.5%")).toBeInTheDocument();
    expect(screen.getByText("8.2 / 16.0 GB")).toBeInTheDocument();
    expect(screen.getByText("78.3%")).toBeInTheDocument();
    expect(screen.getByText("6.4 / 8.0 GB")).toBeInTheDocument();
  });

  test("displays progress bars with correct values", () => {
    render(<SystemMonitor />, { wrapper: createWrapper() });

    const progressBars = screen.getAllByRole("progressbar");
    expect(progressBars).toHaveLength(4);

    expect(progressBars[0]).toHaveAttribute("aria-valuenow", "45.5");
    expect(progressBars[1]).toHaveAttribute("aria-valuenow", "51.25"); // 8.2/16*100
    expect(progressBars[2]).toHaveAttribute("aria-valuenow", "78.3");
    expect(progressBars[3]).toHaveAttribute("aria-valuenow", "80"); // 6.4/8*100
  });

  test("shows warning when VRAM usage is high", () => {
    const highVramStats = {
      ...mockSystemStats,
      vram: { used: 7.5, total: 8.0 }, // 93.75% usage
    };

    mockUseWebSocket.mockReturnValue({
      data: highVramStats,
      isConnected: true,
      error: null,
    });

    render(<SystemMonitor />, { wrapper: createWrapper() });

    expect(screen.getByText(/high vram usage/i)).toBeInTheDocument();
    expect(screen.getByText(/consider reducing/i)).toBeInTheDocument();
  });

  test("displays connection status", () => {
    render(<SystemMonitor />, { wrapper: createWrapper() });

    expect(screen.getByText(/connected/i)).toBeInTheDocument();
  });

  test("shows disconnected state", () => {
    mockUseWebSocket.mockReturnValue({
      data: null,
      isConnected: false,
      error: null,
    });

    render(<SystemMonitor />, { wrapper: createWrapper() });

    expect(screen.getByText(/disconnected/i)).toBeInTheDocument();
    expect(screen.getByText(/reconnecting/i)).toBeInTheDocument();
  });

  test("handles WebSocket errors", () => {
    mockUseWebSocket.mockReturnValue({
      data: null,
      isConnected: false,
      error: new Error("Connection failed"),
    });

    render(<SystemMonitor />, { wrapper: createWrapper() });

    expect(screen.getByText(/connection failed/i)).toBeInTheDocument();
  });

  test("updates metrics in real-time", async () => {
    const { rerender } = render(<SystemMonitor />, {
      wrapper: createWrapper(),
    });

    expect(screen.getByText("45.5%")).toBeInTheDocument();

    // Update metrics
    const updatedStats = { ...mockSystemStats, cpu: 60.2 };
    mockUseWebSocket.mockReturnValue({
      data: updatedStats,
      isConnected: true,
      error: null,
    });

    rerender(<SystemMonitor />);

    await waitFor(() => {
      expect(screen.getByText("60.2%")).toBeInTheDocument();
    });
  });

  test("shows optimization suggestions", () => {
    const highUsageStats = {
      cpu: 95.0,
      ram: { used: 15.5, total: 16.0 },
      gpu: 98.0,
      vram: { used: 7.8, total: 8.0 },
      timestamp: new Date(),
    };

    mockUseWebSocket.mockReturnValue({
      data: highUsageStats,
      isConnected: true,
      error: null,
    });

    render(<SystemMonitor />, { wrapper: createWrapper() });

    expect(screen.getByText(/optimization suggestions/i)).toBeInTheDocument();
    expect(screen.getByText(/reduce batch size/i)).toBeInTheDocument();
    expect(screen.getByText(/enable model offloading/i)).toBeInTheDocument();
  });
});
