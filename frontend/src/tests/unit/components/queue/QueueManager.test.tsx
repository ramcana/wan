import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { QueueManager } from "../../../../components/queue/QueueManager";
import { useQueue } from "../../../../hooks/api/use-queue";

vi.mock("../../../../hooks/api/use-queue");

const mockUseQueue = vi.mocked(useQueue);

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

const mockTasks = [
  {
    id: "1",
    modelType: "T2V-A14B",
    prompt: "Test prompt 1",
    status: "processing" as const,
    progress: 50,
    createdAt: new Date(),
  },
  {
    id: "2",
    modelType: "I2V-A14B",
    prompt: "Test prompt 2",
    status: "pending" as const,
    progress: 0,
    createdAt: new Date(),
  },
];

describe("QueueManager", () => {
  beforeEach(() => {
    mockUseQueue.mockReturnValue({
      tasks: mockTasks,
      cancelTask: vi.fn(),
      clearCompleted: vi.fn(),
      isLoading: false,
      error: null,
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  test("renders queue tasks correctly", () => {
    render(<QueueManager />, { wrapper: createWrapper() });

    expect(screen.getByText("Test prompt 1")).toBeInTheDocument();
    expect(screen.getByText("Test prompt 2")).toBeInTheDocument();
    expect(screen.getByText("T2V-A14B")).toBeInTheDocument();
    expect(screen.getByText("I2V-A14B")).toBeInTheDocument();
  });

  test("displays progress for processing tasks", () => {
    render(<QueueManager />, { wrapper: createWrapper() });

    const progressBars = screen.getAllByRole("progressbar");
    expect(progressBars).toHaveLength(2);

    // Check that processing task shows 50% progress
    expect(progressBars[0]).toHaveAttribute("aria-valuenow", "50");
  });

  test("allows task cancellation", async () => {
    const mockCancel = vi.fn();
    mockUseQueue.mockReturnValue({
      tasks: mockTasks,
      cancelTask: mockCancel,
      clearCompleted: vi.fn(),
      isLoading: false,
      error: null,
    });

    render(<QueueManager />, { wrapper: createWrapper() });

    const cancelButtons = screen.getAllByRole("button", { name: /cancel/i });
    fireEvent.click(cancelButtons[0]);

    await waitFor(() => {
      expect(mockCancel).toHaveBeenCalledWith("1");
    });
  });

  test("shows empty state when no tasks", () => {
    mockUseQueue.mockReturnValue({
      tasks: [],
      cancelTask: vi.fn(),
      clearCompleted: vi.fn(),
      isLoading: false,
      error: null,
    });

    render(<QueueManager />, { wrapper: createWrapper() });

    expect(screen.getByText(/no tasks in queue/i)).toBeInTheDocument();
  });

  test("displays loading state", () => {
    mockUseQueue.mockReturnValue({
      tasks: [],
      cancelTask: vi.fn(),
      clearCompleted: vi.fn(),
      isLoading: true,
      error: null,
    });

    render(<QueueManager />, { wrapper: createWrapper() });

    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });

  test("handles queue errors", () => {
    mockUseQueue.mockReturnValue({
      tasks: [],
      cancelTask: vi.fn(),
      clearCompleted: vi.fn(),
      isLoading: false,
      error: new Error("Queue error"),
    });

    render(<QueueManager />, { wrapper: createWrapper() });

    expect(screen.getByText(/queue error/i)).toBeInTheDocument();
  });

  test("updates progress in real-time", async () => {
    const { rerender } = render(<QueueManager />, { wrapper: createWrapper() });

    // Initial progress
    expect(screen.getByRole("progressbar")).toHaveAttribute(
      "aria-valuenow",
      "50"
    );

    // Update progress
    const updatedTasks = [{ ...mockTasks[0], progress: 75 }, mockTasks[1]];

    mockUseQueue.mockReturnValue({
      tasks: updatedTasks,
      cancelTask: vi.fn(),
      clearCompleted: vi.fn(),
      isLoading: false,
      error: null,
    });

    rerender(<QueueManager />);

    await waitFor(() => {
      expect(screen.getByRole("progressbar")).toHaveAttribute(
        "aria-valuenow",
        "75"
      );
    });
  });
});
