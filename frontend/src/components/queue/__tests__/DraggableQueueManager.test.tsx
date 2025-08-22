import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "react-query";
import { vi } from "vitest";
import DraggableQueueManager from "../DraggableQueueManager";
import * as useQueueHook from "@/hooks/api/use-queue";
import { it } from "zod/v4/locales";
import { it } from "zod/v4/locales";
import { it } from "zod/v4/locales";
import { it } from "zod/v4/locales";
import { it } from "zod/v4/locales";
import { it } from "zod/v4/locales";
import { it } from "zod/v4/locales";
import { it } from "zod/v4/locales";

// Mock framer-motion
vi.mock("framer-motion", () => ({
  motion: {
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  },
  AnimatePresence: ({ children }: any) => <>{children}</>,
}));

// Mock @dnd-kit
vi.mock("@dnd-kit/core", () => ({
  DndContext: ({ children }: any) => <div>{children}</div>,
  closestCenter: vi.fn(),
  KeyboardSensor: vi.fn(),
  PointerSensor: vi.fn(),
  useSensor: vi.fn(),
  useSensors: vi.fn(() => []),
}));

vi.mock("@dnd-kit/sortable", () => ({
  SortableContext: ({ children }: any) => <div>{children}</div>,
  sortableKeyboardCoordinates: vi.fn(),
  verticalListSortingStrategy: vi.fn(),
  arrayMove: vi.fn((array, oldIndex, newIndex) => {
    const result = [...array];
    const [removed] = result.splice(oldIndex, 1);
    result.splice(newIndex, 0, removed);
    return result;
  }),
}));

vi.mock("@dnd-kit/utilities", () => ({
  restrictToVerticalAxis: vi.fn(),
  restrictToWindowEdges: vi.fn(),
}));

// Mock the hooks
vi.mock("@/hooks/api/use-queue");

const mockQueueStatus = {
  total_tasks: 3,
  pending_tasks: 1,
  processing_tasks: 1,
  completed_tasks: 1,
  failed_tasks: 0,
  cancelled_tasks: 0,
  tasks: [
    {
      id: "1",
      model_type: "T2V-A14B",
      prompt: "Test prompt 1",
      status: "pending" as const,
      progress: 0,
      created_at: "2024-01-01T00:00:00Z",
      resolution: "1280x720",
      steps: 50,
    },
    {
      id: "2",
      model_type: "I2V-A14B",
      prompt: "Test prompt 2",
      status: "processing" as const,
      progress: 50,
      created_at: "2024-01-01T01:00:00Z",
      started_at: "2024-01-01T01:00:00Z",
      resolution: "1280x720",
      steps: 50,
    },
    {
      id: "3",
      model_type: "TI2V-5B",
      prompt: "Test prompt 3",
      status: "completed" as const,
      progress: 100,
      created_at: "2024-01-01T02:00:00Z",
      completed_at: "2024-01-01T02:05:00Z",
      output_path: "/outputs/test-3.mp4",
      resolution: "1280x720",
      steps: 50,
    },
  ],
};

const renderWithProviders = (component: React.ReactElement) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  return render(
    <QueryClientProvider client={queryClient}>{component}</QueryClientProvider>
  );
};

describe("DraggableQueueManager", () => {
  beforeEach(() => {
    vi.mocked(useQueueHook.useQueueStatus).mockReturnValue({
      data: mockQueueStatus,
      isLoading: false,
      error: null,
      refetch: vi.fn(),
    } as any);

    vi.mocked(useQueueHook.useClearCompletedTasks).mockReturnValue({
      mutateAsync: vi.fn(),
      isLoading: false,
    } as any);

    vi.mocked(useQueueHook.useReorderTasks).mockReturnValue({
      mutateAsync: vi.fn(),
      isLoading: false,
    } as any);

    // Mock Notification API
    Object.defineProperty(window, "Notification", {
      value: {
        permission: "default",
        requestPermission: vi.fn(() => Promise.resolve("granted")),
      },
      writable: true,
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("renders queue with tasks", async () => {
    renderWithProviders(<DraggableQueueManager />);

    expect(screen.getByText("Task Queue")).toBeInTheDocument();
    expect(screen.getByText("Test prompt 1")).toBeInTheDocument();
    expect(screen.getByText("Test prompt 2")).toBeInTheDocument();
    expect(screen.getByText("Test prompt 3")).toBeInTheDocument();
  });

  it("toggles drag mode", async () => {
    renderWithProviders(<DraggableQueueManager />);

    const dragModeButton = screen.getByRole("button", { name: /reorder/i });
    fireEvent.click(dragModeButton);

    await waitFor(() => {
      expect(screen.getByText("Drag Mode Active")).toBeInTheDocument();
    });
  });

  it("filters tasks by status", async () => {
    renderWithProviders(<DraggableQueueManager />);

    const pendingFilter = screen.getByText("pending");
    fireEvent.click(pendingFilter);

    // Should show only pending tasks
    expect(screen.getByText("Test prompt 1")).toBeInTheDocument();
  });

  it("sorts tasks", async () => {
    renderWithProviders(<DraggableQueueManager />);

    const sortSelect = screen.getByDisplayValue("Created");
    fireEvent.change(sortSelect, { target: { value: "status" } });

    // Should sort by status
    expect(sortSelect).toHaveValue("status");
  });

  it("clears completed tasks", async () => {
    const mockClearCompleted = vi.fn();
    vi.mocked(useQueueHook.useClearCompletedTasks).mockReturnValue({
      mutateAsync: mockClearCompleted,
      isLoading: false,
    } as any);

    renderWithProviders(<DraggableQueueManager />);

    const clearButton = screen.getByRole("button", {
      name: /clear completed/i,
    });
    fireEvent.click(clearButton);

    expect(mockClearCompleted).toHaveBeenCalled();
  });

  it("toggles notifications", async () => {
    renderWithProviders(<DraggableQueueManager />);

    const notificationButton = screen.getByRole("button", {
      name: /notifications/i,
    });
    fireEvent.click(notificationButton);

    // Should request notification permission
    expect(window.Notification.requestPermission).toHaveBeenCalled();
  });

  it("handles loading state", async () => {
    vi.mocked(useQueueHook.useQueueStatus).mockReturnValue({
      data: undefined,
      isLoading: true,
      error: null,
      refetch: vi.fn(),
    } as any);

    renderWithProviders(<DraggableQueueManager />);

    // Should show loading skeleton
    expect(screen.getByText("Task Queue")).toBeInTheDocument();
  });

  it("handles error state", async () => {
    const error = new Error("Failed to load queue");
    vi.mocked(useQueueHook.useQueueStatus).mockReturnValue({
      data: undefined,
      isLoading: false,
      error,
      refetch: vi.fn(),
    } as any);

    renderWithProviders(<DraggableQueueManager />);

    expect(
      screen.getByText("Failed to load queue status. Please try again.")
    ).toBeInTheDocument();
  });
});
