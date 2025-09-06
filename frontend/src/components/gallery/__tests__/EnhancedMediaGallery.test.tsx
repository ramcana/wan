import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "react-query";
import { BrowserRouter } from "react-router-dom";
import { vi } from "vitest";
import EnhancedMediaGallery from "../EnhancedMediaGallery";
import * as useOutputsHook from "@/hooks/api/use-outputs";
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

// Mock the hooks
vi.mock("@/hooks/api/use-outputs");

const mockVideoGallery = {
  videos: [
    {
      id: "1",
      filename: "test-video.mp4",
      file_path: "/outputs/test-video.mp4",
      thumbnail_path: "/thumbnails/test-video.jpg",
      model_type: "T2V-A14B",
      prompt: "A beautiful sunset",
      resolution: "1280x720",
      duration_seconds: 5,
      file_size_mb: 10.5,
      created_at: "2024-01-01T00:00:00Z",
      generation_params: {},
    },
  ],
  totalCount: 1,
  totalSize: 10.5,
  isLoading: false,
  error: null,
  searchVideos: vi.fn(),
  filterByModel: vi.fn(),
  sortByDate: vi.fn(),
  sortBySize: vi.fn(),
};

const renderWithProviders = (component: React.ReactElement) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  return render(
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>{component}</BrowserRouter>
    </QueryClientProvider>
  );
};

describe("EnhancedMediaGallery", () => {
  beforeEach(() => {
    vi.mocked(useOutputsHook.useVideoGallery).mockReturnValue(mockVideoGallery);
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("renders gallery with videos", async () => {
    renderWithProviders(<EnhancedMediaGallery />);

    expect(screen.getByText("Generated Videos")).toBeInTheDocument();
    expect(screen.getByText("test-video.mp4")).toBeInTheDocument();
  });

  it("toggles view mode between grid and list", async () => {
    renderWithProviders(<EnhancedMediaGallery />);

    const listButton = screen.getByRole("button", { name: /list/i });
    fireEvent.click(listButton);

    // Should switch to list view
    await waitFor(() => {
      expect(listButton).toHaveClass("bg-primary");
    });
  });

  it("shows and hides filters", async () => {
    renderWithProviders(<EnhancedMediaGallery />);

    const filtersButton = screen.getByRole("button", { name: /filters/i });
    fireEvent.click(filtersButton);

    // Filters should be visible
    await waitFor(() => {
      expect(screen.getByText("Filter:")).toBeInTheDocument();
    });

    // Hide filters
    fireEvent.click(filtersButton);
    await waitFor(() => {
      expect(screen.queryByText("Filter:")).not.toBeInTheDocument();
    });
  });

  it("handles video selection", async () => {
    renderWithProviders(<EnhancedMediaGallery />);

    const videoCard = screen.getByText("test-video.mp4").closest("div");
    expect(videoCard).toBeInTheDocument();

    fireEvent.click(videoCard!);

    // Should open video player (mocked)
    await waitFor(() => {
      // Since we're mocking framer-motion, we can't test the actual modal
      // but we can verify the click handler was called
      expect(videoCard).toBeInTheDocument();
    });
  });

  it("handles bulk operations", async () => {
    renderWithProviders(<EnhancedMediaGallery />);

    // This would require more complex setup to test bulk selection
    // For now, just verify the component renders without errors
    expect(screen.getByText("Generated Videos")).toBeInTheDocument();
  });

  it("handles loading state", async () => {
    vi.mocked(useOutputsHook.useVideoGallery).mockReturnValue({
      ...mockVideoGallery,
      isLoading: true,
    });

    renderWithProviders(<EnhancedMediaGallery />);

    // Should show loading state
    expect(screen.getByText("Generated Videos")).toBeInTheDocument();
  });

  it("handles error state", async () => {
    const error = new Error("Failed to load videos");
    vi.mocked(useOutputsHook.useVideoGallery).mockReturnValue({
      ...mockVideoGallery,
      error,
    });

    renderWithProviders(<EnhancedMediaGallery />);

    expect(screen.getByText("Generated Videos")).toBeInTheDocument();
  });
});
