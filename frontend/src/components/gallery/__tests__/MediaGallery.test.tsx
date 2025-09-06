import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "react-query";
import { BrowserRouter } from "react-router-dom";
import { vi } from "vitest";
import MediaGallery from "../MediaGallery";
import * as useOutputsHook from "@/hooks/api/use-outputs";

// Mock the hooks
vi.mock("@/hooks/api/use-outputs");

// Mock the toast hook
vi.mock("@/hooks/use-toast", () => ({
  useToast: () => ({
    toast: vi.fn(),
  }),
}));

const mockVideoGallery = useOutputsHook as any;

const mockVideos = [
  {
    id: "1",
    filename: "test-video-1.mp4",
    file_path: "/outputs/test-video-1.mp4",
    thumbnail_path: "/thumbnails/1_thumb.jpg",
    prompt: "A beautiful sunset over mountains",
    model_type: "T2V-A14B",
    resolution: "1280x720",
    duration_seconds: 5,
    file_size_mb: 10.5,
    created_at: "2024-01-15T10:00:00Z",
    generation_time_minutes: 3.5,
  },
  {
    id: "2",
    filename: "test-video-2.mp4",
    file_path: "/outputs/test-video-2.mp4",
    thumbnail_path: "/thumbnails/2_thumb.jpg",
    prompt: "A cat playing in the garden",
    model_type: "I2V-A14B",
    resolution: "1920x1080",
    duration_seconds: 8,
    file_size_mb: 25.2,
    created_at: "2024-01-15T11:00:00Z",
    generation_time_minutes: 5.2,
  },
];

const renderWithProviders = (component: React.ReactElement) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });

  return render(
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>{component}</BrowserRouter>
    </QueryClientProvider>
  );
};

describe("MediaGallery", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders loading state", () => {
    mockVideoGallery.useVideoGallery.mockReturnValue({
      videos: [],
      totalCount: 0,
      totalSize: 0,
      isLoading: true,
      error: null,
      searchVideos: vi.fn(),
      filterByModel: vi.fn(),
      sortByDate: vi.fn(),
      sortBySize: vi.fn(),
    });

    renderWithProviders(<MediaGallery />);

    expect(screen.getByText("Generated Videos")).toBeInTheDocument();
    expect(
      screen.getByText("Browse and manage your generated video content")
    ).toBeInTheDocument();
  });

  it("renders empty state when no videos", () => {
    mockVideoGallery.useVideoGallery.mockReturnValue({
      videos: [],
      totalCount: 0,
      totalSize: 0,
      isLoading: false,
      error: null,
      searchVideos: vi.fn(),
      filterByModel: vi.fn(),
      sortByDate: vi.fn(),
      sortBySize: vi.fn(),
    });

    renderWithProviders(<MediaGallery />);

    expect(screen.getByText("No videos yet")).toBeInTheDocument();
    expect(screen.getByText("Generate Your First Video")).toBeInTheDocument();
  });

  it("renders videos when data is available", async () => {
    mockVideoGallery.useVideoGallery.mockReturnValue({
      videos: mockVideos,
      totalCount: 2,
      totalSize: 35.7,
      isLoading: false,
      error: null,
      searchVideos: vi.fn(),
      filterByModel: vi.fn(),
      sortByDate: vi.fn(),
      sortBySize: vi.fn(),
    });

    renderWithProviders(<MediaGallery />);

    await waitFor(() => {
      expect(screen.getByText("test-video-1.mp4")).toBeInTheDocument();
      expect(screen.getByText("test-video-2.mp4")).toBeInTheDocument();
      expect(
        screen.getByText("A beautiful sunset over mountains")
      ).toBeInTheDocument();
      expect(
        screen.getByText("A cat playing in the garden")
      ).toBeInTheDocument();
    });

    // Check stats - use more specific queries
    expect(screen.getByText("Total Videos")).toBeInTheDocument();
    expect(screen.getByText("Total Storage")).toBeInTheDocument();
    expect(screen.getByText("35.7 MB")).toBeInTheDocument(); // Total size
  });

  it("renders error state", () => {
    const mockError = new Error("Failed to load videos");
    mockVideoGallery.useVideoGallery.mockReturnValue({
      videos: [],
      totalCount: 0,
      totalSize: 0,
      isLoading: false,
      error: mockError,
      searchVideos: vi.fn(),
      filterByModel: vi.fn(),
      sortByDate: vi.fn(),
      sortBySize: vi.fn(),
    });

    renderWithProviders(<MediaGallery />);

    expect(screen.getByText("Error")).toBeInTheDocument();
    expect(screen.getByText("Failed to load videos")).toBeInTheDocument();
  });

  it("shows filter controls", () => {
    mockVideoGallery.useVideoGallery.mockReturnValue({
      videos: mockVideos,
      totalCount: 2,
      totalSize: 35.7,
      isLoading: false,
      error: null,
      searchVideos: vi.fn(),
      filterByModel: vi.fn(),
      sortByDate: vi.fn(),
      sortBySize: vi.fn(),
    });

    renderWithProviders(<MediaGallery />);

    // Should show search input
    expect(
      screen.getByPlaceholderText(
        "Search videos by prompt, filename, or model..."
      )
    ).toBeInTheDocument();

    // Should show filter controls
    expect(screen.getByText("All Models")).toBeInTheDocument();
    expect(screen.getByText("Date Created")).toBeInTheDocument();
  });

  it("shows gallery stats", () => {
    mockVideoGallery.useVideoGallery.mockReturnValue({
      videos: mockVideos,
      totalCount: 2,
      totalSize: 35.7,
      isLoading: false,
      error: null,
      searchVideos: vi.fn(),
      filterByModel: vi.fn(),
      sortByDate: vi.fn(),
      sortBySize: vi.fn(),
    });

    renderWithProviders(<MediaGallery />);

    expect(screen.getByText("Total Videos")).toBeInTheDocument();
    expect(screen.getByText("Total Storage")).toBeInTheDocument();
    expect(screen.getByText("35.7 MB")).toBeInTheDocument();
  });
});
