import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MediaGallery } from "../../../../components/gallery/MediaGallery";
import { useOutputs } from "../../../../hooks/api/use-outputs";

vi.mock("../../../../hooks/api/use-outputs");

const mockUseOutputs = vi.mocked(useOutputs);

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

const mockVideos = [
  {
    id: "1",
    filename: "video1.mp4",
    thumbnail: "thumb1.jpg",
    metadata: {
      prompt: "Beautiful sunset",
      modelType: "T2V-A14B",
      resolution: "1280x720",
      duration: 5,
      createdAt: new Date("2024-01-01"),
    },
  },
  {
    id: "2",
    filename: "video2.mp4",
    thumbnail: "thumb2.jpg",
    metadata: {
      prompt: "Ocean waves",
      modelType: "I2V-A14B",
      resolution: "1920x1080",
      duration: 10,
      createdAt: new Date("2024-01-02"),
    },
  },
];

describe("MediaGallery", () => {
  beforeEach(() => {
    mockUseOutputs.mockReturnValue({
      videos: mockVideos,
      deleteVideo: vi.fn(),
      isLoading: false,
      error: null,
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  test("renders video grid correctly", () => {
    render(<MediaGallery />, { wrapper: createWrapper() });

    expect(screen.getByText("Beautiful sunset")).toBeInTheDocument();
    expect(screen.getByText("Ocean waves")).toBeInTheDocument();
    expect(screen.getByText("T2V-A14B")).toBeInTheDocument();
    expect(screen.getByText("I2V-A14B")).toBeInTheDocument();
  });

  test("displays video metadata", () => {
    render(<MediaGallery />, { wrapper: createWrapper() });

    expect(screen.getByText("1280x720")).toBeInTheDocument();
    expect(screen.getByText("1920x1080")).toBeInTheDocument();
    expect(screen.getByText("5s")).toBeInTheDocument();
    expect(screen.getByText("10s")).toBeInTheDocument();
  });

  test("opens video player on click", async () => {
    render(<MediaGallery />, { wrapper: createWrapper() });

    const videoCard = screen.getByText("Beautiful sunset").closest("div");
    fireEvent.click(videoCard!);

    await waitFor(() => {
      expect(screen.getByRole("dialog")).toBeInTheDocument();
      expect(screen.getByText("Beautiful sunset")).toBeInTheDocument();
    });
  });

  test("allows video deletion", async () => {
    const mockDelete = vi.fn();
    mockUseOutputs.mockReturnValue({
      videos: mockVideos,
      deleteVideo: mockDelete,
      isLoading: false,
      error: null,
    });

    render(<MediaGallery />, { wrapper: createWrapper() });

    const deleteButtons = screen.getAllByRole("button", { name: /delete/i });
    fireEvent.click(deleteButtons[0]);

    await waitFor(() => {
      expect(mockDelete).toHaveBeenCalledWith("1");
    });
  });

  test("shows empty state when no videos", () => {
    mockUseOutputs.mockReturnValue({
      videos: [],
      deleteVideo: vi.fn(),
      isLoading: false,
      error: null,
    });

    render(<MediaGallery />, { wrapper: createWrapper() });

    expect(screen.getByText(/no videos generated yet/i)).toBeInTheDocument();
  });

  test("displays loading state", () => {
    mockUseOutputs.mockReturnValue({
      videos: [],
      deleteVideo: vi.fn(),
      isLoading: true,
      error: null,
    });

    render(<MediaGallery />, { wrapper: createWrapper() });

    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });

  test("handles gallery errors", () => {
    mockUseOutputs.mockReturnValue({
      videos: [],
      deleteVideo: vi.fn(),
      isLoading: false,
      error: new Error("Gallery error"),
    });

    render(<MediaGallery />, { wrapper: createWrapper() });

    expect(screen.getByText(/gallery error/i)).toBeInTheDocument();
  });

  test("filters videos by search term", async () => {
    render(<MediaGallery />, { wrapper: createWrapper() });

    const searchInput = screen.getByPlaceholderText(/search videos/i);
    fireEvent.change(searchInput, { target: { value: "sunset" } });

    await waitFor(() => {
      expect(screen.getByText("Beautiful sunset")).toBeInTheDocument();
      expect(screen.queryByText("Ocean waves")).not.toBeInTheDocument();
    });
  });

  test("sorts videos by date", async () => {
    render(<MediaGallery />, { wrapper: createWrapper() });

    const sortSelect = screen.getByLabelText(/sort by/i);
    fireEvent.change(sortSelect, { target: { value: "date-desc" } });

    await waitFor(() => {
      const videoCards = screen.getAllByTestId("video-card");
      expect(videoCards[0]).toHaveTextContent("Ocean waves"); // Newer video first
      expect(videoCards[1]).toHaveTextContent("Beautiful sunset");
    });
  });

  test("lazy loads video thumbnails", () => {
    // Mock IntersectionObserver
    const mockIntersectionObserver = vi.fn();
    mockIntersectionObserver.mockReturnValue({
      observe: vi.fn(),
      unobserve: vi.fn(),
      disconnect: vi.fn(),
    });
    window.IntersectionObserver = mockIntersectionObserver;

    render(<MediaGallery />, { wrapper: createWrapper() });

    expect(mockIntersectionObserver).toHaveBeenCalled();
  });
});
