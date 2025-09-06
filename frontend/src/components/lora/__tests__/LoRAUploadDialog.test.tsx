import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "react-query";
import { vi } from "vitest";
import { LoRAUploadDialog } from "../LoRAUploadDialog";
import * as loraHooks from "../../../hooks/api/use-lora";

// Mock the hooks
vi.mock("../../../hooks/api/use-lora");

const createTestQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

const renderWithQueryClient = (component: React.ReactElement) => {
  const queryClient = createTestQueryClient();
  return render(
    <QueryClientProvider client={queryClient}>{component}</QueryClientProvider>
  );
};

// Mock file for testing
const createMockFile = (name: string, size: number, type: string) => {
  const file = new File(["mock content"], name, { type });
  Object.defineProperty(file, "size", { value: size });
  return file;
};

describe("LoRAUploadDialog", () => {
  const mockOnOpenChange = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders when open", () => {
    vi.mocked(loraHooks.useLoRAUpload).mockReturnValue({
      mutateAsync: vi.fn(),
      isLoading: false,
      isSuccess: false,
      error: null,
    } as any);

    renderWithQueryClient(
      <LoRAUploadDialog open={true} onOpenChange={mockOnOpenChange} />
    );

    expect(screen.getByText("Upload LoRA File")).toBeInTheDocument();
    expect(
      screen.getByText("Drop your LoRA file here, or click to browse")
    ).toBeInTheDocument();
  });

  it("does not render when closed", () => {
    vi.mocked(loraHooks.useLoRAUpload).mockReturnValue({
      mutateAsync: vi.fn(),
      isLoading: false,
      isSuccess: false,
      error: null,
    } as any);

    renderWithQueryClient(
      <LoRAUploadDialog open={false} onOpenChange={mockOnOpenChange} />
    );

    expect(screen.queryByText("Upload LoRA File")).not.toBeInTheDocument();
  });

  it("handles file selection via input", async () => {
    vi.mocked(loraHooks.useLoRAUpload).mockReturnValue({
      mutateAsync: vi.fn(),
      isLoading: false,
      isSuccess: false,
      error: null,
    } as any);

    renderWithQueryClient(
      <LoRAUploadDialog open={true} onOpenChange={mockOnOpenChange} />
    );

    const fileInput = screen.getByRole("textbox", {
      hidden: true,
    }) as HTMLInputElement;
    const mockFile = createMockFile(
      "test-lora.safetensors",
      1024 * 1024,
      "application/octet-stream"
    );

    fireEvent.change(fileInput, { target: { files: [mockFile] } });

    await waitFor(() => {
      expect(screen.getByText("test-lora.safetensors")).toBeInTheDocument();
      expect(screen.getByText("1.00 MB")).toBeInTheDocument();
    });
  });

  it("validates file extension", async () => {
    vi.mocked(loraHooks.useLoRAUpload).mockReturnValue({
      mutateAsync: vi.fn(),
      isLoading: false,
      isSuccess: false,
      error: null,
    } as any);

    renderWithQueryClient(
      <LoRAUploadDialog open={true} onOpenChange={mockOnOpenChange} />
    );

    const fileInput = screen.getByRole("textbox", {
      hidden: true,
    }) as HTMLInputElement;
    const invalidFile = createMockFile("invalid.txt", 1024, "text/plain");

    fireEvent.change(fileInput, { target: { files: [invalidFile] } });

    await waitFor(() => {
      expect(
        screen.getByText(
          /Only .safetensors, .pt, .pth, and .bin files are supported/
        )
      ).toBeInTheDocument();
    });
  });

  it("validates file size", async () => {
    vi.mocked(loraHooks.useLoRAUpload).mockReturnValue({
      mutateAsync: vi.fn(),
      isLoading: false,
      isSuccess: false,
      error: null,
    } as any);

    renderWithQueryClient(
      <LoRAUploadDialog open={true} onOpenChange={mockOnOpenChange} />
    );

    const fileInput = screen.getByRole("textbox", {
      hidden: true,
    }) as HTMLInputElement;
    const largeFile = createMockFile(
      "large.safetensors",
      600 * 1024 * 1024,
      "application/octet-stream"
    ); // 600MB

    fireEvent.change(fileInput, { target: { files: [largeFile] } });

    await waitFor(() => {
      expect(
        screen.getByText(/File size must be less than 500MB/)
      ).toBeInTheDocument();
    });
  });

  it("handles custom name input", async () => {
    vi.mocked(loraHooks.useLoRAUpload).mockReturnValue({
      mutateAsync: vi.fn(),
      isLoading: false,
      isSuccess: false,
      error: null,
    } as any);

    renderWithQueryClient(
      <LoRAUploadDialog open={true} onOpenChange={mockOnOpenChange} />
    );

    const nameInput = screen.getByPlaceholderText(
      "Enter custom name or leave empty to use filename"
    );
    fireEvent.change(nameInput, { target: { value: "my-custom-lora" } });

    expect(nameInput).toHaveValue("my-custom-lora");
  });

  it("auto-generates name from filename", async () => {
    vi.mocked(loraHooks.useLoRAUpload).mockReturnValue({
      mutateAsync: vi.fn(),
      isLoading: false,
      isSuccess: false,
      error: null,
    } as any);

    renderWithQueryClient(
      <LoRAUploadDialog open={true} onOpenChange={mockOnOpenChange} />
    );

    const fileInput = screen.getByRole("textbox", {
      hidden: true,
    }) as HTMLInputElement;
    const mockFile = createMockFile(
      "anime-style-v2.safetensors",
      1024 * 1024,
      "application/octet-stream"
    );

    fireEvent.change(fileInput, { target: { files: [mockFile] } });

    await waitFor(() => {
      const nameInput = screen.getByPlaceholderText(
        "Enter custom name or leave empty to use filename"
      );
      expect(nameInput).toHaveValue("anime-style-v2");
    });
  });

  it("handles file upload", async () => {
    const mockMutateAsync = vi.fn().mockResolvedValue({
      success: true,
      message: "Upload successful",
      lora_name: "test-lora",
      file_path: "/loras/test-lora.safetensors",
      size_mb: 1.0,
    });

    vi.mocked(loraHooks.useLoRAUpload).mockReturnValue({
      mutateAsync: mockMutateAsync,
      isLoading: false,
      isSuccess: false,
      error: null,
    } as any);

    renderWithQueryClient(
      <LoRAUploadDialog open={true} onOpenChange={mockOnOpenChange} />
    );

    // Select a file
    const fileInput = screen.getByRole("textbox", {
      hidden: true,
    }) as HTMLInputElement;
    const mockFile = createMockFile(
      "test-lora.safetensors",
      1024 * 1024,
      "application/octet-stream"
    );
    fireEvent.change(fileInput, { target: { files: [mockFile] } });

    await waitFor(() => {
      expect(screen.getByText("test-lora.safetensors")).toBeInTheDocument();
    });

    // Click upload button
    const uploadButton = screen.getByText("Upload LoRA");
    fireEvent.click(uploadButton);

    await waitFor(() => {
      expect(mockMutateAsync).toHaveBeenCalledWith({
        file: mockFile,
        name: "test-lora",
        onProgress: expect.any(Function),
      });
    });
  });

  it("shows upload progress", async () => {
    vi.mocked(loraHooks.useLoRAUpload).mockReturnValue({
      mutateAsync: vi.fn(),
      isLoading: true,
      isSuccess: false,
      error: null,
    } as any);

    renderWithQueryClient(
      <LoRAUploadDialog open={true} onOpenChange={mockOnOpenChange} />
    );

    // Should show uploading state
    expect(screen.getByText("Uploading...")).toBeInTheDocument();
  });

  it("shows success message", () => {
    vi.mocked(loraHooks.useLoRAUpload).mockReturnValue({
      mutateAsync: vi.fn(),
      isLoading: false,
      isSuccess: true,
      error: null,
    } as any);

    renderWithQueryClient(
      <LoRAUploadDialog open={true} onOpenChange={mockOnOpenChange} />
    );

    expect(screen.getByText("LoRA uploaded successfully!")).toBeInTheDocument();
  });

  it("handles file removal", async () => {
    vi.mocked(loraHooks.useLoRAUpload).mockReturnValue({
      mutateAsync: vi.fn(),
      isLoading: false,
      isSuccess: false,
      error: null,
    } as any);

    renderWithQueryClient(
      <LoRAUploadDialog open={true} onOpenChange={mockOnOpenChange} />
    );

    // Select a file
    const fileInput = screen.getByRole("textbox", {
      hidden: true,
    }) as HTMLInputElement;
    const mockFile = createMockFile(
      "test-lora.safetensors",
      1024 * 1024,
      "application/octet-stream"
    );
    fireEvent.change(fileInput, { target: { files: [mockFile] } });

    await waitFor(() => {
      expect(screen.getByText("test-lora.safetensors")).toBeInTheDocument();
    });

    // Click remove button
    const removeButton = screen.getByText("Remove");
    fireEvent.click(removeButton);

    await waitFor(() => {
      expect(
        screen.queryByText("test-lora.safetensors")
      ).not.toBeInTheDocument();
      expect(
        screen.getByText("Drop your LoRA file here, or click to browse")
      ).toBeInTheDocument();
    });
  });

  it("disables upload button when no file selected", () => {
    vi.mocked(loraHooks.useLoRAUpload).mockReturnValue({
      mutateAsync: vi.fn(),
      isLoading: false,
      isSuccess: false,
      error: null,
    } as any);

    renderWithQueryClient(
      <LoRAUploadDialog open={true} onOpenChange={mockOnOpenChange} />
    );

    const uploadButton = screen.getByText("Upload LoRA");
    expect(uploadButton).toBeDisabled();
  });

  it("disables upload button when validation error exists", async () => {
    vi.mocked(loraHooks.useLoRAUpload).mockReturnValue({
      mutateAsync: vi.fn(),
      isLoading: false,
      isSuccess: false,
      error: null,
    } as any);

    renderWithQueryClient(
      <LoRAUploadDialog open={true} onOpenChange={mockOnOpenChange} />
    );

    // Select invalid file
    const fileInput = screen.getByRole("textbox", {
      hidden: true,
    }) as HTMLInputElement;
    const invalidFile = createMockFile("invalid.txt", 1024, "text/plain");
    fireEvent.change(fileInput, { target: { files: [invalidFile] } });

    await waitFor(() => {
      const uploadButton = screen.getByText("Upload LoRA");
      expect(uploadButton).toBeDisabled();
    });
  });

  it("handles dialog close", () => {
    vi.mocked(loraHooks.useLoRAUpload).mockReturnValue({
      mutateAsync: vi.fn(),
      isLoading: false,
      isSuccess: false,
      error: null,
    } as any);

    renderWithQueryClient(
      <LoRAUploadDialog open={true} onOpenChange={mockOnOpenChange} />
    );

    const cancelButton = screen.getByText("Cancel");
    fireEvent.click(cancelButton);

    expect(mockOnOpenChange).toHaveBeenCalledWith(false);
  });

  it("prevents closing during upload", () => {
    vi.mocked(loraHooks.useLoRAUpload).mockReturnValue({
      mutateAsync: vi.fn(),
      isLoading: true,
      isSuccess: false,
      error: null,
    } as any);

    renderWithQueryClient(
      <LoRAUploadDialog open={true} onOpenChange={mockOnOpenChange} />
    );

    const cancelButton = screen.getByText("Cancel");
    expect(cancelButton).toBeDisabled();
  });
});
