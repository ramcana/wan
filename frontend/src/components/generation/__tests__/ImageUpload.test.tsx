import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ImageUpload } from "../ImageUpload";

describe("ImageUpload", () => {
  const mockOnImageSelect = vi.fn();

  beforeEach(() => {
    mockOnImageSelect.mockClear();
  });

  it("should render upload area when no image is selected", () => {
    render(
      <ImageUpload onImageSelect={mockOnImageSelect} selectedImage={null} />
    );

    expect(screen.getByText("Upload an image")).toBeInTheDocument();
    expect(
      screen.getByText("Drag and drop an image file, or click to browse")
    ).toBeInTheDocument();
    expect(screen.getByText("JPEG")).toBeInTheDocument();
    expect(screen.getByText("PNG")).toBeInTheDocument();
    expect(screen.getByText("WebP")).toBeInTheDocument();
  });

  it("should show image preview when image is selected", () => {
    const mockFile = new File(["test"], "test.jpg", { type: "image/jpeg" });

    render(
      <ImageUpload onImageSelect={mockOnImageSelect} selectedImage={mockFile} />
    );

    expect(screen.getByText("test.jpg")).toBeInTheDocument();
    expect(screen.getByText("JPEG")).toBeInTheDocument();
    expect(screen.getByRole("button")).toBeInTheDocument(); // Remove button
  });

  it("should call onImageSelect when remove button is clicked", async () => {
    const user = userEvent.setup();
    const mockFile = new File(["test"], "test.jpg", { type: "image/jpeg" });

    render(
      <ImageUpload onImageSelect={mockOnImageSelect} selectedImage={mockFile} />
    );

    const removeButton = screen.getByRole("button");
    await user.click(removeButton);

    expect(mockOnImageSelect).toHaveBeenCalledWith(null);
  });

  it("should display error message", () => {
    render(
      <ImageUpload
        onImageSelect={mockOnImageSelect}
        selectedImage={null}
        error="File too large"
      />
    );

    expect(screen.getByText("File too large")).toBeInTheDocument();
  });

  it("should be disabled when disabled prop is true", () => {
    render(
      <ImageUpload
        onImageSelect={mockOnImageSelect}
        selectedImage={null}
        disabled={true}
      />
    );

    // Find the card element which should have the disabled styling
    const cardElement = screen
      .getByText("Upload an image")
      .closest("[class*='border-2']");
    expect(cardElement).toHaveClass("opacity-50");
  });

  it("should show file size in preview", () => {
    const mockFile = new File(["x".repeat(1024)], "test.jpg", {
      type: "image/jpeg",
    });

    render(
      <ImageUpload onImageSelect={mockOnImageSelect} selectedImage={mockFile} />
    );

    expect(screen.getByText(/1 KB/)).toBeInTheDocument();
  });

  it("should show valid image indicator", () => {
    const mockFile = new File(["test"], "test.jpg", { type: "image/jpeg" });

    render(
      <ImageUpload onImageSelect={mockOnImageSelect} selectedImage={mockFile} />
    );

    expect(screen.getByText("Valid image")).toBeInTheDocument();
  });
});
