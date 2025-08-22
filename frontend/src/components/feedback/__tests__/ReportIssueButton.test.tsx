/**
 * ReportIssueButton Component Tests
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { ReportIssueButton } from "../ReportIssueButton";

// Mock the toast hook
const mockToast = vi.fn();
vi.mock("../../../hooks/use-toast", () => ({
  useToast: () => ({ toast: mockToast }),
}));

describe("ReportIssueButton", () => {
  beforeEach(() => {
    mockToast.mockClear();
  });

  it("should render report issue button", () => {
    render(<ReportIssueButton />);

    const button = screen.getByTestId("report-issue-button");
    expect(button).toBeInTheDocument();
    expect(button).toHaveTextContent("Report Issue");
  });

  it("should open dialog when button is clicked", async () => {
    render(<ReportIssueButton />);

    const button = screen.getByTestId("report-issue-button");
    fireEvent.click(button);

    await waitFor(() => {
      expect(screen.getByText("Report an Issue")).toBeInTheDocument();
    });
  });

  it("should validate form fields", async () => {
    render(<ReportIssueButton />);

    // Open dialog
    fireEvent.click(screen.getByTestId("report-issue-button"));

    await waitFor(() => {
      expect(screen.getByText("Report an Issue")).toBeInTheDocument();
    });

    // Try to submit without filling fields
    const submitButton = screen.getByTestId("submit-issue-button");
    fireEvent.click(submitButton);

    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledWith({
        title: "Validation Error",
        description: "Please select an issue type and provide a description.",
        variant: "destructive",
      });
    });
  });

  it("should submit issue report successfully", async () => {
    render(
      <ReportIssueButton context={{ page: "generation", action: "submit" }} />
    );

    // Open dialog
    fireEvent.click(screen.getByTestId("report-issue-button"));

    await waitFor(() => {
      expect(screen.getByText("Report an Issue")).toBeInTheDocument();
    });

    // Fill form
    const issueTypeSelect = screen.getByTestId("issue-type-select");
    fireEvent.click(issueTypeSelect);

    await waitFor(() => {
      const bugOption = screen.getByText("Bug / Error");
      fireEvent.click(bugOption);
    });

    const descriptionTextarea = screen.getByTestId("issue-description");
    fireEvent.change(descriptionTextarea, {
      target: { value: "Test issue description" },
    });

    // Submit form
    const submitButton = screen.getByTestId("submit-issue-button");
    fireEvent.click(submitButton);

    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledWith({
        title: "Issue Reported",
        description: "Thank you for your feedback. We'll look into this issue.",
        variant: "default",
      });
    });
  });

  it("should display context information when provided", async () => {
    const context = {
      page: "generation",
      action: "submit",
      error: "VRAM exhausted",
    };

    render(<ReportIssueButton context={context} />);

    // Open dialog
    fireEvent.click(screen.getByTestId("report-issue-button"));

    await waitFor(() => {
      expect(screen.getByText("Context Information")).toBeInTheDocument();
      expect(screen.getByText("Page: generation")).toBeInTheDocument();
      expect(screen.getByText("Action: submit")).toBeInTheDocument();
      expect(screen.getByText("Error: VRAM exhausted")).toBeInTheDocument();
    });
  });

  it("should show character counter for description", async () => {
    render(<ReportIssueButton />);

    // Open dialog
    fireEvent.click(screen.getByTestId("report-issue-button"));

    await waitFor(() => {
      expect(screen.getByText("0/1000")).toBeInTheDocument();
    });

    // Type in description
    const descriptionTextarea = screen.getByTestId("issue-description");
    fireEvent.change(descriptionTextarea, {
      target: { value: "Test description" },
    });

    expect(screen.getByText("16/1000")).toBeInTheDocument();
  });

  it("should handle different variants and sizes", () => {
    const { rerender } = render(
      <ReportIssueButton variant="ghost" size="lg" />
    );

    let button = screen.getByTestId("report-issue-button");
    expect(button).toBeInTheDocument();

    rerender(<ReportIssueButton variant="outline" size="sm" />);
    button = screen.getByTestId("report-issue-button");
    expect(button).toBeInTheDocument();
  });
});
