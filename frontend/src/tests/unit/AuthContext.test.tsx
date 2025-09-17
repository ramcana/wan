import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { AuthProvider, useAuth } from "../../contexts/AuthContext";

// Mock fetch globally
global.fetch = vi.fn();

// Test component that uses the auth context
const TestComponent: React.FC = () => {
  const { user, isAuthenticated, login, logout, register } = useAuth();

  return (
    <div>
      <div data-testid="auth-status">
        {isAuthenticated ? "Authenticated" : "Not Authenticated"}
      </div>
      {user && (
        <div data-testid="user-info">
          <span data-testid="username">{user.username}</span>
          <span data-testid="email">{user.email}</span>
        </div>
      )}
      <button
        data-testid="login-btn"
        onClick={() => login("testuser", "password")}
      >
        Login
      </button>
      <button data-testid="logout-btn" onClick={logout}>
        Logout
      </button>
      <button
        data-testid="register-btn"
        onClick={() => register("newuser", "new@example.com", "password")}
      >
        Register
      </button>
    </div>
  );
};

// Wrapper component with AuthProvider
const Wrapper: React.FC = () => (
  <AuthProvider>
    <TestComponent />
  </AuthProvider>
);

describe("AuthContext", () => {
  beforeEach(() => {
    // Clear localStorage before each test
    localStorage.clear();

    // Reset fetch mock
    (fetch as any).mockReset();
  });

  afterEach(() => {
    // Clear localStorage after each test
    localStorage.clear();
  });

  it("should initialize with no user", () => {
    render(<Wrapper />);

    const statusElement = screen.getByTestId("auth-status");
    expect(statusElement.textContent).toBe("Not Authenticated");
    expect(screen.queryByTestId("user-info")).toBeNull();
  });

  it("should login successfully", async () => {
    const user = userEvent.setup();

    // Mock successful login response
    (fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        user: {
          id: "1",
          username: "testuser",
          email: "test@example.com",
          isAdmin: false,
          createdAt: "2023-01-01T00:00:00Z",
        },
        access_token: "access-token",
        refresh_token: "refresh-token",
      }),
    });

    render(<Wrapper />);

    await user.click(screen.getByTestId("login-btn"));

    // Wait for state update
    await new Promise((resolve) => setTimeout(resolve, 100));

    const statusElement = screen.getByTestId("auth-status");
    expect(statusElement.textContent).toBe("Authenticated");

    // Check that tokens are stored in localStorage
    expect(localStorage.getItem("accessToken")).toBe("access-token");
    expect(localStorage.getItem("refreshToken")).toBe("refresh-token");
    expect(localStorage.getItem("user")).toBeTruthy();
  });

  it("should handle login failure", async () => {
    const user = userEvent.setup();

    // Mock failed login response
    (fetch as any).mockResolvedValueOnce({
      ok: false,
      status: 401,
    });

    render(<Wrapper />);

    await user.click(screen.getByTestId("login-btn"));

    // Wait for state update
    await new Promise((resolve) => setTimeout(resolve, 100));

    // Should remain unauthenticated
    const statusElement = screen.getByTestId("auth-status");
    expect(statusElement.textContent).toBe("Not Authenticated");
  });

  it("should logout and clear state", async () => {
    const user = userEvent.setup();

    // Mock successful login response
    (fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        user: {
          id: "1",
          username: "testuser",
          email: "test@example.com",
          isAdmin: false,
          createdAt: "2023-01-01T00:00:00Z",
        },
        access_token: "access-token",
        refresh_token: "refresh-token",
      }),
    });

    render(<Wrapper />);

    // Login first
    await user.click(screen.getByTestId("login-btn"));

    // Wait for state update
    await new Promise((resolve) => setTimeout(resolve, 100));

    const statusElement = screen.getByTestId("auth-status");
    expect(statusElement.textContent).toBe("Authenticated");

    // Then logout
    await user.click(screen.getByTestId("logout-btn"));

    expect(screen.getByTestId("auth-status").textContent).toBe(
      "Not Authenticated"
    );
    expect(screen.queryByTestId("user-info")).toBeNull();

    // Check that tokens are cleared from localStorage
    expect(localStorage.getItem("accessToken")).toBeNull();
    expect(localStorage.getItem("refreshToken")).toBeNull();
    expect(localStorage.getItem("user")).toBeNull();
  });

  it("should register successfully", async () => {
    const user = userEvent.setup();

    // Mock successful registration response
    (fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        user: {
          id: "2",
          username: "newuser",
          email: "new@example.com",
          isAdmin: false,
          createdAt: "2023-01-01T00:00:00Z",
        },
        access_token: "new-access-token",
        refresh_token: "new-refresh-token",
      }),
    });

    render(<Wrapper />);

    await user.click(screen.getByTestId("register-btn"));

    // Wait for state update
    await new Promise((resolve) => setTimeout(resolve, 100));

    const statusElement = screen.getByTestId("auth-status");
    expect(statusElement.textContent).toBe("Authenticated");

    // Check that tokens are stored in localStorage
    expect(localStorage.getItem("accessToken")).toBe("new-access-token");
    expect(localStorage.getItem("refreshToken")).toBe("new-refresh-token");
  });
});
