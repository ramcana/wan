import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  ReactNode,
} from "react";

interface User {
  id: string;
  username: string;
  email: string;
  isAdmin: boolean;
  createdAt: string;
}

interface AuthContextType {
  user: User | null;
  accessToken: string | null;
  refreshToken: string | null;
  isAuthenticated: boolean;
  login: (username: string, password: string) => Promise<boolean>;
  logout: () => void;
  register: (
    username: string,
    email: string,
    password: string
  ) => Promise<boolean>;
  refreshAccessToken: () => Promise<boolean>;
  updateProfile: (userData: Partial<User>) => Promise<boolean>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [accessToken, setAccessToken] = useState<string | null>(null);
  const [refreshToken, setRefreshToken] = useState<string | null>(null);

  // Initialize auth state from localStorage
  useEffect(() => {
    const storedUser = localStorage.getItem("user");
    const storedAccessToken = localStorage.getItem("accessToken");
    const storedRefreshToken = localStorage.getItem("refreshToken");

    if (storedUser && storedAccessToken) {
      setUser(JSON.parse(storedUser));
      setAccessToken(storedAccessToken);
      setRefreshToken(storedRefreshToken || null);
    }
  }, []);

  // Save auth state to localStorage
  const saveAuthState = (
    userData: User | null,
    access: string | null,
    refresh: string | null
  ) => {
    if (userData && access) {
      localStorage.setItem("user", JSON.stringify(userData));
      localStorage.setItem("accessToken", access);
      if (refresh) {
        localStorage.setItem("refreshToken", refresh);
      }
    } else {
      localStorage.removeItem("user");
      localStorage.removeItem("accessToken");
      localStorage.removeItem("refreshToken");
    }
  };

  const login = async (
    username: string,
    password: string
  ): Promise<boolean> => {
    try {
      const response = await fetch("/api/v1/auth/login", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ username, password }),
      });

      if (response.ok) {
        const data = await response.json();
        setUser(data.user);
        setAccessToken(data.access_token);
        setRefreshToken(data.refresh_token);
        saveAuthState(data.user, data.access_token, data.refresh_token);
        return true;
      }
      return false;
    } catch (error) {
      console.error("Login error:", error);
      return false;
    }
  };

  const logout = () => {
    setUser(null);
    setAccessToken(null);
    setRefreshToken(null);
    saveAuthState(null, null, null);
  };

  const register = async (
    username: string,
    email: string,
    password: string
  ): Promise<boolean> => {
    try {
      const response = await fetch("/api/v1/auth/register", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ username, email, password }),
      });

      if (response.ok) {
        const data = await response.json();
        setUser(data.user);
        setAccessToken(data.access_token);
        setRefreshToken(data.refresh_token);
        saveAuthState(data.user, data.access_token, data.refresh_token);
        return true;
      }
      return false;
    } catch (error) {
      console.error("Registration error:", error);
      return false;
    }
  };

  const refreshAccessToken = async (): Promise<boolean> => {
    if (!refreshToken) return false;

    try {
      const response = await fetch("/api/v1/auth/refresh", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ refresh_token: refreshToken }),
      });

      if (response.ok) {
        const data = await response.json();
        setAccessToken(data.access_token);
        saveAuthState(user, data.access_token, refreshToken);
        return true;
      } else {
        // If refresh fails, logout user
        logout();
        return false;
      }
    } catch (error) {
      console.error("Token refresh error:", error);
      logout();
      return false;
    }
  };

  const updateProfile = async (userData: Partial<User>): Promise<boolean> => {
    if (!accessToken) return false;

    try {
      const response = await fetch("/api/v1/auth/profile", {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${accessToken}`,
        },
        body: JSON.stringify(userData),
      });

      if (response.ok) {
        const updatedUser = await response.json();
        setUser(updatedUser);
        saveAuthState(updatedUser, accessToken, refreshToken);
        return true;
      }
      return false;
    } catch (error) {
      console.error("Profile update error:", error);
      return false;
    }
  };

  const value = {
    user,
    accessToken,
    refreshToken,
    isAuthenticated: !!user && !!accessToken,
    login,
    logout,
    register,
    refreshAccessToken,
    updateProfile,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
};
