import axios, {
  AxiosError,
  AxiosInstance,
  AxiosRequestConfig,
  AxiosResponse,
} from "axios";

const API_PREFIX = "/api/v1";
const DEFAULT_PORT = 9000;
const FALLBACK_PORTS = [9000, 8000, 8080, 3001];

const hasWindow = typeof window !== "undefined";
const DEFAULT_PROTOCOL = hasWindow ? window.location.protocol : "http:";
const DEFAULT_HOSTNAME = hasWindow ? window.location.hostname : "localhost";

interface ClientConfig {
  protocol: string;
  hostname: string;
  defaultPort: number;
  origin: string;
  baseUrl: string;
}

export interface PortDetectionResult {
  isHealthy: boolean;
  detectedPort: number;
  baseUrl: string;
  responseTime: number;
  error?: string;
}

export interface ConfigurationValidationResult {
  isValid: boolean;
  issues: string[];
  suggestions: string[];
  detectedConfig: PortDetectionResult;
}

export class ApiError extends Error {
  public status: number;
  public data: unknown;
  public code?: string;

  constructor(status: number, message: string, data?: unknown, code?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.data = data;
    this.code = code;
  }
}

const trimTrailingSlash = (value: string) => value.replace(/\/+$/, "");

const normalizeUrl = (value: string): URL => {
  const trimmed = value.trim();
  if (!trimmed) {
    return new URL(`${DEFAULT_PROTOCOL}//${DEFAULT_HOSTNAME}:${DEFAULT_PORT}`);
  }
  const hasProtocol = /^https?:\/\//i.test(trimmed);
  if (hasProtocol) {
    return new URL(trimmed);
  }
  return new URL(`${DEFAULT_PROTOCOL}//${trimmed}`);
};

type ImportMetaWithEnv = ImportMeta & {
  env: Record<string, string | undefined>;
};

const readEnvVariable = (key: string): string | undefined => {
  try {
    const meta = import.meta as unknown as Partial<ImportMetaWithEnv> | undefined;
    const value = meta?.env?.[key];
    return typeof value === "string" ? value : undefined;
  } catch {
    return undefined;
  }
};

const resolveInitialConfig = (): ClientConfig => {
  const envUrl = readEnvVariable("VITE_API_URL");

  if (envUrl) {
    try {
      const parsed = normalizeUrl(envUrl);
      const port = parsed.port ? Number(parsed.port) : DEFAULT_PORT;
      const baseHost = `${parsed.protocol}//${parsed.hostname}${parsed.port ? `:${parsed.port}` : ""}`;
      const rawPath = trimTrailingSlash(parsed.pathname || "");
      const hasPrefix = rawPath.endsWith(API_PREFIX);
      const origin = hasPrefix
        ? trimTrailingSlash(baseHost + rawPath.slice(0, rawPath.length - API_PREFIX.length))
        : trimTrailingSlash(baseHost + (rawPath && rawPath !== "/" ? rawPath : ""));
      const baseUrl = `${origin}${API_PREFIX}`;
      return {
        protocol: parsed.protocol,
        hostname: parsed.hostname,
        defaultPort: port,
        origin,
        baseUrl,
      };
    } catch (error) {
      console.warn("Failed to parse VITE_API_URL, falling back to defaults", error);
    }
  }

  const origin = `${DEFAULT_PROTOCOL}//${DEFAULT_HOSTNAME}:${DEFAULT_PORT}`;
  return {
    protocol: DEFAULT_PROTOCOL,
    hostname: DEFAULT_HOSTNAME,
    defaultPort: DEFAULT_PORT,
    origin,
    baseUrl: `${origin}${API_PREFIX}`,
  };
};

let clientConfig: ClientConfig = resolveInitialConfig();

const createClient = (): AxiosInstance =>
  axios.create({
    baseURL: clientConfig.baseUrl,
    timeout: 30000,
    withCredentials: false,
  });

export const apiClient = createClient();

const toApiError = (error: unknown): ApiError => {
  if (error instanceof ApiError) {
    return error;
  }

  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError;
    const status = axiosError.response?.status ?? 0;
    const data = axiosError.response?.data;
    const message = axiosError.message || "Request failed";
    return new ApiError(status, message, data, axiosError.code);
  }

  const fallbackMessage = error instanceof Error ? error.message : "Unknown error";
  return new ApiError(0, fallbackMessage, undefined);
};

const request = async <T>(
  method: "get" | "post" | "delete" | "put",
  url: string,
  data?: unknown,
  config?: AxiosRequestConfig
): Promise<T> => {
  try {
    let response: AxiosResponse<T>;
    switch (method) {
      case "get":
        response = await apiClient.get<T>(url, config);
        break;
      case "post":
        response = await apiClient.post<T>(url, data, config);
        break;
      case "put":
        response = await apiClient.put<T>(url, data, config);
        break;
      case "delete":
        response = await apiClient.delete<T>(url, config);
        break;
      default:
        throw new Error(`Unsupported method: ${method}`);
    }
    return response.data;
  } catch (error) {
    throw toApiError(error);
  }
};

export const get = async <T>(url: string, config?: AxiosRequestConfig) =>
  request<T>("get", url, undefined, config);

export const post = async <T>(url: string, data?: unknown, config?: AxiosRequestConfig) =>
  request<T>("post", url, data, config);

export const put = async <T>(url: string, data?: unknown, config?: AxiosRequestConfig) =>
  request<T>("put", url, data, config);

export const del = async <T>(url: string, config?: AxiosRequestConfig) =>
  request<T>("delete", url, undefined, config);

const buildOriginForPort = (port: number): string => {
  return `${clientConfig.protocol}//${clientConfig.hostname}:${port}`;
};

const buildHealthUrl = (port: number): string => {
  return `${buildOriginForPort(port)}${API_PREFIX}/system/health`;
};

const uniquePorts = (ports: number[]): number[] => {
  const seen = new Set<number>();
  const results: number[] = [];
  ports.forEach((port) => {
    if (!seen.has(port)) {
      seen.add(port);
      results.push(port);
    }
  });
  return results;
};

const candidatePorts = (): number[] =>
  uniquePorts([clientConfig.defaultPort, ...FALLBACK_PORTS]);

const logWithTimestamp = (message: string, meta?: unknown) => {
  const timestamp = new Date().toISOString();
  if (meta) {
    console.log(`${message} [${timestamp}]`, meta);
  } else {
    console.log(`${message} [${timestamp}]`);
  }
};

export const portDetectionApi = {
  async testPort(port: number): Promise<PortDetectionResult> {
    const start = typeof performance !== "undefined" ? performance.now() : Date.now();
    const healthUrl = buildHealthUrl(port);

    try {
      const response = await fetch(healthUrl, {
        method: "GET",
        headers: { Accept: "application/json" },
      });
      const end = typeof performance !== "undefined" ? performance.now() : Date.now();
      const responseTime = Math.max(0, Math.round(end - start));

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      await response.json();

      const baseUrl = buildOriginForPort(port);
      logWithTimestamp(`✅ Port ${port} connectivity test successful`, {
        port,
        baseUrl,
        responseTime: `${responseTime}ms`,
      });

      return {
        isHealthy: true,
        detectedPort: port,
        baseUrl,
        responseTime,
      };
    } catch (error) {
      const end = typeof performance !== "undefined" ? performance.now() : Date.now();
      const responseTime = Math.max(0, Math.round(end - start));
      const message = error instanceof Error ? error.message : String(error);
      logWithTimestamp(`⚠️ Port ${port} connectivity test failed`, {
        port,
        error: message,
        responseTime: `${responseTime}ms`,
      });

      return {
        isHealthy: false,
        detectedPort: port,
        baseUrl: buildOriginForPort(port),
        responseTime,
        error: message,
      };
    }
  },

  async detectPort(): Promise<PortDetectionResult> {
    for (const port of candidatePorts()) {
      const result = await this.testPort(port);
      if (result.isHealthy) {
        return result;
      }
    }

    const fallbackBase = buildOriginForPort(clientConfig.defaultPort);
    return {
      isHealthy: false,
      detectedPort: clientConfig.defaultPort,
      baseUrl: fallbackBase,
      responseTime: 0,
      error: "Backend server is not responding on tested ports",
    };
  },

  async validateConfiguration(): Promise<ConfigurationValidationResult> {
    const detection = await this.detectPort();
    const issues: string[] = [];
    const suggestions: string[] = [];

    if (!detection.isHealthy) {
      issues.push("Backend server is not responding");
      suggestions.push(
        "Ensure the backend server is running",
        "Check if the server is running on the expected port"
      );
    } else {
      const expectedBase = buildOriginForPort(clientConfig.defaultPort);
      if (detection.baseUrl !== expectedBase) {
        issues.push(
          `Backend detected on port ${detection.detectedPort}, expected ${clientConfig.defaultPort}`
        );
        suggestions.push(
          `Update frontend configuration to use port ${detection.detectedPort}`
        );
      }

      if (hasWindow) {
        const frontendOrigin = window.location.origin;
        if (frontendOrigin && !detection.baseUrl.includes(new URL(frontendOrigin).hostname)) {
          issues.push(`Frontend origin ${frontendOrigin} may not match backend host`);
          suggestions.push("Verify CORS configuration on the backend");
        }
      }
    }

    return {
      isValid: issues.length === 0,
      issues,
      suggestions,
      detectedConfig: detection,
    };
  },
};

export const setApiBaseOrigin = (origin: string) => {
  try {
    const parsed = normalizeUrl(origin);
    const port = parsed.port ? Number(parsed.port) : clientConfig.defaultPort;
    const baseHost = `${parsed.protocol}//${parsed.hostname}${parsed.port ? `:${parsed.port}` : ""}`;
    clientConfig = {
      protocol: parsed.protocol,
      hostname: parsed.hostname,
      defaultPort: port,
      origin: baseHost,
      baseUrl: `${baseHost}${API_PREFIX}`,
    };
    apiClient.defaults.baseURL = clientConfig.baseUrl;
  } catch (error) {
    console.error("Failed to set API base origin", error);
  }
};

export const getCurrentApiBaseUrl = () => clientConfig.baseUrl;

export const initializePortDetection = async () => {
  try {
    const result = await portDetectionApi.detectPort();
    if (result.isHealthy) {
      setApiBaseOrigin(result.baseUrl);
    }
    return result;
  } catch (error) {
    console.error("Port detection failed", error);
    return {
      isHealthy: false,
      detectedPort: clientConfig.defaultPort,
      baseUrl: buildOriginForPort(clientConfig.defaultPort),
      responseTime: 0,
      error: error instanceof Error ? error.message : "Unknown error",
    } as PortDetectionResult;
  }
};

export const getSystemHealth = async () => {
  return await get(`/system/health`);
};

export const loraApi = {
  list: async () => get(`/lora`),
  getStatus: async (name: string) => get(`/lora/${encodeURIComponent(name)}/status`),
  generatePreview: async (name: string, prompt: string) =>
    post(`/lora/${encodeURIComponent(name)}/preview`, { prompt }),
  estimateMemoryImpact: async (name: string) =>
    get(`/lora/${encodeURIComponent(name)}/memory-impact`),
  upload: async (
    file: File,
    name?: string,
    onProgress?: (progress: number) => void
  ) => {
    const formData = new FormData();
    formData.append("file", file);
    if (name) {
      formData.append("name", name);
    }

    const config: AxiosRequestConfig = {};
    if (onProgress) {
      config.onUploadProgress = (event) => {
        if (event.total) {
          const progress = Math.round((event.loaded / event.total) * 100);
          onProgress(progress);
        }
      };
    }

    return post(`/lora/upload`, formData, config);
  },
  delete: async (name: string) => del(`/lora/${encodeURIComponent(name)}`),
};

export default apiClient;
