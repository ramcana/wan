import axios, {
  AxiosError,
  AxiosInstance,
  AxiosRequestConfig,
  AxiosResponse,
} from "axios";
import {
  LoRAListResponseSchema,
  LoRAMemoryImpactResponseSchema,
  LoRAPreviewResponseSchema,
  LoRAStatusResponseSchema,
  LoRAUploadResponseSchema,
  type LoRAListResponse,
  type LoRAMemoryImpactResponse,
  type LoRAPreviewResponse,
  type LoRAStatusResponse,
  type LoRAUploadResponse,
  validateApiResponse,
} from "./api-schemas";

const DEFAULT_TIMEOUT = 30_000;

const resolveBaseUrl = (): string | undefined => {
  let rawUrl: string | undefined;

  try {
    rawUrl = (import.meta as ImportMeta).env?.VITE_API_URL as
      | string
      | undefined;
  } catch {
    // import.meta is not available in some test environments
    rawUrl = undefined;
  }

  if (!rawUrl && typeof process !== "undefined") {
    rawUrl = process.env?.VITE_API_URL;
  }

  if (!rawUrl) {
    return undefined;
  }

  const trimmed = rawUrl.trim();
  if (!trimmed) {
    return undefined;
  }

  return trimmed.replace(/\/+$/, "");
};

const API_BASE_URL = resolveBaseUrl();

export class ApiError<T = unknown> extends Error {
  readonly status: number;
  readonly details?: T;
  readonly code?: string;

  constructor(
    status: number,
    message: string,
    details?: T,
    code?: string,
    options?: { cause?: unknown }
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.details = details;
    this.code = code;

    if (options?.cause !== undefined) {
      this.cause = options.cause;
    }

    Object.setPrototypeOf(this, new.target.prototype);
  }
}

const apiClientConfig: AxiosRequestConfig = {
  baseURL: API_BASE_URL,
  timeout: DEFAULT_TIMEOUT,
  withCredentials: false,
  headers: {
    Accept: "application/json",
  },
};

export const apiClient: AxiosInstance = axios.create(apiClientConfig);

const isFormData = (value: unknown): value is FormData =>
  typeof FormData !== "undefined" && value instanceof FormData;

const extractErrorDetails = (error: AxiosError): {
  status: number;
  message: string;
  details?: unknown;
  code?: string;
} => {
  const status = error.response?.status ?? 0;
  const data = error.response?.data;

  let message = error.message || "Request failed";
  let details: unknown = undefined;
  let code: string | undefined;

  if (data !== undefined && data !== null) {
    if (typeof data === "string") {
      message = data;
    } else if (typeof data === "object") {
      details = data;
      const detailRecord = data as Record<string, unknown>;
      const derivedMessage =
        (typeof detailRecord.message === "string" && detailRecord.message) ||
        (typeof detailRecord.detail === "string" && detailRecord.detail) ||
        (typeof detailRecord.error === "string" && detailRecord.error);

      if (derivedMessage) {
        message = derivedMessage;
      }

      const derivedCode =
        (typeof detailRecord.error === "string" && detailRecord.error) ||
        (typeof detailRecord.code === "string" && detailRecord.code);
      if (derivedCode) {
        code = derivedCode;
      }
    }
  } else if (error.request && status === 0) {
    message = "Network request failed";
  }

  return { status, message, details, code };
};

const toApiError = (error: unknown): ApiError => {
  if (error instanceof ApiError) {
    return error;
  }

  if (axios.isAxiosError(error)) {
    const details = extractErrorDetails(error);
    return new ApiError(details.status, details.message, details.details, details.code, {
      cause: error,
    });
  }

  if (error instanceof Error) {
    return new ApiError(0, error.message || "Unknown error", undefined, undefined, {
      cause: error,
    });
  }

  return new ApiError(0, "Unknown error", error);
};

const request = async <T>(config: AxiosRequestConfig): Promise<T> => {
  if (!config.url) {
    throw new Error("Request URL is required");
  }

  const requestConfig: AxiosRequestConfig = {
    ...config,
  };

  if (requestConfig.data && isFormData(requestConfig.data)) {
    if (requestConfig.headers) {
      delete (requestConfig.headers as Record<string, unknown>)[
        "Content-Type"
      ];
    }
  }

  try {
    const response: AxiosResponse<T> = await apiClient.request<T>(requestConfig);
    return response.data;
  } catch (error) {
    throw toApiError(error);
  }
};

export const get = async <T = unknown>(
  url: string,
  config?: AxiosRequestConfig
): Promise<T> => {
  const finalConfig: AxiosRequestConfig = { ...(config ?? {}), url, method: "GET" };
  return request<T>(finalConfig);
};

export const post = async <T = unknown, D = unknown>(
  url: string,
  data?: D,
  config?: AxiosRequestConfig
): Promise<T> => {
  const finalConfig: AxiosRequestConfig = {
    ...(config ?? {}),
    url,
    data,
    method: "POST",
  };
  return request<T>(finalConfig);
};

export const del = async <T = unknown>(
  url: string,
  config?: AxiosRequestConfig
): Promise<T> => {
  const finalConfig: AxiosRequestConfig = { ...(config ?? {}), url, method: "DELETE" };
  return request<T>(finalConfig);
};

const LORA_BASE_PATH = "/api/v1/lora";

const encodePathSegment = (segment: string): string =>
  encodeURIComponent(segment).replace(/%2F/g, "/");

export const loraApi = {
  async list(): Promise<LoRAListResponse> {
    const raw = await get<unknown>(`${LORA_BASE_PATH}/list`);
    return validateApiResponse(LoRAListResponseSchema, raw);
  },

  async getStatus(loraName: string): Promise<LoRAStatusResponse> {
    const raw = await get<unknown>(
      `${LORA_BASE_PATH}/${encodePathSegment(loraName)}/status`
    );
    return validateApiResponse(LoRAStatusResponseSchema, raw);
  },

  async generatePreview(
    loraName: string,
    basePrompt: string
  ): Promise<LoRAPreviewResponse> {
    const formData = new FormData();
    formData.append("base_prompt", basePrompt);

    const raw = await post<unknown>(
      `${LORA_BASE_PATH}/${encodePathSegment(loraName)}/preview`,
      formData
    );
    return validateApiResponse(LoRAPreviewResponseSchema, raw);
  },

  async estimateMemoryImpact(loraName: string): Promise<LoRAMemoryImpactResponse> {
    const raw = await get<unknown>(
      `${LORA_BASE_PATH}/${encodePathSegment(loraName)}/memory-impact`
    );
    return validateApiResponse(LoRAMemoryImpactResponseSchema, raw);
  },

  async upload(
    file: File,
    name?: string,
    onProgress?: (progress: number) => void
  ): Promise<LoRAUploadResponse> {
    const formData = new FormData();
    formData.append("file", file);
    if (name) {
      formData.append("name", name);
    }

    const raw = await post<unknown>(`${LORA_BASE_PATH}/upload`, formData, {
      onUploadProgress: (event) => {
        if (!onProgress) {
          return;
        }

        const { loaded, total } = event;
        if (typeof total === "number" && total > 0) {
          const percent = Math.min(100, Math.round((loaded / total) * 100));
          onProgress(percent);
        }
      },
    });

    return validateApiResponse(LoRAUploadResponseSchema, raw);
  },

  async delete(loraName: string): Promise<{ message?: string }> {
    return del<{ message?: string }>(
      `${LORA_BASE_PATH}/${encodePathSegment(loraName)}`
    );
  },
};

export type { AxiosRequestConfig };
