import { get, post, put, del } from './api-client';

// Outputs API
export const outputsApi = {
  list: () => get('/api/v1/outputs'),
  get: (id: string) => get(`/api/v1/outputs/${id}`),
  delete: (id: string) => del(`/api/v1/outputs/${id}`),
  getDownloadUrl: (id: string) => `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/v1/outputs/${id}/download`,
};

// Generation API
export const generationApi = {
  create: (data: any) => post('/api/v1/generation/submit', data),
  getStatus: (taskId: string) => get(`/api/v1/generation/${taskId}`),
  cancel: (taskId: string) => del(`/api/v1/generation/${taskId}`),
};

// Queue API
export const queueApi = {
  getStatus: () => get('/api/v1/queue'),
  clear: () => del('/api/v1/queue'),
};

// System API
export const systemApi = {
  getHealth: () => get('/api/v1/system/health'),
  getDiagnostics: () => get('/api/v1/system/diagnostics'),
  getGpuStatus: () => get('/api/v1/system/gpu'),
};

// Prompt API
export const promptApi = {
  enhance: (data: any) => post('/api/v1/prompt/enhance', data),
};

// Export all APIs
export default {
  outputs: outputsApi,
  generation: generationApi,
  queue: queueApi,
  system: systemApi,
  prompt: promptApi,
};