import { z } from 'zod'

// Task Status Enum
export enum TaskStatus {
  PENDING = 'pending',
  PROCESSING = 'processing', 
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled'
}

// Generation Request Schema
export const GenerationFormDataSchema = z.object({
  prompt: z.string().min(1, "Prompt is required"),
  model_type: z.enum(['t2v', 'i2v', 'ti2v']),
  num_frames: z.number().min(1).max(240).default(120),
  fps: z.number().min(1).max(60).default(24),
  width: z.number().min(64).max(2048).default(1024),
  height: z.number().min(64).max(2048).default(576),
  guidance_scale: z.number().min(1).max(20).default(7.5),
  num_inference_steps: z.number().min(1).max(100).default(50),
  seed: z.number().optional(),
  image_file: z.any().optional(), // File input
  video_file: z.any().optional(), // File input
})

export type GenerationFormData = z.infer<typeof GenerationFormDataSchema>

// Generation Response Schema
export const GenerationResponseSchema = z.object({
  task_id: z.string(),
  status: z.nativeEnum(TaskStatus),
  message: z.string(),
  estimated_time: z.number().optional(),
})

export type GenerationResponse = z.infer<typeof GenerationResponseSchema>

// Task Info Schema
export const TaskInfoSchema = z.object({
  id: z.string(),
  status: z.nativeEnum(TaskStatus),
  prompt: z.string(),
  model_type: z.enum(['t2v', 'i2v', 'ti2v']),
  created_at: z.string(),
  started_at: z.string().optional(),
  completed_at: z.string().optional(),
  progress: z.number().min(0).max(100).default(0),
  estimated_time_remaining: z.number().optional(),
  output_path: z.string().optional(),
  error_message: z.string().optional(),
  parameters: z.record(z.any()).optional(),
})

export type TaskInfo = z.infer<typeof TaskInfoSchema>

// Queue Status Schema
export const QueueStatusSchema = z.object({
  total_tasks: z.number(),
  pending_tasks: z.number(),
  processing_tasks: z.number(),
  completed_tasks: z.number(),
  failed_tasks: z.number(),
  cancelled_tasks: z.number(),
  current_task: TaskInfoSchema.optional(),
  queue: z.array(TaskInfoSchema),
})

export type QueueStatus = z.infer<typeof QueueStatusSchema>

// System Health Schema
export const SystemHealthSchema = z.object({
  status: z.enum(['healthy', 'degraded', 'unhealthy']),
  backend_online: z.boolean(),
  database_connected: z.boolean(),
  gpu_available: z.boolean(),
  memory_usage: z.number().optional(),
  gpu_memory_usage: z.number().optional(),
  active_tasks: z.number().optional(),
  uptime: z.number().optional(),
})

export type SystemHealth = z.infer<typeof SystemHealthSchema>

// Port Detection Response Schema
export const PortDetectionResponseSchema = z.object({
  backend_port: z.number(),
  frontend_port: z.number(),
  status: z.string(),
})

export type PortDetectionResponse = z.infer<typeof PortDetectionResponseSchema>

// API Response Validation Helper
export function validateApiResponse<T>(schema: z.ZodSchema<T>, data: unknown): T {
  try {
    return schema.parse(data)
  } catch (error) {
    if (error instanceof z.ZodError) {
      throw new Error(`API response validation failed: ${error.errors.map(e => e.message).join(', ')}`)
    }
    throw error
  }
}

// Output File Schema
export const OutputFileSchema = z.object({
  id: z.string(),
  filename: z.string(),
  path: z.string(),
  size: z.number(),
  created_at: z.string(),
  task_id: z.string().optional(),
  thumbnail_path: z.string().optional(),
  duration: z.number().optional(),
  width: z.number().optional(),
  height: z.number().optional(),
})

export type OutputFile = z.infer<typeof OutputFileSchema>

// Prompt Enhancement Schemas
export const PromptEnhanceRequestSchema = z.object({
  prompt: z.string().min(1),
  style: z.enum(['cinematic', 'artistic', 'realistic', 'anime']).optional(),
  enhance_creativity: z.boolean().default(true),
})

export type PromptEnhanceRequest = z.infer<typeof PromptEnhanceRequestSchema>

export const PromptEnhanceResponseSchema = z.object({
  enhanced_prompt: z.string(),
  suggested_model: z.enum(['t2v', 'i2v', 'ti2v']).optional(),
  confidence: z.number().min(0).max(1).optional(),
})

export type PromptEnhanceResponse = z.infer<typeof PromptEnhanceResponseSchema>
