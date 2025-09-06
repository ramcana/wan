import { create } from 'zustand';
import { devtools } from 'zustand/middleware';

// Queue task interface
interface QueueTask {
  id: string;
  modelType: string;
  prompt: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  estimatedTime?: number;
  createdAt: Date;
  completedAt?: Date;
  outputPath?: string;
  errorMessage?: string;
}

interface QueueState {
  tasks: QueueTask[];
  isProcessing: boolean;
  totalTasks: number;
  completedTasks: number;
  
  // Actions
  addTask: (task: QueueTask) => void;
  updateTask: (id: string, updates: Partial<QueueTask>) => void;
  removeTask: (id: string) => void;
  clearCompleted: () => void;
  reorderTasks: (startIndex: number, endIndex: number) => void;
  setProcessing: (isProcessing: boolean) => void;
}

export const useQueueStore = create<QueueState>()(
  devtools(
    (set) => ({
      // Initial state
      tasks: [],
      isProcessing: false,
      totalTasks: 0,
      completedTasks: 0,

      // Actions
      addTask: (task) =>
        set((state) => {
          const newTasks = [...state.tasks, task];
          return {
            tasks: newTasks,
            totalTasks: newTasks.length,
          };
        }),

      updateTask: (id, updates) =>
        set((state) => {
          const newTasks = state.tasks.map((task) =>
            task.id === id ? { ...task, ...updates } : task
          );
          const completedTasks = newTasks.filter(t => t.status === 'completed').length;
          
          return {
            tasks: newTasks,
            completedTasks,
          };
        }),

      removeTask: (id) =>
        set((state) => {
          const newTasks = state.tasks.filter((task) => task.id !== id);
          const completedTasks = newTasks.filter(t => t.status === 'completed').length;
          
          return {
            tasks: newTasks,
            totalTasks: newTasks.length,
            completedTasks,
          };
        }),

      clearCompleted: () =>
        set((state) => {
          const newTasks = state.tasks.filter((task) => task.status !== 'completed');
          return {
            tasks: newTasks,
            totalTasks: newTasks.length,
            completedTasks: 0,
          };
        }),

      reorderTasks: (startIndex, endIndex) =>
        set((state) => {
          const newTasks = [...state.tasks];
          const [removed] = newTasks.splice(startIndex, 1);
          newTasks.splice(endIndex, 0, removed);
          
          return { tasks: newTasks };
        }),

      setProcessing: (isProcessing) => set({ isProcessing }),
    }),
    { name: 'QueueStore' }
  )
);