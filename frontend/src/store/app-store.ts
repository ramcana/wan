import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

// Global application state
interface AppState {
  theme: 'light' | 'dark' | 'system';
  isOnline: boolean;
  notifications: Notification[];
  sidebarCollapsed: boolean;
  
  // Actions
  setTheme: (theme: 'light' | 'dark' | 'system') => void;
  setOnlineStatus: (isOnline: boolean) => void;
  addNotification: (notification: Notification) => void;
  removeNotification: (id: string) => void;
  toggleSidebar: () => void;
  setSidebarCollapsed: (collapsed: boolean) => void;
}

interface Notification {
  id: string;
  title: string;
  message: string;
  type: 'info' | 'success' | 'warning' | 'error';
  timestamp: Date;
  read: boolean;
}

export const useAppStore = create<AppState>()(
  devtools(
    persist(
      (set) => ({
        // Initial state
        theme: 'system',
        isOnline: navigator.onLine,
        notifications: [],
        sidebarCollapsed: false,

        // Actions
        setTheme: (theme) => set({ theme }),
        
        setOnlineStatus: (isOnline) => set({ isOnline }),
        
        addNotification: (notification) => 
          set((state) => ({
            notifications: [notification, ...state.notifications].slice(0, 50) // Keep last 50
          })),
        
        removeNotification: (id) =>
          set((state) => ({
            notifications: state.notifications.filter(n => n.id !== id)
          })),
        
        toggleSidebar: () =>
          set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
        
        setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),
      }),
      {
        name: 'wan22-app-store',
        partialize: (state) => ({
          theme: state.theme,
          sidebarCollapsed: state.sidebarCollapsed,
        }),
      }
    ),
    { name: 'AppStore' }
  )
);