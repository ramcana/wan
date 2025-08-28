import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import { createEnvReloadPlugin } from './src/lib/vite-env-reload-plugin'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    createEnvReloadPlugin({
      watchedVars: ['VITE_API_URL', 'VITE_DEV_MODE'],
      onEnvChange: (changedVars) => {
        console.log('Environment variables changed, triggering cache clear:', changedVars);
      }
    })
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test-setup.ts'],
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: process.env.VITE_API_URL || 'http://localhost:9000',
        changeOrigin: true,
        secure: false,
        configure: (proxy, options) => {
          proxy.on('error', (err, req, res) => {
            console.log('Proxy error:', err);
          });
          proxy.on('proxyReq', (proxyReq, req, res) => {
            console.log('Proxying request:', req.method, req.url, 'to', options.target);
          });
        }
      },
    },
  },
  optimizeDeps: {
    include: [
      'react',
      'react-dom',
      'react-router-dom',
      'react-query',
      'zustand',
      'axios',
      'zod',
      'lucide-react',
      '@radix-ui/react-dialog',
      '@radix-ui/react-dropdown-menu',
      '@radix-ui/react-progress',
      '@radix-ui/react-select',
      '@radix-ui/react-slider',
      '@radix-ui/react-switch',
      '@radix-ui/react-toast',
      '@radix-ui/react-label',
      '@radix-ui/react-slot',
      'class-variance-authority',
      'clsx',
      'tailwind-merge'
    ],
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      input: {
        main: path.resolve(__dirname, 'index.html'),
      },
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          router: ['react-router-dom'],
          ui: ['@radix-ui/react-dialog', '@radix-ui/react-dropdown-menu', '@radix-ui/react-progress'],
        },
      },
    },
  },
})