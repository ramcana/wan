import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";
import { createEnvReloadPlugin } from "./src/lib/vite-env-reload-plugin";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    createEnvReloadPlugin({
      watchedVars: ["VITE_API_URL", "VITE_DEV_MODE"],
      onEnvChange: (changedVars) => {
        console.log(
          "Environment variables changed, triggering cache clear:",
          changedVars
        );
      },
    }),
  ],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    host: '127.0.0.1', // Use IPv4 localhost explicitly
    port: 8080, // Use a commonly available port
    strictPort: true,
    cors: true,
    proxy: {
      '/api': {
        target: process.env.VITE_API_URL || 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      }
    }
  },
  optimizeDeps: {
    include: [
      "react",
      "react-dom",
      "react-router-dom",
      "react-query",
      "zustand",
      "axios",
      "zod",
      "lucide-react",
      "@radix-ui/react-dialog",
      "@radix-ui/react-dropdown-menu",
      "@radix-ui/react-progress",
      "@radix-ui/react-select",
      "@radix-ui/react-slider",
      "@radix-ui/react-switch",
      "@radix-ui/react-toast",
      "@radix-ui/react-label",
      "@radix-ui/react-slot",
      "class-variance-authority",
      "clsx",
      "tailwind-merge",
    ],
  },
  build: {
    outDir: "dist",
    sourcemap: true,
    rollupOptions: {
      input: {
        main: path.resolve(__dirname, "index.html"),
      },
      output: {
        manualChunks: {
          vendor: ["react", "react-dom"],
          router: ["react-router-dom"],
          ui: [
            "@radix-ui/react-dialog",
            "@radix-ui/react-dropdown-menu",
            "@radix-ui/react-progress",
          ],
        },
      },
    },
  },
});