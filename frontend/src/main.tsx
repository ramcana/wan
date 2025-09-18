import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "react-query";
import { ReactQueryDevtools } from "react-query/devtools";
import App from "./App.tsx";
import "./index.css";

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
});

// Register service worker
if ("serviceWorker" in navigator) {
  window.addEventListener("load", async () => {
    try {
      // Register service worker
      const registration = await navigator.serviceWorker.register("/sw.js");
      console.log("SW registered: ", registration);

      // Handle service worker updates
      registration.addEventListener("updatefound", () => {
        const newWorker = registration.installing;
        if (newWorker) {
          newWorker.addEventListener("statechange", () => {
            if (
              newWorker.state === "installed" &&
              navigator.serviceWorker.controller
            ) {
              console.log("New service worker available, clearing caches...");
              // Clear caches when new service worker is available
              if ("caches" in window) {
                caches
                  .keys()
                  .then((cacheNames) => {
                    cacheNames.forEach((cacheName) => {
                      caches.delete(cacheName);
                    });
                  })
                  .then(() => {
                    console.log("Caches cleared, reloading...");
                    window.location.reload();
                  });
              }
            }
          });
        }
      });

      // Clear service worker cache on startup to avoid cached configuration issues
      if ("caches" in window) {
        caches.keys().then((cacheNames) => {
          cacheNames.forEach((cacheName) => {
            caches.delete(cacheName);
          });
        });
        console.log("Service worker caches cleared on startup");
      }
    } catch (registrationError) {
      console.log("SW registration failed: ", registrationError);
    }
  });
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
      <ReactQueryDevtools initialIsOpen={false} />
    </QueryClientProvider>
  </React.StrictMode>
);
