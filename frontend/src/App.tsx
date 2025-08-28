import { Routes, Route } from "react-router-dom";
import { Toaster } from "@/components/ui/toaster";
import Layout from "@/components/layout/Layout";
import GenerationPage from "@/pages/GenerationPage";
import QueuePage from "@/pages/QueuePage";
import SystemPage from "@/pages/SystemPage";
import OutputsPage from "@/pages/OutputsPage";
import { LoRAPage } from "@/pages/LoRAPage";
import { ThemeProvider } from "@/components/providers/ThemeProvider";
import {
  AccessibilityProvider,
  RouteFocusManager,
} from "@/components/providers/AccessibilityProvider";
import { useKeyboardNavigation } from "@/hooks/use-keyboard-navigation";
import { useOffline } from "@/hooks/use-offline";
import ErrorBoundary from "@/components/error/ErrorBoundary";
import { OfflineIndicator } from "@/components/ui/offline-indicator";
import {
  KeyboardHelpButton,
  SkipLinks,
  LandmarkNavigation,
} from "@/components/ui/keyboard-navigation";
import StartupValidator from "@/components/startup/StartupValidator";
import { useEffect } from "react";

function App() {
  // Enable keyboard navigation
  useKeyboardNavigation();

  // Initialize offline functionality
  const { requestNotificationPermission } = useOffline();

  // Request notification permission on app load
  useEffect(() => {
    requestNotificationPermission();
  }, [requestNotificationPermission]);

  return (
    <ErrorBoundary>
      <ThemeProvider defaultTheme="light" storageKey="wan22-ui-theme">
        <AccessibilityProvider>
          <StartupValidator>
            <SkipLinks />
            <LandmarkNavigation />
            <Layout>
              <ErrorBoundary>
                <main id="main-content" tabIndex={-1}>
                  <Routes>
                    <Route path="/" element={<GenerationPage />} />
                    <Route path="/generation" element={<GenerationPage />} />
                    <Route path="/queue" element={<QueuePage />} />
                    <Route path="/system" element={<SystemPage />} />
                    <Route path="/outputs" element={<OutputsPage />} />
                    <Route path="/lora" element={<LoRAPage />} />
                  </Routes>
                </main>
              </ErrorBoundary>
            </Layout>
            <RouteFocusManager />
            <OfflineIndicator />
            <KeyboardHelpButton />
            <Toaster />
          </StartupValidator>
        </AccessibilityProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;
