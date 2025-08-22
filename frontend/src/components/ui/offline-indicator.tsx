import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Wifi, WifiOff, Clock, AlertCircle } from "lucide-react";
import { useOffline } from "../../hooks/use-offline";
import { Button } from "./button";
import { Badge } from "./badge";
import { useAccessibilityContext } from "../providers/AccessibilityProvider";

export const OfflineIndicator: React.FC = () => {
  const {
    isOnline,
    queuedRequestsCount,
    syncInProgress,
    lastSyncAttempt,
    processOfflineQueue,
    clearOfflineQueue,
  } = useOffline();

  const { announce } = useAccessibilityContext();

  const handleSyncNow = async () => {
    announce("Syncing queued requests", "polite");
    const success = await processOfflineQueue();

    if (success) {
      announce("Sync completed successfully", "polite");
    } else {
      announce("Sync failed. Please try again.", "assertive");
    }
  };

  const handleClearQueue = async () => {
    if (
      window.confirm(
        "Are you sure you want to clear all queued requests? This action cannot be undone."
      )
    ) {
      announce("Clearing offline queue", "polite");
      const success = await clearOfflineQueue();

      if (success) {
        announce("Offline queue cleared", "polite");
      } else {
        announce("Failed to clear queue", "assertive");
      }
    }
  };

  return (
    <div className="fixed bottom-4 right-4 z-50">
      <AnimatePresence>
        {(!isOnline || queuedRequestsCount > 0) && (
          <motion.div
            initial={{ opacity: 0, y: 50, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 50, scale: 0.9 }}
            transition={{ duration: 0.3, ease: "easeOut" }}
            className="bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 p-4 max-w-sm"
            role="status"
            aria-live="polite"
            aria-label={
              isOnline ? "Online with queued requests" : "Offline mode active"
            }
          >
            {/* Status Header */}
            <div className="flex items-center gap-3 mb-3">
              <div className="flex-shrink-0">
                {isOnline ? (
                  <Wifi className="w-5 h-5 text-green-500" aria-hidden="true" />
                ) : (
                  <WifiOff
                    className="w-5 h-5 text-red-500"
                    aria-hidden="true"
                  />
                )}
              </div>

              <div className="flex-1">
                <h3 className="font-medium text-gray-900 dark:text-gray-100">
                  {isOnline ? "Online" : "Offline Mode"}
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {isOnline
                    ? "All features available"
                    : "Limited functionality"}
                </p>
              </div>
            </div>

            {/* Queued Requests */}
            {queuedRequestsCount > 0 && (
              <div className="mb-3">
                <div className="flex items-center gap-2 mb-2">
                  <Clock
                    className="w-4 h-4 text-orange-500"
                    aria-hidden="true"
                  />
                  <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
                    Queued Requests
                  </span>
                  <Badge variant="secondary" className="ml-auto">
                    {queuedRequestsCount}
                  </Badge>
                </div>

                <p className="text-xs text-gray-600 dark:text-gray-400 mb-3">
                  {isOnline
                    ? "These requests will be processed automatically"
                    : "These will be sent when you're back online"}
                </p>

                {/* Action Buttons */}
                <div className="flex gap-2">
                  {isOnline && (
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={handleSyncNow}
                      disabled={syncInProgress}
                      className="flex-1"
                      aria-describedby="sync-description"
                    >
                      {syncInProgress ? (
                        <>
                          <motion.div
                            animate={{ rotate: 360 }}
                            transition={{
                              duration: 1,
                              repeat: Infinity,
                              ease: "linear",
                            }}
                            className="w-3 h-3 border-2 border-current border-t-transparent rounded-full mr-2"
                            aria-hidden="true"
                          />
                          Syncing...
                        </>
                      ) : (
                        "Sync Now"
                      )}
                    </Button>
                  )}

                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={handleClearQueue}
                    className="flex-shrink-0"
                    aria-describedby="clear-description"
                  >
                    Clear
                  </Button>
                </div>

                {/* Hidden descriptions for screen readers */}
                <div id="sync-description" className="sr-only">
                  Manually sync all queued requests now
                </div>
                <div id="clear-description" className="sr-only">
                  Clear all queued requests. This action cannot be undone.
                </div>
              </div>
            )}

            {/* Last Sync Info */}
            {lastSyncAttempt && (
              <div className="text-xs text-gray-500 dark:text-gray-400 border-t border-gray-200 dark:border-gray-700 pt-2">
                Last sync: {lastSyncAttempt.toLocaleTimeString()}
              </div>
            )}

            {/* Offline Features Info */}
            {!isOnline && (
              <div className="mt-3 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-md">
                <div className="flex items-start gap-2">
                  <AlertCircle
                    className="w-4 h-4 text-blue-500 mt-0.5 flex-shrink-0"
                    aria-hidden="true"
                  />
                  <div>
                    <h4 className="text-sm font-medium text-blue-900 dark:text-blue-100 mb-1">
                      Available Offline
                    </h4>
                    <ul className="text-xs text-blue-800 dark:text-blue-200 space-y-1">
                      <li>• Browse cached content</li>
                      <li>• View generation history</li>
                      <li>• Queue new requests</li>
                      <li>• Access system info</li>
                    </ul>
                  </div>
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

// Compact version for header/navbar
export const OfflineIndicatorCompact: React.FC = () => {
  const { isOnline, queuedRequestsCount } = useOffline();

  if (isOnline && queuedRequestsCount === 0) return null;

  return (
    <div className="flex items-center gap-2">
      {!isOnline && (
        <Badge variant="destructive" className="flex items-center gap-1">
          <WifiOff className="w-3 h-3" aria-hidden="true" />
          <span className="sr-only">Offline mode active</span>
          Offline
        </Badge>
      )}

      {queuedRequestsCount > 0 && (
        <Badge variant="secondary" className="flex items-center gap-1">
          <Clock className="w-3 h-3" aria-hidden="true" />
          <span className="sr-only">{queuedRequestsCount} requests queued</span>
          {queuedRequestsCount}
        </Badge>
      )}
    </div>
  );
};

// Hook for offline-aware components
export const useOfflineAware = () => {
  const { isOnline, queueRequest, getOfflineData } = useOffline();
  const { announceError, announceSuccess } = useAccessibilityContext();

  const handleOfflineAction = async (
    action: () => Promise<any>,
    offlineMessage: string,
    successMessage?: string
  ) => {
    if (!isOnline) {
      announceError(offlineMessage);
      return { success: false, offline: true };
    }

    try {
      const result = await action();
      if (successMessage) {
        announceSuccess(successMessage);
      }
      return { success: true, data: result };
    } catch (error) {
      announceError("Action failed. Please try again.");
      return { success: false, error };
    }
  };

  return {
    isOnline,
    handleOfflineAction,
    queueRequest,
    getOfflineData,
  };
};
