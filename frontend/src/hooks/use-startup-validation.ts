/**
 * React hook for startup validation and port detection
 */

import { useState, useEffect, useCallback } from 'react';
import { initializePortDetection } from '@/lib/api-client';
import { validateStartup, type StartupValidationResult } from '@/lib/startup-validator';

export interface StartupState {
  isInitializing: boolean;
  isValidated: boolean;
  validationResult: StartupValidationResult | null;
  error: string | null;
}

export function useStartupValidation() {
  const [state, setState] = useState<StartupState>({
    isInitializing: true,
    isValidated: false,
    validationResult: null,
    error: null,
  });

  // Run startup validation
  const runValidation = useCallback(async () => {
    try {
      console.log(`🚀 [${new Date().toISOString()}] Starting application initialization...`);
      
      setState(prev => ({
        ...prev,
        isInitializing: true,
        error: null,
      }));

      // Step 1: Initialize port detection
      await initializePortDetection();

      // Step 2: Run comprehensive startup validation
      const validationResult = await validateStartup();

      setState({
        isInitializing: false,
        isValidated: validationResult.isValid,
        validationResult,
        error: null,
      });

      if (validationResult.isValid) {
        console.log(`✅ [${new Date().toISOString()}] Application initialization completed successfully`);
      } else {
        console.warn(`⚠️ [${new Date().toISOString()}] Application initialization completed with issues:`, validationResult.issues);
      }

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error during startup validation';
      console.error(`❌ [${new Date().toISOString()}] Application initialization failed:`, error);
      
      setState({
        isInitializing: false,
        isValidated: false,
        validationResult: null,
        error: errorMessage,
      });
    }
  }, []);

  // Retry validation
  const retryValidation = useCallback(() => {
    runValidation();
  }, [runValidation]);

  // Initialize on mount
  useEffect(() => {
    runValidation();
  }, [runValidation]);

  return {
    ...state,
    retryValidation,
  };
}

export default useStartupValidation;