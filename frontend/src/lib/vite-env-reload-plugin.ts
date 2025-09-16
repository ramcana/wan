import type { Plugin } from 'vite';
import { watch } from 'fs';
import { resolve } from 'path';

export interface EnvReloadPluginOptions {
  watchedVars?: string[];
  onEnvChange?: (changedVars: string[]) => void;
  envFile?: string;
}

export function createEnvReloadPlugin(options: EnvReloadPluginOptions = {}): Plugin {
  const {
    watchedVars = [],
    onEnvChange,
    envFile = '.env'
  } = options;

  let envCache: Record<string, string> = {};
  let watcher: ReturnType<typeof watch> | null = null;

  const loadEnvVars = () => {
    const newCache: Record<string, string> = {};
    
    // Load from process.env
    watchedVars.forEach(varName => {
      const value = process.env[varName];
      if (value !== undefined) {
        newCache[varName] = value;
      }
    });

    return newCache;
  };

  const detectChanges = (oldEnv: Record<string, string>, newEnv: Record<string, string>) => {
    const changedVars: string[] = [];
    
    watchedVars.forEach(varName => {
      if (oldEnv[varName] !== newEnv[varName]) {
        changedVars.push(varName);
      }
    });

    return changedVars;
  };

  return {
    name: 'vite-env-reload',
    configureServer(server) {
      // Initialize env cache
      envCache = loadEnvVars();

      // Watch for env file changes
      const envPath = resolve(process.cwd(), envFile);
      
      try {
        watcher = watch(envPath, (eventType) => {
          if (eventType === 'change') {
            setTimeout(() => {
              const newEnv = loadEnvVars();
              const changedVars = detectChanges(envCache, newEnv);
              
              if (changedVars.length > 0) {
                envCache = newEnv;
                onEnvChange?.(changedVars);
                
                // Trigger HMR update
                server.ws.send({
                  type: 'full-reload'
                });
              }
            }, 100); // Debounce file changes
          }
        });
      } catch (error) {
        console.warn(`Could not watch env file ${envPath}:`, error);
      }
    },
    buildStart() {
      // Load initial env vars
      envCache = loadEnvVars();
    },
    buildEnd() {
      // Cleanup watcher
      if (watcher) {
        watcher.close();
        watcher = null;
      }
    }
  };
}
