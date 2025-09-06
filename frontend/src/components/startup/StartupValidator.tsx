/**
 * Startup Validation Component
 * Shows validation status and handles connectivity issues during app startup
 */

import React from "react";
import {
  AlertCircle,
  CheckCircle,
  Loader2,
  RefreshCw,
  Wifi,
  WifiOff,
} from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useStartupValidation } from "@/hooks/use-startup-validation";

interface StartupValidatorProps {
  children: React.ReactNode;
}

export function StartupValidator({ children }: StartupValidatorProps) {
  const {
    isInitializing,
    isValidated,
    validationResult,
    error,
    retryValidation,
  } = useStartupValidation();

  // Show loading state during initialization
  if (isInitializing) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <Card className="w-full max-w-md">
          <CardHeader className="text-center">
            <div className="flex justify-center mb-4">
              <Loader2 className="h-8 w-8 animate-spin text-primary" />
            </div>
            <CardTitle>Initializing WAN22</CardTitle>
            <CardDescription>
              Detecting backend configuration and validating connectivity...
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 text-sm text-muted-foreground">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-primary rounded-full animate-pulse" />
                Detecting backend port...
              </div>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-muted rounded-full" />
                Validating CORS configuration...
              </div>
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-muted rounded-full" />
                Testing API connectivity...
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Show error state if validation failed completely
  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background p-4">
        <Card className="w-full max-w-lg">
          <CardHeader>
            <div className="flex items-center gap-2">
              <AlertCircle className="h-6 w-6 text-destructive" />
              <CardTitle className="text-destructive">
                Initialization Failed
              </CardTitle>
            </div>
            <CardDescription>
              An unexpected error occurred during startup validation.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle>Error Details</AlertTitle>
              <AlertDescription className="font-mono text-sm">
                {error}
              </AlertDescription>
            </Alert>

            <div className="flex gap-2">
              <Button onClick={retryValidation} className="flex-1">
                <RefreshCw className="h-4 w-4 mr-2" />
                Retry Initialization
              </Button>
              <Button
                variant="outline"
                onClick={() => window.location.reload()}
              >
                Reload Page
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Show validation issues if backend is not healthy
  if (!isValidated && validationResult) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background p-4">
        <Card className="w-full max-w-2xl">
          <CardHeader>
            <div className="flex items-center gap-2">
              <WifiOff className="h-6 w-6 text-warning" />
              <CardTitle className="text-warning">
                Connectivity Issues Detected
              </CardTitle>
            </div>
            <CardDescription>
              The application detected some connectivity issues that may affect
              functionality.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Status Overview */}
            <div className="grid grid-cols-2 gap-4">
              <div className="flex items-center gap-2">
                <Badge
                  variant={
                    validationResult.backendHealthy ? "default" : "destructive"
                  }
                >
                  {validationResult.backendHealthy ? (
                    <CheckCircle className="h-3 w-3 mr-1" />
                  ) : (
                    <AlertCircle className="h-3 w-3 mr-1" />
                  )}
                  Backend{" "}
                  {validationResult.backendHealthy ? "Healthy" : "Unhealthy"}
                </Badge>
              </div>
              <div className="flex items-center gap-2">
                <Badge variant="outline">
                  Port: {validationResult.portDetected || "Not detected"}
                </Badge>
              </div>
            </div>

            {/* System Information */}
            {validationResult.systemInfo && (
              <div className="space-y-2">
                <h4 className="font-medium">System Information</h4>
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div>
                    API Version: {validationResult.systemInfo.apiVersion}
                  </div>
                  <div>Service: {validationResult.systemInfo.service}</div>
                  <div>
                    CORS:{" "}
                    {validationResult.systemInfo.corsEnabled
                      ? "Enabled"
                      : "Disabled"}
                  </div>
                  <div>
                    WebSocket:{" "}
                    {validationResult.systemInfo.websocketAvailable
                      ? "Available"
                      : "Unavailable"}
                  </div>
                </div>
              </div>
            )}

            {/* Issues */}
            {validationResult.issues.length > 0 && (
              <div className="space-y-2">
                <h4 className="font-medium text-destructive">Issues Found</h4>
                <ul className="space-y-1">
                  {validationResult.issues.map((issue, index) => (
                    <li key={index} className="flex items-start gap-2 text-sm">
                      <AlertCircle className="h-4 w-4 text-destructive mt-0.5 flex-shrink-0" />
                      {issue}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Suggestions */}
            {validationResult.suggestions.length > 0 && (
              <div className="space-y-2">
                <h4 className="font-medium text-primary">
                  Suggested Solutions
                </h4>
                <ul className="space-y-1">
                  {validationResult.suggestions.map((suggestion, index) => (
                    <li key={index} className="flex items-start gap-2 text-sm">
                      <CheckCircle className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                      {suggestion}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Actions */}
            <div className="flex gap-2">
              <Button onClick={retryValidation} className="flex-1">
                <RefreshCw className="h-4 w-4 mr-2" />
                Retry Validation
              </Button>
              <Button
                variant="outline"
                onClick={() => {
                  // Continue anyway - some features may not work
                  window.location.hash = "#continue-anyway";
                  window.location.reload();
                }}
              >
                Continue Anyway
              </Button>
            </div>

            <Alert>
              <Wifi className="h-4 w-4" />
              <AlertTitle>Development Note</AlertTitle>
              <AlertDescription>
                If you're in development mode, ensure the backend server is
                running on the expected port. Response time:{" "}
                {validationResult.responseTime}ms
              </AlertDescription>
            </Alert>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Validation passed - render the main application
  return (
    <>
      {children}
      {/* Show a subtle indicator that validation passed */}
      {validationResult && (
        <div className="fixed bottom-4 right-4 z-50">
          <Badge
            variant="outline"
            className="bg-background/80 backdrop-blur-sm"
          >
            <CheckCircle className="h-3 w-3 mr-1 text-green-500" />
            Backend: Port {validationResult.portDetected} (
            {validationResult.responseTime}ms)
          </Badge>
        </div>
      )}
    </>
  );
}

export default StartupValidator;
