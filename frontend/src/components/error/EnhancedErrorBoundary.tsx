import React, { Component, ReactNode } from 'react';
import { AlertTriangle, RefreshCw, Home, Bug } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ConnectionStatus } from '@/components/ui/connection-status';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  showConnectionStatus?: boolean;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: any;
  errorId: string;
}

export class EnhancedErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      errorId: '',
    };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return {
      hasError: true,
      error,
      errorId: `error_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    };
  }

  componentDidCatch(error: Error, errorInfo: any) {
    this.setState({
      error,
      errorInfo,
    });

    // Log error for debugging
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    
    // Report error to monitoring service if available
    if (window.errorReporter) {
      window.errorReporter.reportError(error, {
        errorBoundary: true,
        componentStack: errorInfo.componentStack,
        errorId: this.state.errorId,
      });
    }
  }

  handleRetry = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      errorId: '',
    });
  };

  handleReload = () => {
    window.location.reload();
  };

  handleGoHome = () => {
    window.location.href = '/';
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      const isApiError = this.state.error?.message?.includes('API') || 
                        this.state.error?.message?.includes('fetch') ||
                        this.state.error?.message?.includes('network');

      return (
        <div className="min-h-screen bg-background flex items-center justify-center p-4">
          <Card className="w-full max-w-2xl">
            <CardHeader>
              <div className="flex items-center gap-2">
                <AlertTriangle className="h-6 w-6 text-destructive" />
                <CardTitle>Something went wrong</CardTitle>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Connection Status */}
              {(this.props.showConnectionStatus || isApiError) && (
                <div>
                  <h4 className="text-sm font-medium mb-2">Connection Status</h4>
                  <ConnectionStatus showDetails={true} />
                </div>
              )}

              {/* Error Details */}
              <Alert variant="destructive">
                <Bug className="h-4 w-4" />
                <AlertTitle>Error Details</AlertTitle>
                <AlertDescription>
                  <div className="space-y-2">
                    <p>{this.state.error?.message || 'An unexpected error occurred'}</p>
                    <details className="text-xs">
                      <summary className="cursor-pointer hover:underline">
                        Technical Details (Error ID: {this.state.errorId})
                      </summary>
                      <pre className="mt-2 p-2 bg-muted rounded text-xs overflow-auto">
                        {this.state.error?.stack}
                      </pre>
                    </details>
                  </div>
                </AlertDescription>
              </Alert>

              {/* Recommendations */}
              <Alert>
                <AlertTitle>Recommended Actions</AlertTitle>
                <AlertDescription>
                  <ul className="list-disc list-inside space-y-1 text-sm">
                    {isApiError && (
                      <li>Check if the backend server is running on port 9000</li>
                    )}
                    <li>Try refreshing the page</li>
                    <li>Clear browser cache and cookies</li>
                    <li>Check browser console for additional errors</li>
                    {this.state.errorId && (
                      <li>Report this error with ID: {this.state.errorId}</li>
                    )}
                  </ul>
                </AlertDescription>
              </Alert>

              {/* Action Buttons */}
              <div className="flex flex-wrap gap-2">
                <Button onClick={this.handleRetry} variant="default">
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Try Again
                </Button>
                <Button onClick={this.handleReload} variant="outline">
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Reload Page
                </Button>
                <Button onClick={this.handleGoHome} variant="outline">
                  <Home className="h-4 w-4 mr-2" />
                  Go Home
                </Button>
              </div>

              {/* Development Info */}
              {import.meta.env.VITE_DEV_MODE === 'true' && (
                <details className="text-xs text-muted-foreground">
                  <summary className="cursor-pointer hover:underline">
                    Development Information
                  </summary>
                  <pre className="mt-2 p-2 bg-muted rounded overflow-auto">
                    {JSON.stringify({
                      error: this.state.error?.message,
                      stack: this.state.error?.stack,
                      componentStack: this.state.errorInfo?.componentStack,
                      timestamp: new Date().toISOString(),
                      userAgent: navigator.userAgent,
                      url: window.location.href,
                    }, null, 2)}
                  </pre>
                </details>
              )}
            </CardContent>
          </Card>
        </div>
      );
    }

    return this.props.children;
  }
}

// Declare global error reporter interface
declare global {
  interface Window {
    errorReporter?: {
      reportError: (error: Error, context?: any) => void;
    };
  }
}

export default EnhancedErrorBoundary;
