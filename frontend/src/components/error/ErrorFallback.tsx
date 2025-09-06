import React from "react";
import { AlertCircle, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";

interface ErrorFallbackProps {
  error?: Error;
  resetError?: () => void;
  title?: string;
  message?: string;
}

const ErrorFallback: React.FC<ErrorFallbackProps> = ({
  error,
  resetError,
  title = "Something went wrong",
  message = "An error occurred while loading this component.",
}) => {
  return (
    <div className="flex flex-col items-center justify-center p-8 text-center bg-card border border-border rounded-lg">
      <AlertCircle className="h-12 w-12 text-destructive mb-4" />

      <h2 className="text-lg font-semibold text-foreground mb-2">{title}</h2>

      <p className="text-muted-foreground mb-6 max-w-md">{message}</p>

      {resetError && (
        <Button onClick={resetError} variant="outline">
          <RefreshCw className="h-4 w-4 mr-2" />
          Try Again
        </Button>
      )}

      {process.env.NODE_ENV === "development" && error && (
        <details className="mt-6 w-full text-left">
          <summary className="cursor-pointer text-sm text-muted-foreground hover:text-foreground">
            Error Details (Development)
          </summary>
          <pre className="mt-2 text-xs bg-muted p-3 rounded overflow-auto max-h-32 whitespace-pre-wrap">
            {error.toString()}
          </pre>
        </details>
      )}
    </div>
  );
};

export default ErrorFallback;
