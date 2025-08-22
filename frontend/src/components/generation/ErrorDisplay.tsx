import React from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { AlertTriangle, RefreshCw } from "lucide-react";
import { ApiError } from "@/lib/api-client";

interface ErrorDisplayProps {
  error: string | ApiError | Error | any | null;
  onRetry?: () => void;
  className?: string;
}

export const ErrorDisplay: React.FC<ErrorDisplayProps> = ({
  error,
  onRetry,
  className,
}) => {
  if (!error) return null;

  // Convert any error to a safe string message
  let errorMessage: string = "An error occurred";
  let errorTitle: string = "Error";
  let canRetry = true;

  // Helper function to safely convert any value to string
  const safeStringify = (value: any): string => {
    if (typeof value === "string") return value;
    if (value === null || value === undefined) return "";
    if (typeof value === "object") {
      try {
        return JSON.stringify(value);
      } catch {
        return String(value);
      }
    }
    return String(value);
  };

  try {
    if (typeof error === "string") {
      errorMessage = error;
    } else if (error instanceof ApiError) {
      // Handle ApiError instances specifically
      errorMessage = safeStringify(error.message || "API error occurred");
      errorTitle = "API Error";

      // Set retry capability based on status code
      if (error.status === 422) {
        errorTitle = "Validation Error";
        canRetry = false;
      } else if (error.status >= 500) {
        errorTitle = "Server Error";
        canRetry = true;
      } else if (error.status === 0) {
        errorTitle = "Network Error";
        canRetry = true;
      }

      // Try to extract more detailed error message from details if available
      if (error.details) {
        if (Array.isArray(error.details) && error.details.length > 0) {
          const firstDetail = error.details[0];
          if (
            firstDetail &&
            typeof firstDetail === "object" &&
            "msg" in firstDetail
          ) {
            const location = Array.isArray(firstDetail.loc)
              ? firstDetail.loc.join(".")
              : String(firstDetail.loc || "");
            errorMessage = `${safeStringify(firstDetail.msg)}${
              location ? ` (field: ${location})` : ""
            }`;
          }
        } else if (
          typeof error.details === "object" &&
          error.details !== null
        ) {
          if ("detail" in error.details) {
            if (Array.isArray(error.details.detail)) {
              const firstDetail = error.details.detail[0];
              if (
                firstDetail &&
                typeof firstDetail === "object" &&
                "msg" in firstDetail
              ) {
                const location = Array.isArray(firstDetail.loc)
                  ? firstDetail.loc.join(".")
                  : String(firstDetail.loc || "");
                errorMessage = `${safeStringify(firstDetail.msg)}${
                  location ? ` (field: ${location})` : ""
                }`;
              }
            } else if (typeof error.details.detail === "string") {
              errorMessage = safeStringify(error.details.detail);
            }
          } else if (
            "msg" in error.details &&
            "type" in error.details &&
            "loc" in error.details
          ) {
            // Direct Pydantic error object in details
            const location = Array.isArray(error.details.loc)
              ? error.details.loc.join(".")
              : String(error.details.loc);
            errorMessage = `${safeStringify(error.details.msg)}${
              location ? ` (field: ${location})` : ""
            }`;
            errorTitle = "Validation Error";
            canRetry = false;
          }
        }
      }
    } else if (typeof error === "object" && error !== null) {
      // Handle Pydantic validation errors (array of error objects)
      if (Array.isArray(error)) {
        const firstError = error[0];
        if (
          firstError &&
          typeof firstError === "object" &&
          "msg" in firstError
        ) {
          errorMessage = safeStringify(firstError.msg);
          errorTitle = "Validation Error";
          canRetry = false;
        } else {
          errorMessage = "Multiple validation errors occurred";
          errorTitle = "Validation Error";
          canRetry = false;
        }
      }
      // Handle single validation error object with Pydantic structure
      else if ("msg" in error && "type" in error && "loc" in error) {
        // This is a Pydantic validation error object
        const location = Array.isArray(error.loc)
          ? error.loc.join(".")
          : String(error.loc);
        errorMessage = `${safeStringify(error.msg)}${
          location ? ` (field: ${location})` : ""
        }`;
        errorTitle = "Validation Error";
        canRetry = false;
      }
      // Handle single validation error object (general)
      else if ("msg" in error) {
        errorMessage = safeStringify(error.msg);
        errorTitle = "Validation Error";
        canRetry = false;
      }
      // Handle standard error objects
      else if ("message" in error) {
        errorMessage = safeStringify(error.message);
      }
      // Handle Error instances
      else if (error instanceof Error) {
        errorMessage = safeStringify(error.message || "Error occurred");
      }
      // Handle objects with detail property (FastAPI errors)
      else if ("detail" in error) {
        if (typeof error.detail === "string") {
          errorMessage = safeStringify(error.detail);
        } else if (Array.isArray(error.detail)) {
          const firstDetail = error.detail[0];
          if (
            firstDetail &&
            typeof firstDetail === "object" &&
            "msg" in firstDetail
          ) {
            errorMessage = safeStringify(firstDetail.msg);
            errorTitle = "Validation Error";
            canRetry = false;
          } else {
            errorMessage = "Validation error occurred";
            errorTitle = "Validation Error";
            canRetry = false;
          }
        } else if (typeof error.detail === "object" && error.detail !== null) {
          // Handle cases where detail is an object - convert to string safely
          errorMessage = safeStringify(error.detail);
        } else {
          errorMessage = "Server error occurred";
        }
      }
      // Fallback for unknown object types - safely stringify
      else {
        try {
          // Try to extract any readable message from the object
          if (
            error.toString &&
            typeof error.toString === "function" &&
            error.toString() !== "[object Object]"
          ) {
            errorMessage = error.toString();
          } else {
            errorMessage = safeStringify(error);
          }
        } catch {
          errorMessage = "Invalid input provided";
        }
        errorTitle = "Validation Error";
        canRetry = false;
      }
    } else {
      // Handle non-object, non-string errors
      errorMessage = safeStringify(error);
    }
  } catch (e) {
    errorMessage = "An unexpected error occurred";
  }

  // Final safety check - ensure errorMessage and errorTitle are always strings
  if (typeof errorMessage !== "string") {
    errorMessage = "An error occurred (unable to parse error message)";
  }
  if (typeof errorTitle !== "string") {
    errorTitle = "Error";
  }

  return (
    <Alert variant="destructive" className={className}>
      <AlertTriangle className="h-4 w-4" />
      <AlertTitle className="flex items-center justify-between">
        {errorTitle}
        {canRetry && onRetry && (
          <Button
            variant="outline"
            size="sm"
            onClick={onRetry}
            className="ml-2 h-6 px-2 text-xs"
          >
            <RefreshCw className="h-3 w-3 mr-1" />
            Retry
          </Button>
        )}
      </AlertTitle>
      <AlertDescription className="mt-2">
        <p className="mb-3">{errorMessage}</p>

        <div>
          <p className="font-medium mb-2">Suggestions:</p>
          <ul className="list-disc list-inside space-y-1 text-sm">
            <li>Check that all required fields are filled</li>
            <li>Verify your input format is correct</li>
            <li>Try refreshing the page if the issue persists</li>
          </ul>
        </div>
      </AlertDescription>
    </Alert>
  );
};

export default ErrorDisplay;
