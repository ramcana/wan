import React, { useState, useCallback, useRef } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Upload,
  X,
  Image as ImageIcon,
  FileImage,
  AlertCircle,
  CheckCircle2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { validateImageUpload, type ImageUpload } from "@/lib/api-schemas";

interface ImageUploadProps {
  onImageSelect: (file: File | null) => void;
  selectedImage: File | null;
  error?: string | null;
  disabled?: boolean;
  className?: string;
}

interface ImagePreviewProps {
  file: File;
  onRemove: () => void;
}

const ImagePreview: React.FC<ImagePreviewProps> = ({ file, onRemove }) => {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [imageError, setImageError] = useState(false);

  React.useEffect(() => {
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  return (
    <Card className="relative">
      <CardContent className="p-4">
        <div className="flex items-start gap-4">
          <div className="relative flex-shrink-0">
            {previewUrl && !imageError ? (
              <img
                src={previewUrl}
                alt="Preview"
                className="w-20 h-20 object-cover rounded-md border"
                onError={() => setImageError(true)}
              />
            ) : (
              <div className="w-20 h-20 bg-muted rounded-md border flex items-center justify-center">
                <FileImage className="h-8 w-8 text-muted-foreground" />
              </div>
            )}
            <Button
              variant="destructive"
              size="sm"
              className="absolute -top-2 -right-2 h-6 w-6 rounded-full p-0"
              onClick={onRemove}
            >
              <X className="h-3 w-3" />
            </Button>
          </div>

          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-2">
              <p className="font-medium text-sm truncate">{file.name}</p>
              <Badge variant="secondary" className="text-xs">
                {file.type.split("/")[1].toUpperCase()}
              </Badge>
            </div>

            <div className="space-y-1 text-xs text-muted-foreground">
              <p>Size: {formatFileSize(file.size)}</p>
              {previewUrl && !imageError && <p>Ready for generation</p>}
            </div>

            {!imageError && (
              <div className="flex items-center gap-1 mt-2">
                <CheckCircle2 className="h-3 w-3 text-green-600" />
                <span className="text-xs text-green-600">Valid image</span>
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export const ImageUpload: React.FC<ImageUploadProps> = ({
  onImageSelect,
  selectedImage,
  error,
  disabled = false,
  className,
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateFile = useCallback((file: File): string | null => {
    try {
      validateImageUpload(file);
      return null;
    } catch (err) {
      if (err instanceof Error) {
        return err.message;
      }
      return "Invalid file";
    }
  }, []);

  const handleFileSelect = useCallback(
    (file: File) => {
      const validationErr = validateFile(file);
      if (validationErr) {
        setValidationError(validationErr);
        onImageSelect(null);
        return;
      }

      setValidationError(null);
      onImageSelect(file);
    },
    [validateFile, onImageSelect]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);

      if (disabled) return;

      const files = Array.from(e.dataTransfer.files);
      if (files.length > 0) {
        handleFileSelect(files[0]);
      }
    },
    [disabled, handleFileSelect]
  );

  const handleDragOver = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      if (!disabled) {
        setIsDragOver(true);
      }
    },
    [disabled]
  );

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleFileInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (files && files.length > 0) {
        handleFileSelect(files[0]);
      }
    },
    [handleFileSelect]
  );

  const handleBrowseClick = () => {
    fileInputRef.current?.click();
  };

  const handleRemoveImage = () => {
    setValidationError(null);
    onImageSelect(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const displayError = error || validationError;

  return (
    <div className={cn("space-y-4", className)}>
      {selectedImage ? (
        <ImagePreview file={selectedImage} onRemove={handleRemoveImage} />
      ) : (
        <Card
          className={cn(
            "border-2 border-dashed transition-colors cursor-pointer",
            isDragOver && !disabled && "border-primary bg-primary/5",
            disabled && "opacity-50 cursor-not-allowed",
            displayError && "border-destructive"
          )}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={!disabled ? handleBrowseClick : undefined}
        >
          <CardContent className="p-8">
            <div className="flex flex-col items-center justify-center text-center space-y-4">
              <div
                className={cn(
                  "rounded-full p-4 transition-colors",
                  isDragOver && !disabled ? "bg-primary/10" : "bg-muted"
                )}
              >
                {isDragOver && !disabled ? (
                  <Upload className="h-8 w-8 text-primary" />
                ) : (
                  <ImageIcon className="h-8 w-8 text-muted-foreground" />
                )}
              </div>

              <div className="space-y-2">
                <p className="text-lg font-medium">
                  {isDragOver && !disabled
                    ? "Drop image here"
                    : "Upload an image"}
                </p>
                <p className="text-sm text-muted-foreground">
                  Drag and drop an image file, or click to browse
                </p>
                <div className="flex flex-wrap justify-center gap-2 text-xs text-muted-foreground">
                  <Badge variant="outline">JPEG</Badge>
                  <Badge variant="outline">PNG</Badge>
                  <Badge variant="outline">WebP</Badge>
                  <span>•</span>
                  <span>Max 10MB</span>
                </div>
              </div>

              {!disabled && (
                <Button variant="outline" size="sm" type="button">
                  <Upload className="h-4 w-4 mr-2" />
                  Browse Files
                </Button>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {displayError && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{displayError}</AlertDescription>
        </Alert>
      )}

      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        onChange={handleFileInputChange}
        className="hidden"
        disabled={disabled}
      />
    </div>
  );
};

export default ImageUpload;
