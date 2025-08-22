import React, { useState, useCallback } from "react";
import { Upload, X, AlertCircle, CheckCircle, FileText } from "lucide-react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "../ui/dialog";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { Progress } from "../ui/progress";
import { useLoRAUpload } from "../../hooks/api/use-lora";
import { validateLoRAUpload } from "../../lib/api-schemas";

interface LoRAUploadDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function LoRAUploadDialog({
  open,
  onOpenChange,
}: LoRAUploadDialogProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [customName, setCustomName] = useState("");
  const [uploadProgress, setUploadProgress] = useState(0);
  const [dragActive, setDragActive] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);

  const uploadLoRA = useLoRAUpload();

  const resetForm = () => {
    setSelectedFile(null);
    setCustomName("");
    setUploadProgress(0);
    setValidationError(null);
  };

  const handleClose = () => {
    if (!uploadLoRA.isLoading) {
      resetForm();
      onOpenChange(false);
    }
  };

  const validateFile = (file: File): string | null => {
    try {
      validateLoRAUpload({ file, name: customName });
      return null;
    } catch (error: any) {
      if (error.errors && error.errors.length > 0) {
        return error.errors[0].message;
      }
      return error.message || "Invalid file";
    }
  };

  const handleFileSelect = (file: File) => {
    const error = validateFile(file);
    if (error) {
      setValidationError(error);
      setSelectedFile(null);
    } else {
      setValidationError(null);
      setSelectedFile(file);

      // Auto-generate name from filename if not set
      if (!customName) {
        const nameWithoutExt = file.name.replace(/\.[^/.]+$/, "");
        setCustomName(nameWithoutExt);
      }
    }
  };

  const handleFileInputChange = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = event.target.files?.[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const files = e.dataTransfer.files;
    if (files && files[0]) {
      handleFileSelect(files[0]);
    }
  }, []);

  const handleUpload = async () => {
    if (!selectedFile) return;

    try {
      await uploadLoRA.mutateAsync({
        file: selectedFile,
        name: customName.trim() || undefined,
        onProgress: setUploadProgress,
      });

      // Success - close dialog
      handleClose();
    } catch (error) {
      // Error handling is done in the hook
      setUploadProgress(0);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  const getSupportedFormats = () => [".safetensors", ".pt", ".pth", ".bin"];

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center">
            <Upload className="w-5 h-5 mr-2" />
            Upload LoRA File
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-4">
          {/* File Upload Area */}
          <div
            className={`relative border-2 border-dashed rounded-lg p-6 text-center transition-colors ${
              dragActive
                ? "border-blue-500 bg-blue-50"
                : validationError
                ? "border-red-300 bg-red-50"
                : "border-gray-300 hover:border-gray-400"
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <input
              type="file"
              accept={getSupportedFormats().join(",")}
              onChange={handleFileInputChange}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              disabled={uploadLoRA.isLoading}
            />

            {selectedFile ? (
              <div className="space-y-2">
                <FileText className="w-8 h-8 mx-auto text-green-500" />
                <div>
                  <p className="font-medium text-sm">{selectedFile.name}</p>
                  <p className="text-xs text-gray-500">
                    {formatFileSize(selectedFile.size)}
                  </p>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={(e) => {
                    e.stopPropagation();
                    setSelectedFile(null);
                    setValidationError(null);
                  }}
                  disabled={uploadLoRA.isLoading}
                >
                  <X className="w-4 h-4 mr-1" />
                  Remove
                </Button>
              </div>
            ) : (
              <div className="space-y-2">
                <Upload className="w-8 h-8 mx-auto text-gray-400" />
                <div>
                  <p className="text-sm font-medium">
                    Drop your LoRA file here, or click to browse
                  </p>
                  <p className="text-xs text-gray-500">
                    Supported formats: {getSupportedFormats().join(", ")}
                  </p>
                  <p className="text-xs text-gray-500">Maximum size: 500MB</p>
                </div>
              </div>
            )}
          </div>

          {/* Validation Error */}
          {validationError && (
            <div className="flex items-center space-x-2 text-sm text-red-600 bg-red-50 p-3 rounded-md">
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              <span>{validationError}</span>
            </div>
          )}

          {/* Custom Name Input */}
          <div className="space-y-2">
            <Label htmlFor="lora-name">LoRA Name (optional)</Label>
            <Input
              id="lora-name"
              placeholder="Enter custom name or leave empty to use filename"
              value={customName}
              onChange={(e) => setCustomName(e.target.value)}
              disabled={uploadLoRA.isLoading}
            />
            <p className="text-xs text-gray-500">
              If empty, the filename will be used as the LoRA name
            </p>
          </div>

          {/* Upload Progress */}
          {uploadLoRA.isLoading && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span>Uploading...</span>
                <span>{uploadProgress}%</span>
              </div>
              <Progress value={uploadProgress} className="h-2" />
            </div>
          )}

          {/* Success Message */}
          {uploadLoRA.isSuccess && (
            <div className="flex items-center space-x-2 text-sm text-green-600 bg-green-50 p-3 rounded-md">
              <CheckCircle className="w-4 h-4 flex-shrink-0" />
              <span>LoRA uploaded successfully!</span>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex justify-end space-x-2 pt-4 border-t">
            <Button
              variant="outline"
              onClick={handleClose}
              disabled={uploadLoRA.isLoading}
            >
              Cancel
            </Button>
            <Button
              onClick={handleUpload}
              disabled={
                !selectedFile || !!validationError || uploadLoRA.isLoading
              }
            >
              {uploadLoRA.isLoading ? (
                <>
                  <Upload className="w-4 h-4 mr-2 animate-pulse" />
                  Uploading...
                </>
              ) : (
                <>
                  <Upload className="w-4 h-4 mr-2" />
                  Upload LoRA
                </>
              )}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
