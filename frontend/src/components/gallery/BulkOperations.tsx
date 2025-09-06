import React from "react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Download, Trash2, Share2, CheckSquare, Square, X } from "lucide-react";

interface BulkOperationsProps {
  selectedCount: number;
  onDelete: () => void;
  onDownload: () => void;
  onShare: () => void;
  onSelectAll: () => void;
  onClearSelection: () => void;
}

const BulkOperations: React.FC<BulkOperationsProps> = ({
  selectedCount,
  onDelete,
  onDownload,
  onShare,
  onSelectAll,
  onClearSelection,
}) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.2 }}
    >
      <Card className="border-primary/20 bg-primary/5">
        <CardContent className="p-4">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium">
                {selectedCount} item{selectedCount !== 1 ? "s" : ""} selected
              </span>
            </div>

            <div className="flex items-center gap-2">
              {/* Select All */}
              <Button
                variant="outline"
                size="sm"
                onClick={onSelectAll}
                className="flex items-center gap-2"
              >
                <CheckSquare className="h-4 w-4" />
                Select All
              </Button>

              {/* Download */}
              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Button
                  variant="outline"
                  size="sm"
                  onClick={onDownload}
                  className="flex items-center gap-2"
                >
                  <Download className="h-4 w-4" />
                  Download
                </Button>
              </motion.div>

              {/* Share */}
              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Button
                  variant="outline"
                  size="sm"
                  onClick={onShare}
                  className="flex items-center gap-2"
                >
                  <Share2 className="h-4 w-4" />
                  Share
                </Button>
              </motion.div>

              {/* Delete */}
              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={onDelete}
                  className="flex items-center gap-2"
                >
                  <Trash2 className="h-4 w-4" />
                  Delete
                </Button>
              </motion.div>

              {/* Clear Selection */}
              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={onClearSelection}
                  className="flex items-center gap-2"
                >
                  <X className="h-4 w-4" />
                </Button>
              </motion.div>
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
};

export default BulkOperations;
