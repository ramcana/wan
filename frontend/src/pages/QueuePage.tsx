import React from "react";
import QueueManager from "@/components/queue/QueueManager";

const QueuePage: React.FC = () => {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-foreground">Generation Queue</h1>
        <p className="text-muted-foreground">
          Monitor and manage your video generation tasks
        </p>
      </div>

      <QueueManager />
    </div>
  );
};

export default QueuePage;
