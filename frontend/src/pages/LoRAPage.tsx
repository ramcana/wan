import React from "react";
import { LoRABrowser } from "../components/lora";
import { Container } from "../components/ui/container";

export function LoRAPage() {
  return (
    <Container className="py-6">
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">LoRA Management</h1>
          <p className="text-gray-600 mt-2">
            Manage your LoRA files, upload new ones, and preview their effects
            on your generations.
          </p>
        </div>

        <LoRABrowser />
      </div>
    </Container>
  );
}
