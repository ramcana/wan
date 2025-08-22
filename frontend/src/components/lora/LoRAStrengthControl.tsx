import React, { useState, useEffect } from "react";
import { Slider } from "../ui/slider";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
import { RotateCcw, AlertCircle } from "lucide-react";

interface LoRAStrengthControlProps {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
  showPresets?: boolean;
  label?: string;
  description?: string;
}

export function LoRAStrengthControl({
  value,
  onChange,
  min = 0,
  max = 2,
  step = 0.1,
  disabled = false,
  showPresets = true,
  label = "LoRA Strength",
  description = "Controls how strongly the LoRA affects the generation",
}: LoRAStrengthControlProps) {
  const [inputValue, setInputValue] = useState(value.toString());
  const [isInputFocused, setIsInputFocused] = useState(false);

  // Update input value when prop value changes (unless input is focused)
  useEffect(() => {
    if (!isInputFocused) {
      setInputValue(value.toString());
    }
  }, [value, isInputFocused]);

  const handleSliderChange = (newValue: number[]) => {
    const val = newValue[0];
    onChange(val);
    if (!isInputFocused) {
      setInputValue(val.toString());
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
  };

  const handleInputBlur = () => {
    setIsInputFocused(false);
    const numValue = parseFloat(inputValue);

    if (isNaN(numValue)) {
      setInputValue(value.toString());
    } else {
      const clampedValue = Math.max(min, Math.min(max, numValue));
      onChange(clampedValue);
      setInputValue(clampedValue.toString());
    }
  };

  const handleInputFocus = () => {
    setIsInputFocused(true);
  };

  const handleInputKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleInputBlur();
    }
  };

  const handlePresetClick = (presetValue: number) => {
    onChange(presetValue);
    setInputValue(presetValue.toString());
  };

  const handleReset = () => {
    onChange(1.0);
    setInputValue("1.0");
  };

  const getStrengthDescription = (strength: number) => {
    if (strength === 0) return "Disabled";
    if (strength < 0.5) return "Subtle effect";
    if (strength < 1.0) return "Moderate effect";
    if (strength === 1.0) return "Standard strength";
    if (strength < 1.5) return "Strong effect";
    if (strength < 2.0) return "Very strong effect";
    return "Maximum strength";
  };

  const getStrengthColor = (strength: number) => {
    if (strength === 0) return "text-gray-500";
    if (strength < 0.5) return "text-blue-500";
    if (strength < 1.0) return "text-green-500";
    if (strength === 1.0) return "text-blue-600";
    if (strength < 1.5) return "text-orange-500";
    return "text-red-500";
  };

  const presets = [
    { label: "Off", value: 0 },
    { label: "Subtle", value: 0.3 },
    { label: "Light", value: 0.6 },
    { label: "Standard", value: 1.0 },
    { label: "Strong", value: 1.3 },
    { label: "Max", value: 2.0 },
  ];

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <Label className="text-sm font-medium">{label}</Label>
          {description && (
            <p className="text-xs text-gray-500 mt-1">{description}</p>
          )}
        </div>
        <div className="flex items-center space-x-2">
          <Badge className={`text-xs ${getStrengthColor(value)}`}>
            {getStrengthDescription(value)}
          </Badge>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleReset}
            disabled={disabled || value === 1.0}
            className="h-6 px-2"
          >
            <RotateCcw className="w-3 h-3" />
          </Button>
        </div>
      </div>

      {/* Slider and Input */}
      <div className="space-y-3">
        <div className="flex items-center space-x-4">
          <div className="flex-1">
            <Slider
              value={[value]}
              onValueChange={handleSliderChange}
              min={min}
              max={max}
              step={step}
              disabled={disabled}
              className="w-full"
            />
          </div>
          <div className="w-20">
            <Input
              type="number"
              value={inputValue}
              onChange={handleInputChange}
              onBlur={handleInputBlur}
              onFocus={handleInputFocus}
              onKeyPress={handleInputKeyPress}
              min={min}
              max={max}
              step={step}
              disabled={disabled}
              className="text-center text-sm h-8"
            />
          </div>
        </div>

        {/* Slider Labels */}
        <div className="flex justify-between text-xs text-gray-500 px-1">
          <span>{min}</span>
          <span>1.0</span>
          <span>{max}</span>
        </div>
      </div>

      {/* Presets */}
      {showPresets && (
        <div className="space-y-2">
          <Label className="text-xs font-medium text-gray-600">
            Quick Presets
          </Label>
          <div className="flex flex-wrap gap-1">
            {presets.map((preset) => (
              <Button
                key={preset.value}
                variant={value === preset.value ? "default" : "outline"}
                size="sm"
                onClick={() => handlePresetClick(preset.value)}
                disabled={disabled}
                className="h-7 px-3 text-xs"
              >
                {preset.label}
              </Button>
            ))}
          </div>
        </div>
      )}

      {/* Warning for extreme values */}
      {value > 1.5 && (
        <div className="flex items-start space-x-2 p-2 bg-amber-50 border border-amber-200 rounded-md">
          <AlertCircle className="w-4 h-4 text-amber-600 flex-shrink-0 mt-0.5" />
          <div className="text-xs text-amber-800">
            <p className="font-medium">High strength warning</p>
            <p>Values above 1.5 may produce unexpected or distorted results.</p>
          </div>
        </div>
      )}
    </div>
  );
}
