import React from "react";
import { cn } from "@/lib/utils";
import {
  useBreakpoint,
  getContainerMaxWidth,
  getResponsivePadding,
} from "@/lib/responsive";

interface ContainerProps extends React.HTMLAttributes<HTMLDivElement> {
  size?: "sm" | "md" | "lg" | "xl" | "full";
  padding?: boolean;
  center?: boolean;
}

const Container: React.FC<ContainerProps> = ({
  children,
  className,
  size,
  padding = true,
  center = true,
  ...props
}) => {
  const breakpoint = useBreakpoint();

  const getSizeClass = () => {
    if (size === "full") return "w-full";
    if (size) {
      const sizeMap = {
        sm: "max-w-sm",
        md: "max-w-md",
        lg: "max-w-4xl",
        xl: "max-w-6xl",
      };
      return sizeMap[size];
    }
    return getContainerMaxWidth(breakpoint);
  };

  return (
    <div
      className={cn(
        "w-full",
        getSizeClass(),
        center && "mx-auto",
        padding && getResponsivePadding(breakpoint),
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
};

export { Container };
