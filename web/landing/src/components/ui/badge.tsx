import { cn } from "@/lib/utils";
import { type HTMLAttributes } from "react";

interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  variant?: "default" | "success" | "warning" | "danger" | "outline";
}

export function Badge({ className, variant = "default", ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium",
        {
          "bg-dana-100 dark:bg-dana-900/50 text-dana-700 dark:text-dana-300": variant === "default",
          "bg-emerald-100 dark:bg-emerald-900/50 text-emerald-700 dark:text-emerald-300": variant === "success",
          "bg-amber-100 dark:bg-amber-900/50 text-amber-700 dark:text-amber-300": variant === "warning",
          "bg-red-100 dark:bg-red-900/50 text-red-700 dark:text-red-300": variant === "danger",
          "border border-gray-300 dark:border-gray-600 text-gray-600 dark:text-gray-400": variant === "outline",
        },
        className
      )}
      {...props}
    />
  );
}
