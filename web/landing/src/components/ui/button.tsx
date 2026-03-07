import { cn } from "@/lib/utils";
import { type ButtonHTMLAttributes, forwardRef } from "react";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "outline" | "ghost" | "danger";
  size?: "sm" | "md" | "lg";
}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "primary", size = "md", ...props }, ref) => {
    return (
      <button
        ref={ref}
        className={cn(
          "inline-flex items-center justify-center font-medium rounded-xl transition-all duration-200 disabled:opacity-50 disabled:pointer-events-none cursor-pointer",
          {
            "bg-gradient-to-l from-dana-600 to-dana-500 text-white hover:from-dana-700 hover:to-dana-600 shadow-lg shadow-dana-500/25 hover:shadow-xl hover:shadow-dana-500/30":
              variant === "primary",
            "bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-100 hover:bg-gray-200 dark:hover:bg-gray-700":
              variant === "secondary",
            "border-2 border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-300 hover:border-dana-500 hover:text-dana-600 dark:hover:border-dana-400 dark:hover:text-dana-400 bg-transparent":
              variant === "outline",
            "text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-800 bg-transparent":
              variant === "ghost",
            "bg-red-600 text-white hover:bg-red-700": variant === "danger",
          },
          {
            "px-3 py-1.5 text-sm gap-1.5": size === "sm",
            "px-5 py-2.5 text-sm gap-2": size === "md",
            "px-7 py-3.5 text-base gap-2.5": size === "lg",
          },
          className
        )}
        {...props}
      />
    );
  }
);

Button.displayName = "Button";
export { Button };
