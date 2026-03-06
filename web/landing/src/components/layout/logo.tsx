import Link from "next/link";
import { cn } from "@/lib/utils";

export function Logo({ className }: { className?: string }) {
  return (
    <Link href="/" className={cn("flex items-center gap-2 group", className)}>
      <div className="w-8 h-8 rounded-lg bg-gradient-to-bl from-dana-400 to-dana-600 flex items-center justify-center shadow-lg shadow-dana-500/20 group-hover:shadow-dana-500/40 transition-shadow">
        <span className="text-white font-bold text-sm">د</span>
      </div>
      <span className="text-xl font-bold text-gray-900 dark:text-white">دانا</span>
    </Link>
  );
}
