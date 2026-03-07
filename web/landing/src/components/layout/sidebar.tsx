"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Logo } from "./logo";
import { ThemeToggle } from "./theme-toggle";
import { cn } from "@/lib/utils";
import { useAuth } from "@/hooks/use-auth";
import { LayoutDashboard, Key, BarChart3, CreditCard, LogOut, FlaskConical } from "lucide-react";

const navItems = [
  { href: "/dashboard", label: "داشبورد", icon: LayoutDashboard },
  { href: "/dashboard/usage", label: "مصرف", icon: BarChart3 },
  { href: "/dashboard/keys", label: "کلیدها", icon: Key },
  { href: "/dashboard/billing", label: "صورتحساب", icon: CreditCard },
];

export function Sidebar() {
  const pathname = usePathname();
  const { user, logout } = useAuth();

  const initial = user?.email?.[0]?.toUpperCase() || "ک";

  return (
    <>
      {/* Desktop sidebar */}
      <aside className="hidden md:flex flex-col w-64 border-l border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-950 h-screen sticky top-0">
        <div className="p-5 border-b border-gray-200 dark:border-gray-800">
          <Logo />
        </div>
        <nav className="flex-1 p-3 space-y-1">
          {navItems.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm transition-colors",
                  isActive
                    ? "bg-dana-50 dark:bg-dana-950/50 text-dana-700 dark:text-dana-300 font-medium"
                    : "text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-900 hover:text-gray-900 dark:hover:text-gray-100"
                )}
              >
                <item.icon className="w-4.5 h-4.5" />
                {item.label}
              </Link>
            );
          })}
          <div className="pt-2 mt-2 border-t border-gray-200 dark:border-gray-800">
            <Link
              href="/playground"
              className="flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-900 hover:text-gray-900 dark:hover:text-gray-100 transition-colors"
            >
              <FlaskConical className="w-4.5 h-4.5" />
              آزمایشگاه
            </Link>
          </div>
        </nav>
        <div className="p-4 border-t border-gray-200 dark:border-gray-800 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-full bg-gradient-to-bl from-dana-400 to-dana-600 flex items-center justify-center text-white text-xs font-bold">{initial}</div>
              <div>
                <p className="text-xs font-medium text-gray-900 dark:text-gray-100 truncate max-w-[120px]">{user?.email || "کاربر"}</p>
                <p className="text-[10px] text-gray-500 dark:text-gray-400">{user?.tier || "free"}</p>
              </div>
            </div>
            <ThemeToggle />
          </div>
          <button
            onClick={() => logout()}
            className="flex items-center gap-2 w-full px-3 py-2 rounded-xl text-sm text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-900 transition-colors cursor-pointer"
          >
            <LogOut className="w-4 h-4" />
            خروج
          </button>
        </div>
      </aside>

      {/* Mobile bottom nav */}
      <nav className="md:hidden fixed bottom-0 left-0 right-0 z-50 border-t border-gray-200 dark:border-gray-800 bg-white/90 dark:bg-gray-950/90 backdrop-blur-xl">
        <div className="flex items-center justify-around h-14">
          {navItems.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex flex-col items-center gap-0.5 px-3 py-1.5 rounded-lg transition-colors",
                  isActive
                    ? "text-dana-600 dark:text-dana-400"
                    : "text-gray-400 dark:text-gray-500"
                )}
              >
                <item.icon className="w-5 h-5" />
                <span className="text-[10px]">{item.label}</span>
              </Link>
            );
          })}
        </div>
      </nav>
    </>
  );
}
