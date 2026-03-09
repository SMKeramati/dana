"use client";

import { Card, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useState } from "react";
import { useUsage } from "@/hooks/use-usage";
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, BarChart, Bar } from "recharts";
import { Activity, Hash, Clock, BarChart3 } from "lucide-react";
import { cn, toPersianDigits } from "@/lib/utils";

const periods = [
  { label: "\u06F2\u06F4 \u0633\u0627\u0639\u062A", value: "24h" },
  { label: "\u06F7 \u0631\u0648\u0632", value: "7d" },
  { label: "\u06F3\u06F0 \u0631\u0648\u0632", value: "30d" },
  { label: "\u06F9\u06F0 \u0631\u0648\u0632", value: "90d" },
];

const dayNames = ["\u0634\u0646\u0628\u0647", "\u06CC\u06A9\u0634\u0646\u0628\u0647", "\u062F\u0648\u0634\u0646\u0628\u0647", "\u0633\u0647\u200C\u0634\u0646\u0628\u0647", "\u0686\u0647\u0627\u0631\u0634\u0646\u0628\u0647", "\u067E\u0646\u062C\u0634\u0646\u0628\u0647", "\u062C\u0645\u0639\u0647"];

export default function UsagePage() {
  const [period, setPeriod] = useState("7d");
  const [chartView, setChartView] = useState<"tokens" | "requests">("tokens");
  const { data, loading } = useUsage(period);

  const chartData = data?.daily?.map((d, i) => ({
    name: dayNames[i % 7] || d.date,
    tokens: d.tokens,
    requests: d.requests,
  })) || dayNames.map((name) => ({ name, tokens: 0, requests: 0 }));

  const totalTokens = data?.total_tokens ?? 0;
  const totalRequests = data?.total_requests ?? 0;

  const statCards = [
    { label: "\u06A9\u0644 \u062A\u0648\u06A9\u0646\u200C\u0647\u0627\u06CC \u0645\u0635\u0631\u0641\u06CC", value: toPersianDigits(totalTokens.toLocaleString()), icon: Activity, color: "text-blue-500 bg-blue-500/10" },
    { label: "\u062A\u0639\u062F\u0627\u062F \u062F\u0631\u062E\u0648\u0627\u0633\u062A\u200C\u0647\u0627", value: toPersianDigits(totalRequests.toLocaleString()), icon: Hash, color: "text-emerald-500 bg-emerald-500/10" },
    { label: "\u0645\u06CC\u0627\u0646\u06AF\u06CC\u0646 \u062A\u0627\u062E\u06CC\u0631", value: "\u2014 ms", icon: Clock, color: "text-amber-500 bg-amber-500/10" },
  ];

  if (loading) {
    return (
      <div className="p-6 lg:p-8 max-w-6xl">
        <Skeleton className="h-8 w-48 mb-2" />
        <Skeleton className="h-4 w-64 mb-8" />
        <div className="grid grid-cols-3 gap-4 mb-8">
          {[1, 2, 3].map((i) => <Skeleton key={i} className="h-28 rounded-2xl" />)}
        </div>
        <Skeleton className="h-80 rounded-2xl" />
      </div>
    );
  }

  return (
    <div className="p-6 lg:p-8 max-w-6xl">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-8">
        <div>
          <h1 className="text-2xl font-bold">آمار مصرف</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">جزئیات مصرف API خود را مشاهده کنید</p>
        </div>
        <div className="flex bg-gray-100 dark:bg-gray-800 rounded-xl p-1">
          {periods.map((p) => (
            <button
              key={p.value}
              onClick={() => setPeriod(p.value)}
              className={cn(
                "px-3 py-1.5 text-xs rounded-lg transition-all cursor-pointer",
                period === p.value
                  ? "bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 shadow-sm font-medium"
                  : "text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200"
              )}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
        {statCards.map((stat) => (
          <div key={stat.label}>
            <Card>
              <div className="flex items-start justify-between mb-3">
                <div className={`w-9 h-9 rounded-xl ${stat.color} flex items-center justify-center`}>
                  <stat.icon className="w-4.5 h-4.5" />
                </div>
              </div>
              <div className="text-2xl font-bold">{stat.value}</div>
              <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">{stat.label}</div>
            </Card>
          </div>
        ))}
      </div>

      {/* Chart */}
      <div>
        <Card className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <CardTitle>نمودار مصرف</CardTitle>
            <div className="flex bg-gray-100 dark:bg-gray-800 rounded-lg p-0.5">
              <button onClick={() => setChartView("tokens")} className={cn("px-3 py-1 text-xs rounded-md transition-all cursor-pointer", chartView === "tokens" ? "bg-white dark:bg-gray-700 shadow-sm" : "text-gray-500")}>
                توکن
              </button>
              <button onClick={() => setChartView("requests")} className={cn("px-3 py-1 text-xs rounded-md transition-all cursor-pointer", chartView === "requests" ? "bg-white dark:bg-gray-700 shadow-sm" : "text-gray-500")}>
                درخواست
              </button>
            </div>
          </div>
          <div className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              {chartView === "tokens" ? (
                <AreaChart data={chartData}>
                  <defs>
                    <linearGradient id="usageGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#0c87f0" stopOpacity={0.2} />
                      <stop offset="95%" stopColor="#0c87f0" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="name" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} />
                  <YAxis tick={{ fontSize: 11 }} tickLine={false} axisLine={false} width={40} />
                  <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid #e5e7eb", fontSize: 12 }} />
                  <Area type="monotone" dataKey="tokens" stroke="#0c87f0" strokeWidth={2} fill="url(#usageGrad)" />
                </AreaChart>
              ) : (
                <BarChart data={chartData}>
                  <XAxis dataKey="name" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} />
                  <YAxis tick={{ fontSize: 11 }} tickLine={false} axisLine={false} width={40} />
                  <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid #e5e7eb", fontSize: 12 }} />
                  <Bar dataKey="requests" fill="#10b981" radius={[6, 6, 0, 0]} />
                </BarChart>
              )}
            </ResponsiveContainer>
          </div>
          {totalTokens === 0 && (
            <p className="text-xs text-gray-400 dark:text-gray-500 mt-2 text-center">اولین درخواست API خود را ارسال کنید تا آمار شروع شود</p>
          )}
        </Card>
      </div>

      {/* Recent Requests */}
      <div>
        <Card>
          <CardTitle className="mb-4">آخرین درخواست‌ها</CardTitle>
          <div className="text-center py-12">
            <div className="w-12 h-12 rounded-2xl bg-gray-100 dark:bg-gray-800 flex items-center justify-center mx-auto mb-3">
              <BarChart3 className="w-6 h-6 text-gray-400 dark:text-gray-500" />
            </div>
            <p className="text-sm text-gray-500 dark:text-gray-400">هنوز درخواستی ثبت نشده.</p>
            <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">اولین درخواست API خود را ارسال کنید.</p>
          </div>
        </Card>
      </div>
    </div>
  );
}
