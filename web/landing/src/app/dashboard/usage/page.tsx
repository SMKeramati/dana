"use client";

import { Card, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useState } from "react";
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, BarChart, Bar } from "recharts";
import { Activity, Hash, Clock, TrendingUp, BarChart3 } from "lucide-react";
import { cn } from "@/lib/utils";

const periods = [
  { label: "۲۴ ساعت", value: "24h" },
  { label: "۷ روز", value: "7d" },
  { label: "۳۰ روز", value: "30d" },
  { label: "۹۰ روز", value: "90d" },
];

const chartData = [
  { name: "شنبه", tokens: 0, requests: 0 },
  { name: "یکشنبه", tokens: 0, requests: 0 },
  { name: "دوشنبه", tokens: 0, requests: 0 },
  { name: "سه‌شنبه", tokens: 0, requests: 0 },
  { name: "چهارشنبه", tokens: 0, requests: 0 },
  { name: "پنجشنبه", tokens: 0, requests: 0 },
  { name: "جمعه", tokens: 0, requests: 0 },
];

const statCards = [
  { label: "کل توکن‌های مصرفی", value: "۰", icon: Activity, color: "text-blue-500 bg-blue-500/10", change: null },
  { label: "تعداد درخواست‌ها", value: "۰", icon: Hash, color: "text-emerald-500 bg-emerald-500/10", change: null },
  { label: "میانگین تاخیر", value: "— ms", icon: Clock, color: "text-amber-500 bg-amber-500/10", change: null },
];

export default function UsagePage() {
  const [period, setPeriod] = useState("24h");
  const [chartView, setChartView] = useState<"tokens" | "requests">("tokens");

  return (
    <div className="p-6 lg:p-8 max-w-6xl">
      <div className="flex items-center justify-between mb-8">
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
      <div className="grid grid-cols-3 gap-4 mb-8">
        {statCards.map((stat, i) => (
          <div key={stat.label}>
            <Card>
              <div className="flex items-start justify-between mb-3">
                <div className={`w-9 h-9 rounded-xl ${stat.color} flex items-center justify-center`}>
                  <stat.icon className="w-4.5 h-4.5" />
                </div>
                {stat.change && (
                  <Badge variant="success" className="text-[10px]">
                    <TrendingUp className="w-3 h-3 ml-1" />
                    {stat.change}
                  </Badge>
                )}
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
          <p className="text-xs text-gray-400 dark:text-gray-500 mt-2 text-center">اولین درخواست API خود را ارسال کنید تا آمار شروع شود</p>
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
