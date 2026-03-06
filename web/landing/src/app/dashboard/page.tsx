"use client";

import { Card, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { Zap, Hash, Clock, Crown, ArrowLeft, Key, BookOpen, CheckCircle2, Circle } from "lucide-react";

const statCards = [
  { label: "توکن مصرفی امروز", value: "۰ / ۱,۰۰۰", icon: Zap, color: "text-blue-500 bg-blue-500/10" },
  { label: "درخواست‌های امروز", value: "۰", icon: Hash, color: "text-emerald-500 bg-emerald-500/10" },
  { label: "میانگین تاخیر", value: "— ms", icon: Clock, color: "text-amber-500 bg-amber-500/10" },
  { label: "پلن فعلی", value: "رایگان", icon: Crown, color: "text-purple-500 bg-purple-500/10" },
];

const chartData = [
  { name: "شنبه", tokens: 0 },
  { name: "یکشنبه", tokens: 0 },
  { name: "دوشنبه", tokens: 0 },
  { name: "سه‌شنبه", tokens: 0 },
  { name: "چهارشنبه", tokens: 0 },
  { name: "پنجشنبه", tokens: 0 },
  { name: "جمعه", tokens: 0 },
];

const onboardingSteps = [
  { label: "یک کلید API بسازید", href: "/dashboard/keys", done: false, icon: Key },
  { label: "اولین درخواست خود را ارسال کنید", href: "/playground", done: false, icon: Zap },
  { label: "مستندات را بخوانید", href: "/docs", done: false, icon: BookOpen },
];

export default function DashboardPage() {
  return (
    <div className="p-6 lg:p-8 max-w-6xl">
      {/* Welcome */}
      <div
       
       
        className="mb-8 p-6 rounded-2xl bg-gradient-to-l from-dana-600 to-dana-500 text-white relative overflow-hidden"
      >
        <div className="absolute inset-0 grid-pattern opacity-20" />
        <div className="relative">
          <h1 className="text-2xl font-bold">سلام! خوش آمدید 👋</h1>
          <p className="mt-2 text-dana-100 text-sm">از داشبورد دانا برای مدیریت کلیدها، مصرف و صورتحساب استفاده کنید.</p>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        {statCards.map((stat, i) => (
          <div key={stat.label}>
            <Card className="hover:shadow-md transition-shadow">
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

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Usage Chart */}
        <div className="lg:col-span-2">
          <Card>
            <CardTitle className="mb-4">مصرف هفتگی</CardTitle>
            <div className="h-[220px]">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData}>
                  <defs>
                    <linearGradient id="tokenGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#0c87f0" stopOpacity={0.2} />
                      <stop offset="95%" stopColor="#0c87f0" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="name" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} />
                  <YAxis tick={{ fontSize: 11 }} tickLine={false} axisLine={false} width={40} />
                  <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid #e5e7eb", fontSize: 12 }} />
                  <Area type="monotone" dataKey="tokens" stroke="#0c87f0" strokeWidth={2} fill="url(#tokenGradient)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            <p className="text-xs text-gray-400 dark:text-gray-500 mt-2 text-center">اولین درخواست API خود را ارسال کنید تا نمودار مصرف شروع شود</p>
          </Card>
        </div>

        {/* Onboarding */}
        <div>
          <Card>
            <CardTitle className="mb-1">شروع سریع</CardTitle>
            <p className="text-xs text-gray-500 dark:text-gray-400 mb-5">۳ قدم تا اولین درخواست</p>
            <div className="space-y-4">
              {onboardingSteps.map((step, i) => (
                <Link key={i} href={step.href} className="flex items-center gap-3 group">
                  <div className={`w-7 h-7 rounded-full flex items-center justify-center shrink-0 ${step.done ? "bg-emerald-100 dark:bg-emerald-900/50" : "bg-gray-100 dark:bg-gray-800"}`}>
                    {step.done ? <CheckCircle2 className="w-4 h-4 text-emerald-500" /> : <Circle className="w-4 h-4 text-gray-400" />}
                  </div>
                  <span className={`text-sm ${step.done ? "line-through text-gray-400" : "text-gray-700 dark:text-gray-300 group-hover:text-dana-600 dark:group-hover:text-dana-400"} transition-colors`}>
                    {step.label}
                  </span>
                  <ArrowLeft className="w-3.5 h-3.5 text-gray-300 dark:text-gray-600 mr-auto group-hover:text-dana-500 transition-colors" />
                </Link>
              ))}
            </div>
          </Card>
        </div>
      </div>

      {/* Quick Code */}
      <div className="mt-6">
        <Card>
          <CardTitle className="mb-4">نمونه کد</CardTitle>
          <div className="rounded-xl bg-gray-950 dark:bg-gray-900 overflow-hidden">
            <div className="flex items-center gap-2 px-4 py-2.5 border-b border-gray-800">
              <div className="flex gap-1.5">
                <div className="w-2.5 h-2.5 rounded-full bg-red-500/80" />
                <div className="w-2.5 h-2.5 rounded-full bg-amber-500/80" />
                <div className="w-2.5 h-2.5 rounded-full bg-emerald-500/80" />
              </div>
              <Badge variant="outline" className="text-[10px] border-gray-700 text-gray-400 mr-2">Python</Badge>
            </div>
            <pre className="p-4 text-xs leading-relaxed overflow-x-auto">
              <code className="text-gray-300">
{`from openai import OpenAI

client = OpenAI(
    base_url="https://api.dana.ir/v1",
    api_key="dk-YOUR_KEY"
)

response = client.chat.completions.create(
    model="qwen3-235b-moe",
    messages=[{"role": "user", "content": "سلام! یک تابع فیبوناچی بنویس"}]
)
print(response.choices[0].message.content)`}
              </code>
            </pre>
          </div>
        </Card>
      </div>
    </div>
  );
}
