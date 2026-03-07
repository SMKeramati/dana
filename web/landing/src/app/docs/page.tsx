"use client";

import { Header } from "@/components/layout/header";
import { Footer } from "@/components/layout/footer";
import { Badge } from "@/components/ui/badge";
import { useState } from "react";
import { cn } from "@/lib/utils";
import { BookOpen, Code2, Key, Zap, MessageSquare, Settings2 } from "lucide-react";

const sections = [
  { id: "quickstart", label: "شروع سریع", icon: Zap },
  { id: "auth", label: "احراز هویت", icon: Key },
  { id: "chat", label: "Chat Completions", icon: MessageSquare },
  { id: "models", label: "مدل‌ها", icon: Settings2 },
  { id: "streaming", label: "استریم", icon: Code2 },
  { id: "errors", label: "کدهای خطا", icon: BookOpen },
];

const docs: Record<string, { title: string; content: React.ReactNode }> = {
  quickstart: {
    title: "شروع سریع",
    content: (
      <div className="space-y-6">
        <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
          در کمتر از ۳ دقیقه اولین درخواست API خود را ارسال کنید.
        </p>

        <div>
          <h3 className="text-lg font-semibold mb-3">۱. دریافت کلید API</h3>
          <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
            به <a href="/dashboard/keys" className="text-dana-600 dark:text-dana-400 hover:underline">صفحه کلیدها</a> بروید و یک کلید جدید بسازید.
          </p>
        </div>

        <div>
          <h3 className="text-lg font-semibold mb-3">۲. نصب SDK</h3>
          <div className="rounded-xl bg-gray-950 dark:bg-gray-900 p-4 overflow-x-auto">
            <pre className="text-sm text-gray-300"><code>{`pip install openai`}</code></pre>
          </div>
        </div>

        <div>
          <h3 className="text-lg font-semibold mb-3">۳. ارسال اولین درخواست</h3>
          <div className="rounded-xl bg-gray-950 dark:bg-gray-900 p-4 overflow-x-auto">
            <pre className="text-sm text-gray-300"><code>{`from openai import OpenAI

client = OpenAI(
    base_url="https://api.dana.ir/v1",
    api_key="dk-YOUR_KEY"
)

response = client.chat.completions.create(
    model="qwen3-235b-moe",
    messages=[
        {"role": "user", "content": "سلام!"}
    ]
)

print(response.choices[0].message.content)`}</code></pre>
          </div>
        </div>

        <div>
          <h3 className="text-lg font-semibold mb-3">معادل JavaScript</h3>
          <div className="rounded-xl bg-gray-950 dark:bg-gray-900 p-4 overflow-x-auto">
            <pre className="text-sm text-gray-300"><code>{`import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "https://api.dana.ir/v1",
  apiKey: "dk-YOUR_KEY",
});

const response = await client.chat.completions.create({
  model: "qwen3-235b-moe",
  messages: [{ role: "user", content: "سلام!" }],
});

console.log(response.choices[0].message.content);`}</code></pre>
          </div>
        </div>
      </div>
    ),
  },
  auth: {
    title: "احراز هویت",
    content: (
      <div className="space-y-6">
        <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
          تمام درخواست‌ها به API دانا نیاز به کلید API دارند. کلید را در هدر
          <code className="mx-1 px-1.5 py-0.5 bg-gray-100 dark:bg-gray-800 rounded text-xs" dir="ltr">Authorization</code>
          ارسال کنید.
        </p>

        <div className="rounded-xl bg-gray-950 dark:bg-gray-900 p-4 overflow-x-auto">
          <pre className="text-sm text-gray-300"><code>{`curl https://api.dana.ir/v1/chat/completions \\
  -H "Authorization: Bearer dk-YOUR_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"model": "qwen3-235b-moe", "messages": [{"role": "user", "content": "سلام"}]}'`}</code></pre>
        </div>

        <div className="rounded-xl border border-amber-200 dark:border-amber-800 bg-amber-50 dark:bg-amber-950/30 p-4">
          <p className="text-sm text-amber-800 dark:text-amber-200 font-medium">نکته امنیتی</p>
          <p className="text-sm text-amber-700 dark:text-amber-300 mt-1">
            هرگز کلید API را در کد فرانت‌اند قرار ندهید. کلید را در متغیرهای محیطی سرور نگهداری کنید.
          </p>
        </div>

        <h3 className="text-lg font-semibold">فرمت کلید</h3>
        <p className="text-sm text-gray-600 dark:text-gray-400">
          کلیدهای دانا با پیشوند <code className="px-1.5 py-0.5 bg-gray-100 dark:bg-gray-800 rounded text-xs" dir="ltr">dk-</code> شروع می‌شوند.
          هر کلید شامل اطلاعات tier و permissions است.
        </p>
      </div>
    ),
  },
  chat: {
    title: "Chat Completions",
    content: (
      <div className="space-y-6">
        <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
          اندپوینت Chat Completions کاملا سازگار با OpenAI API است.
        </p>

        <div>
          <h3 className="text-lg font-semibold mb-2">
            <code className="text-sm font-mono" dir="ltr">POST /v1/chat/completions</code>
          </h3>
        </div>

        <h3 className="text-base font-semibold">پارامترها</h3>
        <div className="rounded-xl border border-gray-200 dark:border-gray-800 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-50 dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800">
                <th className="text-right px-4 py-2 font-medium">پارامتر</th>
                <th className="text-right px-4 py-2 font-medium">نوع</th>
                <th className="text-right px-4 py-2 font-medium">الزامی</th>
                <th className="text-right px-4 py-2 font-medium">توضیح</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100 dark:divide-gray-800">
              <tr><td className="px-4 py-2 font-mono text-xs" dir="ltr">model</td><td className="px-4 py-2">string</td><td className="px-4 py-2">بله</td><td className="px-4 py-2 text-gray-500 dark:text-gray-400">شناسه مدل</td></tr>
              <tr><td className="px-4 py-2 font-mono text-xs" dir="ltr">messages</td><td className="px-4 py-2">array</td><td className="px-4 py-2">بله</td><td className="px-4 py-2 text-gray-500 dark:text-gray-400">لیست پیام‌ها</td></tr>
              <tr><td className="px-4 py-2 font-mono text-xs" dir="ltr">temperature</td><td className="px-4 py-2">float</td><td className="px-4 py-2">خیر</td><td className="px-4 py-2 text-gray-500 dark:text-gray-400">خلاقیت (0-2، پیش‌فرض: 0.7)</td></tr>
              <tr><td className="px-4 py-2 font-mono text-xs" dir="ltr">max_tokens</td><td className="px-4 py-2">int</td><td className="px-4 py-2">خیر</td><td className="px-4 py-2 text-gray-500 dark:text-gray-400">حداکثر توکن خروجی</td></tr>
              <tr><td className="px-4 py-2 font-mono text-xs" dir="ltr">stream</td><td className="px-4 py-2">bool</td><td className="px-4 py-2">خیر</td><td className="px-4 py-2 text-gray-500 dark:text-gray-400">استریم SSE</td></tr>
              <tr><td className="px-4 py-2 font-mono text-xs" dir="ltr">top_p</td><td className="px-4 py-2">float</td><td className="px-4 py-2">خیر</td><td className="px-4 py-2 text-gray-500 dark:text-gray-400">نمونه‌برداری هسته‌ای</td></tr>
              <tr><td className="px-4 py-2 font-mono text-xs" dir="ltr">logprobs</td><td className="px-4 py-2">bool</td><td className="px-4 py-2">خیر</td><td className="px-4 py-2 text-gray-500 dark:text-gray-400">بازگشت log probabilities</td></tr>
            </tbody>
          </table>
        </div>

        <h3 className="text-base font-semibold mt-4">نمونه پاسخ</h3>
        <div className="rounded-xl bg-gray-950 dark:bg-gray-900 p-4 overflow-x-auto">
          <pre className="text-sm text-gray-300"><code>{`{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1709820000,
  "model": "qwen3-235b-moe",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "سلام! چطور می‌توانم کمکتان کنم؟"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 15,
    "total_tokens": 25
  }
}`}</code></pre>
        </div>
      </div>
    ),
  },
  models: {
    title: "مدل‌ها",
    content: (
      <div className="space-y-6">
        <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
          مدل‌های در دسترس از طریق API دانا.
        </p>

        <div className="rounded-xl border border-gray-200 dark:border-gray-800 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-50 dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800">
                <th className="text-right px-4 py-2 font-medium">شناسه</th>
                <th className="text-right px-4 py-2 font-medium">پارامترها</th>
                <th className="text-right px-4 py-2 font-medium">حداکثر ورودی</th>
                <th className="text-right px-4 py-2 font-medium">وضعیت</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100 dark:divide-gray-800">
              <tr>
                <td className="px-4 py-3 font-mono text-xs" dir="ltr">qwen3-235b-moe</td>
                <td className="px-4 py-3">۲۳۵ میلیارد (MoE)</td>
                <td className="px-4 py-3">۳۲,۷۶۸ توکن</td>
                <td className="px-4 py-3"><Badge variant="success">فعال</Badge></td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="rounded-xl bg-gray-950 dark:bg-gray-900 p-4 overflow-x-auto">
          <pre className="text-sm text-gray-300"><code>{`# لیست مدل‌ها
curl https://api.dana.ir/v1/models \\
  -H "Authorization: Bearer dk-YOUR_KEY"`}</code></pre>
        </div>
      </div>
    ),
  },
  streaming: {
    title: "استریم (SSE)",
    content: (
      <div className="space-y-6">
        <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
          با تنظیم <code className="px-1.5 py-0.5 bg-gray-100 dark:bg-gray-800 rounded text-xs" dir="ltr">stream: true</code> پاسخ‌ها به صورت Server-Sent Events دریافت می‌شوند.
        </p>

        <div className="rounded-xl bg-gray-950 dark:bg-gray-900 p-4 overflow-x-auto">
          <pre className="text-sm text-gray-300"><code>{`from openai import OpenAI

client = OpenAI(
    base_url="https://api.dana.ir/v1",
    api_key="dk-YOUR_KEY"
)

stream = client.chat.completions.create(
    model="qwen3-235b-moe",
    messages=[{"role": "user", "content": "یک داستان کوتاه بنویس"}],
    stream=True
)

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)`}</code></pre>
        </div>

        <h3 className="text-base font-semibold">فرمت SSE</h3>
        <div className="rounded-xl bg-gray-950 dark:bg-gray-900 p-4 overflow-x-auto">
          <pre className="text-sm text-gray-300"><code>{`data: {"id":"chatcmpl-abc","choices":[{"delta":{"content":"سلام"}}]}

data: {"id":"chatcmpl-abc","choices":[{"delta":{"content":"!"}}]}

data: [DONE]`}</code></pre>
        </div>
      </div>
    ),
  },
  errors: {
    title: "کدهای خطا",
    content: (
      <div className="space-y-6">
        <p className="text-gray-600 dark:text-gray-400 leading-relaxed">
          API دانا از کدهای HTTP استاندارد برای نشان دادن خطا استفاده می‌کند.
        </p>

        <div className="rounded-xl border border-gray-200 dark:border-gray-800 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-gray-50 dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800">
                <th className="text-right px-4 py-2 font-medium">کد</th>
                <th className="text-right px-4 py-2 font-medium">معنی</th>
                <th className="text-right px-4 py-2 font-medium">راه‌حل</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100 dark:divide-gray-800">
              <tr><td className="px-4 py-2 font-mono">400</td><td className="px-4 py-2">درخواست نامعتبر</td><td className="px-4 py-2 text-gray-500 dark:text-gray-400">پارامترها را بررسی کنید</td></tr>
              <tr><td className="px-4 py-2 font-mono">401</td><td className="px-4 py-2">احراز هویت ناموفق</td><td className="px-4 py-2 text-gray-500 dark:text-gray-400">کلید API را بررسی کنید</td></tr>
              <tr><td className="px-4 py-2 font-mono">403</td><td className="px-4 py-2">دسترسی ممنوع</td><td className="px-4 py-2 text-gray-500 dark:text-gray-400">مجوزهای کلید را بررسی کنید</td></tr>
              <tr><td className="px-4 py-2 font-mono">429</td><td className="px-4 py-2">محدودیت نرخ</td><td className="px-4 py-2 text-gray-500 dark:text-gray-400">کمی صبر کنید و دوباره تلاش کنید</td></tr>
              <tr><td className="px-4 py-2 font-mono">500</td><td className="px-4 py-2">خطای سرور</td><td className="px-4 py-2 text-gray-500 dark:text-gray-400">با پشتیبانی تماس بگیرید</td></tr>
              <tr><td className="px-4 py-2 font-mono">503</td><td className="px-4 py-2">سرویس در دسترس نیست</td><td className="px-4 py-2 text-gray-500 dark:text-gray-400">چند دقیقه بعد تلاش کنید</td></tr>
            </tbody>
          </table>
        </div>

        <h3 className="text-base font-semibold">فرمت خطا</h3>
        <div className="rounded-xl bg-gray-950 dark:bg-gray-900 p-4 overflow-x-auto">
          <pre className="text-sm text-gray-300"><code>{`{
  "error": {
    "message": "Invalid API key provided",
    "type": "authentication_error",
    "code": "invalid_api_key"
  }
}`}</code></pre>
        </div>
      </div>
    ),
  },
};

export default function DocsPage() {
  const [activeSection, setActiveSection] = useState("quickstart");

  return (
    <div className="min-h-screen">
      <Header />

      <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="flex gap-8">
          {/* Sidebar Nav */}
          <nav className="hidden md:block w-56 shrink-0">
            <div className="sticky top-24 space-y-1">
              <h2 className="text-xs font-semibold text-gray-400 dark:text-gray-500 uppercase tracking-wider mb-3">مستندات API</h2>
              {sections.map((s) => (
                <button
                  key={s.id}
                  onClick={() => setActiveSection(s.id)}
                  className={cn(
                    "flex items-center gap-2.5 w-full px-3 py-2 rounded-xl text-sm transition-colors cursor-pointer",
                    activeSection === s.id
                      ? "bg-dana-50 dark:bg-dana-950/50 text-dana-700 dark:text-dana-300 font-medium"
                      : "text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-900"
                  )}
                >
                  <s.icon className="w-4 h-4" />
                  {s.label}
                </button>
              ))}
            </div>
          </nav>

          {/* Mobile Nav */}
          <div className="md:hidden mb-6 flex gap-2 flex-wrap">
            {sections.map((s) => (
              <button
                key={s.id}
                onClick={() => setActiveSection(s.id)}
                className={cn(
                  "px-3 py-1.5 rounded-lg text-xs transition-colors cursor-pointer",
                  activeSection === s.id
                    ? "bg-dana-100 dark:bg-dana-900/50 text-dana-700 dark:text-dana-300"
                    : "bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400"
                )}
              >
                {s.label}
              </button>
            ))}
          </div>

          {/* Content */}
          <div className="flex-1 min-w-0">
            <div className="max-w-3xl">
              <h1 className="text-3xl font-bold mb-2">{docs[activeSection].title}</h1>
              <div className="h-1 w-12 bg-gradient-to-l from-dana-400 to-dana-600 rounded-full mb-8" />
              {docs[activeSection].content}
            </div>
          </div>
        </div>
      </div>

      <Footer />
    </div>
  );
}
