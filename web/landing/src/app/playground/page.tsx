"use client";

import { useState } from "react";

export default function PlaygroundPage() {
  const [prompt, setPrompt] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);
  const [apiKey, setApiKey] = useState("");

  const handleSubmit = async () => {
    if (!prompt.trim()) return;
    setLoading(true);
    setResponse("");

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const res = await fetch(`${apiUrl}/v1/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey || "dk-demo"}`,
        },
        body: JSON.stringify({
          model: "qwen3-235b-moe",
          messages: [{ role: "user", content: prompt }],
          stream: false,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        setResponse(`خطا: ${err.detail?.error?.message || res.statusText}`);
        return;
      }

      const data = await res.json();
      setResponse(data.choices?.[0]?.message?.content || "پاسخی دریافت نشد");
    } catch (err) {
      setResponse(`خطای اتصال: ${err}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-white" dir="rtl">
      <header className="border-b border-gray-100">
        <nav className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
          <a href="/" className="text-2xl font-bold text-dana-700">دانا</a>
          <div className="flex gap-6 text-sm text-gray-600">
            <a href="/docs" className="hover:text-dana-600">مستندات</a>
            <a href="/pricing" className="hover:text-dana-600">تعرفه‌ها</a>
            <a href="/dashboard" className="hover:text-dana-600">داشبورد</a>
          </div>
        </nav>
      </header>

      <main className="max-w-4xl mx-auto px-4 py-12">
        <h1 className="text-3xl font-bold mb-2">آزمایشگاه API</h1>
        <p className="text-gray-600 mb-8">
          API دانا را مستقیماً در مرورگر آزمایش کنید.
        </p>

        {/* کلید API */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            کلید API (اختیاری)
          </label>
          <input
            type="text"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="dk-..."
            className="w-full border border-gray-300 rounded-lg px-4 py-2 text-sm focus:ring-2 focus:ring-dana-500 focus:border-transparent"
            dir="ltr"
          />
        </div>

        {/* ورودی پرامپت */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            پیام شما
          </label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={4}
            placeholder="سوال یا درخواست خود را اینجا بنویسید..."
            className="w-full border border-gray-300 rounded-lg px-4 py-3 text-sm focus:ring-2 focus:ring-dana-500 focus:border-transparent resize-none"
          />
        </div>

        <button
          onClick={handleSubmit}
          disabled={loading || !prompt.trim()}
          className="bg-dana-600 text-white px-6 py-2.5 rounded-lg font-medium hover:bg-dana-700 transition disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? "در حال پردازش..." : "ارسال درخواست"}
        </button>

        {/* پاسخ */}
        {response && (
          <div className="mt-8">
            <h2 className="text-sm font-medium text-gray-700 mb-2">پاسخ مدل:</h2>
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-6">
              <pre className="whitespace-pre-wrap text-sm text-gray-800 leading-relaxed" dir="auto">
                {response}
              </pre>
            </div>
          </div>
        )}

        {/* نمونه درخواست cURL */}
        <div className="mt-12">
          <h2 className="text-lg font-semibold mb-4">معادل cURL:</h2>
          <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto" dir="ltr">
            <pre className="text-sm text-gray-300">
{`curl -X POST https://api.dana.ir/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer dk-YOUR_KEY" \\
  -d '{
    "model": "qwen3-235b-moe",
    "messages": [{"role": "user", "content": "${prompt || "سلام"}"}],
    "stream": false
  }'`}
            </pre>
          </div>
        </div>
      </main>
    </div>
  );
}
