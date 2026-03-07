"use client";

import { Header } from "@/components/layout/header";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { useState, useRef } from "react";
import { Send, Trash2, Copy, Check, Settings2, Bot, User } from "lucide-react";
import { cn } from "@/lib/utils";

interface Message {
  role: "user" | "assistant";
  content: string;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "https://api.dana.ir";

export default function PlaygroundPage() {
  const [apiKey, setApiKey] = useState("dk-demo");
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(1024);
  const [showSettings, setShowSettings] = useState(false);
  const [copied, setCopied] = useState(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  async function handleSend() {
    if (!input.trim() || loading) return;
    const userMsg: Message = { role: "user", content: input.trim() };
    const newMessages = [...messages, userMsg];
    setMessages(newMessages);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(`${API_URL}/v1/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
        body: JSON.stringify({
          model: "qwen3-235b-moe",
          messages: newMessages.map((m) => ({ role: m.role, content: m.content })),
          temperature,
          max_tokens: maxTokens,
          stream: false,
        }),
      });
      const data = await res.json();
      const assistantContent = data.choices?.[0]?.message?.content || "خطا در دریافت پاسخ";
      setMessages([...newMessages, { role: "assistant", content: assistantContent }]);
    } catch {
      setMessages([...newMessages, { role: "assistant", content: "خطا در اتصال به سرور. لطفا دوباره تلاش کنید." }]);
    } finally {
      setLoading(false);
    }
  }

  function handleCopy() {
    const code = `curl -X POST ${API_URL}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer ${apiKey}" \\
  -d '{
    "model": "qwen3-235b-moe",
    "messages": ${JSON.stringify(messages.map((m) => ({ role: m.role, content: m.content })))},
    "temperature": ${temperature}
  }'`;
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Header />

      <div className="flex-1 flex">
        {/* Chat Area */}
        <div className="flex-1 flex flex-col">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-6">
            {messages.length === 0 ? (
              <div className="h-full flex items-center justify-center">
                <div className="text-center">
                  <div className="w-16 h-16 rounded-2xl bg-gradient-to-bl from-dana-400 to-dana-600 flex items-center justify-center mx-auto mb-4 shadow-lg shadow-dana-500/20">
                    <Bot className="w-8 h-8 text-white" />
                  </div>
                  <h2 className="text-xl font-bold mb-2">آزمایشگاه API دانا</h2>
                  <p className="text-sm text-gray-500 dark:text-gray-400 max-w-md">
                    یک پیام ارسال کنید تا مکالمه با مدل Qwen3-235B-MoE را شروع کنید.
                  </p>
                </div>
              </div>
            ) : (
              <div className="max-w-3xl mx-auto space-y-4">
                {messages.map((msg, i) => (
                  <div key={i} className={cn("flex gap-3", msg.role === "user" ? "flex-row-reverse" : "")}>
                    <div className={cn("w-8 h-8 rounded-full flex items-center justify-center shrink-0 mt-1", msg.role === "user" ? "bg-dana-100 dark:bg-dana-900/50 text-dana-600 dark:text-dana-400" : "bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400")}>
                      {msg.role === "user" ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
                    </div>
                    <div className={cn("max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed", msg.role === "user" ? "bg-dana-600 text-white rounded-tl-sm" : "bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-100 rounded-tr-sm")}>
                      <pre className="whitespace-pre-wrap font-sans" dir="auto">{msg.content}</pre>
                    </div>
                  </div>
                ))}
                {loading && (
                  <div className="flex gap-3">
                    <div className="w-8 h-8 rounded-full bg-gray-100 dark:bg-gray-800 flex items-center justify-center">
                      <Bot className="w-4 h-4 text-gray-400" />
                    </div>
                    <div className="bg-gray-100 dark:bg-gray-800 rounded-2xl rounded-tr-sm px-4 py-3">
                      <div className="flex gap-1">
                        <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:0ms]" />
                        <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:150ms]" />
                        <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:300ms]" />
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Input */}
          <div className="border-t border-gray-200 dark:border-gray-800 p-4">
            <div className="max-w-3xl mx-auto">
              <div className="flex gap-2 mb-3">
                <button onClick={() => setShowSettings(!showSettings)} className={cn("flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs transition-colors cursor-pointer", showSettings ? "bg-dana-100 dark:bg-dana-900/50 text-dana-600 dark:text-dana-400" : "text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800")}>
                  <Settings2 className="w-3.5 h-3.5" />
                  تنظیمات
                </button>
                {messages.length > 0 && (
                  <>
                    <button onClick={() => setMessages([])} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors cursor-pointer">
                      <Trash2 className="w-3.5 h-3.5" />
                      پاک کردن
                    </button>
                    <button onClick={handleCopy} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors cursor-pointer">
                      {copied ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
                      {copied ? "کپی شد!" : "کپی cURL"}
                    </button>
                  </>
                )}
              </div>
              {showSettings && (
                <div className="mb-3 p-4 rounded-xl bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-800 grid grid-cols-1 sm:grid-cols-3 gap-4">
                  <div>
                    <label className="text-xs text-gray-500 dark:text-gray-400 mb-1 block">کلید API</label>
                    <Input value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder="dk-..." className="text-xs font-mono" dir="ltr" />
                  </div>
                  <div>
                    <label className="text-xs text-gray-500 dark:text-gray-400 mb-1 block">Temperature: {temperature}</label>
                    <input type="range" min="0" max="2" step="0.1" value={temperature} onChange={(e) => setTemperature(parseFloat(e.target.value))} className="w-full accent-dana-500" />
                  </div>
                  <div>
                    <label className="text-xs text-gray-500 dark:text-gray-400 mb-1 block">حداکثر توکن</label>
                    <Input type="number" value={maxTokens} onChange={(e) => setMaxTokens(parseInt(e.target.value) || 1024)} className="text-xs font-mono" dir="ltr" />
                  </div>
                </div>
              )}
              <div className="flex gap-2 items-end">
                <div className="flex-1 relative">
                  <textarea
                    ref={inputRef}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); } }}
                    placeholder="سوال یا درخواست خود را بنویسید..."
                    rows={1}
                    className="w-full rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 px-4 py-3 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-dana-500/50 focus:border-dana-500 transition-colors placeholder:text-gray-400 dark:placeholder:text-gray-500"
                    style={{ minHeight: 44, maxHeight: 200 }}
                  />
                </div>
                <Button onClick={handleSend} disabled={loading || !input.trim()} className="h-[44px] w-[44px] p-0 shrink-0">
                  <Send className="w-4 h-4" />
                </Button>
              </div>
              <div className="mt-2 flex items-center gap-2">
                <Badge variant="outline" className="text-[10px]">مدل: qwen3-235b-moe</Badge>
                <Badge variant="outline" className="text-[10px]">Ctrl+Enter ارسال</Badge>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
