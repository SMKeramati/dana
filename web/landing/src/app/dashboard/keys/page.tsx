"use client";

import { Card, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { useState } from "react";
import { Plus, Copy, Check, Trash2, Key, Eye, EyeOff } from "lucide-react";

interface ApiKey {
  id: string;
  name: string;
  prefix: string;
  fullKey?: string;
  createdAt: string;
  lastUsed: string;
  status: "active" | "revoked";
}

export default function KeysPage() {
  const [keys, setKeys] = useState<ApiKey[]>([]);
  const [newKeyName, setNewKeyName] = useState("");
  const [showCreate, setShowCreate] = useState(false);
  const [newlyCreated, setNewlyCreated] = useState<ApiKey | null>(null);
  const [copied, setCopied] = useState(false);

  function createKey() {
    if (!newKeyName.trim()) return;
    const id = Math.random().toString(36).substring(2, 10);
    const key: ApiKey = {
      id,
      name: newKeyName,
      prefix: `dk-f1_${id}`,
      fullKey: `dk-f1_${id}${"x".repeat(32)}`,
      createdAt: new Date().toLocaleDateString("fa-IR"),
      lastUsed: "هرگز",
      status: "active",
    };
    setKeys((prev) => [key, ...prev]);
    setNewKeyName("");
    setNewlyCreated(key);
    setShowCreate(false);
  }

  function copyKey(text: string) {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  function revokeKey(id: string) {
    setKeys((prev) => prev.map((k) => (k.id === id ? { ...k, status: "revoked" as const } : k)));
  }

  return (
    <div className="p-6 lg:p-8 max-w-4xl">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold">کلیدهای API</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">کلیدهای API خود را مدیریت کنید</p>
        </div>
        <Button onClick={() => setShowCreate(true)} size="sm">
          <Plus className="w-4 h-4 ml-1.5" />
          ساخت کلید جدید
        </Button>
      </div>

      {/* Create Key Dialog */}
      
        {showCreate && (
          <div>
            <Card className="mb-6 border-dana-200 dark:border-dana-800">
              <CardTitle>ساخت کلید جدید</CardTitle>
              <div className="mt-4 flex gap-3">
                <Input
                  value={newKeyName}
                  onChange={(e) => setNewKeyName(e.target.value)}
                  placeholder="نام کلید (مثلاً: پروژه اصلی)"
                  onKeyDown={(e) => e.key === "Enter" && createKey()}
                />
                <Button onClick={createKey} disabled={!newKeyName.trim()}>ایجاد</Button>
                <Button variant="ghost" onClick={() => setShowCreate(false)}>انصراف</Button>
              </div>
            </Card>
          </div>
        )}
      

      {/* Newly Created Key Alert */}
      
        {newlyCreated && (
          <div>
            <Card className="mb-6 border-emerald-300 dark:border-emerald-800 bg-emerald-50 dark:bg-emerald-950/30">
              <div className="flex items-start gap-3">
                <Key className="w-5 h-5 text-emerald-600 dark:text-emerald-400 mt-0.5 shrink-0" />
                <div className="flex-1">
                  <p className="text-sm font-semibold text-emerald-800 dark:text-emerald-200">کلید &quot;{newlyCreated.name}&quot; ساخته شد!</p>
                  <p className="text-xs text-emerald-600 dark:text-emerald-400 mt-1">این کلید فقط یک بار نمایش داده می‌شود. حتما آن را کپی کنید.</p>
                  <div className="mt-3 flex items-center gap-2 bg-white dark:bg-gray-900 rounded-lg border border-emerald-200 dark:border-emerald-800 px-3 py-2">
                    <code className="text-xs font-mono text-gray-700 dark:text-gray-300 flex-1" dir="ltr">{newlyCreated.fullKey}</code>
                    <button onClick={() => copyKey(newlyCreated.fullKey || "")} className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 cursor-pointer">
                      {copied ? <Check className="w-4 h-4 text-emerald-500" /> : <Copy className="w-4 h-4" />}
                    </button>
                  </div>
                  <button onClick={() => setNewlyCreated(null)} className="mt-3 text-xs text-emerald-600 dark:text-emerald-400 hover:underline cursor-pointer">متوجه شدم، بستن</button>
                </div>
              </div>
            </Card>
          </div>
        )}
      

      {/* Keys List */}
      {keys.length === 0 ? (
        <div>
          <Card className="text-center py-16">
            <div className="w-16 h-16 rounded-2xl bg-gray-100 dark:bg-gray-800 flex items-center justify-center mx-auto mb-4">
              <Key className="w-8 h-8 text-gray-400 dark:text-gray-500" />
            </div>
            <h3 className="font-semibold mb-1">هنوز کلیدی ندارید</h3>
            <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">اولین کلید API خود را بسازید تا شروع کنید.</p>
            <Button onClick={() => setShowCreate(true)} size="sm">
              <Plus className="w-4 h-4 ml-1.5" />
              ساخت کلید جدید
            </Button>
          </Card>
        </div>
      ) : (
        <Card className="overflow-hidden p-0">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-900">
                <th className="text-right text-xs font-medium text-gray-500 dark:text-gray-400 px-4 py-3">نام</th>
                <th className="text-right text-xs font-medium text-gray-500 dark:text-gray-400 px-4 py-3">پیشوند</th>
                <th className="text-right text-xs font-medium text-gray-500 dark:text-gray-400 px-4 py-3">تاریخ ساخت</th>
                <th className="text-right text-xs font-medium text-gray-500 dark:text-gray-400 px-4 py-3">آخرین استفاده</th>
                <th className="text-right text-xs font-medium text-gray-500 dark:text-gray-400 px-4 py-3">وضعیت</th>
                <th className="text-right text-xs font-medium text-gray-500 dark:text-gray-400 px-4 py-3">عملیات</th>
              </tr>
            </thead>
            <tbody>
              {keys.map((key) => (
                <tr key={key.id} className="border-b border-gray-100 dark:border-gray-800 last:border-0 hover:bg-gray-50 dark:hover:bg-gray-900/50">
                  <td className="px-4 py-3 text-sm font-medium">{key.name}</td>
                  <td className="px-4 py-3"><code className="text-xs font-mono text-gray-500 dark:text-gray-400 bg-gray-100 dark:bg-gray-800 px-2 py-0.5 rounded" dir="ltr">{key.prefix}...</code></td>
                  <td className="px-4 py-3 text-xs text-gray-500 dark:text-gray-400">{key.createdAt}</td>
                  <td className="px-4 py-3 text-xs text-gray-500 dark:text-gray-400">{key.lastUsed}</td>
                  <td className="px-4 py-3">
                    <Badge variant={key.status === "active" ? "success" : "danger"}>
                      {key.status === "active" ? "فعال" : "غیرفعال"}
                    </Badge>
                  </td>
                  <td className="px-4 py-3">
                    {key.status === "active" && (
                      <button onClick={() => revokeKey(key.id)} className="text-red-500 hover:text-red-600 cursor-pointer">
                        <Trash2 className="w-4 h-4" />
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}
    </div>
  );
}
