"use client";

import { Card, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { useState } from "react";
import { useApiKeys } from "@/hooks/use-api-keys";
import { Plus, Copy, Check, Trash2, Key } from "lucide-react";

export default function KeysPage() {
  const { keys, loading, createKey, deleteKey } = useApiKeys();
  const [newKeyName, setNewKeyName] = useState("");
  const [showCreate, setShowCreate] = useState(false);
  const [newlyCreatedKey, setNewlyCreatedKey] = useState<string | null>(null);
  const [newlyCreatedName, setNewlyCreatedName] = useState("");
  const [copied, setCopied] = useState(false);
  const [creating, setCreating] = useState(false);

  async function handleCreate() {
    if (!newKeyName.trim()) return;
    setCreating(true);
    const result = await createKey(newKeyName);
    setCreating(false);
    if (result && result.key) {
      setNewlyCreatedKey(result.key);
      setNewlyCreatedName(newKeyName);
      setNewKeyName("");
      setShowCreate(false);
    }
  }

  function copyKey(text: string) {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  async function handleDelete(id: number) {
    await deleteKey(id);
  }

  if (loading) {
    return (
      <div className="p-6 lg:p-8 max-w-4xl">
        <Skeleton className="h-8 w-48 mb-2" />
        <Skeleton className="h-4 w-64 mb-8" />
        <Skeleton className="h-64 w-full rounded-2xl" />
      </div>
    );
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
                onKeyDown={(e) => e.key === "Enter" && handleCreate()}
              />
              <Button onClick={handleCreate} disabled={!newKeyName.trim() || creating}>
                {creating ? "..." : "ایجاد"}
              </Button>
              <Button variant="ghost" onClick={() => setShowCreate(false)}>انصراف</Button>
            </div>
          </Card>
        </div>
      )}

      {/* Newly Created Key Alert */}
      {newlyCreatedKey && (
        <div>
          <Card className="mb-6 border-emerald-300 dark:border-emerald-800 bg-emerald-50 dark:bg-emerald-950/30">
            <div className="flex items-start gap-3">
              <Key className="w-5 h-5 text-emerald-600 dark:text-emerald-400 mt-0.5 shrink-0" />
              <div className="flex-1">
                <p className="text-sm font-semibold text-emerald-800 dark:text-emerald-200">کلید &quot;{newlyCreatedName}&quot; ساخته شد!</p>
                <p className="text-xs text-emerald-600 dark:text-emerald-400 mt-1">این کلید فقط یک بار نمایش داده می‌شود. حتما آن را کپی کنید.</p>
                <div className="mt-3 flex items-center gap-2 bg-white dark:bg-gray-900 rounded-lg border border-emerald-200 dark:border-emerald-800 px-3 py-2">
                  <code className="text-xs font-mono text-gray-700 dark:text-gray-300 flex-1" dir="ltr">{newlyCreatedKey}</code>
                  <button onClick={() => copyKey(newlyCreatedKey)} className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 cursor-pointer">
                    {copied ? <Check className="w-4 h-4 text-emerald-500" /> : <Copy className="w-4 h-4" />}
                  </button>
                </div>
                <button onClick={() => setNewlyCreatedKey(null)} className="mt-3 text-xs text-emerald-600 dark:text-emerald-400 hover:underline cursor-pointer">متوجه شدم، بستن</button>
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
          <div className="overflow-x-auto">
          <table className="w-full min-w-[500px]">
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-900">
                <th className="text-right text-xs font-medium text-gray-500 dark:text-gray-400 px-4 py-3">نام</th>
                <th className="text-right text-xs font-medium text-gray-500 dark:text-gray-400 px-4 py-3">پیشوند</th>
                <th className="text-right text-xs font-medium text-gray-500 dark:text-gray-400 px-4 py-3">تاریخ ساخت</th>
                <th className="text-right text-xs font-medium text-gray-500 dark:text-gray-400 px-4 py-3">آخرین استفاده</th>
                <th className="text-right text-xs font-medium text-gray-500 dark:text-gray-400 px-4 py-3">عملیات</th>
              </tr>
            </thead>
            <tbody>
              {keys.map((key) => (
                <tr key={key.id} className="border-b border-gray-100 dark:border-gray-800 last:border-0 hover:bg-gray-50 dark:hover:bg-gray-900/50">
                  <td className="px-4 py-3 text-sm font-medium">{key.name}</td>
                  <td className="px-4 py-3"><code className="text-xs font-mono text-gray-500 dark:text-gray-400 bg-gray-100 dark:bg-gray-800 px-2 py-0.5 rounded" dir="ltr">{key.prefix}...</code></td>
                  <td className="px-4 py-3 text-xs text-gray-500 dark:text-gray-400">{new Date(key.created_at).toLocaleDateString("fa-IR")}</td>
                  <td className="px-4 py-3 text-xs text-gray-500 dark:text-gray-400">{key.last_used ? new Date(key.last_used).toLocaleDateString("fa-IR") : "هرگز"}</td>
                  <td className="px-4 py-3">
                    <button onClick={() => handleDelete(key.id)} className="text-red-500 hover:text-red-600 cursor-pointer">
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          </div>
        </Card>
      )}
    </div>
  );
}
