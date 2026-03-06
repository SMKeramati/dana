"use client";

import { useState } from "react";

interface ApiKey {
  name: string;
  prefix: string;
  created: string;
}

export default function KeysPage() {
  const [keys, setKeys] = useState<ApiKey[]>([]);
  const [newKeyName, setNewKeyName] = useState("");
  const [newKey, setNewKey] = useState<string | null>(null);

  const handleCreate = () => {
    if (!newKeyName.trim()) return;
    // در نسخه واقعی: فراخوانی API auth-service
    const mockKey = `dk-f1_${Math.random().toString(36).substring(2, 26)}`;
    setNewKey(mockKey);
    setKeys((prev) => [
      ...prev,
      {
        name: newKeyName,
        prefix: mockKey.substring(0, 10),
        created: new Date().toLocaleDateString("fa-IR"),
      },
    ]);
    setNewKeyName("");
  };

  return (
    <div className="min-h-screen bg-gray-50" dir="rtl">
      <header className="bg-white border-b border-gray-200">
        <nav className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-8">
            <a href="/" className="text-2xl font-bold text-dana-700">دانا</a>
            <div className="flex gap-6 text-sm text-gray-600">
              <a href="/dashboard" className="hover:text-dana-600">داشبورد</a>
              <a href="/dashboard/usage" className="hover:text-dana-600">مصرف</a>
              <a href="/dashboard/keys" className="text-dana-600 font-medium">کلیدهای API</a>
              <a href="/dashboard/billing" className="hover:text-dana-600">صورتحساب</a>
            </div>
          </div>
        </nav>
      </header>

      <main className="max-w-4xl mx-auto px-4 py-8">
        <h1 className="text-2xl font-bold text-gray-900 mb-8">کلیدهای API</h1>

        {/* ساخت کلید جدید */}
        <div className="bg-white rounded-xl border border-gray-200 p-6 mb-8">
          <h2 className="font-semibold text-gray-900 mb-4">ساخت کلید جدید</h2>
          <div className="flex gap-4">
            <input
              type="text"
              value={newKeyName}
              onChange={(e) => setNewKeyName(e.target.value)}
              placeholder="نام کلید (مثلاً: پروژه اصلی)"
              className="flex-1 border border-gray-300 rounded-lg px-4 py-2 text-sm focus:ring-2 focus:ring-dana-500"
            />
            <button
              onClick={handleCreate}
              className="bg-dana-600 text-white px-6 py-2 rounded-lg text-sm font-medium hover:bg-dana-700"
            >
              ایجاد
            </button>
          </div>

          {newKey && (
            <div className="mt-4 bg-green-50 border border-green-200 rounded-lg p-4">
              <p className="text-sm text-green-800 font-medium mb-2">
                کلید شما ساخته شد! این کلید فقط یکبار نمایش داده می‌شود:
              </p>
              <code
                className="block bg-white border border-green-300 rounded px-3 py-2 text-sm font-mono"
                dir="ltr"
              >
                {newKey}
              </code>
              <button
                onClick={() => {
                  navigator.clipboard.writeText(newKey);
                }}
                className="mt-2 text-sm text-green-700 hover:underline"
              >
                کپی کلید
              </button>
            </div>
          )}
        </div>

        {/* لیست کلیدها */}
        <div className="bg-white rounded-xl border border-gray-200">
          <div className="p-6 border-b border-gray-100">
            <h2 className="font-semibold text-gray-900">کلیدهای فعال</h2>
          </div>
          {keys.length === 0 ? (
            <div className="p-12 text-center text-gray-500">
              هنوز کلید API ساخته نشده. اولین کلید خود را بسازید.
            </div>
          ) : (
            <table className="w-full">
              <thead>
                <tr className="text-sm text-gray-500 border-b border-gray-100">
                  <th className="text-right py-3 px-6">نام</th>
                  <th className="text-right py-3 px-6">پیشوند</th>
                  <th className="text-right py-3 px-6">تاریخ ساخت</th>
                  <th className="text-right py-3 px-6">عملیات</th>
                </tr>
              </thead>
              <tbody>
                {keys.map((key, i) => (
                  <tr key={i} className="border-b border-gray-50">
                    <td className="py-3 px-6 text-sm">{key.name}</td>
                    <td className="py-3 px-6 text-sm font-mono" dir="ltr">
                      {key.prefix}...
                    </td>
                    <td className="py-3 px-6 text-sm text-gray-500">{key.created}</td>
                    <td className="py-3 px-6">
                      <button className="text-sm text-red-600 hover:underline">
                        حذف
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </main>
    </div>
  );
}
