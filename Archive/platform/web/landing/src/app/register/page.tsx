"use client";

import { Logo } from "@/components/layout/logo";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import Link from "next/link";
import { useState } from "react";
import { Eye, EyeOff, ArrowLeft } from "lucide-react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/hooks/use-auth";

export default function RegisterPage() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPass, setShowPass] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const router = useRouter();
  const { register } = useAuth();

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError("");
    const result = await register(email, password);
    if (result.ok) {
      router.push("/dashboard");
    } else {
      setError(result.error || "خطا در ثبت‌نام");
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-950 p-6">
      <div className="absolute inset-0 grid-pattern opacity-50" />
      <div
       
       
        className="relative w-full max-w-sm"
      >
        <div className="text-center mb-8">
          <div className="flex justify-center mb-4">
            <Logo />
          </div>
          <h1 className="text-2xl font-bold">ساخت حساب کاربری</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">رایگان شروع کنید. بدون نیاز به کارت بانکی.</p>
        </div>

        <div className="bg-white dark:bg-gray-900 rounded-2xl border border-gray-200 dark:border-gray-800 p-6 shadow-xl shadow-gray-200/50 dark:shadow-gray-900/50">
          <form onSubmit={handleSubmit} className="space-y-4">
            {error && (
              <div className="p-3 rounded-lg bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800 text-sm text-red-600 dark:text-red-400">
                {error}
              </div>
            )}
            <div>
              <label className="text-sm font-medium mb-1.5 block">نام</label>
              <Input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="نام شما"
                required
              />
            </div>
            <div>
              <label className="text-sm font-medium mb-1.5 block">ایمیل</label>
              <Input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                dir="ltr"
                required
              />
            </div>
            <div>
              <label className="text-sm font-medium mb-1.5 block">رمز عبور</label>
              <div className="relative">
                <Input
                  type={showPass ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="حداقل ۸ کاراکتر"
                  dir="ltr"
                  required
                  minLength={8}
                />
                <button
                  type="button"
                  onClick={() => setShowPass(!showPass)}
                  className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 cursor-pointer"
                >
                  {showPass ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>
            <div className="flex items-start gap-2">
              <input type="checkbox" id="terms" required className="mt-1 accent-dana-500" />
              <label htmlFor="terms" className="text-xs text-gray-500 dark:text-gray-400 leading-relaxed">
                با ثبت‌نام، <Link href="#" className="text-dana-600 dark:text-dana-400 hover:underline">شرایط استفاده</Link> و{" "}
                <Link href="#" className="text-dana-600 dark:text-dana-400 hover:underline">حریم خصوصی</Link> را می‌پذیرم.
              </label>
            </div>
            <Button type="submit" className="w-full" disabled={loading}>
              {loading ? "در حال ثبت‌نام..." : "ثبت‌نام"}
              {!loading && <ArrowLeft className="w-4 h-4 mr-1.5" />}
            </Button>
          </form>
        </div>

        <p className="text-center text-sm text-gray-500 dark:text-gray-400 mt-6">
          قبلا ثبت‌نام کرده‌اید؟{" "}
          <Link href="/login" className="text-dana-600 dark:text-dana-400 font-medium hover:underline">وارد شوید</Link>
        </p>
      </div>
    </div>
  );
}
