"use client";

import { Header } from "@/components/layout/header";
import { Footer } from "@/components/layout/footer";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import Link from "next/link";
import {
  Zap, Shield, Globe, MessageSquare, Code2, Coins,
  ArrowLeft, Sparkles, Server, Clock, CheckCircle2,
} from "lucide-react";

const features = [
  { icon: Code2, title: "سازگار با OpenAI", desc: "از SDK رسمی OpenAI (Python/JS) استفاده کنید. فقط base_url را تغییر دهید.", color: "text-blue-500 bg-blue-500/10" },
  { icon: Zap, title: "سرعت بالا", desc: "موتور Speculative Decoding اختصاصی. تا ۳ برابر سریع‌تر از استنتاج معمول.", color: "text-amber-500 bg-amber-500/10" },
  { icon: Shield, title: "حاکمیت داده", desc: "تمام پردازش‌ها روی سرورهای GPU داخل ایران. هیچ داده‌ای به خارج ارسال نمی‌شود.", color: "text-emerald-500 bg-emerald-500/10" },
  { icon: MessageSquare, title: "پشتیبانی فارسی", desc: "بهینه‌سازی شده برای زبان فارسی. نتایج طبیعی و روان به زبان فارسی.", color: "text-purple-500 bg-purple-500/10" },
  { icon: Globe, title: "استریم بلادرنگ", desc: "پاسخ‌های استریم SSE. تجربه کاربری سریع و تعاملی مانند ChatGPT.", color: "text-cyan-500 bg-cyan-500/10" },
  { icon: Coins, title: "قیمت مقرون‌به‌صرفه", desc: "پلن رایگان با ۱,۰۰۰ توکن روزانه. پلن حرفه‌ای از ۴۹۹,۰۰۰ تومان/ماه.", color: "text-rose-500 bg-rose-500/10" },
];

const stats = [
  { value: "۲۳۵B", label: "پارامتر مدل", icon: Server },
  { value: "~۱۵", label: "توکن بر ثانیه", icon: Zap },
  { value: "۳۲K", label: "طول ورودی", icon: MessageSquare },
  { value: "۹۹.۵٪", label: "آپتایم", icon: Clock },
];

const plans = [
  {
    name: "رایگان", price: "۰", period: "", desc: "برای آشنایی و تست",
    features: ["۱,۰۰۰ توکن روزانه", "۵ درخواست در دقیقه", "استریم بلادرنگ", "پشتیبانی انجمن"],
    cta: "شروع رایگان", variant: "outline" as const, popular: false,
  },
  {
    name: "حرفه‌ای", price: "۴۹۹,۰۰۰", period: "تومان / ماه", desc: "برای توسعه‌دهندگان و تیم‌ها",
    features: ["۱۰۰,۰۰۰ توکن روزانه", "۶۰ درخواست در دقیقه", "اولویت در صف", "پشتیبانی ایمیل", "SLA ۹۹٪"],
    cta: "خرید پلن", variant: "primary" as const, popular: true,
  },
  {
    name: "سازمانی", price: "تماس بگیرید", period: "", desc: "برای شرکت‌ها و سازمان‌ها",
    features: ["توکن نامحدود", "۶۰۰ درخواست در دقیقه", "مدل اختصاصی", "پشتیبانی ۲۴/۷", "SLA ۹۹.۹٪"],
    cta: "ارتباط با فروش", variant: "outline" as const, popular: false,
  },
];

export default function Home() {
  return (
    <div className="min-h-screen">
      <Header />

      {/* Hero */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 grid-pattern" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[600px] bg-dana-500/10 dark:bg-dana-500/5 rounded-full blur-3xl" />
        <div className="relative max-w-7xl mx-auto px-6 pt-20 pb-28">
          <div className="text-center max-w-4xl mx-auto animate-fade-in">
            <Badge className="mb-6 px-4 py-1.5 text-sm">
              <Sparkles className="w-3.5 h-3.5 ml-1.5" />
              Qwen3-235B-MoE • زیرساخت ایرانی
            </Badge>
            <h1 className="text-4xl sm:text-5xl md:text-6xl font-bold tracking-tight leading-[1.15]">
              مدل‌های هوش مصنوعی
              <br />
              <span className="gradient-text">با کیفیت جهانی</span>
            </h1>
            <p className="mt-6 text-lg text-gray-600 dark:text-gray-400 max-w-2xl mx-auto leading-relaxed">
              دسترسی به Qwen3-235B-MoE از طریق API سازگار با OpenAI.
              زیرساخت GPU داخلی، بدون نیاز به VPN، با حاکمیت کامل داده.
            </p>
            <div className="mt-8 flex items-center justify-center gap-4 flex-wrap">
              <Link href="/login">
                <Button size="lg">
                  شروع رایگان
                  <ArrowLeft className="w-4 h-4 mr-2" />
                </Button>
              </Link>
              <Link href="/docs">
                <Button variant="outline" size="lg">مستندات API</Button>
              </Link>
            </div>
          </div>

          {/* Code Example */}
          <div className="mt-16 max-w-3xl mx-auto animate-slide-up">
            <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-gray-950 dark:bg-gray-900 overflow-hidden shadow-2xl glow-dana">
              <div className="flex items-center gap-2 px-4 py-3 border-b border-gray-800">
                <div className="flex gap-1.5">
                  <div className="w-3 h-3 rounded-full bg-red-500/80" />
                  <div className="w-3 h-3 rounded-full bg-amber-500/80" />
                  <div className="w-3 h-3 rounded-full bg-emerald-500/80" />
                </div>
                <span className="text-xs text-gray-500 mr-2 font-mono">main.py</span>
              </div>
              <pre className="p-5 text-sm leading-relaxed overflow-x-auto">
                <code className="text-gray-300">
{`from openai import OpenAI

client = OpenAI(
    base_url="https://api.dana.ir/v1",
    api_key="dk-YOUR_KEY"
)

response = client.chat.completions.create(
    model="qwen3-235b-moe",
    messages=[
        {"role": "user", "content": "سلام! یک تابع فیبوناچی بنویس"}
    ],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content, end="")`}
                </code>
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="py-24 bg-gray-50 dark:bg-gray-900/50">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold">چرا دانا؟</h2>
            <p className="mt-3 text-gray-500 dark:text-gray-400">ابزارهایی که توسعه‌دهندگان ایرانی نیاز دارند</p>
          </div>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-5">
            {features.map((f) => (
              <div key={f.title} className="group p-6 rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 hover:border-dana-300 dark:hover:border-dana-700 hover:shadow-lg hover:shadow-dana-500/5 transition-all duration-300">
                <div className={`w-10 h-10 rounded-xl ${f.color} flex items-center justify-center mb-4`}>
                  <f.icon className="w-5 h-5" />
                </div>
                <h3 className="text-base font-semibold mb-2">{f.title}</h3>
                <p className="text-sm text-gray-500 dark:text-gray-400 leading-relaxed">{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Stats */}
      <section className="py-20 bg-gray-950 relative overflow-hidden">
        <div className="absolute inset-0 grid-pattern opacity-50" />
        <div className="relative max-w-7xl mx-auto px-6">
          <h2 className="text-3xl font-bold text-white text-center mb-12">مشخصات فنی</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            {stats.map((s) => (
              <div key={s.label} className="text-center p-6 rounded-2xl bg-white/5 border border-white/10 backdrop-blur-sm">
                <s.icon className="w-6 h-6 text-dana-400 mx-auto mb-3" />
                <div className="text-3xl font-bold text-white mb-1">{s.value}</div>
                <div className="text-sm text-gray-400">{s.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Pricing */}
      <section className="py-24">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold">تعرفه‌ها</h2>
            <p className="mt-3 text-gray-500 dark:text-gray-400">شروع رایگان، ارتقا بر اساس نیاز</p>
          </div>
          <div className="grid md:grid-cols-3 gap-6 max-w-5xl mx-auto">
            {plans.map((plan) => (
              <div key={plan.name} className={`relative p-8 rounded-2xl border transition-all duration-300 ${plan.popular ? "border-dana-500 bg-dana-50 dark:bg-dana-950/30 shadow-xl shadow-dana-500/10 scale-[1.02]" : "border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 hover:border-gray-300 dark:hover:border-gray-700"}`}>
                {plan.popular && <div className="absolute -top-3 right-6"><Badge className="px-3 py-1">محبوب‌ترین</Badge></div>}
                <h3 className="text-xl font-bold">{plan.name}</h3>
                <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">{plan.desc}</p>
                <div className="mt-6 mb-6">
                  <span className="text-4xl font-bold">{plan.price}</span>
                  {plan.period && <span className="text-sm text-gray-500 dark:text-gray-400 mr-2">{plan.period}</span>}
                </div>
                <Link href={plan.name === "سازمانی" ? "#" : "/login"}>
                  <Button variant={plan.variant} className="w-full">{plan.cta}</Button>
                </Link>
                <ul className="mt-6 space-y-3">
                  {plan.features.map((f) => (
                    <li key={f} className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                      <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0" />
                      {f}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 bg-gradient-to-b from-dana-600 to-dana-800 relative overflow-hidden">
        <div className="absolute inset-0 grid-pattern opacity-20" />
        <div className="relative max-w-3xl mx-auto px-6 text-center">
          <h2 className="text-3xl font-bold text-white">همین الان شروع کنید</h2>
          <p className="mt-4 text-dana-100">با یک کلید API رایگان، اولین درخواست خود را در کمتر از ۳۰ ثانیه ارسال کنید.</p>
          <div className="mt-8">
            <Link href="/login">
              <Button size="lg" className="!bg-white !text-dana-700 hover:!bg-gray-100 shadow-xl">
                ساخت حساب رایگان
                <ArrowLeft className="w-4 h-4 mr-2" />
              </Button>
            </Link>
          </div>
          <p className="mt-4 text-sm text-dana-200">بدون نیاز به کارت بانکی</p>
        </div>
      </section>

      <Footer />
    </div>
  );
}
