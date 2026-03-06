"use client";

import { Header } from "@/components/layout/header";
import { Footer } from "@/components/layout/footer";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import Link from "next/link";
import { Check, Minus, ChevronDown, ChevronUp } from "lucide-react";
import { useState } from "react";

const tiers = ["رایگان", "حرفه‌ای", "سازمانی"];

const comparisonRows = [
  { feature: "توکن روزانه", values: ["۱,۰۰۰", "۱۰۰,۰۰۰", "نامحدود"] },
  { feature: "درخواست در دقیقه", values: ["۵", "۶۰", "۶۰۰"] },
  { feature: "حداکثر طول متن", values: ["۸,۱۹۲", "۳۲,۷۶۸", "۳۲,۷۶۸"] },
  { feature: "استریم بلادرنگ", values: [true, true, true] },
  { feature: "اولویت در صف", values: [false, true, true] },
  { feature: "پشتیبانی", values: ["انجمن", "ایمیل", "۲۴/۷ اختصاصی"] },
  { feature: "SLA", values: [false, "۹۹٪", "۹۹.۹٪"] },
  { feature: "مدل اختصاصی", values: [false, false, true] },
];

const faqs = [
  { q: "آیا می‌توانم بدون پرداخت از دانا استفاده کنم؟", a: "بله! پلن رایگان شامل ۱,۰۰۰ توکن در روز و ۵ درخواست در دقیقه است. برای شروع فقط یک حساب کاربری بسازید." },
  { q: "API دانا با OpenAI سازگار است؟", a: "بله. شما می‌توانید از SDK رسمی OpenAI (JavaScript و Python) استفاده کنید و فقط base_url را به api.dana.ir تغییر دهید." },
  { q: "داده‌های من کجا پردازش می‌شوند؟", a: "تمام پردازش‌ها روی سرورهای GPU داخل ایران انجام می‌شود. هیچ داده‌ای به خارج از کشور ارسال نمی‌شود." },
  { q: "پرداخت چگونه انجام می‌شود؟", a: "از طریق درگاه‌های پرداخت داخلی (زرین‌پال/آیدی‌پی). پرداخت ریالی و بدون نیاز به ارز خارجی." },
];

export default function PricingPage() {
  const [openFaq, setOpenFaq] = useState<number | null>(null);

  return (
    <div className="min-h-screen">
      <Header />

      <section className="py-20">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-16">
            <h1 className="text-4xl font-bold">تعرفه‌ها و قیمت‌گذاری</h1>
            <p className="mt-4 text-gray-500 dark:text-gray-400">قیمت‌گذاری شفاف بر اساس مصرف. بدون هزینه پنهان.</p>
          </div>

          {/* Comparison Table */}
          <div className="max-w-4xl mx-auto mb-20">
            <div className="rounded-2xl border border-gray-200 dark:border-gray-800 overflow-hidden">
              {/* Header */}
              <div className="grid grid-cols-4 bg-gray-50 dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800">
                <div className="p-4 text-sm font-semibold text-gray-500 dark:text-gray-400">ویژگی</div>
                {tiers.map((tier, i) => (
                  <div key={tier} className={`p-4 text-center ${i === 1 ? "bg-dana-50 dark:bg-dana-950/30" : ""}`}>
                    <div className="font-bold text-base">{tier}</div>
                    {i === 1 && <Badge className="mt-1 text-[10px]">محبوب‌ترین</Badge>}
                  </div>
                ))}
              </div>
              {/* Rows */}
              {comparisonRows.map((row, ri) => (
                <div key={row.feature} className={`grid grid-cols-4 border-b border-gray-100 dark:border-gray-800 last:border-0 ${ri % 2 === 0 ? "bg-white dark:bg-gray-950" : "bg-gray-50/50 dark:bg-gray-900/50"}`}>
                  <div className="p-4 text-sm font-medium text-gray-700 dark:text-gray-300">{row.feature}</div>
                  {row.values.map((val, i) => (
                    <div key={i} className={`p-4 text-center text-sm ${i === 1 ? "bg-dana-50/50 dark:bg-dana-950/20" : ""}`}>
                      {val === true ? <Check className="w-5 h-5 text-emerald-500 mx-auto" /> : val === false ? <Minus className="w-5 h-5 text-gray-300 dark:text-gray-600 mx-auto" /> : <span className="text-gray-700 dark:text-gray-300">{val}</span>}
                    </div>
                  ))}
                </div>
              ))}
              {/* Price Row */}
              <div className="grid grid-cols-4 bg-gray-50 dark:bg-gray-900 border-t border-gray-200 dark:border-gray-800">
                <div className="p-6 text-sm font-semibold text-gray-500 dark:text-gray-400">قیمت</div>
                <div className="p-6 text-center">
                  <div className="text-2xl font-bold">رایگان</div>
                  <Link href="/login"><Button variant="outline" size="sm" className="mt-3">شروع کنید</Button></Link>
                </div>
                <div className="p-6 text-center bg-dana-50 dark:bg-dana-950/30">
                  <div className="text-2xl font-bold">۴۹۹,۰۰۰</div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">تومان / ماه</div>
                  <Link href="/login"><Button size="sm" className="mt-3">خرید پلن</Button></Link>
                </div>
                <div className="p-6 text-center">
                  <div className="text-2xl font-bold">تماس بگیرید</div>
                  <Link href="#"><Button variant="outline" size="sm" className="mt-3">ارتباط با فروش</Button></Link>
                </div>
              </div>
            </div>
          </div>

          {/* FAQ */}
          <div className="max-w-3xl mx-auto">
            <h2 className="text-2xl font-bold text-center mb-10">سوالات متداول</h2>
            <div className="space-y-3">
              {faqs.map((faq, i) => (
                <div key={i} className="rounded-xl border border-gray-200 dark:border-gray-800 overflow-hidden">
                  <button
                    onClick={() => setOpenFaq(openFaq === i ? null : i)}
                    className="w-full flex items-center justify-between p-5 text-right hover:bg-gray-50 dark:hover:bg-gray-900 transition-colors cursor-pointer"
                  >
                    <span className="font-medium text-sm">{faq.q}</span>
                    {openFaq === i ? <ChevronUp className="w-4 h-4 text-gray-400 shrink-0" /> : <ChevronDown className="w-4 h-4 text-gray-400 shrink-0" />}
                  </button>
                  {openFaq === i && (
                    <div
                     
                     
                      className="px-5 pb-5"
                    >
                      <p className="text-sm text-gray-500 dark:text-gray-400 leading-relaxed">{faq.a}</p>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
}
