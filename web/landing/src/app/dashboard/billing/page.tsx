"use client";

import { Card, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import Link from "next/link";
import { Crown, CreditCard, FileText, ArrowUpRight, Zap, Hash } from "lucide-react";

export default function BillingPage() {
  return (
    <div className="p-6 lg:p-8 max-w-4xl">
      <div className="mb-8">
        <h1 className="text-2xl font-bold">صورتحساب و پرداخت</h1>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">پلن، روش پرداخت و فاکتورهای خود را مدیریت کنید</p>
      </div>

      {/* Current Plan */}
      <div>
        <Card className="mb-6">
          <div className="flex items-start justify-between">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 rounded-2xl bg-purple-500/10 flex items-center justify-center">
                <Crown className="w-6 h-6 text-purple-500" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <CardTitle>پلن فعلی</CardTitle>
                  <Badge>رایگان</Badge>
                </div>
                <CardDescription className="mt-1">۱,۰۰۰ توکن/روز • ۵ درخواست/دقیقه</CardDescription>
              </div>
            </div>
            <Link href="/pricing">
              <Button size="sm">
                ارتقا به حرفه‌ای
                <ArrowUpRight className="w-3.5 h-3.5 mr-1.5" />
              </Button>
            </Link>
          </div>

          {/* Usage Bars */}
          <div className="mt-6 grid sm:grid-cols-2 gap-4">
            <div>
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-1.5 text-xs text-gray-500 dark:text-gray-400">
                  <Zap className="w-3.5 h-3.5" />
                  توکن مصرفی امروز
                </div>
                <span className="text-xs font-medium">۰ / ۱,۰۰۰</span>
              </div>
              <div className="h-2 bg-gray-100 dark:bg-gray-800 rounded-full overflow-hidden">
                <div className="h-full bg-blue-500 rounded-full" style={{ width: "0%" }} />
              </div>
            </div>
            <div>
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-1.5 text-xs text-gray-500 dark:text-gray-400">
                  <Hash className="w-3.5 h-3.5" />
                  درخواست‌های این دقیقه
                </div>
                <span className="text-xs font-medium">۰ / ۵</span>
              </div>
              <div className="h-2 bg-gray-100 dark:bg-gray-800 rounded-full overflow-hidden">
                <div className="h-full bg-emerald-500 rounded-full" style={{ width: "0%" }} />
              </div>
            </div>
          </div>
        </Card>
      </div>

      {/* Payment Method */}
      <div>
        <Card className="mb-6">
          <div className="flex items-start gap-4">
            <div className="w-12 h-12 rounded-2xl bg-emerald-500/10 flex items-center justify-center">
              <CreditCard className="w-6 h-6 text-emerald-500" />
            </div>
            <div className="flex-1">
              <CardTitle>روش پرداخت</CardTitle>
              <CardDescription className="mt-1">برای ارتقا به پلن حرفه‌ای یا سازمانی، یک روش پرداخت اضافه کنید.</CardDescription>
              <Button variant="outline" size="sm" className="mt-4">
                <CreditCard className="w-3.5 h-3.5 ml-1.5" />
                افزودن روش پرداخت (زرین‌پال)
              </Button>
            </div>
          </div>
        </Card>
      </div>

      {/* Invoice History */}
      <div>
        <Card>
          <CardTitle className="mb-4">تاریخچه فاکتورها</CardTitle>
          <div className="text-center py-12">
            <div className="w-12 h-12 rounded-2xl bg-gray-100 dark:bg-gray-800 flex items-center justify-center mx-auto mb-3">
              <FileText className="w-6 h-6 text-gray-400 dark:text-gray-500" />
            </div>
            <p className="text-sm text-gray-500 dark:text-gray-400">هنوز فاکتوری صادر نشده.</p>
            <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">پس از خرید پلن، فاکتورها اینجا نمایش داده می‌شوند.</p>
          </div>
        </Card>
      </div>
    </div>
  );
}
