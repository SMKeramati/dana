export default function UsagePage() {
  return (
    <div className="min-h-screen bg-gray-50" dir="rtl">
      <header className="bg-white border-b border-gray-200">
        <nav className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-8">
            <a href="/" className="text-2xl font-bold text-dana-700">دانا</a>
            <div className="flex gap-6 text-sm text-gray-600">
              <a href="/dashboard" className="hover:text-dana-600">داشبورد</a>
              <a href="/dashboard/usage" className="text-dana-600 font-medium">مصرف</a>
              <a href="/dashboard/keys" className="hover:text-dana-600">کلیدهای API</a>
              <a href="/dashboard/billing" className="hover:text-dana-600">صورتحساب</a>
            </div>
          </div>
        </nav>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        <h1 className="text-2xl font-bold text-gray-900 mb-8">آمار مصرف</h1>

        {/* فیلتر زمانی */}
        <div className="flex gap-2 mb-6">
          {["۲۴ ساعت", "۷ روز", "۳۰ روز", "۹۰ روز"].map((period, i) => (
            <button
              key={i}
              className={`px-4 py-2 rounded-lg text-sm ${
                i === 0
                  ? "bg-dana-600 text-white"
                  : "bg-white border border-gray-300 text-gray-700 hover:bg-gray-50"
              }`}
            >
              {period}
            </button>
          ))}
        </div>

        {/* خلاصه */}
        <div className="grid md:grid-cols-3 gap-6 mb-8">
          {[
            { label: "کل توکن‌های مصرفی", value: "۰" },
            { label: "تعداد درخواست‌ها", value: "۰" },
            { label: "میانگین تاخیر", value: "— ms" },
          ].map((stat, i) => (
            <div key={i} className="bg-white p-6 rounded-xl border border-gray-200">
              <div className="text-sm text-gray-500 mb-1">{stat.label}</div>
              <div className="text-3xl font-bold text-gray-900">{stat.value}</div>
            </div>
          ))}
        </div>

        {/* نمودار */}
        <div className="bg-white rounded-xl border border-gray-200 p-8">
          <h2 className="font-semibold text-gray-900 mb-6">نمودار مصرف</h2>
          <div className="h-64 flex items-center justify-center text-gray-400">
            <div className="text-center">
              <div className="text-4xl mb-2">📊</div>
              <p>هنوز داده‌ای برای نمایش وجود ندارد.</p>
              <p className="text-sm mt-1">
                اولین درخواست API خود را ارسال کنید تا آمار شروع شود.
              </p>
            </div>
          </div>
        </div>

        {/* جدول درخواست‌ها */}
        <div className="bg-white rounded-xl border border-gray-200 mt-8">
          <div className="p-6 border-b border-gray-100">
            <h2 className="font-semibold text-gray-900">آخرین درخواست‌ها</h2>
          </div>
          <div className="p-12 text-center text-gray-400">
            هنوز درخواستی ثبت نشده.
          </div>
        </div>
      </main>
    </div>
  );
}
