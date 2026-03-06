export default function BillingPage() {
  return (
    <div className="min-h-screen bg-gray-50" dir="rtl">
      <header className="bg-white border-b border-gray-200">
        <nav className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-8">
            <a href="/" className="text-2xl font-bold text-dana-700">دانا</a>
            <div className="flex gap-6 text-sm text-gray-600">
              <a href="/dashboard" className="hover:text-dana-600">داشبورد</a>
              <a href="/dashboard/usage" className="hover:text-dana-600">مصرف</a>
              <a href="/dashboard/keys" className="hover:text-dana-600">کلیدهای API</a>
              <a href="/dashboard/billing" className="text-dana-600 font-medium">صورتحساب</a>
            </div>
          </div>
        </nav>
      </header>

      <main className="max-w-4xl mx-auto px-4 py-8">
        <h1 className="text-2xl font-bold text-gray-900 mb-8">صورتحساب و پرداخت</h1>

        {/* پلن فعلی */}
        <div className="bg-white rounded-xl border border-gray-200 p-6 mb-8">
          <h2 className="font-semibold text-gray-900 mb-4">پلن فعلی</h2>
          <div className="flex items-center justify-between">
            <div>
              <span className="text-lg font-bold text-gray-900">رایگان</span>
              <span className="text-sm text-gray-500 mr-2">
                ۱,۰۰۰ توکن/روز · ۵ درخواست/دقیقه
              </span>
            </div>
            <a
              href="/pricing"
              className="bg-dana-600 text-white px-6 py-2 rounded-lg text-sm font-medium hover:bg-dana-700"
            >
              ارتقا به حرفه‌ای
            </a>
          </div>
        </div>

        {/* روش پرداخت */}
        <div className="bg-white rounded-xl border border-gray-200 p-6 mb-8">
          <h2 className="font-semibold text-gray-900 mb-4">روش پرداخت</h2>
          <p className="text-sm text-gray-600 mb-4">
            برای ارتقا به پلن حرفه‌ای یا سازمانی، یک روش پرداخت اضافه کنید.
          </p>
          <button className="border border-gray-300 text-gray-700 px-4 py-2 rounded-lg text-sm hover:bg-gray-50">
            افزودن روش پرداخت (زرین‌پال)
          </button>
        </div>

        {/* تاریخچه فاکتورها */}
        <div className="bg-white rounded-xl border border-gray-200">
          <div className="p-6 border-b border-gray-100">
            <h2 className="font-semibold text-gray-900">تاریخچه فاکتورها</h2>
          </div>
          <div className="p-12 text-center text-gray-400">
            هنوز فاکتوری صادر نشده.
          </div>
        </div>
      </main>
    </div>
  );
}
