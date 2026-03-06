export default function PricingPage() {
  return (
    <div className="min-h-screen bg-white" dir="rtl">
      <header className="border-b border-gray-100">
        <nav className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
          <a href="/" className="text-2xl font-bold text-dana-700">دانا</a>
          <div className="flex gap-6 text-sm text-gray-600">
            <a href="/docs" className="hover:text-dana-600">مستندات</a>
            <a href="/playground" className="hover:text-dana-600">آزمایشگاه</a>
            <a href="/dashboard" className="hover:text-dana-600">داشبورد</a>
          </div>
        </nav>
      </header>

      <main className="max-w-5xl mx-auto px-4 py-16">
        <h1 className="text-4xl font-bold text-center mb-4">تعرفه‌ها و قیمت‌گذاری</h1>
        <p className="text-center text-gray-600 mb-12">
          قیمت‌گذاری شفاف بر اساس مصرف. بدون هزینه پنهان.
        </p>

        {/* جدول مقایسه */}
        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr className="border-b-2 border-gray-200">
                <th className="text-right py-4 px-4 text-gray-500 font-normal">ویژگی</th>
                <th className="py-4 px-4 text-center">رایگان</th>
                <th className="py-4 px-4 text-center bg-dana-50 rounded-t-lg font-bold text-dana-700">
                  حرفه‌ای
                </th>
                <th className="py-4 px-4 text-center">سازمانی</th>
              </tr>
            </thead>
            <tbody className="text-sm">
              {[
                ["توکن روزانه", "۱,۰۰۰", "۱۰۰,۰۰۰", "نامحدود"],
                ["درخواست در دقیقه", "۵", "۶۰", "۶۰۰"],
                ["حداکثر طول متن", "۸,۱۹۲", "۳۲,۷۶۸", "۳۲,۷۶۸"],
                ["استریم بلادرنگ", "✓", "✓", "✓"],
                ["اولویت در صف", "—", "✓", "✓✓"],
                ["پشتیبانی", "انجمن", "ایمیل", "۲۴/۷ اختصاصی"],
                ["SLA", "—", "۹۹٪", "۹۹.۹٪"],
                ["مدل اختصاصی", "—", "—", "✓"],
              ].map((row, i) => (
                <tr key={i} className="border-b border-gray-100">
                  <td className="py-3 px-4 text-gray-700 font-medium">{row[0]}</td>
                  <td className="py-3 px-4 text-center text-gray-600">{row[1]}</td>
                  <td className="py-3 px-4 text-center bg-dana-50 text-gray-800">{row[2]}</td>
                  <td className="py-3 px-4 text-center text-gray-600">{row[3]}</td>
                </tr>
              ))}
              <tr>
                <td className="py-6 px-4"></td>
                <td className="py-6 px-4 text-center">
                  <div className="text-2xl font-bold text-gray-900 mb-2">رایگان</div>
                  <a href="/dashboard/keys" className="text-sm text-dana-600 hover:underline">
                    شروع کنید
                  </a>
                </td>
                <td className="py-6 px-4 text-center bg-dana-50 rounded-b-lg">
                  <div className="text-2xl font-bold text-gray-900 mb-1">۴۹۹,۰۰۰</div>
                  <div className="text-xs text-gray-500 mb-2">تومان / ماه</div>
                  <a
                    href="/dashboard/keys"
                    className="inline-block bg-dana-600 text-white px-6 py-2 rounded-lg text-sm hover:bg-dana-700"
                  >
                    خرید پلن
                  </a>
                </td>
                <td className="py-6 px-4 text-center">
                  <div className="text-2xl font-bold text-gray-900 mb-2">تماس بگیرید</div>
                  <a href="#" className="text-sm text-dana-600 hover:underline">
                    ارتباط با فروش
                  </a>
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        {/* سوالات متداول */}
        <section className="mt-20">
          <h2 className="text-2xl font-bold text-center mb-8">سوالات متداول</h2>
          <div className="space-y-6 max-w-3xl mx-auto">
            {[
              {
                q: "آیا می‌توانم بدون پرداخت از دانا استفاده کنم؟",
                a: "بله! پلن رایگان شامل ۱,۰۰۰ توکن در روز و ۵ درخواست در دقیقه است. برای شروع فقط یک حساب کاربری بسازید.",
              },
              {
                q: "API دانا با OpenAI سازگار است؟",
                a: "بله. شما می‌توانید از SDK رسمی OpenAI (Python یا JavaScript) استفاده کنید و فقط base_url را به api.dana.ir تغییر دهید.",
              },
              {
                q: "داده‌های من کجا پردازش می‌شوند؟",
                a: "تمام پردازش‌ها روی سرورهای GPU داخل ایران انجام می‌شود. هیچ داده‌ای به خارج از کشور ارسال نمی‌شود.",
              },
              {
                q: "پرداخت چگونه انجام می‌شود؟",
                a: "از طریق درگاه‌های پرداخت داخلی (زرین‌پال/آیدی‌پی). پرداخت ریالی و بدون نیاز به ارز خارجی.",
              },
            ].map((faq, i) => (
              <div key={i} className="border border-gray-200 rounded-lg p-6">
                <h3 className="font-semibold text-gray-900 mb-2">{faq.q}</h3>
                <p className="text-sm text-gray-600 leading-relaxed">{faq.a}</p>
              </div>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}
