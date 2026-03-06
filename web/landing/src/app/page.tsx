export default function Home() {
  return (
    <div className="min-h-screen bg-white">
      {/* هدر */}
      <header className="border-b border-gray-100">
        <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-8">
            <span className="text-2xl font-bold text-dana-700">دانا</span>
            <div className="hidden md:flex gap-6 text-sm text-gray-600">
              <a href="/docs" className="hover:text-dana-600">مستندات</a>
              <a href="/pricing" className="hover:text-dana-600">تعرفه‌ها</a>
              <a href="/playground" className="hover:text-dana-600">آزمایشگاه</a>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <a href="/dashboard" className="text-sm text-gray-600 hover:text-dana-600">
              داشبورد
            </a>
            <a
              href="/dashboard/keys"
              className="bg-dana-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-dana-700 transition"
            >
              دریافت کلید API
            </a>
          </div>
        </nav>
      </header>

      {/* بخش اصلی (Hero) */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 md:py-32">
        <div className="text-center">
          <div className="inline-block bg-dana-50 text-dana-700 text-sm font-medium px-4 py-1.5 rounded-full mb-6">
            هوش مصنوعی پیشرفته · زیرساخت ایرانی
          </div>
          <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6 leading-tight">
            مدل‌های هوش مصنوعی
            <br />
            <span className="text-dana-600">با کیفیت جهانی</span>
          </h1>
          <p className="text-lg md:text-xl text-gray-600 max-w-2xl mx-auto mb-10">
            دسترسی به Qwen3-235B-MoE از طریق API سازگار با OpenAI.
            بهینه‌سازی شده با رمزگشایی حدسی و بارگذاری هوشمند خبرگان.
            ۱۰۰٪ حاکمیت داده.
          </p>
          <div className="flex gap-4 justify-center">
            <a
              href="/dashboard/keys"
              className="bg-dana-600 text-white px-8 py-3 rounded-lg font-medium hover:bg-dana-700 transition text-lg"
            >
              شروع رایگان
            </a>
            <a
              href="/docs"
              className="border border-gray-300 text-gray-700 px-8 py-3 rounded-lg font-medium hover:bg-gray-50 transition text-lg"
            >
              مستندات API
            </a>
          </div>
        </div>

        {/* نمونه کد */}
        <div className="mt-16 max-w-3xl mx-auto">
          <div className="bg-gray-900 rounded-xl overflow-hidden shadow-2xl">
            <div className="flex items-center gap-2 px-4 py-3 bg-gray-800">
              <div className="flex gap-1.5">
                <div className="w-3 h-3 rounded-full bg-red-500" />
                <div className="w-3 h-3 rounded-full bg-yellow-500" />
                <div className="w-3 h-3 rounded-full bg-green-500" />
              </div>
              <span className="text-gray-400 text-xs mr-2">Python</span>
            </div>
            <pre className="p-6 text-sm text-gray-300 overflow-x-auto" dir="ltr">
              <code>{`from openai import OpenAI

client = OpenAI(
    base_url="https://api.dana.ir/v1",
    api_key="dk-..."  # کلید API دانا
)

response = client.chat.completions.create(
    model="qwen3-235b-moe",
    messages=[
        {"role": "user", "content": "یک تابع مرتب‌سازی سریع بنویس"}
    ],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content, end="")`}</code>
            </pre>
          </div>
        </div>
      </section>

      {/* ویژگی‌ها */}
      <section className="bg-gray-50 py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-center text-gray-900 mb-12">
            چرا دانا؟
          </h2>
          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                title: "سازگار با OpenAI",
                desc: "بدون تغییر کد، فقط base_url را عوض کنید. SDK‌های Python و JavaScript پشتیبانی می‌شوند.",
                icon: "🔌",
              },
              {
                title: "سرعت بالا",
                desc: "رمزگشایی حدسی (Speculative Decoding) سرعت تولید توکن را تا ۷۰٪ افزایش می‌دهد.",
                icon: "⚡",
              },
              {
                title: "حاکمیت داده",
                desc: "تمام داده‌ها روی زیرساخت ایرانی پردازش می‌شوند. هیچ داده‌ای به خارج ارسال نمی‌شود.",
                icon: "🛡️",
              },
              {
                title: "پشتیبانی از فارسی",
                desc: "مدل‌های چندزبانه با عملکرد عالی در زبان فارسی برای کدنویسی و پاسخگویی.",
                icon: "🇮🇷",
              },
              {
                title: "استریم بلادرنگ",
                desc: "پاسخ‌ها به صورت استریم (SSE) ارسال می‌شوند. WebSocket نیز پشتیبانی می‌شود.",
                icon: "📡",
              },
              {
                title: "مقرون‌به‌صرفه",
                desc: "پلن رایگان برای شروع. پلن حرفه‌ای با نرخ رقابتی نسبت به سرویس‌های خارجی.",
                icon: "💰",
              },
            ].map((feature, i) => (
              <div
                key={i}
                className="bg-white p-6 rounded-xl border border-gray-100 hover:shadow-lg transition"
              >
                <div className="text-3xl mb-4">{feature.icon}</div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  {feature.title}
                </h3>
                <p className="text-gray-600 text-sm leading-relaxed">
                  {feature.desc}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* عملکرد */}
      <section className="py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-center text-gray-900 mb-12">
            مشخصات فنی
          </h2>
          <div className="grid md:grid-cols-4 gap-6 text-center">
            {[
              { value: "۲۳۵B", label: "پارامتر مدل" },
              { value: "~۱۵", label: "توکن در ثانیه" },
              { value: "۳۲K", label: "طول متن ورودی" },
              { value: "۹۹.۵٪", label: "آپتایم" },
            ].map((stat, i) => (
              <div key={i} className="bg-dana-50 p-8 rounded-xl">
                <div className="text-3xl md:text-4xl font-bold text-dana-700 mb-2">
                  {stat.value}
                </div>
                <div className="text-gray-600">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* تعرفه‌ها */}
      <section className="bg-gray-50 py-20" id="pricing">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-center text-gray-900 mb-4">
            تعرفه‌ها
          </h2>
          <p className="text-center text-gray-600 mb-12">
            با پلن رایگان شروع کنید. هر زمان که آماده بودید ارتقا دهید.
          </p>
          <div className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto">
            {[
              {
                name: "رایگان",
                price: "۰",
                features: [
                  "۱,۰۰۰ توکن در روز",
                  "۵ درخواست در دقیقه",
                  "مدل qwen3-235b-moe",
                  "پشتیبانی انجمن",
                ],
                cta: "شروع رایگان",
                highlight: false,
              },
              {
                name: "حرفه‌ای",
                price: "۴۹۹,۰۰۰",
                unit: "تومان/ماه",
                features: [
                  "۱۰۰,۰۰۰ توکن در روز",
                  "۶۰ درخواست در دقیقه",
                  "مدل qwen3-235b-moe",
                  "پشتیبانی ایمیل",
                  "اولویت در صف",
                ],
                cta: "شروع دوره آزمایشی",
                highlight: true,
              },
              {
                name: "سازمانی",
                price: "تماس",
                features: [
                  "توکن نامحدود",
                  "۶۰۰ درخواست در دقیقه",
                  "SLA اختصاصی",
                  "پشتیبانی ۲۴/۷",
                  "اولویت بالا",
                  "مدل اختصاصی",
                ],
                cta: "تماس با ما",
                highlight: false,
              },
            ].map((plan, i) => (
              <div
                key={i}
                className={`p-8 rounded-xl border ${
                  plan.highlight
                    ? "border-dana-500 bg-white shadow-xl scale-105"
                    : "border-gray-200 bg-white"
                }`}
              >
                {plan.highlight && (
                  <div className="text-dana-600 text-xs font-medium mb-2">
                    پرطرفدار
                  </div>
                )}
                <h3 className="text-xl font-bold text-gray-900 mb-2">
                  {plan.name}
                </h3>
                <div className="mb-6">
                  <span className="text-3xl font-bold text-gray-900">
                    {plan.price}
                  </span>
                  {plan.unit && (
                    <span className="text-gray-500 text-sm mr-1">
                      {plan.unit}
                    </span>
                  )}
                </div>
                <ul className="space-y-3 mb-8">
                  {plan.features.map((f, j) => (
                    <li key={j} className="text-sm text-gray-600 flex items-center gap-2">
                      <span className="text-green-500">✓</span>
                      {f}
                    </li>
                  ))}
                </ul>
                <a
                  href="/dashboard/keys"
                  className={`block text-center py-2.5 rounded-lg font-medium transition ${
                    plan.highlight
                      ? "bg-dana-600 text-white hover:bg-dana-700"
                      : "border border-gray-300 text-gray-700 hover:bg-gray-50"
                  }`}
                >
                  {plan.cta}
                </a>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* فوتر */}
      <footer className="border-t border-gray-100 py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid md:grid-cols-4 gap-8">
            <div>
              <h4 className="text-lg font-bold text-dana-700 mb-4">دانا</h4>
              <p className="text-sm text-gray-600">
                پلتفرم هوش مصنوعی ایرانی. مدل‌های پیشرفته با زیرساخت بومی.
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-gray-900 mb-4">محصول</h4>
              <ul className="space-y-2 text-sm text-gray-600">
                <li><a href="/docs" className="hover:text-dana-600">مستندات API</a></li>
                <li><a href="/pricing" className="hover:text-dana-600">تعرفه‌ها</a></li>
                <li><a href="/playground" className="hover:text-dana-600">آزمایشگاه</a></li>
                <li><a href="/dashboard" className="hover:text-dana-600">داشبورد</a></li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-gray-900 mb-4">توسعه‌دهندگان</h4>
              <ul className="space-y-2 text-sm text-gray-600">
                <li><a href="/docs" className="hover:text-dana-600">شروع سریع</a></li>
                <li><a href="/docs/authentication" className="hover:text-dana-600">احراز هویت</a></li>
                <li><a href="/docs/chat-completions" className="hover:text-dana-600">Chat Completions</a></li>
                <li><a href="/docs/sdk" className="hover:text-dana-600">SDK پایتون</a></li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-gray-900 mb-4">شرکت</h4>
              <ul className="space-y-2 text-sm text-gray-600">
                <li><a href="#" className="hover:text-dana-600">درباره ما</a></li>
                <li><a href="#" className="hover:text-dana-600">تماس</a></li>
                <li><a href="#" className="hover:text-dana-600">حریم خصوصی</a></li>
                <li><a href="#" className="hover:text-dana-600">شرایط استفاده</a></li>
              </ul>
            </div>
          </div>
          <div className="border-t border-gray-100 mt-8 pt-8 text-center text-sm text-gray-500">
            © ۱۴۰۵ دانا. تمامی حقوق محفوظ است.
          </div>
        </div>
      </footer>
    </div>
  );
}
