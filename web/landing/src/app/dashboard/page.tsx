export default function DashboardPage() {
  return (
    <div className="min-h-screen bg-gray-50" dir="rtl">
      <header className="bg-white border-b border-gray-200">
        <nav className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-8">
            <a href="/" className="text-2xl font-bold text-dana-700">دانا</a>
            <div className="flex gap-6 text-sm text-gray-600">
              <a href="/dashboard" className="text-dana-600 font-medium">داشبورد</a>
              <a href="/dashboard/usage" className="hover:text-dana-600">مصرف</a>
              <a href="/dashboard/keys" className="hover:text-dana-600">کلیدهای API</a>
              <a href="/dashboard/billing" className="hover:text-dana-600">صورتحساب</a>
            </div>
          </div>
          <div className="text-sm text-gray-500">user@example.com</div>
        </nav>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        <h1 className="text-2xl font-bold text-gray-900 mb-8">داشبورد</h1>

        {/* کارت‌های آمار */}
        <div className="grid md:grid-cols-4 gap-6 mb-8">
          {[
            { label: "توکن مصرفی امروز", value: "۰", max: "۱,۰۰۰" },
            { label: "درخواست‌های امروز", value: "۰", max: "" },
            { label: "میانگین تاخیر", value: "—", max: "ms" },
            { label: "پلن فعلی", value: "رایگان", max: "" },
          ].map((stat, i) => (
            <div key={i} className="bg-white p-6 rounded-xl border border-gray-200">
              <div className="text-sm text-gray-500 mb-1">{stat.label}</div>
              <div className="text-2xl font-bold text-gray-900">
                {stat.value}
                {stat.max && (
                  <span className="text-sm font-normal text-gray-400 mr-1">
                    / {stat.max}
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* شروع سریع */}
        <div className="bg-white rounded-xl border border-gray-200 p-8">
          <h2 className="text-xl font-bold text-gray-900 mb-4">شروع سریع</h2>
          <div className="space-y-6">
            <div>
              <h3 className="font-medium text-gray-800 mb-2">
                ۱. یک کلید API بسازید
              </h3>
              <p className="text-sm text-gray-600 mb-2">
                به صفحه{" "}
                <a href="/dashboard/keys" className="text-dana-600 hover:underline">
                  کلیدهای API
                </a>{" "}
                بروید و یک کلید جدید بسازید.
              </p>
            </div>

            <div>
              <h3 className="font-medium text-gray-800 mb-2">
                ۲. اولین درخواست خود را ارسال کنید
              </h3>
              <div className="bg-gray-900 rounded-lg p-4" dir="ltr">
                <pre className="text-sm text-gray-300">
{`from openai import OpenAI

client = OpenAI(
    base_url="https://api.dana.ir/v1",
    api_key="dk-YOUR_KEY"
)

response = client.chat.completions.create(
    model="qwen3-235b-moe",
    messages=[
        {"role": "user", "content": "سلام! یک تابع فیبوناچی بنویس"}
    ]
)
print(response.choices[0].message.content)`}
                </pre>
              </div>
            </div>

            <div>
              <h3 className="font-medium text-gray-800 mb-2">
                ۳. مستندات را بخوانید
              </h3>
              <p className="text-sm text-gray-600">
                برای اطلاعات بیشتر،{" "}
                <a href="/docs" className="text-dana-600 hover:underline">
                  مستندات کامل API
                </a>{" "}
                را مطالعه کنید.
              </p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
