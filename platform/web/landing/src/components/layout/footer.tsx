import Link from "next/link";
import { Logo } from "./logo";

const footerLinks = [
  {
    title: "محصول",
    links: [
      { label: "تعرفه‌ها", href: "/pricing" },
      { label: "آزمایشگاه", href: "/playground" },
      { label: "داشبورد", href: "/dashboard" },
    ],
  },
  {
    title: "توسعه‌دهندگان",
    links: [
      { label: "مستندات API", href: "/docs" },
      { label: "SDK پایتون", href: "/docs" },
      { label: "نمونه کدها", href: "/docs" },
    ],
  },
  {
    title: "شرکت",
    links: [
      { label: "درباره ما", href: "#" },
      { label: "بلاگ", href: "#" },
      { label: "تماس", href: "#" },
    ],
  },
];

export function Footer() {
  return (
    <footer className="border-t border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-950">
      <div className="max-w-7xl mx-auto px-6 py-16">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
          <div className="col-span-2 md:col-span-1">
            <Logo />
            <p className="mt-4 text-sm text-gray-500 dark:text-gray-400 leading-relaxed">
              پلتفرم هوش مصنوعی ایرانی با زیرساخت داخلی و کیفیت جهانی.
            </p>
            <div className="mt-4 flex items-center gap-2">
              <span className="text-xs text-gray-400 dark:text-gray-500">ساخته شده در ایران</span>
              <span>🇮🇷</span>
            </div>
          </div>
          {footerLinks.map((group) => (
            <div key={group.title}>
              <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-4">{group.title}</h3>
              <ul className="space-y-3">
                {group.links.map((link) => (
                  <li key={link.label}>
                    <Link href={link.href} className="text-sm text-gray-500 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors">
                      {link.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
        <div className="mt-12 pt-8 border-t border-gray-200 dark:border-gray-800 text-center">
          <p className="text-xs text-gray-400 dark:text-gray-500">
            © ۱۴۰۴ دانا. تمامی حقوق محفوظ است. دانشبنیان - تحقیق و توسعه داخلی
          </p>
        </div>
      </div>
    </footer>
  );
}
