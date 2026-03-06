import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "دانا - پلتفرم هوش مصنوعی ایرانی",
  description:
    "دسترسی به مدل‌های هوش مصنوعی پیشرفته از طریق API سازگار با OpenAI. زیرساخت ایرانی، حاکمیت داده.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="fa" dir="rtl">
      <body className="antialiased">{children}</body>
    </html>
  );
}
