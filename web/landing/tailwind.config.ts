import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Vazirmatn", "Tahoma", "Arial", "sans-serif"],
      },
      colors: {
        dana: {
          50: "#f0f7ff",
          100: "#e0efff",
          200: "#b9dfff",
          300: "#7cc4ff",
          400: "#36a5ff",
          500: "#0c87f0",
          600: "#006bcd",
          700: "#0054a6",
          800: "#054889",
          900: "#0a3d71",
          950: "#07264b",
        },
      },
    },
  },
  plugins: [],
};

export default config;
