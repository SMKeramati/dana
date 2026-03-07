# Dana Platform - Complete UX Redesign Plan

## Executive Summary

Based on full-page screenshot analysis of all 7 pages (landing, pricing, playground, dashboard, keys, usage, billing), this plan outlines a comprehensive redesign to bring Dana to top-tier developer platform UX quality, comparable to Vercel, Linear, Stripe, and OpenAI.

---

## Current State Issues

### Global Issues
1. **No dark mode** - Developer platforms universally offer dark mode (OpenAI, Vercel, GitHub)
2. **No component library** - All UI is hand-written Tailwind; no reusable component system
3. **Flat visual hierarchy** - White-on-white cards with thin borders lack depth
4. **No animations/transitions** - Static pages feel lifeless
5. **No loading skeletons** - Empty states are just text
6. **No auth flow** - Dashboard shows hardcoded `user@example.com`
7. **No responsive mobile design** - Desktop-only layouts
8. **No toast/notification system** - No feedback on user actions
9. **Missing favicon/logo** - Just text "دانا" in the nav

### Landing Page
- Hero has no visual impact (no gradient, illustration, or animation)
- Feature icons are missing (empty cards)
- No social proof section (customer logos, usage stats, testimonials)
- Technical specs are plain number boxes
- No interactive demo/animation showing the API in action
- CTA buttons are standard blue rectangles
- Footer is plain text with no visual treatment

### Pricing Page
- Basic HTML table styling
- FAQ items are static text blocks, not expandable accordions
- No toggle for monthly/annual pricing
- No "popular" badge or visual emphasis on recommended plan
- Missing feature tooltips

### Playground
- Massive whitespace with minimal controls
- No model selector dropdown
- No temperature/top_p/max_tokens controls
- No streaming toggle
- No conversation history (single prompt only)
- No syntax highlighting for code in responses
- No copy button on response
- No token count display

### Dashboard
- Top-only navigation wastes space (should be sidebar for dashboard)
- Stat cards have no visual indicators (no sparklines, progress bars, or color coding)
- Quick start section is text-heavy
- No activity feed or recent events
- No onboarding progress tracker

### API Keys Page
- Empty state is just text (should have illustration + guided action)
- No key permission scoping UI
- No key usage stats per key
- No expiration date display
- No copy-to-clipboard with visual feedback

### Usage Page
- Chart area is an emoji placeholder
- No real chart implementation (need recharts/tremor)
- Time period buttons have no active state styling
- No breakdown by model/endpoint
- No cost estimation alongside token counts
- Request history table is empty with no column headers

### Billing Page
- Three sparse cards with lots of dead space
- No visual plan comparison
- No payment history details
- No usage-based billing projection
- Invoice history has no download links

---

## Redesign Specification

### Phase 1: Design System Foundation

#### 1.1 Install Component Libraries
```bash
npm install @radix-ui/react-* class-variance-authority clsx tailwind-merge
npm install framer-motion
npm install recharts
npm install lucide-react  # Icon library
npm install next-themes   # Dark mode
npm install sonner         # Toast notifications
```

Use **shadcn/ui** approach: copy components into `src/components/ui/` for full control.

#### 1.2 Color System Overhaul
```
Current: Single blue scale (dana-50 to dana-950)
New: Full design token system

Primary:     dana-600 (#006bcd) → Keep, add gradient variants
Accent:      Emerald for success, Amber for warnings, Rose for errors
Surface:
  Light: white / gray-50 / gray-100
  Dark:  gray-950 / gray-900 / gray-800
Border:
  Light: gray-200 (stronger than current gray-100)
  Dark:  gray-800
Text:
  Light: gray-900 primary, gray-500 secondary
  Dark:  gray-50 primary, gray-400 secondary
```

#### 1.3 Typography Scale
```
Display:  text-5xl font-bold tracking-tight (hero headlines)
H1:       text-3xl font-bold
H2:       text-2xl font-semibold
H3:       text-lg font-semibold
Body:     text-sm leading-relaxed
Caption:  text-xs text-gray-500
Mono:     font-mono text-sm (code blocks, API keys)
```

#### 1.4 Spacing & Layout Tokens
```
Page padding:    px-6 lg:px-8
Section gap:     space-y-16 (landing), space-y-6 (dashboard)
Card padding:    p-6
Card radius:     rounded-xl
Card shadow:     shadow-sm hover:shadow-md transition-shadow
Max widths:      max-w-7xl (landing), max-w-6xl (dashboard content)
```

#### 1.5 Dark Mode
- Use `next-themes` with system preference detection
- Toggle button in header (sun/moon icon)
- All colors via CSS variables: `--background`, `--foreground`, `--card`, `--border`

---

### Phase 2: Landing Page Redesign

#### 2.1 Navigation
**Current:** Plain flex row with text links
**New:**
- Sticky/blurred backdrop header (`backdrop-blur-xl bg-white/80 dark:bg-gray-950/80`)
- Logo: Custom SVG mark + "دانا" text
- Nav links with subtle hover underline animation
- "شروع رایگان" CTA button with gradient: `bg-gradient-to-l from-dana-600 to-dana-500`
- GitHub stars badge (social proof)
- Dark mode toggle icon

#### 2.2 Hero Section
**Current:** Centered text with code block below
**New:**
- Split layout: Text left (RTL: right), animated terminal right
- Gradient text for headline: `bg-gradient-to-l from-dana-400 to-dana-600 bg-clip-text text-transparent`
- Animated typing effect in the terminal showing API call + streaming response
- Pill badge above headline: "Qwen3-235B-MoE • زیرساخت ایرانی" with subtle glow
- Two CTA buttons: Primary gradient + Secondary outline with arrow icon
- Subtle grid/dot pattern background with radial gradient fade
- Floating particles or subtle mesh gradient animation (framer-motion)

#### 2.3 Social Proof Bar
**New section** (doesn't exist currently):
- "مورد اعتماد توسعه‌دهندگان ایرانی" heading
- Animated counter: "۱۰,۰۰۰+ درخواست API در روز"
- Logo cloud of partner/customer companies (placeholder logos initially)
- Subtle scroll animation on viewport entry

#### 2.4 Features Section
**Current:** 6 plain cards in 3-column grid
**New:**
- Bento grid layout (asymmetric sizes, like Linear/Vercel)
- Each card gets:
  - Lucide icon in colored circle
  - Title + description
  - Subtle illustration or mini-demo
  - Hover: lift effect + border glow
- Feature 1 (OpenAI compatibility): Show side-by-side code comparison
- Feature 2 (Speed): Animated speedometer or benchmark bar
- Feature 3 (Data sovereignty): Iran map icon with shield
- Feature 4 (Persian): Show RTL text rendering demo
- Feature 5 (Streaming): Animated text streaming effect
- Feature 6 (Pricing): Mini pricing comparison visual

#### 2.5 Technical Specs
**Current:** 4 boxes with numbers
**New:**
- Dark section with gradient background (`bg-gray-950`)
- Animated count-up numbers on scroll (framer-motion)
- Each stat with icon + label + value
- Subtle glow effects on numbers
- Grid with connecting lines/dots pattern

#### 2.6 Code Example Section
**New section:**
- Tabbed interface: Python / JavaScript / cURL / Dana SDK
- Syntax highlighted with proper theme (use `shiki` or pre-styled)
- "Copy" button with checkmark animation
- "Try in Playground" link button

#### 2.7 Pricing Preview
**Current:** 3 equal cards
**New:**
- Center card (Professional) visually elevated with border glow + "محبوب‌ترین" badge
- Toggle: ماهانه / سالانه (with discount badge "۲۰٪ تخفیف")
- Feature list with checkmarks (green) and dashes (gray)
- Enterprise card with gradient border
- CTA buttons match tier: Free=outline, Pro=filled, Enterprise=gradient

#### 2.8 CTA Section
**New section** before footer:
- Dark gradient background
- "همین الان شروع کنید" headline
- Email input + "ساخت حساب" button (newsletter/signup)
- "بدون نیاز به کارت بانکی" subtext

#### 2.9 Footer
**Current:** Plain 4-column text
**New:**
- Dark background (`bg-gray-950`)
- Logo + brief description
- 4 link columns with hover effects
- Social links with icons (GitHub, Twitter/X, Telegram, LinkedIn)
- "ساخته شده در ایران 🇮🇷" badge
- Newsletter signup mini-form

---

### Phase 3: Dashboard Redesign

#### 3.1 Layout Architecture
**Current:** Top nav only, full-width content
**New:**
- **Sidebar navigation** (collapsible):
  - Logo at top
  - Nav items with icons (Lucide): داشبورد، مصرف، کلیدها، صورتحساب
  - Active item: highlighted background + accent border
  - Collapse to icon-only on small screens
  - User profile section at bottom (avatar, email, logout)
  - Plan badge (رایگان / حرفه‌ای / سازمانی)
- **Top bar:**
  - Breadcrumb navigation
  - Search/command palette trigger (⌘K)
  - Notification bell
  - Dark mode toggle
  - User avatar dropdown

#### 3.2 Dashboard Overview
**Current:** 4 stat cards + Quick Start text
**New:**
- Welcome banner: "سلام، [نام]! 👋" with gradient background, dismissible
- 4 stat cards redesigned:
  - Each with: icon, label, value, sparkline chart, trend arrow (↑↓)
  - Color-coded: tokens=blue, requests=green, latency=amber, plan=purple
  - Click to navigate to detail page
- **Usage chart** (real recharts): Area chart showing last 7 days of token usage
- **Recent activity feed**: Last 5 API calls with timestamp, model, tokens, status
- **Quick start** redesigned as stepper/checklist:
  - [ ] ساخت کلید API
  - [ ] اولین درخواست
  - [ ] بررسی مستندات
  - Progress bar at top
  - Each step expandable with inline code example

#### 3.3 API Keys Page
**Current:** Input field + empty table
**New:**
- "ساخت کلید جدید" button (top-right, primary style)
- **Create key modal** (not inline):
  - Key name input
  - Permission checkboxes (read, write, admin)
  - Expiration selector (30d, 90d, 1y, never)
  - Create button
- **Key created success modal:**
  - Full key displayed once (with warning: "این کلید فقط یک بار نمایش داده می‌شود")
  - Copy button with checkmark animation
  - "متوجه شدم" dismiss button
- **Keys table:**
  - Columns: Name, Prefix (dk-xxx...), Created, Last Used, Status, Actions
  - Row actions: Copy prefix, Revoke (with confirmation dialog)
  - Status badges: Active (green), Revoked (red), Expired (amber)
- **Empty state:**
  - Illustration (key icon with sparkles)
  - "هنوز کلیدی ندارید" heading
  - "اولین کلید API خود را بسازید" subtext
  - CTA button

#### 3.4 Usage Page
**Current:** Period buttons + 3 stats + emoji chart placeholder + empty table
**New:**
- **Time period selector:** Segmented control (not separate buttons)
- **Stats row:** 3 cards with:
  - Icon + label + big number
  - vs. previous period comparison ("+۱۲٪" in green or "-۵٪" in red)
  - Mini sparkline
- **Main chart** (recharts):
  - Area/bar chart with gradient fill
  - Tooltip on hover showing exact values
  - Toggle between: tokens, requests, cost
  - Responsive, animated on data change
- **Breakdown table:**
  - Group by: Model, Endpoint, API Key
  - Columns: Name, Requests, Tokens, Avg Latency, Cost
  - Sortable columns
  - Pagination
- **Export button:** CSV download of usage data

#### 3.5 Billing Page
**Current:** 3 sparse white cards
**New:**
- **Current plan card:**
  - Plan name with colored badge
  - Usage bars: tokens (X/Y used), requests (X/Y used) with progress indicators
  - "ارتقا" button with upgrade benefits preview
  - Next billing date
- **Payment methods card:**
  - List of saved methods (ZarinPal account, card last 4 digits)
  - Add new method button → ZarinPal flow
  - Default method indicator
- **Billing history:**
  - Table: Date, Description, Amount, Status, Invoice PDF
  - Status badges: Paid (green), Pending (amber), Failed (red)
  - Download invoice button (PDF icon)
- **Cost projection card:**
  - Based on current usage trend
  - "با نرخ فعلی، هزینه ماهانه شما حدود ۲۵۰,۰۰۰ تومان خواهد بود"
  - Visual bar comparing current vs. projected

---

### Phase 4: Playground Redesign

**Current:** API key input + textarea + submit button + cURL block
**New: Full-featured playground (like OpenAI Playground)**

#### 4.1 Layout
- **Split panel:** Controls (right, RTL) + Conversation (left, RTL)
- Resizable panels with drag handle

#### 4.2 Control Panel (Right Side)
- **Model selector** dropdown: qwen3-235b-moe (with model info tooltip)
- **System prompt** textarea (collapsible)
- **Parameters:**
  - Temperature slider (0.0 - 2.0)
  - Max tokens input
  - Top P slider
  - Frequency penalty slider
  - Presence penalty slider
- **Streaming toggle** switch
- **Response format:** Text / JSON
- Token count display (prompt tokens / max context)

#### 4.3 Conversation Area (Left Side)
- **Chat-style interface** (not single textarea):
  - Message bubbles: User (right, blue) / Assistant (left, gray)
  - Markdown rendering in assistant messages
  - Code blocks with syntax highlighting + copy button
  - Streaming animation (typing indicator dots → streaming text)
- **Input area at bottom:**
  - Multi-line textarea with auto-resize
  - Send button (Ctrl+Enter)
  - Token count for current input
- **Conversation controls:**
  - Clear conversation button
  - Export conversation (JSON/Markdown)
  - Share conversation (generate link)

#### 4.4 Code Generation Panel
- Toggleable bottom panel: "کد معادل"
- Tabs: Python / JavaScript / cURL / Dana SDK
- Auto-generated from current settings
- Copy button per tab

---

### Phase 5: Authentication Pages

**New pages** (don't exist yet):

#### 5.1 Login Page (`/login`)
- Centered card layout
- Dana logo + "ورود به حساب کاربری"
- Email input
- Password input with show/hide toggle
- "ورود" submit button
- "حساب ندارید؟ ثبت‌نام کنید" link
- "فراموشی رمز عبور" link
- Divider: "یا"
- Social login buttons: Google (if available)

#### 5.2 Register Page (`/register`)
- Similar centered card
- Name, Email, Password, Confirm Password
- Terms checkbox
- "ثبت‌نام" button
- Redirect to dashboard after success

#### 5.3 Forgot Password (`/forgot-password`)
- Email input
- "ارسال لینک بازیابی" button
- Success state with email sent message

---

### Phase 6: Micro-interactions & Polish

#### 6.1 Animations (framer-motion)
- Page transitions: Fade + slide up on route change
- Scroll-triggered reveals: Features, stats, pricing cards
- Number count-up: Technical specs, usage stats
- Hover effects: Card lift, button scale, link underline slide
- Loading: Skeleton screens for all data-dependent content
- Success: Checkmark animation on copy, key creation
- Error: Shake animation on form validation errors

#### 6.2 Toast Notifications (sonner)
- Success: "کلید API با موفقیت ساخته شد" (green)
- Error: "خطا در ارسال درخواست" (red)
- Info: "کلید در کلیپ‌بورد کپی شد" (blue)
- Position: bottom-right (RTL: bottom-left)

#### 6.3 Command Palette (⌘K)
- Global search across: docs, API keys, settings
- Quick actions: Create key, Go to usage, Open playground
- Keyboard navigation support

#### 6.4 Responsive Design
- Mobile: Sidebar collapses to bottom tab bar
- Tablet: Sidebar collapses to icon-only
- Landing: Stack columns, adjust font sizes
- Playground: Stack panels vertically on mobile

---

### Phase 7: Technical Implementation

#### 7.1 Component Structure
```
src/
├── components/
│   ├── ui/              # Base components (shadcn/ui style)
│   │   ├── button.tsx
│   │   ├── card.tsx
│   │   ├── input.tsx
│   │   ├── badge.tsx
│   │   ├── dialog.tsx
│   │   ├── dropdown-menu.tsx
│   │   ├── select.tsx
│   │   ├── slider.tsx
│   │   ├── switch.tsx
│   │   ├── table.tsx
│   │   ├── tabs.tsx
│   │   ├── toast.tsx
│   │   ├── skeleton.tsx
│   │   └── tooltip.tsx
│   ├── layout/
│   │   ├── header.tsx        # Landing header
│   │   ├── footer.tsx        # Landing footer
│   │   ├── sidebar.tsx       # Dashboard sidebar
│   │   ├── topbar.tsx        # Dashboard top bar
│   │   └── mobile-nav.tsx    # Mobile navigation
│   ├── landing/
│   │   ├── hero.tsx
│   │   ├── features.tsx
│   │   ├── social-proof.tsx
│   │   ├── tech-specs.tsx
│   │   ├── code-example.tsx
│   │   ├── pricing-preview.tsx
│   │   └── cta-section.tsx
│   ├── dashboard/
│   │   ├── stat-card.tsx
│   │   ├── usage-chart.tsx
│   │   ├── activity-feed.tsx
│   │   ├── onboarding.tsx
│   │   └── empty-state.tsx
│   ├── playground/
│   │   ├── chat-message.tsx
│   │   ├── control-panel.tsx
│   │   ├── code-panel.tsx
│   │   └── model-selector.tsx
│   └── shared/
│       ├── logo.tsx
│       ├── theme-toggle.tsx
│       ├── copy-button.tsx
│       ├── code-block.tsx
│       └── animated-counter.tsx
├── hooks/
│   ├── use-theme.ts
│   ├── use-clipboard.ts
│   └── use-scroll-animation.ts
├── lib/
│   ├── utils.ts           # cn() helper, formatters
│   └── constants.ts       # Colors, routes, etc.
└── styles/
    └── globals.css        # CSS variables, base styles
```

#### 7.2 New Dependencies
```json
{
  "@radix-ui/react-dialog": "^1.0",
  "@radix-ui/react-dropdown-menu": "^2.0",
  "@radix-ui/react-select": "^2.0",
  "@radix-ui/react-slider": "^1.0",
  "@radix-ui/react-switch": "^1.0",
  "@radix-ui/react-tabs": "^1.0",
  "@radix-ui/react-tooltip": "^1.0",
  "class-variance-authority": "^0.7",
  "clsx": "^2.0",
  "tailwind-merge": "^2.0",
  "framer-motion": "^11.0",
  "recharts": "^2.12",
  "lucide-react": "^0.400",
  "next-themes": "^0.3",
  "sonner": "^1.4",
  "react-markdown": "^9.0",
  "react-syntax-highlighter": "^15.0"
}
```

#### 7.3 Performance Targets
- Lighthouse Performance: >90
- First Contentful Paint: <1.5s
- Largest Contentful Paint: <2.5s
- Cumulative Layout Shift: <0.1
- Bundle size: <200KB initial JS (code-split aggressively)

---

## Implementation Priority

| Priority | Task | Impact | Effort |
|----------|------|--------|--------|
| P0 | Design system + component library setup | Foundation | 2 days |
| P0 | Dark mode support | Table stakes | 1 day |
| P0 | Dashboard sidebar navigation | UX critical | 1 day |
| P1 | Landing hero redesign + animations | First impression | 2 days |
| P1 | Playground chat interface | Core feature | 3 days |
| P1 | Usage charts (recharts) | Dashboard value | 1 day |
| P1 | Auth pages (login/register) | Required for real use | 1 day |
| P2 | Feature bento grid | Visual appeal | 1 day |
| P2 | API keys modal flow + empty states | Polish | 1 day |
| P2 | Billing page redesign | Complete experience | 1 day |
| P2 | Toast notifications | Feedback system | 0.5 day |
| P3 | Command palette (⌘K) | Power user feature | 1 day |
| P3 | Animated counters + scroll reveals | Delight | 0.5 day |
| P3 | Mobile responsive polish | Accessibility | 1 day |
| P3 | Code example tabs on landing | Developer trust | 0.5 day |

**Total estimated effort: ~17 days for full redesign**

---

## Design References

- **Vercel** - Clean, dark-mode-first, excellent animations
- **Linear** - Bento grid features, smooth transitions, glassmorphism
- **Stripe** - Developer documentation, playground, pricing page
- **OpenAI Platform** - Playground interface, API key management, usage dashboard
- **Supabase** - Dashboard layout, dark mode, Persian-friendly (RTL aware)

---

## Success Metrics

1. **Visual Quality:** Comparable to Vercel/Linear landing pages
2. **Developer Experience:** Playground usable for real API testing
3. **RTL Quality:** Perfect Persian typography and layout
4. **Performance:** Lighthouse >90 on all pages
5. **Accessibility:** WCAG 2.1 AA compliance
6. **Dark Mode:** Full coverage with system preference respect
