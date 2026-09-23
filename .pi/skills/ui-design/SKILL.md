---
name: ui-design
description: Full light + dark theme UI design system for the AI Agent (数字警察智能体) project. Reference this skill when creating or modifying UI components to ensure visual consistency across both color modes.
created: 2026-03-17
---

# YeZh Design System

数字警察智能体 (AI Agent) 完整 UI 设计规范，涵盖浅色与深色双模式。所有新增/修改的页面和组件必须遵循本规范。

---

## 1. Tech Stack

| Package | Version | Role |
|---------|---------|------|
| Tailwind CSS | 4.2.1 (via `@tailwindcss/vite`) | CSS-first configuration, no `tailwind.config.js` |
| React | 19.2.0 | UI framework |
| React Router | 7.13.1 | Client-side routing |
| React Query | 5.90.21 | Server state management |
| Lucide React | 0.577.0 | Icon library |
| Vite | 7.3.1 | Build tool |
| TypeScript | 5.9.3 | Type checking |

---

## 2. CSS Architecture

### 2.1 Tailwind v4 CSS-First Config

No `tailwind.config.js`. All configuration is in `src/index.css`:

```css
@import "tailwindcss";

/* CRITICAL: Class-based dark mode for Tailwind v4 */
@custom-variant dark (&:where(.dark, .dark *));

@theme {
  --color-bg-primary: #FAF9F6;
  --color-bg-secondary: #F5F0EB;
  --color-bg-tertiary: #EDE8E3;
  --color-accent-blue: #3B82F6;
  --color-accent-cyan: #06B6D4;
  --color-accent-green: #16A34A;
  --color-accent-warning: #EA580C;
  --color-accent-danger: #DC2626;
  --color-accent-purple: #7C3AED;
  --font-sans: 'Noto Sans SC', 'Inter', system-ui, sans-serif;
  --font-heading: 'Inter', 'Noto Sans SC', system-ui, sans-serif;
  --font-mono: 'JetBrains Mono', monospace;
}
```

### 2.2 Layer Structure

```css
@layer base {
  /* body, scrollbar, html defaults */
}

@layer components {
  /* .glass-card, .board-card-3d, status classes */
}

/* Keyframes OUTSIDE layers (unlayered) */
@keyframes float-up { ... }
```

**CRITICAL**: Tailwind v4 default dark mode is `@media(prefers-color-scheme:dark)`. This project uses class-based toggling (`.dark` on `<html>`), so `@custom-variant dark (...)` is **mandatory**. Without it, ALL `dark:` utilities are non-functional.

### 2.3 Dark Mode Implementation

- Hook: `src/hooks/useTheme.ts`
- Three modes: `light` | `dark` | `system`
- Storage key: `aiagent-theme` in localStorage
- Mechanism: Adds/removes `.dark` class on `document.documentElement`
- System mode listens to `matchMedia('(prefers-color-scheme: dark)')` changes

---

## 3. Color Palette

### 3.1 Theme Tokens

| Token | Hex | Usage |
|-------|-----|-------|
| `--color-bg-primary` | `#FAF9F6` | Page background (warm white) |
| `--color-bg-secondary` | `#F5F0EB` | Secondary surface |
| `--color-bg-tertiary` | `#EDE8E3` | Tertiary surface |
| `--color-accent-blue` | `#3B82F6` | Primary brand, selected state, links, AI accent |
| `--color-accent-cyan` | `#06B6D4` | Secondary accent, gradient endpoint |
| `--color-accent-green` | `#16A34A` | Success, completed, positive |
| `--color-accent-warning` | `#EA580C` | Warning, uncertain |
| `--color-accent-danger` | `#DC2626` | Error, critical, negative |
| `--color-accent-purple` | `#7C3AED` | Supplementary accent |

### 3.2 Light Mode Colors

| Usage | Value / Class |
|-------|---------------|
| Page background | `#FAF9F6` (body default) |
| Card background | `#ffffff` (glass-card) |
| Primary text | `#1C1917` (stone-900) / `text-slate-900` |
| Secondary text | `text-slate-500` |
| Tertiary text | `text-slate-400` |
| Disabled text | `text-slate-300` |
| Card border | `#E7E5E4` (stone-200) |
| Card hover border | `#D6D3D1` (stone-300) |
| Dividers | `border-black/5` |
| Input border | `border-slate-200` |
| Light surface | `bg-black/5`, `bg-slate-50/50` |

### 3.3 Dark Mode Colors

| Usage | Value / Class |
|-------|---------------|
| Page background | `#141211` |
| Card background | `#1C1917` |
| Primary text | `#F5F5F4` / `dark:text-slate-100` |
| Secondary text | `dark:text-slate-300` |
| Tertiary text | `dark:text-slate-400` |
| Subdued text | `dark:text-slate-500` |
| Card border | `#3D3835` |
| Card hover border | `#57534E` |
| Dividers | `dark:border-white/[0.08]` |
| Input border | `dark:border-white/[0.1]` ~ `dark:border-white/[0.12]` |
| Light surface | `dark:bg-white/[0.04]` ~ `dark:bg-white/[0.08]` |

### 3.4 Contrast Rules (WCAG AA)

**Minimum contrast ratio: 4.5:1 for normal text.**

| Dark text class | Approx ratio on `#1C1917` | Verdict |
|-----------------|---------------------------|---------|
| `dark:text-slate-600` | ~2.5:1 | FAIL - never use for readable text |
| `dark:text-slate-500` | ~3.8:1 | Borderline - only for decorative/non-essential |
| `dark:text-slate-400` | ~5.5:1 | PASS - tertiary text minimum |
| `dark:text-slate-300` | ~8:1 | PASS - secondary text |
| `dark:text-slate-200` | ~11:1 | PASS - activity text |
| `dark:text-slate-100` | ~14:1 | PASS - primary text, titles, values |

**Dark background opacity rules:**

| Purpose | Light | Dark |
|---------|-------|------|
| Badge background | `bg-{color}-50` | `dark:bg-{color}-500/15` |
| Surface tint | `bg-black/5` | `dark:bg-white/[0.04]` ~ `dark:bg-white/[0.08]` |
| Interactive hover | `bg-black/[0.02]` | `dark:bg-white/[0.02]` ~ `dark:bg-white/[0.06]` |
| Icon container | `bg-slate-100` | `dark:bg-white/[0.08]` |
| AI reason block | `bg-blue-50/50` | `dark:bg-blue-500/[0.06]` |

---

## 4. Typography

### 4.1 Font Stack

```css
--font-sans:    'Noto Sans SC', 'Inter', system-ui, sans-serif
--font-heading: 'Inter', 'Noto Sans SC', system-ui, sans-serif
--font-mono:    'JetBrains Mono', monospace
```

Google Fonts: `Inter` (300-700), `JetBrains Mono` (400-600), `Noto Sans SC` (300-700)

### 4.2 Type Scale

| Usage | Light Classes | Dark Override |
|-------|---------------|--------------|
| Page title | `font-heading text-2xl font-bold text-slate-900` | `dark:text-slate-100` |
| Section heading | `font-heading text-sm font-semibold text-slate-900` | `dark:text-slate-100` |
| Card large value | `font-heading text-2xl font-bold text-slate-900` | `dark:text-slate-100` |
| Body text | `text-sm text-slate-600` | `dark:text-slate-300` |
| Secondary text | `text-sm text-slate-500` | `dark:text-slate-300` |
| Helper text | `text-xs text-slate-500` | `dark:text-slate-400` |
| Tiny label | `text-[10px] text-slate-400` | `dark:text-slate-400` (same) |
| Mono value | `font-mono text-xs` | — |

---

## 5. Layout

### 5.1 Sidebar

| State | Width |
|-------|-------|
| Expanded | `w-60` (240px) |
| Collapsed | `w-[72px]` (72px) |

```
Light: bg-white/70 backdrop-blur-xl border-r border-black/5
Dark:  dark:bg-[#141211]/90 dark:border-white/[0.08]
```

Nav item: `rounded-xl px-3 py-2.5 text-sm font-medium gap-3`

| State | Light | Dark |
|-------|-------|------|
| Active | `bg-accent-blue/10 text-accent-blue` | `dark:bg-blue-500/10 dark:text-blue-400` |
| Inactive | `text-slate-600 hover:bg-black/5` | `dark:text-slate-400 dark:hover:bg-white/[0.06]` |

### 5.2 Top Bar

Height: `h-16` (64px), sticky `z-30`

```
Light: bg-white/60 backdrop-blur-xl border-b border-black/5
Dark:  dark:bg-[#141211]/85 dark:border-white/[0.08]
```

Search input:
```
h-9 rounded-xl border border-black/5 bg-black/[0.02] pl-10 text-sm
Dark: dark:border-white/[0.1] dark:bg-white/[0.04] dark:text-slate-100
      dark:placeholder:text-slate-500
```

### 5.3 Content Area

- Margin: `ml-60` (expanded) / `ml-[72px]` (collapsed)
- Padding: `p-6`
- Transition: `transition-all duration-300`

### 5.4 Grid Patterns

| Pattern | Classes |
|---------|---------|
| KPI cards | `grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4` |
| Two-col + sidebar | `grid grid-cols-1 lg:grid-cols-3 gap-6` |
| Skills grid | `grid grid-cols-1 lg:grid-cols-2 gap-3` |
| Agent profiles | `grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3` |
| Summary stats | `grid grid-cols-2 gap-3` |

---

## 6. Core Components

### 6.1 Glass Card

Class: `.glass-card` — the primary container.

```css
/* Light */
background: #ffffff;
border: 1px solid #E7E5E4;
border-radius: 12px;
box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);

/* Light hover */
border-color: #D6D3D1;
box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);

/* Dark */
background: #1C1917;
border-color: #3D3835;
box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);

/* Dark hover */
border-color: #57534E;
box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
```

Usage: `<div className="glass-card p-4">...</div>`

### 6.2 Board Card 3D

Class: `.board-card-3d` — kanban/board view cards with 3D hover tilt.

```css
border-radius: 14px;
transform: perspective(800px) rotateX(0) rotateY(0) translateZ(0);
transition: transform 0.25s cubic-bezier(0.33, 1, 0.68, 1);

/* Hover */
transform: perspective(800px) rotateX(-1.5deg) rotateY(1.5deg) translateZ(6px) translateY(-3px);

/* Light overlay on hover */
::after { background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, transparent 60%); }
```

Priority background tints:

| Priority | Light Gradient | Dark Gradient |
|----------|---------------|---------------|
| `urgent` | `#FEF2F2 → #FFF5F5 → #fff`, border `#FECACA` | `rgba(220,38,38,0.12) → 0.05 → #1C1917`, border `rgba(220,38,38,0.3)` |
| `important` | `#FFFBEB → #FEF9EE → #fff`, border `#FDE68A` | `rgba(245,158,11,0.12) → 0.05 → #1C1917`, border `rgba(245,158,11,0.3)` |
| `normal` | `#F8FAFC → #FAFBFC → #fff`, border `#E2E8F0` | `rgba(148,163,184,0.08) → 0.03 → #1C1917`, border `rgba(148,163,184,0.18)` |

### 6.3 StatusBadge

Component: `src/components/StatusBadge.tsx`

Base: `inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium`

| Variant | Light | Dark |
|---------|-------|------|
| `pending` | `bg-slate-100 text-slate-600` | `bg-slate-700/50 text-slate-300` |
| `in_progress` | `bg-sky-50 text-sky-600` | `bg-sky-500/15 text-sky-300` |
| `completed` | `bg-emerald-50 text-emerald-600` | `bg-emerald-500/15 text-emerald-300` |
| `postponed` | `bg-stone-100 text-stone-500` | `bg-stone-500/15 text-stone-300` |
| `transferred` | `bg-indigo-50 text-indigo-500` | `bg-indigo-500/15 text-indigo-300` |
| `ignored` | `bg-slate-100 text-slate-500` | `bg-slate-700/50 text-slate-400` |
| `urgent` | `bg-rose-50 text-rose-600` | `bg-rose-500/15 text-rose-300` |
| `important` | `bg-amber-50 text-amber-600` | `bg-amber-500/15 text-amber-300` |
| `normal` | `bg-slate-100 text-slate-600` | `bg-slate-700/50 text-slate-400` |
| `alarm` | `bg-amber-50 text-amber-700` | `bg-amber-500/15 text-amber-300` |
| `case` | `bg-sky-50 text-sky-600` | `bg-sky-500/15 text-sky-300` |
| `review` | `bg-teal-50 text-teal-600` | `bg-teal-500/15 text-teal-300` |
| `center` | `bg-indigo-50 text-indigo-500` | `bg-indigo-500/15 text-indigo-300` |
| `property` | `bg-orange-50 text-orange-500` | `bg-orange-500/15 text-orange-300` |

**Pattern**: Light uses `{color}-50` bg + `{color}-600` text. Dark uses `{color}-500/15` bg + `{color}-300` text.

### 6.4 StatCard

Component: `src/components/StatCard.tsx`

6 color variants: `blue` | `cyan` | `green` | `warning` | `danger` | `purple`

Each variant pattern:
- Light gradient: `from-{color}-500/20 to-{color}-600/5 border-{color}-500/20`
- Dark gradient: `dark:from-{color}-500/15 dark:to-{color}-600/[0.03] dark:border-{color}-500/15`
- Icon: `text-{color}-500 bg-{color}-500/10 rounded-xl p-2.5`
- Value: `font-heading text-2xl font-bold text-slate-900 dark:text-slate-100`
- Title: `text-sm text-slate-500 dark:text-slate-300`

### 6.5 Document Category Colors

| Category | Light | Dark |
|----------|-------|------|
| `legal_document` | `bg-sky-50 text-sky-600` | `bg-sky-500/15 text-sky-300` |
| `approval` | `bg-indigo-50 text-indigo-500` | `bg-indigo-500/15 text-indigo-300` |
| `notification` | `bg-teal-50 text-teal-600` | `bg-teal-500/15 text-teal-300` |
| `record` | `bg-slate-100 text-slate-600` | `bg-white/[0.06] text-slate-300` |
| `report` | `bg-orange-50 text-orange-500` | `bg-orange-500/15 text-orange-300` |

### 6.6 Theme Toggle

Component: `src/components/ThemeToggle.tsx`

Three modes: Light (Sun) | Dark (Moon) | System (Monitor)

Container: `rounded-xl bg-black/5 p-1 dark:border dark:border-white/[0.1] dark:bg-white/[0.05]`

| State | Light | Dark |
|-------|-------|------|
| Active | `bg-white text-accent-blue shadow-sm` | `dark:bg-white/10 dark:text-blue-400` |
| Inactive | `text-slate-500 hover:text-slate-700` | `dark:text-slate-500 dark:hover:text-slate-300` |

### 6.7 AI Status Indicator

Component: `src/components/AIStatusIndicator.tsx`

Dot + label. Text: `text-xs font-medium text-slate-600 dark:text-slate-300`

| Status | Dot Color | Glow |
|--------|-----------|------|
| `idle` | `bg-slate-400` | No |
| `running` | `bg-accent-blue` | Yes (ping animation) |
| `complete` | `bg-accent-green` | No |
| `error` | `bg-accent-danger` | No |

### 6.8 Task Filters

Component: `src/components/TaskFilters.tsx`

Select input:
```
h-8 rounded-lg border border-black/10 bg-white px-3 text-xs text-slate-700
Dark: dark:border-white/[0.12] dark:bg-white/[0.04] dark:text-slate-200
Focus: focus-visible:ring-2 focus-visible:ring-accent-blue/30
```

### 6.9 Buttons

**Primary:**
```
bg-accent-blue text-white rounded-lg px-3 py-1.5 text-xs font-medium
hover:bg-blue-600 transition-colors
```

**Secondary / Ghost:**
```
border border-black/10 text-slate-600 rounded-lg px-3 py-1.5 text-xs font-medium
hover:bg-black/[0.03]
Dark: dark:border-white/[0.12] dark:text-slate-300 dark:hover:bg-white/[0.05]
```

**Filter (active):**
```
bg-accent-blue text-white shadow-sm rounded-lg px-3 py-2 text-xs font-medium
```

**Filter (inactive):**
```
text-slate-500 hover:bg-black/5 rounded-lg px-3 py-2 text-xs font-medium
Dark: dark:hover:bg-white/[0.04]
```

**Icon button:**
```
rounded-xl p-2 text-slate-500 hover:bg-black/5 hover:text-slate-700
Dark: dark:text-slate-400 dark:hover:bg-white/[0.06] dark:hover:text-slate-200
```

---

## 7. Spacing

| Token | Value | Usage |
|-------|-------|-------|
| Card padding | `p-4` ~ `p-6` | 16-24px |
| Grid gap | `gap-3` ~ `gap-6` | 12-24px |
| Section gap | `space-y-6` | 24px between sections |
| List item gap | `space-y-2` ~ `space-y-3` | 8-12px between list items |
| Inline gap | `gap-1.5` ~ `gap-3` | 6-12px inline |

---

## 8. Border Radius

| Size | Class | Value | Usage |
|------|-------|-------|-------|
| Small | `rounded-lg` | 8px | Buttons, badges, inputs |
| Medium | `rounded-xl` | 12px | Cards (glass-card), nav items, icon containers |
| Large | `rounded-2xl` | 16px | Dropdowns, modals |
| Extra | `rounded-[14px]` | 14px | Board cards (board-card-3d) |
| Pill | `rounded-full` | 9999px | Status badges, progress bars, avatars |

---

## 9. Shadows

| Level | Light | Dark |
|-------|-------|------|
| Subtle (card) | `0 1px 3px rgba(0,0,0,0.04)` | `0 1px 3px rgba(0,0,0,0.3)` |
| Elevated (hover) | `0 2px 8px rgba(0,0,0,0.06)` | `0 2px 8px rgba(0,0,0,0.4)` |
| 3D card | Multi-layer: `0 1px 2px`, `0 4px 8px -2px`, `0 8px 16px -4px` | Same with higher opacity |
| 3D hover | Multi-layer + `0 0 0 1px rgba(59,130,246,0.08)` | + `rgba(59,130,246,0.12)` |
| Button | `shadow-sm shadow-accent-blue/25` | — |
| Dropdown | `shadow-lg shadow-black/10` | — |

---

## 10. Animations

### 10.1 Entry Animations (float-up)

```css
@keyframes float-up {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0); }
}
```

| Class | Delay |
|-------|-------|
| `.float-up` | 0s |
| `.float-up-1` | 0.1s |
| `.float-up-2` | 0.2s |
| `.float-up-3` | 0.3s |
| `.float-up-4` | 0.4s |

List item stagger: `style={{ animationDelay: \`${index * 0.05}s\` }}`

### 10.2 Status Animations

| Animation | Duration | Usage |
|-----------|----------|-------|
| `progress-shimmer` | 1.5s infinite | Progress bar fill |
| `priority-pulse` | 2s infinite | Urgent priority indicator |
| `task-reveal` | 0.3s ease-out | Task card slide-in |
| `subtle-pulse` | — | AI analyzing state |

### 10.3 Transitions

- Standard: `transition-all duration-200`
- Smooth: `transition-all duration-300`
- Color only: `transition-colors`
- Body theme: `transition: background-color 0.3s ease, color 0.3s ease`
- 3D card: `transition: transform 0.25s cubic-bezier(0.33, 1, 0.68, 1), box-shadow 0.25s ease`

### 10.4 Reduced Motion

```css
@media (prefers-reduced-motion: reduce) {
  .float-up, .float-up-1, ..., .progress-shimmer, .priority-urgent, .task-reveal {
    animation: none !important;
    opacity: 1 !important;
  }
  .board-card-3d:hover { transform: none !important; }
  * { transition-duration: 0.01ms !important; }
}
```

---

## 11. Scrollbar

```css
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(120, 113, 108, 0.25); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(120, 113, 108, 0.4); }

.dark ::-webkit-scrollbar-thumb { background: rgba(168, 162, 158, 0.25); }
.dark ::-webkit-scrollbar-thumb:hover { background: rgba(168, 162, 158, 0.4); }
```

---

## 12. Icon System

Library: `lucide-react` v0.577.0

### Sizes

| Class | Usage |
|-------|-------|
| `h-2.5 w-2.5` | Badge inline tiny |
| `h-3 w-3` | Inline tiny |
| `h-3.5 w-3.5` | Search/filter/meta |
| `h-4 w-4` | Standard inline |
| `h-4.5 w-4.5` | Agent profile card icon |
| `h-5 w-5` | Card/list item, sidebar nav |
| `h-8 w-8` ~ `h-10 w-10` | Empty state, large indicator |

### Common Icons

| Category | Icons |
|----------|-------|
| Navigation | `LayoutDashboard`, `ListTodo`, `FolderOpen`, `Sparkles`, `Brain`, `Bell` |
| Source | `Siren` (alarm), `FileText` (case), `Search` (review), `Building2` (center), `Package` (property) |
| Case phases | `Siren`, `ClipboardCheck`, `FolderOpen`, `Search`, `HandMetal`, `SendHorizonal` |
| Status | `CheckCircle2`, `Circle`, `Loader2`, `AlertTriangle` |
| Action | `ArrowLeft`, `ArrowRight`, `ChevronRight`, `ChevronDown`, `ChevronUp` |
| Document | `FileSignature`, `FileText`, `FileCheck` |
| People | `User`, `Bot` |
| Time | `Clock`, `Calendar` |
| AI | `Sparkles`, `Bot`, `Zap`, `BarChart3` |
| Security | `Shield`, `Gavel` |

---

## 13. Page Structure Pattern

All pages follow staggered entry:

```tsx
<div className="space-y-6">
  {/* Header */}
  <div className="float-up">
    <h1 className="font-heading text-2xl font-bold text-slate-900 dark:text-slate-100">
      Page Title
    </h1>
    <p className="mt-1 text-sm text-slate-500 dark:text-slate-300">Subtitle</p>
  </div>

  {/* Summary Cards */}
  <div className="float-up-1 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
    <StatCard ... />
  </div>

  {/* Filters */}
  <div className="float-up-2">
    <TaskFilters ... />
  </div>

  {/* Content */}
  <div className="float-up-3 space-y-3">
    {items.map((item, idx) => (
      <div key={item.id} className="glass-card p-4"
           style={{ animationDelay: `${idx * 0.05}s` }}>
        ...
      </div>
    ))}
  </div>
</div>
```

---

## 14. Status Colors (CSS classes in `index.css`)

| Class | Text | Background |
|-------|------|------------|
| `.status-analyzing` | `#D97706` | `rgba(217, 119, 6, 0.08)` |
| `.status-complete` | `#16A34A` | `rgba(22, 163, 74, 0.08)` |
| `.status-warning` | `#EA580C` | `rgba(234, 88, 12, 0.08)` |
| `.status-error` | `#DC2626` | `rgba(220, 38, 38, 0.08)` |

---

## 15. Design Principles

1. **Warm neutrals** — Use stone/warm tones (`#FAF9F6`, `#1C1917`, `#141211`) instead of pure black/white
2. **Subtle depth** — Low-opacity shadows, thin borders, backdrop blur
3. **Progressive disclosure** — Staggered float-up entries, expandable panels
4. **Consistent badge system** — Light: `-50` bg + `-600` text; Dark: `-500/15` bg + `-300` text
5. **3D physicality** — Board cards use perspective transforms and multi-layer shadows
6. **Accessibility** — WCAG AA contrast (4.5:1 minimum), `prefers-reduced-motion` support
7. **Priority visual hierarchy** — Urgent uses warm red tints, important uses amber, normal uses cool slate

---

## 16. File References

| File | Content |
|------|---------|
| `src/index.css` | Theme variables, `@custom-variant dark`, animations, glass-card, board-card-3d, scrollbar |
| `src/hooks/useTheme.ts` | Dark mode management (class-based) |
| `src/components/StatusBadge.tsx` | Status/priority/source badge variants |
| `src/components/StatCard.tsx` | KPI card with 6 color variants |
| `src/components/TaskCard.tsx` | Task card (compact, board, list views) |
| `src/components/Sidebar.tsx` | Navigation sidebar layout |
| `src/components/Layout.tsx` | Main layout, top bar, search |
| `src/components/ThemeToggle.tsx` | Light/Dark/System mode switcher |
| `src/components/AIStatusIndicator.tsx` | AI status dot + label |
| `src/components/TaskFilters.tsx` | Filter dropdowns |
| `src/pages/Dashboard.tsx` | Dashboard page |
| `src/pages/TaskCenter.tsx` | Task list/board view |
| `src/pages/CaseDetail.tsx` | Case detail with phase timeline |
| `src/pages/AgentSkills.tsx` | Agent skills catalog |
| `src/pages/AgentActivity.tsx` | Agent activity timeline |
| `src/pages/Notifications.tsx` | Notification list |
