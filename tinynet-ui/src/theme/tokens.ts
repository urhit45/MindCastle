// ─── THEME TOKENS ─────────────────────────────────────────────────────────────
// Aligned with tinynet-ui/THEME ideas/ (design rationale, motion specs, DESIGN.md).
// Motion timing follows nocturne_sanctuary_motion_specs.md (Quart Out standard, etc.).

export type ThemeName = "tsushima" | "transylvania" | "frieren" | "lofi";

/** Global motion — same for all palettes (Nocturne Sanctuary specs). */
export const APP_SHELL_MOTION = {
  /** Standard UI transition */
  standard: "600ms cubic-bezier(0.22, 1, 0.36, 1)",
  /** Deep immersion / zoom */
  deep: "1200ms cubic-bezier(0.45, 0, 0.55, 1)",
  /** Soft entry (panels, hero) */
  softEntry: "800ms cubic-bezier(0.16, 1, 0.3, 1)",
  /** Ambient background pulse loop */
  ambient: "8000ms ease-in-out",
} as const;

/** Typography stacks (Ethereal Architect: Manrope + Plus Jakarta Sans). */
export const APP_SHELL_TYPOGRAPHY = {
  display:
    '"Manrope", system-ui, -apple-system, "Segoe UI", sans-serif',
  body: '"Plus Jakarta Sans", "Manrope", system-ui, -apple-system, sans-serif',
} as const;

export interface ThemeTokens {
  name: ThemeName;
  // Backgrounds
  bgApp: string;       // root canvas
  bgNav: string;       // sticky nav bar
  bgSurface: string;   // cards, panels
  bgRaised: string;    // modals, popovers
  bgInput: string;     // inputs, textareas
  bgSubtle: string;    // subtle dividers, hover tints
  // Borders
  borderBase: string;
  borderMuted: string;
  borderStrong: string;
  // Text
  textPrimary: string;
  textSecondary: string;
  textMuted: string;
  textGhost: string;
  // Accent
  accent: string;
  accentDim: string;   // accent at ~10% opacity for backgrounds
  /** Secondary jewel accent (lavender / cyan) — kicker, links */
  accentSecondary: string;
  /** Muted positive / completion */
  accentGood: string;
  /** Muted caution */
  accentWarn: string;
  // Status dots (override per theme)
  dotActive: string;
  dotBlocked: string;
  dotPlanned: string;
  dotPlanning: string;
  dotStart: string;
  dotEnd: string;
  dotIdea: string;
  // Mist / atmospheric
  mistColor: string;
  /** Radial hints on app shell (muted jewel rationale) */
  shellGlowA: string;
  shellGlowB: string;
  /** Time-flow / one-next-task ambient shimmer (motion spec teal + violet) */
  shimmerA: string;
  shimmerB: string;
  /** Legacy shell: glass panel background */
  glass: string;
  /** Hairline / ghost structural line */
  line: string;
  // Motion (per-theme can match global; kept for token completeness)
  transition: string;
  // Radius
  radiusSm: string;
  radiusMd: string;
  // Scrollbar
  scrollbarThumb: string;
  scrollbarTrack: string;
}

// ─── PRESETS ──────────────────────────────────────────────────────────────────

const MOTION = APP_SHELL_MOTION.standard;

export const themes: Record<ThemeName, ThemeTokens> = {
  tsushima: {
    name: "tsushima",
    bgApp: "#080e08",
    bgNav: "#080e08",
    bgSurface: "#0e160e",
    bgRaised: "#0d150d",
    bgInput: "#111a11",
    bgSubtle: "#0f160f",
    borderBase: "#1a2a1a",
    borderMuted: "#162016",
    borderStrong: "#2a3e2a",
    textPrimary: "#e8f0e0",
    textSecondary: "#8aaa7a",
    textMuted: "#4a6040",
    textGhost: "#2a3a22",
    accent: "#b8d460",
    accentDim: "#b8d46012",
    accentSecondary: "#9b8fd4",
    accentGood: "#7ec89a",
    accentWarn: "#c8b070",
    dotActive: "#88c840",
    dotBlocked: "#d46060",
    dotPlanned: "#8888cc",
    dotPlanning: "#c8a060",
    dotStart: "#5ba0e0",
    dotEnd: "#445544",
    dotIdea: "#9888cc",
    mistColor: "#0e1a0e",
    shellGlowA: "rgba(155, 143, 212, 0.14)",
    shellGlowB: "rgba(62, 207, 150, 0.11)",
    shimmerA: "#4a6767",
    shimmerB: "#2d3e4e",
    glass: "rgba(12, 18, 12, 0.72)",
    line: "rgba(26, 42, 26, 0.45)",
    transition: MOTION,
    radiusSm: "3px",
    radiusMd: "5px",
    scrollbarThumb: "#1e2e1e",
    scrollbarTrack: "#0a0e0a",
  },

  transylvania: {
    name: "transylvania",
    bgApp: "#06060e",
    bgNav: "#06060e",
    bgSurface: "#0c0c18",
    bgRaised: "#0d0d1a",
    bgInput: "#101020",
    bgSubtle: "#0e0e1c",
    borderBase: "#18182e",
    borderMuted: "#14142a",
    borderStrong: "#28284a",
    textPrimary: "#dcdcf4",
    textSecondary: "#7878a8",
    textMuted: "#404068",
    textGhost: "#242438",
    accent: "#9988cc",
    accentDim: "#9988cc12",
    accentSecondary: "#88b8d4",
    accentGood: "#8ab8a8",
    accentWarn: "#c8a888",
    dotActive: "#88aacc",
    dotBlocked: "#c85878",
    dotPlanned: "#9988cc",
    dotPlanning: "#a88860",
    dotStart: "#5888cc",
    dotEnd: "#484858",
    dotIdea: "#aa88cc",
    mistColor: "#0a0a18",
    shellGlowA: "rgba(153, 136, 204, 0.16)",
    shellGlowB: "rgba(104, 152, 200, 0.1)",
    shimmerA: "#4a5a78",
    shimmerB: "#2d3048",
    glass: "rgba(10, 10, 22, 0.74)",
    line: "rgba(36, 36, 60, 0.5)",
    transition: MOTION,
    radiusSm: "3px",
    radiusMd: "5px",
    scrollbarThumb: "#1a1a30",
    scrollbarTrack: "#08081a",
  },

  frieren: {
    name: "frieren",
    bgApp: "#0e0c10",
    bgNav: "#0e0c10",
    bgSurface: "#16141a",
    bgRaised: "#18151e",
    bgInput: "#1a1820",
    bgSubtle: "#151318",
    borderBase: "#28243a",
    borderMuted: "#221e32",
    borderStrong: "#38325a",
    textPrimary: "#f0ead8",
    textSecondary: "#9090a8",
    textMuted: "#505068",
    textGhost: "#302e48",
    accent: "#d4b860",
    accentDim: "#d4b86012",
    accentSecondary: "#a898d0",
    accentGood: "#98c8a8",
    accentWarn: "#d8b078",
    dotActive: "#a0c8a0",
    dotBlocked: "#c87878",
    dotPlanned: "#8890c8",
    dotPlanning: "#d4b860",
    dotStart: "#60a0d8",
    dotEnd: "#585068",
    dotIdea: "#a898d0",
    mistColor: "#100e18",
    shellGlowA: "rgba(212, 184, 96, 0.12)",
    shellGlowB: "rgba(168, 152, 208, 0.12)",
    shimmerA: "#5a5868",
    shimmerB: "#3a3848",
    glass: "rgba(20, 18, 28, 0.75)",
    line: "rgba(56, 50, 90, 0.42)",
    transition: MOTION,
    radiusSm: "4px",
    radiusMd: "7px",
    scrollbarThumb: "#28243a",
    scrollbarTrack: "#0c0a10",
  },

  lofi: {
    name: "lofi",
    bgApp: "#0e0c0c",
    bgNav: "#0e0c0c",
    bgSurface: "#161212",
    bgRaised: "#181414",
    bgInput: "#1a1616",
    bgSubtle: "#141010",
    borderBase: "#2a2222",
    borderMuted: "#221a1a",
    borderStrong: "#3a2e2e",
    textPrimary: "#eae0dc",
    textSecondary: "#8a8090",
    textMuted: "#504848",
    textGhost: "#2e2828",
    accent: "#c088a8",
    accentDim: "#c088a812",
    accentSecondary: "#9b7fe8",
    accentGood: "#80c0b8",
    accentWarn: "#d8a35f",
    dotActive: "#80c0b8",
    dotBlocked: "#c06870",
    dotPlanned: "#a088b8",
    dotPlanning: "#c0a070",
    dotStart: "#6090b8",
    dotEnd: "#484040",
    dotIdea: "#b088c0",
    mistColor: "#100c0c",
    shellGlowA: "rgba(155, 127, 232, 0.14)",
    shellGlowB: "rgba(62, 207, 207, 0.11)",
    shimmerA: "#4a6767",
    shimmerB: "#2d3e4e",
    glass: "rgba(15, 12, 12, 0.72)",
    line: "rgba(36, 32, 32, 0.45)",
    transition: MOTION,
    radiusSm: "3px",
    radiusMd: "5px",
    scrollbarThumb: "#2a2020",
    scrollbarTrack: "#0c0a0a",
  },
};

export const DEFAULT_THEME: ThemeName = "lofi";
