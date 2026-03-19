// ─── THEME TOKENS ─────────────────────────────────────────────────────────────

export type ThemeName = "tsushima" | "transylvania" | "frieren" | "lofi";

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
  // Motion
  transition: string;
  // Radius
  radiusSm: string;
  radiusMd: string;
  // Scrollbar
  scrollbarThumb: string;
  scrollbarTrack: string;
}

// ─── PRESETS ──────────────────────────────────────────────────────────────────

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
    accent: "#b8d460",        // sunrise amber-green
    accentDim: "#b8d46012",
    dotActive: "#88c840",
    dotBlocked: "#d46060",
    dotPlanned: "#8888cc",
    dotPlanning: "#c8a060",
    dotStart: "#5ba0e0",
    dotEnd: "#445544",
    dotIdea: "#9888cc",
    mistColor: "#0e1a0e",
    transition: "0.2s ease",
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
    accent: "#9988cc",        // muted violet
    accentDim: "#9988cc12",
    dotActive: "#88aacc",
    dotBlocked: "#c85878",
    dotPlanned: "#9988cc",
    dotPlanning: "#a88860",
    dotStart: "#5888cc",
    dotEnd: "#484858",
    dotIdea: "#aa88cc",
    mistColor: "#0a0a18",
    transition: "0.2s ease",
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
    accent: "#d4b860",        // gentle gold
    accentDim: "#d4b86012",
    dotActive: "#a0c8a0",
    dotBlocked: "#c87878",
    dotPlanned: "#8890c8",
    dotPlanning: "#d4b860",
    dotStart: "#60a0d8",
    dotEnd: "#585068",
    dotIdea: "#a898d0",
    mistColor: "#100e18",
    transition: "0.25s ease",
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
    accent: "#c088a8",        // muted magenta
    accentDim: "#c088a812",
    dotActive: "#80c0b8",     // dusty cyan
    dotBlocked: "#c06870",
    dotPlanned: "#a088b8",
    dotPlanning: "#c0a070",
    dotStart: "#6090b8",
    dotEnd: "#484040",
    dotIdea: "#b088c0",
    mistColor: "#100c0c",
    transition: "0.18s ease",
    radiusSm: "3px",
    radiusMd: "5px",
    scrollbarThumb: "#2a2020",
    scrollbarTrack: "#0c0a0a",
  },
};

export const DEFAULT_THEME: ThemeName = "lofi";
