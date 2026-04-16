import { createContext, useContext, useState, useEffect } from "react";
import type { ThemeName, ThemeTokens } from "./tokens";
import {
  themes,
  DEFAULT_THEME,
  APP_SHELL_MOTION,
  APP_SHELL_TYPOGRAPHY,
} from "./tokens";

// ─── CONTEXT ─────────────────────────────────────────────────────────────────

interface ThemeContextValue {
  theme: ThemeTokens;
  themeName: ThemeName;
  setTheme: (name: ThemeName) => void;
}

const ThemeContext = createContext<ThemeContextValue>({
  theme: themes[DEFAULT_THEME],
  themeName: DEFAULT_THEME,
  setTheme: () => {},
});

export function useTheme() {
  return useContext(ThemeContext);
}

// ─── CSS VAR INJECTION ────────────────────────────────────────────────────────

function applyThemeToDom(t: ThemeTokens) {
  const root = document.documentElement;
  root.style.setProperty("--bg-app", t.bgApp);
  root.style.setProperty("--bg-nav", t.bgNav);
  root.style.setProperty("--bg-surface", t.bgSurface);
  root.style.setProperty("--bg-raised", t.bgRaised);
  root.style.setProperty("--bg-input", t.bgInput);
  root.style.setProperty("--bg-subtle", t.bgSubtle);
  root.style.setProperty("--border-base", t.borderBase);
  root.style.setProperty("--border-muted", t.borderMuted);
  root.style.setProperty("--border-strong", t.borderStrong);
  root.style.setProperty("--text-primary", t.textPrimary);
  root.style.setProperty("--text-secondary", t.textSecondary);
  root.style.setProperty("--text-muted", t.textMuted);
  root.style.setProperty("--text-ghost", t.textGhost);
  root.style.setProperty("--accent", t.accent);
  root.style.setProperty("--accent-dim", t.accentDim);
  root.style.setProperty("--accent-secondary", t.accentSecondary);
  root.style.setProperty("--accent-good", t.accentGood);
  root.style.setProperty("--accent-warn", t.accentWarn);
  root.style.setProperty("--dot-active", t.dotActive);
  root.style.setProperty("--dot-blocked", t.dotBlocked);
  root.style.setProperty("--dot-planned", t.dotPlanned);
  root.style.setProperty("--dot-planning", t.dotPlanning);
  root.style.setProperty("--dot-start", t.dotStart);
  root.style.setProperty("--dot-end", t.dotEnd);
  root.style.setProperty("--dot-idea", t.dotIdea);
  root.style.setProperty("--mist", t.mistColor);
  root.style.setProperty("--shell-glow-a", t.shellGlowA);
  root.style.setProperty("--shell-glow-b", t.shellGlowB);
  root.style.setProperty("--shimmer-a", t.shimmerA);
  root.style.setProperty("--shimmer-b", t.shimmerB);
  root.style.setProperty("--glass-tint", t.glass);
  root.style.setProperty("--line-ghost", t.line);
  root.style.setProperty("--transition", t.transition);
  root.style.setProperty("--radius-sm", t.radiusSm);
  root.style.setProperty("--radius-md", t.radiusMd);
  root.style.setProperty("--scrollbar-thumb", t.scrollbarThumb);
  root.style.setProperty("--scrollbar-track", t.scrollbarTrack);

  // Legacy names used across styles.css (mapped from semantic tokens)
  root.style.setProperty("--bg-0", t.bgApp);
  root.style.setProperty("--bg-1", t.bgSurface);
  root.style.setProperty("--bg-2", t.bgRaised);
  root.style.setProperty("--text-0", t.textPrimary);
  root.style.setProperty("--text-1", t.textSecondary);
  root.style.setProperty("--text-2", t.textMuted);
  root.style.setProperty("--accent-2", t.accentSecondary);
  root.style.setProperty("--good", t.accentGood);
  root.style.setProperty("--warn", t.accentWarn);
  root.style.setProperty("--glass", t.glass);
  root.style.setProperty("--line", t.line);

  // Nocturne motion spec (global)
  root.style.setProperty("--motion-standard", APP_SHELL_MOTION.standard);
  root.style.setProperty("--motion-deep", APP_SHELL_MOTION.deep);
  root.style.setProperty("--motion-soft-entry", APP_SHELL_MOTION.softEntry);
  root.style.setProperty("--motion-ambient", APP_SHELL_MOTION.ambient);

  root.style.setProperty("--font-display", APP_SHELL_TYPOGRAPHY.display);
  root.style.setProperty("--font-body", APP_SHELL_TYPOGRAPHY.body);

  root.setAttribute("data-theme", t.name);
}

// ─── PROVIDER ─────────────────────────────────────────────────────────────────

const STORAGE_KEY = "mindcastle_theme";

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [themeName, setThemeName] = useState<ThemeName>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY) as ThemeName | null;
      return saved && themes[saved] ? saved : DEFAULT_THEME;
    } catch {
      return DEFAULT_THEME;
    }
  });

  const theme = themes[themeName];

  useEffect(() => {
    applyThemeToDom(theme);
    try {
      localStorage.setItem(STORAGE_KEY, themeName);
    } catch {
      /* ignore */
    }
  }, [theme, themeName]);

  const setTheme = (name: ThemeName) => {
    if (themes[name]) setThemeName(name);
  };

  return (
    <ThemeContext.Provider value={{ theme, themeName, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}
