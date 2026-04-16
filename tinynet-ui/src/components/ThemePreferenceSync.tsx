import { useEffect, useRef } from "react";
import { useTheme } from "../theme/ThemeProvider";
import { themes, type ThemeName } from "../theme/tokens";
import { getPreferences } from "../api";

function isThemeName(value: string): value is ThemeName {
  return value in themes;
}

/**
 * On load, apply theme from API when the backend has a stored preference.
 * LocalStorage (ThemeProvider) is the first paint; this may override once.
 */
export function ThemePreferenceSync() {
  const { setTheme } = useTheme();
  const ran = useRef(false);

  useEffect(() => {
    if (ran.current) return;
    ran.current = true;
    let cancelled = false;
    void (async () => {
      try {
        const p = await getPreferences();
        if (cancelled || !p.theme || !isThemeName(p.theme)) return;
        setTheme(p.theme);
      } catch {
        /* offline or API down */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [setTheme]);

  return null;
}
