import { useTheme } from "../theme/ThemeProvider";
import { themes, type ThemeName } from "../theme/tokens";
import { patchPreferences } from "../api";

const LABELS: Record<ThemeName, string> = {
  tsushima: "Tsushima",
  transylvania: "Transylvania",
  frieren: "Frieren",
  lofi: "Lo-fi",
};

export function ThemeSwitcher() {
  const { themeName, setTheme } = useTheme();

  return (
    <label className="theme-switcher">
      <span className="theme-switcher-label">Theme</span>
      <select
        aria-label="Color theme"
        className="theme-switcher-select"
        value={themeName}
        onChange={(e) => {
          const name = e.target.value as ThemeName;
          if (!(name in themes)) return;
          setTheme(name);
          void patchPreferences({ theme: name }).catch(() => {
            /* backend optional */
          });
        }}
      >
        {(Object.keys(themes) as ThemeName[]).map((key) => (
          <option key={key} value={key}>
            {LABELS[key]}
          </option>
        ))}
      </select>
    </label>
  );
}
