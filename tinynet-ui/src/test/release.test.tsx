/**
 * Phase V — Release Gate Tests
 *
 * Product experience gate:  theme completeness, CSS var injection, viewport invariants
 * Technical reliability gate: layout constraints, render performance baseline
 *
 * These run as part of `npm run test:run` and block CI/release if they fail.
 */

import { cleanup, render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { CastleViewport } from "../components/CastleViewport";
import { ThemeProvider } from "../theme/ThemeProvider";
import { themes } from "../theme/tokens";
import type { ThemeName } from "../theme/tokens";

afterEach(cleanup);

// All 4 theme names — used as the authoritative list across all tests
const THEME_NAMES: ThemeName[] = ["tsushima", "transylvania", "frieren", "lofi"];

// Reference key set derived from the canonical theme object
const TOKEN_KEYS = Object.keys(themes.lofi) as (keyof typeof themes.lofi)[];

// ─────────────────────────────────────────────────────────────────────────────
// Gate 1A: Theme token completeness
// Every theme must define every key with a non-empty value.
// A missing or empty token means a CSS var will silently be blank.
// ─────────────────────────────────────────────────────────────────────────────

describe("Release: Theme token completeness", () => {
  it("lofi reference theme has at least 28 tokens", () => {
    expect(TOKEN_KEYS.length).toBeGreaterThanOrEqual(28);
  });

  it.each(THEME_NAMES)(
    "theme '%s' defines all %d token keys with non-empty values",
    (name) => {
      const theme = themes[name];
      for (const key of TOKEN_KEYS) {
        expect(
          theme[key],
          `themes.${name}.${key} must be a non-empty string`,
        ).toBeTruthy();
      }
    },
  );

  it("all 4 themes have identical key sets (no drift between presets)", () => {
    const reference = [...TOKEN_KEYS].sort().join(",");
    for (const name of THEME_NAMES) {
      const actual = Object.keys(themes[name]).sort().join(",");
      expect(actual, `themes.${name} key set diverges from lofi`).toBe(
        reference,
      );
    }
  });

  it("every theme's 'name' field matches its key in the themes map", () => {
    for (const name of THEME_NAMES) {
      expect(themes[name].name).toBe(name);
    }
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Gate 1B: CSS var injection — ThemeProvider writes all CSS custom properties
// ─────────────────────────────────────────────────────────────────────────────

describe("Release: ThemeProvider CSS var coverage", () => {
  beforeEach(() => {
    document.documentElement.removeAttribute("style");
    document.documentElement.removeAttribute("data-theme");
    localStorage.clear();
  });

  // The 27 CSS vars that applyThemeToDom() must write (derived from ThemeProvider source)
  const EXPECTED_CSS_VARS = [
    "--bg-app", "--bg-nav", "--bg-surface", "--bg-raised", "--bg-input",
    "--bg-subtle", "--border-base", "--border-muted", "--border-strong",
    "--text-primary", "--text-secondary", "--text-muted", "--text-ghost",
    "--accent", "--accent-dim", "--dot-active", "--dot-blocked",
    "--dot-planned", "--dot-planning", "--dot-start", "--dot-end",
    "--dot-idea", "--mist", "--transition", "--radius-sm", "--radius-md",
    "--scrollbar-thumb", "--scrollbar-track",
  ] as const;

  it.each(THEME_NAMES)(
    "theme '%s' injects all %d CSS vars into :root",
    (name) => {
      localStorage.setItem("mindcastle_theme", name);
      render(<ThemeProvider><div /></ThemeProvider>);
      for (const varName of EXPECTED_CSS_VARS) {
        const value = document.documentElement.style.getPropertyValue(varName);
        expect(
          value,
          `${name} must inject ${varName} (got empty string)`,
        ).not.toBe("");
      }
    },
  );

  it("data-theme attribute is set to the active theme name", () => {
    for (const name of THEME_NAMES) {
      localStorage.setItem("mindcastle_theme", name);
      render(<ThemeProvider><div /></ThemeProvider>);
      expect(document.documentElement.getAttribute("data-theme")).toBe(name);
      cleanup();
    }
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Gate 2A: CastleViewport layout invariants across viewport widths
// Ensures the centered-castle invariant holds on phones → wide monitors.
// ─────────────────────────────────────────────────────────────────────────────

describe("Release: CastleViewport layout invariants", () => {
  const VIEWPORT_WIDTHS = [320, 375, 768, 1024, 1440, 1920, 2560];

  it.each(VIEWPORT_WIDTHS)(
    "maxWidth stays 1200px at %dpx viewport",
    (width) => {
      Object.defineProperty(window, "innerWidth", { value: width, writable: true, configurable: true });
      const { getByRole } = render(<CastleViewport><div /></CastleViewport>);
      const inner = getByRole("main").firstChild as HTMLElement;
      expect(inner.style.maxWidth, `maxWidth at ${width}px`).toBe("1200px");
    },
  );

  it("overflowX:hidden prevents horizontal overflow at any width", () => {
    for (const w of [320, 768, 1440]) {
      Object.defineProperty(window, "innerWidth", { value: w, writable: true, configurable: true });
      const { getByRole } = render(<CastleViewport><div /></CastleViewport>);
      expect(getByRole("main").style.overflowX).toBe("hidden");
      cleanup();
    }
  });

  it("flex:1 allows <main> to consume remaining viewport height", () => {
    const { getByRole } = render(<CastleViewport><div /></CastleViewport>);
    // jsdom expands flex shorthand: "1" → "1 1 0%" — check flexGrow instead
    expect(getByRole("main").style.flexGrow).toBe("1");
  });

  it("minHeight:0 prevents grid blowout inside flex column", () => {
    const { getByRole } = render(<CastleViewport><div /></CastleViewport>);
    // jsdom omits the unit for zero values: 0 → "0" (not "0px")
    expect(getByRole("main").style.minHeight).toBe("0");
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Gate 2B: Render performance baseline
// Thresholds are generous (jsdom ≫ browser) — used as regression guards only.
// ─────────────────────────────────────────────────────────────────────────────

describe("Release: Render performance baseline", () => {
  it("ThemeProvider cold render completes in < 100ms", () => {
    const t0 = performance.now();
    render(<ThemeProvider><div /></ThemeProvider>);
    const elapsed = performance.now() - t0;
    expect(elapsed, `ThemeProvider took ${elapsed.toFixed(1)}ms`).toBeLessThan(100);
  });

  it("CastleViewport cold render completes in < 20ms", () => {
    const t0 = performance.now();
    render(<CastleViewport><div /></CastleViewport>);
    const elapsed = performance.now() - t0;
    expect(elapsed, `CastleViewport took ${elapsed.toFixed(1)}ms`).toBeLessThan(20);
  });

  it("rendering all 4 themes in sequence takes < 200ms total", () => {
    const t0 = performance.now();
    for (const name of THEME_NAMES) {
      localStorage.setItem("mindcastle_theme", name);
      render(<ThemeProvider><div /></ThemeProvider>);
      cleanup();
    }
    const elapsed = performance.now() - t0;
    expect(elapsed, `4-theme render cycle took ${elapsed.toFixed(1)}ms`).toBeLessThan(200);
  });
});
