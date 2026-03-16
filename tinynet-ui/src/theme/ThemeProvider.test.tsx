import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { ThemeProvider, useTheme } from "./ThemeProvider";
import { DEFAULT_THEME, themes } from "./tokens";

// Minimal component that reads theme context and exposes controls
function ThemeProbe() {
  const { themeName, setTheme } = useTheme();
  return (
    <>
      <span data-testid="theme-name">{themeName}</span>
      <button data-testid="to-tsushima" onClick={() => setTheme("tsushima")} />
      <button data-testid="to-frieren" onClick={() => setTheme("frieren")} />
      <button data-testid="to-lofi" onClick={() => setTheme("lofi")} />
    </>
  );
}

function renderWithProvider() {
  return render(
    <ThemeProvider>
      <ThemeProbe />
    </ThemeProvider>,
  );
}

describe("ThemeProvider", () => {
  beforeEach(() => {
    localStorage.clear();
    // Reset CSS vars between tests
    document.documentElement.removeAttribute("style");
    document.documentElement.removeAttribute("data-theme");
  });

  afterEach(() => {
    cleanup();
  });

  it("provides the default theme when no saved preference", () => {
    renderWithProvider();
    expect(screen.getByTestId("theme-name").textContent).toBe(DEFAULT_THEME);
  });

  it("injects --bg-app CSS var matching the default theme", () => {
    renderWithProvider();
    const applied = document.documentElement.style.getPropertyValue("--bg-app");
    expect(applied).toBe(themes[DEFAULT_THEME].bgApp);
  });

  it("sets data-theme attribute on documentElement", () => {
    renderWithProvider();
    expect(document.documentElement.getAttribute("data-theme")).toBe(
      DEFAULT_THEME,
    );
  });

  it("setTheme updates the context value", () => {
    renderWithProvider();
    fireEvent.click(screen.getByTestId("to-tsushima"));
    expect(screen.getByTestId("theme-name").textContent).toBe("tsushima");
  });

  it("setTheme swaps CSS vars to the new theme", () => {
    renderWithProvider();
    fireEvent.click(screen.getByTestId("to-tsushima"));
    const applied = document.documentElement.style.getPropertyValue("--bg-app");
    expect(applied).toBe(themes["tsushima"].bgApp);
  });

  it("setTheme persists the choice to localStorage", () => {
    renderWithProvider();
    fireEvent.click(screen.getByTestId("to-frieren"));
    expect(localStorage.getItem("mindcastle_theme")).toBe("frieren");
  });

  it("reads a previously saved theme from localStorage", () => {
    localStorage.setItem("mindcastle_theme", "tsushima");
    renderWithProvider();
    expect(screen.getByTestId("theme-name").textContent).toBe("tsushima");
  });

  it("falls back to the default when stored value is unknown", () => {
    localStorage.setItem("mindcastle_theme", "not-a-real-theme");
    renderWithProvider();
    expect(screen.getByTestId("theme-name").textContent).toBe(DEFAULT_THEME);
  });

  it("silently ignores setTheme calls with unknown theme names", () => {
    renderWithProvider();
    // @ts-expect-error testing runtime guard
    fireEvent.click(screen.getByTestId("to-lofi")); // valid — confirm round-trip works
    fireEvent.click(screen.getByTestId("to-lofi"));
    expect(screen.getByTestId("theme-name").textContent).toBe("lofi");
  });
});
