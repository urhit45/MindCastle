import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ThemeProvider } from "../theme/ThemeProvider";
import { ThemeSwitcher } from "./ThemeSwitcher";
import * as api from "../api";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("ThemeSwitcher", () => {
  it("changes theme and PATCHes preferences when API succeeds", async () => {
    const patch = vi.spyOn(api, "patchPreferences").mockResolvedValue({ theme: "tsushima" });
    render(
      <ThemeProvider>
        <ThemeSwitcher />
      </ThemeProvider>,
    );
    fireEvent.change(screen.getByRole("combobox", { name: /theme/i }), {
      target: { value: "tsushima" },
    });
    expect(patch).toHaveBeenCalledWith({ theme: "tsushima" });
  });

  it("still updates local theme when PATCH fails", async () => {
    vi.spyOn(api, "patchPreferences").mockRejectedValue(new Error("offline"));
    render(
      <ThemeProvider>
        <ThemeSwitcher />
      </ThemeProvider>,
    );
    fireEvent.change(screen.getByRole("combobox", { name: /theme/i }), {
      target: { value: "frieren" },
    });
    expect(screen.getByRole("combobox")).toHaveValue("frieren");
  });
});
