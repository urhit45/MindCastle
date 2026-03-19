import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { CastleViewport } from "./CastleViewport";

afterEach(cleanup);

describe("CastleViewport", () => {
  it("renders children inside a <main> element", () => {
    const { getByRole, getByText } = render(
      <CastleViewport>
        <p>Hello Castle</p>
      </CastleViewport>,
    );
    expect(getByRole("main")).toBeInTheDocument();
    expect(getByText("Hello Castle")).toBeInTheDocument();
  });

  it("<main> uses CSS grid with place-items start center", () => {
    const { getByRole } = render(
      <CastleViewport>
        <div />
      </CastleViewport>,
    );
    const main = getByRole("main");
    expect(main.style.display).toBe("grid");
    expect(main.style.placeItems).toBe("start center");
  });

  it("inner wrapper has maxWidth 1200px", () => {
    const { getByRole } = render(
      <CastleViewport>
        <div />
      </CastleViewport>,
    );
    const inner = getByRole("main").firstChild as HTMLElement;
    expect(inner.style.maxWidth).toBe("1200px");
  });

  it("inner wrapper occupies full width", () => {
    const { getByRole } = render(
      <CastleViewport>
        <div />
      </CastleViewport>,
    );
    const inner = getByRole("main").firstChild as HTMLElement;
    expect(inner.style.width).toBe("100%");
  });

  it("<main> hides overflowX and allows overflowY scroll", () => {
    const { getByRole } = render(
      <CastleViewport>
        <div />
      </CastleViewport>,
    );
    const main = getByRole("main");
    expect(main.style.overflowX).toBe("hidden");
    expect(main.style.overflowY).toBe("auto");
  });

  it("renders multiple children without wrapper collapse", () => {
    const { getByText } = render(
      <CastleViewport>
        <span>Alpha</span>
        <span>Beta</span>
      </CastleViewport>,
    );
    expect(getByText("Alpha")).toBeInTheDocument();
    expect(getByText("Beta")).toBeInTheDocument();
  });
});
