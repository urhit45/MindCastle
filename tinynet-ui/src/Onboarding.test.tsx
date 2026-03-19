import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Onboarding } from "./Onboarding";

afterEach(cleanup);

// The Onboarding component calls crypto.randomUUID() in helpers —
// jsdom provides this natively in Node 20+.

describe("Onboarding — initial view", () => {
  it("renders the welcome heading", () => {
    render(
      <Onboarding onComplete={vi.fn()} onSkip={vi.fn()} />,
    );
    expect(screen.getByText("Welcome to MindCastle")).toBeInTheDocument();
  });

  it("shows the skip button", () => {
    render(
      <Onboarding onComplete={vi.fn()} onSkip={vi.fn()} />,
    );
    expect(
      screen.getByText(/skip — set up manually/i),
    ).toBeInTheDocument();
  });

  it("calls onSkip when the skip button is clicked", () => {
    const onSkip = vi.fn();
    render(<Onboarding onComplete={vi.fn()} onSkip={onSkip} />);
    fireEvent.click(screen.getByText(/skip — set up manually/i));
    expect(onSkip).toHaveBeenCalledOnce();
  });

  it("renders the three path cards", () => {
    render(
      <Onboarding onComplete={vi.fn()} onSkip={vi.fn()} />,
    );
    // PATHS array defines: "Bring Your Stuff Over", "Build Your Palace", "Text Import"
    expect(screen.getByText("Bring Your Stuff Over")).toBeInTheDocument();
    expect(screen.getByText("Build Your Palace")).toBeInTheDocument();
    expect(screen.getByText("Text Import")).toBeInTheDocument();
  });
});

describe("Onboarding — path navigation", () => {
  it("switches to the ChatGPT wizard when 'Bring Your Stuff Over' is clicked", () => {
    render(
      <Onboarding onComplete={vi.fn()} onSkip={vi.fn()} />,
    );
    fireEvent.click(screen.getByText("Bring Your Stuff Over"));
    // Wizard title appears; welcome heading disappears
    expect(screen.getByText("Import from AI")).toBeInTheDocument();
    expect(screen.queryByText("Welcome to MindCastle")).not.toBeInTheDocument();
  });

  it("switches to the Palace wizard when 'Build Your Palace' is clicked", () => {
    render(
      <Onboarding onComplete={vi.fn()} onSkip={vi.fn()} />,
    );
    fireEvent.click(screen.getByText("Build Your Palace"));
    expect(screen.getByText("Build Your Palace", { selector: "[style]", exact: true }) ||
           screen.getByText(/build your palace/i)).toBeInTheDocument();
    expect(screen.queryByText("Welcome to MindCastle")).not.toBeInTheDocument();
  });

  it("goes back to the select view from a wizard", () => {
    render(
      <Onboarding onComplete={vi.fn()} onSkip={vi.fn()} />,
    );
    fireEvent.click(screen.getByText("Bring Your Stuff Over"));
    // Back arrow button has accessible text or is findable via role
    const backBtn = screen.getByRole("button", { name: /back/i });
    fireEvent.click(backBtn);
    expect(screen.getByText("Welcome to MindCastle")).toBeInTheDocument();
  });
});
