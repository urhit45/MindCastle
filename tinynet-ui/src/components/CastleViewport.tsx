import type { ReactNode } from "react";

interface CastleViewportProps {
  children: ReactNode;
}

/**
 * Centered, full-bleed content container.
 * Always fills the usable area and keeps content centered
 * even when side panels (ArtifactDetail) are open.
 */
export function CastleViewport({ children }: CastleViewportProps) {
  return (
    <main
      style={{
        flex: 1,
        display: "grid",
        placeItems: "start center",
        overflowY: "auto",
        overflowX: "hidden",
        // Consume remaining height after sticky nav
        minHeight: 0,
      }}
    >
      <div
        style={{
          width: "100%",
          maxWidth: 1200,
          padding: "0 clamp(16px, 3vw, 48px) 80px",
        }}
      >
        {children}
      </div>
    </main>
  );
}
