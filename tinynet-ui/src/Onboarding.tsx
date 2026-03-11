import { useState, useEffect } from "react";
import * as api from "./api";

// ─── TYPES ───────────────────────────────────────────────────────────────────

interface Artifact {
  id: string;
  title: string;
  subtitle: string;
  status: string;
  next: string;
  stack: string[];
  progress: { label: string; val: number }[];
  notes: string;
}

interface Engine {
  id: string;
  name: string;
  subtitle: string;
  color: string;
  icon: string;
  context: string;
  contextLabel: string;
  description: string;
  artifacts: Artifact[];
}

interface OnboardingProps {
  onComplete: (engines: Engine[]) => void;
  onSkip: () => void;
}

// ─── CONSTANTS ───────────────────────────────────────────────────────────────

const PALETTE = [
  "#3ecfcf", "#9b7fe8", "#c8a96e", "#e0637a",
  "#a3d977", "#5b8dee", "#f0a05a", "#d46b8a",
];

const ICONS = ["◈", "✦", "◎", "⬡", "⟁", "◇", "⊕", "⬢", "△", "○"];

const PRESETS: { name: string; subtitle: string; icon: string; color: string }[] = [
  { name: "Tech Builder",   subtitle: "Code, systems & side projects",  icon: "◈", color: "#3ecfcf" },
  { name: "Writer",         subtitle: "Essays, docs & storytelling",    icon: "✦", color: "#9b7fe8" },
  { name: "Health",         subtitle: "Fitness, sleep & nutrition",     icon: "◎", color: "#a3d977" },
  { name: "Finance",        subtitle: "Budgets, savings & investing",   icon: "⬡", color: "#c8a96e" },
  { name: "Learning",       subtitle: "Books, courses & deep dives",    icon: "⟁", color: "#5b8dee" },
  { name: "Creative",       subtitle: "Art, design & making things",    icon: "◇", color: "#f0a05a" },
  { name: "Music",          subtitle: "Practice, theory & recording",   icon: "⊕", color: "#d46b8a" },
  { name: "Social",         subtitle: "Relationships & community",      icon: "△", color: "#e0637a" },
];

const CHATGPT_PROMPT = `I'm migrating to MindCastle — a local personal productivity tool that organises my work into "Engines" (life domains) and "Artifacts" (projects/tasks inside them).

Please review our conversation history and extract every project, goal, task, or ongoing topic we've discussed. Group them into logical Engines (e.g. Work, Learning, Health, Finance, Creative). Then output ONLY a JSON block in this exact format — no extra text:

{
  "version": "1",
  "engines": [
    {
      "name": "Engine name",
      "subtitle": "One-line description",
      "artifacts": [
        { "title": "Task or project title", "notes": "Any context", "status": "concept" }
      ]
    }
  ]
}

Use status "active" for things we're actively working on, "concept" for ideas, "planned" for upcoming work, "blocked" for stuck items.`;

// ─── HELPERS ─────────────────────────────────────────────────────────────────

function makeArtifact(title: string, notes = "", status = "concept"): Artifact {
  return {
    id: crypto.randomUUID(),
    title,
    subtitle: "",
    status,
    next: "",
    stack: [],
    progress: [],
    notes,
  };
}

function makeEngine(
  name: string,
  subtitle = "",
  color?: string,
  icon?: string,
  artifacts: Artifact[] = [],
): Engine {
  const idx = Math.floor(Math.random() * PALETTE.length);
  return {
    id: crypto.randomUUID(),
    name,
    subtitle,
    color: color ?? PALETTE[idx],
    icon: icon ?? ICONS[idx % ICONS.length],
    context: "",
    contextLabel: "Context",
    description: "",
    artifacts,
  };
}

const VALID_STATUSES = new Set(["concept", "active", "planned", "live", "blocked", "planning"]);

function parseImportJSON(raw: string): Engine[] | null {
  try {
    // Strip code fences if present
    const cleaned = raw.replace(/^```[a-z]*\n?/m, "").replace(/```$/m, "").trim();
    const parsed = JSON.parse(cleaned);
    const engines: Engine[] = [];

    const rawEngines = Array.isArray(parsed) ? parsed : (parsed.engines ?? []);
    for (const e of rawEngines) {
      const name = (e.name ?? "Untitled").toString().trim();
      if (!name) continue;
      const artifacts: Artifact[] = (e.artifacts ?? []).map((a: Record<string, string>) =>
        makeArtifact(
          (a.title ?? "Untitled").toString().trim(),
          (a.notes ?? "").toString(),
          VALID_STATUSES.has(a.status) ? a.status : "concept",
        ),
      );
      engines.push(makeEngine(
        name,
        (e.subtitle ?? "").toString(),
        e.color && /^#[0-9a-fA-F]{6}$/.test(e.color) ? e.color : undefined,
        e.icon ?? undefined,
        artifacts,
      ));
    }
    return engines.length ? engines : null;
  } catch {
    return null;
  }
}

// ─── ANIMATIONS (injected once) ──────────────────────────────────────────────

const STYLES = `
@keyframes ob-fadeUp {
  from { opacity: 0; transform: translateY(18px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes ob-slideIn {
  from { opacity: 0; transform: translateX(-16px); }
  to   { opacity: 1; transform: translateX(0); }
}
@keyframes ob-castleRise {
  0%   { opacity: 0; transform: scale(0.2) rotate(-25deg); }
  60%  { opacity: 1; transform: scale(1.15) rotate(4deg); }
  100% { opacity: 1; transform: scale(1) rotate(0deg); }
}
@keyframes ob-castleText {
  0%   { opacity: 0; transform: translateY(10px); }
  100% { opacity: 1; transform: translateY(0); }
}
@keyframes ob-pulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(62,207,207,0.18); }
  50%  { box-shadow: 0 0 0 8px rgba(62,207,207,0); }
}
.ob-card-anim { animation: ob-fadeUp 0.45s ease both; }
.ob-step-anim { animation: ob-slideIn 0.3s ease both; }
`;

function InjectStyles() {
  useEffect(() => {
    if (document.getElementById("ob-styles")) return;
    const el = document.createElement("style");
    el.id = "ob-styles";
    el.textContent = STYLES;
    document.head.appendChild(el);
    return () => { document.getElementById("ob-styles")?.remove(); };
  }, []);
  return null;
}

// ─── SHARED UI ────────────────────────────────────────────────────────────────

function Btn({
  children, onClick, disabled, variant = "primary", style: s,
}: {
  children: React.ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  variant?: "primary" | "ghost" | "danger";
  style?: React.CSSProperties;
}) {
  const base: React.CSSProperties = {
    border: "1px solid",
    borderRadius: 3,
    padding: "9px 22px",
    cursor: disabled ? "not-allowed" : "pointer",
    fontSize: 11,
    letterSpacing: "0.14em",
    textTransform: "uppercase",
    opacity: disabled ? 0.45 : 1,
    transition: "opacity 0.15s",
    fontFamily: "inherit",
  };
  const variants: Record<string, React.CSSProperties> = {
    primary: { background: "#3ecfcf18", borderColor: "#3ecfcf66", color: "#3ecfcf" },
    ghost:   { background: "transparent", borderColor: "#333", color: "#666" },
    danger:  { background: "transparent", borderColor: "#e0637a55", color: "#e0637a" },
  };
  return (
    <button onClick={onClick} disabled={disabled} style={{ ...base, ...variants[variant], ...s }}>
      {children}
    </button>
  );
}

function StepDots({ total, current }: { total: number; current: number }) {
  return (
    <div style={{ display: "flex", gap: 6, justifyContent: "center", margin: "24px 0 0" }}>
      {Array.from({ length: total }).map((_, i) => (
        <div key={i} style={{
          width: 6, height: 6, borderRadius: "50%",
          background: i === current ? "#3ecfcf" : "#222",
          border: "1px solid",
          borderColor: i === current ? "#3ecfcf" : "#333",
          transition: "all 0.2s",
        }} />
      ))}
    </div>
  );
}

// ─── PATH SELECTION ───────────────────────────────────────────────────────────

const PATHS = [
  {
    id: "chatgpt",
    icon: "⟁",
    color: "#9b7fe8",
    title: "Bring Your Stuff Over",
    description: "Have ChatGPT, Claude, or another AI tool export your projects and history into your castle.",
    tag: "AI Import",
  },
  {
    id: "palace",
    icon: "⬡",
    color: "#3ecfcf",
    title: "Build Your Palace",
    description: "Select your life domains and shape your castle from scratch with a guided setup.",
    tag: "Guided",
  },
  {
    id: "text",
    icon: "◈",
    color: "#c8a96e",
    title: "Text Import",
    description: "Paste raw notes, tasks, or a brain dump — MindCastle will organise it for you.",
    tag: "Quick",
  },
];

function PathSelection({ onSelect }: { onSelect: (id: string) => void }) {
  return (
    <div className="ob-step-anim" style={{ maxWidth: 720, margin: "0 auto", padding: "0 8px" }}>
      <div style={{ textAlign: "center", marginBottom: 48 }}>
        <div style={{ fontSize: 42, marginBottom: 16, opacity: 0.18 }}>⬢</div>
        <div style={{ fontSize: 24, fontFamily: "'Syne', sans-serif", fontWeight: 700, color: "#f0ece4", marginBottom: 8 }}>
          Welcome to MindCastle
        </div>
        <div style={{ fontSize: 12, color: "#555", lineHeight: 1.8 }}>
          How do you want to start building?
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16 }}>
        {PATHS.map((p, i) => (
          <button
            key={p.id}
            className="ob-card-anim"
            onClick={() => onSelect(p.id)}
            style={{
              animationDelay: `${i * 0.08}s`,
              background: "#0d0d0d",
              border: `1px solid #1a1a1a`,
              borderRadius: 6,
              padding: "28px 20px",
              cursor: "pointer",
              textAlign: "left",
              transition: "border-color 0.18s, background 0.18s",
            }}
            onMouseEnter={e => {
              (e.currentTarget as HTMLElement).style.borderColor = p.color + "66";
              (e.currentTarget as HTMLElement).style.background = p.color + "08";
            }}
            onMouseLeave={e => {
              (e.currentTarget as HTMLElement).style.borderColor = "#1a1a1a";
              (e.currentTarget as HTMLElement).style.background = "#0d0d0d";
            }}
          >
            <div style={{ fontSize: 22, color: p.color, marginBottom: 14 }}>{p.icon}</div>
            <div style={{
              display: "inline-block",
              fontSize: 9,
              letterSpacing: "0.15em",
              textTransform: "uppercase",
              color: p.color,
              background: p.color + "18",
              border: `1px solid ${p.color}44`,
              borderRadius: 2,
              padding: "2px 7px",
              marginBottom: 10,
            }}>{p.tag}</div>
            <div style={{ fontSize: 13, fontWeight: 600, color: "#f0ece4", marginBottom: 8, lineHeight: 1.3 }}>
              {p.title}
            </div>
            <div style={{ fontSize: 11, color: "#555", lineHeight: 1.7 }}>
              {p.description}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

// ─── PATH 1: CHATGPT IMPORT ───────────────────────────────────────────────────

function ChatGPTWizard({ onComplete, onBack }: { onComplete: (e: Engine[]) => void; onBack: () => void }) {
  const [step, setStep] = useState<1 | 2 | 3>(1);
  const [copied, setCopied] = useState(false);
  const [pasted, setPasted] = useState("");
  const [engines, setEngines] = useState<Engine[]>([]);
  const [parseError, setParseError] = useState<string | null>(null);

  function handleCopy() {
    navigator.clipboard.writeText(CHATGPT_PROMPT).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }

  function handleParse() {
    const result = parseImportJSON(pasted);
    if (!result) {
      setParseError("Couldn't parse that — make sure you pasted the JSON block the AI generated, with no extra text.");
      return;
    }
    setParseError(null);
    setEngines(result);
    setStep(3);
  }

  const label: Record<number, string> = { 1: "Copy the prompt", 2: "Paste the response", 3: "Review & import" };

  return (
    <div className="ob-step-anim" style={{ maxWidth: 640, margin: "0 auto" }}>
      <StepHeader title="Import from AI" step={step} total={3} label={label[step]} onBack={step === 1 ? onBack : () => setStep((step - 1) as 1 | 2)} />

      {step === 1 && (
        <div style={{ animation: "ob-slideIn 0.28s ease both" }}>
          <p style={{ fontSize: 12, color: "#666", lineHeight: 1.8, marginBottom: 20 }}>
            Copy the prompt below, paste it into ChatGPT, Claude, or any AI tool you've been using.
            The AI will export your projects and tasks in a format MindCastle can read.
          </p>
          <pre style={{
            background: "#0a0a0a",
            border: "1px solid #1e1e1e",
            borderRadius: 4,
            padding: 16,
            fontSize: 11,
            color: "#888",
            lineHeight: 1.7,
            whiteSpace: "pre-wrap",
            wordBreak: "break-word",
            maxHeight: 280,
            overflow: "auto",
          }}>{CHATGPT_PROMPT}</pre>
          <div style={{ display: "flex", gap: 10, marginTop: 20, justifyContent: "flex-end" }}>
            <Btn variant="ghost" onClick={onBack}>← Back</Btn>
            <Btn onClick={handleCopy}>{copied ? "✓ Copied!" : "Copy Prompt"}</Btn>
            <Btn onClick={() => setStep(2)}>I got a response →</Btn>
          </div>
        </div>
      )}

      {step === 2 && (
        <div style={{ animation: "ob-slideIn 0.28s ease both" }}>
          <p style={{ fontSize: 12, color: "#666", lineHeight: 1.8, marginBottom: 16 }}>
            Paste the JSON block the AI returned below.
          </p>
          <textarea
            value={pasted}
            onChange={e => { setPasted(e.target.value); setParseError(null); }}
            placeholder={'{\n  "version": "1",\n  "engines": [...]\n}'}
            style={{
              width: "100%",
              minHeight: 220,
              background: "#0a0a0a",
              border: `1px solid ${parseError ? "#e0637a66" : "#1e1e1e"}`,
              borderRadius: 4,
              color: "#bbb",
              fontSize: 11,
              fontFamily: "monospace",
              padding: 14,
              resize: "vertical",
              outline: "none",
              lineHeight: 1.6,
              boxSizing: "border-box",
            }}
          />
          {parseError && (
            <div style={{ fontSize: 11, color: "#e0637a", marginTop: 8, lineHeight: 1.6 }}>{parseError}</div>
          )}
          <div style={{ display: "flex", gap: 10, marginTop: 16, justifyContent: "flex-end" }}>
            <Btn variant="ghost" onClick={() => setStep(1)}>← Back</Btn>
            <Btn onClick={handleParse} disabled={!pasted.trim()}>Parse →</Btn>
          </div>
        </div>
      )}

      {step === 3 && (
        <div style={{ animation: "ob-slideIn 0.28s ease both" }}>
          <p style={{ fontSize: 12, color: "#666", marginBottom: 20, lineHeight: 1.8 }}>
            Found <strong style={{ color: "#f0ece4" }}>{engines.length} engine{engines.length !== 1 ? "s" : ""}</strong> with{" "}
            <strong style={{ color: "#f0ece4" }}>{engines.reduce((n, e) => n + e.artifacts.length, 0)} artifacts</strong>. Ready to import?
          </p>
          <div style={{ display: "flex", flexDirection: "column", gap: 8, maxHeight: 300, overflow: "auto" }}>
            {engines.map(e => (
              <EnginePreviewRow key={e.id} engine={e} />
            ))}
          </div>
          <div style={{ display: "flex", gap: 10, marginTop: 24, justifyContent: "flex-end" }}>
            <Btn variant="ghost" onClick={() => setStep(2)}>← Back</Btn>
            <Btn onClick={() => onComplete(engines)}>Build My Castle →</Btn>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── PATH 2: BUILD YOUR PALACE ────────────────────────────────────────────────

function PalaceWizard({ onComplete, onBack }: { onComplete: (e: Engine[]) => void; onBack: () => void }) {
  const [step, setStep] = useState<1 | 2 | 3>(1);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [customised, setCustomised] = useState<{ name: string; subtitle: string }[]>([]);
  const [celebrating, setCelebrating] = useState(false);

  function handleSelectToggle(name: string) {
    setSelected(prev => {
      const next = new Set(prev);
      next.has(name) ? next.delete(name) : next.add(name);
      return next;
    });
  }

  function handleToStep2() {
    const items = PRESETS.filter(p => selected.has(p.name));
    setCustomised(items.map(p => ({ name: p.name, subtitle: p.subtitle })));
    setStep(2);
  }

  function handleBuild() {
    setStep(3);
    setCelebrating(true);
    const engines = PRESETS
      .filter(p => selected.has(p.name))
      .map((p, i) => makeEngine(
        customised[i]?.name || p.name,
        customised[i]?.subtitle || p.subtitle,
        p.color,
        p.icon,
      ));
    setTimeout(() => onComplete(engines), 2200);
  }

  const label: Record<number, string> = {
    1: "Choose your engines",
    2: "Customise names",
    3: "Building your castle",
  };

  return (
    <div className="ob-step-anim" style={{ maxWidth: 700, margin: "0 auto" }}>
      <StepHeader title="Build Your Palace" step={step} total={3} label={label[step]}
        onBack={step === 1 ? onBack : () => setStep((step - 1) as 1 | 2)} />

      {step === 1 && (
        <div style={{ animation: "ob-slideIn 0.28s ease both" }}>
          <p style={{ fontSize: 12, color: "#666", lineHeight: 1.8, marginBottom: 20 }}>
            Pick the life domains you want to track. You can always add or remove engines later.
          </p>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10 }}>
            {PRESETS.map((p, i) => {
              const on = selected.has(p.name);
              return (
                <button
                  key={p.name}
                  className="ob-card-anim"
                  onClick={() => handleSelectToggle(p.name)}
                  style={{
                    animationDelay: `${i * 0.05}s`,
                    background: on ? p.color + "14" : "#0d0d0d",
                    border: `1px solid ${on ? p.color + "88" : "#1a1a1a"}`,
                    borderRadius: 5,
                    padding: "16px 12px",
                    cursor: "pointer",
                    textAlign: "center",
                    transition: "all 0.15s",
                  }}
                >
                  <div style={{ fontSize: 20, color: on ? p.color : "#444", marginBottom: 8, transition: "color 0.15s" }}>
                    {p.icon}
                  </div>
                  <div style={{ fontSize: 11, fontWeight: 600, color: on ? "#f0ece4" : "#666", marginBottom: 4 }}>
                    {p.name}
                  </div>
                  <div style={{ fontSize: 9.5, color: on ? "#888" : "#333", lineHeight: 1.5 }}>
                    {p.subtitle}
                  </div>
                </button>
              );
            })}
          </div>
          <div style={{ display: "flex", gap: 10, marginTop: 24, justifyContent: "flex-end", alignItems: "center" }}>
            <span style={{ fontSize: 11, color: "#555", flex: 1 }}>
              {selected.size === 0 ? "Select at least one" : `${selected.size} selected`}
            </span>
            <Btn variant="ghost" onClick={onBack}>← Back</Btn>
            <Btn onClick={handleToStep2} disabled={selected.size === 0}>Customise →</Btn>
          </div>
        </div>
      )}

      {step === 2 && (
        <div style={{ animation: "ob-slideIn 0.28s ease both" }}>
          <p style={{ fontSize: 12, color: "#666", lineHeight: 1.8, marginBottom: 20 }}>
            Rename your engines if you'd like — or keep the defaults.
          </p>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {PRESETS.filter(p => selected.has(p.name)).map((p, i) => (
              <div key={p.name} style={{
                display: "flex",
                alignItems: "center",
                gap: 12,
                background: "#0d0d0d",
                border: "1px solid #1a1a1a",
                borderRadius: 5,
                padding: "12px 16px",
              }}>
                <div style={{ fontSize: 18, color: p.color, width: 28, textAlign: "center" }}>{p.icon}</div>
                <input
                  value={customised[i]?.name ?? p.name}
                  onChange={e => setCustomised(prev => {
                    const next = [...prev];
                    next[i] = { ...next[i], name: e.target.value };
                    return next;
                  })}
                  style={{
                    background: "transparent",
                    border: "none",
                    borderBottom: "1px solid #2a2a2a",
                    color: "#f0ece4",
                    fontSize: 12,
                    fontFamily: "inherit",
                    outline: "none",
                    flex: 1,
                    padding: "2px 0",
                  }}
                />
                <input
                  value={customised[i]?.subtitle ?? p.subtitle}
                  onChange={e => setCustomised(prev => {
                    const next = [...prev];
                    next[i] = { ...next[i], subtitle: e.target.value };
                    return next;
                  })}
                  placeholder="Subtitle (optional)"
                  style={{
                    background: "transparent",
                    border: "none",
                    borderBottom: "1px solid #1e1e1e",
                    color: "#666",
                    fontSize: 11,
                    fontFamily: "inherit",
                    outline: "none",
                    flex: 1.2,
                    padding: "2px 0",
                  }}
                />
              </div>
            ))}
          </div>
          <div style={{ display: "flex", gap: 10, marginTop: 24, justifyContent: "flex-end" }}>
            <Btn variant="ghost" onClick={() => setStep(1)}>← Back</Btn>
            <Btn onClick={handleBuild}>Build My Castle →</Btn>
          </div>
        </div>
      )}

      {step === 3 && (
        <div style={{ textAlign: "center", padding: "60px 0" }}>
          <div style={{
            fontSize: 72,
            color: "#3ecfcf",
            animation: celebrating ? "ob-castleRise 0.8s cubic-bezier(0.175,0.885,0.32,1.275) both" : "none",
            display: "inline-block",
            marginBottom: 28,
          }}>⬢</div>
          <div style={{
            fontSize: 22,
            fontFamily: "'Syne', sans-serif",
            fontWeight: 700,
            color: "#f0ece4",
            marginBottom: 10,
            animation: celebrating ? "ob-castleText 0.5s ease 0.7s both" : "none",
            opacity: 0,
          }}>Your castle is rising.</div>
          <div style={{
            fontSize: 12,
            color: "#555",
            animation: celebrating ? "ob-castleText 0.5s ease 0.9s both" : "none",
            opacity: 0,
          }}>Setting up {selected.size} engine{selected.size !== 1 ? "s" : ""}…</div>
        </div>
      )}
    </div>
  );
}

// ─── PATH 3: TEXT IMPORT ──────────────────────────────────────────────────────

function TextWizard({ onComplete, onBack }: { onComplete: (e: Engine[]) => void; onBack: () => void }) {
  const [step, setStep] = useState<1 | 2 | 3>(1);
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [engines, setEngines] = useState<Engine[]>([]);
  const [error, setError] = useState<string | null>(null);

  async function handleAnalyse() {
    setLoading(true);
    setError(null);
    try {
      // Split into sentences / paragraphs (short chunks)
      const chunks = text
        .split(/\n+/)
        .map(l => l.trim())
        .filter(l => l.length > 3);

      if (!chunks.length) {
        setError("No text to analyse — try adding some notes or tasks.");
        setLoading(false);
        return;
      }

      // Classify each chunk, collect by top category
      const grouped: Record<string, { text: string; notes: string }[]> = {};
      for (const chunk of chunks.slice(0, 20)) { // cap at 20 chunks for perf
        try {
          const result = await api.classifyText(chunk);
          const topCat = result.categories[0]?.label ?? "General";
          if (!grouped[topCat]) grouped[topCat] = [];
          grouped[topCat].push({ text: chunk, notes: chunk });
        } catch {
          // If API is offline, fall back to a single "General" engine
          if (!grouped["General"]) grouped["General"] = [];
          grouped["General"].push({ text: chunk, notes: chunk });
        }
      }

      const built: Engine[] = Object.entries(grouped).map(([cat, items]) =>
        makeEngine(
          cat,
          `Imported from text`,
          undefined,
          undefined,
          items.map(i => makeArtifact(
            i.text.length > 60 ? i.text.slice(0, 57) + "…" : i.text,
            i.notes,
            "concept",
          )),
        ),
      );

      setEngines(built);
      setStep(3);
    } catch (e) {
      setError("Something went wrong analysing your text. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  }

  const label: Record<number, string> = {
    1: "Paste your text",
    2: "Analysing…",
    3: "Review & import",
  };

  return (
    <div className="ob-step-anim" style={{ maxWidth: 640, margin: "0 auto" }}>
      <StepHeader title="Text Import" step={step} total={3} label={label[loading ? 2 : step]}
        onBack={step === 1 ? onBack : () => { setStep(1); setEngines([]); }} />

      {step === 1 && (
        <div style={{ animation: "ob-slideIn 0.28s ease both" }}>
          <p style={{ fontSize: 12, color: "#666", lineHeight: 1.8, marginBottom: 16 }}>
            Paste any raw notes, tasks, journal entries, or ideas below. MindCastle will group them into engines automatically.
          </p>
          <textarea
            value={text}
            onChange={e => { setText(e.target.value); setError(null); }}
            placeholder={"Ran 5k this morning.\nWorking on the auth module for the API.\nReading Dune — halfway through.\nNeed to call the dentist.\nLearn Rust — chapter 4 done."}
            style={{
              width: "100%",
              minHeight: 200,
              background: "#0a0a0a",
              border: "1px solid #1e1e1e",
              borderRadius: 4,
              color: "#bbb",
              fontSize: 11,
              fontFamily: "monospace",
              padding: 14,
              resize: "vertical",
              outline: "none",
              lineHeight: 1.7,
              boxSizing: "border-box",
            }}
          />
          {error && <div style={{ fontSize: 11, color: "#e0637a", marginTop: 8 }}>{error}</div>}
          <div style={{ display: "flex", gap: 10, marginTop: 16, justifyContent: "flex-end" }}>
            <Btn variant="ghost" onClick={onBack}>← Back</Btn>
            <Btn onClick={handleAnalyse} disabled={!text.trim() || loading}>
              {loading ? "Analysing…" : "Analyse →"}
            </Btn>
          </div>
        </div>
      )}

      {step === 3 && (
        <div style={{ animation: "ob-slideIn 0.28s ease both" }}>
          <p style={{ fontSize: 12, color: "#666", marginBottom: 20, lineHeight: 1.8 }}>
            Grouped into <strong style={{ color: "#f0ece4" }}>{engines.length} engine{engines.length !== 1 ? "s" : ""}</strong> with{" "}
            <strong style={{ color: "#f0ece4" }}>{engines.reduce((n, e) => n + e.artifacts.length, 0)} artifacts</strong>.
          </p>
          <div style={{ display: "flex", flexDirection: "column", gap: 8, maxHeight: 300, overflow: "auto" }}>
            {engines.map(e => (
              <EnginePreviewRow key={e.id} engine={e} />
            ))}
          </div>
          <div style={{ display: "flex", gap: 10, marginTop: 24, justifyContent: "flex-end" }}>
            <Btn variant="ghost" onClick={() => { setStep(1); setEngines([]); }}>← Redo</Btn>
            <Btn onClick={() => onComplete(engines)}>Import →</Btn>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── SHARED SUB-COMPONENTS ────────────────────────────────────────────────────

function StepHeader({
  title, step, total, label, onBack,
}: {
  title: string; step: number; total: number; label: string; onBack: () => void;
}) {
  return (
    <div style={{ marginBottom: 32 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 20 }}>
        <button onClick={onBack} style={{
          background: "transparent",
          border: "none",
          color: "#444",
          cursor: "pointer",
          fontSize: 16,
          padding: "0 4px",
          lineHeight: 1,
        }}>←</button>
        <span style={{ fontSize: 11, color: "#555", letterSpacing: "0.12em", textTransform: "uppercase" }}>
          {title}
        </span>
      </div>
      <div style={{ fontSize: 13, color: "#f0ece4", fontWeight: 600, marginBottom: 4 }}>{label}</div>
      <StepDots total={total} current={step - 1} />
    </div>
  );
}

function EnginePreviewRow({ engine }: { engine: Engine }) {
  return (
    <div style={{
      display: "flex",
      alignItems: "center",
      gap: 12,
      background: "#0d0d0d",
      border: "1px solid #1a1a1a",
      borderRadius: 4,
      padding: "10px 14px",
    }}>
      <div style={{ fontSize: 16, color: engine.color, width: 24, textAlign: "center" }}>{engine.icon}</div>
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: 12, color: "#f0ece4", fontWeight: 600 }}>{engine.name}</div>
        {engine.subtitle && (
          <div style={{ fontSize: 10.5, color: "#555", marginTop: 2 }}>{engine.subtitle}</div>
        )}
      </div>
      <div style={{ fontSize: 10, color: "#444", letterSpacing: "0.1em" }}>
        {engine.artifacts.length} artifact{engine.artifacts.length !== 1 ? "s" : ""}
      </div>
    </div>
  );
}

// ─── ROOT ─────────────────────────────────────────────────────────────────────

type View = "select" | "chatgpt" | "palace" | "text";

export function Onboarding({ onComplete, onSkip }: OnboardingProps) {
  const [view, setView] = useState<View>("select");

  return (
    <div style={{
      minHeight: "60vh",
      display: "flex",
      flexDirection: "column",
      justifyContent: "center",
      padding: "40px 0 60px",
    }}>
      <InjectStyles />

      {view === "select" && (
        <>
          <PathSelection onSelect={(id) => setView(id as View)} />
          <div style={{ textAlign: "center", marginTop: 40 }}>
            <button onClick={onSkip} style={{
              background: "transparent",
              border: "none",
              color: "#333",
              cursor: "pointer",
              fontSize: 11,
              letterSpacing: "0.12em",
              textDecoration: "underline",
              textUnderlineOffset: 3,
            }}>
              skip — set up manually
            </button>
          </div>
        </>
      )}

      {view === "chatgpt" && (
        <ChatGPTWizard
          onComplete={(engines) => onComplete(engines)}
          onBack={() => setView("select")}
        />
      )}

      {view === "palace" && (
        <PalaceWizard
          onComplete={(engines) => onComplete(engines)}
          onBack={() => setView("select")}
        />
      )}

      {view === "text" && (
        <TextWizard
          onComplete={(engines) => onComplete(engines)}
          onBack={() => setView("select")}
        />
      )}
    </div>
  );
}
