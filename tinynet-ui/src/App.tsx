import { useState, useEffect, useRef } from "react";
import * as api from "./api";
import { Onboarding } from "./Onboarding";
import { ThemeProvider, useTheme } from "./theme/ThemeProvider";
import { CastleViewport } from "./components/CastleViewport";
import type { ThemeName } from "./theme/tokens";

// ─── TYPES ───────────────────────────────────────────────────────────────────

interface ProgressItem {
  label: string;
  val: number;
}

interface Artifact {
  id: string;
  node_id?: string;  // backend Node UUID, set after first API sync
  title: string;
  subtitle: string;
  status: string;
  next: string;
  stack: string[];
  progress: ProgressItem[];
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

// ─── CONSTANTS ───────────────────────────────────────────────────────────────

const STATUS_META: Record<string, { dot: string; label: string }> = {
  active:   { dot: "#a3d977", label: "Active" },
  live:     { dot: "#a3d977", label: "Live" },
  concept:  { dot: "#666",    label: "Concept" },
  blocked:  { dot: "#e0637a", label: "Blocked" },
  planned:  { dot: "#9b7fe8", label: "Planned" },
  planning: { dot: "#c8a96e", label: "Planning" },
  // TinyNet-classified states
  start:    { dot: "#5b8dee", label: "Start" },
  continue: { dot: "#a3d977", label: "Continue" },
  pause:    { dot: "#c8a96e", label: "Pause" },
  end:      { dot: "#555",    label: "End" },
  idea:     { dot: "#9b7fe8", label: "Idea" },
};

const PALETTE = [
  "#3ecfcf", "#9b7fe8", "#c8a96e", "#e0637a",
  "#a3d977", "#5b8dee", "#f0a05a", "#d46b8a",
];

const ICONS = ["◈", "✦", "◎", "⬡", "⟁", "◇", "⊕", "⬢", "△", "○"];

const STATUS_OPTIONS = ["active", "live", "concept", "blocked", "planned", "planning"];

const EMPTY_ARTIFACT = (): Artifact => ({
  id: crypto.randomUUID(),
  title: "",
  subtitle: "",
  status: "concept",
  next: "",
  stack: [],
  progress: [],
  notes: "",
});

const EMPTY_ENGINE = (): Engine => ({
  id: crypto.randomUUID(),
  name: "",
  subtitle: "",
  color: PALETTE[Math.floor(Math.random() * PALETTE.length)],
  icon: ICONS[Math.floor(Math.random() * ICONS.length)],
  context: "",
  contextLabel: "Context",
  description: "",
  artifacts: [],
});

const STORAGE_KEY = "mindcastle_v1";

// ─── PERSISTENCE ─────────────────────────────────────────────────────────────

function load(): Engine[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function save(engines: Engine[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(engines));
}

// ─── SMALL COMPONENTS ────────────────────────────────────────────────────────

function StatusDot({ status }: { status: string }) {
  const m = STATUS_META[status] || STATUS_META.concept;
  return (
    <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <span style={{
        width: 7, height: 7, borderRadius: "50%", background: m.dot, flexShrink: 0,
        boxShadow: (status === "active" || status === "live") ? `0 0 6px ${m.dot}` : "none",
      }} />
      <span style={{ fontSize: 10, letterSpacing: "0.12em", textTransform: "uppercase", color: "#666" }}>
        {m.label}
      </span>
    </span>
  );
}

function ProgressBar({ label, val, color }: { label: string; val: number; color: string }) {
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#555", marginBottom: 4 }}>
        <span>{label}</span><span>{val}%</span>
      </div>
      <div style={{ height: 2, background: "#1e1e1e", borderRadius: 2 }}>
        <div style={{ height: "100%", width: `${val}%`, background: color, borderRadius: 2, transition: "width 0.6s ease" }} />
      </div>
    </div>
  );
}

function Tag({ label }: { label: string }) {
  return (
    <span style={{
      fontSize: 9, padding: "2px 7px", background: "#1a1a1a",
      border: "1px solid #222", borderRadius: 2, color: "#555",
      letterSpacing: "0.08em", textTransform: "uppercase",
    }}>{label}</span>
  );
}

function IconBtn({ onClick, children, title }: { onClick: (e: React.MouseEvent<HTMLButtonElement>) => void; children: React.ReactNode; title?: string }) {
  return (
    <button title={title} onClick={onClick} style={{
      background: "none", border: "none", color: "#444", cursor: "pointer",
      fontSize: 13, padding: "2px 5px", borderRadius: 2,
      transition: "color 0.15s",
    }}
      onMouseEnter={e => (e.currentTarget.style.color = "#888")}
      onMouseLeave={e => (e.currentTarget.style.color = "#444")}
    >
      {children}
    </button>
  );
}

// ─── MODAL ───────────────────────────────────────────────────────────────────

function Modal({ onClose, children, width = 560 }: {
  onClose: () => void;
  children: React.ReactNode;
  width?: number;
}) {
  return (
    <div style={{
      position: "fixed", inset: 0, background: "rgba(0,0,0,0.88)", zIndex: 200,
      display: "flex", alignItems: "center", justifyContent: "center", padding: 24,
    }} onClick={onClose}>
      <div style={{
        background: "#0d0d0d", border: "1px solid #222", borderRadius: 6,
        padding: 32, maxWidth: width, width: "100%", maxHeight: "90vh", overflow: "auto",
      }} onClick={e => e.stopPropagation()}>
        {children}
      </div>
    </div>
  );
}

function ModalLabel({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ fontSize: 9, letterSpacing: "0.15em", textTransform: "uppercase", color: "#444", marginBottom: 6, marginTop: 16 }}>
      {children}
    </div>
  );
}

function ModalInput({ value, onChange, placeholder, multiline = false }: {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  multiline?: boolean;
}) {
  const style: React.CSSProperties = {
    width: "100%", background: "#111", border: "1px solid #1e1e1e",
    color: "#c8c4bc", fontSize: 12, padding: "9px 12px", borderRadius: 3,
    fontFamily: "'DM Mono', monospace", outline: "none", resize: "vertical",
  };
  return multiline
    ? <textarea value={value} onChange={e => onChange(e.target.value)} placeholder={placeholder} rows={4} style={style} />
    : <input value={value} onChange={e => onChange(e.target.value)} placeholder={placeholder} style={style} />;
}

// ─── ARTIFACT EDITOR MODAL ────────────────────────────────────────────────────

function ArtifactEditor({ artifact, color, onSave, onClose, onDelete }: {
  artifact: Artifact;
  color: string;
  onSave: (a: Artifact) => void;
  onClose: () => void;
  onDelete: () => void;
}) {
  const [draft, setDraft] = useState<Artifact>({ ...artifact, stack: [...artifact.stack], progress: artifact.progress.map(p => ({ ...p })) });
  const [stackInput, setStackInput] = useState("");

  const set = <K extends keyof Artifact>(key: K, val: Artifact[K]) => setDraft(d => ({ ...d, [key]: val }));

  const addStack = () => {
    const t = stackInput.trim();
    if (t && !draft.stack.includes(t)) {
      set("stack", [...draft.stack, t]);
    }
    setStackInput("");
  };

  const addProgress = () => set("progress", [...draft.progress, { label: "", val: 0 }]);

  const updateProgress = (i: number, key: keyof ProgressItem, val: string | number) => {
    const next = draft.progress.map((p, idx) => idx === i ? { ...p, [key]: val } : p);
    set("progress", next);
  };

  return (
    <Modal onClose={onClose} width={560}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
        <div style={{ fontSize: 10, color: color, letterSpacing: "0.2em", textTransform: "uppercase" }}>
          {artifact.title ? "Edit Artifact" : "New Artifact"}
        </div>
        <button onClick={onDelete} style={{ background: "none", border: "none", color: "#e0637a", cursor: "pointer", fontSize: 10, letterSpacing: "0.1em" }}>
          Delete
        </button>
      </div>

      <ModalLabel>Title</ModalLabel>
      <ModalInput value={draft.title} onChange={v => set("title", v)} placeholder="Artifact name" />

      <ModalLabel>Subtitle</ModalLabel>
      <ModalInput value={draft.subtitle} onChange={v => set("subtitle", v)} placeholder="One-line description" />

      <ModalLabel>Status</ModalLabel>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
        {STATUS_OPTIONS.map(s => {
          const m = STATUS_META[s];
          const active = draft.status === s;
          return (
            <button key={s} onClick={() => set("status", s)} style={{
              background: active ? `${m.dot}15` : "#111",
              border: `1px solid ${active ? m.dot : "#222"}`,
              color: active ? m.dot : "#444",
              borderRadius: 3, padding: "4px 10px", cursor: "pointer",
              fontSize: 10, letterSpacing: "0.1em", textTransform: "uppercase",
            }}>{s}</button>
          );
        })}
      </div>

      <ModalLabel>Next Step</ModalLabel>
      <ModalInput value={draft.next} onChange={v => set("next", v)} placeholder="What happens next?" />

      <ModalLabel>Notes / Context</ModalLabel>
      <ModalInput value={draft.notes} onChange={v => set("notes", v)} placeholder="Background, blockers, decisions..." multiline />

      <ModalLabel>Stack / Tags</ModalLabel>
      <div style={{ display: "flex", gap: 6, marginBottom: 8 }}>
        <input
          value={stackInput}
          onChange={e => setStackInput(e.target.value)}
          onKeyDown={e => e.key === "Enter" && addStack()}
          placeholder="Add tag and press Enter"
          style={{ flex: 1, background: "#111", border: "1px solid #1e1e1e", color: "#c8c4bc", fontSize: 11, padding: "7px 10px", borderRadius: 3, fontFamily: "'DM Mono', monospace", outline: "none" }}
        />
        <button onClick={addStack} style={{ background: "#1a1a1a", border: "1px solid #222", color: "#555", borderRadius: 3, padding: "0 12px", cursor: "pointer", fontSize: 11 }}>+</button>
      </div>
      {draft.stack.length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 5, marginBottom: 4 }}>
          {draft.stack.map(t => (
            <span key={t} onClick={() => set("stack", draft.stack.filter(x => x !== t))}
              style={{ fontSize: 9, padding: "2px 7px", background: "#1a1a1a", border: "1px solid #2a2a2a", borderRadius: 2, color: "#666", cursor: "pointer", letterSpacing: "0.08em", textTransform: "uppercase" }}
              title="Click to remove">{t} ×</span>
          ))}
        </div>
      )}

      <ModalLabel>Progress Bars</ModalLabel>
      {draft.progress.map((p, i) => (
        <div key={i} style={{ display: "flex", gap: 6, marginBottom: 6, alignItems: "center" }}>
          <input value={p.label} onChange={e => updateProgress(i, "label", e.target.value)}
            placeholder="Label" style={{ flex: 1, background: "#111", border: "1px solid #1e1e1e", color: "#c8c4bc", fontSize: 11, padding: "6px 10px", borderRadius: 3, fontFamily: "'DM Mono', monospace", outline: "none" }} />
          <input type="number" min={0} max={100} value={p.val} onChange={e => updateProgress(i, "val", Number(e.target.value))}
            style={{ width: 56, background: "#111", border: "1px solid #1e1e1e", color: "#c8c4bc", fontSize: 11, padding: "6px 8px", borderRadius: 3, fontFamily: "'DM Mono', monospace", outline: "none", textAlign: "center" }} />
          <button onClick={() => set("progress", draft.progress.filter((_, idx) => idx !== i))}
            style={{ background: "none", border: "none", color: "#444", cursor: "pointer", fontSize: 14 }}>×</button>
        </div>
      ))}
      <button onClick={addProgress} style={{ background: "#111", border: "1px solid #1e1e1e", color: "#555", borderRadius: 3, padding: "6px 12px", cursor: "pointer", fontSize: 10, letterSpacing: "0.1em", marginTop: 2 }}>
        + Add Progress Bar
      </button>

      <div style={{ display: "flex", gap: 8, marginTop: 24 }}>
        <button onClick={() => onSave(draft)} style={{
          background: `${color}18`, border: `1px solid ${color}55`, color: color,
          borderRadius: 3, padding: "9px 20px", cursor: "pointer", fontSize: 11, letterSpacing: "0.1em",
        }}>Save</button>
        <button onClick={onClose} style={{
          background: "none", border: "1px solid #1e1e1e", color: "#444",
          borderRadius: 3, padding: "9px 20px", cursor: "pointer", fontSize: 11,
        }}>Cancel</button>
      </div>
    </Modal>
  );
}

// ─── ENGINE EDITOR MODAL ─────────────────────────────────────────────────────

function EngineEditor({ engine, onSave, onClose, onDelete, isNew }: {
  engine: Engine;
  onSave: (e: Engine) => void;
  onClose: () => void;
  onDelete: () => void;
  isNew: boolean;
}) {
  const [draft, setDraft] = useState<Engine>({ ...engine });
  const set = <K extends keyof Engine>(key: K, val: Engine[K]) => setDraft(d => ({ ...d, [key]: val }));

  return (
    <Modal onClose={onClose} width={520}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
        <div style={{ fontSize: 10, color: draft.color, letterSpacing: "0.2em", textTransform: "uppercase" }}>
          {isNew ? "New Engine" : "Edit Engine"}
        </div>
        {!isNew && (
          <button onClick={onDelete} style={{ background: "none", border: "none", color: "#e0637a", cursor: "pointer", fontSize: 10, letterSpacing: "0.1em" }}>
            Delete Engine
          </button>
        )}
      </div>

      <ModalLabel>Engine Name</ModalLabel>
      <ModalInput value={draft.name} onChange={v => set("name", v)} placeholder="e.g. Tech Builder, Writer, Health..." />

      <ModalLabel>Subtitle</ModalLabel>
      <ModalInput value={draft.subtitle} onChange={v => set("subtitle", v)} placeholder="Short descriptor" />

      <ModalLabel>Description</ModalLabel>
      <ModalInput value={draft.description} onChange={v => set("description", v)} placeholder="What this engine covers" />

      <ModalLabel>Icon</ModalLabel>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
        {ICONS.map(ic => (
          <button key={ic} onClick={() => set("icon", ic)} style={{
            background: draft.icon === ic ? `${draft.color}18` : "#111",
            border: `1px solid ${draft.icon === ic ? draft.color : "#222"}`,
            color: draft.icon === ic ? draft.color : "#555",
            borderRadius: 3, padding: "5px 10px", cursor: "pointer", fontSize: 14,
          }}>{ic}</button>
        ))}
      </div>

      <ModalLabel>Color</ModalLabel>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
        {PALETTE.map(c => (
          <button key={c} onClick={() => set("color", c)} style={{
            width: 24, height: 24, background: c, border: `2px solid ${draft.color === c ? "#fff" : "transparent"}`,
            borderRadius: "50%", cursor: "pointer",
          }} />
        ))}
        <input type="color" value={draft.color} onChange={e => set("color", e.target.value)}
          style={{ width: 24, height: 24, padding: 0, border: "1px solid #333", borderRadius: "50%", cursor: "pointer", background: "none" }} />
      </div>

      <ModalLabel>Context Label</ModalLabel>
      <ModalInput value={draft.contextLabel} onChange={v => set("contextLabel", v)} placeholder="e.g. Tech Builder Context" />

      <ModalLabel>Context (injected into AI chats)</ModalLabel>
      <ModalInput value={draft.context} onChange={v => set("context", v)} placeholder="Background, goals, constraints for this engine..." multiline />

      <div style={{ display: "flex", gap: 8, marginTop: 24 }}>
        <button onClick={() => onSave(draft)} style={{
          background: `${draft.color}18`, border: `1px solid ${draft.color}55`, color: draft.color,
          borderRadius: 3, padding: "9px 20px", cursor: "pointer", fontSize: 11, letterSpacing: "0.1em",
        }}>Save Engine</button>
        <button onClick={onClose} style={{
          background: "none", border: "1px solid #1e1e1e", color: "#444",
          borderRadius: 3, padding: "9px 20px", cursor: "pointer", fontSize: 11,
        }}>Cancel</button>
      </div>
    </Modal>
  );
}

// ─── CONTEXT PAGE ────────────────────────────────────────────────────────────

function ContextPage({ engine, onClose }: { engine: Engine; onClose: () => void }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard.writeText(engine.context);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  return (
    <Modal onClose={onClose} width={680}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
        <div>
          <div style={{ fontSize: 10, color: engine.color, letterSpacing: "0.2em", textTransform: "uppercase", marginBottom: 6 }}>
            🔒 Private Context
          </div>
          <div style={{ fontSize: 20, fontFamily: "'Syne', sans-serif", fontWeight: 700, color: "#f0ece4" }}>
            {engine.contextLabel}
          </div>
        </div>
        <button onClick={onClose} style={{ background: "none", border: "none", color: "#555", cursor: "pointer", fontSize: 22 }}>×</button>
      </div>

      <div style={{ fontSize: 11, color: "#444", marginBottom: 16, fontStyle: "italic", lineHeight: 1.6 }}>
        This context is injected when chatting within this engine. Paste it into your AI assistant's system prompt or project instructions.
      </div>

      {engine.context ? (
        <div style={{
          background: "#111", border: "1px solid #1e1e1e", borderRadius: 4,
          padding: 20, fontFamily: "'DM Mono', monospace", fontSize: 11,
          color: "#888", lineHeight: 1.8, whiteSpace: "pre-wrap",
        }}>
          {engine.context}
        </div>
      ) : (
        <div style={{ color: "#333", fontSize: 11, fontStyle: "italic", padding: 20, border: "1px dashed #1e1e1e", borderRadius: 4 }}>
          No context set yet. Edit this engine to add context.
        </div>
      )}

      <div style={{ display: "flex", gap: 10, marginTop: 16 }}>
        <button onClick={copy} style={{
          background: copied ? `${engine.color}22` : "#1a1a1a",
          border: `1px solid ${copied ? engine.color : "#2a2a2a"}`,
          color: copied ? engine.color : "#666",
          borderRadius: 3, padding: "8px 16px", cursor: "pointer",
          fontSize: 11, letterSpacing: "0.1em", transition: "all 0.2s",
        }}>
          {copied ? "✓ Copied" : "Copy to Clipboard"}
        </button>
        <button onClick={onClose} style={{
          background: "none", border: "1px solid #1e1e1e", color: "#444",
          borderRadius: 3, padding: "8px 16px", cursor: "pointer", fontSize: 11,
        }}>Close</button>
      </div>
    </Modal>
  );
}

// ─── ARTIFACT CARD ────────────────────────────────────────────────────────────

function ArtifactCard({ artifact, color, onSelect, selected, onEdit }: {
  artifact: Artifact;
  color: string;
  onSelect: (a: Artifact) => void;
  selected: boolean;
  onEdit: (a: Artifact) => void;
}) {
  return (
    <div
      onClick={() => onSelect(artifact)}
      style={{
        background: selected ? "rgba(255,255,255,0.04)" : "#111",
        border: `1px solid ${selected ? color : "#1e1e1e"}`,
        borderTop: `2px solid ${color}`,
        borderRadius: 4, padding: 20, cursor: "pointer",
        transition: "all 0.2s", position: "relative",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 8 }}>
        <div style={{ fontSize: 13, fontFamily: "'Syne', sans-serif", fontWeight: 700, color: "#f0ece4", lineHeight: 1.2, maxWidth: "72%" }}>
          {artifact.title || <span style={{ color: "#333" }}>Untitled</span>}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <StatusDot status={artifact.status} />
          <IconBtn onClick={e => { e.stopPropagation(); onEdit(artifact); }} title="Edit artifact">✎</IconBtn>
        </div>
      </div>
      <div style={{ fontSize: 11, color: "#555", marginBottom: 14, lineHeight: 1.5 }}>{artifact.subtitle}</div>

      {artifact.progress.slice(0, 2).map(p => (
        <ProgressBar key={p.label} label={p.label} val={p.val} color={color} />
      ))}

      {artifact.stack.length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 10 }}>
          {artifact.stack.slice(0, 5).map(t => <Tag key={t} label={t} />)}
          {artifact.stack.length > 5 && <span style={{ fontSize: 9, color: "#444" }}>+{artifact.stack.length - 5}</span>}
        </div>
      )}

      {artifact.next && (
        <div style={{ borderTop: "1px solid #1e1e1e", paddingTop: 10, marginTop: 12 }}>
          <div style={{ fontSize: 9, letterSpacing: "0.15em", textTransform: "uppercase", color: "#333", marginBottom: 4 }}>Next</div>
          <div style={{ fontSize: 11, color: "#555", lineHeight: 1.5 }}>{artifact.next}</div>
        </div>
      )}
    </div>
  );
}

// ─── LOG PANEL ───────────────────────────────────────────────────────────────

function LogPanel({ artifact, color, onStatusUpdate }: {
  artifact: Artifact;
  color: string;
  onStatusUpdate: (status: string, nextStep: string) => void;
}) {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<api.ClassifyResult | null>(null);
  const [logs, setLogs] = useState<api.ApiLog[]>([]);
  const [error, setError] = useState<string | null>(null);

  // Load existing logs if artifact has a backend node
  useEffect(() => {
    if (!artifact.node_id) return;
    api.getLogs(artifact.node_id)
      .then(r => setLogs(r.items))
      .catch(() => {}); // API offline is fine for MVP
  }, [artifact.node_id]);

  const submit = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const classified = await api.classifyText(text);
      setResult(classified);

      if (artifact.node_id) {
        const log = await api.addLog(
          artifact.node_id,
          text,
          classified.state.label,
          classified.nextStep.template,
        );
        setLogs(prev => [log, ...prev]);
        onStatusUpdate(classified.state.label, classified.nextStep.template);
      }
      setText("");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ marginTop: 20, borderTop: "1px solid #1a1a1a", paddingTop: 16 }}>
      <div style={{ fontSize: 9, letterSpacing: "0.15em", textTransform: "uppercase", color: "#333", marginBottom: 10 }}>
        Log Progress
      </div>

      {!artifact.node_id && (
        <div style={{ fontSize: 10, color: "#333", fontStyle: "italic", marginBottom: 8 }}>
          Save this artifact first to enable logging.
        </div>
      )}

      <textarea
        value={text}
        onChange={e => setText(e.target.value)}
        onKeyDown={e => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit(); }}
        placeholder="What happened? What's next?  (⌘↵ to submit)"
        disabled={!artifact.node_id || loading}
        rows={3}
        style={{
          width: "100%", background: "#111", border: "1px solid #1e1e1e",
          color: "#c8c4bc", fontSize: 11, padding: "9px 12px", borderRadius: 3,
          fontFamily: "'DM Mono', monospace", outline: "none", resize: "vertical",
          opacity: !artifact.node_id ? 0.4 : 1,
        }}
      />
      <button
        onClick={submit}
        disabled={!artifact.node_id || loading || !text.trim()}
        style={{
          marginTop: 6, background: `${color}18`, border: `1px solid ${color}44`,
          color, borderRadius: 3, padding: "6px 14px", cursor: "pointer",
          fontSize: 10, letterSpacing: "0.1em", transition: "all 0.2s",
          opacity: (!artifact.node_id || !text.trim()) ? 0.4 : 1,
        }}
      >
        {loading ? "Classifying..." : "Classify & Log"}
      </button>

      {error && (
        <div style={{ marginTop: 8, fontSize: 10, color: "#e0637a" }}>
          {error.includes("fetch") ? "API offline — log not saved" : error}
        </div>
      )}

      {result && (
        <div style={{ marginTop: 10, padding: 12, background: "#0d0d0d", border: `1px solid ${color}22`, borderRadius: 3 }}>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 5, marginBottom: 8 }}>
            {result.categories.map(c => (
              <span key={c.label} style={{
                fontSize: 9, padding: "2px 7px", background: `${color}18`,
                border: `1px solid ${color}33`, borderRadius: 2, color,
                letterSpacing: "0.08em", textTransform: "uppercase",
              }}>{c.label} {Math.round(c.score * 100)}%</span>
            ))}
          </div>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10 }}>
            <StatusDot status={result.state.label} />
            <span style={{ color: "#555" }}>→ {result.nextStep.template}</span>
          </div>
        </div>
      )}

      {logs.length > 0 && (
        <div style={{ marginTop: 14 }}>
          <div style={{ fontSize: 9, letterSpacing: "0.15em", textTransform: "uppercase", color: "#2a2a2a", marginBottom: 8 }}>
            Recent Logs
          </div>
          {logs.slice(0, 5).map(l => (
            <div key={l.id} style={{ marginBottom: 8, paddingBottom: 8, borderBottom: "1px solid #111" }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                <StatusDot status={l.state} />
                <span style={{ fontSize: 9, color: "#2a2a2a" }}>
                  {new Date(l.time).toLocaleDateString()}
                </span>
              </div>
              <div style={{ fontSize: 11, color: "#555", lineHeight: 1.5 }}>{l.text}</div>
              {l.nextStep && (
                <div style={{ fontSize: 9, color: "#333", marginTop: 3 }}>→ {l.nextStep}</div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── ARTIFACT DETAIL ─────────────────────────────────────────────────────────

function ArtifactDetail({ artifact, color, onClose, onEdit, onStatusUpdate }: {
  artifact: Artifact;
  color: string;
  onClose: () => void;
  onEdit: (a: Artifact) => void;
  onStatusUpdate: (status: string, nextStep: string) => void;
}) {
  return (
    <div style={{
      background: "#0f0f0f", border: `1px solid ${color}33`,
      borderLeft: `3px solid ${color}`, borderRadius: 4,
      padding: 28, position: "sticky", top: 20,
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16 }}>
        <div>
          <div style={{ fontSize: 11, color, letterSpacing: "0.15em", textTransform: "uppercase", marginBottom: 6 }}>Artifact Detail</div>
          <div style={{ fontSize: 18, fontFamily: "'Syne', sans-serif", fontWeight: 700, color: "#f0ece4" }}>{artifact.title}</div>
          <div style={{ fontSize: 11, color: "#555", marginTop: 4 }}>{artifact.subtitle}</div>
        </div>
        <div style={{ display: "flex", gap: 4 }}>
          <IconBtn onClick={() => onEdit(artifact)} title="Edit">✎</IconBtn>
          <button onClick={onClose} style={{ background: "none", border: "none", color: "#444", cursor: "pointer", fontSize: 18 }}>×</button>
        </div>
      </div>

      <StatusDot status={artifact.status} />

      {artifact.progress.length > 0 && (
        <div style={{ marginTop: 20 }}>
          <div style={{ fontSize: 9, letterSpacing: "0.15em", textTransform: "uppercase", color: "#333", marginBottom: 10 }}>Progress</div>
          {artifact.progress.map(p => <ProgressBar key={p.label} label={p.label} val={p.val} color={color} />)}
        </div>
      )}

      {artifact.stack.length > 0 && (
        <div style={{ marginTop: 16 }}>
          <div style={{ fontSize: 9, letterSpacing: "0.15em", textTransform: "uppercase", color: "#333", marginBottom: 8 }}>Stack</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
            {artifact.stack.map(t => <Tag key={t} label={t} />)}
          </div>
        </div>
      )}

      {artifact.notes && (
        <div style={{ marginTop: 16, padding: 14, background: "#141414", borderRadius: 3, borderLeft: `2px solid ${color}44` }}>
          <div style={{ fontSize: 9, letterSpacing: "0.15em", textTransform: "uppercase", color: "#333", marginBottom: 6 }}>Notes / Context</div>
          <div style={{ fontSize: 11, color: "#666", lineHeight: 1.7 }}>{artifact.notes}</div>
        </div>
      )}

      {artifact.next && (
        <div style={{ marginTop: 14, padding: 14, background: `${color}08`, borderRadius: 3 }}>
          <div style={{ fontSize: 9, letterSpacing: "0.15em", textTransform: "uppercase", color: color + "88", marginBottom: 6 }}>Next Step</div>
          <div style={{ fontSize: 12, color: "#999", lineHeight: 1.6 }}>{artifact.next}</div>
        </div>
      )}

      <LogPanel artifact={artifact} color={color} onStatusUpdate={onStatusUpdate} />
    </div>
  );
}

// ─── ENGINE VIEW ─────────────────────────────────────────────────────────────

function EngineView({ engine, onUpdateEngine, onDeleteEngine }: {
  engine: Engine;
  onUpdateEngine: (e: Engine) => void;
  onDeleteEngine: (id: string) => void;
}) {
  const [selectedArtifact, setSelectedArtifact] = useState<Artifact | null>(null);
  const [showContext, setShowContext] = useState(false);
  const [editingEngine, setEditingEngine] = useState(false);
  const [editingArtifact, setEditingArtifact] = useState<Artifact | null>(null);

  const saveArtifact = async (updated: Artifact) => {
    let synced = updated;
    // Sync title + status to backend node (fire-and-forget; localStorage is source of truth)
    if (updated.title) {
      try {
        if (!updated.node_id) {
          const node = await api.createNode(updated.title, false, updated.status);
          synced = { ...updated, node_id: node.id };
        } else {
          await api.updateNode(updated.node_id, { title: updated.title, status: updated.status });
        }
      } catch {
        // API offline — continue with localStorage only
      }
    }
    const artifacts = engine.artifacts.find(a => a.id === synced.id)
      ? engine.artifacts.map(a => a.id === synced.id ? synced : a)
      : [...engine.artifacts, synced];
    onUpdateEngine({ ...engine, artifacts });
    setEditingArtifact(null);
    if (selectedArtifact?.id === synced.id) setSelectedArtifact(synced);
  };

  const deleteArtifact = (id: string) => {
    const artifact = engine.artifacts.find(a => a.id === id);
    if (artifact?.node_id) {
      api.deleteNode(artifact.node_id).catch(() => {});
    }
    onUpdateEngine({ ...engine, artifacts: engine.artifacts.filter(a => a.id !== id) });
    setEditingArtifact(null);
    if (selectedArtifact?.id === id) setSelectedArtifact(null);
  };

  const handleStatusUpdate = (artifactId: string, status: string, next: string) => {
    const updated = engine.artifacts.map(a =>
      a.id === artifactId ? { ...a, status, next } : a
    );
    onUpdateEngine({ ...engine, artifacts: updated });
    setSelectedArtifact(prev => prev?.id === artifactId ? { ...prev, status, next } : prev);
  };

  return (
    <div style={{ padding: "32px 0" }}>
      {showContext && <ContextPage engine={engine} onClose={() => setShowContext(false)} />}
      {editingEngine && (
        <EngineEditor
          engine={engine}
          onSave={updated => { onUpdateEngine(updated); setEditingEngine(false); }}
          onClose={() => setEditingEngine(false)}
          onDelete={() => { onDeleteEngine(engine.id); setEditingEngine(false); }}
          isNew={false}
        />
      )}
      {editingArtifact && (
        <ArtifactEditor
          artifact={editingArtifact}
          color={engine.color}
          onSave={saveArtifact}
          onClose={() => setEditingArtifact(null)}
          onDelete={() => deleteArtifact(editingArtifact.id)}
        />
      )}

      {/* Engine header */}
      <div style={{
        display: "flex", justifyContent: "space-between", alignItems: "flex-end",
        paddingBottom: 20, borderBottom: "1px solid #1a1a1a", marginBottom: 28,
      }}>
        <div>
          <div style={{ fontSize: 28, fontFamily: "'Syne', sans-serif", fontWeight: 800, color: "#f0ece4", marginBottom: 4 }}>
            <span style={{ color: engine.color, marginRight: 10 }}>{engine.icon}</span>
            {engine.name}
          </div>
          <div style={{ fontSize: 12, color: "#555" }}>{engine.description || engine.subtitle}</div>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button onClick={() => setShowContext(true)} style={{
            background: `${engine.color}10`, border: `1px solid ${engine.color}44`,
            color: engine.color, borderRadius: 3, padding: "7px 13px",
            cursor: "pointer", fontSize: 10, letterSpacing: "0.15em", textTransform: "uppercase",
          }}>🔒 Context</button>
          <button onClick={() => setEditingEngine(true)} style={{
            background: "#111", border: "1px solid #1e1e1e",
            color: "#444", borderRadius: 3, padding: "7px 13px",
            cursor: "pointer", fontSize: 10, letterSpacing: "0.15em", textTransform: "uppercase",
          }}>✎ Edit</button>
        </div>
      </div>

      {/* Artifacts */}
      <div style={{ display: "grid", gridTemplateColumns: selectedArtifact ? "1fr 340px" : "1fr", gap: 16 }}>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 14, alignContent: "start" }}>
          {engine.artifacts.map(a => (
            <ArtifactCard
              key={a.id}
              artifact={a}
              color={engine.color}
              onSelect={art => setSelectedArtifact(selectedArtifact?.id === art.id ? null : art)}
              selected={selectedArtifact?.id === a.id}
              onEdit={art => setEditingArtifact(art)}
            />
          ))}

          {/* Add artifact button */}
          <button
            onClick={() => setEditingArtifact(EMPTY_ARTIFACT())}
            style={{
              background: "none", border: `1px dashed #222`, borderRadius: 4,
              padding: 20, cursor: "pointer", color: "#333",
              fontSize: 11, letterSpacing: "0.1em", textTransform: "uppercase",
              display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
              minHeight: 80, transition: "all 0.2s",
            }}
            onMouseEnter={e => { e.currentTarget.style.borderColor = engine.color + "66"; e.currentTarget.style.color = engine.color; }}
            onMouseLeave={e => { e.currentTarget.style.borderColor = "#222"; e.currentTarget.style.color = "#333"; }}
          >
            + Add Artifact
          </button>
        </div>

        {selectedArtifact && (
          <ArtifactDetail
            artifact={selectedArtifact}
            color={engine.color}
            onClose={() => setSelectedArtifact(null)}
            onEdit={art => setEditingArtifact(art)}
            onStatusUpdate={(status, next) => handleStatusUpdate(selectedArtifact.id, status, next)}
          />
        )}
      </div>
    </div>
  );
}

// ─── THEME SWITCHER ───────────────────────────────────────────────────────────

const THEME_LABELS: { name: ThemeName; label: string }[] = [
  { name: "tsushima",     label: "Tsushima" },
  { name: "transylvania", label: "Transylvania" },
  { name: "frieren",      label: "Frieren" },
  { name: "lofi",         label: "Lofi" },
];

function ThemeSwitcher() {
  const { themeName, setTheme } = useTheme();
  return (
    <div className="theme-switcher">
      {THEME_LABELS.map(t => (
        <button
          key={t.name}
          className={`theme-btn${themeName === t.name ? " active" : ""}`}
          onClick={() => setTheme(t.name)}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}

// ─── MAIN APP (inner) ─────────────────────────────────────────────────────────

function MindCastleInner() {
  const { theme } = useTheme();

  const [engines, setEngines] = useState<Engine[]>(() => load());
  const [activeEngineId, setActiveEngineId] = useState<string | null>(() => {
    const saved = load();
    return saved[0]?.id ?? null;
  });
  const [addingEngine, setAddingEngine] = useState(false);
  const prevEngineCount = useRef(engines.length);

  useEffect(() => { save(engines); }, [engines]);

  useEffect(() => {
    if (engines.length > prevEngineCount.current) {
      setActiveEngineId(engines[engines.length - 1].id);
    }
    if (engines.length === 0) setActiveEngineId(null);
    prevEngineCount.current = engines.length;
  }, [engines]);

  const activeEngine = engines.find(e => e.id === activeEngineId) ?? null;

  const updateEngine = (updated: Engine) => setEngines(es => es.map(e => e.id === updated.id ? updated : e));
  const deleteEngine = (id: string) => setEngines(es => es.filter(e => e.id !== id));
  const addEngine = (e: Engine) => {
    if (!e.name.trim()) return;
    setEngines(es => [...es, e]);
    setAddingEngine(false);
  };

  const totalArtifacts = engines.reduce((n, e) => n + e.artifacts.length, 0);

  return (
    <div style={{
      display: "flex", flexDirection: "column", height: "100dvh",
      background: "var(--bg-app)", color: "var(--text-primary)",
      fontFamily: "'DM Mono', monospace", overflow: "hidden",
    }}>
      {addingEngine && (
        <EngineEditor
          engine={EMPTY_ENGINE()}
          onSave={addEngine}
          onClose={() => setAddingEngine(false)}
          onDelete={() => setAddingEngine(false)}
          isNew
        />
      )}

      {/* Top nav */}
      <nav style={{
        display: "flex", alignItems: "center", flexShrink: 0,
        borderBottom: `1px solid ${theme.borderMuted}`,
        padding: "0 clamp(16px, 3vw, 32px)",
        background: "var(--bg-nav)",
        position: "sticky", top: 0, zIndex: 50,
        overflowX: "auto",
      }}>
        <div style={{
          fontSize: 11, fontFamily: "'Syne', sans-serif", fontWeight: 700,
          letterSpacing: "0.2em", color: "var(--text-ghost)", textTransform: "uppercase",
          padding: "18px 0", marginRight: 32, whiteSpace: "nowrap", flexShrink: 0,
        }}>
          MIND · CASTLE
        </div>

        {engines.map(e => (
          <button key={e.id} onClick={() => setActiveEngineId(e.id)} style={{
            background: "none", border: "none",
            borderBottom: activeEngineId === e.id ? `2px solid ${e.color}` : "2px solid transparent",
            color: activeEngineId === e.id ? e.color : "var(--text-muted)",
            padding: "18px 18px 16px", cursor: "pointer",
            fontSize: 11, fontFamily: "'Syne', sans-serif", fontWeight: 700,
            letterSpacing: "0.1em", textTransform: "uppercase",
            display: "flex", alignItems: "center", gap: 8,
            transition: `all var(--transition)`, whiteSpace: "nowrap",
          }}>
            <span style={{ opacity: 0.7 }}>{e.icon}</span>
            {e.name}
            <span style={{
              fontSize: 9,
              background: activeEngineId === e.id ? e.color + "22" : theme.bgSubtle,
              padding: "1px 6px", borderRadius: 10,
              color: activeEngineId === e.id ? e.color : "var(--text-ghost)",
            }}>
              {e.artifacts.length}
            </span>
          </button>
        ))}

        <button onClick={() => setAddingEngine(true)} style={{
          background: "none", border: "none", borderBottom: "2px solid transparent",
          color: "var(--text-ghost)", padding: "18px 14px 16px", cursor: "pointer",
          fontSize: 11, letterSpacing: "0.1em", textTransform: "uppercase",
          whiteSpace: "nowrap", flexShrink: 0, transition: `color var(--transition)`,
        }}
          onMouseEnter={e => (e.currentTarget.style.color = "var(--text-secondary)")}
          onMouseLeave={e => (e.currentTarget.style.color = "var(--text-ghost)")}
        >
          + Engine
        </button>

        <ThemeSwitcher />
      </nav>

      {/* Content — CastleViewport handles centering + scroll */}
      <CastleViewport>
        {activeEngine
          ? <EngineView
              key={activeEngine.id}
              engine={activeEngine}
              onUpdateEngine={updateEngine}
              onDeleteEngine={deleteEngine}
            />
          : <Onboarding
              onComplete={(imported) => setEngines(imported)}
              onSkip={() => setAddingEngine(true)}
            />
        }
      </CastleViewport>

      {/* Footer */}
      <footer style={{
        borderTop: `1px solid ${theme.borderMuted}`,
        padding: "12px clamp(16px, 3vw, 32px)",
        display: "flex", justifyContent: "space-between", alignItems: "center",
        flexShrink: 0,
      }}>
        <span style={{ fontSize: 10, color: "var(--text-ghost)", letterSpacing: "0.1em" }}>MIND CASTLE</span>
        <span style={{ fontSize: 10, color: "var(--text-ghost)" }}>
          {engines.length} engine{engines.length !== 1 ? "s" : ""} · {totalArtifacts} artifact{totalArtifacts !== 1 ? "s" : ""}
        </span>
      </footer>
    </div>
  );
}

// ─── MAIN EXPORT ──────────────────────────────────────────────────────────────

export default function MindCastle() {
  return (
    <ThemeProvider>
      <MindCastleInner />
    </ThemeProvider>
  );
}
