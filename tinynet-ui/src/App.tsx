import { useState } from "react";

// ─── DATA ────────────────────────────────────────────────────────────────────

const ROHIT_CORE_CONTEXT = `You are assisting Rohit — a multidisciplinary builder, writer, and life architect operating across tech, philosophy, and personal systems. He is building TML (a credibility-based art trade platform), a personal cognitive OS (Mind Castle), a philosophy book about individualism and art as resistance, and an Azure provisioning tool for work. He's simultaneously running a fitness transformation (PPL, targeting lean physique by May), high-protein meal prep, and planning a Hawaii hiking trip in May 2026. His three creative engines — Tech Builder, Philosopher, and Life Architect — are converging into one unified architecture. He values systems thinking, aesthetic integrity, and building things that actually matter. Always connect advice back to the larger architecture. Never give isolated answers — Rohit thinks in systems.`;

const PROJECTS = [
  {
    id: "rohit-os",
    name: "Rohit OS",
    subtitle: "Master Context Layer",
    engine: "meta",
    color: "#c8a96e",
    colorDim: "rgba(200,169,110,0.08)",
    icon: "⬡",
    context: ROHIT_CORE_CONTEXT,
    contextLabel: "Master Identity Context",
    description: "The private layer that makes every other project smarter. Injected silently into all conversations.",
    artifacts: [
      {
        id: "identity",
        title: "Identity Brief",
        subtitle: "Who Rohit is, what he's building, how he thinks",
        status: "live",
        next: "Update after each major life shift or project milestone",
        stack: null,
        progress: null,
        notes: "3 engines: Tech Builder · Philosopher · Life Architect. All converging.",
        engine: "meta"
      },
      {
        id: "convergence",
        title: "Convergence Map",
        subtitle: "How TML + Book + Mind Castle + Body form one system",
        status: "live",
        next: "Revisit when a new project is added or one completes",
        stack: null,
        progress: [{ label: "Mapped", val: 85 }],
        notes: "TML is the market. Book is the manifesto. Mind Castle is the OS. Body is the vessel.",
        engine: "meta"
      }
    ]
  },
  {
    id: "tech-builder",
    name: "Tech Builder",
    subtitle: "Engineering · Platforms · AI",
    engine: "tech",
    color: "#3ecfcf",
    colorDim: "rgba(62,207,207,0.08)",
    icon: "◈",
    context: `Rohit is a software engineer building TML (React/Vite/Expo + Node + PostgreSQL + Neo4j), an Azure user provisioning platform (C# Function App + Angular), and an AI Social Media Visa Compliance CLI tool (Python + NLP). He's also on an 8hr/week AI learning roadmap: Math → ML algorithms → LLM architecture → AI Agents. He's investigating agentic coding tools. His infra uses Vercel + Railway + GCP. Railway service denial is a risk to demo stability. DaVinci Resolve automation is blocked pending scripting API enablement. When helping with tech, always consider how it connects to TML's long-term vision.`,
    contextLabel: "Tech Builder Context",
    description: "TML, Azure provisioning, AI tooling, infra, DaVinci automation.",
    artifacts: [
      {
        id: "tml",
        title: "TML — Trade Mode Life",
        subtitle: "Digital art exchange platform with credibility network",
        status: "active",
        next: "Finalize Neo4j graph interaction model for Aura/Credibility social layer",
        stack: ["React", "Vite", "Expo", "Node", "PostgreSQL", "Neo4j", "Vercel", "Railway", "GCP"],
        progress: [
          { label: "Architecture", val: 100 },
          { label: "Graph DB / Social Credibility", val: 35 },
          { label: "Demo Infra Stability", val: 60 }
        ],
        notes: "Roles: Artist · Collector · Seller · Curator · Admin. Railway service denial risk before demo.",
        engine: "tech"
      },
      {
        id: "azure",
        title: "Azure Provisioning Platform",
        subtitle: "Microsoft Entra ID user management — internal work tool",
        status: "active",
        next: "Multi-env rollout: QA → UAT → PROD",
        stack: ["C#", "Azure Function App", "Azure SQL", "Web API", "Angular"],
        progress: [
          { label: "Core Architecture", val: 80 },
          { label: "Group Membership UI", val: 65 },
          { label: "Multi-env Deploy", val: 40 }
        ],
        notes: "Hourly polling via Function App. Onboarding + group assignment features in scope.",
        engine: "tech"
      },
      {
        id: "ai-learning",
        title: "AI Learning Roadmap",
        subtitle: "8 hrs/week: Math → ML → LLM → Agents",
        status: "active",
        next: "Complete core ML algorithms before pivoting to LLM architecture",
        stack: ["Python", "Math", "PyTorch", "Transformers"],
        progress: [
          { label: "Math Foundations", val: 90 },
          { label: "Core ML Algorithms", val: 55 },
          { label: "LLM Architecture", val: 20 },
          { label: "AI Agents", val: 5 }
        ],
        notes: "Constraint: 8hrs/week. Exploring agentic coding tools in parallel.",
        engine: "tech"
      },
      {
        id: "visa-tool",
        title: "Visa Compliance CLI Tool",
        subtitle: "Local social media scanner for immigration safety",
        status: "concept",
        next: "Scaffold scraper + sentiment classifier. Local-only first pass.",
        stack: ["Python", "NLP", "CLI"],
        progress: [{ label: "Concept Definition", val: 90 }, { label: "Implementation", val: 0 }],
        notes: "Features: scrape posts · classify risk · compliance report. Privacy-first, local run.",
        engine: "tech"
      },
      {
        id: "davinci",
        title: "DaVinci Resolve Automation",
        subtitle: "Python video editing workflow automation",
        status: "blocked",
        next: "Open Resolve → Preferences → Enable scripting API → rerun Video.py from Fusion console",
        stack: ["Python", "DaVinci Resolve API"],
        progress: [{ label: "Script Written", val: 100 }, { label: "Connected to Resolve", val: 0 }],
        notes: "Error: Could not connect to DaVinci Resolve. Script at Fusion/Scripts/Utility/Video.py",
        engine: "tech"
      }
    ]
  },
  {
    id: "philosopher",
    name: "Philosopher",
    subtitle: "Writing · Identity · Theory",
    engine: "write",
    color: "#9b7fe8",
    colorDim: "rgba(155,127,232,0.08)",
    icon: "✦",
    context: `Rohit is writing a philosophy book titled "What's WRONG with the WoRLd? Why YOU're Right. How ART Is the Fix." He's written multiple essays including "Man of Two Cities" (NYC vs Mumbai identity), pieces on validation and belonging, and reflections on societal systems. Themes: hyper-individualism, identity, societal validation, art as resistance. Goal: combine essays into a book + Substack + synergy with TML platform. When assisting, treat the writing as both a standalone intellectual project AND a manifesto for TML. Style: bold, personal, culturally sharp. Avoid academic detachment.`,
    contextLabel: "Philosopher Context",
    description: "Book, essays, Substack, identity theory, art as resistance.",
    artifacts: [
      {
        id: "book",
        title: "The Book",
        subtitle: '"What\'s WRONG with the WoRLd? Why YOU\'re Right. How ART Is the Fix."',
        status: "active",
        next: "Compile existing essays into chapter structure. Define arc.",
        stack: null,
        progress: [
          { label: "Essays Written", val: 60 },
          { label: "Chapter Architecture", val: 20 },
          { label: "Book Draft", val: 10 }
        ],
        notes: "Man of Two Cities · Validation & Belonging · Societal Systems. Substack + TML synergy planned.",
        engine: "write"
      },
      {
        id: "substack",
        title: "Substack / Essays",
        subtitle: "Public-facing philosophy engine feeding book and platform",
        status: "planned",
        next: "Publish Man of Two Cities as launch essay. Build audience before book.",
        stack: null,
        progress: [{ label: "Draft essays", val: 60 }, { label: "Published", val: 0 }],
        notes: "Substack is the funnel: audience → book → TML community.",
        engine: "write"
      }
    ]
  },
  {
    id: "life-architect",
    name: "Life Architect",
    subtitle: "Systems · Body · Rhythm",
    engine: "life",
    color: "#c8a96e",
    colorDim: "rgba(200,169,110,0.08)",
    icon: "◎",
    context: `Rohit is building a personal life operating system. Mind Castle is his cognitive architecture app: rooms (life domains), resources, architect points, gamified productivity — built in React/Vite + FastAPI + SQLite/Postgres with local LLM support. His daily rhythm vision includes early rising, morning workouts, focused work blocks, writing/reading, and financial discipline. Fitness: PPL split (Push/Pull/Legs) with abs classes Mon+Wed. Goal: lean aesthetic "ILYA body" by May 2026. Meal prep: high protein, batch cooking — chicken tinga, beef chili, lime soy tofu, egg burritos, protein shakes, yogurt. Hawaii trip May 16–26 (Maui + Volcano National Park). When helping with life systems, always connect to Mind Castle as the container.`,
    contextLabel: "Life Architect Context",
    description: "Mind Castle, fitness, meal prep, daily rhythm, travel.",
    artifacts: [
      {
        id: "mind-castle",
        title: "Mind Castle",
        subtitle: "Personal cognitive OS — rooms, resources, architect points",
        status: "active",
        next: "Build core room architecture + daily rhythm integration module",
        stack: ["React", "Vite", "FastAPI", "SQLite", "Postgres", "Local LLM"],
        progress: [
          { label: "Concept Architecture", val: 85 },
          { label: "Core Rooms UI", val: 30 },
          { label: "Gamification Layer", val: 15 },
          { label: "LLM Integration", val: 5 }
        ],
        notes: "Domains: tasks · finances · health · habits · creative work. Gamified with architect points.",
        engine: "life"
      },
      {
        id: "fitness",
        title: "ILYA Body",
        subtitle: "Lean aesthetic physique — broader shoulders, smaller waist — by May",
        status: "active",
        next: "Track progressive overload weekly. Log protein daily.",
        stack: null,
        progress: [{ label: "Program Adherence", val: 70 }, { label: "Target Physique", val: 35 }],
        notes: "PPL split. Abs Mon+Wed. Target: lower BF%, shoulder width, waist illusion. May 2026 deadline.",
        engine: "life"
      },
      {
        id: "meal-prep",
        title: "Meal Prep OS",
        subtitle: "High protein · batch cook · efficient storage",
        status: "active",
        next: "Add new rotation recipe. Track weekly macros.",
        stack: null,
        progress: [{ label: "System Consistency", val: 75 }],
        notes: "Rotation: chicken tinga · beef chili · lime soy tofu · egg burritos · protein shakes · yogurt.",
        engine: "life"
      },
      {
        id: "hawaii",
        title: "Hawaii Trip",
        subtitle: "Maui + Volcano National Park · May 16–26",
        status: "planning",
        next: "Build day-by-day hiking itinerary. Book accommodation.",
        stack: null,
        progress: [{ label: "Flights / Dates", val: 80 }, { label: "Itinerary", val: 20 }],
        notes: "10 days. Focus: hiking, nature reset. Maui + Big Island Volcano NP.",
        engine: "life"
      }
    ]
  }
];

// ─── HELPERS ─────────────────────────────────────────────────────────────────

const STATUS_META: Record<string, { dot: string; label: string }> = {
  active:   { dot: "#a3d977", label: "Active" },
  live:     { dot: "#a3d977", label: "Live" },
  concept:  { dot: "#666",    label: "Concept" },
  blocked:  { dot: "#e0637a", label: "Blocked" },
  planned:  { dot: "#9b7fe8", label: "Planned" },
  planning: { dot: "#c8a96e", label: "Planning" },
};

// ─── TYPES ───────────────────────────────────────────────────────────────────

interface ProgressItem {
  label: string;
  val: number;
}

interface Artifact {
  id: string;
  title: string;
  subtitle: string;
  status: string;
  next: string;
  stack: string[] | null;
  progress: ProgressItem[] | null;
  notes: string;
  engine: string;
}

interface Project {
  id: string;
  name: string;
  subtitle: string;
  engine: string;
  color: string;
  colorDim: string;
  icon: string;
  context: string;
  contextLabel: string;
  description: string;
  artifacts: Artifact[];
}

// ─── COMPONENTS ──────────────────────────────────────────────────────────────

function StatusDot({ status }: { status: string }) {
  const m = STATUS_META[status] || STATUS_META.concept;
  return (
    <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <span style={{
        width: 7, height: 7, borderRadius: "50%", background: m.dot, flexShrink: 0,
        boxShadow: (status === "active" || status === "live") ? `0 0 6px ${m.dot}` : "none"
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
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#555", marginBottom: 4, letterSpacing: "0.04em" }}>
        <span>{label}</span><span>{val}%</span>
      </div>
      <div style={{ height: 2, background: "#1e1e1e", borderRadius: 2 }}>
        <div style={{ height: "100%", width: `${val}%`, background: color, borderRadius: 2, transition: "width 1s ease" }} />
      </div>
    </div>
  );
}

function ArtifactCard({ artifact, color, onSelect, selected }: {
  artifact: Artifact;
  color: string;
  onSelect: (a: Artifact) => void;
  selected: boolean;
}) {
  return (
    <div
      onClick={() => onSelect(artifact)}
      style={{
        background: selected ? "rgba(255,255,255,0.04)" : "#111",
        border: `1px solid ${selected ? color : "#1e1e1e"}`,
        borderTop: `2px solid ${color}`,
        borderRadius: 4,
        padding: 20,
        cursor: "pointer",
        transition: "all 0.2s",
        position: "relative",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 8 }}>
        <div style={{ fontSize: 13, fontFamily: "'Syne', sans-serif", fontWeight: 700, color: "#f0ece4", lineHeight: 1.2, maxWidth: "75%" }}>
          {artifact.title}
        </div>
        <StatusDot status={artifact.status} />
      </div>
      <div style={{ fontSize: 11, color: "#555", marginBottom: 14, lineHeight: 1.5 }}>{artifact.subtitle}</div>

      {artifact.progress && artifact.progress.slice(0, 2).map(p => (
        <ProgressBar key={p.label} label={p.label} val={p.val} color={color} />
      ))}

      {artifact.stack && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 10 }}>
          {artifact.stack.slice(0, 5).map(t => (
            <span key={t} style={{ fontSize: 9, padding: "2px 7px", background: "#1a1a1a", border: "1px solid #222", borderRadius: 2, color: "#555", letterSpacing: "0.08em", textTransform: "uppercase" }}>
              {t}
            </span>
          ))}
          {artifact.stack.length > 5 && <span style={{ fontSize: 9, color: "#444" }}>+{artifact.stack.length - 5}</span>}
        </div>
      )}

      <div style={{ borderTop: "1px solid #1e1e1e", paddingTop: 10, marginTop: 12 }}>
        <div style={{ fontSize: 9, letterSpacing: "0.15em", textTransform: "uppercase", color: "#333", marginBottom: 4 }}>Next</div>
        <div style={{ fontSize: 11, color: "#555", lineHeight: 1.5 }}>{artifact.next}</div>
      </div>
    </div>
  );
}

function ArtifactDetail({ artifact, color, onClose }: {
  artifact: Artifact;
  color: string;
  onClose: () => void;
}) {
  return (
    <div style={{
      background: "#0f0f0f",
      border: `1px solid ${color}33`,
      borderLeft: `3px solid ${color}`,
      borderRadius: 4,
      padding: 28,
      position: "sticky",
      top: 20,
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16 }}>
        <div>
          <div style={{ fontSize: 11, color: color, letterSpacing: "0.15em", textTransform: "uppercase", marginBottom: 6 }}>Artifact Detail</div>
          <div style={{ fontSize: 18, fontFamily: "'Syne', sans-serif", fontWeight: 700, color: "#f0ece4" }}>{artifact.title}</div>
          <div style={{ fontSize: 11, color: "#555", marginTop: 4 }}>{artifact.subtitle}</div>
        </div>
        <button onClick={onClose} style={{ background: "none", border: "none", color: "#444", cursor: "pointer", fontSize: 18 }}>×</button>
      </div>

      <StatusDot status={artifact.status} />

      {artifact.progress && (
        <div style={{ marginTop: 20 }}>
          <div style={{ fontSize: 9, letterSpacing: "0.15em", textTransform: "uppercase", color: "#333", marginBottom: 10 }}>Progress</div>
          {artifact.progress.map(p => <ProgressBar key={p.label} label={p.label} val={p.val} color={color} />)}
        </div>
      )}

      {artifact.stack && (
        <div style={{ marginTop: 16 }}>
          <div style={{ fontSize: 9, letterSpacing: "0.15em", textTransform: "uppercase", color: "#333", marginBottom: 8 }}>Stack</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 5 }}>
            {artifact.stack.map(t => (
              <span key={t} style={{ fontSize: 10, padding: "3px 9px", background: "#1a1a1a", border: "1px solid #252525", borderRadius: 2, color: "#666", letterSpacing: "0.08em" }}>
                {t}
              </span>
            ))}
          </div>
        </div>
      )}

      <div style={{ marginTop: 16, padding: 14, background: "#141414", borderRadius: 3, borderLeft: `2px solid ${color}44` }}>
        <div style={{ fontSize: 9, letterSpacing: "0.15em", textTransform: "uppercase", color: "#333", marginBottom: 6 }}>Notes / Context</div>
        <div style={{ fontSize: 11, color: "#666", lineHeight: 1.7 }}>{artifact.notes}</div>
      </div>

      <div style={{ marginTop: 14, padding: 14, background: `${color}08`, borderRadius: 3 }}>
        <div style={{ fontSize: 9, letterSpacing: "0.15em", textTransform: "uppercase", color: color + "88", marginBottom: 6 }}>Next Step</div>
        <div style={{ fontSize: 12, color: "#999", lineHeight: 1.6 }}>{artifact.next}</div>
      </div>
    </div>
  );
}

function ContextPage({ project, onClose }: { project: Project; onClose: () => void }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard.writeText(project.context);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  return (
    <div style={{
      position: "fixed", inset: 0, background: "rgba(0,0,0,0.85)", zIndex: 100,
      display: "flex", alignItems: "center", justifyContent: "center", padding: 24
    }} onClick={onClose}>
      <div style={{
        background: "#0d0d0d", border: `1px solid ${project.color}44`,
        borderTop: `3px solid ${project.color}`,
        borderRadius: 6, padding: 36, maxWidth: 680, width: "100%",
        maxHeight: "80vh", overflow: "auto"
      }} onClick={e => e.stopPropagation()}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
          <div>
            <div style={{ fontSize: 10, color: project.color, letterSpacing: "0.2em", textTransform: "uppercase", marginBottom: 6 }}>
              🔒 Private Context Page
            </div>
            <div style={{ fontSize: 20, fontFamily: "'Syne', sans-serif", fontWeight: 700, color: "#f0ece4" }}>
              {project.contextLabel}
            </div>
          </div>
          <button onClick={onClose} style={{ background: "none", border: "none", color: "#555", cursor: "pointer", fontSize: 22 }}>×</button>
        </div>

        <div style={{ fontSize: 11, color: "#444", marginBottom: 16, fontStyle: "italic", lineHeight: 1.6 }}>
          This context is injected silently when chatting within this project. Claude reads this before every response.
        </div>

        <div style={{
          background: "#111", border: "1px solid #1e1e1e", borderRadius: 4,
          padding: 20, fontFamily: "'DM Mono', monospace", fontSize: 11,
          color: "#888", lineHeight: 1.8, whiteSpace: "pre-wrap"
        }}>
          {project.context}
        </div>

        <div style={{ display: "flex", gap: 10, marginTop: 16 }}>
          <button onClick={copy} style={{
            background: copied ? `${project.color}22` : "#1a1a1a",
            border: `1px solid ${copied ? project.color : "#2a2a2a"}`,
            color: copied ? project.color : "#666",
            borderRadius: 3, padding: "8px 16px", cursor: "pointer",
            fontSize: 11, letterSpacing: "0.1em", transition: "all 0.2s"
          }}>
            {copied ? "✓ Copied" : "Copy to Claude Project Instructions"}
          </button>
          <button onClick={onClose} style={{
            background: "none", border: "1px solid #1e1e1e",
            color: "#444", borderRadius: 3, padding: "8px 16px",
            cursor: "pointer", fontSize: 11
          }}>Close</button>
        </div>
      </div>
    </div>
  );
}

function ProjectView({ project }: { project: Project }) {
  const [selectedArtifact, setSelectedArtifact] = useState<Artifact | null>(null);
  const [showContext, setShowContext] = useState(false);

  return (
    <div style={{ padding: "32px 0" }}>
      {showContext && <ContextPage project={project} onClose={() => setShowContext(false)} />}

      {/* Project header */}
      <div style={{
        display: "flex", justifyContent: "space-between", alignItems: "flex-end",
        paddingBottom: 20, borderBottom: "1px solid #1a1a1a", marginBottom: 28
      }}>
        <div>
          <div style={{ fontSize: 28, fontFamily: "'Syne', sans-serif", fontWeight: 800, color: "#f0ece4", marginBottom: 4 }}>
            <span style={{ color: project.color, marginRight: 10 }}>{project.icon}</span>
            {project.name}
          </div>
          <div style={{ fontSize: 12, color: "#555" }}>{project.description}</div>
        </div>
        <button
          onClick={() => setShowContext(true)}
          style={{
            background: project.colorDim,
            border: `1px solid ${project.color}44`,
            color: project.color,
            borderRadius: 3, padding: "8px 14px",
            cursor: "pointer", fontSize: 10,
            letterSpacing: "0.15em", textTransform: "uppercase",
            display: "flex", alignItems: "center", gap: 6
          }}>
          🔒 Context Page
        </button>
      </div>

      {/* Artifacts grid + detail */}
      <div style={{ display: "grid", gridTemplateColumns: selectedArtifact ? "1fr 340px" : "1fr", gap: 16 }}>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: 14, alignContent: "start" }}>
          {project.artifacts.map(a => (
            <ArtifactCard
              key={a.id}
              artifact={a}
              color={project.color}
              onSelect={art => setSelectedArtifact(selectedArtifact?.id === art.id ? null : art)}
              selected={selectedArtifact?.id === a.id}
            />
          ))}
        </div>
        {selectedArtifact && (
          <ArtifactDetail
            artifact={selectedArtifact}
            color={project.color}
            onClose={() => setSelectedArtifact(null)}
          />
        )}
      </div>
    </div>
  );
}

// ─── MAIN APP ─────────────────────────────────────────────────────────────────

export default function RohitOS() {
  const [activeProject, setActiveProject] = useState("rohit-os");
  const project = PROJECTS.find(p => p.id === activeProject)!;

  return (
    <div style={{
      background: "#080808", minHeight: "100vh", color: "#f0ece4",
      fontFamily: "'DM Mono', monospace",
    }}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400&display=swap'); * { box-sizing: border-box; margin: 0; padding: 0; } ::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-track { background: #0a0a0a; } ::-webkit-scrollbar-thumb { background: #222; border-radius: 2px; }`}</style>

      {/* Top nav */}
      <div style={{
        display: "flex", alignItems: "center", gap: 0,
        borderBottom: "1px solid #151515", padding: "0 32px",
        background: "#080808", position: "sticky", top: 0, zIndex: 50
      }}>
        <div style={{ fontSize: 11, fontFamily: "'Syne', sans-serif", fontWeight: 700, letterSpacing: "0.2em", color: "#333", textTransform: "uppercase", padding: "18px 0", marginRight: 32 }}>
          ROHIT · OS
        </div>
        {PROJECTS.map(p => (
          <button
            key={p.id}
            onClick={() => setActiveProject(p.id)}
            style={{
              background: "none",
              border: "none",
              borderBottom: activeProject === p.id ? `2px solid ${p.color}` : "2px solid transparent",
              color: activeProject === p.id ? p.color : "#444",
              padding: "18px 18px 16px",
              cursor: "pointer",
              fontSize: 11,
              fontFamily: "'Syne', sans-serif",
              fontWeight: 700,
              letterSpacing: "0.1em",
              textTransform: "uppercase",
              display: "flex", alignItems: "center", gap: 8,
              transition: "all 0.2s",
              whiteSpace: "nowrap"
            }}
          >
            <span style={{ opacity: 0.7 }}>{p.icon}</span> {p.name}
            <span style={{ fontSize: 9, background: activeProject === p.id ? p.color + "22" : "#151515", padding: "1px 6px", borderRadius: 10, color: activeProject === p.id ? p.color : "#333" }}>
              {p.artifacts.length}
            </span>
          </button>
        ))}
      </div>

      {/* Content */}
      <div style={{ maxWidth: 1100, margin: "0 auto", padding: "0 32px 80px" }}>
        <ProjectView key={activeProject} project={project} />
      </div>

      {/* Footer */}
      <div style={{ borderTop: "1px solid #111", padding: "16px 32px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ fontSize: 10, color: "#2a2a2a", letterSpacing: "0.1em" }}>ROHIT ENGAGEMENT OS · MARCH 2026</span>
        <span style={{ fontSize: 10, color: "#2a2a2a" }}>4 projects · {PROJECTS.reduce((a, p) => a + p.artifacts.length, 0)} artifacts</span>
      </div>
    </div>
  );
}
