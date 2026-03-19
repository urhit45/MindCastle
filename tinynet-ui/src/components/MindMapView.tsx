import React from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { getMindMap, type MindMapLink, type MindMapNode } from '../api/client';

type GraphData = {
  nodes: MindMapNode[];
  links: MindMapLink[];
};

type EngineMapping = {
  name: string;
  icon?: string;
  color?: string;
};

const SIGNAL_COLOR: Record<MindMapLink['dominant_signal'], string> = {
  temporal_co_activation: '#6a7bf7',
  blocked_propagation: '#ff7d7d',
  progress_correlation: '#5fd085',
};

interface MindMapViewProps {
  onNavigateToEngine?: (node: MindMapNode) => void;
}

function dominantSignal(
  breakdown: Partial<Record<MindMapLink['dominant_signal'], number>>,
): MindMapLink['dominant_signal'] {
  const temporal = breakdown.temporal_co_activation ?? 0;
  const blocked = breakdown.blocked_propagation ?? 0;
  const progress = breakdown.progress_correlation ?? 0;
  if (blocked >= temporal && blocked >= progress) return 'blocked_propagation';
  if (progress >= temporal && progress >= blocked) return 'progress_correlation';
  return 'temporal_co_activation';
}

function loadEngineMappingsByNodeId(): Map<string, EngineMapping> {
  const byNodeId = new Map<string, EngineMapping>();
  if (typeof window === 'undefined') return byNodeId;

  try {
    const raw = window.localStorage.getItem('mindcastle_v1');
    if (!raw) return byNodeId;
    const parsed = JSON.parse(raw) as Array<{
      name?: string;
      icon?: string;
      color?: string;
      artifacts?: Array<{ node_id?: string }>;
    }>;

    if (!Array.isArray(parsed)) return byNodeId;

    for (const engine of parsed) {
      const engineName = engine?.name?.trim();
      if (!engineName) continue;
      const artifacts = Array.isArray(engine.artifacts) ? engine.artifacts : [];
      for (const artifact of artifacts) {
        const nodeId = artifact?.node_id?.trim();
        if (!nodeId) continue;
        byNodeId.set(nodeId, {
          name: engineName,
          icon: engine.icon,
          color: engine.color,
        });
      }
    }
  } catch {
    // Ignore malformed local cache; backend graph still renders safely.
  }

  return byNodeId;
}

function aggregateGraphToEngineLevel(raw: GraphData): GraphData {
  const mappingByNodeId = loadEngineMappingsByNodeId();
  if (mappingByNodeId.size === 0) return raw;

  type Group = {
    id: string;
    name: string;
    val: number;
    color: string;
    icon: string;
    status: string;
    memberNodeIds: Set<string>;
  };

  const groupsByName = new Map<string, Group>();
  const groupIdByNodeId = new Map<string, string>();

  for (const node of raw.nodes) {
    const mapped = mappingByNodeId.get(node.id);
    const engineName = mapped?.name ?? node.name;
    const existing = groupsByName.get(engineName);

    if (!existing) {
      groupsByName.set(engineName, {
        id: node.id,
        name: engineName,
        val: Math.max(1, node.val),
        color: mapped?.color || node.color,
        icon: mapped?.icon || node.icon,
        status: node.status,
        memberNodeIds: new Set([node.id]),
      });
      groupIdByNodeId.set(node.id, node.id);
      continue;
    }

    existing.val += Math.max(1, node.val);
    existing.memberNodeIds.add(node.id);
    groupIdByNodeId.set(node.id, existing.id);

    if (node.status === 'blocked') {
      existing.status = 'blocked';
      existing.color = '#ff7d7d';
      existing.icon = '🚧';
    }
  }

  const mergedLinks = new Map<string, MindMapLink>();
  for (const link of raw.links) {
    const sourceId = groupIdByNodeId.get(String(link.source)) ?? String(link.source);
    const targetId = groupIdByNodeId.get(String(link.target)) ?? String(link.target);
    if (sourceId === targetId) continue;

    const pairKey = sourceId < targetId ? `${sourceId}|${targetId}` : `${targetId}|${sourceId}`;
    const existing = mergedLinks.get(pairKey);
    if (!existing) {
      mergedLinks.set(pairKey, {
        source: sourceId,
        target: targetId,
        weight: link.weight,
        width: link.width,
        dominant_signal: link.dominant_signal,
        signal_breakdown: {
          temporal_co_activation: link.signal_breakdown.temporal_co_activation ?? 0,
          blocked_propagation: link.signal_breakdown.blocked_propagation ?? 0,
          progress_correlation: link.signal_breakdown.progress_correlation ?? 0,
        },
      });
      continue;
    }

    existing.weight += link.weight;
    existing.signal_breakdown.temporal_co_activation =
      (existing.signal_breakdown.temporal_co_activation ?? 0) +
      (link.signal_breakdown.temporal_co_activation ?? 0);
    existing.signal_breakdown.blocked_propagation =
      (existing.signal_breakdown.blocked_propagation ?? 0) +
      (link.signal_breakdown.blocked_propagation ?? 0);
    existing.signal_breakdown.progress_correlation =
      (existing.signal_breakdown.progress_correlation ?? 0) +
      (link.signal_breakdown.progress_correlation ?? 0);
    existing.dominant_signal = dominantSignal(existing.signal_breakdown);
    existing.width = Math.max(1, Math.min(8, existing.weight * 4.5));
  }

  return {
    nodes: Array.from(groupsByName.values()).map((g) => ({
      id: g.id,
      name: g.name,
      val: g.val,
      color: g.color,
      icon: g.icon,
      status: g.status,
    })),
    links: Array.from(mergedLinks.values()),
  };
}

export default function MindMapView({ onNavigateToEngine }: MindMapViewProps) {
  const [graph, setGraph] = React.useState<GraphData>({ nodes: [], links: [] });
  const [selectedLink, setSelectedLink] = React.useState<MindMapLink | null>(null);
  const [selectedNode, setSelectedNode] = React.useState<MindMapNode | null>(null);
  const [status, setStatus] = React.useState<'idle' | 'loading' | 'ready' | 'error'>('idle');
  const [error, setError] = React.useState<string | null>(null);
  const graphStageRef = React.useRef<HTMLDivElement | null>(null);
  const [graphSize, setGraphSize] = React.useState({ width: 760, height: 380 });

  React.useEffect(() => {
    let mounted = true;
    const load = async () => {
      setStatus('loading');
      setError(null);
      try {
        const response = await getMindMap(0.04);
        if (!mounted) return;
        setGraph(aggregateGraphToEngineLevel({ nodes: response.nodes, links: response.links }));
        setStatus('ready');
      } catch (e) {
        if (!mounted) return;
        setStatus('error');
        setError(e instanceof Error ? e.message : 'Failed to load mind map');
      }
    };
    load();
    return () => {
      mounted = false;
    };
  }, []);

  React.useEffect(() => {
    const el = graphStageRef.current;
    if (!el) return;

    const updateSize = () => {
      const width = Math.max(320, Math.floor(el.clientWidth));
      const height = Math.max(320, Math.floor(width * 0.52));
      setGraphSize({ width, height });
    };

    updateSize();

    if (typeof ResizeObserver !== 'undefined') {
      const observer = new ResizeObserver(() => updateSize());
      observer.observe(el);
      return () => observer.disconnect();
    }

    window.addEventListener('resize', updateSize);
    return () => window.removeEventListener('resize', updateSize);
  }, [graph.nodes.length]);

  const openNode = (node: MindMapNode) => {
    setSelectedNode(node);
    onNavigateToEngine?.(node);
  };

  return (
    <section className="panel mind-map-panel">
      <div className="panel-header">
        <h2>Mind Map</h2>
        <p>Engines as nodes, learned relationships as weighted edges.</p>
      </div>

      <div className="mind-map-canvas-wrap">
        {status === 'loading' && <p className="mind-map-meta">Learning pattern topology...</p>}
        {status === 'error' && <p className="mind-map-meta">Map unavailable ({error ?? 'unknown error'}).</p>}
        {status === 'ready' && graph.nodes.length === 0 && (
          <div className="mind-map-empty">
            <p>No graph edges yet.</p>
            <p>As you add progress logs, this map learns how engines influence each other.</p>
          </div>
        )}
        {graph.nodes.length > 0 && (
          <div className="mind-map-graph-stage" ref={graphStageRef}>
            <ForceGraph2D
              width={graphSize.width}
              height={graphSize.height}
              graphData={graph}
              backgroundColor="#0f1118"
              nodeLabel={(node) => `${(node as MindMapNode).icon} ${(node as MindMapNode).name}`}
              nodeVal={(node) => (node as MindMapNode).val}
              nodeColor={(node) => (node as MindMapNode).color}
              linkColor={(link) => SIGNAL_COLOR[(link as MindMapLink).dominant_signal]}
              linkWidth={(link) => (link as MindMapLink).width}
              linkDirectionalParticles={1}
              linkDirectionalParticleWidth={1.2}
              onNodeClick={(node) => openNode(node as MindMapNode)}
              onLinkClick={(link) => setSelectedLink(link as MindMapLink)}
              nodeCanvasObject={(node, ctx, globalScale) => {
                const n = node as MindMapNode;
                const point = node as MindMapNode & { x?: number; y?: number };
                const label = `${n.icon} ${n.name}`;
                const fontSize = 12 / globalScale;
                ctx.font = `${fontSize}px Sans-Serif`;
                const textWidth = ctx.measureText(label).width;
                const bckgDimensions: [number, number] = [textWidth + 8, fontSize + 6];
                ctx.fillStyle = 'rgba(9, 14, 27, 0.74)';
                ctx.fillRect(
                  (point.x ?? 0) - bckgDimensions[0] / 2,
                  (point.y ?? 0) - bckgDimensions[1] / 2,
                  bckgDimensions[0],
                  bckgDimensions[1],
                );
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillStyle = '#f4f7ff';
                ctx.fillText(label, point.x ?? 0, point.y ?? 0);
              }}
            />
          </div>
        )}
      </div>

      <div className="mind-map-footnote">
        <p>Edge color: temporal (blue), blocked propagation (red), progress correlation (green).</p>
        {selectedNode && (
          <div className="mind-map-node-nav">
            <strong>Current node:</strong> {selectedNode.icon} {selectedNode.name}
            <button type="button" className="pill-btn" onClick={() => onNavigateToEngine?.(selectedNode)}>
              Open engine
            </button>
          </div>
        )}
        {selectedLink && (
          <div className="mind-map-why">
            <strong>Why connected:</strong>{' '}
            Temporal {(selectedLink.signal_breakdown.temporal_co_activation ?? 0).toFixed(2)} | Blocked{' '}
            {(selectedLink.signal_breakdown.blocked_propagation ?? 0).toFixed(2)} | Progress{' '}
            {(selectedLink.signal_breakdown.progress_correlation ?? 0).toFixed(2)}
          </div>
        )}
      </div>
    </section>
  );
}
