import { useEffect, useState, useCallback } from "react";
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  type Node,
  type Edge,
  BackgroundVariant,
} from "reactflow";
import "reactflow/dist/style.css";
import { Share2, RefreshCw, Info, Maximize2, Filter } from "lucide-react";
import { getKnowledgeGraph } from "../services/api";
import type { GraphData, GraphNode } from "../types";

const TYPE_COLORS: Record<string, string> = {
  PERSON:     "#4f8ef7",
  ORG:        "#9b6dff",
  GPE:        "#00d4b8",
  LOC:        "#00d4b8",
  PRODUCT:    "#f5a623",
  DATE:       "#ff5a6e",
  EVENT:      "#34d058",
  NORP:       "#e879f9",
  WORK_OF_ART:"#38bdf8",
  LAW:        "#fb923c",
  MONEY:      "#a3e635",
  DEFAULT:    "#8899b4",
};

const nodeColor = (type: string) => TYPE_COLORS[type] ?? TYPE_COLORS.DEFAULT;

function buildFlowNodes(nodes: GraphNode[]): Node[] {
  // Simple circular/force-like layout
  const total = nodes.length;
  const sorted = [...nodes].sort((a, b) => b.frequency - a.frequency);

  return sorted.slice(0, 80).map((n, i) => {
    const angle = (i / Math.min(total, 80)) * 2 * Math.PI;
    const radius = 250 + n.degree * 20;
    const color = nodeColor(n.type);
    const size = Math.max(32, Math.min(60, 28 + n.frequency * 2));

    return {
      id: n.id,
      data: { label: n.label, type: n.type, freq: n.frequency, color },
      position: {
        x: 500 + radius * Math.cos(angle),
        y: 400 + radius * Math.sin(angle),
      },
      style: {
        background: color + "22",
        border: `2px solid ${color}`,
        borderRadius: "50%",
        width: size,
        height: size,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color,
        fontSize: Math.max(8, Math.min(11, 6 + n.frequency)),
        fontWeight: 600,
        fontFamily: "Inter, sans-serif",
        padding: 2,
        textAlign: "center",
        lineHeight: 1.2,
        cursor: "pointer",
      },
      title: `${n.label} (${n.type_label}) — freq: ${n.frequency}`,
    };
  });
}

interface RawEdge { source: string; target: string; relationship: string; weight: number; }
function buildFlowEdges(edges: RawEdge[]): Edge[] {
  return edges.slice(0, 200).map((e, i) => ({
    id: `e-${i}`,
    source: e.source,
    target: e.target,
    label: e.relationship.replace(/_/g, " "),
    style: { stroke: "var(--border-bright)", strokeWidth: Math.min(3, e.weight * 0.5 + 0.5) },
    labelStyle: { fill: "var(--text-muted)", fontSize: 9 },
    labelBgStyle: { fill: "var(--bg-card)", fillOpacity: 0.9 },
    animated: e.weight > 3,
    type: "default",
  }));
}

const ALL_TYPES = Object.keys(TYPE_COLORS).filter(t => t !== "DEFAULT");

export default function KnowledgeGraphPage() {
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selected, setSelected] = useState<GraphNode | null>(null);
  const [filterType, setFilterType] = useState<string>("ALL");

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const g = await getKnowledgeGraph();
      setGraphData(g);
    } catch {
      setError("Could not load graph. Ensure backend is running and documents are processed.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const filteredNodes = (graphData?.nodes ?? []).filter(
    n => filterType === "ALL" || n.type === filterType
  );
  const filteredEdges = (graphData?.edges ?? []).filter(e => {
    if (filterType === "ALL") return true;
    const nodeIds = new Set(filteredNodes.map(n => n.id));
    return nodeIds.has(e.source) && nodeIds.has(e.target);
  });

  const flowNodes = buildFlowNodes(filteredNodes);
  const flowEdges = buildFlowEdges(filteredEdges);

  const analytics = graphData?.analytics;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20, height: "calc(100vh - 108px)" }}>

      {/* ── Header ─────────────────────────────────────── */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexShrink: 0 }}>
        <div>
          <h2 style={{ fontSize: 22, fontWeight: 700 }}>Knowledge Graph</h2>
          <p style={{ color: "var(--text-muted)", fontSize: 13, marginTop: 2 }}>
            Entity co-occurrence relationships across all documents
          </p>
        </div>
        <button className="btn-secondary" onClick={load} style={{ fontSize: 13 }}>
          <RefreshCw size={13} /> Refresh
        </button>
      </div>

      {/* ── Stats row ─────────────────────────────────── */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, flexShrink: 0 }}>
        {[
          { label: "Nodes",        value: analytics?.total_nodes ?? 0,  color: "var(--accent)" },
          { label: "Edges",        value: analytics?.total_edges ?? 0,  color: "var(--purple)" },
          { label: "Density",      value: analytics?.density?.toFixed(3) ?? "—", color: "var(--teal)" },
          { label: "Hub Entity",   value: analytics?.most_connected_entity || "—", color: "var(--amber)" },
        ].map(({ label, value, color }) => (
          <div key={label} className="card" style={{ padding: "14px 16px" }}>
            <div style={{ fontSize: 18, fontWeight: 700, color }}>{value}</div>
            <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>{label}</div>
          </div>
        ))}
      </div>

      {/* ── Main panel ────────────────────────────────── */}
      <div style={{ flex: 1, display: "grid", gridTemplateColumns: "1fr 240px", gap: 14, minHeight: 0 }}>

        {/* Graph canvas */}
        <div className="card" style={{ overflow: "hidden", position: "relative" }}>

          {/* Filter bar */}
          <div style={{
            position: "absolute", top: 12, left: 12, zIndex: 10,
            display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap",
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: 4, background: "rgba(10,14,26,0.7)", backdropFilter: "blur(8px)", borderRadius: 8, padding: "4px 8px", border: "1px solid var(--border)" }}>
              <Filter size={11} color="var(--text-muted)" />
              <span style={{ fontSize: 10, color: "var(--text-muted)" }}>Filter:</span>
              <select
                value={filterType}
                onChange={e => setFilterType(e.target.value)}
                style={{
                  background: "transparent", border: "none", color: "var(--text-primary)",
                  fontSize: 11, cursor: "pointer", outline: "none",
                }}
              >
                <option value="ALL">All Types</option>
                {ALL_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
              </select>
            </div>
            <div style={{ background: "rgba(10,14,26,0.7)", backdropFilter: "blur(8px)", borderRadius: 8, padding: "4px 10px", border: "1px solid var(--border)", fontSize: 10, color: "var(--text-muted)" }}>
              Showing {Math.min(flowNodes.length, 80)} nodes
            </div>
          </div>

          {loading ? (
            <div style={{ height: "100%", display: "flex", alignItems: "center", justifyContent: "center", flexDirection: "column", gap: 14 }}>
              <div style={{ width: 40, height: 40, border: "3px solid var(--border)", borderTopColor: "var(--accent)", borderRadius: "50%", animation: "spin 0.8s linear infinite" }} />
              <span style={{ color: "var(--text-muted)", fontSize: 13 }}>Building graph…</span>
            </div>
          ) : error ? (
            <div style={{ height: "100%", display: "flex", alignItems: "center", justifyContent: "center", flexDirection: "column", gap: 12 }}>
              <Share2 size={32} color="var(--text-muted)" />
              <span style={{ color: "var(--text-muted)", fontSize: 13, maxWidth: 300, textAlign: "center" }}>{error}</span>
              <button className="btn-primary" onClick={load} style={{ fontSize: 12 }}><RefreshCw size={12} /> Retry</button>
            </div>
          ) : flowNodes.length === 0 ? (
            <div style={{ height: "100%", display: "flex", alignItems: "center", justifyContent: "center", flexDirection: "column", gap: 12 }}>
              <Share2 size={32} color="var(--text-muted)" />
              <span style={{ color: "var(--text-muted)", fontSize: 13 }}>No graph data. Process a document first.</span>
            </div>
          ) : (
            <ReactFlow
              nodes={flowNodes}
              edges={flowEdges}
              fitView
              minZoom={0.1}
              maxZoom={3}
              onNodeClick={(_, node) => {
                const gn = filteredNodes.find(n => n.id === node.id);
                if (gn) setSelected(gn);
              }}
            >
              <Background variant={BackgroundVariant.Dots} color="var(--border)" gap={20} size={1} />
              <Controls style={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8 }} />
              <MiniMap
                style={{ background: "var(--bg-card)", border: "1px solid var(--border)" }}
                nodeColor={n => (n.style as { background?: string })?.background?.slice(0, 7) ?? "#4f8ef7"}
              />
            </ReactFlow>
          )}
        </div>

        {/* Right panel: legend + selected node details */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>

          {/* Legend */}
          <div className="card" style={{ padding: "16px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 12 }}>
              <Maximize2 size={13} color="var(--accent)" />
              <span style={{ fontWeight: 600, fontSize: 12 }}>Entity Types</span>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {Object.entries(TYPE_COLORS).filter(([k]) => k !== "DEFAULT").map(([type, color]) => (
                <button
                  key={type}
                  onClick={() => setFilterType(filterType === type ? "ALL" : type)}
                  style={{
                    display: "flex", alignItems: "center", gap: 8,
                    background: filterType === type ? color + "15" : "transparent",
                    border: `1px solid ${filterType === type ? color : "transparent"}`,
                    borderRadius: 6, padding: "5px 8px", cursor: "pointer",
                    textAlign: "left", width: "100%",
                  }}
                >
                  <div style={{ width: 10, height: 10, borderRadius: "50%", background: color, flexShrink: 0 }} />
                  <span style={{ fontSize: 11, color: "var(--text-secondary)" }}>{type}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Selected node details */}
          {selected ? (
            <div className="card" style={{ padding: "16px", flex: 1 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 14 }}>
                <Info size={13} color="var(--accent)" />
                <span style={{ fontWeight: 600, fontSize: 12 }}>Node Details</span>
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                <div>
                  <div style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em" }}>Entity</div>
                  <div style={{ fontWeight: 700, fontSize: 14, color: "var(--text-primary)", marginTop: 2 }}>{selected.label}</div>
                </div>
                {[
                  { k: "Type",        v: selected.type_label },
                  { k: "Frequency",   v: selected.frequency },
                  { k: "Connections", v: selected.degree },
                  { k: "Centrality",  v: selected.centrality?.toFixed(4) ?? "—" },
                  { k: "Betweenness", v: selected.betweenness?.toFixed(4) ?? "—" },
                ].map(({ k, v }) => (
                  <div key={k} style={{ display: "flex", justifyContent: "space-between", fontSize: 12, borderBottom: "1px solid var(--border)", paddingBottom: 6 }}>
                    <span style={{ color: "var(--text-muted)" }}>{k}</span>
                    <span style={{ fontWeight: 600, color: "var(--text-primary)" }}>{v}</span>
                  </div>
                ))}
                <button
                  onClick={() => setSelected(null)}
                  className="btn-secondary"
                  style={{ width: "100%", justifyContent: "center", fontSize: 11, padding: "7px" }}
                >
                  Deselect
                </button>
              </div>
            </div>
          ) : (
            <div className="card" style={{ padding: "16px", textAlign: "center", flex: 1 }}>
              <Share2 size={24} color="var(--text-muted)" style={{ margin: "20px auto 10px" }} />
              <div style={{ fontSize: 12, color: "var(--text-muted)", lineHeight: 1.5 }}>
                Click any node to inspect its properties
              </div>
            </div>
          )}
        </div>
      </div>

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
