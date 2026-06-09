import { useEffect, useState, useCallback } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, AreaChart, Area,
} from "recharts";
import {
  FileText, Users, Tag, Share2, RefreshCw,
  Building2, MapPin, Package, TrendingUp,
} from "lucide-react";
import StatCard from "../components/ui/StatCard";
import UploadZone from "../components/ui/UploadZone";
import { getAnalytics, listDocuments, deleteDocument } from "../services/api";
import type { Analytics, Document } from "../types";

const COLORS = ["#4f8ef7", "#9b6dff", "#00d4b8", "#f5a623", "#ff5a6e", "#34d058", "#e879f9", "#38bdf8"];

export default function DashboardPage() {
  const [analytics, setAnalytics] = useState<Analytics | null>(null);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [a, d] = await Promise.all([getAnalytics(), listDocuments()]);
      setAnalytics(a);
      setDocuments(d.documents ?? []);
    } catch {
      setError("Could not load analytics. Ensure the backend is running at http://localhost:8000");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const handleDelete = async (docId: string) => {
    if (!confirm("Delete this document and all its data?")) return;
    await deleteDocument(docId);
    load();
  };

  if (loading) return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "60vh", flexDirection: "column", gap: 16 }}>
      <div style={{ width: 48, height: 48, border: "3px solid var(--border)", borderTopColor: "var(--accent)", borderRadius: "50%", animation: "spin 0.8s linear infinite" }} />
      <span style={{ color: "var(--text-muted)" }}>Loading analytics…</span>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );

  if (error) return (
    <div style={{ padding: 32, textAlign: "center" }}>
      <div style={{ color: "var(--red)", marginBottom: 12 }}>{error}</div>
      <button className="btn-primary" onClick={load}><RefreshCw size={14} /> Retry</button>
    </div>
  );

  const s = analytics?.summary;
  const ea = analytics?.entity_analytics;
  const ta = analytics?.topic_analytics;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>

      {/* ── Header row ──────────────────────────────────── */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div>
          <h2 style={{ fontSize: 22, fontWeight: 700 }}>Analytics Dashboard</h2>
          <p style={{ color: "var(--text-muted)", fontSize: 13, marginTop: 2 }}>
            Cross-document intelligence across {s?.total_documents ?? 0} document(s)
          </p>
        </div>
        <button className="btn-secondary" onClick={load} style={{ fontSize: 13 }}>
          <RefreshCw size={13} /> Refresh
        </button>
      </div>

      {/* ── Stat cards ──────────────────────────────────── */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 14 }}>
        <StatCard value={s?.total_documents ?? 0}   label="Total Documents"   icon={<FileText size={20} color="#4f8ef7" />} color="var(--accent)"  dimColor="var(--accent-dim)" />
        <StatCard value={s?.total_entities ?? 0}    label="Unique Entities"   icon={<Users size={20} color="#9b6dff" />}    color="var(--purple)"  dimColor="var(--purple-dim)" />
        <StatCard value={s?.total_topics ?? 0}      label="Topics Discovered" icon={<Tag size={20} color="#00d4b8" />}       color="var(--teal)"    dimColor="var(--teal-dim)"   />
        <StatCard value={s?.total_graph_nodes ?? 0} label="Graph Nodes"       icon={<Share2 size={20} color="#f5a623" />}    color="var(--amber)"   dimColor="var(--amber-dim)"  />
      </div>

      {/* ── Row 2: Org chart + Topic distribution ─────────── */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>

        {/* Top Organizations bar chart */}
        <div className="card" style={{ padding: "20px 22px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 18 }}>
            <Building2 size={16} color="var(--accent)" />
            <span style={{ fontWeight: 600, fontSize: 14 }}>Top Organizations</span>
          </div>
          {ea?.top_organizations?.length ? (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={ea.top_organizations.slice(0, 8)} layout="vertical" margin={{ left: 10, right: 20 }}>
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis dataKey="entity" type="category" tick={{ fontSize: 11 }} width={90} />
                <Tooltip
                  contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }}
                  labelStyle={{ color: "var(--text-primary)" }}
                />
                <Bar dataKey="frequency" radius={[0, 4, 4, 0]}>
                  {ea.top_organizations.slice(0, 8).map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : <Empty msg="No organization data yet" />}
        </div>

        {/* Topic distribution pie */}
        <div className="card" style={{ padding: "20px 22px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 18 }}>
            <Tag size={16} color="var(--purple)" />
            <span style={{ fontWeight: 600, fontSize: 14 }}>Topic Distribution</span>
          </div>
          {ta?.topic_distribution?.length ? (
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie
                  data={ta.topic_distribution.slice(0, 8).map(t => ({
                    ...t,
                    // Use topic_weight for sizing; fall back to count if missing
                    value: (t as any).topic_weight > 0 ? (t as any).topic_weight : t.count,
                  }))}
                  dataKey="value"
                  nameKey="topic"
                  cx="50%" cy="50%"
                  outerRadius={80}
                  label={({ topic }) => topic?.slice(0, 14)}
                  labelLine={false}
                >
                  {ta.topic_distribution.slice(0, 8).map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
              </PieChart>
            </ResponsiveContainer>
          ) : <Empty msg="No topic data yet" />}
        </div>

      </div>

      {/* ── Row 3: Keywords area + People bar ─────────────── */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>

        {/* Top keywords area chart */}
        <div className="card" style={{ padding: "20px 22px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 18 }}>
            <TrendingUp size={16} color="var(--teal)" />
            <span style={{ fontWeight: 600, fontSize: 14 }}>Keyword Frequency</span>
          </div>
          {ta?.top_keywords?.length ? (
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={ta.top_keywords.slice(0, 15)} margin={{ left: 0, right: 10 }}>
                <defs>
                  <linearGradient id="kw-grad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#00d4b8" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#00d4b8" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="keyword" tick={{ fontSize: 9 }} interval={1} angle={-30} textAnchor="end" height={40} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }} />
                <Area dataKey="count" stroke="#00d4b8" fill="url(#kw-grad)" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          ) : <Empty msg="No keyword data yet" />}
        </div>

        {/* Top people */}
        <div className="card" style={{ padding: "20px 22px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 18 }}>
            <Users size={16} color="var(--purple)" />
            <span style={{ fontWeight: 600, fontSize: 14 }}>Top People Mentioned</span>
          </div>
          {ea?.top_people?.length ? (
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {ea.top_people.slice(0, 8).map((p, i) => (
                <div key={p.entity} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <div style={{ width: 22, height: 22, borderRadius: "50%", background: COLORS[i % COLORS.length] + "30", border: `1px solid ${COLORS[i % COLORS.length]}50`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 9, fontWeight: 700, color: COLORS[i % COLORS.length] }}>
                    {i + 1}
                  </div>
                  <span style={{ flex: 1, fontSize: 13, color: "var(--text-primary)" }}>{p.entity}</span>
                  <span style={{ fontSize: 12, color: "var(--text-muted)", fontVariantNumeric: "tabular-nums" }}>{p.frequency}×</span>
                  <div style={{ width: 60, height: 4, background: "var(--border)", borderRadius: 2, overflow: "hidden" }}>
                    <div style={{ width: `${Math.min(100, (p.frequency / (ea.top_people[0]?.frequency || 1)) * 100)}%`, height: "100%", background: COLORS[i % COLORS.length], borderRadius: 2 }} />
                  </div>
                </div>
              ))}
            </div>
          ) : <Empty msg="No people data yet" />}
        </div>
      </div>

      {/* ── Row 4: Graph stats + Locations ────────────────── */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>

        {/* Graph analytics */}
        <div className="card" style={{ padding: "20px 22px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 18 }}>
            <Share2 size={16} color="var(--amber)" />
            <span style={{ fontWeight: 600, fontSize: 14 }}>Knowledge Graph Stats</span>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            {[
              { label: "Total Nodes",       value: s?.total_graph_nodes ?? 0,      color: "var(--accent)" },
              { label: "Total Edges",       value: s?.total_graph_edges ?? 0,      color: "var(--purple)" },
              { label: "Avg Graph Density", value: (s?.average_graph_density ?? 0).toFixed(3), color: "var(--teal)" },
              { label: "Top Entity",        value: s?.most_connected_entities?.[0] ?? "—", color: "var(--amber)" },
            ].map(({ label, value, color }) => (
              <div key={label} style={{ background: "var(--bg-secondary)", borderRadius: 8, padding: "12px 14px", border: "1px solid var(--border)" }}>
                <div style={{ fontSize: 18, fontWeight: 700, color }}>{value}</div>
                <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>{label}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Top locations */}
        <div className="card" style={{ padding: "20px 22px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 18 }}>
            <MapPin size={16} color="var(--red)" />
            <span style={{ fontWeight: 600, fontSize: 14 }}>Top Locations</span>
          </div>
          {ea?.top_locations?.length ? (
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
              {ea.top_locations.slice(0, 12).map((loc, i) => (
                <div key={loc.entity} style={{
                  display: "flex", alignItems: "center", gap: 6,
                  background: "var(--bg-secondary)", border: "1px solid var(--border)",
                  borderRadius: 20, padding: "6px 12px",
                }}>
                  <MapPin size={10} color={COLORS[i % COLORS.length]} />
                  <span style={{ fontSize: 12 }}>{loc.entity}</span>
                  <span style={{ fontSize: 10, color: "var(--text-muted)" }}>{loc.frequency}</span>
                </div>
              ))}
            </div>
          ) : <Empty msg="No location data yet" />}
        </div>
      </div>

      {/* ── Documents table ────────────────────────────── */}
      <div className="card" style={{ padding: "20px 22px" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 18 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <FileText size={16} color="var(--accent)" />
            <span style={{ fontWeight: 600, fontSize: 14 }}>Documents</span>
          </div>
          <span className="badge badge-blue">{documents.length} total</span>
        </div>

        {documents.length === 0 ? (
          <div style={{ padding: "32px", textAlign: "center" }}>
            <div style={{ color: "var(--text-muted)", marginBottom: 20 }}>No documents yet. Upload one to get started.</div>
            <div style={{ maxWidth: 400, margin: "0 auto" }}>
              <UploadZone onSuccess={load} />
            </div>
          </div>
        ) : (
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
            <thead>
              <tr style={{ borderBottom: "1px solid var(--border)" }}>
                {["Document", "Format", "Pages", "Status", ""].map(h => (
                  <th key={h} style={{ textAlign: "left", padding: "8px 10px", color: "var(--text-muted)", fontWeight: 500, fontSize: 11, textTransform: "uppercase" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {documents.map((doc) => (
                <tr key={doc.doc_id} style={{ borderBottom: "1px solid var(--border)", transition: "background 0.15s" }}
                  onMouseEnter={e => (e.currentTarget.style.background = "var(--bg-card-hover)")}
                  onMouseLeave={e => (e.currentTarget.style.background = "transparent")}>
                  <td style={{ padding: "10px 10px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <FileText size={14} color="var(--accent)" />
                      <span style={{ fontWeight: 500 }}>{doc.original_name ?? doc.document_name}</span>
                    </div>
                  </td>
                  <td style={{ padding: "10px 10px" }}>
                    <span className="badge badge-purple" style={{ fontSize: 9 }}>{(doc.format ?? "?").toUpperCase()}</span>
                  </td>
                  <td style={{ padding: "10px 10px", color: "var(--text-secondary)" }}>{doc.total_pages ?? "—"}</td>
                  <td style={{ padding: "10px 10px" }}>
                    <span className={`badge ${doc.status === "processed" ? "badge-green" : doc.status === "error" ? "badge-red" : "badge-amber"}`} style={{ fontSize: 9 }}>
                      {doc.status ?? "uploaded"}
                    </span>
                  </td>
                  <td style={{ padding: "10px 10px", textAlign: "right" }}>
                    <button
                      onClick={() => handleDelete(doc.doc_id)}
                      style={{ background: "none", border: "1px solid var(--border)", borderRadius: 6, color: "var(--red)", cursor: "pointer", padding: "4px 10px", fontSize: 11, transition: "all 0.15s" }}
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* ── Upload more ────────────────────────────────── */}
      {documents.length > 0 && (
        <div className="card" style={{ padding: "20px 22px" }}>
          <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 14, display: "flex", alignItems: "center", gap: 8 }}>
            <Package size={16} color="var(--teal)" /> Add More Documents
          </div>
          <UploadZone onSuccess={load} />
        </div>
      )}
    </div>
  );
}

function Empty({ msg }: { msg: string }) {
  return (
    <div style={{ height: 120, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--text-muted)", fontSize: 13 }}>
      {msg}
    </div>
  );
}
