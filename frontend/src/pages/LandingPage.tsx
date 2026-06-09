import { useNavigate } from "react-router-dom";
import {
  Upload, Zap, Brain, Share2, Search, BarChart3,
  FileText, FileImage, File, ChevronRight, Cpu,
  Network, TrendingUp, Database,
} from "lucide-react";
import UploadZone from "../components/ui/UploadZone";

const FEATURES = [
  { icon: FileText, color: "#4f8ef7", bg: "rgba(79,142,247,0.12)", label: "Document Processing", desc: "PDF, DOCX, TXT, MD — extract text, pages & paragraphs automatically." },
  { icon: Brain,    color: "#9b6dff", bg: "rgba(155,109,255,0.12)", label: "Entity Extraction",  desc: "spaCy NER identifies people, orgs, products, locations & more." },
  { icon: Cpu,      color: "#00d4b8", bg: "rgba(0,212,184,0.1)",   label: "Topic Modeling",    desc: "BERTopic discovers topics, keywords & trends across documents." },
  { icon: Network,  color: "#f5a623", bg: "rgba(245,166,35,0.12)", label: "Knowledge Graph",   desc: "NetworkX builds entity-relationship graphs from co-occurrence." },
  { icon: Search,   color: "#ff5a6e", bg: "rgba(255,90,110,0.12)", label: "Smart Search",      desc: "TF-IDF inverted index delivers ranked results with highlights." },
  { icon: BarChart3,color: "#34d058", bg: "rgba(52,208,88,0.1)",  label: "Analytics",         desc: "Recharts dashboards visualise trends, frequencies & distributions." },
];

const WORKFLOW = [
  { step: "01", label: "Upload",  icon: Upload,    desc: "Drag & drop any document" },
  { step: "02", label: "Process", icon: Cpu,       desc: "NLP pipeline runs automatically" },
  { step: "03", label: "Explore", icon: TrendingUp, desc: "Insights, graph & search ready" },
];

const TECH = [
  { name: "FastAPI",   badge: "Backend" },
  { name: "spaCy",     badge: "NLP" },
  { name: "BERTopic",  badge: "NLP" },
  { name: "NetworkX",  badge: "Graph" },
  { name: "React 18",  badge: "Frontend" },
  { name: "Tailwind",  badge: "Styling" },
  { name: "Recharts",  badge: "Charts" },
  { name: "React Flow",badge: "Graph UI" },
];

export default function LandingPage() {
  const navigate = useNavigate();

  return (
    <div style={{ minHeight: "100vh", background: "var(--bg-primary)", overflowX: "hidden" }}>

      {/* ── Nav bar ─────────────────────────────────────── */}
      <nav style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "16px 48px",
        borderBottom: "1px solid var(--border)",
        background: "rgba(10,14,26,0.8)",
        backdropFilter: "blur(12px)",
        position: "sticky", top: 0, zIndex: 100,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{
            width: 36, height: 36,
            background: "linear-gradient(135deg, #4f8ef7, #9b6dff)",
            borderRadius: 10,
            display: "flex", alignItems: "center", justifyContent: "center",
          }}>
            <Zap size={18} color="white" />
          </div>
          <span style={{ fontWeight: 700, fontSize: 16 }}>DocAnalytics</span>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button className="btn-secondary" onClick={() => navigate("/dashboard")} style={{ fontSize: 13, padding: "8px 16px" }}>
            Dashboard
          </button>
          <button className="btn-primary" onClick={() => navigate("/search")} style={{ fontSize: 13, padding: "8px 16px" }}>
            <Search size={14} /> Search Docs
          </button>
        </div>
      </nav>

      {/* ── Hero ────────────────────────────────────────── */}
      <section style={{
        padding: "80px 48px 64px",
        display: "grid",
        gridTemplateColumns: "1fr 420px",
        gap: 60,
        maxWidth: 1200,
        margin: "0 auto",
        alignItems: "center",
      }}>
        <div className="animate-fade-up">
          <div className="badge badge-blue" style={{ marginBottom: 20 }}>
            <Zap size={10} /> LLM-Powered Intelligence
          </div>
          <h1 style={{ fontSize: "clamp(36px, 5vw, 56px)", fontWeight: 900, lineHeight: 1.1, marginBottom: 20 }}>
            <span className="gradient-text">Document Analytics</span>
            <br />Engine
          </h1>
          <p style={{ fontSize: 17, color: "var(--text-secondary)", lineHeight: 1.7, marginBottom: 16, maxWidth: 500 }}>
            Reduce manual document analysis time by <strong style={{ color: "var(--accent)" }}>60%</strong> through
            automated processing, entity extraction, topic discovery, and knowledge graph generation.
          </p>

          {/* Student info */}
          <div className="card" style={{ padding: "14px 18px", marginBottom: 28, display: "inline-flex", gap: 24, flexWrap: "wrap" }}>
            <div>
              <div style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.07em" }}>Student</div>
              <div style={{ fontWeight: 600, fontSize: 14, color: "var(--text-primary)" }}>Swyra</div>
            </div>
            <div style={{ width: 1, background: "var(--border)" }} />
            <div>
              <div style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.07em" }}>Email</div>
              <div style={{ fontWeight: 600, fontSize: 14, color: "var(--accent)" }}>swyra@example.com</div>
            </div>
            <div style={{ width: 1, background: "var(--border)" }} />
            <div>
              <div style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.07em" }}>Topic</div>
              <div style={{ fontWeight: 600, fontSize: 14, color: "var(--purple)" }}>LLM Powered Document Analytics Engine</div>
            </div>
          </div>

          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            <button className="btn-primary" onClick={() => navigate("/dashboard")} style={{ padding: "12px 24px", fontSize: 14 }}>
              <BarChart3 size={16} /> View Dashboard <ChevronRight size={14} />
            </button>
            <button className="btn-secondary" onClick={() => navigate("/graph")} style={{ padding: "12px 24px", fontSize: 14 }}>
              <Share2 size={16} /> Knowledge Graph
            </button>
          </div>

          {/* Supported formats */}
          <div style={{ marginTop: 24, display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ fontSize: 12, color: "var(--text-muted)" }}>Supported formats:</span>
            {[
              { ext: "PDF", icon: FileImage },
              { ext: "DOCX", icon: FileText },
              { ext: "TXT", icon: File },
              { ext: "MD", icon: File },
            ].map(({ ext }) => (
              <span key={ext} className="badge badge-blue" style={{ fontSize: 10 }}>{ext}</span>
            ))}
          </div>
        </div>

        {/* Upload card */}
        <div className="card gradient-border animate-fade-up" style={{ padding: 24, animationDelay: "0.15s" }}>
          <div style={{ marginBottom: 20 }}>
            <div style={{ fontWeight: 700, fontSize: 16, marginBottom: 4 }}>Upload & Analyze</div>
            <div style={{ fontSize: 13, color: "var(--text-muted)" }}>
              Drop any document to run the full NLP pipeline
            </div>
          </div>
          <UploadZone onSuccess={() => setTimeout(() => navigate("/dashboard"), 1500)} />
        </div>
      </section>

      {/* ── Workflow ─────────────────────────────────────── */}
      <section style={{ padding: "48px 48px", maxWidth: 1200, margin: "0 auto" }}>
        <div style={{ textAlign: "center", marginBottom: 40 }}>
          <h2 style={{ fontSize: 28, fontWeight: 800, marginBottom: 8 }}>How It Works</h2>
          <p style={{ color: "var(--text-muted)", fontSize: 14 }}>Three steps from raw document to actionable insights</p>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 20 }}>
          {WORKFLOW.map(({ step, label, icon: Icon, desc }, i) => (
            <div key={step} className="card card-hover" style={{ padding: "28px 24px", textAlign: "center", position: "relative" }}>
              <div style={{
                fontSize: 48, fontWeight: 900,
                color: "var(--border-bright)",
                lineHeight: 1,
                fontVariantNumeric: "tabular-nums",
                marginBottom: 16,
              }}>
                {step}
              </div>
              <div style={{
                width: 48, height: 48,
                background: "var(--accent-dim)",
                borderRadius: 12,
                display: "flex", alignItems: "center", justifyContent: "center",
                margin: "0 auto 14px",
              }}>
                <Icon size={22} color="var(--accent)" />
              </div>
              <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 6 }}>{label}</div>
              <div style={{ fontSize: 13, color: "var(--text-muted)" }}>{desc}</div>
              {i < WORKFLOW.length - 1 && (
                <ChevronRight size={20} color="var(--border-bright)"
                  style={{ position: "absolute", right: -12, top: "50%", transform: "translateY(-50%)", zIndex: 1 }} />
              )}
            </div>
          ))}
        </div>
      </section>

      {/* ── Feature cards ─────────────────────────────────── */}
      <section style={{ padding: "48px 48px", maxWidth: 1200, margin: "0 auto" }}>
        <div style={{ textAlign: "center", marginBottom: 40 }}>
          <h2 style={{ fontSize: 28, fontWeight: 800, marginBottom: 8 }}>Platform Capabilities</h2>
          <p style={{ color: "var(--text-muted)", fontSize: 14 }}>Six intelligent modules working in harmony</p>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16 }}>
          {FEATURES.map(({ icon: Icon, color, bg, label, desc }) => (
            <div key={label} className="card card-hover" style={{ padding: "22px 20px" }}>
              <div style={{
                width: 44, height: 44, background: bg, borderRadius: 10,
                display: "flex", alignItems: "center", justifyContent: "center",
                marginBottom: 14,
              }}>
                <Icon size={20} color={color} />
              </div>
              <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 6, color: "var(--text-primary)" }}>{label}</div>
              <div style={{ fontSize: 13, color: "var(--text-muted)", lineHeight: 1.5 }}>{desc}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Tech stack ─────────────────────────────────────── */}
      <section style={{
        margin: "0 48px 64px",
        borderRadius: 16,
        background: "var(--bg-secondary)",
        border: "1px solid var(--border)",
        padding: "36px 40px",
        maxWidth: 1200 - 96,
        marginLeft: "auto",
        marginRight: "auto",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 24 }}>
          <Database size={18} color="var(--accent)" />
          <h3 style={{ fontWeight: 700, fontSize: 17 }}>Technology Stack</h3>
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
          {TECH.map(({ name, badge }) => (
            <div key={name} style={{
              display: "flex", alignItems: "center", gap: 8,
              background: "var(--bg-card)", border: "1px solid var(--border)",
              borderRadius: 8, padding: "8px 14px",
            }}>
              <span style={{ fontWeight: 600, fontSize: 13 }}>{name}</span>
              <span className="badge badge-purple" style={{ fontSize: 9 }}>{badge}</span>
            </div>
          ))}
        </div>
      </section>

      {/* ── Footer ──────────────────────────────────────── */}
      <footer style={{
        borderTop: "1px solid var(--border)",
        padding: "20px 48px",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        color: "var(--text-muted)",
        fontSize: 12,
      }}>
        <span>© 2024 LLM Document Analytics Engine — Internship Project</span>
        <span style={{ color: "var(--accent)" }}>Built with FastAPI + React + spaCy + BERTopic + NetworkX</span>
      </footer>
    </div>
  );
}
