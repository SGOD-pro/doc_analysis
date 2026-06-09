import { useLocation, useNavigate } from "react-router-dom";
import { Home, Upload, Bell } from "lucide-react";

const TITLES: Record<string, string> = {
  "/dashboard":  "Analytics Dashboard",
  "/search":     "Document Search",
  "/graph":      "Knowledge Graph",
};

export default function Header() {
  const location = useLocation();
  const navigate = useNavigate();
  const title = TITLES[location.pathname] ?? "DocAnalytics";

  return (
    <header style={{
      height: 60,
      background: "var(--bg-secondary)",
      borderBottom: "1px solid var(--border)",
      display: "flex",
      alignItems: "center",
      padding: "0 24px",
      gap: 12,
      position: "sticky",
      top: 0,
      zIndex: 10,
    }}>
      <button
        onClick={() => navigate("/")}
        style={{
          background: "none", border: "none", cursor: "pointer",
          color: "var(--text-muted)", display: "flex", alignItems: "center",
          padding: 4, borderRadius: 6, transition: "color 0.2s",
        }}
        title="Back to Home"
      >
        <Home size={16} />
      </button>

      <div style={{ width: 1, height: 20, background: "var(--border)" }} />

      <h1 style={{ fontSize: 15, fontWeight: 600, color: "var(--text-primary)", flex: 1 }}>
        {title}
      </h1>

      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <div style={{
          display: "flex", alignItems: "center", gap: 6,
          background: "var(--accent-dim)", border: "1px solid rgba(79,142,247,0.2)",
          borderRadius: 20, padding: "4px 12px",
        }}>
          <div style={{ width: 7, height: 7, borderRadius: "50%", background: "#34d058", boxShadow: "0 0 6px #34d058" }} />
          <span style={{ fontSize: 11, color: "var(--accent)", fontWeight: 600 }}>API Live</span>
        </div>
        <button style={{
          background: "var(--bg-card)", border: "1px solid var(--border)",
          borderRadius: 8, color: "var(--text-secondary)", cursor: "pointer",
          display: "flex", alignItems: "center", padding: 7,
          transition: "all 0.2s",
        }}>
          <Bell size={14} />
        </button>
      </div>
    </header>
  );
}
