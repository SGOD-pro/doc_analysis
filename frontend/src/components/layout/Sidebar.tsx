import { NavLink, useLocation } from "react-router-dom";
import {
  LayoutDashboard,
  Search,
  Share2,
  FileText,
  Zap,
  ChevronRight,
} from "lucide-react";

const NAV = [
  { to: "/dashboard", icon: LayoutDashboard, label: "Dashboard" },
  { to: "/search",    icon: Search,          label: "Search" },
  { to: "/graph",     icon: Share2,           label: "Knowledge Graph" },
];

export default function Sidebar() {
  const location = useLocation();

  return (
    <aside style={{
      width: 220,
      background: "var(--bg-secondary)",
      borderRight: "1px solid var(--border)",
      display: "flex",
      flexDirection: "column",
      flexShrink: 0,
      position: "sticky",
      top: 0,
      height: "100vh",
    }}>
      {/* Logo */}
      <div style={{
        padding: "20px 16px",
        borderBottom: "1px solid var(--border)",
        display: "flex",
        alignItems: "center",
        gap: 10,
      }}>
        <div style={{
          width: 36, height: 36,
          background: "linear-gradient(135deg, #4f8ef7, #9b6dff)",
          borderRadius: 10,
          display: "flex", alignItems: "center", justifyContent: "center",
        }}>
          <Zap size={18} color="white" />
        </div>
        <div>
          <div style={{ fontWeight: 700, fontSize: 14, color: "var(--text-primary)", lineHeight: 1.2 }}>
            DocAnalytics
          </div>
          <div style={{ fontSize: 10, color: "var(--text-muted)", letterSpacing: "0.05em" }}>
            LLM ENGINE
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav style={{ flex: 1, padding: "12px 8px" }}>
        <div style={{ fontSize: 10, fontWeight: 600, color: "var(--text-muted)", padding: "8px 8px 4px", letterSpacing: "0.08em", textTransform: "uppercase" }}>
          Navigation
        </div>
        {NAV.map(({ to, icon: Icon, label }) => {
          const active = location.pathname.startsWith(to);
          return (
            <NavLink
              key={to}
              to={to}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                padding: "9px 10px",
                borderRadius: 8,
                marginBottom: 2,
                textDecoration: "none",
                color: active ? "var(--accent)" : "var(--text-secondary)",
                background: active ? "var(--accent-dim)" : "transparent",
                fontWeight: active ? 600 : 400,
                fontSize: 13,
                transition: "all 0.15s ease",
                border: active ? "1px solid rgba(79,142,247,0.2)" : "1px solid transparent",
              }}
            >
              <Icon size={16} />
              <span style={{ flex: 1 }}>{label}</span>
              {active && <ChevronRight size={12} />}
            </NavLink>
          );
        })}
      </nav>

      {/* Footer */}
      <div style={{
        padding: "14px 16px",
        borderTop: "1px solid var(--border)",
        display: "flex",
        alignItems: "center",
        gap: 8,
      }}>
        <FileText size={14} color="var(--text-muted)" />
        <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
          <div>Internship Project</div>
          <div style={{ color: "var(--accent)", fontWeight: 500 }}>v1.0.0</div>
        </div>
      </div>
    </aside>
  );
}
