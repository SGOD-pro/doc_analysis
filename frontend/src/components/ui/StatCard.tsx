interface Props {
  value: number | string;
  label: string;
  icon: React.ReactNode;
  color?: string;
  dimColor?: string;
  subtitle?: string;
}

export default function StatCard({ value, label, icon, color = "var(--accent)", dimColor = "var(--accent-dim)", subtitle }: Props) {
  return (
    <div className="card card-hover" style={{ padding: "20px 22px" }}>
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 12 }}>
        <div>
          <div style={{ fontSize: 28, fontWeight: 800, color, lineHeight: 1.1, fontVariantNumeric: "tabular-nums" }}>
            {typeof value === "number" ? value.toLocaleString() : value}
          </div>
          <div style={{ fontSize: 13, color: "var(--text-secondary)", marginTop: 4 }}>{label}</div>
          {subtitle && (
            <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>{subtitle}</div>
          )}
        </div>
        <div style={{
          width: 44, height: 44,
          background: dimColor,
          borderRadius: 10,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexShrink: 0,
        }}>
          {icon}
        </div>
      </div>
    </div>
  );
}
