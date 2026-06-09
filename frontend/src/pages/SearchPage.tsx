import { useState, useRef } from "react";
import { Search, X, FileText, ChevronRight, AlertCircle, Loader2, Hash } from "lucide-react";
import { searchDocuments } from "../services/api";
import type { SearchResult } from "../types";

const SUGGESTIONS = [
  "machine learning", "attention mechanism", "transformer architecture",
  "neural network", "natural language processing", "deep learning",
];

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [searched, setSearched] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const doSearch = async (q?: string) => {
    const term = (q ?? query).trim();
    if (!term) return;
    setQuery(term);
    setLoading(true);
    setError("");
    setSearched(false);
    try {
      const res = await searchDocuments(term, 30);
      setResults(res.results ?? []);
      setSearched(true);
    } catch {
      setError("Search failed. Make sure the backend is running and documents are indexed.");
    } finally {
      setLoading(false);
    }
  };

  const clear = () => {
    setQuery("");
    setResults([]);
    setSearched(false);
    inputRef.current?.focus();
  };

  // Highlight matched terms in snippet
  const highlight = (text: string) => {
    const parts = text.split(/(\*\*[^*]+\*\*)/g);
    return parts.map((part, i) =>
      part.startsWith("**") && part.endsWith("**")
        ? <mark key={i} style={{ background: "rgba(79,142,247,0.25)", color: "var(--accent)", borderRadius: 3, padding: "0 2px", fontWeight: 600 }}>
            {part.slice(2, -2)}
          </mark>
        : <span key={i}>{part}</span>
    );
  };

  const scoreColor = (score: number) =>
    score > 3 ? "var(--green)" : score > 1.5 ? "var(--amber)" : "var(--text-muted)";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24, maxWidth: 900, margin: "0 auto", width: "100%" }}>

      {/* ── Title ──────────────────────────────────────── */}
      <div>
        <h2 style={{ fontSize: 22, fontWeight: 700 }}>Document Search</h2>
        <p style={{ color: "var(--text-muted)", fontSize: 13, marginTop: 4 }}>
          TF-IDF ranked search across all processed documents
        </p>
      </div>

      {/* ── Search box ─────────────────────────────────── */}
      <div className="card" style={{ padding: "20px 22px" }}>
        <div style={{ display: "flex", gap: 10 }}>
          <div style={{ flex: 1, position: "relative" }}>
            <Search size={16} color="var(--text-muted)" style={{ position: "absolute", left: 12, top: "50%", transform: "translateY(-50%)", pointerEvents: "none" }} />
            <input
              ref={inputRef}
              className="input"
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={e => e.key === "Enter" && doSearch()}
              placeholder='Try: "attention mechanism", "neural network", "GPT"…'
              style={{ paddingLeft: 38, paddingRight: query ? 38 : 12, fontSize: 15 }}
            />
            {query && (
              <button onClick={clear} style={{ position: "absolute", right: 10, top: "50%", transform: "translateY(-50%)", background: "none", border: "none", cursor: "pointer", color: "var(--text-muted)" }}>
                <X size={14} />
              </button>
            )}
          </div>
          <button className="btn-primary" onClick={() => doSearch()} disabled={loading} style={{ padding: "10px 22px", minWidth: 100, justifyContent: "center" }}>
            {loading ? <Loader2 size={15} style={{ animation: "spin 0.8s linear infinite" }} /> : <><Search size={15} /> Search</>}
          </button>
        </div>

        {/* Suggestions */}
        <div style={{ marginTop: 12, display: "flex", flexWrap: "wrap", gap: 6, alignItems: "center" }}>
          <span style={{ fontSize: 11, color: "var(--text-muted)" }}>Try:</span>
          {SUGGESTIONS.map(s => (
            <button
              key={s}
              onClick={() => doSearch(s)}
              style={{
                background: "var(--bg-secondary)", border: "1px solid var(--border)",
                borderRadius: 20, padding: "4px 12px", fontSize: 11,
                color: "var(--text-secondary)", cursor: "pointer",
                transition: "all 0.15s",
              }}
              onMouseEnter={e => { (e.target as HTMLElement).style.borderColor = "var(--accent)"; (e.target as HTMLElement).style.color = "var(--accent)"; }}
              onMouseLeave={e => { (e.target as HTMLElement).style.borderColor = "var(--border)"; (e.target as HTMLElement).style.color = "var(--text-secondary)"; }}
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      {/* ── Error ──────────────────────────────────────── */}
      {error && (
        <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "12px 16px", background: "var(--red-dim)", border: "1px solid rgba(255,90,110,0.3)", borderRadius: 8, color: "var(--red)", fontSize: 13 }}>
          <AlertCircle size={14} /> {error}
        </div>
      )}

      {/* ── Results ────────────────────────────────────── */}
      {searched && (
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 14 }}>
            <span style={{ fontWeight: 600, fontSize: 14 }}>Results</span>
            <span className="badge badge-blue">{results.length} found</span>
            <span style={{ fontSize: 12, color: "var(--text-muted)" }}>for "{query}"</span>
          </div>

          {results.length === 0 ? (
            <div className="card" style={{ padding: "40px", textAlign: "center" }}>
              <Search size={32} color="var(--text-muted)" style={{ marginBottom: 12 }} />
              <div style={{ color: "var(--text-muted)", fontSize: 14 }}>No results found for "{query}"</div>
              <div style={{ color: "var(--text-muted)", fontSize: 12, marginTop: 6 }}>Try different keywords or upload more documents</div>
            </div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {results.map((r, i) => (
                <div key={i} className="card card-hover" style={{ padding: "18px 20px" }}>
                  <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 12, marginBottom: 10 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <FileText size={15} color="var(--accent)" />
                      <span style={{ fontWeight: 600, fontSize: 14 }}>{r.document_name}</span>
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
                      <span style={{ fontSize: 11, color: "var(--text-muted)" }}>Score</span>
                      <span style={{ fontWeight: 700, fontSize: 13, color: scoreColor(r.relevance_score), fontVariantNumeric: "tabular-nums" }}>
                        {r.relevance_score.toFixed(2)}
                      </span>
                    </div>
                  </div>

                  {/* Ref */}
                  <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 10 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 11, color: "var(--text-muted)" }}>
                      <ChevronRight size={11} />
                      Page {r.page}
                    </div>
                    <div style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 11, color: "var(--text-muted)" }}>
                      <Hash size={11} />
                      Para {r.paragraph_idx + 1}
                    </div>
                  </div>

                  {/* Snippet */}
                  <div style={{
                    fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.65,
                    background: "var(--bg-secondary)", borderRadius: 6, padding: "10px 12px",
                    fontFamily: "inherit",
                    borderLeft: "2px solid var(--accent)",
                  }}>
                    {highlight(r.snippet)}
                  </div>

                  {/* Matched terms */}
                  {r.matched_terms.length > 0 && (
                    <div style={{ marginTop: 10, display: "flex", flexWrap: "wrap", gap: 4 }}>
                      {r.matched_terms.map(t => (
                        <span key={t} className="badge badge-teal" style={{ fontSize: 9 }}>{t}</span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {!searched && !loading && (
        <div style={{ textAlign: "center", padding: "60px 0", color: "var(--text-muted)" }}>
          <Search size={40} style={{ marginBottom: 16, opacity: 0.3 }} />
          <div style={{ fontSize: 15 }}>Enter a query to search across all documents</div>
          <div style={{ fontSize: 12, marginTop: 6 }}>Results are ranked by TF-IDF relevance score</div>
        </div>
      )}

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
