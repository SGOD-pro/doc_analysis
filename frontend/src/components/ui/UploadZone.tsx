import { useState, useCallback } from "react";
import { Upload, X, File, CheckCircle, AlertCircle, Loader2 } from "lucide-react";
import { uploadDocument, processDocument } from "../../services/api";

interface Props {
  onSuccess?: () => void;
}

const ACCEPTED = ".pdf,.docx,.txt,.md,.rst,.csv";

export default function UploadZone({ onSuccess }: Props) {
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<"idle" | "uploading" | "processing" | "done" | "error">("idle");
  const [message, setMessage] = useState("");
  const [result, setResult] = useState<Record<string, unknown> | null>(null);

  const handleFile = useCallback((f: File) => {
    setFile(f);
    setStatus("idle");
    setMessage("");
    setResult(null);
  }, []);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, [handleFile]);

  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) handleFile(f);
  };

  const run = async () => {
    if (!file) return;
    try {
      setStatus("uploading");
      setMessage("Uploading file...");
      const up = await uploadDocument(file);

      setStatus("processing");
      setMessage("Running NLP pipeline (this may take a minute)...");
      const proc = await processDocument(up.doc_id);

      setStatus("done");
      setMessage("Document processed successfully!");
      setResult(proc as unknown as Record<string, unknown>);
      onSuccess?.();
    } catch (err: unknown) {
      setStatus("error");
      const msg = err instanceof Error ? err.message : String(err);
      setMessage(`Error: ${msg}`);
    }
  };

  const reset = () => {
    setFile(null);
    setStatus("idle");
    setMessage("");
    setResult(null);
  };

  const borderColor = dragging ? "var(--accent)" :
    status === "done" ? "var(--green)" :
    status === "error" ? "var(--red)" : "var(--border-bright)";

  return (
    <div style={{ width: "100%" }}>
      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        style={{
          border: `2px dashed ${borderColor}`,
          borderRadius: 12,
          padding: "40px 24px",
          textAlign: "center",
          cursor: "pointer",
          background: dragging ? "var(--accent-dim)" : "var(--bg-secondary)",
          transition: "all 0.2s ease",
        }}
        onClick={() => document.getElementById("file-input")?.click()}
      >
        <input
          id="file-input"
          type="file"
          accept={ACCEPTED}
          style={{ display: "none" }}
          onChange={onInputChange}
        />

        {file ? (
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 12 }}>
            <File size={24} color="var(--accent)" />
            <div style={{ textAlign: "left" }}>
              <div style={{ fontWeight: 600, color: "var(--text-primary)", fontSize: 14 }}>{file.name}</div>
              <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
                {(file.size / 1024 / 1024).toFixed(2)} MB
              </div>
            </div>
            <button
              onClick={(e) => { e.stopPropagation(); reset(); }}
              style={{ marginLeft: 8, background: "none", border: "none", cursor: "pointer", color: "var(--text-muted)" }}
            >
              <X size={16} />
            </button>
          </div>
        ) : (
          <>
            <div style={{
              width: 56, height: 56,
              background: "var(--accent-dim)",
              borderRadius: "50%",
              display: "flex", alignItems: "center", justifyContent: "center",
              margin: "0 auto 16px",
            }}>
              <Upload size={24} color="var(--accent)" />
            </div>
            <div style={{ fontWeight: 600, color: "var(--text-primary)", marginBottom: 6 }}>
              Drop your document here
            </div>
            <div style={{ fontSize: 13, color: "var(--text-muted)" }}>
              or click to browse • PDF, DOCX, TXT, MD, RST, CSV
            </div>
          </>
        )}
      </div>

      {/* Status */}
      {message && (
        <div style={{
          marginTop: 12,
          padding: "10px 14px",
          borderRadius: 8,
          display: "flex",
          alignItems: "center",
          gap: 8,
          background: status === "done" ? "var(--green-dim)" :
                      status === "error" ? "var(--red-dim)" : "var(--accent-dim)",
          border: `1px solid ${status === "done" ? "rgba(52,208,88,0.3)" : status === "error" ? "rgba(255,90,110,0.3)" : "rgba(79,142,247,0.3)"}`,
          fontSize: 13,
          color: status === "done" ? "var(--green)" : status === "error" ? "var(--red)" : "var(--accent)",
        }}>
          {status === "uploading" || status === "processing"
            ? <Loader2 size={14} style={{ animation: "spin 1s linear infinite" }} />
            : status === "done" ? <CheckCircle size={14} />
            : status === "error" ? <AlertCircle size={14} /> : null}
          {message}
        </div>
      )}

      {/* Result summary */}
      {result && status === "done" && (
        <div style={{
          marginTop: 12,
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: 8,
        }}>
          {[
            { label: "Pages", value: result.total_pages as number },
            { label: "Entities", value: result.entities_found as number },
            { label: "Topics", value: result.topics_found as number },
          ].map(({ label, value }) => (
            <div key={label} style={{
              background: "var(--bg-card)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              padding: "10px 12px",
              textAlign: "center",
            }}>
              <div style={{ fontSize: 20, fontWeight: 700, color: "var(--accent)" }}>{value}</div>
              <div style={{ fontSize: 11, color: "var(--text-muted)" }}>{label}</div>
            </div>
          ))}
        </div>
      )}

      {/* Action button */}
      {file && status === "idle" && (
        <button
          onClick={run}
          className="btn-primary"
          style={{ marginTop: 14, width: "100%", justifyContent: "center", padding: "12px" }}
        >
          <Upload size={16} />
          Analyze Document
        </button>
      )}

      {(status === "done" || status === "error") && (
        <button
          onClick={reset}
          className="btn-secondary"
          style={{ marginTop: 10, width: "100%", justifyContent: "center" }}
        >
          Upload Another
        </button>
      )}

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}
