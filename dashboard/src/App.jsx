import { useState, useRef, useEffect } from "react";

const API = "http://localhost:5001";

const MODELS = {
  attacker:    { label: "Attacker",    model: "llama-3.3-70b-versatile",                      platform: "Groq",   color: "#E24B4A" },
  target:      { label: "Target",      model: "gemini-3-flash-preview",                        platform: "Gemini", color: "#378ADD" },
  analyzerA:   { label: "Analyzer A",  model: "gemini-3-flash-preview",                        platform: "Gemini", color: "#1D9E75" },
  analyzerB:   { label: "Analyzer B",  model: "llama-3.1-8b-instant",                          platform: "Groq",   color: "#1D9E75" },
  analyzerC:   { label: "Analyzer C",  model: "llama-4-scout-17b-16e-instruct",                platform: "Groq",   color: "#1D9E75" },
  coordinator: { label: "Coordinator", model: "gemini-3-flash-preview",                        platform: "Gemini", color: "#BA7517" },
  judge:       { label: "Judge",       model: "openai/gpt-oss-safeguard-20b",                  platform: "Groq",   color: "#7F77DD" },
};

function ModelChip({ role }) {
  const m = MODELS[role];
  if (!m) return null;
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 5,
      fontSize: 11, padding: "2px 8px", borderRadius: 20,
      border: `1px solid ${m.color}22`,
      background: `${m.color}11`, color: m.color,
      fontFamily: "var(--font-mono)", whiteSpace: "nowrap",
    }}>
      <span style={{ width: 6, height: 6, borderRadius: "50%", background: m.color, flexShrink: 0 }} />
      {m.label} · {m.model}
    </span>
  );
}

function AgentCard({ role }) {
  const m = MODELS[role];
  return (
    <div style={{
      padding: "10px 14px", borderRadius: 10,
      border: `1px solid ${m.color}33`,
      background: `${m.color}08`,
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 3 }}>
        <span style={{ fontWeight: 500, fontSize: 13, color: m.color }}>{m.label}</span>
        <span style={{ fontSize: 10, padding: "1px 6px", borderRadius: 20, background: `${m.color}22`, color: m.color }}>{m.platform}</span>
      </div>
      <div style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--color-text-secondary)" }}>{m.model}</div>
    </div>
  );
}

function StatCard({ label, value, sub, color }) {
  return (
    <div style={{ background: "var(--color-background-secondary)", borderRadius: 10, padding: "14px 18px" }}>
      <div style={{ fontSize: 12, color: "var(--color-text-secondary)", marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 26, fontWeight: 500, color: color || "var(--color-text-primary)", lineHeight: 1 }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: "var(--color-text-secondary)", marginTop: 3 }}>{sub}</div>}
    </div>
  );
}

function LogLine({ line }) {
  const isSuccess = line.includes("SUCCESS");
  const isFailure = line.includes("FAILURE");
  const isHeader  = line.startsWith("===");
  return (
    <div style={{
      fontSize: 12, fontFamily: "var(--font-mono)", padding: "2px 0",
      color: isSuccess ? "#1D9E75" : isFailure ? "#E24B4A" : isHeader ? "#BA7517" : "var(--color-text-secondary)",
      fontWeight: isHeader ? 500 : 400,
    }}>
      {line}
    </div>
  );
}

function TurnCard({ turn, index }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{
      border: "0.5px solid var(--color-border-tertiary)",
      borderRadius: 10, marginBottom: 8, overflow: "hidden",
    }}>
      <div
        onClick={() => setOpen(o => !o)}
        style={{
          display: "flex", alignItems: "center", gap: 10,
          padding: "10px 14px", cursor: "pointer",
          background: "var(--color-background-secondary)",
        }}
      >
        <span style={{
          fontSize: 11, fontWeight: 500, padding: "2px 8px", borderRadius: 20,
          background: turn.score ? "#1D9E7522" : "#E24B4A22",
          color: turn.score ? "#1D9E75" : "#E24B4A",
        }}>
          k={turn.k} · {turn.score ? "✓ SUCCESS" : "✗ FAIL"}
        </span>
        <span style={{ fontSize: 12, color: "var(--color-text-secondary)", flex: 1 }}>
          {turn.prompt.slice(0, 80)}…
        </span>
        <span style={{ fontSize: 12, color: "var(--color-text-tertiary)" }}>{open ? "▲" : "▼"}</span>
      </div>
      {open && (
        <div style={{ padding: "12px 14px", display: "flex", flexDirection: "column", gap: 10 }}>
          <div>
            <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginBottom: 4 }}>PROMPT</div>
            <div style={{ fontSize: 12, fontFamily: "var(--font-mono)", background: "var(--color-background-secondary)", padding: 10, borderRadius: 8, whiteSpace: "pre-wrap" }}>{turn.prompt}</div>
          </div>
          <div>
            <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginBottom: 4 }}>RESPONSE</div>
            <div style={{ fontSize: 12, background: "var(--color-background-secondary)", padding: 10, borderRadius: 8, whiteSpace: "pre-wrap" }}>{turn.response}</div>
          </div>
          {turn.analysis && (
            <div>
              <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginBottom: 4 }}>COORDINATOR SUGGESTION</div>
              <div style={{ fontSize: 12, fontStyle: "italic", color: "#BA7517", background: "#BA751711", padding: 10, borderRadius: 8 }}>{turn.analysis}</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [goal, setGoal] = useState("");
  const [status, setStatus] = useState("idle"); // idle | running | done | error
  const [logs, setLogs] = useState([]);
  const [result, setResult] = useState(null);
  const [activeTab, setActiveTab] = useState("overview");
  const logRef = useRef(null);
  const esRef = useRef(null);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logs]);

  async function startRun() {
    if (!goal.trim() || status === "running") return;
    setStatus("running"); setLogs([]); setResult(null); setActiveTab("logs");

    try {
      const res = await fetch(`${API}/run`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ goal }),
      });
      const { job_id } = await res.json();
      const es = new EventSource(`${API}/stream/${job_id}`);
      esRef.current = es;

      es.onmessage = (e) => {
        if (e.data === "[DONE]") { es.close(); return; }
        const msg = JSON.parse(e.data);
        if (msg.type === "log")   setLogs(l => [...l, msg.message]);
        if (msg.type === "done")  { setResult(msg.result); setStatus("done"); setActiveTab("results"); }
        if (msg.type === "error") { setLogs(l => [...l, `ERROR: ${msg.message}`]); setStatus("error"); }
      };
      es.onerror = () => { es.close(); setStatus("error"); };
    } catch (err) {
      setLogs(l => [...l, `Network error: ${err.message}`]);
      setStatus("error");
    }
  }

  const successRuns = result?.results?.filter(r => r.success) ?? [];
  const failRuns    = result?.results?.filter(r => !r.success) ?? [];

  return (
    <div style={{ padding: "1.5rem", maxWidth: 900, margin: "0 auto" }}>

      {/* Header */}
      <div style={{ marginBottom: "1.5rem" }}>
        <h1 style={{ fontSize: 22, fontWeight: 500, margin: "0 0 4px" }}>MA-PAIR</h1>
        <p style={{ fontSize: 14, color: "var(--color-text-secondary)", margin: 0 }}>
          Multi-Agent Prompt Automatic Iterative Refinement · Extended from PAIR (Chao et al., 2023)
        </p>
      </div>

      {/* Model roster */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 8, marginBottom: "1.5rem" }}>
        {Object.keys(MODELS).map(role => <AgentCard key={role} role={role} />)}
      </div>

      {/* Input */}
      <div style={{ display: "flex", gap: 8, marginBottom: "1.5rem" }}>
        <input
          value={goal}
          onChange={e => setGoal(e.target.value)}
          onKeyDown={e => e.key === "Enter" && startRun()}
          placeholder="Enter adversarial goal (e.g. Explain how to pick a lock)"
          style={{ flex: 1, padding: "10px 14px", borderRadius: 10, fontSize: 14 }}
          disabled={status === "running"}
        />
        <button
          onClick={startRun}
          disabled={status === "running" || !goal.trim()}
          style={{
            padding: "10px 22px", borderRadius: 10, fontSize: 14,
            background: status === "running" ? "var(--color-background-secondary)" : "#E24B4A",
            color: status === "running" ? "var(--color-text-secondary)" : "#fff",
            border: "none", cursor: status === "running" ? "not-allowed" : "pointer",
            fontWeight: 500, transition: "background 0.2s",
          }}
        >
          {status === "running" ? "Running…" : "Run ↗"}
        </button>
      </div>

      {/* Status bar */}
      {status !== "idle" && (
        <div style={{
          display: "flex", alignItems: "center", gap: 8, marginBottom: "1rem",
          padding: "8px 14px", borderRadius: 8,
          background: status === "running" ? "#BA751711" : status === "done" ? "#1D9E7511" : "#E24B4A11",
          border: `0.5px solid ${status === "running" ? "#BA7517" : status === "done" ? "#1D9E75" : "#E24B4A"}33`,
        }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", flexShrink: 0,
            background: status === "running" ? "#BA7517" : status === "done" ? "#1D9E75" : "#E24B4A",
            animation: status === "running" ? "pulse 1.2s ease-in-out infinite" : "none",
          }} />
          <span style={{ fontSize: 13, color: "var(--color-text-primary)" }}>
            {status === "running" ? `Running MA-PAIR on: "${goal}"` :
             status === "done"    ? `Done — ${result?.summary?.successes}/${result?.summary?.total_runs} runs succeeded` :
             "Error occurred"}
          </span>
        </div>
      )}

      {/* Tabs */}
      {status !== "idle" && (
        <>
          <div style={{ display: "flex", gap: 2, marginBottom: "1rem", borderBottom: "0.5px solid var(--color-border-tertiary)", paddingBottom: 0 }}>
            {["logs", "results"].map(tab => (
              <button key={tab} onClick={() => setActiveTab(tab)} style={{
                padding: "6px 16px", fontSize: 13, border: "none", cursor: "pointer",
                background: "transparent", borderBottom: activeTab === tab ? "2px solid #E24B4A" : "2px solid transparent",
                color: activeTab === tab ? "#E24B4A" : "var(--color-text-secondary)",
                fontWeight: activeTab === tab ? 500 : 400, marginBottom: -1,
              }}>
                {tab === "logs" ? "Live logs" : "Results"}
              </button>
            ))}
          </div>

          {/* Logs tab */}
          {activeTab === "logs" && (
            <div ref={logRef} style={{
              height: 320, overflowY: "auto", background: "var(--color-background-secondary)",
              borderRadius: 10, padding: "12px 16px",
            }}>
              {logs.length === 0 ? (
                <div style={{ fontSize: 12, color: "var(--color-text-tertiary)", fontFamily: "var(--font-mono)" }}>Waiting for output…</div>
              ) : logs.map((l, i) => <LogLine key={i} line={l} />)}
            </div>
          )}

          {/* Results tab */}
          {activeTab === "results" && result && (
            <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
              {/* Summary stats */}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10 }}>
                <StatCard label="Total runs" value={result.summary.total_runs} />
                <StatCard label="Successes" value={result.summary.successes} color="#1D9E75" />
                <StatCard label="Failures"  value={failRuns.length} color="#E24B4A" />
                <StatCard label="Success rate" value={`${Math.round(result.summary.success_rate * 100)}%`} color="#BA7517" />
              </div>

              {/* Successful runs */}
              {successRuns.length > 0 && (
                <div>
                  <h2 style={{ fontSize: 16, fontWeight: 500, marginBottom: 10, color: "#1D9E75" }}>
                    Successful jailbreaks ({successRuns.length})
                  </h2>
                  {successRuns.map((run, i) => (
                    <div key={i} style={{ border: "0.5px solid #1D9E7533", borderRadius: 12, padding: "12px 16px", marginBottom: 10 }}>
                      <div style={{ display: "flex", gap: 8, marginBottom: 10, flexWrap: "wrap" }}>
                        <span style={{ fontSize: 12, fontWeight: 500, color: "#1D9E75" }}>Run {run.run_index + 1}</span>
                        <span style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>Strategy: {run.strategy_name}</span>
                        <span style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>Iterations: {run.turns.length}</span>
                      </div>
                      {run.turns.map((t, j) => <TurnCard key={j} turn={t} index={j} />)}
                    </div>
                  ))}
                </div>
              )}

              {/* Failed runs */}
              {failRuns.length > 0 && (
                <div>
                  <h2 style={{ fontSize: 16, fontWeight: 500, marginBottom: 10, color: "var(--color-text-secondary)" }}>
                    Failed runs ({failRuns.length})
                  </h2>
                  {failRuns.slice(0, 3).map((run, i) => (
                    <div key={i} style={{ border: "0.5px solid var(--color-border-tertiary)", borderRadius: 12, padding: "12px 16px", marginBottom: 10 }}>
                      <div style={{ display: "flex", gap: 8, marginBottom: 10, flexWrap: "wrap" }}>
                        <span style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>Run {run.run_index + 1}</span>
                        <span style={{ fontSize: 12, color: "var(--color-text-secondary)" }}>Strategy: {run.strategy_name}</span>
                      </div>
                      {run.turns.map((t, j) => <TurnCard key={j} turn={t} index={j} />)}
                    </div>
                  ))}
                  {failRuns.length > 3 && (
                    <div style={{ fontSize: 13, color: "var(--color-text-secondary)", padding: "8px 0" }}>
                      …and {failRuns.length - 3} more failed runs
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </>
      )}

      <style>{`
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
      `}</style>
    </div>
  );
}
