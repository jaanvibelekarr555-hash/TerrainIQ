import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, type Benchmark, type Cell, type Config, type FrameData, type ResultImage, type Sequence } from "./api";
import Plot from "./components/Plot";
import Evidence from "./components/Evidence";
import {
  compareTraces, fmtRes, navTraces, rangeChart, resolutionTraces, terrainTraces, trendChart,
  TRAV_COLOR, TRAV_LABEL, vehicleAt, type ColorMode,
} from "./traces";
import type { Metrics } from "./api";

type View = "drive" | "resolution" | "compare" | "terrain";
const VIEWS: { id: View; label: string; hint: string }[] = [
  { id: "drive", label: "Driving map", hint: "Traversability + planned path" },
  { id: "resolution", label: "Adaptive resolution", hint: "Quadtree leaves by cell size" },
  { id: "compare", label: "Fixed vs adaptive", hint: "Same mapped points, uniform 5 cm vs adaptive" },
  { id: "terrain", label: "3D terrain", hint: "Real LiDAR point cloud" },
];
const SPEEDS = [0.5, 1, 2, 4];

function download(name: string, body: string, type: string) {
  const url = URL.createObjectURL(new Blob([body], { type }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function cellsCsv(fd: FrameData) {
  const head = "cell_x,cell_y,x_center,y_center,z_mean,points,traversability,slope_deg,roughness_m,complexity,priority,confidence,finest_resolution_m";
  const rows = fd.cells.map((c) =>
    [c.id[0], c.id[1], c.x, c.y, c.z, c.n, `"${c.trav}"`, c.slope, c.rough, c.complexity, c.priority, c.confidence, c.res]
      .map((v) => (v == null ? "" : v))
      .join(","),
  );
  return [head, ...rows].join("\n");
}
const STEP_MS = 60;
const FRAME_DURATION_MS = 4200;

const PIPELINE = [
  { k: "LiDAR input", d: "Real RELLIS-3D scan · XYZ + intensity" },
  { k: "Ground separation", d: "RANSAC plane · ground / non-ground" },
  { k: "Terrain analysis", d: "1 m cells · slope, roughness, complexity" },
  { k: "Traversability", d: "Drivable / non-drivable / sparse" },
  { k: "Adaptive resolution", d: "Distance + terrain-priority quadtree" },
  { k: "2.5D map", d: "Elevation summary per adaptive leaf" },
  { k: "Path planning", d: "A* over drivable cells, terrain-weighted cost" },
];

const n = (v: number) => v.toLocaleString();

export default function App() {
  const [status, setStatus] = useState<"connecting" | "online" | "offline">("connecting");
  const [cfg, setCfg] = useState<Config | null>(null);
  const [seqs, setSeqs] = useState<Sequence[]>([]);
  const [seq, setSeq] = useState<string>("");
  const [frameIdx, setFrameIdx] = useState(0);
  const [cache, setCache] = useState<Record<string, FrameData>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<View>("drive");
  const [playing, setPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [colorMode, setColorMode] = useState<ColorMode>("height");
  const [selected, setSelected] = useState<Cell | null>(null);
  const [bench, setBench] = useState<Benchmark | null>(null);
  const [images, setImages] = useState<ResultImage[]>([]);
  const [speed, setSpeed] = useState(1);
  const [summary, setSummary] = useState<Record<string, Metrics[]>>({});
  const [showKeys, setShowKeys] = useState(false);
  const viewerRef = useRef<HTMLDivElement>(null);

  // Bootstrap: health, config, sequences, evidence.
  useEffect(() => {
    (async () => {
      try {
        await api.health();
        const [c, s] = await Promise.all([api.config(), api.sequences()]);
        setCfg(c);
        setSeqs(s);
        setSeq(s[0]?.id ?? "");
        setStatus("online");
        api.benchmark().then(setBench).catch(() => {});
        api.results().then(setImages).catch(() => {});
      } catch (e) {
        setStatus("offline");
        setError(`Backend unreachable: ${(e as Error).message}`);
      }
    })();
  }, []);

  const frames = seqs.find((s) => s.id === seq)?.frames ?? [];
  const frameName = frames[frameIdx];
  const key = `${seq}/${frameName}`;
  const fd = cache[key];

  const inflight = useRef(new Set<string>());
  const load = useCallback(async (s: string, f: string) => {
    const k = `${s}/${f}`;
    if (inflight.current.has(k)) return;
    inflight.current.add(k);
    try {
      const d = await api.frame(s, f);
      setCache((c) => ({ ...c, [k]: d }));
      setError(null);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      inflight.current.delete(k);
    }
  }, []);

  useEffect(() => {
    if (!seq || !frameName || cache[key]) return;
    setLoading(true);
    load(seq, frameName).finally(() => setLoading(false));
  }, [seq, frameName, key, cache, load]);

  // Preload the next frame so replay stays smooth.
  useEffect(() => {
    const next = frames[frameIdx + 1];
    if (fd && next && !cache[`${seq}/${next}`]) load(seq, next);
  }, [fd, frames, frameIdx, seq, cache, load]);

  // Replay loop: vehicle advances along the path (wall-clock based, so render cost
  // doesn't slow it down), then steps to the next real frame once it is loaded.
  const progressRef = useRef(progress);
  progressRef.current = progress;
  useEffect(() => {
    if (!playing || !fd) return;
    let last = performance.now();
    const id = setInterval(() => {
      const now = performance.now();
      const np = progressRef.current + ((now - last) * speed) / FRAME_DURATION_MS;
      last = now;
      if (np < 1) {
        setProgress(np);
        return;
      }
      const nextIdx = frameIdx + 1 < frames.length ? frameIdx + 1 : 0;
      if (cache[`${seq}/${frames[nextIdx]}`]) {
        setFrameIdx(nextIdx);
        setProgress(0);
      } else {
        setProgress(1); // hold at the goal until the next frame arrives
      }
    }, STEP_MS);
    return () => clearInterval(id);
  }, [playing, fd, frameIdx, frames, seq, cache, speed]);

  useEffect(() => setSelected(null), [key]);

  // Whole-sequence metrics for the trend chart (also warms the server cache).
  useEffect(() => {
    if (!seq || summary[seq]) return;
    api.summary(seq).then((r) => setSummary((s) => ({ ...s, [seq]: r.frames }))).catch(() => {});
  }, [seq, summary]);

  const goFrame = useCallback(
    (i: number) => {
      if (!frames.length) return;
      setFrameIdx(((i % frames.length) + frames.length) % frames.length);
      setProgress(0);
    },
    [frames.length],
  );

  const toggleFullscreen = useCallback(() => {
    if (document.fullscreenElement) document.exitFullscreen();
    else viewerRef.current?.requestFullscreen?.();
  }, []);

  // Keyboard shortcuts.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === " ") { e.preventDefault(); setPlaying((p) => !p); }
      else if (e.key === "ArrowRight") goFrame(frameIdx + 1);
      else if (e.key === "ArrowLeft") goFrame(frameIdx - 1);
      else if (e.key >= "1" && e.key <= "4") setView(VIEWS[Number(e.key) - 1].id);
      else if (e.key.toLowerCase() === "f") toggleFullscreen();
      else if (e.key === "?") setShowKeys((s) => !s);
      else if (e.key === "Escape") { setShowKeys(false); setSelected(null); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [frameIdx, goFrame, toggleFullscreen]);

  const plot = useMemo(() => {
    if (!fd || !cfg) return null;
    if (view === "drive") return navTraces(fd, progress, selected);
    if (view === "resolution") return resolutionTraces(fd, cfg, progress);
    if (view === "compare") return compareTraces(fd, cfg);
    return terrainTraces(fd, progress, colorMode);
  }, [fd, cfg, view, progress, selected, colorMode]);

  const range = useMemo(() => (fd && cfg ? rangeChart(fd, cfg) : null), [fd, cfg]);
  const seqSummary = summary[seq];
  const trend = useMemo(() => (seqSummary ? trendChart(seqSummary, frameIdx) : null), [seqSummary, frameIdx]);

  const vehicle = fd ? vehicleAt(fd.path, progress) : null;
  const vehicleRes = useMemo(() => {
    if (!fd || !vehicle) return null;
    let best: Cell | null = null;
    let bd = Infinity;
    for (const c of fd.cells) {
      const d = (c.x - vehicle.x) ** 2 + (c.y - vehicle.y) ** 2;
      if (d < bd) [bd, best] = [d, c];
    }
    return best?.res ?? null;
  }, [fd, vehicle?.x, vehicle?.y]); // eslint-disable-line react-hooks/exhaustive-deps

  const onPlotClick = useCallback(
    (p: { customdata?: unknown }) => {
      if (view !== "drive" || !fd || typeof p.customdata !== "number") return;
      setSelected(fd.cells[p.customdata]);
    },
    [view, fd],
  );

  const m = fd?.metrics;
  const resTotal = m ? Object.values(m.resolution_counts).reduce((a, b) => a + b, 0) : 0;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <svg width="28" height="28" viewBox="0 0 32 32" aria-hidden>
            <rect width="32" height="32" rx="8" fill="#0f1a2b" />
            <circle cx="16" cy="16" r="3.5" fill="#22d3ee" />
            <circle cx="16" cy="16" r="8.5" fill="none" stroke="#22d3ee" strokeOpacity=".55" strokeWidth="1.6" />
            <path d="M16 3.5a12.5 12.5 0 0 1 12.5 12.5" fill="none" stroke="#22d3ee" strokeWidth="1.6" strokeLinecap="round" />
          </svg>
          <div>
            <strong>Adaptive LiDAR</strong>
            <span>Variable-resolution 2.5D terrain mapping</span>
          </div>
        </div>
        <nav className="topnav">
          <a href="#viewer">Replay</a>
          <a href="#evidence">Evidence</a>
        </nav>
        <div className={`status ${status}`}>
          <span className="pulse" />
          {status === "online" ? "API online" : status === "offline" ? "API offline" : "Connecting…"}
        </div>
      </header>

      <main className="container">
        <section className="hero">
          <div>
            <p className="eyebrow">RELLIS-3D · offline replay of real scans</p>
            <h1>Spend resolution where the terrain needs it.</h1>
            <p className="lede">
              Each LiDAR frame is split into ground and obstacles, analysed per 1 m cell for slope, roughness and
              complexity, then refined with a quadtree from 50 cm down to 5 cm. A terrain-weighted A* then plans a
              drivable route across the result.
            </p>
          </div>
          <div className="controls card">
            <label className="field">
              <span>Sequence</span>
              <div className="seg">
                {seqs.map((s) => (
                  <button
                    key={s.id}
                    className={s.id === seq ? "on" : ""}
                    onClick={() => {
                      setSeq(s.id);
                      setFrameIdx(0);
                      setProgress(0);
                    }}
                  >
                    {s.id}
                  </button>
                ))}
              </div>
            </label>
            <label className="field">
              <span>Frame</span>
              <div className="seg">
                {frames.map((f, i) => (
                  <button
                    key={f}
                    className={i === frameIdx ? "on" : ""}
                    onClick={() => {
                      setFrameIdx(i);
                      setProgress(0);
                    }}
                  >
                    {f.replace(/^0+(?=\d)/, "")}
                    {cache[`${seq}/${f}`] ? <i className="ready" title="Processed" /> : null}
                  </button>
                ))}
              </div>
            </label>
          </div>
        </section>

        {error && (
          <div className="alert" role="alert">
            <strong>Something went wrong.</strong> {error}
            {status === "offline" && (
              <span className="muted"> Start the API with <code>python -m uvicorn backend.app:app --port 8000</code>.</span>
            )}
          </div>
        )}

        <section className="kpis">
          <Kpi label="LiDAR points" value={m ? n(m.lidar_points) : "—"} sub={m ? `${n(m.ground_points)} ground (${((m.ground_points / m.lidar_points) * 100).toFixed(1)}%)` : ""} />
          <Kpi label="Adaptive cells" value={m ? n(m.adaptive_cells) : "—"} sub={m ? `from ${n(m.parent_cells)} terrain cells` : ""} />
          <Kpi
            label="Saved vs fixed 5 cm"
            value={m ? `${m.cells_saved_pct >= 0 ? "−" : "+"}${Math.abs(m.cells_saved_pct).toFixed(2)}%` : "—"}
            sub={m ? `${n(m.fixed_cells - m.adaptive_cells)} fewer cells than ${n(m.fixed_cells)}` : ""}
            accent
          />
          <Kpi label="Drivable cells" value={m ? n(m.drivable_cells) : "—"} sub={m ? `${m.non_drivable_cells} blocked · ${m.unknown_cells} sparse` : ""} />
          <Kpi label="Planned path" value={m ? (m.path_available ? `${m.path_length_m.toFixed(1)} m` : "None") : "—"} sub={m ? `${m.path_cells} cells · ${m.holes} depression${m.holes === 1 ? "" : "s"}` : ""} />
          <Kpi label="Pipeline time" value={m ? `${Math.round(m.processing_ms)} ms` : "—"} sub="server-side, this frame" />
        </section>

        <section className="workspace" id="viewer">
          <div className="card viewer" ref={viewerRef}>
            <div className="viewer-head">
              <div className="tabs" role="tablist">
                {VIEWS.map((v, i) => (
                  <button key={v.id} role="tab" aria-selected={view === v.id} className={view === v.id ? "on" : ""} onClick={() => setView(v.id)} title={`${v.hint} (${i + 1})`}>
                    {v.label}
                  </button>
                ))}
              </div>
              <div className="head-actions">
                <span className="muted small hint">{VIEWS.find((v) => v.id === view)?.hint}</span>
                <button className="icon-btn" disabled={!fd} onClick={() => fd && download(`lidar_${seq}_${frameName}.json`, JSON.stringify(fd), "application/json")} title="Download frame as JSON">
                  <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"><path d="M8 2v8m0 0L5 7m3 3 3-3M3 12.5h10" /></svg>
                  JSON
                </button>
                <button className="icon-btn" disabled={!fd} onClick={() => fd && download(`terrain_cells_${seq}_${frameName}.csv`, cellsCsv(fd), "text/csv")} title="Download terrain cells as CSV">
                  <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"><path d="M8 2v8m0 0L5 7m3 3 3-3M3 12.5h10" /></svg>
                  CSV
                </button>
                <button className="icon-btn square" onClick={toggleFullscreen} title="Fullscreen (F)" aria-label="Toggle fullscreen">
                  <svg width="15" height="15" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"><path d="M2.5 6V2.5H6M10 2.5h3.5V6M13.5 10v3.5H10M6 13.5H2.5V10" /></svg>
                </button>
              </div>
            </div>

            <div className="plot-area">
              {plot && <Plot data={plot.data} layout={plot.layout} onClick={onPlotClick} />}
              {(loading || !fd) && status !== "offline" && (
                <div className="loading">
                  <div className="spinner" />
                  <span>Processing frame {frameName ?? ""} through the pipeline…</span>
                </div>
              )}
              <div className="legend-float">
                {view === "drive" &&
                  (Object.keys(TRAV_COLOR) as (keyof typeof TRAV_COLOR)[]).map((t) => (
                    <span key={t} className="legend-item">
                      <span className="swatch" style={{ background: TRAV_COLOR[t] }} />
                      {TRAV_LABEL[t]}
                    </span>
                  ))}
                {view === "compare" && (
                  <span className="legend-item">
                    <span className="swatch" style={{ background: "#94a3b8" }} />
                    Fixed 5 cm
                  </span>
                )}
                {(view === "resolution" || view === "compare") &&
                  cfg?.resolutions.map((r) => (
                    <span key={r.value} className="legend-item">
                      <span className="swatch" style={{ background: r.color }} />
                      {r.label}
                    </span>
                  ))}
                {view === "terrain" && (
                  <div className="seg small">
                    {(["height", "intensity", "ground"] as ColorMode[]).map((c) => (
                      <button key={c} className={colorMode === c ? "on" : ""} onClick={() => setColorMode(c)}>
                        {c === "height" ? "Height" : c === "intensity" ? "Intensity" : "Ground"}
                      </button>
                    ))}
                  </div>
                )}
                {(view === "drive" || view === "resolution") && (
                  <>
                    <span className="legend-item">
                      <span className="swatch line" />
                      Path
                    </span>
                    {fd && fd.holes.length > 0 && (
                      <span className="legend-item">
                        <span className="swatch ring" />
                        Depression
                      </span>
                    )}
                  </>
                )}
              </div>
            </div>

            <div className="timeline">
              <button className="btn primary icon" onClick={() => setPlaying((p) => !p)} aria-label={playing ? "Pause" : "Play"} disabled={!fd}>
                {playing ? (
                  <svg width="16" height="16" viewBox="0 0 16 16"><rect x="3" y="2" width="3.5" height="12" rx="1" fill="currentColor" /><rect x="9.5" y="2" width="3.5" height="12" rx="1" fill="currentColor" /></svg>
                ) : (
                  <svg width="16" height="16" viewBox="0 0 16 16"><path d="M4 2.5v11l9.5-5.5z" fill="currentColor" /></svg>
                )}
              </button>
              <div className="scrub">
                <div className="scrub-track">
                  {frames.map((f, i) => (
                    <button
                      key={f}
                      className={`scrub-seg ${i < frameIdx ? "done" : ""} ${i === frameIdx ? "cur" : ""}`}
                      onClick={() => {
                        setFrameIdx(i);
                        setProgress(0);
                      }}
                      aria-label={`Frame ${f}`}
                    >
                      {i === frameIdx && <span className="scrub-fill" style={{ width: `${progress * 100}%` }} />}
                    </button>
                  ))}
                </div>
                <div className="scrub-meta mono">
                  <span>SEQ {seq} · FRAME {frameName ?? "—"}</span>
                  <span>
                    {vehicle ? `vehicle x ${vehicle.x.toFixed(1)} m · y ${vehicle.y.toFixed(1)} m · ${fmtRes(vehicleRes)}` : "no drivable path"}
                  </span>
                </div>
              </div>
              <div className="seg small speed" role="group" aria-label="Playback speed">
                {SPEEDS.map((s) => (
                  <button key={s} className={speed === s ? "on" : ""} onClick={() => setSpeed(s)}>
                    {s}×
                  </button>
                ))}
              </div>
              <button className="icon-btn square" onClick={() => setShowKeys((s) => !s)} title="Keyboard shortcuts (?)" aria-label="Keyboard shortcuts">
                ?
              </button>
            </div>
            {showKeys && (
              <div className="keys" role="dialog" aria-label="Keyboard shortcuts">
                {[
                  ["Space", "Play / pause"],
                  ["← →", "Previous / next frame"],
                  ["1 – 4", "Switch view"],
                  ["F", "Fullscreen viewer"],
                  ["Esc", "Clear selection"],
                ].map(([k, d]) => (
                  <div key={k}>
                    <kbd>{k}</kbd>
                    <span>{d}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          <aside className="side">
            <div className="card">
              <div className="card-head">
                <h3>Resolution mix</h3>
                <span className="muted small mono">{n(resTotal)} leaves</span>
              </div>
              <div className="stack-bar">
                {cfg?.resolutions.map((r) => {
                  const c = m?.resolution_counts[r.value.toFixed(3)] ?? 0;
                  return c ? <span key={r.value} style={{ width: `${(c / Math.max(1, resTotal)) * 100}%`, background: r.color }} title={`${r.label}: ${c}`} /> : null;
                })}
              </div>
              <ul className="res-list">
                {cfg?.resolutions.map((r) => {
                  const c = m?.resolution_counts[r.value.toFixed(3)] ?? 0;
                  return (
                    <li key={r.value}>
                      <span className="swatch" style={{ background: r.color }} />
                      <span>{r.label}</span>
                      <span className="mono">{n(c)}</span>
                      <span className="mono muted">{resTotal ? ((c / resTotal) * 100).toFixed(1) : "0.0"}%</span>
                    </li>
                  );
                })}
              </ul>
            </div>

            <div className="card">
              <div className="card-head">
                <h3>Cell inspector</h3>
                {selected && (
                  <button className="btn ghost small" onClick={() => setSelected(null)}>
                    Clear
                  </button>
                )}
              </div>
              {selected ? (
                <dl className="inspect">
                  <div><dt>Class</dt><dd><span className="pill" style={{ color: TRAV_COLOR[selected.trav], borderColor: TRAV_COLOR[selected.trav] + "66" }}>{TRAV_LABEL[selected.trav]}</span></dd></div>
                  <div><dt>Cell</dt><dd className="mono">({selected.id.join(", ")})</dd></div>
                  <div><dt>Centre</dt><dd className="mono">{selected.x.toFixed(1)}, {selected.y.toFixed(1)} m</dd></div>
                  <div><dt>Points</dt><dd className="mono">{n(selected.n)}</dd></div>
                  <div><dt>Mean Z</dt><dd className="mono">{selected.z.toFixed(2)} m</dd></div>
                  <div><dt>Slope</dt><dd className="mono">{selected.slope?.toFixed(2) ?? "—"}°</dd></div>
                  <div><dt>Roughness</dt><dd className="mono">{selected.rough != null ? `${(selected.rough * 100).toFixed(2)} cm` : "—"}</dd></div>
                  <Meter label="Complexity" v={selected.complexity} />
                  <Meter label="Priority" v={selected.priority} />
                  <Meter label="Confidence" v={selected.confidence} />
                  <div><dt>Finest leaf</dt><dd className="mono">{fmtRes(selected.res)}</dd></div>
                </dl>
              ) : (
                <p className="muted small empty">
                  {view === "drive" ? "Click any terrain cell on the driving map to inspect its geometry." : "Switch to the driving map and click a cell to inspect it."}
                </p>
              )}
            </div>

            <div className="card">
              <div className="card-head">
                <h3>Pipeline</h3>
              </div>
              <ol className="pipeline">
                {PIPELINE.map((p, i) => (
                  <li key={p.k} className={fd ? "done" : loading && i < 3 ? "active" : ""}>
                    <span className="step">{i + 1}</span>
                    <div>
                      <strong>{p.k}</strong>
                      <span>{p.d}</span>
                    </div>
                  </li>
                ))}
              </ol>
            </div>
          </aside>
        </section>

        <section className="analytics">
          <div className="card chart-card">
            <div className="card-head">
              <div>
                <h3>Resolution vs. range</h3>
                <p className="muted small sub">Leaf count per 2 m ring (bars, coloured by mean size) and mean cell size (line)</p>
              </div>
            </div>
            <div className="chart-box sm">{range ? <Plot data={range.data} layout={range.layout} /> : <div className="skeleton" />}</div>
          </div>
          <div className="card chart-card">
            <div className="card-head">
              <div>
                <h3>Sequence {seq} across frames</h3>
                <p className="muted small sub">Adaptive vs fixed 5 cm cells and planned path length · click a frame to jump</p>
              </div>
              {seqSummary && (
                <span className="pill good">
                  avg −{(seqSummary.reduce((a, f) => a + f.cells_saved_pct, 0) / seqSummary.length).toFixed(2)}%
                </span>
              )}
            </div>
            <div className="legend-row">
              <span className="legend-item"><span className="swatch line" style={{ background: "#94a3b8" }} />Fixed 5 cm cells</span>
              <span className="legend-item"><span className="swatch line" />Adaptive cells</span>
              <span className="legend-item"><span className="swatch line" style={{ background: "#a78bfa" }} />Path length</span>
            </div>
            <div className="chart-box sm">
              {trend ? (
                <Plot data={trend.data} layout={trend.layout} onClick={(p) => goFrame(frames.indexOf(String(p.x).padStart(6, "0")))} />
              ) : (
                <div className="skeleton">
                  <span>Processing all {frames.length} frames…</span>
                </div>
              )}
            </div>
          </div>
        </section>

        <Evidence bench={bench} images={images} />

        <footer className="footer muted small">
          Offline replay of real RELLIS-3D LiDAR scans with a terrain-derived drivable path. Not a live vehicle system and
          no trained hazard detector is used.
        </footer>
      </main>
    </div>
  );
}

function Kpi({ label, value, sub, accent }: { label: string; value: string; sub?: string; accent?: boolean }) {
  return (
    <div className={`card kpi ${accent ? "accent" : ""}`}>
      <span className="kpi-label">{label}</span>
      <span className="kpi-value">{value}</span>
      <span className="kpi-sub">{sub}</span>
    </div>
  );
}

function Meter({ label, v }: { label: string; v: number | null }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd className="meter-dd">
        <span className="meter"><span style={{ width: `${Math.max(0, Math.min(1, v ?? 0)) * 100}%` }} /></span>
        <span className="mono">{v != null ? v.toFixed(2) : "—"}</span>
      </dd>
    </div>
  );
}
