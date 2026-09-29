import type { Cell, Config, FrameData, Metrics, Trav } from "./api";
import { axis, baseLayout } from "./components/Plot";

export const TRAV_COLOR: Record<Trav, string> = {
  DRIVABLE: "#22c55e",
  "NON-DRIVABLE": "#ef4444",
  "SPARSE / UNKNOWN": "#64748b",
};
export const TRAV_LABEL: Record<Trav, string> = {
  DRIVABLE: "Drivable",
  "NON-DRIVABLE": "Non-drivable",
  "SPARSE / UNKNOWN": "Sparse / unknown",
};
const PATH = "#22d3ee";

// Static (progress-independent) traces are cached per frame so replay ticks only
// rebuild the vehicle marker; Plotly.react then skips unchanged arrays.
const staticCache = new WeakMap<FrameData, Map<string, unknown[]>>();
function cached(fd: FrameData, key: string, build: () => unknown[]) {
  let m = staticCache.get(fd);
  if (!m) staticCache.set(fd, (m = new Map()));
  if (!m.has(key)) m.set(key, build());
  return m.get(key)!;
}
const HOLE = "#c084fc";

export const fmtRes = (r: number | null | undefined) =>
  r == null ? "—" : r * 100 >= 1 ? `${+(r * 100).toFixed(1)} cm` : `${r} m`;

/** Point on the path at progress 0..1, plus heading in degrees (map frame). */
export function vehicleAt(path: FrameData["path"], progress: number) {
  if (!path.length) return null;
  if (path.length === 1) return { x: path[0][0], y: path[0][1], z: path[0][2], heading: 0, idx: 0 };
  const pos = Math.max(0, Math.min(1, progress)) * (path.length - 1);
  const i = Math.min(Math.floor(pos), path.length - 2);
  const t = pos - i;
  const [a, b] = [path[i], path[i + 1]];
  return {
    x: a[0] + t * (b[0] - a[0]),
    y: a[1] + t * (b[1] - a[1]),
    z: a[2] + t * (b[2] - a[2]),
    heading: (Math.atan2(b[1] - a[1], b[0] - a[0]) * 180) / Math.PI,
    idx: Math.round(pos),
  };
}

// The map is drawn "driving-up": forward (x) is vertical, lateral (y) is horizontal
// with +y (vehicle left) on the left, so plot-x = y with a reversed axis.
function rectPolys(rects: { x0: number; y0: number; s: number }[]) {
  const xs: (number | null)[] = [];
  const ys: (number | null)[] = [];
  for (const { x0, y0, s } of rects) {
    xs.push(y0, y0 + s, y0 + s, y0, y0, null);
    ys.push(x0, x0, x0 + s, x0 + s, x0, null);
  }
  return { xs, ys };
}

function mapLayout(fd: FrameData, extra: Record<string, unknown> = {}) {
  return baseLayout({
    margin: { l: 52, r: 12, t: 8, b: 44 },
    dragmode: "pan",
    uirevision: "map",
    xaxis: axis("Lateral Y (m)", { range: [fd.bounds.y[1] + 0.5, fd.bounds.y[0] - 0.5], constrain: "domain" }),
    yaxis: axis("Forward X (m)", {
      range: [fd.bounds.x[0] - 1, fd.bounds.x[1] + 1],
      scaleanchor: "x",
      scaleratio: 1,
      constrain: "domain",
    }),
    ...extra,
  });
}

function vehicleTrace(fd: FrameData, progress: number) {
  const v = vehicleAt(fd.path, progress);
  if (!v) return [];
  return [{
    type: "scatter",
    mode: "markers",
    x: [v.y],
    y: [v.x],
    hovertemplate: "<b>Vehicle</b><br>X %{y:.2f} m · Y %{x:.2f} m<extra></extra>",
    marker: { symbol: "triangle-up", size: 20, color: "#f8fafc", angle: -v.heading, line: { width: 2, color: PATH } },
  }];
}

function pathTraces(fd: FrameData, progress: number) {
  return [...cached(fd, "path2d", () => pathStatic(fd)), ...vehicleTrace(fd, progress)];
}

function pathStatic(fd: FrameData) {
  const out: unknown[] = [];
  if (fd.path.length) {
    const px = fd.path.map((p) => p[1]);
    const py = fd.path.map((p) => p[0]);
    out.push(
      { type: "scatter", mode: "lines", x: px, y: py, hoverinfo: "skip", line: { color: PATH, width: 12 }, opacity: 0.16 },
      { type: "scatter", mode: "lines", x: px, y: py, hoverinfo: "skip", line: { color: PATH, width: 3, shape: "spline" } },
      {
        type: "scatter",
        mode: "markers",
        x: [px[0], px[px.length - 1]],
        y: [py[0], py[py.length - 1]],
        text: ["Start", "Goal"],
        hovertemplate: "<b>%{text}</b><br>X %{y:.1f} m · Y %{x:.1f} m<extra></extra>",
        marker: { size: 12, color: ["#f8fafc", "#a78bfa"], line: { width: 3, color: "#0b1220" } },
      },
    );
  }
  if (fd.holes.length) {
    out.push({
      type: "scatter",
      mode: "markers",
      x: fd.holes.map((h) => h.y),
      y: fd.holes.map((h) => h.x),
      customdata: fd.holes.map((h) => h.depth * 100),
      hovertemplate: "<b>Surface depression</b><br>Depth %{customdata:.1f} cm<extra></extra>",
      marker: { symbol: "circle-open", size: 22, color: HOLE, line: { width: 3 } },
    });
  }
  return out;
}

export function navTraces(fd: FrameData, progress: number, selected: Cell | null) {
  const traces = [...cached(fd, "nav", () => navStatic(fd))];
  if (selected) {
    const { xs, ys } = rectPolys([{ x0: selected.x - 0.5, y0: selected.y - 0.5, s: 1 }]);
    traces.push({ type: "scatter", mode: "lines", x: xs, y: ys, hoverinfo: "skip", line: { color: "#f8fafc", width: 2 } });
  }
  return { data: [...traces, ...pathTraces(fd, progress)], layout: cachedLayout(fd) };
}

const layoutCache = new WeakMap<FrameData, Record<string, unknown>>();
function cachedLayout(fd: FrameData) {
  if (!layoutCache.has(fd)) layoutCache.set(fd, mapLayout(fd));
  return layoutCache.get(fd)!;
}

function navStatic(fd: FrameData) {
  const traces: unknown[] = [];
  (Object.keys(TRAV_COLOR) as Trav[]).forEach((t) => {
    const cells = fd.cells.filter((c) => c.trav === t);
    const { xs, ys } = rectPolys(cells.map((c) => ({ x0: c.x - 0.46, y0: c.y - 0.46, s: 0.92 })));
    traces.push({
      type: "scatter",
      mode: "lines",
      x: xs,
      y: ys,
      fill: "toself",
      fillcolor: TRAV_COLOR[t] + (t === "DRIVABLE" ? "38" : "4d"),
      line: { width: 1, color: TRAV_COLOR[t] + "99" },
      hoverinfo: "skip",
    });
  });
  // Invisible hit targets for hover/click on each 1 m terrain cell.
  traces.push({
    type: "scatter",
    mode: "markers",
    x: fd.cells.map((c) => c.y),
    y: fd.cells.map((c) => c.x),
    customdata: fd.cells.map((_, i) => i),
    text: fd.cells.map(
      (c) =>
        `<b>${TRAV_LABEL[c.trav]}</b><br>Cell ${c.id.join(", ")} · ${c.n} pts<br>` +
        `Slope ${c.slope?.toFixed(1) ?? "—"}° · Rough ${c.rough != null ? (c.rough * 100).toFixed(1) + " cm" : "—"}<br>` +
        `Resolution ${fmtRes(c.res)}`,
    ),
    hovertemplate: "%{text}<extra></extra>",
    marker: { size: 14, symbol: "square", color: "rgba(0,0,0,0)" },
  });
  return traces;
}

export function resolutionTraces(fd: FrameData, cfg: Config, progress: number) {
  const traces = cached(fd, "res", () => resolutionStatic(fd, cfg));
  return { data: [...traces, ...pathTraces(fd, progress)], layout: cachedLayout(fd) };
}

function resolutionStatic(fd: FrameData, cfg: Config) {
  const traces: unknown[] = [];
  for (const r of cfg.resolutions) {
    const leaves = fd.leaves.filter((l) => Math.abs(l[2] - r.value) < 1e-6);
    if (!leaves.length) continue;
    const { xs, ys } = rectPolys(leaves.map((l) => ({ x0: l[0], y0: l[1], s: l[2] })));
    traces.push({
      type: "scatter",
      mode: "lines",
      x: xs,
      y: ys,
      fill: "toself",
      fillcolor: r.color + "8c",
      line: { width: r.value <= 0.05 ? 0.3 : 0.6, color: r.color },
      hoverinfo: "skip",
    });
    traces.push({
      type: "scattergl",
      mode: "markers",
      x: leaves.map((l) => l[1] + l[2] / 2),
      y: leaves.map((l) => l[0] + l[2] / 2),
      customdata: leaves.map((l) => [l[3], l[4]]),
      hovertemplate: `<b>${r.label} leaf</b><br>z̄ %{customdata[0]:.2f} m · %{customdata[1]} pts<extra></extra>`,
      marker: { size: 4, color: "rgba(0,0,0,0)" },
    });
  }
  return traces;
}

export type ColorMode = "height" | "intensity" | "ground";

const terrainLayoutCache = { value: null as Record<string, unknown> | null };

export function terrainTraces(fd: FrameData, progress: number, mode: ColorMode) {
  const data = [...cached(fd, "terrain-" + mode, () => terrainStatic(fd, mode))];
  const v = vehicleAt(fd.path, progress);
  if (v)
    data.push({
      type: "scatter3d",
      mode: "markers",
      x: [v.x],
      y: [v.y],
      z: [v.z + 0.35],
      marker: { size: 7, color: "#f8fafc", line: { color: PATH, width: 2 } },
      hoverinfo: "skip",
    });
  return { data, layout: (terrainLayoutCache.value ??= terrainLayout()) };
}

function percentile(values: number[], q: number) {
  const s = [...values].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.max(0, Math.floor(q * (s.length - 1))))];
}

function terrainStatic(fd: FrameData, mode: ColorMode) {
  const pts = fd.raw.points;
  const x: number[] = [], y: number[] = [], z: number[] = [], c: (number | string)[] = [];
  for (const p of pts) {
    if (p[0] < fd.bounds.x[0] - 5 || p[0] > fd.bounds.x[1] + 10 || Math.abs(p[1]) > 20) continue;
    x.push(p[0]);
    y.push(p[1]);
    z.push(p[2]);
    c.push(mode === "height" ? p[2] : mode === "intensity" ? p[3] : p[4] ? "#22c55e" : "#475569");
  }
  const marker: Record<string, unknown> = { size: 1.8, color: c, opacity: 0.9 };
  if (mode !== "ground") {
    marker.colorscale = mode === "height"
      ? [[0, "#1e3a8a"], [0.35, "#0ea5e9"], [0.6, "#22d3ee"], [0.8, "#facc15"], [1, "#f97316"]]
      : "Viridis";
    if (mode === "height" && z.length) {
      marker.cmin = percentile(z, 0.02);
      marker.cmax = percentile(z, 0.98);
    }
  }
  const data: unknown[] = [
    { type: "scatter3d", mode: "markers", x, y, z, marker, hoverinfo: "skip" },
  ];
  if (fd.path.length) {
    data.push({
      type: "scatter3d",
      mode: "lines",
      x: fd.path.map((p) => p[0]),
      y: fd.path.map((p) => p[1]),
      z: fd.path.map((p) => p[2] + 0.15),
      line: { color: PATH, width: 8 },
      hoverinfo: "skip",
    });
  }
  return data;
}

function terrainLayout() {
  const ax = (title: string) => ({
    title: { text: title, font: { size: 10, color: "#64748b" } },
    backgroundcolor: "rgba(0,0,0,0)",
    gridcolor: "rgba(148,163,184,0.12)",
    zerolinecolor: "rgba(148,163,184,0.2)",
    showspikes: false,
    tickfont: { size: 9, color: "#64748b" },
  });
  return baseLayout({
    margin: { l: 0, r: 0, t: 0, b: 0 },
    uirevision: "terrain",
    scene: {
      xaxis: ax("X forward (m)"),
      yaxis: ax("Y lateral (m)"),
      zaxis: ax("Z (m)"),
      aspectmode: "manual",
      aspectratio: { x: 2, y: 1.4, z: 0.35 },
      camera: { eye: { x: -1.35, y: -1.1, z: 0.75 }, center: { x: 0.05, y: 0, z: -0.15 } },
    },
  });
}

// ---------------------------------------------------------------------------
// Compare: uniform 5 cm grid vs adaptive leaves over the same mapped points.
// ---------------------------------------------------------------------------
export function compareTraces(fd: FrameData, cfg: Config) {
  const data = cached(fd, "compare", () => {
    const res = fd.fixed_grid.res;
    const fixed = rectPolys(fd.fixed_grid.bins.map(([ix, iy]) => ({ x0: ix * res, y0: iy * res, s: res })));
    const out: unknown[] = [
      {
        type: "scatter", mode: "lines", x: fixed.xs, y: fixed.ys, fill: "toself",
        fillcolor: "#94a3b88c", line: { width: 0.3, color: "#94a3b8" }, hoverinfo: "skip",
        xaxis: "x", yaxis: "y",
      },
    ];
    for (const t of resolutionStatic(fd, cfg) as Record<string, unknown>[]) {
      if (t.type === "scatter") out.push({ ...t, xaxis: "x2", yaxis: "y2" });
    }
    return out;
  });
  return { data, layout: compareLayout(fd) };
}

const compareLayoutCache = new WeakMap<FrameData, Record<string, unknown>>();
function compareLayout(fd: FrameData) {
  if (compareLayoutCache.has(fd)) return compareLayoutCache.get(fd)!;
  const m = fd.metrics;
  const yr = [fd.bounds.y[1] + 0.5, fd.bounds.y[0] - 0.5];
  const xr = [fd.bounds.x[0] - 1, fd.bounds.x[1] + 1];
  const title = (text: string, x: number) => ({
    text, x, y: 1, xref: "paper", yref: "paper", xanchor: "center", yanchor: "bottom", showarrow: false,
    font: { size: 12, color: "#cbd5e1", family: "Inter, sans-serif" },
  });
  const layout = baseLayout({
    margin: { l: 44, r: 12, t: 30, b: 40 },
    dragmode: "pan",
    uirevision: "compare",
    xaxis: axis("Lateral Y (m)", { domain: [0, 0.48], range: yr }),
    yaxis: axis("Forward X (m)", { range: xr, scaleanchor: "x", constrain: "domain" }),
    xaxis2: axis("Lateral Y (m)", { domain: [0.52, 1], range: yr, matches: "x" }),
    yaxis2: axis("", { anchor: "x2", range: xr, matches: "y", showticklabels: false }),
    annotations: [
      title(`Fixed 5 cm · ${m.fixed_cells.toLocaleString()} cells`, 0.24),
      title(`Adaptive · ${m.adaptive_cells.toLocaleString()} cells (${m.cells_saved_pct > 0 ? "−" : "+"}${Math.abs(m.cells_saved_pct).toFixed(2)}%)`, 0.76),
    ],
  });
  compareLayoutCache.set(fd, layout);
  return layout;
}

// ---------------------------------------------------------------------------
// Analytics charts.
// ---------------------------------------------------------------------------
export function rangeChart(fd: FrameData, cfg: Config) {
  const p = fd.distance_profile;
  const colorFor = (cm: number) => {
    let best = cfg.resolutions[0];
    for (const r of cfg.resolutions) if (Math.abs(r.value * 100 - cm) < Math.abs(best.value * 100 - cm)) best = r;
    return best.color;
  };
  return {
    data: [
      {
        type: "bar", x: p.map((b) => b.range_m), y: p.map((b) => b.count), width: 1.6,
        marker: { color: p.map((b) => colorFor(b.mean_cm)), opacity: 0.85 },
        customdata: p.map((b) => [b.mean_cm, b.min_cm, b.max_cm]),
        hovertemplate: "<b>%{x} m range</b><br>%{y} leaves<br>mean %{customdata[0]:.1f} cm (min %{customdata[1]}, max %{customdata[2]})<extra></extra>",
      },
      {
        type: "scatter", mode: "lines+markers", x: p.map((b) => b.range_m), y: p.map((b) => b.mean_cm), yaxis: "y2",
        line: { color: "#f8fafc", width: 2, shape: "spline" }, marker: { size: 5, color: "#f8fafc" }, hoverinfo: "skip",
      },
    ],
    layout: baseLayout({
      margin: { l: 44, r: 44, t: 8, b: 36 },
      bargap: 0.1,
      xaxis: axis("Range from sensor (m)", { showgrid: false }),
      yaxis: axis("Leaves"),
      yaxis2: axis("Mean cell (cm)", { overlaying: "y", side: "right", showgrid: false, rangemode: "tozero" }),
    }),
  };
}

export function trendChart(frames: Metrics[], current: number) {
  const x = frames.map((f) => Number(f.frame));
  const ys = (k: keyof Metrics) => frames.map((f) => f[k] as number);
  const cur = frames[current];
  const data: unknown[] = [
    { type: "scatter", mode: "lines", x, y: ys("fixed_cells"), name: "Fixed 5 cm cells", line: { color: "#94a3b8", width: 2, dash: "dot" }, hovertemplate: "Fixed 5 cm: %{y:,}<extra></extra>" },
    { type: "scatter", mode: "lines+markers", x, y: ys("adaptive_cells"), name: "Adaptive cells", line: { color: "#22d3ee", width: 2.5 }, marker: { size: 6 }, fill: "tonexty", fillcolor: "rgba(34,211,238,0.07)", hovertemplate: "Adaptive: %{y:,}<extra></extra>" },
    { type: "scatter", mode: "lines+markers", x, y: ys("path_length_m"), name: "Path length", yaxis: "y2", line: { color: "#a78bfa", width: 2 }, marker: { size: 5 }, hovertemplate: "Path: %{y:.1f} m<extra></extra>" },
  ];
  if (cur)
    data.push({ type: "scatter", mode: "markers", x: [Number(cur.frame)], y: [cur.adaptive_cells], marker: { size: 13, color: "#22d3ee", line: { color: "#f8fafc", width: 2 } }, hoverinfo: "skip" });
  return {
    data,
    layout: baseLayout({
      margin: { l: 52, r: 44, t: 8, b: 36 },
      hovermode: "x unified",
      xaxis: axis("Frame", { dtick: 1, showgrid: false }),
      yaxis: axis("Map cells"),
      yaxis2: axis("Path (m)", { overlaying: "y", side: "right", showgrid: false, rangemode: "tozero" }),
    }),
  };
}
