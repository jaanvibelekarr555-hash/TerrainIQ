import { useMemo, useState } from "react";
import type { Benchmark, ResultImage } from "../api";
import Plot, { axis, baseLayout } from "./Plot";

const MODE_COLORS = { A: "#64748b", B: "#38bdf8", C: "#22d3ee" } as const;
const MODES = ["A", "B", "C"] as const;

const METRICS = [
  { key: "cells", label: "Map cells", fmt: (v: number) => v.toLocaleString() },
  { key: "compact_bytes", label: "Compact map size", fmt: (v: number) => `${(v / 1024).toFixed(0)} KB` },
  { key: "peak_bytes", label: "Peak memory", fmt: (v: number) => `${(v / 1048576).toFixed(2)} MB` },
  { key: "build_s", label: "Build time", fmt: (v: number) => `${v.toFixed(2)} s` },
] as const;

export default function Evidence({ bench, images }: { bench: Benchmark | null; images: ResultImage[] }) {
  const [metric, setMetric] = useState<(typeof METRICS)[number]["key"]>("cells");
  const [lightbox, setLightbox] = useState<ResultImage | null>(null);

  const chart = useMemo(() => {
    if (!bench) return null;
    const seqs = Object.keys(bench.summary.sequences);
    const m = METRICS.find((x) => x.key === metric)!;
    const data = MODES.map((mode) => {
      const vals = seqs.map((s) => bench.summary.sequences[s][mode][metric]);
      return {
        type: "bar",
        name: bench.summary.labels[mode],
        x: seqs.map((s) => `Seq ${s}`),
        y: vals,
        text: vals.map((v) => m.fmt(v)),
        textposition: "none",
        hovertemplate: `<b>${bench.summary.labels[mode]}</b><br>%{x}: %{text}<extra></extra>`,
        marker: { color: MODE_COLORS[mode], line: { width: 0 } },
      };
    });
    return {
      data,
      layout: baseLayout({
        barmode: "group",
        bargap: 0.28,
        bargroupgap: 0.08,
        margin: { l: 56, r: 8, t: 8, b: 32 },
        xaxis: axis("", { showgrid: false }),
        yaxis: axis(m.label),
      }),
    };
  }, [bench, metric]);

  if (!bench) return null;
  const o = bench.summary.overall;

  return (
    <section className="evidence" id="evidence">
      <div className="section-head">
        <div>
          <p className="eyebrow">Measured evidence</p>
          <h2>Fixed 5 cm vs. adaptive resolution</h2>
          <p className="muted">
            Repeated-run benchmark ({o.A.runs} runs per mode) across real RELLIS-3D sequences. Negative deltas are
            savings versus the fixed 5 cm baseline.
          </p>
        </div>
      </div>

      <div className="delta-grid">
        {METRICS.map((m) => (
          <div className="card delta" key={m.key}>
            <span className="kpi-label">{m.label}</span>
            <div className="delta-rows">
              {MODES.map((mode) => {
                const s = o[mode];
                const d = (s as unknown as Record<string, number | undefined>)[
                  m.key === "compact_bytes" ? "compact_delta_pct" : m.key === "peak_bytes" ? "peak_delta_pct" : m.key === "build_s" ? "build_delta_pct" : "cells_delta_pct"
                ];
                return (
                  <div className="delta-row" key={mode}>
                    <span className="dot" style={{ background: MODE_COLORS[mode] }} />
                    <span className="delta-name">{bench.summary.labels[mode]}</span>
                    <span className="mono">{m.fmt(s[m.key])}</span>
                    <span className={`pill ${d == null ? "" : d <= 0 ? "good" : "bad"}`}>
                      {d == null ? "baseline" : `${d > 0 ? "+" : ""}${d.toFixed(2)}%`}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      <div className="card chart-card">
        <div className="card-head">
          <h3>Per-sequence comparison</h3>
          <div className="seg">
            {METRICS.map((m) => (
              <button key={m.key} className={metric === m.key ? "on" : ""} onClick={() => setMetric(m.key)}>
                {m.label}
              </button>
            ))}
          </div>
        </div>
        <div className="legend-row">
          {MODES.map((mode) => (
            <span key={mode} className="legend-item">
              <span className="swatch" style={{ background: MODE_COLORS[mode] }} />
              {bench.summary.labels[mode]}
            </span>
          ))}
        </div>
        <div className="chart-box">{chart && <Plot data={chart.data} layout={chart.layout} />}</div>
        <ul className="notes">
          {bench.summary.notes.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      </div>

      {images.length > 0 && (
        <>
          <div className="section-head">
            <div>
              <p className="eyebrow">Result artifacts</p>
              <h2>Pipeline output gallery</h2>
            </div>
          </div>
          <div className="gallery">
            {images.map((img) => (
              <button className="card thumb" key={img.file} onClick={() => setLightbox(img)}>
                <img src={img.url} alt={img.title} loading="lazy" />
                <span>{img.title}</span>
              </button>
            ))}
          </div>
        </>
      )}

      {lightbox && (
        <div className="lightbox" onClick={() => setLightbox(null)} role="dialog" aria-label={lightbox.title}>
          <figure onClick={(e) => e.stopPropagation()}>
            <img src={lightbox.url} alt={lightbox.title} />
            <figcaption>
              {lightbox.title}
              <button className="btn ghost" onClick={() => setLightbox(null)}>
                Close
              </button>
            </figcaption>
          </figure>
        </div>
      )}
    </section>
  );
}
