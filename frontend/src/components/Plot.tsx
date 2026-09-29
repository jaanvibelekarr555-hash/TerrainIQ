import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";

type ClickPoint = { x: number; y: number; customdata?: unknown };

interface Props {
  data: unknown[];
  layout: Record<string, unknown>;
  onClick?: (point: ClickPoint) => void;
}

export const PLOT_FONT = { family: "Inter, system-ui, sans-serif", color: "#94a3b8", size: 11 };

export const baseLayout = (extra: Record<string, unknown> = {}) => ({
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: PLOT_FONT,
  margin: { l: 48, r: 16, t: 12, b: 40 },
  showlegend: false,
  hoverlabel: {
    bgcolor: "#0f1a2b",
    bordercolor: "#1f2e45",
    font: { family: "Inter, sans-serif", color: "#e2e8f0", size: 12 },
  },
  ...extra,
});

export const axis = (title: string, extra: Record<string, unknown> = {}) => ({
  title: { text: title, font: { size: 11, color: "#64748b" } },
  gridcolor: "rgba(148,163,184,0.08)",
  zerolinecolor: "rgba(148,163,184,0.18)",
  linecolor: "rgba(148,163,184,0.18)",
  tickfont: { family: "JetBrains Mono, monospace", size: 10, color: "#64748b" },
  ...extra,
});

export default function Plot({ data, layout, onClick }: Props) {
  const ref = useRef<HTMLDivElement & { on?: Function }>(null);
  const bound = useRef(false);
  const clickRef = useRef(onClick);
  clickRef.current = onClick;

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    Promise.resolve(
      Plotly.react(el, data, { ...layout, autosize: true }, {
        displayModeBar: false,
        responsive: true,
        scrollZoom: true,
      }),
    ).then(() => {
      if (!bound.current && el.on) {
        bound.current = true;
        el.on("plotly_click", (ev: { points: ClickPoint[] }) => {
          const p = ev.points?.[0];
          if (p) clickRef.current?.(p);
        });
      }
    });
  }, [data, layout]);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => Plotly.Plots.resize(el));
    ro.observe(el);
    return () => {
      ro.disconnect();
      Plotly.purge(el);
    };
  }, []);

  return <div ref={ref} style={{ width: "100%", height: "100%" }} />;
}
