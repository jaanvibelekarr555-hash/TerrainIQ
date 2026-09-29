export type Trav = "DRIVABLE" | "NON-DRIVABLE" | "SPARSE / UNKNOWN";

export interface Cell {
  id: [number, number];
  x: number;
  y: number;
  z: number;
  n: number;
  trav: Trav;
  slope: number | null;
  rough: number | null;
  complexity: number | null;
  priority: number | null;
  confidence: number | null;
  res: number | null;
}

export interface Metrics {
  frame: string;
  lidar_points: number;
  ground_points: number;
  adaptive_cells: number;
  drivable_cells: number;
  non_drivable_cells: number;
  unknown_cells: number;
  path_cells: number;
  path_length_m: number;
  path_available: boolean;
  holes: number;
  vehicle_resolution: string;
  resolution_counts: Record<string, number>;
  parent_cells: number;
  processing_ms: number;
  fixed_cells: number;
  cells_saved_pct: number;
  fixed_bytes: number;
  adaptive_bytes: number;
}

export interface RangeBin {
  range_m: number;
  count: number;
  mean_cm: number;
  min_cm: number;
  max_cm: number;
}

export interface FrameData {
  frame: string;
  metrics: Metrics;
  cells: Cell[];
  /** [x_min, y_min, size, z_mean, point_count] */
  leaves: [number, number, number, number | null, number][];
  path: [number, number, number][];
  holes: { x: number; y: number; depth: number }[];
  raw: { columns: string[]; points: number[][]; total: number };
  bounds: { x: [number, number]; y: [number, number] };
  distance_profile: RangeBin[];
  fixed_grid: { res: number; bins: [number, number][] };
}

export interface Sequence {
  id: string;
  frames: string[];
}

export interface Config {
  resolutions: { value: number; label: string; color: string }[];
  traversability: { label: Trav; color: string }[];
  parent_cell_size: number;
  min_resolution: number;
}

export interface ModeStats {
  cells: number;
  compact_bytes: number;
  peak_bytes: number;
  build_s: number;
  runs: number;
  cells_delta_pct?: number;
  compact_delta_pct?: number;
  peak_delta_pct?: number;
  build_delta_pct?: number;
}

export interface Benchmark {
  summary: {
    labels: Record<"A" | "B" | "C", string>;
    sequences: Record<string, Record<"A" | "B" | "C", ModeStats>>;
    overall: Record<"A" | "B" | "C", ModeStats>;
    notes: string[];
  };
  tables: Record<string, Record<string, string>[]>;
}

export interface ResultImage {
  file: string;
  url: string;
  title: string;
}

async function get<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body.slice(0, 200)}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => get<{ status: string; sequences: number; cached_frames: number }>("/api/health"),
  config: () => get<Config>("/api/config"),
  sequences: () => get<Sequence[]>("/api/sequences"),
  frame: (seq: string, frame: string) => get<FrameData>(`/api/sequences/${seq}/frames/${frame}`),
  summary: (seq: string) => get<{ sequence: string; frames: Metrics[] }>(`/api/sequences/${seq}/summary`),
  benchmark: () => get<Benchmark>("/api/benchmark"),
  results: () => get<ResultImage[]>("/api/results"),
};
