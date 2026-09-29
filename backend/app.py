"""FastAPI backend for the Adaptive LiDAR Resolution dashboard.

Wraps the verified pipeline in scripts/ (process_frame + drivable path planner)
and exposes it as JSON. Results are cached per (sequence, frame).
"""
from __future__ import annotations

import csv
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
for p in (PROJECT_ROOT, SCRIPTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(PROJECT_ROOT)  # pipeline uses relative DATA_ROOT

import smart_vehicle_real_environment_replay as replay  # noqa: E402
import final_adaptive_lidar_demo as demo  # noqa: E402

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FRONTEND_DIST = os.path.join(PROJECT_ROOT, "frontend", "dist")
RAW_POINT_LIMIT = 24000

app = FastAPI(title="Adaptive LiDAR Resolution API", version="1.0.0")
app.add_middleware(GZipMiddleware, minimum_size=2048)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# The pipeline reads the module-level replay.SEQUENCE, so frame processing is serialized.
_pipeline_lock = threading.Lock()
_cache: dict[tuple[str, str], dict] = {}
_prefetch = ThreadPoolExecutor(max_workers=1)


def _f(value, digits: int = 4):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return round(v, digits) if math.isfinite(v) else None


def list_sequences() -> list[dict]:
    root = replay.DATA_ROOT
    if not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root)):
        frames = demo.available_sequence_frames(name)
        if frames:
            out.append({"id": name, "frames": frames})
    return out


def _serialize(result: dict, path: list[dict], elapsed_ms: float) -> dict:
    data = result["data"]
    parent_res = demo.resolution_by_parent(result)

    cells = []
    for r in result["terrain_records"].values():
        key = (int(r["cell_id"][0]), int(r["cell_id"][1]))
        cells.append({
            "id": list(key),
            "x": _f(r["x_center"], 3),
            "y": _f(r["y_center"], 3),
            "z": _f(r["z_mean"], 3),
            "n": int(r["point_count"]),
            "trav": r["traversability"],
            "slope": _f(r["slope"], 3),
            "rough": _f(r["roughness"], 4),
            "complexity": _f(r["terrain_complexity"], 3),
            "priority": _f(r["terrain_priority"], 3),
            "confidence": _f(r["terrain_confidence"], 3),
            "res": _f(parent_res.get(key), 4),
        })

    z_all = data[:, 2]
    leaves = []
    leaf_ids = []
    for leaf in result["adaptive_leaves"]:
        b = leaf["bbox"]
        idx = np.asarray(leaf["point_indices"], dtype=np.int64)
        leaf_ids.append(idx)
        leaves.append([
            _f(b["x_min"], 3), _f(b["y_min"], 3),
            _f(leaf["cell_size"], 4),
            _f(z_all[idx].mean(), 3) if len(idx) else None,
            int(len(idx)),
        ])

    # Downsampled raw cloud (x, y, z, intensity, ground) for 3D view.
    n = len(data)
    step = max(1, n // RAW_POINT_LIMIT)
    sel = np.arange(0, n, step)
    ground = np.asarray(result["ground_mask"], dtype=bool)
    raw = np.column_stack([data[sel, :4].astype(np.float64), ground[sel].astype(np.float64)])
    raw = np.round(raw, 3)

    metrics = demo.frame_metrics(result, path)
    metrics["parent_cells"] = int(result["parent_cells"])
    metrics["processing_ms"] = round(elapsed_ms, 1)
    baseline = _fixed_baseline(data, leaf_ids, len(leaves))
    fixed_bins = baseline.pop("_fixed_bins", [])
    metrics.update(baseline)

    return {
        "frame": result["frame_name"],
        "metrics": metrics,
        "cells": cells,
        "leaves": leaves,
        "path": [[_f(p["x_center"], 3), _f(p["y_center"], 3), _f(p["z_mean"], 3)] for p in path],
        "holes": [
            {"x": _f(h["x_center"], 3), "y": _f(h["y_center"], 3), "depth": _f(h["hole_depth_m"], 3)}
            for h in result["hole_candidates"]
        ],
        "raw": {"columns": ["x", "y", "z", "intensity", "ground"], "points": raw.tolist(), "total": int(n)},
        "bounds": {"x": [replay.X_MIN, replay.X_MAX], "y": [replay.Y_MIN, replay.Y_MAX]},
        "distance_profile": _distance_profile(leaves),
        "fixed_grid": {"res": FIXED_BASELINE_RES, "bins": fixed_bins},
    }


FIXED_BASELINE_RES = 0.05
LEAF_RECORD_BYTES = 55  # compact bytes per cell, same model as results/ benchmarks


def _fixed_baseline(data: np.ndarray, leaf_ids: list[np.ndarray], adaptive_cells: int) -> dict:
    """Cells a uniform 5 cm grid would need to cover exactly the same mapped points."""
    if not leaf_ids:
        return {"fixed_cells": 0, "cells_saved_pct": 0.0}
    idx = np.concatenate(leaf_ids)
    bins = np.floor(data[idx, :2].astype(np.float64) / FIXED_BASELINE_RES + 1e-9).astype(np.int64)
    uniq = np.unique(bins, axis=0)
    fixed = int(len(uniq))
    saved = (1 - adaptive_cells / fixed) * 100 if fixed else 0.0
    return {
        "_fixed_bins": uniq.tolist(),
        "fixed_cells": fixed,
        "cells_saved_pct": round(saved, 2),
        "fixed_bytes": fixed * LEAF_RECORD_BYTES,
        "adaptive_bytes": adaptive_cells * LEAF_RECORD_BYTES,
    }


def _distance_profile(leaves: list[list]) -> list[dict]:
    """Leaf count and mean/min cell size per 2 m range bin (range from the sensor)."""
    bins: dict[int, list[float]] = {}
    for x0, y0, size, _z, _n in leaves:
        cx, cy = x0 + size / 2, y0 + size / 2
        b = int(math.hypot(cx, cy) // 2)
        bins.setdefault(b, []).append(size)
    return [
        {
            "range_m": b * 2 + 1,
            "count": len(v),
            "mean_cm": round(float(np.mean(v)) * 100, 2),
            "min_cm": round(float(np.min(v)) * 100, 2),
            "max_cm": round(float(np.max(v)) * 100, 2),
        }
        for b, v in sorted(bins.items())
    ]


def compute_frame(sequence: str, frame: str) -> dict:
    key = (sequence, frame)
    if key in _cache:
        return _cache[key]
    with _pipeline_lock:
        if key in _cache:
            return _cache[key]
        prev = replay.SEQUENCE
        replay.SEQUENCE = sequence
        try:
            t0 = time.perf_counter()
            result = replay.process_frame(frame)
            path = demo.plan_drivable_path(result["terrain_records"])
            elapsed = (time.perf_counter() - t0) * 1000
        finally:
            replay.SEQUENCE = prev
        payload = _serialize(result, path, elapsed)
        _cache[key] = payload
        return payload


def _prefetch_sequence(sequence: str, frames: list[str]) -> None:
    for fr in frames:
        try:
            compute_frame(sequence, fr)
        except Exception as exc:  # keep prefetching the rest
            print(f"[prefetch] {sequence}/{fr} failed: {exc}")


def _validate(sequence: str, frame: str | None = None) -> list[str]:
    seqs = {s["id"]: s["frames"] for s in list_sequences()}
    if sequence not in seqs:
        raise HTTPException(404, f"Unknown sequence {sequence}")
    if frame is not None and frame not in seqs[sequence]:
        raise HTTPException(404, f"Unknown frame {frame} in sequence {sequence}")
    return seqs[sequence]


@app.get("/api/health")
def health():
    return {"status": "ok", "sequences": len(list_sequences()), "cached_frames": len(_cache)}


@app.get("/api/config")
def config():
    return {
        "resolutions": [
            {"value": r, "label": replay.RESOLUTION_LABELS[r], "color": demo.RESOLUTION_COLORS[r]}
            for r in replay.RESOLUTIONS
        ],
        "traversability": [{"label": k, "color": v} for k, v in replay.TRAVERSABILITY_COLORS.items()],
        "distance_bands": [
            {"from": 0, "to": 10, "res": 0.10},
            {"from": 10, "to": 30, "res": 0.20},
            {"from": 30, "to": 50, "res": 0.35},
            {"from": 50, "to": None, "res": 0.50},
        ],
        "parent_cell_size": replay.PARENT_CELL_SIZE,
        "min_resolution": replay.MIN_RESOLUTION,
    }


@app.get("/api/sequences")
def sequences():
    return list_sequences()


@app.get("/api/sequences/{sequence}/frames/{frame}")
async def frame(sequence: str, frame: str):
    frames = _validate(sequence, frame)
    payload = await run_in_threadpool(compute_frame, sequence, frame)
    # Warm the rest of the sequence so scrubbing/replay is instant.
    _prefetch.submit(_prefetch_sequence, sequence, frames)
    return payload


@app.get("/api/sequences/{sequence}/status")
def sequence_status(sequence: str):
    frames = _validate(sequence)
    return {"sequence": sequence, "frames": frames, "ready": [f for f in frames if (sequence, f) in _cache]}


@app.get("/api/sequences/{sequence}/summary")
async def sequence_summary(sequence: str):
    """Per-frame metrics for the whole sequence (processes any uncached frames)."""
    frames = _validate(sequence)
    out = []
    for fr in frames:
        payload = await run_in_threadpool(compute_frame, sequence, fr)
        out.append(payload["metrics"])
    return {"sequence": sequence, "frames": out}


@lru_cache(maxsize=1)
def _read_csvs() -> dict:
    out = {}
    for name in ("memory_benchmark.csv", "road_benchmark.csv", "terrain_impact_repeated_runs.csv"):
        path = os.path.join(RESULTS_DIR, name)
        if os.path.isfile(path):
            with open(path, newline="", encoding="utf-8") as fh:
                out[name[:-4]] = list(csv.DictReader(fh))
    return out


@app.get("/api/benchmark")
def benchmark():
    return {"summary": demo.BENCHMARK_DATA, "tables": _read_csvs()}


@app.get("/api/results")
def results():
    if not os.path.isdir(RESULTS_DIR):
        return []
    items = []
    for name in sorted(os.listdir(RESULTS_DIR)):
        if name.lower().endswith(".png"):
            title = os.path.splitext(name)[0].replace("_", " ")
            items.append({"file": name, "url": f"/results/{name}", "title": title})
    return items


if os.path.isdir(RESULTS_DIR):
    app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")

if os.path.isdir(FRONTEND_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets")), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        candidate = os.path.join(FRONTEND_DIST, full_path)
        if full_path and os.path.isfile(candidate):
            return FileResponse(candidate)
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))
