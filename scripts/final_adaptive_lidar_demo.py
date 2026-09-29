from __future__ import annotations

"""Final presentation dashboard for Adaptive Variable-Resolution 2.5D LiDAR.

Purpose:
- Use the verified real RELLIS-3D terrain pipeline as the data source.
- Add a terrain-derived drivable path and vehicle replay.
- Present the model through a clean, judge-friendly dashboard.
- Keep the underlying terrain/resolution implementation unchanged.

Current demo target: multi-sequence RELLIS-3D replay. Simulation sequences
are discovered from the real LiDAR files so frame numbering is never inferred.
"""

import heapq
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.utils import PlotlyJSONEncoder
import plotly.io as pio

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import smart_vehicle_real_environment_replay as replay_module

from smart_vehicle_real_environment_replay import (
    DATA_ROOT,
    FRAME_NAMES,
    RESOLUTIONS,
    RESOLUTION_LABELS,
    TRAVERSABILITY_COLORS,
    process_frame,
    frame_traces,
)

SEQUENCE = "00000"
DEMO_DIR = os.path.join(PROJECT_ROOT, "demo")
X_MIN, X_MAX = 0.0, 30.0
Y_MIN, Y_MAX = -10.0, 10.0
PATH_COLOR = "#22d3ee"
VEHICLE_COLOR = "#f8fafc"
GOAL_COLOR = "#a78bfa"

RESOLUTION_COLORS = {
    0.05: "#38bdf8",
    0.10: "#22c55e",
    0.125: "#facc15",
    0.25: "#fb923c",
    0.50: "#ef4444",
}


# ---------------------------------------------------------------------------
# Path planning: same verified terrain inputs, no change to terrain analysis.
# ---------------------------------------------------------------------------

def _cell_key(record: dict) -> tuple[int, int]:
    cell_id = record["cell_id"]
    return int(cell_id[0]), int(cell_id[1])


def _path_cost(record: dict) -> float:
    slope = record.get("slope")
    roughness = record.get("roughness")
    complexity = record.get("terrain_complexity")

    slope_n = 0.0 if not np.isfinite(slope) else min(float(slope) / 30.0, 1.0)
    rough_n = 0.0 if not np.isfinite(roughness) else min(float(roughness) / 0.20, 1.0)
    complexity_n = 0.0 if not np.isfinite(complexity) else float(complexity)
    return 1.0 + 0.70 * slope_n + 0.80 * rough_n + 0.90 * complexity_n


def _connected_components(drivable: dict[tuple[int, int], dict]) -> list[list[dict]]:
    neighbours = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )
    remaining = set(drivable)
    components = []

    while remaining:
        seed = remaining.pop()
        stack = [seed]
        component = [drivable[seed]]
        while stack:
            current = stack.pop()
            for dx, dy in neighbours:
                nxt = (current[0] + dx, current[1] + dy)
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
                    component.append(drivable[nxt])
        components.append(component)
    return components


def plan_drivable_path(records: dict) -> list[dict]:
    drivable = {
        _cell_key(r): r
        for r in records.values()
        if r["traversability"] == "DRIVABLE"
    }
    if len(drivable) < 2:
        return []

    components = [c for c in _connected_components(drivable) if len(c) >= 2]
    if not components:
        return []

    def component_score(component: list[dict]) -> tuple[float, float, int]:
        xs = [float(r["x_center"]) for r in component]
        span = max(xs) - min(xs)
        mean_y_abs = sum(abs(float(r["y_center"])) for r in component) / len(component)
        return span, -mean_y_abs, len(component)

    component = max(components, key=component_score)
    start_record = min(
        component,
        key=lambda r: (
            (float(r["x_center"]) - 1.0) ** 2 + float(r["y_center"]) ** 2,
            float(r["x_center"]),
        ),
    )
    goal_record = max(
        component,
        key=lambda r: (float(r["x_center"]), -abs(float(r["y_center"])))
    )

    start = _cell_key(start_record)
    goal = _cell_key(goal_record)
    if start == goal:
        return [start_record]

    def heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    neighbours = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )
    open_heap = [(heuristic(start, goal), 0.0, start)]
    came_from = {}
    g_score = {start: 0.0}

    while open_heap:
        _, current_cost, current = heapq.heappop(open_heap)
        if current_cost > g_score.get(current, float("inf")) + 1e-9:
            continue
        if current == goal:
            break

        for dx, dy in neighbours:
            nxt = (current[0] + dx, current[1] + dy)
            if nxt not in drivable:
                continue
            distance_cost = math.sqrt(2.0) if dx and dy else 1.0
            new_cost = current_cost + distance_cost * _path_cost(drivable[nxt])
            if new_cost < g_score.get(nxt, float("inf")):
                g_score[nxt] = new_cost
                came_from[nxt] = current
                heapq.heappush(open_heap, (new_cost + heuristic(nxt, goal), new_cost, nxt))

    if goal not in came_from:
        return []

    keys = [goal]
    current = goal
    while current != start:
        current = came_from[current]
        keys.append(current)
    keys.reverse()
    return [drivable[key] for key in keys]


def path_length(path: list[dict]) -> float:
    if len(path) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(path, path[1:]):
        total += math.hypot(
            float(b["x_center"]) - float(a["x_center"]),
            float(b["y_center"]) - float(a["y_center"]),
        )
    return total


def vehicle_at_path(path: list[dict], progress: float) -> tuple[float, float, float]:
    if not path:
        return 0.0, 0.0, 0.0
    if len(path) == 1:
        r = path[0]
        return float(r["x_center"]), float(r["y_center"]), float(r["z_mean"]) + 0.20

    progress = max(0.0, min(1.0, progress))
    position = progress * (len(path) - 1)
    i = min(int(position), len(path) - 2)
    t = position - i
    a, b = path[i], path[i + 1]
    return (
        float(a["x_center"] + t * (b["x_center"] - a["x_center"])),
        float(a["y_center"] + t * (b["y_center"] - a["y_center"])),
        float(a["z_mean"] + t * (b["z_mean"] - a["z_mean"])) + 0.20,
    )


# ---------------------------------------------------------------------------
# Derived presentation data.
# ---------------------------------------------------------------------------

def resolution_by_parent(result: dict) -> dict[tuple[int, int], float]:
    best = {}
    for leaf in result["adaptive_leaves"]:
        parent_id = leaf.get("_parent_cell_id")
        if parent_id is None:
            continue
        key = (int(parent_id[0]), int(parent_id[1]))
        resolution = round(float(leaf["cell_size"]), 6)
        if key not in best or resolution < best[key]:
            best[key] = resolution
    return best


def vehicle_resolution_label(result: dict, path: list[dict], progress: float) -> str:
    """Return the adaptive resolution at the vehicle's current parent cell."""
    if not path:
        return "N/A"
    progress = max(0.0, min(1.0, float(progress)))
    idx = min(int(progress * (len(path) - 1)), len(path) - 1)
    record = path[idx]
    cell_id = (int(record["cell_id"][0]), int(record["cell_id"][1]))
    resolution = resolution_by_parent(result).get(cell_id)
    if resolution is None:
        return "N/A"
    resolution = round(float(resolution), 6)
    return RESOLUTION_LABELS.get(resolution, f"{resolution * 100:.1f} cm")


def frame_metrics(result: dict, path: list[dict]) -> dict:
    counts = result["traversability_counts"]
    progress = 0.0
    # Per-frame replay position is aligned with the frame index in main().
    return {
        "frame": result["frame_name"],
        "lidar_points": int(len(result["data"])),
        "ground_points": int(result["ground_points"]),
        "adaptive_cells": int(len(result["adaptive_leaves"])),
        "drivable_cells": int(counts["DRIVABLE"]),
        "non_drivable_cells": int(counts["NON-DRIVABLE"]),
        "unknown_cells": int(counts["SPARSE / UNKNOWN"]),
        "path_cells": int(len(path)),
        "path_length_m": round(path_length(path), 2),
        "path_available": bool(path),
        "holes": int(len(result["hole_candidates"])),
        "vehicle_resolution": vehicle_resolution_label(result, path, 0.0),
        "resolution_counts": {
            f"{resolution:.3f}": int(count)
            for resolution, count in result["resolution_counts"].items()
        },
    }


# ---------------------------------------------------------------------------
# 2D navigation hero plot.
# ---------------------------------------------------------------------------

def resolution_overlay_traces(result: dict) -> list:
    """Subtle spatial overlay showing where each adaptive resolution is active.

    Each marker represents a 1 m terrain parent cell and is colored by the
    finest adaptive leaf resolution present in that parent cell. The overlay
    is intentionally low-opacity so traversability/path information remains
    dominant in the hero view.
    """
    parent_resolution = resolution_by_parent(result)
    grouped = defaultdict(lambda: {"x": [], "y": []})

    for record in result["terrain_records"].values():
        key = (int(record["cell_id"][0]), int(record["cell_id"][1]))
        resolution = parent_resolution.get(key)
        if resolution is None:
            continue
        resolution = round(float(resolution), 6)
        grouped[resolution]["x"].append(float(record["x_center"]))
        grouped[resolution]["y"].append(float(record["y_center"]))

    traces = []
    for resolution in RESOLUTIONS:
        resolution = round(float(resolution), 6)
        label = RESOLUTION_LABELS[resolution]
        traces.append(
            go.Scatter(
                x=grouped[resolution]["x"],
                y=grouped[resolution]["y"],
                mode="markers",
                marker={
                    "size": 24,
                    "symbol": "square",
                    "color": RESOLUTION_COLORS[resolution],
                    "opacity": 0.14,
                    "line": {"width": 1.4, "color": RESOLUTION_COLORS[resolution]},
                },
                name=f"RESOLUTION ZONE {label}",
                showlegend=False,
                hovertemplate=(
                    f"<b>Adaptive resolution zone</b><br>"
                    f"{label}<br>X: %{{x:.1f}} m<br>Y: %{{y:.1f}} m"
                    "<extra></extra>"
                ),
            )
        )
    return traces


def nav_frame_traces(result: dict, path: list[dict], progress: float) -> list:
    grouped = defaultdict(lambda: {"x": [], "y": [], "z": []})
    for record in result["terrain_records"].values():
        label = record["traversability"]
        if label not in TRAVERSABILITY_COLORS:
            label = "SPARSE / UNKNOWN"
        grouped[label]["x"].append(float(record["x_center"]))
        grouped[label]["y"].append(float(record["y_center"]))

    traces = resolution_overlay_traces(result)
    for label, symbol, size in (
        ("DRIVABLE", "square", 10),
        ("NON-DRIVABLE", "x", 10),
        ("SPARSE / UNKNOWN", "diamond", 9),
    ):
        traces.append(
            go.Scatter(
                x=grouped[label]["x"],
                y=grouped[label]["y"],
                mode="markers",
                marker={
                    "size": size,
                    "symbol": symbol,
                    "color": TRAVERSABILITY_COLORS[label],
                    "opacity": 0.48 if label == "SPARSE / UNKNOWN" else 0.70,
                    "line": {"width": 0.5, "color": "#0b1220"},
                },
                name=label,
                hovertemplate=f"<b>{label}</b><br>X: %{{x:.1f}} m<br>Y: %{{y:.1f}} m<extra></extra>",
            )
        )

    if path:
        px = [float(r["x_center"]) for r in path]
        py = [float(r["y_center"]) for r in path]
    else:
        px, py = [], []

    traces.append(
        go.Scatter(
            x=px,
            y=py,
            mode="lines+markers",
            line={"color": PATH_COLOR, "width": 5},
            marker={"size": 5, "color": PATH_COLOR},
            name="DRIVABLE PATH",
            hovertemplate="<b>Terrain-aware path</b><br>X: %{x:.1f} m<br>Y: %{y:.1f} m<extra></extra>",
        )
    )

    vx, vy, _ = vehicle_at_path(path, progress)
    traces.append(
        go.Scatter(
            x=[vx], y=[vy],
            mode="markers+text",
            marker={
                "size": 15,
                "symbol": "diamond",
                "color": VEHICLE_COLOR,
                "line": {"width": 2, "color": PATH_COLOR},
            },
            text=["🚗 VEHICLE"],
            textposition="top center",
            name="VEHICLE",
            hovertemplate="<b>Vehicle</b><br>X: %{x:.1f} m<br>Y: %{y:.1f} m<extra></extra>",
        )
    )

    if path:
        goal = path[-1]
        traces.append(
            go.Scatter(
                x=[float(goal["x_center"])],
                y=[float(goal["y_center"])],
                mode="markers+text",
                marker={"size": 12, "symbol": "diamond", "color": GOAL_COLOR},
                text=["GOAL"],
                textposition="top center",
                name="PATH GOAL",
                hovertemplate="<b>Path goal</b><br>X: %{x:.1f} m<br>Y: %{y:.1f} m<extra></extra>",
            )
        )
    else:
        traces.append(go.Scatter(x=[], y=[], mode="markers", name="PATH GOAL", showlegend=False))

    hx = [float(r["x_center"]) for r in result["hole_candidates"]]
    hy = [float(r["y_center"]) for r in result["hole_candidates"]]
    traces.append(
        go.Scatter(
            x=hx, y=hy,
            mode="markers",
            marker={"size": 10, "symbol": "diamond", "color": "#f59e0b"},
            name="DEPRESSION CANDIDATE",
            hovertemplate="<b>Surface depression / hole candidate</b><br>X: %{x:.1f} m<br>Y: %{y:.1f} m<extra></extra>",
        )
    )
    return traces


# ---------------------------------------------------------------------------
# 2D adaptive-resolution overview.
# ---------------------------------------------------------------------------

def resolution_frame_traces(result: dict, path: list[dict], progress: float) -> list:
    parent_resolution = resolution_by_parent(result)
    grouped = defaultdict(lambda: {"x": [], "y": []})

    for record in result["terrain_records"].values():
        key = (int(record["cell_id"][0]), int(record["cell_id"][1]))
        resolution = parent_resolution.get(key)
        if resolution is None:
            continue
        resolution = round(resolution, 6)
        grouped[resolution]["x"].append(float(record["x_center"]))
        grouped[resolution]["y"].append(float(record["y_center"]))

    traces = []
    for resolution in RESOLUTIONS:
        resolution = round(float(resolution), 6)
        label = RESOLUTION_LABELS[resolution]
        traces.append(
            go.Scatter(
                x=grouped[resolution]["x"],
                y=grouped[resolution]["y"],
                mode="markers",
                marker={
                    "size": 19,
                    "symbol": "square",
                    "color": RESOLUTION_COLORS[resolution],
                    "opacity": 0.82,
                    "line": {"width": 1, "color": "#111827"},
                },
                name=label,
                hovertemplate=f"<b>Adaptive resolution</b><br>{label}<br>X: %{{x:.1f}} m<br>Y: %{{y:.1f}} m<extra></extra>",
            )
        )

    if path:
        traces.append(
            go.Scatter(
                x=[float(r["x_center"]) for r in path],
                y=[float(r["y_center"]) for r in path],
                mode="lines",
                line={"color": PATH_COLOR, "width": 4},
                name="DRIVABLE PATH",
                hovertemplate="<b>Path</b><br>X: %{x:.1f} m<br>Y: %{y:.1f} m<extra></extra>",
            )
        )
    else:
        traces.append(go.Scatter(x=[], y=[], mode="lines", name="DRIVABLE PATH", showlegend=False))

    vx, vy, _ = vehicle_at_path(path, progress)
    traces.append(
        go.Scatter(
            x=[vx], y=[vy],
            mode="markers",
            marker={"size": 15, "symbol": "diamond", "color": VEHICLE_COLOR, "line": {"width": 2, "color": PATH_COLOR}},
            name="VEHICLE",
            hovertemplate="<b>Vehicle</b><br>X: %{x:.1f} m<br>Y: %{y:.1f} m<extra></extra>",
        )
    )
    return traces


# ---------------------------------------------------------------------------
# 3D evidence plot using the existing real-model traces.
# ---------------------------------------------------------------------------

def enriched_3d_traces(result: dict, path: list[dict], progress: float) -> list:
    traces = frame_traces(result)
    if len(traces) != 11:
        raise RuntimeError(f"Expected 11 base terrain traces, got {len(traces)}")
    vx, vy, vz = vehicle_at_path(path, progress)
    traces[10] = go.Scatter3d(
        x=[vx], y=[vy], z=[vz],
        mode="markers+text",
        marker={"size": 11, "symbol": "diamond", "color": VEHICLE_COLOR, "line": {"width": 2, "color": PATH_COLOR}},
        text=["VEHICLE"],
        textposition="top center",
        name="VEHICLE",
        legendgroup="VEHICLE",
        showlegend=True,
        hovertemplate="<b>Vehicle</b><br>X: %{x:.1f} m<br>Y: %{y:.1f} m<extra></extra>",
    )

    if path:
        traces.append(
            go.Scatter3d(
                x=[float(r["x_center"]) for r in path],
                y=[float(r["y_center"]) for r in path],
                z=[float(r["z_mean"]) + 0.12 for r in path],
                mode="lines+markers",
                line={"color": PATH_COLOR, "width": 7},
                marker={"size": 3, "color": PATH_COLOR},
                name="DRIVABLE PATH",
                legendgroup="PATH",
            )
        )
        goal = path[-1]
        traces.append(
            go.Scatter3d(
                x=[float(goal["x_center"])],
                y=[float(goal["y_center"])],
                z=[float(goal["z_mean"]) + 0.18],
                mode="markers+text",
                marker={"size": 8, "symbol": "diamond", "color": GOAL_COLOR},
                text=["GOAL"],
                textposition="top center",
                name="PATH GOAL",
                legendgroup="PATH",
            )
        )
    else:
        traces.extend([
            go.Scatter3d(x=[], y=[], z=[], name="DRIVABLE PATH", showlegend=False),
            go.Scatter3d(x=[], y=[], z=[], name="PATH GOAL", showlegend=False),
        ])

    return traces


# ---------------------------------------------------------------------------
# Plotly figure builders.
# ---------------------------------------------------------------------------

def figure_with_frames(initial_traces: list, frames: list, layout: dict) -> go.Figure:
    fig = go.Figure(data=initial_traces, frames=frames)
    fig.update_layout(**layout)
    return fig


def build_nav_figure(processed: list, paths: list) -> go.Figure:
    frames = []
    for idx, (result, path) in enumerate(zip(processed, paths)):
        progress = idx / max(1, len(processed) - 1)
        frames.append(go.Frame(
            name=result["frame_name"],
            data=nav_frame_traces(result, path, progress),
            traces=list(range(12)),
        ))
    initial = nav_frame_traces(processed[0], paths[0], 0.0)
    return figure_with_frames(
        initial,
        frames,
        {
            "paper_bgcolor": "#07111c",
            "plot_bgcolor": "#07111c",
            "font": {"color": "#dbe7ee", "size": 11},
            "margin": {"l": 55, "r": 25, "t": 45, "b": 45},
            "xaxis": {"title": "Forward X (m)", "range": [X_MIN, X_MAX], "gridcolor": "#1b2c3a", "zeroline": False},
            "yaxis": {"title": "Lateral Y (m)", "range": [Y_MIN, Y_MAX], "gridcolor": "#1b2c3a", "zeroline": False, "scaleanchor": "x", "scaleratio": 1},
            "hovermode": "closest",
            "showlegend": True,
            "legend": {"orientation": "h", "x": 0.0, "y": -0.13, "font": {"size": 10}},
            "title": {"text": "DRIVABLE SPACE + TERRAIN-AWARE PATH", "x": 0.015, "xanchor": "left", "font": {"size": 16}},
        },
    )


def build_resolution_figure(processed: list, paths: list) -> go.Figure:
    frames = []
    for idx, (result, path) in enumerate(zip(processed, paths)):
        progress = idx / max(1, len(processed) - 1)
        frames.append(go.Frame(
            name=result["frame_name"],
            data=resolution_frame_traces(result, path, progress),
            traces=list(range(7)),
        ))
    initial = resolution_frame_traces(processed[0], paths[0], 0.0)
    return figure_with_frames(
        initial,
        frames,
        {
            "paper_bgcolor": "#07111c",
            "plot_bgcolor": "#07111c",
            "font": {"color": "#dbe7ee", "size": 11},
            "margin": {"l": 55, "r": 25, "t": 45, "b": 50},
            "xaxis": {"title": "Forward X (m)", "range": [X_MIN, X_MAX], "gridcolor": "#1b2c3a", "zeroline": False},
            "yaxis": {"title": "Lateral Y (m)", "range": [Y_MIN, Y_MAX], "gridcolor": "#1b2c3a", "zeroline": False, "scaleanchor": "x", "scaleratio": 1},
            "hovermode": "closest",
            "legend": {"orientation": "h", "x": 0.0, "y": -0.13, "font": {"size": 10}},
            "title": {"text": "ADAPTIVE RESOLUTION MAP", "x": 0.015, "xanchor": "left", "font": {"size": 16}},
        },
    )


def build_3d_figure(processed: list, paths: list) -> go.Figure:
    frames = []
    for idx, (result, path) in enumerate(zip(processed, paths)):
        progress = idx / max(1, len(processed) - 1)
        traces = enriched_3d_traces(result, path, progress)
        frames.append(go.Frame(name=result["frame_name"], data=traces, traces=list(range(13))))

    initial = enriched_3d_traces(processed[0], paths[0], 0.0)

    fig = go.Figure(data=initial, frames=frames)
    # Default: terrain-resolution evidence + traversability + path + vehicle.
    # Trace layout: 0 raw, 1-3 traversability, 4-8 resolutions, 9 hole, 10 vehicle, 11 path, 12 goal.
    for i, trace in enumerate(fig.data):
        trace.visible = i in set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    fig.update_layout(
        paper_bgcolor="#07111c",
        font={"color": "#dbe7ee", "size": 10},
        margin={"l": 0, "r": 0, "t": 45, "b": 75},
        title={"text": "3D TERRAIN + ADAPTIVE 2.5D EVIDENCE", "x": 0.015, "xanchor": "left", "font": {"size": 16}},
        scene={
            "xaxis": {"title": "FORWARD X (m)", "range": [X_MIN, X_MAX], "showspikes": False, "gridcolor": "#263746", "zeroline": False},
            "yaxis": {"title": "LATERAL Y (m)", "range": [Y_MIN, Y_MAX], "showspikes": False, "gridcolor": "#263746", "zeroline": False},
            "zaxis": {"title": "ELEVATION Z (m)", "showspikes": False, "gridcolor": "#263746", "zeroline": False},
            "aspectmode": "manual",
            "aspectratio": {"x": 1.65, "y": 1.15, "z": 0.60},
            "camera": {
                "projection": {"type": "perspective"},
                "eye": {"x": -1.35, "y": -0.55, "z": 0.70},
                "center": {"x": 0.0, "y": 0.0, "z": 0.0},
                "up": {"x": 0.0, "y": 0.0, "z": 1.0},
            },
            "bgcolor": "#07111c",
        },
        legend={"orientation": "h", "x": 0.0, "y": -0.08, "font": {"size": 9}},
    )
    return fig


# ---------------------------------------------------------------------------
# Dashboard HTML. Plotly.js is embedded once; plots share one custom replay UI.
# ---------------------------------------------------------------------------

def available_sequence_frames(sequence: str) -> list[str]:
    """Return existing LiDAR frame IDs for one RELLIS-3D sequence."""
    lidar_dir = os.path.join(
        DATA_ROOT,
        sequence,
        "vel_cloud_node_kitti_bin",
    )
    if not os.path.isdir(lidar_dir):
        return []
    return sorted(
        entry[:-4]
        for entry in os.listdir(lidar_dir)
        if entry.lower().endswith(".bin")
    )


def build_sequence_bundle(sequence: str) -> dict:
    """Process every real LiDAR frame available for one sequence."""
    previous_sequence = replay_module.SEQUENCE
    replay_module.SEQUENCE = sequence
    try:
        frame_names = available_sequence_frames(sequence)
        if not frame_names:
            raise RuntimeError(
                f"No LiDAR .bin frames found for RELLIS-3D sequence {sequence}."
            )

        processed = []
        paths = []
        print()
        print(f"=== SEQUENCE {sequence} ({len(frame_names)} frames) ===")

        for frame_name in frame_names:
            print(f"Processing frame {frame_name}...")
            result = process_frame(frame_name)
            path = plan_drivable_path(result["terrain_records"])
            processed.append(result)
            paths.append(path)
            print(f"  LiDAR points     : {len(result['data'])}")
            print(f"  Ground points    : {result['ground_points']}")
            print(f"  Adaptive cells   : {len(result['adaptive_leaves'])}")
            print(f"  Drivable cells   : {result['traversability_counts']['DRIVABLE']}")
            print(f"  Path cells       : {len(path)}")
            print(f"  Path length (m)  : {path_length(path):.2f}")
            print(f"  Hole candidates  : {len(result['hole_candidates'])}")

        metrics = []
        workflow_data = []
        for index, (result, path) in enumerate(zip(processed, paths)):
            metric = frame_metrics(result, path)
            progress = index / max(1, len(processed) - 1)
            metric["vehicle_resolution"] = vehicle_resolution_label(result, path, progress)
            metrics.append(metric)

            sample = path[0] if path else next(iter(result["terrain_records"].values()))
            res_mix = metric["resolution_counts"]
            workflow_data.append({
                "frame": result["frame_name"],
                "lidar": [
                    {"k":"POINTS","v":f"{metric['lidar_points']:,}","s":"real RELLIS-3D input"},
                    {"k":"FRAME","v":result["frame_name"],"s":"selected replay frame"},
                    {"k":"SOURCE","v":"RELLIS-3D","s":"offline real scan"},
                    {"k":"INPUT","v":"XYZ + intensity","s":"LiDAR point record"},
                ],
                "ground": [
                    {"k":"GROUND POINTS","v":f"{metric['ground_points']:,}","s":"preserved real point IDs"},
                    {"k":"GROUND SHARE","v":f"{metric['ground_points']/max(1,metric['lidar_points'])*100:.1f}%","s":"of LiDAR points"},
                    {"k":"PARENT CELLS","v":f"{len(result['terrain_records']):,}","s":"terrain analysis cells"},
                    {"k":"STEP","v":"GROUND → TERRAIN","s":"next pipeline stage"},
                ],
                "terrain": [
                    {"k":"SLOPE","v":f"{float(sample.get('slope', float('nan'))):.2f}°" if np.isfinite(sample.get('slope', np.nan)) else "N/A","s":"sample path-entry terrain cell"},
                    {"k":"ROUGHNESS","v":f"{float(sample.get('roughness', float('nan'))):.3f} m" if np.isfinite(sample.get('roughness', np.nan)) else "N/A","s":"RMS to local plane"},
                    {"k":"COMPLEXITY","v":f"{float(sample.get('terrain_complexity', float('nan'))):.3f}" if np.isfinite(sample.get('terrain_complexity', np.nan)) else "N/A","s":"normalized terrain complexity"},
                    {"k":"PRIORITY","v":f"{float(sample.get('terrain_priority', float('nan'))):.3f}" if np.isfinite(sample.get('terrain_priority', np.nan)) else "N/A","s":"refinement / information priority"},
                ],
                "traversability": [
                    {"k":"DRIVABLE","v":f"{metric['drivable_cells']:,}","s":"cells available to path planner"},
                    {"k":"NON-DRIVABLE","v":f"{metric['non_drivable_cells']:,}","s":"terrain cells rejected"},
                    {"k":"UNKNOWN","v":f"{metric['unknown_cells']:,}","s":"sparse / insufficient geometry"},
                    {"k":"CONFIDENCE","v":f"{float(sample.get('terrain_confidence', float('nan'))):.3f}" if np.isfinite(sample.get('terrain_confidence', np.nan)) else "N/A","s":"sample cell geometry confidence"},
                ],
                "resolution": [
                    {"k":"5 cm","v":f"{res_mix['0.050']:,}","s":"finest level"},
                    {"k":"10 cm","v":f"{res_mix['0.100']:,}","s":"near-field base level"},
                    {"k":"12.5 cm","v":f"{res_mix['0.125']:,}","s":"terrain-refined 25 cm base"},
                    {"k":"25 / 50 cm","v":f"{res_mix['0.250'] + res_mix['0.500']:,}","s":"coarser representation"},
                ],
                "map": [
                    {"k":"ADAPTIVE CELLS","v":f"{metric['adaptive_cells']:,}","s":"2.5D adaptive leaves"},
                    {"k":"LEVELS","v":"5 / 10 / 12.5 / 25 / 50 cm","s":"observed resolution set"},
                    {"k":"MAP TYPE","v":"2.5D","s":"terrain surface representation"},
                    {"k":"FRAME","v":result["frame_name"],"s":"same real scan"},
                ],
                "path": [
                    {"k":"PATH STATUS","v":"AVAILABLE" if path else "NOT AVAILABLE","s":"terrain-derived drivable route"},
                    {"k":"PATH CELLS","v":f"{len(path):,}","s":"connected drivable cells"},
                    {"k":"PATH LENGTH","v":f"{path_length(path):.2f} m","s":"planned route length"},
                    {"k":"HOLE CANDIDATES","v":f"{metric['holes']:,}","s":"surface-depression candidates"},
                ],
                "replay": [
                    {"k":"FRAME","v":result["frame_name"],"s":"real LiDAR frame"},
                    {"k":"VEHICLE STATE","v":"MOVING" if path else "NO PATH","s":"presentation replay"},
                    {"k":"CURRENT RESOLUTION","v":metric["vehicle_resolution"],"s":"vehicle-cell adaptive resolution"},
                    {"k":"MODE","v":"OFFLINE REPLAY","s":"not a measured live vehicle system"},
                ],
            })

        nav_fig = build_nav_figure(processed, paths)
        resolution_fig = build_resolution_figure(processed, paths)
        terrain_fig = build_3d_figure(processed, paths)

        return {
            "sequence": sequence,
            "frames": frame_names,
            "metrics": metrics,
            "workflow": workflow_data,
            "nav_frames": [
                {"name": frame.name, "data": frame.data}
                for frame in nav_fig.frames
            ],
            "resolution_frames": [
                {"name": frame.name, "data": frame.data}
                for frame in resolution_fig.frames
            ],
            "terrain_frames": [
                {"name": frame.name, "data": frame.data}
                for frame in terrain_fig.frames
            ],
            "path_success": sum(bool(path) for path in paths),
            "_nav_fig": nav_fig,
            "_resolution_fig": resolution_fig,
            "_terrain_fig": terrain_fig,
        }
    finally:
        replay_module.SEQUENCE = previous_sequence


def build_dashboard(nav_fig: go.Figure, resolution_fig: go.Figure, terrain_fig: go.Figure, metrics: list[dict], workflow_data: list[dict], benchmark_data: dict, simulation_data: dict) -> str:
    # Do not rely on Plotly's internal frame/animation queue for replay.
    # Store each precomputed frame as plain data and render it with Plotly.react().
    nav_frames_json = json.dumps([
        {"name": frame.name, "data": frame.data}
        for frame in nav_fig.frames
    ], cls=PlotlyJSONEncoder, separators=(",", ":"))
    resolution_frames_json = json.dumps([
        {"name": frame.name, "data": frame.data}
        for frame in resolution_fig.frames
    ], cls=PlotlyJSONEncoder, separators=(",", ":"))
    terrain_frames_json = json.dumps([
        {"name": frame.name, "data": frame.data}
        for frame in terrain_fig.frames
    ], cls=PlotlyJSONEncoder, separators=(",", ":"))

    # Remove Plotly's native frames from the embedded figures. The dashboard
    # owns replay timing and therefore never enters Plotly's animation queue.
    nav_fig.frames = []
    resolution_fig.frames = []
    terrain_fig.frames = []

    nav_html = pio.to_html(nav_fig, full_html=False, include_plotlyjs=False, div_id="navPlotGraph", config={"displayModeBar": False, "responsive": True})
    resolution_html = pio.to_html(resolution_fig, full_html=False, include_plotlyjs=False, div_id="resolutionPlotGraph", config={"displayModeBar": False, "responsive": True})
    terrain_html = pio.to_html(terrain_fig, full_html=False, include_plotlyjs=False, div_id="terrainPlotGraph", config={"displayModeBar": False, "responsive": True})
    plotly_js = get_plotlyjs()

    metrics_json = json.dumps(metrics, separators=(",", ":"))
    workflow_json = json.dumps(workflow_data, separators=(",", ":"), allow_nan=False)
    simulation_json = json.dumps(simulation_data, separators=(",", ":"), cls=PlotlyJSONEncoder)
    benchmark_json = json.dumps(benchmark_data, separators=(",", ":"), allow_nan=False)
    resolution_labels = ["5 cm", "10 cm", "12.5 cm", "25 cm", "50 cm"]
    resolution_values = [0.05, 0.10, 0.125, 0.25, 0.50]
    resolution_colors_json = json.dumps([RESOLUTION_COLORS[r] for r in resolution_values])
    resolution_labels_json = json.dumps(resolution_labels)
    resolution_values_json = json.dumps(resolution_values)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Adaptive Variable-Resolution 2.5D LiDAR — Final Demo</title>
<style>
:root {{
  --bg:#030a11; --panel:#07131f; --panel2:#0a1825; --line:#173044;
  --text:#e7f1f5; --muted:#7f99a8; --cyan:#22d3ee; --green:#22c55e;
  --red:#ef4444; --amber:#f59e0b; --purple:#a78bfa;
}}
* {{ box-sizing:border-box; }}
html,body {{ margin:0; min-height:100%; background:var(--bg); color:var(--text); font-family:Inter,Segoe UI,Arial,sans-serif; }}
body {{ overflow:hidden; }}
.app {{ height:100vh; display:grid; grid-template-rows:68px 1fr; }}
.header {{ display:flex; align-items:center; justify-content:space-between; padding:0 22px; border-bottom:1px solid var(--line); background:#050d15; }}
.brand {{ display:flex; gap:12px; align-items:center; min-width:0; }}
.logo {{ width:34px; height:34px; border:1px solid #22506a; display:grid; place-items:center; color:var(--cyan); border-radius:9px; font-size:18px; }}
.title {{ font-size:17px; font-weight:700; letter-spacing:.02em; }}
.subtitle {{ margin-top:3px; color:var(--muted); font-size:10px; letter-spacing:.11em; }}
.statuses {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; justify-content:flex-end; }}
.badge {{ border:1px solid var(--line); background:#081723; color:#a9c3cf; border-radius:999px; padding:7px 10px; font-size:9px; letter-spacing:.08em; }}
.badge.live {{ color:#9cecc0; border-color:#1c5b42; }}
.main {{ min-height:0; padding:14px; display:grid; grid-template-columns:minmax(0,1fr) 286px; gap:14px; }}
.left {{ min-width:0; min-height:0; display:grid; grid-template-rows:auto auto minmax(0,1fr) auto; gap:10px; }}
.toolbar {{ display:flex; align-items:center; gap:7px; flex-wrap:nowrap; overflow:hidden; }}
.sequence-bar {{ display:flex; align-items:center; gap:8px; min-width:0; padding:7px 9px; border:1px solid var(--line); border-radius:9px; background:#06111a; }}
.sequence-caption {{ color:#78919f; font-size:9px; font-weight:800; letter-spacing:.10em; white-space:nowrap; }}
.sequence-buttons {{ display:flex; gap:6px; min-width:0; flex-wrap:wrap; }}
.sequence-btn {{ border:1px solid #173044; background:#081723; color:#a6bbc5; border-radius:7px; padding:7px 10px; font-size:10px; font-weight:800; cursor:pointer; white-space:nowrap; }}
.sequence-btn:hover {{ border-color:#28677d; background:#0a1d29; }}
.sequence-btn.active {{ color:#e9fbff; border-color:#28708a; background:#0c2532; }}
.sequence-status {{ margin-left:auto; color:#6f8794; font-size:9px; letter-spacing:.04em; white-space:nowrap; }}
.tab {{ border:1px solid var(--line); background:#07131f; color:#8fa8b5; border-radius:7px; padding:9px 10px; font-size:10.5px; letter-spacing:.04em; cursor:pointer; white-space:nowrap; }}
.tab.active {{ color:#e9fbff; border-color:#28708a; background:#0c2532; }}
.spacer {{ flex:1; }}
.action {{ border:1px solid #24566b; background:#0b202d; color:#dffaff; border-radius:7px; padding:9px 10px; font-size:10.5px; cursor:pointer; white-space:nowrap; }}
.action.primary {{ color:#071018; background:var(--cyan); border-color:var(--cyan); font-weight:700; }}
.stage {{ position:relative; min-height:0; border:1px solid var(--line); border-radius:10px; background:var(--panel); overflow:hidden; }}
.plot {{ position:absolute; inset:0; display:none; }}
.plot.active {{ display:block; }}
.flow-view {{ position:absolute; inset:0; display:none; padding:22px; overflow:auto; background:radial-gradient(circle at 18% 10%, rgba(34,211,238,.07), transparent 34%), #07131f; }}
.flow-view.active {{ display:block; }}

.evidence-view {{ position:absolute; inset:0; display:none; padding:14px 18px; overflow:hidden; background:radial-gradient(circle at 80% 10%, rgba(167,139,250,.08), transparent 34%), #07131f; }}
.evidence-view.active {{ display:block; }}
.evidence-head {{ display:flex; justify-content:space-between; align-items:flex-start; gap:14px; margin-bottom:9px; }}
.evidence-title {{ font-size:18px; font-weight:800; letter-spacing:.03em; }}
.evidence-sub {{ margin-top:5px; color:#9ab0bc; font-size:11px; letter-spacing:.01em; line-height:1.45; }}
.evidence-selector {{ display:flex; gap:7px; flex-wrap:wrap; margin-bottom:8px; }}
.evidence-seq {{ border:1px solid #173044; background:#081723; color:#a6bbc5; border-radius:7px; padding:8px 12px; font-size:10.5px; font-weight:700; cursor:pointer; }}
.evidence-seq.active {{ color:#f3f0ff; border-color:#6d57a5; background:#17152a; }}
.evidence-kpis {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:8px; margin-bottom:9px; }}
.evidence-kpi {{ border:1px solid #18384b; border-radius:9px; background:#06131d; padding:9px 11px; }}
.evidence-kpi .k {{ color:#8ea6b2; font-size:9px; letter-spacing:.09em; font-weight:600; }}
.evidence-kpi .v {{ margin-top:3px; font-size:18px; font-weight:800; }}
.evidence-kpi .s {{ margin-top:3px; color:#79919e; font-size:9px; }}
.evidence-table-wrap {{ overflow:auto; border:1px solid #18384b; border-radius:10px; background:#06131d; }}
.evidence-table {{ width:100%; border-collapse:collapse; font-size:12px; min-width:720px; }}
.evidence-table th, .evidence-table td {{ padding:11px 12px; border-bottom:1px solid #112839; text-align:left; }}
.evidence-table th {{ color:#b7cbd4; font-size:11px; letter-spacing:.06em; background:#071723; }}
.evidence-table td {{ color:#e1edf1; line-height:1.35; }}
.evidence-table td:first-child {{ color:#a2b6bf; width:30%; font-weight:600; }}
.evidence-table tr:last-child td {{ border-bottom:none; }}
.evidence-delta {{ color:#8bd5e3; font-size:10.5px; margin-left:7px; white-space:nowrap; font-weight:700; }}
.evidence-note {{ margin-top:8px; color:#93aab5; font-size:10px; line-height:1.45; }}
.evidence-scope {{ margin:6px 0 8px; display:flex; flex-wrap:wrap; gap:7px; }}
.evidence-scope-pill {{ border:1px solid #21495b; background:#081b25; color:#b5c8d1; border-radius:7px; padding:7px 9px; font-size:10px; line-height:1.3; }}
.evidence-scope-pill strong {{ color:#f0fbff; }}
.evidence-scope-pill .dim {{ color:#7f99a6; }}
@media(max-width:1100px) {{ .evidence-kpis {{ grid-template-columns:1fr; }} }}
.flow-head {{ display:flex; justify-content:space-between; align-items:flex-start; gap:14px; margin-bottom:16px; }}
.flow-title {{ font-size:18px; font-weight:800; letter-spacing:.02em; }}
.flow-sub {{ margin-top:4px; color:#78919f; font-size:9px; line-height:1.5; letter-spacing:.06em; }}
.flow-frame {{ border:1px solid #21495b; background:#0a1c27; border-radius:8px; padding:9px 11px; color:var(--cyan); font-size:9px; letter-spacing:.07em; white-space:nowrap; }}
.flow-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:9px; }}
.flow-step {{ border:1px solid #173044; background:#081723; border-radius:9px; padding:11px; text-align:left; cursor:pointer; color:#9db5c0; min-height:94px; transition:border-color .12s, background .12s, transform .12s; }}
.flow-step:hover {{ border-color:#28677d; background:#0a1d29; transform:translateY(-1px); }}
.flow-step.active {{ border-color:#2b849e; background:#0b2633; color:#e6fbff; box-shadow:inset 0 0 0 1px rgba(34,211,238,.10); }}
.flow-num {{ color:var(--cyan); font-size:8px; letter-spacing:.10em; }}
.flow-name {{ margin-top:6px; font-weight:800; font-size:11px; }}
.flow-mini {{ margin-top:6px; color:#6f8794; font-size:8px; line-height:1.45; }}
.flow-detail {{ margin-top:12px; border:1px solid #18384b; border-radius:10px; background:#06131d; padding:14px; }}
.flow-detail-head {{ display:flex; justify-content:space-between; align-items:center; gap:10px; }}
.flow-detail-title {{ font-size:12px; font-weight:800; letter-spacing:.06em; }}
.flow-detail-note {{ color:#617b89; font-size:8px; }}
.flow-values {{ margin-top:12px; display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px; }}
.flow-value {{ border:1px solid #142d3d; background:#081723; border-radius:8px; padding:9px; }}
.flow-value .k {{ color:#708997; font-size:7px; letter-spacing:.09em; }}
.flow-value .v {{ margin-top:4px; font-size:15px; font-weight:800; }}
.flow-value .s {{ margin-top:3px; color:#5e7684; font-size:7px; line-height:1.35; }}
.flow-chain {{ margin-top:12px; padding:10px 12px; border-top:1px solid #132c3b; color:#718b98; font-size:8px; line-height:1.6; }}
@media(max-width:1100px) {{ .flow-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .flow-values {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
.presentation-overlay {{ position:absolute; inset:0; display:none; align-items:center; justify-content:center; padding:22px; background:rgba(2,8,15,.58); backdrop-filter:blur(3px); z-index:30; }}
.presentation-overlay.open {{ display:flex; }}
.presentation-card {{ width:min(760px,92%); border:1px solid #285b72; border-radius:14px; background:linear-gradient(180deg,#0a1d29,#07131f); box-shadow:0 24px 80px rgba(0,0,0,.48); padding:20px; }}
.presentation-top {{ display:flex; justify-content:space-between; gap:16px; align-items:flex-start; }}
.presentation-kicker {{ color:var(--cyan); font-size:9px; font-weight:800; letter-spacing:.14em; }}
.presentation-title {{ margin-top:6px; font-size:22px; font-weight:900; letter-spacing:.03em; }}
.presentation-copy {{ margin-top:7px; color:#9cb4c0; font-size:11px; line-height:1.55; max-width:620px; }}
.presentation-close {{ border:1px solid #24495d; background:#081723; color:#9eb7c2; border-radius:7px; width:32px; height:32px; cursor:pointer; }}
.presentation-steps {{ display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:7px; margin-top:18px; }}
.presentation-step {{ border:1px solid #173044; background:#081723; border-radius:8px; padding:9px 7px; cursor:pointer; min-height:72px; text-align:left; }}
.presentation-step.active {{ border-color:#2b849e; background:#0b2633; box-shadow:inset 0 0 0 1px rgba(34,211,238,.10); }}
.presentation-step-num {{ color:var(--cyan); font-size:8px; font-weight:800; }}
.presentation-step-name {{ margin-top:5px; color:#dcecf1; font-size:9px; font-weight:800; line-height:1.25; }}
.presentation-step-sub {{ margin-top:4px; color:#6f8794; font-size:7px; line-height:1.35; }}
.presentation-footer {{ margin-top:18px; display:flex; justify-content:space-between; align-items:center; gap:12px; color:#708996; font-size:9px; }}
.presentation-actions {{ display:flex; gap:7px; }}
@media(max-width:1100px) {{ .presentation-steps {{ grid-template-columns:repeat(3,minmax(0,1fr)); }} }}
.hud {{ position:absolute; left:14px; top:13px; display:flex; gap:7px; flex-wrap:wrap; pointer-events:none; z-index:5; }}
.hud span {{ background:rgba(4,13,21,.88); border:1px solid #183447; border-radius:6px; padding:6px 8px; color:#9fb6c2; font-size:9px; letter-spacing:.05em; }}
.hud .accent {{ color:var(--cyan); border-color:#18586e; }}
.resolution-hud {{ position:absolute; right:14px; top:13px; width:182px; padding:10px 11px; border:1px solid #1a4659; border-radius:9px; background:rgba(4,13,21,.90); z-index:5; pointer-events:none; box-shadow:0 10px 30px rgba(0,0,0,.20); }}
.rh-title {{ color:#8fa8b5; font-size:8px; letter-spacing:.12em; }}
.rh-value {{ margin-top:4px; color:var(--cyan); font-size:19px; font-weight:800; }}
.rh-sub {{ margin-top:2px; color:#667f8d; font-size:7px; letter-spacing:.08em; }}
.rh-pills {{ display:flex; gap:4px; flex-wrap:wrap; margin-top:8px; }}
.rh-pills span {{ padding:4px 5px; border:1px solid #173044; border-radius:5px; color:#6f8794; font-size:7px; background:#07131f; }}
.rh-pills span.active {{ color:#eafcff; border-color:#28708a; background:#0c2532; }}
.bottom {{ display:grid; grid-template-columns:1fr auto; gap:10px; align-items:center; padding:8px 10px; border:1px solid var(--line); border-radius:9px; background:#06111a; }}
.timeline {{ display:grid; grid-template-columns:auto 1fr auto; align-items:center; gap:10px; min-width:0; }}
.timeline strong {{ font-size:10px; white-space:nowrap; }}
.timeline input {{ width:100%; accent-color:var(--cyan); }}
.frame-labels {{ display:flex; justify-content:space-between; color:#69818e; font-size:8px; margin-top:3px; }}
.side {{ min-height:0; display:flex; flex-direction:column; gap:10px; }}
.card {{ border:1px solid var(--line); border-radius:10px; background:var(--panel); padding:12px; }}
.card h3 {{ margin:0 0 9px; font-size:10px; letter-spacing:.10em; color:#9eb7c5; }}
.kpis {{ display:grid; grid-template-columns:1fr 1fr; gap:7px; }}
.kpi {{ border:1px solid #142b3b; background:#081723; border-radius:8px; padding:9px; }}
.kpi .label {{ color:#708997; font-size:8px; letter-spacing:.08em; }}
.kpi .value {{ margin-top:3px; font-size:17px; font-weight:700; }}
.kpi .small {{ margin-top:3px; color:#607987; font-size:8px; }}
.path-status {{ display:flex; align-items:center; justify-content:space-between; gap:8px; padding:9px; border-radius:8px; border:1px solid #1b4658; background:#071923; }}
.path-dot {{ width:9px; height:9px; border-radius:50%; background:var(--cyan); box-shadow:0 0 0 3px rgba(34,211,238,.10); }}
.path-copy {{ flex:1; }}
.path-copy b {{ display:block; font-size:10px; }}
.path-copy span {{ display:block; color:#76919e; margin-top:2px; font-size:8px; }}
.res-list {{ display:grid; gap:6px; }}
.res-row {{ display:grid; grid-template-columns:50px 1fr 38px; align-items:center; gap:7px; font-size:8px; }}
.res-bar {{ height:7px; border-radius:99px; background:#0c2230; overflow:hidden; }}
.res-fill {{ height:100%; border-radius:99px; }}
.res-count {{ text-align:right; color:#9eb5c0; }}
.notes {{ color:#6c8591; font-size:8px; line-height:1.45; }}
.legend-title {{ color:#879eaa; font-size:8px; letter-spacing:.08em; margin-bottom:7px; }}
.legend-row {{ display:flex; align-items:center; gap:7px; margin:6px 0; font-size:8px; color:#9cb1bc; }}
.sw {{ width:10px; height:10px; border-radius:2px; border:1px solid #173044; }}
@media(max-width:980px) {{
  body {{ overflow:auto; }}
  .app {{ height:auto; min-height:100vh; }}
  .main {{ grid-template-columns:1fr; }}
  .stage {{ height:620px; }}
  .side {{ display:grid; grid-template-columns:1fr 1fr; }}
}}
</style>
<script>{plotly_js}</script>
</head>
<body>
<div class="app">
<header class="header">
  <div class="brand">
    <div class="logo">◈</div>
    <div>
      <div class="title">ADAPTIVE VARIABLE-RESOLUTION 2.5D LiDAR</div>
      <div class="subtitle">REAL RELLIS-3D • TERRAIN INTELLIGENCE • TERRAIN-AWARE VEHICLE SIMULATION</div>
    </div>
  </div>
  <div class="statuses">
    <div class="badge live">● REAL DATA</div>
    <div class="badge">SEQ <span id="headerSequence">{SEQUENCE}</span></div>
    <div class="badge">5 / 10 / 12.5 / 25 / 50 cm</div>
  </div>
</header>

<main class="main">
  <section class="left">
    <div class="toolbar">
      <button class="tab active" data-view="driving">01 · DRIVING</button>
      <button class="tab" data-view="resolution">02 · ADAPTIVE RESOLUTION</button>
      <button class="tab" data-view="terrain">03 · 3D TERRAIN</button>
      <button class="tab" data-view="raw">04 · RAW LiDAR</button>
      <button class="tab" data-view="diagnostic">05 · DIAGNOSTIC</button>
      <button class="tab" data-view="flow">06 · HOW IT WORKS</button>
      <button class="tab" data-view="evidence">07 · EVIDENCE</button>
      <span class="spacer"></span>
      <button class="action" id="present">▣ PRESENT</button>
      <button class="action primary" id="play">▶ PLAY</button>
      <button class="action" id="pause">Ⅱ PAUSE</button>
    </div>


    <div class="sequence-bar">
      <div class="sequence-caption">SIMULATION SEQUENCE</div>
      <div class="sequence-buttons" id="sequenceButtons"></div>
      <div class="sequence-status" id="sequenceStatus">6 frames</div>
    </div>
    <div class="stage">
      <div id="presentationOverlay" class="presentation-overlay">
        <div class="presentation-card">
          <div class="presentation-top">
            <div>
              <div class="presentation-kicker">GUIDED DEMO</div>
              <div class="presentation-title" id="presentationTitle">THE RESULT</div>
              <div class="presentation-copy" id="presentationCopy">Start with the real LiDAR scene, then walk through how terrain intelligence produces the path.</div>
            </div>
            <button class="presentation-close" id="presentationClose">✕</button>
          </div>
          <div class="presentation-steps" id="presentationSteps"></div>
          <div class="presentation-footer">
            <span id="presentationCounter">1 / 6</span>
            <div class="presentation-actions">
              <button class="action" id="presentationBack">← BACK</button>
              <button class="action primary" id="presentationNext">NEXT →</button>
            </div>
          </div>
        </div>
      </div>
      <div id="navPlot" class="plot active">{nav_html}</div>
      <div id="resolutionPlot" class="plot">{resolution_html}</div>
      <div id="terrainPlot" class="plot">{terrain_html}</div>
      <div id="flowView" class="flow-view">
        <div class="flow-head">
          <div>
            <div class="flow-title">HOW THE SYSTEM TURNS REAL LiDAR INTO A DRIVABLE PATH</div>
            <div class="flow-sub">Every stage below is backed by the current RELLIS-3D frame data already processed by the demo. Click a stage to inspect the real output.</div>
          </div>
          <div class="flow-frame" id="flowFrameBadge">FRAME {metrics[0]['frame']}</div>
        </div>

        <div class="flow-grid">
          <button class="flow-step active" data-stage="lidar"><div class="flow-num">01</div><div class="flow-name">REAL LiDAR</div><div class="flow-mini">Raw 3D scan enters the pipeline.</div></button>
          <button class="flow-step" data-stage="ground"><div class="flow-num">02</div><div class="flow-name">GROUND EXTRACTION</div><div class="flow-mini">Ground points are separated while preserving point identity.</div></button>
          <button class="flow-step" data-stage="terrain"><div class="flow-num">03</div><div class="flow-name">TERRAIN UNDERSTANDING</div><div class="flow-mini">Slope, roughness, elevation variation and complexity are measured.</div></button>
          <button class="flow-step" data-stage="traversability"><div class="flow-num">04</div><div class="flow-name">TRAVERSABILITY</div><div class="flow-mini">Terrain is labeled drivable, non-drivable or unknown.</div></button>
          <button class="flow-step" data-stage="resolution"><div class="flow-num">05</div><div class="flow-name">ADAPTIVE RESOLUTION</div><div class="flow-mini">Distance base resolution is refined where terrain needs more detail.</div></button>
          <button class="flow-step" data-stage="map"><div class="flow-num">06</div><div class="flow-name">2.5D MAP</div><div class="flow-mini">The adaptive cell representation becomes the planning surface.</div></button>
          <button class="flow-step" data-stage="path"><div class="flow-num">07</div><div class="flow-name">DRIVABLE PATH</div><div class="flow-mini">A terrain-aware path is planned through connected drivable space.</div></button>
          <button class="flow-step" data-stage="replay"><div class="flow-num">08</div><div class="flow-name">VEHICLE REPLAY</div><div class="flow-mini">The simulated vehicle follows the terrain-derived path through real frames.</div></button>
        </div>

        <div class="flow-detail">
          <div class="flow-detail-head">
            <div class="flow-detail-title" id="flowDetailTitle">REAL LiDAR</div>
            <div class="flow-detail-note" id="flowDetailNote">Measured from the selected frame</div>
          </div>
          <div class="flow-values" id="flowValues"></div>
          <div class="flow-chain" id="flowChain">REAL LiDAR → GROUND → TERRAIN → TRAVERSABILITY → ADAPTIVE RESOLUTION → 2.5D MAP → PATH → VEHICLE</div>
        </div>
      </div>
      <div id="evidenceView" class="evidence-view">
        <div class="evidence-head">
          <div>
            <div class="evidence-title">MEASURED COMPARISON — REAL BENCHMARK DATA</div>
            <div class="evidence-sub">Five repeated runs per mode • benchmark sequences • values shown as recorded averages</div>
          </div>
          <div class="flow-frame">4 SEQUENCES</div>
        </div>
        <div class="evidence-scope">
          <div class="evidence-scope-pill" id="evidenceBenchmarkScope"><strong>Benchmark scope:</strong> SEQ 00000 • <span class="dim">frame not recorded in benchmark file</span></div>
          <div class="evidence-scope-pill" id="evidenceLiveScope"><strong>Live replay:</strong> SEQ {SEQUENCE} • FRAME {metrics[0]['frame']}</div>
        </div>
        <div class="evidence-selector" id="evidenceSelector"></div>
        <div class="evidence-kpis" id="evidenceKpis"></div>
        <div class="evidence-table-wrap">
          <table class="evidence-table">
            <thead><tr><th>MEASURE</th><th>FIXED 5 cm</th><th>DISTANCE ADAPTIVE</th><th>DISTANCE + TERRAIN</th></tr></thead>
            <tbody id="evidenceBody"></tbody>
          </table>
        </div>
        <div class="evidence-note" id="evidenceNote"></div>
      </div>
      <div class="hud">
        <span>FRAME <b id="hudFrame">{metrics[0]['frame']}</b></span>
        <span>LiDAR <b id="hudPoints">{metrics[0]['lidar_points']:,}</b></span>
        <span>ADAPTIVE CELLS <b id="hudCells">{metrics[0]['adaptive_cells']:,}</b></span>
        <span class="accent">PATH <b id="hudPath">{'AVAILABLE' if metrics[0]['path_available'] else 'NOT AVAILABLE'}</b></span>
        <span>RESOLUTION ZONES <b>ACTIVE</b></span>
      </div>
      <div class="resolution-hud">
        <div class="rh-title">ADAPTIVE PERCEPTION</div>
        <div class="rh-value" id="hudResolution">{metrics[0]['vehicle_resolution']}</div>
        <div class="rh-sub" id="hudResolutionSub">CURRENT VEHICLE CELL</div>
        <div class="rh-pills">
          <span data-res="0.050">5 cm</span>
          <span data-res="0.100">10 cm</span>
          <span data-res="0.125">12.5 cm</span>
          <span data-res="0.250">25 cm</span>
          <span data-res="0.500">50 cm</span>
        </div>
      </div>
    </div>

    <div class="bottom">
      <div class="timeline">
        <strong id="timelineFrame">FRAME {metrics[0]['frame']}</strong>
        <div>
          <input id="frameSlider" type="range" min="0" max="0" value="0" step="1" aria-label="Replay frame">
          <div class="frame-labels" id="frameLabels"></div>
        </div>
        <strong id="progressLabel">0%</strong>
      </div>
      <div class="notes">Offline replay of real LiDAR scans • vehicle path is terrain-derived</div>
    </div>
  </section>

  <aside class="side">
    <div class="card">
      <h3>CURRENT FRAME</h3>
      <div class="kpis">
        <div class="kpi"><div class="label">LiDAR POINTS</div><div class="value" id="kpiPoints">{metrics[0]['lidar_points']:,}</div><div class="small">real input</div></div>
        <div class="kpi"><div class="label">GROUND</div><div class="value" id="kpiGround">{metrics[0]['ground_points']:,}</div><div class="small">ground points</div></div>
        <div class="kpi"><div class="label">ADAPTIVE CELLS</div><div class="value" id="kpiCells">{metrics[0]['adaptive_cells']:,}</div><div class="small">2.5D leaves</div></div>
        <div class="kpi"><div class="label">DRIVABLE CELLS</div><div class="value" id="kpiDrive">{metrics[0]['drivable_cells']:,}</div><div class="small">terrain cells</div></div>
      </div>
    </div>

    <div class="card">
      <h3>NAVIGATION STATE</h3>
      <div class="path-status">
        <span class="path-dot"></span>
        <div class="path-copy"><b id="navState">{'PATH AVAILABLE' if metrics[0]['path_available'] else 'PATH NOT AVAILABLE'}</b><span id="navMeta">{metrics[0]['path_cells']} cells • {metrics[0]['path_length_m']:.2f} m</span></div>
      </div>
      <div style="height:8px"></div>
      <div class="kpis">
        <div class="kpi"><div class="label">NON-DRIVABLE</div><div class="value" id="kpiNonDrive">{metrics[0]['non_drivable_cells']:,}</div></div>
        <div class="kpi"><div class="label">UNKNOWN</div><div class="value" id="kpiUnknown">{metrics[0]['unknown_cells']:,}</div></div>
        <div class="kpi"><div class="label">HOLE CANDIDATES</div><div class="value" id="kpiHoles">{metrics[0]['holes']:,}</div></div>
        <div class="kpi"><div class="label">VEHICLE</div><div class="value" id="kpiVehicle">MOVING</div></div>
      </div>
    </div>

    <div class="card">
      <h3>ADAPTIVE RESOLUTION MIX <span id="resFrameTag" style="color:#536c79;font-size:8px;font-weight:500;margin-left:4px"></span></h3>
      <div id="resolutionList" class="res-list"></div>
      <div class="notes" style="margin-top:9px">Resolution level is selected from the existing distance base plus terrain-driven refinement logic.</div>
    </div>

    <div class="card">
      <div class="legend-title">WHAT THE DEMO SHOWS</div>
      <div class="legend-row"><span class="sw" style="background:#22c55e"></span>Drivable terrain</div>
      <div class="legend-row"><span class="sw" style="background:#ef4444"></span>Non-drivable terrain</div>
      <div class="legend-row"><span class="sw" style="background:#94a3b8"></span>Sparse / unknown</div>
      <div class="legend-row"><span class="sw" style="background:#22d3ee"></span>Terrain-aware path</div>
      <div class="notes" style="margin-top:8px">Hole/depression markers are local-surface candidates, not confirmed hazard labels.</div>
    </div>
  </aside>
</main>
</div>

<script>
const METRICS = {metrics_json};
const WORKFLOW_DATA = {workflow_json};
const SIMULATION_DATA = {simulation_json};
let currentSequence = "{SEQUENCE}";
const BENCHMARK_DATA = {benchmark_json};
const NAV_FRAMES = {nav_frames_json};
const RESOLUTION_FRAMES = {resolution_frames_json};
const TERRAIN_FRAMES = {terrain_frames_json};
const RES_LABELS = {resolution_labels_json};
const RES_VALUES = {resolution_values_json};
const RES_COLORS = {resolution_colors_json};
const graphIds = ['navPlot','resolutionPlot','terrainPlot'];
let currentIndex = 0;
let playing = false;
let playTimer = null;
let presentationIndex = 0;
const PRESENTATION_STEPS = [
  {{ view:'driving', title:'THE RESULT', copy:'Start with the real RELLIS-3D scene: terrain classification, drivable path and vehicle position are visible together.', name:'DRIVING', sub:'Show the outcome' }},
  {{ view:'flow', title:'HOW IT WORKS', copy:'Connect the result to the pipeline: LiDAR → ground → terrain → traversability → adaptive resolution → map → path → vehicle.', name:'HOW IT WORKS', sub:'Explain the pipeline' }},
  {{ view:'resolution', title:'WHERE DETAIL GOES', copy:'Show that resolution changes spatially instead of using one resolution everywhere.', name:'ADAPTIVE RESOLUTION', sub:'Explain adaptation' }},
  {{ view:'evidence', title:'MEASURED EVIDENCE', copy:'Show the repeated-run comparison for representation size, peak Python memory and build time.', name:'MEASURED EVIDENCE', sub:'Show measurements' }},
  {{ view:'terrain', title:'3D TERRAIN PROOF', copy:'Use the 3D view to connect the terrain surface, traversability and adaptive representation back to the real LiDAR scan.', name:'3D TERRAIN', sub:'Show geometry' }},
  {{ view:'driving', title:'REAL-FRAME REPLAY', copy:'Return to the driving view and use PLAY to replay the verified real frames through the terrain-derived path.', name:'REAL-FRAME REPLAY', sub:'Finish with replay' }}
];

function renderPresentation() {{
  const step = PRESENTATION_STEPS[presentationIndex];
  document.getElementById('presentationTitle').textContent = step.title;
  document.getElementById('presentationCopy').textContent = step.copy;
  document.getElementById('presentationCounter').textContent = (presentationIndex + 1) + ' / ' + PRESENTATION_STEPS.length;
  document.getElementById('presentationBack').disabled = presentationIndex === 0;
  document.getElementById('presentationNext').textContent = presentationIndex === PRESENTATION_STEPS.length - 1 ? 'FINISH ✓' : 'NEXT →';
  document.getElementById('presentationSteps').innerHTML = PRESENTATION_STEPS.map((s,i) => `
    <button class="presentation-step ${{i===presentationIndex?'active':''}}" data-present-index="${{i}}">
      <div class="presentation-step-num">0${{i+1}}</div>
      <div class="presentation-step-name">${{s.name}}</div>
      <div class="presentation-step-sub">${{s.sub}}</div>
    </button>`).join('');
  document.querySelectorAll('.presentation-step').forEach(btn => btn.addEventListener('click', () => {{
    presentationIndex = Number(btn.dataset.presentIndex);
    openPresentationStep();
  }}));
}}

function openPresentationStep() {{
  const step = PRESENTATION_STEPS[presentationIndex];
  document.getElementById('presentationOverlay').classList.add('open');
  renderPresentation();
  stopReplay();
  setTab(step.view);
}}

function startPresentation() {{
  presentationIndex = 0;
  openPresentationStep();
}}

function closePresentation() {{
  document.getElementById('presentationOverlay').classList.remove('open');
}}

function nextPresentationStep() {{
  if (presentationIndex >= PRESENTATION_STEPS.length - 1) {{
    closePresentation();
    setTab('driving');
    return;
  }}
  presentationIndex += 1;
  openPresentationStep();
}}

function backPresentationStep() {{
  if (presentationIndex === 0) return;
  presentationIndex -= 1;
  openPresentationStep();
}}

function graphNode(id) {{
  return document.querySelector('#' + id + ' .plotly-graph-div') || document.getElementById(id).querySelector('.js-plotly-plot');
}}


const WORKFLOW_STAGE_ORDER = ['lidar','ground','terrain','traversability','resolution','map','path','replay'];
const WORKFLOW_STAGE_TITLES = {{
  lidar:'REAL LiDAR', ground:'GROUND EXTRACTION', terrain:'TERRAIN UNDERSTANDING', traversability:'TRAVERSABILITY',
  resolution:'ADAPTIVE RESOLUTION', map:'2.5D MAP', path:'DRIVABLE PATH', replay:'VEHICLE REPLAY'
}};
let workflowStage = 'lidar';

function renderWorkflow(index) {{
  const f = WORKFLOW_DATA[index] || WORKFLOW_DATA[0];
  document.getElementById('flowFrameBadge').textContent = 'FRAME ' + f.frame;
  const values = f[workflowStage] || [];
  document.getElementById('flowDetailTitle').textContent = WORKFLOW_STAGE_TITLES[workflowStage];
  document.getElementById('flowDetailNote').textContent = 'REAL RELLIS-3D output • FRAME ' + f.frame;
  document.getElementById('flowValues').innerHTML = values.map(item => `
    <div class="flow-value"><div class="k">${{item.k}}</div><div class="v">${{item.v}}</div><div class="s">${{item.s || ''}}</div></div>`).join('');
  document.querySelectorAll('.flow-step').forEach(b => b.classList.toggle('active', b.dataset.stage === workflowStage));
}}

document.querySelectorAll('.flow-step').forEach(btn => btn.addEventListener('click', () => {{
  workflowStage = btn.dataset.stage;
  renderWorkflow(currentIndex);
}}));

function fmtBytes(bytes) {{
  if (bytes >= 1024*1024) return (bytes/(1024*1024)).toFixed(2) + ' MiB';
  return (bytes/1024).toFixed(1) + ' KiB';
}}
function deltaText(v) {{
  if (v === undefined) return '';
  if (v < 0) return `<span class="evidence-delta">${{Math.abs(v).toFixed(2)}}% fewer vs fixed</span>`;
  if (v > 0) return `<span class="evidence-delta">${{v.toFixed(2)}}% higher vs fixed</span>`;
  return `<span class="evidence-delta">same as fixed</span>`;
}}
function renderEvidence() {{
  const seqs = Object.keys(BENCHMARK_DATA.sequences);
  document.getElementById('evidenceSelector').innerHTML = seqs.map(seq =>
    `<button class="evidence-seq ${{seq===evidenceSequence?'active':''}}" data-seq="${{seq}}">SEQ ${{seq}}</button>`
  ).join('');
  document.querySelectorAll('.evidence-seq').forEach(b => b.addEventListener('click', () => {{ evidenceSequence=b.dataset.seq; renderEvidence(); }}));

  const d = BENCHMARK_DATA.sequences[evidenceSequence];
  document.getElementById('evidenceBenchmarkScope').innerHTML = `<strong>Benchmark scope:</strong> SEQ ${{evidenceSequence}} • <span class="dim">frame not recorded in benchmark file</span>`;
  document.getElementById('evidenceLiveScope').innerHTML = `<strong>Live replay:</strong> SEQ {SEQUENCE} • FRAME ${{METRICS[currentIndex].frame}}`;
  const avgCells = Math.round((d.A.cells + d.B.cells + d.C.cells)/3);
  document.getElementById('evidenceKpis').innerHTML = `
    <div class="evidence-kpi"><div class="k">BENCHMARK REPEATS</div><div class="v">5 / MODE</div><div class="s">Repeated runs for sequence ${{evidenceSequence}}</div></div>
    <div class="evidence-kpi"><div class="k">FIXED REPRESENTATION</div><div class="v">${{d.A.cells.toLocaleString()}} CELLS</div><div class="s">${{fmtBytes(d.A.compact_bytes)}} compact map</div></div>
    <div class="evidence-kpi"><div class="k">ADAPTIVE REPRESENTATION</div><div class="v">${{d.C.cells.toLocaleString()}} CELLS</div><div class="s">${{fmtBytes(d.C.compact_bytes)}} compact map</div></div>`;

  document.getElementById('evidenceBody').innerHTML = `
    <tr><td>Occupied / representation cells</td><td>${{d.A.cells.toLocaleString()}}</td><td>${{d.B.cells.toLocaleString()}}${{deltaText(d.B.cells_delta_pct)}}</td><td>${{d.C.cells.toLocaleString()}}${{deltaText(d.C.cells_delta_pct)}}</td></tr>
    <tr><td>Compact map size</td><td>${{fmtBytes(d.A.compact_bytes)}}</td><td>${{fmtBytes(d.B.compact_bytes)}}${{deltaText(d.B.compact_delta_pct)}}</td><td>${{fmtBytes(d.C.compact_bytes)}}${{deltaText(d.C.compact_delta_pct)}}</td></tr>
    <tr><td>Peak Python memory</td><td>${{fmtBytes(d.A.peak_bytes)}}</td><td>${{fmtBytes(d.B.peak_bytes)}}${{deltaText(d.B.peak_delta_pct)}}</td><td>${{fmtBytes(d.C.peak_bytes)}}${{deltaText(d.C.peak_delta_pct)}}</td></tr>
    <tr><td>Mean build time</td><td>${{d.A.build_s.toFixed(3)}} s</td><td>${{d.B.build_s.toFixed(3)}} s${{deltaText(d.B.build_delta_pct)}}</td><td>${{d.C.build_s.toFixed(3)}} s${{deltaText(d.C.build_delta_pct)}}</td></tr>`;

  document.getElementById('evidenceNote').textContent = BENCHMARK_DATA.notes.join(' • ');
}}

function setTab(view) {{
  document.querySelectorAll('.tab').forEach(b => b.classList.toggle('active', b.dataset.view === view));
  document.querySelectorAll('.plot').forEach(p => p.classList.remove('active'));
  document.getElementById('flowView').classList.remove('active');
  document.getElementById('evidenceView').classList.remove('active');

  const special = ['flow','evidence'].includes(view);
  document.querySelector('.hud').style.display = special ? 'none' : 'flex';
  document.querySelector('.resolution-hud').style.display = special ? 'none' : 'block';
  document.querySelector('.bottom').style.display = special ? 'none' : 'grid';

  if (view === 'flow') {{
    document.getElementById('flowView').classList.add('active');
    renderWorkflow(currentIndex);
    return;
  }}
  if (view === 'evidence') {{
    document.getElementById('evidenceView').classList.add('active');
    document.getElementById('evidenceView').scrollTop = 0;
    renderEvidence();
    return;
  }}

  if (view === 'driving') document.getElementById('navPlot').classList.add('active');
  if (view === 'resolution') document.getElementById('resolutionPlot').classList.add('active');
  if (['terrain','raw','diagnostic'].includes(view)) document.getElementById('terrainPlot').classList.add('active');

  if (['terrain','raw','diagnostic'].includes(view)) {{
    terrainView = view;
    applyTerrainVisibility();
  }}

  renderFrame(currentIndex);
}}

document.querySelectorAll('.tab').forEach(btn => btn.addEventListener('click', () => setTab(btn.dataset.view)));
document.getElementById('present').addEventListener('click', startPresentation);
document.getElementById('presentationClose').addEventListener('click', closePresentation);
document.getElementById('presentationNext').addEventListener('click', nextPresentationStep);
document.getElementById('presentationBack').addEventListener('click', backPresentationStep);
document.addEventListener('keydown', e => {{
  if (!document.getElementById('presentationOverlay').classList.contains('open')) return;
  if (e.key === 'Escape') closePresentation();
  if (e.key === 'ArrowRight') nextPresentationStep();
  if (e.key === 'ArrowLeft') backPresentationStep();
}});

let evidenceSequence = '00000';
function fmtBytes(bytes) {{
  if (bytes >= 1024*1024) return (bytes/(1024*1024)).toFixed(2) + ' MiB';
  return (bytes/1024).toFixed(1) + ' KiB';
}}
function deltaText(v) {{
  if (v === undefined) return '';
  if (v < 0) return `<span class="evidence-delta">${{Math.abs(v).toFixed(2)}}% fewer vs fixed</span>`;
  if (v > 0) return `<span class="evidence-delta">${{v.toFixed(2)}}% higher vs fixed</span>`;
  return `<span class="evidence-delta">same as fixed</span>`;
}}
function renderEvidence() {{
  const seqs = Object.keys(BENCHMARK_DATA.sequences);
  document.getElementById('evidenceSelector').innerHTML = seqs.map(seq =>
    `<button class="evidence-seq ${{seq===evidenceSequence?'active':''}}" data-seq="${{seq}}">SEQ ${{seq}}</button>`
  ).join('');
  document.querySelectorAll('.evidence-seq').forEach(b => b.addEventListener('click', () => {{ evidenceSequence=b.dataset.seq; renderEvidence(); }}));

  const d = BENCHMARK_DATA.sequences[evidenceSequence];
  document.getElementById('evidenceBenchmarkScope').innerHTML = `<strong>Benchmark scope:</strong> SEQ ${{evidenceSequence}} • <span class="dim">frame not recorded in benchmark file</span>`;
  document.getElementById('evidenceLiveScope').innerHTML = `<strong>Live replay:</strong> SEQ ${{currentSequence}} • FRAME ${{currentMetrics().frame}}`;
  document.getElementById('evidenceKpis').innerHTML = `
    <div class="evidence-kpi"><div class="k">BENCHMARK REPEATS</div><div class="v">5 / MODE</div><div class="s">Repeated runs for sequence ${{evidenceSequence}}</div></div>
    <div class="evidence-kpi"><div class="k">FIXED REPRESENTATION</div><div class="v">${{d.A.cells.toLocaleString()}} CELLS</div><div class="s">${{fmtBytes(d.A.compact_bytes)}} compact map</div></div>
    <div class="evidence-kpi"><div class="k">ADAPTIVE REPRESENTATION</div><div class="v">${{d.C.cells.toLocaleString()}} CELLS</div><div class="s">${{fmtBytes(d.C.compact_bytes)}} compact map</div></div>`;

  document.getElementById('evidenceBody').innerHTML = `
    <tr><td>Occupied / representation cells</td><td>${{d.A.cells.toLocaleString()}}</td><td>${{d.B.cells.toLocaleString()}}${{deltaText(d.B.cells_delta_pct)}}</td><td>${{d.C.cells.toLocaleString()}}${{deltaText(d.C.cells_delta_pct)}}</td></tr>
    <tr><td>Compact map size</td><td>${{fmtBytes(d.A.compact_bytes)}}</td><td>${{fmtBytes(d.B.compact_bytes)}}${{deltaText(d.B.compact_delta_pct)}}</td><td>${{fmtBytes(d.C.compact_bytes)}}${{deltaText(d.C.compact_delta_pct)}}</td></tr>
    <tr><td>Peak Python memory</td><td>${{fmtBytes(d.A.peak_bytes)}}</td><td>${{fmtBytes(d.B.peak_bytes)}}${{deltaText(d.B.peak_delta_pct)}}</td><td>${{fmtBytes(d.C.peak_bytes)}}${{deltaText(d.C.peak_delta_pct)}}</td></tr>
    <tr><td>Mean build time</td><td>${{d.A.build_s.toFixed(3)}} s</td><td>${{d.B.build_s.toFixed(3)}} s${{deltaText(d.B.build_delta_pct)}}</td><td>${{d.C.build_s.toFixed(3)}} s${{deltaText(d.C.build_delta_pct)}}</td></tr>`;

  document.getElementById('evidenceNote').textContent = BENCHMARK_DATA.notes.join(' • ');
}}

function currentPayload() {{ return SIMULATION_DATA[currentSequence]; }}
function currentMetrics() {{ return currentPayload().metrics; }}
function currentWorkflow() {{ return currentPayload().workflow; }}
function currentNavFrames() {{ return currentPayload().nav_frames; }}
function currentResolutionFrames() {{ return currentPayload().resolution_frames; }}
function currentTerrainFrames() {{ return currentPayload().terrain_frames; }}

function renderResolutionMix(m) {{
  const counts = RES_VALUES.map(v => m.resolution_counts[v.toFixed(3)] || 0);
  const max = Math.max(1, ...counts);
  document.getElementById('resolutionList').innerHTML = counts.map((count, i) => `
    <div class="res-row">
      <span>${{RES_LABELS[i]}}</span>
      <div class="res-bar"><div class="res-fill" style="width:${{(count/max)*100}}%;background:${{RES_COLORS[i]}}"></div></div>
      <span class="res-count">${{count.toLocaleString()}}</span>
    </div>`).join('');
}}

function updateSequenceControls() {{
  const sequences = Object.keys(SIMULATION_DATA);
  document.getElementById('sequenceButtons').innerHTML = sequences.map(seq =>
    `<button class="sequence-btn ${{seq===currentSequence?'active':''}}" data-sequence="${{seq}}">SEQ ${{seq}}</button>`
  ).join('');
  document.querySelectorAll('.sequence-btn').forEach(btn => btn.addEventListener('click', () => selectSequence(btn.dataset.sequence)));
  const frames = currentPayload().frames;
  document.getElementById('sequenceStatus').textContent = `${{frames.length}} frames • ${{frames[0]}} → ${{frames[frames.length-1]}}`;
  document.getElementById('headerSequence').textContent = currentSequence;
  const labels = document.getElementById('frameLabels');
  labels.innerHTML = frames.map(frame => `<span>${{frame}}</span>`).join('');
  const slider = document.getElementById('frameSlider');
  slider.max = Math.max(0, frames.length - 1);
  slider.value = currentIndex;
}}

function updateHud(index) {{
  const metrics = currentMetrics();
  currentIndex = Math.max(0, Math.min(metrics.length - 1, Number(index)));
  const m = metrics[currentIndex];
  const percent = metrics.length <= 1 ? 100 : Math.round((currentIndex/(metrics.length-1))*100);
  document.getElementById('hudFrame').textContent = m.frame;
  document.getElementById('hudPoints').textContent = m.lidar_points.toLocaleString();
  document.getElementById('hudCells').textContent = m.adaptive_cells.toLocaleString();
  document.getElementById('hudPath').textContent = m.path_available ? 'AVAILABLE' : 'NOT AVAILABLE';
  document.getElementById('hudResolution').textContent = m.vehicle_resolution;
  document.querySelectorAll('.rh-pills span').forEach(p => p.classList.toggle('active', p.textContent === m.vehicle_resolution));
  document.getElementById('timelineFrame').textContent = `FRAME ${{m.frame}}`;
  document.getElementById('progressLabel').textContent = percent + '%';
  document.getElementById('kpiPoints').textContent = m.lidar_points.toLocaleString();
  document.getElementById('kpiGround').textContent = m.ground_points.toLocaleString();
  document.getElementById('kpiCells').textContent = m.adaptive_cells.toLocaleString();
  document.getElementById('kpiDrive').textContent = m.drivable_cells.toLocaleString();
  document.getElementById('kpiNonDrive').textContent = m.non_drivable_cells.toLocaleString();
  document.getElementById('kpiUnknown').textContent = m.unknown_cells.toLocaleString();
  document.getElementById('kpiHoles').textContent = m.holes.toLocaleString();
  document.getElementById('navState').textContent = m.path_available ? 'PATH AVAILABLE' : 'PATH NOT AVAILABLE';
  document.getElementById('navMeta').textContent = `${{m.path_cells}} cells • ${{m.path_length_m.toFixed(2)}} m`;
  renderResolutionMix(m);
  document.getElementById('resFrameTag').textContent = `• FRAME ${{m.frame}}`;
  const liveScope = document.getElementById('evidenceLiveScope');
  if (liveScope) liveScope.innerHTML = `<strong>Live replay:</strong> SEQ ${{currentSequence}} • FRAME ${{m.frame}}`;
}}

function renderWorkflow(index) {{
  const data = currentWorkflow();
  const f = data[index] || data[0];
  document.getElementById('flowFrameBadge').textContent = `SEQ ${{currentSequence}} • FRAME ${{f.frame}}`;
  const values = f[workflowStage] || [];
  document.getElementById('flowDetailTitle').textContent = WORKFLOW_STAGE_TITLES[workflowStage];
  document.getElementById('flowDetailNote').textContent = `REAL RELLIS-3D output • SEQ ${{currentSequence}} • FRAME ${{f.frame}}`;
  document.getElementById('flowValues').innerHTML = values.map(item => `
    <div class="flow-value"><div class="k">${{item.k}}</div><div class="v">${{item.v}}</div><div class="s">${{item.s || ''}}</div></div>`).join('');
  document.querySelectorAll('.flow-step').forEach(b => b.classList.toggle('active', b.dataset.stage === workflowStage));
}}

function frameData(list, index) {{
  return list[index] ? list[index].data : [];
}}

function activeViewPlot() {{
  const active = document.querySelector('.tab.active');
  const view = active ? active.dataset.view : 'driving';
  if (view === 'driving') return ['navPlot', currentNavFrames()];
  if (view === 'resolution') return ['resolutionPlot', currentResolutionFrames()];
  if (view === 'flow' || view === 'evidence') return [null, null];
  return ['terrainPlot', currentTerrainFrames()];
}}

let terrainView = 'terrain';
let renderToken = 0;

function graphNode(id) {{
  const host = document.getElementById(id);
  if (!host) return null;
  return host.querySelector('.js-plotly-plot') || host;
}}

function terrainVisibility() {{
  if (terrainView === 'raw') return [true,false,false,false,false,false,false,false,false,false,true,true,true];
  if (terrainView === 'diagnostic') return new Array(13).fill(true);
  return [false,true,true,true,true,true,true,true,true,true,true,true,true];
}}

function applyTerrainVisibility() {{
  const terrain = graphNode('terrainPlot');
  if (!terrain || !window.Plotly) return;
  try {{ Plotly.restyle(terrain, {{visible: terrainVisibility()}}); }} catch (err) {{ console.warn('Terrain visibility update failed', err); }}
}}

async function renderFrame(index) {{
  const token = renderToken;
  const metrics = currentMetrics();
  currentIndex = Math.max(0, Math.min(metrics.length - 1, Number(index)));
  document.getElementById('frameSlider').value = currentIndex;
  updateHud(currentIndex);

  const active = document.querySelector('.tab.active');
  const activeView = active ? active.dataset.view : 'driving';
  if (activeView === 'flow') {{ renderWorkflow(currentIndex); return; }}
  if (activeView === 'evidence') {{ renderEvidence(); return; }}
  if (!window.Plotly) return;

  const [id, states] = activeViewPlot();
  const plot = graphNode(id);
  const state = states[currentIndex];
  if (!plot || !state) return;

  try {{
    await Plotly.react(plot, frameData(states, currentIndex), plot.layout || {{}});
  }} catch (err) {{
    console.warn('Frame render failed for ' + id, err);
  }}

  if (token !== renderToken) return;
  if (id === 'terrainPlot') applyTerrainVisibility();
}}

function stopReplay() {{
  playing = false;
  clearTimeout(playTimer);
  playTimer = null;
  renderToken += 1;
  document.getElementById('play').textContent = '▶ PLAY';
}}

async function runReplay() {{
  if (playing) return;
  playing = true;
  clearTimeout(playTimer);
  playTimer = null;
  renderToken += 1;
  const thisRun = renderToken;
  document.getElementById('play').textContent = '▶ PLAYING';

  // Start from the first real frame available in the selected sequence.
  currentIndex = 0;
  await renderFrame(0);
  if (!playing || thisRun !== renderToken) return;

  const nextFrame = async () => {{
    if (!playing || thisRun !== renderToken) return;
    const next = currentIndex + 1;
    if (next >= currentMetrics().length) {{ stopReplay(); return; }}
    await renderFrame(next);
    if (!playing || thisRun !== renderToken) return;
    playTimer = setTimeout(nextFrame, 4000);
  }};
  playTimer = setTimeout(nextFrame, 4000);
}}

async function selectSequence(sequence) {{
  if (!SIMULATION_DATA[sequence] || sequence === currentSequence) return;
  stopReplay();
  currentSequence = sequence;
  currentIndex = 0;
  updateSequenceControls();
  updateHud(0);
  const activeView = document.querySelector('.tab.active')?.dataset.view || 'driving';
  if (activeView === 'flow') {{ renderWorkflow(0); return; }}
  if (activeView === 'evidence') {{ renderEvidence(); return; }}
  await renderFrame(0);
}}

document.getElementById('frameSlider').addEventListener('input', async e => {{
  stopReplay();
  await renderFrame(Number(e.target.value));
}});
document.getElementById('play').addEventListener('click', runReplay);
document.getElementById('pause').addEventListener('click', stopReplay);

document.querySelectorAll('.flow-step').forEach(btn => btn.addEventListener('click', () => {{ workflowStage = btn.dataset.stage; renderWorkflow(currentIndex); }}));

updateSequenceControls();
updateHud(0);
renderResolutionMix(currentMetrics()[0]);
renderWorkflow(0);
renderEvidence();
renderPresentation();
</script>
</body>
</html>'''


BENCHMARK_DATA = {'labels': {'A': 'FIXED 5 cm', 'B': 'DISTANCE ADAPTIVE', 'C': 'DISTANCE + TERRAIN'}, 'sequences': {'00000': {'A': {'cells': 4482, 'compact_bytes': 246510, 'peak_bytes': 5137200, 'build_s': 4.218, 'runs': 5}, 'B': {'cells': 4162, 'compact_bytes': 228910, 'peak_bytes': 4796647, 'build_s': 4.126, 'runs': 5, 'cells_delta_pct': -7.14, 'compact_delta_pct': -7.14, 'peak_delta_pct': -6.63, 'build_delta_pct': -2.18}, 'C': {'cells': 4174, 'compact_bytes': 229570, 'peak_bytes': 4820891, 'build_s': 4.229, 'runs': 5, 'cells_delta_pct': -6.87, 'compact_delta_pct': -6.87, 'peak_delta_pct': -6.16, 'build_delta_pct': 0.26}}, '00001': {'A': {'cells': 4002, 'compact_bytes': 220110, 'peak_bytes': 4597324, 'build_s': 3.992, 'runs': 5}, 'B': {'cells': 3704, 'compact_bytes': 203720, 'peak_bytes': 4279040, 'build_s': 3.807, 'runs': 5, 'cells_delta_pct': -7.45, 'compact_delta_pct': -7.45, 'peak_delta_pct': -6.92, 'build_delta_pct': -4.63}, 'C': {'cells': 3726, 'compact_bytes': 204930, 'peak_bytes': 4303937, 'build_s': 3.873, 'runs': 5, 'cells_delta_pct': -6.9, 'compact_delta_pct': -6.9, 'peak_delta_pct': -6.38, 'build_delta_pct': -2.98}}, '00002': {'A': {'cells': 8045, 'compact_bytes': 442475, 'peak_bytes': 9178985, 'build_s': 7.797, 'runs': 5}, 'B': {'cells': 7520, 'compact_bytes': 413600, 'peak_bytes': 8622893, 'build_s': 8.026, 'runs': 5, 'cells_delta_pct': -6.53, 'compact_delta_pct': -6.53, 'peak_delta_pct': -6.06, 'build_delta_pct': 2.94}, 'C': {'cells': 7520, 'compact_bytes': 413600, 'peak_bytes': 8623057, 'build_s': 8.391, 'runs': 5, 'cells_delta_pct': -6.53, 'compact_delta_pct': -6.53, 'peak_delta_pct': -6.06, 'build_delta_pct': 7.62}}, '00003': {'A': {'cells': 3491, 'compact_bytes': 192005, 'peak_bytes': 4093961, 'build_s': 3.775, 'runs': 5}, 'B': {'cells': 3300, 'compact_bytes': 181500, 'peak_bytes': 3892070, 'build_s': 3.871, 'runs': 5, 'cells_delta_pct': -5.47, 'compact_delta_pct': -5.47, 'peak_delta_pct': -4.93, 'build_delta_pct': 2.54}, 'C': {'cells': 3309, 'compact_bytes': 181995, 'peak_bytes': 3899421, 'build_s': 3.829, 'runs': 5, 'cells_delta_pct': -5.21, 'compact_delta_pct': -5.21, 'peak_delta_pct': -4.75, 'build_delta_pct': 1.43}}}, 'overall': {'A': {'cells': 5005, 'compact_bytes': 275275, 'peak_bytes': 5751868, 'build_s': 4.946, 'runs': 5}, 'B': {'cells': 4672, 'compact_bytes': 256932, 'peak_bytes': 5397662, 'build_s': 4.957, 'runs': 5, 'cells_delta_pct': -6.65, 'compact_delta_pct': -6.66, 'peak_delta_pct': -6.16, 'build_delta_pct': 0.22}, 'C': {'cells': 4682, 'compact_bytes': 257524, 'peak_bytes': 5411826, 'build_s': 5.08, 'runs': 5, 'cells_delta_pct': -6.45, 'compact_delta_pct': -6.45, 'peak_delta_pct': -5.91, 'build_delta_pct': 2.71}}, 'notes': ['Repeated-run terrain impact benchmark: 5 runs per mode.', 'Compact map is representation size; peak Python memory is process-level allocation.', 'Fidelity error columns are not displayed because the captured terminal export truncated those values.', 'Benchmark frame ID is not recorded in this sequence-level repeated-run file.']}


def main() -> None:
    print("FINAL ADAPTIVE LiDAR VEHICLE DEMO — UI V7 + MULTI-SEQUENCE REPLAY")
    print("Real RELLIS-3D + terrain-aware drivable path + adaptive 2.5D")
    print()

    default_sequence = SEQUENCE
    sequence_names = [
        entry
        for entry in sorted(os.listdir(DATA_ROOT))
        if os.path.isdir(os.path.join(DATA_ROOT, entry)) and available_sequence_frames(entry)
    ]
    if not sequence_names:
        raise RuntimeError(f"No RELLIS-3D LiDAR sequences found under {DATA_ROOT}")
    if default_sequence not in sequence_names:
        default_sequence = sequence_names[0]

    simulation_data = {}
    default_bundle = None
    default_nav_fig = None
    default_resolution_fig = None
    default_terrain_fig = None

    for sequence in sequence_names:
        bundle = build_sequence_bundle(sequence)
        if sequence == default_sequence:
            default_bundle = bundle
            default_nav_fig = bundle.pop("_nav_fig")
            default_resolution_fig = bundle.pop("_resolution_fig")
            default_terrain_fig = bundle.pop("_terrain_fig")
        else:
            bundle.pop("_nav_fig")
            bundle.pop("_resolution_fig")
            bundle.pop("_terrain_fig")
        simulation_data[sequence] = bundle
        print(f"  Frames with drivable path: {bundle['path_success']}/{len(bundle['frames'])}")

    if default_bundle is None or default_nav_fig is None or default_resolution_fig is None or default_terrain_fig is None:
        raise RuntimeError(f"Default sequence {default_sequence} could not be processed")

    replay_module.SEQUENCE = default_sequence

    os.makedirs(DEMO_DIR, exist_ok=True)
    output_path = os.path.join(DEMO_DIR, "index.html")
    html = build_dashboard(
        default_nav_fig,
        default_resolution_fig,
        default_terrain_fig,
        default_bundle["metrics"],
        default_bundle["workflow"],
        BENCHMARK_DATA,
        simulation_data,
    )
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(html)

    print()
    print("SEQUENCES ENABLED:", ", ".join(sequence_names))
    print("DEFAULT SIMULATION SEQUENCE:", default_sequence)
    print("FINAL DEMO OUTPUT:", output_path)
    print("Views: driving / adaptive resolution / 3D terrain / raw LiDAR / diagnostic / how it works / evidence")
    print("Navigation: terrain-derived drivable path + vehicle replay")
    print("Resolution levels: 5 / 10 / 12.5 / 25 / 50 cm")
    print("Plotly modebar: hidden for presentation clarity")
    print("Dashboard layout: toolbar / sequence selector / plot / replay timeline")
    print("Important: offline replay of real LiDAR scans; not a measured live vehicle system.")


if __name__ == "__main__":
    main()
