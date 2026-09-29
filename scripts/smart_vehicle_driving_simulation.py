from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import plotly.graph_objects as go

from src.road_processor import extract_ground_mask
from src.terrain_analysis import analyze_terrain_cells
from src.terrain_cells import build_terrain_cells
from src.terrain_quadtree import build_terrain_quadtree_for_cell
from src.terrain_resolution import build_base_resolution_by_cell


DATA_ROOT = "data/RELLIS3D"
SEQUENCE = "00000"

FRAME_NAMES = (
    "000000",
    "000001",
    "000002",
    "000003",
    "000004",
    "000005",
)

RESULTS_DIR = "results"

PARENT_CELL_SIZE = 1.0
MIN_RESOLUTION = 0.05
NO_ERROR_BOUND = 1_000_000.0

RAW_MAX_POINTS = 16000

HOLE_MIN_DEPTH_M = 0.10
HOLE_MIN_NEIGHBORS = 3

X_MIN = 0.0
X_MAX = 30.0
Y_MIN = -10.0
Y_MAX = 10.0

RESOLUTIONS = (
    0.05,
    0.10,
    0.125,
    0.25,
    0.50,
)

RESOLUTION_LABELS = {
    0.05: "5 cm",
    0.10: "10 cm",
    0.125: "12.5 cm",
    0.25: "25 cm",
    0.50: "50 cm",
}

RESOLUTION_COLORS = {
    0.05: "#38bdf8",
    0.10: "#22c55e",
    0.125: "#facc15",
    0.25: "#fb923c",
    0.50: "#ef4444",
}

TRAVERSABILITY_COLORS = {
    "DRIVABLE": "#22c55e",
    "NON-DRIVABLE": "#ef4444",
    "SPARSE / UNKNOWN": "#94a3b8",
}

TRAVERSABILITY_SYMBOLS = {
    "DRIVABLE": "circle",
    "NON-DRIVABLE": "x",
    "SPARSE / UNKNOWN": "diamond",
}


def frame_path(frame_name: str) -> str:
    return os.path.join(
        DATA_ROOT,
        SEQUENCE,
        "vel_cloud_node_kitti_bin",
        f"{frame_name}.bin",
    )


def load_frame(frame_name: str) -> np.ndarray:
    path = frame_path(frame_name)

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"LiDAR frame not found: {path}"
        )

    data = np.fromfile(
        path,
        dtype=np.float32,
    ).reshape(-1, 4)

    if len(data) == 0:
        raise ValueError(
            f"LiDAR frame is empty: {path}"
        )

    return data


def safe_float(
    value,
    default: float = np.nan,
) -> float:

    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_parent_inputs(
    data: np.ndarray,
):
    ground_mask, point_id, _ = (
        extract_ground_mask(data)
    )

    ground_point_ids = point_id[
        ground_mask
    ]

    cells = build_terrain_cells(
        data=data,
        ground_mask=ground_mask,
        point_id=point_id,
        cell_size=PARENT_CELL_SIZE,
    )

    base_resolution_by_cell = (
        build_base_resolution_by_cell(
            cells
        )
    )

    analyzed_cells = analyze_terrain_cells(
        data=data,
        cells=cells,
        base_resolution_by_cell=(
            base_resolution_by_cell
        ),
    )

    return (
        ground_point_ids,
        cells,
        base_resolution_by_cell,
        analyzed_cells,
    )


def build_adaptive_leaves(
    data: np.ndarray,
    cells: dict,
    base_resolution_by_cell: dict,
    analyzed_cells: dict,
) -> list[dict]:

    leaves = []

    for cell_id, cell in cells.items():

        base_resolution = float(
            base_resolution_by_cell[
                cell_id
            ]
        )

        request = analyzed_cells[
            cell_id
        ][
            "refinement_request"
        ]

        requested_resolution = float(
            request[
                "requested_resolution"
            ]
        )

        priority = safe_float(
            analyzed_cells[
                cell_id
            ].get(
                "terrain_priority"
            ),
            0.0,
        )

        refinement_request = {
            "region": {
                "cell_id": cell_id,
                "bbox": cell["bbox"],
            },
            "base_resolution": base_resolution,
            "requested_resolution": requested_resolution,
            "priority": priority,
            "reason": "distance + terrain",
        }

        result = build_terrain_quadtree_for_cell(
            data=data,
            cell=cell,
            refinement_request=refinement_request,
            max_plane_residual=NO_ERROR_BOUND,
            max_elevation_variation=NO_ERROR_BOUND,
            min_resolution=MIN_RESOLUTION,
        )

        for leaf in result["leaves"]:

            if len(
                leaf["point_indices"]
            ) == 0:
                continue

            leaf_copy = dict(
                leaf
            )

            leaf_copy[
                "_parent_cell_id"
            ] = cell_id

            leaves.append(
                leaf_copy
            )

    return leaves


def build_terrain_records(
    data: np.ndarray,
    cells: dict,
    analyzed_cells: dict,
) -> dict:

    records = {}

    for cell_id, cell in cells.items():

        point_indices = np.asarray(
            cell["point_indices"],
            dtype=np.int64,
        )

        if len(point_indices) == 0:
            continue

        z_values = data[
            point_indices,
            2,
        ].astype(
            np.float64
        )

        bbox = cell["bbox"]

        x_center = (
            float(bbox["x_min"])
            + float(bbox["x_max"])
        ) / 2.0

        y_center = (
            float(bbox["y_min"])
            + float(bbox["y_max"])
        ) / 2.0

        terrain = analyzed_cells[
            cell_id
        ]

        traversability = str(
            terrain.get(
                "traversability",
                "SPARSE / UNKNOWN",
            )
        )

        if traversability not in TRAVERSABILITY_COLORS:
            traversability = (
                "SPARSE / UNKNOWN"
            )

        records[cell_id] = {
            "x_center": x_center,
            "y_center": y_center,
            "z_mean": float(
                np.mean(z_values)
            ),
            "point_count": int(
                len(point_indices)
            ),
            "slope": safe_float(
                terrain.get(
                    "slope"
                )
            ),
            "roughness": safe_float(
                terrain.get(
                    "roughness"
                )
            ),
            "terrain_complexity": safe_float(
                terrain.get(
                    "terrain_complexity"
                )
            ),
            "terrain_confidence": safe_float(
                terrain.get(
                    "terrain_confidence"
                )
            ),
            "terrain_priority": safe_float(
                terrain.get(
                    "terrain_priority"
                )
            ),
            "traversability": traversability,
        }

    return records


def detect_hole_candidates(
    terrain_records: dict,
) -> list[dict]:

    records = list(
        terrain_records.values()
    )

    if len(records) < (
        HOLE_MIN_NEIGHBORS + 1
    ):
        return []

    candidates = []

    for index, record in enumerate(
        records
    ):

        if (
            record["traversability"]
            == "SPARSE / UNKNOWN"
        ):
            continue

        neighbours = []

        for other_index, other in enumerate(
            records
        ):

            if index == other_index:
                continue

            distance = float(
                np.hypot(
                    record["x_center"]
                    - other["x_center"],
                    record["y_center"]
                    - other["y_center"],
                )
            )

            if distance <= 1.5:
                neighbours.append(
                    other["z_mean"]
                )

        if len(neighbours) < HOLE_MIN_NEIGHBORS:
            continue

        reference_z = float(
            np.median(
                np.asarray(
                    neighbours,
                    dtype=np.float64,
                )
            )
        )

        depth = (
            reference_z
            - record["z_mean"]
        )

        if depth < HOLE_MIN_DEPTH_M:
            continue

        candidates.append(
            {
                **record,
                "local_reference_z": reference_z,
                "hole_depth_m": float(depth),
            }
        )

    candidates.sort(
        key=lambda item: item[
            "hole_depth_m"
        ],
        reverse=True,
    )

    selected = []

    for candidate in candidates:

        too_close = False

        for existing in selected:

            separation = float(
                np.hypot(
                    candidate["x_center"]
                    - existing["x_center"],
                    candidate["y_center"]
                    - existing["y_center"],
                )
            )

            if separation < 0.8:
                too_close = True
                break

        if not too_close:
            selected.append(
                candidate
            )

    return selected


def raw_trace(
    data: np.ndarray,
    visible: bool,
) -> go.Scatter3d:

    display = data

    if len(display) > RAW_MAX_POINTS:

        stride = int(
            np.ceil(
                len(display)
                / RAW_MAX_POINTS
            )
        )

        display = display[
            ::stride
        ]

    return go.Scatter3d(
        x=display[:, 0],
        y=display[:, 1],
        z=display[:, 2],
        mode="markers",
        marker={
            "size": 1.5,
            "opacity": 0.32,
            "color": display[:, 2],
            "colorscale": "Turbo",
            "showscale": True,
            "colorbar": {
                "title": "Elevation Z (m)",
                "len": 0.48,
            },
        },
        name="REAL LiDAR ENVIRONMENT",
        legendgroup="RAW",
        showlegend=True,
        visible=visible,
        hovertemplate=(
            "<b>LiDAR</b>"
            "<br>X: %{x:.2f} m"
            "<br>Y: %{y:.2f} m"
            "<br>Z: %{z:.2f} m"
            "<extra></extra>"
        ),
    )


def mesh_by_traversability(
    data: np.ndarray,
    leaves: list[dict],
    records: dict,
    visible: bool,
):
    groups = defaultdict(
        lambda: {
            "x": [],
            "y": [],
            "z": [],
            "i": [],
            "j": [],
            "k": [],
        }
    )

    for leaf in leaves:

        parent_id = leaf.get(
            "_parent_cell_id"
        )

        record = records.get(
            parent_id
        )

        if record is None:
            continue

        label = record[
            "traversability"
        ]

        indices = np.asarray(
            leaf["point_indices"],
            dtype=np.int64,
        )

        if len(indices) == 0:
            continue

        z = float(
            np.mean(
                data[
                    indices,
                    2,
                ]
            )
        )

        bbox = leaf["bbox"]

        x0 = float(
            bbox["x_min"]
        )
        x1 = float(
            bbox["x_max"]
        )
        y0 = float(
            bbox["y_min"]
        )
        y1 = float(
            bbox["y_max"]
        )

        group = groups[
            label
        ]

        offset = len(
            group["x"]
        )

        group["x"].extend(
            [x0, x1, x1, x0]
        )

        group["y"].extend(
            [y0, y0, y1, y1]
        )

        group["z"].extend(
            [z, z, z, z]
        )

        group["i"].extend(
            [offset, offset]
        )

        group["j"].extend(
            [
                offset + 1,
                offset + 2,
            ]
        )

        group["k"].extend(
            [
                offset + 2,
                offset + 3,
            ]
        )

    traces = []

    for label in (
        "DRIVABLE",
        "NON-DRIVABLE",
        "SPARSE / UNKNOWN",
    ):

        group = groups[
            label
        ]

        if not group["x"]:
            continue

        traces.append(
            go.Mesh3d(
                x=group["x"],
                y=group["y"],
                z=group["z"],
                i=group["i"],
                j=group["j"],
                k=group["k"],
                color=TRAVERSABILITY_COLORS[
                    label
                ],
                opacity=0.42,
                flatshading=True,
                lighting={
                    "ambient": 0.88,
                    "diffuse": 0.65,
                    "specular": 0.08,
                },
                name=label,
                legendgroup="TRAVERSABILITY",
                showlegend=True,
                visible=visible,
                hovertemplate=(
                    f"<b>{label}</b>"
                    "<br>Adaptive terrain surface"
                    "<extra></extra>"
                ),
            )
        )

    return traces


def mesh_by_resolution(
    data: np.ndarray,
    leaves: list[dict],
    visible: bool,
):
    groups = defaultdict(
        lambda: {
            "x": [],
            "y": [],
            "z": [],
            "i": [],
            "j": [],
            "k": [],
        }
    )

    for leaf in leaves:

        resolution = round(
            float(
                leaf["cell_size"]
            ),
            6,
        )

        if resolution not in RESOLUTION_COLORS:
            continue

        indices = np.asarray(
            leaf["point_indices"],
            dtype=np.int64,
        )

        if len(indices) == 0:
            continue

        z = float(
            np.mean(
                data[
                    indices,
                    2,
                ]
            )
        )

        bbox = leaf["bbox"]

        x0 = float(
            bbox["x_min"]
        )
        x1 = float(
            bbox["x_max"]
        )
        y0 = float(
            bbox["y_min"]
        )
        y1 = float(
            bbox["y_max"]
        )

        group = groups[
            resolution
        ]

        offset = len(
            group["x"]
        )

        group["x"].extend(
            [x0, x1, x1, x0]
        )

        group["y"].extend(
            [y0, y0, y1, y1]
        )

        group["z"].extend(
            [z, z, z, z]
        )

        group["i"].extend(
            [offset, offset]
        )

        group["j"].extend(
            [
                offset + 1,
                offset + 2,
            ]
        )

        group["k"].extend(
            [
                offset + 2,
                offset + 3,
            ]
        )

    traces = []

    for resolution in RESOLUTIONS:

        group = groups[
            resolution
        ]

        if not group["x"]:
            continue

        label = RESOLUTION_LABELS[
            resolution
        ]

        traces.append(
            go.Mesh3d(
                x=group["x"],
                y=group["y"],
                z=group["z"],
                i=group["i"],
                j=group["j"],
                k=group["k"],
                color=RESOLUTION_COLORS[
                    resolution
                ],
                opacity=0.48,
                flatshading=True,
                name=f"Resolution {label}",
                legendgroup="RESOLUTION",
                showlegend=True,
                visible=visible,
                hovertemplate=(
                    f"<b>Resolution {label}</b>"
                    "<br>Adaptive map cell"
                    "<extra></extra>"
                ),
            )
        )

    return traces


def traversability_markers(
    records: dict,
    visible: bool,
) -> list[go.Scatter3d]:

    grouped = defaultdict(
        lambda: {
            "x": [],
            "y": [],
            "z": [],
            "customdata": [],
        }
    )

    for record in records.values():

        label = record[
            "traversability"
        ]

        grouped[
            label
        ]["x"].append(
            record["x_center"]
        )

        grouped[
            label
        ]["y"].append(
            record["y_center"]
        )

        grouped[
            label
        ]["z"].append(
            record["z_mean"]
            + 0.07
        )

        grouped[
            label
        ]["customdata"].append(
            [
                record["slope"],
                record["roughness"],
                record["terrain_complexity"],
                record["terrain_confidence"],
                record["terrain_priority"],
                record["point_count"],
            ]
        )

    traces = []

    for label in (
        "DRIVABLE",
        "NON-DRIVABLE",
        "SPARSE / UNKNOWN",
    ):

        group = grouped[
            label
        ]

        if not group["x"]:
            continue

        traces.append(
            go.Scatter3d(
                x=group["x"],
                y=group["y"],
                z=group["z"],
                mode="markers",
                marker={
                    "size": 5,
                    "symbol": TRAVERSABILITY_SYMBOLS[
                        label
                    ],
                    "color": TRAVERSABILITY_COLORS[
                        label
                    ],
                },
                customdata=group[
                    "customdata"
                ],
                name=label,
                legendgroup="TRAVERSABILITY_MARKERS",
                showlegend=True,
                visible=visible,
                hovertemplate=(
                    f"<b>{label}</b>"
                    "<br>Slope: %{customdata[0]:.2f}°"
                    "<br>Roughness: %{customdata[1]:.3f} m"
                    "<br>Complexity: %{customdata[2]:.3f}"
                    "<br>Confidence: %{customdata[3]:.3f}"
                    "<br>Priority: %{customdata[4]:.3f}"
                    "<br>Points: %{customdata[5]}"
                    "<extra></extra>"
                ),
            )
        )

    return traces


def hole_trace(
    candidates: list[dict],
    visible: bool,
) -> go.Scatter3d:

    return go.Scatter3d(
        x=[
            item["x_center"]
            for item in candidates
        ],
        y=[
            item["y_center"]
            for item in candidates
        ],
        z=[
            item["z_mean"] + 0.14
            for item in candidates
        ],
        mode="markers+text",
        marker={
            "size": 9,
            "symbol": "diamond",
            "color": "#f59e0b",
        },
        text=[
            f"HOLE\n{item['hole_depth_m']:.2f} m"
            for item in candidates
        ],
        textposition="top center",
        name="HOLE / DEPRESSION CANDIDATE",
        legendgroup="HAZARD",
        showlegend=True,
        visible=visible,
        customdata=[
            [
                item["hole_depth_m"],
                item["terrain_complexity"],
                item["terrain_confidence"],
            ]
            for item in candidates
        ],
        hovertemplate=(
            "<b>HOLE / DEPRESSION CANDIDATE</b>"
            "<br>Estimated depth: %{customdata[0]:.3f} m"
            "<br>Terrain complexity: %{customdata[1]:.3f}"
            "<br>Confidence: %{customdata[2]:.3f}"
            "<extra></extra>"
        ),
    )


def vehicle_trace() -> go.Scatter3d:

    return go.Scatter3d(
        x=[0.0],
        y=[0.0],
        z=[0.0],
        mode="markers+text",
        marker={
            "size": 11,
            "symbol": "diamond",
            "color": "#60a5fa",
        },
        text=["VEHICLE"],
        textposition="top center",
        name="VEHICLE / LiDAR",
        legendgroup="VEHICLE",
        showlegend=True,
        visible=True,
    )


def frame_traces(
    result: dict,
) -> list:

    # Fixed 11-trace layout:
    # 0 raw environment
    # 1-5 adaptive resolution surfaces
    # 6-8 traversability surfaces
    # 9-11 traversability markers
    # 12 hole candidates
    # 13 vehicle
    #
    # The 14-trace layout is used for deterministic frame replay.

    traces = []

    traces.append(
        raw_trace(
            result["data"],
            visible=True,
        )
    )

    resolution_figure = go.Figure()

    resolution_surface_traces = (
        mesh_by_resolution(
            result["data"],
            result["adaptive_leaves"],
            visible=True,
        )
    )

    for trace in resolution_surface_traces:
        traces.append(
            trace
        )

    while len(traces) < 6:
        traces.append(
            go.Mesh3d(
                x=[],
                y=[],
                z=[],
                i=[],
                j=[],
                k=[],
                visible=False,
                showlegend=False,
            )
        )

    traversal_surface_traces = (
        mesh_by_traversability(
            result["data"],
            result["adaptive_leaves"],
            result["terrain_records"],
            visible=True,
        )
    )

    for trace in traversal_surface_traces:
        traces.append(
            trace
        )

    while len(traces) < 9:
        traces.append(
            go.Mesh3d(
                x=[],
                y=[],
                z=[],
                i=[],
                j=[],
                k=[],
                visible=False,
                showlegend=False,
            )
        )

    marker_traces = traversability_markers(
        result["terrain_records"],
        visible=True,
    )

    for trace in marker_traces:
        traces.append(
            trace
        )

    while len(traces) < 12:
        traces.append(
            go.Scatter3d(
                x=[],
                y=[],
                z=[],
                visible=False,
                showlegend=False,
            )
        )

    traces.append(
        hole_trace(
            result["hole_candidates"],
            visible=True,
        )
    )

    traces.append(
        vehicle_trace()
    )

    while len(traces) < 14:
        traces.append(
            go.Scatter3d(
                x=[],
                y=[],
                z=[],
                visible=False,
                showlegend=False,
            )
        )

    return traces


def driving_visibility(
    count: int = 14,
) -> list[bool]:

    visibility = [
        False
        for _ in range(count)
    ]

    # Raw environment as context.
    visibility[0] = True

    # Resolution layers hidden in clean driving view.

    # Traversability surfaces.
    visibility[6] = True
    visibility[7] = True
    visibility[8] = True

    # Small diagnostic traversability markers.
    visibility[9] = True
    visibility[10] = True
    visibility[11] = True

    # Hole candidate.
    visibility[12] = True

    # Vehicle.
    visibility[13] = True

    return visibility


def resolution_visibility(
    count: int = 14,
) -> list[bool]:

    visibility = [
        False
        for _ in range(count)
    ]

    visibility[0] = True

    for index in range(
        1,
        6,
    ):
        visibility[index] = True

    visibility[13] = True

    return visibility


def raw_visibility(
    count: int = 14,
) -> list[bool]:

    visibility = [
        False
        for _ in range(count)
    ]

    visibility[0] = True

    visibility[13] = True

    return visibility


def diagnostic_visibility(
    count: int = 14,
) -> list[bool]:

    visibility = [
        True
        for _ in range(count)
    ]

    return visibility


def frame_title(
    result: dict,
) -> str:

    counts = result[
        "traversability_counts"
    ]

    return (
        f"<b>REAL RELLIS-3D</b> | "
        f"SEQUENCE {SEQUENCE} | "
        f"FRAME {result['frame_name']}"
        f"<br>"
        f"<span style='color:#22c55e'>DRIVABLE {counts['DRIVABLE']}</span> "
        f"| <span style='color:#ef4444'>"
        f"NON-DRIVABLE {counts['NON-DRIVABLE']}</span> "
        f"| <span style='color:#94a3b8'>"
        f"UNKNOWN {counts['SPARSE / UNKNOWN']}</span> "
        f"| <span style='color:#f59e0b'>"
        f"HOLES {len(result['hole_candidates'])}</span>"
    )


def process_frame(
    frame_name: str,
) -> dict:

    data = load_frame(
        frame_name
    )

    (
        ground_ids,
        cells,
        base_resolution_by_cell,
        analyzed_cells,
    ) = build_parent_inputs(
        data
    )

    terrain_records = (
        build_terrain_records(
            data=data,
            cells=cells,
            analyzed_cells=analyzed_cells,
        )
    )

    hole_candidates = detect_hole_candidates(
        terrain_records
    )

    adaptive_leaves = build_adaptive_leaves(
        data=data,
        cells=cells,
        base_resolution_by_cell=(
            base_resolution_by_cell
        ),
        analyzed_cells=analyzed_cells,
    )

    counts = {
        "DRIVABLE": 0,
        "NON-DRIVABLE": 0,
        "SPARSE / UNKNOWN": 0,
    }

    for record in terrain_records.values():

        label = record[
            "traversability"
        ]

        if label not in counts:
            label = "SPARSE / UNKNOWN"

        counts[
            label
        ] += 1

    return {
        "frame_name": frame_name,
        "data": data,
        "ground_points": int(
            len(ground_ids)
        ),
        "parent_cells": int(
            len(cells)
        ),
        "adaptive_leaves": adaptive_leaves,
        "terrain_records": terrain_records,
        "hole_candidates": hole_candidates,
        "traversability_counts": counts,
    }


def main() -> None:

    print(
        "SMART VEHICLE REAL-LiDAR DRIVING SIMULATION"
    )

    print()
    print(
        "Processing six real RELLIS-3D frames..."
    )

    processed = []

    for frame_name in FRAME_NAMES:

        print(
            f"  Processing {frame_name}..."
        )

        result = process_frame(
            frame_name
        )

        processed.append(
            result
        )

        print(
            f"    LiDAR points     : {len(result['data'])}"
        )

        print(
            f"    Ground points    : {result['ground_points']}"
        )

        print(
            f"    Adaptive cells   : {len(result['adaptive_leaves'])}"
        )

        print(
            f"    Hole candidates  : {len(result['hole_candidates'])}"
        )

    initial_traces = frame_traces(
        processed[0]
    )

    if len(initial_traces) != 14:
        raise RuntimeError(
            "Unexpected trace count."
        )

    figure = go.Figure(
        data=initial_traces
    )

    initial_view = driving_visibility()

    for index, trace in enumerate(
        figure.data
    ):

        trace.visible = (
            initial_view[index]
        )

    frames = []

    for result in processed:

        traces = frame_traces(
            result
        )

        for trace in traces:
            trace.visible = True

        frames.append(
            go.Frame(
                name=result[
                    "frame_name"
                ],
                data=traces,
                traces=list(
                    range(14)
                ),
                layout=go.Layout(
                    title={
                        "text": frame_title(
                            result
                        )
                    }
                ),
            )
        )

    figure.frames = frames

    slider_steps = []

    for result in processed:

        slider_steps.append(
            {
                "label": result[
                    "frame_name"
                ],
                "method": "animate",
                "args": [
                    [
                        result[
                            "frame_name"
                        ]
                    ],
                    {
                        "mode": "immediate",
                        "frame": {
                            "duration": 650,
                            "redraw": True,
                        },
                        "transition": {
                            "duration": 100,
                        },
                    },
                ],
            }
        )

    figure.update_layout(
        title={
            "text": frame_title(
                processed[0]
            )
        },

        scene={
            "xaxis": {
                "title": "FORWARD X (m)",
                "range": [
                    X_MIN,
                    X_MAX,
                ],
                "showspikes": False,
            },
            "yaxis": {
                "title": "LATERAL Y (m)",
                "range": [
                    Y_MIN,
                    Y_MAX,
                ],
                "showspikes": False,
            },
            "zaxis": {
                "title": "ELEVATION Z (m)",
                "showspikes": False,
            },

            "aspectmode": "manual",

            "aspectratio": {
                "x": 2.4,
                "y": 1.35,
                "z": 0.62,
            },

            "camera": {
                "projection": {
                    "type": "perspective",
                },

                # Vehicle-mounted forward camera.
                "eye": {
                    "x": -1.45,
                    "y": 0.0,
                    "z": 0.22,
                },

                "center": {
                    "x": 0.85,
                    "y": 0.0,
                    "z": -0.10,
                },

                "up": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 1.0,
                },
            },
        },

        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0.01,
                "y": 1.10,

                "buttons": [
                    {
                        "label": "DRIVING VIEW",
                        "method": "update",
                        "args": [
                            {
                                "visible":
                                    driving_visibility()
                            }
                        ],
                    },

                    {
                        "label": "RESOLUTION VIEW",
                        "method": "update",
                        "args": [
                            {
                                "visible":
                                    resolution_visibility()
                            }
                        ],
                    },

                    {
                        "label": "RAW LiDAR",
                        "method": "update",
                        "args": [
                            {
                                "visible":
                                    raw_visibility()
                            }
                        ],
                    },

                    {
                        "label": "DIAGNOSTIC",
                        "method": "update",
                        "args": [
                            {
                                "visible":
                                    diagnostic_visibility()
                            }
                        ],
                    },

                    {
                        "label": "▶ PLAY",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": True,
                                "mode": "immediate",
                                "frame": {
                                    "duration": 650,
                                    "redraw": True,
                                },
                                "transition": {
                                    "duration": 100,
                                },
                            },
                        ],
                    },

                    {
                        "label": "⏸ PAUSE",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {
                                    "duration": 0,
                                    "redraw": False,
                                },
                            },
                        ],
                    },
                ],
            }
        ],

        sliders=[
            {
                "active": 0,
                "x": 0.10,
                "y": 0.02,
                "len": 0.80,
                "currentvalue": {
                    "prefix": "REAL LiDAR FRAME: "
                },
                "steps": slider_steps,
            }
        ],

        legend={
            "orientation": "h",
            "x": 0.50,
            "xanchor": "center",
            "y": -0.08,
            "yanchor": "top",
            "bgcolor": "rgba(15,23,42,0.82)",
            "font": {
                "size": 12,
                "color": "#e5e7eb",
            },
        },

        paper_bgcolor="#020617",

        plot_bgcolor="#020617",

        font={
            "color": "#e5e7eb",
        },

        margin={
            "l": 0,
            "r": 0,
            "t": 105,
            "b": 115,
        },

        hovermode="closest",

        template="plotly_dark",
    )

    # Add a small operator note as an annotation.
    figure.add_annotation(
        x=0.01,
        y=0.01,
        xref="paper",
        yref="paper",
        text=(
            "GREEN = DRIVABLE   |   "
            "RED = NON-DRIVABLE   |   "
            "GREY = UNKNOWN   |   "
            "ORANGE = HOLE / DEPRESSION CANDIDATE"
        ),
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font={
            "size": 11,
            "color": "#cbd5e1",
        },
        bgcolor="rgba(15,23,42,0.75)",
    )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    output_path = os.path.join(
        RESULTS_DIR,
        (
            f"smart_vehicle_driving_"
            f"simulation_{SEQUENCE}.html"
        ),
    )

    figure.write_html(
        output_path,
        include_plotlyjs=True,
        full_html=True,
    )

    print()
    print(
        "Output:",
        output_path,
    )

    print()
    print(
        "DEFAULT VIEW: DRIVING VIEW"
    )

    print(
        "The operator sees:"
    )

    print(
        "  REAL LiDAR environment"
    )

    print(
        "  Drivable / Non-drivable / Unknown terrain"
    )

    print(
        "  Hole / depression candidates"
    )

    print(
        "  Adaptive terrain resolution"
    )

    print(
        "  Vehicle-forward perspective"
    )

    print(
        "  Six real LiDAR frames for replay"
    )

    print()
    print(
        "Important: this is an offline replay using real LiDAR data."
    )

    print(
        "It is not a measured live vehicle sensor loop."
    )

    print()
    print(
        "Smart vehicle visualization complete."
    )


if __name__ == "__main__":
    main()
