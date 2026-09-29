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

RAW_MAX_POINTS = 14000

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
        ground_mask,
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
            base_resolution_by_cell[cell_id]
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

        bbox = cell["bbox"]

        x_center = (
            float(bbox["x_min"])
            + float(bbox["x_max"])
        ) / 2.0

        y_center = (
            float(bbox["y_min"])
            + float(bbox["y_max"])
        ) / 2.0

        z_values = data[
            point_indices,
            2,
        ].astype(
            np.float64
        )

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
            "cell_id": cell_id,
            "x_center": x_center,
            "y_center": y_center,
            "z_mean": float(
                np.mean(z_values)
            ),
            "point_indices": point_indices,
            "point_count": int(
                len(point_indices)
            ),
            "slope": safe_float(
                terrain.get("slope")
            ),
            "roughness": safe_float(
                terrain.get("roughness")
            ),
            "terrain_complexity": safe_float(
                terrain.get("terrain_complexity")
            ),
            "terrain_confidence": safe_float(
                terrain.get("terrain_confidence")
            ),
            "terrain_priority": safe_float(
                terrain.get("terrain_priority")
            ),
            "traversability": traversability,
        }

    return records


def classify_ground_points(
    ground_point_ids: np.ndarray,
    terrain_records: dict,
    total_points: int,
) -> dict[str, np.ndarray]:

    point_to_label = np.full(
        total_points,
        "SPARSE / UNKNOWN",
        dtype="<U18",
    )

    for record in terrain_records.values():

        label = record[
            "traversability"
        ]

        for point_id in record[
            "point_indices"
        ]:
            point_to_label[
                int(point_id)
            ] = label

    labels = {
        "DRIVABLE": [],
        "NON-DRIVABLE": [],
        "SPARSE / UNKNOWN": [],
    }

    for point_id in ground_point_ids:

        label = point_to_label[
            int(point_id)
        ]

        if label not in labels:
            label = "SPARSE / UNKNOWN"

        labels[label].append(
            int(point_id)
        )

    return {
        label: np.asarray(
            indices,
            dtype=np.int64,
        )
        for label, indices
        in labels.items()
    }


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


def raw_context_trace(
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
            "size": 1.4,
            "opacity": 0.28,
            "color": "#a7b2c4",
        },
        name="Real LiDAR Environment",
        legendgroup="RAW",
        showlegend=True,
        visible=visible,
        hovertemplate=(
            "LiDAR point"
            "<br>Forward X: %{x:.2f} m"
            "<br>Lateral Y: %{y:.2f} m"
            "<br>Elevation Z: %{z:.2f} m"
            "<extra></extra>"
        ),
    )


def ground_point_trace(
    data: np.ndarray,
    point_ids: np.ndarray,
    label: str,
    visible: bool,
) -> go.Scatter3d:

    if len(point_ids) == 0:

        return go.Scatter3d(
            x=[],
            y=[],
            z=[],
            mode="markers",
            visible=False,
            showlegend=False,
        )

    points = data[
        point_ids
    ]

    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker={
            "size": 2.6,
            "opacity": 0.82,
            "color": (
                TRAVERSABILITY_COLORS[
                    label
                ]
            ),
        },
        name=label,
        legendgroup="DRIVING",
        showlegend=True,
        visible=visible,
        hovertemplate=(
            f"<b>{label}</b>"
            "<br>X: %{x:.2f} m"
            "<br>Y: %{y:.2f} m"
            "<br>Z: %{z:.2f} m"
            "<extra></extra>"
        ),
    )


def make_resolution_mesh(
    data: np.ndarray,
    leaves: list[dict],
    resolution: float,
    visible: bool,
) -> go.Mesh3d:

    x_values = []
    y_values = []
    z_values = []
    i_values = []
    j_values = []
    k_values = []

    for leaf in leaves:

        leaf_resolution = round(
            float(
                leaf["cell_size"]
            ),
            6,
        )

        if not np.isclose(
            leaf_resolution,
            resolution,
            rtol=0.0,
            atol=1e-6,
        ):
            continue

        point_ids = np.asarray(
            leaf["point_indices"],
            dtype=np.int64,
        )

        if len(point_ids) == 0:
            continue

        z = float(
            np.mean(
                data[
                    point_ids,
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

        offset = len(
            x_values
        )

        x_values.extend(
            [
                x0,
                x1,
                x1,
                x0,
            ]
        )

        y_values.extend(
            [
                y0,
                y0,
                y1,
                y1,
            ]
        )

        z_values.extend(
            [
                z,
                z,
                z,
                z,
            ]
        )

        i_values.extend(
            [
                offset,
                offset,
            ]
        )

        j_values.extend(
            [
                offset + 1,
                offset + 2,
            ]
        )

        k_values.extend(
            [
                offset + 2,
                offset + 3,
            ]
        )

    label = RESOLUTION_LABELS[
        resolution
    ]

    return go.Mesh3d(
        x=x_values,
        y=y_values,
        z=z_values,
        i=i_values,
        j=j_values,
        k=k_values,
        color=RESOLUTION_COLORS[
            resolution
        ],
        opacity=0.48,
        flatshading=True,
        lighting={
            "ambient": 0.88,
            "diffuse": 0.65,
            "specular": 0.10,
        },
        name=f"Resolution {label}",
        legendgroup="RESOLUTION",
        showlegend=True,
        visible=visible,
        hovertemplate=(
            f"<b>Adaptive cell</b>"
            f"<br>Resolution: {label}"
            "<extra></extra>"
        ),
    )


def hole_trace(
    candidates: list[dict],
    visible: bool,
) -> go.Scatter3d:

    return go.Scatter3d(
        x=[
            candidate["x_center"]
            for candidate
            in candidates
        ],
        y=[
            candidate["y_center"]
            for candidate
            in candidates
        ],
        z=[
            candidate["z_mean"] + 0.12
            for candidate
            in candidates
        ],
        mode="markers+text",
        marker={
            "size": 9,
            "symbol": "diamond",
            "color": "#f59e0b",
        },
        text=[
            f"HOLE\n"
            f"{candidate['hole_depth_m']:.2f} m"
            for candidate
            in candidates
        ],
        textposition="top center",
        name="Hole / Depression Candidate",
        legendgroup="HAZARD",
        showlegend=True,
        visible=visible,
        customdata=[
            [
                candidate[
                    "hole_depth_m"
                ],
                candidate[
                    "terrain_confidence"
                ],
            ]
            for candidate
            in candidates
        ],
        hovertemplate=(
            "<b>HOLE / DEPRESSION CANDIDATE</b>"
            "<br>Estimated depth: "
            "%{customdata[0]:.3f} m"
            "<br>Terrain confidence: "
            "%{customdata[1]:.3f}"
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
            "size": 12,
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


def process_frame(
    frame_name: str,
) -> dict:

    data = load_frame(
        frame_name
    )

    (
        ground_mask,
        ground_point_ids,
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

    ground_by_label = (
        classify_ground_points(
            ground_point_ids=ground_point_ids,
            terrain_records=terrain_records,
            total_points=len(data),
        )
    )

    adaptive_leaves = (
        build_adaptive_leaves(
            data=data,
            cells=cells,
            base_resolution_by_cell=(
                base_resolution_by_cell
            ),
            analyzed_cells=analyzed_cells,
        )
    )

    hole_candidates = (
        detect_hole_candidates(
            terrain_records
        )
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

    resolution_counts = {
        resolution: 0
        for resolution in RESOLUTIONS
    }

    for leaf in adaptive_leaves:

        resolution = round(
            float(
                leaf["cell_size"]
            ),
            6,
        )

        if resolution in resolution_counts:

            resolution_counts[
                resolution
            ] += 1

    return {
        "frame_name": frame_name,
        "data": data,
        "ground_mask": ground_mask,
        "ground_points": len(
            ground_point_ids
        ),
        "parent_cells": len(
            cells
        ),
        "terrain_records": terrain_records,
        "ground_by_label": ground_by_label,
        "adaptive_leaves": adaptive_leaves,
        "hole_candidates": hole_candidates,
        "traversability_counts": counts,
        "resolution_counts": resolution_counts,
    }


def frame_traces(
    result: dict,
) -> list:

    traces = []

    # 0: Real environment context.
    traces.append(
        raw_context_trace(
            result["data"],
            visible=True,
        )
    )

    # 1-3: Ground status.
    for label in (
        "DRIVABLE",
        "NON-DRIVABLE",
        "SPARSE / UNKNOWN",
    ):

        traces.append(
            ground_point_trace(
                data=result["data"],
                point_ids=result[
                    "ground_by_label"
                ][label],
                label=label,
                visible=True,
            )
        )

    # 4-8: Resolution surfaces.
    for resolution in RESOLUTIONS:

        traces.append(
            make_resolution_mesh(
                data=result["data"],
                leaves=result[
                    "adaptive_leaves"
                ],
                resolution=resolution,
                visible=False,
            )
        )

    # 9: Hole candidate.
    traces.append(
        hole_trace(
            result["hole_candidates"],
            visible=True,
        )
    )

    # 10: Vehicle.
    traces.append(
        vehicle_trace()
    )

    if len(traces) != 11:

        raise RuntimeError(
            f"Expected 11 traces, got {len(traces)}."
        )

    return traces


def driving_visibility() -> list[bool]:

    return [
        True,   # 0 real environment
        True,   # 1 drivable
        True,   # 2 non-drivable
        True,   # 3 unknown
        False,  # 4 5 cm
        False,  # 5 10 cm
        False,  # 6 12.5 cm
        False,  # 7 25 cm
        False,  # 8 50 cm
        True,   # 9 hole
        True,   # 10 vehicle
    ]


def resolution_visibility() -> list[bool]:

    return [
        False,  # 0 raw environment
        False,  # 1 drivable
        False,  # 2 non-drivable
        False,  # 3 unknown
        True,   # 4 5 cm
        True,   # 5 10 cm
        True,   # 6 12.5 cm
        True,   # 7 25 cm
        True,   # 8 50 cm
        True,   # 9 hole
        True,   # 10 vehicle
    ]


def raw_visibility() -> list[bool]:

    return [
        True,   # raw environment
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,   # vehicle reference
    ]


def diagnostic_visibility() -> list[bool]:

    return [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]


def frame_title(
    result: dict,
) -> str:

    counts = result[
        "traversability_counts"
    ]

    holes = len(
        result[
            "hole_candidates"
        ]
    )

    return (
        f"<b>SMART VEHICLE — REAL RELLIS-3D</b>"
        f"<br>"
        f"Sequence {SEQUENCE} | "
        f"Frame {result['frame_name']} | "
        f"Forward range 0–30 m"
        f"<br>"
        f"<span style='color:#22c55e'>"
        f"DRIVABLE: {counts['DRIVABLE']}"
        f"</span>"
        f" &nbsp; "
        f"<span style='color:#ef4444'>"
        f"NON-DRIVABLE: "
        f"{counts['NON-DRIVABLE']}"
        f"</span>"
        f" &nbsp; "
        f"<span style='color:#94a3b8'>"
        f"UNKNOWN: "
        f"{counts['SPARSE / UNKNOWN']}"
        f"</span>"
        f" &nbsp; "
        f"<span style='color:#f59e0b'>"
        f"HOLE CANDIDATES: {holes}"
        f"</span>"
    )


def main() -> None:

    print(
        "SMART VEHICLE REAL-LiDAR ENVIRONMENT SIMULATION"
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
            f"    LiDAR points    : "
            f"{len(result['data'])}"
        )

        print(
            f"    Ground points   : "
            f"{result['ground_points']}"
        )

        print(
            f"    Adaptive cells  : "
            f"{len(result['adaptive_leaves'])}"
        )

        print(
            f"    Holes           : "
            f"{len(result['hole_candidates'])}"
        )

    initial = processed[0]

    figure = go.Figure(
        data=frame_traces(
            initial
        )
    )

    initial_visibility = (
        driving_visibility()
    )

    for index, trace in enumerate(
        figure.data
    ):

        trace.visible = (
            initial_visibility[index]
        )

    frames = []

    for result in processed:

        traces = frame_traces(
            result
        )

        frames.append(
            go.Frame(
                name=result[
                    "frame_name"
                ],
                data=traces,
                traces=list(
                    range(11)
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
                initial
            )
        },

        scene={
            "xaxis": {
                "title": "FORWARD X (m)",
                "range": [
                    X_MIN,
                    X_MAX,
                ],
                "showgrid": True,
                "gridcolor": "#334155",
                "zeroline": False,
                "showspikes": False,
            },
            "yaxis": {
                "title": "LATERAL Y (m)",
                "range": [
                    Y_MIN,
                    Y_MAX,
                ],
                "showgrid": True,
                "gridcolor": "#334155",
                "zeroline": False,
                "showspikes": False,
            },
            "zaxis": {
                "title": "ELEVATION Z (m)",
                "showgrid": True,
                "gridcolor": "#334155",
                "zeroline": False,
                "showspikes": False,
            },

            "bgcolor": "#020617",

            "aspectmode": "manual",

            "aspectratio": {
                "x": 2.55,
                "y": 1.55,
                "z": 0.72,
            },

            "camera": {
                "projection": {
                    "type": "perspective",
                },

                # Default vehicle-mounted forward view:
                # behind LiDAR, looking along +X.
                "eye": {
                    "x": -1.90,
                    "y": 0.0,
                    "z": 0.24,
                },

                "center": {
                    "x": 0.85,
                    "y": 0.0,
                    "z": -0.03,
                },

                "up": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 1.0,
                },
            },
        },

        paper_bgcolor="#020617",

        font={
            "color": "#e5e7eb",
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
                    "prefix": "REAL FRAME: "
                },
                "steps": slider_steps,
            }
        ],

        legend={
            "orientation": "h",
            "x": 0.50,
            "xanchor": "center",
            "y": -0.09,
            "yanchor": "top",
            "bgcolor": "rgba(2,6,23,0.88)",
            "font": {
                "size": 12,
                "color": "#e5e7eb",
            },
        },

        margin={
            "l": 0,
            "r": 0,
            "t": 115,
            "b": 105,
        },

        hovermode="closest",

        template="plotly_dark",
    )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    output_path = os.path.join(
        RESULTS_DIR,
        (
            f"smart_vehicle_real_environment_"
            f"replay_{SEQUENCE}.html"
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
        "Default: vehicle-forward DRIVING VIEW"
    )

    print(
        "Driving view shows:"
    )

    print(
        "  Real LiDAR environment"
    )

    print(
        "  Drivable terrain"
    )

    print(
        "  Non-drivable terrain"
    )

    print(
        "  Sparse / unknown terrain"
    )

    print(
        "  Hole / depression candidates"
    )

    print(
        "  Vehicle / LiDAR position"
    )

    print()
    print(
        "Resolution view shows:"
    )

    for resolution in RESOLUTIONS:

        print(
            f"  {RESOLUTION_LABELS[resolution]}"
        )

    print()
    print(
        "Six real LiDAR frames can be replayed."
    )

    print(
        "Important: this remains an offline replay "
        "of real LiDAR scans, not a measured live vehicle system."
    )

    print()
    print(
        "Smart vehicle real-environment simulation complete."
    )


if __name__ == "__main__":
    main()
