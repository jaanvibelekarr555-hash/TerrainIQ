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

# The map pipeline uses every LiDAR point. Only the visualization
# samples the raw point cloud so the browser remains responsive.
RAW_MAX_POINTS = 18000

# Conservative visualization-only depression candidate threshold.
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

# Resolution colors are deliberately different so the adaptive
# cell-size changes can be seen in the 3D environment.
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


def frame_path(
    frame_name: str,
) -> str:

    return os.path.join(
        DATA_ROOT,
        SEQUENCE,
        "vel_cloud_node_kitti_bin",
        f"{frame_name}.bin",
    )


def load_frame(
    frame_name: str,
) -> np.ndarray:

    path = frame_path(
        frame_name
    )

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
    default: float = 0.0,
) -> float:

    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_parent_data(
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
                ),
                np.nan,
            ),
            "roughness": safe_float(
                terrain.get(
                    "roughness"
                ),
                np.nan,
            ),
            "terrain_complexity": safe_float(
                terrain.get(
                    "terrain_complexity"
                ),
                np.nan,
            ),
            "terrain_confidence": safe_float(
                terrain.get(
                    "terrain_confidence"
                ),
                np.nan,
            ),
            "terrain_priority": safe_float(
                terrain.get(
                    "terrain_priority"
                ),
                np.nan,
            ),
            "traversability": str(
                terrain.get(
                    "traversability",
                    "SPARSE / UNKNOWN",
                )
            ),
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

        neighbour_z = []

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
                neighbour_z.append(
                    other["z_mean"]
                )

        if len(
            neighbour_z
        ) < HOLE_MIN_NEIGHBORS:
            continue

        reference_z = float(
            np.median(
                np.asarray(
                    neighbour_z,
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
                "hole_depth_m": float(
                    depth
                ),
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

        for previous in selected:

            separation = float(
                np.hypot(
                    candidate["x_center"]
                    - previous["x_center"],
                    candidate["y_center"]
                    - previous["y_center"],
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


def sample_raw_points(
    data: np.ndarray,
) -> np.ndarray:

    if len(data) <= RAW_MAX_POINTS:
        return data

    step = int(
        np.ceil(
            len(data)
            / RAW_MAX_POINTS
        )
    )

    return data[
        ::step
    ]


def build_environment_trace(
    data: np.ndarray,
    visible: bool,
) -> go.Scatter3d:

    display = sample_raw_points(
        data
    )

    return go.Scatter3d(
        x=display[:, 0],
        y=display[:, 1],
        z=display[:, 2],
        mode="markers",
        marker={
            "size": 1.6,
            "opacity": 0.72,
            "color": display[:, 2],
            "colorscale": "Turbo",
            "showscale": True,
            "colorbar": {
                "title": "Elevation Z (m)",
                "len": 0.55,
            },
        },
        name="REAL LiDAR ENVIRONMENT",
        legendgroup="RAW",
        showlegend=True,
        visible=visible,
        hovertemplate=(
            "LiDAR point"
            "<br>X: %{x:.2f} m"
            "<br>Y: %{y:.2f} m"
            "<br>Z: %{z:.2f} m"
            "<extra></extra>"
        ),
    )


def add_resolution_surface_traces(
    figure: go.Figure,
    data: np.ndarray,
    leaves: list[dict],
    visible: bool,
) -> list[int]:

    grouped = {
        resolution: {
            "x": [],
            "y": [],
            "z": [],
            "i": [],
            "j": [],
            "k": [],
        }
        for resolution in RESOLUTIONS
    }

    for leaf in leaves:

        resolution = round(
            float(
                leaf["cell_size"]
            ),
            6,
        )

        if resolution not in grouped:
            continue

        point_indices = np.asarray(
            leaf["point_indices"],
            dtype=np.int64,
        )

        if len(point_indices) == 0:
            continue

        z = float(
            np.mean(
                data[
                    point_indices,
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

        group = grouped[
            resolution
        ]

        offset = len(
            group["x"]
        )

        group["x"].extend(
            [
                x0,
                x1,
                x1,
                x0,
            ]
        )

        group["y"].extend(
            [
                y0,
                y0,
                y1,
                y1,
            ]
        )

        group["z"].extend(
            [
                z,
                z,
                z,
                z,
            ]
        )

        group["i"].extend(
            [
                offset,
                offset,
            ]
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

    indices = []

    for resolution in RESOLUTIONS:

        group = grouped[
            resolution
        ]

        if not group["x"]:
            continue

        label = RESOLUTION_LABELS[
            resolution
        ]

        figure.add_trace(
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
                opacity=0.36,
                flatshading=True,
                lighting={
                    "ambient": 0.85,
                    "diffuse": 0.65,
                    "specular": 0.10,
                },
                name=f"Adaptive {label}",
                legendgroup="RESOLUTION",
                showlegend=True,
                visible=visible,
                hovertemplate=(
                    f"Adaptive terrain cell"
                    f"<br>Resolution: {label}"
                    "<extra></extra>"
                ),
            )
        )

        indices.append(
            len(figure.data) - 1
        )

    return indices


def make_traversability_trace(
    records: dict,
    label: str,
    visible: bool,
) -> go.Scatter3d:

    x_values = []
    y_values = []
    z_values = []
    customdata = []

    for record in records.values():

        if record[
            "traversability"
        ] != label:
            continue

        x_values.append(
            record["x_center"]
        )

        y_values.append(
            record["y_center"]
        )

        z_values.append(
            record["z_mean"]
            + 0.06
        )

        customdata.append(
            [
                record["slope"],
                record["roughness"],
                record["terrain_complexity"],
                record["terrain_confidence"],
                record["terrain_priority"],
                record["point_count"],
            ]
        )

    return go.Scatter3d(
        x=x_values,
        y=y_values,
        z=z_values,
        mode="markers",
        marker={
            "size": 5,
            "symbol": (
                TRAVERSABILITY_SYMBOLS[
                    label
                ]
            ),
            "color": (
                TRAVERSABILITY_COLORS[
                    label
                ]
            ),
        },
        customdata=customdata,
        name=label,
        legendgroup="TRAVERSABILITY",
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


def make_hole_trace(
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
            item["z_mean"] + 0.12
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
            "<b>Surface depression / hole candidate</b>"
            "<br>Depth estimate: %{customdata[0]:.3f} m"
            "<br>Complexity: %{customdata[1]:.3f}"
            "<br>Confidence: %{customdata[2]:.3f}"
            "<extra></extra>"
        ),
    )


def make_vehicle_trace() -> go.Scatter3d:

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
        text=["VEHICLE / LiDAR"],
        textposition="top center",
        name="VEHICLE / LiDAR",
        legendgroup="VEHICLE",
        showlegend=True,
        visible=True,
    )


def build_frame_traces(
    result: dict,
    driving_view: bool,
) -> list:

    traces = []

    # Trace 0: full real environment.
    traces.append(
        build_environment_trace(
            result["data"],
            visible=True,
        )
    )

    surface_visible = driving_view

    surface_figure = go.Figure()

    resolution_indices = (
        add_resolution_surface_traces(
            surface_figure,
            result["data"],
            result["adaptive_leaves"],
            visible=surface_visible,
        )
    )

    # Extract traces from the temporary figure.
    traces.extend(
        list(
            surface_figure.data
        )
    )

    # Keep the expected fixed trace slots.
    # Missing resolution groups are replaced with empty traces.
    existing_resolution_traces = len(
        resolution_indices
    )

    for _ in range(
        len(RESOLUTIONS)
        - existing_resolution_traces
    ):

        traces.append(
            go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode="markers",
                visible=False,
                showlegend=False,
            )
        )

    traces.append(
        make_traversability_trace(
            result["terrain_records"],
            "DRIVABLE",
            visible=driving_view,
        )
    )

    traces.append(
        make_traversability_trace(
            result["terrain_records"],
            "NON-DRIVABLE",
            visible=driving_view,
        )
    )

    traces.append(
        make_traversability_trace(
            result["terrain_records"],
            "SPARSE / UNKNOWN",
            visible=driving_view,
        )
    )

    traces.append(
        make_hole_trace(
            result["hole_candidates"],
            visible=driving_view,
        )
    )

    traces.append(
        make_vehicle_trace()
    )

    return traces


def frame_title(
    result: dict,
) -> str:

    counts = result[
        "traversability_counts"
    ]

    return (
        f"REAL RELLIS-3D | "
        f"SEQUENCE {SEQUENCE} | "
        f"FRAME {result['frame_name']}"
        "<br>"
        f"Drivable: {counts['DRIVABLE']} | "
        f"Non-drivable: {counts['NON-DRIVABLE']} | "
        f"Unknown: {counts['SPARSE / UNKNOWN']} | "
        f"Hole candidates: "
        f"{len(result['hole_candidates'])}"
    )


def process_frame(
    frame_name: str,
) -> dict:

    data = load_frame(
        frame_name
    )

    (
        ground_point_ids,
        cells,
        base_resolution_by_cell,
        analyzed_cells,
    ) = build_parent_data(
        data
    )

    terrain_records = (
        build_terrain_records(
            data=data,
            cells=cells,
            analyzed_cells=analyzed_cells,
        )
    )

    hole_candidates = (
        detect_hole_candidates(
            terrain_records
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

    traversability_counts = {
        "DRIVABLE": 0,
        "NON-DRIVABLE": 0,
        "SPARSE / UNKNOWN": 0,
    }

    for record in terrain_records.values():

        label = record[
            "traversability"
        ]

        if label not in traversability_counts:
            label = "SPARSE / UNKNOWN"

        traversability_counts[
            label
        ] += 1

    return {
        "frame_name": frame_name,
        "data": data,
        "ground_points": int(
            len(ground_point_ids)
        ),
        "parent_cells": int(
            len(cells)
        ),
        "adaptive_leaves": adaptive_leaves,
        "terrain_records": terrain_records,
        "hole_candidates": hole_candidates,
        "traversability_counts": (
            traversability_counts
        ),
    }


def build_view_visibility(
    trace_count: int,
    driving_view: bool,
) -> list[bool]:

    visibility = [
        False
        for _ in range(
            trace_count
        )
    ]

    # 0 = raw environment.
    visibility[0] = True

    # 1-5 = adaptive resolutions.
    for index in range(
        1,
        6,
    ):
        visibility[index] = (
            driving_view
        )

    # 6-8 = traversability.
    for index in range(
        6,
        9,
    ):
        visibility[index] = (
            driving_view
        )

    # 9 = holes.
    visibility[9] = (
        driving_view
    )

    # 10 = vehicle.
    visibility[10] = True

    return visibility


def main() -> None:

    print(
        "REAL RELLIS-3D ENVIRONMENT + ADAPTIVE DRIVING SIMULATION"
    )

    print()
    print(
        "The raw point cloud is the environment."
    )

    print(
        "The adaptive 2.5D terrain layer is overlaid on it."
    )

    print()
    print(
        "Processing real frames..."
    )

    processed = []

    for frame_name in FRAME_NAMES:

        print(
            f"  Frame {frame_name}..."
        )

        result = process_frame(
            frame_name
        )

        processed.append(
            result
        )

        print(
            f"    LiDAR points: "
            f"{len(result['data'])}"
        )

        print(
            f"    Ground points: "
            f"{result['ground_points']}"
        )

        print(
            f"    Adaptive cells: "
            f"{len(result['adaptive_leaves'])}"
        )

        print(
            f"    Hole candidates: "
            f"{len(result['hole_candidates'])}"
        )

    # The first frame establishes the trace layout.
    first_traces = build_frame_traces(
        processed[0],
        driving_view=True,
    )

    if len(first_traces) != 11:
        raise RuntimeError(
            "Internal trace-layout error. "
            "Expected exactly 11 traces per frame."
        )

    figure = go.Figure(
        data=first_traces
    )

    # Six fixed trace slots per frame are represented by the 11 traces:
    #
    # 0 raw environment
    # 1-5 resolution layers
    # 6-8 traversability
    # 9 hole candidates
    # 10 vehicle
    #
    # The frame data always use the same 11 trace positions.
    frames = []

    for result in processed:

        frame_traces = build_frame_traces(
            result,
            driving_view=True,
        )

        frames.append(
            go.Frame(
                name=result[
                    "frame_name"
                ],
                data=frame_traces,
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

    trace_count = 11

    driving_visibility = (
        build_view_visibility(
            trace_count,
            driving_view=True,
        )
    )

    raw_visibility = (
        build_view_visibility(
            trace_count,
            driving_view=False,
        )
    )

    # Raw-only view should hide the vehicle and adaptive layers.
    raw_visibility = [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
    ]

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

    initial = processed[0]

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
                "showspikes": False,
                "gridcolor": "#334155",
                "zerolinecolor": "#475569",
            },
            "yaxis": {
                "title": "LATERAL Y (m)",
                "range": [
                    Y_MIN,
                    Y_MAX,
                ],
                "showspikes": False,
                "gridcolor": "#334155",
                "zerolinecolor": "#475569",
            },
            "zaxis": {
                "title": "ELEVATION Z (m)",
                "showspikes": False,
                "gridcolor": "#334155",
                "zerolinecolor": "#475569",
            },
            "bgcolor": "#050505",
            "aspectmode": "manual",
            "aspectratio": {
                "x": 2.25,
                "y": 1.30,
                "z": 0.72,
            },
            "camera": {
                "projection": {
                    "type": "perspective",
                },
                # Vehicle-forward camera:
                # behind LiDAR, looking ahead in +X.
                "eye": {
                    "x": -1.75,
                    "y": 0.0,
                    "z": 0.42,
                },
                "center": {
                    "x": 0.65,
                    "y": 0.0,
                    "z": 0.0,
                },
                "up": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 1.0,
                },
            },
        },

        paper_bgcolor="#050505",
        plot_bgcolor="#050505",

        font={
            "color": "#e5e7eb",
        },

        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0.01,
                "y": 1.12,
                "buttons": [
                    {
                        "label": "DRIVING VIEW",
                        "method": "update",
                        "args": [
                            {
                                "visible":
                                    driving_visibility
                            }
                        ],
                    },
                    {
                        "label": "RAW ENVIRONMENT",
                        "method": "update",
                        "args": [
                            {
                                "visible":
                                    raw_visibility
                            }
                        ],
                    },
                    {
                        "label": "PLAY",
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
                        "label": "PAUSE",
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
                "x": 0.08,
                "y": 0.035,
                "len": 0.84,
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
            "y": -0.12,
            "yanchor": "top",
            "bgcolor": "rgba(5,5,5,0.72)",
        },

        margin={
            "l": 0,
            "r": 0,
            "t": 105,
            "b": 125,
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
            f"real_environment_adaptive_"
            f"driving_replay_{SEQUENCE}.html"
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
        "Default view: DRIVING VIEW"
    )

    print(
        "The view contains:"
    )

    print(
        "  REAL LiDAR environment"
    )

    print(
        "  Adaptive terrain cells"
    )

    print(
        "  Resolution: 5 / 10 / 12.5 / 25 / 50 cm"
    )

    print(
        "  DRIVABLE / NON-DRIVABLE / UNKNOWN"
    )

    print(
        "  Hole / depression candidates"
    )

    print(
        "  Vehicle / LiDAR reference"
    )

    print(
        "  Six-frame real-data replay"
    )

    print()
    print(
        "Important: this is an offline replay of real LiDAR frames."
    )

    print(
        "Hole markers are conservative surface-depression candidates, "
        "not a trained pothole detector."
    )

    print()
    print(
        "Real environment driving simulation complete."
    )


if __name__ == "__main__":
    main()
