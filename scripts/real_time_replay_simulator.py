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

RAW_MAX_POINTS = 12000

HOLE_MIN_DEPTH_M = 0.10
HOLE_MIN_NEIGHBORS = 3

MAX_X = 30.0
MIN_Y = -10.0
MAX_Y = 10.0

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
    0.05: "#2563eb",
    0.10: "#16a34a",
    0.125: "#eab308",
    0.25: "#f97316",
    0.50: "#dc2626",
}

TRAVERSABILITY_STYLE = {
    "DRIVABLE": {
        "color": "#22c55e",
        "symbol": "circle",
    },
    "NON-DRIVABLE": {
        "color": "#ef4444",
        "symbol": "x",
    },
    "SPARSE / UNKNOWN": {
        "color": "#94a3b8",
        "symbol": "diamond",
    },
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
        result = float(value)
    except (TypeError, ValueError):
        return default

    return result


def build_parent_inputs(
    data: np.ndarray,
):
    ground_mask, point_id, _ = extract_ground_mask(
        data
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
            base_resolution_by_cell[cell_id]
        )

        request = analyzed_cells[
            cell_id
        ]["refinement_request"]

        requested_resolution = float(
            request["requested_resolution"]
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

        points = data[
            point_indices
        ]

        z_values = points[
            :,
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
                terrain.get("slope")
            ),
            "roughness": safe_float(
                terrain.get("roughness")
            ),
            "elevation_variation": safe_float(
                terrain.get(
                    "elevation_variation"
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

    candidates = []

    if len(records) < (
        HOLE_MIN_NEIGHBORS + 1
    ):
        return candidates

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

        local_reference = float(
            np.median(
                np.asarray(
                    neighbours,
                    dtype=np.float64,
                )
            )
        )

        depth = (
            local_reference
            - record["z_mean"]
        )

        if depth < HOLE_MIN_DEPTH_M:
            continue

        candidates.append(
            {
                **record,
                "local_reference_z": local_reference,
                "hole_depth_m": float(depth),
            }
        )

    candidates.sort(
        key=lambda item: item["hole_depth_m"],
        reverse=True,
    )

    # Avoid marking the same local depression many times.
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


def mesh_trace_for_resolution(
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

        if not np.isclose(
            float(leaf["cell_size"]),
            resolution,
            rtol=0.0,
            atol=1e-6,
        ):
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

    color = RESOLUTION_COLORS[
        resolution
    ]

    return go.Mesh3d(
        x=x_values,
        y=y_values,
        z=z_values,
        i=i_values,
        j=j_values,
        k=k_values,
        color=color,
        opacity=0.68,
        flatshading=True,
        lighting={
            "ambient": 0.8,
            "diffuse": 0.7,
            "specular": 0.15,
        },
        hovertemplate=(
            f"<b>Adaptive terrain cell</b>"
            f"<br>Resolution: {label}"
            "<extra></extra>"
        ),
        name=f"Resolution {label}",
        legendgroup="RESOLUTION",
        showlegend=True,
        visible=visible,
    )


def traversability_trace(
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
            + 0.035
        )

        customdata.append(
            [
                safe_float(
                    record["slope"]
                ),
                safe_float(
                    record["roughness"]
                ),
                safe_float(
                    record[
                        "terrain_complexity"
                    ]
                ),
                safe_float(
                    record[
                        "terrain_confidence"
                    ]
                ),
                safe_float(
                    record[
                        "terrain_priority"
                    ]
                ),
                record[
                    "point_count"
                ],
            ]
        )

    style = TRAVERSABILITY_STYLE[
        label
    ]

    return go.Scatter3d(
        x=x_values,
        y=y_values,
        z=z_values,
        mode="markers",
        marker={
            "size": 5,
            "symbol": style[
                "symbol"
            ],
            "color": style[
                "color"
            ],
        },
        customdata=customdata,
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
        name=label,
        legendgroup="TRAVERSABILITY",
        showlegend=True,
        visible=visible,
    )


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
            item["z_mean"] + 0.10
            for item in candidates
        ],
        mode="markers+text",
        marker={
            "size": 8,
            "symbol": "diamond",
            "color": "#f59e0b",
        },
        text=[
            f"HOLE {item['hole_depth_m']:.2f} m"
            for item in candidates
        ],
        textposition="top center",
        customdata=[
            [
                item["hole_depth_m"],
                item["local_reference_z"],
                item["terrain_complexity"],
                item["terrain_confidence"],
            ]
            for item in candidates
        ],
        hovertemplate=(
            "<b>Surface depression / hole candidate</b>"
            "<br>Depth estimate: %{customdata[0]:.3f} m"
            "<br>Local reference Z: %{customdata[1]:.3f} m"
            "<br>Complexity: %{customdata[2]:.3f}"
            "<br>Confidence: %{customdata[3]:.3f}"
            "<extra></extra>"
        ),
        name="Hole / Depression Candidate",
        legendgroup="HAZARD",
        showlegend=True,
        visible=visible,
    )


def vehicle_trace() -> go.Scatter3d:

    return go.Scatter3d(
        x=[0.0],
        y=[0.0],
        z=[0.0],
        mode="markers+text",
        marker={
            "size": 10,
            "symbol": "diamond",
            "color": "#1d4ed8",
        },
        text=["VEHICLE / LiDAR"],
        textposition="top center",
        name="Vehicle / LiDAR",
        legendgroup="VEHICLE",
        showlegend=True,
        visible=True,
    )


def raw_trace(
    data: np.ndarray,
    visible: bool,
) -> go.Scatter3d:

    display_data = data

    if len(
        display_data
    ) > RAW_MAX_POINTS:

        step = int(
            np.ceil(
                len(display_data)
                / RAW_MAX_POINTS
            )
        )

        display_data = display_data[
            ::step
        ]

    return go.Scatter3d(
        x=display_data[:, 0],
        y=display_data[:, 1],
        z=display_data[:, 2],
        mode="markers",
        marker={
            "size": 1.5,
            "opacity": 0.40,
            "color": display_data[:, 2],
            "colorscale": "Viridis",
            "showscale": True,
            "colorbar": {
                "title": "Elevation Z (m)",
                "len": 0.5,
            },
        },
        name="Raw LiDAR",
        legendgroup="RAW",
        showlegend=True,
        visible=visible,
    )


def make_frame_traces(
    data: np.ndarray,
    leaves: list[dict],
    records: dict,
    holes: list[dict],
) -> list[go.BaseTraceType]:

    traces = []

    # 0-4: resolution surfaces
    for index, resolution in enumerate(
        RESOLUTIONS
    ):
        traces.append(
            mesh_trace_for_resolution(
                data=data,
                leaves=leaves,
                resolution=resolution,
                visible=True,
            )
        )

    # 5-7: traversability
    for label in TRAVERSABILITY_STYLE:
        traces.append(
            traversability_trace(
                records=records,
                label=label,
                visible=True,
            )
        )

    # 8: hole candidates
    traces.append(
        hole_trace(
            holes,
            visible=True,
        )
    )

    # 9: vehicle
    traces.append(
        vehicle_trace()
    )

    # 10: raw point cloud
    traces.append(
        raw_trace(
            data,
            visible=False,
        )
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
        f"Adaptive cells: "
        f"{len(result['adaptive_leaves'])} | "
        f"Drivable: "
        f"{counts['DRIVABLE']} | "
        f"Non-drivable: "
        f"{counts['NON-DRIVABLE']} | "
        f"Unknown: "
        f"{counts['SPARSE / UNKNOWN']} | "
        f"Hole candidates: "
        f"{len(result['hole_candidates'])}"
    )


def main() -> None:

    print(
        "CLEAN REAL-FRAME DRIVING REPLAY"
    )

    print()
    print(
        "Processing six actual RELLIS-3D scans..."
    )

    processed = []

    for frame_name in FRAME_NAMES:

        print(
            f"  Processing frame {frame_name}..."
        )

        data = load_frame(
            frame_name
        )

        (
            ground_point_ids,
            cells,
            base_resolution_by_cell,
            analyzed_cells,
        ) = build_parent_inputs(
            data
        )

        records = build_terrain_records(
            data=data,
            cells=cells,
            analyzed_cells=analyzed_cells,
        )

        holes = detect_hole_candidates(
            records
        )

        leaves = build_adaptive_leaves(
            data=data,
            cells=cells,
            base_resolution_by_cell=(
                base_resolution_by_cell
            ),
            analyzed_cells=analyzed_cells,
        )

        traversability_counts = {
            label: 0
            for label
            in TRAVERSABILITY_STYLE
        }

        for record in records.values():

            label = record[
                "traversability"
            ]

            if label not in traversability_counts:
                label = "SPARSE / UNKNOWN"

            traversability_counts[
                label
            ] += 1

        result = {
            "frame_name": frame_name,
            "data": data,
            "ground_points": len(
                ground_point_ids
            ),
            "parent_cells": len(
                cells
            ),
            "adaptive_leaves": leaves,
            "terrain_records": records,
            "hole_candidates": holes,
            "traversability_counts": (
                traversability_counts
            ),
        }

        processed.append(
            result
        )

        print(
            f"    LiDAR points: {len(data)}"
        )

        print(
            f"    Ground points: "
            f"{len(ground_point_ids)}"
        )

        print(
            f"    Adaptive cells: "
            f"{len(leaves)}"
        )

        print(
            f"    Hole candidates: "
            f"{len(holes)}"
        )

    base_traces = make_frame_traces(
        data=processed[0]["data"],
        leaves=processed[0]["adaptive_leaves"],
        records=processed[0]["terrain_records"],
        holes=processed[0]["hole_candidates"],
    )

    figure = go.Figure(
        data=base_traces
    )

    # Only the first frame is visible initially.
    # Base traces 0-9 are the adaptive driving view;
    # trace 10 is raw LiDAR and remains hidden.
    for index, trace in enumerate(
        figure.data
    ):

        if index == 9:
            trace.visible = True

        elif index == 10:
            trace.visible = False

        else:
            trace.visible = True

    frames = []

    for result in processed:

        frame_traces = make_frame_traces(
            data=result["data"],
            leaves=result["adaptive_leaves"],
            records=result["terrain_records"],
            holes=result["hole_candidates"],
        )

        frames.append(
            go.Frame(
                name=result[
                    "frame_name"
                ],
                data=frame_traces,
                traces=list(
                    range(
                        11
                    )
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
                            "duration": 700,
                            "redraw": True,
                        },
                        "transition": {
                            "duration": 120,
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
                    0,
                    MAX_X,
                ],
                "showspikes": False,
            },
            "yaxis": {
                "title": "LATERAL Y (m)",
                "range": [
                    MIN_Y,
                    MAX_Y,
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
                "y": 1.25,
                "z": 0.60,
            },
            "camera": {
                "projection": {
                    "type": "perspective",
                },
                "eye": {
                    "x": -1.60,
                    "y": 0.0,
                    "z": 0.32,
                },
                "center": {
                    "x": 0.40,
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

        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0.01,
                "y": 1.10,
                "buttons": [
                    {
                        "label": "▶ PLAY",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": True,
                                "mode": "immediate",
                                "frame": {
                                    "duration": 700,
                                    "redraw": True,
                                },
                                "transition": {
                                    "duration": 120,
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
                    {
                        "label": "DRIVING VIEW",
                        "method": "update",
                        "args": [
                            {
                                "visible": [
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
                                    False,
                                ]
                            }
                        ],
                    },
                    {
                        "label": "RAW LiDAR",
                        "method": "update",
                        "args": [
                            {
                                "visible": [
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
                                    True,
                                ]
                            }
                        ],
                    },
                ],
            }
        ],

        sliders=[
            {
                "active": 0,
                "x": 0.08,
                "y": 0.02,
                "len": 0.84,
                "currentvalue": {
                    "prefix": "Real LiDAR scan: "
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
        },

        margin={
            "l": 0,
            "r": 0,
            "t": 105,
            "b": 125,
        },

        hovermode="closest",

        template="plotly_white",
    )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    output_path = os.path.join(
        RESULTS_DIR,
        (
            f"real_time_vehicle_driving_"
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
        "Default view:"
    )

    print(
        "  Vehicle-forward adaptive terrain view"
    )

    print()
    print(
        "Interactive controls:"
    )

    print(
        "  PLAY / PAUSE six real LiDAR frames"
    )

    print(
        "  DRIVING VIEW"
    )

    print(
        "  RAW LiDAR"
    )

    print(
        "  Resolution-coded adaptive terrain"
    )

    print(
        "  Drivable / Non-drivable / Unknown"
    )

    print(
        "  Hole / Depression candidates"
    )

    print()
    print(
        "Important: this is an offline replay of real scans, "
        "not a measured live vehicle system."
    )

    print()
    print(
        "Clean driving replay complete."
    )


if __name__ == "__main__":
    main()
