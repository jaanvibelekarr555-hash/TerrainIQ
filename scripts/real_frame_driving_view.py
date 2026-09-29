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
FRAME = "000000"

RESULTS_DIR = "results"

PARENT_CELL_SIZE = 1.0

FIXED_RESOLUTION = 0.05
MIN_RESOLUTION = 0.05

NO_ERROR_BOUND = 1_000_000.0

RAW_MAX_POINTS = 18000

# Hole/depression detection is intentionally a conservative
# visualization heuristic, not a certified road-hazard detector.
HOLE_MIN_DEPTH_M = 0.10
HOLE_MIN_NEIGHBORS = 3


RESOLUTION_LABELS = {
    0.05: "5 cm",
    0.10: "10 cm",
    0.125: "12.5 cm",
    0.25: "25 cm",
    0.50: "50 cm",
}


TRAVERSABILITY_LABELS = (
    "DRIVABLE",
    "NON-DRIVABLE",
    "SPARSE / UNKNOWN",
)


def data_file_path() -> str:
    return os.path.join(
        DATA_ROOT,
        SEQUENCE,
        "vel_cloud_node_kitti_bin",
        f"{FRAME}.bin",
    )


def load_data() -> np.ndarray:
    path = data_file_path()

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


def build_parent_data(
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


def build_mode_leaves(
    data: np.ndarray,
    cells: dict,
    base_resolution_by_cell: dict,
    analyzed_cells: dict,
    mode: str,
) -> list[dict]:

    leaves = []

    for cell_id, cell in cells.items():

        base_resolution = float(
            base_resolution_by_cell[cell_id]
        )

        if mode == "FIXED":

            base_resolution = FIXED_RESOLUTION
            requested_resolution = FIXED_RESOLUTION
            priority = 0.0
            reason = "fixed 5 cm baseline"

        elif mode == "DISTANCE":

            requested_resolution = base_resolution
            priority = 0.0
            reason = "distance-only"

        elif mode == "TERRAIN":

            request = analyzed_cells[
                cell_id
            ]["refinement_request"]

            requested_resolution = float(
                request["requested_resolution"]
            )

            priority = float(
                analyzed_cells[
                    cell_id
                ]["terrain_priority"]
            )

            reason = "distance + terrain"

        else:

            raise ValueError(
                "mode must be FIXED, DISTANCE, or TERRAIN."
            )

        refinement_request = {
            "region": {
                "cell_id": cell_id,
                "bbox": cell["bbox"],
            },
            "base_resolution": base_resolution,
            "requested_resolution": requested_resolution,
            "priority": priority,
            "reason": reason,
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


def leaf_z(
    data: np.ndarray,
    leaf: dict,
) -> float:

    point_indices = np.asarray(
        leaf["point_indices"],
        dtype=np.int64,
    )

    if len(point_indices) == 0:
        return float("nan")

    return float(
        np.mean(
            data[
                point_indices,
                2,
            ]
        )
    )


def leaf_center(
    leaf: dict,
) -> tuple[float, float]:

    bbox = leaf["bbox"]

    return (
        (
            float(bbox["x_min"])
            + float(bbox["x_max"])
        ) / 2.0,
        (
            float(bbox["y_min"])
            + float(bbox["y_max"])
        ) / 2.0,
    )


def resolution_counts(
    leaves: list[dict],
) -> dict[float, int]:

    counts: dict[float, int] = {}

    for leaf in leaves:

        resolution = round(
            float(leaf["cell_size"]),
            6,
        )

        counts[resolution] = (
            counts.get(
                resolution,
                0,
            )
            + 1
        )

    return dict(
        sorted(
            counts.items()
        )
    )


def safe_float(value, default=np.nan) -> float:
    """Convert optional numeric terrain values safely.

    Sparse/unknown terrain cells can legitimately expose None for
    geometry-derived quantities such as slope or roughness. Those
    values must not crash the visualization pipeline.
    """

    if value is None:
        return float(default)

    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)

    if not np.isfinite(result):
        return float(default)

    return result


def build_parent_terrain_records(
    data: np.ndarray,
    cells: dict,
    analyzed_cells: dict,
) -> dict:

    records = {}

    for cell_id, cell in cells.items():

        indices = np.asarray(
            cell["point_indices"],
            dtype=np.int64,
        )

        if len(indices) == 0:
            continue

        points = data[
            indices
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

        z_mean = float(
            np.mean(z_values)
        )

        if not np.isfinite(z_mean):
            continue

        records[cell_id] = {
            "cell_id": cell_id,
            "x_center": x_center,
            "y_center": y_center,
            "z_mean": z_mean,
            "z_min": float(
                np.min(z_values)
            ),
            "z_max": float(
                np.max(z_values)
            ),
            "point_count": int(
                len(indices)
            ),
            "slope": safe_float(
                terrain.get("slope")
            ),
            "roughness": safe_float(
                terrain.get("roughness")
            ),
            "elevation_variation": safe_float(
                terrain.get("elevation_variation")
            ),
            "terrain_complexity": safe_float(
                terrain.get("terrain_complexity")
            ),
            "traversability": terrain[
                "traversability"
            ],
            "terrain_confidence": safe_float(
                terrain.get("terrain_confidence")
            ),
            "terrain_priority": safe_float(
                terrain.get("terrain_priority")
            ),
        }

    return records


def detect_hole_candidates(
    terrain_records: dict,
) -> list[dict]:

    record_list = list(
        terrain_records.values()
    )

    if len(record_list) < (
        HOLE_MIN_NEIGHBORS + 1
    ):
        return []

    candidates = []

    for record in record_list:

        neighbor_elevations = []

        x0 = record["x_center"]
        y0 = record["y_center"]

        for other in record_list:

            if other is record:
                continue

            dx = abs(
                other["x_center"]
                - x0
            )

            dy = abs(
                other["y_center"]
                - y0
            )

            # Parent terrain cells are approximately 1 m x 1 m.
            # Nearby cells inside a 1.5 m radius provide the
            # local surface reference.
            if (
                dx <= 1.5
                and dy <= 1.5
            ):

                neighbor_elevations.append(
                    other["z_mean"]
                )

        if len(
            neighbor_elevations
        ) < HOLE_MIN_NEIGHBORS:
            continue

        neighbor_array = np.asarray(
            neighbor_elevations,
            dtype=np.float64,
        )

        neighbor_array = neighbor_array[
            np.isfinite(neighbor_array)
        ]

        if len(neighbor_array) < HOLE_MIN_NEIGHBORS:
            continue

        local_surface = float(
            np.median(neighbor_array)
        )

        record_z = safe_float(
            record.get("z_mean")
        )

        if not np.isfinite(record_z):
            continue

        depth = (
            local_surface
            - record_z
        )

        if depth < HOLE_MIN_DEPTH_M:
            continue

        candidates.append(
            {
                **record,
                "local_surface_z": local_surface,
                "hole_depth_m": float(
                    depth
                ),
                "candidate_type": (
                    "surface depression / hole candidate"
                ),
            }
        )

    return candidates


def build_surface_mesh(
    data: np.ndarray,
    leaves: list[dict],
    traversability_by_parent: dict,
):
    groups = defaultdict(
        lambda: {
            "x": [],
            "y": [],
            "z": [],
            "i": [],
            "j": [],
            "k": [],
            "hover": [],
        }
    )

    for leaf in leaves:

        point_indices = np.asarray(
            leaf["point_indices"],
            dtype=np.int64,
        )

        if len(point_indices) == 0:
            continue

        resolution = round(
            float(
                leaf["cell_size"]
            ),
            6,
        )

        if resolution not in RESOLUTION_LABELS:
            continue

        parent_id = leaf.get(
            "_parent_cell_id"
        )

        terrain = traversability_by_parent.get(
            parent_id
        )

        if terrain is None:
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

        group = groups[
            (
                resolution,
                terrain[
                    "traversability"
                ],
            )
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

    return groups


def add_adaptive_surface_traces(
    figure: go.Figure,
    data: np.ndarray,
    leaves: list[dict],
    terrain_records: dict,
    mode_name: str,
    visible: bool,
) -> list[int]:

    groups = build_surface_mesh(
        data=data,
        leaves=leaves,
        traversability_by_parent=terrain_records,
    )

    trace_indices = []

    for (
        resolution,
        traversability,
    ), group in sorted(
        groups.items(),
        key=lambda item: (
            item[0][1],
            item[0][0],
        ),
    ):

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
                opacity=0.60,
                flatshading=True,
                name=(
                    f"{mode_name}: "
                    f"{label} / {traversability}"
                ),
                legendgroup=(
                    f"{mode_name}_{traversability}"
                ),
                showlegend=False,
                visible=visible,
                hovertemplate=(
                    f"{mode_name}<br>"
                    f"Resolution: {label}<br>"
                    f"Traversability: {traversability}"
                    "<extra></extra>"
                ),
            )
        )

        trace_indices.append(
            len(figure.data) - 1
        )

    return trace_indices


def add_traversability_markers(
    figure: go.Figure,
    terrain_records: dict,
    visible: bool,
) -> list[int]:

    grouped = defaultdict(
        lambda: {
            "x": [],
            "y": [],
            "z": [],
            "custom": [],
        }
    )

    for record in terrain_records.values():

        traversability = record[
            "traversability"
        ]

        grouped[
            traversability
        ]["x"].append(
            record["x_center"]
        )

        grouped[
            traversability
        ]["y"].append(
            record["y_center"]
        )

        grouped[
            traversability
        ]["z"].append(
            record["z_mean"]
            + 0.025
        )

        grouped[
            traversability
        ]["custom"].append(
            [
                record["slope"],
                record["roughness"],
                record["terrain_complexity"],
                record["terrain_confidence"],
                record["terrain_priority"],
                record["point_count"],
            ]
        )

    symbols = {
        "DRIVABLE": "circle",
        "NON-DRIVABLE": "x",
        "SPARSE / UNKNOWN": "diamond",
    }

    trace_indices = []

    for label in TRAVERSABILITY_LABELS:

        group = grouped[
            label
        ]

        if not group["x"]:
            continue

        figure.add_trace(
            go.Scatter3d(
                x=group["x"],
                y=group["y"],
                z=group["z"],
                mode="markers",
                marker={
                    "size": 5,
                    "symbol": symbols[label],
                },
                name=label,
                customdata=group[
                    "custom"
                ],
                hovertemplate=(
                    f"{label}<br>"
                    "Slope: %{customdata[0]:.2f}°<br>"
                    "Roughness: %{customdata[1]:.3f} m<br>"
                    "Complexity: %{customdata[2]:.3f}<br>"
                    "Confidence: %{customdata[3]:.3f}<br>"
                    "Priority: %{customdata[4]:.3f}<br>"
                    "Points: %{customdata[5]}<extra></extra>"
                ),
                legendgroup="TRAVERSABILITY",
                showlegend=True,
                visible=visible,
            )
        )

        trace_indices.append(
            len(figure.data) - 1
        )

    return trace_indices


def add_hole_markers(
    figure: go.Figure,
    hole_candidates: list[dict],
    visible: bool,
) -> list[int]:

    if not hole_candidates:
        return []

    figure.add_trace(
        go.Scatter3d(
            x=[
                candidate["x_center"]
                for candidate
                in hole_candidates
            ],
            y=[
                candidate["y_center"]
                for candidate
                in hole_candidates
            ],
            z=[
                candidate["z_mean"]
                + 0.08
                for candidate
                in hole_candidates
            ],
            mode="markers+text",
            marker={
                "size": 7,
                "symbol": "diamond",
            },
            text=[
                f"{candidate['hole_depth_m']:.2f} m"
                for candidate
                in hole_candidates
            ],
            textposition="top center",
            name="Hole / Depression Candidate",
            customdata=[
                [
                    candidate[
                        "hole_depth_m"
                    ],
                    candidate[
                        "local_surface_z"
                    ],
                    candidate[
                        "terrain_complexity"
                    ],
                    candidate[
                        "terrain_confidence"
                    ],
                ]
                for candidate
                in hole_candidates
            ],
            hovertemplate=(
                "Surface depression / hole candidate"
                "<br>Depth estimate: %{customdata[0]:.3f} m"
                "<br>Local reference Z: %{customdata[1]:.3f} m"
                "<br>Terrain complexity: %{customdata[2]:.3f}"
                "<br>Terrain confidence: %{customdata[3]:.3f}"
                "<extra></extra>"
            ),
            legendgroup="HAZARD",
            showlegend=True,
            visible=visible,
        )
    )

    return [
        len(figure.data) - 1
    ]


def add_raw_trace(
    figure: go.Figure,
    data: np.ndarray,
    visible: bool,
) -> int:

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

    figure.add_trace(
        go.Scatter3d(
            x=display_data[:, 0],
            y=display_data[:, 1],
            z=display_data[:, 2],
            mode="markers",
            marker={
                "size": 1.3,
                "opacity": 0.35,
            },
            name="Raw LiDAR",
            visible=visible,
            legendgroup="RAW",
            showlegend=True,
        )
    )

    return len(figure.data) - 1


def add_vehicle_marker(
    figure: go.Figure,
) -> int:

    figure.add_trace(
        go.Scatter3d(
            x=[0.0],
            y=[0.0],
            z=[0.0],
            mode="markers+text",
            marker={
                "size": 8,
                "symbol": "diamond",
            },
            text=["Vehicle / LiDAR"],
            textposition="top center",
            name="Vehicle / LiDAR",
            showlegend=True,
            visible=True,
        )
    )

    return len(figure.data) - 1


def add_resolution_legend(
    figure: go.Figure,
) -> list[int]:

    indices = []

    for resolution in (
        0.05,
        0.10,
        0.125,
        0.25,
        0.50,
    ):

        label = RESOLUTION_LABELS[
            resolution
        ]

        figure.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker={
                    "size": 8,
                },
                name=label,
                legendgroup="RESOLUTION",
                showlegend=True,
                visible=True,
            )
        )

        indices.append(
            len(figure.data) - 1
        )

    return indices


def visibility_array(
    total_traces: int,
    visible_indices: list[int],
) -> list[bool]:

    visibility = [
        False
        for _ in range(
            total_traces
        )
    ]

    for index in visible_indices:

        visibility[
            index
        ] = True

    return visibility


def main() -> None:

    print(
        "REAL-FRAME VEHICLE 3D TERRAIN SIMULATION"
    )

    print()
    print(
        "Sequence:",
        SEQUENCE,
    )

    print(
        "Frame:",
        FRAME,
    )

    print(
        "Input:",
        data_file_path(),
    )

    data = load_data()

    print()
    print(
        "LiDAR points:",
        len(data),
    )

    (
        ground_point_ids,
        cells,
        base_resolution_by_cell,
        analyzed_cells,
    ) = build_parent_data(
        data
    )

    print(
        "Ground points:",
        len(ground_point_ids),
    )

    print(
        "Terrain parent cells:",
        len(cells),
    )

    terrain_records = (
        build_parent_terrain_records(
            data=data,
            cells=cells,
            analyzed_cells=analyzed_cells,
        )
    )

    hole_candidates = detect_hole_candidates(
        terrain_records
    )

    print(
        "Hole/depression candidates:",
        len(hole_candidates),
    )

    print()
    print(
        "Building fixed 5 cm map..."
    )

    fixed_leaves = build_mode_leaves(
        data=data,
        cells=cells,
        base_resolution_by_cell=(
            base_resolution_by_cell
        ),
        analyzed_cells=analyzed_cells,
        mode="FIXED",
    )

    print(
        "Fixed occupied cells:",
        len(fixed_leaves),
    )

    print()
    print(
        "Building distance-adaptive map..."
    )

    distance_leaves = build_mode_leaves(
        data=data,
        cells=cells,
        base_resolution_by_cell=(
            base_resolution_by_cell
        ),
        analyzed_cells=analyzed_cells,
        mode="DISTANCE",
    )

    print(
        "Distance occupied cells:",
        len(distance_leaves),
    )

    print()
    print(
        "Building distance + terrain map..."
    )

    terrain_leaves = build_mode_leaves(
        data=data,
        cells=cells,
        base_resolution_by_cell=(
            base_resolution_by_cell
        ),
        analyzed_cells=analyzed_cells,
        mode="TERRAIN",
    )

    print(
        "Distance + Terrain occupied cells:",
        len(terrain_leaves),
    )

    print(
        "Distance + Terrain resolution distribution:",
        resolution_counts(
            terrain_leaves
        ),
    )

    figure = go.Figure()

    raw_index = add_raw_trace(
        figure,
        data,
        visible=False,
    )

    vehicle_index = add_vehicle_marker(
        figure
    )

    fixed_indices = add_adaptive_surface_traces(
        figure,
        data,
        fixed_leaves,
        terrain_records,
        "FIXED 5 cm",
        visible=False,
    )

    distance_indices = add_adaptive_surface_traces(
        figure,
        data,
        distance_leaves,
        terrain_records,
        "DISTANCE ADAPTIVE",
        visible=False,
    )

    terrain_indices = add_adaptive_surface_traces(
        figure,
        data,
        terrain_leaves,
        terrain_records,
        "DISTANCE + TERRAIN",
        visible=True,
    )

    traversability_indices = (
        add_traversability_markers(
            figure,
            terrain_records,
            visible=True,
        )
    )

    hole_indices = add_hole_markers(
        figure,
        hole_candidates,
        visible=True,
    )

    resolution_indices = (
        add_resolution_legend(
            figure
        )
    )

    total_traces = len(
        figure.data
    )

    user_view_indices = (
        [vehicle_index]
        + terrain_indices
        + traversability_indices
        + hole_indices
        + resolution_indices
    )

    raw_view_indices = (
        [raw_index, vehicle_index]
    )

    fixed_view_indices = (
        [vehicle_index]
        + fixed_indices
        + resolution_indices
    )

    distance_view_indices = (
        [vehicle_index]
        + distance_indices
        + traversability_indices
        + resolution_indices
    )

    terrain_view_indices = (
        [vehicle_index]
        + terrain_indices
        + traversability_indices
        + hole_indices
        + resolution_indices
    )

    driving_view_indices = (
        [raw_index, vehicle_index]
        + terrain_indices
        + traversability_indices
        + hole_indices
        + resolution_indices
    )

    user_visibility = visibility_array(
        total_traces,
        user_view_indices,
    )

    raw_visibility = visibility_array(
        total_traces,
        raw_view_indices,
    )

    fixed_visibility = visibility_array(
        total_traces,
        fixed_view_indices,
    )

    distance_visibility = visibility_array(
        total_traces,
        distance_view_indices,
    )

    terrain_visibility = visibility_array(
        total_traces,
        terrain_view_indices,
    )

    driving_visibility = visibility_array(
        total_traces,
        driving_view_indices,
    )

    figure.update_layout(
        title=(
            f"REAL RELLIS-3D {SEQUENCE}/{FRAME} — "
            "Vehicle-Forward Driving View"
        ),

        scene={
            "xaxis": {
                "title": "Forward X (m)",
                "range": [
                    0,
                    30,
                ],
                "showspikes": False,
            },
            "yaxis": {
                "title": "Lateral Y (m)",
                "range": [
                    -10,
                    10,
                ],
                "showspikes": False,
            },
            "zaxis": {
                "title": "Elevation Z (m)",
                "showspikes": False,
            },
            "aspectmode": "manual",
            "aspectratio": {
                "x": 2.2,
                "y": 1.4,
                "z": 0.75,
            },
            "camera": {
                # Vehicle-view default:
                # the camera sits behind the LiDAR/vehicle and
                # looks forward along +X, with Z as the vertical axis.
                "eye": {
                    "x": -1.80,
                    "y": 0.0,
                    "z": 0.22,
                },
                "center": {
                    "x": 0.85,
                    "y": 0.0,
                    "z": 0.03,
                },
                "up": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 1.0,
                },
                "projection": {
                    "type": "perspective",
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
                                    driving_visibility
                            },
                            {
                                "title": (
                                    f"REAL RELLIS-3D "
                                    f"{SEQUENCE}/{FRAME} — "
                                    "Vehicle-Forward Driving View"
                                )
                            },
                        ],
                    },
                    {
                        "label": "USER VIEW",
                        "method": "update",
                        "args": [
                            {
                                "visible":
                                    user_visibility
                            },
                            {
                                "title": (
                                    f"REAL RELLIS-3D "
                                    f"{SEQUENCE}/{FRAME} — "
                                    "Adaptive 2.5D Terrain + Hazards"
                                )
                            },
                        ],
                    },
                    {
                        "label": "RAW LiDAR",
                        "method": "update",
                        "args": [
                            {
                                "visible":
                                    raw_visibility
                            },
                            {
                                "title": (
                                    f"REAL RELLIS-3D "
                                    f"{SEQUENCE}/{FRAME} — "
                                    "Raw LiDAR Point Cloud"
                                )
                            },
                        ],
                    },
                    {
                        "label": "FIXED 5 cm",
                        "method": "update",
                        "args": [
                            {
                                "visible":
                                    fixed_visibility
                            },
                            {
                                "title": (
                                    f"REAL RELLIS-3D "
                                    f"{SEQUENCE}/{FRAME} — "
                                    "Fixed-Resolution Baseline"
                                )
                            },
                        ],
                    },
                    {
                        "label": "DISTANCE ADAPTIVE",
                        "method": "update",
                        "args": [
                            {
                                "visible":
                                    distance_visibility
                            },
                            {
                                "title": (
                                    f"REAL RELLIS-3D "
                                    f"{SEQUENCE}/{FRAME} — "
                                    "Distance-Adaptive 2.5D"
                                )
                            },
                        ],
                    },
                    {
                        "label": "DISTANCE + TERRAIN",
                        "method": "update",
                        "args": [
                            {
                                "visible":
                                    terrain_visibility
                            },
                            {
                                "title": (
                                    f"REAL RELLIS-3D "
                                    f"{SEQUENCE}/{FRAME} — "
                                    "Proposed Distance + Terrain"
                                )
                            },
                        ],
                    },
                ],
            }
        ],

        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.10,
            "yanchor": "top",
        },

        margin={
            "l": 0,
            "r": 0,
            "t": 95,
            "b": 100,
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
            f"real_frame_driving_view_"
            f"{SEQUENCE}_{FRAME}.html"
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
        "Driving-view features:"
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
        "  Surface depression / hole candidates"
    )

    print(
        "  Slope, roughness, complexity, confidence"
    )

    print(
        "  Adaptive 5 / 10 / 12.5 / 25 / 50 cm resolution"
    )

    print(
        "  Raw LiDAR context"
    )

    print()
    print(
        "Important: hole candidates are local-surface "
        "depression heuristics, not confirmed object/hazard labels."
    )

    print()
    print(
        "Real-frame driving simulation complete."
    )


if __name__ == "__main__":
    main()
