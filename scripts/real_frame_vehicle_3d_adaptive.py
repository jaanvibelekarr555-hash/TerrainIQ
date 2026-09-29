from __future__ import annotations

import os

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

RAW_MAX_POINTS = 20000

RESOLUTION_INFO = {
    0.05: {
        "label": "5 cm",
        "color": "#2563eb",
    },
    0.10: {
        "label": "10 cm",
        "color": "#16a34a",
    },
    0.125: {
        "label": "12.5 cm",
        "color": "#eab308",
    },
    0.25: {
        "label": "25 cm",
        "color": "#f97316",
    },
    0.50: {
        "label": "50 cm",
        "color": "#dc2626",
    },
}


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


def build_leaves(
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
            ) > 0:

                leaves.append(
                    leaf
                )

    return leaves


def leaf_z(
    data: np.ndarray,
    leaf: dict,
) -> float:

    indices = np.asarray(
        leaf["point_indices"],
        dtype=np.int64,
    )

    return float(
        np.mean(
            data[
                indices,
                2,
            ]
        )
    )


def build_mesh_group(
    data: np.ndarray,
    leaves: list[dict],
) -> dict[float, dict]:

    grouped = {
        resolution: {
            "x": [],
            "y": [],
            "z": [],
            "i": [],
            "j": [],
            "k": [],
        }
        for resolution in RESOLUTION_INFO
    }

    for leaf in leaves:

        resolution = round(
            float(leaf["cell_size"]),
            6,
        )

        if resolution not in grouped:
            continue

        bbox = leaf["bbox"]

        z = leaf_z(
            data,
            leaf,
        )

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
            [offset + 1, offset + 2]
        )

        group["k"].extend(
            [offset + 2, offset + 3]
        )

    return grouped


def add_mesh_traces(
    figure: go.Figure,
    data: np.ndarray,
    leaves: list[dict],
    mode_name: str,
    visible: bool,
    opacity: float,
) -> list[int]:

    grouped = build_mesh_group(
        data,
        leaves,
    )

    trace_indices = []

    for resolution in RESOLUTION_INFO:

        group = grouped[
            resolution
        ]

        if not group["x"]:
            continue

        label = RESOLUTION_INFO[
            resolution
        ]["label"]

        color = RESOLUTION_INFO[
            resolution
        ]["color"]

        figure.add_trace(
            go.Mesh3d(
                x=group["x"],
                y=group["y"],
                z=group["z"],
                i=group["i"],
                j=group["j"],
                k=group["k"],
                color=color,
                opacity=opacity,
                flatshading=True,
                hovertemplate=(
                    f"{mode_name}<br>"
                    f"Resolution: {label}"
                    "<extra></extra>"
                ),
                name=f"{mode_name}: {label}",
                legendgroup=mode_name,
                visible=visible,
                showlegend=False,
            )
        )

        trace_indices.append(
            len(figure.data) - 1
        )

    return trace_indices


def add_raw_trace(
    figure: go.Figure,
    data: np.ndarray,
    visible: bool,
) -> int:

    display_data = data

    if len(display_data) > RAW_MAX_POINTS:

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
                "size": 1.6,
                "opacity": 0.60,
                "color": display_data[:, 2],
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {
                    "title": "Elevation Z (m)",
                    "len": 0.55,
                },
            },
            name="Raw LiDAR",
            visible=visible,
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
                "size": 7,
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

    for resolution in RESOLUTION_INFO:

        label = RESOLUTION_INFO[
            resolution
        ]["label"]

        color = RESOLUTION_INFO[
            resolution
        ]["color"]

        figure.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker={
                    "size": 9,
                    "color": color,
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
        visibility[index] = True

    return visibility


def main() -> None:

    print(
        "REAL-FRAME 3D ENVIRONMENT SIMULATION"
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

    print()
    print(
        "Building fixed 5 cm map..."
    )

    fixed_leaves = build_leaves(
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

    distance_leaves = build_leaves(
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

    terrain_leaves = build_leaves(
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

    figure = go.Figure()

    raw_index = add_raw_trace(
        figure,
        data,
        visible=True,
    )

    vehicle_index = add_vehicle_marker(
        figure
    )

    fixed_indices = add_mesh_traces(
        figure,
        data,
        fixed_leaves,
        "FIXED 5 cm",
        visible=False,
        opacity=0.72,
    )

    distance_indices = add_mesh_traces(
        figure,
        data,
        distance_leaves,
        "DISTANCE ADAPTIVE",
        visible=False,
        opacity=0.72,
    )

    terrain_indices = add_mesh_traces(
        figure,
        data,
        terrain_leaves,
        "DISTANCE + TERRAIN",
        visible=True,
        opacity=0.72,
    )

    resolution_legend_indices = (
        add_resolution_legend(
            figure
        )
    )

    total_traces = len(
        figure.data
    )

    user_view_indices = (
        [raw_index, vehicle_index]
        + terrain_indices
        + resolution_legend_indices
    )

    raw_view_indices = (
        [raw_index, vehicle_index]
        + resolution_legend_indices
    )

    fixed_view_indices = (
        [vehicle_index]
        + fixed_indices
        + resolution_legend_indices
    )

    distance_view_indices = (
        [vehicle_index]
        + distance_indices
        + resolution_legend_indices
    )

    terrain_view_indices = (
        [vehicle_index]
        + terrain_indices
        + resolution_legend_indices
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

    figure.update_layout(
        title=(
            f"REAL RELLIS-3D {SEQUENCE}/{FRAME} — "
            "Vehicle-Centric 3D Environment & Adaptive Mapping"
        ),

        scene={
            "xaxis": {
                "title": "Forward X (m)",
                "range": [
                    0,
                    30,
                ],
                "backgroundcolor": "rgb(245,245,245)",
                "showspikes": False,
            },
            "yaxis": {
                "title": "Lateral Y (m)",
                "range": [
                    -10,
                    10,
                ],
                "backgroundcolor": "rgb(245,245,245)",
                "showspikes": False,
            },
            "zaxis": {
                "title": "Elevation Z (m)",
                "backgroundcolor": "rgb(250,250,250)",
                "showspikes": False,
            },
            "aspectmode": "manual",
            "aspectratio": {
                "x": 1.8,
                "y": 1.2,
                "z": 0.65,
            },
            "camera": {
                "eye": {
                    "x": -1.75,
                    "y": -1.45,
                    "z": 0.90,
                },
                "center": {
                    "x": 0.45,
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
                                    "Vehicle-Centric 3D Environment "
                                    "+ Proposed Adaptive 2.5D"
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
                                    "Raw LiDAR Environment"
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
                                    "Fixed-Resolution 5 cm Map"
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
                                    "Distance-Adaptive Map"
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
                                    "Proposed Distance + Terrain Map"
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
            "b": 90,
        },

        hovermode="closest",

        paper_bgcolor="white",

        template="plotly_white",
    )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    output_path = os.path.join(
        RESULTS_DIR,
        (
            f"real_frame_vehicle_3d_adaptive_"
            f"{SEQUENCE}_{FRAME}.html"
        ),
    )

    # Start directly in the user view.
    figure.update_traces(
        selector=dict(
            name="Raw LiDAR"
        ),
        visible=True,
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
        "Interactive views:"
    )

    print(
        "  USER VIEW"
    )

    print(
        "  RAW LiDAR"
    )

    print(
        "  FIXED 5 cm"
    )

    print(
        "  DISTANCE ADAPTIVE"
    )

    print(
        "  DISTANCE + TERRAIN"
    )

    print()
    print(
        "The USER VIEW combines the real LiDAR environment "
        "with the proposed adaptive terrain representation."
    )

    print()
    print(
        "Real-frame 3D simulation complete."
    )


if __name__ == "__main__":
    main()
