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

# To keep the browser visualization responsive, the raw point cloud
# is displayed at this stride. The mapping itself still uses every point.
RAW_POINT_DISPLAY_STRIDE = 1


RESOLUTION_INFO = {
    0.05: ("5 cm", "#2563eb"),
    0.10: ("10 cm", "#16a34a"),
    0.125: ("12.5 cm", "#eab308"),
    0.25: ("25 cm", "#f97316"),
    0.50: ("50 cm", "#dc2626"),
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
            ) > 0:

                leaves.append(leaf)

    return leaves


def leaf_resolution(
    leaf: dict,
) -> float:

    return round(
        float(
            leaf["cell_size"]
        ),
        6,
    )


def leaf_polygon_data(
    data: np.ndarray,
    leaves: list[dict],
):
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

        resolution = leaf_resolution(
            leaf
        )

        if resolution not in grouped:
            continue

        point_indices = np.asarray(
            leaf["point_indices"],
            dtype=np.int64,
        )

        if len(point_indices) == 0:
            continue

        z_value = float(
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
            grouped[resolution]["x"]
        )

        grouped[resolution]["x"].extend(
            [x0, x1, x1, x0]
        )

        grouped[resolution]["y"].extend(
            [y0, y0, y1, y1]
        )

        grouped[resolution]["z"].extend(
            [z_value] * 4
        )

        grouped[resolution]["i"].append(
            offset
        )
        grouped[resolution]["j"].append(
            offset + 1
        )
        grouped[resolution]["k"].append(
            offset + 2
        )

        grouped[resolution]["i"].append(
            offset
        )
        grouped[resolution]["j"].append(
            offset + 2
        )
        grouped[resolution]["k"].append(
            offset + 3
        )

    return grouped


def add_map_traces(
    figure: go.Figure,
    data: np.ndarray,
    leaves: list[dict],
    visible: bool,
    group_name: str,
) -> list[int]:

    grouped = leaf_polygon_data(
        data=data,
        leaves=leaves,
    )

    trace_indices = []

    for resolution in RESOLUTION_INFO:

        group = grouped[resolution]

        if not group["x"]:
            continue

        label, color = RESOLUTION_INFO[
            resolution
        ]

        trace = go.Mesh3d(
            x=group["x"],
            y=group["y"],
            z=group["z"],
            i=group["i"],
            j=group["j"],
            k=group["k"],
            color=color,
            opacity=0.72,
            flatshading=True,
            name=f"{group_name}: {label}",
            legendgroup=group_name,
            visible=visible,
            hoverinfo="name",
            showlegend=True,
        )

        figure.add_trace(
            trace
        )

        trace_indices.append(
            len(figure.data) - 1
        )

    return trace_indices


def add_raw_point_trace(
    figure: go.Figure,
    data: np.ndarray,
    visible: bool,
) -> int:

    display_data = data[
        ::RAW_POINT_DISPLAY_STRIDE
    ]

    if len(display_data) > 50000:

        stride = int(
            np.ceil(
                len(display_data)
                / 50000
            )
        )

        display_data = display_data[
            ::stride
        ]

    figure.add_trace(
        go.Scatter3d(
            x=display_data[:, 0],
            y=display_data[:, 1],
            z=display_data[:, 2],
            mode="markers",
            marker={
                "size": 1.8,
                "opacity": 0.75,
                "color": display_data[:, 2],
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {
                    "title": "Z (m)"
                },
            },
            name="Raw LiDAR",
            visible=visible,
            legendgroup="RAW",
        )
    )

    return len(figure.data) - 1


def add_title_trace(
    figure: go.Figure,
    text: str,
) -> None:

    figure.add_annotation(
        text=text,
        x=0.5,
        y=1.02,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={
            "size": 18,
        },
    )


def add_resolution_legend(
    figure: go.Figure,
) -> None:

    for resolution in RESOLUTION_INFO:

        label, color = RESOLUTION_INFO[
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
                    "color": color,
                },
                name=label,
                legendgroup="RESOLUTION",
                visible=True,
                showlegend=True,
            )
        )


def make_visibility_matrix(
    figure: go.Figure,
    raw_index: int,
    fixed_indices: list[int],
    distance_indices: list[int],
    terrain_indices: list[int],
    total_traces_before_legend: int,
):
    base_false = [
        False
        for _ in range(
            total_traces_before_legend
        )
    ]

    visibility = []

    raw_visibility = list(
        base_false
    )

    raw_visibility[
        raw_index
    ] = True

    visibility.append(
        raw_visibility
    )

    for target_indices in (
        fixed_indices,
        distance_indices,
        terrain_indices,
    ):

        current = list(
            base_false
        )

        for index in target_indices:
            current[index] = True

        visibility.append(
            current
        )

    return visibility


def main() -> None:

    print(
        "REAL RELLIS-3D 3D MAPPING SIMULATION"
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
        "Using actual LiDAR frame:"
    )

    print(
        " ",
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
    ) = build_parent_inputs(
        data
    )

    print(
        "Ground points:",
        len(ground_point_ids),
    )

    print(
        "Parent terrain cells:",
        len(cells),
    )

    print()
    print(
        "Building fixed 5 cm representation..."
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
        "Building distance-adaptive representation..."
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
        "Building distance + terrain representation..."
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

    figure = go.Figure()

    raw_index = add_raw_point_trace(
        figure,
        data,
        visible=True,
    )

    fixed_indices = add_map_traces(
        figure,
        data,
        fixed_leaves,
        visible=False,
        group_name="FIXED",
    )

    distance_indices = add_map_traces(
        figure,
        data,
        distance_leaves,
        visible=False,
        group_name="DISTANCE",
    )

    terrain_indices = add_map_traces(
        figure,
        data,
        terrain_leaves,
        visible=False,
        group_name="TERRAIN",
    )

    total_map_traces = len(
        figure.data
    )

    add_resolution_legend(
        figure
    )

    total_traces = len(
        figure.data
    )

    # Legends are kept visible while map traces are switched.
    # Build complete visibility arrays accordingly.
    raw_visibility = [
        False
        for _ in range(
            total_traces
        )
    ]

    raw_visibility[
        raw_index
    ] = True

    fixed_visibility = [
        False
        for _ in range(
            total_traces
        )
    ]

    for index in fixed_indices:
        fixed_visibility[
            index
        ] = True

    distance_visibility = [
        False
        for _ in range(
            total_traces
        )
    ]

    for index in distance_indices:
        distance_visibility[
            index
        ] = True

    terrain_visibility = [
        False
        for _ in range(
            total_traces
        )
    ]

    for index in terrain_indices:
        terrain_visibility[
            index
        ] = True

    for index in range(
        total_map_traces,
        total_traces,
    ):
        raw_visibility[
            index
        ] = True

        fixed_visibility[
            index
        ] = True

        distance_visibility[
            index
        ] = True

        terrain_visibility[
            index
        ] = True

    figure.update_layout(
        title=(
            f"Real RELLIS-3D Frame {SEQUENCE}/{FRAME} — "
            "3D LiDAR / Adaptive 2.5D Simulation"
        ),

        scene={
            "xaxis_title": "X (m)",
            "yaxis_title": "Y (m)",
            "zaxis_title": "Elevation Z (m)",
            "xaxis": {
                "range": [
                    0,
                    30,
                ]
            },
            "yaxis": {
                "range": [
                    -10,
                    10,
                ]
            },
        },

        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0.02,
                "y": 1.12,
                "buttons": [
                    {
                        "label": "Raw LiDAR",
                        "method": "update",
                        "args": [
                            {
                                "visible":
                                    raw_visibility
                            },
                            {
                                "title": (
                                    f"Real RELLIS-3D "
                                    f"{SEQUENCE}/{FRAME} — "
                                    "Raw LiDAR Point Cloud"
                                )
                            },
                        ],
                    },
                    {
                        "label": "Fixed 5 cm",
                        "method": "update",
                        "args": [
                            {
                                "visible":
                                    fixed_visibility
                            },
                            {
                                "title": (
                                    f"Real RELLIS-3D "
                                    f"{SEQUENCE}/{FRAME} — "
                                    "Fixed-Resolution Baseline"
                                )
                            },
                        ],
                    },
                    {
                        "label": "Distance Adaptive",
                        "method": "update",
                        "args": [
                            {
                                "visible":
                                    distance_visibility
                            },
                            {
                                "title": (
                                    f"Real RELLIS-3D "
                                    f"{SEQUENCE}/{FRAME} — "
                                    "Distance-Adaptive 2.5D"
                                )
                            },
                        ],
                    },
                    {
                        "label": "Distance + Terrain",
                        "method": "update",
                        "args": [
                            {
                                "visible":
                                    terrain_visibility
                            },
                            {
                                "title": (
                                    f"Real RELLIS-3D "
                                    f"{SEQUENCE}/{FRAME} — "
                                    "Proposed Distance + Terrain 2.5D"
                                )
                            },
                        ],
                    },
                ],
            }
        ],

        legend={
            "orientation": "h",
            "y": -0.12,
        },

        margin={
            "l": 0,
            "r": 0,
            "t": 80,
            "b": 80,
        },
    )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    output_path = os.path.join(
        RESULTS_DIR,
        (
            f"real_3d_mapping_simulation_"
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
        "Simulation output:",
        output_path,
    )

    print()
    print(
        "Available views:"
    )

    print(
        "  Raw LiDAR"
    )

    print(
        "  Fixed 5 cm"
    )

    print(
        "  Distance Adaptive"
    )

    print(
        "  Distance + Terrain"
    )

    print()
    print(
        "3D real-frame simulation complete."
    )


if __name__ == "__main__":
    main()
