from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.terrain_analysis import analyze_terrain_cells
from src.terrain_cells import build_terrain_cells
from src.terrain_quadtree import build_terrain_quadtree_for_cell
from src.terrain_resolution import build_base_resolution_by_cell
from src.road_processor import extract_ground_mask


DATA_FILE = (
    "data/RELLIS3D/00000/"
    "vel_cloud_node_kitti_bin/000000.bin"
)

RESULTS_DIR = "results"

PARENT_CELL_SIZE = 1.0

FIXED_RESOLUTION = 0.05
MIN_RESOLUTION = 0.05

NO_ERROR_BOUND = 1_000_000.0

WINDOW_WIDTH = 4.0
WINDOW_HEIGHT = 4.0
BIN_SIZE = 1.0

RESOLUTIONS = (
    0.05,
    0.10,
    0.125,
    0.25,
)

RESOLUTION_LABELS = {
    0.05: "5 cm",
    0.10: "10 cm",
    0.125: "12.5 cm",
    0.25: "25 cm",
}


def load_data() -> np.ndarray:
    data = np.fromfile(
        DATA_FILE,
        dtype=np.float32,
    ).reshape(-1, 4)

    if len(data) == 0:
        raise ValueError(
            "LiDAR frame is empty."
        )

    return data


def build_parent_data(
    data: np.ndarray,
):
    ground_mask, point_id, _ = extract_ground_mask(
        data
    )

    cells = build_terrain_cells(
        data=data,
        ground_mask=ground_mask,
        point_id=point_id,
        cell_size=PARENT_CELL_SIZE,
    )

    base_resolution_by_cell = build_base_resolution_by_cell(
        cells
    )

    analyzed_cells = analyze_terrain_cells(
        data=data,
        cells=cells,
        base_resolution_by_cell=base_resolution_by_cell,
    )

    return (
        cells,
        base_resolution_by_cell,
        analyzed_cells,
    )


def build_map_leaves(
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
            ][
                "refinement_request"
            ]

            requested_resolution = float(
                request[
                    "requested_resolution"
                ]
            )

            priority = float(
                analyzed_cells[
                    cell_id
                ][
                    "terrain_priority"
                ]
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


def leaf_resolution(
    leaf: dict,
) -> float:

    return round(
        float(
            leaf["cell_size"]
        ),
        6,
    )


def resolution_counts(
    leaves: list[dict],
) -> dict[float, int]:

    counts = {}

    for leaf in leaves:

        resolution = leaf_resolution(
            leaf
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


def select_mixed_resolution_region(
    terrain_leaves: list[dict],
) -> tuple[
    float,
    float,
    float,
    float,
]:

    if not terrain_leaves:
        raise ValueError(
            "No terrain leaves available."
        )

    xs = []
    ys = []
    resolutions = []

    for leaf in terrain_leaves:

        x, y = leaf_center(
            leaf
        )

        xs.append(x)
        ys.append(y)
        resolutions.append(
            leaf_resolution(leaf)
        )

    xs = np.asarray(
        xs,
        dtype=np.float64,
    )

    ys = np.asarray(
        ys,
        dtype=np.float64,
    )

    resolutions = np.asarray(
        resolutions,
        dtype=np.float64,
    )

    x_origin = 0.0
    y_origin = -10.0

    x_bins = np.floor(
        (xs - x_origin)
        / BIN_SIZE
    ).astype(
        np.int64
    )

    y_bins = np.floor(
        (ys - y_origin)
        / BIN_SIZE
    ).astype(
        np.int64
    )

    max_x_bin = int(
        np.floor(
            (30.0 - x_origin)
            / BIN_SIZE
        )
    )

    max_y_bin = int(
        np.floor(
            (10.0 - y_origin)
            / BIN_SIZE
        )
    )

    candidate_bins = set(
        zip(
            x_bins.tolist(),
            y_bins.tolist(),
        )
    )

    best_key = None
    best_bin = None

    for x0, y0 in candidate_bins:

        x_mask = (
            (x_bins >= x0)
            & (
                x_bins
                < x0
                + int(
                    round(
                        WINDOW_WIDTH
                        / BIN_SIZE
                    )
                )
            )
        )

        y_mask = (
            (y_bins >= y0)
            & (
                y_bins
                < y0
                + int(
                    round(
                        WINDOW_HEIGHT
                        / BIN_SIZE
                    )
                )
            )
        )

        mask = x_mask & y_mask

        if not np.any(mask):
            continue

        regional_resolutions = resolutions[
            mask
        ]

        distinct_count = len(
            np.unique(
                regional_resolutions
            )
        )

        non_fine_count = int(
            np.sum(
                regional_resolutions
                > 0.05
            )
        )

        total_count = int(
            np.sum(mask)
        )

        key = (
            distinct_count,
            non_fine_count,
            total_count,
        )

        if (
            best_key is None
            or key > best_key
        ):

            best_key = key
            best_bin = (
                x0,
                y0,
            )

    if best_bin is None:
        raise ValueError(
            "Unable to find a mixed-resolution region."
        )

    x0, y0 = best_bin

    x_min = (
        x_origin
        + x0 * BIN_SIZE
    )

    y_min = (
        y_origin
        + y0 * BIN_SIZE
    )

    x_max = min(
        30.0,
        x_min + WINDOW_WIDTH,
    )

    y_max = min(
        10.0,
        y_min + WINDOW_HEIGHT,
    )

    return (
        x_min,
        x_max,
        y_min,
        y_max,
    )


def filter_leaves(
    leaves: list[dict],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> list[dict]:

    selected = []

    for leaf in leaves:

        x, y = leaf_center(
            leaf
        )

        if (
            x_min <= x <= x_max
            and y_min <= y <= y_max
        ):

            selected.append(
                leaf
            )

    return selected


def make_polygon(
    data: np.ndarray,
    leaf: dict,
) -> list[
    tuple[
        float,
        float,
        float,
    ]
]:

    indices = np.asarray(
        leaf["point_indices"],
        dtype=np.int64,
    )

    z_value = float(
        np.mean(
            data[
                indices,
                2,
            ]
        )
    )

    bbox = leaf["bbox"]

    return [
        (
            float(bbox["x_min"]),
            float(bbox["y_min"]),
            z_value,
        ),
        (
            float(bbox["x_max"]),
            float(bbox["y_min"]),
            z_value,
        ),
        (
            float(bbox["x_max"]),
            float(bbox["y_max"]),
            z_value,
        ),
        (
            float(bbox["x_min"]),
            float(bbox["y_max"]),
            z_value,
        ),
    ]


def render_panel(
    ax,
    data: np.ndarray,
    leaves: list[dict],
    title: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> None:

    polygons = [
        make_polygon(
            data,
            leaf,
        )
        for leaf in leaves
    ]

    if not polygons:
        raise ValueError(
            f"No leaves available for panel: {title}"
        )

    collection = Poly3DCollection(
        polygons,
        alpha=0.72,
        linewidths=[
            max(
                0.25,
                1.6
                * (
                    0.25
                    / max(
                        leaf_resolution(leaf),
                        0.05,
                    )
                ),
            )
            for leaf in leaves
        ],
    )

    ax.add_collection3d(
        collection
    )

    z_values = [
        vertex[2]
        for polygon in polygons
        for vertex in polygon
    ]

    z_min = float(
        np.min(z_values)
    )

    z_max = float(
        np.max(z_values)
    )

    z_range = max(
        0.05,
        z_max - z_min,
    )

    ax.set_xlim(
        x_min,
        x_max,
    )

    ax.set_ylim(
        y_min,
        y_max,
    )

    ax.set_zlim(
        z_min - 0.10 * z_range,
        z_max + 0.10 * z_range,
    )

    ax.set_xlabel(
        "X (m)"
    )

    ax.set_ylabel(
        "Y (m)"
    )

    ax.set_zlabel(
        "Elevation Z (m)"
    )

    ax.set_title(
        title
    )

    ax.view_init(
        elev=48,
        azim=-58,
    )

    ax.set_box_aspect(
        (
            x_max - x_min,
            y_max - y_min,
            max(
                0.5,
                0.45 * z_range,
            ),
        )
    )


def main() -> None:

    print(
        "3-STAGE 3D RESOLUTION COMPARISON"
    )

    print(
        "Fixed 5 cm -> Distance Adaptive -> Distance + Terrain"
    )

    print()
    print(
        "Frame:",
        DATA_FILE,
    )

    data = load_data()

    (
        cells,
        base_resolution_by_cell,
        analyzed_cells,
    ) = build_parent_data(
        data
    )

    print()
    print(
        "Parent terrain cells:",
        len(cells),
    )

    print()
    print(
        "Building fixed 5 cm map..."
    )

    fixed_leaves = build_map_leaves(
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

    print(
        "Fixed resolution distribution:",
        resolution_counts(
            fixed_leaves
        ),
    )

    print()
    print(
        "Building distance-adaptive map..."
    )

    distance_leaves = build_map_leaves(
        data=data,
        cells=cells,
        base_resolution_by_cell=(
            base_resolution_by_cell
        ),
        analyzed_cells=analyzed_cells,
        mode="DISTANCE",
    )

    print(
        "Distance-adaptive occupied cells:",
        len(distance_leaves),
    )

    print(
        "Distance resolution distribution:",
        resolution_counts(
            distance_leaves
        ),
    )

    print()
    print(
        "Building distance + terrain map..."
    )

    terrain_leaves = build_map_leaves(
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
        "Terrain resolution distribution:",
        resolution_counts(
            terrain_leaves
        ),
    )

    print()
    print(
        "Selecting a region containing mixed adaptive resolutions..."
    )

    (
        x_min,
        x_max,
        y_min,
        y_max,
    ) = select_mixed_resolution_region(
        terrain_leaves
    )

    fixed_zoom = filter_leaves(
        fixed_leaves,
        x_min,
        x_max,
        y_min,
        y_max,
    )

    distance_zoom = filter_leaves(
        distance_leaves,
        x_min,
        x_max,
        y_min,
        y_max,
    )

    terrain_zoom = filter_leaves(
        terrain_leaves,
        x_min,
        x_max,
        y_min,
        y_max,
    )

    print(
        "Selected region:"
    )

    print(
        f"  X: {x_min:.2f} to {x_max:.2f} m"
    )

    print(
        f"  Y: {y_min:.2f} to {y_max:.2f} m"
    )

    print()
    print(
        "Regional fixed cells:",
        len(fixed_zoom),
    )

    print(
        "Regional distance cells:",
        len(distance_zoom),
    )

    print(
        "Regional terrain cells:",
        len(terrain_zoom),
    )

    fixed_distribution = resolution_counts(
        fixed_zoom
    )

    distance_distribution = resolution_counts(
        distance_zoom
    )

    terrain_distribution = resolution_counts(
        terrain_zoom
    )

    print()
    print(
        "Regional fixed distribution:",
        fixed_distribution,
    )

    print(
        "Regional distance distribution:",
        distance_distribution,
    )

    print(
        "Regional terrain distribution:",
        terrain_distribution,
    )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    output_path = os.path.join(
        RESULTS_DIR,
        "3d_fixed_distance_terrain_resolution_comparison_00000_000000.png",
    )

    figure = plt.figure(
        figsize=(20, 8)
    )

    axes = [
        figure.add_subplot(
            131,
            projection="3d",
        ),
        figure.add_subplot(
            132,
            projection="3d",
        ),
        figure.add_subplot(
            133,
            projection="3d",
        ),
    ]

    render_panel(
        axes[0],
        data,
        fixed_zoom,
        "1. Fixed-Resolution Baseline\nUniform 5 cm",
        x_min,
        x_max,
        y_min,
        y_max,
    )

    render_panel(
        axes[1],
        data,
        distance_zoom,
        "2. Distance-Adaptive\n5 / 10 / 25 / 50 cm",
        x_min,
        x_max,
        y_min,
        y_max,
    )

    render_panel(
        axes[2],
        data,
        terrain_zoom,
        "3. Proposed Adaptive 2.5D\nDistance + Terrain",
        x_min,
        x_max,
        y_min,
        y_max,
    )

    figure.suptitle(
        (
            "Same RELLIS-3D Region — How Spatial Resolution Changes "
            "Through the Mapping Pipeline"
        )
    )

    legend_handles = [
        Patch(
            label="5 cm"
        ),
        Patch(
            label="10 cm"
        ),
        Patch(
            label="12.5 cm"
        ),
        Patch(
            label="25 cm"
        ),
        Patch(
            label="50 cm base distance band"
        ),
    ]

    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(
            0.5,
            0.01,
        ),
    )

    figure.tight_layout(
        rect=(
            0,
            0.08,
            1,
            0.94,
        )
    )

    print()
    print(
        "Rendering 3-stage 3D comparison..."
    )

    figure.savefig(
        output_path,
        dpi=230,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print()
    print(
        "Output:",
        output_path,
    )

    print()
    print(
        "3D comparison complete."
    )


if __name__ == "__main__":
    main()
