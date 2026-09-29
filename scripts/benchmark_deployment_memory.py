from __future__ import annotations

import gc
import time
import tracemalloc

import numpy as np

from src.road_processor import extract_ground_mask
from src.terrain_cells import build_terrain_cells
from src.terrain_resolution import build_base_resolution_by_cell
from src.terrain_analysis import analyze_terrain_cells
from src.terrain_quadtree import (
    build_terrain_quadtree_for_cell,
)


DATA_FILE = (
    "data/RELLIS3D/00000/"
    "vel_cloud_node_kitti_bin/000000.bin"
)

MIN_RESOLUTION = 0.05
NO_ERROR_BOUND = 1_000_000.0


COMPACT_DTYPE = np.dtype(
    [
        ("x", np.float32),
        ("y", np.float32),
        ("elevation", np.float32),
        ("z_min", np.float32),
        ("z_max", np.float32),
        ("slope", np.float32),
        ("roughness", np.float32),
        ("elevation_variation", np.float32),
        ("terrain_complexity", np.float32),
        ("geometry_status", np.uint8),
        ("traversability", np.uint8),
        ("terrain_confidence", np.float32),
        ("terrain_priority", np.float32),
        ("resolution", np.float32),
        ("depth", np.uint8),
        ("point_count", np.uint32),
    ]
)


GEOMETRY_STATUS_CODES = {
    "valid": 0,
    "sparse": 1,
}


TRAVERSABILITY_CODES = {
    "DRIVABLE": 0,
    "NON-DRIVABLE": 1,
    "SPARSE / UNKNOWN": 2,
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


def build_inputs(data):
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
        cell_size=1.0,
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
    data,
    cells,
    base_resolution_by_cell,
    analyzed_cells,
    mode,
):
    leaves = []

    for cell_id, cell in cells.items():

        base_resolution = float(
            base_resolution_by_cell[
                cell_id
            ]
        )

        if mode == "A":
            requested_resolution = 0.05

        elif mode == "B":
            requested_resolution = (
                base_resolution
            )

        elif mode == "C":
            requested_resolution = float(
                analyzed_cells[
                    cell_id
                ]["refinement_request"][
                    "requested_resolution"
                ]
            )

        else:
            raise ValueError(
                "mode must be A, B, or C."
            )

        refinement_request = {
            "region": {
                "cell_id": cell_id,
                "bbox": cell["bbox"],
            },
            "requested_resolution": (
                requested_resolution
            ),
            "priority": float(
                analyzed_cells[
                    cell_id
                ]["terrain_priority"]
            ),
            "reason": mode,
        }

        result = (
            build_terrain_quadtree_for_cell(
                data=data,
                cell=cell,
                refinement_request=(
                    refinement_request
                ),
                max_plane_residual=(
                    NO_ERROR_BOUND
                ),
                max_elevation_variation=(
                    NO_ERROR_BOUND
                ),
                min_resolution=(
                    MIN_RESOLUTION
                ),
            )
        )

        for leaf in result["leaves"]:

            if len(leaf["point_indices"]) == 0:
                continue

            leaf = dict(leaf)

            # Connect each leaf to the original
            # analyzed parent terrain cell so the
            # deployment representation can inherit
            # the actual terrain attributes.
            leaf["_parent_cell_id"] = cell_id

            leaves.append(leaf)

    return leaves


def geometry_status_code(value):
    if value not in GEOMETRY_STATUS_CODES:
        raise ValueError(
            f"Invalid geometry status: {value}"
        )

    return GEOMETRY_STATUS_CODES[value]


def terrain_code(value):
    if value not in TRAVERSABILITY_CODES:
        raise ValueError(
            f"Invalid traversability: {value}"
        )

    return TRAVERSABILITY_CODES[value]


def optional_float(value):
    if value is None:
        return np.nan

    return float(value)


def build_deployment_map(
    data,
    leaves,
    analyzed_cells,
):
    compact = np.empty(
        len(leaves),
        dtype=COMPACT_DTYPE,
    )

    for index, leaf in enumerate(leaves):

        point_indices = np.asarray(
            leaf["point_indices"],
            dtype=np.int64,
        )

        if len(point_indices) == 0:
            continue

        parent_cell_id = leaf.get(
            "_parent_cell_id"
        )

        if parent_cell_id not in analyzed_cells:
            raise ValueError(
                "Leaf is missing a valid parent "
                "terrain analysis entry."
            )

        terrain_features = analyzed_cells[
            parent_cell_id
        ]

        points = data[
            point_indices
        ]

        bbox = leaf["bbox"]

        x = (
            bbox["x_min"]
            + bbox["x_max"]
        ) / 2.0

        y = (
            bbox["y_min"]
            + bbox["y_max"]
        ) / 2.0

        z_values = (
            points[:, 2]
            .astype(np.float64)
        )

        compact[index]["x"] = x

        compact[index]["y"] = y

        compact[index]["elevation"] = (
            np.mean(z_values)
        )

        compact[index]["z_min"] = (
            np.min(z_values)
        )

        compact[index]["z_max"] = (
            np.max(z_values)
        )

        compact[index]["slope"] = (
            optional_float(
                terrain_features[
                    "slope"
                ]
            )
        )

        compact[index]["roughness"] = (
            optional_float(
                terrain_features[
                    "roughness"
                ]
            )
        )

        compact[index][
            "elevation_variation"
        ] = optional_float(
            terrain_features[
                "elevation_variation"
            ]
        )

        compact[index][
            "terrain_complexity"
        ] = optional_float(
            terrain_features[
                "terrain_complexity"
            ]
        )

        compact[index][
            "geometry_status"
        ] = geometry_status_code(
            terrain_features[
                "geometry_status"
            ]
        )

        compact[index][
            "traversability"
        ] = terrain_code(
            terrain_features[
                "traversability"
            ]
        )

        compact[index][
            "terrain_confidence"
        ] = float(
            terrain_features[
                "terrain_confidence"
            ]
        )

        compact[index][
            "terrain_priority"
        ] = float(
            terrain_features[
                "terrain_priority"
            ]
        )

        compact[index]["resolution"] = (
            leaf["cell_size"]
        )

        compact[index]["depth"] = (
            leaf["depth"]
        )

        compact[index]["point_count"] = (
            len(point_indices)
        )

    return compact


def calculate_elevation_error(
    data,
    leaves,
    ground_point_ids,
):
    reconstruction = np.full(
        len(data),
        np.nan,
        dtype=np.float64,
    )

    actual_ids = []

    for leaf in leaves:

        ids = np.asarray(
            leaf["point_indices"],
            dtype=np.int64,
        )

        if len(ids) == 0:
            continue

        z_mean = float(
            np.mean(
                data[
                    ids,
                    2,
                ]
            )
        )

        reconstruction[ids] = z_mean

        actual_ids.extend(
            ids.tolist()
        )

    actual_ids = np.asarray(
        actual_ids,
        dtype=np.int64,
    )

    expected = np.asarray(
        ground_point_ids,
        dtype=np.int64,
    )

    conservation = (
        len(actual_ids) == len(expected)
        and len(np.unique(actual_ids))
        == len(actual_ids)
        and np.array_equal(
            np.sort(actual_ids),
            np.sort(expected),
        )
    )

    reconstructed = reconstruction[
        expected
    ]

    original = data[
        expected,
        2,
    ].astype(np.float64)

    if np.any(
        ~np.isfinite(reconstructed)
    ):
        raise ValueError(
            "Some ground points were not reconstructed."
        )

    errors = np.abs(
        original
        - reconstructed
    )

    return {
        "point_conservation": bool(
            conservation
        ),
        "mae_m": float(
            np.mean(errors)
        ),
        "rmse_m": float(
            np.sqrt(
                np.mean(errors ** 2)
            )
        ),
        "p95_m": float(
            np.percentile(
                errors,
                95,
            )
        ),
        "max_m": float(
            np.max(errors)
        ),
    }


def calculate_resolution_distribution(
    leaves,
):
    counts = {}

    for leaf in leaves:

        resolution = round(
            float(
                leaf["cell_size"]
            ),
            6,
        )

        counts[resolution] = (
            counts.get(
                resolution,
                0,
            )
            + 1
        )

    return counts


def run_mode(
    data,
    ground_point_ids,
    cells,
    base_resolution_by_cell,
    analyzed_cells,
    mode,
):
    gc.collect()

    tracemalloc.start()

    start = time.perf_counter()

    leaves = build_leaves(
        data=data,
        cells=cells,
        base_resolution_by_cell=(
            base_resolution_by_cell
        ),
        analyzed_cells=analyzed_cells,
        mode=mode,
    )

    compact = build_deployment_map(
        data=data,
        leaves=leaves,
        analyzed_cells=analyzed_cells,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    _, peak = (
        tracemalloc.get_traced_memory()
    )

    tracemalloc.stop()

    elevation = calculate_elevation_error(
        data=data,
        leaves=leaves,
        ground_point_ids=(
            ground_point_ids
        ),
    )

    return (
        leaves,
        compact,
        elapsed,
        peak,
        elevation,
    )


def main():
    print(
        "DEPLOYMENT-STYLE 2.5D MEMORY BENCHMARK"
    )

    print(
        "Uniform vs distance-adaptive vs "
        "distance + terrain"
    )

    data = load_data()

    (
        ground_point_ids,
        cells,
        base_resolution_by_cell,
        analyzed_cells,
    ) = build_inputs(data)

    print()
    print(
        "LiDAR points          :",
        len(data),
    )

    print(
        "Ground points         :",
        len(ground_point_ids),
    )

    print(
        "Parent terrain cells  :",
        len(cells),
    )

    baseline_bytes = None
    baseline_cells = None

    for mode in (
        "A",
        "B",
        "C",
    ):

        (
            leaves,
            compact,
            elapsed,
            peak,
            elevation,
        ) = run_mode(
            data=data,
            ground_point_ids=(
                ground_point_ids
            ),
            cells=cells,
            base_resolution_by_cell=(
                base_resolution_by_cell
            ),
            analyzed_cells=analyzed_cells,
            mode=mode,
        )

        if mode == "A":

            baseline_bytes = (
                compact.nbytes
            )

            baseline_cells = len(
                leaves
            )

        memory_reduction = (
            100.0
            * (
                1.0
                - compact.nbytes
                / baseline_bytes
            )
        )

        cell_reduction = (
            100.0
            * (
                1.0
                - len(leaves)
                / baseline_cells
            )
        )

        print()
        print(
            f"MODE {mode}"
        )

        if mode == "A":

            print(
                "  meaning              : "
                "uniform 5 cm"
            )

        elif mode == "B":

            print(
                "  meaning              : "
                "distance-adaptive"
            )

        else:

            print(
                "  meaning              : "
                "distance + terrain"
            )

        print(
            "  cells                :",
            len(leaves),
        )

        print(
            "  record bytes         :",
            COMPACT_DTYPE.itemsize,
        )

        print(
            "  compact map bytes    :",
            compact.nbytes,
        )

        print(
            "  compact map KiB      :",
            round(
                compact.nbytes / 1024.0,
                3,
            ),
        )

        print(
            "  memory reduction %   :",
            round(
                memory_reduction,
                3,
            ),
        )

        print(
            "  cell reduction %     :",
            round(
                cell_reduction,
                3,
            ),
        )

        print(
            "  peak Python KiB      :",
            round(
                peak / 1024.0,
                3,
            ),
        )

        print(
            "  build time seconds   :",
            round(
                elapsed,
                4,
            ),
        )

        print(
            "  point conservation   :",
            elevation[
                "point_conservation"
            ],
        )

        print(
            "  elevation MAE m      :",
            round(
                elevation["mae_m"],
                6,
            ),
        )

        print(
            "  elevation RMSE m     :",
            round(
                elevation["rmse_m"],
                6,
            ),
        )

        print(
            "  elevation P95 m      :",
            round(
                elevation["p95_m"],
                6,
            ),
        )

        print(
            "  elevation max m      :",
            round(
                elevation["max_m"],
                6,
            )
        )

        print(
            "  resolutions:"
        )

        distribution = (
            calculate_resolution_distribution(
                leaves
            )
        )

        for resolution, count in sorted(
            distribution.items()
        ):

            print(
                f"    {resolution:.6f} m"
                f" : {count}"
            )

    print()
    print(
        "Benchmark complete."
    )


if __name__ == "__main__":
    main()