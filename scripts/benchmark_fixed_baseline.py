from __future__ import annotations

import gc
import time
import tracemalloc

import numpy as np

from src.road_processor import extract_ground_mask
from src.terrain_analysis import analyze_terrain_cells
from src.terrain_cells import build_terrain_cells
from src.terrain_resolution import build_base_resolution_by_cell
from src.terrain_quadtree import (
    build_terrain_quadtree_for_cell,
)


DATA_FILE = (
    "data/RELLIS3D/00000/"
    "vel_cloud_node_kitti_bin/000000.bin"
)

FIXED_RESOLUTION = 0.05
MIN_RESOLUTION = 0.05
NO_ERROR_BOUND = 1_000_000.0


# The same compact record layout is used for all
# three representations so the memory comparison
# is based on the same stored fields.
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


def build_inputs(data: np.ndarray):
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


def build_fixed_5cm_grid(
    data: np.ndarray,
    ground_point_ids: np.ndarray,
):
    """
    Build a conventional fixed-resolution 2.5D
    occupied grid from ground points.

    Every occupied XY cell has exactly 5 cm
    spatial resolution.

    One cell stores one mean elevation plus
    basic elevation statistics.
    """

    cells = {}

    ids = np.asarray(
        ground_point_ids,
        dtype=np.int64,
    )

    for point_id in ids:

        x = float(
            data[
                point_id,
                0,
            ]
        )

        y = float(
            data[
                point_id,
                1,
            ]
        )

        z = float(
            data[
                point_id,
                2,
            ]
        )

        grid_x = int(
            np.floor(
                x / FIXED_RESOLUTION
            )
        )

        grid_y = int(
            np.floor(
                y / FIXED_RESOLUTION
            )
        )

        key = (
            grid_x,
            grid_y,
        )

        if key not in cells:

            cells[key] = {
                "x_min": (
                    grid_x
                    * FIXED_RESOLUTION
                ),
                "x_max": (
                    (grid_x + 1)
                    * FIXED_RESOLUTION
                ),
                "y_min": (
                    grid_y
                    * FIXED_RESOLUTION
                ),
                "y_max": (
                    (grid_y + 1)
                    * FIXED_RESOLUTION
                ),
                "z_values": [],
                "point_ids": [],
            }

        cells[key][
            "z_values"
        ].append(z)

        cells[key][
            "point_ids"
        ].append(int(point_id))

    return cells


def build_fixed_compact_map(
    grid_cells,
):
    """
    Convert the fixed grid into the same compact
    record layout used for the adaptive maps.

    Terrain-specific fields are not used by the
    fixed baseline. They remain empty/default,
    while spatial/elevation fields are populated.
    """

    compact = np.empty(
        len(grid_cells),
        dtype=COMPACT_DTYPE,
    )

    for index, (
        key,
        cell,
    ) in enumerate(
        grid_cells.items()
    ):

        z_values = np.asarray(
            cell["z_values"],
            dtype=np.float64,
        )

        x = (
            cell["x_min"]
            + cell["x_max"]
        ) / 2.0

        y = (
            cell["y_min"]
            + cell["y_max"]
        ) / 2.0

        compact[index][
            "x"
        ] = x

        compact[index][
            "y"
        ] = y

        compact[index][
            "elevation"
        ] = np.mean(
            z_values
        )

        compact[index][
            "z_min"
        ] = np.min(
            z_values
        )

        compact[index][
            "z_max"
        ] = np.max(
            z_values
        )

        compact[index][
            "slope"
        ] = np.nan

        compact[index][
            "roughness"
        ] = np.nan

        compact[index][
            "elevation_variation"
        ] = (
            np.max(z_values)
            - np.min(z_values)
        )

        compact[index][
            "terrain_complexity"
        ] = np.nan

        compact[index][
            "geometry_status"
        ] = GEOMETRY_STATUS_CODES[
            "sparse"
        ]

        compact[index][
            "traversability"
        ] = TRAVERSABILITY_CODES[
            "SPARSE / UNKNOWN"
        ]

        compact[index][
            "terrain_confidence"
        ] = 0.0

        compact[index][
            "terrain_priority"
        ] = 0.0

        compact[index][
            "resolution"
        ] = FIXED_RESOLUTION

        compact[index][
            "depth"
        ] = 0

        compact[index][
            "point_count"
        ] = len(
            cell["point_ids"]
        )

    return compact


def build_adaptive_leaves(
    data: np.ndarray,
    cells: dict,
    base_resolution_by_cell: dict,
    analyzed_cells: dict,
    use_terrain: bool,
):
    leaves = []

    for cell_id, cell in cells.items():

        base_resolution = float(
            base_resolution_by_cell[
                cell_id
            ]
        )

        if use_terrain:

            requested_resolution = float(
                analyzed_cells[
                    cell_id
                ][
                    "refinement_request"
                ][
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

            reason = (
                "distance + terrain"
            )

        else:

            requested_resolution = (
                base_resolution
            )

            priority = 0.0

            reason = (
                "distance-only"
            )

        refinement_request = {
            "region": {
                "cell_id": cell_id,
                "bbox": cell["bbox"],
            },
            "requested_resolution": (
                requested_resolution
            ),
            "priority": priority,
            "reason": reason,
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

        for leaf in result[
            "leaves"
        ]:

            if len(
                leaf[
                    "point_indices"
                ]
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


def optional_float(
    value,
):
    if value is None:
        return np.nan

    return float(value)


def build_adaptive_compact_map(
    data: np.ndarray,
    leaves: list[dict],
    analyzed_cells: dict,
):
    compact = np.empty(
        len(leaves),
        dtype=COMPACT_DTYPE,
    )

    for index, leaf in enumerate(
        leaves
    ):

        point_indices = np.asarray(
            leaf[
                "point_indices"
            ],
            dtype=np.int64,
        )

        parent_cell_id = leaf[
            "_parent_cell_id"
        ]

        terrain = analyzed_cells[
            parent_cell_id
        ]

        points = data[
            point_indices
        ]

        bbox = leaf[
            "bbox"
        ]

        x = (
            bbox["x_min"]
            + bbox["x_max"]
        ) / 2.0

        y = (
            bbox["y_min"]
            + bbox["y_max"]
        ) / 2.0

        z_values = (
            points[
                :,
                2,
            ].astype(
                np.float64
            )
        )

        compact[index][
            "x"
        ] = x

        compact[index][
            "y"
        ] = y

        compact[index][
            "elevation"
        ] = np.mean(
            z_values
        )

        compact[index][
            "z_min"
        ] = np.min(
            z_values
        )

        compact[index][
            "z_max"
        ] = np.max(
            z_values
        )

        compact[index][
            "slope"
        ] = optional_float(
            terrain[
                "slope"
            ]
        )

        compact[index][
            "roughness"
        ] = optional_float(
            terrain[
                "roughness"
            ]
        )

        compact[index][
            "elevation_variation"
        ] = optional_float(
            terrain[
                "elevation_variation"
            ]
        )

        compact[index][
            "terrain_complexity"
        ] = optional_float(
            terrain[
                "terrain_complexity"
            ]
        )

        compact[index][
            "geometry_status"
        ] = (
            GEOMETRY_STATUS_CODES[
                terrain[
                    "geometry_status"
                ]
            ]
        )

        compact[index][
            "traversability"
        ] = (
            TRAVERSABILITY_CODES[
                terrain[
                    "traversability"
                ]
            ]
        )

        compact[index][
            "terrain_confidence"
        ] = float(
            terrain[
                "terrain_confidence"
            ]
        )

        compact[index][
            "terrain_priority"
        ] = float(
            terrain[
                "terrain_priority"
            ]
        )

        compact[index][
            "resolution"
        ] = float(
            leaf[
                "cell_size"
            ]
        )

        compact[index][
            "depth"
        ] = int(
            leaf[
                "depth"
            ]
        )

        compact[index][
            "point_count"
        ] = len(
            point_indices
        )

    return compact


def calculate_elevation_error(
    data: np.ndarray,
    cells: list[dict] | dict,
    ground_point_ids: np.ndarray,
):
    reconstruction = np.full(
        len(data),
        np.nan,
        dtype=np.float64,
    )

    actual_ids = []

    if isinstance(
        cells,
        dict,
    ):

        iterable = cells.values()

    else:

        iterable = cells

    for cell in iterable:

        if "point_indices" in cell:

            ids = np.asarray(
                cell[
                    "point_indices"
                ],
                dtype=np.int64,
            )

        else:

            ids = np.asarray(
                cell[
                    "point_ids"
                ],
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

        reconstruction[
            ids
        ] = z_mean

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
        len(actual_ids)
        == len(expected)
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

    if np.any(
        ~np.isfinite(
            reconstructed
        )
    ):
        raise ValueError(
            "Some ground points were not "
            "reconstructed."
        )

    original = data[
        expected,
        2,
    ].astype(
        np.float64
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
                np.mean(
                    errors ** 2
                )
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


def run_fixed(
    data,
    ground_point_ids,
):

    gc.collect()

    tracemalloc.start()

    start = time.perf_counter()

    grid = build_fixed_5cm_grid(
        data,
        ground_point_ids,
    )

    compact = build_fixed_compact_map(
        grid
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    _, peak = (
        tracemalloc.get_traced_memory()
    )

    tracemalloc.stop()

    metrics = calculate_elevation_error(
        data=data,
        cells=grid,
        ground_point_ids=(
            ground_point_ids
        ),
    )

    return (
        grid,
        compact,
        elapsed,
        peak,
        metrics,
    )


def run_adaptive(
    data,
    ground_point_ids,
    cells,
    base_resolution_by_cell,
    analyzed_cells,
    use_terrain,
):

    gc.collect()

    tracemalloc.start()

    start = time.perf_counter()

    leaves = build_adaptive_leaves(
        data=data,
        cells=cells,
        base_resolution_by_cell=(
            base_resolution_by_cell
        ),
        analyzed_cells=analyzed_cells,
        use_terrain=use_terrain,
    )

    compact = build_adaptive_compact_map(
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

    metrics = calculate_elevation_error(
        data=data,
        cells=leaves,
        ground_point_ids=(
            ground_point_ids
        ),
    )

    return (
        leaves,
        compact,
        elapsed,
        peak,
        metrics,
    )


def print_result(
    name,
    compact,
    elapsed,
    peak,
    metrics,
    baseline_bytes=None,
):
    print()
    print(
        name
    )

    print(
        "  cells                 :",
        len(compact),
    )

    print(
        "  bytes per record      :",
        COMPACT_DTYPE.itemsize,
    )

    print(
        "  map memory bytes      :",
        compact.nbytes,
    )

    print(
        "  map memory KiB        :",
        round(
            compact.nbytes
            / 1024.0,
            3,
        ),
    )

    if baseline_bytes is not None:

        reduction = (
            100.0
            * (
                1.0
                - compact.nbytes
                / baseline_bytes
            )
        )

        print(
            "  memory reduction %    :",
            round(
                reduction,
                3,
            ),
        )

    print(
        "  peak Python KiB       :",
        round(
            peak / 1024.0,
            3,
        ),
    )

    print(
        "  build time seconds    :",
        round(
            elapsed,
            4,
        ),
    )

    print(
        "  point conservation    :",
        metrics[
            "point_conservation"
        ],
    )

    print(
        "  elevation MAE m       :",
        round(
            metrics[
                "mae_m"
            ],
            6,
        ),
    )

    print(
        "  elevation RMSE m      :",
        round(
            metrics[
                "rmse_m"
            ],
            6,
        ),
    )

    print(
        "  elevation P95 m       :",
        round(
            metrics[
                "p95_m"
            ],
            6,
        ),
    )

    print(
        "  elevation max m       :",
        round(
            metrics[
                "max_m"
            ],
            6,
        ),
    )


def main():

    print(
        "FIXED BASELINE VS ADAPTIVE 2.5D"
    )

    print(
        "Frame:",
        DATA_FILE,
    )

    data = load_data()

    (
        ground_point_ids,
        cells,
        base_resolution_by_cell,
        analyzed_cells,
    ) = build_inputs(
        data
    )

    print()
    print(
        "LiDAR points           :",
        len(data),
    )

    print(
        "Ground points          :",
        len(ground_point_ids),
    )

    print(
        "Parent terrain cells   :",
        len(cells),
    )

    (
        fixed_grid,
        fixed_compact,
        fixed_time,
        fixed_peak,
        fixed_metrics,
    ) = run_fixed(
        data=data,
        ground_point_ids=(
            ground_point_ids
        ),
    )

    print_result(
        "MODE A - Fixed 5 cm baseline",
        fixed_compact,
        fixed_time,
        fixed_peak,
        fixed_metrics,
    )

    (
        distance_leaves,
        distance_compact,
        distance_time,
        distance_peak,
        distance_metrics,
    ) = run_adaptive(
        data=data,
        ground_point_ids=(
            ground_point_ids
        ),
        cells=cells,
        base_resolution_by_cell=(
            base_resolution_by_cell
        ),
        analyzed_cells=analyzed_cells,
        use_terrain=False,
    )

    print_result(
        "MODE B - Distance adaptive",
        distance_compact,
        distance_time,
        distance_peak,
        distance_metrics,
        baseline_bytes=(
            fixed_compact.nbytes
        ),
    )

    (
        terrain_leaves,
        terrain_compact,
        terrain_time,
        terrain_peak,
        terrain_metrics,
    ) = run_adaptive(
        data=data,
        ground_point_ids=(
            ground_point_ids
        ),
        cells=cells,
        base_resolution_by_cell=(
            base_resolution_by_cell
        ),
        analyzed_cells=analyzed_cells,
        use_terrain=True,
    )

    print_result(
        "MODE C - Distance + terrain",
        terrain_compact,
        terrain_time,
        terrain_peak,
        terrain_metrics,
        baseline_bytes=(
            fixed_compact.nbytes
        ),
    )

    print()
    print(
        "=" * 70
    )

    print(
        "TERRAIN IMPACT"
    )

    print(
        "Distance cells         :",
        len(distance_compact),
    )

    print(
        "Terrain cells          :",
        len(terrain_compact),
    )

    print(
        "Additional cells       :",
        len(terrain_compact)
        - len(distance_compact),
    )

    print(
        "Distance memory KiB    :",
        round(
            distance_compact.nbytes
            / 1024.0,
            3,
        ),
    )

    print(
        "Terrain memory KiB     :",
        round(
            terrain_compact.nbytes
            / 1024.0,
            3,
        ),
    )

    print(
        "Memory change B -> C % :",
        round(
            100.0
            * (
                terrain_compact.nbytes
                / distance_compact.nbytes
                - 1.0
            ),
            3,
        ),
    )

    print(
        "RMSE change B -> C %  :",
        round(
            100.0
            * (
                terrain_metrics[
                    "rmse_m"
                ]
                / distance_metrics[
                    "rmse_m"
                ]
                - 1.0
            ),
            3,
        ),
    )

    print()
    print(
        "Benchmark complete."
    )


if __name__ == "__main__":
    main()