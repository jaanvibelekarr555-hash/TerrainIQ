from __future__ import annotations

import csv
import gc
import os
import statistics
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


DATA_ROOT = "data/RELLIS3D"

SEQUENCES = (
    "00000",
    "00001",
    "00002",
    "00003",
)

FRAME = "000000"

CELL_SIZE = 1.0

MIN_RESOLUTION = 0.05

MEASURED_RUNS = 5


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


MODE_NAMES = {
    "A": "Uniform 5 cm",
    "B": "Distance-adaptive",
    "C": "Distance + Terrain",
}


def load_data(
    sequence: str,
) -> np.ndarray:

    file_path = (
        f"{DATA_ROOT}/{sequence}/"
        f"vel_cloud_node_kitti_bin/{FRAME}.bin"
    )

    data = np.fromfile(
        file_path,
        dtype=np.float32,
    ).reshape(-1, 4)

    if len(data) == 0:
        raise ValueError(
            f"LiDAR frame is empty: {file_path}"
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
        cell_size=CELL_SIZE,
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
            base_resolution_by_cell[
                cell_id
            ]
        )

        if mode == "A":

            # Uniform 5 cm baseline.
            base_resolution = 0.05
            requested_resolution = 0.05

        elif mode == "B":

            # Distance determines the base resolution.
            requested_resolution = (
                base_resolution
            )

        elif mode == "C":

            # Distance determines the base resolution.
            # Terrain priority requests additional
            # local refinement.
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

        if requested_resolution > (
            base_resolution + 1e-12
        ):
            raise ValueError(
                "Requested resolution cannot be "
                "coarser than base resolution."
            )

        refinement_request = {
            "region": {
                "cell_id": cell_id,
                "bbox": cell["bbox"],
            },

            # IMPORTANT:
            # The new terrain hierarchy needs to know
            # the actual distance-based base resolution.
            "base_resolution": (
                base_resolution
            ),

            "requested_resolution": (
                requested_resolution
            ),

            "priority": float(
                analyzed_cells[
                    cell_id
                ]["terrain_priority"]
            ),

            "reason": MODE_NAMES[
                mode
            ],
        }

        result = (
            build_terrain_quadtree_for_cell(
                data=data,
                cell=cell,
                refinement_request=(
                    refinement_request
                ),
                min_resolution=MIN_RESOLUTION,
            )
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


def geometry_status_code(
    value: str,
) -> int:

    if value not in GEOMETRY_STATUS_CODES:
        raise ValueError(
            f"Invalid geometry status: {value}"
        )

    return GEOMETRY_STATUS_CODES[
        value
    ]


def terrain_code(
    value: str,
) -> int:

    if value not in TRAVERSABILITY_CODES:
        raise ValueError(
            f"Invalid traversability: {value}"
        )

    return TRAVERSABILITY_CODES[
        value
    ]


def optional_float(
    value,
) -> float:

    if value is None:
        return np.nan

    return float(value)


def build_deployment_map(
    data: np.ndarray,
    leaves: list[dict],
    analyzed_cells: dict,
) -> np.ndarray:

    compact = np.empty(
        len(leaves),
        dtype=COMPACT_DTYPE,
    )

    for index, leaf in enumerate(
        leaves
    ):

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
                "Leaf is missing a valid "
                "parent terrain analysis entry."
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
    data: np.ndarray,
    leaves: list[dict],
    ground_point_ids: np.ndarray,
) -> dict:

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

    original = data[
        expected,
        2,
    ].astype(np.float64)

    if np.any(
        ~np.isfinite(
            reconstructed
        )
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


def calculate_resolution_distribution(
    leaves: list[dict],
) -> dict:

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


def run_single_measurement(
    data: np.ndarray,
    ground_point_ids: np.ndarray,
    cells: dict,
    base_resolution_by_cell: dict,
    analyzed_cells: dict,
    mode: str,
) -> dict:

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

    elevation = (
        calculate_elevation_error(
            data=data,
            leaves=leaves,
            ground_point_ids=(
                ground_point_ids
            ),
        )
    )

    return {
        "cells": len(
            leaves
        ),

        "compact_map_bytes": int(
            compact.nbytes
        ),

        "peak_python_bytes": int(
            peak
        ),

        "build_time_seconds": float(
            elapsed
        ),

        **elevation,

        "resolution_distribution": (
            calculate_resolution_distribution(
                leaves
            )
        ),
    }


def run_repeated_measurements(
    data: np.ndarray,
    ground_point_ids: np.ndarray,
    cells: dict,
    base_resolution_by_cell: dict,
    analyzed_cells: dict,
    mode: str,
) -> tuple[dict, list[dict]]:

    # Warm-up run is excluded from measured runs.
    run_single_measurement(
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

    measurements = []

    for run_index in range(
        MEASURED_RUNS
    ):

        result = run_single_measurement(
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

        result["run"] = (
            run_index + 1
        )

        measurements.append(
            result
        )

    median_result = {
        "cells": int(
            statistics.median(
                result["cells"]
                for result in measurements
            )
        ),

        "compact_map_bytes": int(
            statistics.median(
                result[
                    "compact_map_bytes"
                ]
                for result in measurements
            )
        ),

        "peak_python_bytes": int(
            statistics.median(
                result[
                    "peak_python_bytes"
                ]
                for result in measurements
            )
        ),

        "build_time_seconds": float(
            statistics.median(
                result[
                    "build_time_seconds"
                ]
                for result in measurements
            )
        ),

        "mae_m": float(
            statistics.median(
                result["mae_m"]
                for result in measurements
            )
        ),

        "rmse_m": float(
            statistics.median(
                result["rmse_m"]
                for result in measurements
            )
        ),

        "p95_m": float(
            statistics.median(
                result["p95_m"]
                for result in measurements
            )
        ),

        "max_m": float(
            statistics.median(
                result["max_m"]
                for result in measurements
            )
        ),

        "point_conservation": all(
            result[
                "point_conservation"
            ]
            for result in measurements
        ),

        "resolution_distribution": (
            measurements[0][
                "resolution_distribution"
            ]
        ),
    }

    return (
        median_result,
        measurements,
    )


def percent_reduction(
    baseline: float,
    value: float,
) -> float:

    if baseline == 0:
        return 0.0

    return (
        100.0
        * (
            1.0
            - value / baseline
        )
    )


def calculate_terrain_refinement_stats(
    cells: dict,
    base_resolution_by_cell: dict,
    analyzed_cells: dict,
) -> dict:

    refined = 0
    high_priority = 0
    high_priority_refined = 0

    for cell_id in cells:

        base = float(
            base_resolution_by_cell[
                cell_id
            ]
        )

        priority = float(
            analyzed_cells[
                cell_id
            ]["terrain_priority"]
        )

        request = analyzed_cells[
            cell_id
        ]["refinement_request"]

        requested = float(
            request[
                "requested_resolution"
            ]
        )

        if requested < (
            base - 1e-12
        ):
            refined += 1

        if priority >= 0.70:

            high_priority += 1

            if requested < (
                base - 1e-12
            ):
                high_priority_refined += 1

    refinement_rate = (
        100.0
        * refined
        / len(cells)
        if cells
        else 0.0
    )

    high_priority_rate = (
        100.0
        * high_priority_refined
        / high_priority
        if high_priority
        else 0.0
    )

    return {
        "refined_parent_cells": refined,
        "refinement_rate_percent": (
            refinement_rate
        ),
        "high_priority_cells": (
            high_priority
        ),
        "high_priority_refined": (
            high_priority_refined
        ),
        "high_priority_refinement_rate_percent": (
            high_priority_rate
        ),
    }


def main():

    print(
        "REPEATED TERRAIN-IMPACT BENCHMARK"
    )

    print(
        f"Measured runs per mode: "
        f"{MEASURED_RUNS}"
    )

    print(
        "Modes: Uniform 5 cm, "
        "Distance-adaptive, "
        "Distance + Terrain"
    )

    all_rows = []

    for sequence in SEQUENCES:

        print()
        print(
            "=" * 78
        )

        print(
            f"Sequence {sequence} "
            f"Frame {FRAME}"
        )

        data = load_data(
            sequence
        )

        (
            ground_point_ids,
            cells,
            base_resolution_by_cell,
            analyzed_cells,
        ) = build_parent_inputs(
            data
        )

        terrain_stats = (
            calculate_terrain_refinement_stats(
                cells=cells,
                base_resolution_by_cell=(
                    base_resolution_by_cell
                ),
                analyzed_cells=(
                    analyzed_cells
                ),
            )
        )

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

        print(
            "Terrain-refined parent:",
            terrain_stats[
                "refined_parent_cells"
            ],
            f"({terrain_stats['refinement_rate_percent']:.2f}%)",
        )

        print(
            "High-priority parents :",
            terrain_stats[
                "high_priority_cells"
            ],
        )

        print(
            "High-priority refined :",
            terrain_stats[
                "high_priority_refined"
            ],
            f"({terrain_stats['high_priority_refinement_rate_percent']:.2f}%)",
        )

        sequence_results = {}

        for mode in (
            "A",
            "B",
            "C",
        ):

            median_result, measurements = (
                run_repeated_measurements(
                    data=data,
                    ground_point_ids=(
                        ground_point_ids
                    ),
                    cells=cells,
                    base_resolution_by_cell=(
                        base_resolution_by_cell
                    ),
                    analyzed_cells=(
                        analyzed_cells
                    ),
                    mode=mode,
                )
            )

            sequence_results[
                mode
            ] = median_result

            for measurement in measurements:

                all_rows.append(
                    {
                        "sequence": sequence,
                        "mode": mode,
                        "run": measurement[
                            "run"
                        ],
                        "cells": measurement[
                            "cells"
                        ],
                        "compact_map_bytes": (
                            measurement[
                                "compact_map_bytes"
                            ]
                        ),
                        "peak_python_bytes": (
                            measurement[
                                "peak_python_bytes"
                            ]
                        ),
                        "build_time_seconds": (
                            measurement[
                                "build_time_seconds"
                            ]
                        ),
                        "mae_m": measurement[
                            "mae_m"
                        ],
                        "rmse_m": measurement[
                            "rmse_m"
                        ],
                        "p95_m": measurement[
                            "p95_m"
                        ],
                        "max_m": measurement[
                            "max_m"
                        ],
                        "point_conservation": (
                            measurement[
                                "point_conservation"
                            ]
                        ),
                    }
                )

            print()
            print(
                f"MODE {mode} "
                f"({MODE_NAMES[mode]})"
            )

            print(
                "  median cells          :",
                median_result[
                    "cells"
                ],
            )

            print(
                "  compact map KiB       :",
                round(
                    median_result[
                        "compact_map_bytes"
                    ]
                    / 1024.0,
                    3,
                ),
            )

            print(
                "  median peak Python KiB:",
                round(
                    median_result[
                        "peak_python_bytes"
                    ]
                    / 1024.0,
                    3,
                ),
            )

            print(
                "  median build time s   :",
                round(
                    median_result[
                        "build_time_seconds"
                    ],
                    4,
                ),
            )

            print(
                "  median elevation RMSE:",
                round(
                    median_result[
                        "rmse_m"
                    ],
                    6,
                ),
            )

            print(
                "  median elevation MAE :",
                round(
                    median_result[
                        "mae_m"
                    ],
                    6,
                ),
            )

            print(
                "  median elevation P95 :",
                round(
                    median_result[
                        "p95_m"
                    ],
                    6,
                ),
            )

            print(
                "  median elevation max :",
                round(
                    median_result[
                        "max_m"
                    ],
                    6,
                ),
            )

            print(
                "  point conservation   :",
                median_result[
                    "point_conservation"
                ],
            )

            print(
                "  resolutions:"
            )

            for resolution, count in sorted(
                median_result[
                    "resolution_distribution"
                ].items()
            ):

                print(
                    f"    {resolution:.6f} m"
                    f" : {count}"
                )

        uniform = sequence_results[
            "A"
        ]

        distance = sequence_results[
            "B"
        ]

        terrain = sequence_results[
            "C"
        ]

        print()
        print(
            "TERRAIN IMPACT (B -> C)"
        )

        print(
            "  cell change            :",
            terrain["cells"]
            - distance["cells"],
        )

        print(
            "  cell change %          :",
            round(
                100.0
                * (
                    terrain["cells"]
                    / distance["cells"]
                    - 1.0
                ),
                3,
            ),
        )

        print(
            "  compact memory change :",
            round(
                (
                    terrain[
                        "compact_map_bytes"
                    ]
                    - distance[
                        "compact_map_bytes"
                    ]
                )
                / 1024.0,
                3,
            ),
            "KiB",
        )

        print(
            "  RMSE change            :",
            round(
                terrain[
                    "rmse_m"
                ]
                - distance[
                    "rmse_m"
                ],
                6,
            ),
            "m",
        )

        print(
            "  RMSE change %          :",
            round(
                100.0
                * (
                    terrain[
                        "rmse_m"
                    ]
                    / distance[
                        "rmse_m"
                    ]
                    - 1.0
                )
                if distance[
                    "rmse_m"
                ] > 0
                else 0.0,
                3,
            ),
        )

        print(
            "  build-time change %    :",
            round(
                100.0
                * (
                    terrain[
                        "build_time_seconds"
                    ]
                    / distance[
                        "build_time_seconds"
                    ]
                    - 1.0
                ),
                3,
            ),
        )

        print()
        print(
            "UNIFORM -> TERRAIN"
        )

        print(
            "  cell reduction %       :",
            round(
                percent_reduction(
                    uniform["cells"],
                    terrain["cells"],
                ),
                3,
            ),
        )

        print(
            "  compact memory reduction %:",
            round(
                percent_reduction(
                    uniform[
                        "compact_map_bytes"
                    ],
                    terrain[
                        "compact_map_bytes"
                    ],
                ),
                3,
            ),
        )

    os.makedirs(
        "results",
        exist_ok=True,
    )

    csv_path = (
        "results/"
        "terrain_impact_repeated_runs.csv"
    )

    if all_rows:

        fieldnames = list(
            all_rows[0].keys()
        )

        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:

            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            writer.writerows(
                all_rows
            )

        print()
        print(
            "CSV output:",
            csv_path,
        )

    print()
    print(
        "=" * 78
    )

    print(
        "Benchmark complete."
    )


if __name__ == "__main__":
    main()