from __future__ import annotations

import gc
import pickle
import time
import tracemalloc

import numpy as np

from src.road_processor import extract_ground_mask
from src.terrain_cells import build_terrain_cells
from src.terrain_resolution import (
    build_base_resolution_by_cell,
)
from src.terrain_analysis import analyze_terrain_cells
from src.terrain_quadtree import (
    build_terrain_quadtree_for_cell,
    validate_terrain_quadtree_point_conservation,
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

# A/B/C disable error-bound refinement by using thresholds
# far above any realistic terrain measurement.
NO_ERROR_BOUND_PLANE = 1_000_000.0
NO_ERROR_BOUND_ELEVATION = 1_000_000.0

FULL_ERROR_BOUND_PLANE = 0.08
FULL_ERROR_BOUND_ELEVATION = 0.20

MIN_RESOLUTION = 0.05


def load_frame(sequence: str) -> np.ndarray:
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
            f"Empty LiDAR frame: {file_path}"
        )

    return data


def build_parent_inputs(
    data: np.ndarray,
):
    ground_mask, point_id, plane_model = (
        extract_ground_mask(data)
    )

    ground_point_ids = point_id[ground_mask]

    cells = build_terrain_cells(
        data=data,
        ground_mask=ground_mask,
        point_id=point_id,
        cell_size=CELL_SIZE,
    )

    base_resolution_by_cell = (
        build_base_resolution_by_cell(cells)
    )

    analyzed_cells = analyze_terrain_cells(
        data=data,
        cells=cells,
        base_resolution_by_cell=base_resolution_by_cell,
    )

    return (
        ground_point_ids,
        cells,
        base_resolution_by_cell,
        analyzed_cells,
    )


def build_mode_map(
    data: np.ndarray,
    cells: dict,
    base_resolution_by_cell: dict,
    analyzed_cells: dict,
    mode: str,
):
    mode = mode.upper()

    if mode not in {"A", "B", "C", "D"}:
        raise ValueError(
            "Mode must be A, B, C, or D."
        )

    terrain_leaf_results = {}

    for cell_id, cell in cells.items():

        base_resolution = (
            base_resolution_by_cell[cell_id]
        )

        if mode == "A":
            requested_resolution = 0.05
            priority = 0.0
            reason = "uniform 5 cm baseline"

            error_plane = (
                NO_ERROR_BOUND_PLANE
            )
            error_elevation = (
                NO_ERROR_BOUND_ELEVATION
            )

        elif mode == "B":
            requested_resolution = (
                base_resolution
            )
            priority = 0.0
            reason = "distance-only base resolution"

            error_plane = (
                NO_ERROR_BOUND_PLANE
            )
            error_elevation = (
                NO_ERROR_BOUND_ELEVATION
            )

        elif mode == "C":
            request = analyzed_cells[
                cell_id
            ]["refinement_request"]

            requested_resolution = float(
                request["requested_resolution"]
            )

            priority = float(
                request["priority"]
            )

            reason = (
                "distance + terrain priority"
            )

            error_plane = (
                NO_ERROR_BOUND_PLANE
            )
            error_elevation = (
                NO_ERROR_BOUND_ELEVATION
            )

        else:
            request = analyzed_cells[
                cell_id
            ]["refinement_request"]

            requested_resolution = float(
                request["requested_resolution"]
            )

            priority = float(
                request["priority"]
            )

            reason = (
                "distance + terrain + error bound"
            )

            error_plane = (
                FULL_ERROR_BOUND_PLANE
            )
            error_elevation = (
                FULL_ERROR_BOUND_ELEVATION
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

        result = build_terrain_quadtree_for_cell(
            data=data,
            cell=cell,
            refinement_request=refinement_request,
            max_plane_residual=error_plane,
            max_elevation_variation=(
                error_elevation
            ),
            min_resolution=MIN_RESOLUTION,
        )

        terrain_leaf_results[cell_id] = result

    return terrain_leaf_results


def flatten_leaves(
    terrain_quadtree_map: dict,
) -> list[dict]:
    leaves = []

    for result in terrain_quadtree_map.values():

        for leaf in result["leaves"]:

            if len(leaf["point_indices"]) == 0:
                continue

            leaves.append(leaf)

    return leaves


def reconstruct_elevation(
    data: np.ndarray,
    leaves: list[dict],
) -> np.ndarray:
    reconstruction = np.full(
        len(data),
        np.nan,
        dtype=np.float64,
    )

    for leaf in leaves:

        indices = np.asarray(
            leaf["point_indices"],
            dtype=np.int64,
        )

        if len(indices) == 0:
            continue

        z_values = data[
            indices,
            2,
        ].astype(float)

        z_mean = float(
            np.mean(z_values)
        )

        reconstruction[indices] = z_mean

    return reconstruction


def calculate_elevation_metrics(
    data: np.ndarray,
    leaves: list[dict],
    ground_point_ids: np.ndarray,
) -> dict:

    reconstruction = reconstruct_elevation(
        data,
        leaves,
    )

    ids = np.asarray(
        ground_point_ids,
        dtype=np.int64,
    )

    original_z = data[
        ids,
        2,
    ].astype(float)

    reconstructed_z = reconstruction[
        ids
    ]

    if np.any(
        ~np.isfinite(reconstructed_z)
    ):
        raise ValueError(
            "Some ground points were not reconstructed."
        )

    errors = np.abs(
        original_z
        - reconstructed_z
    )

    squared_errors = (
        original_z
        - reconstructed_z
    ) ** 2

    return {
        "mae_m": float(
            np.mean(errors)
        ),
        "rmse_m": float(
            np.sqrt(
                np.mean(
                    squared_errors
                )
            )
        ),
        "p95_error_m": float(
            np.percentile(
                errors,
                95,
            )
        ),
        "max_error_m": float(
            np.max(errors)
        ),
    }


def calculate_resolution_distribution(
    leaves: list[dict],
) -> dict:

    counts = {}

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

    return counts


def calculate_parent_refinement(
    cells: dict,
    base_resolution_by_cell: dict,
    analyzed_cells: dict,
    mode: str,
) -> dict:

    mode = mode.upper()

    if mode == "A":
        return {
            "refined_parent_cells": len(cells),
            "parent_refinement_rate_percent": 100.0,
            "extra_refined_parent_cells": 0,
        }

    refined = 0
    extra_refined = 0

    for cell_id in cells:

        base = float(
            base_resolution_by_cell[cell_id]
        )

        if mode == "B":
            requested = base

        else:
            requested = float(
                analyzed_cells[cell_id][
                    "refinement_request"
                ][
                    "requested_resolution"
                ]
            )

        if requested < base:
            refined += 1

            if mode in {"C", "D"}:
                extra_refined += 1

    rate = (
        100.0
        * refined
        / len(cells)
        if cells
        else 0.0
    )

    return {
        "refined_parent_cells": refined,
        "parent_refinement_rate_percent": rate,
        "extra_refined_parent_cells": (
            extra_refined
        ),
    }


def calculate_priority_groups(
    cells: dict,
    analyzed_cells: dict,
    mode: str,
) -> dict:

    if mode not in {"C", "D"}:
        return {
            "high_priority_parent_cells": 0,
            "high_priority_refined_cells": 0,
            "high_priority_refinement_rate_percent": 0.0,
        }

    high_priority = 0
    high_priority_refined = 0

    for cell_id in cells:

        priority = float(
            analyzed_cells[cell_id][
                "terrain_priority"
            ]
        )

        if priority >= 0.70:

            high_priority += 1

            request = analyzed_cells[
                cell_id
            ]["refinement_request"]

            base = float(
                analyzed_cells[
                    cell_id
                ].get(
                    "_base_resolution",
                    0.0,
                )
            )

            requested = float(
                request[
                    "requested_resolution"
                ]
            )

            # Recalculate base from cell geometry
            # when it is not stored in analyzed_cells.
            if base <= 0.0:

                bbox = cells[
                    cell_id
                ]["bbox"]

                x_center = (
                    bbox["x_min"]
                    + bbox["x_max"]
                ) / 2.0

                y_center = (
                    bbox["y_min"]
                    + bbox["y_max"]
                ) / 2.0

                distance = float(
                    np.hypot(
                        x_center,
                        y_center,
                    )
                )

                from src.resolution_engine import (
                    distance_to_resolution,
                )

                base = distance_to_resolution(
                    distance
                )

            if requested < base:
                high_priority_refined += 1

    rate = (
        100.0
        * high_priority_refined
        / high_priority
        if high_priority
        else 0.0
    )

    return {
        "high_priority_parent_cells": (
            high_priority
        ),
        "high_priority_refined_cells": (
            high_priority_refined
        ),
        "high_priority_refinement_rate_percent": (
            rate
        ),
    }


def calculate_map_size(
    leaves: list[dict],
) -> int:

    return len(
        pickle.dumps(
            leaves,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    )


def run_mode(
    data: np.ndarray,
    ground_point_ids: np.ndarray,
    cells: dict,
    base_resolution_by_cell: dict,
    analyzed_cells: dict,
    mode: str,
) -> tuple[dict, list[dict]]:

    gc.collect()

    tracemalloc.start()

    start = time.perf_counter()

    terrain_quadtree_map = build_mode_map(
        data=data,
        cells=cells,
        base_resolution_by_cell=(
            base_resolution_by_cell
        ),
        analyzed_cells=analyzed_cells,
        mode=mode,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    current_bytes, peak_bytes = (
        tracemalloc.get_traced_memory()
    )

    tracemalloc.stop()

    leaves = flatten_leaves(
        terrain_quadtree_map
    )

    conservation = validate_terrain_quadtree_point_conservation(
        terrain_quadtree_map,
        ground_point_ids,
    )

    # validate against the actual expected ground IDs.
    actual_ids = []

    for leaf in leaves:
        actual_ids.extend(
            leaf["point_indices"]
        )

    actual_ids = np.asarray(
        actual_ids,
        dtype=np.int64,
    )

    conservation = (
        conservation
        and len(actual_ids)
        == len(ground_point_ids)
        and len(np.unique(actual_ids))
        == len(actual_ids)
        and np.array_equal(
            np.sort(actual_ids),
            np.sort(
                ground_point_ids
            ),
        )
    )

    elevation = calculate_elevation_metrics(
        data=data,
        leaves=leaves,
        ground_point_ids=(
            ground_point_ids
        ),
    )

    refinement = calculate_parent_refinement(
        cells=cells,
        base_resolution_by_cell=(
            base_resolution_by_cell
        ),
        analyzed_cells=analyzed_cells,
        mode=mode,
    )

    priority = calculate_priority_groups(
        cells=cells,
        analyzed_cells=analyzed_cells,
        mode=mode,
    )

    result = {
        "mode": mode,
        "cell_count": len(leaves),
        "serialized_map_bytes": calculate_map_size(
            leaves
        ),
        "peak_python_memory_bytes": int(
            peak_bytes
        ),
        "build_time_seconds": float(
            elapsed
        ),
        "point_conservation": bool(
            conservation
        ),
        **elevation,
        **refinement,
        **priority,
        "resolution_distribution": (
            calculate_resolution_distribution(
                leaves
            )
        ),
    }

    return result, leaves


def main():

    print(
        "TERRAIN BRANCH IMPACT BENCHMARK"
    )

    print(
        "Modes: A=Uniform, B=Distance, "
        "C=Terrain, D=Full"
    )

    all_results = {}

    for sequence in SEQUENCES:

        print(
            f"\n{'=' * 70}"
        )

        print(
            f"Sequence {sequence} "
            f"Frame {FRAME}"
        )

        data = load_frame(
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

        print(
            f"LiDAR points        : "
            f"{len(data)}"
        )

        print(
            f"Ground points       : "
            f"{len(ground_point_ids)}"
        )

        print(
            f"Terrain parent cells: "
            f"{len(cells)}"
        )

        sequence_results = {}

        for mode in (
            "A",
            "B",
            "C",
            "D",
        ):

            result, leaves = run_mode(
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

            sequence_results[mode] = result

            print(
                f"\nMode {mode}"
            )

            print(
                "  cells              :",
                result["cell_count"],
            )

            print(
                "  serialized map KiB :",
                round(
                    result[
                        "serialized_map_bytes"
                    ]
                    / 1024.0,
                    2,
                ),
            )

            print(
                "  peak Python KiB    :",
                round(
                    result[
                        "peak_python_memory_bytes"
                    ]
                    / 1024.0,
                    2,
                ),
            )

            print(
                "  build time s       :",
                round(
                    result[
                        "build_time_seconds"
                    ],
                    4,
                ),
            )

            print(
                "  point conservation :",
                result[
                    "point_conservation"
                ],
            )

            print(
                "  elevation MAE m    :",
                round(
                    result["mae_m"],
                    6,
                ),
            )

            print(
                "  elevation RMSE m   :",
                round(
                    result["rmse_m"],
                    6,
                ),
            )

            print(
                "  elevation P95 m    :",
                round(
                    result["p95_error_m"],
                    6,
                ),
            )

            print(
                "  elevation max m    :",
                round(
                    result["max_error_m"],
                    6,
                ),
            )

            if mode in {"B", "C", "D"}:

                print(
                    "  refined parents    :",
                    result[
                        "refined_parent_cells"
                    ],
                )

                print(
                    "  refinement rate %  :",
                    round(
                        result[
                            "parent_refinement_rate_percent"
                        ],
                        2,
                    ),
                )

            if mode in {"C", "D"}:

                print(
                    "  high-priority cells:",
                    result[
                        "high_priority_parent_cells"
                    ],
                )

                print(
                    "  high-priority refined:",
                    result[
                        "high_priority_refined_cells"
                    ],
                )

                print(
                    "  high-priority rate %:",
                    round(
                        result[
                            "high_priority_refinement_rate_percent"
                        ],
                        2,
                    ),
                )

            print(
                "  resolutions:"
            )

            for resolution, count in sorted(
                result[
                    "resolution_distribution"
                ].items()
            ):

                print(
                    f"    {resolution:.6f} m"
                    f" : {count}"
                )

        all_results[
            sequence
        ] = sequence_results

    print(
        f"\n{'=' * 70}"
    )

    print(
        "SUMMARY"
    )

    for sequence, results in (
        all_results.items()
    ):

        print(
            f"\nSequence {sequence}"
        )

        baseline_cells = results[
            "A"
        ]["cell_count"]

        baseline_memory = results[
            "A"
        ]["serialized_map_bytes"]

        baseline_rmse = results[
            "A"
        ]["rmse_m"]

        for mode in (
            "A",
            "B",
            "C",
            "D",
        ):

            result = results[
                mode
            ]

            cell_reduction = (
                100.0
                * (
                    1.0
                    - result[
                        "cell_count"
                    ]
                    / baseline_cells
                )
            )

            memory_reduction = (
                100.0
                * (
                    1.0
                    - result[
                        "serialized_map_bytes"
                    ]
                    / baseline_memory
                )
            )

            rmse_increase = (
                100.0
                * (
                    result["rmse_m"]
                    / baseline_rmse
                    - 1.0
                )
                if baseline_rmse > 0
                else 0.0
            )

            print(
                f"  {mode}: "
                f"cells={result['cell_count']} "
                f"({cell_reduction:.2f}% "
                f"reduction), "
                f"memory={result['serialized_map_bytes'] / 1024.0:.2f} KiB "
                f"({memory_reduction:.2f}% reduction), "
                f"RMSE={result['rmse_m']:.6f} m "
                f"({rmse_increase:+.2f}% vs uniform)"
            )

    print(
        "\nBenchmark complete."
    )


if __name__ == "__main__":
    main()
