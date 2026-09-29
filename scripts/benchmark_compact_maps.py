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
        ("z", np.float32),
        ("resolution", np.float32),
        ("point_count", np.uint32),
    ]
)


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


def build_parent_inputs(data):

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


def build_mode(
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
                "Mode must be A, B, or C."
            )

        refinement_request = {
            "region": {
                "cell_id": cell_id,
                "bbox": cell["bbox"],
            },
            "requested_resolution": (
                requested_resolution
            ),
            "priority": 0.0,
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

        leaves.extend(
            leaf
            for leaf in result["leaves"]
            if len(leaf["point_indices"]) > 0
        )

    return leaves


def build_compact_representation(
    data,
    leaves,
):

    compact = np.empty(
        len(leaves),
        dtype=COMPACT_DTYPE,
    )

    for i, leaf in enumerate(leaves):

        indices = np.asarray(
            leaf["point_indices"],
            dtype=np.int64,
        )

        points = data[
            indices
        ]

        bbox = leaf["bbox"]

        compact[i]["x"] = (
            (
                bbox["x_min"]
                + bbox["x_max"]
            )
            / 2.0
        )

        compact[i]["y"] = (
            (
                bbox["y_min"]
                + bbox["y_max"]
            )
            / 2.0
        )

        compact[i]["z"] = np.mean(
            points[:, 2]
        )

        compact[i]["resolution"] = (
            leaf["cell_size"]
        )

        compact[i]["point_count"] = (
            len(indices)
        )

    return compact


def reconstruct_and_validate(
    data,
    leaves,
    ground_point_ids,
):

    reconstruction = np.full(
        len(data),
        np.nan,
        dtype=np.float64,
    )

    all_ids = []

    for leaf in leaves:

        indices = np.asarray(
            leaf["point_indices"],
            dtype=np.int64,
        )

        if len(indices) == 0:
            continue

        z_mean = float(
            np.mean(
                data[
                    indices,
                    2,
                ]
            )
        )

        reconstruction[
            indices
        ] = z_mean

        all_ids.extend(
            indices.tolist()
        )

    all_ids = np.asarray(
        all_ids,
        dtype=np.int64,
    )

    expected = np.asarray(
        ground_point_ids,
        dtype=np.int64,
    )

    conservation = (
        len(all_ids) == len(expected)
        and len(np.unique(all_ids))
        == len(all_ids)
        and np.array_equal(
            np.sort(all_ids),
            np.sort(expected),
        )
    )

    errors = np.abs(
        data[
            expected,
            2,
        ].astype(np.float64)
        - reconstruction[expected]
    )

    return (
        conservation,
        float(np.mean(errors)),
        float(np.sqrt(np.mean(errors ** 2))),
        float(np.percentile(errors, 95)),
        float(np.max(errors)),
    )


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

    leaves = build_mode(
        data=data,
        cells=cells,
        base_resolution_by_cell=(
            base_resolution_by_cell
        ),
        analyzed_cells=analyzed_cells,
        mode=mode,
    )

    compact = build_compact_representation(
        data=data,
        leaves=leaves,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    _, peak = (
        tracemalloc.get_traced_memory()
    )

    tracemalloc.stop()

    (
        conservation,
        mae,
        rmse,
        p95,
        max_error,
    ) = reconstruct_and_validate(
        data=data,
        leaves=leaves,
        ground_point_ids=(
            ground_point_ids
        ),
    )

    print(
        f"\nMode {mode}"
    )

    print(
        "  cells                  :",
        len(leaves),
    )

    print(
        "  compact map bytes      :",
        compact.nbytes,
    )

    print(
        "  compact map KiB        :",
        round(
            compact.nbytes / 1024.0,
            3,
        ),
    )

    print(
        "  record size bytes      :",
        COMPACT_DTYPE.itemsize,
    )

    print(
        "  peak Python memory KiB :",
        round(
            peak / 1024.0,
            3,
        ),
    )

    print(
        "  build time seconds     :",
        round(
            elapsed,
            4,
        ),
    )

    print(
        "  point conservation     :",
        conservation,
    )

    print(
        "  elevation MAE m        :",
        round(mae, 6),
    )

    print(
        "  elevation RMSE m       :",
        round(rmse, 6),
    )

    print(
        "  elevation P95 m        :",
        round(p95, 6),
    )

    print(
        "  elevation max m        :",
        round(max_error, 6),
    )

    resolution_counts = {}

    for leaf in leaves:

        resolution = round(
            float(
                leaf["cell_size"]
            ),
            6,
        )

        resolution_counts[
            resolution
        ] = (
            resolution_counts.get(
                resolution,
                0,
            )
            + 1
        )

    print(
        "  resolutions:"
    )

    for resolution, count in sorted(
        resolution_counts.items()
    ):

        print(
            f"    {resolution:.6f} m"
            f" : {count}"
        )


def main():

    print(
        "COMPACT MAP MEMORY BENCHMARK"
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
    ) = build_parent_inputs(
        data
    )

    print(
        "LiDAR points:",
        len(data),
    )

    print(
        "Ground points:",
        len(ground_point_ids),
    )

    print(
        "Parent terrain cells:",
        len(cells),
    )

    for mode in (
        "A",
        "B",
        "C",
    ):

        run_mode(
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

    print(
        "\nBenchmark complete."
    )


if __name__ == "__main__":
    main()
