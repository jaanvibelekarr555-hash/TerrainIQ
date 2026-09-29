import csv
from pathlib import Path

import numpy as np
import open3d as o3d

from src.config import ROAD_CELL_SIZE
from src.road_geometry import analyze_road_cells


DATA_DIR = Path(
    "data/semantic_kitti/sequences/00/velodyne"
)

RESULTS_FILE = Path(
    "results/memory_benchmark.csv"
)


def measure_frame(file_path: Path) -> dict:
    """
    Measure raw 4D LiDAR representation versus
    the current complete 2.5D cell representation
    for one real LiDAR frame.
    """

    # --------------------------------------------------
    # 1. Load REAL LiDAR frame
    # --------------------------------------------------

    data = np.fromfile(
        file_path,
        dtype=np.float32,
    ).reshape(-1, 4)

    total_points = len(data)

    # --------------------------------------------------
    # 2. Select road-analysis region
    # --------------------------------------------------

    road_points = data[
        (data[:, 0] > 0)
        & (data[:, 0] < 30)
        & (np.abs(data[:, 1]) < 10)
    ]

    selected_points = len(road_points)

    if selected_points == 0:
        raise ValueError(
            f"No road points found in {file_path.name}"
        )

    # --------------------------------------------------
    # 3. Ground extraction using RANSAC
    # --------------------------------------------------

    xyz = road_points[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.15,
        ransac_n=3,
        num_iterations=1000,
    )

    ground_points_4d = road_points[inliers]

    ground_points = len(ground_points_4d)

    if ground_points == 0:
        raise ValueError(
            f"No ground points detected in {file_path.name}"
        )

    ground_xyz = ground_points_4d[:, :3]

    # --------------------------------------------------
    # 4. Create real 1 m × 1 m cells
    # --------------------------------------------------

    x_min_global = ground_xyz[:, 0].min()
    y_min_global = ground_xyz[:, 1].min()

    cell_x = np.floor(
        (ground_xyz[:, 0] - x_min_global)
        / ROAD_CELL_SIZE
    ).astype(np.int32)

    cell_y = np.floor(
        (ground_xyz[:, 1] - y_min_global)
        / ROAD_CELL_SIZE
    ).astype(np.int32)

    cells = {}

    for i in range(len(ground_xyz)):

        key = (
            int(cell_x[i]),
            int(cell_y[i]),
        )

        if key not in cells:
            cells[key] = []

        cells[key].append(ground_xyz[i])

    # --------------------------------------------------
    # 5. Analyze real cells
    # --------------------------------------------------

    road_results = analyze_road_cells(
        cells,
        height_threshold=0.20,
    )

    # --------------------------------------------------
    # 6. Complete 2.5D representation
    # --------------------------------------------------

    cell_dtype = np.dtype(
        [
            ("cell_x", np.int32),
            ("cell_y", np.int32),

            ("point_count", np.int32),

            ("x_min", np.float32),
            ("x_max", np.float32),

            ("y_min", np.float32),
            ("y_max", np.float32),

            ("z_mean", np.float32),
            ("z_min", np.float32),
            ("z_max", np.float32),

            ("height_variation", np.float32),

            # 0 = sparse
            # 1 = drivable
            # 2 = non-drivable
            ("classification", np.uint8),
        ]
    )

    compact_cells = np.zeros(
        len(road_results),
        dtype=cell_dtype,
    )

    classification_map = {
        "sparse": 0,
        "drivable": 1,
        "non-drivable": 2,
    }

    # --------------------------------------------------
    # 7. Fill 2.5D cells using REAL LiDAR values
    # --------------------------------------------------

    for i, result in enumerate(road_results):

        cx = result["cell_x"]
        cy = result["cell_y"]

        compact_cells[i]["cell_x"] = cx
        compact_cells[i]["cell_y"] = cy

        compact_cells[i]["point_count"] = (
            result["point_count"]
        )

        mask = (
            (cell_x == cx)
            & (cell_y == cy)
        )

        cell_points = ground_xyz[mask]

        if len(cell_points) > 0:

            compact_cells[i]["x_min"] = np.min(
                cell_points[:, 0]
            )

            compact_cells[i]["x_max"] = np.max(
                cell_points[:, 0]
            )

            compact_cells[i]["y_min"] = np.min(
                cell_points[:, 1]
            )

            compact_cells[i]["y_max"] = np.max(
                cell_points[:, 1]
            )

            compact_cells[i]["z_mean"] = np.mean(
                cell_points[:, 2]
            )

            compact_cells[i]["z_min"] = np.min(
                cell_points[:, 2]
            )

            compact_cells[i]["z_max"] = np.max(
                cell_points[:, 2]
            )

            compact_cells[i]["height_variation"] = (
                compact_cells[i]["z_max"]
                - compact_cells[i]["z_min"]
            )

        compact_cells[i]["classification"] = (
            classification_map[
                result["classification"]
            ]
        )

    # --------------------------------------------------
    # 8. Raw LiDAR memory
    #
    # X + Y + Z + Intensity
    # 4 × float32 = 16 bytes/point
    # --------------------------------------------------

    raw_ground_lidar = ground_points_4d.astype(
        np.float32
    )

    raw_memory = raw_ground_lidar.nbytes

    raw_bytes_per_point = (
        raw_memory // len(raw_ground_lidar)
    )

    # --------------------------------------------------
    # 9. 2.5D memory
    # --------------------------------------------------

    cell_memory = compact_cells.nbytes

    bytes_per_cell = compact_cells.itemsize

    # --------------------------------------------------
    # 10. Memory reduction
    # --------------------------------------------------

    memory_reduction = (
        (raw_memory - cell_memory)
        / raw_memory
    ) * 100

    # --------------------------------------------------
    # 11. Representation ratio
    # --------------------------------------------------

    representation_ratio = (
        raw_memory / cell_memory
    )

    # --------------------------------------------------
    # 12. Return measurements
    # --------------------------------------------------

    return {
        "frame": file_path.stem,
        "total_points": total_points,
        "selected_points": selected_points,
        "ground_points": ground_points,
        "cells": len(compact_cells),
        "raw_bytes": raw_memory,
        "cell_bytes": cell_memory,
        "raw_kib": raw_memory / 1024,
        "cell_kib": cell_memory / 1024,
        "raw_bytes_per_point": raw_bytes_per_point,
        "cell_bytes_per_cell": bytes_per_cell,
        "memory_reduction_percent": memory_reduction,
        "representation_ratio": representation_ratio,
    }


def main():

    # --------------------------------------------------
    # 1. Process all 5 real frames
    # --------------------------------------------------

    frame_files = sorted(
        DATA_DIR.glob("00000[0-4].bin")
    )

    if len(frame_files) != 5:
        raise ValueError(
            "Expected exactly 5 frames: "
            "000000.bin to 000004.bin"
        )

    results = []

    print()
    print("=" * 90)
    print("REAL LiDAR vs COMPLETE 2.5D MEMORY BENCHMARK")
    print("=" * 90)

    for file_path in frame_files:

        result = measure_frame(file_path)

        results.append(result)

        print()
        print(f"Frame: {result['frame']}")
        print("-" * 90)

        print(
            f"Total LiDAR points:        "
            f"{result['total_points']}"
        )

        print(
            f"Selected road points:     "
            f"{result['selected_points']}"
        )

        print(
            f"Ground LiDAR points:      "
            f"{result['ground_points']}"
        )

        print(
            f"2.5D cells:               "
            f"{result['cells']}"
        )

        print(
            f"Raw LiDAR memory:         "
            f"{result['raw_bytes']} bytes "
            f"({result['raw_kib']:.2f} KiB)"
        )

        print(
            f"Complete 2.5D memory:     "
            f"{result['cell_bytes']} bytes "
            f"({result['cell_kib']:.2f} KiB)"
        )

        print(
            f"Memory reduction:         "
            f"{result['memory_reduction_percent']:.2f}%"
        )

        print(
            f"Representation ratio:    "
            f"{result['representation_ratio']:.2f} : 1"
        )

    # --------------------------------------------------
    # 2. Calculate overall statistics
    # --------------------------------------------------

    reductions = np.array(
        [
            r["memory_reduction_percent"]
            for r in results
        ]
    )

    ratios = np.array(
        [
            r["representation_ratio"]
            for r in results
        ]
    )

    print()
    print("=" * 90)
    print("FIVE-FRAME SUMMARY")
    print("=" * 90)

    print(
        f"Minimum memory reduction:  "
        f"{np.min(reductions):.2f}%"
    )

    print(
        f"Maximum memory reduction:  "
        f"{np.max(reductions):.2f}%"
    )

    print(
        f"Average memory reduction:  "
        f"{np.mean(reductions):.2f}%"
    )

    print(
        f"Minimum representation:    "
        f"{np.min(ratios):.2f} : 1"
    )

    print(
        f"Maximum representation:    "
        f"{np.max(ratios):.2f} : 1"
    )

    print(
        f"Average representation:    "
        f"{np.mean(ratios):.2f} : 1"
    )

    print("=" * 90)

    # --------------------------------------------------
    # 3. Save results to CSV
    # --------------------------------------------------

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "frame",
        "total_points",
        "selected_points",
        "ground_points",
        "cells",
        "raw_bytes",
        "cell_bytes",
        "raw_kib",
        "cell_kib",
        "raw_bytes_per_point",
        "cell_bytes_per_cell",
        "memory_reduction_percent",
        "representation_ratio",
    ]

    with RESULTS_FILE.open(
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print(
        f"Detailed results saved to: "
        f"{RESULTS_FILE}"
    )


if __name__ == "__main__":
    main()