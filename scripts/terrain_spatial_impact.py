from __future__ import annotations

import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from src.road_processor import extract_ground_mask
from src.terrain_analysis import analyze_terrain_cells
from src.terrain_cells import build_terrain_cells
from src.terrain_quadtree import build_terrain_quadtree_for_cell
from src.terrain_resolution import (
    build_base_resolution_by_cell,
    calculate_cell_distance,
)

DATA_FILE = (
    "data/RELLIS3D/00000/"
    "vel_cloud_node_kitti_bin/000000.bin"
)

RESULTS_DIR = "results"
MIN_RESOLUTION = 0.05
NO_ERROR_BOUND = 1_000_000.0


def load_data() -> np.ndarray:
    data = np.fromfile(DATA_FILE, dtype=np.float32).reshape(-1, 4)
    if len(data) == 0:
        raise ValueError("LiDAR frame is empty.")
    return data


def build_parent_data(data: np.ndarray):
    ground_mask, point_id, _ = extract_ground_mask(data)
    ground_point_ids = point_id[ground_mask]

    cells = build_terrain_cells(
        data=data,
        ground_mask=ground_mask,
        point_id=point_id,
        cell_size=1.0,
    )

    base_resolution_by_cell = build_base_resolution_by_cell(cells)

    analyzed_cells = analyze_terrain_cells(
        data=data,
        cells=cells,
        base_resolution_by_cell=base_resolution_by_cell,
    )

    return ground_point_ids, cells, base_resolution_by_cell, analyzed_cells


def build_mode_leaves(
    data: np.ndarray,
    cells: dict,
    base_resolution_by_cell: dict,
    analyzed_cells: dict,
    mode: str,
):
    leaves_by_parent = {}

    for cell_id, cell in cells.items():
        base_resolution = float(base_resolution_by_cell[cell_id])

        if mode == "B":
            requested_resolution = base_resolution
            priority = 0.0
            reason = "distance-only"

        elif mode == "C":
            request = analyzed_cells[cell_id]["refinement_request"]
            requested_resolution = float(request["requested_resolution"])
            priority = float(analyzed_cells[cell_id]["terrain_priority"])
            reason = "distance + terrain"

        else:
            raise ValueError("mode must be B or C.")

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

        occupied_leaves = [
            leaf
            for leaf in result["leaves"]
            if len(leaf["point_indices"]) > 0
        ]
        leaves_by_parent[cell_id] = occupied_leaves

    return leaves_by_parent


def calculate_actual_resolution(leaves: list[dict]) -> float:
    if not leaves:
        return np.nan
    return float(min(leaf["cell_size"] for leaf in leaves))


def calculate_parent_results(
    cells: dict,
    base_resolution_by_cell: dict,
    analyzed_cells: dict,
    leaves_b: dict,
    leaves_c: dict,
) -> list[dict]:
    results = []

    for cell_id, cell in cells.items():
        bbox = cell["bbox"]
        x_center = (bbox["x_min"] + bbox["x_max"]) / 2.0
        y_center = (bbox["y_min"] + bbox["y_max"]) / 2.0
        distance = float(calculate_cell_distance(cell))

        terrain = analyzed_cells[cell_id]
        base_resolution = float(base_resolution_by_cell[cell_id])
        requested_resolution = float(
            terrain["refinement_request"]["requested_resolution"]
        )
        priority = float(terrain["terrain_priority"])
        complexity = terrain["terrain_complexity"]
        traversability = terrain["traversability"]

        b_leaves = leaves_b[cell_id]
        c_leaves = leaves_c[cell_id]

        b_resolution = calculate_actual_resolution(b_leaves)
        c_resolution = calculate_actual_resolution(c_leaves)
        b_leaf_count = len(b_leaves)
        c_leaf_count = len(c_leaves)
        added_leaves = c_leaf_count - b_leaf_count
        terrain_refined = requested_resolution < base_resolution
        high_priority = priority >= 0.70
        medium_priority = 0.40 <= priority < 0.70

        if high_priority:
            priority_band = "HIGH"
        elif medium_priority:
            priority_band = "MEDIUM"
        else:
            priority_band = "LOW"

        results.append(
            {
                "cell_id": cell_id,
                "x_center": x_center,
                "y_center": y_center,
                "distance_m": distance,
                "base_resolution_m": base_resolution,
                "requested_resolution_m": requested_resolution,
                "terrain_priority": priority,
                "priority_band": priority_band,
                "terrain_complexity": complexity,
                "traversability": traversability,
                "distance_only_resolution_m": b_resolution,
                "terrain_adaptive_resolution_m": c_resolution,
                "distance_only_leaf_count": b_leaf_count,
                "terrain_adaptive_leaf_count": c_leaf_count,
                "additional_leaves": added_leaves,
                "terrain_refined": terrain_refined,
                "high_priority": high_priority,
            }
        )

    return results


def print_summary(results: list[dict]) -> None:
    total = len(results)
    refined = sum(result["terrain_refined"] for result in results)
    high_priority = sum(result["high_priority"] for result in results)
    high_priority_refined = sum(
        result["high_priority"] and result["terrain_refined"]
        for result in results
    )
    additional_leaves = sum(result["additional_leaves"] for result in results)
    distance_only_leaves = sum(
        result["distance_only_leaf_count"] for result in results
    )
    terrain_leaves = sum(
        result["terrain_adaptive_leaf_count"] for result in results
    )

    print()
    print("=" * 78)
    print("SPATIAL TERRAIN IMPACT SUMMARY")
    print("=" * 78)
    print("Parent terrain cells      :", total)
    print(
        "Terrain-refined parents   :",
        refined,
        f"({100.0 * refined / total:.2f}%)" if total else "(0.00%)",
    )
    print("High-priority parents     :", high_priority)
    print(
        "High-priority refined     :",
        high_priority_refined,
        f"({100.0 * high_priority_refined / high_priority:.2f}%)"
        if high_priority
        else "(0.00%)",
    )
    print("Distance-only leaves      :", distance_only_leaves)
    print("Terrain-adaptive leaves   :", terrain_leaves)
    print("Additional leaves         :", additional_leaves)

    if distance_only_leaves:
        print(
            "Additional leaves %       :",
            f"{100.0 * additional_leaves / distance_only_leaves:.3f}%",
        )

    print()
    print("Priority-band impact:")

    for band in ("LOW", "MEDIUM", "HIGH"):
        band_results = [
            result for result in results if result["priority_band"] == band
        ]
        if not band_results:
            continue

        refined_count = sum(result["terrain_refined"] for result in band_results)
        average_b = np.mean(
            [result["distance_only_resolution_m"] for result in band_results]
        )
        average_c = np.mean(
            [result["terrain_adaptive_resolution_m"] for result in band_results]
        )
        extra = sum(result["additional_leaves"] for result in band_results)

        print()
        print(f"  {band} priority")
        print("    parent cells       :", len(band_results))
        print("    refined parents    :", refined_count)
        print("    average B res (m) :", f"{average_b:.6f}")
        print("    average C res (m) :", f"{average_c:.6f}")
        print("    additional leaves :", extra)


def save_csv(results: list[dict]) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(
        RESULTS_DIR,
        "terrain_spatial_impact_00000_000000.csv",
    )

    fieldnames = list(results[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    return path


def make_priority_spatial_plot(results: list[dict]) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    x = np.array([result["x_center"] for result in results], dtype=float)
    y = np.array([result["y_center"] for result in results], dtype=float)
    priority = np.array(
        [result["terrain_priority"] for result in results],
        dtype=float,
    )
    sizes = 20.0 + 120.0 * priority

    plt.figure(figsize=(9, 7))
    plt.scatter(x, y, s=sizes, alpha=0.75)
    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.title("Terrain Priority Spatial Distribution")
    plt.grid(True, alpha=0.25)
    plt.axis("equal")

    path = os.path.join(
        RESULTS_DIR,
        "terrain_priority_spatial_00000_000000.png",
    )
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def make_resolution_comparison_plot(results: list[dict]) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    bands = ["LOW", "MEDIUM", "HIGH"]
    distance_values = []
    terrain_values = []

    for band in bands:
        band_results = [
            result for result in results if result["priority_band"] == band
        ]
        if band_results:
            distance_values.append(
                np.mean(
                    [result["distance_only_resolution_m"] for result in band_results]
                )
            )
            terrain_values.append(
                np.mean(
                    [result["terrain_adaptive_resolution_m"] for result in band_results]
                )
            )
        else:
            distance_values.append(np.nan)
            terrain_values.append(np.nan)

    x = np.arange(len(bands))
    width = 0.35

    plt.figure(figsize=(9, 6))
    plt.bar(x - width / 2.0, distance_values, width, label="Distance-only")
    plt.bar(x + width / 2.0, terrain_values, width, label="Distance + Terrain")
    plt.xticks(x, bands)
    plt.ylabel("Average actual leaf resolution (m)")
    plt.xlabel("Terrain priority band")
    plt.title("Actual Resolution by Terrain Priority")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)

    path = os.path.join(
        RESULTS_DIR,
        "terrain_resolution_by_priority_00000_000000.png",
    )
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def make_added_leaf_plot(results: list[dict]) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    bands = ["LOW", "MEDIUM", "HIGH"]
    values = [
        sum(
            result["additional_leaves"]
            for result in results
            if result["priority_band"] == band
        )
        for band in bands
    ]

    plt.figure(figsize=(9, 6))
    plt.bar(bands, values)
    plt.xlabel("Terrain priority band")
    plt.ylabel("Additional leaves from terrain")
    plt.title("Additional Spatial Detail Added by Terrain")
    plt.grid(axis="y", alpha=0.25)

    path = os.path.join(
        RESULTS_DIR,
        "terrain_added_leaves_by_priority_00000_000000.png",
    )
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def make_refinement_spatial_plot(results: list[dict]) -> str:
    """Show which 1 m parent terrain cells request local refinement."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    refined = [result for result in results if result["terrain_refined"]]
    not_refined = [result for result in results if not result["terrain_refined"]]

    plt.figure(figsize=(9, 7))

    if not_refined:
        x_not = [result["x_center"] for result in not_refined]
        y_not = [result["y_center"] for result in not_refined]
        plt.scatter(
            x_not,
            y_not,
            s=18,
            alpha=0.45,
            marker="o",
            label="No terrain refinement",
        )

    if refined:
        x_ref = [result["x_center"] for result in refined]
        y_ref = [result["y_center"] for result in refined]
        plt.scatter(
            x_ref,
            y_ref,
            s=45,
            alpha=0.90,
            marker="s",
            label="Terrain refinement requested",
        )

    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.title("Terrain-Requested Refinement by Parent Region")
    plt.grid(True, alpha=0.25)
    plt.axis("equal")
    plt.legend()

    path = os.path.join(
        RESULTS_DIR,
        "terrain_refinement_requested_spatial_00000_000000.png",
    )
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def main():
    print("TERRAIN SPATIAL IMPACT SIMULATION")
    print("Distance-only vs Distance + Terrain")
    print()
    print("Frame:", DATA_FILE)

    data = load_data()

    (
        ground_point_ids,
        cells,
        base_resolution_by_cell,
        analyzed_cells,
    ) = build_parent_data(data)

    print()
    print("LiDAR points      :", len(data))
    print("Ground points     :", len(ground_point_ids))
    print("Parent cells      :", len(cells))

    print()
    print("Building distance-only map...")
    leaves_b = build_mode_leaves(
        data=data,
        cells=cells,
        base_resolution_by_cell=base_resolution_by_cell,
        analyzed_cells=analyzed_cells,
        mode="B",
    )

    print("Building terrain-adaptive map...")
    leaves_c = build_mode_leaves(
        data=data,
        cells=cells,
        base_resolution_by_cell=base_resolution_by_cell,
        analyzed_cells=analyzed_cells,
        mode="C",
    )

    results = calculate_parent_results(
        cells=cells,
        base_resolution_by_cell=base_resolution_by_cell,
        analyzed_cells=analyzed_cells,
        leaves_b=leaves_b,
        leaves_c=leaves_c,
    )

    print_summary(results)

    csv_path = save_csv(results)
    priority_plot = make_priority_spatial_plot(results)
    resolution_plot = make_resolution_comparison_plot(results)
    leaf_plot = make_added_leaf_plot(results)
    refinement_plot = make_refinement_spatial_plot(results)

    print()
    print("=" * 78)
    print("OUTPUT FILES")
    print("CSV:", csv_path)
    print("Priority map:", priority_plot)
    print("Resolution comparison:", resolution_plot)
    print("Added-detail plot:", leaf_plot)
    print("Refinement map:", refinement_plot)
    print()
    print("Simulation complete.")


if __name__ == "__main__":
    main()
