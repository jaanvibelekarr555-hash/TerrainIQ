import csv
import time
from pathlib import Path

from src.road_processor import process_road_frame


DATA_DIR = Path("data/semantic_kitti/sequences/00/velodyne")
OUTPUT_DIR = Path("results")

HEIGHT_THRESHOLD = 0.20


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = []

    bin_files = sorted(DATA_DIR.glob("*.bin"))

    if not bin_files:
        raise FileNotFoundError(
            f"No .bin files found in {DATA_DIR}"
        )

    for file_path in bin_files:

        start_time = time.perf_counter()

        results = process_road_frame(
            str(file_path),
            height_threshold=HEIGHT_THRESHOLD,
        )

        processing_time = time.perf_counter() - start_time

        drivable = sum(
            result["classification"] == "drivable"
            for result in results
        )

        non_drivable = sum(
            result["classification"] == "non-drivable"
            for result in results
        )

        sparse = sum(
            result["classification"] == "sparse"
            for result in results
        )

        summary.append(
            {
                "frame": file_path.stem,
                "total_cells": len(results),
                "drivable_cells": drivable,
                "non_drivable_cells": non_drivable,
                "sparse_cells": sparse,
                "processing_time_seconds": round(
                    processing_time,
                    4,
                ),
            }
        )

        output_file = (
            OUTPUT_DIR
            / f"road_cells_{file_path.stem}.csv"
        )

        with open(
            output_file,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "cell_x",
                    "cell_y",
                    "point_count",
                    "height_variation",
                    "classification",
                ],
            )

            writer.writeheader()
            writer.writerows(results)

    summary_file = OUTPUT_DIR / "road_benchmark.csv"

    with open(
        summary_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "frame",
                "total_cells",
                "drivable_cells",
                "non_drivable_cells",
                "sparse_cells",
                "processing_time_seconds",
            ],
        )

        writer.writeheader()
        writer.writerows(summary)

    print("\nRoad processing benchmark")
    print("=" * 70)

    for row in summary:
        print(
            f"{row['frame']}: "
            f"total={row['total_cells']}, "
            f"drivable={row['drivable_cells']}, "
            f"non-drivable={row['non_drivable_cells']}, "
            f"sparse={row['sparse_cells']}, "
            f"time={row['processing_time_seconds']} s"
        )

    print("\nGenerated:")
    print(f"  {summary_file}")


if __name__ == "__main__":
    main()