import numpy as np
import open3d as o3d

from src.config import ROAD_CELL_SIZE
from src.road_geometry import analyze_road_cells


FILE_PATH = "data/semantic_kitti/sequences/00/velodyne/000000.bin"


# --------------------------------------------------
# 1. Load the REAL LiDAR frame
# --------------------------------------------------

data = np.fromfile(
    FILE_PATH,
    dtype=np.float32,
).reshape(-1, 4)

print("Total LiDAR points:", len(data))


# --------------------------------------------------
# 2. Select the same real road-analysis region
# --------------------------------------------------

road_points = data[
    (data[:, 0] > 0)
    & (data[:, 0] < 30)
    & (np.abs(data[:, 1]) < 10)
]

print("Selected road-region points:", len(road_points))

if len(road_points) == 0:
    raise ValueError(
        "No LiDAR points found in selected region."
    )


# --------------------------------------------------
# 3. Extract XYZ for ground detection
# --------------------------------------------------

xyz = road_points[:, :3]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)


# --------------------------------------------------
# 4. Detect ground using RANSAC
# --------------------------------------------------

plane_model, inliers = pcd.segment_plane(
    distance_threshold=0.15,
    ransac_n=3,
    num_iterations=1000,
)


# --------------------------------------------------
# 5. Keep the COMPLETE original LiDAR information
#
# Each point contains:
# X
# Y
# Z
# Intensity
#
# Therefore:
# 4 float32 values × 4 bytes = 16 bytes/point
# --------------------------------------------------

ground_points_4d = road_points[inliers]

print("Ground LiDAR points:", len(ground_points_4d))


if len(ground_points_4d) == 0:
    raise ValueError(
        "No ground points detected."
    )


# XYZ is used for geometric processing
ground_xyz = ground_points_4d[:, :3]


# --------------------------------------------------
# 6. Create REAL 1 m × 1 m spatial cells
# --------------------------------------------------

cell_size = ROAD_CELL_SIZE

x_min_global = ground_xyz[:, 0].min()
y_min_global = ground_xyz[:, 1].min()

cell_x = np.floor(
    (ground_xyz[:, 0] - x_min_global)
    / cell_size
).astype(np.int32)

cell_y = np.floor(
    (ground_xyz[:, 1] - y_min_global)
    / cell_size
).astype(np.int32)


# --------------------------------------------------
# 7. Group REAL LiDAR points into cells
# --------------------------------------------------

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
# 8. Analyze the REAL cells
# --------------------------------------------------

results = analyze_road_cells(
    cells,
    height_threshold=0.20,
)


# --------------------------------------------------
# 9. Define the COMPLETE current 2.5D cell
#
# Information retained:
#
# cell_x
# cell_y
# point_count
# x_min
# x_max
# y_min
# y_max
# z_mean
# z_min
# z_max
# height_variation
# classification
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
    len(results),
    dtype=cell_dtype,
)


classification_map = {
    "sparse": 0,
    "drivable": 1,
    "non-drivable": 2,
}


# --------------------------------------------------
# 10. Fill every cell using REAL LiDAR values
# --------------------------------------------------

for i, result in enumerate(results):

    cx = result["cell_x"]
    cy = result["cell_y"]

    compact_cells[i]["cell_x"] = cx
    compact_cells[i]["cell_y"] = cy

    compact_cells[i]["point_count"] = (
        result["point_count"]
    )

    # Find the real points belonging to this cell
    mask = (
        (cell_x == cx)
        & (cell_y == cy)
    )

    cell_points = ground_xyz[mask]

    if len(cell_points) > 0:

        # Real spatial extent
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

        # Real height information
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

    # Real classification generated by our road module
    compact_cells[i]["classification"] = (
        classification_map[
            result["classification"]
        ]
    )


# --------------------------------------------------
# 11. Calculate RAW LiDAR memory
#
# Raw representation keeps:
# X + Y + Z + Intensity
#
# Every value = float32 = 4 bytes
# Every point = 16 bytes
# --------------------------------------------------

raw_ground_lidar = ground_points_4d.astype(
    np.float32
)

raw_memory = raw_ground_lidar.nbytes

raw_bytes_per_point = (
    raw_memory // len(raw_ground_lidar)
)


# --------------------------------------------------
# 12. Calculate COMPLETE 2.5D memory
# --------------------------------------------------

cell_memory = compact_cells.nbytes

bytes_per_cell = compact_cells.itemsize


# --------------------------------------------------
# 13. Calculate memory reduction
# --------------------------------------------------

memory_reduction = (
    (raw_memory - cell_memory)
    / raw_memory
) * 100


# --------------------------------------------------
# 14. Calculate representation ratio
# --------------------------------------------------

representation_ratio = (
    raw_memory / cell_memory
)


# --------------------------------------------------
# 15. Print detailed comparison
# --------------------------------------------------

print()
print("=" * 70)
print("REAL LiDAR vs COMPLETE 2.5D REPRESENTATION")
print("=" * 70)

print(
    f"Raw ground LiDAR points:       "
    f"{len(raw_ground_lidar):>8}"
)

print(
    f"2.5D cells:                    "
    f"{len(compact_cells):>8}"
)

print()

print(
    f"Raw point size:                "
    f"{raw_bytes_per_point:>8} bytes/point"
)

print(
    f"2.5D cell size:               "
    f"{bytes_per_cell:>8} bytes/cell"
)

print()

print(
    f"Raw LiDAR memory:              "
    f"{raw_memory:>8} bytes "
    f"({raw_memory / 1024:.2f} KiB)"
)

print(
    f"Complete 2.5D memory:          "
    f"{cell_memory:>8} bytes "
    f"({cell_memory / 1024:.2f} KiB)"
)

print()

print(
    f"Memory reduction:              "
    f"{memory_reduction:>8.2f}%"
)

print(
    f"Representation ratio:         "
    f"{representation_ratio:>8.2f} : 1"
)

print("=" * 70)