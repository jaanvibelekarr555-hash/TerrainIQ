import numpy as np
import open3d as o3d

from src.config import ROAD_CELL_SIZE, ROAD_MIN_POINTS
from src.road_geometry import analyze_road_cell


file_path = "data/semantic_kitti/sequences/00/velodyne/000000.bin"


# --------------------------------------------------
# 1. Load LiDAR data
# --------------------------------------------------

data = np.fromfile(
    file_path,
    dtype=np.float32,
).reshape(-1, 4)

# Use the region in front of the vehicle
points = data[
    (data[:, 0] > 0)
    & (data[:, 0] < 30)
    & (np.abs(data[:, 1]) < 10)
]

xyz = points[:, :3]

print("Input points:", len(xyz))


# --------------------------------------------------
# 2. Extract dominant ground plane
# --------------------------------------------------

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

plane_model, inliers = pcd.segment_plane(
    distance_threshold=0.15,
    ransac_n=3,
    num_iterations=1000,
)

ground_points = xyz[inliers]

print("Ground points:", len(ground_points))


# --------------------------------------------------
# 3. Create local cells
# --------------------------------------------------

cell_size = ROAD_CELL_SIZE

x_min = ground_points[:, 0].min()
y_min = ground_points[:, 1].min()

cell_x = np.floor(
    (ground_points[:, 0] - x_min) / cell_size
).astype(int)

cell_y = np.floor(
    (ground_points[:, 1] - y_min) / cell_size
).astype(int)

cells = {}

for i in range(len(ground_points)):

    key = (cell_x[i], cell_y[i])

    if key not in cells:
        cells[key] = []

    cells[key].append(ground_points[i])

print("Total cells:", len(cells))


# --------------------------------------------------
# 4. Analyze every road cell
# --------------------------------------------------

drivable_cells = []
non_drivable_cells = []
sparse_cells = []

cell_results = {}

for key, cell_points in cells.items():

    cell_points = np.asarray(cell_points)

    result = analyze_road_cell(
        cell_points,
        cell_x=key[0],
        cell_y=key[1],
        height_threshold=0.20,
        plane_model=plane_model,
    )

    cell_results[key] = result

    if result["classification"] == "drivable":
        drivable_cells.append(key)

    elif result["classification"] == "non-drivable":
        non_drivable_cells.append(key)

    else:
        sparse_cells.append(key)


# --------------------------------------------------
# 5. Print results
# --------------------------------------------------

print("\nRoad classification:")
print("Drivable cells:", len(drivable_cells))
print("Non-drivable cells:", len(non_drivable_cells))
print("Sparse cells:", len(sparse_cells))

print("\nExample cell results:")

shown = 0

for key, result in cell_results.items():

    print(
        f"Cell {key}: "
        f"points={result['point_count']}, "
        f"variation={result['height_variation']}, "
        f"classification={result['classification']}"
    )

    shown += 1

    if shown >= 10:
        break