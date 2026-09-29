import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from src.config import ROAD_CELL_SIZE, ROAD_MIN_POINTS
from src.road_geometry import analyze_road_cell


FILE_PATH = "data/semantic_kitti/sequences/00/velodyne/000000.bin"
OUTPUT_PATH = "results/road_map_000000.png"

# Prototype threshold: 20 cm
HEIGHT_THRESHOLD = 0.20


# --------------------------------------------------
# 1. Load LiDAR
# --------------------------------------------------

data = np.fromfile(
    FILE_PATH,
    dtype=np.float32,
).reshape(-1, 4)


# --------------------------------------------------
# 2. Select region in front of vehicle
# --------------------------------------------------

points = data[
    (data[:, 0] > 0)
    & (data[:, 0] < 30)
    & (np.abs(data[:, 1]) < 10)
]

xyz = points[:, :3]

print("Input points:", len(xyz))


# --------------------------------------------------
# 3. Extract dominant ground points
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
# 4. Create 1 m × 1 m cells
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

    key = (
        int(cell_x[i]),
        int(cell_y[i]),
    )

    if key not in cells:
        cells[key] = []

    cells[key].append(ground_points[i])


# --------------------------------------------------
# 5. Analyze every cell
# --------------------------------------------------

results = []

for key, cell_points in cells.items():

    cell_points = np.asarray(cell_points)

    result = analyze_road_cell(
        cell_points,
        cell_x=key[0],
        cell_y=key[1],
        height_threshold=HEIGHT_THRESHOLD,
    )

    results.append(result)


# --------------------------------------------------
# 6. Separate classifications
# --------------------------------------------------

drivable_x = []
drivable_y = []

non_drivable_x = []
non_drivable_y = []

sparse_x = []
sparse_y = []

for result in results:

    x = result["cell_x"]
    y = result["cell_y"]

    if result["classification"] == "drivable":

        drivable_x.append(x)
        drivable_y.append(y)

    elif result["classification"] == "non-drivable":

        non_drivable_x.append(x)
        non_drivable_y.append(y)

    else:

        sparse_x.append(x)
        sparse_y.append(y)


# --------------------------------------------------
# 7. Print summary
# --------------------------------------------------

print("\nRoad classification:")
print("Drivable cells:", len(drivable_x))
print("Non-drivable cells:", len(non_drivable_x))
print("Sparse cells:", len(sparse_x))


# --------------------------------------------------
# 8. Create top-down map
# --------------------------------------------------

plt.figure(figsize=(10, 7))

plt.scatter(
    drivable_x,
    drivable_y,
    s=35,
    marker="s",
    label="Drivable",
)

plt.scatter(
    non_drivable_x,
    non_drivable_y,
    s=35,
    marker="s",
    label="Non-drivable",
)

plt.scatter(
    sparse_x,
    sparse_y,
    s=35,
    marker="s",
    label="Sparse",
)

plt.xlabel("Cell X")
plt.ylabel("Cell Y")
plt.title("Road Geometry: Drivable vs Non-Drivable")
plt.legend()
plt.axis("equal")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
)

print("\nSaved visualization:")
print(OUTPUT_PATH)

plt.show()