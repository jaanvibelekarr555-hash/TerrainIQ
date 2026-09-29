import numpy as np
import open3d as o3d


file_path = "data/semantic_kitti/sequences/00/velodyne/000000.bin"

data = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)

# Use the area in front of the vehicle.
points = data[
    (data[:, 0] > 0) &
    (data[:, 0] < 30) &
    (np.abs(data[:, 1]) < 10)
]

xyz = points[:, :3]

print("Input points:", len(xyz))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

# Find the dominant plane.
plane_model, inliers = pcd.segment_plane(
    distance_threshold=0.15,
    ransac_n=3,
    num_iterations=1000,
)

a, b, c, d = plane_model

ground = pcd.select_by_index(inliers)
non_ground = pcd.select_by_index(inliers, invert=True)

print("Plane equation:")
print(f"{a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

print("Ground points:", len(ground.points))
print("Non-ground points:", len(non_ground.points))

# Give ground and non-ground different colors.
ground.paint_uniform_color([1.0, 0.0, 0.0])
non_ground.paint_uniform_color([0.7, 0.7, 0.7])

o3d.visualization.draw_geometries(
    [ground, non_ground],
    window_name="Ground vs Non-Ground",
)