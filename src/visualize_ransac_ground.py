import numpy as np
import open3d as o3d


file_path = "data/semantic_kitti/sequences/00/velodyne/000000.bin"

# Load LiDAR frame
data = np.fromfile(
    file_path,
    dtype=np.float32,
).reshape(-1, 4)

# Select area in front of the vehicle
points = data[
    (data[:, 0] > 0)
    & (data[:, 0] < 30)
    & (np.abs(data[:, 1]) < 10)
]

xyz = points[:, :3]

print("Input points:", len(xyz))

# Create point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

# RANSAC ground-plane extraction
plane_model, inliers = pcd.segment_plane(
    distance_threshold=0.15,
    ransac_n=3,
    num_iterations=1000,
)

ground = pcd.select_by_index(inliers)

print("RANSAC ground points:", len(ground.points))

# Show ONLY the points selected by RANSAC
ground.paint_uniform_color([0.0, 1.0, 0.0])

o3d.visualization.draw_geometries(
    [ground],
    window_name="RANSAC Selected Ground Points",
)