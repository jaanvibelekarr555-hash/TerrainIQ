import numpy as np
import open3d as o3d

file_path = "data/semantic_kitti/sequences/00/velodyne/000000.bin"

data = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)

# Keep only points in front of the vehicle
# x = forward direction in KITTI coordinates
front_points = data[
    (data[:, 0] > 0) &
    (data[:, 0] < 30) &
    (np.abs(data[:, 1]) < 10)
]

xyz = front_points[:, :3]

print("Selected points:", len(xyz))
print("Z range:", xyz[:, 2].min(), "to", xyz[:, 2].max())

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

o3d.visualization.draw_geometries(
    [pcd],
    window_name="Front Ground Region"
)