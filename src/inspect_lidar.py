import numpy as np
import open3d as o3d

file_path = "data/semantic_kitti/sequences/00/velodyne/000000.bin"

data = np.fromfile(file_path, dtype=np.float32)
points = data.reshape(-1, 4)

xyz = points[:, :3]

print("Total points:", len(xyz))
print("X range:", xyz[:, 0].min(), "to", xyz[:, 0].max())
print("Y range:", xyz[:, 1].min(), "to", xyz[:, 1].max())
print("Z range:", xyz[:, 2].min(), "to", xyz[:, 2].max())

point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(xyz)

o3d.visualization.draw_geometries(
    [point_cloud],
    window_name="Raw LiDAR - 000000"
)