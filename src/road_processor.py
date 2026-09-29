import numpy as np
import open3d as o3d

from src.config import ROAD_CELL_SIZE
from src.road_geometry import analyze_road_cells


def extract_ground_mask(
    data: np.ndarray,
    x_min: float = 0.0,
    x_max: float = 30.0,
    y_limit: float = 10.0,
    distance_threshold: float = 0.15,
    ransac_n: int = 3,
    num_iterations: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract ground points using RANSAC while preserving
    correspondence with the original LiDAR point cloud.

    Args:
        data:
            LiDAR data with shape (N, 4):
            [X, Y, Z, Intensity]

    Returns:
        ground_mask:
            Boolean array of shape (N,).
            True means the original LiDAR point was
            classified as a ground point.

        point_id:
            Integer array of shape (N,).
            Contains the original point index.

        plane_model:
            RANSAC plane model [a, b, c, d].
    """

    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(
            "LiDAR data must have shape (N, 3) or (N, 4)."
        )

    if len(data) == 0:
        raise ValueError("LiDAR data is empty.")

    # --------------------------------------------------
    # 1. Preserve original point IDs
    # --------------------------------------------------

    point_id = np.arange(
        len(data),
        dtype=np.int64,
    )

    # --------------------------------------------------
    # 2. Select region in front of the vehicle
    # --------------------------------------------------

    region_mask = (
        (data[:, 0] > x_min)
        & (data[:, 0] < x_max)
        & (np.abs(data[:, 1]) < y_limit)
    )

    region_points = data[region_mask]
    region_point_ids = point_id[region_mask]

    if len(region_points) == 0:
        raise ValueError(
            "No LiDAR points found in the selected region."
        )

    xyz = region_points[:, :3]

    # --------------------------------------------------
    # 3. Estimate dominant ground plane using RANSAC
    # --------------------------------------------------

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # Make RANSAC reproducible.
    o3d.utility.random.seed(0)

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
        probability=1.0,
    )

    if len(inliers) == 0:
        raise ValueError("No ground points detected.")

    # --------------------------------------------------
    # 4. Convert local RANSAC indices back to original
    #    LiDAR point indices
    # --------------------------------------------------

    inliers = np.asarray(
        inliers,
        dtype=np.int64,
    )

    original_ground_ids = region_point_ids[inliers]

    # --------------------------------------------------
    # 5. Create full-size ground mask
    # --------------------------------------------------

    ground_mask = np.zeros(
        len(data),
        dtype=bool,
    )

    ground_mask[original_ground_ids] = True

    return (
        ground_mask,
        point_id,
        np.asarray(plane_model, dtype=np.float64),
    )


def process_road_frame(
    file_path: str,
    height_threshold: float = 0.20,
) -> list[dict]:
    """
    Process one LiDAR frame and classify local ground/road cells.

    Returns:
        List of dictionaries containing:
        - cell_x
        - cell_y
        - point_count
        - height_variation
        - classification
    """

    # --------------------------------------------------
    # 1. Load LiDAR frame
    # --------------------------------------------------

    data = np.fromfile(
        file_path,
        dtype=np.float32,
    ).reshape(-1, 4)

    if len(data) == 0:
        raise ValueError("LiDAR file is empty.")

    # --------------------------------------------------
    # 2. Extract ground mask while preserving
    #    original point correspondence
    # --------------------------------------------------

    ground_mask, point_id, plane_model = extract_ground_mask(
        data
    )

    # --------------------------------------------------
    # 3. Get ground points using the original mask
    # --------------------------------------------------

    ground_points = data[ground_mask, :3]

    if len(ground_points) == 0:
        raise ValueError("No ground points detected.")

    # --------------------------------------------------
    # 4. Divide ground into local cells
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
    # 5. Analyze all road cells
    # --------------------------------------------------

    results = analyze_road_cells(
        cells,
        height_threshold=height_threshold,
    )

    return results