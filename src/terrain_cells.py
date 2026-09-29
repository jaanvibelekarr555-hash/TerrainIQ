import numpy as np


def build_terrain_cells(
    data: np.ndarray,
    ground_mask: np.ndarray,
    point_id: np.ndarray,
    cell_size: float = 1.0,
) -> dict[tuple[int, int], dict]:
    """
    Build spatial terrain cells from ground LiDAR points.

    Args:
        data:
            Original LiDAR data with shape (N, 4):
            [X, Y, Z, Intensity]

        ground_mask:
            Boolean array of shape (N,).
            True indicates a ground point.

        point_id:
            Original point IDs with shape (N,).

        cell_size:
            XY cell size in metres.

    Returns:
        Dictionary mapping cell_id -> cell information.
    """

    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(
            "LiDAR data must have shape (N, 3) or (N, 4)."
        )

    if len(ground_mask) != len(data):
        raise ValueError(
            "ground_mask must have the same length as data."
        )

    if len(point_id) != len(data):
        raise ValueError(
            "point_id must have the same length as data."
        )

    if cell_size <= 0:
        raise ValueError(
            "cell_size must be greater than zero."
        )

    ground_indices = point_id[ground_mask]
    ground_points = data[ground_mask, :3]

    if len(ground_points) == 0:
        return {}

    cell_x = np.floor(
        ground_points[:, 0] / cell_size
    ).astype(int)

    cell_y = np.floor(
        ground_points[:, 1] / cell_size
    ).astype(int)

    cells = {}

    for i in range(len(ground_points)):

        cell_id = (
            int(cell_x[i]),
            int(cell_y[i]),
        )

        if cell_id not in cells:

            x_min = cell_id[0] * cell_size
            x_max = x_min + cell_size

            y_min = cell_id[1] * cell_size
            y_max = y_min + cell_size

            cells[cell_id] = {
                "cell_id": cell_id,
                "bbox": {
                    "x_min": float(x_min),
                    "x_max": float(x_max),
                    "y_min": float(y_min),
                    "y_max": float(y_max),
                },
                "point_indices": [],
                "point_count": 0,
            }

        cells[cell_id]["point_indices"].append(
            int(ground_indices[i])
        )

        cells[cell_id]["point_count"] += 1

    return cells