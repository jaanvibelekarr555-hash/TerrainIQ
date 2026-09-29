import numpy as np

from src.resolution_engine import distance_to_resolution


def calculate_cell_distance(cell: dict) -> float:
    """
    Calculate horizontal distance from the LiDAR origin
    to the center of a terrain cell.

    Distance uses only X and Y.
    """
    if not isinstance(cell, dict):
        raise ValueError("cell must be a dictionary.")

    if "bbox" not in cell:
        raise ValueError("cell must contain a bbox.")

    bbox = cell["bbox"]

    required_keys = {
        "x_min",
        "x_max",
        "y_min",
        "y_max",
    }

    if not required_keys.issubset(bbox):
        raise ValueError(
            "bbox must contain x_min, x_max, y_min, and y_max."
        )

    x_min = float(bbox["x_min"])
    x_max = float(bbox["x_max"])
    y_min = float(bbox["y_min"])
    y_max = float(bbox["y_max"])

    if not np.isfinite(
        [x_min, x_max, y_min, y_max]
    ).all():
        raise ValueError("bbox values must be finite.")

    if x_max <= x_min:
        raise ValueError(
            "x_max must be greater than x_min."
        )

    if y_max <= y_min:
        raise ValueError(
            "y_max must be greater than y_min."
        )

    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0

    return float(np.hypot(x_center, y_center))


def calculate_cell_base_resolution(cell: dict) -> float:
    """
    Calculate the distance-based base resolution
    for a terrain cell.
    """
    distance = calculate_cell_distance(cell)

    return distance_to_resolution(distance)


def build_base_resolution_by_cell(
    cells: dict[tuple[int, int], dict],
) -> dict[tuple[int, int], float]:
    """
    Calculate the distance-based base resolution
    for every terrain cell.
    """
    if not isinstance(cells, dict):
        raise ValueError("cells must be a dictionary.")

    base_resolution_by_cell = {}

    for cell_id, cell in cells.items():
        base_resolution_by_cell[cell_id] = (
            calculate_cell_base_resolution(cell)
        )

    return base_resolution_by_cell