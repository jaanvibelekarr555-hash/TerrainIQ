import numpy as np

from src.config import (
    ROAD_HEIGHT_THRESHOLD,
    ROAD_MIN_POINTS,
)


def calculate_height_variation(points: np.ndarray) -> float:
    """
    Calculate the raw height variation of points in one road cell.

    Points must have shape (N, 3) or (N, 4).
    Z is stored in column 2.
    """

    if points.size == 0:
        raise ValueError("No points provided.")

    z_values = points[:, 2]

    return float(np.max(z_values) - np.min(z_values))


def calculate_plane_deviation(
    points: np.ndarray,
    plane_model: np.ndarray,
) -> float:
    """
    Calculate the maximum perpendicular distance of points
    from a fitted plane.

    Plane equation:
        ax + by + cz + d = 0

    Returns:
        Maximum perpendicular deviation in metres.
    """

    if points.size == 0:
        raise ValueError("No points provided.")

    if len(plane_model) != 4:
        raise ValueError("Plane model must contain [a, b, c, d].")

    xyz = points[:, :3]

    a, b, c, d = plane_model

    numerator = np.abs(
        a * xyz[:, 0]
        + b * xyz[:, 1]
        + c * xyz[:, 2]
        + d
    )

    denominator = np.sqrt(a**2 + b**2 + c**2)

    if denominator == 0:
        raise ValueError("Invalid plane model.")

    distances = numerator / denominator

    return float(np.max(distances))


def classify_road_cell(
    points: np.ndarray,
    height_threshold: float = ROAD_HEIGHT_THRESHOLD,
    plane_model: np.ndarray | None = None,
) -> str:
    """
    Classify one road cell as drivable or non-drivable.

    If a plane model is provided, plane deviation is used.
    Otherwise, raw Z variation is used.
    """

    if plane_model is not None:
        variation = calculate_plane_deviation(
            points,
            plane_model,
        )
    else:
        variation = calculate_height_variation(points)

    if variation <= height_threshold:
        return "drivable"

    return "non-drivable"


def analyze_road_cell(
    points: np.ndarray,
    cell_x: int,
    cell_y: int,
    height_threshold: float = ROAD_HEIGHT_THRESHOLD,
    plane_model: np.ndarray | None = None,
) -> dict:
    """
    Analyze one road cell and return its information.
    """

    point_count = len(points)

    result = {
        "cell_x": int(cell_x),
        "cell_y": int(cell_y),
        "point_count": int(point_count),
        "height_variation": None,
        "classification": "sparse",
    }

    if point_count < ROAD_MIN_POINTS:
        return result

    if plane_model is not None:
        variation = calculate_plane_deviation(
            points,
            plane_model,
        )
    else:
        variation = calculate_height_variation(points)

    classification = (
        "drivable"
        if variation <= height_threshold
        else "non-drivable"
    )

    result["height_variation"] = variation
    result["classification"] = classification

    return result


def analyze_road_cells(
    cells: dict,
    height_threshold: float = ROAD_HEIGHT_THRESHOLD,
    plane_model: np.ndarray | None = None,
) -> list[dict]:
    """
    Analyze all road cells.

    The same fitted ground plane can be supplied to every cell.
    """

    results = []

    for (cell_x, cell_y), cell_points in cells.items():

        cell_points = np.asarray(cell_points)

        result = analyze_road_cell(
            cell_points,
            cell_x=cell_x,
            cell_y=cell_y,
            height_threshold=height_threshold,
            plane_model=plane_model,
        )

        results.append(result)

    return results