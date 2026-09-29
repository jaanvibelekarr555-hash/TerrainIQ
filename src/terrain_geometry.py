import numpy as np


def fit_local_plane_pca(
    points: np.ndarray,
) -> np.ndarray:
    """
    Fit a local plane to terrain points using PCA.

    Points must have shape (N, 3) or (N, 4).

    Returns:
        Plane model [a, b, c, d] satisfying:

            ax + by + cz + d = 0
    """

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(
            "Points must have shape (N, 3) or (N, 4)."
        )

    if len(points) < 3:
        raise ValueError(
            "At least 3 points are required."
        )

    xyz = points[:, :3]

    centroid = np.mean(
        xyz,
        axis=0,
    )

    centered = xyz - centroid

    covariance = (
        centered.T @ centered
    ) / len(xyz)

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    # Eigenvector corresponding to the smallest
    # eigenvalue is the local surface normal.
    normal = eigenvectors[:, np.argmin(eigenvalues)]

    normal = normal / np.linalg.norm(normal)

    # Keep the normal direction consistent.
    if normal[2] < 0:
        normal = -normal

    a, b, c = normal

    d = -np.dot(
        normal,
        centroid,
    )

    return np.array(
        [a, b, c, d],
        dtype=np.float64,
    )


def calculate_local_slope(
    points: np.ndarray,
) -> float:
    """
    Calculate local terrain slope in degrees.

    Returns:
        Slope angle between 0 and 90 degrees.
    """

    plane_model = fit_local_plane_pca(
        points
    )

    a, b, c, _ = plane_model

    horizontal_component = np.sqrt(
        a**2 + b**2
    )

    slope_radians = np.arctan2(
        horizontal_component,
        abs(c),
    )

    slope_degrees = np.degrees(
        slope_radians
    )

    return float(slope_degrees)


def calculate_roughness(
    points: np.ndarray,
) -> float:
    """
    Calculate terrain roughness as the RMS
    perpendicular distance of points from the
    locally fitted PCA plane.

    Returns:
        Roughness in metres.
    """

    plane_model = fit_local_plane_pca(
        points
    )

    xyz = points[:, :3]

    a, b, c, d = plane_model

    numerator = np.abs(
        a * xyz[:, 0]
        + b * xyz[:, 1]
        + c * xyz[:, 2]
        + d
    )

    denominator = np.sqrt(
        a**2 + b**2 + c**2
    )

    if denominator == 0:
        raise ValueError(
            "Invalid local plane model."
        )

    distances = numerator / denominator

    roughness = np.sqrt(
        np.mean(distances**2)
    )

    return float(roughness)


def calculate_elevation_variation(
    points: np.ndarray,
) -> float:
    """
    Calculate vertical elevation variation
    inside one terrain cell.

    Returns:
        Maximum Z - minimum Z in metres.
    """

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(
            "Points must have shape (N, 3) or (N, 4)."
        )

    if len(points) == 0:
        raise ValueError(
            "No points provided."
        )

    z_values = points[:, 2]

    return float(
        np.max(z_values)
        - np.min(z_values)
    )


def analyze_terrain_geometry(
    points: np.ndarray,
) -> dict:
    """
    Calculate all terrain geometry features
    for one terrain cell.
    """

    if len(points) < 3:
        return {
            "slope": None,
            "roughness": None,
            "elevation_variation": None,
            "geometry_status": "sparse",
        }

    return {
        "slope": calculate_local_slope(points),
        "roughness": calculate_roughness(points),
        "elevation_variation": (
            calculate_elevation_variation(points)
        ),
        "geometry_status": "valid",
    }