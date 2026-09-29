import numpy as np


MAX_RANGE = 100.0
UNIFORM_RESOLUTION = 0.05

DISTANCE_BANDS = (
    (0.0, 10.0, 0.05),
    (10.0, 25.0, 0.10),
    (25.0, 50.0, 0.25),
    (50.0, 100.0, 0.50),
)


def distance_to_resolution(distance_m: float) -> float:
    """
    Convert horizontal LiDAR distance to base spatial resolution.

    Boundary convention:
        0 <= d < 10      -> 0.05 m
        10 <= d < 25     -> 0.10 m
        25 <= d < 50     -> 0.25 m
        50 <= d <= 100   -> 0.50 m
    """
    if not np.isfinite(distance_m):
        raise ValueError("distance_m must be finite.")

    if distance_m < 0:
        raise ValueError("distance_m must be non-negative.")

    if distance_m > MAX_RANGE:
        raise ValueError(
            f"distance_m must not exceed {MAX_RANGE} m."
        )

    for min_distance, max_distance, resolution in DISTANCE_BANDS:
        if min_distance <= distance_m < max_distance:
            return resolution

    if distance_m == MAX_RANGE:
        return DISTANCE_BANDS[-1][2]

    raise ValueError(
        f"No resolution band found for distance {distance_m} m."
    )


def distance_array_to_resolution(
    distance_m: np.ndarray,
) -> np.ndarray:
    """
    Vectorized distance-to-resolution conversion.

    Parameters
    ----------
    distance_m : np.ndarray
        One-dimensional array of horizontal LiDAR distances.

    Returns
    -------
    np.ndarray
        Resolution for every input distance.
    """
    distances = np.asarray(distance_m, dtype=float)

    if distances.ndim != 1:
        raise ValueError("distance_m must be a one-dimensional array.")

    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_m must contain only finite values.")

    if np.any(distances < 0):
        raise ValueError("distance_m must be non-negative.")

    if np.any(distances > MAX_RANGE):
        raise ValueError(
            f"distance_m must not exceed {MAX_RANGE} m."
        )

    resolutions = np.empty_like(distances)

    for min_distance, max_distance, resolution in DISTANCE_BANDS:
        mask = (
            (distances >= min_distance)
            & (distances < max_distance)
        )
        resolutions[mask] = resolution

    resolutions[distances == MAX_RANGE] = DISTANCE_BANDS[-1][2]

    return resolutions