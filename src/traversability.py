import numpy as np


DEFAULT_MAX_SLOPE = 30.0
DEFAULT_MAX_ROUGHNESS = 0.20
DEFAULT_MAX_ELEVATION_VARIATION = 0.50


def classify_traversability(
    slope: float | None,
    roughness: float | None,
    elevation_variation: float | None,
    geometry_status: str,
    max_slope: float = DEFAULT_MAX_SLOPE,
    max_roughness: float = DEFAULT_MAX_ROUGHNESS,
    max_elevation_variation: float = (
        DEFAULT_MAX_ELEVATION_VARIATION
    ),
) -> str:
    """
    Classify a terrain cell as:

        DRIVABLE
        NON-DRIVABLE
        SPARSE / UNKNOWN

    Geometry thresholds are configurable and are
    prototype values until validated on real terrain.
    """

    if geometry_status == "sparse":
        return "SPARSE / UNKNOWN"

    if geometry_status != "valid":
        raise ValueError(
            "geometry_status must be 'valid' or 'sparse'."
        )

    if slope is None:
        raise ValueError(
            "Slope is required for a valid cell."
        )

    if roughness is None:
        raise ValueError(
            "Roughness is required for a valid cell."
        )

    if elevation_variation is None:
        raise ValueError(
            "Elevation variation is required for a valid cell."
        )

    if max_slope <= 0:
        raise ValueError(
            "max_slope must be greater than zero."
        )

    if max_roughness <= 0:
        raise ValueError(
            "max_roughness must be greater than zero."
        )

    if max_elevation_variation <= 0:
        raise ValueError(
            "max_elevation_variation must be greater than zero."
        )

    if (
        slope > max_slope
        or roughness > max_roughness
        or elevation_variation > max_elevation_variation
    ):
        return "NON-DRIVABLE"

    return "DRIVABLE"


def calculate_traversability_confidence(
    point_count: int,
    complexity: float | None,
    geometry_status: str,
) -> float:
    """
    Calculate a prototype confidence score in [0, 1].

    Confidence represents how much geometric evidence
    is available for the cell. It is NOT a probability
    that the terrain is drivable.
    """

    if point_count < 0:
        raise ValueError(
            "point_count cannot be negative."
        )

    if geometry_status == "sparse":
        return 0.0

    if geometry_status != "valid":
        raise ValueError(
            "geometry_status must be 'valid' or 'sparse'."
        )

    if complexity is None:
        raise ValueError(
            "Complexity is required for a valid cell."
        )

    if not 0.0 <= complexity <= 1.0:
        raise ValueError(
            "Complexity must be in the range [0, 1]."
        )

    # More points provide stronger geometric evidence.
    # Saturates at 20 points.
    point_evidence = min(
        point_count / 20.0,
        1.0,
    )

    # Higher complexity does not mean lower confidence.
    # It means the terrain is more complex.
    # Confidence here is based on point evidence.
    confidence = point_evidence

    return float(
        np.clip(confidence, 0.0, 1.0)
    )