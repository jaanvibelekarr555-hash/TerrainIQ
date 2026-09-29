import numpy as np


def normalize_feature(
    value: float,
    limit: float,
) -> float:
    """
    Normalize a non-negative terrain feature to [0, 1].

    Values at or above the limit become 1.0.
    """

    if value < 0:
        raise ValueError(
            "Feature value must be non-negative."
        )

    if limit <= 0:
        raise ValueError(
            "Normalization limit must be greater than zero."
        )

    return float(
        np.clip(value / limit, 0.0, 1.0)
    )


def calculate_terrain_complexity(
    slope: float,
    roughness: float,
    elevation_variation: float,
    slope_limit: float = 30.0,
    roughness_limit: float = 0.20,
    elevation_limit: float = 0.50,
    weights: tuple[float, float, float] | None = None,
) -> float:
    """
    Calculate terrain complexity in the range [0, 1].

    Inputs:
        slope:
            Local terrain slope in degrees.

        roughness:
            RMS distance from the local terrain plane in metres.

        elevation_variation:
            Maximum Z - minimum Z in metres.

    The default prototype uses equal weights.
    """

    if slope < 0:
        raise ValueError(
            "Slope must be non-negative."
        )

    if roughness < 0:
        raise ValueError(
            "Roughness must be non-negative."
        )

    if elevation_variation < 0:
        raise ValueError(
            "Elevation variation must be non-negative."
        )

    if weights is None:
        weights = (1.0, 1.0, 1.0)

    if len(weights) != 3:
        raise ValueError(
            "Three weights are required."
        )

    if any(weight < 0 for weight in weights):
        raise ValueError(
            "Weights must be non-negative."
        )

    weight_sum = sum(weights)

    if weight_sum == 0:
        raise ValueError(
            "At least one weight must be greater than zero."
        )

    slope_score = normalize_feature(
        slope,
        slope_limit,
    )

    roughness_score = normalize_feature(
        roughness,
        roughness_limit,
    )

    elevation_score = normalize_feature(
        elevation_variation,
        elevation_limit,
    )

    complexity = (
        weights[0] * slope_score
        + weights[1] * roughness_score
        + weights[2] * elevation_score
    ) / weight_sum

    return float(
        np.clip(complexity, 0.0, 1.0)
    )