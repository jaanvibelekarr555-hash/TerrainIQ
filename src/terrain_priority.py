import numpy as np


DEFAULT_COMPLEXITY_WEIGHT = 0.50
DEFAULT_RISK_WEIGHT = 0.30
DEFAULT_UNCERTAINTY_WEIGHT = 0.20


def calculate_terrain_priority(
    terrain_complexity: float | None,
    traversability: str,
    terrain_confidence: float,
    complexity_weight: float = DEFAULT_COMPLEXITY_WEIGHT,
    risk_weight: float = DEFAULT_RISK_WEIGHT,
    uncertainty_weight: float = DEFAULT_UNCERTAINTY_WEIGHT,
) -> float:
    """
    Calculate terrain refinement priority in [0, 1].

    Priority represents the need for additional spatial
    detail or information in a terrain region.

    It is NOT a probability of danger and NOT a
    probability of traversability.
    """

    if not 0.0 <= terrain_confidence <= 1.0:
        raise ValueError(
            "terrain_confidence must be in [0, 1]."
        )

    if terrain_complexity is not None:
        if not 0.0 <= terrain_complexity <= 1.0:
            raise ValueError(
                "terrain_complexity must be in [0, 1]."
            )

    weights = (
        complexity_weight,
        risk_weight,
        uncertainty_weight,
    )

    if any(weight < 0 for weight in weights):
        raise ValueError(
            "Priority weights must be non-negative."
        )

    weight_sum = sum(weights)

    if weight_sum == 0:
        raise ValueError(
            "At least one priority weight must be greater than zero."
        )

    # Sparse or unknown terrain has maximum
    # information/refinement need.
    if traversability == "SPARSE / UNKNOWN":
        return 1.0

    elif traversability == "DRIVABLE":
        risk_score = 0.0

    elif traversability == "NON-DRIVABLE":
        risk_score = 1.0

    else:
        raise ValueError(
            "Invalid traversability state."
        )

    # Lower confidence means greater uncertainty.
    uncertainty_score = 1.0 - terrain_confidence

    # A valid terrain cell always has a complexity value.
    if terrain_complexity is None:
        complexity_score = 1.0
    else:
        complexity_score = terrain_complexity

    priority = (
        complexity_weight * complexity_score
        + risk_weight * risk_score
        + uncertainty_weight * uncertainty_score
    ) / weight_sum

    return float(
        np.clip(priority, 0.0, 1.0)
    )