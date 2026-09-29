import numpy as np


DEFAULT_HIGH_PRIORITY_THRESHOLD = 0.70
DEFAULT_MEDIUM_PRIORITY_THRESHOLD = 0.40

DEFAULT_HIGH_REFINEMENT_FACTOR = 0.50
DEFAULT_MEDIUM_REFINEMENT_FACTOR = 0.75

DEFAULT_MIN_RESOLUTION = 0.05


def calculate_requested_resolution(
    base_resolution: float,
    priority: float,
    high_priority_threshold: float = (
        DEFAULT_HIGH_PRIORITY_THRESHOLD
    ),
    medium_priority_threshold: float = (
        DEFAULT_MEDIUM_PRIORITY_THRESHOLD
    ),
    high_refinement_factor: float = (
        DEFAULT_HIGH_REFINEMENT_FACTOR
    ),
    medium_refinement_factor: float = (
        DEFAULT_MEDIUM_REFINEMENT_FACTOR
    ),
    min_resolution: float = DEFAULT_MIN_RESOLUTION,
) -> float:
    """
    Calculate the terrain-requested spatial resolution.

    The terrain branch does not decide the final map
    resolution. It only requests a finer resolution
    when terrain priority is sufficiently high.

    Args:
        base_resolution:
            Current resolution provided by the main
            distance-based resolution system.

        priority:
            Terrain priority in [0, 1].

    Returns:
        Requested resolution in metres.
    """

    if base_resolution <= 0:
        raise ValueError(
            "base_resolution must be greater than zero."
        )

    if not 0.0 <= priority <= 1.0:
        raise ValueError(
            "priority must be in [0, 1]."
        )

    if not (
        0.0 <= medium_priority_threshold
        <= high_priority_threshold
        <= 1.0
    ):
        raise ValueError(
            "Priority thresholds must satisfy "
            "0 <= medium <= high <= 1."
        )

    if high_refinement_factor <= 0:
        raise ValueError(
            "high_refinement_factor must be greater than zero."
        )

    if medium_refinement_factor <= 0:
        raise ValueError(
            "medium_refinement_factor must be greater than zero."
        )

    if min_resolution <= 0:
        raise ValueError(
            "min_resolution must be greater than zero."
        )

    if priority >= high_priority_threshold:
        requested_resolution = (
            base_resolution * high_refinement_factor
        )

    elif priority >= medium_priority_threshold:
        requested_resolution = (
            base_resolution * medium_refinement_factor
        )

    else:
        requested_resolution = base_resolution

    return float(
        max(
            requested_resolution,
            min_resolution,
        )
    )


def build_terrain_refinement_request(
    region: dict,
    base_resolution: float,
    priority: float,
    traversability: str,
    terrain_complexity: float | None,
    terrain_confidence: float,
) -> dict:
    """
    Build the terrain refinement request passed to
    the adaptive resolution system.

    The request contains both:

        base_resolution:
            Distance-based base map resolution.

        requested_resolution:
            Terrain-requested refinement resolution.
            This remains an upper bound for local
            hierarchical refinement.

    Returns:
        Dictionary containing:
        - region
        - base_resolution
        - requested_resolution
        - priority
        - reason
    """

    if base_resolution <= 0:
        raise ValueError(
            "base_resolution must be greater than zero."
        )

    if not 0.0 <= priority <= 1.0:
        raise ValueError(
            "priority must be in [0, 1]."
        )

    if not 0.0 <= terrain_confidence <= 1.0:
        raise ValueError(
            "terrain_confidence must be in [0, 1]."
        )

    if terrain_complexity is not None:
        if not 0.0 <= terrain_complexity <= 1.0:
            raise ValueError(
                "terrain_complexity must be in [0, 1]."
            )

    if not isinstance(region, dict):
        raise ValueError(
            "region must be a dictionary."
        )

    requested_resolution = calculate_requested_resolution(
        base_resolution=base_resolution,
        priority=priority,
    )

    reasons = []

    if traversability == "SPARSE / UNKNOWN":
        reasons.append(
            "sparse or unknown terrain"
        )

    elif traversability == "NON-DRIVABLE":
        reasons.append(
            "non-drivable terrain"
        )

    elif traversability == "DRIVABLE":
        pass

    else:
        raise ValueError(
            "Invalid traversability state."
        )

    if (
        terrain_complexity is not None
        and terrain_complexity >= 0.70
    ):
        reasons.append(
            "high terrain complexity"
        )

    if terrain_confidence < 0.50:
        reasons.append(
            "low terrain confidence"
        )

    if priority >= 0.70:
        reasons.append(
            "high terrain refinement priority"
        )

    elif priority >= 0.40:
        reasons.append(
            "moderate terrain refinement priority"
        )

    if not reasons:
        reasons.append(
            "low-priority terrain; retain base resolution"
        )

    reason = "; ".join(reasons)

    return {
        "region": region,

        "base_resolution": float(
            base_resolution
        ),

        "requested_resolution": (
            requested_resolution
        ),

        "priority": float(
            np.clip(priority, 0.0, 1.0)
        ),

        "reason": reason,
    }