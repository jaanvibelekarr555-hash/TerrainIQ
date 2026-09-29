from __future__ import annotations

import numpy as np

from src.terrain_geometry import fit_local_plane_pca


DEFAULT_MAX_PLANE_RESIDUAL = 0.08
DEFAULT_MAX_ELEVATION_VARIATION = 0.20


def calculate_plane_residuals(
    points: np.ndarray,
) -> np.ndarray:
    """
    Calculate the absolute perpendicular distance of every
    terrain point from the locally fitted PCA plane.

    Returns:
        One residual value per point, in metres.
    """

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(
            "Points must have shape (N, 3) or (N, 4)."
        )

    if len(points) < 3:
        raise ValueError(
            "At least 3 points are required."
        )

    xyz = np.asarray(
        points[:, :3],
        dtype=float,
    )

    if not np.isfinite(xyz).all():
        raise ValueError(
            "Points must contain finite values."
        )

    plane_model = fit_local_plane_pca(
        points
    )

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

    if denominator == 0.0:
        raise ValueError(
            "Invalid local plane model."
        )

    return numerator / denominator


def calculate_terrain_error_metrics(
    points: np.ndarray,
) -> dict:
    """
    Calculate terrain geometric error metrics for one region.

    Metrics:
        - max_plane_residual
        - rms_plane_residual
        - elevation_variation
        - point_count
    """

    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(
            "Points must have shape (N, 3) or (N, 4)."
        )

    point_count = len(points)

    if point_count < 3:
        return {
            "max_plane_residual": None,
            "rms_plane_residual": None,
            "elevation_variation": None,
            "point_count": point_count,
            "geometry_status": "sparse",
        }

    xyz = np.asarray(
        points[:, :3],
        dtype=float,
    )

    if not np.isfinite(xyz).all():
        raise ValueError(
            "Points must contain finite values."
        )

    residuals = calculate_plane_residuals(
        points
    )

    elevation_variation = float(
        np.max(xyz[:, 2])
        - np.min(xyz[:, 2])
    )

    max_plane_residual = float(
        np.max(residuals)
    )

    rms_plane_residual = float(
        np.sqrt(
            np.mean(residuals**2)
        )
    )

    return {
        "max_plane_residual": max_plane_residual,
        "rms_plane_residual": rms_plane_residual,
        "elevation_variation": elevation_variation,
        "point_count": point_count,
        "geometry_status": "valid",
    }


def decide_terrain_error_bound(
    points: np.ndarray,
    max_plane_residual: float = (
        DEFAULT_MAX_PLANE_RESIDUAL
    ),
    max_elevation_variation: float = (
        DEFAULT_MAX_ELEVATION_VARIATION
    ),
    max_points_per_region: int | None = None,
) -> dict:
    """
    Decide whether a terrain region should be subdivided.

    A region requests splitting when at least one enabled
    condition exceeds its threshold:

        1. maximum PCA plane residual
        2. elevation variation
        3. maximum points per region

    Default development thresholds:

        max_plane_residual = 0.08 m
        max_elevation_variation = 0.20 m

    These defaults are configurable and may be revalidated
    as additional frames are evaluated.

    Returns:
        {
            "split": bool,
            "reason": str,
            "metrics": {...}
        }
    """

    if max_plane_residual <= 0:
        raise ValueError(
            "max_plane_residual must be greater than zero."
        )

    if max_elevation_variation <= 0:
        raise ValueError(
            "max_elevation_variation must be greater than zero."
        )

    if (
        max_points_per_region is not None
        and max_points_per_region <= 0
    ):
        raise ValueError(
            "max_points_per_region must be greater than zero "
            "when provided."
        )

    metrics = calculate_terrain_error_metrics(
        points
    )

    if metrics["geometry_status"] == "sparse":
        return {
            "split": False,
            "reason": "sparse geometry; error bound not evaluated",
            "metrics": metrics,
        }

    reasons = []

    if (
        metrics["max_plane_residual"]
        > max_plane_residual
    ):
        reasons.append(
            "max plane residual exceeds threshold"
        )

    if (
        metrics["elevation_variation"]
        > max_elevation_variation
    ):
        reasons.append(
            "elevation variation exceeds threshold"
        )

    if (
        max_points_per_region is not None
        and metrics["point_count"]
        > max_points_per_region
    ):
        reasons.append(
            "point count exceeds threshold"
        )

    if reasons:
        return {
            "split": True,
            "reason": "; ".join(reasons),
            "metrics": metrics,
        }

    return {
        "split": False,
        "reason": "terrain error within bounds",
        "metrics": metrics,
    }