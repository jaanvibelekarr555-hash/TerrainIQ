from __future__ import annotations

import numpy as np

from src.terrain_complexity import (
    calculate_terrain_complexity,
)
from src.terrain_geometry import (
    analyze_terrain_geometry,
)
from src.terrain_priority import (
    calculate_terrain_priority,
)
from src.traversability import (
    calculate_traversability_confidence,
    classify_traversability,
)


def _validate_data(
    data: np.ndarray,
) -> None:
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(
            "data must have shape (N, 3) or (N, 4)."
        )

    if len(data) == 0:
        raise ValueError(
            "data must not be empty."
        )


def _get_points(
    data: np.ndarray,
    point_indices: list[int],
) -> np.ndarray:
    indices = np.asarray(
        point_indices,
        dtype=np.int64,
    )

    if indices.ndim != 1:
        raise ValueError(
            "point_indices must have shape (N,)."
        )

    if np.any(indices < 0) or np.any(
        indices >= len(data)
    ):
        raise ValueError(
            "point_indices contains invalid indices."
        )

    return np.asarray(
        data[indices, :3],
        dtype=float,
    )


def _build_parent_terrain_features(
    data: np.ndarray,
    terrain_quadtree_result: dict,
) -> dict:
    """
    Calculate terrain interpretation for the original
    terrain region before quadtree refinement.

    These features are intentionally inherited by all
    final leaves belonging to this parent terrain cell.

    This avoids recalculating terrain geometry on tiny
    leaves that may contain only 1-2 points.
    """

    parent_point_indices = []

    for leaf in terrain_quadtree_result["leaves"]:

        if "point_indices" not in leaf:
            raise ValueError(
                "Each leaf must contain point_indices."
            )

        parent_point_indices.extend(
            int(point_id)
            for point_id in leaf["point_indices"]
        )

    if len(parent_point_indices) == 0:
        return {
            "slope": None,
            "roughness": None,
            "elevation_variation": None,
            "geometry_status": "sparse",
            "terrain_complexity": None,
            "traversability": "SPARSE / UNKNOWN",
            "terrain_confidence": 0.0,
            "terrain_priority": (
                calculate_terrain_priority(
                    terrain_complexity=None,
                    traversability="SPARSE / UNKNOWN",
                    terrain_confidence=0.0,
                )
            ),
        }

    # Every parent terrain point should occur once.
    if len(
        np.unique(
            np.asarray(
                parent_point_indices,
                dtype=np.int64,
            )
        )
    ) != len(parent_point_indices):
        raise ValueError(
            "Parent terrain region contains duplicate point indices."
        )

    parent_points = _get_points(
        data=data,
        point_indices=parent_point_indices,
    )

    geometry = analyze_terrain_geometry(
        parent_points
    )

    if geometry["geometry_status"] != "valid":

        terrain_priority = (
            calculate_terrain_priority(
                terrain_complexity=None,
                traversability="SPARSE / UNKNOWN",
                terrain_confidence=0.0,
            )
        )

        return {
            "slope": geometry["slope"],
            "roughness": geometry["roughness"],
            "elevation_variation": (
                geometry["elevation_variation"]
            ),
            "geometry_status": (
                geometry["geometry_status"]
            ),
            "terrain_complexity": None,
            "traversability": "SPARSE / UNKNOWN",
            "terrain_confidence": 0.0,
            "terrain_priority": terrain_priority,
        }

    terrain_complexity = (
        calculate_terrain_complexity(
            slope=geometry["slope"],
            roughness=geometry["roughness"],
            elevation_variation=(
                geometry["elevation_variation"]
            ),
        )
    )

    traversability = classify_traversability(
        slope=geometry["slope"],
        roughness=geometry["roughness"],
        elevation_variation=(
            geometry["elevation_variation"]
        ),
        geometry_status="valid",
    )

    terrain_confidence = (
        calculate_traversability_confidence(
            point_count=len(parent_point_indices),
            complexity=terrain_complexity,
            geometry_status="valid",
        )
    )

    terrain_priority = (
        calculate_terrain_priority(
            terrain_complexity=terrain_complexity,
            traversability=traversability,
            terrain_confidence=terrain_confidence,
        )
    )

    return {
        "slope": geometry["slope"],
        "roughness": geometry["roughness"],
        "elevation_variation": (
            geometry["elevation_variation"]
        ),
        "geometry_status": (
            geometry["geometry_status"]
        ),
        "terrain_complexity": terrain_complexity,
        "traversability": traversability,
        "terrain_confidence": terrain_confidence,
        "terrain_priority": terrain_priority,
    }


def build_terrain_2_5d_cell(
    data: np.ndarray,
    terrain_quadtree_result: dict,
) -> list[dict]:
    """
    Convert one terrain quadtree result into 2.5D
    terrain-map cells.

    Terrain interpretation is calculated once for the
    parent terrain region and inherited by all final
    quadtree leaves.

    Each occupied quadtree leaf contains:
        - spatial bounding box
        - centroid X/Y
        - elevation statistics
        - parent terrain slope
        - parent terrain roughness
        - parent elevation variation
        - parent terrain complexity
        - parent traversability
        - parent terrain confidence
        - parent terrain priority
        - resolution
        - depth
        - point count
        - point indices
        - occupancy
        - error metrics
    """

    _validate_data(data)

    if not isinstance(
        terrain_quadtree_result,
        dict,
    ):
        raise ValueError(
            "terrain_quadtree_result must be a dictionary."
        )

    required_keys = {
        "cell_id",
        "bbox",
        "leaves",
    }

    if not required_keys.issubset(
        terrain_quadtree_result
    ):
        raise ValueError(
            "terrain_quadtree_result is missing required fields."
        )

    parent_terrain_features = (
        _build_parent_terrain_features(
            data=data,
            terrain_quadtree_result=terrain_quadtree_result,
        )
    )

    cells_2_5d = []

    parent_cell_id = (
        terrain_quadtree_result["cell_id"]
    )

    parent_bbox = (
        terrain_quadtree_result["bbox"]
    )

    for leaf_index, leaf in enumerate(
        terrain_quadtree_result["leaves"]
    ):

        if "point_indices" not in leaf:
            raise ValueError(
                "Each leaf must contain point_indices."
            )

        if "bbox" not in leaf:
            raise ValueError(
                "Each leaf must contain bbox."
            )

        point_indices = [
            int(point_id)
            for point_id in leaf["point_indices"]
        ]

        if len(point_indices) == 0:
            continue

        points = _get_points(
            data=data,
            point_indices=point_indices,
        )

        bbox = {
            "x_min": float(
                leaf["bbox"]["x_min"]
            ),
            "x_max": float(
                leaf["bbox"]["x_max"]
            ),
            "y_min": float(
                leaf["bbox"]["y_min"]
            ),
            "y_max": float(
                leaf["bbox"]["y_max"]
            ),
        }

        x_center = (
            bbox["x_min"]
            + bbox["x_max"]
        ) / 2.0

        y_center = (
            bbox["y_min"]
            + bbox["y_max"]
        ) / 2.0

        z_values = points[:, 2]

        cell = {
            "cell_id": parent_cell_id,

            "leaf_id": (
                parent_cell_id,
                int(leaf["depth"]),
                int(leaf_index),
            ),

            "bbox": bbox,

            "centroid": {
                "x": float(x_center),
                "y": float(y_center),
            },

            # Leaf-level elevation information.
            "z_mean": float(
                np.mean(z_values)
            ),

            "z_min": float(
                np.min(z_values)
            ),

            "z_max": float(
                np.max(z_values)
            ),

            "elevation": float(
                np.mean(z_values)
            ),

            # Parent-level terrain interpretation.
            "slope": parent_terrain_features[
                "slope"
            ],

            "roughness": parent_terrain_features[
                "roughness"
            ],

            "elevation_variation": (
                parent_terrain_features[
                    "elevation_variation"
                ]
            ),

            "geometry_status": (
                parent_terrain_features[
                    "geometry_status"
                ]
            ),

            "terrain_complexity": (
                parent_terrain_features[
                    "terrain_complexity"
                ]
            ),

            "traversability": (
                parent_terrain_features[
                    "traversability"
                ]
            ),

            "terrain_confidence": (
                parent_terrain_features[
                    "terrain_confidence"
                ]
            ),

            "terrain_priority": (
                parent_terrain_features[
                    "terrain_priority"
                ]
            ),

            # Hierarchical spatial information.
            "resolution": float(
                leaf["cell_size"]
            ),

            "depth": int(
                leaf["depth"]
            ),

            "point_count": len(
                point_indices
            ),

            "point_indices": point_indices,

            "occupied": True,

            "parent_bbox": dict(
                parent_bbox
            ),

            # Error-bound refinement information.
            "error_bound_split": bool(
                leaf.get(
                    "error_bound_split",
                    False,
                )
            ),

            "error_bound_reason": leaf.get(
                "error_bound_reason"
            ),

            "error_metrics": leaf.get(
                "error_metrics"
            ),
        }

        cells_2_5d.append(cell)

    return cells_2_5d


def build_terrain_2_5d_map(
    data: np.ndarray,
    terrain_quadtree_map: dict[
        tuple[int, int],
        dict,
    ],
) -> list[dict]:
    """
    Convert the complete terrain quadtree map into
    a flat list of occupied 2.5D terrain cells.

    Terrain attributes are inherited from the parent
    terrain region of each quadtree result.
    """

    _validate_data(data)

    if not isinstance(
        terrain_quadtree_map,
        dict,
    ):
        raise ValueError(
            "terrain_quadtree_map must be a dictionary."
        )

    terrain_map = []

    for result in terrain_quadtree_map.values():

        cells = build_terrain_2_5d_cell(
            data=data,
            terrain_quadtree_result=result,
        )

        terrain_map.extend(cells)

    return terrain_map


def validate_terrain_2_5d_point_conservation(
    terrain_map: list[dict],
    expected_point_ids: np.ndarray,
) -> bool:
    """
    Verify that every expected original point ID occurs
    exactly once in the final 2.5D map.
    """

    if not isinstance(
        terrain_map,
        list,
    ):
        raise ValueError(
            "terrain_map must be a list."
        )

    expected = np.asarray(
        expected_point_ids,
        dtype=np.int64,
    )

    if expected.ndim != 1:
        raise ValueError(
            "expected_point_ids must be one-dimensional."
        )

    if len(np.unique(expected)) != len(expected):
        return False

    actual_ids = []

    for cell in terrain_map:

        if "point_indices" not in cell:
            return False

        actual_ids.extend(
            cell["point_indices"]
        )

    actual = np.asarray(
        actual_ids,
        dtype=np.int64,
    )

    if len(actual) != len(expected):
        return False

    if len(np.unique(actual)) != len(actual):
        return False

    return bool(
        np.array_equal(
            np.sort(actual),
            np.sort(expected),
        )
    )