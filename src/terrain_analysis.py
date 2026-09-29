import numpy as np

from src.terrain_geometry import analyze_terrain_geometry
from src.terrain_complexity import calculate_terrain_complexity
from src.traversability import (
    classify_traversability,
    calculate_traversability_confidence,
)
from src.terrain_priority import calculate_terrain_priority
from src.terrain_refinement import build_terrain_refinement_request


def analyze_terrain_cells(
    data: np.ndarray,
    cells: dict[tuple[int, int], dict],
    base_resolution_by_cell: (
        dict[tuple[int, int], float] | None
    ) = None,
) -> dict[tuple[int, int], dict]:
    """
    Analyze every terrain cell.

    Adds:
        - slope
        - roughness
        - elevation_variation
        - geometry_status
        - terrain_complexity
        - traversability
        - terrain_confidence
        - terrain_priority
        - refinement_request

    Args:
        data:
            Original LiDAR data with shape (N, 3) or (N, 4).

        cells:
            Terrain cells containing original LiDAR point indices.

        base_resolution_by_cell:
            Optional mapping from cell_id to the base
            resolution supplied by the shared distance-based
            resolution system.

            Example:
                {
                    (0, 0): 0.05,
                    (1, 0): 0.10,
                    (2, 0): 0.25,
                }

            If this is not provided, terrain analysis still
            works, but refinement requests are not created.
    """

    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(
            "LiDAR data must have shape (N, 3) or (N, 4)."
        )

    if base_resolution_by_cell is not None:
        if not isinstance(
            base_resolution_by_cell,
            dict,
        ):
            raise ValueError(
                "base_resolution_by_cell must be a dictionary."
            )

        for cell_id, resolution in (
            base_resolution_by_cell.items()
        ):
            if resolution <= 0:
                raise ValueError(
                    f"Base resolution for cell {cell_id} "
                    "must be greater than zero."
                )

    analyzed_cells = {}

    for cell_id, cell in cells.items():

        point_indices = np.asarray(
            cell["point_indices"],
            dtype=np.int64,
        )

        if np.any(point_indices < 0) or np.any(
            point_indices >= len(data)
        ):
            raise ValueError(
                f"Cell {cell_id} contains invalid point indices."
            )

        points = data[point_indices, :3]

        geometry = analyze_terrain_geometry(
            points
        )

        result = {
            **cell,
            "slope": geometry["slope"],
            "roughness": geometry["roughness"],
            "elevation_variation": (
                geometry["elevation_variation"]
            ),
            "geometry_status": geometry["geometry_status"],
            "terrain_complexity": None,
            "traversability": "SPARSE / UNKNOWN",
            "terrain_confidence": 0.0,
            "terrain_priority": None,
            "refinement_request": None,
        }

        if geometry["geometry_status"] == "valid":

            complexity = calculate_terrain_complexity(
                slope=geometry["slope"],
                roughness=geometry["roughness"],
                elevation_variation=(
                    geometry["elevation_variation"]
                ),
            )

            traversability = classify_traversability(
                slope=geometry["slope"],
                roughness=geometry["roughness"],
                elevation_variation=(
                    geometry["elevation_variation"]
                ),
                geometry_status="valid",
            )

            confidence = calculate_traversability_confidence(
                point_count=len(point_indices),
                complexity=complexity,
                geometry_status="valid",
            )

            priority = calculate_terrain_priority(
                terrain_complexity=complexity,
                traversability=traversability,
                terrain_confidence=confidence,
            )

            result["terrain_complexity"] = complexity
            result["traversability"] = traversability
            result["terrain_confidence"] = confidence
            result["terrain_priority"] = priority

        else:

            priority = calculate_terrain_priority(
                terrain_complexity=None,
                traversability="SPARSE / UNKNOWN",
                terrain_confidence=0.0,
            )

            result["terrain_priority"] = priority

        # --------------------------------------------------
        # Build refinement request only when the shared
        # distance-based base resolution is available.
        # --------------------------------------------------

        if base_resolution_by_cell is not None:

            if cell_id not in base_resolution_by_cell:
                raise ValueError(
                    f"No base resolution supplied for cell {cell_id}."
                )

            base_resolution = (
                base_resolution_by_cell[cell_id]
            )

            region = {
                "cell_id": cell_id,
                "bbox": result["bbox"],
            }

            refinement_request = (
                build_terrain_refinement_request(
                    region=region,
                    base_resolution=base_resolution,
                    priority=priority,
                    traversability=result["traversability"],
                    terrain_complexity=result[
                        "terrain_complexity"
                    ],
                    terrain_confidence=result[
                        "terrain_confidence"
                    ],
                )
            )

            result["refinement_request"] = (
                refinement_request
            )

        analyzed_cells[cell_id] = result

    return analyzed_cells