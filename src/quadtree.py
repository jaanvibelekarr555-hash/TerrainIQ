from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class QuadTreeNode:
    """
    One node of a 2D quadtree.

    The node stores original LiDAR point IDs.
    Z is not used for spatial subdivision.
    """

    bbox: dict[str, float]
    point_indices: list[int]
    depth: int = 0
    children: list["QuadTreeNode"] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def point_count(self) -> int:
        return len(self.point_indices)

    @property
    def cell_size(self) -> float:
        return self.bbox["x_max"] - self.bbox["x_min"]

    def subdivide(
        self,
        points: np.ndarray,
        point_ids: np.ndarray | None = None,
    ) -> None:
        """
        Split this node into four children.

        The children receive the original point IDs.
        """

        if not self.is_leaf:
            raise ValueError(
                "Node is already subdivided."
            )

        if points.ndim != 2 or points.shape[1] < 2:
            raise ValueError(
                "points must have shape (N, 2) or greater."
            )

        if len(points) == 0:
            return

        if point_ids is None:
            source_ids = np.arange(
                len(points),
                dtype=np.int64,
            )
        else:
            source_ids = np.asarray(
                point_ids,
                dtype=np.int64,
            )

            if source_ids.ndim != 1 or len(source_ids) != len(points):
                raise ValueError(
                    "point_ids must have shape (N,)."
                )

        id_to_local_index = {
            int(point_id): index
            for index, point_id in enumerate(source_ids)
        }

        node_ids = np.asarray(
            self.point_indices,
            dtype=np.int64,
        )

        missing_ids = [
            int(point_id)
            for point_id in node_ids
            if int(point_id) not in id_to_local_index
        ]

        if missing_ids:
            raise ValueError(
                "Node contains point IDs that are not present "
                "in point_ids."
            )

        local_indices = np.asarray(
            [
                id_to_local_index[int(point_id)]
                for point_id in node_ids
            ],
            dtype=np.int64,
        )

        x_min = self.bbox["x_min"]
        x_max = self.bbox["x_max"]
        y_min = self.bbox["y_min"]
        y_max = self.bbox["y_max"]

        x_mid = (x_min + x_max) / 2.0
        y_mid = (y_min + y_max) / 2.0

        child_bboxes = [
            {
                "x_min": x_min,
                "x_max": x_mid,
                "y_min": y_min,
                "y_max": y_mid,
            },
            {
                "x_min": x_mid,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_mid,
            },
            {
                "x_min": x_min,
                "x_max": x_mid,
                "y_min": y_mid,
                "y_max": y_max,
            },
            {
                "x_min": x_mid,
                "x_max": x_max,
                "y_min": y_mid,
                "y_max": y_max,
            },
        ]

        child_indices = [
            [],
            [],
            [],
            [],
        ]

        for local_index in local_indices:
            x = float(points[local_index, 0])
            y = float(points[local_index, 1])

            if not np.isfinite([x, y]).all():
                raise ValueError(
                    "points must contain finite X/Y values."
                )

            child_number = 0

            if x >= x_mid:
                child_number += 1

            if y >= y_mid:
                child_number += 2

            child_indices[child_number].append(
                int(source_ids[local_index])
            )

        self.children = [
            QuadTreeNode(
                bbox=child_bboxes[i],
                point_indices=child_indices[i],
                depth=self.depth + 1,
            )
            for i in range(4)
        ]

        # Once the node is split, its points are owned by its children.
        self.point_indices = []


def _validate_bbox(
    bbox: dict[str, float],
) -> dict[str, float]:
    required = {
        "x_min",
        "x_max",
        "y_min",
        "y_max",
    }

    if not isinstance(bbox, dict) or not required.issubset(bbox):
        raise ValueError(
            "bbox must contain x_min, x_max, y_min, and y_max."
        )

    result = {
        key: float(bbox[key])
        for key in required
    }

    if not np.isfinite(
        list(result.values())
    ).all():
        raise ValueError(
            "bbox values must be finite."
        )

    if result["x_max"] <= result["x_min"]:
        raise ValueError(
            "x_max must be greater than x_min."
        )

    if result["y_max"] <= result["y_min"]:
        raise ValueError(
            "y_max must be greater than y_min."
        )

    if not np.isclose(
        result["x_max"] - result["x_min"],
        result["y_max"] - result["y_min"],
    ):
        raise ValueError(
            "Quadtree root bbox must be square."
        )

    return result


def _build_root_bbox(
    points: np.ndarray,
) -> dict[str, float]:
    x_min = float(np.min(points[:, 0]))
    x_max = float(np.max(points[:, 0]))
    y_min = float(np.min(points[:, 1]))
    y_max = float(np.max(points[:, 1]))

    span = max(
        x_max - x_min,
        y_max - y_min,
    )

    if span == 0.0:
        span = 1.0

    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    half_span = span / 2.0

    return {
        "x_min": x_center - half_span,
        "x_max": x_center + half_span,
        "y_min": y_center - half_span,
        "y_max": y_center + half_span,
    }


def build_quadtree(
    points: np.ndarray,
    point_ids: np.ndarray | None = None,
    bbox: dict[str, float] | None = None,
    max_points_per_leaf: int = 0,
    max_depth: int = 0,
) -> QuadTreeNode:
    """
    Build a 2D quadtree from XY coordinates.

    Parameters
    ----------
    points:
        Array with shape (N, 2) or greater.

    point_ids:
        Original point IDs. If omitted, 0..N-1 are used.

    bbox:
        Optional square root bounding box.

    max_points_per_leaf:
        Automatically split nodes containing more than this
        many points. 0 disables point-count-based splitting.

    max_depth:
        Maximum tree depth when automatic splitting is enabled.
        0 means no depth limit is applied.
    """

    points = np.asarray(points)

    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(
            "points must have shape (N, 2) or greater."
        )

    if len(points) == 0:
        raise ValueError(
            "points must not be empty."
        )

    xy = np.asarray(
        points[:, :2],
        dtype=float,
    )

    if not np.isfinite(xy).all():
        raise ValueError(
            "points must contain finite X/Y values."
        )

    if point_ids is None:
        point_ids = np.arange(
            len(points),
            dtype=np.int64,
        )
    else:
        point_ids = np.asarray(
            point_ids,
            dtype=np.int64,
        )

        if (
            point_ids.ndim != 1
            or len(point_ids) != len(points)
        ):
            raise ValueError(
                "point_ids must have shape (N,)."
            )

    if max_points_per_leaf < 0:
        raise ValueError(
            "max_points_per_leaf must be non-negative."
        )

    if max_depth < 0:
        raise ValueError(
            "max_depth must be non-negative."
        )

    root_bbox = (
        _build_root_bbox(xy)
        if bbox is None
        else _validate_bbox(bbox)
    )

    root = QuadTreeNode(
        bbox=root_bbox,
        point_indices=point_ids.tolist(),
    )

    def split_recursively(
        node: QuadTreeNode,
    ) -> None:

        if node.depth >= max_depth and max_depth > 0:
            return

        if (
            max_points_per_leaf <= 0
            or node.point_count <= max_points_per_leaf
        ):
            return

        before_count = node.point_count

        node.subdivide(
            xy,
            point_ids=point_ids,
        )

        child_count = sum(
            child.point_count
            for child in node.children
        )

        if child_count != before_count:
            raise RuntimeError(
                "Quadtree subdivision changed the point count."
            )

        # If all points remain in one child, further splitting
        # cannot improve the spatial partition at this level.
        non_empty_children = [
            child
            for child in node.children
            if child.point_count > 0
        ]

        if (
            len(non_empty_children) == 1
            and non_empty_children[0].point_count == before_count
        ):
            return

        for child in node.children:
            split_recursively(child)

    split_recursively(root)

    return root


def get_leaf_nodes(
    root: QuadTreeNode,
) -> list[QuadTreeNode]:
    """
    Return all leaf nodes in depth-first order.
    """

    if not isinstance(root, QuadTreeNode):
        raise ValueError(
            "root must be a QuadTreeNode."
        )

    if root.is_leaf:
        return [root]

    leaves = []

    for child in root.children:
        leaves.extend(
            get_leaf_nodes(child)
        )

    return leaves


def collect_leaf_point_ids(
    root: QuadTreeNode,
) -> np.ndarray:
    """
    Collect all original point IDs stored by leaf nodes.
    """

    leaves = get_leaf_nodes(root)

    point_ids = []

    for leaf in leaves:
        point_ids.extend(
            leaf.point_indices
        )

    return np.asarray(
        point_ids,
        dtype=np.int64,
    )


def validate_point_conservation(
    root: QuadTreeNode,
    expected_point_ids: np.ndarray,
) -> bool:
    """
    Verify that every expected point ID appears exactly once
    across all leaf nodes.
    """

    expected = np.asarray(
        expected_point_ids,
        dtype=np.int64,
    )

    if expected.ndim != 1:
        raise ValueError(
            "expected_point_ids must be one-dimensional."
        )

    actual = collect_leaf_point_ids(root)

    if len(actual) != len(expected):
        return False

    return bool(
        np.array_equal(
            np.sort(actual),
            np.sort(expected),
        )
    )
def refine_node_to_resolution(
    node: QuadTreeNode,
    points: np.ndarray,
    point_ids: np.ndarray,
    target_resolution: float,
) -> None:
    """
    Recursively refine a quadtree node until its cell size
    is less than or equal to the requested resolution.

    The requested resolution is treated as an upper bound
    on the final leaf cell size.
    """

    if not isinstance(node, QuadTreeNode):
        raise ValueError(
            "node must be a QuadTreeNode."
        )

    if target_resolution <= 0:
        raise ValueError(
            "target_resolution must be greater than zero."
        )

    points = np.asarray(points)

    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(
            "points must have shape (N, 2) or greater."
        )

    point_ids = np.asarray(
        point_ids,
        dtype=np.int64,
    )

    if (
        point_ids.ndim != 1
        or len(point_ids) != len(points)
    ):
        raise ValueError(
            "point_ids must have shape (N,)."
        )

    if node.cell_size <= target_resolution:
        return

    if node.point_count == 0:
        return

    before_count = node.point_count

    node.subdivide(
        points,
        point_ids=point_ids,
    )

    after_count = sum(
        child.point_count
        for child in node.children
    )

    if after_count != before_count:
        raise RuntimeError(
            "Resolution refinement changed the point count."
        )

    for child in node.children:
        refine_node_to_resolution(
            node=child,
            points=points,
            point_ids=point_ids,
            target_resolution=target_resolution,
        )