"""Cylinder fitting utilities."""

from dataclasses import dataclass
import numpy as np
import open3d as o3d

from visualization.colors import PALETTE
from primitives.utils import normalize


@dataclass
class CylinderPatch:
    center: np.ndarray
    axis: np.ndarray
    radius: float
    t_min: float
    t_max: float
    mesh: o3d.geometry.TriangleMesh


def _axis_rotation_from_z(target_axis):
    """Rotation matrix that aligns +Z with ``target_axis``."""
    z = np.array([0.0, 0.0, 1.0])
    a = normalize(np.asarray(target_axis, float))
    dot = np.clip(np.dot(z, a), -1.0, 1.0)

    if np.isclose(dot, 1.0):
        return np.eye(3)
    if np.isclose(dot, -1.0):
        return o3d.geometry.get_rotation_matrix_from_axis_angle(
            np.array([1.0, 0.0, 0.0]) * np.pi
        )

    axis = normalize(np.cross(z, a))
    angle = np.arccos(dot)
    return o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)


def _angle_coverage(points, center, axis, bins=24):
    """Fraction of azimuth bins occupied around the cylinder axis."""
    h = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = normalize(h - np.dot(h, axis) * axis)
    v = np.cross(axis, u)

    q = points - center
    x = q @ u
    y = q @ v
    ang = np.arctan2(y, x)

    edges = np.linspace(-np.pi, np.pi, bins + 1)
    hist, _ = np.histogram(ang, bins=edges)
    return float(np.count_nonzero(hist)) / float(bins)


def _fit_single_cylinder(points, diag, min_coverage=0.45):
    """Estimate one cylinder from a set of points using PCA + radial consistency."""
    if len(points) < 250:
        return None, None

    c = points.mean(axis=0)
    centered = points - c
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = normalize(vh[0])

    t = centered @ axis
    closest_axis = c + np.outer(t, axis)
    radial = np.linalg.norm(points - closest_axis, axis=1)

    r = float(np.median(radial))
    dist_thr = max(0.004 * diag, 1e-4)
    inliers = np.where(np.abs(radial - r) <= dist_thr)[0]

    if len(inliers) < max(250, int(0.03 * len(points))):
        return None, None

    P = points[inliers]
    c2 = P.mean(axis=0)
    centered2 = P - c2
    _, _, vh2 = np.linalg.svd(centered2, full_matrices=False)
    axis2 = normalize(vh2[0])

    t2 = (P - c2) @ axis2
    closest2 = c2 + np.outer(t2, axis2)
    radial2 = np.linalg.norm(P - closest2, axis=1)
    r2 = float(np.median(radial2))

    # quality gates
    radial_mad = float(np.median(np.abs(radial2 - r2)))
    h2 = float(np.max(t2) - np.min(t2))
    coverage = _angle_coverage(P, c2, axis2, bins=30)

    if radial_mad > 0.02 * diag:
        return None, None
    if coverage < min_coverage:
        return None, None
    if h2 < 0.12 * diag or r2 < 0.015 * diag:
        return None, None

    return {
        "center": c2,
        "axis": axis2,
        "radius": r2,
        "t_min": float(np.min(t2)),
        "t_max": float(np.max(t2)),
        "coverage": coverage,
    }, inliers


def fit_cylinders(pcd, max_cylinders=2):
    """Fit coarse cylindrical patches from a point cloud.

    Returns
    -------
    cylinders: list[CylinderPatch]
    meshes: list[o3d.geometry.TriangleMesh]
    """
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return [], []

    bbox = pcd.get_axis_aligned_bounding_box()
    diag = float(np.linalg.norm(bbox.get_extent()))

    cylinders = []
    remaining = points.copy()

    for i in range(max_cylinders):
        fit, inliers = _fit_single_cylinder(remaining, diag)
        if fit is None:
            break

        h = fit["t_max"] - fit["t_min"]
        mesh = o3d.geometry.TriangleMesh.create_cylinder(
            radius=fit["radius"],
            height=h,
            resolution=60,
            split=6,
        )
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(PALETTE[(i + 7) % len(PALETTE)].tolist())
        mesh.rotate(_axis_rotation_from_z(fit["axis"]), center=np.zeros(3))

        mid = fit["center"] + 0.5 * (fit["t_min"] + fit["t_max"]) * fit["axis"]
        mesh.translate(mid)

        cylinders.append(
            CylinderPatch(
                center=fit["center"],
                axis=fit["axis"],
                radius=fit["radius"],
                t_min=fit["t_min"],
                t_max=fit["t_max"],
                mesh=mesh,
            )
        )

        mask = np.ones(len(remaining), dtype=bool)
        mask[inliers] = False
        remaining = remaining[mask]
        if len(remaining) < 250:
            break

    return cylinders, [c.mesh for c in cylinders]
