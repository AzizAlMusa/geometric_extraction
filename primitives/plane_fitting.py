import numpy as np
import open3d as o3d
from dataclasses import dataclass
from visualization.colors import PALETTE
from primitives.utils import normalize, project_to_plane, plane_frame
from geometry.hulls import hull2d


@dataclass
class PlanePatch:
    n: np.ndarray
    d: float
    o: np.ndarray
    R: np.ndarray
    hull_uv: np.ndarray
    hull_xyz: np.ndarray
    mesh: o3d.geometry.TriangleMesh


def poly_mesh_double_sided(X):
    if len(X) < 3:
        return None
    tris = np.array([[0, i, i + 1] for i in range(1, len(X) - 1)], int)
    tris = np.vstack([tris, tris[:, [0, 2, 1]]])  # double-sided
    m = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(X),
        triangles=o3d.utility.Vector3iVector(tris)
    )
    m.compute_triangle_normals()
    return m


def plane_from_model(model):
    a, b, c, d = model
    n_raw = np.array([a, b, c], float)
    n = normalize(n_raw)
    d = d / max(np.linalg.norm(n_raw), 1e-12)
    return n, d


def fit_planes(
    pcd,
    max_planes=15,
    inflate=1.5,
    min_inlier_ratio_remaining=0.05,
    max_aspect_ratio=6.0,
    return_remaining=False,
):
    """Fit multiple planes with quality-gated acceptance.

    The additional acceptance gates are designed to reject spurious plane fits
    on curved structures (for example strips on a cylindrical wall).
    """
    planes = []
    remaining = pcd

    bbox = pcd.get_axis_aligned_bounding_box()
    diag = float(np.linalg.norm(bbox.get_extent()))
    dist_thr = 0.01 * diag
    min_inlier = max(400, int(0.003 * len(pcd.points)))
    Cscene = np.asarray(pcd.points).mean(axis=0) if len(pcd.points) else np.zeros(3)

    for i in range(max_planes):
        n_remaining = len(remaining.points)
        if n_remaining < 200:
            break

        model, inliers = remaining.segment_plane(
            distance_threshold=dist_thr, ransac_n=3, num_iterations=1000
        )
        inlier_count = len(inliers)
        if inlier_count < min_inlier:
            break

        # Stop when best available plane explains only a tiny part of remaining cloud.
        if inlier_count < min_inlier_ratio_remaining * n_remaining:
            break

        cloud = remaining.select_by_index(inliers)
        n, d = plane_from_model(model)

        o_tmp = project_to_plane(n, d, cloud.get_center())
        if np.dot(n, o_tmp - Cscene) < 0:
            n, d = -n, -d

        P = np.asarray(cloud.points)
        o, R = plane_frame(n, d, P)
        uv = (P - o) @ R[:, :2]

        bins = 100
        margin = 0.05
        u = uv[:, 0]
        v = uv[:, 1]

        hu, edges_u = np.histogram(u, bins=bins)
        hv, edges_v = np.histogram(v, bins=bins)

        def dense_range(hist, edges, frac=0.2):
            if np.all(hist == 0):
                return edges[0], edges[-1]
            thr = frac * hist.max()
            idx = np.where(hist > thr)[0]
            if len(idx) == 0:
                return edges[0], edges[-1]
            lo, hi = idx[0], idx[-1]
            return edges[lo], edges[hi + 1]

        umin, umax = dense_range(hu, edges_u, frac=0.6)
        vmin, vmax = dense_range(hv, edges_v, frac=0.6)

        du = umax - umin
        dv = vmax - vmin
        umin -= margin * du
        umax += margin * du
        vmin -= margin * dv
        vmax += margin * dv

        mask = (u >= umin) & (u <= umax) & (v >= vmin) & (v <= vmax)
        uv_trim = uv[mask]
        if len(uv_trim) < 30:
            remaining = remaining.select_by_index(inliers, invert=True)
            continue

        hull_uv = hull2d(uv_trim)
        if len(hull_uv) < 3:
            remaining = remaining.select_by_index(inliers, invert=True)
            continue

        # Reject long thin strips (common false planes on curved surfaces).
        span_u = max(float(uv_trim[:, 0].ptp()), 1e-12)
        span_v = max(float(uv_trim[:, 1].ptp()), 1e-12)
        aspect = max(span_u, span_v) / min(span_u, span_v)
        if aspect > max_aspect_ratio:
            remaining = remaining.select_by_index(inliers, invert=True)
            continue

        c2 = hull_uv.mean(0)
        hull_uv = c2 + inflate * (hull_uv - c2)

        UVZ = np.hstack([hull_uv, np.zeros((len(hull_uv), 1))])
        XYZ = o + UVZ @ R.T

        mesh = poly_mesh_double_sided(XYZ)
        if mesh is None:
            remaining = remaining.select_by_index(inliers, invert=True)
            continue
        mesh.paint_uniform_color(PALETTE[i % len(PALETTE)].tolist())

        planes.append(
            PlanePatch(n=n, d=d, o=o, R=R, hull_uv=hull_uv, hull_xyz=XYZ, mesh=mesh)
        )

        remaining = remaining.select_by_index(inliers, invert=True)

    patches = [p.mesh for p in planes]
    if return_remaining:
        return planes, patches, remaining
    return planes, patches
