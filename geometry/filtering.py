# geometry/filtering.py
import numpy as np
import open3d as o3d

def filter_edges(edges, V, point_cloud, tol=0.01):
    """
    Keep only edges that are supported by nearby points in the original cloud.

    Parameters
    ----------
    edges : list of (i, j)
        Edges built from segment intersections.
    V : (N,3) array
        Vertex coordinates.
    point_cloud : open3d.geometry.PointCloud
        The raw or downsampled cloud.
    tol : float
        Maximum distance (fraction of scene diagonal) allowed between edge and points.
    """
    if not edges:
        return edges

    pts = np.asarray(point_cloud.points)
    diag = np.linalg.norm(pts.ptp(axis=0))
    max_dist = tol * diag

    valid_edges = []
    for i, j in edges:
        a, b = V[i], V[j]
        ab = b - a
        L2 = np.dot(ab, ab)
        if L2 < 1e-14:
            continue
        # project all points to the edge and measure distance
        t = np.clip(((pts - a) @ ab) / L2, 0, 1)
        closest = a + np.outer(t, ab)
        dists = np.linalg.norm(pts - closest, axis=1)
        if np.any(dists < max_dist):
            valid_edges.append((i, j))

    return valid_edges
