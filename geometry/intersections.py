# geometry/intersections.py
import numpy as np
from primitives.utils import normalize


def plane_plane_line(n1, d1, n2, d2):
    u = np.cross(n1, n2)
    if np.dot(u, u) < 1e-14:
        return None
    nu2 = np.dot(u, u)
    p0 = (-d1 * np.cross(n2, u) - d2 * np.cross(u, n1)) / nu2
    return p0, normalize(u)


def clip_line_t(p0, u, poly_uv):
    hits = []
    N = len(poly_uv)
    for i in range(N):
        v0 = poly_uv[i]
        v1 = poly_uv[(i + 1) % N]
        e = v1 - v0
        M = np.column_stack([u, -e])
        det = np.linalg.det(M)
        if abs(det) < 1e-14:
            continue
        t, s = np.linalg.solve(M, v0 - p0)
        if -1e-10 <= s <= 1.0 + 1e-10:
            hits.append(t)
    if len(hits) < 2:
        return None
    hits.sort()
    return hits[0], hits[-1]


def segment_intersection_3d(a1, a2, b1, b2, tol):
    u = a2 - a1
    v = b2 - b1
    w = a1 - b1
    A = np.dot(u, u)
    B = np.dot(u, v)
    C = np.dot(v, v)
    D = np.dot(u, w)
    E = np.dot(v, w)
    den = A * C - B * B
    if abs(den) < 1e-14:
        return None
    sc = (B * E - C * D) / den
    tc = (A * E - B * D) / den
    p1 = a1 + sc * u
    p2 = b1 + tc * v
    if np.linalg.norm(p1 - p2) > tol:
        return None
    if not (0.0 <= sc <= 1.0 and 0.0 <= tc <= 1.0):
        return None
    return 0.5 * (p1 + p2)


def plane_cylinder_intersection(cyl, plane_n, plane_d, n_samples=180, tol=1e-8):
    """Approximate intersection curve between finite cylinder and plane.

    Parameters
    ----------
    cyl : object
        Any object with fields: center, axis, radius, t_min, t_max.
    plane_n, plane_d : ndarray, float
        Plane equation is dot(plane_n, x) + plane_d = 0.
    n_samples : int
        Number of azimuth samples around the cylinder.

    Returns
    -------
    np.ndarray shape (K,3)
        Ordered polyline points (possibly empty) on the finite cylinder.
    """
    axis = normalize(np.asarray(cyl.axis, float))
    n = normalize(np.asarray(plane_n, float))

    # orthonormal frame around cylinder axis
    h = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = normalize(h - np.dot(h, axis) * axis)
    v = np.cross(axis, u)

    c = np.asarray(cyl.center, float)
    ts = np.linspace(0.0, 2.0 * np.pi, max(16, int(n_samples)), endpoint=False)
    pts = []

    denom = np.dot(n, axis)
    for theta in ts:
        radial = cyl.radius * (np.cos(theta) * u + np.sin(theta) * v)

        # point on infinite cylinder line at fixed theta is c + radial + t*axis
        if abs(denom) < tol:
            # plane parallel to axis -> either no hit for this generator or full line;
            # we skip this degenerate path in this coarse helper.
            continue

        t = -(np.dot(n, c + radial) + plane_d) / denom
        if cyl.t_min - 1e-9 <= t <= cyl.t_max + 1e-9:
            pts.append(c + radial + t * axis)

    if not pts:
        return np.empty((0, 3))

    return np.asarray(pts)


def cylinder_cylinder_intersection(cyl_a, cyl_b, n_samples=180, tol=2e-3):
    """Approximate finite-cylinder intersection points by sampling cyl_a.

    This is a numeric helper (not closed-form): sample points on ``cyl_a`` and keep
    those that satisfy ``cyl_b`` radial and axial constraints.
    """
    axis_a = normalize(np.asarray(cyl_a.axis, float))
    h = np.array([1.0, 0.0, 0.0]) if abs(axis_a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    ua = normalize(h - np.dot(h, axis_a) * axis_a)
    va = np.cross(axis_a, ua)

    ta = np.linspace(cyl_a.t_min, cyl_a.t_max, 60)
    th = np.linspace(0.0, 2.0 * np.pi, max(32, int(n_samples // 2)), endpoint=False)

    c_a = np.asarray(cyl_a.center, float)
    c_b = np.asarray(cyl_b.center, float)
    axis_b = normalize(np.asarray(cyl_b.axis, float))

    hits = []
    for t in ta:
        center_ring = c_a + t * axis_a
        for theta in th:
            p = center_ring + cyl_a.radius * (np.cos(theta) * ua + np.sin(theta) * va)

            tb = np.dot(p - c_b, axis_b)
            if tb < cyl_b.t_min - tol or tb > cyl_b.t_max + tol:
                continue
            proj = c_b + tb * axis_b
            rb = np.linalg.norm(p - proj)
            if abs(rb - cyl_b.radius) <= tol:
                hits.append(p)

    if not hits:
        return np.empty((0, 3))

    # simple spatial dedup
    arr = np.asarray(hits)
    q = np.round(arr / max(tol, 1e-6)).astype(int)
    _, idx = np.unique(q, axis=0, return_index=True)
    return arr[np.sort(idx)]
