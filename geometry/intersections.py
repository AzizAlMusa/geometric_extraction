# geometry/intersections.py
import numpy as np
from primitives.utils import normalize

def plane_plane_line(n1, d1, n2, d2):
    u = np.cross(n1, n2)
    if np.dot(u, u) < 1e-14: return None
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
        if abs(det) < 1e-14: continue
        t, s = np.linalg.solve(M, v0 - p0)
        if -1e-10 <= s <= 1.0 + 1e-10:
            hits.append(t)
    if len(hits) < 2: return None
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
    if abs(den) < 1e-14: return None
    sc = (B * E - C * D) / den
    tc = (A * E - B * D) / den
    p1 = a1 + sc * u
    p2 = b1 + tc * v
    if np.linalg.norm(p1 - p2) > tol: return None
    if not (0.0 <= sc <= 1.0 and 0.0 <= tc <= 1.0): return None
    return 0.5 * (p1 + p2)
