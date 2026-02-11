# primitives/utils.py
import numpy as np

def normalize(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def project_to_plane(n, d, p):
    return p - (np.dot(n, p) + d) * n

def plane_frame(n, d, pts):
    o = project_to_plane(n, d, np.asarray(pts).mean(0))
    z = n
    h = np.array([1, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1, 0])
    x = normalize(h - np.dot(h, z) * z)
    y = np.cross(z, x)
    return o, np.column_stack([x, y, z])
