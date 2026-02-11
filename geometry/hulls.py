# geometry/hulls.py
import numpy as np

def hull2d(points):
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[0] < 3: return pts
    p = np.unique(pts, axis=0)
    if len(p) < 3: return p
    p = p[np.lexsort((p[:, 1], p[:, 0]))]
    def cross(o, a, b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lo, up = [], []
    for q in p:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], q) <= 0: lo.pop()
        lo.append(tuple(q))
    for q in p[::-1]:
        while len(up) >= 2 and cross(up[-2], up[-1], q) <= 0: up.pop()
        up.append(tuple(q))
    return np.array(lo[:-1] + up[:-1], float)
