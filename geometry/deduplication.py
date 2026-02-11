# geometry/deduplication.py
import numpy as np

def dedup_points(pts, eps):
    out=[]; seen=set(); inv=1/eps
    for p in pts:
        k=tuple(np.round(np.asarray(p)*inv).astype(int))
        if k not in seen:
            seen.add(k); out.append(np.asarray(p))
    return np.array(out)
