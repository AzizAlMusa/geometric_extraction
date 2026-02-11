# config.py
# utils/config.py
"""
Configuration constants and tolerances.
"""

class Config:
    dist_threshold = 0.01
    min_inliers = 400
    vtx_tol = 1e-4
    dedup_eps = 0.005
    sphere_radius_factor = 0.012
    inflate_factor = 1.5
