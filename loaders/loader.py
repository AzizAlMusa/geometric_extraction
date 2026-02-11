# loaders/loader.py
import os
import numpy as np
import open3d as o3d

def load_point_cloud(path, n_samples=30000, force_gray=True):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb", ".pts"]:
        pcd = o3d.io.read_point_cloud(path)
    else:
        mesh = o3d.io.read_triangle_mesh(path)
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_uniformly(number_of_points=n_samples)

    if force_gray and len(pcd.points) > 0:
        gray = np.tile([0.5, 0.5, 0.5], (len(pcd.points), 1))
        pcd.colors = o3d.utility.Vector3dVector(gray)

    bbox = pcd.get_axis_aligned_bounding_box()
    diag = float(np.linalg.norm(bbox.get_extent()))
    return pcd, diag
