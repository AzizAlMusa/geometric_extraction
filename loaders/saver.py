# saver.py
# io/saver.py
"""
File saving utilities for edges, vertices, or meshes.
"""

import open3d as o3d
import os

def save_point_cloud(pcd, filename):
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Saved point cloud to {filename}")

def save_mesh(mesh, filename):
    o3d.io.write_triangle_mesh(filename, mesh)
    print(f"Saved mesh to {filename}")
