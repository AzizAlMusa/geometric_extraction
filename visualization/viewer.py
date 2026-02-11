# visualization/viewer.py
import numpy as np
import open3d as o3d
from visualization.colors import BLACK, DARK_GREEN


def show_raw(pcd, title="1) raw cloud"):
    o3d.visualization.draw_geometries([pcd], window_name=title)


def show_planes(mesh_list, title="2) fitted planes"):
    if mesh_list:
        o3d.visualization.draw_geometries(mesh_list, window_name=title)


def show_lines_vertices(segments, V, sphere_r, title="3) lines + dark-green vertices"):
    """Black thick edges + dark-green vertices (light background)."""
    geoms = []
    if segments:
        pts, lines = [], []
        for a, b in segments:
            i = len(pts)
            pts += [a, b]
            lines.append([i, i + 1])
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.asarray(pts)),
            lines=o3d.utility.Vector2iVector(np.asarray(lines, int))
        )
        ls.colors = o3d.utility.Vector3dVector(np.tile(BLACK, (len(lines), 1)))
        geoms.append(ls)

    for v in (V if V is not None else []):
        s = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_r)
        s.paint_uniform_color(DARK_GREEN)
        s.translate(v)
        geoms.append(s)

    if geoms:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title)
        for g in geoms:
            vis.add_geometry(g)
        opt = vis.get_render_option()
        opt.line_width = 4.0
        vis.run()
        vis.destroy_window()


def show_clean(mesh_list, edges, V, sphere_r,
               title="4) clean planes + black edges + black vertices"):
    """Palette faces + visible thick black edges + black vertices."""
    if not mesh_list:
        return

    geoms = list(mesh_list)

    # --- edges (black & thick) ---
    if edges and V is not None and len(V):
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(np.asarray(V))
        ls.lines  = o3d.utility.Vector2iVector(np.asarray(edges, int))
        ls.colors = o3d.utility.Vector3dVector(
            np.tile(np.array([0, 0, 0]), (len(edges), 1))
        )
        geoms.append(ls)

    # --- vertices (black spheres) ---
    if V is not None and len(V):
        for v in V:
            s = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_r)
            s.paint_uniform_color([0, 0, 0])
            s.translate(v)
            geoms.append(s)

    # --- draw everything in one call (ensures lines are on top) ---
    o3d.visualization.draw_geometries(
        geoms,
        window_name=title,
        width=1280,
        height=960,
        left=50,
        top=50,
        mesh_show_back_face=True,
        point_show_normal=False
    )

    # now set the line thickness globally
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title)
    for g in geoms:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.line_width = 8.0          # strong visible black edges
    opt.mesh_show_back_face = True
    vis.run()
    vis.destroy_window()

