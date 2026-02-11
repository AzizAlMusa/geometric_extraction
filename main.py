# main.py
# Four-stage orchestration

from loaders.loader import load_point_cloud
from primitives.plane_fitting import fit_planes
from primitives.cylinder_fitting import fit_cylinders
from geometry.edges_vertices import build_segments_vertices_edges, rebuild_clean_faces
from visualization.viewer import show_raw, show_planes, show_lines_vertices, show_clean
from geometry.filtering import filter_edges


FILE = "models/cube_with_square_hole.pcd"
N_SAMPLES = 30000
INFLATE = 1.5
MAX_PLANES = 15
MAX_CYLINDERS = 2

def main():
    # 1) Raw gray cloud
    pcd, diag = load_point_cloud(FILE, n_samples=N_SAMPLES, force_gray=True)
    show_raw(pcd, title="1) raw cloud")

    # 2) Fitted planes + cylinders (double-sided, palette)
    planes, patches, remaining = fit_planes(
        pcd,
        max_planes=MAX_PLANES,
        inflate=INFLATE,
        min_inlier_ratio_remaining=0.06,
        max_aspect_ratio=5.0,
        return_remaining=True,
    )
    cylinders, cyl_meshes = fit_cylinders(remaining, max_cylinders=MAX_CYLINDERS)
    show_planes(patches + cyl_meshes, title="2) fitted planes + cylinders")

    # 3) Intersections → segments + vertices (red lines + dark green verts only)
    segments, V, edges, vtx_tol, sphere_r = build_segments_vertices_edges(planes, diag)
    edges = filter_edges(edges, V, pcd, tol=0.02)

    show_lines_vertices(segments, V, sphere_r, title="3) lines + dark-green vertices")


    # 4) Clean planes + black edges + black vertices
    cleaned = rebuild_clean_faces(planes, V, vtx_tol, edges=edges)


    show_clean(cleaned if cleaned else patches, edges, V, sphere_r,
               title="4) clean planes + black edges + black vertices")

if __name__ == "__main__":
    main()
