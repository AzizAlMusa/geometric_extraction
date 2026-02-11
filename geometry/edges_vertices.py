# geometry/edges_vertices.py
import numpy as np
import itertools
import open3d as o3d

from geometry.intersections import plane_plane_line, clip_line_t, segment_intersection_3d
from visualization.colors import PALETTE

from shapely.geometry import Polygon, LineString
from shapely.ops import triangulate, unary_union, polygonize_full


# ------------------------- utils -------------------------
def _dedup_points(pts, eps):
    out, seen = [], set()
    inv = 1.0 / eps
    for p in pts:
        k = tuple(np.round(np.asarray(p) * inv).astype(int))
        if k not in seen:
            seen.add(k)
            out.append(np.asarray(p))
    return np.array(out) if out else np.empty((0, 3))


# ----------------- segments & vertices -------------------
def build_segments_vertices(planes, diag):
    """Find clipped plane–plane intersection segments and deduplicated vertices."""
    vtx_tol   = 1e-4 * diag
    dedup_eps = 0.005 * diag
    sphere_r  = 0.012 * diag

    segments = []
    for A, B in itertools.combinations(planes, 2):
        ll = plane_plane_line(A.n, A.d, B.n, B.d)
        if not ll:
            continue
        p0, u = ll

        # clip by A
        pA = ((p0 - A.o) @ A.R)[:2]
        uA = (u @ A.R)[:2]
        if np.linalg.norm(uA) < 1e-12:  # parallel to A
            continue
        uA /= np.linalg.norm(uA)
        tA = clip_line_t(pA, uA, A.hull_uv)
        if not tA:
            continue

        # clip by B
        pB = ((p0 - B.o) @ B.R)[:2]
        uB = (u @ B.R)[:2]
        if np.linalg.norm(uB) < 1e-12:  # parallel to B
            continue
        uB /= np.linalg.norm(uB)
        tB = clip_line_t(pB, uB, B.hull_uv)
        if not tB:
            continue

        t0, t1 = max(tA[0], tB[0]), min(tA[1], tB[1])
        if t0 >= t1:
            continue
        a, b = p0 + t0 * u, p0 + t1 * u
        if np.linalg.norm(a - b) < 1e-10:
            continue
        segments.append((a, b))

    # vertices = segment–segment intersections
    raw_v = []
    for (a1, a2), (b1, b2) in itertools.combinations(segments, 2):
        p = segment_intersection_3d(np.asarray(a1), np.asarray(a2),
                                    np.asarray(b1), np.asarray(b2),
                                    vtx_tol)
        if p is not None:
            raw_v.append(p)

    V = _dedup_points(raw_v, dedup_eps)
    return segments, V, vtx_tol, sphere_r


def build_edges_from_segments_vertices(segments, V, vtx_tol):
    """Project vertices onto each segment, sort by t, connect consecutive pairs."""
    edges_set = set()
    if V is None or len(V) == 0 or not segments:
        return []

    Varr = np.asarray(V)
    for a, b in segments:
        a = np.asarray(a); b = np.asarray(b)
        ab = b - a
        L2 = float(np.dot(ab, ab))
        if L2 < 1e-18:
            continue

        ts = []
        for i, v in enumerate(Varr):
            t = np.dot(v - a, ab) / L2
            if t < -1e-9 or t > 1 + 1e-9:
                continue
            closest = a + t * ab
            if np.linalg.norm(v - closest) <= vtx_tol:
                ts.append((t, i))

        if len(ts) >= 2:
            ts.sort(key=lambda x: x[0])
            for (_, i0), (_, i1) in zip(ts[:-1], ts[1:]):
                if i0 == i1:
                    continue
                e = tuple(sorted((int(i0), int(i1))))
                edges_set.add(e)

    return sorted(list(edges_set))


# -------------- polygonization-based rebuild --------------
def _polygons_from_local_edges(uv_points, local_edges):
    """
    Build Shapely polygons (with holes) directly from linework using polygonize_full.
    This is robust to loop ordering and automatically preserves holes.
    """
    if len(local_edges) < 3 or len(uv_points) < 3:
        return []

    lines = [LineString([tuple(uv_points[i]), tuple(uv_points[j])]) for i, j in local_edges]
    merged = unary_union(lines)
    polys, _, _, _ = polygonize_full(merged)  # GeometryCollection of Polygons
    if polys.is_empty:
        return []

    # explode to list
    out = []
    if polys.geom_type == "Polygon":
        out = [polys]
    else:
        out = [g for g in polys.geoms if g.geom_type == "Polygon"]

    # filter out tiny slivers
    out = [p for p in out if p.area > 1e-12]
    return out


# ---------------- diagnostics + face reconstruction ----------------
def _extract_plane_loops_uv(pl, Varr, vtx_tol, edges):
    """
    For a single plane, return:
      uv: (M,2) UV coords for on-plane vertices
      local_edges: list of (i,j) within uv
      loops: list of loops, each is list of vertex indices in uv order
    Only loops whose vertices all have degree 2 are returned.
    """
    local_indices = [k for k, v in enumerate(Varr)
                     if abs(np.dot(pl.n, v) + pl.d) <= 2.0 * vtx_tol]
    if len(local_indices) < 3:
        return None, None, []

    local_points = Varr[local_indices]
    uv = ((local_points - pl.o) @ pl.R)[:, :2]

    idx_map = {g: l for l, g in enumerate(local_indices)}
    local_edges = [(idx_map[g1], idx_map[g2])
                   for (g1, g2) in edges if g1 in idx_map and g2 in idx_map]
    if len(local_edges) < 3:
        return uv, local_edges, []

    # adjacency list
    adj = {i: [] for i in range(len(uv))}
    for i, j in local_edges:
        if i == j:
            continue
        adj[i].append(j)
        adj[j].append(i)

    # find pure closed rings (degree == 2)
    ring_nodes = [i for i in range(len(uv)) if len(adj[i]) == 2]
    visited = set()
    loops = []

    for start in ring_nodes:
        if start in visited:
            continue
        loop = [start]
        visited.add(start)
        prev = None
        curr = start
        ok = True
        for _ in range(len(uv) + 5):
            nbrs = [n for n in adj[curr] if n != prev]
            if not nbrs:
                ok = False
                break
            nxt = nbrs[0]
            if nxt == start:
                loop.append(nxt)
                break
            loop.append(nxt)
            visited.add(nxt)
            prev, curr = curr, nxt
        if ok and len(loop) > 3 and loop[0] == loop[-1]:
            loop = loop[:-1]
            loops.append(loop)

    return uv, local_edges, loops


def rebuild_clean_faces(planes, V, vtx_tol, edges=None):
    """
    Combine both:
      1) Slotted quads for planes that have an inner+outer rectangle (holes)
      2) Normal polygonization for all other planes
    """
    cleaned = []
    if V is None or len(V) == 0 or edges is None or len(edges) == 0:
        return cleaned

    Varr = np.asarray(V)

    for i, pl in enumerate(planes):
        uv, local_edges, loops = _extract_plane_loops_uv(pl, Varr, vtx_tol, edges)
        if uv is None or not loops:
            continue

        # ---------------------------------------------------------------
        # Detect if this plane has outer+inner rectangle loops
        # ---------------------------------------------------------------
        polys = [Polygon(uv[loop]) for loop in loops if len(loop) >= 3]
        polys = [p for p in polys if p.is_valid and p.area > 1e-10]
        if len(polys) >= 2:
            # find outer/inner
            areas = [p.area for p in polys]
            outer_idx = np.argmax(areas)
            inner_idx = np.argmin(areas)
            outer = np.array(polys[outer_idx].exterior.coords)[:-1]
            inner = np.array(polys[inner_idx].exterior.coords)[:-1]

            # confirm both are rectangles
            if len(outer) == 4 and len(inner) == 4:
                # consistent orientation
                if Polygon(outer).exterior.is_ccw != Polygon(inner).exterior.is_ccw:
                    inner = inner[::-1]

                # build 4 frame quads
                for k in range(4):
                    a0, a1 = outer[k], outer[(k + 1) % 4]
                    b0, b1 = inner[k], inner[(k + 1) % 4]
                    quad = np.array([a0, a1, b1, b0])
                    tris = [[0, 1, 2], [0, 2, 3]]
                    XYZ = pl.o + np.c_[quad, np.zeros(4)] @ pl.R.T
                    m = o3d.geometry.TriangleMesh(
                        vertices=o3d.utility.Vector3dVector(XYZ),
                        triangles=o3d.utility.Vector3iVector(np.array(tris)),
                    )
                    m.compute_triangle_normals()
                    m.paint_uniform_color(PALETTE[i % len(PALETTE)])
                    cleaned.append(m)

                # we already handled this plane, go to next
                continue

        # ---------------------------------------------------------------
        # fallback: normal polygonization for solid/simple planes
        # ---------------------------------------------------------------
        local_indices = [k for k, v in enumerate(Varr)
                         if abs(np.dot(pl.n, v) + pl.d) <= 2.0 * vtx_tol]
        if len(local_indices) < 3:
            continue

        local_points = Varr[local_indices]
        uv2 = ((local_points - pl.o) @ pl.R)[:, :2]
        idx_map = {g: l for l, g in enumerate(local_indices)}
        local_edges2 = [(idx_map[g1], idx_map[g2])
                        for (g1, g2) in edges if g1 in idx_map and g2 in idx_map]
        if len(local_edges2) < 3:
            continue

        lines = [LineString([tuple(uv2[i]), tuple(uv2[j])]) for i, j in local_edges2]
        merged = unary_union(lines)
        polyset, _, _, _ = polygonize_full(merged)
        if polyset.is_empty:
            continue

        polys2 = [polyset] if polyset.geom_type == "Polygon" else [
            g for g in polyset.geoms if g.geom_type == "Polygon"
        ]
        for poly in polys2:
            if poly.area < 1e-12:
                continue
            tris2d = triangulate(poly)
            for tri in tris2d:
                coords = np.array(tri.exterior.coords)[:-1]
                if len(coords) != 3:
                    continue
                XYZ = pl.o + np.c_[coords, np.zeros(len(coords))] @ pl.R.T
                tri_idx = np.array([[0, 1, 2]], dtype=int)
                m = o3d.geometry.TriangleMesh(
                    vertices=o3d.utility.Vector3dVector(XYZ),
                    triangles=o3d.utility.Vector3iVector(tri_idx),
                )
                m.compute_triangle_normals()
                m.paint_uniform_color(PALETTE[i % len(PALETTE)])
                cleaned.append(m)

    return cleaned





# ---------- Compatibility wrapper (matches your main.py import) ----------
def build_segments_vertices_edges(planes, diag):
    """
    Returns: segments, V, edges, vtx_tol, sphere_r
    """
    segments, V, vtx_tol, sphere_r = build_segments_vertices(planes, diag)
    edges = build_edges_from_segments_vertices(segments, V, vtx_tol)
    return segments, V, edges, vtx_tol, sphere_r
