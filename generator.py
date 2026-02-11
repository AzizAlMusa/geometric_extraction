import numpy as np
import open3d as o3d

def generate_cube_with_square_hole(n_side=60, side=1.0, hole_half=0.2, noise=0.002):
    """
    Generates a cube with a square hole through its center (along Z-axis).
    Returns: (N,3) array of 3D surface points.
    """
    rng = np.random.default_rng(0)
    half = side / 2
    pts = []

    # -------------------------------------------------------
    # 1. Cube outer faces (remove central hole)
    # -------------------------------------------------------
    grid = np.linspace(-half, half, n_side)
    X, Y = np.meshgrid(grid, grid)

    def outside_hole(x, y):
        return (np.abs(x) > hole_half) | (np.abs(y) > hole_half)

    mask = outside_hole(X, Y)
    faces = [
        np.c_[X[mask], Y[mask], np.full_like(X[mask], half)],   # top
        np.c_[X[mask], Y[mask], np.full_like(X[mask], -half)]   # bottom
    ]

    # Side faces (X=±half)
    Z, Y = np.meshgrid(grid, grid)
    faces.append(np.c_[np.full(Z.shape, half).ravel(), Y.ravel(), Z.ravel()])   # +X
    faces.append(np.c_[np.full(Z.shape, -half).ravel(), Y.ravel(), Z.ravel()])  # -X

    # Side faces (Y=±half)
    Z, X = np.meshgrid(grid, grid)
    faces.append(np.c_[X.ravel(), np.full(Z.shape, half).ravel(), Z.ravel()])   # +Y
    faces.append(np.c_[X.ravel(), np.full(Z.shape, -half).ravel(), Z.ravel()])  # -Y

    pts = np.vstack(faces)

    # -------------------------------------------------------
    # 2. Inner square tunnel walls
    # -------------------------------------------------------
    z = np.linspace(-half, half, n_side)
    inner = np.linspace(-hole_half, hole_half, n_side)
    Z, T = np.meshgrid(z, inner)

    # Four inner walls (each Nx3)
    faces_hole = [
        np.c_[np.full(Z.shape,  hole_half).ravel(), T.ravel(), Z.ravel()],  # x = +hole_half
        np.c_[np.full(Z.shape, -hole_half).ravel(), T.ravel(), Z.ravel()],  # x = -hole_half
        np.c_[T.ravel(), np.full(Z.shape,  hole_half).ravel(), Z.ravel()],  # y = +hole_half
        np.c_[T.ravel(), np.full(Z.shape, -hole_half).ravel(), Z.ravel()]   # y = -hole_half
    ]

    pts = np.vstack([pts, *faces_hole])

    # -------------------------------------------------------
    # 3. Add small Gaussian noise
    # -------------------------------------------------------
    pts += rng.normal(0, noise, pts.shape)

    print(f"Generated {len(pts)} surface points for cube-with-square-hole.")
    return pts


# ============================================================
# Generate + visualize + save
# ============================================================
points = generate_cube_with_square_hole(n_side=80, side=1.0, hole_half=0.2)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

o3d.visualization.draw_geometries([pcd])
o3d.io.write_point_cloud("cube_with_square_hole.pcd", pcd)
print("Saved to cube_with_square_hole.pcd")
