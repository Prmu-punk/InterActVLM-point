"""
Compute geodesic distance matrix for SMPLX mesh (10475 vertices).
This only needs to be run once and saved for future use.

Usage:
    python compute_smplx_geodesic.py --smplx_path /path/to/SMPLX_NEUTRAL.npz --output smplx_geodesic_dist.npy
"""

import argparse
import numpy as np
from pathlib import Path


def compute_geodesic_matrix_scipy(vertices, faces):
    """
    Compute geodesic distance matrix using scipy's sparse graph shortest path.
    This is an approximation using mesh edge distances.

    Args:
        vertices: (N, 3) vertex positions
        faces: (F, 3) face indices

    Returns:
        dist_matrix: (N, N) geodesic distance matrix
    """
    from scipy.sparse import lil_matrix
    from scipy.sparse.csgraph import shortest_path

    n_verts = len(vertices)
    print(f"Computing geodesic matrix for {n_verts} vertices...")

    # Build adjacency matrix with edge lengths as weights
    adj = lil_matrix((n_verts, n_verts), dtype=np.float32)

    # Add edges from faces
    for face in faces:
        for i in range(3):
            v1, v2 = face[i], face[(i + 1) % 3]
            dist = np.linalg.norm(vertices[v1] - vertices[v2])
            adj[v1, v2] = dist
            adj[v2, v1] = dist

    adj = adj.tocsr()
    print("Computing shortest paths (this may take a while for 10475 vertices)...")

    # Compute all-pairs shortest path
    dist_matrix = shortest_path(adj, method='D', directed=False)

    return dist_matrix.astype(np.float32)


def compute_geodesic_matrix_igl(vertices, faces):
    """
    Compute exact geodesic distance matrix using libigl (if available).
    This is more accurate but requires igl to be installed.

    pip install libigl
    """
    try:
        import igl
    except ImportError:
        print("libigl not installed. Using scipy approximation instead.")
        print("For exact geodesic, install: pip install libigl")
        return None

    n_verts = len(vertices)
    print(f"Computing exact geodesic matrix for {n_verts} vertices using libigl...")

    # Compute geodesic distances from each vertex
    dist_matrix = np.zeros((n_verts, n_verts), dtype=np.float32)

    for i in range(n_verts):
        if i % 500 == 0:
            print(f"  Progress: {i}/{n_verts} ({100*i/n_verts:.1f}%)")

        # Compute geodesic distance from vertex i to all other vertices
        vs = np.array([i], dtype=np.int32)
        vt = np.arange(n_verts, dtype=np.int32)
        d = igl.exact_geodesic(vertices, faces, vs, vt)
        dist_matrix[i] = d

    return dist_matrix


def main():
    parser = argparse.ArgumentParser(description="Compute SMPLX geodesic distance matrix")
    parser.add_argument('--smplx_path', type=str,
                        default='/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/boran/InterActVLM-point/smpl_models/SMPLX_NEUTRAL.npz',
                        help='Path to SMPLX model file')
    parser.add_argument('--output', type=str,
                        default='/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/boran/InterActVLM-point/data/smplx_geodesic_dist.npy',
                        help='Output path for geodesic matrix')
    parser.add_argument('--method', type=str, choices=['scipy', 'igl'], default='scipy',
                        help='Method to compute geodesic: scipy (fast approximation) or igl (exact but slower)')
    args = parser.parse_args()

    # Load SMPLX model
    print(f"Loading SMPLX model from: {args.smplx_path}")
    smplx_data = np.load(args.smplx_path, allow_pickle=True)

    vertices = smplx_data['v_template']  # (10475, 3)
    faces = smplx_data['f']  # (F, 3)

    print(f"Vertices shape: {vertices.shape}")
    print(f"Faces shape: {faces.shape}")

    # Compute geodesic matrix
    if args.method == 'igl':
        dist_matrix = compute_geodesic_matrix_igl(vertices, faces)
        if dist_matrix is None:
            print("Falling back to scipy method...")
            dist_matrix = compute_geodesic_matrix_scipy(vertices, faces)
    else:
        dist_matrix = compute_geodesic_matrix_scipy(vertices, faces)

    # Verify
    print(f"\nGeodesic matrix shape: {dist_matrix.shape}")
    print(f"Min distance: {dist_matrix.min():.4f}")
    print(f"Max distance: {dist_matrix.max():.4f}")
    print(f"Mean distance: {dist_matrix.mean():.4f}")
    print(f"Diagonal (should be 0): {dist_matrix.diagonal()[:5]}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, dist_matrix)
    print(f"\nSaved geodesic matrix to: {args.output}")
    print(f"File size: {output_path.stat().st_size / 1e6:.1f} MB")


if __name__ == '__main__':
    main()
