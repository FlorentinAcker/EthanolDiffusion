"""
Data-agnostic tangent-geometry toolkit: reads local tangent structure from
ANY source of estimated Jacobians (a trained score's full_jacobian, or the
exact data.tangent_projector, or anything else with the same shape), and
tests its coherence via discrete parallel transport and holonomy on small
triangles.

This module never imports models.scoreMLP or knows about sigma/training --
the caller (typically a notebook) computes the Jacobians and passes them in.
This lets the exact same code be applied to a learned score AND to the
exact analytical tangent projector, for direct comparison.
"""
import numpy as np
import torch
from scipy.spatial import Delaunay, cKDTree

# %%

def detect_k(jacobians: torch.Tensor, energy_threshold: float = 0.9) -> int:
    """
    jacobians: (N, d, d) tensor of estimated Jacobians (e.g. full_jacobian
      output, or data.tangent_projector), one per anchor point. Not
      assumed symmetric.

    For each anchor, symmetrize and diagonalize to get a local eigenvalue
    spectrum; find the smallest k_i capturing energy_threshold of the
    total spectral energy; return the mode of the k_i across anchors.
    """
    sym = 0.5 * (jacobians + jacobians.transpose(-2, -1))     # (N, d, d)
    eigvals, _ = torch.linalg.eigh(sym)                        # (N, d), ascending
    eigvals_desc = eigvals.flip(dims=[-1])                     # (N, d), descending

    total_energy = eigvals_desc.sum(dim=-1, keepdim=True)      # (N, 1)
    cumulative = torch.cumsum(eigvals_desc, dim=-1)             # (N, d)
    fraction = cumulative / total_energy                       # (N, d)

    # k_i = smallest k such that fraction[i, k-1] >= energy_threshold
    meets_threshold = fraction >= energy_threshold             # (N, d), bool
    k_per_anchor = meets_threshold.float().argmax(dim=-1) + 1  # (N,), 1-indexed

    values, counts = torch.unique(k_per_anchor, return_counts=True)
    mode_k = values[torch.argmax(counts)].item()
    return mode_k

# %%

def tangent_basis(jacobians: torch.Tensor, k: int) -> torch.Tensor:
    """
    jacobians: (N, d, d) tensor of estimated Jacobians, one per anchor.
    k: fixed intrinsic dimension, same for every anchor (per project
      hypothesis -- no locally-varying dimension).

    Symmetrizes each Jacobian and keeps the k eigenvectors of largest
    eigenvalue as the estimated tangent basis at each anchor.

    Returns: (N, d, k) tensor, one orthonormal basis per anchor.
    """
    sym = 0.5 * (jacobians + jacobians.transpose(-2, -1))     # (N, d, d)
    eigvals, eigvecs = torch.linalg.eigh(sym)                  # ascending
    return eigvecs[:, :, -k:]                                   # (N, d, k), top-k columns

# %%

def filtered_neighbors(anchors: torch.Tensor, bases: torch.Tensor,
                        epsilon: float = 0.1, n_presel: int = 40) -> list[torch.Tensor]:
    """
    anchors: (N, d) points. bases: (N, d, k) tangent bases (tangent_basis output).
    epsilon: max relative normal-component allowed for a candidate neighbor
      to be kept (see project discussion: filters out points that are
      close in raw Euclidean distance but poorly aligned with the local
      tangent plane, e.g. "the other side of the torus").
    n_presel: how many nearest neighbors (raw Euclidean) to consider
      before filtering.

    Returns: list of length N; entry i is a LongTensor of anchor indices
      (excluding i itself) that passed the normal-component filter.
    """
    anchors_np = anchors.detach().numpy()
    tree = cKDTree(anchors_np)
    N = anchors.shape[0]
    neighbor_lists = []

    for i in range(N):
        _, idx = tree.query(anchors_np[i], k=n_presel + 1)   # includes i itself
        idx = torch.tensor([j for j in idx if j != i], dtype=torch.long)

        diffs = anchors[idx] - anchors[i]                     # (m, d)
        B_i = bases[i]                                         # (d, k)
        tangent_part = diffs @ B_i @ B_i.transpose(-2, -1)      # (m, d), projection onto tangent
        normal_part = diffs - tangent_part                      # (m, d)

        total_norm = diffs.norm(dim=-1)
        normal_norm = normal_part.norm(dim=-1)
        relative_normal = normal_norm / total_norm.clamp_min(1e-12)

        keep = relative_normal < epsilon
        neighbor_lists.append(idx[keep])

    return neighbor_lists

# %%

def local_delaunay(anchors: torch.Tensor, bases: torch.Tensor,
                    neighbor_lists: list[torch.Tensor]) -> list[np.ndarray]:
    """
    anchors: (N, d). bases: (N, d, k). neighbor_lists: filtered_neighbors output.

    For each anchor i, projects its filtered neighbors onto the local
    tangent basis B_i (giving k-dimensional coordinates), then computes a
    k-dimensional Delaunay triangulation on them (tangential Delaunay
    complex, simplified -- see project discussion: no sliver removal or
    formal refinement, just the core projection + Delaunay idea).

    Returns: list of length N; entry i is the array of simplices (as
    returned by scipy.spatial.Delaunay.simplices), indexing INTO
    neighbor_lists[i] (not into the full anchor set).
    """
    N = anchors.shape[0]
    simplices_per_anchor = []

    for i in range(N):
        idx = neighbor_lists[i]
        if idx.numel() < bases.shape[-1] + 1:
            simplices_per_anchor.append(np.empty((0, bases.shape[-1] + 1), dtype=int))
            continue

        diffs = anchors[idx] - anchors[i]                # (m, d)
        B_i = bases[i]                                    # (d, k)
        local_coords = (diffs @ B_i).detach().numpy()      # (m, k)

        try:
            tri = Delaunay(local_coords)
            simplices_per_anchor.append(tri.simplices)
        except Exception:
            simplices_per_anchor.append(np.empty((0, bases.shape[-1] + 1), dtype=int))

    return simplices_per_anchor

# %%

def procrustes_rotation(B_i: torch.Tensor, B_j: torch.Tensor) -> torch.Tensor:
    """
    B_i, B_j: (d, k) tangent bases. Finds the k x k orthogonal transform
    O_ij (rotation OR reflection, det = +1 or -1) minimizing
    || B_i @ O_ij - B_j ||, via orthogonal Procrustes (Schonemann 1966):
    M = B_j^T @ B_i = U S V^T, O_ij = U @ V^T.

    NOTE: deliberately unconstrained (a reflection is a valid, sometimes
    necessary, answer here) -- see synchronize_orientation for why: the
    sign/orientation of eigh's eigenvectors is arbitrary per anchor, so
    some neighbor pairs genuinely need a reflection to align well. Do not
    force det=+1 here; that was tried and made things worse (see project
    discussion) because it silently distorts genuinely-mismatched pairs
    rather than fixing the real issue at its source.

    Returns: (k, k) orthogonal matrix, det = +1 or -1.
    """
    M = B_j.transpose(-2, -1) @ B_i                # (k, k)
    U, _, Vh = torch.linalg.svd(M)
    return U @ Vh

# %%

def synchronize_orientation(bases: torch.Tensor,
                             neighbor_lists: list[torch.Tensor]) -> torch.Tensor:
    """
    Resolves the orientation ambiguity of eigh's eigenvectors (see project
    discussion) via the spectral synchronization method of Singer & Wu
    ("Orientability and Diffusion Maps", 2011): builds a graph where each
    edge (i, j) carries z_ij = det(procrustes_rotation(B_i, B_j)) in
    {+1, -1}, then finds the assignment z_i in {+1, -1} per anchor that is
    most consistent with all edges at once (z_i * z_j = z_ij), via the top
    eigenvector of the normalized reflection matrix Z = D^-1 @ Z_raw.

    This is robust to noisy/wrong individual edges (unlike a greedy BFS
    propagation, which would irreversibly propagate a single bad edge);
    it also reveals non-orientability if no consistent solution exists
    (the eigenvector values would not cluster around +-1/sqrt(N)).

    Implementation note: Z_raw is built and solved as a SPARSE matrix
    (scipy.sparse + ARPACK via eigs, requesting only the top eigenvalue)
    rather than a dense torch matrix -- for N in the thousands, each
    anchor only has ~n_presel << N neighbors, so a dense N x N eig (cubic
    in N) is drastically more work than necessary; sparse eigs scales
    with the number of edges instead. See project discussion: dense
    version measured ~11s at N=2000, growing worse than linearly.

    bases: (N, d, k). neighbor_lists: filtered_neighbors output.

    Returns: (N, d, k) tensor, bases with column k flipped in sign for
    every anchor where synchronization disagrees with eigh's raw choice.
    """
    import scipy.sparse
    import scipy.sparse.linalg

    N = bases.shape[0]

    rows, cols, vals = [], [], []
    for i in range(N):
        idx = neighbor_lists[i]
        if idx.numel() == 0:
            continue
        B_i = bases[i:i+1].expand(idx.numel(), -1, -1)     # (m, d, k)
        B_j = bases[idx]                                     # (m, d, k)
        M = B_j.transpose(-2, -1) @ B_i                       # (m, k, k)
        U, _, Vh = torch.linalg.svd(M)
        O = U @ Vh
        dets = torch.linalg.det(O)                            # (m,)

        rows.extend([i] * idx.numel())
        cols.extend(idx.tolist())
        vals.extend(dets.tolist())

    Z_raw = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    degree = np.asarray(np.abs(Z_raw).sum(axis=1)).flatten()
    degree[degree == 0] = 1.0
    D_inv = scipy.sparse.diags(1.0 / degree)
    Z = D_inv @ Z_raw

    eigvals, eigvecs = scipy.sparse.linalg.eigs(Z, k=1, which='LR')
    v1 = eigvecs[:, 0].real

    z_hat = np.sign(v1)                              # (N,), each +1, -1, or 0
    z_hat[z_hat == 0] = 1.0

    flipped = bases.clone()
    flip_mask = torch.tensor(z_hat < 0)
    flipped[flip_mask, :, -1] *= -1                 # flip last basis column
    return flipped

# %%

def triangle_shape_quality(p_a: torch.Tensor, p_b: torch.Tensor, p_c: torch.Tensor) -> float:
    """
    p_a, p_b, p_c: (2,) projected 2D coordinates of a triangle's vertices.

    Returns a shape-quality score in (0, 1] -- normalized ratio of area
    to squared perimeter (max for an equilateral triangle). Near-zero for
    both near-zero-area AND long/thin ("sliver") triangles, which pure
    area filtering misses (see project discussion: a triangle can have
    non-negligible area yet still be nearly collinear, causing an
    ill-conditioned Procrustes alignment).
    """
    ab = (p_b - p_a).norm()
    bc = (p_c - p_b).norm()
    ca = (p_a - p_c).norm()
    perimeter = ab + bc + ca
    area = 0.5 * abs((p_b[0]-p_a[0])*(p_c[1]-p_a[1]) - (p_c[0]-p_a[0])*(p_b[1]-p_a[1]))
    # normalization constant so an equilateral triangle scores 1.0
    return (4 * np.sqrt(3) * area / perimeter.clamp_min(1e-12)**2).item()

# %%

def holonomy_error(R_ij: torch.Tensor, R_jk: torch.Tensor, R_ki: torch.Tensor) -> float:
    """
    Composes the three discrete parallel-transport rotations around a
    closed triangle i -> j -> k -> i, and returns the residual rotation
    angle (radians) -- zero would mean perfect holonomy (flat manifold,
    consistent tangent estimates).

    For k=2, the composed rotation is in SO(2); its angle is
    atan2(R[1,0], R[0,0]).
    """
    R_loop = R_ij @ R_jk @ R_ki
    if R_loop.shape[-1] == 2:
        angle = torch.atan2(R_loop[1, 0], R_loop[0, 0])
    else:
        # general k: angle from the trace via the closest rotation's eigenvalues
        trace = torch.trace(R_loop)
        cos_angle = ((trace - (R_loop.shape[-1] - 2)) / 2).clamp(-1.0, 1.0)
        angle = torch.arccos(cos_angle)
    return angle.abs().item()

# %%

def grassmann_distance(B1: torch.Tensor, B2: torch.Tensor) -> float:
    """
    B1, B2: (d, k) orthonormal bases. Distance between the SUBSPACES they
    span (not the specific vectors), via principal angles: singular
    values of B1^T @ B2 are cosines of the principal angles theta_l;
    distance = sqrt(sum theta_l^2).
    """
    M = B1.transpose(-2, -1) @ B2                  # (k, k)
    cos_angles = torch.linalg.svdvals(M).clamp(-1.0, 1.0)
    angles = torch.arccos(cos_angles)
    return angles.norm().item()

# %%

def delaunay_neighbors(anchors: torch.Tensor, bases: torch.Tensor,
                        neighbor_lists_raw: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    Alternative to using filtered_neighbors' k-NN graph directly: restricts
    each anchor's neighbor set to those anchors that are its actual
    Delaunay neighbors (share a triangle edge WITH IT), rather than just
    "one of the k nearest, epsilon-aligned" candidates. This gives a more
    geometrically principled notion of neighborhood (points naturally
    surrounding i, not just the closest ones which can all sit on the
    same side).

    NOTE (bug found and fixed during testing): local_delaunay triangulates
    the neighbor set alone, WITHOUT including i as a vertex -- so simply
    collecting "any point appearing in any triangle of i's neighbor set"
    picks up essentially everyone (a well-populated 2D Delaunay
    triangulation uses nearly all its input points as vertices somewhere),
    not specifically i's neighbors. Anchor i itself must be included in
    the point set being triangulated, and only triangles touching i's own
    (local index 0) vertex give i's true Delaunay neighbors.

    anchors: (N, d). bases: (N, d, k).
    neighbor_lists_raw: filtered_neighbors output (the candidate pool to
      triangulate; Delaunay only RESTRICTS this set, it cannot add points
      outside of it).

    Returns: list of length N, entry i = LongTensor of anchor indices that
      are Delaunay-adjacent to i (subset of neighbor_lists_raw[i]).
    """
    N = anchors.shape[0]
    delaunay_lists = []

    for i in range(N):
        idx = neighbor_lists_raw[i]
        if idx.numel() < bases.shape[-1] + 1:
            delaunay_lists.append(idx[:0])
            continue

        patch_idx = torch.cat([torch.tensor([i]), idx])   # i is LOCAL INDEX 0
        diffs = anchors[patch_idx] - anchors[i]
        local_coords = (diffs @ bases[i]).detach().numpy()

        try:
            tri = Delaunay(local_coords)
        except Exception:
            delaunay_lists.append(idx[:0])
            continue

        adjacent_local = set()
        for simplex in tri.simplices:
            if 0 in simplex:                       # touches i's own vertex
                adjacent_local.update(v for v in simplex if v != 0)

        # local index l (l >= 1) in patch_idx corresponds to idx[l-1]
        adjacent_global = idx[torch.tensor(sorted(adjacent_local), dtype=torch.long) - 1] \
            if adjacent_local else idx[:0]
        delaunay_lists.append(adjacent_global)

    return delaunay_lists

# %%

def build_alignment_matrix(anchors: torch.Tensor, bases: torch.Tensor,
                            neighbor_lists: list[torch.Tensor]):
    """
    LTSA (Local Tangent Space Alignment, Zhang & Zha 2004) global
    alignment matrix S, built by eliminating the best-fit local affine
    reconstruction (c_i, L_i) at each patch analytically, leaving a
    single quadratic form in the global coordinates y_i alone.

    KNOWN LIMITATION (see project discussion): LTSA embeds into flat
    R^k, with no mechanism to recognize or preserve periodicity -- on a
    genuinely periodic manifold like the torus, the expected failure
    mode is that the map "cuts" the manifold open rather than closing it
    into a loop. This function is used here as a deliberate baseline to
    document that failure empirically, not as the final answer (see
    circular coordinates / persistent cohomology as the intended fix).

    anchors: (N, d). bases: (N, d, k). neighbor_lists: filtered_neighbors
      output (entry i = neighbor indices of i, NOT including i itself).

    Returns: (N, N) sparse CSR matrix S.
    """
    import scipy.sparse

    N = anchors.shape[0]
    rows, cols, vals = [], [], []

    for i in range(N):
        idx = neighbor_lists[i]
        patch_idx = torch.cat([torch.tensor([i]), idx])   # include i itself
        m1 = patch_idx.numel()
        if m1 <= bases.shape[-1] + 1:
            continue  # not enough points to fit a local affine model

        theta = (anchors[patch_idx] - anchors[i]) @ bases[i]   # (m1, k)

        # Design matrix D = [ones, theta]: projecting onto its column space
        # must annihilate BOTH the constant AND the local coordinates.
        # NOTE: centering theta alone is NOT enough -- centered theta is
        # already orthogonal to `ones` (sums to zero by construction), so
        # projecting only onto span(theta_centered) leaves `ones` entirely
        # in the residual (untouched), rather than annihilating it. The
        # ones column must be included explicitly in D (bug found and
        # fixed during testing: S @ ones was far from zero before this).
        ones_col = torch.ones(m1, 1)
        D = torch.cat([ones_col, theta], dim=1)                # (m1, k+1)

        gram = D.transpose(-2, -1) @ D                           # (k+1, k+1)
        gram_reg = gram + 1e-8 * torch.eye(gram.shape[0])        # numerical safety
        proj = D @ torch.linalg.solve(gram_reg, D.transpose(-2, -1))
        W = torch.eye(m1) - proj                                  # (m1, m1)
        S_i = (W @ W.transpose(-2, -1)).numpy()                    # (m1, m1)

        patch_idx_np = patch_idx.numpy()
        # vectorized fill: all (a, b) pairs at once, instead of a python double loop
        rows.append(np.repeat(patch_idx_np, m1))
        cols.append(np.tile(patch_idx_np, m1))
        vals.append(S_i.ravel())

    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    S = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    return S

# %%

def global_coordinates(S, k: int) -> np.ndarray:
    """
    S: (N, N) sparse alignment matrix (build_alignment_matrix output).
    k: number of global coordinates to extract.

    LTSA's global coordinates are the eigenvectors of S associated with
    the k SMALLEST eigenvalues, EXCLUDING the trivial constant
    eigenvector (eigenvalue 0, corresponds to a global translation, not
    informative). We therefore request k+1 smallest eigenvalues and drop
    the first (smallest) one.

    Returns: (N, k) array of global coordinates, up to an unknown affine
    transform (rotation/scale/translation) relative to any true
    parametrization.
    """
    import scipy.sparse.linalg

    eigvals, eigvecs = scipy.sparse.linalg.eigsh(S, k=k + 1, which='SM')
    order = np.argsort(eigvals)
    eigvecs = eigvecs[:, order]
    return eigvecs[:, 1:k+1]   # drop the trivial constant eigenvector