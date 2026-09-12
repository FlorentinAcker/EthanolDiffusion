#%%
import numpy as np
from collections import deque, defaultdict

# %%

def build_face_adjacency(triangles: set) -> dict:
    """
    triangles: set of sorted 3-tuples (a, b, c), global vertex indices.

    Returns: dict mapping each edge (sorted 2-tuple) to the list of
    triangles (as 3-tuples) containing it -- used to find, for each
    triangle, its neighbors across each of its three edges.
    """
    edge_to_faces = defaultdict(list)
    for tri in triangles:
        a, b, c = tri
        for e in [(a, b), (b, c), (a, c)]:
            edge_to_faces[tuple(sorted(e))].append(tri)
    return edge_to_faces

# %%

def place_seed_face(tri: tuple, edge_lengths: dict) -> dict:
    """
    tri: (v0, v1, v2), a single triangle to place first.
    edge_lengths: dict mapping sorted (a, b) -> length l_ab.

    Places v0 at the origin, v1 on the positive x-axis at distance l01,
    and v2 using the law of cosines for the angle at v0 -- exactly the
    "flatten a seed face" step of the paper (section 4.1).

    Returns: dict mapping vertex index -> (x, y) position, for these 3
    vertices only.
    """
    v0, v1, v2 = tri
    l01 = edge_lengths[tuple(sorted((v0, v1)))]
    l02 = edge_lengths[tuple(sorted((v0, v2)))]
    l12 = edge_lengths[tuple(sorted((v1, v2)))]

    cos_theta0 = (l01**2 + l02**2 - l12**2) / (2 * l01 * l02)
    cos_theta0 = np.clip(cos_theta0, -1.0, 1.0)
    theta0 = np.arccos(cos_theta0)

    positions = {
        v0: np.array([0.0, 0.0]),
        v1: np.array([l01, 0.0]),
        v2: np.array([l02 * np.cos(theta0), l02 * np.sin(theta0)]),
    }
    return positions

# %%

def circle_intersections(c1: np.ndarray, r1: float, c2: np.ndarray, r2: float):
    """
    Intersection points of two circles in the plane: center c1, radius r1,
    and center c2, radius r2. Standard closed-form (two circles intersect
    in 0, 1, or 2 points).

    Returns: list of 0, 1, or 2 candidate (x, y) points. An empty list
    signals a numerical inconsistency (e.g. triangle inequality violated
    by accumulated propagation error, or by genuine curvature -- see
    module docstring).
    """
    d = np.linalg.norm(c2 - c1)
    if d > r1 + r2 or d < abs(r1 - r2) or d == 0:
        return []

    a = (r1**2 - r2**2 + d**2) / (2 * d)
    h_sq = r1**2 - a**2
    if h_sq < 0:
        return []
    h = np.sqrt(h_sq)

    mid = c1 + a * (c2 - c1) / d
    perp = np.array([-(c2 - c1)[1], (c2 - c1)[0]]) / d

    p1 = mid + h * perp
    p2 = mid - h * perp
    if h < 1e-12:
        return [p1]
    return [p1, p2]

# %%

def place_third_vertex(vi: int, vj: int, vk: int, positions: dict,
                        edge_lengths: dict) -> np.ndarray:
    """
    vi, vj already placed in `positions`; find the position of vk, the
    third vertex of a triangle, using the two known edge lengths l_ik and
    l_jk, via circle intersection (section 4.2 of the paper).

    Orientation test: among the (up to two) candidate points, keep the
    one giving (tau(vj)-tau(vi)) x (tau(vk)-tau(vi)) > 0 -- i.e. vi, vj,
    vk in counterclockwise order, kept consistent across the whole layout.

    Returns: (x, y) position for vk. Raises ValueError if the two circles
    do not intersect (numerical breakdown -- see module docstring on how
    this signals residual curvature when run on a raw, non-flat metric).
    """
    pi, pj = positions[vi], positions[vj]
    l_ik = edge_lengths[tuple(sorted((vi, vk)))]
    l_jk = edge_lengths[tuple(sorted((vj, vk)))]

    candidates = circle_intersections(pi, l_ik, pj, l_jk)
    if len(candidates) == 0:
        raise ValueError(
            f"circles for vertex {vk} (centers {vi},{vj}) do not intersect -- "
            "triangle inequality violated by accumulated/curved metric"
        )

    if len(candidates) == 1:
        return candidates[0]

    # orientation test: keep the CCW candidate
    for p in candidates:
        cross = (pj[0] - pi[0]) * (p[1] - pi[1]) - (pj[1] - pi[1]) * (p[0] - pi[0])
        if cross > 0:
            return p
    # fallback (should not happen if candidates are the two genuine roots)
    return candidates[0]

# %%

def unfold_triangulation(triangles: set, edge_lengths: dict, seed: int = 0):
    """
    Full layout pipeline (paper section 4.1-4.2): place a seed face, then
    propagate to all triangles reachable from it via shared edges,
    breadth-first, using place_third_vertex for each newly-reached vertex.

    triangles: set of sorted 3-tuples.
    edge_lengths: dict (sorted 2-tuple) -> length.
    seed: index into the (arbitrary, set-iteration-order) list of
      triangles, to pick the starting seed face -- exposed so different
      seeds can be tried (the paper notes different seed faces should
      agree up to a rigid motion, once the metric is truly flat).

    Returns: (positions, conflicts)
      positions: dict vertex_index -> (x, y), for every vertex reached.
      conflicts: list of (vertex, existing_position, new_position, gap)
        recording every time a vertex was reached a SECOND time via a
        different propagation path and the new position disagreed with
        the existing one by more than a small tolerance -- this is the
        direct empirical signal of residual curvature (see module
        docstring): on a truly flat metric this list should be empty
        (modulo genuine non-contractible loops, which are NOT flagged
        here -- see unfolding.py's companion period-detection code).
    """
    tri_list = list(triangles)
    seed_tri = tri_list[seed % len(tri_list)]

    edge_to_faces = build_face_adjacency(triangles)

    positions = place_seed_face(seed_tri, edge_lengths)
    visited_faces = {seed_tri}
    conflicts = []

    queue = deque([seed_tri])
    while queue:
        tri = queue.popleft()
        for e in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[0], tri[2])]:
            e_sorted = tuple(sorted(e))
            for neighbor in edge_to_faces[e_sorted]:
                if neighbor == tri or neighbor in visited_faces:
                    continue
                # the vertex of `neighbor` not on the shared edge
                vk = [v for v in neighbor if v not in e_sorted][0]
                vi, vj = e_sorted

                if vi not in positions or vj not in positions:
                    # shared edge not yet fully placed from this side;
                    # will be revisited when its own edge is processed
                    continue

                try:
                    new_pos = place_third_vertex(vi, vj, vk, positions, edge_lengths)
                except ValueError:
                    # circles fail to intersect: exactly the empirical signal
                    # of residual curvature described in the module docstring.
                    # Record it as a conflict (infinite gap) rather than
                    # crashing, so unfolding a raw (curved) metric can be
                    # fully characterized rather than stopping at the first
                    # failure.
                    conflicts.append((vk, None, None, np.inf))
                    visited_faces.add(neighbor)
                    continue

                if vk in positions:
                    gap = np.linalg.norm(new_pos - positions[vk])
                    if gap > 1e-4:
                        conflicts.append((vk, positions[vk].copy(), new_pos, gap))
                else:
                    positions[vk] = new_pos

                visited_faces.add(neighbor)
                queue.append(neighbor)

    return positions, conflicts

# %%

def edge_lengths_from_positions(x: np.ndarray, triangles: set) -> dict:
    """
    Convenience: build the edge_lengths dict directly from real ambient
    positions x (n, d) -- l_ij = ||x_i - x_j|| -- for the "raw metric"
    baseline test (no Ricci flow applied yet).
    """
    edge_lengths = {}
    for tri in triangles:
        a, b, c = tri
        for i, j in [(a, b), (b, c), (a, c)]:
            key = tuple(sorted((i, j)))
            if key not in edge_lengths:
                edge_lengths[key] = float(np.linalg.norm(x[i] - x[j]))
    return edge_lengths