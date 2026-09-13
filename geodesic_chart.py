"""
Riemann normal coordinates via ODE integration of the tangent field, built
from ANY source of jacobians (learned score's full_jacobian, or the exact
data.tangent_projector) -- data-agnostic like tangent_geometry.py.

Unlike everything else tried in this project (LTSA, tangential Delaunay,
discrete Ricci flow), this approach never builds a discrete triangulation
from a finite point cloud. It treats the tangent field as a genuine
continuous vector field, evaluable at ANY point (not just sample points),
and integrates it directly -- avoiding the patch-reconciliation problem
entirely (no two independent local triangulations to disagree with each
other, since there is no triangulation at all).
"""
import numpy as np
import torch

# %%

def local_tangent_basis(x: torch.Tensor, jacobian_fn, k: int,
                         prev_basis: torch.Tensor = None) -> torch.Tensor:
    """
    x: (1, d) point. jacobian_fn: callable, x -> (1, d, d) jacobian at x
      (e.g. lambda x: full_jacobian(x, sigma, model), or the exact
      tangent_projector wrapped the same way).
    k: intrinsic dimension.
    prev_basis: (d, k) basis from a previous nearby point, used to
      continuously align the new basis to it.

    IMPORTANT (bug found and fixed while testing): with (near-)degenerate
    tangent eigenvalues (exactly degenerate on the exact ground truth,
    since both tangent eigenvalues equal 1), eigh can return ANY
    orthonormal basis of the eigenspace at each point -- not just an
    arbitrary SIGN per eigenvector, but an arbitrary ROTATION within the
    plane. A naive per-column sign fix (dot product > 0) does not
    prevent this rotation from drifting between consecutive integration
    steps, which silently made the integrated direction swap between the
    two physical angles at a different step size in testing. Fixed by
    reusing the same orthogonal Procrustes alignment already used for
    global orientation synchronization in tangent_geometry.py, applied
    here locally (one step at a time) instead of globally.

    Returns: (d, k) orthonormal tangent basis at x, aligned to prev_basis
      if given (by full rotation, not just per-column sign).
    """
    jac = jacobian_fn(x)[0]                          # (d, d)
    sym = 0.5 * (jac + jac.transpose(-2, -1))
    eigvals, eigvecs = torch.linalg.eigh(sym)          # ascending
    B = eigvecs[:, -k:]                                 # (d, k)

    if prev_basis is not None:
        M = prev_basis.transpose(-2, -1) @ B
        U, _, Vh = torch.linalg.svd(M)
        R = U @ Vh
        B = B @ R.transpose(-2, -1)
    return B

# %%

def integrate_direction(x0: torch.Tensor, direction_index: int, jacobian_fn,
                         retract_fn, k: int, step_size: float, n_steps: int,
                         retract_every: int = 5) -> torch.Tensor:
    """
    Integrates the direction_index-th tangent direction (of k) starting
    from x0, via simple forward Euler, with periodic retraction back onto
    the (learned or exact) manifold to correct numerical drift.

    x0: (1, d). retract_fn: callable, x -> (1, d) retracted point (e.g.
      lambda x: v(x, sigma, model), or data.project wrapped the same way).

    Returns: (n_steps+1, d) tensor, the integrated path (including x0).
    """
    x = x0.clone()
    B_prev = local_tangent_basis(x, jacobian_fn, k)
    path = [x.clone()]

    for step in range(n_steps):
        B = local_tangent_basis(x, jacobian_fn, k, prev_basis=B_prev)
        direction = B[:, direction_index].unsqueeze(0)     # (1, d)
        x = x + step_size * direction

        if (step + 1) % retract_every == 0:
            x = retract_fn(x)

        B_prev = B
        path.append(x.clone())

    return torch.cat(path, dim=0)

# %%

def build_grid(x0: torch.Tensor, jacobian_fn, retract_fn, k: int,
                step_size: float, n1: int, n2: int,
                retract_every: int = 5) -> torch.Tensor:
    """
    Builds a 2D coordinate grid (Riemann-normal-coordinates style) by
    first integrating direction 0 from x0 (n1 steps), then, from EVERY
    point of that curve, integrating direction 1 (n2 steps).

    Returns: (n1+1, n2+1, d) tensor of grid positions.
    """
    curve0 = integrate_direction(x0, 0, jacobian_fn, retract_fn, k,
                                   step_size, n1, retract_every)   # (n1+1, d)

    rows = []
    for i in range(curve0.shape[0]):
        xi = curve0[i:i+1]
        curve1 = integrate_direction(xi, 1, jacobian_fn, retract_fn, k,
                                       step_size, n2, retract_every)  # (n2+1, d)
        rows.append(curve1)

    return torch.stack(rows, dim=0)   # (n1+1, n2+1, d)