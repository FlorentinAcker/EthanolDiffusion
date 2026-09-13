#%%
import numpy as np
import torch

from models.scoreMLP import v

# %%

def sample_via_probability_flow_ode(n_samples: int, dim: int, model,
                                     sigma_max: float = 3.0, sigma_min: float = 1e-3,
                                     n_steps: int = 200) -> torch.Tensor:
    """
    Generates samples from a trained score by integrating the probability
    flow ODE, dx/dsigma = (x - v(x,sigma)) / sigma (Euler method), from
    pure Gaussian noise at sigma_max down to sigma_min.

    This follows directly from Tweedie's identity (score = -(x-v)/sigma^2)
    and the variance-exploding SDE's probability flow ODE -- it is a
    short, direct consequence of what v() already computes, not a
    separate piece of machinery. sigma_max/sigma_min should match the
    range train.py's sample_sigma used during training.

    Returns: (n_samples, dim) generated points.
    """
    sigmas = torch.exp(torch.linspace(np.log(sigma_max), np.log(sigma_min), n_steps))
    x = torch.randn(n_samples, dim) * sigma_max

    for i in range(n_steps - 1):
        sigma_i, sigma_next = sigmas[i], sigmas[i + 1]
        sigma_batch = torch.full((n_samples,), sigma_i.item())
        v_pred = v(x, sigma_batch, model)
        dx = (x - v_pred) / sigma_i * (sigma_next - sigma_i)
        x = x + dx

    return x

# %%

def _manifold_membership(query: np.ndarray, reference: np.ndarray, k: int) -> np.ndarray:
    """
    For each point in `query`, checks whether it falls within the
    k-NN-ball manifold estimated from `reference` (Kynkaanniemi et al.
    2019): a point q is "in" the reference manifold if some reference
    point r has q within r's own k-th-nearest-neighbor distance (among
    other reference points).

    Returns: (len(query),) boolean array.
    """
    from scipy.spatial import cKDTree

    tree_ref = cKDTree(reference)
    # k-th NN distance of each reference point to the REST of reference
    # (k+1 because the nearest neighbor of a reference point within its
    # own set is itself, at distance 0)
    ref_knn_dist, _ = tree_ref.query(reference, k=k + 1)
    radii = ref_knn_dist[:, -1]  # (n_reference,)

    dists_query_to_ref = torch.cdist(
        torch.tensor(query, dtype=torch.float32),
        torch.tensor(reference, dtype=torch.float32),
    ).numpy()  # (n_query, n_reference)

    within_radius = dists_query_to_ref <= radii[None, :]
    return within_radius.any(axis=1)

# %%

def precision_recall(real_points: np.ndarray, generated_points: np.ndarray,
                      k: int = 3) -> tuple:
    """
    Standard k-NN precision/recall for generative models (Sajjadi et al.
    2018; Kynkaanniemi et al. 2019). Both inputs are (n, dim) arrays in
    the SAME ambient space (no feature extractor needed here, unlike the
    image-domain version which uses e.g. Inception/DINO features -- our
    ambient R^d coordinates already are a meaningful feature space).

    Precision: fraction of generated_points that fall within the real
      data's estimated manifold -- "are the generated samples realistic?"
    Recall: fraction of real_points that fall within the generated data's
      estimated manifold -- "does generation cover the full real support?"

    Returns: (precision, recall), both in [0, 1].
    """
    precision = _manifold_membership(generated_points, real_points, k).mean()
    recall = _manifold_membership(real_points, generated_points, k).mean()
    return precision, recall