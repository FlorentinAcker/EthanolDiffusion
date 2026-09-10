#%%
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import math

from data import sample, embed
from models.diffusion import ScoreMLP, score

# %%
# Hyperparameters (Kharitenko et al., Appendix H.1.1)
SIGMA_MAX = 3.0
LR_MAX = 1e-3
LR_MIN = 5e-5
N_WARMUP = 500        # not in the paper; added as a standard safety net
N_ITERS = 10_000
BATCH_SIZE = 512
GRAD_CLIP_NORM = 1.0
DIM = 4

# NOTE ON SCORE BLOWUP: by construction, the score explodes like 1/sigma^2
# as sigma -> 0, because the data manifold has positive codimension in R^4.
# The sigma^2 weighting in the CSM loss (see csm_loss below) keeps the loss
# itself bounded even as sigma -> 0; gradient clipping is the remaining
# safety net for any residual instability. No epsilon lower bound on sigma
# is used here (unlike the paper's eps=1e-4) -- open question to check with
# the authors whether this matters in practice.
# NOTE ON MEMORIZATION: DSM's empirical-optimal solution is known to
# memorize training points, especially at small sigma. Worth checking
# post-training: compare basin_occupancy of model-generated samples to
# the exact basin_weights, to see whether the model generalizes across
# the mixture or collapses onto the training points.

# %%

def get_batch(batch_size: int, seed: int) -> torch.Tensor:
    """
    Draw a batch of clean points x_0 on the torus, embedded in R^4.
    """
    s = sample(batch_size, seed)
    points = embed(s)
    return torch.tensor(points, dtype=torch.float32)

# %%

def sample_sigma(batch_size: int) -> torch.Tensor:
    """
    Draw one sigma per point, uniformly on [0, SIGMA_MAX].
    """
    return torch.rand(batch_size) * SIGMA_MAX

# %%

def csm_loss(x0: torch.Tensor, model: ScoreMLP) -> torch.Tensor:
    """
    One Monte-Carlo estimate of the conditional score matching loss
    (paper's eq. 26), for a batch of clean points x0.

    x0: (batch, DIM). Returns: scalar loss tensor.
    """
    sigma = sample_sigma(x0.shape[0])              # (batch,)
    eps = torch.randn_like(x0)                      # (batch, DIM)

    if sigma.dim() == 1:
        sigma = sigma.unsqueeze(-1)                 # (batch, 1)
    x = x0 + sigma * eps                             # (batch, DIM)

    s = score(x, sigma, model)                       # (batch, DIM)
    target = -eps / sigma                             # (batch, DIM)

    l = (((s - target) ** 2).sum(dim=-1, keepdim=True) * sigma ** 2).mean()
    return l

# %%

def make_lr_scheduler(optimizer: Adam) -> LambdaLR:
    """
    Linear warmup over N_WARMUP steps, then cosine decay from LR_MAX to
    LR_MIN over the remaining (N_ITERS - N_WARMUP) steps.

    Convention: the optimizer's base lr is set to LR_MAX (in train()),
    and this scheduler returns a MULTIPLIER in [LR_MIN/LR_MAX, 1] applied
    to that base lr -- never an absolute lr value.
    """
    def lr_lambda(step: int) -> float:
        if step < N_WARMUP:
            return step / max(1, N_WARMUP)
        progress = (step - N_WARMUP) / max(1, N_ITERS - N_WARMUP)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return (LR_MIN + (LR_MAX - LR_MIN) * cosine) / LR_MAX

    return LambdaLR(optimizer, lr_lambda)
# %%
