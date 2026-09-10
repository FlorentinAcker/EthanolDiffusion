#%%
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import math

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

    x0: (batch, dim). Returns: scalar loss tensor.
    """
    sigma = sample_sigma(x0.shape[0])              # (batch,)
    eps = torch.randn_like(x0)                      # (batch, dim)

    if sigma.dim() == 1:
        sigma = sigma.unsqueeze(-1)                 # (batch, 1)
    x = x0 + sigma * eps                             # (batch, dim)

    s = score(x, sigma, model)                       # (batch, dim)
    target = -eps / sigma                             # (batch, dim)

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

def train(dataset: torch.Tensor, dim: int, seed: int = 0, log_every: int = 500) -> ScoreMLP:
    """
    Main training loop, following the paper's protocol (App. H.1.1):
    train on mini-batches resampled from a FIXED dataset, with
    warmup+cosine lr and gradient clipping.

    dataset: (n_data, dim) tensor of clean points, already prepared by
      the caller (e.g. the notebook, via data.sample + data.embed for
      the torus, or an equivalent pipeline for another dataset).
    dim: dimension of the ambient space (must match dataset.shape[-1]).
    """
    assert dataset.shape[-1] == dim, 
    torch.manual_seed(seed)
    n_data = dataset.shape[0]

    model = ScoreMLP(dim=dim)
    optimizer = Adam(model.parameters(), lr=LR_MAX)
    scheduler = make_lr_scheduler(optimizer)

    for step in range(N_ITERS):
        idx = torch.randint(0, n_data, (BATCH_SIZE,))
        x0 = dataset[idx]

        loss = csm_loss(x0, model)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        scheduler.step()

        if step % log_every == 0:
            print(f"step {step}: loss={loss.item():.4f}")

    return model