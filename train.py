#%%
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import math

from models.scoreMLP import ScoreMLP, score

# %%
# Hyperparameters (Kharitenko et al., Appendix H.1.1)
SIGMA_MIN=1e-4
SIGMA_MAX = 3.0
LR_MAX = 1e-3
LR_MIN = 5e-5
N_WARMUP = 500        # not in the paper; added as a standard safety net
BATCH_SIZE = 512
GRAD_CLIP_NORM = 1.0

# %%

def sample_sigma(batch_size: int) -> torch.Tensor:
    """
    Draw one sigma per point, log-uniformly on [SIGMA_MIN, SIGMA_MAX].
    Log-uniform (rather than linear) sampling gives equal weight to each
    decade of sigma, so small sigma -- where the network otherwise sees
    very few training examples under linear sampling -- gets properly
    represented during training.
    """
    log_min = math.log(SIGMA_MIN)
    log_max = math.log(SIGMA_MAX)
    u = torch.rand(batch_size)
    log_sigma = log_min + u * (log_max - log_min)
    return torch.exp(log_sigma)

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

def make_lr_scheduler(optimizer: Adam, n_iters: int) -> LambdaLR:
    """
    Linear warmup over N_WARMUP steps, then cosine decay from LR_MAX to
    LR_MIN over the remaining (n_iters - N_WARMUP) steps.

    Convention: the optimizer's base lr is set to LR_MAX (in train()),
    and this scheduler returns a MULTIPLIER in [LR_MIN/LR_MAX, 1] applied
    to that base lr -- never an absolute lr value.

    n_iters: total number of optimizer steps over the whole training run
      (n_epochs * batches_per_epoch), computed by train() and passed in
      here so the cosine decay is correctly calibrated to the run length.
    """
    def lr_lambda(step: int) -> float:
        if step < N_WARMUP:
            return step / max(1, N_WARMUP)
        progress = (step - N_WARMUP) / max(1, n_iters - N_WARMUP)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return (LR_MIN + (LR_MAX - LR_MIN) * cosine) / LR_MAX

    return LambdaLR(optimizer, lr_lambda)

# %%

def train(loader: DataLoader, dim: int, n_epochs: int, seed: int = 0,
          log_every: int = 500) -> tuple[ScoreMLP, list[float]]:
    """
    Main training loop, following the paper's protocol (App. H.1.1):
    train with warmup+cosine lr and gradient clipping, iterating over a
    PyTorch DataLoader for n_epochs full passes over the dataset -- not
    tied to any particular in-memory tensor, so this stays usable when
    the dataset does not fit in memory (e.g. a future molecular dataset).

    loader: a DataLoader yielding batches of (batch, dim) clean points.
      Use shuffle=True for i.i.d.-like mini-batches across epochs.
    dim: dimension of the ambient space (must match the batches' last dim).
    n_epochs: number of full passes over the dataset. Total optimizer
      steps = n_epochs * len(loader), used to calibrate the lr schedule.

    Returns: (trained model, list of per-step loss values).
    """
    torch.manual_seed(seed)

    n_iters = n_epochs * len(loader)

    model = ScoreMLP(dim=dim)
    optimizer = Adam(model.parameters(), lr=LR_MAX)
    scheduler = make_lr_scheduler(optimizer, n_iters=n_iters)
    losses = []

    step = 0
    for epoch in range(n_epochs):
        for x0 in loader:
            assert x0.shape[-1] == dim, "batch's last dimension must match `dim`"

            loss = csm_loss(x0, model)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())

            if step % log_every == 0:
                print(f"epoch {epoch}, step {step}/{n_iters}: loss={loss.item():.4f}")

            step += 1

    return model, losses