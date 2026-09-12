#%%
import torch
import torch.nn as nn

# %%

class ScoreMLP(nn.Module):
    """Network predicting the non-reparametrized score s_tilde(x, sigma).

    The true (reparametrized) score is s(x, sigma) = s_tilde(x, sigma) / sigma,
    computed by the separate `score` function below, not inside this module.
    """

    def __init__(self, dim: int = 4, hidden: int = 128, n_layers: int = 4):
        super().__init__()
        layers = [nn.Linear(dim + 1, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """(n, dim) x, (n,) or (n, 1) sigma -> (n, dim) raw score s_tilde."""
        if sigma.dim() == 1:
            sigma = sigma.unsqueeze(-1)
        inp = torch.cat([x, sigma], dim=-1)
        return self.net(inp)

# %%

def score(x: torch.Tensor, sigma: torch.Tensor, model: ScoreMLP) -> torch.Tensor:
    """Reparametrized score s(x, sigma) = model(x, sigma) / sigma. (n, dim) -> (n, dim)."""
    s_tilde = model(x, sigma)
    if sigma.dim() == 1:
        sigma = sigma.unsqueeze(-1)
    return s_tilde / sigma

# %%

def v(x: torch.Tensor, sigma: torch.Tensor, model: ScoreMLP) -> torch.Tensor:
    """v(x, sigma) = x + sigma^2 * score(x, sigma, model): learned approx. of pi(x)."""
    if sigma.dim() == 1:
        sigma = sigma.unsqueeze(-1)
    return sigma**2 * score(x, sigma, model) + x

# %%

def v_jvp(x: torch.Tensor, sigma: torch.Tensor, model: ScoreMLP, g: torch.Tensor) -> torch.Tensor:
    """Compute v'(x)^T g without forming the full Jacobian (paper's Remark 4).

    Single forward pass (keeping the graph) + single backward pass, via the
    scalar y = <v(x), g> with g treated as a constant. Returns v'(x)^T g,
    i.e. exactly v'(x) g only when v'(x) happens to be symmetric.

    x: (n, dim), sigma: (n,) or (n, 1), g: (n, dim) treated as a constant.
    Returns: (n, dim).
    """
    x = x.detach().requires_grad_(True)
    p = v(x, sigma, model)
    g = g.detach()
    y = (p * g).sum()
    grad = torch.autograd.grad(y, x)[0]
    return grad

# %%

def full_jacobian(x: torch.Tensor, sigma: torch.Tensor, model: ScoreMLP) -> torch.Tensor:
    """Full Jacobian v'(x) for a batch of n points (n can be 1 or more),
    built from v_jvp on the canonical basis.

    Note: v_jvp(x, sigma, model, e_i) returns the i-th column of v'(x)^T,
    which is also the i-th ROW of v'(x). Stacking these rows with dim=1
    therefore reconstructs v'(x) directly (checked numerically against
    torch.autograd.functional.jacobian). The loop below runs exactly
    `dim` times regardless of n, since v_jvp already processes the whole
    batch in a single call per direction -- there is no separate
    single-point vs. batched code path.

    x: (n, dim), sigma: (n,) or (n, 1). Returns: (n, dim, dim).
    """
    d = x.shape[-1]
    rows = []
    for i in range(d):
        g = torch.zeros_like(x)
        g[:, i] = 1.0
        rows.append(v_jvp(x, sigma, model, g))
    return torch.stack(rows, dim=1)