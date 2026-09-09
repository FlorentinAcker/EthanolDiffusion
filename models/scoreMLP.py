#%%
import torch
import torch.nn as nn
#%%
class ScoreMLP(nn.Module):
    def __init__(self, dim:int=4, hidden: int=128, n_layers: int=4):
        super().__init__()
        layers = [nn.Linear(dim + 1, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x:torch.Tensor, sigma: torch.Tensor):
        if sigma.dim() == 1:
            sigma = sigma.unsqueeze(-1)
        inp = torch.cat([x, sigma], dim=-1)
        return self.net(inp)

def score(x, sigma, model):
    s_tilde = model(x, sigma)
    if sigma.dim() == 1:
        sigma = sigma.unsqueeze(-1)
    s = s_tilde / sigma
    return s
# %%
def v(x, sigma, model):
    if sigma.dim() == 1:
        sigma = sigma.unsqueeze(-1)
    return sigma**2 * score(x, sigma, model) + x

def v_jvp(x, sigma, model, g):
    x = x.detach().requires_grad_(True)
    p = v(x, sigma, model)
    g = g.detach()
    y = (p * g).sum()
    grad = torch.autograd.grad(y, x)[0]
    return grad

def full_jacobian(x: torch.Tensor, sigma: torch.Tensor, model: ScoreMLP) -> torch.Tensor:
    d = x.shape[-1]
    rows = []
    for i in range(d):
        g = torch.zeros_like(x)
        g[:, i] = 1.0
        rows.append(v_jvp(x, sigma, model, g))
    return torch.stack(rows, dim=1)