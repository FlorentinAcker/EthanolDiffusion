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

        