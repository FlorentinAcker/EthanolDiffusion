#%%
import numpy as np
from scipy.stats import vonmises
from scipy.special import logsumexp
from scipy.integrate import quad
from scipy.special import iv

#%%
MU = np.array([0., 2*np.pi/3, 4*np.pi/3])
KAPPA = 8.0
WEIGHTS1 = np.array([0.60, 0.30, 0.10])
WEIGHTS2 = np.array([0.50, 0.35, 0.15])
BOUNDS = [(-np.pi/3, np.pi/3), (np.pi/3, np.pi), (np.pi, 5*np.pi/3)]

#%%

def embed(angles: np.ndarray) -> np.ndarray:
    """(n, 2) angles -> (n, 4) points on the torus embedded in R^4."""
    th1, th2 = angles[:, 0], angles[:, 1]
    return np.column_stack([np.cos(th1), np.sin(th1), np.cos(th2), np.sin(th2)])

def unembed(x: np.ndarray) -> np.ndarray:
    """(n, 4) -> (n, 2) angles in [0, 2pi), left inverse of embed."""
    th1 = np.arctan2(x[:, 1], x[:, 0])
    th2 = np.arctan2(x[:, 3], x[:, 2])
    return np.column_stack([th1, th2]) % (2 * np.pi)


# %%

def log_density_1d(theta: np.ndarray, weights: np.ndarray, kappa: float) -> np.ndarray:
    """(n,) angles -> (n,) log-density of the three-von-Mises mixture."""
    z = np.log(weights) + vonmises.logpdf(theta[:, None], kappa, loc=MU)
    return logsumexp(z, axis=1)

# %%

def density_sigma_1d(theta: np.ndarray, weights: np.ndarray, kappa: float,
                      sigma: float, n_harmonics: int = 30) -> np.ndarray:
    """
    (n,) angles -> (n,) density of the mixture wrap-blurred by a Gaussian
    noise of scale sigma on the circle (sigma=0 reduces to the exact
    density), via truncated Fourier series.

    Convolution on the circle becomes a product of Fourier coefficients:
    mixture coefficient c_n = sum_j w_j * exp(i*n*mu_j) * I_n(kappa)/I_0(kappa),
    wrapped-Gaussian kernel coefficient g_n(sigma) = exp(-n^2 sigma^2 / 2),
    blurred density coefficients = c_n * g_n(sigma), reconstructed by an
    inverse (truncated) Fourier series.
    """
    n = np.arange(-n_harmonics, n_harmonics + 1)
    ratio = iv(np.abs(n), kappa) / iv(0, kappa)                      # I_|n|(kappa)/I_0(kappa)
    c_n = ratio * (weights[None, :] * np.exp(1j * np.outer(n, MU))).sum(axis=1)  # (2*n_harmonics+1,)
    g_n = np.exp(-0.5 * (n ** 2) * sigma ** 2)
    coeffs = c_n * g_n / (2 * np.pi)

    phase = np.exp(-1j * np.outer(theta, n))                          # (n_theta, 2*n_harmonics+1)
    return (phase @ coeffs).real

# %%
def sample(n: int, seed: int) -> np.ndarray:
    """Exact draw of n points (n, 2) from the separable mixture."""
    rng = np.random.default_rng(seed)

    idx1 = rng.choice(3, size=n, p=WEIGHTS1)
    theta1 = vonmises.rvs(KAPPA, loc=MU[idx1], size=n) % (2*np.pi)

    idx2 = rng.choice(3, size=n, p=WEIGHTS2)
    theta2 = vonmises.rvs(KAPPA, loc=MU[idx2], size=n) % (2*np.pi)

    return np.column_stack([theta1, theta2])

# %%

def basin_index(theta: np.ndarray) -> np.ndarray:
    """(n,) angles -> (n,) sector indices in {0, 1, 2}."""
    shifted = (theta + np.pi/3) % (2*np.pi)
    return (shifted // (2*np.pi/3)).astype(int)

# %%

def basin_weights(kappa: float, n_grid: int | None = None) -> np.ndarray:
    """Exact (3, 3) weights of the nine basins, by quadrature (n_grid unused here)."""
    def density(theta: float, weights: np.ndarray) -> float:
        theta_arr = np.array([theta])
        return np.exp(log_density_1d(theta_arr, weights, kappa))[0]

    w1 = np.array([quad(density, a, b, args=(WEIGHTS1,))[0] for a, b in BOUNDS])
    w2 = np.array([quad(density, a, b, args=(WEIGHTS2,))[0] for a, b in BOUNDS])

    return np.outer(w1, w2)

# %%

def basin_occupancy(angles: np.ndarray) -> np.ndarray:
    """(n, 2) angles -> (3, 3) empirical proportions per basin."""
    idx1 = basin_index(angles[:, 0])
    idx2 = basin_index(angles[:, 1])
    flat = 3 * idx1 + idx2
    counts = np.bincount(flat, minlength=9).reshape(3, 3)
    return counts / len(angles)

# %%

def potential(angles: np.ndarray, kappa: float) -> np.ndarray:
    """(n, 2) angles -> (n,) potential energy U = -log p."""
    ld1 = log_density_1d(angles[:, 0], WEIGHTS1, kappa)
    ld2 = log_density_1d(angles[:, 1], WEIGHTS2, kappa)
    return -(ld1 + ld2)

# %%

def project(x: np.ndarray) -> np.ndarray:
    """(n, 4) -> (n, 4) closest point on the torus (per-pair normalization)."""
    n1 = np.linalg.norm(x[:, :2], axis=1, keepdims=True)
    n2 = np.linalg.norm(x[:, 2:], axis=1, keepdims=True)
    return np.column_stack([x[:, :2] / n1, x[:, 2:] / n2])

# %%

def tangent_projector(x: np.ndarray) -> np.ndarray:
    """(n, 4) -> (n, 4, 4) orthogonal projector onto the tangent at the projected point."""
    xp = project(x)
    n1 = np.zeros((x.shape[0], 4))
    n1[:, :2] = xp[:, :2]
    n2 = np.zeros((x.shape[0], 4))
    n2[:, 2:] = xp[:, 2:]

    P = np.eye(4)[None, :, :] - np.einsum('ni,nj->nij', n1, n1) - np.einsum('ni,nj->nij', n2, n2)
    return P

# %%

def sample_tubular(n: int, tau: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """n points (n, 4) in a tubular shell of thickness tau (< 1) around the torus, and their amplitudes (n,)."""
    rng = np.random.default_rng(seed)

    angles = rng.uniform(0, 2*np.pi, size=(n, 2))
    x = embed(angles)

    n1 = np.zeros((n, 4)); n1[:, :2] = x[:, :2]
    n2 = np.zeros((n, 4)); n2[:, 2:] = x[:, 2:]

    g = rng.normal(size=(n, 2))
    g = g / np.linalg.norm(g, axis=1, keepdims=True)
    direction = g[:, 0:1] * n1 + g[:, 1:2] * n2

    amplitudes = rng.uniform(0, tau, size=n)
    x_tubular = x + amplitudes[:, None] * direction

    return x_tubular, amplitudes

# %%

import torch
from torch.utils.data import Dataset

class TorusDataset(Dataset):
    """Fixed dataset of n points on the torus, embedded in R^4, for use with a DataLoader."""

    def __init__(self, n: int, seed: int):
        angles = sample(n, seed)
        self.points = torch.tensor(embed(angles), dtype=torch.float32)

    def __len__(self) -> int:
        return self.points.shape[0]

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.points[i]