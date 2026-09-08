#%%
import numpy as np
from scipy.stats import vonmises
from scipy.special import logsumexp
from scipy.integrate import quad

#%%
MU = np.array([0., 2*np.pi/3, 4*np.pi/3])
KAPPA = 8.0
WEIGHTS1 = np.array([0.60, 0.30, 0.10])
WEIGHTS2 = np.array([0.50, 0.35, 0.15])
BOUNDS = [(-np.pi/3, np.pi/3), (np.pi/3, np.pi), (np.pi, 5*np.pi/3)]

#%%

def embed(angles: np.ndarray) -> np.ndarray:
    """(n, 2) angles -> (n, 4) points sur le tore plonge dans R^4."""
    th1, th2 = angles[:, 0], angles[:, 1]
    return np.column_stack([np.cos(th1), np.sin(th1), np.cos(th2), np.sin(th2)])

def unembed(x: np.ndarray) -> np.ndarray:
    """(n, 4) -> (n, 2) angles dans [0, 2pi), inverse a gauche de embed."""
    th1 = np.arctan2(x[:, 1], x[:, 0])
    th2 = np.arctan2(x[:, 3], x[:, 2])
    return np.column_stack([th1, th2]) % (2 * np.pi)


# %%

def log_density_1d(theta: np.ndarray, weights: np.ndarray, kappa: float) -> np.ndarray:
    """(n,) angles -> (n,) log-densite du melange de trois von Mises."""
    z = np.log(weights) + vonmises.logpdf(theta[:, None], kappa, loc=MU)
    return logsumexp(z, axis=1)

# %%
def sample(n: int, seed: int) -> np.ndarray:
    """Tirage exact de n points (n, 2) selon le melange separable."""
    rng = np.random.default_rng(seed)

    idx1 = rng.choice(3, size=n, p=WEIGHTS1)
    theta1 = vonmises.rvs(KAPPA, loc=MU[idx1], size=n) % (2*np.pi)

    idx2 = rng.choice(3, size=n, p=WEIGHTS2)
    theta2 = vonmises.rvs(KAPPA, loc=MU[idx2], size=n) % (2*np.pi)

    return np.column_stack([theta1, theta2])

# %%

def basin_index(theta: np.ndarray) -> np.ndarray:
    """(n,) angles -> (n,) indices de secteur dans {0, 1, 2}."""
    shifted = (theta + np.pi/3) % (2*np.pi)
    return (shifted // (2*np.pi/3)).astype(int)

# %%

def basin_weights(kappa: float, n_grid: int | None = None) -> np.ndarray:
    """Poids exacts (3, 3) des neuf bassins, par quadrature (n_grid inutilise ici)."""
    def density(theta: float, weights: np.ndarray) -> float:
        theta_arr = np.array([theta])
        return np.exp(log_density_1d(theta_arr, weights, kappa))[0]

    w1 = np.array([quad(density, a, b, args=(WEIGHTS1,))[0] for a, b in BOUNDS])
    w2 = np.array([quad(density, a, b, args=(WEIGHTS2,))[0] for a, b in BOUNDS])

    return np.outer(w1, w2)

# %%

def basin_occupancy(angles: np.ndarray) -> np.ndarray:
    """(n, 2) angles -> (3, 3) proportions empiriques par bassin."""
    idx1 = basin_index(angles[:, 0])
    idx2 = basin_index(angles[:, 1])
    flat = 3 * idx1 + idx2
    counts = np.bincount(flat, minlength=9).reshape(3, 3)
    return counts / len(angles)

# %%

def potential(angles: np.ndarray, kappa: float) -> np.ndarray:
    """(n, 2) angles -> (n,) energie potentielle U = -log p."""
    ld1 = log_density_1d(angles[:, 0], WEIGHTS1, kappa)
    ld2 = log_density_1d(angles[:, 1], WEIGHTS2, kappa)
    return -(ld1 + ld2)

# %%

def project(x: np.ndarray) -> np.ndarray:
    """(n, 4) -> (n, 4) point du tore le plus proche (normalisation par paire)."""
    n1 = np.linalg.norm(x[:, :2], axis=1, keepdims=True)
    n2 = np.linalg.norm(x[:, 2:], axis=1, keepdims=True)
    return np.column_stack([x[:, :2] / n1, x[:, 2:] / n2])

# %%

def tangent_projector(x: np.ndarray) -> np.ndarray:
    """(n, 4) -> (n, 4, 4) projecteur orthogonal sur le tangent au point projete."""
    xp = project(x)
    n1 = np.zeros((x.shape[0], 4))
    n1[:, :2] = xp[:, :2]
    n2 = np.zeros((x.shape[0], 4))
    n2[:, 2:] = xp[:, 2:]

    P = np.eye(4)[None, :, :] - np.einsum('ni,nj->nij', n1, n1) - np.einsum('ni,nj->nij', n2, n2)
    return P

# %%

def sample_tubular(n: int, tau: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """n points (n, 4) dans une gaine d'epaisseur tau (< 1) autour du tore, et leurs amplitudes (n,)."""
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