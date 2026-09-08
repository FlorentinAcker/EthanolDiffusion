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

def embed(angles):
    th1, th2 = angles[:, 0], angles[:, 1]
    return np.column_stack([np.cos(th1), np.sin(th1), np.cos(th2), np.sin(th2)])

def unembed(x):
    th1 = np.arctan2(x[:, 1], x[:, 0])
    th2 = np.arctan2(x[:, 3], x[:, 2])
    return np.column_stack([th1, th2]) % (2 * np.pi)


# %%

def log_density_1d(theta, weights, kappa):
    z = np.log(weights) + vonmises.logpdf(theta[:, None], kappa, loc=MU)
    return logsumexp(z, axis=1)

# %%
def sample(n, seed):
    rng = np.random.default_rng(seed)

    idx1 = rng.choice(3, size=n, p=WEIGHTS1)
    theta1 = vonmises.rvs(KAPPA, loc=MU[idx1], size=n) % (2*np.pi)

    idx2 = rng.choice(3, size=n, p=WEIGHTS2)
    theta2 = vonmises.rvs(KAPPA, loc=MU[idx2], size=n) % (2*np.pi)

    return np.column_stack([theta1, theta2])
# %%
def basin_weights(kappa):
    def density(theta, weights):
        theta_arr = np.array([theta])
        return np.exp(log_density_1d(theta_arr, weights, kappa))
    w1 = np.array([quad(density, a, b, args=(WEIGHTS1,))[0] for a, b in BOUNDS])
    w2 = np.array([quad(density, a, b, args=(WEIGHTS2,))[0] for a, b in BOUNDS])

    return np.outer(w1, w2)