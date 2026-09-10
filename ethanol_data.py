#%%
import numpy as np
import torch

from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation

# %%

def load_md17_ethanol(root: str = "./md17_data") -> tuple[np.ndarray, np.ndarray]:
    """
    Load MD17 ethanol trajectory via torch_geometric.

    Returns: (positions, atomic_numbers)
      positions: (n_frames, 9, 3) float64, Angstrom.
      atomic_numbers: (9,) int, shared across all frames.
    """
    from torch_geometric.datasets import MD17
    ds = MD17(root=root, name="ethanol")

    n = len(ds)
    positions = np.stack([ds[i].pos.numpy() for i in range(n)])   # (n, 9, 3)
    atomic_numbers = ds[0].z.numpy()                                # (9,)
    return positions, atomic_numbers

# %%

def kabsch_align(positions: np.ndarray, ref_idx: int = 0) -> np.ndarray:
    """
    Rigid alignment (translation + rotation) of every frame onto a single
    reference frame, via the Kabsch algorithm. This removes the E(3)
    symmetry (rotations/translations) that has nothing to do with the
    molecule's internal (conformational) degrees of freedom -- so the
    network only has to learn the true internal geometry, not the extra
    6 "fake" dimensions coming from rigid motion.

    positions: (n_frames, n_atoms, 3).
    Returns: aligned positions, same shape, centered at the origin,
      rotated to best match frame `ref_idx` (least-squares, via SVD).
    """
    positions = positions - positions.mean(axis=1, keepdims=True)   # center each frame
    ref = positions[ref_idx]                                         # (n_atoms, 3), already centered

    aligned = np.empty_like(positions)
    for i in range(positions.shape[0]):
        # scipy's Rotation.align_vectors solves exactly the Kabsch problem:
        # find R minimizing sum_i || R @ positions[i,j] - ref[j] ||^2
        rot, _ = Rotation.align_vectors(ref, positions[i])
        aligned[i] = rot.apply(positions[i])

    return aligned

# %%

class EthanolDataset(Dataset):
    """
    Fixed dataset of Kabsch-aligned ethanol conformations, flattened to
    (n_frames, 27) for use with ScoreMLP(dim=27) and train().
    """

    def __init__(self, root: str = "./md17_data", n_frames: int | None = None,
                 ref_idx: int = 0):
        positions, self.atomic_numbers = load_md17_ethanol(root)
        if n_frames is not None:
            positions = positions[:n_frames]
        aligned = kabsch_align(positions, ref_idx=ref_idx)
        self.points = torch.tensor(
            aligned.reshape(aligned.shape[0], -1), dtype=torch.float32
        )  # (n_frames, 27)

    def __len__(self) -> int:
        return self.points.shape[0]

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.points[i]
# %%
