import random

import numpy as np
import torch


def set_global_seeds(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (every device)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
