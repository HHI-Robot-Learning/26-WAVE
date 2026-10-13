from pathlib import Path

import h5py
import numpy as np


def pool_name(scene: str, n_objects: int, seed: int, min_separation: float,
              anchor_stride: int) -> str:
    """The file a layout pool lives in. A pool is a set of resting configurations of the
    ``n_objects``-object version of ``scene`` — robot pose and every object's free joint, in
    qpos order — drawn with ``seed``; every episode's start and goal layout is a row of it."""
    name = f"{scene}_n{n_objects}_seed{seed}"
    if min_separation:
        name += f"_sep{min_separation:g}"
    if anchor_stride:
        name += f"_anchor{anchor_stride}"
    return name + ".h5"


def load_pool(pool_dir: str, scene: str, n_objects: int, seed: int, min_separation: float,
              anchor_stride: int, n_configs: int) -> tuple[np.ndarray, Path]:
    """The first ``n_configs`` rows of a pool, and its path."""
    path = Path(pool_dir) / pool_name(scene, n_objects, seed, min_separation, anchor_stride)
    if not path.exists():
        shipped = sorted(p.name for p in Path(pool_dir).glob(
            pool_name(scene, 0, seed, min_separation, anchor_stride).replace("_n0_", "_n*_")))
        raise SystemExit(
            f"no layout pool {path}. The pools of every run in the paper are a release asset "
            f"(python download.py pools); for this scene and seed they are: "
            f"{', '.join(shipped) or 'none'}. An evaluation draws its layouts from the pool "
            f"of the largest count it sweeps.")
    with h5py.File(path, "r") as f:
        q = f["q"][:n_configs]
    if len(q) < n_configs:
        raise SystemExit(f"layout pool {path} holds {len(q)} rows, fewer than the "
                         f"{n_configs} requested")
    print(f"-- layout pool {path} ({len(q)} rows)")
    return q, path
