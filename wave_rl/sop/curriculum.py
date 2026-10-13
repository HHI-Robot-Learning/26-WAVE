import h5py
import mujoco
import numpy as np
from scipy.spatial import KDTree

from wave_rl.config import CurriculumConfig, EmbeddingConfig


def load_configs(path: str) -> np.ndarray:
    """(n, nq) the configurations of a pool file: stable, physically valid scenes of the
    robot and its one object."""
    with h5py.File(path, "r") as f:
        return np.asarray(f["q"], dtype=np.float64)


def embed(model: mujoco.MjModel, q: np.ndarray, cfg: EmbeddingConfig) -> np.ndarray:
    """(n, d) the curriculum embedding of each configuration: the world positions of
    `cfg.points`, the joint positions (`scale_q`, without the last three quaternion entries),
    and per support a closeness in [0, 1] that is 1 in contact and 0 beyond `d_sigma`."""
    data = mujoco.MjData(model)
    points = [model.body(p).id for p in cfg.points]
    obj_geoms = _collision_geoms(model, "obj")
    supports = [_collision_geoms(model, s) for s in cfg.supports]
    out = []
    for row in q:
        data.qpos[:] = row
        mujoco.mj_kinematics(model, data)
        z = [data.xpos[points].reshape(-1), cfg.scale_q * row[:-3]]
        if supports:
            d = np.array([min(mujoco.mj_geomDistance(model, data, g, o, cfg.d_sigma, None)
                              for g in geoms for o in obj_geoms) for geoms in supports])
            z.append(cfg.scale_d * np.clip(1. - d / cfg.d_sigma, 0., 1.))
        out.append(np.concatenate(z))
    return np.stack(out)


def _collision_geoms(model: mujoco.MjModel, body: str) -> list[int]:
    b = model.body(body).id
    return [g for g in range(model.ngeom) if model.geom_bodyid[g] == b
            and (model.geom_contype[g] or model.geom_conaffinity[g])]


class Curriculum:
    """Start/goal pairs for the episodes of one training block. `direct` pairs a random start
    with a random goal. `interpG` blends the two in the embedding with a factor drawn around
    the training progress `alpha` and starts at the start-pool configuration nearest to the
    blend: early episodes begin next to their goal, at alpha = 1 the pairing is `direct`."""

    def __init__(self, cfg: CurriculumConfig, model: mujoco.MjModel, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.goals = load_configs(cfg.goals)
        self.shared = cfg.starts is None
        self.starts = self.goals if self.shared else load_configs(cfg.starts)
        print(f"-- curriculum {cfg.mode}: {self.goals.shape[0]} goal configurations, "
              + ("the same pool for starts" if self.shared
                 else f"{self.starts.shape[0]} start configurations"))
        if cfg.mode == "interpG":
            self.goals_z = embed(model, self.goals, cfg.embedding)
            self.starts_z = self.goals_z if self.shared else embed(model, self.starts,
                                                                   cfg.embedding)
            self.tree = KDTree(self.starts_z)

    def progress(self, t: int) -> float:
        """alpha in (0, 1]: the share of the curriculum reached by the end of the block
        starting at env tick `t`."""
        return min((t + self.cfg.T_block) / (self.cfg.T_curr + 1), 1.)

    def sample(self, n: int, alpha: float | None) -> tuple[np.ndarray, np.ndarray]:
        """(starts, goals), each (n, nq). `alpha` None draws the full task (`direct`)."""
        goal_i = self.rng.integers(self.goals.shape[0], size=n)
        if self.shared:
            start_i = self.rng.integers(self.goals.shape[0] - 1, size=n)
            start_i = start_i + (start_i >= goal_i)
        else:
            start_i = self.rng.integers(self.starts.shape[0], size=n)
        if self.cfg.mode == "interpG" and alpha is not None:
            t = np.clip(self.rng.normal(alpha, self.cfg.interp_t_sigma, size=n), 0., 1.)[:, None]
            blend = t * self.starts_z[start_i] + (1. - t) * self.goals_z[goal_i]
            _, nearest = self.tree.query(blend, k=2)
            start_i = nearest[:, 0]
            if self.shared:
                start_i = np.where(start_i == goal_i, nearest[:, 1], start_i)
        return self.starts[start_i], self.goals[goal_i]
