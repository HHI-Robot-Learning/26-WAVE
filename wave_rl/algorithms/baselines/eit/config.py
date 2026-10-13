from dataclasses import dataclass, field

from wave_rl.algorithms.base import AlgorithmConfig


@dataclass
class EITConfig(AlgorithmConfig):
    """The Entity Interaction Transformer, trained end to end under a TD3 matched to ECRL's."""

    embed_dim: int = 128                 # entity embedding width
    eit_hidden: int = 256                # MLP hidden width inside every EIT block / head
    n_head: int = 4                      # attention heads
    n_blocks: int = 3                    # self-attention blocks (SA1 + (n_blocks-1) after CA)
    #: gradient updates per env step of all scenes, so the effective UTD is utd/num_scenes
    utd: int = 2
    tau: float = 0.005                   # soft (polyak) target-update rate
    #: action-magnitude regulariser in the actor loss: -Q.mean() + a_reg*|a|^2. Keeps the
    #: policy off the tanh rails so da/dtheta stays alive.
    a_reg_coef: float = 0.01
    learning_rate: float = 3e-4
    batch_size: int = 8192
    buffer_size: int = 4_000_000
    policy_freq: int = 2                 # delayed actor update period
    target_policy_noise: float = 0.2     # target-policy smoothing noise
    noise_clip: float = 0.5              # target-noise clip
    grad_clip: float = 1.0               # global grad-norm clip on critic and actor
    min_priority: float = 1.0            # LAP min priority
    alpha: float = 0.4                   # LAP priority exponent
    #: fixed value-clip bounds (the full return range), applied to the whole target
    value_clip: list[float] = field(default_factory=lambda: [-10.0, 30.0])
    ou_noise: float = 0.2                # stationary std of the OU exploration noise
    ou_theta: float = 0.15               # OU mean-reversion rate
    her_ratio: float = 0.5               # relabelled transitions per real transition
    obs_pos_scale: float = 1.0
    obs_vel_scale: float = 0.05
    obs_referr_scale: float = 20.0
    obs_clip: float = 10.0               # clip of the assembled network input
    warmup_steps: int = 25_000

    #: a frozen single-object skill (.torch, with its .yaml sidecar) to distil the actor from
    #: before TD3 starts; None trains from a cold start
    distill_teacher: str | None = None
    #: env steps of the teacher's rollout that seed the replay buffer
    distill_steps: int = 100_000
    #: behaviour-cloning gradient steps of the actor on that rollout
    distill_actor_steps: int = 50_000
    distill_actor_lr: float = 1e-3
    distill_batch_size: int = 512
