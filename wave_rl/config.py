from dataclasses import dataclass, field
from pathlib import Path

from wave_rl.td7.agent import TD7Config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@dataclass
class GoalConfig:
    """The objective. Every LENGTH the objective needs beyond the tolerances comes from the
    scene's own object geometry, so a scene that rescales its objects rescales the structure
    with it and nothing here restates a half-extent or a rest height."""

    type: str = "towers"                 # towers | shape | pyramid | wall
    eps_pos: float | None = None         # null -> the frozen skill's own goal_feat_eps
    eps_vel: float = 0.0                 # rest-speed tolerance; 0 -> position-only
    #: 'latch_freeze' — an object that reaches its goal latches done and (optionally) freezes,
    #: and the task is solved once all have EVER been reached.
    #: 'simultaneous'  — nothing latches: the task is solved only when every object is at its
    #: goal AT THE SAME INSTANT, so a later manipulation that disturbs an earlier object
    #: un-solves it.
    success_mode: str = "latch_freeze"
    success_dwell: int = 1               # consecutive all-satisfied ticks a terminal needs
    stack_dz: float = 0.16               # vertical rung spacing (m) between courses
    tower_height: int = 3                # towers: objects per pile; 0 -> one pile of N

    lattice_xy_frac: float = 0.45        # lateral acceptance radius, as a fraction of pitch
    wall_width: int = 4                  # 0 -> one wall of wall_height courses grows longer
    wall_height: int = 0                 # 0 -> one wall grows upward as N rises
    pyramid_base: int = 3                # pyramid: 0 -> one pyramid grows with N
    shape_pattern: str = "line"          # line | rect | square
    shape_cols: int = 3                  # rect: columns

    #: the statics predicate, in multiples of h: an object rests on the floor within
    #: `floor_tolerance_h` of the floor reference, or on a support one course down (its
    #: height off by under `support_level_dz_h`) that sits under its centre within
    #: `single_support_xy_h`, or that straddles it with another, each within
    #: `overlap_support_xy_h`.
    floor_tolerance_h: float = 0.5
    support_level_dz_h: float = 0.6
    single_support_xy_h: float = 1.2
    overlap_support_xy_h: float = 1.9


@dataclass
class TrainConfig:
    enabled: bool = False
    budget: int = 1_000_000              # learn() total env steps (across all parallel scenes)
    n_train_max: int = 3                 # max objects per scene during training
    num_scenes: int = 64                 # parallel training scenes
    n_configs: int = 2000                # rows drawn from the layout pool
    pool_dir: str = "data/pools"         # the layout pools, one file per scene/count/seed
    #: minimum centre distance [m] between any two resting objects in a drawn layout. 0 keeps
    #: the footprint-contact rule, under which two objects may rest a millimetre apart — and
    #: since a tower is anchored at its base object's drawn pose, two PILES may too. A scene
    #: that dense is adversarial in a way the object count has nothing to do with. The pool
    #: sampler backs the number off when a count cannot hold it, so a sweep's high counts stay
    #: measurable.
    min_separation: float = 0.0
    #: every Nth object is the one a pile is anchored at, and is drawn from the FAR third of
    #: the reach axis while the rest are drawn from the near two thirds -- so the arm fetches
    #: from in front of the piles instead of reaching out past one to bring a cube back
    #: (which is half of all carries when the whole box is uniform). Set it to the structure
    #: height. 0 disables the split, and it is dropped automatically at a count that cannot
    #: hold it.
    anchor_stride: int = 0
    time_limit: float = 10.0             # flat per-episode sim-seconds; used if time_per_object null
    #: if set, the per-episode budget scales with the scene's object count, giving low counts
    #: SHORT episodes (less actor-drift room) while high counts get enough time.
    time_per_object: float | None = None
    remove_on_completion: bool = True    # latch_freeze: pin completed objects at their goal
    save_path: str | None = None         # training WRITES it; an eval-only run READS it
    #: a finished head to START training from, so a run continues one that ran out of budget
    #: instead of beginning at identity. The checkpoint carries its own step count, so
    #: `budget` is the TOTAL: a head stopped at 786M continued under budget 1.5G trains
    #: another 786M. Adam's moments are not in the checkpoint and restart from zero.
    init_from: str | None = None
    seed: int = 1
    log_freq: int = 2_000_000            # env steps between `ep/*` points (also the block size)
    stats_window_per_count: int = 200    # finished episodes per count behind each `ep/*` point
    checkpoint_freq: int = 19_660_800    # env steps between checkpoint writes
    snapshot_every_steps: int = 25_000_000   # env steps between RETAINED checkpoints
    video_freq: int = 19_660_800         # env steps between montage renders (0 -> never)
    video: bool = False                  # capture training episodes, render out of process
    video_episodes: int = 3              # training scenes captured side by side
    video_stride: int = 4                # ticks between captured frames


@dataclass
class EvalConfig:
    """The final sweep: the ONLY held-out, greedy-policy measurement a run produces, so it is
    definitive by default — every count from 1 to 30. Counts a task cannot physically lay out
    are dropped by the sweep itself, and counts whose contact budget is starved are reported
    AS starved rather than silently producing a fake cliff."""

    enabled: bool = True                 # false -> the run ends after training, no sweep
    start: int = 1                       # object counts start..stop inclusive, by step
    stop: int = 30
    step: int = 1
    n_episodes: int = 100
    time_limit: float = 20.0             # flat per-episode budget; ignored if time_per_object set
    time_per_object: float | None = None
    keep_completed: bool = False         # true -> completed objects stay dynamic
    #: the layout pool's seed; null -> the training seed, i.e. the pool the head trained on
    configs_seed: int | None = None
    #: algorithm config keys the sweep changes from training, for an eval whose difference
    #: from training IS the experiment. (An eval-only replay otherwise takes the algorithm
    #: config from the checkpoint: a head replayed under a different projection, measure or
    #: decision clock scores confidently and wrongly.)
    algorithm_overrides: dict = field(default_factory=dict)
    seed: int = 0


@dataclass
class PlantConfig:
    """The simulated robot: contact physics, the control tick and the actuation contract the
    frozen skill was trained with. One file per robot under ``configs/plant``."""

    tau_step: float                      # control tick (s)
    tau_sim: float                       # physics substep (s)
    action_scale: float                  # reference change per unit action, per sqrt(s)
    friction: str                        # contact parameters of every `geom_contact` geom
    solref: str
    solimp: str
    impratio: float
    #: box-box contacts from mujoco's primitive collider instead of its convex CCD
    box_box_primitive: bool
    #: the per-world contact allocation is this many contacts per object
    contacts_per_object: int
    grip_binary: bool                    # the last action channel opens or closes the jaw
    grip_hysteresis: float               # Schmitt band of the jaw command; 0 = sign
    grip_closed_at_upper: bool           # the jaw's closed stop is its upper ctrl limit
    ctrl_antiwindup: bool                # clamp the reference to the joint limits
    grasp_ee_point: str | None           # body published as the end effector; None = carriage


@dataclass
class TaskConfig:
    """The fully-resolved config for one run: a ``configs/tasks/<task>.yaml`` file with its
    plant loaded from ``configs/plant/<plant>.yaml``, loaded onto this schema by OmegaConf (so
    values are type-checked and unknown keys fail at startup)."""

    scene: str                           # base scene under scenes/, multiplied per object count
    plant: PlantConfig
    algorithm: str = "wave"
    algorithm_cfg: dict = field(default_factory=dict)   # validated against the algorithm's own schema
    skill: str | None = None             # frozen skill .torch (sidecar: same stem, .yaml)
    run_name: str | None = None          # null -> auto-built
    wandb: bool = False

    goal: GoalConfig = field(default_factory=GoalConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


@dataclass
class SkillConfig:
    """A frozen skill's sidecar (``<skill>.yaml`` next to ``<skill>.torch``): what it was
    trained on and how it reads its observation."""

    embodiment: str                      # gantry_gripper | panda_gripper | panda_pusher
    scene: str                           # its single-object scene under scenes/skills/
    eps_pos: float                       # the goal tolerance it was trained to reach
    vel_scale: float                     # observation scale of the joint velocities
    referr_scale: float                  # observation scale of the servo error
    td7: TD7Config = field(default_factory=TD7Config)


@dataclass
class SkillRewardConfig:
    """The single-object task a skill is trained on: reach the goal pose of the object
    (its centre and corner markers, relative to `obj_base`) within `eps_pos`. The reward is
    1 at the goal plus potential-based shaping terms, each off at scale 0."""

    eps_pos: float                       # goal tolerance on the position-first distance
    ori_weight: float                    # weight of the orientation in that distance
    yaw_symmetry: int = 0                # k > 1: orientation compared modulo k-fold yaw
    pos_scale: float = 4.0               # shaping of the centre distance ...
    pos_reach: float = 1.0               # ... over this range
    ori_scale: float = 4.0               # shaping of the marker (orientation) distance ...
    ori_reach: float = 0.5               # ... over this range
    ori_gate: float | None = None        # ... only within this centre distance (null: always)
    combined_scale: float = 8.0          # shaping of the combined distance near the goal ...
    combined_reach: float = 0.2          # ... over this range
    reach_scale: float = 0.0             # potential pulling the hand to the object ...
    reach_len: float | None = None       # ... over this range (null: 10 eps_pos)
    pregrasp_scale: float = 0.0          # potential for closing the jaw at the object ...
    pregrasp_len: float | None = None    # ... over this range (null: 4 eps_pos)
    grasp: float = 0.0                   # per-tick bonus while the object is lifted and centred
    release: float = 0.0                 # penalty for dropping a lifted object short of the goal
    grasp_rest_z: float = 0.0            # object-centre z resting on its support
    grasp_lift_scale: float = 0.05       # lift above rest at which the grasp gate saturates
    grasp_reach_sigma: float = 0.06      # xy spread of the grasp gate around the hand
    bounds_margin: float = 0.5           # an episode ends this far outside the joint limits
    vel_scale: float = 0.05              # observation scale of the joint velocities
    referr_scale: float = 20.0           # observation scale of the servo error


@dataclass
class EmbeddingConfig:
    """The configuration embedding the interpolating curriculum blends starts and goals in:
    the world positions of `points`, the joint positions, and how close the object is to
    each of `supports`."""

    points: list[str] = field(default_factory=lambda: ["obj", "goal1", "goal2", "goal3",
                                                       "goal4"])
    scale_q: float = 0.2
    supports: list[str] = field(default_factory=list)
    scale_d: float = 0.1
    d_sigma: float = 0.1


@dataclass
class CurriculumConfig:
    """Where episodes start and what they aim at. `direct` pairs a random start with a random
    goal. `interpG` starts each episode at the pool configuration nearest to a blend of the
    two, so early episodes begin close to their goal and, by `T_curr`, the full task."""

    mode: str                            # direct | interpG
    goals: str                           # configuration pool (h5) the goals are drawn from
    starts: str | None = None            # pool the starts are drawn from (null: `goals`)
    T_curr: int = 0                      # env ticks over which interpG reaches the full task
    T_block: int = 5000                  # env ticks between curriculum updates
    nominal_time_limit: float = 30.0     # episodes last up to twice this (scaled by progress)
    interp_t_sigma: float = 0.25         # spread of the blend factor around the progress
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)


@dataclass
class SkillTrainConfig:
    """``configs/skills/<embodiment>.yaml``: training one single-object skill with TD7."""

    embodiment: str                      # gantry_gripper | panda_gripper | panda_pusher
    scene: str                           # single-object scene under scenes/skills/
    plant: PlantConfig
    reward: SkillRewardConfig
    curriculum: CurriculumConfig
    td7: TD7Config
    T_end: int                           # env ticks to train for
    num_scenes: int = 1024               # parallel worlds
    discount_half_life: float = 2.0      # seconds; sets TD7's discount from the control tick
    seed: int = 0
    init_from: str | None = None         # a skill .torch to warm-start from
    eval_every: int = 10_000             # env ticks between evaluations
    eval_scenes: int = 64                # worlds of the evaluation env
    eval_episodes: int = 1000            # episodes of the final evaluation
    eval_time_limit: float = 60.0        # seconds per evaluation episode
    log_every: int = 1000                # env ticks between log lines
    wandb: bool = False
