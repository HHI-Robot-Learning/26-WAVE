from dataclasses import dataclass, field

from wave_rl.algorithms.base import AlgorithmConfig


@dataclass
class PPOConfig:
    """The warper's PPO."""

    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    update_epochs: int = 4
    minibatch_size: int = 8192
    entropy_coef: float = 5e-4
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    #: the floor under the cosine that runs the rate down over the run's step budget
    learning_rate_final: float = 1e-4
    #: beta of a fixed penalty beta*KL(old || new) on the JOINT warp distribution,
    #: k3-estimated from the same samples as the surrogate. It works on top of the ratio clip,
    #: not instead of it.
    kl_penalty_coef: float = 0.05
    #: Advantages are normalised by max(remembered, batch) std rather than per-batch alone,
    #: to avoid unit-scale gradients during reward droughts; this is the decay applied to the
    #: remembered scale each iteration.
    advantage_scale_decay: float = 0.99


@dataclass
class WarpHeadConfig:
    """The warp head: its bin supports, its init posture and its transformer."""

    #: The multiplier rungs: graded damping steps between identity (1.0) and takeover (0.0),
    #: plus boost up to 2x — seven damping rungs, 25 bins.
    damping_multipliers: list[float] = field(default_factory=lambda: [
        2.0, 1.75, 1.5, 1.25, 1.0, 0.75, 0.5, 0.3, 0.18, 0.1, 0.06, 0.03, 0.0])
    #: The support symmetric in slope, with the geometric {1,2,4,8}*u offsets: the calibration
    #: reference. The decay calibration matches the shipped grid's expected init warp
    #: magnitude to this one, which holds the init posture fixed however fine and far the
    #: grid reaches.
    reference_slope_bins: list[float] = field(default_factory=lambda: [
        0.0, -0.25, 0.25, -0.5, 0.5, -1.0, 1.0, -1.5, 1.5, -2.0, 2.0])
    reference_max_quanta: int = 8
    #: Offset ceiling, in quanta, for the sparse ladder {1,2,4,8} then strides of 4; 32 = 10
    #: magnitudes / 21 bins, fitting under the 25-bin multiplier ladder.
    offset_max_quanta: int = 32
    #: u, the offset quantum of the input position family, in units of h (a length, so it
    #: rescales with object size).
    quantum_position_h: float = 0.1875
    quantum_rotation: float = 0.05
    quantum_velocity: float = 0.02
    #: Quantum of the action/grip offset ladder; dimensionless (act_t and grip live in the
    #: clamped [-1, 1] action space), not rescaled per embodiment.
    quantum_action: float = 0.015
    #: Initial probability mass on the identity (no-warp) bin of each parameter's distribution.
    p_zero_init: float = 0.8
    #: Initial mass on the "carry the previous applied value" bin, a slot per channel distinct
    #: from the identity, fed back as input to keep the process Markov.
    p_hold_init: float = 0.1
    #: Base of the geometric decay over bin magnitude, shaping both the exploration quota and
    #: the init bias; the exponent is per-channel rescaled to keep the expected init warp
    #: magnitude fixed as the grid is widened.
    magnitude_decay_base: float = 2.0
    #: Permanent per-channel exploration quota, exact in the log-prob (`mixture_probs`).
    epsilon_exploration: float = 0.05
    #: Bound on the per-bin logits before the softmax. At |logit| = 30 the softmax is already
    #: one-hot to ~1e-26, so clamping there costs nothing behaviourally for a healthy policy
    #: while preventing unbounded parameter growth and overflow.
    logit_clamp: float = 30.0
    #: the transformer: width, encoder and decoder depth, heads, feed-forward ratio
    d_model: int = 128
    n_layers: int = 2
    decoder_layers: int = 2
    n_heads: int = 4
    ff_mult: int = 4
    #: width of the per-pair score network. Small on purpose: it scores ONE pair, and the work
    #: of combining pairs is done by the min, not by capacity.
    pair_hidden: int = 64


@dataclass
class SelectorConfig:
    """The learned selector: its scorer, its option critic and their optimisation."""

    #: width and depth of the per-object scorer's trunk and of the option critic
    hidden: int = 64
    n_layers: int = 2
    #: K, the neighbour cap of the learned local context, per anchor (the object and its
    #: goal). With the compact-support radius below this bounds the context's domain at every
    #: N; a packing argument bounds the number of objects that can even be within R at
    #: (1 + 2R/delta)^2 = 49 for touching cubes, so K < that is a capacity choice. K must
    #: exceed the number of lattice neighbours that decide whether a slot can be filled: an
    #: interior slot of a 3-D lattice has 4 supports below, 4 same-course sides and 4
    #: dependents above.
    context_k: int = 12
    #: R of the compact-support neighbour kernel relu(1 - d/R), in units of h. A LOCAL length,
    #: never the arena: relu is exactly zero past R, so an object beyond R of every existing
    #: object and goal cannot change any existing score. A Gaussian tail would sum over all N
    #: and leak the count straight back in.
    context_radius_h: float = 6.0
    #: F, the bound on the learned priority |F tanh f|. Together with the travel cost it
    #: confines the competition for the argmax to a ball of radius 2F/alpha around the nearest
    #: feasible candidate: a learned preference reorders within a travel band and never
    #: inverts a large distance gap. 4 logits = 8 half-widths of travel at alpha.
    priority_bound: float = 4.0
    #: cap on log(1/sigma) inside the attempt cost — the largest number of attempts the rule
    #: will price (e^2 = 7.4), and the bound on how far a learned estimate may move a score.
    feasibility_floor: float = 2.0
    #: alpha, the travel cost per object half-width. FIXED, not learned: with F bounded, it is
    #: what confines the argmax's competition to a ball around the hand, and a premise the
    #: optimiser can delete is not a structural guarantee.
    travel_alpha: float = 0.5
    #: beta_0 / alpha: the incumbent's stickiness at init, in object half-widths of travel.
    #: beta is LEARNED from the return and is the whole anti-oscillation device: a switch
    #: needs to clear the incumbent by beta and switching back needs to clear it by beta again.
    stickiness_h: float = 4.0

    #: below the warper's on purpose: authority grows from near-identity. The rate follows
    #: the same cosine over the same budget.
    learning_rate: float = 3e-4
    learning_rate_final: float = 3e-5
    #: Deliberately loose since options outlive the rollout: the behaviour policy can be
    #: several updates stale.
    clip_ratio: float = 0.2
    update_epochs: int = 2
    minibatch_size: int = 512
    value_coef: float = 0.5
    #: its OWN coefficient on the entropy over the pool NORMALISED by log|pool|, so its
    #: strength does not drift with the object count.
    entropy_coef: float = 3e-3
    #: lambda_feas, the weight of the feasibility head's outcome-label BCE next to the policy
    #: loss.
    feasibility_coef: float = 1.0
    max_grad_norm: float = 0.5
    #: the selector's OWN per-window discount. Horizon 1/(1-g) = 1667 windows ~ one full
    #: episode at the largest training count, so a pick is credited with the whole structure
    #: it enables rather than its own transit. MUST stay < 1: at 1 the option return grows
    #: linearly with N and V_sel stops being count-free.
    gamma: float = 0.9994
    #: permanent coverage FLOOR as an exact MIXTURE weight on the UNIFORM pool distribution,
    #: training only. Charged ONCE PER COMMITMENT, not per decision: the floor is blind to the
    #: incumbent's stickiness, so spending it every decision would re-create by exploration
    #: the churn beta exists to prevent.
    epsilon_uniform: float = 0.05
    #: EVERY decision is a sample, so this ring fills ~10k pushes per iteration (2048 scenes,
    #: 48 windows, a decision every ~8 windows). A row waits in the ring, unlabelled, until its
    #: ATTEMPT closes, and an attempt held to the episode end can span hundreds of windows, so
    #: the ring spans ~12 iterations.
    option_capacity: int = 131_072

    #: The re-selection cooldown after a completion, so the settle bounce cannot re-elect the
    #: object: `cooldown_settles` settle times, a settle being `settle_fall_times` free-fall
    #: times from goal height with a floor of `settle_floor_seconds`.
    settle_fall_times: float = 3.0
    settle_floor_seconds: float = 0.1
    cooldown_settles: int = 2


@dataclass
class WaveConfig(AlgorithmConfig):
    """Configuration knobs for the WAVE lifter. An empty ``algorithm_cfg`` uses these defaults.

    The top-level knobs are the ones a released head records and varies; the groups below
    hold the numbers every head shares.
    """

    #: R of the crop's hand-anchored view, in units of h; 0 takes the skill's own disc. An
    #: arm's skill declares no disc, and without one its crop is anchored on the WORKED
    #: OBJECT, so at release the warper looks at the object it is about to fetch and not at
    #: the one its hand is standing in. The panda tasks set 9.375 h, the disc the gantry's
    #: skill uses (0.75 m at h = 0.08 m).
    crop_radius_h: float = 0.0
    #: L, the length crop magnitudes are compressed about, in units of h (2h = object width).
    crop_length_scale_h: float = 2.0
    #: V, the outer normaliser of log(1+|v|/L)/log(1+V/L).
    crop_length_outer: float = 5.0
    #: If true, includes the unwarped pi_0 action in the crop (costs one extra skill forward
    #: per decision).
    crop_tool_action: bool = True

    #: THE SELECTION CLOCK, as a fraction of the task's own per-object time budget. The
    #: selector is asked every `selector_decide_fraction * time_per_object` seconds, and on
    #: top of that whenever the task's own predicate fires (an object completes, or anything
    #: satisfied is knocked loose). 0.05 x 20 s = 1.0 s on the gantry.
    #:
    #: This is the only time constant in the decision path, and it is expressed in a unit
    #: every task declares, so it does not need re-tuning per embodiment.
    selector_decide_fraction: float = 0.05

    #: Warper re-decides every k env ticks (typically 1-2 at train time).
    cadence_ticks: int = 2
    decisions_per_iteration: int = 48
    learning_rate: float = 1e-3

    #: full_io | input_only | output_only
    warp_side: str = "full_io"
    #: both | slope_only | offset_only
    warp_component: str = "both"
    #: Comma list of channel-group names to disable, or "name@k" to pin channels k.. of
    #: one, or "all" for the identity warp (pass-through, no modification).
    disabled_warp_groups: str = ""

    #: learned | analytic | random; `random` is a null-calibration baseline for reordering
    #: churn.
    selector_mode: str = "learned"

    ppo: PPOConfig = field(default_factory=PPOConfig)
    warp_head: WarpHeadConfig = field(default_factory=WarpHeadConfig)
    selector: SelectorConfig = field(default_factory=SelectorConfig)
