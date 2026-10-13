# WAVE: Warping Actions and Views of an Expert

Code for the paper **"Object Count Generalization Is an Interface Problem: Warping a Frozen
Skill Instead of Retraining It"**.

This repository contains:

- the MuJoCo scenes of the gantry robot and the Franka Panda (parallel-jaw gripper and
  sphere-tip pusher), simulated on the GPU with [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp)
- the multi-object environment with the stacking, line, square, pyramid and wall tasks
- WAVE, the baselines and the ablations
- the training of the single-object skills (TD7) that WAVE warps
- the released skills, layout pools and trained WAVE heads

## Installation

A Linux machine with an NVIDIA GPU is required (the simulator runs only on CUDA). The paper's
runs used one A100 each.

```bash
git clone https://github.com/HHI-Robot-Learning/26-WAVE.git
cd 26-WAVE
uv sync                      # or: pip install -e .  (with torch 2.11 for your CUDA)
source .venv/bin/activate
python download.py           # skills, layout pools and WAVE heads (~0.8 GB)
```

`python download.py skills pools` fetches only what training needs; `heads` and `ablations`
are the trained WAVE heads. Training videos are encoded with the `ffmpeg` binary and rendered
offscreen with EGL (`MUJOCO_GL=egl`).

## Quick start

Evaluate a released head on cube stacking (h = 3) for 1 to 30 cubes:

```bash
python eval.py gantry_towers data/heads/gantry_towers_seed1.pt train.seed=1
```

Train WAVE on the same task from scratch, then sweep it (500,000,000 environment steps):

```bash
python train.py gantry_towers train.seed=1
```

A run writes its head to `data/runs/<run>.pt`, retained snapshots to `data/runs/snapshots/`,
every training and sweep record to `data/runs/<run>_metrics.jsonl` and the sweep to
`data/runs/<run>_sweep.csv`. With `wandb=true` the run is also logged to Weights & Biases.

An evaluation builds one scene with as many cubes as the largest count it sweeps (surplus
cubes are parked outside the arena) and draws the layouts of every count from that scene's
pool. `train.seed` selects the pool's seed: in the paper each head was evaluated on the pool
of its own seed.


## Reproducing the paper

Every result was produced with three seeds, `1`, `42` and `100`. The commands below evaluate the released heads, `data/heads/<task>_seed<s>.pt`; replace
`eval.py <task> <head>` by `train.py <task>` to train the same head from scratch.

```bash
for s in 1 42 100; do python eval.py gantry_towers data/heads/gantry_towers_seed$s.pt train.seed=$s; done
```

### Count generalization on the gantry and the Panda arm

| Paper | Task config | Head | Trained on | Evaluated on |
|---|---|---|---|---|
| gantry, stacking h = 3 | `gantry_towers` | `gantry_towers` | 1-7 | 1-30 |
| gantry, stacking h = 6 | `gantry_towers_h6_n13` | `gantry_towers_h6_n13` | 1-13, continued from `gantry_towers_h6` (1-7) | 1-30, 40 s per cube |
| gantry, line | `gantry_line` | `gantry_line` | 1-6 | 1-30 |
| gantry, square | `gantry_square` | `gantry_square` | 1-9 | 1-30 |
| gantry, pyramid | `gantry_pyramid` | `gantry_pyramid` | 1-6 | 1-30 |
| gantry, wall (two courses, growing longer) | `gantry_wall_long` | `gantry_wall` | 1-8 on `gantry_wall` (four wide, growing upward) | 1-30 |
| Panda gripper, stacking h = 3 | `panda_towers` | `panda_towers` | 1-7 | 1-15 |
| Panda gripper, pyramid | `panda_pyramid` | `panda_pyramid` | 1-6 | 1-15 |
| Panda gripper, wall | `panda_wall` | `panda_wall` | 1-8 | 1-15 |
| Panda pusher, line | `push_line` | `push_line` | 1-9 | 1-20 |
| Panda pusher, square | `push_square` | `push_square` | 1-9 | 1-30 |


### Baselines

The baselines are trained on cube stacking (`gantry_towers_<suffix>`) and on the line
(`gantry_line_<suffix>`) with three seeds each. Every
learner, WAVE and the baselines alike, trains on the same scene reward (closeness shaping,
occupancy income, the success bonus and the first-placement bonus) under the per-scene discount
the algorithm base derives from the episode length; the environment computes no reward.

| Paper | Suffix | Algorithm |
|---|---|---|
| Skill alone | `skill_alone` | the frozen skill with the scripted selection, no training |
| Fine-tuning μ | `finetune` | the skill's own actor and critic trained further |
| JSRL | `jsrl` | the skill guides, a TD7 policy initialised from it takes over |
| HRL subgoals | `hrl` | a TD7 manager proposes goals to the frozen skill |
| EIT | `eit` | entity transformer, every object and goal a token |
| EIT distilled from μ | `eit_distill` | EIT initialised by imitating the skill |
| TD7 on the crop | `td7_crop` | a TD7 policy on WAVE's crop, no skill; evaluated on 1-10 |
| Fine-tuning μ on the crop, ω = 0 | `finetune_crop` | the skill's actor reads the crop through zero-initialised weights; evaluated on 1-10 |
| Fine-tuning μ on the crop, ω = 1 | `finetune_crop algorithm_cfg.anchor_alpha=1.0` | the same with the anchor to the skill's action |

```bash
for s in 1 42 100; do python train.py gantry_towers_jsrl train.seed=$s; done
python train.py gantry_line_skill_alone        # no training, the sweep only
```

### Ablations

On cube stacking (`gantry_towers`), the square (`gantry_square`) and the pyramid
(`gantry_pyramid`), seeds 1 and 42. Each ablation is one override of the WAVE config. The
released heads are `data/heads/ablations/<task>_<variant>_seed<s>.pt`; an evaluation replays a
head under the configuration it was trained with, so no override is needed to evaluate one.

| Variant | Override |
|---|---|
| `selanalytic`: nearest-first selection | `algorithm_cfg.selector_mode=analytic` |
| `selrandom`: random selection | `algorithm_cfg.selector_mode=random` |
| `inputwarponly`: the warp edits the skill's inputs only | `algorithm_cfg.warp_side=input_only` |
| `outputwarponly`: the warp edits the skill's outputs only | `algorithm_cfg.warp_side=output_only` |
| `slopeonly`: multipliers only | `algorithm_cfg.warp_component=slope_only` |
| `offsetonly`: offsets only | `algorithm_cfg.warp_component=offset_only` |
| `cadence1` / `cadence4` (stacking only): a new warp every 1 / 4 control steps | `algorithm_cfg.cadence_ticks=1` / `=4` |

```bash
python train.py gantry_square algorithm_cfg.selector_mode=random train.seed=42 \
    run_name=gantry_square_selrandom_seed42
python eval.py gantry_square data/heads/ablations/gantry_square_selrandom_seed42.pt train.seed=42
```

### Measured constants of the bound

The success probability per attempt `p_e` and the mean number of cubes an attempt unplaces `λ`
are the sweep columns `sweep/windows/success_p_e` and `sweep/windows/unplaced_per_window_lambda`. `λ_max`
and the crop-coverage term `d_cov` came from a separate replay analysis that is not part of
this release.

### Interventional estimation of warp necessity

```bash
python necessity.py gantry_towers_h6 data/heads/gantry_towers_h6_seed42.pt --out-dir data/necessity
```

Every control step of 32 deployed four-cube episodes is branched into copies that run with the
warper, its input side or its output side switched off, and the extra time each copy needs to
finish the current phase is recorded. The script prints the per-phase table and draws one
figure per rendered episode (the paper shows episode 20); `--report-only` redraws both from the
recording.


## Training a skill

The skills WAVE warps are single-object TD7 policies that move one cube to a goal pose. The
released skills are in `skills/`. To train one and run WAVE on it:

```bash
python train_skill.py gantry_gripper       # or panda_gripper, panda_pusher
python train.py gantry_towers skill=data/skills/gantry_gripper_seed0.torch
```

`train_skill.py` writes `data/skills/<name>.torch` with its sidecar `<name>.yaml` (the goal
tolerance and how the skill reads its observation), keeping the policy that scores best on a
fixed set of evaluation tasks. Episodes start and end at physically valid configurations from
the pools in `data/sop_pools/`. The gantry skill uses a curriculum that starts each episode
close to its goal and widens the gap over training; the Panda skills pair random starts with
random goals from the beginning.

📌 Each released skill is the last stage of a longer training lineage: it was warm-started
from an earlier policy, and `configs/skills/` holds the recipe of that last stage
(`init_from=<skill.torch>` repeats it from a given policy). Training from scratch with these
configs is not guaranteed to give the released skill. In particular, the last stage of the
gantry skill starts every episode with the cube already between the jaws, so a gantry skill
trained from scratch with it never learns to pick a cube up from the floor.
