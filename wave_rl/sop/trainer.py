import dataclasses
import math
from pathlib import Path

import numpy as np
import omegaconf
import torch

from wave_rl.config import SkillConfig, SkillTrainConfig
from wave_rl.logger import Logger
from wave_rl.seeding import set_global_seeds
from wave_rl.sim.scene import skill_scene
from wave_rl.sim.sim import Sim
from wave_rl.sop.curriculum import Curriculum
from wave_rl.sop.env import SkillEnv
from wave_rl.td7.agent import TD7Agent

OUT_DIR = Path("data/skills")


def evaluate(agent: TD7Agent, env: SkillEnv, starts: np.ndarray, goals: np.ndarray,
             time_limit: float, per_world: int) -> dict:
    """Success rate, mean steps of the successes and mean final goal distance of the noise-free
    policy over `per_world` episodes in every world (a fixed quota per world, so fast
    episodes are not over-sampled)."""
    env.set_starts_goals(starts, goals)
    env.time_limit = time_limit
    W = env.num_scenes
    done = torch.zeros(W, dtype=torch.long, device=env.device)
    success, steps, dist = [], [], []
    obs = env.reset()
    while bool((done < per_world).any()):
        obs, _, terminated, truncated = env.step(agent.select_action(obs, use_exploration=False))
        counted = (terminated | truncated) & (done < per_world)
        success.append(terminated[counted])
        steps.append(env.scene_time[counted] / env.cfg.tau_step)
        dist.append(env.goal_distance(env.feat)[counted])
        done = done + (terminated | truncated).long()
        obs = env.auto_reset()
    success, steps, dist = torch.cat(success).float(), torch.cat(steps), torch.cat(dist)
    return {"success": float(success.mean()),
            "steps": float(steps[success > 0].mean()) if bool(success.any()) else math.inf,
            "final_dist": float(dist.mean()), "episodes": int(success.numel())}


def write_skill(agent: TD7Agent, cfg: SkillTrainConfig, path: Path) -> None:
    """``<path>.torch`` and the sidecar ``<path>.yaml`` the skill loader reads."""
    agent.save(path.with_suffix(".torch"))
    sidecar = SkillConfig(embodiment=cfg.embodiment, scene=cfg.scene,
                          eps_pos=cfg.reward.eps_pos, vel_scale=cfg.reward.vel_scale,
                          referr_scale=cfg.reward.referr_scale, td7=agent.hp)
    omegaconf.OmegaConf.save(omegaconf.OmegaConf.structured(sidecar), path.with_suffix(".yaml"))


def train_skill(cfg: SkillTrainConfig, name: str) -> Path:
    """Train one skill with TD7 on its single-object task, keep the policy that scores best on
    a fixed evaluation set, and write it to ``data/skills/<name>.torch`` with its sidecar."""
    set_global_seeds(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    p = cfg.plant
    scene = skill_scene(cfg.scene, p.friction, p.solref, p.solimp)
    env = SkillEnv(Sim(scene, p, cfg.num_scenes), p, cfg.reward)
    eval_env = SkillEnv(Sim(scene, p, cfg.eval_scenes), p, cfg.reward)
    td7 = dataclasses.replace(cfg.td7, discount=0.5 ** (p.tau_step / cfg.discount_half_life))
    agent = TD7Agent(env.observation.shape[1], env.action_dim, td7)
    warm = cfg.init_from is not None
    if warm:
        print(f"-- warm start from {cfg.init_from}")
        agent.load(cfg.init_from)
    curriculum = Curriculum(cfg.curriculum, env.sim.model, rng)
    eval_starts, eval_goals = curriculum.sample(cfg.eval_scenes, None)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / name
    logger = Logger(name, env.device, {"project": "wave-skills"} if cfg.wandb else None)
    logger.config(omegaconf.OmegaConf.to_container(omegaconf.OmegaConf.structured(cfg)))
    print(f"-- skill {name}: {cfg.embodiment} on {cfg.scene}, {cfg.num_scenes} worlds, "
          f"{cfg.T_end} ticks, discount {td7.discount:.4f}")

    W = cfg.num_scenes
    ep_reward = torch.zeros(W, device=env.device)
    noise0 = td7.exploration_noise
    best = -1.
    t = 0
    while t < cfg.T_end:
        alpha = curriculum.progress(t)
        env.time_limit = 2. * alpha * cfg.curriculum.nominal_time_limit
        # enough pairs for episodes lasting a fifth of the nominal limit, then twice what the
        # previous block used
        n = env.starts_goals_counter * 2 if t else int(min(
            1e6, cfg.curriculum.T_block * W
            / (0.2 * cfg.curriculum.nominal_time_limit / p.tau_step)))
        env.set_starts_goals(*curriculum.sample(n, alpha))
        logger.add("curriculum/alpha", alpha)
        logger.add("curriculum/time_limit", env.time_limit)
        print(f"== t {t}: alpha {alpha:.3f}, {n} start/goal pairs, "
              f"time limit {env.time_limit:.1f} s")
        for _ in range(cfg.curriculum.T_block):
            if td7.expl_noise_final is not None and td7.expl_noise_anneal_end > \
                    td7.expl_noise_anneal_start:
                frac = min(1., max(0., (t - td7.expl_noise_anneal_start)
                                   / (td7.expl_noise_anneal_end - td7.expl_noise_anneal_start)))
                agent.hp.exploration_noise = noise0 + frac * (td7.expl_noise_final - noise0)
            learning = agent.replay_buffer.size >= td7.timesteps_before_training and (
                not warm or t >= td7.warm_min_ticks)

            obs = env.auto_reset()
            if learning or warm:
                action = agent.select_action(obs)
            else:
                action = torch.rand(W, env.action_dim, device=env.device) * 2. - 1.
            next_obs, reward, terminated, truncated = env.step(action)
            agent.replay_buffer.add(obs, action, next_obs, reward,
                                    td7.discount * (~terminated).float())
            if learning:
                agent.train(logger, td7.training_steps)

            ep_reward += reward
            finished = (terminated | truncated).float()
            logger.add("ep/success", terminated.float(), finished)
            logger.add("ep/reward", ep_reward * finished, finished)
            logger.add("ep/steps", env.scene_time / p.tau_step * finished, finished)
            logger.add("ep/final_dist", env.goal_distance(env.feat) * finished, finished)
            logger.add("ep/episodes_per_tick", finished.sum())
            ep_reward *= 1. - finished
            t += 1

            if t % cfg.eval_every == 0:
                result = evaluate(agent, eval_env, eval_starts, eval_goals,
                                  cfg.eval_time_limit, 1)
                for k, v in result.items():
                    logger.add(f"eval/{k}", v)
                write_skill(agent, cfg, out.with_name(f"{name}_last"))
                if result["success"] > best:
                    best = result["success"]
                    write_skill(agent, cfg, out.with_name(f"{name}_best"))
                print(f"-- eval at {t}: {result} (best {best:.3f})")
            if t % cfg.log_every == 0:
                record = logger.flush(t * W, {"train/exploration_noise": agent.hp.exploration_noise,
                                              "buffer/size": agent.replay_buffer.size,
                                              "env/nan_resets": env.nan_resets})
                print(f"-- t={t}  success={record['ep/success']:.3f}  "
                      f"reward={record['ep/reward']:.3f}  "
                      f"{record['time/env_steps_per_second']:,.0f} steps/s")
            if t >= cfg.T_end:
                break

    if best < 0.:
        write_skill(agent, cfg, out.with_name(f"{name}_best"))
    agent.load(out.with_name(f"{name}_best").with_suffix(".torch"))
    per_world = math.ceil(cfg.eval_episodes / cfg.eval_scenes)
    starts, goals = curriculum.sample(per_world * cfg.eval_scenes, None)
    result = evaluate(agent, eval_env, starts, goals, cfg.eval_time_limit, per_world)
    logger.flush(t * W, {f"final/{k}": v for k, v in result.items()})
    logger.finish()
    write_skill(agent, cfg, out)
    print(f"== skill {name}: final evaluation of the best policy {result}; "
          f"written to {out.with_suffix('.torch')}")
    return out.with_suffix(".torch")
