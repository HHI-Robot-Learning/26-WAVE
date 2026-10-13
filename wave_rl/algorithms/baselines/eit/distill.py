import torch
import torch.nn.functional as F

from wave_rl.algorithms.heuristic import scripted_selection
from wave_rl.skills.embodiments import load_skill


def distill_from_teacher(eit) -> None:
    """Warm-start ``eit.agent.actor`` by behaviour cloning a frozen skill and seed its replay
    buffer with the skill's own multi-object rollout, in place of EIT's cold start. Called
    once from :meth:`EIT.learn`'s first-call setup; it leaves ``eit._rows`` and
    ``eit.total_steps`` where the rollout stopped.

    Two phases:

      1. **Rollout**: the frozen teacher, ``cfg.distill_teacher`` (EIT is end-to-end and names
         no skill of its own), drives ``eit.env`` for ``cfg.distill_steps`` steps, one object
         at a time (:func:`~wave_rl.algorithms.heuristic.scripted_selection`), through the skill's own single-object
         observation and deterministic action. Every transition is stored in EIT's own
         multi-object rows, so the student sees the whole scene (distractors included) even
         though the label only ever moves one object, which is what teaches the attention to
         ignore objects the teacher isn't acting on.
      2. **Behaviour cloning**: ``cfg.distill_actor_steps`` MSE-regression steps of the actor
         against the stored teacher actions, sampled uniformly from that buffer.
         ``actor_target`` is synced afterwards.

    The critic is left untrained: the teacher's Q-values were fit under its single-object
    reward and discount, not the scene reward, so it warms up the normal way, via TD3 on the
    buffer this function populated.
    """
    cfg = eit.cfg
    env = eit.env
    agent = eit.agent
    device = eit.device
    b = env.num_scenes

    teacher = load_skill(cfg.distill_teacher, env.scene, env.n_robot_qpos, env.n_robot_qvel,
                         env.n_ctrl, env.action_dim, env.device)
    print(f"-- Distilling teacher '{cfg.distill_teacher}' into EIT's actor "
          f"({cfg.distill_steps} rollout steps, {cfg.distill_actor_steps} BC steps)...")

    obs = env.reset()
    eit.reward.reset(obs)
    rows = eit.encode(obs)

    steps = 0
    while steps < cfg.distill_steps:
        k = scripted_selection(eit.measure, obs)
        with torch.no_grad():
            action = teacher.act(teacher.present(obs, k))

        next_obs, next_obs_true, terminated, truncated, info = env.step(action)
        rewards = eit.reward.step(next_obs_true, next_obs, terminated, truncated, info)
        true_rows = eit.encode(next_obs_true)
        next_rows = eit.encode(next_obs)
        agent.replay_buffer.add(rows, action, true_rows, rewards,
                                eit.reward.discount * (~terminated).float())
        eit.reward.refresh(next_obs)

        rows = next_rows
        obs = next_obs
        steps += b
        eit.total_steps += b

    eit._rows = rows
    print(f"-- Collected {agent.replay_buffer.size} teacher transitions; running "
          f"{cfg.distill_actor_steps} actor BC steps...")

    opt = torch.optim.Adam(agent.actor.parameters(), lr=cfg.distill_actor_lr)
    buf = agent.replay_buffer
    for _ in range(cfg.distill_actor_steps):
        idx = torch.randint(0, buf.size, (cfg.distill_batch_size,), device=device)
        robot, tokens, goals, mask = agent.entities(buf.state[idx])
        loss = F.mse_loss(agent.actor(robot, tokens, goals, mask), buf.action[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        eit.logger.add("distill/bc_loss", loss.detach())

    agent.actor_target.load_state_dict(agent.actor.state_dict())
    print(f"-- Distillation done (final BC loss={float(loss.detach()):.4f}); "
          f"handing off to normal EIT training.")
